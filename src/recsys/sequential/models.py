"""GPT-style causal Transformer user tower trained on next-item prediction.

The user state after event ``t`` is the Transformer output at position ``t``;
relevance is ``r(u_t, i) = <h_t, E_i>`` with the item table ``E`` shared between
the input embeddings and the item tower. Training uses every position of every
window at once; inference uses the state after the last event.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from recsys.two_tower.data import pad_sequences


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where position ``t`` sees only valid positions ``<= t``."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} is not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        q, k, v = (
            self.qkv(x).view(batch, seq_len, 3, self.n_heads, d_model // self.n_heads).permute(2, 0, 3, 1, 4)
        )
        causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril()
        # Every query row keeps at least position 0, so no row is fully masked.
        attn_mask = causal & mask[:, None, None, :]
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(out.transpose(1, 2).reshape(batch, seq_len, d_model))


class MLP(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Pre-LayerNorm Transformer block: attention and MLP sublayers with residuals."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln_mlp = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln_attn(x), mask))
        return x + self.dropout(self.mlp(self.ln_mlp(x)))


class GPT(nn.Module):
    """Input dropout, ``n_layers`` causal blocks and a final LayerNorm: ``(B, S, D) -> (B, S, D)``."""

    def __init__(self, n_layers: int, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, dropout) for _ in range(n_layers))
        self.ln_final = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, mask)
        return self.ln_final(x)


class TransformerUserEncoder(nn.Module):
    """Item + learned position embeddings through a :class:`GPT`.

    Returns padded hidden states ``(B, max_seq_len, D)`` and the validity mask.
    Batches are always padded to ``max_seq_len`` so that every batch has the same
    shape (see :func:`recsys.ops.pad_to_bucket` for why this matters on MPS).
    """

    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
        max_seq_len: int = 100,
        n_layers: int = 2,
        n_heads: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.position_embeddings = nn.Embedding(max_seq_len, embedding_dim)
        self.gpt = GPT(n_layers, embedding_dim, n_heads, dropout)

    def pad(self, values: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        padded, mask = pad_sequences(values, lengths)
        extra = self.max_seq_len - padded.shape[1]
        if extra < 0:
            raise ValueError(f"history of length {padded.shape[1]} exceeds max_seq_len={self.max_seq_len}")
        return F.pad(padded, (0, extra)), F.pad(mask, (0, extra))

    def forward(self, history: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        item_ids, mask = self.pad(history["item_id"], history["length"])
        x = self.item_embeddings(item_ids) + self.position_embeddings.weight
        return self.gpt(x, mask), mask


class NextItemTransformer(nn.Module):
    """Returns a scalar loss in train mode and ``(batch, num_items)`` scores in eval mode.

    The loss is a sampled softmax at every valid position. ``num_negatives``
    negatives are drawn per batch from the batch's own targets (so in proportion
    to item popularity) and shared by all positions. When ``log_q`` is given,
    ``log q(i)`` is subtracted from every logit, positive and negatives alike.
    Negatives that coincide with a position's positive are masked out.
    """

    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
        log_q: Optional[torch.Tensor] = None,
        num_negatives: int = 512,
        max_seq_len: int = 100,
        n_layers: int = 2,
        n_heads: int = 2,
        dropout: float = 0.1,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.encoder = TransformerUserEncoder(num_items, embedding_dim, max_seq_len, n_layers, n_heads, dropout)
        self.num_negatives = num_negatives
        self.register_buffer("log_q", None if log_q is None else log_q.float())
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.trunc_normal_(module.weight, std=init_std, a=-2 * init_std, b=2 * init_std)
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

    @property
    def item_matrix(self) -> torch.Tensor:
        return self.encoder.item_embeddings.weight

    def forward(self, inputs: Dict[str, Any]) -> torch.Tensor:
        hidden, mask = self.encoder(inputs["history"])
        if self.training:
            targets, _ = self.encoder.pad(inputs["targets"], inputs["history"]["length"])
            return self.compute_loss(hidden, mask, targets)
        last = inputs["history"]["length"] - 1
        user_repr = hidden[torch.arange(len(last), device=hidden.device), last]
        return user_repr @ self.item_matrix.T

    def compute_loss(self, hidden: torch.Tensor, mask: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """``hidden`` ``(B, S, D)``, ``mask`` and padded ``targets`` ``(B, S)`` -> mean loss over valid positions."""
        sampled = torch.multinomial(mask.flatten().float(), self.num_negatives, replacement=True)
        negatives = targets.flatten()[sampled]

        pos = (hidden * self.item_matrix[targets]).sum(-1, keepdim=True)
        neg = hidden @ self.item_matrix[negatives].T
        if self.log_q is not None:
            pos = pos - self.log_q[targets].unsqueeze(-1)
            neg = neg - self.log_q[negatives]
        neg = neg.masked_fill(targets.unsqueeze(-1) == negatives, float("-inf"))

        logits = torch.cat([pos, neg], dim=-1)
        loss = -F.log_softmax(logits, dim=-1)[..., 0]
        return (loss * mask).sum() / mask.sum()
