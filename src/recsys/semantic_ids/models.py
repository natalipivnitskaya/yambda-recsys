"""Two ways to use semantic IDs on top of the causal Transformer user tower.

* :class:`SemanticIdEmbedding` builds an item's vector from its codes (sum of
  one embedding per level, optionally plus a per-item embedding). It drops into
  :class:`recsys.sequential.NextItemTransformer` in place of ``nn.Embedding``.
* :class:`GenerativeRetrieval` predicts the next item's code level by level
  (TIGER-style generative retrieval) and recommends with a beam search that is
  constrained to codes of real items.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from recsys.semantic_ids.quantize import SemanticIds
from recsys.sequential.models import TransformerUserEncoder, init_weights


class SemanticIdEmbedding(nn.Module):
    """``E[i] = sum_l C_l[code_l(i)]`` (+ ``V[i]`` when ``use_item_ids``), with the full table as ``.weight``."""

    def __init__(self, ids: SemanticIds, embedding_dim: int, use_item_ids: bool = True) -> None:
        super().__init__()
        offsets = torch.tensor([0] + ids.codebook_sizes[:-1]).cumsum(0)
        self.register_buffer("flat_codes", ids.codes + offsets)
        self.code_embeddings = nn.Embedding(sum(ids.codebook_sizes), embedding_dim)
        self.item_embeddings = nn.Embedding(len(ids.codes), embedding_dim) if use_item_ids else None

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        out = self.code_embeddings(self.flat_codes[item_ids]).sum(-2)
        if self.item_embeddings is not None:
            out = out + self.item_embeddings(item_ids)
        return out

    @property
    def weight(self) -> torch.Tensor:
        return self.forward(torch.arange(len(self.flat_codes), device=self.flat_codes.device))


def build_prefix_tables(codes: torch.Tensor, codebook_sizes: List[int]) -> List[torch.Tensor]:
    """Trie of valid codes as dense lookup tables.

    ``tables[l][p, c]`` is the id of the prefix reached by appending code ``c`` to
    level-``l`` prefix ``p`` (``-1`` if no item has that prefix). At the last level
    the "prefix id" is the item id, since full codes are unique.
    """
    num_items, levels = codes.shape
    prefix = torch.zeros(num_items, dtype=torch.long)
    num_prefixes = 1
    tables = []
    for level in range(levels):
        if level == levels - 1:
            child = torch.arange(num_items)
        else:
            _, child = torch.unique(prefix * codebook_sizes[level] + codes[:, level], return_inverse=True)
        table = torch.full((num_prefixes, codebook_sizes[level]), -1, dtype=torch.int32)
        table[prefix, codes[:, level]] = child.int()
        tables.append(table)
        prefix, num_prefixes = child, int(child.max()) + 1
    return tables


class GenerativeRetrieval(nn.Module):
    """Predicts the next item's semantic ID one level at a time.

    The decoder state starts as the user state ``h_t`` and after each level adds
    the embedding of the chosen code: ``s_0 = h_t``, ``s_{l+1} = s_l + C_l[c_l]``.
    Level ``l`` is a softmax over its codebook with logits ``f_l(s_l) · C_l``,
    so ``log p(item) = sum_l log p(c_l | h_t, c_<l)``.

    Train mode returns the summed per-level cross-entropy at every valid position.
    Eval mode returns ``(batch, num_items)`` scores: the log-probability of each
    item found by a beam search of width ``beam_size`` over valid codes, ``-inf``
    for all other items.
    """

    def __init__(
        self,
        ids: SemanticIds,
        embedding_dim: int,
        max_seq_len: int = 100,
        n_layers: int = 2,
        n_heads: int = 2,
        dropout: float = 0.1,
        use_item_ids: bool = False,
        beam_size: int = 100,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        num_items = len(ids.codes)
        self.encoder = TransformerUserEncoder(
            num_items, embedding_dim, max_seq_len, n_layers, n_heads, dropout,
            item_embeddings=SemanticIdEmbedding(ids, embedding_dim, use_item_ids),
        )
        self.register_buffer("codes", ids.codes.clone())
        self.code_embeddings = nn.ModuleList(nn.Embedding(size, embedding_dim) for size in ids.codebook_sizes)
        self.heads = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, embedding_dim), nn.GELU(),
                          nn.Linear(embedding_dim, embedding_dim))
            for _ in ids.codebook_sizes
        )
        for level, table in enumerate(build_prefix_tables(ids.codes, ids.codebook_sizes)):
            self.register_buffer(f"prefix_table_{level}", table)
        self.beam_size = beam_size
        init_weights(self, init_std)

    @property
    def num_levels(self) -> int:
        return self.codes.shape[1]

    def level_logits(self, state: torch.Tensor, level: int) -> torch.Tensor:
        return self.heads[level](state) @ self.code_embeddings[level].weight.T

    def forward(self, inputs: Dict[str, Any]) -> torch.Tensor:
        hidden, mask = self.encoder(inputs["history"])
        if self.training:
            targets, _ = self.encoder.pad(inputs["targets"], inputs["history"]["length"])
            return self.compute_loss(hidden, mask, self.codes[targets])
        last = inputs["history"]["length"] - 1
        user = hidden[torch.arange(len(last), device=hidden.device), last]
        items, scores = self.beam_search(user, self.beam_size)
        return self._dense_scores(items, scores)

    def compute_loss(self, hidden: torch.Tensor, mask: torch.Tensor, target_codes: torch.Tensor) -> torch.Tensor:
        """``hidden`` ``(B, S, D)``, ``mask`` ``(B, S)``, ``target_codes`` ``(B, S, L)``."""
        state, loss = hidden, hidden.new_zeros(())
        for level in range(self.num_levels):
            logits = self.level_logits(state, level)
            nll = F.cross_entropy(logits.flatten(0, 1), target_codes[..., level].flatten(), reduction="none")
            loss = loss + (nll.view_as(mask) * mask).sum() / mask.sum()
            state = state + self.code_embeddings[level](target_codes[..., level])
        return loss

    @torch.no_grad()
    def beam_search(self, user: torch.Tensor, beam_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Top items by ``log p(item | user)`` among valid codes: ``(B, beam)`` item ids and scores.

        Slots without a valid item (catalog smaller than the beam) get id ``-1`` and score ``-inf``.
        """
        batch = len(user)
        prefix = torch.zeros(batch, 1, dtype=torch.long, device=user.device)
        score = torch.zeros(batch, 1, device=user.device)
        state = user.unsqueeze(1)
        for level in range(self.num_levels):
            table = getattr(self, f"prefix_table_{level}")
            children = table[prefix.clamp(min=0)].long()                      # (B, W, K)
            children = children.masked_fill(prefix.unsqueeze(-1) < 0, -1)
            log_p = F.log_softmax(self.level_logits(state, level), dim=-1)
            total = (score.unsqueeze(-1) + log_p).masked_fill(children < 0, float("-inf"))
            k = min(beam_size, total.shape[1] * total.shape[2])
            score, flat = total.flatten(1).topk(k, dim=1)
            beam, code = flat // total.shape[2], flat % total.shape[2]
            prefix = children.flatten(1).gather(1, flat).masked_fill(torch.isinf(score), -1)
            state = state.gather(1, beam.unsqueeze(-1).expand(-1, -1, state.shape[-1]))
            state = state + self.code_embeddings[level](code)
        return prefix, score

    @torch.no_grad()
    def exact_scores(self, user: torch.Tensor) -> torch.Tensor:
        """``log p(item | user)`` for every item, ``(B, num_items)``. Memory ``B x num_items x D``: small catalogs only."""
        codes = self.codes
        state = user.unsqueeze(1).expand(-1, len(codes), -1)
        total = user.new_zeros(len(user), len(codes))
        for level in range(self.num_levels):
            log_p = F.log_softmax(self.level_logits(state, level), dim=-1)
            total = total + log_p.gather(-1, codes[:, level].expand(len(user), -1).unsqueeze(-1)).squeeze(-1)
            state = state + self.code_embeddings[level](codes[:, level])
        return total

    def _dense_scores(self, items: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        num_items = len(self.codes)
        dense = scores.new_full((len(items), num_items + 1), float("-inf"))
        dense.scatter_(1, items.masked_fill(items < 0, num_items), scores)
        return dense[:, :num_items]
