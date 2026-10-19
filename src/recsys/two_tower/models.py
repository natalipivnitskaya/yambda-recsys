"""Bag-of-items two-tower model and the training objectives compared in this project.

Relevance is a dot product ``r(u, i) = <P_u, E_i>`` where ``E`` is the item
embedding table and ``P_u`` is the sum of embeddings of the user's history.
The item tower and the user tower share ``E``.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class UserEncoder(nn.Module):
    """``P_u = sum_k E[i_k]`` over a flattened batch of histories.

    The flattened history is padded up to a multiple of ``bucket`` so that only a
    few distinct tensor shapes occur: the MPS backend compiles and caches a graph
    per shape, and per-batch lengths would otherwise grow memory without bound.
    Padding positions are summed into an extra row that is dropped.
    """

    def __init__(self, num_items: int, embedding_dim: int, bucket: int = 4096) -> None:
        super().__init__()
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.bucket = bucket

    def forward(self, history: Dict[str, torch.Tensor]) -> torch.Tensor:
        item_ids, lengths = history["item_id"], history["length"]
        batch_size, total = len(lengths), len(item_ids)
        padded_total = -(-total // self.bucket) * self.bucket
        item_ids = F.pad(item_ids, (0, padded_total - total))

        # segment[k] = user that flattened position k belongs to; padding maps to row `batch_size`
        # (searchsorted instead of repeat_interleave, which is broken on MPS)
        positions = torch.arange(padded_total, device=lengths.device)
        segment = torch.searchsorted(torch.cumsum(lengths, 0), positions, right=True)

        embeddings = self.item_embeddings(item_ids)
        out = embeddings.new_zeros((batch_size + 1, embeddings.shape[1]))
        return out.index_add_(0, segment, embeddings)[:batch_size]


class TwoTower(nn.Module):
    """Returns a scalar loss in train mode and ``(batch, num_items)`` scores in eval mode.

    Subclasses implement :meth:`compute_loss`.
    """

    def __init__(self, num_items: int, embedding_dim: int, init_std: float = 0.02) -> None:
        super().__init__()
        self.encoder = UserEncoder(num_items=num_items, embedding_dim=embedding_dim)
        nn.init.trunc_normal_(self.item_matrix, std=init_std, a=-2 * init_std, b=2 * init_std)

    @property
    def item_matrix(self) -> torch.Tensor:
        return self.encoder.item_embeddings.weight

    def compute_loss(self, user_repr: torch.Tensor, inputs: Dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, inputs: Dict[str, Any]) -> torch.Tensor:
        user_repr = self.encoder(inputs["history"])
        if self.training:
            return self.compute_loss(user_repr, inputs)
        return user_repr @ self.item_matrix.T

    def _positive_scores(self, user_repr: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return (user_repr * self.item_matrix[labels]).sum(dim=-1)

    def _uniform_items(self, *shape: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.item_matrix.shape[0], shape, device=device)

    def _per_user_scores(self, user_repr: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Scores of ``items`` of shape ``(batch, n)`` against their own user: ``(batch, n)``."""
        return torch.bmm(self.item_matrix[items], user_repr.unsqueeze(2)).squeeze(2)


class FullSoftmaxModel(TwoTower):
    """Cross-entropy over the whole catalog. The reference objective, but O(|I|) per sample."""

    def compute_loss(self, user_repr, inputs):
        return F.cross_entropy(user_repr @ self.item_matrix.T, inputs["label"])


class BCEModel(TwoTower):
    """Pointwise: positive vs. ``num_negatives`` uniform negatives, each as a binary label."""

    def __init__(self, num_items: int, embedding_dim: int, num_negatives: int = 2048) -> None:
        super().__init__(num_items, embedding_dim)
        self.num_negatives = num_negatives

    def compute_loss(self, user_repr, inputs):
        pos = self._positive_scores(user_repr, inputs["label"])
        negatives = self._uniform_items(len(user_repr), self.num_negatives, device=user_repr.device)
        neg = self._per_user_scores(user_repr, negatives)
        return -(F.logsigmoid(pos).mean() + F.logsigmoid(-neg).mean())


class BPRModel(TwoTower):
    """Pairwise: ``-log sigma(r(u, i+) - r(u, i-))`` with one uniform negative per user."""

    def compute_loss(self, user_repr, inputs):
        pos = self._positive_scores(user_repr, inputs["label"])
        negatives = self._uniform_items(len(user_repr), device=user_repr.device)
        neg = self._positive_scores(user_repr, negatives)
        return -F.logsigmoid(pos - neg).mean()


class SampledSoftmaxModel(TwoTower):
    """Softmax over the positive and ``num_negatives`` per-user uniform negatives."""

    def __init__(self, num_items: int, embedding_dim: int, num_negatives: int = 2048) -> None:
        super().__init__(num_items, embedding_dim)
        self.num_negatives = num_negatives

    def compute_loss(self, user_repr, inputs):
        pos = self._positive_scores(user_repr, inputs["label"]).unsqueeze(1)
        negatives = self._uniform_items(len(user_repr), self.num_negatives, device=user_repr.device)
        neg = self._per_user_scores(user_repr, negatives)
        logits = torch.cat([pos, neg], dim=1)
        return F.cross_entropy(logits, torch.zeros(len(user_repr), dtype=torch.long, device=user_repr.device))


class InBatchSoftmaxModel(TwoTower):
    """Softmax where other users' positives act as negatives.

    Optionally ``num_uniform_negatives`` catalog-uniform items shared across the
    batch are appended, which counteracts the popularity bias of in-batch sampling.
    """

    def __init__(self, num_items: int, embedding_dim: int, num_uniform_negatives: int = 0) -> None:
        super().__init__(num_items, embedding_dim)
        self.num_uniform_negatives = num_uniform_negatives

    def compute_loss(self, user_repr, inputs):
        logits = user_repr @ self.item_matrix[inputs["label"]].T
        if self.num_uniform_negatives:
            uniform = self._uniform_items(self.num_uniform_negatives, device=user_repr.device)
            logits = torch.cat([logits, user_repr @ self.item_matrix[uniform].T], dim=1)
        target = torch.arange(len(user_repr), device=user_repr.device)
        return F.cross_entropy(logits, target)


class InBatchLogQModel(TwoTower):
    """In-batch softmax with logQ correction: ``r(u, i) - log q(i)``.

    ``q(i)`` is the probability of item ``i`` being a positive (its share of train
    interactions), i.e. the rate at which it is sampled as an in-batch negative.
    """

    def __init__(self, num_items: int, embedding_dim: int, log_q: torch.Tensor) -> None:
        super().__init__(num_items, embedding_dim)
        self.register_buffer("log_q", log_q.float())

    def compute_loss(self, user_repr, inputs):
        labels = inputs["label"]
        logits = user_repr @ self.item_matrix[labels].T - self.log_q[labels].unsqueeze(0)
        return F.cross_entropy(logits, torch.arange(len(user_repr), device=user_repr.device))


def item_log_frequencies(item_ids: np.ndarray, num_items: int, eps: float = 1e-12) -> torch.Tensor:
    counts = np.bincount(item_ids, minlength=num_items).astype(np.float64)
    return torch.from_numpy(np.log(counts / counts.sum() + eps)).float()
