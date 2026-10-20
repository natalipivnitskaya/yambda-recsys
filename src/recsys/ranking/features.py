"""Feature encoders for the neural ranker.

* Numeric counters: piecewise-linear encoding over quantile bins
  (Gorishniy et al., "On Embeddings for Numerical Features in Tabular Deep Learning", 2022).
* Categorical ids and id lists: one shared hashed table looked up with several
  hash functions per feature (Coleman et al., "Unified Embedding", 2023).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from recsys.ops import pad_to_bucket, segment_mean


class PiecewiseLinearEncoding(nn.Module):
    """Encode each numeric feature as its position inside quantile bins, then embed it.

    For bin ``t`` with edges ``[b_{t-1}, b_t)`` the code is ``clip((x - b_{t-1}) / (b_t - b_{t-1}), 0, 1)``:
    1 for bins fully below ``x``, a fraction for the bin containing ``x``, 0 above.
    A per-feature linear layer maps the codes to ``dim`` values, followed by ReLU.
    """

    def __init__(self, edges: List[np.ndarray], dim: int) -> None:
        super().__init__()
        num_features, num_bins = len(edges), max(len(e) - 1 for e in edges)
        left = torch.zeros(num_features, num_bins)
        width = torch.ones(num_features, num_bins)
        mask = torch.zeros(num_features, num_bins)
        for f, e in enumerate(edges):
            n = len(e) - 1
            left[f, :n] = torch.as_tensor(e[:-1], dtype=torch.float32)
            width[f, :n] = torch.as_tensor(np.diff(e), dtype=torch.float32)
            mask[f, :n] = 1.0
        self.register_buffer("left", left)
        self.register_buffer("width", width)
        self.register_buffer("mask", mask)
        self.weight = nn.Parameter(torch.randn(num_features, num_bins, dim) / num_bins ** 0.5)
        self.bias = nn.Parameter(torch.zeros(num_features, dim))

    @classmethod
    def from_data(cls, x: np.ndarray, n_bins: int, dim: int, max_rows: int = 2_000_000, seed: int = 0):
        """Fit quantile edges on (a sample of) ``x`` with shape ``(rows, features)``."""
        if len(x) > max_rows:
            x = x[np.random.default_rng(seed).choice(len(x), max_rows, replace=False)]
        edges = []
        for column in x.T:
            # "lower" keeps edges on observed values, so discrete counters get one bin per distinct step
            e = np.unique(np.quantile(column, np.linspace(0, 1, n_bins + 1), method="lower"))
            edges.append(e if len(e) > 1 else np.array([e[0], e[0] + 1.0]))
        return cls(edges, dim)

    @property
    def n_bins(self) -> torch.Tensor:
        return self.mask.sum(dim=1).long()

    @property
    def out_dim(self) -> int:
        return self.weight.shape[0] * self.weight.shape[2]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """``(batch, features)`` -> piecewise-linear codes ``(batch, features, bins)``."""
        return ((x.unsqueeze(-1) - self.left) / self.width).clamp(0, 1) * self.mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.einsum("bft,ftd->bfd", self.encode(x), self.weight) + self.bias
        return torch.relu(out).flatten(1)


class MultiHash(nn.Module):
    """``num_hashes`` independent universal hashes ``((a * x + b) mod p) mod table_size``."""

    PRIME = 2 ** 31 - 1  # keeps a * x within int64

    def __init__(self, num_hashes: int, table_size: int, seed: int) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("a", torch.randint(1, self.PRIME, (num_hashes,), generator=g))
        self.register_buffer("b", torch.randint(0, self.PRIME, (num_hashes,), generator=g))
        self.table_size = table_size

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """``(...,)`` integer ids -> ``(..., num_hashes)`` row indices."""
        x = (ids.long() % self.PRIME).unsqueeze(-1)
        return ((x * self.a + self.b) % self.PRIME) % self.table_size


class UnifiedEmbedding(nn.Module):
    """One hashed embedding table shared by all categorical features.

    Feature ``f`` is looked up with its own ``num_hashes[f]`` hash functions and the
    vectors are concatenated, so features get differently sized representations
    (``num_hashes[f] * dim``) while sharing parameters. Collisions of one hash are
    disambiguated by the others.
    """

    def __init__(self, num_hashes: Dict[str, int], table_size: int, dim: int, bucket: int = 8192, seed: int = 0) -> None:
        super().__init__()
        self.table = nn.Embedding(table_size, dim)
        nn.init.normal_(self.table.weight, std=0.02)
        self.hashes = nn.ModuleDict({
            name: MultiHash(n, table_size, seed=seed + 1000 * i) for i, (name, n) in enumerate(num_hashes.items())
        })
        self.dims = {name: n * dim for name, n in num_hashes.items()}
        self.bucket = bucket

    def lookup(self, name: str, ids: torch.Tensor) -> torch.Tensor:
        """Single-valued feature: ``(batch,)`` -> ``(batch, num_hashes * dim)``."""
        return self.table(self.hashes[name](ids)).flatten(1)

    def lookup_bag(self, name: str, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Multivalent feature (flattened ids + lengths): mean of the list's embeddings."""
        embeddings = self.table(self.hashes[name](pad_to_bucket(ids, self.bucket))).flatten(1)
        return segment_mean(embeddings, lengths)
