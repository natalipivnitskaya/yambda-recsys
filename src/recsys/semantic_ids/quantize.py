"""Semantic IDs: residual k-means over item embeddings.

Level 1 clusters the embeddings, level 2 clusters what level 1 left unexplained
(the residuals), and so on. An item's code ``(c1, c2, ..., cL)`` is coarse to
fine, so items with a shared prefix are similar. Items that end up with the
same code get one extra level, ``0, 1, 2, ...``, so that every code is unique.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch


def kmeans(x: torch.Tensor, k: int, iters: int = 30, seed: int = 0, chunk: int = 32768) -> torch.Tensor:
    """Lloyd's k-means with k-means++ init; empty clusters are reseeded. Returns ``(k, dim)`` centroids."""
    generator = torch.Generator().manual_seed(seed)
    centroids = kmeans_plus_plus(x, k, generator)
    for _ in range(iters):
        assign = assign_clusters(x, centroids, chunk)
        sums = torch.zeros_like(centroids).index_add_(0, assign, x)
        counts = torch.bincount(assign, minlength=k)
        empty = counts == 0
        centroids = sums / counts.clamp(min=1).unsqueeze(1).to(x.dtype)
        if empty.any():
            reseed = torch.randint(0, len(x), (int(empty.sum()),), generator=generator).to(x.device)
            centroids[empty] = x[reseed]
    return centroids


def kmeans_plus_plus(x: torch.Tensor, k: int, generator: torch.Generator) -> torch.Tensor:
    """Pick each next centroid with probability proportional to its squared distance to the chosen ones."""
    first = int(torch.randint(0, len(x), (1,), generator=generator))
    centroids = [x[first]]
    dist = ((x - x[first]) ** 2).sum(1)
    for _ in range(k - 1):
        weights = dist.cpu().double()
        nxt = int(torch.multinomial(weights / weights.sum(), 1, generator=generator)) if weights.sum() > 0 else 0
        centroids.append(x[nxt])
        dist = torch.minimum(dist, ((x - x[nxt]) ** 2).sum(1))
    return torch.stack(centroids)


def assign_clusters(x: torch.Tensor, centroids: torch.Tensor, chunk: int = 32768) -> torch.Tensor:
    """Index of the nearest centroid (squared Euclidean) for every row of ``x``."""
    c_norm = (centroids ** 2).sum(1)
    return torch.cat([
        (c_norm - 2 * block @ centroids.T).argmin(1) for block in x.split(chunk)
    ])


@dataclass
class SemanticIds:
    codes: torch.Tensor          # (num_items, levels + 1), long; the last column deduplicates
    codebook_sizes: List[int]    # number of distinct codes per level, including the dedup level
    reconstruction_error: List[float]  # mean squared residual norm after each level

    @property
    def num_levels(self) -> int:
        return self.codes.shape[1]


def residual_kmeans(
    embeddings: torch.Tensor,
    levels: int = 3,
    codebook_size: int = 256,
    iters: int = 30,
    normalize: bool = True,
    seed: int = 0,
) -> SemanticIds:
    """Quantize ``(num_items, dim)`` embeddings into unique semantic IDs (row ``i`` = item ``i``)."""
    x = embeddings.float()
    if normalize:
        x = torch.nn.functional.normalize(x, dim=1)
    residual = x.clone()
    codes, errors = [], []
    for level in range(levels):
        centroids = kmeans(residual, codebook_size, iters=iters, seed=seed + level)
        assign = assign_clusters(residual, centroids)
        residual = residual - centroids[assign]
        codes.append(assign)
        errors.append(float((residual ** 2).sum(1).mean()))

    codes = torch.stack(codes, dim=1).cpu()
    dedup = torch.from_numpy(duplicate_rank(codes.numpy()))
    codes = torch.cat([codes, dedup.unsqueeze(1)], dim=1)
    sizes = [codebook_size] * levels + [int(dedup.max()) + 1]
    return SemanticIds(codes=codes, codebook_sizes=sizes, reconstruction_error=errors)


def duplicate_rank(codes: np.ndarray) -> np.ndarray:
    """For each row, how many earlier rows have exactly the same code (0 for the first one)."""
    _, inverse = np.unique(codes, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    order = np.argsort(inverse, kind="stable")
    sorted_groups = inverse[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_groups)) + 1]
    group_start = np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    rank = np.empty(len(codes), dtype=np.int64)
    rank[order] = np.arange(len(order)) - group_start
    return rank


def code_stats(ids: SemanticIds) -> dict:
    """Prefix sharing and collision statistics for logging."""
    codes = ids.codes.numpy()
    base = codes[:, :-1]
    _, group_sizes = np.unique(base, axis=0, return_counts=True)
    stats = {
        "items": len(codes),
        "unique_codes_before_dedup": len(group_sizes),
        "items_sharing_a_code": float((group_sizes[group_sizes > 1]).sum() / len(codes)),
        "max_items_per_code": int(group_sizes.max()),
        "dedup_codebook_size": ids.codebook_sizes[-1],
    }
    for level in range(1, base.shape[1] + 1):
        stats[f"distinct_prefixes_{level}"] = len(np.unique(base[:, :level], axis=0))
    return stats
