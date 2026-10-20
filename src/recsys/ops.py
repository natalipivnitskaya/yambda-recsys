"""Tensor helpers for ragged (flattened + lengths) batches."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def pad_to_bucket(values: torch.Tensor, bucket: int) -> torch.Tensor:
    """Right-pad a 1-D tensor with zeros to a multiple of ``bucket``.

    Keeps the number of distinct shapes small: the MPS backend compiles and caches
    a graph per input shape, so per-batch lengths would grow memory without bound.
    """
    padded = -(-len(values) // bucket) * bucket
    return F.pad(values, (0, padded - len(values)))


def segment_sum(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Sum consecutive rows of ``values`` into ``len(lengths)`` segments.

    The first ``lengths.sum()`` rows belong to the segments in order; any rows
    after that (bucket padding) are ignored.
    """
    positions = torch.arange(len(values), device=values.device)
    # searchsorted instead of repeat_interleave, which is broken on MPS
    segment = torch.searchsorted(torch.cumsum(lengths, 0), positions, right=True)
    out = values.new_zeros((len(lengths) + 1, values.shape[1]))
    return out.index_add_(0, segment, values)[: len(lengths)]


def segment_mean(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Like :func:`segment_sum` but averaged; empty segments give zero vectors."""
    return segment_sum(values, lengths) / lengths.clamp(min=1).unsqueeze(1).to(values.dtype)
