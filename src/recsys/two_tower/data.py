"""Next-item prediction samples and flattened batches.

Batches store all histories concatenated into one 1-D tensor plus a ``length``
tensor with each user's history size, which avoids padding:

    item_id = [u1_t1, ..., u1_tL1, u2_t1, ..., u2_tL2, ...]
    length  = [L1, L2, ...]
"""
from __future__ import annotations

from typing import Any, Dict, Hashable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def pad_sequences(data: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Turn a flattened batch into a right-padded tensor and a validity mask.

    ``data`` has shape ``(sum(lengths), *feature_dims)``; the result has shape
    ``(batch, max_len, *feature_dims)`` and the mask ``(batch, max_len)``.
    """
    max_len = int(lengths.max()) if lengths.numel() else 0
    mask = torch.arange(max_len, device=lengths.device) < lengths.unsqueeze(1)
    padded = data.new_zeros((len(lengths), max_len) + tuple(data.shape[1:]))
    padded[mask] = data
    return padded, mask


class SequenceDataset(Dataset):
    """User histories served as next-item prediction samples.

    Train mode turns a history ``[i1, ..., iT]`` into ``T - 1`` samples
    ``(history[:t] -> history[t])``. Only ``(user, t)`` pointers are stored;
    prefixes are sliced lazily and truncated to the last ``max_seq_len`` items.

    Eval mode yields one sample per user that has targets in ``labels``,
    containing the (truncated) full history.
    """

    def __init__(
        self,
        histories: Dict[Hashable, List[int]],
        labels: Optional[Dict[Hashable, List[int]]] = None,
        is_train: bool = True,
        max_seq_len: int = 100,
    ) -> None:
        self.histories = histories
        self.labels = labels or {}
        self.is_train = is_train
        self.max_seq_len = max_seq_len

        if is_train:
            self.uids = list(histories)
            sizes = np.array([max(len(histories[u]) - 1, 0) for u in self.uids], dtype=np.int64)
            self._user = np.repeat(np.arange(len(self.uids), dtype=np.int32), sizes)
            starts = np.cumsum(sizes) - sizes
            self._position = (np.arange(sizes.sum()) - np.repeat(starts, sizes) + 1).astype(np.int32)
        else:
            self.uids = [u for u in self.labels if u in histories]
            self._user = np.arange(len(self.uids), dtype=np.int32)
            self._position = np.array([len(histories[u]) for u in self.uids], dtype=np.int32)

    def __len__(self) -> int:
        return len(self._user)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        uid = self.uids[self._user[idx]]
        t = int(self._position[idx])
        history = self.histories[uid]
        prefix = list(history[max(0, t - self.max_seq_len):t])

        sample = {"uid": uid, "history": {"item_id": prefix, "length": len(prefix)}}
        if self.is_train:
            sample["label"] = int(history[t])
        return sample


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Concatenate histories into the flattened batch format (all tensors ``long``)."""
    result = {
        "uid": torch.tensor([s["uid"] for s in batch], dtype=torch.long),
        "history": {
            "item_id": torch.tensor([i for s in batch for i in s["history"]["item_id"]], dtype=torch.long),
            "length": torch.tensor([s["history"]["length"] for s in batch], dtype=torch.long),
        },
    }
    if batch and "label" in batch[0]:
        result["label"] = torch.tensor([s["label"] for s in batch], dtype=torch.long)
    return result
