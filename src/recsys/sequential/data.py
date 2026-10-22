"""Chunked next-item training samples for causal sequence encoders.

A causal encoder produces a state for every prefix of the history in one pass,
so one sample covers a whole window of next-item steps instead of one:

    history = [i1, ..., iL],  targets = [i2, ..., iL+1]

Histories longer than ``max_seq_len`` steps are split into non-overlapping
windows. Evaluation reuses :class:`recsys.two_tower.SequenceDataset` (the last
``max_seq_len`` items of each test user).
"""
from __future__ import annotations

from typing import Any, Dict, Hashable, List

import numpy as np
import torch
from torch.utils.data import Dataset

from recsys.two_tower.data import collate_fn as _collate_histories


class ChunkedSequenceDataset(Dataset):
    """One sample per window of up to ``max_seq_len`` next-item steps.

    Only ``(user, window start)`` pointers are stored; windows are sliced lazily.
    """

    def __init__(self, histories: Dict[Hashable, List[int]], max_seq_len: int = 100) -> None:
        self.histories = histories
        self.max_seq_len = max_seq_len
        self.uids = list(histories)
        steps = np.array([max(len(histories[u]) - 1, 0) for u in self.uids], dtype=np.int64)
        windows = -(-steps // max_seq_len)
        self._user = np.repeat(np.arange(len(self.uids), dtype=np.int32), windows)
        first = np.cumsum(windows) - windows
        self._start = ((np.arange(windows.sum()) - np.repeat(first, windows)) * max_seq_len).astype(np.int64)

    def __len__(self) -> int:
        return len(self._user)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        uid = self.uids[self._user[idx]]
        start = int(self._start[idx])
        history = self.histories[uid]
        end = min(start + self.max_seq_len, len(history) - 1)
        return {
            "uid": uid,
            "history": {"item_id": list(history[start:end]), "length": end - start},
            "targets": list(history[start + 1:end + 1]),
        }

    def target_items(self) -> np.ndarray:
        """Every train target (all history items except each user's first), for logQ."""
        return np.concatenate([np.asarray(h[1:], dtype=np.int64) for h in self.histories.values()])


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Flattened batch as in two-tower, plus ``targets`` aligned with ``history.item_id``."""
    result = _collate_histories(batch)
    if batch and "targets" in batch[0]:
        result["targets"] = torch.tensor([i for s in batch for i in s["targets"]], dtype=torch.long)
    return result
