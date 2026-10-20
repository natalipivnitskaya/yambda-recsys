"""In-memory tabular batches and the training loop for rankers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Sequence

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from tqdm import tqdm

from recsys.ranking.data import DENSE_COLUMNS, LABEL_COLUMNS, MULTIVALENT_COLUMNS, SPARSE_COLUMNS


@dataclass
class TabularData:
    """Column arrays of a listens table; multivalent columns as flat values + offsets."""

    dense: np.ndarray                    # (n, num_dense) float32
    sparse: np.ndarray                   # (n, num_sparse) int64
    labels: np.ndarray                   # (n, num_labels) float32
    multivalent: Dict[str, tuple]        # name -> (values int64, offsets int64 of length n + 1)
    uid: np.ndarray
    timestamp: np.ndarray

    @classmethod
    def from_frame(cls, df: pl.DataFrame) -> "TabularData":
        multivalent = {}
        for col in MULTIVALENT_COLUMNS:
            lengths = df[col].list.len().to_numpy().astype(np.int64)
            offsets = np.concatenate([[0], np.cumsum(lengths)])
            values = df[col].explode().drop_nulls().to_numpy().astype(np.int64)
            multivalent[col.removesuffix("_ids")] = (values, offsets)   # "artist_ids" -> "artist"
        return cls(
            dense=df.select(DENSE_COLUMNS).to_numpy().astype(np.float32),
            sparse=df.select(SPARSE_COLUMNS).to_numpy().astype(np.int64),
            labels=df.select(LABEL_COLUMNS).to_numpy().astype(np.float32),
            multivalent=multivalent,
            uid=df["uid"].to_numpy(),
            timestamp=df["timestamp"].to_numpy(),
        )

    def __len__(self) -> int:
        return len(self.dense)

    def batch(self, idx: np.ndarray) -> Dict[str, torch.Tensor]:
        out = {
            "dense": torch.from_numpy(self.dense[idx]),
            "sparse": torch.from_numpy(self.sparse[idx]),
            "labels": torch.from_numpy(self.labels[idx]),
        }
        for name, (values, offsets) in self.multivalent.items():
            ids, lengths = ragged_take(values, offsets, idx)
            out[f"{name}_ids"], out[f"{name}_lengths"] = torch.from_numpy(ids), torch.from_numpy(lengths)
        return out

    def batches(self, batch_size: int, shuffle: bool, seed: int = 0) -> Iterator[Dict[str, torch.Tensor]]:
        order = np.random.default_rng(seed).permutation(len(self)) if shuffle else np.arange(len(self))
        for start in range(0, len(self), batch_size):
            yield self.batch(order[start:start + batch_size])


def ragged_take(values: np.ndarray, offsets: np.ndarray, idx: np.ndarray):
    """Gather rows ``idx`` of a ragged array; returns ``(flat values, lengths)``."""
    starts, lengths = offsets[idx], offsets[idx + 1] - offsets[idx]
    within = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    return values[np.repeat(starts, lengths) + within], lengths


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def predict(model: torch.nn.Module, data: TabularData, device: torch.device, batch_size: int = 16384) -> np.ndarray:
    """Logits ``(n, num_tasks)`` in input order."""
    model.eval()
    return np.concatenate([
        model(to_device(b, device)).float().cpu().numpy() for b in data.batches(batch_size, shuffle=False)
    ])


def fit(
    model: torch.nn.Module,
    train: TabularData,
    num_epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    task_label_index: Sequence[int],
    on_epoch_end: Optional[Callable[[int], Dict[str, float]]] = None,
    weight_decay: float = 1e-5,
    seed: int = 0,
) -> List[Dict[str, float]]:
    """Minimise the sum of per-task BCE losses; ``task_label_index[k]`` is the label column of head ``k``."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = -(-len(train) // batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=num_epochs * steps_per_epoch, pct_start=0.05
    )
    label_index = torch.tensor(list(task_label_index), device=device)
    history = []
    for epoch in range(1, num_epochs + 1):
        model.train()
        total = 0.0
        progress = tqdm(train.batches(batch_size, shuffle=True, seed=seed + epoch), total=steps_per_epoch,
                        desc=f"epoch {epoch}/{num_epochs}")
        for step, batch in enumerate(progress, start=1):
            batch = to_device(batch, device)
            logits = model(batch)
            loss = F.binary_cross_entropy_with_logits(logits, batch["labels"][:, label_index], reduction="none").mean(0).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item()
            if step % 200 == 0:
                progress.set_postfix(loss=f"{total / step:.4f}")
        log = {"epoch": epoch, "loss": total / steps_per_epoch}
        if on_epoch_end is not None:
            log.update(on_epoch_end(epoch))
        print(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in log.items()))
        history.append(log)
    return history
