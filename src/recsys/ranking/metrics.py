"""Pairwise accuracy on neighbouring listens of the same user."""
from __future__ import annotations

import numpy as np

SESSION_GAP_SECONDS = 15 * 60


def pairwise_accuracy(
    uids: np.ndarray,
    timestamps: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    session_gap: int = SESSION_GAP_SECONDS,
) -> float:
    """Share of adjacent same-user pairs with different labels that the scores order correctly.

    Events are ordered by ``(uid, timestamp)``, keeping input order for ties. A pair
    counts if both events are at most ``session_gap`` seconds apart and exactly one
    is positive; it is correct when the positive event scores higher. Ties count 0.5,
    so a constant or random scorer gets 0.5.
    """
    order = np.lexsort((np.asarray(timestamps), np.asarray(uids)))
    u = np.asarray(uids)[order]
    t = np.asarray(timestamps, dtype=np.int64)[order]
    y = np.asarray(labels).astype(bool)[order]
    s = np.asarray(scores, dtype=np.float64)[order]

    valid = (u[1:] == u[:-1]) & (t[1:] - t[:-1] <= session_gap) & (y[1:] != y[:-1])
    if not valid.any():
        return float("nan")
    margin = np.where(y[1:], s[1:] - s[:-1], s[:-1] - s[1:])[valid]
    return float(np.mean((margin > 0) + 0.5 * (margin == 0)))
