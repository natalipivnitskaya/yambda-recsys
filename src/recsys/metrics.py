"""Top-K retrieval metrics: HitRate, Recall, nDCG and catalog Coverage."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

_DISCOUNTS = 1.0 / np.log2(np.arange(2, 10_002))


def user_metrics(targets: Sequence[int], candidates: Sequence[int], topk: int) -> Dict[str, float]:
    """Metrics for one user.

    Recall and nDCG are normalised by ``min(|targets|, topk)`` so a perfect
    top-K list always scores 1.0. Duplicate targets are counted once.
    """
    relevant = set(targets)
    hits = np.fromiter((c in relevant for c in candidates[:topk]), dtype=np.float64)
    n_ideal = min(len(relevant), topk)
    if n_ideal == 0:
        return {"hitrate": 0.0, "recall": 0.0, "ndcg": 0.0}

    dcg = float(hits @ _DISCOUNTS[: len(hits)])
    idcg = float(_DISCOUNTS[:n_ideal].sum())
    return {
        "hitrate": float(hits.any()),
        "recall": float(hits.sum()) / n_ideal,
        "ndcg": dcg / idcg,
    }


def evaluate(
    targets: Dict[int, List[int]],
    candidates: Dict[int, List[int]],
    catalog_size: int,
    topk: int = 100,
) -> Dict[str, float]:
    """Average per-user metrics over all users in ``targets`` plus Coverage@K.

    Coverage is the share of the catalog that appears in at least one user's top-K.
    """
    totals = {"hitrate": 0.0, "recall": 0.0, "ndcg": 0.0}
    for uid, user_targets in targets.items():
        for name, value in user_metrics(user_targets, candidates[uid], topk).items():
            totals[name] += value

    result = {name: value / len(targets) for name, value in totals.items()}
    recommended = {item for user_cands in candidates.values() for item in user_cands[:topk]}
    result["coverage"] = len(recommended) / catalog_size
    return result
