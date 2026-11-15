"""Persist leaderboards as CSV (machine-readable) and Markdown (for the README)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import polars as pl

METRICS = ["hitrate", "recall", "ndcg", "coverage"]
LABELS = {
    "hitrate": "HitRate", "recall": "Recall", "ndcg": "nDCG", "coverage": "Coverage",
    "best_epoch": "Best epoch", "pair_acc_like": "PairAcc (like)", "pair_acc_full_play": "PairAcc (full play)",
}


def save_leaderboard(
    rows: List[Dict[str, object]],
    path: Union[str, Path],
    topk: Optional[int] = 100,
    metrics: Sequence[str] = METRICS,
    sort_by: Optional[str] = None,
) -> pl.DataFrame:
    """Merge ``rows`` into the CSV at ``path`` (same method name is replaced) and write a .md next to it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = list(metrics)
    board = pl.DataFrame(rows).select(["method", *metrics])
    if path.exists():
        previous = pl.read_csv(path).filter(~pl.col("method").is_in(board["method"].to_list()))
        board = pl.concat([previous, board], how="vertical_relaxed")
    board = board.sort(sort_by or ("recall" if "recall" in metrics else metrics[0]), descending=True)
    board.write_csv(path)

    suffix = f"@{topk}" if topk else ""
    header = "| Method | " + " | ".join(f"{LABELS.get(m, m)}{suffix if m in METRICS else ''}" for m in metrics) + " |"
    lines = [header, "|---|" + "---:|" * len(metrics)]
    for row in board.iter_rows(named=True):
        lines.append(f"| {row['method']} | " + " | ".join(_fmt(row[m]) for m in metrics) + " |")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return board


def _fmt(value: object) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def save_best_epoch(method: str, history: List[Dict[str, float]], path: Union[str, Path],
                    topk: int = 100, metric: str = "recall") -> Dict[str, object]:
    """Record the epoch with the best ``metric`` from a validation run, as a leaderboard row."""
    best = max(history, key=lambda log: log[metric])
    row = {"method": method, "best_epoch": int(best["epoch"]), **{m: best[m] for m in METRICS}}
    print(f"best epoch by validation {metric}: {row['best_epoch']}")
    save_leaderboard([row], path, topk, metrics=["best_epoch", *METRICS], sort_by=metric)
    return row
