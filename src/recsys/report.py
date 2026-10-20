"""Persist leaderboards as CSV (machine-readable) and Markdown (for the README)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import polars as pl

METRICS = ["hitrate", "recall", "ndcg", "coverage"]
LABELS = {
    "hitrate": "HitRate", "recall": "Recall", "ndcg": "nDCG", "coverage": "Coverage",
    "pair_acc_like": "PairAcc (like)", "pair_acc_full_play": "PairAcc (full play)",
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
    header = "| Method | " + " | ".join(f"{LABELS.get(m, m)}{suffix}" for m in metrics) + " |"
    lines = [header, "|---|" + "---:|" * len(metrics)]
    for row in board.iter_rows(named=True):
        lines.append(f"| {row['method']} | " + " | ".join(f"{row[m]:.4f}" for m in metrics) + " |")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return board
