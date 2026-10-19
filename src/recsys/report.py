"""Persist leaderboards as CSV (machine-readable) and Markdown (for the README)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

import polars as pl

METRICS = ["hitrate", "recall", "ndcg", "coverage"]
LABELS = {"hitrate": "HitRate", "recall": "Recall", "ndcg": "nDCG", "coverage": "Coverage"}


def save_leaderboard(rows: List[Dict[str, object]], path: Union[str, Path], topk: int = 100) -> pl.DataFrame:
    """Merge ``rows`` into the CSV at ``path`` (same method name is replaced) and write a .md next to it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    board = pl.DataFrame(rows).select(["method", *METRICS])
    if path.exists():
        previous = pl.read_csv(path).filter(~pl.col("method").is_in(board["method"].to_list()))
        board = pl.concat([previous, board], how="vertical_relaxed")
    board = board.sort("recall", descending=True)
    board.write_csv(path)

    header = "| Method | " + " | ".join(f"{LABELS[m]}@{topk}" for m in METRICS) + " |"
    lines = [header, "|---|" + "---:|" * len(METRICS)]
    for row in board.iter_rows(named=True):
        lines.append(f"| {row['method']} | " + " | ".join(f"{row[m]:.4f}" for m in METRICS) + " |")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return board
