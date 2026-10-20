"""Listens with lagged counters: metadata join, temporal split, preference filtering.

Source: ``matfu21/yambda-50m-lag-features`` (organic-free Yambda-50M listens). Every
counter is computed at ``timestamp - 15 min`` so a listen never sees its own or its
neighbours' feedback. ``is_like`` marks listens a like was attributed to (nearest
listen of the same track within 24 h); ``is_full_play`` means >= 95% played.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import polars as pl

from recsys.data import DEFAULT_DATA_DIR

DEFAULT_RANKING_DIR = DEFAULT_DATA_DIR / "ranking"
DAY_SECONDS = 24 * 60 * 60

DENSE_COLUMNS = tuple(
    f"{scope}_lag_{stat}"
    for scope in ("user", "item", "ui")
    for stat in ("listen_cnt", "like_cnt", "full_play_cnt", "skip_cnt", "avg_played_ratio")
)
SPARSE_COLUMNS = ("uid", "item_id")
MULTIVALENT_COLUMNS = ("artist_ids", "album_ids")
LABEL_COLUMNS = ("is_like", "is_full_play")


def load_ranking_data(
    data_dir: Union[str, Path] = DEFAULT_RANKING_DIR,
    test_days: int = 30,
    cache: bool = True,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Return ``(train, test)``: preference events, the last ``test_days`` in test.

    The prepared split is cached as parquet next to the raw files.
    """
    data_dir = Path(data_dir)
    cache_dir = data_dir / f"prepared_{test_days}d"
    if cache and (cache_dir / "test.parquet").exists():
        return pl.read_parquet(cache_dir / "train.parquet"), pl.read_parquet(cache_dir / "test.parquet")

    listens = pl.read_parquet(data_dir / "listens.parquet").with_row_index("row_id")
    listens = add_item_metadata(
        listens,
        pl.read_parquet(data_dir / "artist_item_mapping.parquet"),
        pl.read_parquet(data_dir / "album_item_mapping.parquet"),
    )
    train, test = temporal_split(listens, test_days * DAY_SECONDS)
    train, test = preference_events(train), preference_events(test)

    if cache:
        cache_dir.mkdir(exist_ok=True)
        train.write_parquet(cache_dir / "train.parquet")
        test.write_parquet(cache_dir / "test.parquet")
    return train, test


def add_item_metadata(listens: pl.DataFrame, artists: pl.DataFrame, albums: pl.DataFrame) -> pl.DataFrame:
    """Attach ``artist_ids`` / ``album_ids`` lists per track (empty list when unknown)."""
    for mapping, id_col, list_col in ((artists, "artist_id", "artist_ids"), (albums, "album_id", "album_ids")):
        lists = mapping.group_by("item_id").agg(pl.col(id_col).unique().sort().alias(list_col))
        listens = listens.join(lists, on="item_id", how="left").with_columns(
            pl.col(list_col).fill_null(pl.lit([], dtype=pl.List(pl.UInt32)))
        )
    return listens


def temporal_split(df: pl.DataFrame, test_seconds: int) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Events in the last ``test_seconds`` (boundary included) go to test."""
    split_point = df["timestamp"].max() - test_seconds
    return df.filter(pl.col("timestamp") < split_point), df.filter(pl.col("timestamp") >= split_point)


def preference_events(df: pl.DataFrame) -> pl.DataFrame:
    """Keep listens whose ``is_like`` or ``is_full_play`` differs from the previous or next listen of the user.

    Events with identical feedback on both sides carry no ranking signal for a
    small model. Applying this to test does not change pairwise accuracy: a removed
    event has the same labels as both neighbours, so any newly adjacent pair has
    equal labels and is skipped by the metric.
    """
    differs = pl.lit(False)
    for label in LABEL_COLUMNS:
        for shift in (1, -1):
            neighbour = pl.col(label).shift(shift).over("uid")
            differs = differs | (neighbour.is_not_null() & (pl.col(label) != neighbour))
    return df.sort(["uid", "timestamp", "row_id"]).filter(differs)
