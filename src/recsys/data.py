"""Loading, filtering and temporal train/test split of Yambda likes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import polars as pl

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
WEEK_SECONDS = 7 * 24 * 60 * 60


@dataclass
class Dataset:
    train: pl.DataFrame
    test: pl.DataFrame
    embeddings: pl.DataFrame
    artists: pl.DataFrame
    test_targets: Dict[int, List[int]]

    @property
    def catalog_size(self) -> int:
        """Number of items that can be recommended (items seen in train)."""
        return self.train["item_id"].n_unique()

    def user_histories(self) -> Dict[int, List[int]]:
        """Chronologically ordered train history for every user."""
        return dict(
            self.train.group_by("uid")
            .agg(pl.col("item_id").sort_by("timestamp"))
            .iter_rows()
        )


def load_dataset(
    data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
    min_item_interactions: int = 5,
    test_seconds: int = WEEK_SECONDS,
    reindex_items: bool = False,
) -> Dataset:
    """Read the parquet files and build a leave-last-week-out split.

    Steps:
      1. keep interactions whose items have a pretrained embedding;
      2. keep items with at least ``min_item_interactions`` likes;
      3. left-join artist metadata;
      4. put the last ``test_seconds`` into test, restricted to users seen in train.

    With ``reindex_items=True`` item ids are remapped to ``[0, n_items)`` so they
    can index an ``nn.Embedding`` directly.
    """
    data_dir = Path(data_dir)
    interactions = pl.read_parquet(data_dir / "interactions.parquet")
    embeddings = pl.read_parquet(data_dir / "embeddings.parquet")
    artists = pl.read_parquet(data_dir / "artists.parquet")

    interactions = interactions.join(embeddings, on="item_id", how="semi")
    frequent_items = (
        interactions["item_id"].value_counts()
        .filter(pl.col("count") >= min_item_interactions)
        .select("item_id")
    )
    interactions = interactions.join(frequent_items, on="item_id", how="semi")
    interactions = interactions.join(artists, on="item_id", how="left")
    embeddings = embeddings.join(frequent_items, on="item_id", how="semi")

    if reindex_items:
        mapping = (
            interactions.select("item_id").unique().sort("item_id")
            .with_row_index("item_idx")
            .with_columns(pl.col("item_idx").cast(pl.UInt32))
        )
        interactions = _apply_mapping(interactions, mapping)
        embeddings = _apply_mapping(embeddings, mapping)
        artists = _apply_mapping(artists, mapping)

    split_point = interactions["timestamp"].max() - test_seconds
    train = interactions.filter(pl.col("timestamp") < split_point)
    test = (
        interactions.filter(pl.col("timestamp") >= split_point)
        .join(train.select("uid").unique(), on="uid", how="semi")
    )
    test_targets = dict(test.group_by("uid").agg(pl.col("item_id")).iter_rows())

    return Dataset(train=train, test=test, embeddings=embeddings, artists=artists, test_targets=test_targets)


def _apply_mapping(df: pl.DataFrame, mapping: pl.DataFrame) -> pl.DataFrame:
    columns = df.columns
    return (
        df.join(mapping, on="item_id", how="inner")
        .drop("item_id")
        .rename({"item_idx": "item_id"})
        .select(columns)
    )
