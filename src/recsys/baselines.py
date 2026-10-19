"""Count-based recommenders: global popularity and favourite artists."""
from __future__ import annotations

from typing import Dict, Iterable, List

import polars as pl


def top_popular(train: pl.DataFrame, topk: int) -> List[int]:
    """Most-liked items in train."""
    return (
        train["item_id"].value_counts()
        .sort(["count", "item_id"], descending=[True, False])
        .head(topk)["item_id"].to_list()
    )


def fill_with_popular(
    candidates: Dict[int, List[int]],
    popular: List[int],
    topk: int,
    users: Iterable[int] = (),
) -> Dict[int, List[int]]:
    """Pad every user's list up to ``topk`` with popular items not already in it.

    Users listed in ``users`` but missing from ``candidates`` get the popular list.
    """
    result = {}
    for uid in set(candidates) | set(users):
        items = candidates.get(uid, [])[:topk]
        if len(items) < topk:
            seen = set(items)
            items = items + [i for i in popular if i not in seen][: topk - len(items)]
        result[uid] = items
    return result


def artist_recommendations(
    train: pl.DataFrame,
    artists: pl.DataFrame,
    n_last_events: int = 100,
    tracks_per_artist: int = 20,
) -> Dict[int, List[int]]:
    """Recommend popular tracks of the artists a user liked most recently.

    Artists are ranked by the number of likes among the user's last
    ``n_last_events``; each contributes its ``tracks_per_artist`` most popular
    tracks, minus the ones the user already liked. The per-artist cut is taken
    *before* removing seen tracks, so artists the user has exhausted are skipped.
    """
    recent = train.sort("timestamp").group_by("uid").tail(n_last_events)
    artist_affinity = recent.group_by(["uid", "artist_id"]).len(name="user_likes")

    artist_top_tracks = (
        train.group_by("item_id").len(name="popularity")
        .join(artists, on="item_id", how="inner")
        .sort("popularity", descending=True)
        .group_by("artist_id")
        .head(tracks_per_artist)
    )

    ranked = (
        artist_affinity
        .join(artist_top_tracks, on="artist_id", how="inner")
        .join(train.select(["uid", "item_id"]), on=["uid", "item_id"], how="anti")
        .sort(["uid", "user_likes", "popularity"], descending=[False, True, True])
        .group_by("uid", maintain_order=True)
        .agg(pl.col("item_id"))
    )
    return dict(ranked.iter_rows())
