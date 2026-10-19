"""Sparse user-item matrix and ALS matrix factorisation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix


@dataclass
class InteractionMatrix:
    matrix: csr_matrix          # users x items, values = number of likes
    user_ids: np.ndarray        # row index -> uid
    item_ids: np.ndarray        # column index -> item_id

    @classmethod
    def from_interactions(cls, train: pl.DataFrame) -> "InteractionMatrix":
        users = train.select("uid").unique().sort("uid").with_row_index("row")
        items = train.select("item_id").unique().sort("item_id").with_row_index("col")
        indexed = train.join(users, on="uid").join(items, on="item_id")
        matrix = csr_matrix(
            (
                np.ones(indexed.height, dtype=np.float32),
                (indexed["row"].to_numpy(), indexed["col"].to_numpy()),
            ),
            shape=(users.height, items.height),
        )
        matrix.sum_duplicates()
        return cls(matrix, users["uid"].to_numpy(), items["item_id"].to_numpy())

    def tfidf(self) -> "InteractionMatrix":
        """Down-weight popular items: ``sqrt(count) * idf(item)``."""
        from implicit.nearest_neighbours import tfidf_weight

        return InteractionMatrix(tfidf_weight(self.matrix).tocsr().astype(np.float32), self.user_ids, self.item_ids)


def als_recommendations(
    interactions: InteractionMatrix,
    users: Iterable[int],
    factors: int = 128,
    regularization: float = 0.01,
    alpha: float = 1.0,
    iterations: int = 30,
    topk: int = 100,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """Fit implicit-feedback ALS and return top-``topk`` unseen items per user."""
    from implicit.als import AlternatingLeastSquares

    model = AlternatingLeastSquares(
        factors=factors, regularization=regularization, alpha=alpha,
        iterations=iterations, random_state=seed, dtype=np.float32,
    )
    model.fit(interactions.matrix, show_progress=True)

    row_of = {int(uid): row for row, uid in enumerate(interactions.user_ids.tolist())}
    rows = np.array([row_of[int(uid)] for uid in users if int(uid) in row_of], dtype=np.int64)
    item_idx, _ = model.recommend(rows, interactions.matrix[rows], N=topk, filter_already_liked_items=True)
    return {
        int(interactions.user_ids[row]): interactions.item_ids[recs].tolist()
        for row, recs in zip(rows, item_idx)
    }
