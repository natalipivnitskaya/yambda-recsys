"""Benchmark count-based, item-to-item and matrix-factorisation recommenders.

    python scripts/run_classic.py                      # all methods
    python scripts/run_classic.py --methods toppop als  # a subset
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np

from recsys.baselines import artist_recommendations, fill_with_popular, top_popular
from recsys.data import DEFAULT_DATA_DIR, load_dataset
from recsys.item2item import (
    aggregate_neighbours,
    cooccurrence_neighbours,
    embedding_neighbours,
    item2vec_embeddings,
)
from recsys.matrix import InteractionMatrix, als_recommendations
from recsys.metrics import evaluate
from recsys.report import save_leaderboard

METHODS = ["toppop", "artist", "i2i-content", "item2vec", "cf", "cf-tfidf", "als", "als-tfidf"]
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "classic.csv")
    args = parser.parse_args()
    np.random.seed(42)
    topk = args.topk

    ds = load_dataset(args.data_dir)
    users = list(ds.test_targets)
    print(f"train={ds.train.height:,} test={ds.test.height:,} test_users={len(users):,} catalog={ds.catalog_size:,}")
    popular = top_popular(ds.train, topk)

    def i2i(neighbours, **kwargs):
        return aggregate_neighbours(ds.train, neighbours, users=users, topk=topk, **kwargs)

    def build(method):
        if method == "toppop":
            return "TopPop", {uid: popular for uid in users}
        if method == "artist":
            return "Favourite artists", artist_recommendations(ds.train, ds.artists)
        if method == "i2i-content":
            emb = ds.embeddings.join(ds.train.select("item_id").unique(), on="item_id", how="semi")
            neighbours = embedding_neighbours(emb["item_id"].to_numpy(), np.vstack(emb["embed"].to_numpy()), 2 * topk)
            return "Item2Item (content embeddings)", i2i(neighbours, n_last=50, half_life=25)
        if method == "item2vec":
            item_ids, vectors = item2vec_embeddings(ds.train)
            return "Item2Vec", i2i(embedding_neighbours(item_ids, vectors, 2 * topk), n_last=50, half_life=25)
        if method in ("cf", "cf-tfidf", "als", "als-tfidf"):
            matrix = InteractionMatrix.from_interactions(ds.train)
            weighting = "TF-IDF" if method.endswith("tfidf") else "raw counts"
            if method.endswith("tfidf"):
                matrix = matrix.tfidf()
            if method.startswith("cf"):
                neighbours = cooccurrence_neighbours(matrix.matrix, matrix.item_ids, topk=4 * topk)
                return f"Item-based CF ({weighting})", i2i(neighbours, n_last=80, half_life=20)
            return f"ALS ({weighting})", als_recommendations(matrix, users, topk=topk)
        raise ValueError(method)

    rows = []
    for method in args.methods:
        start = time.time()
        name, candidates = build(method)
        candidates = fill_with_popular(candidates, popular, topk, users=users)
        metrics = evaluate(ds.test_targets, candidates, ds.catalog_size, topk)
        print(f"{name}: {metrics}  ({time.time() - start:.0f}s)")
        rows.append({"method": name, **metrics})
        gc.collect()

    save_leaderboard(rows, args.output, topk)


if __name__ == "__main__":
    main()
