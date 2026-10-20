"""Benchmark rankers by pairwise accuracy on neighbouring listens (like and full-play targets).

    python scripts/run_ranking.py --model random popularity catboost
    python scripts/run_ranking.py --model dcn-mix-mt
    python scripts/run_ranking.py --model mlp --train-fraction 0.05     # quick smoke run

Single-task models predict full play and use that score for both metrics.
The multi-task model is reported twice: each head on its own target, and one
joint score log P(full play) + log P(like) used for both targets.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from recsys.item2item import default_device
from recsys.ranking.data import DEFAULT_RANKING_DIR, DENSE_COLUMNS, LABEL_COLUMNS, SPARSE_COLUMNS, load_ranking_data
from recsys.ranking.features import PiecewiseLinearEncoding, UnifiedEmbedding
from recsys.ranking.metrics import pairwise_accuracy
from recsys.ranking.models import FeatureEncoder, Ranker
from recsys.ranking.train import TabularData, fit, predict
from recsys.report import save_leaderboard

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["pair_acc_like", "pair_acc_full_play"]
LIKE, FULL = LABEL_COLUMNS.index("is_like"), LABEL_COLUMNS.index("is_full_play")

NEURAL = {
    # name: (display name, tasks, cross layers, cross experts)
    "mlp": ("Concat + MLP", ["is_full_play"], 0, 1),
    "dcn": ("DCN-v2 + MLP", ["is_full_play"], 3, 1),
    "dcn-mix": ("DCN-v2 (mixture of low-rank experts) + MLP", ["is_full_play"], 3, 4),
    "dcn-mix-mt": ("DCN-v2 mixture, multi-task", ["is_like", "is_full_play"], 3, 4),
}
MODELS = ["random", "popularity", "catboost", *NEURAL]


def scores_row(name, test: TabularData, like_scores, full_scores=None):
    full_scores = like_scores if full_scores is None else full_scores
    return {
        "method": name,
        "pair_acc_like": pairwise_accuracy(test.uid, test.timestamp, test.labels[:, LIKE], like_scores),
        "pair_acc_full_play": pairwise_accuracy(test.uid, test.timestamp, test.labels[:, FULL], full_scores),
    }


def popularity_scores(train_df: pl.DataFrame, test_df: pl.DataFrame) -> np.ndarray:
    counts = train_df.group_by("item_id").len("popularity")
    return test_df.join(counts, on="item_id", how="left")["popularity"].fill_null(0).to_numpy().astype(np.float64)


def catboost_scores(train: TabularData, test: TabularData, max_rows: int, seed: int) -> np.ndarray:
    import pandas as pd
    from catboost import CatBoostClassifier, Pool

    rows = np.random.default_rng(seed).choice(len(train), min(max_rows, len(train)), replace=False)

    def pool(data: TabularData, idx=None, with_label=True):
        idx = np.arange(len(data)) if idx is None else idx
        frame = pd.DataFrame(data.dense[idx], columns=list(DENSE_COLUMNS))
        for i, c in enumerate(SPARSE_COLUMNS):
            frame[c] = data.sparse[idx, i].astype(str)
        return Pool(frame, label=data.labels[idx, FULL] if with_label else None, cat_features=list(SPARSE_COLUMNS))

    model = CatBoostClassifier(iterations=500, depth=8, learning_rate=0.1, random_seed=seed,
                               thread_count=-1, verbose=100, allow_writing_files=False)
    model.fit(pool(train, rows))
    return model.predict_proba(pool(test, with_label=False))[:, 1]


def build_ranker(kind: str, train: TabularData, args) -> Ranker:
    _, tasks, cross_layers, experts = NEURAL[kind]
    numeric = PiecewiseLinearEncoding.from_data(train.dense, n_bins=args.bins, dim=args.numeric_dim, seed=args.seed)
    categorical = UnifiedEmbedding(
        {"uid": 2, "item_id": 4, "artist": 2, "album": 2}, table_size=args.table_size, dim=args.embedding_dim, seed=args.seed
    )
    encoder = FeatureEncoder(numeric, categorical, SPARSE_COLUMNS, ["artist", "album"])
    return Ranker(encoder, tasks, hidden=(512, 256), cross_layers=cross_layers, cross_experts=experts,
                  cross_rank=args.cross_rank, dropout=args.dropout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_RANKING_DIR)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--numeric-dim", type=int, default=8)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--table-size", type=int, default=2 ** 20)
    parser.add_argument("--cross-rank", type=int, default=64)
    parser.add_argument("--catboost-rows", type=int, default=3_000_000)
    parser.add_argument("--train-fraction", type=float, default=1.0, help="subsample train rows for quick runs")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "ranking.csv")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_df, test_df = load_ranking_data(args.data_dir)
    if args.train_fraction < 1.0:
        train_df = train_df.sample(fraction=args.train_fraction, seed=args.seed)
    train, test = TabularData.from_frame(train_df), TabularData.from_frame(test_df)
    print(f"train={len(train):,} test={len(test):,} like_rate={test.labels[:, LIKE].mean():.4f} "
          f"full_play_rate={test.labels[:, FULL].mean():.4f} device={device}")

    rows = []
    for kind in args.model:
        start = time.time()
        if kind == "random":
            rows.append(scores_row("Random", test, np.random.default_rng(args.seed).random(len(test))))
        elif kind == "popularity":
            rows.append(scores_row("Popularity (train listens)", test, popularity_scores(train_df, test_df)))
        elif kind == "catboost":
            rows.append(scores_row("CatBoost (full play)", test, catboost_scores(train, test, args.catboost_rows, args.seed)))
        else:
            name, tasks = NEURAL[kind][:2]
            model = build_ranker(kind, train, args).to(device)
            print(f"{name}: {sum(p.numel() for p in model.parameters()):,} parameters")
            label_index = [LABEL_COLUMNS.index(t) for t in tasks]

            def evaluate(epoch, model=model, tasks=tasks, name=name):
                logits = predict(model, test, device)
                if len(tasks) == 1:
                    row = scores_row(name, test, logits[:, 0])
                else:
                    row = scores_row(name, test, logits[:, tasks.index("is_like")], logits[:, tasks.index("is_full_play")])
                return {m: row[m] for m in METRICS}

            fit(model, train, args.epochs, args.batch_size, args.lr, device, label_index, on_epoch_end=evaluate, seed=args.seed)
            logits = predict(model, test, device)
            if len(tasks) == 1:
                rows.append(scores_row(name, test, logits[:, 0]))
            else:
                like, full = logits[:, tasks.index("is_like")], logits[:, tasks.index("is_full_play")]
                rows.append(scores_row(f"{name} (per-task heads)", test, like, full))
                joint = (F.logsigmoid(torch.from_numpy(like)) + F.logsigmoid(torch.from_numpy(full))).numpy()
                rows.append(scores_row(f"{name} (joint score)", test, joint))
        print(f"{rows[-1]}  ({time.time() - start:.0f}s)")

    if args.train_fraction == 1.0:
        save_leaderboard(rows, args.output, topk=None, metrics=METRICS, sort_by="pair_acc_like")


if __name__ == "__main__":
    main()
