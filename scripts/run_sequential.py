"""Train the causal Transformer user tower on next-item prediction and log metrics.

    python scripts/run_sequential.py                              # 2 layers, in-batch + logQ
    python scripts/run_sequential.py --no-logq
    python scripts/run_sequential.py --epochs 1 --user-fraction 0.05   # quick smoke run
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from recsys.data import DEFAULT_DATA_DIR, load_dataset
from recsys.item2item import default_device
from recsys.report import save_leaderboard
from recsys.sequential import ChunkedSequenceDataset, NextItemTransformer, collate_fn
from recsys.two_tower import SequenceDataset, fit, item_log_frequencies

ROOT = Path(__file__).resolve().parents[1]


def method_name(args: argparse.Namespace) -> str:
    negatives = "in-batch negatives + logQ" if args.logq else "in-batch negatives"
    return (f"Transformer ({args.num_layers} layers, d={args.embedding_dim}, len {args.max_seq_len}, "
            f"{args.epochs} ep), {negatives}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128, help="windows (not events) per batch")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-negatives", type=int, default=512)
    parser.add_argument("--no-logq", dest="logq", action="store_false")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-seq-len", type=int, default=100)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--user-fraction", type=float, default=1.0, help="subsample users for quick runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "sequential.csv")
    parser.add_argument("--checkpoint", type=Path, help="where to save the trained state dict")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    ds = load_dataset(args.data_dir, reindex_items=True)
    num_items = ds.embeddings.height  # after reindexing, ids are exactly [0, num_items)
    histories = ds.user_histories()
    targets = ds.test_targets
    if args.user_fraction < 1.0:
        rng = np.random.default_rng(args.seed)
        keep = set(rng.choice(list(histories), size=int(len(histories) * args.user_fraction), replace=False).tolist())
        histories = {u: h for u, h in histories.items() if u in keep}
        targets = {u: t for u, t in targets.items() if u in keep}

    train_dataset = ChunkedSequenceDataset(histories, max_seq_len=args.max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              collate_fn=collate_fn)
    eval_loader = DataLoader(
        SequenceDataset(histories, targets, is_train=False, max_seq_len=args.max_seq_len),
        batch_size=256, shuffle=False, collate_fn=collate_fn,
    )
    print(f"items={num_items:,} train_windows={len(train_dataset):,} "
          f"eval_users={len(eval_loader.dataset):,} device={device}")

    log_q = item_log_frequencies(train_dataset.target_items(), num_items) if args.logq else None
    model = NextItemTransformer(
        num_items, args.embedding_dim, log_q=log_q, num_negatives=args.num_negatives,
        max_seq_len=args.max_seq_len, n_layers=args.num_layers, n_heads=args.num_heads, dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = fit(model, train_loader, eval_loader, optimizer, args.epochs, num_items, args.topk, device)

    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.checkpoint)
    if args.user_fraction == 1.0:
        final = {k: v for k, v in history[-1].items() if k not in ("epoch", "loss")}
        save_leaderboard([{"method": method_name(args), **final}], args.output, args.topk)


if __name__ == "__main__":
    main()
