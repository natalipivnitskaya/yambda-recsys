"""Train the bag-of-items two-tower model with a chosen loss and log metrics.

    python scripts/run_two_tower.py --loss inbatch-logq
    python scripts/run_two_tower.py --loss softmax --batch-size 256 --device cpu
    python scripts/run_two_tower.py --loss bpr --user-fraction 0.05   # quick smoke run
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
from recsys.two_tower import (
    BCEModel,
    BPRModel,
    FullSoftmaxModel,
    InBatchLogQModel,
    InBatchSoftmaxModel,
    SampledSoftmaxModel,
    SequenceDataset,
    collate_fn,
    fit,
    item_log_frequencies,
)

ROOT = Path(__file__).resolve().parents[1]
LOSSES = {
    "softmax": "Full softmax",
    "bce": "BCE, uniform negatives",
    "bpr": "BPR, uniform negatives",
    "sampled": "Sampled softmax, uniform negatives",
    "inbatch": "Sampled softmax, in-batch negatives",
    "inbatch-mixed": "Sampled softmax, in-batch + uniform negatives",
    "inbatch-logq": "Sampled softmax, in-batch negatives + logQ",
}


def build_model(loss: str, num_items: int, dim: int, num_negatives: int, train_items: np.ndarray):
    if loss == "softmax":
        return FullSoftmaxModel(num_items, dim)
    if loss == "bce":
        return BCEModel(num_items, dim, num_negatives)
    if loss == "bpr":
        return BPRModel(num_items, dim)
    if loss == "sampled":
        return SampledSoftmaxModel(num_items, dim, num_negatives)
    if loss == "inbatch":
        return InBatchSoftmaxModel(num_items, dim)
    if loss == "inbatch-mixed":
        return InBatchSoftmaxModel(num_items, dim, num_uniform_negatives=num_negatives)
    if loss == "inbatch-logq":
        return InBatchLogQModel(num_items, dim, item_log_frequencies(train_items, num_items))
    raise ValueError(loss)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--loss", choices=LOSSES, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--num-negatives", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-seq-len", type=int, default=100)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--user-fraction", type=float, default=1.0, help="subsample users for quick runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "two_tower.csv")
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

    train_loader = DataLoader(
        SequenceDataset(histories, is_train=True, max_seq_len=args.max_seq_len),
        batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn,
    )
    eval_loader = DataLoader(
        SequenceDataset(histories, targets, is_train=False, max_seq_len=args.max_seq_len),
        batch_size=256, shuffle=False, collate_fn=collate_fn,
    )
    print(f"items={num_items:,} train_samples={len(train_loader.dataset):,} "
          f"eval_users={len(eval_loader.dataset):,} device={device}")

    model = build_model(args.loss, num_items, args.embedding_dim, args.num_negatives,
                        ds.train["item_id"].to_numpy()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = fit(model, train_loader, eval_loader, optimizer, args.epochs, num_items, args.topk, device)

    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.checkpoint)
    if args.user_fraction == 1.0:
        final = {k: v for k, v in history[-1].items() if k not in ("epoch", "loss")}
        save_leaderboard([{"method": LOSSES[args.loss], **final}], args.output, args.topk)


if __name__ == "__main__":
    main()
