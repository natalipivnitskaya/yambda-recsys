"""Semantic IDs from content embeddings, used by the causal Transformer in two ways.

    python scripts/run_semantic_ids.py --model embed                 # item vector = sum of code vectors + id vector
    python scripts/run_semantic_ids.py --model embed --no-item-ids   # codes only
    python scripts/run_semantic_ids.py --model generative            # predict the next item's code, beam search
    python scripts/run_semantic_ids.py --model generative --epochs 1 --user-fraction 0.05   # quick smoke run
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
from recsys.semantic_ids import GenerativeRetrieval, SemanticIdEmbedding, code_stats, residual_kmeans
from recsys.sequential import ChunkedSequenceDataset, NextItemTransformer, collate_fn
from recsys.two_tower import SequenceDataset, fit, item_log_frequencies

ROOT = Path(__file__).resolve().parents[1]


def method_name(args: argparse.Namespace) -> str:
    codes = f"{args.levels}x{args.codebook_size} codes"
    inputs = "codes + item ids" if args.item_ids else "codes only"
    if args.model == "embed":
        return f"Transformer, semantic ID embeddings ({inputs}, {codes}, {args.epochs} ep), in-batch negatives + logQ"
    return f"Generative retrieval over semantic IDs ({inputs}, {codes}, beam {args.beam_size}, {args.epochs} ep)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["embed", "generative"], required=True)
    parser.add_argument("--no-item-ids", dest="item_ids", action="store_false",
                        help="represent items by their codes only (default: codes + a per-item vector for "
                             "--model embed, codes only for --model generative)")
    parser.add_argument("--item-ids", dest="item_ids", action="store_true")
    parser.set_defaults(item_ids=None)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--kmeans-iters", type=int, default=30)
    parser.add_argument("--beam-size", type=int, default=100)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128, help="windows (not events) per batch")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-negatives", type=int, default=512, help="--model embed only")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-seq-len", type=int, default=100)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--user-fraction", type=float, default=1.0, help="subsample users for quick runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "sequential.csv")
    parser.add_argument("--checkpoint", type=Path, help="where to save the trained state dict")
    args = parser.parse_args()
    if args.item_ids is None:
        args.item_ids = args.model == "embed"
    if args.model == "generative" and args.beam_size < args.topk:
        parser.error("--beam-size must be at least --topk")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    ds = load_dataset(args.data_dir, reindex_items=True)
    embeddings = ds.embeddings.sort("item_id")
    num_items = embeddings.height  # after reindexing, ids are exactly [0, num_items)
    vectors = torch.from_numpy(np.stack(embeddings["embed"].to_list()).astype(np.float32)).to(device)
    ids = residual_kmeans(vectors, levels=args.levels, codebook_size=args.codebook_size,
                          iters=args.kmeans_iters, seed=args.seed)
    print("semantic ids:", code_stats(ids), "residual error per level:",
          [round(e, 3) for e in ids.reconstruction_error])

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

    if args.model == "embed":
        model = NextItemTransformer(
            num_items, args.embedding_dim, log_q=item_log_frequencies(train_dataset.target_items(), num_items),
            num_negatives=args.num_negatives, max_seq_len=args.max_seq_len, n_layers=args.num_layers,
            n_heads=args.num_heads, dropout=args.dropout,
            item_embeddings=SemanticIdEmbedding(ids, args.embedding_dim, use_item_ids=args.item_ids),
        )
    else:
        model = GenerativeRetrieval(
            ids, args.embedding_dim, max_seq_len=args.max_seq_len, n_layers=args.num_layers,
            n_heads=args.num_heads, dropout=args.dropout, use_item_ids=args.item_ids, beam_size=args.beam_size,
        )
    model = model.to(device)
    print(f"parameters={sum(p.numel() for p in model.parameters()):,}")
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
