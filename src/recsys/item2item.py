"""Item-to-item recommendations over any item similarity.

Neighbour lists come from one of three sources:
  * pretrained audio/content embeddings shipped with Yambda,
  * Item2Vec embeddings trained on user sequences (word2vec),
  * sparse user-item co-occurrence vectors (item-based collaborative filtering).

A user's candidates are the time-decayed sum of neighbour similarities over
their most recent interactions.
"""
from __future__ import annotations

import heapq
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from tqdm import tqdm

Neighbours = Dict[int, List[Tuple[int, float]]]


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def embedding_neighbours(
    item_ids: np.ndarray,
    item_embeddings: np.ndarray,
    topk: int = 200,
    batch_size: int = 1024,
    device: Optional[str] = None,
) -> Neighbours:
    """Exact top-``topk`` cosine neighbours for every item, computed in batches."""
    device = device or default_device()
    emb = F.normalize(torch.as_tensor(item_embeddings, dtype=torch.float32, device=device), dim=1)
    k = min(topk + 1, len(emb))

    neighbours: Neighbours = {}
    for start in tqdm(range(0, len(emb), batch_size), desc="cosine top-k"):
        sims, idx = torch.topk(emb[start:start + batch_size] @ emb.T, k=k, dim=1)
        sims, idx = sims.cpu().numpy(), idx.cpu().numpy()
        for row in range(len(idx)):
            item = int(item_ids[start + row])
            cand_ids = item_ids[idx[row]]
            keep = cand_ids != item
            neighbours[item] = list(zip(cand_ids[keep][:topk].tolist(), sims[row][keep][:topk].tolist()))
    return neighbours


def item2vec_embeddings(
    train: pl.DataFrame,
    vector_size: int = 100,
    window: int = 5,
    epochs: int = 5,
    seed: int = 42,
    workers: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train CBOW word2vec on chronological user sequences; returns ``(item_ids, vectors)``."""
    from gensim.models import Word2Vec

    sequences = train.group_by("uid").agg(pl.col("item_id").sort_by("timestamp"))["item_id"].to_list()
    corpus = [[str(i) for i in seq] for seq in sequences]
    w2v = Word2Vec(
        sentences=corpus, vector_size=vector_size, window=window, min_count=1,
        sg=0, epochs=epochs, seed=seed, workers=workers,
    )
    item_ids = np.array([int(token) for token in w2v.wv.index_to_key], dtype=np.int64)
    return item_ids, w2v.wv.vectors


def cooccurrence_neighbours(user_items: csr_matrix, item_ids: np.ndarray, topk: int = 400) -> Neighbours:
    """Cosine neighbours over columns of the user-item matrix (item-based CF)."""
    from implicit.nearest_neighbours import CosineRecommender

    model = CosineRecommender(K=topk + 1)
    model.fit(user_items.tocsr().astype(np.float32), show_progress=False)
    similarity = model.similarity.tocsr()

    neighbours: Neighbours = {}
    for idx, item in enumerate(item_ids.tolist()):
        row = similarity[idx]
        order = np.argsort(-row.data)
        pairs = [(int(item_ids[j]), float(s)) for j, s in zip(row.indices[order], row.data[order]) if j != idx]
        neighbours[item] = pairs[:topk]
    return neighbours


def aggregate_neighbours(
    train: pl.DataFrame,
    neighbours: Neighbours,
    users: Optional[Iterable[int]] = None,
    n_last: int = 50,
    half_life: float = 25.0,
    topk: int = 100,
) -> Dict[int, List[int]]:
    """Score candidates for each user from the neighbours of their recent items.

    ``score(u, j) = sum_t 2^{-(L - t) / half_life} * sim(i_t, j)`` over the user's
    last ``L <= n_last`` items. Items already in the user's history are dropped.
    """
    if users is not None:
        train = train.join(pl.DataFrame({"uid": list(users)}, schema={"uid": train["uid"].dtype}), on="uid", how="semi")

    per_user = (
        train.group_by("uid")
        .agg(
            pl.col("item_id").sort_by("timestamp").tail(n_last).alias("recent"),
            pl.col("item_id").unique().alias("seen"),
        )
    )

    candidates = {}
    for uid, recent, seen in tqdm(per_user.iter_rows(), total=per_user.height, desc="aggregate"):
        seen = set(seen)
        length = len(recent)
        scores: Dict[int, float] = defaultdict(float)
        for position, item in enumerate(recent, start=1):
            weight = 2.0 ** (-(length - position) / half_life)
            for cand, sim in neighbours.get(item, ()):
                if cand not in seen:
                    scores[cand] += weight * sim
        best = heapq.nlargest(topk, scores.items(), key=lambda kv: kv[1])
        candidates[uid] = [cand for cand, _ in best]
    return candidates
