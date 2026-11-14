import numpy as np
import torch

from recsys.semantic_ids import (
    GenerativeRetrieval,
    SemanticIdEmbedding,
    SemanticIds,
    build_prefix_tables,
    code_stats,
    duplicate_rank,
    kmeans,
    residual_kmeans,
)
from recsys.sequential import ChunkedSequenceDataset, NextItemTransformer, collate_fn


def test_kmeans_separates_clear_clusters():
    torch.manual_seed(0)
    centers = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    x = torch.cat([c + 0.1 * torch.randn(50, 2) for c in centers])
    found = kmeans(x, 3, iters=20)
    assert torch.cdist(centers, found).min(dim=1).values.max() < 0.5


def test_duplicate_rank():
    codes = np.array([[1, 2], [3, 4], [1, 2], [1, 2], [3, 4], [5, 6]])
    assert duplicate_rank(codes).tolist() == [0, 0, 1, 2, 1, 0]


def test_residual_kmeans_codes_are_unique_and_error_decreases():
    torch.manual_seed(0)
    ids = residual_kmeans(torch.randn(500, 16), levels=3, codebook_size=8, iters=10)
    assert ids.codes.shape == (500, 4)
    assert len(torch.unique(ids.codes, dim=0)) == 500
    assert ids.reconstruction_error == sorted(ids.reconstruction_error, reverse=True)
    assert ids.codebook_sizes[:3] == [8, 8, 8] and ids.codebook_sizes[3] == int(ids.codes[:, 3].max()) + 1
    assert code_stats(ids)["items"] == 500


CODES = torch.tensor([[0, 1, 0], [0, 1, 1], [0, 2, 0], [1, 0, 0], [1, 2, 0]])
IDS = SemanticIds(codes=CODES, codebook_sizes=[2, 3, 2], reconstruction_error=[])


def test_prefix_tables_follow_codes_to_items():
    tables = build_prefix_tables(CODES, IDS.codebook_sizes)
    assert [t.shape for t in tables] == [(1, 2), (2, 3), (4, 2)]
    for item, code in enumerate(CODES.tolist()):
        prefix = 0
        for level, c in enumerate(code):
            prefix = int(tables[level][prefix, c])
        assert prefix == item
    assert int(tables[1][1, 1]) == -1  # no item starts with (1, 1)


def test_semantic_id_embedding_sums_code_vectors():
    emb = SemanticIdEmbedding(IDS, embedding_dim=4, use_item_ids=False)
    table = emb.code_embeddings.weight
    expected = table[0] + table[2 + 1] + table[5 + 1]  # item 1: codes (0, 1, 1) with offsets 0, 2, 5
    assert torch.allclose(emb(torch.tensor([1]))[0], expected)
    assert emb.weight.shape == (5, 4)
    with_ids = SemanticIdEmbedding(IDS, embedding_dim=4, use_item_ids=True)
    expected = with_ids.code_embeddings(with_ids.flat_codes).sum(1) + with_ids.item_embeddings.weight
    assert torch.allclose(with_ids.weight, expected)


def _generative(**kwargs):
    torch.manual_seed(0)
    return GenerativeRetrieval(IDS, embedding_dim=8, max_seq_len=4, dropout=0.0, **kwargs)


def test_wide_beam_search_matches_exact_scores():
    model = _generative().eval()
    with torch.no_grad():  # spread the logits so the ranking is not a tie
        for emb in model.code_embeddings:
            emb.weight.normal_()
    user = torch.randn(3, 8)
    exact = model.exact_scores(user)
    assert (exact.exp().sum(1) <= 1 + 1e-5).all()  # some probability goes to codes no item has
    items, scores = model.beam_search(user, beam_size=5)
    assert torch.equal(items, exact.argsort(dim=1, descending=True))
    assert torch.allclose(scores, exact.sort(dim=1, descending=True).values, atol=1e-5)


def test_beam_larger_than_catalog_pads_with_invalid_slots():
    model = _generative().eval()
    items, scores = model.beam_search(torch.randn(2, 8), beam_size=8)
    assert (items[:, :5] >= 0).all() and set(items[0, :5].tolist()) == set(range(5))
    assert (items[:, 5:] == -1).all() and torch.isinf(scores[:, 5:]).all()


def test_generative_train_and_eval_modes():
    model = _generative(use_item_ids=True, beam_size=3)
    ds = ChunkedSequenceDataset({u: [u % 5, (u + 1) % 5, (u + 2) % 5] for u in range(6)}, max_seq_len=4)
    batch = collate_fn([ds[i] for i in range(len(ds))])
    model.train()
    loss = model(batch)
    loss.backward()
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert model.code_embeddings[0].weight.grad is not None
    model.eval()
    scores = model(batch)
    assert scores.shape == (len(ds), 5)
    assert (torch.isfinite(scores).sum(1) == 3).all()


def test_next_item_transformer_accepts_semantic_id_embeddings():
    model = NextItemTransformer(5, 8, num_negatives=4, max_seq_len=4, dropout=0.0,
                                item_embeddings=SemanticIdEmbedding(IDS, 8))
    ds = ChunkedSequenceDataset({u: [u % 5, (u + 1) % 5, (u + 2) % 5] for u in range(6)}, max_seq_len=4)
    batch = collate_fn([ds[i] for i in range(len(ds))])
    model.train()
    model(batch).backward()
    assert model.encoder.item_embeddings.code_embeddings.weight.grad is not None
    model.eval()
    assert model(batch).shape == (len(ds), 5)
