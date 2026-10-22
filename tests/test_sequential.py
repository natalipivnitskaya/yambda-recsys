import torch

from recsys.sequential import (
    Block,
    CausalSelfAttention,
    ChunkedSequenceDataset,
    NextItemTransformer,
    TransformerUserEncoder,
    collate_fn,
)
from recsys.two_tower import SequenceDataset, item_log_frequencies

HISTORIES = {1: [1, 2, 3, 4, 5, 6], 2: [10, 11], 3: [100], 4: [20, 21, 22]}


def test_chunked_dataset_splits_into_non_overlapping_windows():
    ds = ChunkedSequenceDataset(HISTORIES, max_seq_len=2)
    samples = [ds[i] for i in range(len(ds))]
    got = [(s["uid"], s["history"]["item_id"], s["targets"]) for s in samples]
    assert got == [
        (1, [1, 2], [2, 3]), (1, [3, 4], [4, 5]), (1, [5], [6]),
        (2, [10], [11]),
        (4, [20, 21], [21, 22]),
    ]
    assert all(s["history"]["length"] == len(s["targets"]) for s in samples)
    assert sorted(ds.target_items().tolist()) == sorted([2, 3, 4, 5, 6, 11, 21, 22])


def test_collate_adds_flat_targets():
    ds = ChunkedSequenceDataset(HISTORIES, max_seq_len=2)
    batch = collate_fn([ds[0], ds[3]])
    assert torch.equal(batch["history"]["item_id"], torch.tensor([1, 2, 10]))
    assert torch.equal(batch["history"]["length"], torch.tensor([2, 1]))
    assert torch.equal(batch["targets"], torch.tensor([2, 3, 11]))
    eval_batch = collate_fn([SequenceDataset(HISTORIES, {1: [7]}, is_train=False)[0]])
    assert "targets" not in eval_batch


def test_attention_is_causal_and_ignores_padding():
    torch.manual_seed(0)
    attn = CausalSelfAttention(d_model=8, n_heads=2).eval()
    x = torch.randn(1, 5, 8)
    mask = torch.tensor([[True, True, True, False, False]])
    out = attn(x, mask)

    future = x.clone()
    future[0, 2:] = torch.randn(3, 8)  # changes position 2 and the padding
    out_future = attn(future, mask)
    assert torch.allclose(out[0, :2], out_future[0, :2], atol=1e-6)
    assert not torch.allclose(out[0, 2], out_future[0, 2])

    padding = x.clone()
    padding[0, 3:] = torch.randn(2, 8)
    assert torch.allclose(out[0, :3], attn(padding, mask)[0, :3], atol=1e-6)


def test_block_keeps_shape():
    block = Block(d_model=8, n_heads=2)
    x = torch.randn(3, 4, 8)
    assert block(x, torch.ones(3, 4, dtype=torch.bool)).shape == x.shape


def test_encoder_pads_to_max_seq_len_and_matches_unbatched():
    torch.manual_seed(0)
    encoder = TransformerUserEncoder(num_items=30, embedding_dim=8, max_seq_len=6, dropout=0.0).eval()
    batch = {"item_id": torch.tensor([1, 2, 3, 4, 5]), "length": torch.tensor([3, 2])}
    hidden, mask = encoder(batch)
    assert hidden.shape == (2, 6, 8) and mask.sum().item() == 5
    alone, _ = encoder({"item_id": torch.tensor([4, 5]), "length": torch.tensor([2])})
    assert torch.allclose(hidden[1, :2], alone[0, :2], atol=1e-6)


def _model(log_q=None, **kwargs):
    return NextItemTransformer(num_items=50, embedding_dim=8, log_q=log_q, num_negatives=16,
                               max_seq_len=4, dropout=0.0, **kwargs)


def test_eval_scores_use_last_position():
    torch.manual_seed(0)
    model = _model().eval()
    batch = collate_fn([SequenceDataset(HISTORIES, {1: [7], 4: [7]}, is_train=False, max_seq_len=4)[i]
                        for i in range(2)])
    scores = model(batch)
    assert scores.shape == (2, 50)
    hidden, _ = model.encoder(batch["history"])
    assert torch.allclose(scores[0], hidden[0, 3] @ model.item_matrix.T, atol=1e-5)  # u1: last 4 items
    assert torch.allclose(scores[1], hidden[1, 2] @ model.item_matrix.T, atol=1e-5)  # u4: 3 items


def test_loss_is_finite_with_gradients():
    log_q = item_log_frequencies(torch.randint(0, 50, (500,)).numpy(), 50)
    ds = ChunkedSequenceDataset({u: list(range(u, u + 7)) for u in range(10)}, max_seq_len=4)
    batch = collate_fn([ds[i] for i in range(len(ds))])
    for model in (_model(), _model(log_q)):
        model.train()
        loss = model(batch)
        loss.backward()
        assert loss.ndim == 0 and torch.isfinite(loss)
        assert model.item_matrix.grad is not None


def test_uniform_log_q_does_not_change_loss():
    log_q = item_log_frequencies(torch.arange(50).numpy(), 50)
    plain, corrected = _model(), _model(log_q)
    corrected.load_state_dict({**plain.state_dict(), "log_q": log_q})
    hidden = torch.randn(2, 4, 8)
    mask = torch.tensor([[True, True, True, False], [True, True, True, True]])
    targets = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]])
    torch.manual_seed(1)
    expected = plain.compute_loss(hidden, mask, targets)
    torch.manual_seed(1)
    assert torch.allclose(corrected.compute_loss(hidden, mask, targets), expected)


def test_negatives_equal_to_positive_are_masked():
    model = _model()
    hidden = torch.randn(1, 4, 8)
    mask = torch.tensor([[True, True, False, False]])
    same = torch.tensor([[3, 3, 0, 0]])  # every sampled negative is the positive itself
    assert torch.allclose(model.compute_loss(hidden, mask, same), torch.tensor(0.0))
