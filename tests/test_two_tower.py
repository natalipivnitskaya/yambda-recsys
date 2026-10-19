import torch
import torch.nn.functional as F

from recsys.two_tower import (
    BCEModel,
    BPRModel,
    FullSoftmaxModel,
    InBatchLogQModel,
    InBatchSoftmaxModel,
    SampledSoftmaxModel,
    SequenceDataset,
    TwoTower,
    UserEncoder,
    collate_fn,
    item_log_frequencies,
    pad_sequences,
)


def test_pad_sequences_indices():
    data = torch.tensor([1, 2, 3, 4, 5, 6])
    padded, mask = pad_sequences(data, torch.tensor([2, 3, 1]))
    assert torch.equal(padded, torch.tensor([[1, 2, 0], [3, 4, 5], [6, 0, 0]]))
    assert torch.equal(mask, torch.tensor([[True, True, False], [True, True, True], [True, False, False]]))


def test_pad_sequences_features():
    data = torch.arange(20).view(5, 4)
    padded, mask = pad_sequences(data, torch.tensor([2, 3]))
    assert padded.shape == (2, 3, 4) and mask.shape == (2, 3)
    assert (padded[~mask] == 0).all()
    assert torch.equal(padded[mask], data)


HISTORIES = {"u1": [1, 2, 3, 4], "u2": [10, 11], "u3": [100, 101, 102, 103, 104]}


def test_train_dataset_yields_every_prefix():
    ds = SequenceDataset(HISTORIES, is_train=True, max_seq_len=2)
    assert len(ds) == 3 + 1 + 4
    samples = [ds[i] for i in range(len(ds))]
    assert {(s["uid"], s["label"]) for s in samples} == {
        ("u1", 2), ("u1", 3), ("u1", 4), ("u2", 11),
        ("u3", 101), ("u3", 102), ("u3", 103), ("u3", 104),
    }
    s = next(s for s in samples if s["uid"] == "u1" and s["label"] == 4)
    assert s["history"] == {"item_id": [2, 3], "length": 2}


def test_eval_dataset_one_sample_per_labelled_user():
    ds = SequenceDataset(HISTORIES, {"u1": [999], "u3": [888]}, is_train=False, max_seq_len=2)
    assert len(ds) == 2
    samples = {ds[i]["uid"]: ds[i] for i in range(len(ds))}
    assert set(samples) == {"u1", "u3"}
    assert "label" not in samples["u3"]
    assert samples["u3"]["history"] == {"item_id": [103, 104], "length": 2}


def test_collate_flattens_histories():
    batch = [
        {"uid": 7, "history": {"item_id": [1, 2], "length": 2}, "label": 3},
        {"uid": 8, "history": {"item_id": [10], "length": 1}, "label": 11},
    ]
    out = collate_fn(batch)
    assert torch.equal(out["history"]["item_id"], torch.tensor([1, 2, 10]))
    assert torch.equal(out["history"]["length"], torch.tensor([2, 1]))
    assert torch.equal(out["uid"], torch.tensor([7, 8]))
    assert torch.equal(out["label"], torch.tensor([3, 11]))
    assert "label" not in collate_fn([{k: v for k, v in s.items() if k != "label"} for s in batch])


def _set_weights(model, weight):
    with torch.no_grad():
        model.item_matrix.copy_(weight)


def test_user_encoder_sums_history():
    encoder = UserEncoder(num_items=10, embedding_dim=3)
    weight = torch.arange(30, dtype=torch.float32).view(10, 3)
    with torch.no_grad():
        encoder.item_embeddings.weight.copy_(weight)
    out = encoder({"item_id": torch.arange(6), "length": torch.tensor([2, 3, 1])})
    expected = torch.stack([weight[[0, 1]].sum(0), weight[[2, 3, 4]].sum(0), weight[5]])
    assert torch.allclose(out, expected)


def test_two_tower_train_and_eval_modes():
    class SquaredNorm(TwoTower):
        def compute_loss(self, user_repr, inputs):
            return user_repr.pow(2).mean()

    model = SquaredNorm(num_items=6, embedding_dim=3)
    weight = torch.arange(18, dtype=torch.float32).view(6, 3)
    _set_weights(model, weight)
    inputs = {"uid": torch.tensor([10, 11]), "history": {"item_id": torch.tensor([0, 1, 2]), "length": torch.tensor([2, 1])}}

    model.train()
    assert model(inputs).ndim == 0
    model.eval()
    user = torch.stack([weight[[0, 1]].sum(0), weight[2]])
    assert torch.allclose(model(inputs), user @ weight.T)


def test_full_softmax_matches_cross_entropy():
    model = FullSoftmaxModel(num_items=4, embedding_dim=2)
    W = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]])
    _set_weights(model, W)
    user = torch.tensor([[1.0, 2.0], [0.5, -1.0]])
    labels = torch.tensor([2, 1])
    assert torch.allclose(model.compute_loss(user, {"label": labels}), F.cross_entropy(user @ W.T, labels))


def test_logq_with_uniform_frequencies_equals_plain_in_batch():
    num_items = 5
    log_q = item_log_frequencies(torch.arange(num_items).numpy(), num_items)
    plain, corrected = InBatchSoftmaxModel(num_items, 3), InBatchLogQModel(num_items, 3, log_q)
    corrected.load_state_dict({**plain.state_dict(), "log_q": log_q})
    user = torch.randn(4, 3)
    inputs = {"label": torch.tensor([0, 1, 2, 3])}
    assert torch.allclose(plain.compute_loss(user, inputs), corrected.compute_loss(user, inputs))


def test_all_losses_are_finite_scalars_with_gradients():
    num_items, dim = 50, 8
    log_q = item_log_frequencies(torch.randint(0, num_items, (500,)).numpy(), num_items)
    models = [
        FullSoftmaxModel(num_items, dim), BCEModel(num_items, dim, 16), BPRModel(num_items, dim),
        SampledSoftmaxModel(num_items, dim, 16), InBatchSoftmaxModel(num_items, dim),
        InBatchSoftmaxModel(num_items, dim, num_uniform_negatives=16), InBatchLogQModel(num_items, dim, log_q),
    ]
    batch = collate_fn([
        {"uid": u, "history": {"item_id": [u, u + 1], "length": 2}, "label": u + 2} for u in range(8)
    ])
    for model in models:
        model.train()
        loss = model(batch)
        loss.backward()
        assert loss.ndim == 0 and torch.isfinite(loss), type(model).__name__
        assert model.item_matrix.grad is not None
