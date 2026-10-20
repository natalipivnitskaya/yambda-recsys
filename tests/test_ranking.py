import numpy as np
import polars as pl
import pytest
import torch

from recsys.ops import segment_mean
from recsys.ranking.data import add_item_metadata, preference_events, temporal_split
from recsys.ranking.features import MultiHash, PiecewiseLinearEncoding, UnifiedEmbedding
from recsys.ranking.metrics import pairwise_accuracy
from recsys.ranking.models import CrossNetwork, FeatureEncoder, Ranker
from recsys.ranking.train import ragged_take


def _listens(likes, fulls, uid=1, start=0, step=60):
    n = len(likes)
    return pl.DataFrame({
        "uid": [uid] * n, "timestamp": [start + i * step for i in range(n)], "row_id": list(range(n)),
        "is_like": [bool(x) for x in likes], "is_full_play": [bool(x) for x in fulls],
    })


# ---------------------------------------------------------------- metric

def test_pairwise_accuracy_counts_only_differing_adjacent_pairs():
    uids = np.array([1, 1, 1, 2, 2])
    ts = np.array([0, 10, 20, 0, 10])
    labels = np.array([1, 0, 0, 0, 1])
    scores = np.array([0.9, 0.1, 0.5, 0.3, 0.2])
    # user 1: pair (0,1) correct, pair (1,2) same labels; user 2: pair wrong
    assert pairwise_accuracy(uids, ts, labels, scores) == pytest.approx(0.5)


def test_pairwise_accuracy_respects_session_gap_and_ties():
    uids, labels = np.array([1, 1, 1]), np.array([1, 0, 1])
    ts = np.array([0, 10, 10_000])
    assert pairwise_accuracy(uids, ts, labels, np.array([1.0, 0.0, 0.0]), session_gap=100) == 1.0
    assert pairwise_accuracy(uids, ts, labels, np.zeros(3), session_gap=100) == 0.5


def test_pairwise_accuracy_sorts_by_user_and_time():
    uids = np.array([1, 1])
    ts = np.array([10, 0])
    assert pairwise_accuracy(uids, ts, np.array([0, 1]), np.array([0.0, 1.0])) == 1.0


# ---------------------------------------------------------------- data

def test_preference_events_keeps_rows_with_a_differing_neighbour():
    df = _listens(likes=[0, 0, 0, 1, 1, 1], fulls=[1, 1, 1, 1, 1, 1])
    assert preference_events(df)["row_id"].to_list() == [2, 3]


def test_preference_filter_does_not_change_pairwise_accuracy():
    rng = np.random.default_rng(0)
    frames = [_listens(rng.random(200) < 0.3, rng.random(200) < 0.6, uid=u, step=int(rng.integers(30, 600)))
              for u in range(20)]
    df = pl.concat(frames).with_columns(pl.Series("score", rng.random(4000)))
    kept = preference_events(df)
    assert kept.height < df.height
    for label in ("is_like", "is_full_play"):
        def acc(d):
            return pairwise_accuracy(d["uid"].to_numpy(), d["timestamp"].to_numpy(), d[label].to_numpy(), d["score"].to_numpy())
        assert acc(kept) == pytest.approx(acc(df))


def test_temporal_split_puts_boundary_in_test():
    df = pl.DataFrame({"timestamp": [0, 50, 90, 100]})
    train, test = temporal_split(df, test_seconds=10)
    assert train["timestamp"].to_list() == [0, 50] and test["timestamp"].to_list() == [90, 100]


def test_add_item_metadata_builds_lists_and_fills_missing():
    listens = pl.DataFrame({"item_id": pl.Series([1, 2, 3], dtype=pl.UInt32)})
    artists = pl.DataFrame({"artist_id": [10, 11], "item_id": [1, 1]}, schema={"artist_id": pl.UInt32, "item_id": pl.UInt32})
    albums = pl.DataFrame({"album_id": [7], "item_id": [2]}, schema={"album_id": pl.UInt32, "item_id": pl.UInt32})
    out = add_item_metadata(listens, artists, albums).sort("item_id")
    assert out["artist_ids"].to_list() == [[10, 11], [], []]
    assert out["album_ids"].to_list() == [[], [7], []]


def test_ragged_take():
    values, offsets = np.array([1, 2, 3, 4, 5, 6]), np.array([0, 2, 2, 5, 6])  # [1,2], [], [3,4,5], [6]
    ids, lengths = ragged_take(values, offsets, np.array([2, 0, 1, 3]))
    assert ids.tolist() == [3, 4, 5, 1, 2, 6] and lengths.tolist() == [3, 2, 0, 1]


# ---------------------------------------------------------------- features & models

def test_piecewise_linear_codes():
    enc = PiecewiseLinearEncoding([np.array([0.0, 1.0, 3.0])], dim=2)
    codes = enc.encode(torch.tensor([[-1.0], [0.5], [2.0], [5.0]]))[:, 0]
    assert torch.allclose(codes, torch.tensor([[0, 0], [0.5, 0], [1, 0.5], [1, 1]]))


def test_piecewise_linear_from_data_handles_constant_and_discrete_columns():
    x = np.stack([np.zeros(1000), np.repeat([0.0, 1.0, 2.0, 3.0], 250), np.linspace(0, 1, 1000)], axis=1)
    enc = PiecewiseLinearEncoding.from_data(x, n_bins=8, dim=4)
    assert enc.n_bins.tolist() == [1, 3, 8]
    assert enc(torch.tensor(x[:5], dtype=torch.float32)).shape == (5, 3 * 4)


def test_multihash_is_deterministic_and_in_range():
    h = MultiHash(num_hashes=3, table_size=100, seed=1)
    ids = torch.tensor([0, 5, 123_456_789, 2 ** 32 - 1])
    out = h(ids)
    assert out.shape == (4, 3) and out.min() >= 0 and out.max() < 100
    assert torch.equal(out, MultiHash(3, 100, seed=1)(ids))
    assert not torch.equal(out, MultiHash(3, 100, seed=2)(ids))


def test_unified_embedding_bag_is_mean_of_lookups():
    emb = UnifiedEmbedding({"artist": 2}, table_size=50, dim=4, bucket=8)
    ids, lengths = torch.tensor([3, 7, 9]), torch.tensor([2, 0, 1])
    bag = emb.lookup_bag("artist", ids, lengths)
    single = emb.lookup("artist", ids)
    assert torch.allclose(bag[0], single[:2].mean(0))
    assert torch.allclose(bag[1], torch.zeros(8))
    assert torch.allclose(bag[2], single[2])


def test_segment_mean_ignores_padding():
    values = torch.tensor([[1.0], [3.0], [5.0], [100.0]])  # last row is padding
    assert segment_mean(values, torch.tensor([2, 1])).flatten().tolist() == [2.0, 5.0]


def test_cross_network_single_expert_formula():
    torch.manual_seed(0)
    net = CrossNetwork(dim=4, num_layers=1, rank=2, num_experts=1)
    x0 = torch.randn(3, 4)
    W = net.V[0][0] @ net.U[0][0]                       # (dim, dim) low-rank weight
    assert torch.allclose(net(x0), x0 + x0 * (x0 @ W + net.b[0][0]), atol=1e-6)


def test_ranker_forward_and_backward():
    rng = np.random.default_rng(0)
    numeric = PiecewiseLinearEncoding.from_data(rng.random((100, 3)), n_bins=4, dim=2)
    cat = UnifiedEmbedding({"uid": 1, "item_id": 2, "artist": 1, "album": 1}, table_size=64, dim=4, bucket=8)
    encoder = FeatureEncoder(numeric, cat, ["uid", "item_id"], ["artist", "album"])
    model = Ranker(encoder, ["is_like", "is_full_play"], hidden=(16,), cross_layers=2, cross_rank=4, cross_experts=3)
    batch = {
        "dense": torch.rand(5, 3), "sparse": torch.randint(0, 1000, (5, 2)),
        "artist_ids": torch.arange(6), "artist_lengths": torch.tensor([1, 2, 0, 2, 1]),
        "album_ids": torch.arange(5), "album_lengths": torch.ones(5, dtype=torch.long),
    }
    logits = model(batch)
    assert logits.shape == (5, 2)
    logits.sum().backward()
    assert cat.table.weight.grad is not None and model.cross.U[0].grad is not None
