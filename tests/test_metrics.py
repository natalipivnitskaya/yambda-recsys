import numpy as np
import pytest

from recsys.baselines import fill_with_popular
from recsys.metrics import evaluate, user_metrics


def test_user_metrics_toy():
    m = user_metrics(targets=[1, 2, 3], candidates=[10, 2, 30, 1], topk=4)
    assert m["hitrate"] == 1.0
    assert m["recall"] == pytest.approx(2 / 3)
    # DCG = 1/log2(3) + 1/log2(5); IDCG = 1 + 1/log2(3) + 1/log2(4)
    assert m["ndcg"] == pytest.approx(0.49818925746641285)


def test_duplicate_targets_are_counted_once():
    assert user_metrics([1, 2, 2, 2, 3], [10, 2, 30, 1], 4) == user_metrics([1, 2, 3], [10, 2, 30, 1], 4)


def test_perfect_ranking_scores_one():
    m = user_metrics(targets=list(range(10)), candidates=list(range(5)), topk=5)
    assert m == {"hitrate": 1.0, "recall": 1.0, "ndcg": pytest.approx(1.0)}


def test_no_hits():
    assert user_metrics([1], [2, 3], 2) == {"hitrate": 0.0, "recall": 0.0, "ndcg": 0.0}


def test_evaluate_averages_users_and_computes_coverage():
    out = evaluate({7: [1, 2], 8: [9]}, {7: [1, 3, 4], 8: [10, 9, 11]}, catalog_size=100, topk=3)
    assert out["hitrate"] == pytest.approx(1.0)
    assert out["recall"] == pytest.approx(0.75)
    assert out["ndcg"] == pytest.approx(0.622038473168458)
    assert out["coverage"] == pytest.approx(0.06)


def test_fill_with_popular_pads_without_duplicates():
    out = fill_with_popular({1: [5, 6], 2: list(range(10))}, popular=[6, 7, 8, 9], topk=4, users=[3])
    assert out[1] == [5, 6, 7, 8]
    assert out[2] == [0, 1, 2, 3]
    assert out[3] == [6, 7, 8, 9]
    assert all(len(set(v)) == len(v) for v in out.values())
    assert np.all([len(v) == 4 for v in out.values()])
