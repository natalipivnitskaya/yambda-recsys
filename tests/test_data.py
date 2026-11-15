import polars as pl

from recsys.data import WEEK_SECONDS, load_dataset
from recsys.report import save_best_epoch


def _write_data(path):
    # user u likes item i at day d; items 0..2 appear often enough to survive filtering
    rows = [(u, i, d * 24 * 3600) for u in range(4) for i in range(3) for d in range(0, 21, 3)]
    rows += [(9, 7, 0)]  # rare item, filtered out
    pl.DataFrame(rows, schema=["uid", "item_id", "timestamp"], orient="row").write_parquet(path / "interactions.parquet")
    pl.DataFrame({"item_id": [0, 1, 2, 7], "embed": [[0.0, 1.0]] * 4}).write_parquet(path / "embeddings.parquet")
    pl.DataFrame({"item_id": [0, 1, 2, 7], "artist_id": [1, 1, 2, 3]}).write_parquet(path / "artists.parquet")


def test_validation_split_uses_week_before_test(tmp_path):
    _write_data(tmp_path)
    test = load_dataset(tmp_path)
    val = load_dataset(tmp_path, validation=True)
    end = 18 * 24 * 3600
    assert test.test["timestamp"].min() >= end - WEEK_SECONDS
    assert val.train["timestamp"].max() < end - 2 * WEEK_SECONDS
    assert end - 2 * WEEK_SECONDS <= val.test["timestamp"].min() <= val.test["timestamp"].max() < end - WEEK_SECONDS
    assert set(val.embeddings["item_id"]) == set(test.embeddings["item_id"]) == {0, 1, 2}


def test_save_best_epoch(tmp_path):
    history = [{"epoch": e, "loss": 1.0, "hitrate": r, "recall": r, "ndcg": r, "coverage": r}
               for e, r in [(1, 0.1), (2, 0.3), (3, 0.2)]]
    row = save_best_epoch("m", history, tmp_path / "v.csv")
    assert row["best_epoch"] == 2 and row["recall"] == 0.3
    assert "| m | 2 | 0.3000" in (tmp_path / "v.md").read_text()
