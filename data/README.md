# Data

Nothing in this folder except this README is committed. Use the scripts to download the data.

## Retrieval: `scripts/download_data.sh`

| File | Contents |
|---|---|
| `interactions.parquet` | `uid, timestamp, item_id, is_organic`: likes from Yambda-500M |
| `embeddings.parquet` | `item_id, embed`: pretrained track embeddings shipped with Yambda |
| `artists.parquet` | `item_id, artist_id`: track to artist mapping |

These files are a filtered and repacked subset of [Yambda](https://huggingface.co/datasets/yandex/yambda) (Apache-2.0).

## Ranking: `scripts/download_ranking_data.sh` (~800 MB)

| File | Contents |
|---|---|
| `ranking/listens.parquet` | Yambda-50M listens from recommendation surfaces only, with `is_like` / `is_full_play` labels and 15 lagged counters, from [`matfu21/yambda-50m-lag-features`](https://huggingface.co/datasets/matfu21/yambda-50m-lag-features) |
| `ranking/artist_item_mapping.parquet` | track to artists (Yambda) |
| `ranking/album_item_mapping.parquet` | track to albums (Yambda) |

The first run of `scripts/run_ranking.py` caches the prepared split in `ranking/prepared_30d/`.
