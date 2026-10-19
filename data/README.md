# Data

Run `scripts/download_data.sh` to fill this folder. It should then contain:

| File | Contents |
|---|---|
| `interactions.parquet` | `uid, timestamp, item_id, is_organic`: likes from Yambda-500M |
| `embeddings.parquet` | `item_id, embed`: pretrained track embeddings shipped with Yambda |
| `artists.parquet` | `item_id, artist_id`: track to artist mapping |

The files are a filtered and repacked subset of [Yambda](https://huggingface.co/datasets/yandex/yambda) (Apache-2.0).
