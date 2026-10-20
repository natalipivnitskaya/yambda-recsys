#!/usr/bin/env bash
# Download the ranking dataset (~800 MB) into data/ranking/:
#   listens.parquet               Yambda-50M listens with lagged counters (matfu21/yambda-50m-lag-features)
#   artist_item_mapping.parquet   track -> artists (yandex/yambda)
#   album_item_mapping.parquet    track -> albums  (yandex/yambda)
set -euo pipefail
mkdir -p "$(dirname "$0")/../data/ranking"
cd "$(dirname "$0")/../data/ranking"

HF=https://huggingface.co/datasets
fetch() {  # url file
  if [[ ! -f "$2" ]]; then
    curl -fL --retry 3 -o "$2.part" "$1"
    mv "$2.part" "$2"
  fi
}
fetch "$HF/matfu21/yambda-50m-lag-features/resolve/main/listens.parquet" listens.parquet
fetch "$HF/yandex/yambda/resolve/main/artist_item_mapping.parquet" artist_item_mapping.parquet
fetch "$HF/yandex/yambda/resolve/main/album_item_mapping.parquet" album_item_mapping.parquet
ls -lh ./*.parquet
