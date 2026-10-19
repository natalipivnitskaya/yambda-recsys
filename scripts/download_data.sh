#!/usr/bin/env bash
# Download the preprocessed Yambda subset (~360 MB zip) into data/.
set -euo pipefail
cd "$(dirname "$0")/../data"

if [[ ! -f interactions.parquet || ! -f embeddings.parquet || ! -f artists.parquet ]]; then
  [[ -f dataset.zip ]] || gdown 1PojPVpXGBAqzHQi97QAFhJ9gnPsXxveS -O dataset.zip
  unzip -oq dataset.zip
  rm dataset.zip
fi
ls -lh ./*.parquet
