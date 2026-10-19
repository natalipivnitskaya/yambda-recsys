#!/usr/bin/env bash
# Train the two-tower model with every loss (1 epoch each) and collect results/two_tower.md.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-.venv/bin/python}

for loss in bce bpr sampled inbatch inbatch-mixed inbatch-logq; do
  "$PY" scripts/run_two_tower.py --loss "$loss" "$@"
done
# Full softmax needs a (batch x catalog) logit matrix; smaller batches = more updates per epoch.
"$PY" scripts/run_two_tower.py --loss softmax --batch-size 256 "$@"
