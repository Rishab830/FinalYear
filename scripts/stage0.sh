#!/usr/bin/env bash
# Run all deterministic extraction. Safe to re-run. Usage: bash scripts/stage0.sh
set -e
cd "$(dirname "$0")/.."
python3 scripts/extract_notebooks.py .
python3 scripts/collect_metrics.py .
python3 scripts/dataset_stats.py .
python3 scripts/inventory.py .
echo "Stage 0 done. Look in facts/"
