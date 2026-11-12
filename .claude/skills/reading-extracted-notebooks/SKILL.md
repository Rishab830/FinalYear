---
name: reading-extracted-notebooks
description: How to read the stripped notebook files in facts/nb/, the metrics tables and the other generated evidence files, and how to cite them. Read before reading any file under facts/.
---
# Reading the generated evidence

Never open raw `.ipynb` files or anything under `models/` directly. Scripts have already turned them into small readable files under `facts/`. Those files are your only evidence.

## facts/nb/<notebook>.md
- Header lists cell counts and how many image outputs were dropped.
- Each cell starts with `## [cell N] code` or `## [cell N] markdown`. Cite as `cell N`.
- Code is in ```python fences; text output follows under `OUTPUT:`. Long outputs are cut in the middle with `[... chars truncated ...]`: never quote across that marker and never assume what was in the gap.
- Charts and images are NOT in these files. Never describe a plot.
- Some notebooks are small because their outputs were cleared. If a result is not in the notebook, take it from the metrics files, not from guessing.
- Read the whole notebook in chunks (use Read with offset/limit). Do not skim and then summarize.
- **Ignore notebook titles, markdown headings and markdown text that describes settings.** Only code cells, code outputs, config files, metrics files and run folder names count as evidence. If a heading disagrees with code, say nothing about the heading.

## facts/metrics/metrics_long.csv
Columns `run,source,key,value`. One citable line per value. Use Grep for `^<run name>,` to get one run. Quote a full line.
- `source` says which original file the value came from (test_metrics.json, run_config.json, best_lora_adapters/adapter_config.json, training_history.csv).
- `row0.col` style keys are training-history rows (row0 = first epoch row).

## facts/metrics/metrics_table.csv
One row per run, wide format. For comparisons only. Do not quote from it in the ledger (use the long file). Columns starting `name_parsed_` come from folder names and are NOT confirmed facts.

## facts/metrics/history/<run>.csv
Copy of the training history. Cite rows from here or from the long file.

## facts/dataset/dataset_stats.md, facts/inventory.md, facts/fewshot/*
Generated summaries and copies of small result files. Quote lines exactly. Inventory flags (missing configs, partial files, small notebooks) are important caveats.

## Name decoding
See the project-glossary skill. Folder-name meanings are hypotheses until a config or the human confirms them.
