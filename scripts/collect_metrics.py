#!/usr/bin/env python3
"""Stage 0b: gather every number/config from json/csv files into citable tables. No LLM.
Outputs: facts/metrics/metrics_long.csv (run,source,key,value - one citable line per value)
         facts/metrics/metrics_table.csv (one row per run, for comparisons)
         facts/metrics/history/<run>.csv, facts/fewshot/* (copies of small few-shot result files)"""
import csv, json, re, shutil, sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
models = root / "models"
out = root / "facts" / "metrics"
(out / "history").mkdir(parents=True, exist_ok=True)
fewshot = root / "facts" / "fewshot"
fewshot.mkdir(parents=True, exist_ok=True)

def flatten(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from flatten(v, f"{p}.{k}" if p else str(k))
    elif isinstance(o, list):
        if all(not isinstance(i, (dict, list)) for i in o):
            yield p, ";".join(map(str, o))
        else:
            for i, v in enumerate(o):
                yield from flatten(v, f"{p}[{i}]")
    else:
        yield p, o

def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

long_rows, wide_rows = [], []
for d in sorted(p for p in models.iterdir() if p.is_dir()):
    if not (d / "best_lora_adapters").exists():
        for f in d.iterdir():  # few-shot / non-LoRA folder: copy small result files
            if f.is_file() and f.suffix in (".csv", ".json") and "predictions" not in f.name:
                shutil.copy(f, fewshot / f.name)
        continue
    run, row = d.name, {"run": d.name}
    m = re.search(r"lora_(balanced_qkvo|efficient_qv|high_capacity_all_linear)_r(\d+)$", run)
    row["name_parsed_scheme"] = m.group(1) if m else ""
    row["name_parsed_rank"] = m.group(2) if m else ""
    row["name_parsed_family"] = "qwen2_1_5b" if "qwen2_1_5b" in run else "unlabeled"

    json_sources = [("test_metrics.json", d / "test_metrics.json", "test_metrics"),
                    ("run_config.json", d / "run_config.json", "run_config")]
    for n, p in enumerate(sorted((d / "best_lora_adapters").rglob("adapter_config.json"))):
        json_sources.append((p.relative_to(d).as_posix(), p, "adapter_config" + (str(n + 1) if n else "")))
    for src, path, prefix in json_sources:
        if not path.exists():
            continue
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            long_rows.append([run, src, "__parse_error__", str(e)]); continue
        for k, v in flatten(obj):
            long_rows.append([run, src, k, v]); row[f"{prefix}.{k}"] = v

    hist = d / "training_history.csv"
    if hist.exists():
        shutil.copy(hist, out / "history" / f"{run}.csv")
        with open(hist, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        row["history_n_rows"] = len(rows)
        for i, r in enumerate(rows):
            for col, val in r.items():
                long_rows.append([run, "training_history.csv", f"row{i}.{col}", val])
        for col in (rows[0].keys() if rows else []):
            vals = [num(r[col]) for r in rows if num(r[col]) is not None]
            if vals:
                row[f"history_min.{col}"], row[f"history_max.{col}"], row[f"history_last.{col}"] = min(vals), max(vals), vals[-1]
    wide_rows.append(row)

with open(out / "metrics_long.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, lineterminator="\n"); w.writerow(["run", "source", "key", "value"]); w.writerows(long_rows)
cols = []
for r in wide_rows:
    cols += [c for c in r if c not in cols]
with open(out / "metrics_table.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n"); w.writeheader(); w.writerows(wide_rows)
print(f"{len(wide_rows)} LoRA runs, {len(long_rows)} long rows, {len(cols)} table columns")
print(f"few-shot files copied: {sorted(p.name for p in fewshot.iterdir())}")
