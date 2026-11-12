# Project: FireVLM final year project knowledge base

Goal: produce an accurate `knowledge.md` describing this project (dataset, models, training runs, results).

## Non-negotiable rules for every agent
- Numbers and configs come from generated files in `facts/` (made by scripts in `scripts/`), never from memory or general knowledge.
- Every claim needs a verbatim quote from a source file (see .claude/skills/citation-format). No quote, no claim. Write UNKNOWN instead of guessing.
- Never open raw `.ipynb` files or model weight files. Use `facts/nb/*.md` and `facts/metrics/*`.
- `decisions.md` and the glossary are written by the human. Never invent reasons for design decisions.
- Do not edit anything under `models/` or `notebooks/`.

## Layout
- `models/`, `notebooks/`: original project files (read-only)
- `scripts/`: deterministic extraction + validators (`stage0.sh`, `validate.py`)
- `facts/`: generated evidence; `facts/ledger/*.jsonl` per-agent claims; `facts/facts.jsonl` validated merge; `facts/verification_report.md`
- `knowledge.md`: the final output
- `.claude/agents`, `.claude/skills`, `.claude/commands/build-knowledge.md`: the pipeline

Run the whole pipeline with `/build-knowledge`.
