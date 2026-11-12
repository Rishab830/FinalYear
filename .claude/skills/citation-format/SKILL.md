---
name: citation-format
description: The fact schema and the "no quote, no claim" rule for the project knowledge base. Read before writing or checking any fact ledger entry or any cited sentence.
---
# Citation format

Core rule: **no quote, no claim.** A fact exists only if you can copy an exact passage from a source file that supports it. If you cannot, the answer is UNKNOWN. Guessing is the worst possible output here.

## One fact = one JSON object per line (JSONL)

```json
{"id":"balanced_qkvo_r16-007","claim":"LoRA rank r is 16","value":"16","source":"facts/metrics/metrics_long.csv","locator":"line starting with run name, key r","quote":"firevlm_lora_balanced_qkvo_r16,best_lora_adapters/adapter_config.json,r,16","type":"extracted"}
```

Fields (all required):
- `id`: SLUG-NNN, three digits, unique, with the slug you were given.
- `claim`: one atomic statement in plain words. One number or one setting per fact.
- `value`: the exact value as a string, or `"UNKNOWN"`.
- `source`: path relative to the project root of the file you read (e.g. `facts/nb/xxx.md`).
- `locator`: for notebooks, `cell N` using the `## [cell N]` header the quote sits under; for other files, a short description such as `row 3` or `line for key rougeL`.
- `quote`: text copied EXACTLY (character for character) from the source, 3 to 300 characters. Do not retype from memory, do not fix typos, do not merge two places. Copy a contiguous piece.
- `type`: `extracted`, `unknown`, or `computed`.

### type "unknown"
Use when a topic is expected but the sources do not show it. `value` must be `"UNKNOWN"`, `quote` can be empty, `claim` says what could not be determined and where you looked.

### type "computed" (rare)
Only if a number must be derived from other facts. Add `"inputs":["ID1","ID2"]` and `"method":"description of arithmetic"`. Inputs must exist. Prefer citing the two input facts and letting the verifier recompute.

### optional field
`"note"`: e.g. `"CONFLICT"` when two sources disagree.

## Rules
1. The quote must contain the evidence for the value. If the value is `16`, the quote must show 16.
2. Never state something because it is "standard" for Qwen, LoRA or any library.
3. Names are not evidence. A folder called `..._qkvo_...` does not prove the target modules; the config does.
4. If notebook code and a config file disagree, write both and a CONFLICT note.
5. Numbers keep their exact digits as written in the source.

## How sentences in knowledge.md cite
End every sentence, bullet and table row with the ids in square brackets: `LoRA rank is 16 [balanced_qkvo_r16-007].` Several: `[FS-004, FS-005]`. Decisions written by the human are cited as `[D-01]`. The only uncited text allowed is headings and the Open questions section.
