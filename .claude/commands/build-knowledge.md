---
description: Build knowledge.md from the project files using extraction scripts, reader agents, a writer agent and an adversarial verifier.
---
You are the orchestrator. You do NOT read project notebooks or model files yourself and you do NOT write knowledge.md yourself. You run the pipeline below, step by step, and report progress briefly after each step. Stop and tell the user if a step fails twice.

## Step 1: deterministic extraction
Run `bash scripts/stage0.sh`. Then read facts/inventory.md (only that file) to get the list of LoRA runs and the notebook-to-run map. If any run says "NO RUN MATCH", ask the user which notebook belongs to it before continuing.

## Step 2: reader agents (run in parallel where possible)
Invoke these subagents, giving each only what it needs:
- `dataset-reader`
- `fewshot-reader`
- `lora-reader`, ONE invocation per LoRA run. Tell it exactly: RUN=<folder name>, NOTEBOOK=facts/nb/<notebook>.md, SLUG=<folder name without the leading "firevlm_" and without "lora_", keep underscores>. Example: RUN=firevlm_lora_balanced_qkvo_r16 -> SLUG=balanced_qkvo_r16; RUN=firevlm_qwen2_1_5b_lora_efficient_qv_r8 -> SLUG=qwen2_1_5b_efficient_qv_r8. SLUGs must be unique.

## Step 3: validate the ledger
Run `python3 scripts/validate.py ledger`. For every ERROR, re-invoke the responsible reader and quote the error lines so it fixes only those facts (max 2 rounds). Facts that still fail are dropped automatically and will not reach the writer; list them for the user.

## Step 4: write
Invoke `knowledge-writer` (normal mode). Then run `python3 scripts/validate.py doc`.
If there are ERRORs (uncited lines, nonexistent ids), invoke `knowledge-writer` in FIX MODE with the error list pasted in, then re-run the check.

## Step 5: verify (loop, max 2 rounds)
Invoke `claim-verifier`. Read the OVERALL line of facts/verification_report.md.
- If FAIL: invoke `knowledge-writer` in FIX MODE ("fix the lines flagged in facts/verification_report.md"), re-run `python3 scripts/validate.py doc`, then invoke `claim-verifier` again.
- After 2 rounds stop regardless.

## Step 6: report to the user
Give: facts count, rejected facts count, verifier verdict, remaining FAIL/UNSUPPORTED lines (copy them), the Open questions list, and the Anomalies list. End by telling the user to review the Open questions and fill decisions.md or the glossary, then re-run /build-knowledge.
Never hide failures and never describe the document as "verified" unless the last verifier line was OVERALL: PASS.
