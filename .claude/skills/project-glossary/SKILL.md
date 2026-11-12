---
name: project-glossary
description: Human-confirmed project terms and the unverified meanings of folder names. Read to avoid inventing definitions.
---
# Project glossary

## A. Confirmed by the project owner (safe to use; cite as [D-xx] where one exists)
- Goal: a visual question answering (VQA) system for wildfires [D-07]
- Dataset: built by another paper on top of the existing FLAME-3 dataset [D-05]. TODO: paper title and dataset name
- Primary metric in the owner's view: Exact Match, because expected outputs are very limited [D-06]. TODO: confirm Exact Match is actually computed in the code and what higher/lower means for each metric
- Zero-shot / few-shot baseline uses a Qwen-based VLM and is kept for comparison only [D-03]
- Wildfire_Generative_Custom_VLM_Kaggle.ipynb is a trial run, not a final result [D-08]
- TODO: what "FireVLM" means in this project
- TODO: base LLM / vision encoder of EACH LoRA family. The owner mentioned "llama models" as well as Qwen; the config files decide which runs are which
- TODO: what the projector is and whether it is trained in every run (owner is unsure; the code must answer it, otherwise it stays an Open question)

## B. Folder-name decoding (UNVERIFIED hypotheses; never state as fact)
- `balanced_qkvo` : probably LoRA on q, k, v, o attention projections
- `efficient_qv` : probably LoRA on q and v only
- `high_capacity_all_linear` : probably LoRA on all linear layers
- `r4 / r8 / r16 / r32` : probably LoRA rank
- `qwen2_1_5b` in a folder name : probably a Qwen2-VL 1.5B base. Folders without it: base model UNKNOWN (the owner mentioned Llama-based models, so do NOT assume Qwen 3B; only the frozen benchmark folder says `qwen25vl_3b`)
These only become facts when an adapter_config / run_config / notebook cell shows them. If the config contradicts the name, the config wins and the contradiction goes in Anomalies.

## C. Rules
- If something is not in A, in decisions.md, or in a cited source, it is UNKNOWN.
