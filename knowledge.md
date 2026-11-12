# Project knowledge base

## 1. Project overview

The project is a visual question answering (VQA) system for wildfires [D-07].
The dataset is built by another paper on top of the FLAME-3 dataset [DS-001, D-05].
The trial notebook Wildfire_Generative_Custom_VLM_Kaggle.ipynb was an early experiment, not the final pipeline [DS-034, D-08].

## 2. Dataset and splits

The final pipeline loaded 207,306 raw annotation rows [DS-005].
The pipeline filtered for site prefix 'Sycan', leaving 25,092 rows [DS-006, DS-008].
The raw annotations contained three sites: Shoetank (120,360 rows), Wilamette (61,846 rows), and Sycan (25,092 rows) [DS-007].

The preprocessing identified 738 unique source image pairs after strict radiometric pairing [DS-009].
The audit CSV contains 738 rows documenting source-to-Kaggle image pairing [DS-010].
The audit CSV is byte-identical across all 16 LoRA runs [DS-011].
The audit CSV columns are: source_image_key, source_collection, source_class, source_filename, canonical_stem, match_mode, max_signature_error_celsius, expected_signature, observed_signature [DS-015].

The source image pairs include 622 Fire class images and 116 No Fire class images [DS-012].
The source collections are Sycan 2A (364 rows) and Sycan 2D (374 rows) [DS-016].
Each image pair has exactly 34 questions [DS-013].

The pairing audit used radiometric TIFF signatures, with 734 exact matches and 4 tolerance matches [DS-014, DS-018].
The preprocessing includes deduplication by subset: image_id, question_id, question, answer, rgb_path, thermal_path [DS-036].
The preprocessing verifies no duplicate questions within a source image and no contradictory answers [DS-037].
Canonical image paths are stored in rgb_resolved and thermal_resolved fields [DS-038].

The split procedure groups all questions for each image into one split [DS-022].
Validation fraction is 0.10 and test fraction is 0.10 [DS-022].
The random seed is 42 [DS-039].
The train split contains 592 unique images [DS-023].
The validation split contains 73 unique images [DS-024].
The test split contains 73 unique images [DS-025].
The splits are verified to have no image leakage [DS-026].

Train manifest contains 20,128 rows [DS-019].
Validation manifest contains 2,482 rows [DS-020].
The train, validation, and test manifest files are byte-identical across all 8 Qwen2-1.5B LoRA runs [DS-027].

Manifest records contain 19 top-level keys [DS-028].
Each manifest record includes a temp_summary field with 7 radiometric statistics (min, max, mean, std, top3_mean, pct_over_200, pct_over_400) [DS-029].
The answer field contains ground truth answer text [DS-030].
The options field contains a list of candidate answers [DS-031].
An example question category is 'Presence and Detection' [DS-043, DS-044].

Test predictions CSV for LoRA runs contains 2,482 rows with 12 columns: pair_id, source_collection, canonical_stem, rgb_path, thermal_path, category, question, target, prediction, exact_match, bleu, rouge_l [DS-032, DS-033].

## 3. Pipeline and models

### Base models and vision encoder

Llama-based runs use meta-llama/Llama-3.2-1B-Instruct [balanced_qkvo_r16-001].
Qwen2-VL-based runs use mit-han-lab/Qwen2-VL-1.5B-Instruct [qwen2_1_5b_balanced_qkvo_r16-001].

All LoRA runs use frozen CLIP vision encoder openai/clip-vit-base-patch32 [balanced_qkvo_r16-021].
The CLIP encoder produces 49 patch tokens per image [efficient_qv_r8-023].

All LoRA runs train separate RGB and thermal projectors with hidden dimension 1024 [balanced_qkvo_r16-022, balanced_qkvo_r16-006].
All LoRA runs train type embeddings (rgb_type, thermal_type) [balanced_qkvo_r16-007].

### LoRA configurations

The project uses LoRA fine-tuning with different configurations compared against each other [D-01].
Rank sweeps were not run equally: balanced_qkvo has more ranks than efficient_qv and all_linear [D-02].
The efficient (q,v) setup did not show good results in other experiments [D-02].

All LoRA runs use 4-bit NF4 quantization with double quantization [balanced_qkvo_r16-018].
All LoRA runs use PEFT type LORA and task type CAUSAL_LM [balanced_qkvo_r16-049, balanced_qkvo_r16-048].
All LoRA runs use bias setting 'none' [balanced_qkvo_r16-047].

| Run | Base LLM | Rank | Alpha | Dropout | Target modules | Projector hidden | Optimizer | LR | Scheduler | Warmup | Epochs planned | Epochs completed | Batch size | Grad accum | Effective BS | Grad clip | Best val epoch | [structural]
|-----|----------|------|-------|---------|----------------|-----------------|-----------|-----|-----------|--------|---------------|-----------------|------------|-----------|-------------|----------|----------------|-------------|
| firevlm_lora_balanced_qkvo_r16 | Llama-3.2-1B [balanced_qkvo_r16-001] | 16 [balanced_qkvo_r16-002] | 32 [balanced_qkvo_r16-003] | 0.05 [balanced_qkvo_r16-004] | q_proj, k_proj, v_proj, o_proj [balanced_qkvo_r16-005] | 1024 [balanced_qkvo_r16-022] | AdamW [balanced_qkvo_r16-008] | 1.5e-4 [balanced_qkvo_r16-009] | cosine+warmup [balanced_qkvo_r16-011] | 0.05 [balanced_qkvo_r16-012] | 8 [balanced_qkvo_r16-013] | 8 [balanced_qkvo_r16-014] | 4 [balanced_qkvo_r16-015] | 4 [balanced_qkvo_r16-016] | 16 [balanced_qkvo_r16-017] | 1.0 [balanced_qkvo_r16-025] | 6 [balanced_qkvo_r16-026] |
| firevlm_lora_balanced_qkvo_r32 | Llama-3.2-1B [balanced_qkvo_r32-001] | 32 [balanced_qkvo_r32-002] | 64 [balanced_qkvo_r32-003] | 0.05 [balanced_qkvo_r32-004] | q_proj, k_proj, v_proj, o_proj [balanced_qkvo_r32-005] | 1024 [balanced_qkvo_r32-021] | AdamW [balanced_qkvo_r32-008] | 1.5e-4 [balanced_qkvo_r32-009] | cosine+warmup [balanced_qkvo_r32-010] | 0.05 [balanced_qkvo_r32-011] | 10 [balanced_qkvo_r32-013] | 8 [balanced_qkvo_r32-014] | 4 [balanced_qkvo_r32-015] | 4 [balanced_qkvo_r32-016] | 16 [balanced_qkvo_r32-017] | 1.0 [balanced_qkvo_r32-018] | 5 [balanced_qkvo_r32-028] |
| firevlm_lora_balanced_qkvo_r4 | Llama-3.2-1B [balanced_qkvo_r4-001] | 4 [balanced_qkvo_r4-002] | 8 [balanced_qkvo_r4-003] | 0.05 [balanced_qkvo_r4-004] | q_proj, k_proj, v_proj, o_proj [balanced_qkvo_r4-005] | 1024 [balanced_qkvo_r4-011] | UNKNOWN [balanced_qkvo_r4-013] | 1.5e-4 [balanced_qkvo_r4-014] | UNKNOWN [balanced_qkvo_r4-015] | 0.05 [balanced_qkvo_r4-026] | 10 [balanced_qkvo_r4-020] | 8 [balanced_qkvo_r4-021] | 4 [balanced_qkvo_r4-016] | 4 [balanced_qkvo_r4-017] | 16 [balanced_qkvo_r4-018] | 1.0 [balanced_qkvo_r4-025] | 5 [balanced_qkvo_r4-030] |
| firevlm_lora_balanced_qkvo_r8 | Llama-3.2-1B [balanced_qkvo_r8-001] | 8 [balanced_qkvo_r8-002] | 16 [balanced_qkvo_r8-003] | 0.05 [balanced_qkvo_r8-004] | q_proj, k_proj, v_proj, o_proj [balanced_qkvo_r8-005] | 1024 [balanced_qkvo_r8-032] | AdamW [balanced_qkvo_r8-008] | 1.5e-4 [balanced_qkvo_r8-009] | cosine+warmup [balanced_qkvo_r8-010] | 0.05 [balanced_qkvo_r8-011] | 10 [balanced_qkvo_r8-014] | 9 [balanced_qkvo_r8-015] | 4 [balanced_qkvo_r8-016] | 4 [balanced_qkvo_r8-017] | 16 [balanced_qkvo_r8-018] | 1.0 [balanced_qkvo_r8-013] | 6 [balanced_qkvo_r8-023] |
| firevlm_lora_efficient_qv_r16 | Llama-3.2-1B [efficient_qv_r16-001] | 16 [efficient_qv_r16-002] | 32 [efficient_qv_r16-003] | 0.05 [efficient_qv_r16-004] | q_proj, v_proj [efficient_qv_r16-005] | 1024 [efficient_qv_r16-020] | UNKNOWN [efficient_qv_r16-008] | 1.5e-4 [efficient_qv_r16-009] | cosine+warmup [efficient_qv_r16-010] | 0.05 [efficient_qv_r16-011] | 10 [efficient_qv_r16-013] | 7 [efficient_qv_r16-014] | 4 [efficient_qv_r16-016] | 4 [efficient_qv_r16-017] | 16 [efficient_qv_r16-018] | 1.0 [efficient_qv_r16-030] | 4 [efficient_qv_r16-024] |
| firevlm_lora_efficient_qv_r8 | Llama-3.2-1B [efficient_qv_r8-001] | 8 [efficient_qv_r8-002] | 16 [efficient_qv_r8-003] | 0.05 [efficient_qv_r8-004] | q_proj, v_proj [efficient_qv_r8-005] | 1024 [efficient_qv_r8-024] | AdamW [efficient_qv_r8-009] | 2e-4 [efficient_qv_r8-010] | cosine+warmup [efficient_qv_r8-012] | 0.05 [efficient_qv_r8-013] | 100 [efficient_qv_r8-014] | 10 [efficient_qv_r8-015] | 4 [efficient_qv_r8-016] | 4 [efficient_qv_r8-017] | 16 [efficient_qv_r8-018] | 1.0 [efficient_qv_r8-028] | 6 [efficient_qv_r8-036] |
| firevlm_lora_high_capacity_all_linear_r16 | Llama-3.2-1B [high_capacity_all_linear_r16-001] | 16 [high_capacity_all_linear_r16-002] | 32 [high_capacity_all_linear_r16-003] | 0.05 [high_capacity_all_linear_r16-004] | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj [high_capacity_all_linear_r16-005] | 1024 [high_capacity_all_linear_r16-021] | AdamW [high_capacity_all_linear_r16-007] | 1e-4 [high_capacity_all_linear_r16-008] | cosine+warmup [high_capacity_all_linear_r16-010] | 0.05 [high_capacity_all_linear_r16-011] | 100 [high_capacity_all_linear_r16-012] | 9 [high_capacity_all_linear_r16-013] | 2 [high_capacity_all_linear_r16-014] | 8 [high_capacity_all_linear_r16-015] | 16 [high_capacity_all_linear_r16-016] | 1.0 [high_capacity_all_linear_r16-029] | 6 [high_capacity_all_linear_r16-024] |
| firevlm_lora_high_capacity_all_linear_r8 | Llama-3.2-1B [high_capacity_all_linear_r8-001] | 8 [high_capacity_all_linear_r8-002] | 16 [high_capacity_all_linear_r8-003] | 0.05 [high_capacity_all_linear_r8-004] | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj [high_capacity_all_linear_r8-005] | 1024 [high_capacity_all_linear_r8-027] | AdamW [high_capacity_all_linear_r8-007] | 1.5e-4 [high_capacity_all_linear_r8-008] | cosine+warmup [high_capacity_all_linear_r8-009] | 0.05 [high_capacity_all_linear_r8-010] | 10 [high_capacity_all_linear_r8-018] | 9 [high_capacity_all_linear_r8-019] | 4 [high_capacity_all_linear_r8-012] | 4 [high_capacity_all_linear_r8-013] | 16 [high_capacity_all_linear_r8-014] | 1.0 [high_capacity_all_linear_r8-015] | 6 [high_capacity_all_linear_r8-031] |
| firevlm_qwen2_1_5b_lora_balanced_qkvo_r16 | Qwen2-VL-1.5B [qwen2_1_5b_balanced_qkvo_r16-001] | 16 [qwen2_1_5b_balanced_qkvo_r16-002] | 32 [qwen2_1_5b_balanced_qkvo_r16-003] | 0.05 [qwen2_1_5b_balanced_qkvo_r16-004] | q_proj, k_proj, v_proj, o_proj [qwen2_1_5b_balanced_qkvo_r16-005] | 1024 [qwen2_1_5b_balanced_qkvo_r16-030] | AdamW [qwen2_1_5b_balanced_qkvo_r16-009] | 2e-4 [qwen2_1_5b_balanced_qkvo_r16-010] | cosine+warmup [qwen2_1_5b_balanced_qkvo_r16-013] | 0.05 [qwen2_1_5b_balanced_qkvo_r16-012] | 100 [qwen2_1_5b_balanced_qkvo_r16-014] | 10 [qwen2_1_5b_balanced_qkvo_r16-015] | 4 [qwen2_1_5b_balanced_qkvo_r16-018] | 4 [qwen2_1_5b_balanced_qkvo_r16-019] | 16 [qwen2_1_5b_balanced_qkvo_r16-020] | 1.0 [qwen2_1_5b_balanced_qkvo_r16-021] | 7 [qwen2_1_5b_balanced_qkvo_r16-034] |
| firevlm_qwen2_1_5b_lora_balanced_qkvo_r32 | Qwen2-VL-1.5B [qwen2_1_5b_balanced_qkvo_r32-001] | 32 [qwen2_1_5b_balanced_qkvo_r32-002] | 64 [qwen2_1_5b_balanced_qkvo_r32-003] | 0.05 [qwen2_1_5b_balanced_qkvo_r32-004] | q_proj, k_proj, v_proj, o_proj [qwen2_1_5b_balanced_qkvo_r32-005] | 1024 [qwen2_1_5b_balanced_qkvo_r32-035] | AdamW [qwen2_1_5b_balanced_qkvo_r32-007] | 2e-4 [qwen2_1_5b_balanced_qkvo_r32-008] | cosine+warmup [qwen2_1_5b_balanced_qkvo_r32-009] | 0.05 [qwen2_1_5b_balanced_qkvo_r32-010] | 100 [qwen2_1_5b_balanced_qkvo_r32-011] | 13 [qwen2_1_5b_balanced_qkvo_r32-012] | 4 [qwen2_1_5b_balanced_qkvo_r32-013] | 4 [qwen2_1_5b_balanced_qkvo_r32-014] | 16 [qwen2_1_5b_balanced_qkvo_r32-015] | 1.0 [qwen2_1_5b_balanced_qkvo_r32-031] | 5 [qwen2_1_5b_balanced_qkvo_r32-022] |
| firevlm_qwen2_1_5b_lora_balanced_qkvo_r4 | Qwen2-VL-1.5B [qwen2_1_5b_balanced_qkvo_r4-001] | 4 [qwen2_1_5b_balanced_qkvo_r4-002] | 8 [qwen2_1_5b_balanced_qkvo_r4-003] | 0.05 [qwen2_1_5b_balanced_qkvo_r4-004] | q_proj, k_proj, v_proj, o_proj [qwen2_1_5b_balanced_qkvo_r4-005] | 1024 [qwen2_1_5b_balanced_qkvo_r4-007] | AdamW [qwen2_1_5b_balanced_qkvo_r4-008] | 2e-4 [qwen2_1_5b_balanced_qkvo_r4-009] | cosine+warmup [qwen2_1_5b_balanced_qkvo_r4-010] | 0.05 [qwen2_1_5b_balanced_qkvo_r4-011] | 100 [qwen2_1_5b_balanced_qkvo_r4-012] | 14 [qwen2_1_5b_balanced_qkvo_r4-013] | 4 [qwen2_1_5b_balanced_qkvo_r4-016] | 4 [qwen2_1_5b_balanced_qkvo_r4-017] | 16 [qwen2_1_5b_balanced_qkvo_r4-018] | 1.0 [qwen2_1_5b_balanced_qkvo_r4-024] | 11 [qwen2_1_5b_balanced_qkvo_r4-025] |
| firevlm_qwen2_1_5b_lora_balanced_qkvo_r8 | Qwen2-VL-1.5B [qwen2_1_5b_balanced_qkvo_r8-001] | 8 [qwen2_1_5b_balanced_qkvo_r8-002] | 16 [qwen2_1_5b_balanced_qkvo_r8-003] | 0.05 [qwen2_1_5b_balanced_qkvo_r8-004] | q_proj, k_proj, v_proj, o_proj [qwen2_1_5b_balanced_qkvo_r8-005] | 1024 [balanced_qkvo_r16-022] | AdamW [qwen2_1_5b_balanced_qkvo_r8-007] | 2e-4 [qwen2_1_5b_balanced_qkvo_r8-008] | cosine+warmup [qwen2_1_5b_balanced_qkvo_r8-009] | 0.05 [qwen2_1_5b_balanced_qkvo_r8-023] | 100 [qwen2_1_5b_balanced_qkvo_r8-010] | 13 [qwen2_1_5b_balanced_qkvo_r8-011] | 4 [qwen2_1_5b_balanced_qkvo_r8-012] | 4 [qwen2_1_5b_balanced_qkvo_r8-013] | 16 [qwen2_1_5b_balanced_qkvo_r8-012, qwen2_1_5b_balanced_qkvo_r8-013] | 1.0 [qwen2_1_5b_balanced_qkvo_r8-024] | 8 [qwen2_1_5b_balanced_qkvo_r8-017] |
| firevlm_qwen2_1_5b_lora_efficient_qv_r16 | Qwen2-VL-1.5B [qwen2_1_5b_efficient_qv_r16-001] | 16 [qwen2_1_5b_efficient_qv_r16-002] | 32 [qwen2_1_5b_efficient_qv_r16-003] | 0.05 [qwen2_1_5b_efficient_qv_r16-004] | q_proj, v_proj [qwen2_1_5b_efficient_qv_r16-005] | 1024 [qwen2_1_5b_efficient_qv_r16-027] | AdamW [qwen2_1_5b_efficient_qv_r16-009] | 2e-4 [qwen2_1_5b_efficient_qv_r16-008] | cosine+warmup [qwen2_1_5b_efficient_qv_r16-012] | 0.05 [qwen2_1_5b_efficient_qv_r16-011] | 100 [qwen2_1_5b_efficient_qv_r16-018] | 12 [qwen2_1_5b_efficient_qv_r16-019] | 4 [qwen2_1_5b_efficient_qv_r16-013] | 4 [qwen2_1_5b_efficient_qv_r16-014] | 16 [qwen2_1_5b_efficient_qv_r16-015] | 1.0 [qwen2_1_5b_efficient_qv_r16-016] | 9 [qwen2_1_5b_efficient_qv_r16-022] |
| firevlm_qwen2_1_5b_lora_efficient_qv_r8 | Qwen2-VL-1.5B [qwen2_1_5b_efficient_qv_r8-001] | 8 [qwen2_1_5b_efficient_qv_r8-002] | 16 [qwen2_1_5b_efficient_qv_r8-003] | 0.05 [qwen2_1_5b_efficient_qv_r8-004] | q_proj, v_proj [qwen2_1_5b_efficient_qv_r8-005] | 1024 [balanced_qkvo_r16-022] | AdamW [qwen2_1_5b_efficient_qv_r8-008] | 2e-4 [qwen2_1_5b_efficient_qv_r8-009] | cosine+warmup [qwen2_1_5b_efficient_qv_r8-010] | 0.05 [qwen2_1_5b_efficient_qv_r8-026] | 100 [qwen2_1_5b_efficient_qv_r8-011] | 12 [qwen2_1_5b_efficient_qv_r8-012] | 4 [qwen2_1_5b_efficient_qv_r8-013] | 4 [qwen2_1_5b_efficient_qv_r8-014] | 16 [qwen2_1_5b_efficient_qv_r8-013, qwen2_1_5b_efficient_qv_r8-014] | 1.0 [qwen2_1_5b_efficient_qv_r8-027] | 9 [qwen2_1_5b_efficient_qv_r8-024] |
| firevlm_qwen2_1_5b_lora_high_capacity_all_linear_r16 | Qwen2-VL-1.5B [qwen2_1_5b_high_capacity_all_linear_r16-001] | 16 [qwen2_1_5b_high_capacity_all_linear_r16-002] | 32 [qwen2_1_5b_high_capacity_all_linear_r16-003] | 0.05 [qwen2_1_5b_high_capacity_all_linear_r16-004] | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj [qwen2_1_5b_high_capacity_all_linear_r16-005] | 1024 [qwen2_1_5b_high_capacity_all_linear_r16-039] | AdamW [qwen2_1_5b_high_capacity_all_linear_r16-010] | 2e-4 [qwen2_1_5b_high_capacity_all_linear_r16-011] | cosine+warmup [qwen2_1_5b_high_capacity_all_linear_r16-012] | 0.05 [qwen2_1_5b_high_capacity_all_linear_r16-013] | 100 [qwen2_1_5b_high_capacity_all_linear_r16-014] | 13 [qwen2_1_5b_high_capacity_all_linear_r16-015] | 4 [qwen2_1_5b_high_capacity_all_linear_r16-016] | 4 [qwen2_1_5b_high_capacity_all_linear_r16-017] | 16 [qwen2_1_5b_high_capacity_all_linear_r16-018] | 1.0 [qwen2_1_5b_high_capacity_all_linear_r16-034] | 8 [qwen2_1_5b_high_capacity_all_linear_r16-032] |
| firevlm_qwen2_1_5b_lora_high_capacity_all_linear_r8 | Qwen2-VL-1.5B [qwen2_1_5b_high_capacity_all_linear_r8-001] | 8 [qwen2_1_5b_high_capacity_all_linear_r8-003] | 16 [qwen2_1_5b_high_capacity_all_linear_r8-004] | 0.05 [qwen2_1_5b_high_capacity_all_linear_r8-005] | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj [qwen2_1_5b_high_capacity_all_linear_r8-006] | 1024 [qwen2_1_5b_high_capacity_all_linear_r8-033] | AdamW [qwen2_1_5b_high_capacity_all_linear_r8-010] | 2e-4 [qwen2_1_5b_high_capacity_all_linear_r8-011] | cosine+warmup [qwen2_1_5b_high_capacity_all_linear_r8-012] | 0.05 [qwen2_1_5b_high_capacity_all_linear_r8-013] | 100 [qwen2_1_5b_high_capacity_all_linear_r8-014] | 15 [qwen2_1_5b_high_capacity_all_linear_r8-015] | 4 [qwen2_1_5b_high_capacity_all_linear_r8-016] | 4 [qwen2_1_5b_high_capacity_all_linear_r8-017] | 16 [qwen2_1_5b_high_capacity_all_linear_r8-016, qwen2_1_5b_high_capacity_all_linear_r8-017] | 1.0 [qwen2_1_5b_high_capacity_all_linear_r8-030] | 9 [qwen2_1_5b_high_capacity_all_linear_r8-023] |

All LoRA runs use weight decay 0.01 [balanced_qkvo_r16-010].
Early stopping patience is 3 for all runs except Llama balanced_qkvo_r16 which uses patience 2 [balanced_qkvo_r32-026, balanced_qkvo_r4-022, efficient_qv_r16-031, efficient_qv_r8-033, high_capacity_all_linear_r16-031, high_capacity_all_linear_r8-020, qwen2_1_5b_balanced_qkvo_r16-016, qwen2_1_5b_balanced_qkvo_r32-024, qwen2_1_5b_balanced_qkvo_r4-014, qwen2_1_5b_balanced_qkvo_r8-025, qwen2_1_5b_efficient_qv_r16-020, qwen2_1_5b_efficient_qv_r8-018, qwen2_1_5b_high_capacity_all_linear_r16-024, qwen2_1_5b_high_capacity_all_linear_r8-024, balanced_qkvo_r16-056].
Training prompt includes system message: "You are an expert wildfire intelligence system analyzing multimodal aerial imagery." [balanced_qkvo_r4-029].
Training prompts do not include options or thermal summary [balanced_qkvo_r16-023, balanced_qkvo_r16-024].
Max target tokens and max generate tokens are 32 for all runs [balanced_qkvo_r16-019, balanced_qkvo_r16-020].

Best checkpoint selection uses minimum validation loss [balanced_qkvo_r16-026].

## 4. Few-shot baseline

The few-shot baseline uses a frozen Qwen2.5-VL-3B-Instruct model [FS-001, D-03].
The baseline was kept for comparison only; comparing Llama and Qwen in zero/few-shot is not appropriate [D-03].

Shot counts evaluated range from 0 through 10 [FS-002].
Shot means the number of example question-answer pairs shown before the test query [FS-003].
Support examples were selected from the same category as the query when possible [FS-004].
Each query received a deterministic ordered 10-example support pool [FS-005].
k-shot runs used the first k examples from the same 10-example pool [FS-006].

System prompt describes the model as a wildfire analyst [FS-007].
Each demonstration and query includes both RGB and thermal images [FS-008].
Decoding uses greedy search with max_new_tokens=32, do_sample=False, num_beams=1 [FS-009, FS-010, FS-011].
Model quantization uses 4-bit NF4 [FS-012].

Metrics computed include exact_match, normalized_exact_match, bleu, rouge_l, answer_rate, latency_seconds, and input_tokens [FS-013, FS-014, FS-015, FS-016, FS-017, FS-018, FS-019].

The model was evaluated in frozen mode with no gradient computation [FS-072].
Number of weight updates is 0 [FS-059].
Total generations across all shot counts is 2,200 [FS-058].
Test set size per shot count is 200 queries [FS-057].
Bootstrap confidence intervals used 1,000 samples [FS-056].

Category-level results computed for: Classification, Cross-Modal Reasoning, Distribution and Segmentation, Flight Planning, Localization and Direction, and Presence and Detection [FS-062, FS-063, FS-064, FS-065, FS-066, FS-067].

The notebook completed all 11 shot counts from 0 through 10 [FS-071].

## 5. Results

### Few-shot baseline results

| Shot count | Exact match | ROUGE-L | Mean latency (s) | Mean input tokens | [structural]
|------------|-------------|---------|------------------|-------------------|-------------|
| 0 | 0.15 [FS-020] | 0.24378422484115084 [FS-031] | 0.40358946204999313 [FS-052] | 210.785 [FS-054] |
| 1 | 0.2 [FS-021] | 0.3112567783094099 [FS-032] | UNKNOWN | UNKNOWN |
| 2 | 0.235 [FS-022] | 0.3280119047619047 [FS-033] | UNKNOWN | UNKNOWN |
| 3 | 0.235 [FS-023] | 0.32900000000000007 [FS-034] | UNKNOWN | UNKNOWN |
| 4 | 0.245 [FS-024] | 0.3362222222222222 [FS-035] | UNKNOWN | UNKNOWN |
| 5 | 0.255 [FS-025] | 0.3420793650793651 [FS-036] | UNKNOWN | UNKNOWN |
| 6 | 0.26 [FS-026] | 0.3474285714285714 [FS-037] | UNKNOWN | UNKNOWN |
| 7 | 0.27 [FS-027] | 0.35850000000000004 [FS-038] | UNKNOWN | UNKNOWN |
| 8 | 0.26 [FS-028] | 0.3468333333333334 [FS-039] | UNKNOWN | UNKNOWN |
| 9 | 0.275 [FS-029] | 0.36683333333333334 [FS-040] | UNKNOWN | UNKNOWN |
| 10 | 0.28 [FS-030] | 0.376 [FS-041] | 2.231058431765 [FS-053] | 1919.29 [FS-055] |

Best shot count by exact_match metric is 10 [FS-060].
Best shot count by rouge_l metric is 10 [FS-061].

### Llama-3.2-1B LoRA results

| Run | Test exact match | Test BLEU | Test ROUGE-L | Test samples | [structural]
|-----|------------------|-----------|--------------|--------------|-------------|
| firevlm_lora_balanced_qkvo_r16 | 0.8815471394037067 [balanced_qkvo_r16-043] | 0.14565290091097322 [balanced_qkvo_r16-044] | 0.907313610375657 [balanced_qkvo_r16-045] | 2482 [balanced_qkvo_r16-046] |
| firevlm_lora_balanced_qkvo_r32 | 0.8682514101531024 [balanced_qkvo_r32-031] | 0.13982483811524185 [balanced_qkvo_r32-032] | 0.897375388511569 [balanced_qkvo_r32-033] | 2482 [balanced_qkvo_r32-034] |
| firevlm_lora_balanced_qkvo_r4 | 0.871474617244158 [balanced_qkvo_r4-047] | 0.14671102734219488 [balanced_qkvo_r4-048] | 0.8982080503434252 [balanced_qkvo_r4-049] | 2482 [balanced_qkvo_r4-050] |
| firevlm_lora_balanced_qkvo_r8 | 0.8847703464947623 [balanced_qkvo_r8-024] | 0.14579161116912756 [balanced_qkvo_r8-025] | 0.9121282759679215 [balanced_qkvo_r8-026] | 2482 [balanced_qkvo_r8-027] |
| firevlm_lora_efficient_qv_r16 | 0.8763094278807413 [efficient_qv_r16-026] | 0.14494801047911868 [efficient_qv_r16-027] | 0.9042592379417521 [efficient_qv_r16-028] | 2482 [efficient_qv_r16-029] |
| firevlm_lora_efficient_qv_r8 | 0.8638195004029009 [efficient_qv_r8-038] | 0.145080802066314 [efficient_qv_r8-039] | 0.8910853382448871 [efficient_qv_r8-040] | 2482 [efficient_qv_r8-041] |
| firevlm_lora_high_capacity_all_linear_r16 | 0.854552780016116 [high_capacity_all_linear_r16-025] | 0.14473430834259518 [high_capacity_all_linear_r16-026] | 0.8877940217182764 [high_capacity_all_linear_r16-027] | 2482 [high_capacity_all_linear_r16-028] |
| firevlm_lora_high_capacity_all_linear_r8 | 0.8763094278807413 [high_capacity_all_linear_r8-033] | 0.14303217021981574 [high_capacity_all_linear_r8-034] | 0.9061778135911899 [high_capacity_all_linear_r8-035] | 2482 [high_capacity_all_linear_r8-036] |

### Qwen2-VL-1.5B LoRA results

| Run | Test exact match | Test BLEU | Test ROUGE-L | Test samples | [structural]
|-----|------------------|-----------|--------------|--------------|-------------|
| firevlm_qwen2_1_5b_lora_balanced_qkvo_r16 | 0.8412570507655117 [qwen2_1_5b_balanced_qkvo_r16-036] | 0.1459590245474915 [qwen2_1_5b_balanced_qkvo_r16-037] | 0.87728694217413 [qwen2_1_5b_balanced_qkvo_r16-038] | 2482 [qwen2_1_5b_balanced_qkvo_r16-039] |
| firevlm_qwen2_1_5b_lora_balanced_qkvo_r32 | 0.854955680902498 [qwen2_1_5b_balanced_qkvo_r32-026] | 0.14002498442321482 [qwen2_1_5b_balanced_qkvo_r32-027] | 0.8868980468899887 [qwen2_1_5b_balanced_qkvo_r32-028] | 2482 [qwen2_1_5b_balanced_qkvo_r32-029] |
| firevlm_qwen2_1_5b_lora_balanced_qkvo_r4 | 0.8662369057211926 [qwen2_1_5b_balanced_qkvo_r4-027] | 0.1403136761760804 [qwen2_1_5b_balanced_qkvo_r4-028] | 0.8978943632247419 [qwen2_1_5b_balanced_qkvo_r4-029] | 2482 [qwen2_1_5b_balanced_qkvo_r4-030] |
| firevlm_qwen2_1_5b_lora_balanced_qkvo_r8 | 0.863013698630137 [qwen2_1_5b_balanced_qkvo_r8-018] | 0.14969716674577332 [qwen2_1_5b_balanced_qkvo_r8-019] | 0.8927995855876598 [qwen2_1_5b_balanced_qkvo_r8-020] | 2482 [qwen2_1_5b_balanced_qkvo_r8-021] |
| firevlm_qwen2_1_5b_lora_efficient_qv_r16 | 0.8605962933118453 [qwen2_1_5b_efficient_qv_r16-029] | 0.13964606127463883 [qwen2_1_5b_efficient_qv_r16-030] | 0.8936878861133494 [qwen2_1_5b_efficient_qv_r16-031] | 2482 [qwen2_1_5b_efficient_qv_r16-032] |
| firevlm_qwen2_1_5b_lora_efficient_qv_r8 | 0.8614020950846092 [qwen2_1_5b_efficient_qv_r8-019] | 0.13744401364244885 [qwen2_1_5b_efficient_qv_r8-020] | 0.8931861785810213 [qwen2_1_5b_efficient_qv_r8-021] | 2482 [qwen2_1_5b_efficient_qv_r8-022] |
| firevlm_qwen2_1_5b_lora_high_capacity_all_linear_r16 | 0.8573730862207897 [qwen2_1_5b_high_capacity_all_linear_r16-026] | 0.14073647668631462 [qwen2_1_5b_high_capacity_all_linear_r16-027] | 0.8919036107593722 [qwen2_1_5b_high_capacity_all_linear_r16-028] | 2482 [qwen2_1_5b_high_capacity_all_linear_r16-029] |
| firevlm_qwen2_1_5b_lora_high_capacity_all_linear_r8 | 0.8485092667203867 [qwen2_1_5b_high_capacity_all_linear_r8-025] | 0.1418188261375277 [qwen2_1_5b_high_capacity_all_linear_r8-026] | 0.8857871915889644 [qwen2_1_5b_high_capacity_all_linear_r8-027] | 2482 [qwen2_1_5b_high_capacity_all_linear_r8-028] |

## 6. Comparisons

Few-shot exact_match delta versus zero-shot: 1-shot +0.05, 2-shot +0.085, 3-shot +0.085, 4-shot +0.095, 5-shot +0.105, 6-shot +0.11, 7-shot +0.12, 8-shot +0.11, 9-shot +0.125, 10-shot +0.13 [FS-042, FS-043, FS-044, FS-045, FS-046, FS-047, FS-048, FS-049, FS-050, FS-051].

Exact Match is the most important metric according to the project owner [D-06].

## 7. Anomalies and caveats

The notebook for firevlm_lora_balanced_qkvo_r8 has markdown title "rank 16" but the run is rank 8 [balanced_qkvo_r8-035].

The trial notebook (Wildfire_Generative_Custom_VLM_Kaggle.ipynb) has no text outputs, so row counts cannot be cited from it [DS-004].

No run_config.json file exists for firevlm_lora_balanced_qkvo_r4; hyperparameters must come from notebook or adapter_config [balanced_qkvo_r4-051].

## 8. Open questions

What is the paper title and dataset name for the dataset built on top of FLAME-3?
What does "FireVLM" mean in this project context?
What are the exact training and validation losses for few-shot baseline shots 1-9?
What are the mean latency and input token counts for few-shot baseline shots 1-9?
What is the optimizer for runs firevlm_lora_efficient_qv_r16 and firevlm_lora_balanced_qkvo_r4?
What is the scheduler for runs firevlm_lora_efficient_qv_r16 and firevlm_lora_balanced_qkvo_r4?
What is the max sequence length configuration for firevlm_qwen2_1_5b_lora_high_capacity_all_linear_r16?
