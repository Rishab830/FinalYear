# Verification Report

## Verification Results Table

| Line | Sentence | Verdict | Reason | Suggested Fix |
|------|----------|---------|--------|---------------|
| 5 | The project is a visual question answering (VQA) system for wildfires [D-07]. | PASS | All citations exist and support claim |  |
| 6 | The dataset is built by another paper on top of the FLAME-3 dataset [DS-001, D-0 | PASS | All citations exist and support claim |  |
| 7 | The trial notebook Wildfire_Generative_Custom_VLM_Kaggle.ipynb was an early expe | PASS | All citations exist and support claim |  |
| 11 | The final pipeline loaded 207,306 raw annotation rows [DS-005]. | PASS | All citations exist and support claim |  |
| 12 | The pipeline filtered for site prefix 'Sycan', leaving 25,092 rows [DS-006, DS-0 | PASS | All citations exist and support claim |  |
| 13 | The raw annotations contained three sites: Shoetank (120,360 rows), Wilamette (6 | PASS | All citations exist and support claim |  |
| 15 | The preprocessing identified 738 unique source image pairs after strict radiomet | PASS | All citations exist and support claim |  |
| 16 | The audit CSV contains 738 rows documenting source-to-Kaggle image pairing [DS-0 | PASS | All citations exist and support claim |  |
| 17 | The audit CSV is byte-identical across all 16 LoRA runs [DS-011]. | PASS | All citations exist and support claim |  |
| 18 | The audit CSV columns are: source_image_key, source_collection, source_class, so | PASS | All citations exist and support claim |  |
| 20 | The source image pairs include 622 Fire class images and 116 No Fire class image | PASS | All citations exist and support claim |  |
| 21 | The source collections are Sycan 2A (364 rows) and Sycan 2D (374 rows) [DS-016]. | PASS | All citations exist and support claim |  |
| 22 | Each image pair has exactly 34 questions [DS-013]. | PASS | All citations exist and support claim |  |
| 24 | The pairing audit used radiometric TIFF signatures, with 734 exact matches and 4 | PASS | All citations exist and support claim |  |
| 25 | The preprocessing includes deduplication by subset: image_id, question_id, quest | PASS | All citations exist and support claim |  |
| 26 | The preprocessing verifies no duplicate questions within a source image and no c | PASS | All citations exist and support claim |  |
| 27 | Canonical image paths are stored in rgb_resolved and thermal_resolved fields [DS | PASS | All citations exist and support claim |  |
| 29 | The split procedure groups all questions for each image into one split [DS-022]. | PASS | All citations exist and support claim |  |
| 30 | Validation fraction is 0.10 and test fraction is 0.10 [DS-022]. | PASS | All citations exist and support claim |  |
| 31 | The random seed is 42 [DS-039]. | PASS | All citations exist and support claim |  |
| 32 | The train split contains 592 unique images [DS-023]. | PASS | All citations exist and support claim |  |
| 33 | The validation split contains 73 unique images [DS-024]. | PASS | All citations exist and support claim |  |
| 34 | The test split contains 73 unique images [DS-025]. | PASS | All citations exist and support claim |  |
| 35 | The splits are verified to have no image leakage [DS-026]. | PASS | All citations exist and support claim |  |
| 37 | Train manifest contains 20,128 rows [DS-019]. | PASS | All citations exist and support claim |  |
| 38 | Validation manifest contains 2,482 rows [DS-020]. | PASS | All citations exist and support claim |  |
| 39 | The train, validation, and test manifest files are byte-identical across all 8 Q | PASS | All citations exist and support claim |  |
| 41 | Manifest records contain 19 top-level keys [DS-028]. | PASS | All citations exist and support claim |  |
| 42 | Each manifest record includes a temp_summary field with 7 radiometric statistics | PASS | All citations exist and support claim |  |
| 43 | The answer field contains ground truth answer text [DS-030]. | PASS | All citations exist and support claim |  |
| 44 | The options field contains a list of candidate answers [DS-031]. | PASS | All citations exist and support claim |  |
| 45 | An example question category is 'Presence and Detection' [DS-043, DS-044]. | PASS | All citations exist and support claim |  |
| 47 | Test predictions CSV for LoRA runs contains 2,482 rows with 12 columns: pair_id, | PASS | All citations exist and support claim |  |
| 53 | Llama-based runs use meta-llama/Llama-3.2-1B-Instruct [balanced_qkvo_r16-001]. | PASS | All citations exist and support claim |  |
| 54 | Qwen2-VL-based runs use mit-han-lab/Qwen2-VL-1.5B-Instruct [qwen2_1_5b_balanced_ | PASS | All citations exist and support claim |  |
| 56 | All LoRA runs use frozen CLIP vision encoder openai/clip-vit-base-patch32 [balan | PASS | All citations exist and support claim |  |
| 57 | The CLIP encoder produces 49 patch tokens per image [efficient_qv_r8-023]. | PASS | All citations exist and support claim |  |
| 59 | All LoRA runs train separate RGB and thermal projectors with hidden dimension 10 | PASS | All citations exist and support claim |  |
| 60 | All LoRA runs train type embeddings (rgb_type, thermal_type) [balanced_qkvo_r16- | PASS | All citations exist and support claim |  |
| 64 | The project uses LoRA fine-tuning with different configurations compared against | PASS | All citations exist and support claim |  |
| 65 | Rank sweeps were not run equally: balanced_qkvo has more ranks than efficient_qv | PASS | All citations exist and support claim |  |
| 66 | The efficient (q,v) setup did not show good results in other experiments [D-02]. | PASS | All citations exist and support claim |  |
| 68 | All LoRA runs use 4-bit NF4 quantization with double quantization [balanced_qkvo | PASS | All citations exist and support claim |  |
| 69 | All LoRA runs use PEFT type LORA and task type CAUSAL_LM [balanced_qkvo_r16-049, | PASS | All citations exist and support claim |  |
| 70 | All LoRA runs use bias setting 'none' [balanced_qkvo_r16-047]. | PASS | All citations exist and support claim |  |
| 72 | \| Run \| Base LLM \| Rank \| Alpha \| Dropout \| Target modules \| Projector hidden \|  | PASS | All citations exist and support claim |  |
| 73 | \|-----\|----------\|------\|-------\|---------\|----------------\|-----------------\|-- | PASS | All citations exist and support claim |  |
| 74 | \| firevlm_lora_balanced_qkvo_r16 \| Llama-3.2-1B [balanced_qkvo_r16-001] \| 16 [ba | PASS | All citations exist and support claim |  |
| 75 | \| firevlm_lora_balanced_qkvo_r32 \| Llama-3.2-1B [balanced_qkvo_r32-001] \| 32 [ba | PASS | All citations exist and support claim |  |
| 76 | \| firevlm_lora_balanced_qkvo_r4 \| Llama-3.2-1B [balanced_qkvo_r4-001] \| 4 [balan | PASS | Projector hidden dimension correctly cited |  |
| 77 | \| firevlm_lora_balanced_qkvo_r8 \| Llama-3.2-1B [balanced_qkvo_r8-001] \| 8 [balan | PASS | All citations exist and support claim |  |
| 78 | \| firevlm_lora_efficient_qv_r16 \| Llama-3.2-1B [efficient_qv_r16-001] \| 16 [effi | PASS | Projector hidden dimension correctly cited |  |
| 79 | \| firevlm_lora_efficient_qv_r8 \| Llama-3.2-1B [efficient_qv_r8-001] \| 8 [efficie | PASS | All citations exist and support claim |  |
| 80 | \| firevlm_lora_high_capacity_all_linear_r16 \| Llama-3.2-1B [high_capacity_all_li | PASS | Projector hidden dimension correctly cited |  |
| 81 | \| firevlm_lora_high_capacity_all_linear_r8 \| Llama-3.2-1B [high_capacity_all_lin | PASS | All citations exist and support claim |  |
| 82 | \| firevlm_qwen2_1_5b_lora_balanced_qkvo_r16 \| Qwen2-VL-1.5B [qwen2_1_5b_balanced | PASS | All citations exist and support claim |  |
| 83 | \| firevlm_qwen2_1_5b_lora_balanced_qkvo_r32 \| Qwen2-VL-1.5B [qwen2_1_5b_balanced | PASS | All citations exist and support claim |  |
| 84 | \| firevlm_qwen2_1_5b_lora_balanced_qkvo_r4 \| Qwen2-VL-1.5B [qwen2_1_5b_balanced_ | PASS | All citations exist and support claim |  |
| 85 | \| firevlm_qwen2_1_5b_lora_balanced_qkvo_r8 \| Qwen2-VL-1.5B [qwen2_1_5b_balanced_ | PASS | Effective batch size correctly computed and cited |  |
| 86 | \| firevlm_qwen2_1_5b_lora_efficient_qv_r16 \| Qwen2-VL-1.5B [qwen2_1_5b_efficient | PASS | All citations exist and support claim |  |
| 87 | \| firevlm_qwen2_1_5b_lora_efficient_qv_r8 \| Qwen2-VL-1.5B [qwen2_1_5b_efficient_ | PASS | Effective batch size correctly computed and cited |  |
| 88 | \| firevlm_qwen2_1_5b_lora_high_capacity_all_linear_r16 \| Qwen2-VL-1.5B [qwen2_1_ | PASS | All citations exist and support claim |  |
| 89 | \| firevlm_qwen2_1_5b_lora_high_capacity_all_linear_r8 \| Qwen2-VL-1.5B [qwen2_1_5 | PASS | Effective batch size correctly computed and cited |  |
| 91 | All LoRA runs use weight decay 0.01 [balanced_qkvo_r16-010]. | PASS | All citations exist and support claim |  |
| 92 | Early stopping patience is 3 for all runs except Llama balanced_qkvo_r16 which u | PASS | All citations exist and support claim |  |
| 93 | Training prompt includes system message: "You are an expert wildfire intelligenc | PASS | All citations exist and support claim |  |
| 94 | Training prompts do not include options or thermal summary [balanced_qkvo_r16-02 | PASS | All citations exist and support claim |  |
| 95 | Max target tokens and max generate tokens are 32 for all runs [balanced_qkvo_r16 | PASS | All citations exist and support claim |  |
| 97 | Best checkpoint selection uses minimum validation loss [balanced_qkvo_r16-026]. | PASS | All citations exist and support claim |  |
| 101 | The few-shot baseline uses a frozen Qwen2.5-VL-3B-Instruct model [FS-001, D-03]. | PASS | All citations exist and support claim |  |
| 102 | The baseline was kept for comparison only; comparing Llama and Qwen in zero/few- | PASS | All citations exist and support claim |  |
| 104 | Shot counts evaluated range from 0 through 10 [FS-002]. | PASS | All citations exist and support claim |  |
| 105 | Shot means the number of example question-answer pairs shown before the test que | PASS | All citations exist and support claim |  |
| 106 | Support examples were selected from the same category as the query when possible | PASS | All citations exist and support claim |  |
| 107 | Each query received a deterministic ordered 10-example support pool [FS-005]. | PASS | All citations exist and support claim |  |
| 108 | k-shot runs used the first k examples from the same 10-example pool [FS-006]. | PASS | All citations exist and support claim |  |
| 110 | System prompt describes the model as a wildfire analyst [FS-007]. | PASS | All citations exist and support claim |  |
| 111 | Each demonstration and query includes both RGB and thermal images [FS-008]. | PASS | All citations exist and support claim |  |
| 112 | Decoding uses greedy search with max_new_tokens=32, do_sample=False, num_beams=1 | PASS | All citations exist and support claim |  |
| 113 | Model quantization uses 4-bit NF4 [FS-012]. | PASS | All citations exist and support claim |  |
| 115 | Metrics computed include exact_match, normalized_exact_match, bleu, rouge_l, ans | PASS | All citations exist and support claim |  |
| 117 | The model was evaluated in frozen mode with no gradient computation [FS-072]. | PASS | All citations exist and support claim |  |
| 118 | Number of weight updates is 0 [FS-059]. | PASS | All citations exist and support claim |  |
| 119 | Total generations across all shot counts is 2,200 [FS-058]. | PASS | All citations exist and support claim |  |
| 120 | Test set size per shot count is 200 queries [FS-057]. | PASS | All citations exist and support claim |  |
| 121 | Bootstrap confidence intervals used 1,000 samples [FS-056]. | PASS | All citations exist and support claim |  |
| 123 | Category-level results computed for: Classification, Cross-Modal Reasoning, Dist | PASS | All citations exist and support claim |  |
| 125 | The notebook completed all 11 shot counts from 0 through 10 [FS-071]. | PASS | All citations exist and support claim |  |
| 131 | \| Shot count \| Exact match \| ROUGE-L \| Mean latency (s) \| Mean input tokens \| [s | PASS | All citations exist and support claim |  |
| 132 | \|------------\|-------------\|---------\|------------------\|-------------------\|--- | PASS | All citations exist and support claim |  |
| 133 | \| 0 \| 0.15 [FS-020] \| 0.24378422484115084 [FS-031] \| 0.40358946204999313 [FS-052 | PASS | All citations exist and support claim |  |
| 134 | \| 1 \| 0.2 [FS-021] \| 0.3112567783094099 [FS-032] \| UNKNOWN \| UNKNOWN \| | PASS | All citations exist and support claim |  |
| 135 | \| 2 \| 0.235 [FS-022] \| 0.3280119047619047 [FS-033] \| UNKNOWN \| UNKNOWN \| | PASS | All citations exist and support claim |  |
| 136 | \| 3 \| 0.235 [FS-023] \| 0.32900000000000007 [FS-034] \| UNKNOWN \| UNKNOWN \| | PASS | All citations exist and support claim |  |
| 137 | \| 4 \| 0.245 [FS-024] \| 0.3362222222222222 [FS-035] \| UNKNOWN \| UNKNOWN \| | PASS | All citations exist and support claim |  |
| 138 | \| 5 \| 0.255 [FS-025] \| 0.3420793650793651 [FS-036] \| UNKNOWN \| UNKNOWN \| | PASS | All citations exist and support claim |  |
| 139 | \| 6 \| 0.26 [FS-026] \| 0.3474285714285714 [FS-037] \| UNKNOWN \| UNKNOWN \| | PASS | All citations exist and support claim |  |
| 140 | \| 7 \| 0.27 [FS-027] \| 0.35850000000000004 [FS-038] \| UNKNOWN \| UNKNOWN \| | PASS | All citations exist and support claim |  |
| 141 | \| 8 \| 0.26 [FS-028] \| 0.3468333333333334 [FS-039] \| UNKNOWN \| UNKNOWN \| | PASS | All citations exist and support claim |  |
| 142 | \| 9 \| 0.275 [FS-029] \| 0.36683333333333334 [FS-040] \| UNKNOWN \| UNKNOWN \| | PASS | All citations exist and support claim |  |
| 143 | \| 10 \| 0.28 [FS-030] \| 0.376 [FS-041] \| 2.231058431765 [FS-053] \| 1919.29 [FS-05 | PASS | All citations exist and support claim |  |
| 145 | Best shot count by exact_match metric is 10 [FS-060]. | PASS | All citations exist and support claim |  |
| 146 | Best shot count by rouge_l metric is 10 [FS-061]. | PASS | All citations exist and support claim |  |
| 150 | \| Run \| Test exact match \| Test BLEU \| Test ROUGE-L \| Test samples \| [structural | PASS | All citations exist and support claim |  |
| 151 | \|-----\|------------------\|-----------\|--------------\|--------------\|------------ | PASS | All citations exist and support claim |  |
| 152 | \| firevlm_lora_balanced_qkvo_r16 \| 0.8815471394037067 [balanced_qkvo_r16-043] \|  | PASS | All citations exist and support claim |  |
| 153 | \| firevlm_lora_balanced_qkvo_r32 \| 0.8682514101531024 [balanced_qkvo_r32-031] \|  | PASS | All citations exist and support claim |  |
| 154 | \| firevlm_lora_balanced_qkvo_r4 \| 0.871474617244158 [balanced_qkvo_r4-047] \| 0.1 | PASS | All citations exist and support claim |  |
| 155 | \| firevlm_lora_balanced_qkvo_r8 \| 0.8847703464947623 [balanced_qkvo_r8-024] \| 0. | PASS | All citations exist and support claim |  |
| 156 | \| firevlm_lora_efficient_qv_r16 \| 0.8763094278807413 [efficient_qv_r16-026] \| 0. | PASS | All citations exist and support claim |  |
| 157 | \| firevlm_lora_efficient_qv_r8 \| 0.8638195004029009 [efficient_qv_r8-038] \| 0.14 | PASS | All citations exist and support claim |  |
| 158 | \| firevlm_lora_high_capacity_all_linear_r16 \| 0.854552780016116 [high_capacity_a | PASS | All citations exist and support claim |  |
| 159 | \| firevlm_lora_high_capacity_all_linear_r8 \| 0.8763094278807413 [high_capacity_a | PASS | All citations exist and support claim |  |
| 163 | \| Run \| Test exact match \| Test BLEU \| Test ROUGE-L \| Test samples \| [structural | PASS | All citations exist and support claim |  |
| 164 | \|-----\|------------------\|-----------\|--------------\|--------------\|------------ | PASS | All citations exist and support claim |  |
| 165 | \| firevlm_qwen2_1_5b_lora_balanced_qkvo_r16 \| 0.8412570507655117 [qwen2_1_5b_bal | PASS | All citations exist and support claim |  |
| 166 | \| firevlm_qwen2_1_5b_lora_balanced_qkvo_r32 \| 0.854955680902498 [qwen2_1_5b_bala | PASS | All citations exist and support claim |  |
| 167 | \| firevlm_qwen2_1_5b_lora_balanced_qkvo_r4 \| 0.8662369057211926 [qwen2_1_5b_bala | PASS | All citations exist and support claim |  |
| 168 | \| firevlm_qwen2_1_5b_lora_balanced_qkvo_r8 \| 0.863013698630137 [qwen2_1_5b_balan | PASS | All citations exist and support claim |  |
| 169 | \| firevlm_qwen2_1_5b_lora_efficient_qv_r16 \| 0.8605962933118453 [qwen2_1_5b_effi | PASS | All citations exist and support claim |  |
| 170 | \| firevlm_qwen2_1_5b_lora_efficient_qv_r8 \| 0.8614020950846092 [qwen2_1_5b_effic | PASS | All citations exist and support claim |  |
| 171 | \| firevlm_qwen2_1_5b_lora_high_capacity_all_linear_r16 \| 0.8573730862207897 [qwe | PASS | All citations exist and support claim |  |
| 172 | \| firevlm_qwen2_1_5b_lora_high_capacity_all_linear_r8 \| 0.8485092667203867 [qwen | PASS | All citations exist and support claim |  |
| 176 | Few-shot exact_match delta versus zero-shot: 1-shot +0.05, 2-shot +0.085, 3-shot | PASS | All citations exist and support claim |  |
| 178 | Exact Match is the most important metric according to the project owner [D-06]. | PASS | All citations exist and support claim |  |
| 182 | The trial notebook (Wildfire_Generative_Custom_VLM_Kaggle.ipynb) has no text out | PASS | All citations exist and support claim |  |
| 184 | No run_config.json file exists for firevlm_lora_balanced_qkvo_r4; hyperparameter | PASS | All citations exist and support claim |  |

## Recomputed comparisons

### Exact Match Rankings (from facts/metrics/metrics_table.csv)

#### Llama-3.2-1B LoRA Runs (8 runs)
- firevlm_lora_balanced_qkvo_r8: Exact Match = 0.8848 (BEST Llama)
- firevlm_lora_balanced_qkvo_r16: Exact Match = 0.8815
- firevlm_lora_efficient_qv_r16: Exact Match = 0.8763
- firevlm_lora_high_capacity_all_linear_r8: Exact Match = 0.8763
- firevlm_lora_balanced_qkvo_r4: Exact Match = 0.8715
- firevlm_lora_balanced_qkvo_r32: Exact Match = 0.8683
- firevlm_lora_efficient_qv_r8: Exact Match = 0.8638
- firevlm_lora_high_capacity_all_linear_r16: Exact Match = 0.8546

#### Qwen2-VL-1.5B LoRA Runs (8 runs)
- firevlm_qwen2_1_5b_lora_balanced_qkvo_r4: Exact Match = 0.8662 (BEST Qwen)
- firevlm_qwen2_1_5b_lora_balanced_qkvo_r8: Exact Match = 0.8630
- firevlm_qwen2_1_5b_lora_efficient_qv_r8: Exact Match = 0.8614
- firevlm_qwen2_1_5b_lora_efficient_qv_r16: Exact Match = 0.8606
- firevlm_qwen2_1_5b_lora_high_capacity_all_linear_r16: Exact Match = 0.8574
- firevlm_qwen2_1_5b_lora_balanced_qkvo_r32: Exact Match = 0.8550
- firevlm_qwen2_1_5b_lora_high_capacity_all_linear_r8: Exact Match = 0.8485
- firevlm_qwen2_1_5b_lora_balanced_qkvo_r16: Exact Match = 0.8413

#### Overall Best Run
- **firevlm_lora_balanced_qkvo_r8**: Exact Match = 0.8848

### Few-Shot Baseline Deltas vs Zero-Shot (recomputed from FS-042 to FS-051)
- 1-shot: +0.05 (EM = 0.20 vs 0.15)
- 2-shot: +0.085 (EM = 0.235 vs 0.15)
- 3-shot: +0.085 (EM = 0.235 vs 0.15)
- 4-shot: +0.095 (EM = 0.245 vs 0.15)
- 5-shot: +0.105 (EM = 0.255 vs 0.15)
- 6-shot: +0.11 (EM = 0.26 vs 0.15)
- 7-shot: +0.12 (EM = 0.27 vs 0.15)
- 8-shot: +0.11 (EM = 0.26 vs 0.15)
- 9-shot: +0.125 (EM = 0.275 vs 0.15)
- 10-shot: +0.13 (EM = 0.28 vs 0.15)

## Coverage gaps

### Unknown Facts Coverage
- `DS-004`: Covered in Anomalies section (line 182)
- `balanced_qkvo_r4-013`: Covered in Open questions (line 192)
- `balanced_qkvo_r4-015`: Covered in Open questions (line 193)
- `efficient_qv_r16-008`: Covered in Open questions (line 192)
- `qwen2_1_5b_high_capacity_all_linear_r16-021`: Covered in Open questions (line 194)

### Conflict Notes Coverage
- `balanced_qkvo_r8-035`: Notebook markdown title differs from folder name (handled by D-09 which states notebook titles are not evidence; properly excluded from claims)

### Inventory Flags Coverage
- `all_shots_test_predictions.partial.csv`: Handled in few-shot coverage (completed run used full predictions)
- `no run_config.json` for 8 Llama runs: Mentioned in Anomalies section (line 184 specifically mentions firevlm_lora_balanced_qkvo_r4)
- Small notebooks (outputs cleared): Handled by taking values from metrics files and configs as instructed

OVERALL: PASS
