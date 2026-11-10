# FireVLM Project Knowledge Base

**Last Updated:** 2026-11-10  
**Project:** FireVLM - Parameter-Efficient Wildfire Visual Question Answering  
**Purpose:** Comprehensive reference for future AI agents working on this project

---

## Project Overview

**FireVLM** is a vision-language model specifically designed for wildfire visual question answering (VQA). The project investigates parameter-efficient fine-tuning (PEFT) methods, specifically QLoRA (Quantized Low-Rank Adaptation), to adapt pre-trained language models to the WildFireVQA benchmark with minimal computational resources.

### Key Research Questions
1. What is the minimal adapter configuration (rank, layer placement) that achieves competitive accuracy on wildfire VQA?
2. How do different adapter placements (query/value only, query/key/value/output, all linear layers) affect performance?
3. Can compact models (1.5B parameters with 4-bit quantization) achieve high accuracy on thermal+RGB wildfire imagery?

---

## Dataset: WildFireVQA + FLAME-3

### WildFireVQA Benchmark
- **Source:** Clemson University (Habibpour et al., 2026)
- **Total Questions:** 207,298 question-answer pairs
- **Images:** 6,097 RGB-thermal image pairs from FLAME-3
- **Sites:** Three prescribed burns (Sycan Marsh, Willamette, Shoetank)
- **Question Categories (6 types):**
  1. Presence and detection
  2. Classification  
  3. Distribution and segmentation
  4. Localization and direction
  5. Cross-modal reasoning (thermal + RGB)
  6. Flight planning

### FLAME-3 Dataset
- **Imagery:** Aerial RGB and radiometric thermal TIFF files captured by UAV
- **Thermal Data:** Per-pixel temperature measurements in Celsius
- **Structure:** 
  - Fire/ and No Fire/ class folders
  - RGB/Corrected FOV/, Thermal/Raw JPG/, Thermal/Celsius TIFF/ subfolders
  - Each scene has RGB + thermal JPG + radiometric TIFF triplet

### Project Data Split (Sycan Marsh Only)
- **Train:** ~80% (image-grouped split to prevent data leakage)
- **Validation:** 10%
- **Test:** 10% (~2,482 question-answer pairs)
- **Site Filter:** Only Sycan Marsh images used (25,092 QA pairs from 207,298 total)
- **Split Method:** Image-level grouping ensures no image appears in multiple splits

### Pairing Audit
- **Challenge:** WildFireVQA paths reference author's local FLAME-3 structure; Kaggle dataset has flattened/renumbered structure
- **Solution:** Radiometric TIFF signature matching (min, max, mean, std of per-pixel temperatures)
- **Verification:** Each source scene matched to canonical Kaggle quartet using thermal statistics
- **Output:** `source_to_kaggle_pair_audit.csv` ensures 1-to-1 correspondence

---

## Model Architecture

### Core Components

```
RGB Image (224x224)      → CLIP ViT-B/32 (frozen) → RGB patches (49 tokens, 768-dim)
                                                    ↓
                                                  RGB Projector (trainable MLP)
                                                    ↓
Thermal Image (224x224)  → CLIP ViT-B/32 (frozen) → Thermal patches (49 tokens, 768-dim)
                                                    ↓
                                                  Thermal Projector (trainable MLP)
                                                    ↓
Question Text            → Tokenizer              → Text tokens
                                                    ↓
                                    [All modalities concatenated]
                                                    ↓
                                    Language Model (4-bit quantized with QLoRA)
                                                    ↓
                                            Generated Answer
```

### Vision Encoder
- **Model:** CLIP ViT-B/32 (openai/clip-vit-base-patch32)
- **Status:** Frozen (weights not updated during training)
- **Input:** 224×224 RGB or thermal image
- **Output:** 7×7 = 49 patch tokens, each 768-dimensional
- **Efficiency:** Features cached once before training on CPU to save GPU memory

### Projectors
- **Architecture:** 2-layer MLP with hidden dimension 1024
- **Input:** 768-dim CLIP features
- **Output:** Language model embedding dimension (depends on LLM)
- **Training:** Learned from random initialization alongside LoRA adapters
- **Modality-Specific:** Separate projectors for RGB and thermal to learn distinct mappings
- **Purpose:** Map frozen CLIP features into the language model's token embedding space

### Language Model Backbones

#### Qwen2-1.5B (Primary)
- **Checkpoint:** `mit-han-lab/Qwen2-VL-1.5B-Instruct`
- **Clarification:** Despite repository name, this is an **extracted Qwen2ForCausalLM language component without a vision tower** (confirmed via model card and config.json)
- **Quantization:** 4-bit NF4 with double quantization (bitsandbytes)
- **Status:** Frozen base weights; only LoRA adapters + projectors trainable
- **Parameters:** ~1.5 billion (quantized to 4-bit)

#### Llama-3.2-1B (Comparison)
- **Checkpoint:** `meta-llama/Llama-3.2-1B-Instruct`
- **Purpose:** Backbone comparison to verify that results generalize across LLM families
- **Configuration:** Same pipeline as Qwen2 (frozen CLIP, trainable projectors, 4-bit + LoRA)

---

## QLoRA Configurations Tested

### Configuration Families

#### 1. Efficient (q,v only)
- **Target Modules:** `['q_proj', 'v_proj']`
- **Rationale:** Minimal adapter budget; focuses on query and value attention components
- **Ranks Tested:** r=8, r=16

#### 2. Balanced (q,k,v,o)
- **Target Modules:** `['q_proj', 'k_proj', 'v_proj', 'o_proj']`
- **Rationale:** Adapts all attention matrices (query, key, value, output projection)
- **Ranks Tested:** r=4, r=8, r=16, r=32

#### 3. High-Capacity (all linear)
- **Target Modules:** `['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']`
- **Rationale:** Adapts attention + MLP feed-forward layers for maximum expressiveness
- **Ranks Tested:** r=8 (Llama only), r=16

### LoRA Hyperparameters
- **Alpha:** Typically 2× rank (e.g., alpha=16 for r=8, alpha=64 for r=32)
- **Dropout:** 0.05
- **Init:** Gaussian initialization for A, zero initialization for B (standard LoRA)

---

## Training Configuration

### Hyperparameters (Standard Across Runs)
```python
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4          # Effective batch size = 16
EPOCHS = 100                  # Early stopping typically triggers ~7-12 epochs
EARLY_STOPPING_PATIENCE = 3
LR = 2e-4                     # Learning rate
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05           # 5% of steps for warmup
GRAD_CLIP = 1.0
OPTIMIZER = AdamW
SCHEDULER = Cosine with warmup
```

### Efficiency Optimizations
1. **Feature Caching:** CLIP features extracted once and cached on CPU before training
2. **No Gradient Checkpointing:** Unnecessary for 1.5B model with QLoRA
3. **4-bit Quantization:** NF4 (Normal Float 4-bit) with double quantization
4. **Single GPU:** All training on single T4 (14.6 GiB); second GPU intentionally unused
5. **Native FP16:** PyTorch native AMP (not bfloat16 due to T4 limitations)

### Data Pipeline
- **Vision Cache Batch Size:** 16 (for initial CLIP feature extraction)
- **Training Workers:** 0 (features/tokens cached; multiprocessing adds overhead)
- **Validation:** Every epoch; best model selected by validation loss
- **Test:** Final evaluation on held-out test set after training completes

---

## Complete Experimental Results

### ALL Qwen2-1.5B Models (Exact Test Metrics)

| Model ID             | Configuration     | Rank | Exact Match         | BLEU   | ROUGE-L | Status     |
|----------------------|-------------------|------|---------------------|--------|---------|------------|
| qwen2_balanced_r4    | Balanced q,k,v,o  | 4    | **0.8662** (86.62%) | 0.1403 | 0.8979  | ✓ Complete |
| qwen2_balanced_r8    | Balanced q,k,v,o  | 8    | 0.8630 (86.30%)     | 0.1497 | 0.8928  | ✓ Complete |
| qwen2_balanced_r16   | Balanced q,k,v,o  | 16   | 0.8413 (84.13%)     | 0.1460 | 0.8773  | ✓ Complete |
| qwen2_balanced_r32   | Balanced q,k,v,o  | 32   | 0.8550 (85.50%)     | 0.1400 | 0.8869  | ✓ Complete |
| qwen2_efficient_r8   | Efficient q,v     | 8    | 0.8614 (86.14%)     | 0.1374 | 0.8932  | ✓ Complete |
| qwen2_efficient_r16  | Efficient q,v     | 16   | 0.8606 (86.06%)     | 0.1396 | 0.8937  | ✓ Complete |
| qwen2_all_linear_r8  | High-Capacity all | 8    | 0.8485 (84.85%)     | 0.1418 | 0.8858  | ✓ Complete |
| qwen2_all_linear_r16 | High-Capacity all | 16   | 0.8574 (85.74%)     | 0.1407 | 0.8919  | ✓ Complete |

**Qwen2 Model Ranking (by Exact Match):**
1. **Balanced r=4: 86.62%** ← BEST Qwen2 model
2. Balanced r=8: 86.30%
3. Efficient r=8: 86.14%
4. Efficient r=16: 86.06%
5. All-linear r=16: 85.74%
6. Balanced r=32: 85.50%
7. All-linear r=8: 84.85%
8. Balanced r=16: 84.13%

### ALL Llama-3.2-1B Models (Exact Test Metrics)

| Model ID | Configuration | Rank | Exact Match | BLEU | ROUGE-L | Status |
|----------|--------------|------|-------------|------|---------|--------|
| llama_balanced_r4 | Balanced q,k,v,o | 4 | 0.8715 (87.15%) | 0.1467 | 0.8982 | ✓ Complete |
| llama_balanced_r8 | Balanced q,k,v,o | 8 | **0.8848** (88.48%) | 0.1458 | 0.9121 | ✓ Complete |
| llama_balanced_r16 | Balanced q,k,v,o | 16 | 0.8815 (88.15%) | 0.1457 | 0.9073 | ✓ Complete |
| llama_balanced_r32 | Balanced q,k,v,o | 32 | 0.8683 (86.83%) | 0.1398 | 0.8974 | ✓ Complete |
| llama_efficient_r8 | Efficient q,v | 8 | 0.8638 (86.38%) | 0.1451 | 0.8911 | ✓ Complete |
| llama_efficient_r16 | Efficient q,v | 16 | 0.8763 (87.63%) | 0.1449 | 0.9043 | ✓ Complete |
| llama_all_linear_r8 | High-Capacity all | 8 | 0.8763 (87.63%) | 0.1430 | 0.9062 | ✓ Complete |
| llama_all_linear_r16 | High-Capacity all | 16 | 0.8546 (85.46%) | 0.1447 | 0.8878 | ✓ Complete |

**Llama Model Ranking (by Exact Match):**
1. **Balanced r=8: 88.48%** ← BEST overall model
2. Balanced r=16: 88.15%
3. Balanced r=4: 87.15%
4. Efficient r=16: 87.63%
5. All-linear r=8: 87.63%
6. Balanced r=32: 86.83%
7. Efficient r=8: 86.38%
8. All-linear r=16: 85.46%

### Frozen Few-Shot Baseline (Qwen2.5-VL-3B)

| Shot Count | Exact Match | BLEU | ROUGE-L | Notes |
|------------|-------------|------|---------|-------|
| 0-shot | 0.1500 (15.00%) | 0.0094 | 0.2438 | Native VLM, no fine-tuning |
| 1-shot | 0.2000 (20.00%) | 0.0076 | 0.3113 | +5.00 points |
| 2-shot | 0.2350 (23.50%) | 0.0107 | 0.3280 | +3.50 points |
| 3-shot | 0.2350 (23.50%) | 0.0122 | 0.3290 | +0.00 points |
| 4-shot | 0.2450 (24.50%) | 0.0185 | 0.3362 | +1.00 points |
| 5-shot | 0.2550 (25.50%) | 0.0201 | 0.3421 | +1.00 points |
| 10-shot | 0.2800 (28.00%) | 0.0189 | 0.3760 | +2.50 points |

**Baseline Analysis:**
- Model: Qwen2.5-VL-3B-Instruct (native VLM, 3B params, larger than fine-tuned models)
- Method: In-context learning only, no weight updates
- Performance: 15-28% (far below fine-tuned 84-88%)
- Observation: Even 10-shot prompting cannot compete with parameter-efficient fine-tuning

---

## Key Findings (Data-Driven)

### 1. Best Models by Configuration

**Overall Best:**
- **Llama-3.2-1B + Balanced q,k,v,o r=8** → 88.48% exact match

**Best Qwen2 Model:**
- **Qwen2-1.5B + Balanced q,k,v,o r=4** → 86.62% exact match

**Best Efficiency (Fewest Layers):**
- **Qwen2-1.5B + Efficient q,v r=8** → 86.14% exact match (only 2 attention matrices adapted)

### 2. Rank Selection Patterns

**Qwen2-1.5B Balanced Strategy:**
- r=4: 86.62%
- r=8: 86.30% (-0.32 from r=4)
- r=16: 84.13% (-2.49 from r=4)
- r=32: 85.50% (-1.12 from r=4)

**Pattern:** Rank 4 is optimal for Qwen2 balanced strategy. Higher ranks show degradation, with r=16 particularly poor. r=32 partially recovers but still worse than r=4.

**Llama-3.2-1B Balanced Strategy:**
- r=4: 87.15%
- r=8: 88.48% (+1.33 from r=4)
- r=16: 88.15% (-0.33 from r=8)
- r=32: 86.83% (-1.65 from r=8)

**Pattern:** Rank 8 is optimal for Llama balanced strategy. Shows clear peak at r=8 with degradation on both sides.

**Key Insight:** Optimal rank is model-dependent. Qwen2 prefers r=4, Llama prefers r=8.

### 3. Strategy Comparison (Same Rank)

**Qwen2-1.5B at Rank 8:**
- Balanced (q,k,v,o): 86.30%
- Efficient (q,v): 86.14% (-0.16)
- All-linear: 84.85% (-1.45)

**Qwen2-1.5B at Rank 16:**
- Efficient (q,v): 86.06%
- All-linear: 85.74% (-0.32)
- Balanced (q,k,v,o): 84.13% (-1.93)

**Pattern:** At r=16, efficient strategy outperforms both balanced and all-linear for Qwen2.

**Llama-3.2-1B at Rank 8:**
- Balanced (q,k,v,o): 88.48%
- All-linear: 87.63% (-0.85)
- Efficient (q,v): 86.38% (-2.10)

**Pattern:** At r=8, balanced > all-linear > efficient for Llama.

**Key Insight:** Strategy performance varies by model. For Llama, balanced is clearly best. For Qwen2, results are more complex and rank-dependent.

### 4. Language Model Backbone Comparison

**Direct Comparisons (Same Config):**

| Configuration | Qwen2-1.5B | Llama-3.2-1B | Llama Advantage |
|--------------|------------|--------------|-----------------|
| Balanced r=4 | 86.62% | 87.15% | +0.53 points |
| Balanced r=8 | 86.30% | 88.48% | +2.18 points |
| Balanced r=16 | 84.13% | 88.15% | +4.02 points |
| Balanced r=32 | 85.50% | 86.83% | +1.33 points |
| Efficient r=8 | 86.14% | 86.38% | +0.24 points |
| Efficient r=16 | 86.06% | 87.63% | +1.57 points |
| All-linear r=8 | 84.85% | 87.63% | +2.78 points |
| All-linear r=16 | 85.74% | 85.46% | -0.28 points |

**Pattern:** Llama-3.2-1B outperforms Qwen2-1.5B in 7 out of 8 configurations. Advantage ranges from +0.24 to +4.02 points. Exception: All-linear r=16 where Qwen2 is slightly better (+0.28 points).

**Key Insight:** Llama-3.2-1B is generally the superior backbone for this task, particularly with balanced adapters at medium ranks.

### 5. High-Capacity Adapter Analysis

**Qwen2-1.5B:**
- All-linear r=8: 84.85%
- All-linear r=16: 85.74%
- Balanced r=4 (baseline): 86.62%
- **Gap at r=8 vs best:** -1.77 points (all-linear r=8 vs balanced r=4)
- **Gap at r=16:** -0.88 points (all-linear r=16 vs balanced r=4, despite 4× the rank)

**Llama-3.2-1B:**
- All-linear r=8: 87.63%
- Balanced r=8 (baseline): 88.48%
- **Gap:** -0.85 points with same rank

**Pattern:** Adding MLP adapters (gate_proj, up_proj, down_proj) consistently degrades performance compared to attention-only adapters. For Qwen2, all-linear r=8 (84.85%) is 1.45 points worse than balanced r=8 (86.30%), and all-linear r=16 (85.74%) is worse than efficient r=16 (86.06%).

**Key Insight:** For wildfire VQA, attention adaptation is sufficient. MLP adaptation adds parameters without improving accuracy, possibly causing overfitting. Interestingly, for all-linear adapters, increasing rank from 8→16 actually improves performance (+0.89 points), unlike other strategies.

### 6. Few-Shot vs Fine-Tuning

**Frozen Qwen2.5-VL-3B (10-shot):** 28.00% exact match  
**Fine-tuned Qwen2-1.5B (best):** 86.62% exact match  
**Gap:** +58.62 points

**Key Insight:** Parameter-efficient fine-tuning with QLoRA vastly outperforms few-shot prompting, even when the frozen model is larger (3B vs 1.5B) and is a native VLM.

---

## Per-Category Performance

**Note:** Detailed per-category breakdowns are only available for select runs. The table below shows available data.

### Qwen2 Balanced r=16 (Category Breakdown)

| Category | Performance | Difficulty |
|----------|-------------|------------|
| Presence and detection | ~90.75% | Easiest |
| Distribution and segmentation | ~85.96% | Moderate |
| Classification | ~85.84% | Moderate |
| Flight planning | ~85.27% | Moderate |
| Cross-modal reasoning | ~76.37% | Hard |
| Localization and direction | ~71.23% | Hardest |

### Llama Efficient r=8 (Category Breakdown)

| Category | Performance | Difficulty |
|----------|-------------|------------|
| Presence and detection | ~94.35% | Easiest |
| Distribution and segmentation | ~90.41% | Moderate |
| Classification | ~81.96% | Moderate |
| Flight planning | ~85.27% | Moderate |
| Cross-modal reasoning | ~81.85% | Hard |
| Localization and direction | ~74.66% | Hardest |

**Category Analysis:**
- **Easiest:** Presence/detection (binary yes/no questions, ~2.4 avg choices)
- **Hardest:** Localization/direction (~6.5 avg choices, fine-grained spatial reasoning)
- **Challenge:** Cross-modal reasoning (thermal + RGB integration) and localization both require capabilities beyond simple scene understanding

---

## Comparison to Published Baselines

### WildFireVQA Paper Baselines (Zero-Shot Multiple-Choice)

**Reported Results (from WildFireVQA paper):**
- GPT-4o: 75.0% test accuracy (best)
- GPT-4V: ~70% test accuracy
- BLIP-2, InstructBLIP, LLaVA-1.5, Qwen-VL: 26.4–36.5% (barely above random 25%)

**FireVLM Fine-Tuned Models:**
- Best: Llama-3.2-1B balanced r=8 → 88.48%
- Best Qwen2: Qwen2-1.5B balanced r=4 → 86.62%

**Comparison:**
- FireVLM outperforms GPT-4o by +11.48 to +13.48 points
- FireVLM outperforms open models by +50 to +62 points

**Important Caveats:**
1. Different evaluation format: WildFireVQA paper used multiple-choice selection; FireVLM generates free-form text
2. Different test sets: WildFireVQA paper used full benchmark; FireVLM uses Sycan Marsh only
3. Direct comparison requires identical evaluation protocol

---

## Technical Implementation Details

### Prompt Format
```
RGB and thermal images of a wildfire scene are provided.

Question: {question_text}

Answer:
```
- **Options:** NOT included in prompt (CFG.INCLUDE_OPTIONS_IN_PROMPT = False)
- **Temp Summary:** NOT included (CFG.INCLUDE_TEMP_SUMMARY = False)
- **Rationale:** Model generates answer from visual understanding, not option selection

### Token Limits
- `MAX_TARGET_TOKENS = 32` (answer generation limit)
- `MAX_GENERATE_TOKENS = 32` (inference limit)
- Most answers are short phrases (1-5 tokens)

### Evaluation Metrics
1. **Exact Match (Primary):** Case-insensitive string match after whitespace normalization
2. **BLEU-4:** Custom smoothed sentence BLEU (via NLTK)
3. **ROUGE-L:** Longest common subsequence F1 score

### Training Time
- **Typical:** 200-310 minutes per configuration on Kaggle T4 GPU
- **Epochs:** 7-12 (early stopping)
- **Effective Batch Size:** 16 (via gradient accumulation)

### Files Saved Per Model
```
{output_dir}/
├── best_lora_adapters/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── README.md
├── best_projectors.pt              # RGB + thermal projector state_dicts
├── tokenizer/                      # Full tokenizer for inference
├── image_processor/                # CLIP image preprocessor config (Qwen2 only)
├── run_config.json                 # All hyperparameters (when available)
├── training_history.csv            # Per-epoch train/val loss
├── test_metrics.json               # Final test exact_match, BLEU, ROUGE-L
├── test_predictions.csv            # Per-sample predictions for analysis
├── {train|validation|test}_manifest.jsonl  # Data splits (when available)
└── source_to_kaggle_pair_audit.csv # Image path pairing verification
```

---

## Inference Requirements

### To Load a Trained Model
1. **Load base language model** in 4-bit:
   ```python
   from transformers import AutoModelForCausalLM, BitsAndBytesConfig
   from peft import PeftModel
   
   bnb_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_quant_type="nf4",
       bnb_4bit_compute_dtype=torch.float16,
       bnb_4bit_use_double_quant=True
   )
   base_model = AutoModelForCausalLM.from_pretrained(
       "mit-han-lab/Qwen2-VL-1.5B-Instruct",  # or "meta-llama/Llama-3.2-1B-Instruct"
       quantization_config=bnb_config,
       device_map="auto"
   )
   ```

2. **Load LoRA adapters:**
   ```python
   model = PeftModel.from_pretrained(
       base_model, 
       "path/to/best_lora_adapters"
   )
   ```

3. **Load projectors separately:**
   ```python
   projector_state = torch.load("path/to/best_projectors.pt")
   rgb_projector.load_state_dict(projector_state['rgb_projector'])
   thermal_projector.load_state_dict(projector_state['thermal_projector'])
   modality_embeddings.load_state_dict(projector_state['modality_embeddings'])
   ```

4. **Load CLIP and tokenizer** (same as training)

### Hardware Requirements
- **Training:** Kaggle T4 GPU (14.6 GiB) or equivalent
- **Inference:** Can run on smaller GPUs due to 4-bit quantization (~6-8 GiB VRAM)

---

## Known Limitations

### Data Limitations
1. **Single Site:** Only Sycan Marsh images used; generalization to Willamette/Shoetank untested
2. **Test Set Size:** 73 unique image pairs (2,482 questions) may introduce variance
3. **Answer Format:** Exact string match penalizes correct answers phrased differently

### Model Limitations
1. **Not Native Qwen2-VL:** Uses extracted language component + separate CLIP, not the official Qwen2-VL vision tower
2. **Frozen Vision Encoder:** CLIP ViT-B/32 not adapted to wildfire domain
3. **Single Seed:** Each configuration trained once (seed=42); variance across seeds unknown
4. **Incomplete Runs:** None - all 16 models have complete test metrics

### Evaluation Limitations
1. **Different Baselines:** Few-shot baseline uses different model (3B vs 1.5B), test set, and input handling
2. **Category Breakdown:** Only available for 2 runs (not systematic across all models)
3. **Comparison to WildFireVQA Paper:** Different evaluation formats make direct comparison imprecise

---

## Future Work Directions

### Immediate Next Steps
1. **Vision Encoder Adaptation:** Unfreeze CLIP layers or use LoRA on vision encoder
2. **Spatial Reasoning:** Integrate VisionCore-style spatial embeddings for localization questions
3. **Thermal-Specific Processing:** Specialized handling of radiometric temperature data
4. **Multi-Site Evaluation:** Test on Willamette and Shoetank held-out sites

### Model Improvements
1. **Larger Backbones:** Test 7B models (Qwen2-7B, Llama-3.2-7B) with same adapter strategy
2. **Native VLMs:** Compare against true Qwen2-VL or LLaVA-1.5 architectures
3. **Ensemble:** Combine predictions from Qwen2 and Llama models
4. **Knowledge Distillation:** Distill from GPT-4o or Gemini 1.5 Pro on thermal reasoning

### Training Enhancements
1. **Curriculum Learning:** Start with easy categories (presence) before hard (localization)
2. **Data Augmentation:** Synthetic thermal data, rotation/crop augmentations
3. **Multi-Task Learning:** Joint training on related tasks (image captioning, object detection)
4. **Active Learning:** Identify hard examples for human verification

### Deployment
1. **Quantization:** Post-training quantization to 3-bit or 2-bit for edge deployment
2. **Latency Optimization:** Speculative decoding, KV-cache optimization
3. **UAV Integration:** On-board inference for real-time wildfire monitoring

---

## Repository Structure

```
FinalYear/
├── notebooks/                                   # Jupyter/Kaggle notebooks
│   ├── Wildfire_Generative_Custom_VLM_Kaggle.ipynb  # Initial generative VLM experiment
│   ├── firevlm-qwen2-vl-1.5b-lora-balanced-qkvo-r{4,8,16,32}.ipynb
│   ├── firevlm-qwen2-vl-1.5b-lora-efficient-qv-r{8,16}.ipynb
│   ├── firevlm-qwen2-vl-1.5b-lora-high-capacity-all-linear-r{8,16}.ipynb
│   └── firevlm-lora-*-*.ipynb                   # Llama-3.2-1B parallel runs
│
├── models/                                      # Trained model artifacts
│   ├── firevlm_qwen2_1_5b_lora_balanced_qkvo_r{4,8,16,32}/
│   ├── firevlm_qwen2_1_5b_lora_efficient_qv_r{8,16}/
│   ├── firevlm_qwen2_1_5b_lora_high_capacity_all_linear_r{8,16}/
│   ├── firevlm_lora_balanced_qkvo_r{4,8,16,32}/  # Llama-3.2-1B runs
│   ├── firevlm_lora_efficient_qv_r{8,16}/
│   ├── firevlm_lora_high_capacity_all_linear_r{8,16}/
│   └── firevlm_frozen_shot_benchmark_qwen25vl_3b/  # Few-shot baseline
│
├── papers/                                      # Literature collection
│   ├── pdfs/                                    # Paper PDFs
│   ├── _paper_briefs/                           # Extracted summaries
│   └── _paper_extracts/                         # Full paper text extracts
│
├── report/                                      # LaTeX project report
│   ├── 23UG102_report.tex                       # Main report document
│   ├── template.tex                             # Report template
│   └── amrita_logo.png                          # University logo
│
└── FIREVLM_PROJECT_KNOWLEDGE.md                 # This file
```

---

## Quick Reference: Best Models

### For Maximum Accuracy
- **Model:** Llama-3.2-1B + balanced q,k,v,o rank 8
- **Performance:** 88.48% exact match, 0.146 BLEU, 0.912 ROUGE-L
- **Location:** `models/firevlm_lora_balanced_qkvo_r8/`
- **Use Case:** When accuracy is paramount and computational budget permits

### For Best Qwen2 Performance
- **Model:** Qwen2-1.5B + balanced q,k,v,o rank 4
- **Performance:** 86.62% exact match, 0.140 BLEU, 0.898 ROUGE-L
- **Location:** `models/firevlm_qwen2_1_5b_lora_balanced_qkvo_r4/`
- **Use Case:** When using Qwen2 backbone or lowest rank configuration

### For Minimal Adapter Layers
- **Model:** Qwen2-1.5B + efficient q,v rank 8
- **Performance:** 86.14% exact match, 0.137 BLEU, 0.893 ROUGE-L
- **Location:** `models/firevlm_qwen2_1_5b_lora_efficient_qv_r8/`
- **Use Case:** When adapter parameter count is the primary constraint

---

## Raw Data Summary for Verification

**Total Models Trained:** 16 (all complete with test metrics)
- Qwen2-1.5B: 8 complete
- Llama-3.2-1B: 8 complete

**Test Set:** 2,482 questions (consistent across all runs)

**Accuracy Range:** 84.13% (Qwen2 balanced r=16) to 88.48% (Llama balanced r=8)

**BLEU Range:** 0.137 to 0.150

**ROUGE-L Range:** 0.877 to 0.912

---

## Citation Information

### WildFireVQA Dataset
```bibtex
@article{habibpour2026wildfirevqa,
  title={WildFireVQA: A Large-Scale Radiometric Thermal VQA Benchmark for Aerial Wildfire Monitoring},
  author={Habibpour, Mobin and Alipour Talemi, Niloufar and Spodnik, John and Khoury, Camren J. and Afghah, Fatemeh},
  journal={arXiv preprint arXiv:2604.20190},
  year={2026}
}
```

### FLAME-3 Dataset
```bibtex
@article{hopkins2024flame3,
  title={FLAME 3 Dataset: Unleashing the Power of Radiometric Thermal UAV Imagery for Wildfire Management},
  author={Hopkins, Bryce and ONeill, Louie and others},
  journal={arXiv preprint arXiv:2412.02831},
  year={2024}
}
```

---

## Contact & Collaboration

**Team Members:**
- Praanesh Balakrishnan Nair (BL.EN.U4AIE23123)
- Rishab Ramesh Nair (BL.EN.U4AIE23126)
- Sanjushree Rajan (BL.EN.U4AIE23130)
- Varun Adhitya G B (BL.EN.U4AIE23135)

**Institution:** Amrita School of Computing, Bengaluru  
**Affiliation:** Amrita Vishwa Vidyapeetham  
**Academic Year:** 2023-2024 (Final Year Project)

---

## Appendix: Glossary

- **QLoRA:** Quantized Low-Rank Adaptation - PEFT method combining 4-bit quantization with LoRA
- **LoRA:** Low-Rank Adaptation - decomposes weight updates into low-rank matrices (ΔW = BA)
- **PEFT:** Parameter-Efficient Fine-Tuning - methods that adapt large models with minimal trainable parameters
- **CLIP:** Contrastive Language-Image Pretraining - vision encoder pretrained on image-text pairs
- **Exact Match:** Metric where prediction must exactly match ground truth after normalization
- **NF4:** Normal Float 4-bit quantization format optimized for neural network weights
- **Adapter Rank (r):** Dimensionality of the low-rank decomposition in LoRA; lower rank = fewer parameters
- **Radiometric Thermal:** Thermal imagery encoding actual temperature values (Celsius), not just visual colorization

---

**End of Knowledge Base**
