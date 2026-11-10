# FireVLM Phase 2 Review - Presentation Slides

---

## Slide 1: Project Overview

**FireVLM: Multimodal Wildfire Visual Question Answering**

- **Goal**: Answer questions about wildfire scenes using RGB + Thermal aerial imagery
- **Dataset**: WildFireVQA (25K QA pairs, 738 image pairs from FLAME-3)
- **Challenge**: Fuse complementary modalities (RGB: smoke/structures, Thermal: hotspots/temperature)

**Approaches Tested:**
1. Few-Shot Baseline (frozen Qwen2.5-VL-3B)
2. High-Capacity QLoRA (17M params, attention + MLP)
3. Balanced QLoRA (9.2M params, attention-only) ⭐
4. Efficient QLoRA (6.6M params, minimal adapters)

---

## Slide 2: Architecture

```
┌───────────────────────────────────────────────────────────┐
│                   FireVLM Architecture                     │
└───────────────────────────────────────────────────────────┘

RGB Image (224×224)          Thermal Image (224×224)
      │                              │
      ├──▶ CLIP ViT-Base (frozen)   │
      │                              │
      ├──▶ 49 patches × 768-dim     │
      │                              │
      ▼                              ▼
┌─────────────────┐        ┌─────────────────┐
│ RGB Projector   │        │Thermal Projector│
│   (768→2048)    │        │   (768→2048)    │
│   + Type Embed  │        │   + Type Embed  │
└────────┬────────┘        └────────┬────────┘
         │                          │
         └──────────┬───────────────┘
                    ▼
         ┌──────────────────────┐
         │   Llama-3.2-1B-Inst  │
         │   + 4-bit QLoRA      │
         │   (Q/K/V/O adapters) │
         └──────────┬───────────┘
                    ▼
              [Answer Token]
```

**Key Design Choices:**
- Frozen vision encoder (cached features)
- Modality-specific projectors (RGB vs Thermal)
- Parameter-efficient fine-tuning (QLoRA)

---

## Slide 3: Results Summary

| Approach | Params (M) | Exact Match | ROUGE-L | Val Loss |
|----------|-----------|-------------|---------|----------|
| **Few-Shot (0-shot)** | 0 | TBD* | TBD* | N/A |
| High-Capacity QLoRA | 17.0 (2.00%) | 85.46% | 0.8878 | 0.1382 |
| **Balanced QLoRA** ⭐ | **9.2 (1.09%)** | **88.15%** | **0.9073** | **0.1017** |
| Efficient QLoRA | 6.6 (0.79%) | 86.38% | 0.8911 | 0.1233 |

*Need to extract from few-shot notebook

**Key Findings:**
- ✅ **Fewer parameters = Better performance** (9.2M > 17M)
- ✅ **Attention-only LoRA sufficient** (MLP adapters hurt)
- ✅ **88.15% accuracy** with only 1.09% trainable params

---

## Slide 4: Efficiency-Accuracy Tradeoff

```
Performance vs Parameters
═════════════════════════

90% ┤
    │              ● Balanced
    │             /│\ (9.2M, 88.15%)
88% ┤            / │ \
    │           /  │  \
86% ┤          /   │   ● Efficient
    │         /    │    (6.6M, 86.38%)
84% ┤        /     │
    │ ● High-Cap   │
82% ┤  (17M, 85.46%)
    └──────┴───────┴────────┴──────────▶
       5M    10M    15M    20M
           Trainable Parameters

Pareto Frontier: Balanced is optimal
```

**Insight**: Adding MLP adapters (gate/up/down_proj) **decreases** accuracy
- Hypothesis: Low-rank regularization prevents overfitting
- MLP layers add noise rather than signal for this task

---

## Slide 5: Training Dynamics

**Convergence Comparison:**

| Approach | Best Epoch | Final Val Loss | Training Time |
|----------|-----------|----------------|---------------|
| High-Capacity | 9 | 0.1382 | ~9h (slowest) |
| **Balanced** | **6** | **0.1017** | **~4h (fastest)** |
| Efficient | 7 | 0.1233 | ~5h |

**Observations:**
1. Balanced reaches best model in 6 epochs (33% faster than High-Cap)
2. Efficient achieves lowest val loss (0.1233) but NOT highest test accuracy
3. Possible val-test distribution mismatch for Efficient

**Hyperparameters:**
- Batch size: 2 (High-Cap) vs 4 (others)
- Learning rate: 1e-4 → 1.5e-4 → 2e-4 (increasing)
- Early stopping: 2-3 epochs patience

---

## Slide 6: Sample Predictions

**Example 1: Thermal-Critical Question**
```
Q: Are active thermal hotspots detected?
RGB:     [Shows smoke and trees]
Thermal: [Shows bright heat signatures]

Ground Truth: "Yes"
Prediction:   "Yes"  ✅

Category: Thermal Detection (requires thermal modality)
```

**Example 2: RGB-Sufficient Question**
```
Q: Is smoke visible?
RGB:     [Shows gray smoke plumes]
Thermal: [Shows same scene in IR]

Ground Truth: "Yes"
Prediction:   "Yes"  ✅

Category: Visual Detection (RGB likely sufficient)
```

**Error Analysis Needed:**
- Which categories benefit most from thermal?
- Where does the model fail?

---

## Slide 7: What's Working

✅ **Architecture Validated:**
- Dual-modality fusion via learned projectors works
- Type embeddings enable LLM to distinguish RGB vs Thermal
- QLoRA effectively adapts frozen LLM to multimodal task

✅ **Efficient Training:**
- Cached CLIP features (no repeated encoding)
- 4-bit quantization fits 1B model on single T4
- Early stopping prevents overfitting

✅ **Strong Performance:**
- 88.15% exact match on held-out test set
- 90.73% ROUGE-L (near-perfect semantic overlap)
- Generalizes within Sycan site

---

## Slide 8: What's Missing

⚠️ **Critical Experiments Not Yet Run:**

1. **Modality Ablation**: Does thermal actually help?
   - Need RGB-only and Thermal-only baselines
   - Compare per-category performance
   - Current assumption: thermal is beneficial (not proven)

2. **Few-Shot Baseline**: How much does supervised training help?
   - Extract 0-shot and 10-shot results from baseline notebook
   - Quantify gain from fine-tuning vs prompting

3. **Cross-Site Generalization**: Does it work on new locations?
   - Trained on Sycan subset only
   - Should test on Shoetank and Wilamette
   - Assess domain shift robustness

4. **Error Analysis**: Where and why does it fail?
   - Confusion matrix by category
   - Qualitative review of mispredictions
   - Identify failure modes

---

## Slide 9: Phase 3 Roadmap

**High Priority (Do First):**
1. ✅ **Deploy Balanced QLoRA** (best model)
   - Save checkpoints for inference
   - Build evaluation pipeline
   
2. 🔬 **Modality Ablations** (~2 training runs)
   - RGB-only: Remove thermal input
   - Thermal-only: Remove RGB input
   - Compare to dual-modality (88.15% baseline)

3. 📊 **Extract Few-Shot Results**
   - Parse 0-shot through 10-shot performance
   - Calculate supervised fine-tuning gain

**Medium Priority (If Time Permits):**
4. 🌍 **Cross-Site Evaluation**
   - Test on Shoetank/Wilamette (zero-shot)
   - Fine-tune on all sites (data augmentation)

5. 🐛 **Error Analysis**
   - Per-category breakdown
   - Failure mode taxonomy

6. 🎯 **Hyperparameter Ablations**
   - Projector hidden dim (512, 1024, 2048)
   - LoRA rank sweep (4, 8, 16, 32)

---

## Slide 10: Key Takeaways

**For Technical Reviewers:**
1. **Architecture is sound**: Frozen CLIP + learned projectors + QLoRA LLM works
2. **Best model**: Balanced QLoRA (Q/K/V/O, R=16, 9.2M params, 88.15% EM)
3. **Surprising finding**: Attention-only adapters > attention+MLP adapters
4. **Efficiency**: 1.09% trainable params achieves 88% accuracy

**For Next Phase:**
1. Run modality ablations to validate thermal contribution
2. Extract few-shot baseline for comparison
3. Deploy model and collect feedback
4. Test cross-site generalization

**Open Questions:**
- Does thermal imagery actually improve accuracy? (hypothesis: yes, but unproven)
- What's the failure mode distribution? (need error analysis)
- Will it generalize to new fire sites? (need cross-site eval)

---

## Slide 11: Budget & Resources

**Compute Used (Phase 2):**
- Hardware: 2× Tesla T4 GPUs (Kaggle)
- Training time: ~4-9 hours per approach
- Total: ~20-30 GPU-hours

**Dataset Statistics:**
- Training: 20,128 QA pairs (592 image pairs)
- Validation: 2,482 QA pairs (73 image pairs)
- Test: 2,482 QA pairs (73 image pairs)
- Modalities: RGB (corrected FOV) + Thermal (raw JPG)
- Source: FLAME-3 Sycan Marsh subset

**Model Specs:**
- Base LLM: Llama-3.2-1B (1B params, 4-bit)
- Vision: CLIP ViT-Base-32 (frozen)
- Trainable: 9.2M params (best model)
- Inference: ~200ms per query (single T4)

---

## Slide 12: Questions?

**Contact:**
- Your Name
- Your Email
- GitHub: [repository link]

**Resources:**
- Full Analysis: `PHASE2_RESULTS_ANALYSIS.md`
- Notebooks: All 4 approaches in `/FinalYear/`
- Extracted Metrics: `extracted_metrics.json`

**Next Review:**
- Phase 3 (after modality ablations)
- Target: [Date]

---

## Backup Slides

### Backup: LoRA Configuration Details

**High-Capacity:**
- Targets: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Rank: 16, Alpha: 32, Dropout: 0.05
- Params: 17.0M (2.00%)

**Balanced:**
- Targets: q_proj, k_proj, v_proj, o_proj
- Rank: 16, Alpha: 32, Dropout: 0.05
- Params: 9.2M (1.09%)

**Efficient:**
- Targets: q_proj, v_proj
- Rank: 8, Alpha: 16, Dropout: 0.05
- Params: 6.6M (0.79%)

### Backup: Data Processing Pipeline

1. **Path Resolution**: Strict structural mapping from VQA annotations to Kaggle images
2. **Radiometric Matching**: TIFF temperature signatures ensure correct RGB-thermal pairing
3. **Image-Level Splits**: No leakage (train/val/test on different image pairs)
4. **Quality Checks**: 
   - RGB ≠ Thermal (verified via feature difference)
   - One-to-one source-to-Kaggle mapping
   - No ambiguous multi-site images

### Backup: Training Configuration

**Common Settings:**
- Optimizer: AdamW (weight decay 0.01)
- Scheduler: Cosine with warmup (5% warmup ratio)
- Gradient clipping: 1.0
- Accumulation steps: 4
- Early stopping: 2-3 epochs patience
- Effective batch size: 8-16

**Hardware:**
- GPU: 2× Tesla T4 (14.6 GiB each)
- Memory: Only GPU 0 used (1B model fits on one T4)
- Precision: FP16 autocast, 4-bit base model

