# FireVLM Phase 3 - Systematic Experiment Plan

**Goal**: Clean ablation study with **controlled hyperparameters** to isolate architectural effects

**Problem with Phase 2**: Different batch sizes (2 vs 4) and learning rates (1e-4 vs 1.5e-4 vs 2e-4) confounded the comparison

---

## 🎯 **Phase 3 Strategy: Controlled Ablation**

### **Fixed Hyperparameters (Lock These Across ALL Runs)**

```python
# Use Balanced QLoRA's hyperparameters as baseline
BATCH_SIZE = 4
LEARNING_RATE = 1.5e-4
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
GRAD_ACCUM_STEPS = 4  # Effective BS = 16
GRAD_CLIP = 1.0
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 1e-3  # More relaxed than 1e-8
MAX_EPOCHS = 10
SEED = 42

# Vision & LLM (NEVER change these)
VISION_MODEL = 'openai/clip-vit-base-patch32'
LLM_MODEL = 'meta-llama/Llama-3.2-1B-Instruct'
USE_4BIT = True
PROJECTOR_HIDDEN_DIM = 1024
GRADIENT_CHECKPOINTING = False  # Balanced used False, keep it
```

**Why these values?**
- Balanced QLoRA achieved best performance with these
- Eliminates optimization as a confound
- Only architecture (LoRA config) varies

---

## 📊 **Experiment Matrix: 6 Controlled Runs**

### **Dimension 1: Rank Sweep on Attention-Only** (Best from Phase 2)
**Goal**: Is rank 16 optimal for attention-only adapters?

| Run | Name | LoRA Targets | Rank | Expected Outcome |
|-----|------|--------------|------|------------------|
| **1** | Balanced-r4 | Q/K/V/O | **4** | Lower capacity → expect 84-86% EM |
| **2** | **Balanced-r16** (baseline) | Q/K/V/O | **16** | Already done: 88.2% EM |
| **3** | Balanced-r32 | Q/K/V/O | **32** | Higher capacity → might overfit or marginally improve |

**Why this matters**: Validates whether r=16 is a sweet spot or arbitrary

---

### **Dimension 2: Module Coverage Sweep at Rank 8** (Efficiency Frontier)
**Goal**: Which modules are essential at lower rank?

| Run | Name | LoRA Targets | Rank | Expected Outcome |
|-----|------|--------------|------|------------------|
| **4** | **Efficient-r8** (baseline) | Q/V | **8** | Already done: 86.4% EM |
| **5** | AllLinear-r8 | Q/K/V/O + gate/up/down | **8** | Test if MLP helps at lower rank |
| **6** | Attention-r8 | Q/K/V/O | **8** | Between Efficient and AllLinear |

**Why this matters**: 
- Phase 2 showed MLP hurts at r=16 (High-Capacity: 85.5% vs Balanced: 88.2%)
- But maybe MLP helps at lower rank (r=8) by adding capacity?
- This tests the "MLP adapters add noise" hypothesis

---

## 🔬 **What This Experiment Design Achieves**

### **1. Rank Hypothesis Test**
```
Question: Is rank 16 necessary for attention-only adapters?
Answer: Compare runs 1, 2, 3
  - If r=4 performs well (>86%): Lower rank is sufficient
  - If r=32 overfits (worse val loss): r=16 is optimal
  - If r=32 improves: We underfit at r=16
```

### **2. MLP Value at Lower Rank**
```
Question: Do MLP adapters help when rank is constrained?
Answer: Compare runs 4, 5, 6
  - Run 4 (Q/V, r=8): 86.4% EM baseline
  - Run 6 (Q/K/V/O, r=8): Should be ~87-88% if attention is key
  - Run 5 (Q/K/V/O + MLP, r=8): If > Run 6, MLP adds capacity
                                 If < Run 6, MLP still hurts
```

### **3. Parameter Efficiency Frontier**
```
Plot: Trainable Params vs Exact Match
  - Find the Pareto frontier (minimal params for target accuracy)
  - Identify deployment-optimal configurations
```

---

## 📈 **Expected Results & Insights**

### **Scenario A: MLP Hurts Regardless of Rank**
```
Outcome: Run 5 (AllLinear-r8) < Run 6 (Attention-r8)
Conclusion: MLP adapters introduce noise even at low rank
Action: Stick with attention-only for deployment
```

### **Scenario B: MLP Helps at Low Rank Only**
```
Outcome: Run 5 (AllLinear-r8) > Run 6 (Attention-r8)
         Run 2 (Balanced-r16) > High-Capacity (from Phase 2)
Conclusion: MLP helps when rank-constrained, hurts at high rank
Insight: Attention layers need higher rank than MLP to be effective
Action: Use AllLinear-r8 for deployment (smaller + similar accuracy)
```

### **Scenario C: Rank 4 is Sufficient**
```
Outcome: Run 1 (Balanced-r4) ≈ Run 2 (Balanced-r16)
Conclusion: r=16 was overkill, r=4 captures most capacity
Action: Deploy Balanced-r4 (even fewer parameters)
```

---

## 💰 **Compute Budget**

### **Runs Needed**
- **3 new runs** (Balanced-r4, Balanced-r32, Attention-r8)
- **1 re-run** (AllLinear-r8 with fixed hyperparameters)

**Total: 4 training runs**

### **Cost per Run**
- Training time: ~6-7 hours (based on Phase 2)
- Hardware: 2× Tesla T4
- Epochs: ~6-10 (early stopping)

**Total compute: ~24-28 GPU-hours (1.5 days on 2× T4)**

---

## 🔧 **Implementation: Unified Config Template**

```python
# phase3_config_template.py

FIXED_HYPERPARAMS = {
    'BATCH_SIZE': 4,
    'LEARNING_RATE': 1.5e-4,
    'WARMUP_RATIO': 0.05,
    'WEIGHT_DECAY': 0.01,
    'GRAD_ACCUM_STEPS': 4,
    'GRAD_CLIP': 1.0,
    'EARLY_STOPPING_PATIENCE': 3,
    'EARLY_STOPPING_MIN_DELTA': 1e-3,
    'EPOCHS': 10,
    'SEED': 42,
    'VISION_MODEL': 'openai/clip-vit-base-patch32',
    'LLM_MODEL': 'meta-llama/Llama-3.2-1B-Instruct',
    'USE_4BIT': True,
    'PROJECTOR_HIDDEN_DIM': 1024,
    'GRADIENT_CHECKPOINTING': False,
    'LORA_ALPHA': None,  # Will be set as 2 * LORA_R
    'LORA_DROPOUT': 0.05,
}

# Variable configurations
EXPERIMENTS = {
    'balanced_r4': {
        'LORA_R': 4,
        'LORA_TARGET_MODULES': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        'OUTPUT_DIR': 'firevlm_phase3_balanced_r4',
    },
    'balanced_r32': {
        'LORA_R': 32,
        'LORA_TARGET_MODULES': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        'OUTPUT_DIR': 'firevlm_phase3_balanced_r32',
    },
    'attention_r8': {
        'LORA_R': 8,
        'LORA_TARGET_MODULES': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        'OUTPUT_DIR': 'firevlm_phase3_attention_r8',
    },
    'alllinear_r8': {
        'LORA_R': 8,
        'LORA_TARGET_MODULES': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                                 'gate_proj', 'up_proj', 'down_proj'],
        'OUTPUT_DIR': 'firevlm_phase3_alllinear_r8',
    },
}

# Auto-set LORA_ALPHA = 2 * LORA_R (standard practice)
for config in EXPERIMENTS.values():
    config['LORA_ALPHA'] = 2 * config['LORA_R']
```

---

## 📋 **Execution Plan**

### **Week 1: Run Experiments**
**Day 1-2**: 
- Set up unified config template
- Run **Balanced-r4** and **Balanced-r32**
- Monitor convergence

**Day 3-4**:
- Run **Attention-r8** and **AllLinear-r8**
- Ensure all use same hyperparameters

**Day 5**:
- Collect all results
- Verify no confounds (check training logs)

### **Week 2: Analysis & Write-Up**
**Day 6-7**:
- Generate comparison tables
- Plot parameter efficiency curves
- Per-category analysis for each model

**Day 8-9**:
- Write Phase 3 report
- Create updated slides
- Prepare for final review

**Day 10**:
- Buffer for re-runs if needed
- Final presentation prep

---

## 📊 **Expected Deliverables**

### **1. Clean Comparison Table**
| Model | Rank | Targets | Params (M) | EM (%) | ROUGE-L | Val Loss | Notes |
|-------|------|---------|------------|--------|---------|----------|-------|
| Balanced-r4 | 4 | Q/K/V/O | ~4.6M | TBD | TBD | TBD | Minimal |
| Efficient-r8 | 8 | Q/V | 6.6M | 86.4 | 0.891 | 0.123 | Phase 2 |
| Attention-r8 | 8 | Q/K/V/O | ~7.5M | TBD | TBD | TBD | Full attention |
| AllLinear-r8 | 8 | Q/K/V/O+MLP | ~12M | TBD | TBD | TBD | With MLP |
| Balanced-r16 | 16 | Q/K/V/O | 9.2M | 88.2 | 0.907 | 0.102 | Phase 2 best |
| Balanced-r32 | 32 | Q/K/V/O | ~18M | TBD | TBD | TBD | High capacity |

### **2. Pareto Frontier Plot**
```
90% EM │                    ● Balanced-r16 (9.2M)
       │                   /│
88% EM │                  / │  ● Balanced-r32? (18M)
       │                 /  │ /
86% EM │      ● Attention-r8? (7.5M)
       │     /          /  │/
84% EM │    /  ● Efficient-r8 (6.6M)
       │   /  /
82% EM │  ● Balanced-r4? (4.6M)
       │ /
       └────────────────────────────────
         4M     8M     12M    16M    20M
              Trainable Parameters

Goal: Find the knee of the curve (best accuracy/param ratio)
```

### **3. Hypothesis Test Results**
- **H1**: Rank 16 is optimal for attention-only adapters
  - ✓ Confirmed / ✗ Rejected / ⚠️ Inconclusive
  
- **H2**: MLP adapters hurt at high rank but help at low rank
  - ✓ Confirmed / ✗ Rejected / ⚠️ Inconclusive

- **H3**: Attention layers need higher rank than MLP layers
  - ✓ Confirmed / ✗ Rejected / ⚠️ Inconclusive

### **4. Per-Category Analysis**
- Which configurations excel at Cross-Modal Reasoning?
- Does rank affect Localization performance (hardest category)?
- Are MLP adapters helpful for any specific category?

---

## ⚠️ **What We're NOT Doing (And Why)**

### **NOT Running:**
1. ❌ **Cross-site generalization** - No Shoetank/Wilamette data available
2. ❌ **RGB-only / Thermal-only** - No time for proper ablation
3. ❌ **Different learning rates** - Would re-introduce confounds
4. ❌ **Larger models** - Stay with Llama-1B for fair comparison
5. ❌ **Thermal-specific encoder** - Too expensive, low expected ROI

### **Why This is Still Valuable:**
- Answers the core architectural question: **Which LoRA configuration is optimal?**
- Clean comparison enables publication (no confounds)
- Identifies deployment-ready model (Pareto frontier)
- Validates or refutes Phase 2 findings with controlled experiments

---

## 🎯 **Success Criteria**

### **Minimum Viable Outcome:**
- 4 runs complete with same hyperparameters ✓
- Clean comparison table ✓
- At least one hypothesis confirmed or rejected ✓

### **Strong Outcome:**
- Identify Pareto-optimal configuration (best accuracy/param ratio)
- Per-category insights explain when MLP helps vs hurts
- Clear recommendation for deployment

### **Ideal Outcome:**
- All 3 hypotheses tested
- Training dynamics explain why some configs converge faster
- Generalizable insight: "Attention layers need rank X, MLP needs rank Y"

---

## 📝 **Phase 3 Presentation Outline**

### **Slide 1: Phase 2 Recap**
- Best model: Balanced-r16 (88.2% EM)
- Problem: Confounded hyperparameters (different BS/LR)
- Question: Was the architecture or optimization responsible?

### **Slide 2: Phase 3 Design**
- Controlled ablation: Fixed all hyperparameters
- 6 configurations tested
- 2 hypotheses: Rank optimality + MLP value at low rank

### **Slide 3: Results**
- Clean comparison table
- Pareto frontier plot
- Hypothesis test outcomes

### **Slide 4: Insights**
- Which configuration is deployment-optimal?
- When do MLP adapters help vs hurt?
- Per-category breakdown

### **Slide 5: Recommendations**
- Deployment model
- Future work (cross-site, thermal ablation)
- Final conclusions

---

## 🚀 **Next Steps**

1. **Copy Balanced QLoRA notebook** → rename to `phase3_template.ipynb`
2. **Replace CFG class** with the unified config from above
3. **Run experiments** in order: Balanced-r4 → Balanced-r32 → Attention-r8 → AllLinear-r8
4. **Track results** in a shared spreadsheet as each run completes
5. **Generate plots** after all 4 runs finish
6. **Write Phase 3 report** with hypothesis tests and insights

**Estimated timeline: 10 days (1.5 days compute + 3 days analysis + buffer)**

---

## 💡 **Pro Tips**

1. **Monitor Val Loss**: If val loss plateaus early, might need higher patience
2. **Check Training Logs**: Ensure LR schedule is identical across runs
3. **Save Checkpoints**: Keep best model from each run for error analysis
4. **Track GPU Utilization**: Ensure all runs use same hardware/memory
5. **Document Everything**: Note any anomalies (OOM, longer epochs, etc.)

---

**Status**: Ready to execute Phase 3 controlled ablation study

**Expected Completion**: 2 weeks from start

**Deliverable**: Clean architectural comparison + deployment recommendation
