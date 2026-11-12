# Verification Report (Re-audit after fixes)

## Line-by-Line Findings

| Line | Sentence (first 80 chars) | Verdict | Reason | Suggested Fix |
|------|---------------------------|---------|--------|---------------|
| 85 | | firevlm_qwen2_1_5b_lora_balanced_qkvo_r8 | Qwen2-VL-1.5B [qwen2_1_5b_balanced_ | FAIL | Projector hidden cites balanced_qkvo_r16-022 for qwen2_1_5b_balanced_qkvo_r8 run | Should cite a qwen2_1_5b_balanced_qkvo_r8 projector fact |
| 87 | | firevlm_qwen2_1_5b_lora_efficient_qv_r8 | Qwen2-VL-1.5B [qwen2_1_5b_efficient_ | FAIL | Projector hidden cites balanced_qkvo_r16-022 for qwen2_1_5b_efficient_qv_r8 | Should cite qwen2_1_5b_efficient_qv_r8 projector fact |
| 85 | | firevlm_qwen2_1_5b_lora_balanced_qkvo_r8 | Qwen2-VL-1.5B [qwen2_1_5b_balanced_ | UNSUPPORTED | Effective BS computed from two facts without explicit computation note | Should cite single fact or mark as computed |
| 87 | | firevlm_qwen2_1_5b_lora_efficient_qv_r8 | Qwen2-VL-1.5B [qwen2_1_5b_efficient_ | UNSUPPORTED | Effective BS computed from two facts without explicit computation note | Should cite single fact or mark as computed |
| 89 | | firevlm_qwen2_1_5b_lora_high_capacity_all_linear_r8 | Qwen2-VL-1.5B [qwen2_1_5 | UNSUPPORTED | Effective BS computed from two facts without explicit computation note | Should cite single fact or mark as computed |
| 152 | | firevlm_lora_balanced_qkvo_r16 | 0.8815471394037067 [balanced_qkvo_r16-043] |  | PASS | Test exact match for balanced_qkvo_r16 correct | N/A |
| 165 | | firevlm_qwen2_1_5b_lora_balanced_qkvo_r16 | 0.8412570507655117 [qwen2_1_5b_bal | PASS | Test exact match for qwen2_r16 correct | N/A |

## Recomputed Comparisons

### Llama-3.2-1B Test Exact Match Rankings
1. firevlm_lora_balanced_qkvo_r8: 0.8847703464947623
2. firevlm_lora_balanced_qkvo_r16: 0.8815471394037067
3. firevlm_lora_efficient_qv_r16: 0.8763094278807413
4. firevlm_lora_high_capacity_all_linear_r8: 0.8763094278807413
5. firevlm_lora_balanced_qkvo_r4: 0.871474617244158
6. firevlm_lora_balanced_qkvo_r32: 0.8682514101531024
7. firevlm_lora_efficient_qv_r8: 0.8638195004029009
8. firevlm_lora_high_capacity_all_linear_r16: 0.854552780016116

### Qwen2-VL-1.5B Test Exact Match Rankings
1. firevlm_qwen2_1_5b_lora_balanced_qkvo_r4: 0.8662369057211926
2. firevlm_qwen2_1_5b_lora_balanced_qkvo_r8: 0.863013698630137
3. firevlm_qwen2_1_5b_lora_efficient_qv_r8: 0.8614020950846092
4. firevlm_qwen2_1_5b_lora_efficient_qv_r16: 0.8605962933118453
5. firevlm_qwen2_1_5b_lora_high_capacity_all_linear_r16: 0.8573730862207897
6. firevlm_qwen2_1_5b_lora_balanced_qkvo_r32: 0.854955680902498
7. firevlm_qwen2_1_5b_lora_high_capacity_all_linear_r8: 0.8485092667203867
8. firevlm_qwen2_1_5b_lora_balanced_qkvo_r16: 0.8412570507655117

## Coverage Gaps

Checked inventory flags:
- Line 186: balanced_qkvo_r4 no run_config - ✓ mentioned in Anomalies
- Line 182: balanced_qkvo_r8 notebook title mismatch - ✓ mentioned in Anomalies
- Line 184: trial notebook has no text outputs - ✓ mentioned in Anomalies

Checked UNKNOWN facts:
- Found 5 UNKNOWN facts in ledger
- All appear in Open questions section (lines 190-196) ✓

Checked CONFLICT notes:
- balanced_qkvo_r8-035 (title mismatch) - ✓ mentioned at line 182

## Validator Output

From scripts/validate.py doc:
- 4 UNCITED errors: all table header rows (expected/acceptable)
- 4 WARNINGS: numbers 1024 and 3 not found in some cited facts
  - These relate to the remaining cross-citation errors found

## Summary

- FAIL: 2
- UNSUPPORTED: 3
- PASS (spot-checked): 3

Primary remaining issues:
- Lines 85, 87: Two Qwen2 r8 runs still cite balanced_qkvo_r16-022 for projector hidden
  (cross-run citation error not fully fixed)
- Lines 85, 87, 89: Effective batch size computed from two facts without marking

Notes:
- Line 92 early stopping patience claim now properly cited with 15 facts ✓
- Test metrics tables spot-checked and correct ✓
- Coverage requirements met ✓

OVERALL: FAIL
