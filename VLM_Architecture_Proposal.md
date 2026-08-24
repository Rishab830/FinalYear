# Proposed VLM Architecture

Source: `Research.pdf`

This proposal keeps the VLM modular so Kaggle experiments can swap backbones, fusion layers, PEFT methods, and training stages without rewriting the full pipeline.

## High-Level Architecture

```mermaid
flowchart LR
    I[Image Input] --> IP[Image Preprocessing<br/>resize, normalize, augment]
    T[Text Input] --> TP[Text Preprocessing<br/>tokenize, prompt template]

    IP --> VE{Vision Encoder}
    VE --> VE1[CNN]
    VE --> VE2[ViT]
    VE --> VE3[Swin Transformer]
    VE --> VE4[Hybrid CNN + Transformer]

    TP --> TE{Text Encoder}
    TE --> TE1[LSTM]
    TE --> TE2[Transformer Encoder]
    TE --> TE3[Small LLM / Decoder LM]

    VE1 --> VP[Vision Projection / Adapter]
    VE2 --> VP
    VE3 --> VP
    VE4 --> VP

    TE1 --> TPJ[Text Projection / Adapter]
    TE2 --> TPJ
    TE3 --> TPJ

    VP --> F{Cross-Modal Fusion}
    TPJ --> F

    F --> F1[Embedding Concatenation]
    F --> F2[Cross-Attention]
    F --> F3[Q-Former / Query Tokens]
    F --> F4[Mixture of Experts]

    F1 --> MM[Multimodal Representation]
    F2 --> MM
    F3 --> MM
    F4 --> MM

    MM --> H{Task Head}
    H --> R[Retrieval / Similarity Head]
    H --> C[Captioning Decoder]
    H --> VQA[VQA / Instruction Head]
    H --> CLS[Classification Head]
```

## Training Flow

```mermaid
flowchart TD
    D[Paired Image-Text Dataset<br/>COCO, SynthVLM-100K, BIOMEDICA, RS5M, task data]
    D --> S1[Stage 1: Alignment]
    S1 --> L1[Contrastive Loss<br/>image-text embedding alignment]
    S1 --> L2[Optional Next-Token Prediction]

    L1 --> S2[Stage 2: Instruction Tuning]
    L2 --> S2
    S2 --> L3[Captioning / VQA / QA Loss]

    L3 --> S3[Stage 3: Domain Adaptation]
    S3 --> L4[Domain-Specific Fine-Tuning<br/>medical, remote sensing, custom Kaggle task]

    L4 --> E[Evaluation]
    E --> M[Metrics<br/>accuracy, BLEU, retrieval precision, validation loss]
    M --> EXP[Experiment Tracker<br/>compare layer permutations]
```

## Configurable Experiment Axes

| Layer / Module | Candidate Options | Kaggle-Oriented Default |
|---|---|---|
| Base model | CLIP, InternVL, SynthVLM, Molmo | CLIP or small open-weight VLM first |
| Vision encoder | CNN, ViT, Swin, Hybrid CNN + Swin/ViT | ViT for simplicity, Swin for efficiency/local-global features |
| Text encoder | LSTM, Transformer, small decoder LM | Transformer |
| Projection layer | Linear, MLP, adapter block | Linear first, adapter/MLP if underfitting |
| Fusion layer | Concatenation, cross-attention, Q-Former, Mixture of Experts | Concatenation baseline, cross-attention upgrade |
| PEFT method | LoRA, AdaLoRA, QLoRA, adapters, prompt tuning, normalization-only tuning | LoRA or QLoRA |
| Training loss | Contrastive, next-token prediction, captioning CE, VQA CE | Contrastive + task loss |
| Fine-tuning stage | Alignment, instruction tuning, domain adaptation | Alignment -> instruction tuning -> domain adaptation |
| Inference strategy | Single model, zero-shot ensemble, confidence merge, test-time prompt tuning | Single model baseline, ensemble later |

## Proposed Baseline

Start with the smallest useful architecture:

```text
Image -> ViT/CLIP vision encoder -> projection layer
Text  -> Transformer text encoder -> projection layer
Projected embeddings -> contrastive alignment
Aligned representation -> captioning/VQA/classification head
Fine-tune with LoRA or QLoRA
```

This baseline is practical for Kaggle because it keeps most pretrained weights frozen and only trains lightweight adaptation layers.

## Recommended Experiment Order

1. **Baseline retrieval/alignment**
   - Vision encoder: CLIP ViT
   - Text encoder: CLIP text transformer
   - Fusion: shared embedding space
   - Loss: contrastive loss
   - PEFT: none or LoRA on projection layers

2. **Task-specific VQA or captioning**
   - Add decoder/task head.
   - Keep the vision backbone frozen.
   - Tune projection, fusion, and task head.

3. **Fusion-layer comparison**
   - Compare concatenation vs cross-attention vs Q-Former/query-token fusion.
   - Keep the same dataset split and metrics.

4. **PEFT comparison**
   - Compare LoRA, AdaLoRA, QLoRA, adapters, prompt tuning, and normalization-only tuning.
   - Track trainable parameter count, GPU memory, validation score, and epoch time.

5. **Backbone comparison**
   - Compare ViT, Swin, and hybrid CNN-transformer variants.
   - Use smaller image resolution first to stay within Kaggle GPU limits.

6. **Domain adaptation**
   - Use BIOMEDICA-like data for medical tasks, RS5M-like data for remote sensing, or the target Kaggle dataset.
   - Add instruction tuning after alignment if the task requires reasoning or natural-language answers.

7. **Ensembling and test-time adaptation**
   - Try confidence-based model merging or zero-shot ensembles after strong single-model baselines exist.
   - Add test-time prompt tuning only if validation behavior suggests prompt sensitivity.

## Kaggle Constraints To Design Around

- Freeze large pretrained backbones wherever possible.
- Prefer LoRA, QLoRA, adapters, prompt tuning, or normalization-only tuning over full fine-tuning.
- Use small batches with gradient accumulation.
- Cache image/text embeddings for early fusion experiments.
- Track every permutation with the same validation split.
- Keep the architecture config-driven so layers can be swapped cleanly.

## Suggested Config Skeleton

```yaml
model:
  base_model: clip
  vision_encoder: vit
  text_encoder: transformer
  projection: linear
  fusion: contrastive_shared_space
  task_head: retrieval

training:
  stages:
    - alignment
    - instruction_tuning
    - domain_adaptation
  losses:
    alignment: contrastive
    language: next_token_prediction
    task: cross_entropy

peft:
  method: lora
  target_modules:
    - projection
    - fusion
    - task_head

kaggle:
  freeze_backbones: true
  mixed_precision: true
  gradient_accumulation: true
  cache_embeddings: true
```

## Notes From Research.pdf

- Open-weight VLM choices mentioned: CLIP, InternVL, SynthVLM, and Molmo.
- Vision backbone trade-offs: ViT has global patch attention and is strong for large-scale tasks; Swin is more efficient and captures local plus global structure; hybrid designs can help domain-specific image types.
- Training should be multi-stage: alignment first, then instruction tuning, then domain adaptation.
- PEFT methods are important for Kaggle constraints: LoRA, AdaLoRA, QLoRA, adapters, prompt tuning, skip tuning, LoRA-XS, and normalization-only tuning are all candidate knobs.
- Later experiments can include mixture-of-experts, uncertainty-guided merging, zero-shot ensembles, and privacy-preserving variants when the baseline is stable.
