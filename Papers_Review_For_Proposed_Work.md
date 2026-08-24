# Paper Review for Proposed VLM Work

Source folder: `papers/`

This document summarizes each PDF with the fields needed for the proposed work: author names, full title with year, datasets used, inference from the paper, and open problems that can guide our VLM architecture and Kaggle experiments.

## Quick Comparison

| # | Paper | Main Use For Proposed Work |
|---|---|---|
| 1 | Comprehensive VLM survey | Overall research map: PEFT, prompts, datasets, benchmarks, deployment gaps |
| 2 | CLIP-AST | Adaptive layer/parameter selection instead of manually choosing tuning locations |
| 3 | Contrastive alignment | Use contrastive loss with next-token prediction during alignment |
| 4 | KDA-Tuning | Preserve general knowledge while learning task-specific adapters |
| 5 | Molmo and PixMo | Open-data VLM recipe and importance of high-quality non-distilled data |
| 6 | PEFT post-transformer survey | Compare LoRA, QLoRA, DoRA, adapters, prompts under resource limits |
| 7 | SynthVLM | Synthetic image-caption data generation and data selection |
| 8 | Remote sensing VLM survey | Domain-specific VLM gaps: multimodal RS data, explainability, continual learning |
| 9 | VisionCore | Spatial reasoning with coordinate-aware LoRA tuning |
| 10 | VL-PET | Granularity-controlled PET modules and layer placement experiments |

---

## 1. Comprehensive VLM Survey

**Authors:** Sufyan Danish, Abolghasem Sadeghi-Niaraki, Samee Ullah Khan, L. Minh Dang, Lilia Tightiz, Hyeonjoon Moon

**Full title with year:** *A comprehensive survey of Vision-Language Models: Pretrained models, fine-tuning, prompt engineering, adapters, and benchmark datasets* (2026)

**Datasets used / reviewed:** This is a survey, so it does not introduce one experimental dataset. It reviews major VLM datasets and benchmarks including MS COCO, VQAv2, GQA, CLEVR, Open Images, ADE20K, Cityscapes, Flickr30k, COCO Captions, Conceptual Captions, SBU Captions, Flickr30k Entities, RS5M, and medical VLM datasets such as VQA-Med, MIMIC-NLE, SLAKE, GEMeX, MS-CXR, 3D-RAD, MEDVQA-GI, PMC-OA, and BIOMEDICA.

**Inference from the paper:** The paper shows that modern VLM progress is not only about larger models. Practical performance depends on a combination of pretrained architectures, prompt engineering, adapter-based tuning, dataset quality, benchmark choice, and deployment constraints. The most relevant point for our work is that PEFT methods such as LoRA, BitFit, adapters, and prompt tuning can preserve most of the performance of full fine-tuning while greatly reducing trainable parameters and compute cost.

**Open problem for proposed work:** Build a modular VLM experimentation framework that jointly evaluates architecture choices, PEFT choices, and dataset choices under fixed compute limits. Important gaps to address are benchmark inconsistency, weak generalization to low-resource/domain-specific settings, limited interpretability, dataset bias, and efficient deployment.

---

## 2. Adaptive Parameter Selection for Tuning VLMs

**Authors:** Yi Zhang, Yi-Xuan Deng, Meng-Hao Guo, Shi-Min Hu

**Full title with year:** *Adaptive Parameter Selection for Tuning Vision-Language Models* (2025)

**Datasets used:** Caltech101, DTD, EuroSAT, FGVC Aircraft, Flowers102, Food101, ImageNet, OxfordPets, StanfordCars, SUN397, UCF101, ImageNet-Sketch, and ImageNetV2.

**Inference from the paper:** The paper proposes CLIP-AST, which automatically selects important CLIP parameters for fine-tuning using AdamW second-moment gradient statistics. Instead of adding prompts or adapters at manually chosen locations, it first estimates parameter importance and then fine-tunes the top selected sub-layers. This improves few-shot, base-to-novel, and out-of-distribution performance without adding inference-time parameters.

**Open problem for proposed work:** Our VLM pipeline should not only compare fixed LoRA/adapters at predefined layers. We should add an adaptive layer-selection experiment where the model identifies which vision, projection, fusion, or language layers need tuning for each dataset. A gap remains in extending CLIP-AST-style selection beyond CLIP classification to generative VLMs, VQA, captioning, and domain-specific data.

---

## 3. Contrastive Alignment for VLMs

**Authors:** Kenan E. Ak, Jay Mohta, Dimitris Dimitriadis, Saurav Manchanda, Yan Xu, Mingwei Shen

**Full title with year:** *Aligning Vision Language Models with Contrastive Learning* (2025)

**Datasets used:** Stage 1 uses 558K image-text pairs from LAION, Conceptual Captions, and SBU. Stage 2 uses LLaVA-style instruction tuning data including COCO, GQA, OCR-VQA, Text-VQA, and Visual Genome. Evaluation uses MMBench, MMBench-CN, MME, Seed-Bench, MM-VET, and MMMU.

**Inference from the paper:** The paper identifies a weakness in next-token-prediction-only alignment: image and text embeddings can remain poorly aligned even after pretraining. Adding contrastive loss alongside next-token prediction improves multimodal performance by about 2 percent without extra training data or major extra compute. The paper also shows that projection layers, vision encoder strength, and LLM choice affect alignment quality.

**Open problem for proposed work:** Our baseline should include an explicit ablation between next-token-only alignment and joint contrastive plus next-token alignment. We should also test whether this benefit remains under small Kaggle-scale datasets and with lightweight projection layers.

---

## 4. KDA-Tuning

**Authors:** Zhengdong Zhou, Chenhao Ding, Qilong Xue

**Full title with year:** *KDA-Tuning: Knowledge-Decoupled Adapter Tuning for Vision-Language Models* (2025)

**Datasets used:** Few-shot experiments use ImageNet, StanfordCars, Caltech101, UCF101, Flowers102, Food101, DTD, EuroSAT, FGVCAircraft, OxfordPets, and SUN397. Domain generalization uses ImageNet as source and ImageNet-V2, ImageNet-Sketch, ImageNet-A, and ImageNet-R as shifted target datasets.

**Inference from the paper:** KDA-Tuning addresses overfitting in adapter tuning by separating general knowledge and task-specific knowledge into two adapter branches. The general branch is supervised to preserve frozen CLIP behavior, while the task branch improves task-specific visual-text alignment. Dynamic fusion combines the branches. The result is better few-shot performance and stronger domain generalization than several prompt/adaptation baselines.

**Open problem for proposed work:** For our VLM, adapters should not be treated as one generic module. We can test dual-branch adapters that separately preserve base-model knowledge and learn domain-specific task features. The open gap is whether this idea transfers from CLIP-style classification to VQA/captioning and to multimodal fusion layers.

---

## 5. Molmo and PixMo

**Authors:** Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, Jiasen Lu, Taira Anderson, Erin Bransom, Kiana Ehsani, Huong Ngo, YenSung Chen, Ajay Patel, Mark Yatskar, Chris Callison-Burch, Andrew Head, Rose Hendrix, Favyen Bastani, Eli VanderBilt, Nathan Lambert, Yvonne Chou, Arnavi Chheda, Jenna Sparks, Sam Skjonsberg, Michael Schmitz, Aaron Sarnat, Byron Bischoff, Pete Walsh, Chris Newell, Piper Wolters, Tanmay Gupta, Kuo-Hao Zeng, Jon Borchardt, Dirk Groeneveld, Crystal Nam, Sophie Lebrecht, Caitlin Wittlif, Carissa Schoenick, Oscar Michel, Ranjay Krishna, Luca Weihs, Noah A. Smith, Hannaneh Hajishirzi, Ross Girshick, Ali Farhadi, Aniruddha Kembhavi

**Full title with year:** *Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models* (2025)

**Datasets used:** The paper introduces PixMo, including PixMo-Cap, PixMo-AskModelAnything, PixMo-Points, PixMo-CapQA, PixMo-Docs, PixMo-Clocks, and PixMo-Count. It also uses open-source training/evaluation datasets such as VQA v2.0, TextVQA, OK-VQA, ChartQA, DocVQA, InfographicVQA, AI2D, A-OKVQA, AndroidControl, ScienceQA, TabMWP, ST-VQA, TallyQA, DVQA, FigureQA, PlotQA, RealWorldQA, MMMU, MathVista, CountBenchQA, and PixMo-Count.

**Inference from the paper:** Molmo shows that strong open VLMs can be built without distilling proprietary VLM outputs if the dataset is carefully designed. Its key contribution is not just model scale, but high-quality data collection: dense spoken captions, free-form Q&A, pointing supervision, and targeted synthetic data. The architecture itself is relatively standard: vision encoder, connector, tokenizer, and decoder-only LLM.

**Open problem for proposed work:** Our proposed VLM should treat data quality and supervision type as experimental variables, not just model layers. A useful direction is to add pointing/grounding or coordinate supervision to improve spatial answers. A remaining open problem is how to reproduce Molmo-like data quality at small scale on Kaggle without expensive annotation.

---

## 6. PEFT Post-Transformer Survey

**Authors:** Patalee Narasinghe, B.H. Sudantha

**Full title with year:** *Parameter-Efficient Fine-Tuning for Vision-Language Models: The Post-Transformer Evolution* (2026)

**Datasets used / benchmarked:** This is a survey/analysis paper. It discusses benchmark results on ImageNet, VQAv2, GQA, VisWiz, ScienceQA, TextVQA, POPE, and MMBench, and references domain datasets such as PubMed Central biomedical image-text data through BiomedCLIP and remote-sensing data through RemoteCLIP.

**Inference from the paper:** PEFT is essential for VLMs because full fine-tuning is expensive, storage-heavy, and vulnerable to catastrophic forgetting. The paper groups PEFT into input-level prompting, feature-level adapters, and weight-level reparameterization such as LoRA, QLoRA, DoRA, PiSSA, and LoftQ. Different methods have different trade-offs: adapters may add latency, LoRA can be merged for inference, QLoRA reduces memory, and DoRA/PiSSA improve LoRA stability and convergence.

**Open problem for proposed work:** The proposed work should compare PEFT methods using the same validation split, memory budget, trainable parameter count, and inference latency. A strong open problem is choosing PEFT dynamically based on the target task: retrieval, VQA, captioning, domain adaptation, or edge deployment.

---

## 7. SynthVLM

**Authors:** Zheng Liu, Hao Liang, Bozhou Li, Wentao Xiong, Chong Chen, Conghui He, Wentao Zhang, Bin Cui

**Full title with year:** *SynthVLM: Towards High-Quality and Efficient Synthesis of Image-Caption Datasets for Vision-Language Models* (2025)

**Datasets used:** The paper introduces SynthVLM-100K, generated from a 1M-caption pool. Caption sources include LAION, Conceptual Captions, SBU, COCO, and BLIP2-DataComp-style captions. It compares against COCO-Caption, BLIP-LCS, ShareGPT4V, ShareGPT4V-PT, and LLaVA-558K. SFT uses LLaVA-665K. Evaluation includes ScienceQA, image-based ScienceQA, MMVet, VizWiz, VQAv2, GQA, MMBench, MME, POPE, and MMLU.

**Inference from the paper:** The paper argues that low-quality web images and weak image-text alignment are major bottlenecks. SynthVLM reverses the usual image-to-caption pipeline by filtering high-quality captions, generating images using diffusion models, and then selecting aligned image-caption pairs using CLIPScore and SSIM. With only 100K synthetic pairs, the resulting models outperform LLaVA baselines trained with much more pretraining data.

**Open problem for proposed work:** Our Kaggle experiments should include a data-quality axis: raw data, filtered data, synthetic data, and filtered synthetic data. The open problem is how to verify that synthetic image-caption pairs improve real-world generalization rather than only improving benchmark alignment.

---

## 8. Remote Sensing VLM Survey

**Authors:** Xingxing Weng, Chao Pang, Gui-Song Xia

**Full title with year:** *Vision-Language Modeling Meets Remote Sensing: Models, datasets, and perspectives* (2025)

**Datasets used / reviewed:** This is a remote-sensing VLM survey. It reviews pretraining, instruction-following, and benchmark datasets including UCM-captions, Sydney-captions, RSICD, RS5M, GeoRSCLIP-related datasets, RemoteCLIP-related datasets, DIOR, DOTA, FAIR1M, LEVIR, NWPU-RESISC45, RSVQA-LR, RSVQA-HR, RSVG, DIOR-RSVG, RSITMD, NWPU-Captions, LHRS-Align, LHRS-Bench, VRSBench, GEOBench-VLM, and other optical, SAR, IR, detection, captioning, VQA, grounding, and generation datasets.

**Inference from the paper:** Remote sensing VLMs need more than generic natural-image VLM design. They must handle sensor variation, geographic distribution shifts, scale changes, domain-specific semantics, and specialized tasks such as classification, captioning, VQA, retrieval, grounding, text-conditioned generation, change detection, and urban prediction. The survey emphasizes the pretraining-then-fine-tuning paradigm and divides RS VLMs into contrastive, instruction-based, and generation-based methods.

**Open problem for proposed work:** If our proposed work includes a remote-sensing or domain-specific track, it should test domain adaptation explicitly. Key gaps are alignment across optical/SAR/IR/geospatial/vector/social data, vague natural-language task requirements, expert explanations for reliability, continual adaptation without forgetting, richer multimodal datasets, and harder application-specific benchmarks.

---

## 9. VisionCore

**Authors:** Ufuk Ozkul, Levent Karacan, Cemil Zalluhoglu

**Full title with year:** *VisionCore: Adapting Efficient Visual Answering Through Enhanced Spatial Reasoning* (2025)

**Datasets used:** GQA is the main dataset, using scene descriptions, bounding boxes, coordinates, and directional relationships. The paper evaluates on GQA test-dev and VQAv2 validation. It also experiments with HaloQuest for hallucination/non-existing-object questions.

**Inference from the paper:** VisionCore shows that small VLMs can improve spatial reasoning through coordinate-aware instruction tuning and LoRA. It uses DeepSeek-VL 1.3B, freezes the base model, and trains only about 15M LoRA parameters. GQA scene descriptions inject explicit spatial priors such as object coordinates and left/right/above/below relations. The method reaches 60.3% on GQA, approaching larger models while using fewer parameters.

**Open problem for proposed work:** Add spatial metadata as an optional training feature and test whether coordinate-aware prompts improve VQA/captioning. The main gap is robustness: VisionCore overfits to a single conversation format, so proposed work should use multi-style prompt augmentation and adaptive fine-tuning across prompt formats.

---

## 10. VL-PET

**Authors:** Zi-Yuan Hu, Yanyang Li, Michael R. Lyu, Liwei Wang

**Full title with year:** *VL-PET: Vision-and-Language Parameter-Efficient Tuning via Granularity Control* (2023)

**Datasets used:** Image-text tasks use VQAv2, GQA, NLVR2, and MSCOCO. Video-text tasks use TVQA, How2QA, TVC, and YC2C from the VALUE benchmark.

**Inference from the paper:** VL-PET shows that PET modules need careful control and placement. Excessive modular modifications can harm performance, and encoder-decoder VLMs have different needs: encoders need stronger visual-language alignment and modeling, while decoders should preserve text generation ability. The proposed granularity-controlled mechanism creates variants with different parameter-efficiency/performance trade-offs and shows strong results on image-text and video-text tasks.

**Open problem for proposed work:** This paper directly supports our layer-permutation plan. We should test where PEFT modules are inserted: vision encoder, text encoder, projection layer, fusion layer, decoder cross-attention, and task head. The open problem is to design a search strategy that finds the best PET placement automatically under Kaggle compute limits.

---

## Consolidated Open Problems for Proposed Work

1. **Adaptive tuning location:** Automatically choose which layers/modules to tune instead of fixing LoRA or adapters manually.
2. **Joint alignment objective:** Compare next-token-only training with contrastive plus next-token training.
3. **Knowledge retention:** Prevent PEFT from overfitting to small task data by preserving base-model general knowledge.
4. **Data quality:** Include dataset filtering, synthetic data, and alignment scoring as first-class experiment variables.
5. **Spatial reasoning:** Add coordinate, bounding-box, or pointing supervision for tasks requiring object relationships.
6. **Domain adaptation:** Test whether the architecture survives domain shifts such as remote sensing, medical, or other specialized Kaggle datasets.
7. **Prompt robustness:** Train and evaluate using multiple prompt styles to avoid overfitting to one instruction template.
8. **Compute-aware search:** Track trainable parameters, GPU memory, epoch time, and inference latency for every architecture/PEFT variant.
9. **Benchmark reliability:** Use fixed validation splits and consistent metrics so layer permutations are comparable.
10. **Explainability:** Add interpretable outputs such as evidence regions, pointing tokens, or structured rationales where possible.
