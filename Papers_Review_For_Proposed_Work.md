# Paper Review for Proposed VLM Work

Source folder: `papers/`

This document summarizes each PDF with the fields needed for the proposed work: author names, full title with year, datasets used, inference from the paper, and open problems that can guide our VLM architecture and Kaggle experiments.

## Quick Comparison

| # | Paper | Main Use For Proposed Work |
|---|---|---|
| 1 | WildFireVQA | Target dataset and benchmark for RGB-thermal wildfire VQA |
| 2 | UAV Swarms and VLM Forest Fire System | System architecture for VLM-assisted wildfire detection and response |
| 3 | CLIP-AST | Adaptive layer/parameter selection instead of manually choosing tuning locations |
| 4 | Contrastive alignment | Use contrastive loss with next-token prediction during alignment |
| 5 | KDA-Tuning | Preserve general knowledge while learning task-specific adapters |
| 6 | Molmo and PixMo | Open-data VLM recipe and importance of high-quality non-distilled data |
| 7 | PEFT post-transformer survey | Compare LoRA, QLoRA, DoRA, adapters, prompts under resource limits |
| 8 | SynthVLM | Synthetic image-caption data generation and data selection |
| 9 | Remote sensing VLM survey | Domain-specific VLM gaps: multimodal RS data, explainability, continual learning |
| 10 | VisionCore | Spatial reasoning with coordinate-aware LoRA tuning |
| 11 | VL-PET | Granularity-controlled PET modules and layer placement experiments |

---

## 1. WildFireVQA

**Authors:** Mobin Habibpour, Niloufar Alipour Talemi, John Spodnik, Camren J. Khoury, Fatemeh Afghah

**Full title with year:** *WildFireVQA: A Large-Scale Radiometric Thermal VQA Benchmark for Aerial Wildfire Monitoring* (2026)

**Datasets used:** The paper introduces WildFireVQA, built on the FLAME-3 UAV wildfire dataset. It uses paired RGB imagery, color-mapped thermal visualizations, and radiometric thermal TIFFs from Sycan Marsh, Willamette, and Shoetank prescribed burns. The benchmark contains 6,097 RGB-thermal samples, 34 questions per sample, and 207,298 multiple-choice questions. Evaluation uses representative MLLMs including Qwen3-VL-8B, LLaVA-v1.6-Mistral-7B, InternVL2-8B, and MiniCPM-V2 under RGB, thermal, and retrieval-augmented settings.

**Inference from the paper:**

- Wildfire VQA needs domain-specific evaluation.
- RGB alone is strongest for current MLLMs.
- Thermal metadata helps stronger models.
- Radiometric TIFFs enable temperature-grounded labels.
- Label quality needs deterministic checks and manual review.

**Open problem for proposed work:** Build a generative RGB-thermal VLM for WildFireVQA that produces free-form answers instead of scoring candidate options. Important gaps are stronger RGB-thermal fusion, better use of radiometric temperature summaries, generalization across burn sites, and evaluation with text-generation metrics such as BLEU, ROUGE-L, exact match, and semantic similarity.

---

## 2. A Smart System for Early Detection and Prevention of Forest Fire Using UAV Swarms and VLM

**Authors:** Sarah Basahel, Adnan Ahmed Abi Sen, Nour Mahmoud Bahbouh, Omar Tayan, Adel Ben Mnaouer, Mohammad Yamin

**Full title with year:** *A Smart System for Early Detection and Prevention of Forest Fire Using UAV Swarms and VLM* (2026)

**Datasets used:** The paper does not introduce a new dataset. For preliminary image-analysis testing, it uses `Fire-Detection-Image-Dataset-master`, which includes real fire scenes and visually confusing non-fire scenes such as sunsets or red skies. The proposed operational system assumes UAV-mounted RGB cameras, thermal cameras, and sensors such as smoke, CO2, and temperature sensors, but the full UAV swarm system is not evaluated on a deployed dataset.

**Inference from the paper:**

- UAV swarms can improve forest coverage.
- Spiral coverage supports systematic monitoring.
- Fog nodes reduce response latency.
- VLMs detect smoke, flames, and thermal anomalies.
- Response drones can act before fire spreads.
- The evaluation is still preliminary.

**Open problem for proposed work:** Convert this conceptual UAV-VLM framework into a measurable VQA or detection pipeline. Important gaps are full-system simulation, real UAV deployment, latency measurement, false-alarm control, RGB-thermal fusion, and evaluation on wildfire-specific datasets such as WildFireVQA instead of only binary fire/non-fire images.

---

## 3. Adaptive Parameter Selection for Tuning VLMs

**Authors:** Yi Zhang, Yi-Xuan Deng, Meng-Hao Guo, Shi-Min Hu

**Full title with year:** *Adaptive Parameter Selection for Tuning Vision-Language Models* (2025)

**Datasets used:** Caltech101, DTD, EuroSAT, FGVC Aircraft, Flowers102, Food101, ImageNet, OxfordPets, StanfordCars, SUN397, UCF101, ImageNet-Sketch, and ImageNetV2.

**Inference from the paper:**

- CLIP-AST selects tunable CLIP parameters automatically.
- It uses AdamW second-moment gradient statistics.
- It fine-tunes only the most important sub-layers.
- It improves few-shot, novel-class, and OOD accuracy.
- It adds no inference-time parameters.

**Open problem for proposed work:** Our VLM pipeline should not only compare fixed LoRA/adapters at predefined layers. We should add an adaptive layer-selection experiment where the model identifies which vision, projection, fusion, or language layers need tuning for each dataset. A gap remains in extending CLIP-AST-style selection beyond CLIP classification to generative VLMs, VQA, captioning, and domain-specific data.

---

## 4. Contrastive Alignment for VLMs

**Authors:** Kenan E. Ak, Jay Mohta, Dimitris Dimitriadis, Saurav Manchanda, Yan Xu, Mingwei Shen

**Full title with year:** *Aligning Vision Language Models with Contrastive Learning* (2025)

**Datasets used:** Stage 1 uses 558K image-text pairs from LAION, Conceptual Captions, and SBU. Stage 2 uses LLaVA-style instruction tuning data including COCO, GQA, OCR-VQA, Text-VQA, and Visual Genome. Evaluation uses MMBench, MMBench-CN, MME, Seed-Bench, MM-VET, and MMMU.

**Inference from the paper:**

- Next-token loss alone can leave weak image-text alignment.
- Contrastive loss improves alignment and benchmark performance.
- The gain is about 2 percent in their experiments.
- Projection design strongly affects results.
- Vision encoder and LLM choice also matter.

**Open problem for proposed work:** Our baseline should include an explicit ablation between next-token-only alignment and joint contrastive plus next-token alignment. We should also test whether this benefit remains under small Kaggle-scale datasets and with lightweight projection layers.

---

## 5. KDA-Tuning

**Authors:** Zhengdong Zhou, Chenhao Ding, Qilong Xue

**Full title with year:** *KDA-Tuning: Knowledge-Decoupled Adapter Tuning for Vision-Language Models* (2025)

**Datasets used:** Few-shot experiments use ImageNet, StanfordCars, Caltech101, UCF101, Flowers102, Food101, DTD, EuroSAT, FGVCAircraft, OxfordPets, and SUN397. Domain generalization uses ImageNet as source and ImageNet-V2, ImageNet-Sketch, ImageNet-A, and ImageNet-R as shifted target datasets.

**Inference from the paper:**

- KDA-Tuning separates general and task-specific knowledge.
- One branch preserves frozen CLIP behavior.
- The other branch learns task-specific alignment.
- Dynamic fusion combines both branches.
- It improves few-shot and domain generalization results.

**Open problem for proposed work:** For our VLM, adapters should not be treated as one generic module. We can test dual-branch adapters that separately preserve base-model knowledge and learn domain-specific task features. The open gap is whether this idea transfers from CLIP-style classification to VQA/captioning and to multimodal fusion layers.

---

## 6. Molmo and PixMo

**Authors:** Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, Jiasen Lu, Taira Anderson, Erin Bransom, Kiana Ehsani, Huong Ngo, YenSung Chen, Ajay Patel, Mark Yatskar, Chris Callison-Burch, Andrew Head, Rose Hendrix, Favyen Bastani, Eli VanderBilt, Nathan Lambert, Yvonne Chou, Arnavi Chheda, Jenna Sparks, Sam Skjonsberg, Michael Schmitz, Aaron Sarnat, Byron Bischoff, Pete Walsh, Chris Newell, Piper Wolters, Tanmay Gupta, Kuo-Hao Zeng, Jon Borchardt, Dirk Groeneveld, Crystal Nam, Sophie Lebrecht, Caitlin Wittlif, Carissa Schoenick, Oscar Michel, Ranjay Krishna, Luca Weihs, Noah A. Smith, Hannaneh Hajishirzi, Ross Girshick, Ali Farhadi, Aniruddha Kembhavi

**Full title with year:** *Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models* (2025)

**Datasets used:** The paper introduces PixMo, including PixMo-Cap, PixMo-AskModelAnything, PixMo-Points, PixMo-CapQA, PixMo-Docs, PixMo-Clocks, and PixMo-Count. It also uses open-source training/evaluation datasets such as VQA v2.0, TextVQA, OK-VQA, ChartQA, DocVQA, InfographicVQA, AI2D, A-OKVQA, AndroidControl, ScienceQA, TabMWP, ST-VQA, TallyQA, DVQA, FigureQA, PlotQA, RealWorldQA, MMMU, MathVista, CountBenchQA, and PixMo-Count.

**Inference from the paper:**

- Strong open VLMs do not require proprietary distillation.
- Data quality is the main contribution.
- Useful supervision includes captions, Q&A, and points.
- Targeted synthetic data also helps.
- The architecture is fairly standard.

**Open problem for proposed work:** Our proposed VLM should treat data quality and supervision type as experimental variables, not just model layers. A useful direction is to add pointing/grounding or coordinate supervision to improve spatial answers. A remaining open problem is how to reproduce Molmo-like data quality at small scale on Kaggle without expensive annotation.

---

## 7. PEFT Post-Transformer Survey

**Authors:** Patalee Narasinghe, B.H. Sudantha

**Full title with year:** *Parameter-Efficient Fine-Tuning for Vision-Language Models: The Post-Transformer Evolution* (2026)

**Datasets used / benchmarked:** This is a survey/analysis paper. It discusses benchmark results on ImageNet, VQAv2, GQA, VisWiz, ScienceQA, TextVQA, POPE, and MMBench, and references domain datasets such as PubMed Central biomedical image-text data through BiomedCLIP and remote-sensing data through RemoteCLIP.

**Inference from the paper:**

- Full fine-tuning is costly and storage-heavy.
- It can also cause catastrophic forgetting.
- PEFT includes prompts, adapters, and reparameterization.
- Key methods include LoRA, QLoRA, DoRA, PiSSA, and LoftQ.
- Each method trades off memory, latency, and stability.

**Open problem for proposed work:** The proposed work should compare PEFT methods using the same validation split, memory budget, trainable parameter count, and inference latency. A strong open problem is choosing PEFT dynamically based on the target task: retrieval, VQA, captioning, domain adaptation, or edge deployment.

---

## 8. SynthVLM

**Authors:** Zheng Liu, Hao Liang, Bozhou Li, Wentao Xiong, Chong Chen, Conghui He, Wentao Zhang, Bin Cui

**Full title with year:** *SynthVLM: Towards High-Quality and Efficient Synthesis of Image-Caption Datasets for Vision-Language Models* (2025)

**Datasets used:** The paper introduces SynthVLM-100K, generated from a 1M-caption pool. Caption sources include LAION, Conceptual Captions, SBU, COCO, and BLIP2-DataComp-style captions. It compares against COCO-Caption, BLIP-LCS, ShareGPT4V, ShareGPT4V-PT, and LLaVA-558K. SFT uses LLaVA-665K. Evaluation includes ScienceQA, image-based ScienceQA, MMVet, VizWiz, VQAv2, GQA, MMBench, MME, POPE, and MMLU.

**Inference from the paper:**

- Poor web data limits VLM training.
- Weak image-text alignment is a major bottleneck.
- SynthVLM filters captions before generating images.
- It uses CLIPScore and SSIM for pair selection.
- 100K synthetic pairs outperform larger LLaVA pretraining sets.

**Open problem for proposed work:** Our Kaggle experiments should include a data-quality axis: raw data, filtered data, synthetic data, and filtered synthetic data. The open problem is how to verify that synthetic image-caption pairs improve real-world generalization rather than only improving benchmark alignment.

---

## 9. Remote Sensing VLM Survey

**Authors:** Xingxing Weng, Chao Pang, Gui-Song Xia

**Full title with year:** *Vision-Language Modeling Meets Remote Sensing: Models, datasets, and perspectives* (2025)

**Datasets used / reviewed:** This is a remote-sensing VLM survey. It reviews pretraining, instruction-following, and benchmark datasets including UCM-captions, Sydney-captions, RSICD, RS5M, GeoRSCLIP-related datasets, RemoteCLIP-related datasets, DIOR, DOTA, FAIR1M, LEVIR, NWPU-RESISC45, RSVQA-LR, RSVQA-HR, RSVG, DIOR-RSVG, RSITMD, NWPU-Captions, LHRS-Align, LHRS-Bench, VRSBench, GEOBench-VLM, and other optical, SAR, IR, detection, captioning, VQA, grounding, and generation datasets.

**Inference from the paper:**

- Remote sensing needs domain-specific VLM design.
- Models must handle sensor and geography shifts.
- Scale variation is a core challenge.
- Tasks include VQA, captioning, retrieval, and grounding.
- RS VLMs are contrastive, instruction-based, or generative.

**Open problem for proposed work:** If our proposed work includes a remote-sensing or domain-specific track, it should test domain adaptation explicitly. Key gaps are alignment across optical/SAR/IR/geospatial/vector/social data, vague natural-language task requirements, expert explanations for reliability, continual adaptation without forgetting, richer multimodal datasets, and harder application-specific benchmarks.

---

## 10. VisionCore

**Authors:** Ufuk Ozkul, Levent Karacan, Cemil Zalluhoglu

**Full title with year:** *VisionCore: Adapting Efficient Visual Answering Through Enhanced Spatial Reasoning* (2025)

**Datasets used:** GQA is the main dataset, using scene descriptions, bounding boxes, coordinates, and directional relationships. The paper evaluates on GQA test-dev and VQAv2 validation. It also experiments with HaloQuest for hallucination/non-existing-object questions.

**Inference from the paper:**

- Small VLMs can improve spatial reasoning.
- Coordinate-aware instruction tuning helps.
- VisionCore freezes DeepSeek-VL 1.3B.
- It trains only about 15M LoRA parameters.
- GQA scene metadata adds spatial priors.
- It reaches 60.3% on GQA.

**Open problem for proposed work:** Add spatial metadata as an optional training feature and test whether coordinate-aware prompts improve VQA/captioning. The main gap is robustness: VisionCore overfits to a single conversation format, so proposed work should use multi-style prompt augmentation and adaptive fine-tuning across prompt formats.

---

## 11. VL-PET

**Authors:** Zi-Yuan Hu, Yanyang Li, Michael R. Lyu, Liwei Wang

**Full title with year:** *VL-PET: Vision-and-Language Parameter-Efficient Tuning via Granularity Control* (2023)

**Datasets used:** Image-text tasks use VQAv2, GQA, NLVR2, and MSCOCO. Video-text tasks use TVQA, How2QA, TVC, and YC2C from the VALUE benchmark.

**Inference from the paper:**

- PET placement strongly affects performance.
- Too many module changes can hurt.
- Encoders need stronger visual-language alignment.
- Decoders must preserve generation ability.
- Granularity control balances efficiency and accuracy.

**Open problem for proposed work:** This paper directly supports our layer-permutation plan. We should test where PEFT modules are inserted: vision encoder, text encoder, projection layer, fusion layer, decoder cross-attention, and task head. The open problem is to design a search strategy that finds the best PET placement automatically under Kaggle compute limits.

---

## Consolidated Research Gap for Proposed Work

Existing wildfire VLM work is still limited in three ways: most systems focus on fire detection or multiple-choice scoring, thermal evidence is not deeply fused with RGB reasoning, and few methods test lightweight generative VLMs under realistic Kaggle-scale compute limits. The proposed work should fill this gap by building a custom RGB-thermal VLM that generates answers for WildFireVQA and evaluates them with text-generation metrics.

1. **Free-form VQA generation:** Current WildFireVQA baselines score options; our model should generate answers directly.
2. **RGB-thermal fusion:** Existing models use RGB better than thermal; stronger fusion is needed.
3. **Radiometric reasoning:** Thermal summaries and TIFF-derived statistics should guide answer generation.
4. **Wildfire-specific adaptation:** Generic VLMs need tuning for smoke, hotspots, burn regions, and UAV flight context.
5. **Layer permutation search:** Vision, projection, fusion, and decoder layers should be compared systematically.
6. **Parameter-efficient tuning:** LoRA, adapters, and selective tuning should be tested under the same split.
7. **Generative evaluation:** BLEU, ROUGE-L, exact match, and semantic similarity should replace option accuracy.
8. **Cross-site generalization:** The model should be tested across Sycan, Willamette, and Shoetank-style shifts.
9. **Data availability handling:** The pipeline must resolve separate WildFireVQA annotations and FLAME-3 images.
10. **Operational reliability:** False alarms, weak thermal interpretation, and missing visual evidence remain open issues.
