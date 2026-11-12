# firevlm-qwen2-vl-1.5b-lora-balanced-qkvo-r8.ipynb
- cells: 16
- code cells with text output: 11
- non-text outputs omitted (images/html): 63
- cells whose output contains an error: 0

## [cell 0] markdown
# FireVLM: Qwen2 1.5B + balanced Q/K/V/O QLoRA, rank 8

Qwen2 parallel of `firevlm-lora-balanced-qkvo-r8.ipynb`, adapted for
`mit-han-lab/Qwen2-VL-1.5B-Instruct`. Run all cells in order on a CUDA GPU
(Kaggle T4 or better, Internet enabled; Python 3.10+ and PyTorch 2.3+).
The dependency cell uses the environment's existing CUDA-enabled PyTorch.

**Checkpoint detail:** despite the repository name, its model card and configuration
identify an extracted **Qwen2ForCausalLM language component**, without a vision tower.
This notebook therefore retains the reference's frozen CLIP ViT-B/32 encoder,
separate trainable RGB/thermal projectors, and modality embeddings. It does not
load this checkpoint as a native Qwen2-VL vision model. The projectors start from
random weights and are learned along with the LoRA adapters.

- Language backbone: 4-bit NF4 with double quantization; frozen base weights.
- LoRA: `q_proj`, `k_proj`, `v_proj`, and `o_proj`, **r=8**, alpha=16, dropout=0.05.
- Efficiency: cache frozen CLIP patch features once on CPU, use native FP16 on Kaggle T4,
  and match the proven Llama notebook's gradient-accumulation training path.
- Data: the same WildFireVQA + FLAME-3 Sycan RGB/thermal pairing audit and
  image-grouped train/validation/test split as the reference. Canonical pairing
  runs before dropping unresolved paths, avoiding premature filename filtering.
- Validation loss selects the best adapters/projectors; held-out test generation
  reports exact match, the reference's custom smoothed sentence BLEU-4, and ROUGE-L.

Configure paths and hyperparameters in `CFG`. Public model downloads require no
Hugging Face token. Kaggle datasets may require your Kaggle account credentials.
For Colab, set `CFG.OUTPUT_DIR` to a persistent location if desired.
The feature cache uses host RAM proportional to the number of unique images.

Saved artifacts include the best LoRA adapters, both projectors/type embeddings,
tokenizer, run configuration, split manifests, pairing audit, history and test results.
The adapter alone is insufficient for image inference: load the projectors as well.

Sources: [model card](https://huggingface.co/mit-han-lab/Qwen2-VL-1.5B-Instruct),
[model config](https://huggingface.co/mit-han-lab/Qwen2-VL-1.5B-Instruct/blob/main/config.json),
[PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization).

## [cell 1] code
```python
# Run in a fresh kernel. Restart the kernel after installation if these packages
# were already imported. PyTorch/torchvision are supplied by Kaggle or Colab.
%pip install -q "transformers==4.57.1" "peft==0.17.1" "accelerate==1.10.1" "bitsandbytes==0.48.1" kagglehub pandas matplotlib pillow tqdm pyarrow jinja2
```
OUTPUT:
```text
[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
gradio 6.26.0 requires huggingface-hub<2.0,>=1.16.0, but you have huggingface-hub 0.36.2 which is incompatible.
gradio 6.26.0 requires pillow<13.0,>=8.0, but you have pillow 13.0.0 which is incompatible.
diffusers 0.40.0 requires huggingface-hub<2.0,>=1.23.0, but you have huggingface-hub 0.36.2 which is incompatible.[0m[31m
[0mNote: you may need to restart the kernel to use updated packages.
```

## [cell 2] code
```python
# Optional: the requested model and CLIP are public, so login is unnecessary.
# If your environment requires authentication, expose HF_TOKEN through an
# environment variable or Kaggle Secrets; never paste a token into the notebook.
import os
from huggingface_hub import login

hf_token = os.environ.get('HF_TOKEN')
if hf_token:
    login(token=hf_token, add_to_git_credential=False)
del hf_token
```

## [cell 3] code
```python
import os
import re
import json
import math
import random
import time
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

# Reduce CUDA fragmentation during long QLoRA runs. Must be set before importing torch.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm

try:
    import kagglehub
except Exception as e:
    kagglehub = None
    print('[WARN] kagglehub import failed:', repr(e))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    CLIPVisionModel,
    CLIPImageProcessor,
    get_cosine_schedule_with_warmup,
)

print('Torch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA device count:', torch.cuda.device_count())
    for gpu_idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(gpu_idx)
        print(f'GPU {gpu_idx}: {props.name} ({props.total_memory / 2**30:.1f} GiB)')
```
OUTPUT:
```text
Torch version: 2.11.0+cu128
CUDA available: True
CUDA device count: 2
GPU 0: Tesla T4 (14.6 GiB)
GPU 1: Tesla T4 (14.6 GiB)
```

## [cell 4] code
```python
class CFG:
    # ----------------- Data Sources -----------------
    DATA_SOURCE = 'kagglehub'
    KAGGLE_DATASET = 'caseypiere/wildfire-vqa'
    FLAME3_KAGGLE_DATASET = 'brycehopkins/flame-3-computer-vision-subset-sycan-marsh'
    DATA_ROOT = Path('/kaggle/input/datasets/caseypiere/wildfire-vqa')
    FLAME3_ROOT = None
    FLAME3_MARKER_DIR = 'Flame 3 Computer Vision Sets'
    ANNOTATION_FILE = None

    # Only rows whose site folder starts with this prefix are kept.
    ALLOWED_SITE_PREFIX = 'Sycan'

    # ----------------- Data Sampling -----------------
    MAX_TRAIN_SAMPLES = None  # Use every resolved training row
    MAX_VAL_SAMPLES = None
    VAL_FRAC = 0.10
    TEST_FRAC = 0.10
    MAX_TEST_SAMPLES = None
    SEED = 42

    # ----------------- Models -----------------
    VISION_MODEL = 'openai/clip-vit-base-patch32'  # 7x7 = 49 patch tokens per image
    LLM_MODEL = 'mit-han-lab/Qwen2-VL-1.5B-Instruct'
    USE_4BIT = True

    # ----------------- Balanced QLoRA -----------------
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    # Attention-only LoRA is much smaller and faster than adapting all MLP layers.
    LORA_TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    PROJECTOR_HIDDEN_DIM = 1024
    # Match the fast Llama baseline: recomputation is unnecessary for this 1.5B QLoRA run.
    GRADIENT_CHECKPOINTING = False

    # ----------------- Prompts & Tokens -----------------
    # The model generates an answer from the imagery rather than selecting an option.
    INCLUDE_OPTIONS_IN_PROMPT = False
    INCLUDE_TEMP_SUMMARY = False
    MAX_TARGET_TOKENS = 32
    MAX_GENERATE_TOKENS = 32

    # ----------------- Training -----------------
    BATCH_SIZE = 4
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 3
    EARLY_STOPPING_MIN_DELTA = 1e-8
    LR = 2e-4
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.05
    GRAD_ACCUM_STEPS = 4     # Effective batch size = 16
    GRAD_CLIP = 1.0
    NUM_WORKERS = 0          # Features/tokens are cached; worker processes add overhead.
    VISION_CACHE_BATCH_SIZE = 16
    LOG_EVERY = 25
    OUTPUT_DIR = Path('/kaggle/working/firevlm_qwen2_1_5b_lora_balanced_qkvo_r8')

CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
random.seed(CFG.SEED)
np.random.seed(CFG.SEED)
torch.manual_seed(CFG.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG.SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
```

## [cell 5] code
```python
def load_json_any(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def flatten_checkpoint_json(obj: Dict[str, Any], source_name: str) -> List[Dict[str, Any]]:
    rows = []
    items = obj.get('items', {})
    if not isinstance(items, dict):
        return rows
    for ck, entry in items.items():
        if not isinstance(entry, dict):
            continue
        meta = entry.get('meta', {})
        if not isinstance(meta, dict):
            continue
        row = dict(meta)
        row['gt_answer'] = entry.get('human_answer', row.get('gt_answer', row.get('answer')))
        row['checkpoint_key'] = ck
        row['source_input'] = source_name
        rows.append(row)
    return rows

def normalize_options(x):
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except Exception:
            pass
        if '|' in s:
            return [v.strip() for v in s.split('|') if v.strip()]
        if ';' in s:
            return [v.strip() for v in s.split(';') if v.strip()]
        return [s]
    return []

def rows_from_json(path: Path) -> List[Dict[str, Any]]:
    obj = load_json_any(path)
    if isinstance(obj, dict) and obj.get('type') == 'checkpoint':
        return flatten_checkpoint_json(obj, str(path))
    if isinstance(obj, dict):
        for key in ['items', 'annotations', 'data', 'records', 'questions']:
            if isinstance(obj.get(key), list):
                return [r for r in obj[key] if isinstance(r, dict)]
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    return []

def discover_annotation_files(root: Path):
    exts = ['*.json', '*.jsonl', '*.csv', '*.parquet']
    files = []
    if root.exists():
        for ext in exts:
            files.extend(root.rglob(ext))
    priority = ('vqa', 'question', 'answer', 'annotation', 'response', 'train', 'val')
    return sorted(files, key=lambda p: (not any(w in p.name.lower() for w in priority), len(str(p))))

def load_rows_from_files():
    if CFG.ANNOTATION_FILE:
        files = [Path(CFG.ANNOTATION_FILE)]
    else:
        files = discover_annotation_files(CFG.DATA_ROOT)
    if not files:
        raise FileNotFoundError(f'No annotation files found under {CFG.DATA_ROOT}')

    all_rows = []
    for p in files:
        try:
            if p.suffix.lower() == '.json':
                rows = rows_from_json(p)
            elif p.suffix.lower() == '.jsonl':
                with open(p, 'r', encoding='utf-8') as f:
                    rows = [json.loads(line) for line in f if line.strip()]
            elif p.suffix.lower() == '.csv':
                rows = pd.read_csv(p).to_dict('records')
            elif p.suffix.lower() == '.parquet':
                rows = pd.read_parquet(p).to_dict('records')
            else:
                rows = []
            for r in rows:
                r.setdefault('source_input', str(p))
            all_rows.extend(rows)
        except Exception as e:
            print('[WARN] failed to load', p, repr(e))
    return all_rows

def load_rows_from_kagglehub():
    if kagglehub is None:
        raise ImportError('kagglehub is unavailable.')
    try:
        dataset_root = Path(kagglehub.dataset_download(CFG.KAGGLE_DATASET))
        CFG.DATA_ROOT = dataset_root
    except Exception as e:
        print('[WARN] kagglehub.dataset_download failed; using local path:', repr(e))
    return load_rows_from_files()

if CFG.DATA_SOURCE == 'kagglehub':
    raw_rows = load_rows_from_kagglehub()
else:
    raw_rows = load_rows_from_files()

print(f'Loaded {len(raw_rows):,} raw annotation rows.')
```
OUTPUT:
```text
Loaded 207,306 raw annotation rows.
```

## [cell 6] code
```python
def get_first(row, names, default=None):
    for n in names:
        if n in row and row[n] is not None:
            return row[n]
    return default

def normalize_answer_text(x):
    if isinstance(x, list):
        vals = [normalize_answer_text(v) for v in x]
        vals = [v for v in vals if v]
        return Counter(vals).most_common(1)[0][0] if vals else ''
    if isinstance(x, dict):
        for key in ['answer', 'label', 'text', 'gt_answer', 'human_answer']:
            if key in x:
                return normalize_answer_text(x[key])
        return json.dumps(x, sort_keys=True)
    return re.sub(r'\s+', ' ', str(x).strip()) if x is not None else ''

records = []
for row in raw_rows:
    question = get_first(row, ['question', 'query', 'prompt'], '')
    answer = get_first(row, ['gt_answer', 'answer', 'human_answer', 'label', 'target'], '')
    options = normalize_options(get_first(row, ['options', 'choices', 'candidate_answers'], []))
    rgb_path = get_first(row, ['rgb_path', 'image_path', 'image', 'img_path', 'rgb', 'file_name', 'filename'], None)
    thermal_path = get_first(row, ['thermal_path', 'thermal', 'ir_path', 'thermal_image', 'thermal_jpg'], None)
    temp_summary = get_first(row, ['temp_summary', 'temperature_summary', 'thermal_summary'], None)

    rec = {
        'image_id': str(get_first(row, ['image_id', 'frame_id', 'id'], '')),
        'question_id': str(get_first(row, ['question_id', 'qid'], '')),
        'category': str(get_first(row, ['category', 'question_category', 'group'], 'unknown')),
        'question': str(question).strip(),
        'answer': normalize_answer_text(answer),
        'options': options,
        'rgb_path': rgb_path,
        'thermal_path': thermal_path,
        'temp_summary': temp_summary if isinstance(temp_summary, dict) else None,
        'source_input': str(get_first(row, ['source_input'], '')),
    }
    if rec['question'] and rec['answer']:
        records.append(rec)

df = pd.DataFrame(records).drop_duplicates(
    subset=['image_id', 'question_id', 'question', 'answer', 'rgb_path', 'thermal_path']
).reset_index(drop=True)

# ---------------------------------------------------------------
# 1) Which FLAME-3 site does each annotation row come from?
# ---------------------------------------------------------------
MARKER = CFG.FLAME3_MARKER_DIR

def site_of(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return 'unknown'
    parts = str(p).replace('\\', '/').split('/')
    if MARKER in parts:
        i = parts.index(MARKER)
        if i + 1 < len(parts):
            return parts[i + 1]
    return 'unknown'

df['site'] = df['rgb_path'].apply(site_of)
print('Rows per site in annotations:')
print(df['site'].value_counts().to_string())

# Keep only sites present in the Kaggle subset (Sycan*).
df = df[df['site'].str.startswith(CFG.ALLOWED_SITE_PREFIX)].reset_index(drop=True)
print(f'\nRows after site filter ({CFG.ALLOWED_SITE_PREFIX}*): {len(df):,}')

# ---------------------------------------------------------------
# 2) Locate the folder on Kaggle that directly contains Fire/ and No Fire/
# ---------------------------------------------------------------
def discover_flame_base():
    roots = []
    if CFG.FLAME3_ROOT is not None:
        roots.append(Path(CFG.FLAME3_ROOT))
    if kagglehub is not None:
        try:
            roots.append(Path(kagglehub.dataset_download(CFG.FLAME3_KAGGLE_DATASET)))
        except Exception as e:
            print('[WARN] kagglehub flame3 download failed:', repr(e))
    if Path('/kaggle/input').exists():
        roots.append(Path('/kaggle/input'))
    for r in roots:
        if not r.exists():
            continue
        hits = sorted([p for p in r.rglob('Fire') if p.is_dir()], key=lambda p: len(p.parts))
        if hits:
            return hits[0].parent
    return None

FLAME_BASE = discover_flame_base()
assert FLAME_BASE is not None, 'Could not find a folder containing Fire/ in the FLAME-3 dataset.'
print('\nFLAME base folder:', FLAME_BASE)
print('Contents:', sorted(p.name for p in FLAME_BASE.iterdir()))

# Resolve canonical RGB/thermal paths in the radiometric pairing audit below.
assert len(df) > 0, 'No Sycan annotations found. Check the annotation source.'
print(f'Rows ready for canonical pairing: {len(df):,}')
```
OUTPUT:
```text
Rows per site in annotations:
site
Shoetank     120360
Wilamette     61846
Sycan         25092

Rows after site filter (Sycan*): 25,092

FLAME base folder: /kaggle/input/datasets/brycehopkins/flame-3-computer-vision-subset-sycan-marsh/FLAME 3 CV Dataset (Sycan Marsh)
Contents: ['Fire', 'No Fire']
Rows ready for canonical pairing: 25,092
```

## [cell 7] code
```python
# Strict source-to-Kaggle pairing audit.
#
# The VQA paths reset filenames inside Sycan 2A and Sycan 2D, while the Kaggle
# CV subset flattens and renumbers both collections under Fire/. Filename-only
# matching therefore maps distinct source scenes onto the same image. Match each
# source scene to its canonical Kaggle quartet using radiometric TIFF statistics.

def parse_source_identity(path_value):
    parts = str(path_value).replace('\\', '/').split('/')
    image_positions = [i for i, part in enumerate(parts) if part == 'Images']
    if not image_positions:
        raise ValueError(f'Cannot parse source identity from: {path_value}')
    index = image_positions[-1]
    if index < 1 or index + 2 >= len(parts):
        raise ValueError(f'Incomplete source path: {path_value}')
    return {
        'collection': parts[index - 1],
        'class': parts[index + 1],
        'filename': parts[-1],
    }

def files_by_stem(directory, suffixes):
    suffixes = {suffix.lower() for suffix in suffixes}
    files = {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    }
    if not files:
        raise FileNotFoundError(f'No {sorted(suffixes)} files found in {directory}')
    return files

def summary_signature(summary):
    keys = ('min', 'max', 'mean', 'std')
    if not isinstance(summary, dict) or not all(key in summary for key in keys):
        raise ValueError(f'Missing radiometric signature fields: {summary}')
    return tuple(round(float(summary[key]), 3) for key in keys)

def tiff_signature(path):
    with Image.open(path) as image:
        values = np.asarray(image, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError(f'No finite temperatures in {path}')
    return tuple(round(value, 3) for value in (
        values.min(), values.max(), values.mean(), values.std()
    ))

rgb_inventory = {}
thermal_inventory = {}
tiff_inventory = {}
canonical_stats = {}

for fire_class in ('Fire', 'No Fire'):
    class_root = FLAME_BASE / fire_class
    rgb_inventory[fire_class] = files_by_stem(
        class_root / 'RGB' / 'Corrected FOV', {'.jpg', '.jpeg', '.png'}
    )
    thermal_inventory[fire_class] = files_by_stem(
        class_root / 'Thermal' / 'Raw JPG', {'.jpg', '.jpeg', '.png'}
    )
    tiff_inventory[fire_class] = files_by_stem(
        class_root / 'Thermal' / 'Celsius TIFF', {'.tif', '.tiff'}
    )
    common_stems = (
        set(rgb_inventory[fire_class])
        & set(thermal_inventory[fire_class])
        & set(tiff_inventory[fire_class])
    )
    assert common_stems == set(rgb_inventory[fire_class]), (
        f'Incomplete RGB/thermal/TIFF quartets for {fire_class}.'
    )
    canonical_stats[fire_class] = {
        stem: tiff_signature(tiff_inventory[fire_class][stem])
        for stem in tqdm(sorted(common_stems), desc=f'Auditing {fire_class} TIFF signatures')
    }
    signature_counts = Counter(canonical_stats[fire_class].values())
    duplicate_signatures = [signature for signature, count in signature_counts.items() if count > 1]
    assert not duplicate_signatures, (
        f'Radiometric signatures are not unique for {fire_class}: {duplicate_signatures[:5]}'
    )

rgb_identity = df['rgb_path'].apply(parse_source_identity)
thermal_identity = df['thermal_path'].apply(parse_source_identity)
for field in ('collection', 'class', 'filename'):
    rgb_values = rgb_identity.apply(lambda value: value[field])
    thermal_values = thermal_identity.apply(lambda value: value[field])
    assert (rgb_values == thermal_values).all(), f'RGB/thermal source {field} mismatch.'
    df[f'source_{field}'] = rgb_values

df['source_image_key'] = (
    df['source_collection'] + '|' + df['source_class'] + '|' + df['source_filename']
)

source_meta = df.drop_duplicates('source_image_key')[
    ['source_image_key', 'source_collection', 'source_class', 'source_filename', 'temp_summary']
].copy()
assert source_meta['source_image_key'].is_unique
source_class_counts = source_meta.groupby('source_class').size().to_dict()
for fire_class, stats in canonical_stats.items():
    assert source_class_counts.get(fire_class, 0) <= len(stats), (
        f'More source images than Kaggle quartets for {fire_class}: '
        f'{source_class_counts.get(fire_class, 0)} vs {len(stats)}'
    )

used_targets = {fire_class: set() for fire_class in canonical_stats}
source_to_target = {}
audit_rows = []

for row in source_meta.itertuples(index=False):
    expected = summary_signature(row.temp_summary)
    candidates = [
        stem for stem, observed in canonical_stats[row.source_class].items()
        if observed == expected and stem not in used_targets[row.source_class]
    ]

    # The TIFF values should reproduce the annotation summary exactly after
    # rounding. A tiny fallback tolerance handles TIFF decoder precision only.
    match_mode = 'exact'
    max_abs_error = 0.0
    if len(candidates) != 1:
        available = {
            stem: observed
            for stem, observed in canonical_stats[row.source_class].items()
            if stem not in used_targets[row.source_class]
        }
        ranked = sorted(
            (
                max(abs(a - b) for a, b in zip(expected, observed)),
                stem,
                observed,
            )
            for stem, observed in available.items()
        )
        eligible = [candidate for candidate in ranked if candidate[0] <= 0.05]
        if len(eligible) != 1:
            raise RuntimeError(
                f'Expected one radiometric match for {row.source_image_key}; '
                f'found {len(eligible)} within tolerance; expected={expected}, '
                f'nearest={ranked[:3]}'
            )
        max_abs_error, matched_stem, _ = eligible[0]
        candidates = [matched_stem]
        match_mode = 'tolerance'

    matched_stem = candidates[0]
    used_targets[row.source_class].add(matched_stem)
    source_to_target[row.source_image_key] = matched_stem
    audit_rows.append({
        'source_image_key': row.source_image_key,
        'source_collection': row.source_collection,
        'source_class': row.source_class,
        'source_filename': row.source_filename,
        'canonical_stem': matched_stem,
        'match_mode': match_mode,
        'max_signature_error_celsius': float(max_abs_error),
        'expected_signature': json.dumps(expected),
        'observed_signature': json.dumps(canonical_stats[row.source_class][matched_stem]),
    })

assert len(source_to_target) == len(source_meta)
assert len(set((row.source_class, source_to_target[row.source_image_key]) for row in source_meta.itertuples())) == len(source_meta), (
    'Source-to-Kaggle mapping is not one-to-one.'
)

df['canonical_stem'] = df['source_image_key'].map(source_to_target)
df['rgb_resolved'] = df.apply(
    lambda row: str(rgb_inventory[row['source_class']][row['canonical_stem']]), axis=1
)
df['thermal_resolved'] = df.apply(
    lambda row: str(thermal_inventory[row['source_class']][row['canonical_stem']]), axis=1
)
df['pair_id'] = df['source_image_key']

assert df['rgb_resolved'].notna().all() and df['thermal_resolved'].notna().all()
assert (df['rgb_resolved'] != df['thermal_resolved']).all()
assert df.groupby('pair_id')['rgb_resolved'].nunique().eq(1).all()
assert df.groupby('pair_id')['thermal_resolved'].nunique().eq(1).all()
assert df.drop_duplicates('pair_id')['rgb_resolved'].is_unique
assert df.drop_duplicates('pair_id')['thermal_resolved'].is_unique

duplicate_questions = df.duplicated(['pair_id', 'question'], keep=False)
assert not duplicate_questions.any(), 'Duplicate question records exist within a source image.'
answer_conflicts = df.groupby(['pair_id', 'question'])['answer'].nunique()
assert answer_conflicts.max() == 1, 'Contradictory answers exist within a true source image.'

pair_audit_df = pd.DataFrame(audit_rows).sort_values(
    ['source_class', 'source_collection', 'source_filename']
)
pair_audit_df.to_csv(CFG.OUTPUT_DIR / 'source_to_kaggle_pair_audit.csv', index=False)

questions_per_image = df.groupby('pair_id').size()
print('\nSTRICT PAIRING AUDIT PASSED')
print('Source image pairs:', df['pair_id'].nunique())
print('Canonical RGB paths:', df['rgb_resolved'].nunique())
print('Canonical thermal paths:', df['thermal_resolved'].nunique())
print('Pairs by source collection and class:')
display(df.drop_duplicates('pair_id').groupby(['source_collection', 'source_class']).size().rename('pairs'))
print('Questions per image distribution:')
display(questions_per_image.value_counts().sort_index().rename('image_count'))
print('Tolerance matches:', int((pair_audit_df['match_mode'] == 'tolerance').sum()))

# Visual spot-checks show a true source pair once, followed by several of its QA labels.
audit_samples = df.drop_duplicates('pair_id').sample(
    n=min(4, df['pair_id'].nunique()), random_state=CFG.SEED
)
for sample in audit_samples.itertuples():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    axes[0].imshow(Image.open(sample.rgb_resolved).convert('RGB'))
    axes[0].set_title('Resolved RGB')
    axes[1].imshow(Image.open(sample.thermal_resolved).convert('RGB'))
    axes[1].set_title('Resolved thermal')
    for axis in axes:
        axis.axis('off')
    plt.suptitle(f'{sample.source_image_key} -> canonical {sample.canonical_stem}')
    plt.tight_layout()
    plt.show()
    display(
        df[df['pair_id'] == sample.pair_id][['question_id', 'question', 'answer']]
        .head(8)
        .reset_index(drop=True)
    )
```
OUTPUT:
```text
Auditing Fire TIFF signatures:   0%|          | 0/622 [00:00<?, ?it/s]
Auditing No Fire TIFF signatures:   0%|          | 0/116 [00:00<?, ?it/s]

STRICT PAIRING AUDIT PASSED
Source image pairs: 738
Canonical RGB paths: 738
Canonical thermal paths: 738
Pairs by source collection and class:

source_collection  source_class
Sycan 2A           Fire            364
Sycan 2D           Fire            258
                   No Fire         116
Name: pairs, dtype: int64
Questions per image distribution:

34    738
Name: image_count, dtype: int64
Tolerance matches: 4

<Figure size 800x320 with 2 Axes>
  question_id                                           question answer
0         PD1              Are active thermal hotspots detected?    Yes
1         PD2                                  Is smoke visible?    Yes
2         PD3                        Are visible flames present?     No
3         PD4  Are any buildings or residential structures vi...     No
4         PD5  Are natural fuel breaks like rock outcroppings...     No
5         PD6  Are there standing dead trees that could contr...     No
6         PD7  Are there isolated heat sources far from the m...     No
7         PD8  How many emergency vehicles are visible in the...      0
<Figure size 800x320 with 2 Axes>
  question_id                                           question answer
0         PD1              Are active thermal hotspots detected?    Yes
1         PD2                                  Is smoke visible?    Yes
2  
[... 310 chars truncated ...]
isolated heat sources far from the m...     No
7         PD8  How many emergency vehicles are visible in the...      0
<Figure size 800x320 with 2 Axes>
  question_id                                           question answer
0         PD1              Are active thermal hotspots detected?    Yes
1         PD2                                  Is smoke visible?    Yes
2         PD3                        Are visible flames present?     No
3         PD4  Are any buildings or residential structures vi...     No
4         PD5  Are natural fuel breaks like rock outcroppings...    Yes
5         PD6  Are there standing dead trees that could contr...     No
6         PD7  Are there isolated heat sources far from the m...     No
7         PD8  How many emergency vehicles are visible in the...      0
<Figure size 800x320 with 2 Axes>
  question_id                                           question   answer
0         PD1              Are active thermal hotspots detected?       No
1         PD2                                  Is smoke visible?      Yes
2         PD3                        Are visible flames present?      Yes
3         PD4  Are any buildings or residential structures vi...       No
4         PD5  Are natural fuel breaks like rock outcroppings...      Yes
5         PD6  Are there standing dead trees that could contr...      Yes
6         PD7  Are there isolated heat sources far from the m...  No Fire
7         PD8  How many emergency vehicles are visible in the...        0
```

## [cell 8] code
```python
# Grouped three-way split: every question for an image stays in exactly one split.
assert 0 < CFG.VAL_FRAC < 1 and 0 < CFG.TEST_FRAC < 1
assert CFG.VAL_FRAC + CFG.TEST_FRAC < 1
unique_images = df['rgb_resolved'].unique().tolist()
assert len(unique_images) >= 3, 'Need at least three image pairs for three splits.'
rng = np.random.default_rng(CFG.SEED)
rng.shuffle(unique_images)
n_test = max(1, int(len(unique_images) * CFG.TEST_FRAC))
n_val = max(1, int(len(unique_images) * CFG.VAL_FRAC))
test_images = set(unique_images[:n_test])
val_images = set(unique_images[n_test:n_test + n_val])
train_df = df[~df['rgb_resolved'].isin(test_images | val_images)].copy()
val_df = df[df['rgb_resolved'].isin(val_images)].copy()
test_df = df[df['rgb_resolved'].isin(test_images)].copy()

if CFG.MAX_TRAIN_SAMPLES is not None and len(train_df) > CFG.MAX_TRAIN_SAMPLES:
    train_df = train_df.sample(n=CFG.MAX_TRAIN_SAMPLES, random_state=CFG.SEED)
if CFG.MAX_VAL_SAMPLES is not None and len(val_df) > CFG.MAX_VAL_SAMPLES:
    val_df = val_df.sample(n=CFG.MAX_VAL_SAMPLES, random_state=CFG.SEED)
if CFG.MAX_TEST_SAMPLES is not None and len(test_df) > CFG.MAX_TEST_SAMPLES:
    test_df = test_df.sample(n=CFG.MAX_TEST_SAMPLES, random_state=CFG.SEED)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
assert all(len(frame) > 0 for frame in (train_df, val_df, test_df)), 'Empty split.'
for split_name, split_frame in [('train', train_df), ('validation', val_df), ('test', test_df)]:
    split_frame.to_json(CFG.OUTPUT_DIR / f'{split_name}_manifest.jsonl', orient='records', lines=True)
print(f'Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}')
print('Unique images:', {
    'train': train_df['rgb_resolved'].nunique(),
    'val': val_df['rgb_resolved'].nunique(),
    'test': test_df['rgb_resolved'].nunique(),
})
train_paths = set(train_df['rgb_resolved'])
val_paths = set(val_df['rgb_resolved'])
test_paths = set(test_df['rgb_resolved'])
assert train_paths.isdisjoint(val_paths)
assert train_paths.isdisjoint(test_paths)
assert val_paths.isdisjoint(test_paths)
train_thermal = set(train_df['thermal_resolved'])
val_thermal = set(val_df['thermal_resolved'])
test_thermal = set(test_df['thermal_resolved'])
assert train_thermal.isdisjoint(val_thermal), 'Thermal leakage between train and validation.'
assert train_thermal.isdisjoint(test_thermal), 'Thermal leakage between train and test.'
assert val_thermal.isdisjoint(test_thermal), 'Thermal leakage between validation and test.'

def format_temp_summary(temp_summary):
    if not isinstance(temp_summary, dict):
        return ''
    parts = []
    for key in ['min', 'max', 'mean', 'std', 'top3_mean', 'pct_200', 'pct_400']:
        if key in temp_summary:
            parts.append(f'{key}: {temp_summary[key]}')
    return '; '.join(parts)

def make_user_prompt(row):
    parts = []
    if CFG.INCLUDE_TEMP_SUMMARY:
        summary = format_temp_summary(row.get('temp_summary'))
        if summary:
            parts.append(f'Thermal sensor summary: {summary}')
    parts.append(f'Question: {row["question"]}')
    if CFG.INCLUDE_OPTIONS_IN_PROMPT and row.get('options'):
        parts.append(f'Candidate options: {" | ".join(row["options"])}')
    parts.append('Provide a direct and concise answer.')
    return '\n'.join(parts)

def fast_load_and_resize(path_str, size=(224, 224)):
    if path_str is None or pd.isna(path_str):
        raise ValueError('Missing image path')
    with Image.open(path_str) as img:
        return img.convert('RGB').resize(size, Image.BILINEAR)
```
OUTPUT:
```text
Train: 20,128 | Val: 2,482 | Test: 2,482
Unique images: {'train': 592, 'val': 73, 'test': 73}
```

## [cell 9] code
```python
if not torch.cuda.is_available():
    raise RuntimeError('This notebook requires a CUDA GPU. Enable a CUDA GPU accelerator in Kaggle or Colab.')

primary_device = torch.device('cuda:0')
device = primary_device
gpu_count = torch.cuda.device_count()
print(f'Using {gpu_count} CUDA device(s).')

# Kaggle T4 (compute capability 7.5) has native FP16 Tensor Cores but no native BF16.
# Do not use torch.cuda.is_bf16_supported() here: recent PyTorch versions may count
# emulated BF16 support, which is much slower on T4. This matches the Llama baseline.
COMPUTE_DTYPE = torch.float16
print('Compute dtype:', COMPUTE_DTYPE)
assert CFG.LORA_R > 0 and CFG.LORA_ALPHA == 2 * CFG.LORA_R
assert CFG.LORA_TARGET_MODULES and len(CFG.LORA_TARGET_MODULES) == len(set(CFG.LORA_TARGET_MODULES))
backbone_config = AutoConfig.from_pretrained(CFG.LLM_MODEL)
assert backbone_config.model_type == 'qwen2', (
    f'Expected the extracted Qwen2 language model, got {backbone_config.model_type}'
)
image_processor = CLIPImageProcessor.from_pretrained(CFG.VISION_MODEL)
tokenizer = AutoTokenizer.from_pretrained(CFG.LLM_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
assert tokenizer.chat_template, 'The checkpoint must provide its Qwen chat template.'
assert tokenizer.eos_token == '<|im_end|>', 'Unexpected Qwen conversation end token.'

class MultimodalProjector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, llm_dim),
        )

    def forward(self, x):
        return self.net(x)

class WildfireQLoRAVLM(nn.Module):
    def __init__(self, vision_model_name: str, llm_model_name: str):
        super().__init__()

        # CLIP is frozen and used only once to build the feature cache.
        self.vision_encoder = CLIPVisionModel.from_pretrained(
            vision_model_name,
            torch_dtype=COMPUTE_DTYPE,
        )
        self.vision_encoder.requires_grad_(False)
        self.vision_encoder.eval()
        self.vision_dim = self.vision_encoder.config.hidden_size

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=COMPUTE_DTYPE,
        ) if CFG.USE_4BIT else None

        # Keep the 1.5B backbone on one GPU; avoid training with inference CPU offload.
        device_map = {'': 0}
        max_memory = {0: '13GiB', 'cpu': '30GiB'}

        base_llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            max_memory=max_memory,
            torch_dtype=COMPUTE_DTYPE,
            low_cpu_mem_usage=True,
        )
        base_llm.config.use_cache = False
        base_llm.config.pad_token_id = tokenizer.pad_token_id
        for target in CFG.LORA_TARGET_MODULES:
            assert any(name.endswith('.' + target) for name, _ in base_llm.named_modules()), target
        base_llm = prepare_model_for_kbit_training(
            base_llm,
            use_gradient_checkpointing=CFG.GRADIENT_CHECKPOINTING,
        )

        lora_config = LoraConfig(
            r=CFG.LORA_R,
            lora_alpha=CFG.LORA_ALPHA,
            target_modules=CFG.LORA_TARGET_MODULES,
            lora_dropout=CFG.LORA_DROPOUT,
            bias='none',
            task_type=TaskType.CAUSAL_LM,
        )
        self.llm = get_peft_model(base_llm, lora_config)
        self.llm_dim = self.llm.config.hidden_size

        self.rgb_projector = MultimodalProjector(
            self.vision_dim, self.llm_dim, CFG.PROJECTOR_HIDDEN_DIM
        )
        self.thermal_projector = MultimodalProjector(
            self.vision_dim, self.llm_dim, CFG.PROJECTOR_HIDDEN_DIM
        )
        self.rgb_type = nn.Parameter(torch.randn(1, 1, self.llm_dim) * 0.02)
        self.thermal_type = nn.Parameter(torch.randn(1, 1, self.llm_dim) * 0.02)

    def project_visual_patches(self, patch_tokens, projector, type_embed):
        patch_tokens = patch_tokens.to(primary_device, dtype=COMPUTE_DTYPE, non_blocking=True)
        with torch.amp.autocast('cuda', dtype=COMPUTE_DTYPE):
            projected = projector(patch_tokens)
            projected = projected + type_embed.to(dtype=projected.dtype)
        return projected

    def forward(self, inputs_embeds, attention_mask, labels=None):
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(self, inputs_embeds, attention_mask, **gen_kwargs):
        eos_ids = [tokenizer.eos_token_id]
        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_ids,
            **gen_kwargs,
        )

model = WildfireQLoRAVLM(CFG.VISION_MODEL, CFG.LLM_MODEL)
model.vision_encoder.to(primary_device)
# Keep FP32 master weights; autocast uses the selected compute dtype.
model.rgb_projector.to(primary_device, dtype=torch.float32)
model.thermal_projector.to(primary_device, dtype=torch.float32)
model.rgb_type.data = model.rgb_type.data.to(primary_device, dtype=torch.float32)
model.thermal_type.data = model.thermal_type.data.to(primary_device, dtype=torch.float32)

trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_count = sum(p.numel() for p in model.parameters())
print(f'Trainable Parameters: {trainable_count:,} / {total_count:,} ({100*trainable_count/total_count:.2f}%)')
model.llm.print_trainable_parameters()

# PEFT wrappers and some Transformers/Accelerate versions do not expose
# hf_device_map. Report it when available, otherwise inspect real placement.
base_llm_for_report = model.llm.get_base_model()
llm_device_map = getattr(base_llm_for_report, 'hf_device_map', None)
if llm_device_map is not None:
    print('LLM device map:', llm_device_map)
else:
    parameters_by_device = Counter()
    for parameter in base_llm_for_report.parameters():
        parameters_by_device[str(parameter.device)] += parameter.numel()
    print('hf_device_map is unavailable in this package version.')
    print('LLM parameters by device:', {
        name: f'{count:,}' for name, count in parameters_by_device.items()
    })

used_cuda_devices = {
    str(parameter.device)
    for parameter in base_llm_for_report.parameters()
    if parameter.device.type == 'cuda'
}
assert used_cuda_devices == {'cuda:0'}, (
    f'Expected the Qwen2 1.5B model on cuda:0, but found: {sorted(used_cuda_devices)}'
)
if gpu_count > 1:
    print('GPU 1 is intentionally unused; this run places the backbone on cuda:0.')

# Confirm that every requested target receives exactly one adapter per Qwen layer.
adapter_layers = [(name, module) for name, module in model.llm.named_modules()
                  if hasattr(module, 'lora_A') and 'default' in module.lora_A]
adapter_suffixes = tuple(f'.{target}' for target in CFG.LORA_TARGET_MODULES)
assert len(adapter_layers) == len(CFG.LORA_TARGET_MODULES) * backbone_config.num_hidden_layers
assert all(name.endswith(adapter_suffixes) for name, _ in adapter_layers)
assert all(module.lora_A['default'].weight.shape[0] == CFG.LORA_R
           for _, module in adapter_layers)
assert all('lora_' in name for name, parameter in model.llm.named_parameters()
           if parameter.requires_grad)

run_config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(CFG).items() if key.isupper()}
run_config['compute_dtype'] = str(COMPUTE_DTYPE)
run_config['backbone_architecture'] = backbone_config.architectures
run_config['backbone_commit'] = getattr(backbone_config, '_commit_hash', None)
import importlib.metadata
run_config['versions'] = {name: importlib.metadata.version(name)
                          for name in ['torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes']}
with open(CFG.OUTPUT_DIR / 'run_config.json', 'w', encoding='utf-8') as handle:
    json.dump(run_config, handle, indent=2)
image_processor.save_pretrained(CFG.OUTPUT_DIR / 'image_processor')
```
OUTPUT:
```text
Using 2 CUDA device(s).
Compute dtype: torch.float16

config.json:   0%|          | 0.00/660 [00:00<?, ?B/s]
preprocessor_config.json:   0%|          | 0.00/316 [00:00<?, ?B/s]
tokenizer_config.json: 0.00B [00:00, ?B/s]
vocab.json: 0.00B [00:00, ?B/s]
merges.txt: 0.00B [00:00, ?B/s]
tokenizer.json: 0.00B [00:00, ?B/s]
`torch_dtype` is deprecated! Use `dtype` instead!

config.json: 0.00B [00:00, ?B/s]
pytorch_model.bin:   0%|          | 0.00/605M [00:00<?, ?B/s]
model.safetensors:   0%|          | 0.00/605M [00:00<?, ?B/s]
`torch_dtype` is deprecated! Use `dtype` instead!

model.safetensors.index.json: 0.00B [00:00, ?B/s]
Fetching 2 files:   0%|          | 0/2 [00:00<?, ?it/s]
model-00001-of-00002.safetensors:   0%|          | 0.00/3.99G [00:00<?, ?B/s]
model-00002-of-00002.safetensors:   0%|          | 0.00/429M [00:00<?, ?B/s]
Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]
/usr/local/lib/python3.13/dist-packages/bitsandbytes/backends/cuda/ops.py:212: FutureWarning: _check_is_size will be removed in a future PyTorch release along with guard_size_oblivious.     Use _check(i >= 0) instead.
  torch._check_is_size(blocksize)

generation_config.json:   0%|          | 0.00/242 [00:00<?, ?B/s]
Trainable Parameters: 6,905,856 / 982,978,304 (0.70%)
trainable params: 2,179,072 || all params: 1,545,893,376 || trainable%: 0.1410
LLM device map: {'': 0}
GPU 1 is intentionally unused; this run places the backbone on cuda:0.

['/kaggle/working/firevlm_qwen2_1_5b_lora_balanced_qkvo_r8/image_processor/preprocessor_config.json']
```

## [cell 10] code
```python
SYSTEM_PROMPT = 'You are an expert wildfire intelligence system analyzing multimodal aerial imagery.'
RGB_MARKER = '[[FIREVLM_RGB_PATCHES]]'
THERMAL_MARKER = '[[FIREVLM_THERMAL_PATCHES]]'

def build_qwen_tokens(user_text: str, assistant_text: Optional[str] = None):
    # Render the checkpoint's chat template, then replace the two placeholders
    # with projected patch embeddings. These markers are never vocabulary additions.
    if RGB_MARKER in user_text or THERMAL_MARKER in user_text:
        raise ValueError('The question contains a reserved visual placeholder.')
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': (
            f'RGB Imagery Evidence: {RGB_MARKER}\n'
            f'Thermal Imagery Evidence: {THERMAL_MARKER}\n{user_text}'
        )},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    header, remainder = rendered.split(RGB_MARKER, 1)
    middle, prompt = remainder.split(THERMAL_MARKER, 1)
    header_ids, middle_ids, prompt_ids = [
        tokenizer.encode(part, add_special_tokens=False) for part in (header, middle, prompt)
    ]
    target_ids = []
    if assistant_text is not None:
        assert CFG.MAX_TARGET_TOKENS >= 2
        answer_ids = tokenizer.encode(str(assistant_text), add_special_tokens=False)
        target_ids = answer_ids[:CFG.MAX_TARGET_TOKENS - 1] + [tokenizer.eos_token_id]
    return header_ids, middle_ids, prompt_ids, target_ids

@torch.no_grad()
def build_vision_feature_cache(paths, batch_size):
    cache = {}
    ordered_paths = sorted(set(str(p) for p in paths))
    model.vision_encoder.eval()
    started = time.perf_counter()

    for start in tqdm(range(0, len(ordered_paths), batch_size), desc='Caching frozen CLIP features'):
        chunk = ordered_paths[start:start + batch_size]
        images = [fast_load_and_resize(path) for path in chunk]
        pixels = image_processor(images=images, return_tensors='pt').pixel_values
        pixels = pixels.to(primary_device, dtype=COMPUTE_DTYPE, non_blocking=True)
        with torch.amp.autocast('cuda', dtype=COMPUTE_DTYPE):
            patches = model.vision_encoder(pixel_values=pixels).last_hidden_state[:, 1:, :]
        patches = patches.to('cpu', dtype=torch.float16).contiguous()
        for path, feature in zip(chunk, patches):
            cache[path] = feature
        del images, pixels, patches

    elapsed = time.perf_counter() - started
    print(f'Cached {len(cache):,} unique images in {elapsed / 60:.2f} minutes.')
    return cache

all_image_paths = pd.concat([
    train_df['rgb_resolved'], train_df['thermal_resolved'],
    val_df['rgb_resolved'], val_df['thermal_resolved'],
    test_df['rgb_resolved'], test_df['thermal_resolved'],
]).dropna().astype(str).tolist()

VISION_FEATURE_CACHE = build_vision_feature_cache(
    all_image_paths,
    batch_size=CFG.VISION_CACHE_BATCH_SIZE,
)

# CLIP is no longer needed after caching, so release its GPU memory before training.
model.vision_encoder.to('cpu')
torch.cuda.empty_cache()

class WildfireTokenizedDataset(Dataset):
    def __init__(self, frame):
        self.records = []
        for row in frame.to_dict('records'):
            prompt = make_user_prompt(row)
            header_ids, middle_ids, prompt_ids, target_ids = build_qwen_tokens(prompt, str(row['answer']))
            self.records.append({
                'rgb_path': str(row['rgb_resolved']),
                'thermal_path': str(row['thermal_resolved']),
                'header_ids': header_ids,
                'middle_ids': middle_ids,
                'prompt_ids': prompt_ids,
                'target_ids': target_ids,
                'question': str(row['question']),
                'target': str(row['answer']),
                'category': str(row.get('category', 'unknown')),
                'pair_id': str(row['pair_id']),
                'source_collection': str(row['source_collection']),
                'canonical_stem': str(row['canonical_stem']),
            })

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]

def collate_fn(batch, is_eval: bool = False):
    return {
        'rgb_patches': torch.stack([VISION_FEATURE_CACHE[b['rgb_path']] for b in batch]),
        'thermal_patches': torch.stack([VISION_FEATURE_CACHE[b['thermal_path']] for b in batch]),
        'header_ids': [b['header_ids'] for b in batch],
        'middle_ids': [b['middle_ids'] for b in batch],
        'prompt_ids': [b['prompt_ids'] for b in batch],
        'target_ids': [[] if is_eval else b['target_ids'] for b in batch],
        'rgb_paths': [b['rgb_path'] for b in batch],
        'thermal_paths': [b['thermal_path'] for b in batch],
        'questions': [b['question'] for b in batch],
        'targets': [b['target'] for b in batch],
        'categories': [b['category'] for b in batch],
        'pair_ids': [b['pair_id'] for b in batch],
        'source_collections': [b['source_collection'] for b in batch],
        'canonical_stems': [b['canonical_stem'] for b in batch],
    }

train_dataset = WildfireTokenizedDataset(train_df)
val_dataset = WildfireTokenizedDataset(val_df)
test_dataset = WildfireTokenizedDataset(test_df)

train_loader = DataLoader(
    train_dataset,
    batch_size=CFG.BATCH_SIZE,
    shuffle=True,
    num_workers=CFG.NUM_WORKERS,
    pin_memory=True,
    collate_fn=lambda batch: collate_fn(batch, is_eval=False),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=CFG.BATCH_SIZE,
    shuffle=False,
    num_workers=CFG.NUM_WORKERS,
    pin_memory=True,
    collate_fn=lambda batch: collate_fn(batch, is_eval=False),
)

print(f'Train batches: {len(train_loader)} | Val batches: {len(val_loader)}')
_b = next(iter(val_loader))
_diff = (_b['rgb_patches'] - _b['thermal_patches']).abs().mean().item()
print(f'Mean |RGB - thermal| CLIP feature difference: {_diff:.4f}')
assert _diff > 1e-4, 'RGB and thermal features look identical; check path resolution.'

fig, ax = plt.subplots(1, 2, figsize=(7, 3.2))
ax[0].imshow(fast_load_and_resize(_b['rgb_paths'][0])); ax[0].set_title('RGB'); ax[0].axis('off')
ax[1].imshow(fast_load_and_resize(_b['thermal_paths'][0])); ax[1].set_title('Thermal'); ax[1].axis('off')
plt.show()
```
OUTPUT:
```text
Caching frozen CLIP features:   0%|          | 0/93 [00:00<?, ?it/s]
Cached 1,476 unique images in 1.68 minutes.
Train batches: 5032 | Val batches: 621
Mean |RGB - thermal| CLIP feature difference: 0.3909

<Figure size 700x320 with 2 Axes>
```

## [cell 11] code
```python
def assemble_batch_embeddings(model: WildfireQLoRAVLM, batch_data: dict, compute_labels: bool = True):
    rgb_patches = batch_data['rgb_patches'].to(primary_device, non_blocking=True)
    thermal_patches = batch_data['thermal_patches'].to(primary_device, non_blocking=True)

    rgb_embeds = model.project_visual_patches(rgb_patches, model.rgb_projector, model.rgb_type)
    thermal_embeds = model.project_visual_patches(
        thermal_patches, model.thermal_projector, model.thermal_type
    )
    batch_size = rgb_embeds.size(0)

    embed_layer = model.llm.get_input_embeddings()
    embedding_device = embed_layer.weight.device
    embedding_dtype = embed_layer.weight.dtype
    rgb_embeds = rgb_embeds.to(embedding_device, dtype=embedding_dtype)
    thermal_embeds = thermal_embeds.to(embedding_device, dtype=embedding_dtype)

    sequence_embeddings = []
    sequence_labels = []

    for index in range(batch_size):
        header_ids = torch.as_tensor(
            batch_data['header_ids'][index], dtype=torch.long, device=embedding_device
        )
        middle_ids = torch.as_tensor(
            batch_data['middle_ids'][index], dtype=torch.long, device=embedding_device
        )
        prompt_ids = torch.as_tensor(
            batch_data['prompt_ids'][index], dtype=torch.long, device=embedding_device
        )
        target_ids = torch.as_tensor(
            batch_data['target_ids'][index], dtype=torch.long, device=embedding_device
        ) if compute_labels else None

        header_embeddings = embed_layer(header_ids)
        prompt_embeddings = embed_layer(prompt_ids)
        parts = [
            header_embeddings,
            rgb_embeds[index],
            embed_layer(middle_ids),
            thermal_embeds[index],
            prompt_embeddings,
        ]
        non_target_length = sum(part.size(0) for part in parts)
        if compute_labels and target_ids is not None and target_ids.numel() > 0:
            parts.append(embed_layer(target_ids))

        full_embeddings = torch.cat(parts, dim=0)
        sequence_embeddings.append(full_embeddings)

        if compute_labels:
            target_length = target_ids.numel() if target_ids is not None else 0
            labels = torch.full(
                (non_target_length + target_length,),
                -100,
                dtype=torch.long,
                device=embedding_device,
            )
            if target_length:
                labels[non_target_length:] = target_ids
            sequence_labels.append(labels)

    max_length = max(embeddings.size(0) for embeddings in sequence_embeddings)
    if max_length > model.llm.config.max_position_embeddings:
        raise ValueError(f'Sequence exceeds context window: {max_length}')
    padded_embeddings = torch.zeros(
        batch_size,
        max_length,
        model.llm_dim,
        dtype=embedding_dtype,
        device=embedding_device,
    )
    attention_mask = torch.zeros(
        batch_size, max_length, dtype=torch.long, device=embedding_device
    )
    padded_labels = torch.full(
        (batch_size, max_length), -100, dtype=torch.long, device=embedding_device
    ) if compute_labels else None

    for index, embeddings in enumerate(sequence_embeddings):
        sequence_length = embeddings.size(0)
        if compute_labels:
            padded_embeddings[index, :sequence_length] = embeddings
            attention_mask[index, :sequence_length] = 1
            padded_labels[index, :sequence_length] = sequence_labels[index]
        else:
            padded_embeddings[index, -sequence_length:] = embeddings
            attention_mask[index, -sequence_length:] = 1

    return padded_embeddings, attention_mask, padded_labels
```

## [cell 12] code
```python
def simple_tokens(s: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", str(s).lower())

def ngram_counts(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

def sentence_bleu(reference: str, prediction: str, max_n: int = 4) -> float:
    ref = simple_tokens(reference)
    pred = simple_tokens(prediction)
    if not pred:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        pred_counts = ngram_counts(pred, n)
        ref_counts = ngram_counts(ref, n)
        if not pred_counts:
            precisions.append(1e-9)
            continue
        overlap = sum(min(c, ref_counts[g]) for g, c in pred_counts.items())
        precisions.append((overlap + 1) / (sum(pred_counts.values()) + 1))
    bp = 1.0 if len(pred) > len(ref) else math.exp(1 - len(ref) / max(len(pred), 1))
    return bp * math.exp(sum(math.log(p) for p in precisions) / max_n)

def lcs_len(a: List[str], b: List[str]) -> int:
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            cur = dp[j]
            if x == y:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j-1])
            prev = cur
    return dp[-1]

def rouge_l(reference: str, prediction: str) -> float:
    ref = simple_tokens(reference)
    pred = simple_tokens(prediction)
    if not ref or not pred:
        return 0.0
    lcs = lcs_len(ref, pred)
    prec = lcs / len(pred)
    rec = lcs / len(ref)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def exact_match(reference: str, prediction: str) -> int:
    return int(str(reference).strip().lower() == str(prediction).strip().lower())
```

## [cell 13] code
```python
trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
optimizer = torch.optim.AdamW(
    trainable_params,
    lr=CFG.LR,
    weight_decay=CFG.WEIGHT_DECAY,
)

steps_per_epoch = math.ceil(len(train_loader) / CFG.GRAD_ACCUM_STEPS)
total_steps = CFG.EPOCHS * steps_per_epoch
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * CFG.WARMUP_RATIO),
    num_training_steps=total_steps,
)

best_val_loss = float('inf')
epochs_without_improvement = 0
history = []
model.llm.config.use_cache = False

for epoch in range(1, CFG.EPOCHS + 1):
    epoch_started = time.perf_counter()
    model.train()
    running_loss = 0.0
    pending_loss = None
    seen_samples = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{CFG.EPOCHS}', mininterval=1.0)
    for step, batch in enumerate(pbar, 1):
        inputs_embeds, attention_mask, labels = assemble_batch_embeddings(
            model, batch, compute_labels=True
        )

        with torch.amp.autocast('cuda', dtype=COMPUTE_DTYPE):
            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / CFG.GRAD_ACCUM_STEPS

        loss.backward()

        if step % CFG.GRAD_ACCUM_STEPS == 0 or step == len(train_loader):
            torch.nn.utils.clip_grad_norm_(trainable_params, CFG.GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        batch_size = inputs_embeds.size(0)
        weighted_loss = loss.detach().float() * CFG.GRAD_ACCUM_STEPS * batch_size
        pending_loss = weighted_loss if pending_loss is None else pending_loss + weighted_loss
        seen_samples += batch_size

        if step % CFG.LOG_EVERY == 0 or step == len(train_loader):
            running_loss += pending_loss.item()
            pending_loss = None
            pbar.set_postfix(train_loss=f'{running_loss / seen_samples:.4f}')

    model.eval()
    val_loss_tensor = None
    val_samples = 0
    with torch.inference_mode():
        for batch in tqdm(val_loader, desc='Validating', leave=False, mininterval=1.0):
            inputs_embeds, attention_mask, labels = assemble_batch_embeddings(
                model, batch, compute_labels=True
            )
            with torch.amp.autocast('cuda', dtype=COMPUTE_DTYPE):
                outputs = model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            batch_size = inputs_embeds.size(0)
            weighted_loss = outputs.loss.detach().float() * batch_size
            val_loss_tensor = weighted_loss if val_loss_tensor is None else val_loss_tensor + weighted_loss
            val_samples += batch_size

    val_loss = val_loss_tensor.item() / max(val_samples, 1)
    if not math.isfinite(val_loss):
        raise FloatingPointError('Non-finite validation loss; no checkpoint will be selected.')
    train_loss_epoch = running_loss / max(seen_samples, 1)
    elapsed_minutes = (time.perf_counter() - epoch_started) / 60
    record = {
        'epoch': epoch,
        'train_loss': train_loss_epoch,
        'val_loss': val_loss,
        'minutes': elapsed_minutes,
    }
    history.append(record)
    print(f'Epoch {epoch} Results:', record)

    improved = val_loss < (best_val_loss - CFG.EARLY_STOPPING_MIN_DELTA)
    if improved:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save({
            'rgb_projector': model.rgb_projector.state_dict(),
            'thermal_projector': model.thermal_projector.state_dict(),
            'rgb_type': model.rgb_type.detach().cpu(),
            'thermal_type': model.thermal_type.detach().cpu(),
        }, CFG.OUTPUT_DIR / 'best_projectors.pt')
        model.llm.save_pretrained(CFG.OUTPUT_DIR / 'best_lora_adapters')
        tokenizer.save_pretrained(CFG.OUTPUT_DIR / 'tokenizer')
        print('--> Saved new best projectors + LoRA adapters!')
    else:
        epochs_without_improvement += 1
        print(
            f'Validation loss did not improve by at least '
            f'{CFG.EARLY_STOPPING_MIN_DELTA:g}. '
            f'Patience: {epochs_without_improvement}/'
            f'{CFG.EARLY_STOPPING_PATIENCE}'
        )
        if epochs_without_improvement >= CFG.EARLY_STOPPING_PATIENCE:
            print(f'Early stopping after epoch {epoch}.')
            break

pd.DataFrame(history).to_csv(CFG.OUTPUT_DIR / 'training_history.csv', index=False)
```
OUTPUT:
```text
Epoch 1/100:   0%|          | 0/5032 [00:00<?, ?it/s]
Validating:   0%|          | 0/621 [00:00<?, ?it/s]
Epoch 1 Results: {'epoch': 1, 'train_loss': 0.7642609992068977, 'val_loss': 0.20561220720830606, 'minutes': 39.52046296646667}
--> Saved new best projectors + LoRA adapters!

Epoch 2/100:   0%|          | 0/5032 [00:00<?, ?it/s]
Validating:   0%|          | 0/621 [00:00<?, ?it/s]
Epoch 2 Results: {'epoch': 2, 'train_loss': 0.1876516419630172, 'val_loss': 0.19274328141131775, 'minutes': 39.496494433516666}
--> Saved new best projectors + LoRA adapters!

Epoch 3/100:   0%|          | 0/5032 [00:00<?, ?it/s]
Validating:   0%|          | 0/621 [00:00<?, ?it/s]
Epoch 3 Results: {'epoch': 3, 'train_loss': 0.1619909280150039, 'val_loss': 0.16868282690055902, 'minutes': 39.453014356433336}
--> Saved new best projectors + LoRA adapters!

Epoch 4/100:   0%|          | 0/5032 [00:00<?, ?it/s]
Validating:   0%|          | 0/621 [00:00<?, ?it/s]
Epoch 4 Results: {'epoch': 4, 'train_loss': 0.15039446686997132, 'val_loss': 0.1641320067965917, 'minutes': 39.425964215533334}
--> Saved new best projectors + LoRA adapters!

Epoch 5/100:   0%|          | 0/5032 [00:00<?, ?it/s]
Validating:   0%|          | 0/621 [00:00<?, ?it/s]
Epoch 5 Results: {'epoch': 5, 'train_loss': 0.14767355252714717, 'val_loss': 0.14781798824583878, 'minutes': 39.395472528533354}
--> Saved new best projectors + LoRA adapters!

Epoch 6/100:   0%|          | 0/5032 [00:00<?, ?it/s]
Validating:   0%|          | 0/621 [
[... 798 chars truncated ...]
     | 0/5032 [00:00<?, ?it/s]
Validating:   0%|          | 0/621 [00:00<?, ?it/s]
Epoch 9 Results: {'epoch': 9, 'train_loss': 0.10228708392343763, 'val_loss': 0.1286368420006862, 'minutes': 39.29041765356669}
Validation loss did not improve by at least 1e-08. Patience: 1/3

Epoch 10/100:   0%|          | 0/5032 [00:00<?, ?it/s]
Validating:   0%|          | 0/621 [00:00<?, ?it/s]
Epoch 10 Results: {'epoch': 10, 'train_loss': 0.09489985396093518, 'val_loss': 0.11587510888947296, 'minutes': 39.29230087863331}
--> Saved new best projectors + LoRA adapters!

Epoch 11/100:   0%|          | 0/5032 [00:00<?, ?it/s]
Validating:   0%|          | 0/621 [00:00<?, ?it/s]
Epoch 11 Results: {'epoch': 11, 'train_loss': 0.08559134815758855, 'val_loss': 0.14114260615887514, 'minutes': 39.259181194350035}
Validation loss did not improve by at least 1e-08. Patience: 1/3

Epoch 12/100:   0%|          | 0/5032 [00:00<?, ?it/s]
Validating:   0%|          | 0/621 [00:00<?, ?it/s]
Epoch 12 Results: {'epoch': 12, 'train_loss': 0.07921187979996867, 'val_loss': 0.11616106671157332, 'minutes': 39.19379889558338}
Validation loss did not improve by at least 1e-08. Patience: 2/3

Epoch 13/100:   0%|          | 0/5032 [00:00<?, ?it/s]
Validating:   0%|          | 0/621 [00:00<?, ?it/s]
Epoch 13 Results: {'epoch': 13, 'train_loss': 0.07610870861938439, 'val_loss': 0.11985001729247456, 'minutes': 39.19631257643335}
Validation loss did not improve by at least 1e-08. Patience: 3/3
Early stopping after epoch 13.
```

## [cell 14] code
```python
best_proj_path = CFG.OUTPUT_DIR / 'best_projectors.pt'
best_lora_path = CFG.OUTPUT_DIR / 'best_lora_adapters'
primary_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

assert best_proj_path.is_file() and best_lora_path.is_dir(), 'Train before evaluating.'
if best_proj_path.exists():
    state = torch.load(best_proj_path, map_location=primary_device, weights_only=True)
    model.rgb_projector.load_state_dict(state['rgb_projector'])
    model.thermal_projector.load_state_dict(state['thermal_projector'])
    model.rgb_type.data.copy_(state['rgb_type'].to(primary_device))
    model.thermal_type.data.copy_(state['thermal_type'].to(primary_device))
    print(f"Loaded best projectors from: {best_proj_path}")

if best_lora_path.exists():
    # Actually load the best adapter weights (the old code only printed a message)
    model.llm.load_adapter(str(best_lora_path), adapter_name='best', is_trainable=False)
    model.llm.set_adapter('best')
    print(f"Loaded best LoRA adapters from: {best_lora_path}")

model.eval()
model.llm.config.use_cache = True

# 2. Evaluation DataLoader (is_eval=True -> No ground-truth answer in prompt)
test_eval_loader = DataLoader(
    test_dataset,
    batch_size=CFG.BATCH_SIZE,
    shuffle=False,
    num_workers=CFG.NUM_WORKERS,
    pin_memory=True,
    collate_fn=lambda b: collate_fn(b, is_eval=True),
)

pred_rows = []
sample_visualizations = []  # Stores one prediction per unique image pair for plotting
visualized_rgb_paths = set()

# 3. Autoregressive Generation Loop
with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(test_eval_loader, desc="Generating answers")):
        inputs_embeds, attention_mask, _ = assemble_batch_embeddings(
            model, batch, compute_labels=False
        )

        with torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
            gen_output = model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=CFG.MAX_GENERATE_TOKENS,
                num_beams=1,
                do_sample=False,
            )

        preds = tokenizer.batch_decode(gen_output, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        for i, (q, target, raw_pred, cat) in enumerate(zip(
            batch["questions"], batch["targets"], preds, batch["categories"]
        )):
            clean_pred = raw_pred.strip().split("\n")[0].strip()
            clean_pred = re.sub(r'[\r\t]', '', clean_pred)

            em = exact_match(target, clean_pred)
            bleu = sentence_bleu(target, clean_pred)
            r_l = rouge_l(target, clean_pred)

            record = {
                "pair_id": batch["pair_ids"][i],
                "source_collection": batch["source_collections"][i],
                "canonical_stem": batch["canonical_stems"][i],
                "rgb_path": batch["rgb_paths"][i],
                "thermal_path": batch["thermal_paths"][i],
                "category": cat,
                "question": q,
                "target": target,
                "prediction": clean_pred,
                "exact_match": em,
                "bleu": bleu,
                "rouge_l": r_l,
            }
            pred_rows.append(record)

            # Collect up to 8 unique image pairs rather than 8 QA rows from one image.
            if len(sample_visualizations) < 8 and batch["rgb_paths"][i] not in visualized_rgb_paths:
                visualized_rgb_paths.add(batch["rgb_paths"][i])
                rgb_img = np.asarray(fast_load_and_resize(batch["rgb_paths"][i])) / 255.0
                th_img = np.asarray(fast_load_and_resize(batch["thermal_paths"][i])) / 255.0

                sample_visualizations.append({
                    "rgb": rgb_img,
                    "thermal": th_img,
                    "question": q,
                    "target": target,
                    "prediction": clean_pred,
                    "category": cat,
                    "em": em,
                })

# 4. Save Predictions & Summary
pred_df = pd.DataFrame(pred_rows)
pred_df.to_csv(CFG.OUTPUT_DIR / "test_predictions.csv", index=False)

summary = {
    "exact_match": float(pred_df["exact_match"].mean()) if len(pred_df) else 0.0,
    "bleu": float(pred_df["bleu"].mean()) if len(pred_df) else 0.0,
    "rouge_l": float(pred_df["rouge_l"].mean()) if len(pred_df) else 0.0,
    "n_samples": int(len(pred_df)),
}

with open(CFG.OUTPUT_DIR / "test_metrics.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*60)
print("                HELD-OUT TEST METRICS (QLoRA)")
print("="*60)
print(f"Total Samples Evaluated: {summary['n_samples']}")
print(f"Exact Match (EM):        {summary['exact_match'] * 100:.2f}%")
print(f"BLEU-4 Score:            {summary['bleu']:.4f}")
print(f"ROUGE-L Score:           {summary['rouge_l']:.4f}")

print("\n" + "="*60)
print("             PERFORMANCE BY QUESTION CATEGORY")
print("="*60)
category_df = (
    pred_df.groupby("category")[["exact_match", "bleu", "rouge_l"]]
    .mean()
    .sort_values("exact_match", ascending=False)
)
category_df["exact_match"] = category_df["exact_match"] * 100
category_df.columns = ["Exact Match (%)", "BLEU-4", "ROUGE-L"]
display(category_df)

# ---------------------------------------------------------
# 5. Visual Sample Inspection (RGB + Thermal Pairs)
# ---------------------------------------------------------
print("\n" + "="*60)
print("         SAMPLE PREDICTIONS WITH DUAL IMAGERY")
print("="*60)

for idx, item in enumerate(sample_visualizations):
    same = np.allclose(item["rgb"], item["thermal"], atol=1e-3)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))

    axes[0].imshow(item["rgb"])
    axes[0].set_title("RGB Aerial Image", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(item["thermal"])
    axes[1].set_title("Thermal / IR Image", fontsize=10, fontweight="bold")
    axes[1].axis("off")

    status_icon = " Correct" if item["em"] == 1 else " Mismatch"
    warn = "  [WARNING: RGB == thermal]" if same else ""
    plt.suptitle(
        f"Sample {idx+1} [{item['category']}] - {status_icon}{warn}\n"
        f"Q: {item['question']}\n"
        f"Ground Truth: '{item['target']}' | Prediction: '{item['prediction']}'",
        fontsize=10,
        y=1.12,
        ha="center"
    )
    plt.tight_layout()
    plt.show()
```
OUTPUT:
```text
Loaded best projectors from: /kaggle/working/firevlm_qwen2_1_5b_lora_balanced_qkvo_r8/best_projectors.pt
Loaded best LoRA adapters from: /kaggle/working/firevlm_qwen2_1_5b_lora_balanced_qkvo_r8/best_lora_adapters

Generating answers:   0%|          | 0/621 [00:00<?, ?it/s]
The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.


============================================================
                HELD-OUT TEST METRICS (QLoRA)
============================================================
Total Samples Evaluated: 2482
Exact Match (EM):        86.30%
BLEU-4 Score:            0.1497
ROUGE-L Score:           0.8928

============================================================
             PERFORMANCE BY QUESTION CATEGORY
============================================================

                               Exact Match (%)    BLEU-4   ROUGE-L
category                                                          
Presence and Detection               90.753425  0.000002  0.925799
Distribution and Segmentation        88.527397  0.329941  0.915117
Flight Planning                      88.356164  0.484644  0.917808
Cross-Modal Reasoning                85.616438  0.001146  0.907192
Classification                       84.931507  0.082900  0.871233
Localization and Direction           73.630137  0.002401  0.775114

============================================================
         SAMPLE PREDICTIONS WITH DUAL IMAGERY
============================================================

<Figure size 700x320 with 2 Axes>
<Figure size 700x320 with 2 Axes>
<Figure size 700x320 with 2 Axes>
<Figure size 700x320 with 2 Axes>
<Figure size 700x320 with 2 Axes>
<Figure size 700x320 with 2 Axes>
<Figure size 700x320 with 2 Axes>
<Figure size 700x320 with 2 Axes>
```

## [cell 15] code
```python
# Archive the output directory (works on Kaggle, Colab and local Python).
import shutil
archive = shutil.make_archive(str(CFG.OUTPUT_DIR), 'zip', root_dir=CFG.OUTPUT_DIR)
print('Saved artifacts:', CFG.OUTPUT_DIR)
print('Download archive:', archive)
```
OUTPUT:
```text
Saved artifacts: /kaggle/working/firevlm_qwen2_1_5b_lora_balanced_qkvo_r8
Download archive: /kaggle/working/firevlm_qwen2_1_5b_lora_balanced_qkvo_r8.zip
```
