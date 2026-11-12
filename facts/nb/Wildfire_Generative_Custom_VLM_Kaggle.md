# Wildfire_Generative_Custom_VLM_Kaggle.ipynb
- cells: 30
- code cells with text output: 0
- non-text outputs omitted (images/html): 0
- cells whose output contains an error: 0

## [cell 0] markdown
# Generative Custom VLM for WildFireVQA

This Kaggle notebook trains a small **generative** VLM for WildFireVQA.

It does **not** score options and pick the best option. It generates answer text autoregressively and evaluates generated text against the ground-truth answer using text metrics such as BLEU and ROUGE-L.

Architecture:

```text
RGB image      -> CLIP vision encoder -> RGB visual token ----
Thermal image  -> CLIP vision encoder -> thermal visual token -+-> T5 encoder -> T5 decoder -> generated answer
Question/text  -> T5 token embeddings -------------------------
```

Why this design:

- WildFireVQA is a multimodal wildfire VQA dataset with RGB and thermal images.
- The dataset is multiple-choice by release format, but we train the model to generate the target answer text.
- The design is closest in spirit to VLM alignment papers such as **Aligning Vision Language Models with Contrastive Learning** and practical encoder-connector-LLM VLMs, but kept small enough for Kaggle.

## [cell 1] markdown
## 0. Kaggle Setup

This notebook uses **two Kaggle datasets**:

1. WildFireVQA annotations: `caseypiere/wildfire-vqa`
2. FLAME-3 Sycan Marsh image subset: `brycehopkins/flame-3-computer-vision-subset-sycan-marsh`

The WildFireVQA JSON rows contain `rgb_path` and `thermal_path`, but those paths point to the author's local FLAME-3 tree, for example:

```text
/home/mhabibp/Downloads/flame3333/Flame 3 Computer Vision Sets/Sycan_2A/Images/Fire/RGB/Corrected FOV/00001.JPG
```

The resolver cell downloads/searches the FLAME-3 Kaggle dataset and maps those author-local paths onto the Kaggle-mounted image files.

Note: this FLAME-3 Kaggle link is a **Sycan Marsh subset**, so rows from Shoetank or Willamette will still not resolve unless you add those image subsets too.

## [cell 2] code
```python
# Install dependencies as needed in Kaggle:
# !pip install -q "kagglehub[pandas-datasets]" transformers accelerate datasets pyarrow
```

## [cell 3] code
```python
import os
import re
import json
import math
import random
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

try:
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
except Exception as e:
    kagglehub = None
    KaggleDatasetAdapter = None
    print('[WARN] kagglehub import failed. Uncomment the pip install cell above and rerun. Error:', repr(e))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    CLIPVisionModel,
    CLIPImageProcessor,
    T5ForConditionalGeneration,
    T5TokenizerFast,
    get_cosine_schedule_with_warmup,
)

print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available())
```

## [cell 4] markdown
## 1. Configuration

Start with `MAX_TRAIN_SAMPLES` small for a smoke test. Set it to `None` for full training.

## [cell 5] code
```python
class CFG:
    # Data source. Keep KAGGLEHUB for Kaggle runs.
    DATA_SOURCE = 'kagglehub'  # one of: 'kagglehub', 'local', 'hf'
    KAGGLE_DATASET = 'caseypiere/wildfire-vqa'
    FLAME3_KAGGLE_DATASET = 'brycehopkins/flame-3-computer-vision-subset-sycan-marsh'

    # KaggleHub needs the path to a tabular annotation file inside the dataset.
    # Start with empty string, exactly as in KaggleHub examples. If it fails,
    # set this to the CSV/JSON/Parquet path printed by the loader cell.
    KAGGLEHUB_FILE_PATH = ''

    # WildFireVQA annotation root. KaggleHub may overwrite this with its downloaded/mounted path.
    DATA_ROOT = Path('/kaggle/input/datasets/caseypiere/wildfire-vqa')

    # FLAME-3 image root. Leave as None to download/auto-discover through KaggleHub and /kaggle/input.
    # If needed, set to something like Path('/kaggle/input/flame-3-computer-vision-subset-sycan-marsh').
    FLAME3_ROOT = None
    FLAME3_MARKER_DIR = 'Flame 3 Computer Vision Sets'

    # Optional Hugging Face fallback.
    HF_DATASET_NAME = 'mobiiin/WildFire_VQA'

    # If None, local fallback searches DATA_ROOT for json/jsonl/csv/parquet files.
    ANNOTATION_FILE = None

    # Training subset for quick Kaggle iteration. Set to None for full data.
    MAX_TRAIN_SAMPLES = 3000
    MAX_VAL_SAMPLES = 500
    VAL_FRAC = 0.10
    SEED = 42

    # Model.
    VISION_MODEL = 'openai/clip-vit-base-patch32'
    TEXT_MODEL = 't5-small'
    FREEZE_VISION = True
    FREEZE_T5 = False

    # Prompting.
    INCLUDE_OPTIONS_IN_PROMPT = False   # True is easier, False is more free-form.
    INCLUDE_TEMP_SUMMARY = True
    MAX_INPUT_TOKENS = 192
    MAX_TARGET_TOKENS = 32
    MAX_GENERATE_TOKENS = 32

    # Training.
    BATCH_SIZE = 8
    EPOCHS = 2
    LR = 3e-4
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.05
    GRAD_ACCUM_STEPS = 1
    GRAD_CLIP = 1.0
    NUM_WORKERS = 2
    MIXED_PRECISION = True

    OUTPUT_DIR = Path('/kaggle/working/wildfire_generative_vlm')

CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
random.seed(CFG.SEED)
np.random.seed(CFG.SEED)
torch.manual_seed(CFG.SEED)
```

## [cell 6] markdown
## 2. Load WildFireVQA Rows with KaggleHub

The primary path is KaggleHub loading into a Pandas dataframe.

Supported annotation formats for fallback discovery:

- list-style JSON: `[{...}, {...}]`
- checkpoint-style JSON: `{"type":"checkpoint", "items": ...}`
- JSONL
- CSV
- Parquet

Expected fields are inferred from the repo evaluator:

`question_id`, `question`, `options`, `gt_answer` or `answer`, `rgb_path`, `thermal_path`, optional `temp_summary`, `image_id`, and `category`.

## [cell 7] code
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
        # Fallback for semicolon/pipe separated options.
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


def rows_from_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def discover_annotation_files(root: Path):
    exts = ['*.json', '*.jsonl', '*.csv', '*.parquet']
    files = []
    if root.exists():
        for ext in exts:
            files.extend(root.rglob(ext))
    priority = ('vqa', 'question', 'answer', 'annotation', 'response', 'train', 'val')
    return sorted(files, key=lambda p: (not any(w in p.name.lower() for w in priority), len(str(p))))


def print_dataset_files(root: Path, limit=80):
    if not root.exists():
        print('Dataset root does not exist yet:', root)
        return
    files = [p for p in root.rglob('*') if p.is_file()]
    print(f'Dataset root: {root}')
    print(f'Total files found: {len(files)}')
    for p in files[:limit]:
        print(' -', p.relative_to(root))
    if len(files) > limit:
        print(f'... {len(files) - limit} more files')


def load_rows_from_files():
    if CFG.ANNOTATION_FILE:
        files = [Path(CFG.ANNOTATION_FILE)]
    else:
        files = discover_annotation_files(CFG.DATA_ROOT)
    if not files:
        raise FileNotFoundError(f'No annotation files found under {CFG.DATA_ROOT}')

    all_rows = []
    print('Annotation files considered:')
    for p in files[:20]:
        print(' -', p)

    for p in files:
        try:
            if p.suffix.lower() == '.json':
                rows = rows_from_json(p)
            elif p.suffix.lower() == '.jsonl':
                rows = rows_from_jsonl(p)
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
    if kagglehub is None or KaggleDatasetAdapter is None:
        raise ImportError('kagglehub is unavailable. Uncomment the pip install cell, restart the notebook, and rerun.')

    # Download/list the annotation dataset. FLAME-3 images are resolved separately in the image path cell.
    try:
        dataset_root = Path(kagglehub.dataset_download(CFG.KAGGLE_DATASET))
        CFG.DATA_ROOT = dataset_root
        print_dataset_files(CFG.DATA_ROOT)
    except Exception as e:
        print('[WARN] kagglehub.dataset_download failed; annotation paths may need /kaggle/input fallback:', repr(e))

    # KaggleHub Pandas loader requested by the user.
    file_path = CFG.KAGGLEHUB_FILE_PATH
    try:
        kh_df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            CFG.KAGGLE_DATASET,
            file_path,
        )
        print('First 5 records:', kh_df.head())
        print('KaggleHub dataframe shape:', kh_df.shape)
        return kh_df.to_dict('records')
    except Exception as e:
        print('[WARN] kagglehub.load_dataset failed with file_path =', repr(file_path))
        print('[WARN] Error:', repr(e))
        print('Set CFG.KAGGLEHUB_FILE_PATH to one of the tabular files printed above, then rerun this cell.')
        print('Falling back to local annotation discovery under CFG.DATA_ROOT.')
        return load_rows_from_files()


def load_rows_from_hf():
    from datasets import load_dataset
    ds = load_dataset(CFG.HF_DATASET_NAME)
    rows = []
    for split_name, split in ds.items():
        for r in split:
            r = dict(r)
            r['split'] = split_name
            rows.append(r)
    return rows


if CFG.DATA_SOURCE == 'kagglehub':
    raw_rows = load_rows_from_kagglehub()
elif CFG.DATA_SOURCE == 'hf':
    raw_rows = load_rows_from_hf()
elif CFG.DATA_SOURCE == 'local':
    raw_rows = load_rows_from_files()
else:
    raise ValueError(f'Unknown CFG.DATA_SOURCE: {CFG.DATA_SOURCE}')

print('Raw rows:', len(raw_rows))
print('Example keys:', list(raw_rows[0].keys()) if raw_rows else None)
```

## [cell 8] code
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

df = pd.DataFrame(records).drop_duplicates(subset=['image_id', 'question_id', 'question', 'answer', 'rgb_path', 'thermal_path'])
print('Clean rows:', len(df))
df.head()
```

## [cell 9] markdown
## 3. Resolve FLAME-3 Image Paths

This cell downloads/searches the FLAME-3 Sycan Marsh Kaggle dataset:

`brycehopkins/flame-3-computer-vision-subset-sycan-marsh`

Then it maps WildFireVQA paths such as:

```text
.../Flame 3 Computer Vision Sets/Sycan_2A/Images/Fire/RGB/Corrected FOV/00001.JPG
```

to the real Kaggle file path.

Because the linked FLAME-3 dataset is Sycan-only, only Sycan rows should resolve. Shoetank/Willamette rows need their own image datasets.

## [cell 10] code
```python
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def build_file_index(root: Path):
    idx = defaultdict(list)
    if root and root.exists():
        for p in root.rglob('*'):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                idx[p.name].append(p)
    return idx


def unique_existing(paths):
    seen = set()
    out = []
    for p in paths:
        if p is None:
            continue
        p = Path(p)
        key = str(p.resolve()) if p.exists() else str(p)
        if p.exists() and key not in seen:
            seen.add(key)
            out.append(p)
    return out


def download_flame3_kaggle_dataset():
    roots = []
    if kagglehub is None:
        print('[WARN] kagglehub unavailable; cannot download FLAME-3 Kaggle dataset automatically.')
        return roots
    try:
        flame_root = Path(kagglehub.dataset_download(CFG.FLAME3_KAGGLE_DATASET))
        print('Downloaded/mounted FLAME-3 Kaggle dataset:', flame_root)
        roots.append(flame_root)
    except Exception as e:
        print('[WARN] kagglehub.dataset_download failed for FLAME-3:', CFG.FLAME3_KAGGLE_DATASET)
        print('[WARN] Error:', repr(e))
    return roots


def discover_flame3_roots():
    candidates = []

    if CFG.FLAME3_ROOT is not None:
        manual = Path(CFG.FLAME3_ROOT)
        candidates.append(manual)
        if manual.name != CFG.FLAME3_MARKER_DIR:
            candidates.append(manual / CFG.FLAME3_MARKER_DIR)

    # Use the explicit FLAME-3 Kaggle dataset slug, same style as WildFireVQA.
    candidates.extend(download_flame3_kaggle_dataset())

    kaggle_input = Path('/kaggle/input')
    if kaggle_input.exists():
        # Exact marker directory is the most reliable signal when the full FLAME-3 folder is preserved.
        candidates.extend([p for p in kaggle_input.rglob(CFG.FLAME3_MARKER_DIR) if p.is_dir()])

        # Fallback: include likely mounted dataset roots/subroots for subset layouts.
        candidates.extend([p for p in kaggle_input.iterdir() if p.is_dir() and 'flame' in p.name.lower()])
        candidates.extend([p / CFG.FLAME3_MARKER_DIR for p in kaggle_input.rglob('flame3333') if p.is_dir()])

    return unique_existing(candidates)


flame3_roots = discover_flame3_roots()
print('WildFireVQA annotation root:', CFG.DATA_ROOT)
print('FLAME-3 Kaggle dataset:', CFG.FLAME3_KAGGLE_DATASET)
print('Discovered FLAME-3 roots:')
for r in flame3_roots:
    print(' -', r)

if not flame3_roots:
    print('[WARN] No FLAME-3 image root found under /kaggle/input or KaggleHub download cache.')
    print('Check that the Kaggle dataset is accessible:', CFG.FLAME3_KAGGLE_DATASET)
    print('Or set CFG.FLAME3_ROOT manually.')

file_index = defaultdict(list)
for root in flame3_roots:
    for name, paths in build_file_index(root).items():
        file_index[name].extend(paths)
print('Indexed FLAME-3 image-like files:', sum(len(v) for v in file_index.values()))


def candidate_flame_paths(s: str):
    marker = CFG.FLAME3_MARKER_DIR
    norm = s.replace(chr(92), '/')

    candidates = []
    if marker in norm:
        suffix = norm.split(marker, 1)[1].lstrip('/')
        for root in flame3_roots:
            # Layout A: root == .../Flame 3 Computer Vision Sets
            if root.name == marker:
                candidates.append(root / suffix)
            # Layout B: root contains Flame 3 Computer Vision Sets
            candidates.append(root / marker / suffix)
            # Layout C: subset root starts directly with Sycan_2A/Sycan_2D/etc.
            candidates.append(root / suffix)

    # If the JSON has relative paths, try them below every FLAME-3 root.
    rel = Path(norm.lstrip('/'))
    for root in flame3_roots:
        candidates.append(root / rel)

    return candidates


def resolve_path(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    # HF datasets may already contain PIL images or dict image objects; leave those alone.
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict):
        # datasets Image feature can look like {'path': ..., 'bytes': ...}
        if value.get('path'):
            value = value['path']
        else:
            return value

    s = str(value).strip()
    if not s:
        return None

    p = Path(s)
    if p.is_absolute() and p.exists():
        return str(p)

    # First map absolute author/HPC paths onto the Kaggle-mounted FLAME-3 tree.
    for candidate in candidate_flame_paths(s):
        if candidate.exists():
            return str(candidate)

    # Then try annotation root for unusual local layouts.
    joined = CFG.DATA_ROOT / s
    if joined.exists():
        return str(joined)

    # Last resort: basename match. This can be ambiguous, so marker-based mapping above is preferred.
    matches = file_index.get(Path(s).name, [])
    if matches:
        return str(matches[0])

    return None


df['rgb_resolved'] = df['rgb_path'].apply(resolve_path)
df['thermal_resolved'] = df['thermal_path'].apply(resolve_path)

# Keep rows with at least one usable visual modality.
df = df[(df['rgb_resolved'].notna()) | (df['thermal_resolved'].notna())].reset_index(drop=True)
print('Rows with image data:', len(df))
print('RGB available:', df['rgb_resolved'].notna().mean() if len(df) else 0)
print('Thermal available:', df['thermal_resolved'].notna().mean() if len(df) else 0)

if len(df) == 0:
    print('[WARN] No rows have resolved images. Check the FLAME-3 Sycan dataset mount/layout.')
    print('Useful Kaggle checks:')
    print('!find /kaggle/input -type f -iname "*.JPG" | head -50')
    print('!find /kaggle/input -type d -iname "*Sycan*" -print')

df[['question', 'answer', 'rgb_resolved', 'thermal_resolved', 'temp_summary']].head()
```

## [cell 11] markdown
## 4. Visual Sanity Check

Plot a few resolved RGB/thermal samples with their question and ground-truth answer before training.

## [cell 12] code
```python
def safe_open_display_image(path_or_obj):
    if isinstance(path_or_obj, Image.Image):
        return path_or_obj.convert('RGB')
    if isinstance(path_or_obj, dict) and path_or_obj.get('bytes'):
        import io
        return Image.open(io.BytesIO(path_or_obj['bytes'])).convert('RGB')
    if path_or_obj is None or (isinstance(path_or_obj, float) and np.isnan(path_or_obj)):
        return None
    try:
        return Image.open(path_or_obj).convert('RGB')
    except Exception as e:
        print('[WARN] could not open image:', path_or_obj, repr(e))
        return None


def shorten_text(text, width=95):
    text = re.sub(r'\s+', ' ', str(text)).strip()
    if len(text) <= width:
        return text
    return text[:width - 3] + '...'


def plot_vqa_samples(frame, n=4, seed=CFG.SEED):
    if len(frame) == 0:
        print('No resolved image rows to plot.')
        return

    sample = frame.sample(min(n, len(frame)), random_state=seed).reset_index(drop=True)
    fig, axes = plt.subplots(len(sample), 2, figsize=(13, 4.2 * len(sample)))
    if len(sample) == 1:
        axes = np.array([axes])

    for i, row in sample.iterrows():
        rgb = safe_open_display_image(row.get('rgb_resolved'))
        thermal = safe_open_display_image(row.get('thermal_resolved'))

        for j, (img, label, path_col) in enumerate([
            (rgb, 'RGB', 'rgb_resolved'),
            (thermal, 'Thermal', 'thermal_resolved'),
        ]):
            ax = axes[i, j]
            ax.axis('off')
            if img is not None:
                ax.imshow(img)
                ax.set_title(f'{label}\n{Path(str(row.get(path_col))).name}', fontsize=10)
            else:
                ax.text(0.5, 0.5, f'{label} missing', ha='center', va='center')

        question = shorten_text(row.get('question', ''), 130)
        answer = shorten_text(row.get('answer', ''), 90)
        category = shorten_text(row.get('category', 'unknown'), 45)
        fig.text(
            0.5,
            1 - (i / max(len(sample), 1)) - 0.03,
            f'Q: {question}\nA: {answer}    Category: {category}',
            ha='center',
            va='top',
            fontsize=11,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()


plot_vqa_samples(df, n=4)
```

## [cell 13] markdown
## 5. Train/Validation Split

If the dataset provides a split column, you can customize this. By default, we split by `image_id` to reduce leakage across questions from the same frame.

## [cell 14] code
```python
unique_images = df['image_id'].replace('', np.nan).dropna().unique().tolist()
if len(unique_images) >= 10:
    rng = np.random.default_rng(CFG.SEED)
    rng.shuffle(unique_images)
    n_val = max(1, int(len(unique_images) * CFG.VAL_FRAC))
    val_images = set(unique_images[:n_val])
    train_df = df[~df['image_id'].isin(val_images)].copy()
    val_df = df[df['image_id'].isin(val_images)].copy()
else:
    val_df = df.sample(frac=CFG.VAL_FRAC, random_state=CFG.SEED)
    train_df = df.drop(val_df.index).copy()

if CFG.MAX_TRAIN_SAMPLES is not None and len(train_df) > CFG.MAX_TRAIN_SAMPLES:
    train_df = train_df.sample(n=CFG.MAX_TRAIN_SAMPLES, random_state=CFG.SEED)
if CFG.MAX_VAL_SAMPLES is not None and len(val_df) > CFG.MAX_VAL_SAMPLES:
    val_df = val_df.sample(n=CFG.MAX_VAL_SAMPLES, random_state=CFG.SEED)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
print('train:', len(train_df), 'val:', len(val_df))
print('Categories:', train_df['category'].value_counts().head(20).to_dict())
```

## [cell 15] markdown
## 6. Prompt and Dataset

`INCLUDE_OPTIONS_IN_PROMPT=False` keeps the task closer to free-form VQA. If you enable it, the model still generates text, but it sees the candidate answers.

## [cell 16] code
```python
def format_temp_summary(temp_summary):
    if not isinstance(temp_summary, dict):
        return ''
    parts = []
    for key in ['min', 'max', 'mean', 'std', 'top3_mean', 'pct_200', 'pct_400']:
        if key in temp_summary:
            parts.append(f'{key}: {temp_summary[key]}')
    return '; '.join(parts)


def build_prompt(row):
    parts = [
        'Task: Generate a short answer for a wildfire visual question answering problem.',
        'Use the available RGB image and/or thermal image evidence.',
    ]
    if CFG.INCLUDE_TEMP_SUMMARY:
        ts = format_temp_summary(row.get('temp_summary'))
        if ts:
            parts.append('Temperature summary: ' + ts)
    parts.append('Question: ' + str(row['question']))
    if CFG.INCLUDE_OPTIONS_IN_PROMPT and row.get('options'):
        parts.append('Candidate options: ' + ' | '.join(row['options']))
    parts.append('Answer:')
    return '\n'.join(parts)


def load_image_any(x, fallback_color=(0, 0, 0)):
    if isinstance(x, Image.Image):
        return x.convert('RGB')
    if isinstance(x, dict) and x.get('bytes'):
        import io
        return Image.open(io.BytesIO(x['bytes'])).convert('RGB')
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return Image.new('RGB', (224, 224), fallback_color)
    try:
        return Image.open(x).convert('RGB')
    except Exception:
        return Image.new('RGB', (224, 224), fallback_color)

class WildfireGenerativeDataset(Dataset):
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx].to_dict()
        rgb_present = row.get('rgb_resolved') is not None and not pd.isna(row.get('rgb_resolved'))
        thermal_present = row.get('thermal_resolved') is not None and not pd.isna(row.get('thermal_resolved'))
        return {
            'rgb_image': load_image_any(row.get('rgb_resolved'), fallback_color=(0, 0, 0)),
            'thermal_image': load_image_any(row.get('thermal_resolved'), fallback_color=(0, 0, 0)),
            'rgb_present': float(rgb_present),
            'thermal_present': float(thermal_present),
            'prompt': build_prompt(row),
            'target': str(row['answer']),
            'question': str(row['question']),
            'category': str(row.get('category', 'unknown')),
        }
```

## [cell 17] markdown
## 7. Model Definition

This is a minimal custom VLM:

- CLIP vision encoder processes RGB and thermal images independently.
- Two projected visual tokens are prepended to the T5 encoder input.
- T5 decoder generates the answer string.

## [cell 18] code
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_processor = CLIPImageProcessor.from_pretrained(CFG.VISION_MODEL)
tokenizer = T5TokenizerFast.from_pretrained(CFG.TEXT_MODEL)

class WildfireGenerativeVLM(nn.Module):
    def __init__(self, vision_model_name, text_model_name):
        super().__init__()
        self.vision = CLIPVisionModel.from_pretrained(vision_model_name)
        self.t5 = T5ForConditionalGeneration.from_pretrained(text_model_name)
        vision_dim = self.vision.config.hidden_size
        text_dim = self.t5.config.d_model
        self.visual_proj = nn.Sequential(
            nn.Linear(vision_dim, text_dim),
            nn.LayerNorm(text_dim),
            nn.Tanh(),
        )
        self.rgb_type = nn.Parameter(torch.zeros(1, 1, text_dim))
        self.thermal_type = nn.Parameter(torch.zeros(1, 1, text_dim))
        nn.init.normal_(self.rgb_type, std=0.02)
        nn.init.normal_(self.thermal_type, std=0.02)

        if CFG.FREEZE_VISION:
            for p in self.vision.parameters():
                p.requires_grad = False
        if CFG.FREEZE_T5:
            for p in self.t5.parameters():
                p.requires_grad = False
            for p in self.t5.encoder.embed_tokens.parameters():
                p.requires_grad = False

    def encode_visual_token(self, pixel_values, type_embed):
        out = self.vision(pixel_values=pixel_values)
        pooled = out.pooler_output
        token = self.visual_proj(pooled).unsqueeze(1) + type_embed
        return token

    def build_encoder_inputs(self, input_ids, attention_mask, rgb_pixels, thermal_pixels, rgb_present, thermal_present):
        rgb_tok = self.encode_visual_token(rgb_pixels, self.rgb_type)
        th_tok = self.encode_visual_token(thermal_pixels, self.thermal_type)
        text_embeds = self.t5.encoder.embed_tokens(input_ids)
        inputs_embeds = torch.cat([rgb_tok, th_tok, text_embeds], dim=1)
        visual_mask = torch.stack([rgb_present, thermal_present], dim=1).long()
        full_attention_mask = torch.cat([visual_mask, attention_mask], dim=1)
        return inputs_embeds, full_attention_mask

    def forward(self, input_ids, attention_mask, rgb_pixels, thermal_pixels, rgb_present, thermal_present, labels=None):
        inputs_embeds, full_attention_mask = self.build_encoder_inputs(
            input_ids, attention_mask, rgb_pixels, thermal_pixels, rgb_present, thermal_present
        )
        return self.t5(inputs_embeds=inputs_embeds, attention_mask=full_attention_mask, labels=labels)

    @torch.no_grad()
    def generate_answers(self, input_ids, attention_mask, rgb_pixels, thermal_pixels, rgb_present, thermal_present, **gen_kwargs):
        inputs_embeds, full_attention_mask = self.build_encoder_inputs(
            input_ids, attention_mask, rgb_pixels, thermal_pixels, rgb_present, thermal_present
        )
        return self.t5.generate(inputs_embeds=inputs_embeds, attention_mask=full_attention_mask, **gen_kwargs)

model = WildfireGenerativeVLM(CFG.VISION_MODEL, CFG.TEXT_MODEL).to(device)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f'trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)')
```

## [cell 19] code
```python
def collate_fn(batch):
    rgb_images = [b['rgb_image'] for b in batch]
    thermal_images = [b['thermal_image'] for b in batch]
    prompts = [b['prompt'] for b in batch]
    targets = [b['target'] for b in batch]

    rgb = image_processor(images=rgb_images, return_tensors='pt')
    thermal = image_processor(images=thermal_images, return_tensors='pt')
    tok = tokenizer(prompts, padding=True, truncation=True, max_length=CFG.MAX_INPUT_TOKENS, return_tensors='pt')
    lab = tokenizer(targets, padding=True, truncation=True, max_length=CFG.MAX_TARGET_TOKENS, return_tensors='pt')
    labels = lab.input_ids
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        'input_ids': tok.input_ids,
        'attention_mask': tok.attention_mask,
        'rgb_pixels': rgb.pixel_values,
        'thermal_pixels': thermal.pixel_values,
        'rgb_present': torch.tensor([b['rgb_present'] for b in batch], dtype=torch.float32),
        'thermal_present': torch.tensor([b['thermal_present'] for b in batch], dtype=torch.float32),
        'labels': labels,
        'targets': targets,
        'questions': [b['question'] for b in batch],
        'categories': [b['category'] for b in batch],
    }

train_loader = DataLoader(
    WildfireGenerativeDataset(train_df),
    batch_size=CFG.BATCH_SIZE,
    shuffle=True,
    num_workers=CFG.NUM_WORKERS,
    collate_fn=collate_fn,
    pin_memory=torch.cuda.is_available(),
)
val_loader = DataLoader(
    WildfireGenerativeDataset(val_df),
    batch_size=CFG.BATCH_SIZE,
    shuffle=False,
    num_workers=CFG.NUM_WORKERS,
    collate_fn=collate_fn,
    pin_memory=torch.cuda.is_available(),
)

batch = next(iter(train_loader))
{k: (v.shape if torch.is_tensor(v) else type(v)) for k, v in batch.items()}
```

## [cell 20] markdown
## 8. Text Metrics: BLEU and ROUGE-L

These implementations are lightweight and avoid external metric packages.

## [cell 21] code
```python
def simple_tokens(s):
    return re.findall(r"\w+|[^\w\s]", str(s).lower())


def ngram_counts(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))


def sentence_bleu(reference, prediction, max_n=4):
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


def lcs_len(a, b):
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


def rouge_l(reference, prediction):
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


def exact_match(reference, prediction):
    return int(str(reference).strip().lower() == str(prediction).strip().lower())
```

## [cell 22] markdown
## 9. Training

## [cell 23] code
```python
def move_to_device(batch):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
steps_per_epoch = math.ceil(len(train_loader) / CFG.GRAD_ACCUM_STEPS)
total_steps = CFG.EPOCHS * steps_per_epoch
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * CFG.WARMUP_RATIO),
    num_training_steps=total_steps,
)
scaler = torch.cuda.amp.GradScaler(enabled=CFG.MIXED_PRECISION and device.type == 'cuda')

best_val_loss = float('inf')
history = []

for epoch in range(1, CFG.EPOCHS + 1):
    model.train()
    running = 0.0
    seen = 0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(train_loader, desc=f'epoch {epoch}/{CFG.EPOCHS}')
    for step, batch in enumerate(pbar, 1):
        batch = move_to_device(batch)
        with torch.cuda.amp.autocast(enabled=CFG.MIXED_PRECISION and device.type == 'cuda'):
            out = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                rgb_pixels=batch['rgb_pixels'],
                thermal_pixels=batch['thermal_pixels'],
                rgb_present=batch['rgb_present'],
                thermal_present=batch['thermal_present'],
                labels=batch['labels'],
            )
            loss = out.loss / CFG.GRAD_ACCUM_STEPS
        scaler.scale(loss).backward()

        if step % CFG.GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], CFG.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        bs = batch['input_ids'].size(0)
        running += float(loss.detach().cpu()) * CFG.GRAD_ACCUM_STEPS * bs
        seen += bs
        pbar.set_postfix(train_loss=running / max(seen, 1))

    # Validation loss.
    model.eval()
    val_loss = 0.0
    val_seen = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='val loss', leave=False):
            batch = move_to_device(batch)
            out = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                rgb_pixels=batch['rgb_pixels'],
                thermal_pixels=batch['thermal_pixels'],
                rgb_present=batch['rgb_present'],
                thermal_present=batch['thermal_present'],
                labels=batch['labels'],
            )
            bs = batch['input_ids'].size(0)
            val_loss += float(out.loss.detach().cpu()) * bs
            val_seen += bs
    val_loss = val_loss / max(val_seen, 1)
    row = {'epoch': epoch, 'train_loss': running / max(seen, 1), 'val_loss': val_loss}
    history.append(row)
    print(row)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), CFG.OUTPUT_DIR / 'best_model.pt')
        tokenizer.save_pretrained(CFG.OUTPUT_DIR / 'tokenizer')
        print('saved best model')

pd.DataFrame(history).to_csv(CFG.OUTPUT_DIR / 'training_history.csv', index=False)
```

## [cell 24] markdown
## 10. Generate Answers and Evaluate BLEU/ROUGE

## [cell 25] code
```python
# Load best checkpoint.
best_path = CFG.OUTPUT_DIR / 'best_model.pt'
if best_path.exists():
    model.load_state_dict(torch.load(best_path, map_location=device))
model.eval()

pred_rows = []
with torch.no_grad():
    for batch in tqdm(val_loader, desc='generating'):
        batch_dev = move_to_device(batch)
        gen_ids = model.generate_answers(
            input_ids=batch_dev['input_ids'],
            attention_mask=batch_dev['attention_mask'],
            rgb_pixels=batch_dev['rgb_pixels'],
            thermal_pixels=batch_dev['thermal_pixels'],
            rgb_present=batch_dev['rgb_present'],
            thermal_present=batch_dev['thermal_present'],
            max_new_tokens=CFG.MAX_GENERATE_TOKENS,
            num_beams=3,
            early_stopping=True,
        )
        preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        for q, target, pred, cat in zip(batch['questions'], batch['targets'], preds, batch['categories']):
            pred_rows.append({
                'category': cat,
                'question': q,
                'target': target,
                'prediction': pred,
                'exact_match': exact_match(target, pred),
                'bleu': sentence_bleu(target, pred),
                'rouge_l': rouge_l(target, pred),
            })

pred_df = pd.DataFrame(pred_rows)
pred_df.to_csv(CFG.OUTPUT_DIR / 'val_predictions.csv', index=False)
summary = {
    'exact_match': float(pred_df['exact_match'].mean()) if len(pred_df) else 0.0,
    'bleu': float(pred_df['bleu'].mean()) if len(pred_df) else 0.0,
    'rouge_l': float(pred_df['rouge_l'].mean()) if len(pred_df) else 0.0,
    'n': int(len(pred_df)),
}
with open(CFG.OUTPUT_DIR / 'val_metrics.json', 'w') as f:
    json.dump(summary, f, indent=2)
summary
```

## [cell 26] code
```python
# Metrics by category.
if len(pred_df):
    display(pred_df.groupby('category')[['exact_match', 'bleu', 'rouge_l']].mean().sort_values('rouge_l', ascending=False))
    display(pred_df[['question', 'target', 'prediction', 'bleu', 'rouge_l']].head(20))
```

## [cell 27] markdown
## 11. Save/Inspect Outputs

## [cell 28] code
```python
print('Saved files:')
for p in sorted(CFG.OUTPUT_DIR.rglob('*')):
    if p.is_file():
        print(p)
```

## [cell 29] markdown
## Suggested Experiments

1. Run RGB-only by replacing thermal images with blanks or dropping thermal visual tokens.
2. Run thermal-only by replacing RGB images with blanks.
3. Set `INCLUDE_OPTIONS_IN_PROMPT=True` and compare against the free-form setting.
4. Freeze T5 and train only `visual_proj` for a very cheap baseline.
5. Swap `t5-small` for `google/flan-t5-base` if Kaggle GPU memory allows.
6. Add contrastive image-question alignment loss before answer generation, following the alignment paper more closely.
