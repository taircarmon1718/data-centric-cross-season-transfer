#!/usr/bin/env python3
"""
extract_test_embeddings_2024_2025.py

Extract embeddings for the TEST image folders for 2024 and 2025 using the
exact same YOLO backbone extraction pipeline used by existing scripts.

Behavior:
- Loads the backbone via `utils.model_utils.load_yolo_backbone`
- Uses `preprocess` and `extract_fn` from the backend
- Collects images only from the explicit test folders listed in the prompt
- Produces `test_embeddings_meta.csv` and `test_embeddings_vectors.npy` under
  `scripts/representation_analysis/test_embeddings/`

Metadata columns (CSV): image_path, season, split, subtype
Ordering: deterministic sorted order over file paths per subtype, preserved in output
"""

import os
import random
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch

# follow same utility functions
from utils.model_utils import load_yolo_backbone
from utils.io import save_embeddings

# Deterministic seed (match other scripts)
RANDOM_SEED = 12345
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

# Model path (must be identical to original)
MODEL_PATH = PROJECT_ROOT / 'models' / '2024' / 'all-ponds' / 'weights' / 'best.pt'

# Test folders (explicit from prompt)
TEST_FOLDERS = [
    # 2024 body
    (PROJECT_ROOT / 'data' / 'images' / 'test_2024_originalSize' / 'test_images_2024_orginalSize' / 'body', '2024', 'test', 'body'),
    # 2024 carapace
    (PROJECT_ROOT / 'data' / 'images' / 'test_2024_originalSize' / 'test_images_2024_orginalSize' / 'carapace', '2024', 'test', 'carapace'),
    # 2025 all images
    (PROJECT_ROOT / 'data' / 'images' / 'test_2025_gamma' / 'test_images_2025_gamma' / 'ALL_IMAGES_640x360', '2025', 'test', 'all'),
]

# Output dir
OUT_DIR = PROJECT_ROOT / 'scripts' / 'representation_analysis' / 'test_embeddings'
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def collect_test_records(folders):
    records = []
    for folder, season, split, subtype in folders:
        folder = Path(folder)
        if not folder.exists():
            print(f"Warning: test folder {folder} not found; skipping.")
            continue
        files = [p for p in sorted(folder.rglob('*')) if p.suffix.lower() in IMAGE_EXTS]
        for p in files:
            records.append({
                'image_path': str(p),
                'season': season,
                'split': split,
                'subtype': subtype,
            })
    # ensure deterministic global order
    records = sorted(records, key=lambda r: (r['season'], r['subtype'], r['image_path']))
    return records


def extract_test_embeddings(model_path: Path, out_dir: Path, device: str = None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Device:', device, 'CUDA available:', torch.cuda.is_available())

    records = collect_test_records(TEST_FOLDERS)
    if len(records) == 0:
        print('No test images found; exiting.')
        return

    print(f'Found {len(records)} test images; loading backbone from {model_path} ...')
    backend = load_yolo_backbone(str(model_path), device=device)
    preprocess = backend['preprocess']
    extract_fn = backend['extract_fn']
    handle = backend.get('hook_handle', None)

    embeddings = []
    processed_meta = []

    for rec in tqdm(records, desc='Extracting test embeddings'):
        img_path = Path(rec['image_path'])
        try:
            img = Image.open(img_path).convert('RGB')
            tensor = preprocess(img).to(device)
            with torch.no_grad():
                emb = extract_fn(tensor)
            if emb is None:
                print(f'Warning: extract_fn returned None for {img_path}; skipping')
                continue
            emb = np.asarray(emb).reshape(-1)
            embeddings.append(emb)
            # ensure saved meta matches required CSV columns
            processed_meta.append({
                'image_path': str(img_path),
                'season': rec['season'],
                'split': rec['split'],
                'subtype': rec['subtype'],
            })
        except Exception as e:
            print(f'Warning: failed to process {img_path}: {e}')
            continue

    # remove hook if present
    if handle is not None:
        try:
            handle.remove()
        except Exception:
            pass

    if len(embeddings) == 0:
        print('No embeddings extracted from test images; exiting.')
        return

    E = np.stack(embeddings, axis=0)
    print('Extracted embeddings shape:', E.shape)

    # Save in same format as other scripts: use save_embeddings (which writes embeddings_meta.csv and embeddings_vectors.npy)
    # But `save_embeddings` expects records with keys image_path, season, split; we'll adapt by writing our own CSV to include subtype.

    # Use utils.io.save_embeddings to save basic CSV (image_path, season, split) and npy, then overwrite CSV to include subtype column
    basic_records = [{'image_path': r['image_path'], 'season': r['season'], 'split': r['split']} for r in processed_meta]
    # save vectors and basic CSV
    save_embeddings(str(out_dir), basic_records, E)

    # Now rewrite meta CSV to include subtype column and follow original column order: image_path, season, split, subtype
    meta_csv = out_dir / 'embeddings_meta.csv'
    # read existing meta and write augmented
    import csv
    tmp_rows = []
    with open(meta_csv, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            tmp_rows.append(row)
    # header is ['image_path','season','split']
    new_csv_path = out_dir / 'test_embeddings_meta.csv'
    with open(new_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'season', 'split', 'subtype'])
        for orig, meta in zip(tmp_rows, processed_meta):
            writer.writerow([meta['image_path'], meta['season'], meta['split'], meta['subtype']])

    # move vectors file to expected output name
    vectors_src = out_dir / 'embeddings_vectors.npy'
    vectors_dst = out_dir / 'test_embeddings_vectors.npy'
    vectors_src.replace(vectors_dst)

    print('Saved test embeddings:')
    print(' - meta:', new_csv_path)
    print(' - vectors:', vectors_dst)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=str(MODEL_PATH))
    parser.add_argument('--out', type=str, default=str(OUT_DIR))
    parser.add_argument('--device', type=str, default=( 'cuda' if torch.cuda.is_available() else 'cpu'))
    args = parser.parse_args()
    extract_test_embeddings(Path(args.model), Path(args.out), device=args.device)

