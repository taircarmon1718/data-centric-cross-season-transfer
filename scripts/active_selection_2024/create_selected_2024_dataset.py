#!/usr/bin/env python3
"""
create_selected_2024_dataset.py

Create a YOLO-pose fine-tuning dataset from selected 2024 images.
This mirrors the 2025 script but adapts paths for 2024.

Usage:
    python scripts/active_selection_2024/create_selected_2024_dataset.py
"""

from pathlib import Path
import shutil
import argparse
import sys
import numpy as np
import pandas as pd
import math

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SELECTED_CSV = PROJECT_ROOT / 'scripts' / 'active_selection_2024' / 'selected_images_2024.csv'
SRC_IMG_DIRS = [
    PROJECT_ROOT / 'datasets' / 'train_on_all' / 'images',
    PROJECT_ROOT / 'datasets' / 'train_on_all' / 'val' / 'images',
]
SRC_LABEL_DIRS = [
    PROJECT_ROOT / 'datasets' / 'train_on_all' / 'labels',
    PROJECT_ROOT / 'datasets' / 'train_on_all' / 'val' / 'labels',
]
OUT_DATASET_BASE = PROJECT_ROOT / 'datasets'

SPLIT_SEED = 0
TRAIN_FRACTION = 0.8

# Helper functions

def load_selected_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f'Selected images CSV not found: {path}')
    df = pd.read_csv(path)
    if 'image_path' not in df.columns:
        raise KeyError('selected_images CSV must contain image_path column')
    df['image_path'] = df['image_path'].astype(str)
    df['basename'] = df['image_path'].apply(lambda p: Path(p).name)
    return df


def build_image_index(src_dirs):
    idx = {}
    for src in src_dirs:
        if not src.exists():
            continue
        for p in sorted(src.rglob('*')):
            if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}:
                idx.setdefault(p.name, []).append(p.resolve())
    return idx


def find_label_for_image(image_path: Path):
    for img_root, lbl_root in zip(SRC_IMG_DIRS, SRC_LABEL_DIRS):
        try:
            rel = image_path.relative_to(img_root.resolve())
            candidate = (lbl_root / rel).with_suffix('.txt')
            if candidate.exists():
                return candidate.resolve()
        except Exception:
            continue
    stem = image_path.stem
    for lbl_root in SRC_LABEL_DIRS:
        if not lbl_root.exists():
            continue
        for p in lbl_root.rglob('*.txt'):
            if p.stem == stem:
                return p.resolve()
    return None


def copy_files(file_pairs, train_idx_set, out_dataset):
    train_img = out_dataset / 'train' / 'images'
    train_lbl = out_dataset / 'train' / 'labels'
    val_img = out_dataset / 'val' / 'images'
    val_lbl = out_dataset / 'val' / 'labels'
    train_img.mkdir(parents=True, exist_ok=True)
    train_lbl.mkdir(parents=True, exist_ok=True)
    val_img.mkdir(parents=True, exist_ok=True)
    val_lbl.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    for i, (img_src, lbl_src) in enumerate(file_pairs):
        if i in train_idx_set:
            dst_img = train_img / Path(img_src).name
            dst_lbl = train_lbl / Path(lbl_src).name
        else:
            dst_img = val_img / Path(img_src).name
            dst_lbl = val_lbl / Path(lbl_src).name
        try:
            shutil.copy2(img_src, dst_img)
            shutil.copy2(lbl_src, dst_lbl)
            copied += 1
        except Exception as e:
            print(f'Warning: failed to copy {img_src} or {lbl_src}: {e}')
            skipped += 1
    return copied, skipped


def write_data_yaml(out_dataset: Path):
    train_abs = str((out_dataset / 'train' / 'images').resolve())
    val_abs = str((out_dataset / 'val' / 'images').resolve())
    yaml_lines = [
        'nc: 1',
        "names: ['prawn']",
        'kpt_shape: [4, 3]',
        'flip_idx: [0, 1, 2, 3]',
        f'train: {train_abs}',
        f'val: {val_abs}',
    ]
    yaml_path = out_dataset / 'data.yaml'
    with open(yaml_path, 'w', newline='') as f:
        f.write('\n'.join(yaml_lines) + '\n')
    return yaml_path


def main(argv=None):
    parser = argparse.ArgumentParser(description='Create selected 2024 dataset')
    parser.add_argument('--selected_csv', type=str, default=str(SELECTED_CSV))
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--seed', type=int, default=SPLIT_SEED)
    parser.add_argument('--train_frac', type=float, default=TRAIN_FRACTION)
    args = parser.parse_args(argv)

    df = load_selected_csv(Path(args.selected_csv))
    total_selected = len(df)
    img_index = build_image_index(SRC_IMG_DIRS)

    successful_pairs = []
    skipped = 0
    for _, row in df.iterrows():
        candidate_path = Path(row['image_path'])
        if candidate_path.is_file():
            img_src = candidate_path.resolve()
        else:
            b = row['basename']
            if b in img_index and len(img_index[b]) > 0:
                img_src = img_index[b][0]
            else:
                print(f'Warning: selected image {row["image_path"]} not found in source folders; skipping')
                skipped += 1
                continue
        lbl_src = find_label_for_image(img_src)
        if lbl_src is None or not lbl_src.exists():
            print(f'Warning: label not found for image {img_src}; skipping this image')
            skipped += 1
            continue
        successful_pairs.append((img_src, lbl_src))

    copied_total_candidates = len(successful_pairs)
    if copied_total_candidates == 0:
        print('No images with labels found among selected images; exiting.')
        sys.exit(0)

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(copied_total_candidates)
    n_train = int(math.floor(args.train_frac * copied_total_candidates))
    train_idx = set(perm[:n_train].tolist())

    out_dataset = OUT_DATASET_BASE / f'train_on_2024_selected_k{args.k}'
    copied, copy_skipped = copy_files(successful_pairs, train_idx, out_dataset)

    yaml_path = write_data_yaml(out_dataset)

    print('\nSummary:')
    print(f'  Total selected images (from CSV): {total_selected}')
    print(f'  Successfully matched (image+label): {copied_total_candidates}')
    print(f'  Skipped due to missing files/labels: {skipped}')
    print(f'  Files copied: {copied} (failed during copy: {copy_skipped})')
    print(f'  Train count: {n_train}')
    print(f'  Val count: {copied_total_candidates - n_train}')
    print(f'  data.yaml written to: {yaml_path}')


if __name__ == '__main__':
    main()

