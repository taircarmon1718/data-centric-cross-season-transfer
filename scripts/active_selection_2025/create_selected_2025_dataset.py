#!/usr/bin/env python3
"""
create_selected_2025_dataset.py

Create a YOLO-pose fine-tuning dataset from selected 2025 images.

Behavior
- Reads selected images from `scripts/active_selection_2025/selected_images.csv`.
- Locates each selected image in the original 2025 dataset (train and val folders).
- Copies image and corresponding label (.txt) into a new dataset:
  `datasets/train_on_2025_selected_k100/` with structure:
    train/images  train/labels
    val/images    val/labels
- Splits selected images 80% train / 20% val deterministically (random_state=0)
- Skips images with missing label files and reports warnings.
- Writes `data.yaml` with the exact required content.

Usage:
    python scripts/active_selection_2025/create_selected_2025_dataset.py

All comments are in English.
"""

from pathlib import Path
import shutil
import argparse
import csv
import json
import sys
import numpy as np
import pandas as pd

# Configuration (change if needed)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SELECTED_CSV = PROJECT_ROOT / 'scripts' / 'active_selection_2025' / 'selected_images.csv'
SRC_IMG_DIRS = [
    PROJECT_ROOT / 'datasets' / 'train_on_2025_all' / 'images',
    PROJECT_ROOT / 'datasets' / 'train_on_2025_all' / 'val' / 'images',
]
SRC_LABEL_DIRS = [
    PROJECT_ROOT / 'datasets' / 'train_on_2025_all' / 'labels',
    PROJECT_ROOT / 'datasets' / 'train_on_2025_all' / 'val' / 'labels',
]
OUT_DATASET = PROJECT_ROOT / 'datasets' / 'train_on_2025_selected_k100'
TRAIN_IMG_OUT = OUT_DATASET / 'train' / 'images'
TRAIN_LABEL_OUT = OUT_DATASET / 'train' / 'labels'
VAL_IMG_OUT = OUT_DATASET / 'val' / 'images'
VAL_LABEL_OUT = OUT_DATASET / 'val' / 'labels'

SPLIT_SEED = 0
TRAIN_FRACTION = 0.8

# Helper functions

def load_selected_csv(path: Path):
    """Load selected_images.csv and return list of image basenames (and optional image_path if present).
    Expected columns: image_path, uncertainty_score, selected_rank (image_path may be relative or absolute)
    """
    if not path.exists():
        raise FileNotFoundError(f'Selected images CSV not found: {path}')
    df = pd.read_csv(path)
    # Ensure image_path column exists
    if 'image_path' not in df.columns:
        raise KeyError('selected_images.csv must contain an `image_path` column')
    # Normalize paths to strings
    df['image_path'] = df['image_path'].astype(str)
    # Use basename for matching
    df['basename'] = df['image_path'].apply(lambda p: Path(p).name)
    return df


def build_image_index(src_dirs):
    """Scan source image directories and build a mapping basename -> list(paths)
    Returned mapping contains resolved Path objects in deterministic sorted order.
    """
    idx = {}
    for src in src_dirs:
        if not src.exists():
            continue
        for p in sorted(src.rglob('*')):
            if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}:
                basename = p.name
                idx.setdefault(basename, []).append(p.resolve())
    return idx


def find_label_for_image(image_path: Path):
    """Given an image Path (resolved), attempt to find the corresponding label .txt.
    Strategy:
      - If image_path is under a known images folder, construct label path by substituting labels root.
      - Otherwise search SRC_LABEL_DIRS for a file with same basename stem + .txt.
    Returns Path to label if found, else None.
    """
    # Try structured replacement: check each source images root
    for img_root, lbl_root in zip(SRC_IMG_DIRS, SRC_LABEL_DIRS):
        try:
            rel = image_path.relative_to(img_root.resolve())
            # label path mirrors relative path but under labels and with .txt suffix
            candidate = (lbl_root / rel).with_suffix('.txt')
            if candidate.exists():
                return candidate.resolve()
        except Exception:
            continue
    # Fallback: search label directories for matching stem
    stem = image_path.stem
    for lbl_root in SRC_LABEL_DIRS:
        if not lbl_root.exists():
            continue
        for p in lbl_root.rglob('*.txt'):
            if p.stem == stem:
                return p.resolve()
    return None


def copy_files(file_pairs, train_idx_set):
    """Copy files to output tree according to train/val membership.
    file_pairs: list of tuples (image_src_path, label_src_path)
    train_idx_set: set of indices (into file_pairs) that belong to train split
    Returns counts: copied_count, skipped_count (should be 0 here as missing labels filtered earlier)
    """
    copied = 0
    skipped = 0
    # ensure output dirs exist
    TRAIN_IMG_OUT.mkdir(parents=True, exist_ok=True)
    TRAIN_LABEL_OUT.mkdir(parents=True, exist_ok=True)
    VAL_IMG_OUT.mkdir(parents=True, exist_ok=True)
    VAL_LABEL_OUT.mkdir(parents=True, exist_ok=True)

    for i, (img_src, lbl_src) in enumerate(file_pairs):
        if i in train_idx_set:
            dst_img = TRAIN_IMG_OUT / Path(img_src).name
            dst_lbl = TRAIN_LABEL_OUT / Path(lbl_src).name
        else:
            dst_img = VAL_IMG_OUT / Path(img_src).name
            dst_lbl = VAL_LABEL_OUT / Path(lbl_src).name
        try:
            shutil.copy2(img_src, dst_img)
            shutil.copy2(lbl_src, dst_lbl)
            copied += 1
        except Exception as e:
            print(f'Warning: failed to copy {img_src} or {lbl_src}: {e}')
            skipped += 1
    return copied, skipped


def write_data_yaml(out_dataset: Path):
    """Write data.yaml with exact required content using absolute paths resolved.
    Format must be:
    nc: 1
    names: ['prawn']
    kpt_shape: [4, 3]
    flip_idx: [0, 1, 2, 3]
    train: <absolute_path_to_train_images>
    val: <absolute_path_to_val_images>
    """
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
    parser = argparse.ArgumentParser(description='Create YOLO-pose dataset from selected 2025 images')
    parser.add_argument('--selected_csv', type=str, default=str(SELECTED_CSV), help='Path to selected_images.csv')
    parser.add_argument('--out_dataset', type=str, default=str(OUT_DATASET), help='Output dataset root')
    parser.add_argument('--train_frac', type=float, default=TRAIN_FRACTION, help='Train fraction (default 0.8)')
    parser.add_argument('--seed', type=int, default=SPLIT_SEED, help='Random seed for split')
    args = parser.parse_args(argv)

    selected_path = Path(args.selected_csv)
    out_dataset = Path(args.out_dataset)

    # Load selected CSV
    try:
        df = load_selected_csv(selected_path)
    except Exception as e:
        print('ERROR: failed to load selected images CSV:', e)
        sys.exit(1)

    total_selected = len(df)

    # Build image index for fast lookup
    img_index = build_image_index(SRC_IMG_DIRS)

    # For each selected image, find source image path and label path
    successful_pairs = []  # tuples (img_src_path, label_src_path)
    skipped = 0
    for _, row in df.iterrows():
        # Preferred: if image_path points to an existing file, use it
        candidate_path = Path(row['image_path'])
        if candidate_path.is_file():
            img_src = candidate_path.resolve()
        else:
            # fallback: find by basename in index
            b = row['basename']
            if b in img_index and len(img_index[b]) > 0:
                img_src = img_index[b][0]
            else:
                print(f'Warning: selected image {row["image_path"]} not found in source folders; skipping')
                skipped += 1
                continue
        # find label
        lbl_src = find_label_for_image(img_src)
        if lbl_src is None or not lbl_src.exists():
            print(f'Warning: label not found for image {img_src}; skipping this image')
            skipped += 1
            continue
        successful_pairs.append((img_src, lbl_src))

    # Now we have successful_pairs list
    copied_total_candidates = len(successful_pairs)
    if copied_total_candidates == 0:
        print('No images with labels found among selected images; exiting.')
        sys.exit(0)

    # Deterministic split into train/val
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(copied_total_candidates)
    n_train = int(math.floor(args.train_frac * copied_total_candidates))
    train_idx = set(perm[:n_train].tolist())

    # Copy files
    copied, copy_skipped = copy_files(successful_pairs, train_idx)

    # Write data.yaml
    yaml_path = write_data_yaml(out_dataset)

    # Print summary
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

