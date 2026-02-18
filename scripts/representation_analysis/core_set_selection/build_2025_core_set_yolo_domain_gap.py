#!/usr/bin/env python3
"""
build_2025_core_set_yolo_domain_gap.py

Select Season-2025 core-sets by distance from the Season-2024 centroid in the
YOLO embedding space. Builds YOLO-style datasets for multiple K values.

Usage:
    python scripts/representation_analysis/core_set_selection/build_2025_core_set_yolo_domain_gap.py

Requirements / behavior (summary):
- Load embeddings_meta.csv and embeddings_vectors.npy (do NOT modify them)
- Split by season column
- L2-normalize embeddings (float64)
- Compute centroid of 2024 embeddings and distances for 2025 samples
- For each k in K_LIST select top-n farthest samples
- Save selection CSVs and build datasets preserving original train/val split
- Deterministic (numpy seed = 0)

Outputs:
- outputs/rep_analysis/core_set_selection/yolo_domain_gap/kXX/core_set_kXX.csv
- datasets/train_on_2025_core_set_yolo_domain_gap_kXX/ with images/train, images/val, labels/train, labels/val and data.yaml

"""

from pathlib import Path
import argparse
import math
import csv
import shutil
import sys
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

# Deterministic
NP_SEED = 0
np = np  # alias used later
np.random.seed(NP_SEED)

# Constants
K_LIST = [1, 2, 5, 10, 20, 50]
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Default embedding paths (repository layout)
DEFAULT_EMB_META = PROJECT_ROOT / 'scripts' / 'representation_analysis' / 'outputs_repreasentation' / 'rep_analysis' / 'embeddings_meta.csv'
DEFAULT_EMB_VEC = PROJECT_ROOT / 'scripts' / 'representation_analysis' / 'outputs_repreasentation' / 'rep_analysis' / 'embeddings_vectors.npy'

# candidate image roots
SRC_DATASET_ROOT = PROJECT_ROOT / 'datasets' / 'train_on_2025_all'
SRC_IMG_TRAIN = SRC_DATASET_ROOT / 'images'
SRC_IMG_VAL = SRC_DATASET_ROOT / 'val' / 'images'
SRC_LABEL_TRAIN = SRC_DATASET_ROOT / 'labels'
SRC_LABEL_VAL = SRC_DATASET_ROOT / 'val' / 'labels'

OUT_BASE = PROJECT_ROOT / 'outputs' / 'rep_analysis' / 'core_set_selection' / 'yolo_domain_gap'
OUT_BASE.mkdir(parents=True, exist_ok=True)

DATASETS_OUT_ROOT = PROJECT_ROOT / 'datasets'

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# ------------------- Helper functions -------------------

def load_embeddings(meta_path: Path, vec_path: Path):
    """Load metadata CSV and vectors .npy, align by min length, return (meta_df, vectors)
    Meta must contain at least image_path and season and ideally a split/dataset_type column.
    """
    if not meta_path.exists():
        raise FileNotFoundError(f"Embeddings meta not found: {meta_path}")
    if not vec_path.exists():
        raise FileNotFoundError(f"Embeddings vectors not found: {vec_path}")

    # read meta robustly
    try:
        meta = pd.read_csv(meta_path)
    except Exception:
        meta = pd.read_csv(meta_path, header=None)

    vecs = np.load(vec_path)

    if len(meta) != vecs.shape[0]:
        m = min(len(meta), vecs.shape[0])
        warnings.warn(f"Meta rows ({len(meta)}) != vec rows ({vecs.shape[0]}). Aligning by min length={m}.")
        meta = meta.iloc[:m].reset_index(drop=True)
        vecs = vecs[:m]

    # normalize column names
    meta.columns = [c.strip() for c in meta.columns]
    if 'image_path' not in meta.columns:
        meta = meta.rename(columns={meta.columns[0]: 'image_path'})
    if 'season' not in meta.columns:
        # try second column
        if len(meta.columns) > 1:
            meta = meta.rename(columns={meta.columns[1]: 'season'})
        else:
            meta['season'] = ''
    # support 'split' or 'dataset_type'
    if 'dataset_type' not in meta.columns:
        if 'split' in meta.columns:
            meta = meta.rename(columns={'split': 'dataset_type'})
        else:
            meta['dataset_type'] = ''

    # add basename for matching
    meta['basename'] = meta['image_path'].astype(str).apply(lambda p: Path(str(p).replace('\\', '/')).name)

    return meta, vecs


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    X = x.astype(np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def find_source_image_from_meta(image_path_str: str):
    """Given image_path from meta, try resolving to an existing file.
    Try several fallbacks: absolute, PROJECT_ROOT relative, search by basename.
    Returns (resolved_path, split) where split is 'train' or 'val' or None if unknown.
    """
    # Try as absolute
    p = Path(image_path_str)
    if p.is_file():
        # determine split by parent path
        if str(SRC_IMG_TRAIN) in str(p):
            return p.resolve(), 'train'
        if str(SRC_IMG_VAL) in str(p):
            return p.resolve(), 'val'
        return p.resolve(), None
    # Try relative to project root
    p2 = (PROJECT_ROOT / image_path_str).resolve()
    if p2.is_file():
        if str(SRC_IMG_TRAIN) in str(p2):
            return p2, 'train'
        if str(SRC_IMG_VAL) in str(p2):
            return p2, 'val'
        return p2, None
    # Fallback: search by basename in dataset folders
    basename = Path(image_path_str).name
    # check train then val (deterministic)
    for d, split in ((SRC_IMG_TRAIN, 'train'), (SRC_IMG_VAL, 'val')):
        if d.exists():
            candidate = next(d.rglob(basename), None)
            if candidate is not None and candidate.is_file():
                return candidate.resolve(), split
    return None, None


def find_label_for_image(image_path: Path, split: str):
    """Find the corresponding label file for given image.
    Try structured mapping (replace images root with labels root) and fall back to searching by stem.
    Return Path or None.
    """
    stem = image_path.stem
    # structured attempt: if image under train images, map to train labels
    if split == 'train' and SRC_IMG_TRAIN in image_path.parents:
        rel = image_path.relative_to(SRC_IMG_TRAIN)
        candidate = SRC_LABEL_TRAIN / rel
        candidate = candidate.with_suffix('.txt')
        if candidate.exists():
            return candidate.resolve()
    if split == 'val' and SRC_IMG_VAL in image_path.parents:
        rel = image_path.relative_to(SRC_IMG_VAL)
        candidate = SRC_LABEL_VAL / rel
        candidate = candidate.with_suffix('.txt')
        if candidate.exists():
            return candidate.resolve()
    # fallback: search label dirs by stem
    for lab_root in (SRC_LABEL_TRAIN, SRC_LABEL_VAL):
        if not lab_root.exists():
            continue
        # prefer deterministic first match
        for p in lab_root.rglob('*.txt'):
            if p.stem == stem:
                return p.resolve()
    return None


def write_core_csv(out_csv: Path, rows):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'distance_to_2024_centroid', 'rank'])
        for r in rows:
            writer.writerow([r['image_path'], f"{r['distance']:.6f}", r['rank']])


def copy_selected_to_dataset(selected_rows, dataset_out_root: Path):
    """Copy selected images and labels into dataset_out_root preserving train/val split.
    Returns missing_label_count and copied_count
    """
    images_train = dataset_out_root / 'images' / 'train'
    images_val = dataset_out_root / 'images' / 'val'
    labels_train = dataset_out_root / 'labels' / 'train'
    labels_val = dataset_out_root / 'labels' / 'val'
    for d in (images_train, images_val, labels_train, labels_val):
        d.mkdir(parents=True, exist_ok=True)

    missing_labels = 0
    copied = 0

    for row in selected_rows:
        src_img = Path(row['resolved_path'])
        split = row.get('split', 'train') if row.get('split') in ('train', 'val') else 'train'
        # choose destination
        if split == 'train':
            dst_img = images_train / src_img.name
        else:
            dst_img = images_val / src_img.name
        # find label
        label_path = find_label_for_image(src_img, split)
        if label_path is None or (not label_path.exists()):
            missing_labels += 1
            continue
        if split == 'train':
            dst_lbl = labels_train / label_path.name
        else:
            dst_lbl = labels_val / label_path.name
        try:
            shutil.copy2(src_img, dst_img)
            shutil.copy2(label_path, dst_lbl)
            copied += 1
        except Exception as e:
            warnings.warn(f"Failed copying {src_img} or {label_path}: {e}")
    return missing_labels, copied


def write_data_yaml(dataset_out_root: Path):
    train_abs = str((dataset_out_root / 'images' / 'train').resolve())
    val_abs = str((dataset_out_root / 'images' / 'val').resolve())
    content = [
        'nc: 1',
        "names: ['prawn']",
        'kpt_shape: [4, 3]',
        'flip_idx: [0, 1, 2, 3]',
        f'train: {train_abs}',
        f'val: {val_abs}',
    ]
    yaml_path = dataset_out_root / 'data.yaml'
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text('\n'.join(content) + '\n')
    return yaml_path


# ------------------- Main script -------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description='Build 2025 core-sets by YOLO domain-gap (distance to 2024 centroid)')
    parser.add_argument('--emb_meta', type=str, default=str(DEFAULT_EMB_META), help='Path to embeddings_meta.csv')
    parser.add_argument('--emb_vec', type=str, default=str(DEFAULT_EMB_VEC), help='Path to embeddings_vectors.npy')
    parser.add_argument('--k_list', type=str, default=','.join(map(str, K_LIST)), help='Comma-separated percent values e.g. 1,2,5')
    parser.add_argument('--out_base', type=str, default=str(OUT_BASE), help='Base output folder for core set CSVs')
    parser.add_argument('--datasets_root', type=str, default=str(DATASETS_OUT_ROOT), help='Root where output datasets will be created')
    args = parser.parse_args(argv)

    emb_meta_path = Path(args.emb_meta)
    emb_vec_path = Path(args.emb_vec)

    # load embeddings
    try:
        meta, vecs = load_embeddings(emb_meta_path, emb_vec_path)
    except Exception as e:
        print('ERROR loading embeddings:', e)
        sys.exit(1)

    # split by season
    is_2024 = meta['season'].astype(str) == '2024'
    is_2025 = meta['season'].astype(str) == '2025'
    idx24 = np.where(is_2024.values)[0]
    idx25 = np.where(is_2025.values)[0]

    N25 = len(idx25)
    print(f'Total 2025 embeddings in meta: {N25}')
    if N25 == 0:
        print('No 2025 samples found in embeddings meta; aborting')
        sys.exit(0)

    # L2 normalize all embeddings (float64)
    vecs_n = l2_normalize_rows(np.asarray(vecs))

    # compute centroid of 2024
    if len(idx24) == 0:
        print('No 2024 samples found in embeddings meta; cannot compute centroid')
        sys.exit(1)
    centroid_2024 = np.mean(vecs_n[idx24, :], axis=0)

    # compute distances for each 2025 embedding
    vecs25 = vecs_n[idx25, :]
    diffs = vecs25 - centroid_2024[None, :]
    dists = np.linalg.norm(diffs, axis=1)

    # attach distances to meta rows for 2025
    meta25 = meta.iloc[idx25].copy().reset_index(drop=True)
    meta25['distance_to_2024_centroid'] = dists
    # rank descending
    meta25 = meta25.sort_values('distance_to_2024_centroid', ascending=False).reset_index(drop=True)
    meta25['rank'] = meta25.index + 1

    # Prepare dictionary mapping basename->list of meta25 rows (keep deterministic order)
    basename_to_rows = {}
    for i, r in meta25.iterrows():
        b = r['basename']
        basename_to_rows.setdefault(b, []).append((i, r))

    # For each k produce selection and dataset
    k_vals = [int(x.strip()) for x in args.k_list.split(',') if x.strip()]
    missing_labels_total = 0
    for k in k_vals:
        n_select = int(math.ceil((k / 100.0) * N25))
        if n_select <= 0:
            print(f'k={k} produced n_select=0; skipping')
            continue
        selected = meta25.iloc[:n_select].copy().reset_index(drop=True)
        # For saving, produce rows with image_path and distance and rank
        rows_to_save = []
        resolved_selected = []
        missing_labels = 0
        for _, row in selected.iterrows():
            image_path_str = str(row['image_path'])
            resolved, split = find_source_image_from_meta(image_path_str)
            if resolved is None:
                # fallback to search by basename
                candidate_basename = row['basename']
                # search train then val directories
                found = None
                for d in (SRC_IMG_TRAIN, SRC_IMG_VAL):
                    if d.exists():
                        for p in d.rglob(candidate_basename):
                            found = p
                            break
                    if found:
                        break
                if found:
                    resolved = found.resolve()
                    split = 'train' if str(SRC_IMG_TRAIN) in str(found) else 'val'
            if resolved is None:
                warnings.warn(f'Could not resolve image for {image_path_str}; skipping')
                continue
            # find label
            label_path = find_label_for_image(resolved, split)
            if label_path is None:
                missing_labels += 1
            rows_to_save.append({'image_path': str(resolved), 'distance': float(row['distance_to_2024_centroid']), 'rank': int(row['rank'])})
            resolved_selected.append({'resolved_path': str(resolved), 'split': split, 'distance': float(row['distance_to_2024_centroid']), 'rank': int(row['rank'])})

        # save selection CSV
        k_str = f'k{int(k):02d}'
        out_k_dir = Path(args.out_base) / k_str
        out_k_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_k_dir / f'core_set_{k_str}.csv'
        write_core_csv(csv_path, rows_to_save)

        # build dataset
        dataset_name = f'train_on_2025_core_set_yolo_domain_gap_{k_str}'
        dataset_out_root = Path(args.datasets_root) / dataset_name
        missing_lbl_count, copied_count = copy_selected_to_dataset(resolved_selected, dataset_out_root)
        yaml_path = write_data_yaml(dataset_out_root)

        print(f'k={k}: selected {len(rows_to_save)} samples, copied {copied_count} images to {dataset_out_root}, missing labels={missing_lbl_count}')
        print(f'  selection CSV: {csv_path}')
        print(f'  data.yaml: {yaml_path}\n')
        missing_labels_total += missing_lbl_count

    # final summary
    print('=== Summary ===')
    print(f'Total 2025 samples considered: {N25}')
    for k in k_vals:
        print(f'  k={k}% -> selection folder: {OUT_BASE}/k{int(k):02d}')
    print(f'Total missing labels across all selections: {missing_labels_total}')


if __name__ == '__main__':
    main()

