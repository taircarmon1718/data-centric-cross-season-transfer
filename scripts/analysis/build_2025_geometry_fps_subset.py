#!/usr/bin/env python3
"""
build_2025_geometry_fps_subset.py

Create a small Season-2025 dataset using Geometry-based Farthest Point Sampling (FPS)
over features: total_length, ratio (carapace/total), angle.

The script:
 - Detects PROJECT_ROOT automatically
 - Parses YOLO-pose labels (4 keypoints: 0 carapace, 1 eyes, 2 rostrum, 3 tail)
 - Computes per-image metrics (carapace_length, total_length, ratio, angle)
 - Normalizes each feature to [0,1] across the dataset
 - Runs greedy FPS in 3D feature space to pick subset
 - Copies selected images and labels into a new YOLO-style dataset
 - Writes data.yaml with absolute train/val image paths
 - Prints summary statistics

Allowed imports: pathlib, shutil, math, numpy, pandas
"""
from pathlib import Path
import math
import shutil
import sys

import numpy as np
import pandas as pd

# ---------------- Configuration ----------------
SUBSET_PERCENT = 0.01  # 1%
MIN_IMAGES = 10
SEED = 0
np.random.seed(SEED)

# PROJECT_ROOT detection
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Source dataset
SRC_ROOT = PROJECT_ROOT / 'datasets' / 'train_on_2025_all'
SRC_IMAGES_TRAIN = SRC_ROOT / 'images'
SRC_IMAGES_VAL = SRC_ROOT / 'val' / 'images'
SRC_LABELS_TRAIN = SRC_ROOT / 'labels'
SRC_LABELS_VAL = SRC_ROOT / 'val' / 'labels'

# Output dataset
OUT_DATASET = PROJECT_ROOT / 'datasets' / 'train_on_2025_geometry_fps_k01'
OUT_IMAGES_TRAIN = OUT_DATASET / 'images' / 'train'
OUT_IMAGES_VAL = OUT_DATASET / 'images' / 'val'
OUT_LABELS_TRAIN = OUT_DATASET / 'labels' / 'train'
OUT_LABELS_VAL = OUT_DATASET / 'labels' / 'val'

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# ---------------- Helpers ----------------

def collect_label_files():
    files = []
    for d in (SRC_LABELS_TRAIN, SRC_LABELS_VAL):
        if not d.exists():
            continue
        for p in sorted(d.rglob('*.txt')):
            files.append(p)
    return files


def parse_label_kpts(label_path: Path):
    try:
        txt = label_path.read_text().strip()
        if not txt:
            return None
        line = txt.splitlines()[0].strip()
        parts = line.split()
        vals = [float(x) for x in parts]
        if len(vals) < 5 + 8:
            return None
        kps = vals[5:5+8]
        kpts = [(kps[i], kps[i+1]) for i in range(0, len(kps), 2)]
        if len(kpts) < 4:
            return None
        return kpts[:4]
    except Exception:
        return None


def find_image_for_label(label_path: Path):
    stem = label_path.stem
    # check train images first then val
    for root in (SRC_IMAGES_TRAIN, SRC_IMAGES_VAL):
        if not root.exists():
            continue
        # try direct candidate with .jpg
        cand = root / (stem + '.jpg')
        if cand.exists():
            return cand, ('train' if root == SRC_IMAGES_TRAIN else 'val')
        # search other extensions
        for ext in IMAGE_EXTS:
            cand = root / (stem + ext)
            if cand.exists():
                return cand, ('train' if root == SRC_IMAGES_TRAIN else 'val')
        # recursive search
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.stem == stem:
                return p, ('train' if root == SRC_IMAGES_TRAIN else 'val')
    return None, None


def compute_distance(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.hypot(dx, dy)


def normalize_minmax(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return arr, 0.0, 1.0
    mn = float(np.nanmin(arr))
    mx = float(np.nanmax(arr))
    if not (np.isfinite(mn) and np.isfinite(mx)) or mx == mn:
        return np.zeros_like(arr, dtype=np.float64), mn, mx
    norm = (arr - mn) / (mx - mn)
    return norm, mn, mx


def fps_greedy(X, n_select, initial_idx=0):
    # X is (M,D) numpy array
    M = X.shape[0]
    if n_select >= M:
        return list(range(M))
    selected = [int(initial_idx)]
    norms = np.sum(X * X, axis=1, keepdims=True)
    min_sq = np.full(M, np.inf)
    x0 = X[selected[0]]
    dots = X.dot(x0)
    min_sq = np.minimum(min_sq, (norms.flatten() + float(np.sum(x0 * x0)) - 2.0 * dots))
    for _ in range(1, n_select):
        nxt = int(np.argmax(min_sq))
        selected.append(nxt)
        xnew = X[nxt]
        dots = X.dot(xnew)
        sq = (norms.flatten() + float(np.sum(xnew * xnew)) - 2.0 * dots)
        sq[sq < 0] = 0.0
        min_sq = np.minimum(min_sq, sq)
    return selected

# ---------------- Main ----------------

def main():
    print('PROJECT_ROOT:', PROJECT_ROOT)
    label_files = collect_label_files()
    print('Found label files:', len(label_files))

    records = []
    for lbl in label_files:
        kpts = parse_label_kpts(lbl)
        if kpts is None:
            continue
        img_path, split = find_image_for_label(lbl)
        if img_path is None:
            continue
        # compute metrics
        car_len = compute_distance(kpts[0], kpts[1])
        tot_len = compute_distance(kpts[2], kpts[3])
        if not math.isfinite(car_len) or not math.isfinite(tot_len) or tot_len <= 0:
            continue
        ratio = car_len / tot_len if tot_len != 0 else float('nan')
        dx = kpts[3][0] - kpts[2][0]
        dy = kpts[3][1] - kpts[2][1]
        angle = math.degrees(math.atan2(dy, dx))
        # normalize angle to [-180,180)
        if angle >= 180.0:
            angle -= 360.0
        if angle < -180.0:
            angle += 360.0
        records.append({
            'label_path': str(lbl.resolve()),
            'image_path': str(img_path.resolve()),
            'split': split,
            'carapace_len': float(car_len),
            'total_len': float(tot_len),
            'ratio': float(ratio),
            'angle': float(angle),
        })

    df = pd.DataFrame.from_records(records)
    if df.empty:
        print('No valid labeled samples found; aborting.')
        return

    N = len(df)
    subset_n = max(MIN_IMAGES, int(math.ceil(SUBSET_PERCENT * N)))
    print(f'Total valid samples: {N}, requested subset size: {subset_n}')

    # Normalize features to [0,1]
    total_norm, tmin, tmax = normalize_minmax(df['total_len'].to_numpy())
    ratio_norm, rmin, rmax = normalize_minmax(df['ratio'].to_numpy())
    # angle normalization: map to [0, 360) then scale
    angles = np.array(df['angle'].to_numpy(), dtype=np.float64)
    ang360 = (angles + 360.0) % 360.0
    ang_norm, amin, amax = normalize_minmax(ang360)

    features = np.stack([total_norm, ratio_norm, ang_norm], axis=1)

    # initial index: farthest from centroid
    centroid = features.mean(axis=0)
    dists_to_centroid = np.linalg.norm(features - centroid[None, :], axis=1)
    init_idx = int(np.argmax(dists_to_centroid))

    selected_idx_local = fps_greedy(features, subset_n, initial_idx=init_idx)
    selected = df.iloc[selected_idx_local].reset_index(drop=True)

    # deterministic split 80/20 using indices
    indices = np.arange(len(selected))
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(indices)
    n_train = int(math.floor(0.8 * len(selected)))
    train_idx = set(perm[:n_train].tolist())

    # create output dirs
    for d in (OUT_IMAGES_TRAIN, OUT_IMAGES_VAL, OUT_LABELS_TRAIN, OUT_LABELS_VAL):
        d.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing_labels = 0
    for i, row in selected.iterrows():
        src_img = Path(row['image_path'])
        src_lbl = Path(row['label_path'])
        split = row['split'] if row['split'] in ('train', 'val') else ('train' if i in train_idx else 'val')
        dst_img = (OUT_IMAGES_TRAIN if split == 'train' else OUT_IMAGES_VAL) / src_img.name
        dst_lbl = (OUT_LABELS_TRAIN if split == 'train' else OUT_LABELS_VAL) / Path(src_lbl.name).name
        try:
            shutil.copy2(src_img, dst_img)
        except Exception:
            print('Warning: failed to copy image', src_img)
        if src_lbl.exists():
            try:
                shutil.copy2(src_lbl, dst_lbl)
            except Exception:
                print('Warning: failed to copy label', src_lbl)
                missing_labels += 1
        else:
            missing_labels += 1
        copied += 1

    # write data.yaml
    train_abs = str(OUT_IMAGES_TRAIN.resolve())
    val_abs = str(OUT_IMAGES_VAL.resolve())
    yaml_lines = [
        'nc: 1',
        "names: ['prawn']",
        'kpt_shape: [4, 3]',
        'flip_idx: [0, 1, 2, 3]',
        f'train: {train_abs}',
        f'val: {val_abs}',
    ]
    with open(OUT_DATASET / 'data.yaml', 'w', encoding='utf-8') as f:
        f.write('\n'.join(yaml_lines) + '\n')

    # summary statistics
    full_var = np.nanvar(df[['total_len', 'ratio', 'angle']].to_numpy(dtype=np.float64), axis=0)
    subset_var = np.nanvar(selected[['total_len', 'ratio', 'angle']].to_numpy(dtype=np.float64), axis=0)
    min_cov = features[selected_idx_local].min(axis=0)
    max_cov = features[selected_idx_local].max(axis=0)

    print('\n=== Summary ===')
    print('Subset size:', len(selected))
    print('Full variance (total, ratio, angle):', full_var.tolist())
    print('Subset variance (total, ratio, angle):', subset_var.tolist())
    print('Min coverage per dimension (normalized):', min_cov.tolist())
    print('Max coverage per dimension (normalized):', max_cov.tolist())
    print('Images copied:', copied)
    print('Missing labels:', missing_labels)
    print('\nDone.')


if __name__ == '__main__':
    main()

