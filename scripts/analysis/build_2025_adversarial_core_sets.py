#!/usr/bin/env python3
"""
build_2025_adversarial_core_sets.py

Create intentionally biased small datasets (5% subsets) from Season 2025
for adversarial / sensitivity testing of cross-season transfer.

Behavior:
- Auto-detect PROJECT_ROOT
- Parse YOLO-pose labels (4 keypoints: 0 carapace,1 eyes,2 rostrum,3 tail)
- Compute per-image metrics (total_length, carapace_length, ratio, angle)
- Build five adversarial subsets (each 5% of valid labeled images):
  - smallest_only_k05
  - largest_only_k05
  - central_band_k05
  - narrow_angle_k05
  - extreme_ratio_k05
- Copy images and labels into datasets/train_on_2025_adversarial/<subset>/
- Reuse original data.yaml (copied into each subset folder if present)
- Print per-subset summary and perform a simple variance reduction verification

Notes:
- Deterministic selection, seed=0
- Uses only numpy, pandas, shutil, pathlib, math
- Handles corrupted/missing labels safely and ignores images without 4 keypoints
"""
from pathlib import Path
import math
import shutil
import sys
import json

import numpy as np
import pandas as pd

# ---------------- Configuration ----------------
SEED = 0
np.random.seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DATASET = PROJECT_ROOT / 'datasets' / 'train_on_2025_all'
SRC_IMAGES_DIRS = [SRC_DATASET / 'images', SRC_DATASET / 'val' / 'images']
SRC_LABEL_DIRS = [SRC_DATASET / 'labels', SRC_DATASET / 'val' / 'labels']
SRC_DATA_YAML = SRC_DATASET / 'data.yaml'

OUT_BASE = PROJECT_ROOT / 'datasets' / 'train_on_2025_adversarial'
OUT_BASE.mkdir(parents=True, exist_ok=True)

SUBSETS = [
    'smallest_only_k05',
    'largest_only_k05',
    'central_band_k05',
    'narrow_angle_k05',
    'extreme_ratio_k05',
]

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# ---------------- Helpers ----------------

def collect_label_files():
    files = []
    for d in SRC_LABEL_DIRS:
        if not d.exists():
            continue
        for p in sorted(d.rglob('*.txt')):
            files.append(p)
    # deduplicate by resolved path
    seen = set()
    unique = []
    for p in files:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        if str(rp) in seen:
            continue
        seen.add(str(rp))
        unique.append(p)
    return unique


def parse_label(label_path: Path):
    """Parse first line of YOLO-pose label. Return 4 keypoints as tuples or None.
    Accepts either normalized coordinates (0-1) or absolute pixels.
    Does not open images. If normalized, returns normalized coords.
    """
    try:
        txt = label_path.read_text().strip()
        if not txt:
            return None
        line = txt.splitlines()[0].strip()
        parts = line.split()
        vals = [float(x) for x in parts]
        if len(vals) < 5 + 8:
            return None
        kps = vals[5:5 + 8]
        kpts = [(kps[i], kps[i + 1]) for i in range(0, len(kps), 2)]
        if len(kpts) < 4:
            return None
        # determine normalized if max <= 1.5
        flat = np.array(kps)
        is_norm = False
        try:
            if flat.max() <= 1.5:
                is_norm = True
        except Exception:
            is_norm = False
        return {'kpts': kpts[:4], 'is_normalized': is_norm}
    except Exception:
        return None


def find_image_for_label(label_path: Path):
    stem = label_path.stem
    # try corresponding file name in images dirs
    for d in SRC_IMAGES_DIRS:
        if not d.exists():
            continue
        # direct mapping
        for ext in IMAGE_EXTS:
            cand = d / (stem + ext)
            if cand.exists():
                return cand
        # recursive search deterministic
        for p in sorted(d.rglob('*')):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.stem == stem:
                return p
    return None


def angular_diff(a, b):
    # minimal absolute angular difference in degrees
    diff = abs(a - b) % 360.0
    if diff > 180.0:
        diff = 360.0 - diff
    return diff

# ---------------- Main ----------------

def main():
    print('PROJECT_ROOT:', PROJECT_ROOT)
    label_files = collect_label_files()
    print('Found label files:', len(label_files))

    records = []
    for lbl in label_files:
        parsed = parse_label(lbl)
        if parsed is None:
            continue
        img = find_image_for_label(lbl)
        if img is None or not img.exists():
            continue
        kpts = parsed['kpts']
        is_norm = parsed['is_normalized']
        # kpts: indices 0..3
        try:
            x0, y0 = kpts[0]
            x1, y1 = kpts[1]
            x2, y2 = kpts[2]
            x3, y3 = kpts[3]
        except Exception:
            continue
        # compute lengths in label units (normalized or pixels)
        carapace_len = math.hypot(x1 - x0, y1 - y0)
        total_len = math.hypot(x3 - x2, y3 - y2)
        if total_len == 0 or not math.isfinite(total_len):
            continue
        ratio = carapace_len / total_len if math.isfinite(carapace_len) else float('nan')
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2))
        # normalize angle to [-180, 180)
        if angle >= 180.0:
            angle -= 360.0
        if angle < -180.0:
            angle += 360.0
        records.append({'label_path': str(lbl.resolve()), 'image_path': str(img.resolve()), 'is_norm': bool(is_norm), 'total_len': float(total_len), 'carapace_len': float(carapace_len), 'ratio': float(ratio), 'angle': float(angle)})

    df = pd.DataFrame.from_records(records)
    if df.empty:
        print('No valid labeled images with 4 keypoints found; exiting')
        return
    N = len(df)
    n_subset = max(1, int(round(0.05 * N)))
    print(f'Total valid images: {N}, subset size (5%): {n_subset}')

    # compute full stats
    full_stats = {
        'total_var': float(np.nanvar(df['total_len'], ddof=1)),
        'ratio_var': float(np.nanvar(df['ratio'], ddof=1)),
        'angle_var': float(np.nanvar(df['angle'], ddof=1)),
    }

    # prepare subsets
    subsets = {}
    # 1 smallest_only_k05
    subsets['smallest_only_k05'] = df.nsmallest(n_subset, 'total_len')
    # 2 largest_only_k05
    subsets['largest_only_k05'] = df.nlargest(n_subset, 'total_len')
    # 3 central_band_k05: closest to mean total_len
    mean_total = float(df['total_len'].mean())
    subsets['central_band_k05'] = df.assign(dist_to_mean=(df['total_len'] - mean_total).abs()).nsmallest(n_subset, 'dist_to_mean')
    # 4 narrow_angle_k05: closest to mean angle (consider wrap)
    mean_angle = float(np.mean(df['angle']))
    subsets['narrow_angle_k05'] = df.assign(angle_diff=df['angle'].apply(lambda a: abs(((a - mean_angle + 180) % 360) - 180))).nsmallest(n_subset, 'angle_diff')
    # 5 extreme_ratio_k05: highest ratio
    subsets['extreme_ratio_k05'] = df.nlargest(n_subset, 'ratio')

    # create datasets and summaries
    summary = {}
    for name, sdf in subsets.items():
        out_root = OUT_BASE / name
        images_out = out_root / 'images'
        labels_out = out_root / 'labels'
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)
        # copy data.yaml if exists
        if SRC_DATA_YAML.exists():
            shutil.copy2(SRC_DATA_YAML, out_root / 'data.yaml')
        # copy files
        copied = 0
        for _, row in sdf.iterrows():
            src_img = Path(row['image_path'])
            src_lbl = Path(row['label_path'])
            if src_img.exists():
                try:
                    shutil.copy2(src_img, images_out / src_img.name)
                except Exception:
                    pass
            if src_lbl.exists():
                try:
                    shutil.copy2(src_lbl, labels_out / src_lbl.name)
                except Exception:
                    pass
            copied += 1
        # compute stats for subset
        total_mean = float(sdf['total_len'].mean())
        total_std = float(sdf['total_len'].std(ddof=1)) if len(sdf) > 1 else 0.0
        ratio_mean = float(sdf['ratio'].mean())
        ratio_std = float(sdf['ratio'].std(ddof=1)) if len(sdf) > 1 else 0.0
        angle_mean = float(sdf['angle'].mean())
        angle_std = float(sdf['angle'].std(ddof=1)) if len(sdf) > 1 else 0.0
        # variance reduction verification: subset variance < 50% of full variance
        total_var_full = full_stats['total_var']
        total_var_sub = float(np.nanvar(sdf['total_len'], ddof=1)) if len(sdf) > 1 else float('nan')
        variance_reduced = (not math.isnan(total_var_sub)) and (total_var_sub < 0.5 * total_var_full)
        summary[name] = {
            'n_images': int(len(sdf)),
            'total_mean': total_mean,
            'total_std': total_std,
            'ratio_mean': ratio_mean,
            'ratio_std': ratio_std,
            'angle_mean': angle_mean,
            'angle_std': angle_std,
            'variance_reduced': bool(variance_reduced),
        }
        print(f"Subset {name}: n_images={len(sdf)}, total_mean={total_mean:.4f}, total_std={total_std:.4f}, ratio_mean={ratio_mean:.4f}, ratio_std={ratio_std:.4f}, angle_mean={angle_mean:.2f}, angle_std={angle_std:.2f}, variance_reduced={variance_reduced}")

    # write summary JSON
    with open(OUT_BASE / 'adversarial_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('\nAdversarial subsets successfully created.')


if __name__ == '__main__':
    main()

