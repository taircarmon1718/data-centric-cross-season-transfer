#!/usr/bin/env python3
"""
build_2025_core_set_yolo_adaptive_full.py

Adaptive domain-gap core-set selection for Season 2025 using YOLO embeddings only.

Standalone script that:
 - Loads existing embeddings (do NOT modify them)
 - Measures domain shift and computes an adaptive annotation budget
 - Selects N most informative 2025 images using FPS on the top-shifted subset
 - Builds a YOLO-pose dataset by copying images and labels
 - Writes selection CSV and JSON summary

Requirements: pathlib, numpy, pandas, tqdm

Author: generated
"""
from pathlib import Path
import argparse
import math
import csv
import json
import shutil
import sys
import warnings

from tqdm import tqdm
import numpy as np
import pandas as pd

# -------------------------- Configuration --------------------------
# Deterministic seed
SEED = 0
np.random.seed(SEED)

# PROJECT_ROOT as requested (Windows path)
PROJECT_ROOT = Path(r"C:\Users\carmonta\Desktop\data-centric-cross-season-transfer")

# Embedding files (existing)
EMB_META = PROJECT_ROOT / "outputs" / "rep_analysis" / "embeddings_meta.csv"
EMB_VEC = PROJECT_ROOT / "outputs" / "rep_analysis" / "embeddings_vectors.npy"

# Source dataset (2025)
SRC_DATASET_ROOT = PROJECT_ROOT / "datasets" / "train_on_2025_all"
SRC_IMG_TRAIN = SRC_DATASET_ROOT / "images"
SRC_IMG_VAL = SRC_DATASET_ROOT / "val" / "images"
SRC_LABEL_TRAIN = SRC_DATASET_ROOT / "labels"
SRC_LABEL_VAL = SRC_DATASET_ROOT / "val" / "labels"

# Output locations
OUT_ROOT = PROJECT_ROOT / "outputs" / "rep_analysis" / "core_set_selection" / "yolo_adaptive_full_experiment"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

OUT_DATASET_ROOT = PROJECT_ROOT / "datasets" / "train_on_2025_core_set_yolo_adaptive"
OUT_IMAGES_TRAIN = OUT_DATASET_ROOT / "images" / "train"
OUT_IMAGES_VAL = OUT_DATASET_ROOT / "images" / "val"
OUT_LABELS_TRAIN = OUT_DATASET_ROOT / "labels" / "train"
OUT_LABELS_VAL = OUT_DATASET_ROOT / "labels" / "val"

# Parameters
ALPHA = 0.2
N_MIN_FRAC = 0.05
N_MAX_FRAC = 0.5
TOP_CANDIDATE_FRAC = 0.40  # top 40% most shifted
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# -------------------------- Helpers --------------------------

def robust_read_csv(path: Path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, header=None)


def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def fps_greedy(E: np.ndarray, n_select: int, initial_index: int = None):
    # E: (M,D) numpy float64
    M = E.shape[0]
    if n_select >= M:
        return list(range(M))
    if initial_index is None:
        # pick farthest from centroid
        centroid = np.mean(E, axis=0)
        d = np.linalg.norm(E - centroid[None, :], axis=1)
        initial_index = int(np.argmax(d))
    selected = [int(initial_index)]
    norms = np.sum(E * E, axis=1, keepdims=True)
    min_sq = np.full(M, np.inf)
    x0 = E[selected[0]]
    dots = E.dot(x0)
    min_sq = np.minimum(min_sq, (norms.flatten() + float(np.sum(x0 * x0)) - 2.0 * dots))
    for _ in range(1, n_select):
        next_idx = int(np.argmax(min_sq))
        selected.append(next_idx)
        xnew = E[next_idx]
        dots = E.dot(xnew)
        sq_to_new = (norms.flatten() + float(np.sum(xnew * xnew)) - 2.0 * dots)
        sq_to_new[sq_to_new < 0] = 0.0
        min_sq = np.minimum(min_sq, sq_to_new)
    return selected


def resolve_image_path(meta_path_str: str):
    p = Path(meta_path_str)
    if p.is_file():
        return p.resolve(), ("val" if "val" in str(p).lower() else ("train" if "images" in str(p).lower() else None))
    p2 = (PROJECT_ROOT / meta_path_str)
    if p2.is_file():
        return p2.resolve(), ("val" if "val" in str(p2).lower() else ("train" if "images" in str(p2).lower() else None))
    basename = p.name
    # deterministic search order: train then val
    for root, split in ((SRC_IMG_TRAIN, "train"), (SRC_IMG_VAL, "val")):
        if root.exists():
            for cand in root.rglob(basename):
                if cand.is_file():
                    return cand.resolve(), split
    return None, None


def find_label_for_image(img_path: Path, split_hint: str = None):
    stem = img_path.stem
    # structured attempt
    try:
        if split_hint == "train" or (SRC_IMG_TRAIN in img_path.parents):
            rel = img_path.relative_to(SRC_IMG_TRAIN)
            candidate = (SRC_LABEL_TRAIN / rel).with_suffix('.txt')
            if candidate.exists():
                return candidate.resolve()
        if split_hint == "val" or (SRC_IMG_VAL in img_path.parents):
            rel = img_path.relative_to(SRC_IMG_VAL)
            candidate = (SRC_LABEL_VAL / rel).with_suffix('.txt')
            if candidate.exists():
                return candidate.resolve()
    except Exception:
        pass
    # fallback search
    for ld in (SRC_LABEL_TRAIN, SRC_LABEL_VAL):
        if not ld.exists():
            continue
        for p in ld.rglob('*.txt'):
            if p.stem == stem:
                return p.resolve()
    return None


def write_selected_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'distance_to_centroid', 'selection_order'])
        for r in rows:
            writer.writerow([r['image_path'], f"{r['distance']:.6f}", r['order']])


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)


# -------------------------- Main --------------------------

def main():
    parser = argparse.ArgumentParser(description='Adaptive domain-gap core-set selection for 2025')
    parser.add_argument('--emb_meta', type=str, default=str(EMB_META))
    parser.add_argument('--emb_vec', type=str, default=str(EMB_VEC))
    parser.add_argument('--out_root', type=str, default=str(OUT_ROOT))
    parser.add_argument('--dataset_out', type=str, default=str(OUT_DATASET_ROOT))
    args = parser.parse_args()

    emb_meta_path = Path(args.emb_meta)
    emb_vec_path = Path(args.emb_vec)
    out_root = Path(args.out_root)
    dataset_out = Path(args.dataset_out)

    if not emb_meta_path.exists() or not emb_vec_path.exists():
        print('ERROR: embedding files missing at:')
        print(' META:', emb_meta_path)
        print(' VEC :', emb_vec_path)
        sys.exit(1)

    # Load
    try:
        meta = pd.read_csv(emb_meta_path)
    except Exception:
        meta = pd.read_csv(emb_meta_path, header=None)
    vecs = np.load(emb_vec_path)
    # align lengths
    if len(meta) != vecs.shape[0]:
        m = min(len(meta), vecs.shape[0])
        warnings.warn(f'Meta rows ({len(meta)}) != vec rows ({vecs.shape[0]}). Aligning to {m}.')
        meta = meta.iloc[:m].reset_index(drop=True)
        vecs = vecs[:m]

    # normalize columns
    meta.columns = [c.strip() for c in meta.columns]
    if 'image_path' not in meta.columns:
        meta = meta.rename(columns={meta.columns[0]: 'image_path'})
    if 'season' not in meta.columns:
        if len(meta.columns) > 1:
            meta = meta.rename(columns={meta.columns[1]: 'season'})
        else:
            meta['season'] = ''
    meta['basename'] = meta['image_path'].astype(str).apply(lambda p: Path(str(p).replace('\\', '/')).name)

    # split indices
    mask24 = meta['season'].astype(str) == '2024'
    mask25 = meta['season'].astype(str) == '2025'
    idx24 = np.where(mask24.values)[0]
    idx25 = np.where(mask25.values)[0]
    N25 = len(idx25)
    if N25 == 0:
        print('No 2025 samples found; aborting')
        sys.exit(0)

    # Step 2: compute domain shift
    vecs_f = np.asarray(vecs).astype(np.float64)
    centroid_2024 = vecs_f[idx24].mean(axis=0)
    centroid_2025 = vecs_f[idx25].mean(axis=0)
    vecs25 = vecs_f[idx25]
    dists25 = np.linalg.norm(vecs25 - centroid_2024[None, :], axis=1)
    mean_2025_distance = float(np.mean(dists25))
    vecs24 = vecs_f[idx24]
    dists24 = np.linalg.norm(vecs24 - centroid_2024[None, :], axis=1)
    mean_2024_spread = float(np.mean(dists24)) if dists24.size > 0 else 1e-9
    if mean_2024_spread <= 0:
        mean_2024_spread = 1e-9
    ShiftScore = mean_2025_distance / mean_2024_spread

    # Step 3: compute annotation budget
    N_total = N25
    N_raw = ALPHA * ShiftScore * N_total
    N_min = max(1, int(math.ceil(N_MIN_FRAC * N_total)))
    N_max = max(1, int(math.floor(N_MAX_FRAC * N_total)))
    N_final = int(round(max(N_min, min(N_raw, N_max))))
    N_final = max(N_min, min(N_final, N_max))

    metrics = {
        'ShiftScore': float(ShiftScore),
        'mean_2025_distance': float(mean_2025_distance),
        'mean_2024_spread': float(mean_2024_spread),
        'N_total_2025': int(N_total),
        'N_raw': float(N_raw),
        'N_min': int(N_min),
        'N_max': int(N_max),
        'N_final': int(N_final),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    write_json(out_root / 'domain_shift_metrics.json', metrics)

    print('ShiftScore:', ShiftScore)
    print('N_total (2025):', N_total)
    print('N_final (selected):', N_final)
    print(f'Percentage selected: {100.0 * N_final / float(N_total):.2f}%')

    # Step 4: select images
    meta25 = meta.iloc[idx25].copy().reset_index(drop=True)
    meta25['distance_to_2024_centroid'] = dists25
    meta25 = meta25.sort_values('distance_to_2024_centroid', ascending=False).reset_index(drop=True)

    top_k = max(1, int(math.ceil(TOP_CANDIDATE_FRAC * N_total)))
    candidates = meta25.iloc[:top_k].copy().reset_index(drop=True)

    # candidate embeddings normalized
    candidate_indices_original = idx25[candidates.index.values]
    E_candidates = vecs_f[candidate_indices_original, :]
    E_candidates_n = l2_normalize_rows(E_candidates)

    if N_final > E_candidates_n.shape[0]:
        N_final = E_candidates_n.shape[0]
        print('Adjusted N_final to candidate pool size:', N_final)

    # initial index: farthest (candidates already sorted by distance so index 0)
    init_idx = 0
    selected_local = fps_greedy(E_candidates_n, n_select=N_final, initial_index=init_idx)

    # map to resolved paths and copy later
    selected_rows = []
    for order, local_idx in enumerate(selected_local, start=1):
        row = candidates.iloc[local_idx]
        meta_img = row['image_path']
        resolved, split = resolve_image_path(str(meta_img))
        if resolved is None:
            warnings.warn(f'Could not resolve image for {meta_img}; skipping')
            continue
        selected_rows.append({'image_path': str(resolved), 'distance': float(row['distance_to_2024_centroid']), 'order': int(order), 'split': split})

    # save selected CSV
    selected_csv = out_root / 'selected_images.csv'
    write_selected_csv(selected_csv, [{'image_path': r['image_path'], 'distance': r['distance'], 'order': r['order']} for r in selected_rows])

    # save summary
    selection_summary = {
        'ShiftScore': float(ShiftScore),
        'N_total_2025': int(N_total),
        'N_selected': len(selected_rows),
        'percentage_selected': float(len(selected_rows)) / float(N_total) if N_total > 0 else 0.0,
    }
    write_json(out_root / 'selection_summary.json', selection_summary)

    # Step 5: build dataset and copy files
    for d in (OUT_IMAGES_TRAIN, OUT_IMAGES_VAL, OUT_LABELS_TRAIN, OUT_LABELS_VAL):
        d.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing_labels = 0
    for r in tqdm(selected_rows, desc='Copying selected files'):
        src_img = Path(r['image_path'])
        split = r.get('split') or ("val" if "val" in str(src_img).lower() else "train")
        dst_img = (OUT_IMAGES_VAL if split == 'val' else OUT_IMAGES_TRAIN) / src_img.name
        label_path = find_label_for_image(src_img, split_hint=split)
        if label_path is None or not label_path.exists():
            missing_labels += 1
            warnings.warn(f'Label missing for {src_img}; skipping copy for this image')
            continue
        dst_lbl = (OUT_LABELS_VAL if split == 'val' else OUT_LABELS_TRAIN) / label_path.name
        try:
            shutil.copy2(src_img, dst_img)
            shutil.copy2(label_path, dst_lbl)
            copied += 1
        except Exception as e:
            warnings.warn(f'Failed copying {src_img} or {label_path}: {e}')

    # write data.yaml exactly
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
    yaml_path = OUT_DATASET_ROOT / 'data.yaml'
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(yaml_lines) + '\n')

    # Final summary
    print('\n=== SUMMARY ===')
    print(f"ShiftScore: {ShiftScore:.6f}")
    print(f"N_total_2025: {N_total}")
    print(f"N_selected: {len(selected_rows)}")
    print(f"Percentage selected: {100.0 * len(selected_rows) / float(N_total):.2f}%")
    print(f"Selected dataset created at: {OUT_DATASET_ROOT}")
    print(f"Images copied: {copied}")
    print(f"Missing labels: {missing_labels}")
    print(f"domain metrics: {out_root / 'domain_shift_metrics.json'}")
    print(f"selected CSV: {selected_csv}")
    print(f"selection summary: {out_root / 'selection_summary.json'}")
    print('\nDone.')


if __name__ == '__main__':
    main()

