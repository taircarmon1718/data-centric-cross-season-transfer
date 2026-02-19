#!/usr/bin/env python3
"""
build_shifted_core_set_yolo_adaptive.py

Adaptive domain-gap core-set selection for SHIFTED 2025 using existing unified embeddings.

- Uses embeddings at scripts/shift_experiments/embeddings/all_embeddings.npy
  and scripts/shift_experiments/embeddings/all_meta.csv
- Filters dataset == '2025_shifted' for target embeddings
- Uses 2024 embeddings from same file as reference centroid
- Selection logic: computes ShiftScore, uses EXCESS SHIFT, adaptive budget
- Selects via FPS on L2-normalized embeddings among top 40% candidates
- Builds dataset at datasets/train_on_2025_shifted_core_set_yolo_adaptive/
- Saves outputs under outputs/shift_experiments/yolo_adaptive_shifted/

Strict rules: deterministic, pathlib, no modification of original datasets, handle missing files
"""
from pathlib import Path
import math
import json
import warnings
import shutil
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

# Configuration
SEED = 0
np.random.seed(SEED)
ALPHA = 0.2
TOP_CANDIDATE_FRAC = 0.40
N_MIN_FRAC = 0.01  # 1% of shifted dataset
N_MAX_FRAC = 0.5   # 50% of shifted dataset

PROJECT_ROOT = Path(r"C:\Users\carmonta\Desktop\data-centric-cross-season-transfer")
EMB_DIR = PROJECT_ROOT / 'scripts' / 'shift_experiments' / 'embeddings'
EMB_VEC = EMB_DIR / 'all_embeddings.npy'
EMB_META = EMB_DIR / 'all_meta.csv'

# shifted dataset source (preferred)
SHIFTED_SRC_ROOT = PROJECT_ROOT / 'datasets' / 'train_on_2025_shifted_all'
SHIFTED_SRC_IMG_TRAIN = SHIFTED_SRC_ROOT / 'images'
SHIFTED_SRC_IMG_VAL = SHIFTED_SRC_ROOT / 'val' / 'images'
SHIFTED_SRC_LABEL_TRAIN = SHIFTED_SRC_ROOT / 'labels'
SHIFTED_SRC_LABEL_VAL = SHIFTED_SRC_ROOT / 'val' / 'labels'
# fallback to outputs shifted experiment
FALLBACK_SHIFTED_ROOT = PROJECT_ROOT / 'outputs' / 'rep_analysis' / 'core_set_selection' / 'shifted_2025_experiment' / 'shifted_2025_experiment'

OUT_ROOT = PROJECT_ROOT / 'outputs' / 'shift_experiments' / 'yolo_adaptive_shifted'
OUT_ROOT.mkdir(parents=True, exist_ok=True)

OUT_DATASET = PROJECT_ROOT / 'datasets' / 'train_on_2025_shifted_core_set_yolo_adaptive'
OUT_IMAGES_TRAIN = OUT_DATASET / 'train' / 'images'
OUT_IMAGES_VAL = OUT_DATASET / 'val' / 'images'
OUT_LABELS_TRAIN = OUT_DATASET / 'train' / 'labels'
OUT_LABELS_VAL = OUT_DATASET / 'val' / 'labels'

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# Helpers

def load_embeddings(meta_path: Path, vec_path: Path):
    if not meta_path.exists() or not vec_path.exists():
        raise FileNotFoundError(f'Embeddings not found at {meta_path} or {vec_path}')
    meta = pd.read_csv(meta_path)
    vecs = np.load(vec_path)
    if len(meta) != vecs.shape[0]:
        m = min(len(meta), vecs.shape[0])
        warnings.warn(f'Meta rows ({len(meta)}) != vec rows ({vecs.shape[0]}). Aligning to {m}.')
        meta = meta.iloc[:m].reset_index(drop=True)
        vecs = vecs[:m]
    # normalize paths
    meta['image_path'] = meta['image_path'].astype(str).apply(lambda p: str(Path(p).resolve()))
    return meta, vecs


def l2_normalize_rows(X: np.ndarray):
    X = X.astype(np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def fps_greedy(E: np.ndarray, n_select: int, initial_index: int = 0):
    N = E.shape[0]
    if n_select >= N:
        return list(range(N))
    selected = [int(initial_index)]
    norms = np.sum(E * E, axis=1, keepdims=True)
    min_sq = np.full(N, np.inf)
    x0 = E[selected[0]]
    dots = E.dot(x0)
    min_sq = np.minimum(min_sq, (norms.flatten() + float(np.sum(x0 * x0)) - 2.0 * dots))
    for _ in range(1, n_select):
        nxt = int(np.argmax(min_sq))
        selected.append(nxt)
        xnew = E[nxt]
        dots = E.dot(xnew)
        sq = (norms.flatten() + float(np.sum(xnew * xnew)) - 2.0 * dots)
        sq[sq < 0] = 0.0
        min_sq = np.minimum(min_sq, sq)
    return selected


def resolve_shifted_image(meta_path_str: str):
    p = Path(meta_path_str)
    if p.is_file():
        return p.resolve(), ('val' if 'val' in str(p).lower() else 'train')
    p2 = (PROJECT_ROOT / meta_path_str)
    if p2.is_file():
        return p2.resolve(), ('val' if 'val' in str(p2).lower() else 'train')
    # search in preferred shifted source
    basename = p.name
    for root, split in ((SHIFTED_SRC_IMG_TRAIN, 'train'), (SHIFTED_SRC_IMG_VAL, 'val')):
        if root.exists():
            for cand in root.rglob(basename):
                if cand.is_file():
                    return cand.resolve(), split
    # fallback to outputs shifted experiment
    for root, split in ((FALLBACK_SHIFTED_ROOT / 'images' / 'train', 'train'), (FALLBACK_SHIFTED_ROOT / 'images' / 'val', 'val')):
        if root.exists():
            for cand in root.rglob(basename):
                if cand.is_file():
                    return cand.resolve(), split
    return None, None


def find_label_for_image(img_path: Path, split_hint: str = None):
    stem = img_path.stem
    # try structured mapping to preferred source
    try:
        if split_hint == 'train' or (SHIFTED_SRC_IMG_TRAIN in img_path.parents):
            rel = img_path.relative_to(SHIFTED_SRC_IMG_TRAIN)
            cand = (SHIFTED_SRC_LABEL_TRAIN if 'SHIFTED_SRC_LABEL_TRAIN' in globals() else SHIFTED_SRC_ROOT) / rel
    except Exception:
        pass
    # simple search in shifted label dirs (preferred then fallback)
    for ld in (SHIFTED_SRC_LABEL_TRAIN, SHIFTED_SRC_LABEL_VAL):
        if not ld.exists():
            continue
        candidate = ld / (stem + '.txt')
        if candidate.exists():
            return candidate.resolve()
        # recursive search
        for p in ld.rglob('*.txt'):
            if p.stem == stem:
                return p.resolve()
    # fallback to outputs labels
    for ld in (FALLBACK_SHIFTED_ROOT / 'labels' / 'train', FALLBACK_SHIFTED_ROOT / 'labels' / 'val'):
        if not ld.exists():
            continue
        for p in ld.rglob('*.txt'):
            if p.stem == stem:
                return p.resolve()
    return None


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)


def main():
    # Load embeddings
    try:
        meta, vecs = load_embeddings(EMB_META, EMB_VEC)
    except Exception as e:
        print('ERROR loading embeddings:', e)
        sys.exit(1)

    # Split 2024 and shifted 2025
    mask24 = meta['season'].astype(str) == '2024'
    mask25s = meta['dataset'].astype(str) == '2025_shifted'
    idx24 = np.where(mask24.values)[0]
    idx25s = np.where(mask25s.values)[0]
    N_shifted = len(idx25s)
    if N_shifted == 0:
        print('No shifted 2025 embeddings found; aborting')
        return

    # Normalize float
    vecs_f = np.asarray(vecs).astype(np.float64)

    # centroid and distances
    centroid_2024 = vecs_f[idx24].mean(axis=0)
    vecs25s = vecs_f[idx25s]
    dists25s = np.linalg.norm(vecs25s - centroid_2024[None, :], axis=1)
    mean_2025_shifted = float(np.mean(dists25s))
    dists24 = np.linalg.norm(vecs_f[idx24] - centroid_2024[None, :], axis=1)
    mean_2024_spread = float(np.mean(dists24)) if dists24.size > 0 else 1e-9

    ShiftScore = mean_2025_shifted / mean_2024_spread
    excess_shift = max(0.0, ShiftScore - 1.0)

    N_total = N_shifted
    N_raw = ALPHA * excess_shift * N_total
    N_min = max(1, int(math.ceil(N_MIN_FRAC * N_total)))
    N_max = max(1, int(math.floor(N_MAX_FRAC * N_total)))
    N_final = int(round(max(N_min, min(N_raw, N_max))))
    N_final = max(N_min, min(N_final, N_max))

    # Prepare output dirs
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    metrics = {
        'ShiftScore': float(ShiftScore),
        'excess_shift': float(excess_shift),
        'mean_2025_shifted': float(mean_2025_shifted),
        'mean_2024_spread': float(mean_2024_spread),
        'N_total_shifted': int(N_total),
        'N_raw': float(N_raw),
        'N_min': int(N_min),
        'N_max': int(N_max),
        'N_final': int(N_final),
    }
    write_json(OUT_ROOT / 'domain_shift_metrics.json', metrics)

    # Sort shifted by distance desc and candidate pool top 40%
    order_idx = np.argsort(-dists25s)
    sorted_meta25s = meta.iloc[idx25s].reset_index(drop=True).iloc[order_idx].reset_index(drop=True)
    sorted_dists = dists25s[order_idx]
    candidate_count = max(1, int(math.ceil(TOP_CANDIDATE_FRAC * N_total)))
    candidates_meta = sorted_meta25s.iloc[:candidate_count].reset_index(drop=True)

    # build embeddings matrix for candidates
    # need mapping from meta index to vecs index
    # idx25s[order_idx] gives original indices
    candidate_orig_indices = idx25s[order_idx][:candidate_count]
    E = vecs_f[candidate_orig_indices, :]
    E_n = l2_normalize_rows(E)

    if N_final > E_n.shape[0]:
        N_final = E_n.shape[0]

    selected_local = fps_greedy(E_n, N_final, initial_index=0)

    selected_rows = []
    copied = 0
    missing_labels = 0
    unresolved = 0

    # ensure dataset output dirs
    for d in (OUT_IMAGES_TRAIN, OUT_IMAGES_VAL, OUT_LABELS_TRAIN, OUT_LABELS_VAL):
        d.mkdir(parents=True, exist_ok=True)

    for order, local_idx in enumerate(selected_local, start=1):
        rec = candidates_meta.iloc[local_idx]
        meta_img_path = str(rec['image_path'])
        resolved, split = resolve_shifted_image(meta_img_path)
        if resolved is None:
            unresolved += 1
            warnings.warn(f'Could not resolve image {meta_img_path}; skipping')
            continue
        # copy image
        dst_img = (OUT_IMAGES_VAL if split == 'val' else OUT_IMAGES_TRAIN) / Path(resolved).name
        try:
            shutil.copy2(resolved, dst_img)
            copied += 1
        except Exception:
            warnings.warn(f'Failed to copy image {resolved} to {dst_img}')
        # copy label
        label = find_label_for_image(Path(resolved), split)
        if label is None or not label.exists():
            missing_labels += 1
            warnings.warn(f'Label missing for {resolved}')
        else:
            dst_lbl = (OUT_LABELS_VAL if split == 'val' else OUT_LABELS_TRAIN) / label.name
            try:
                shutil.copy2(label, dst_lbl)
            except Exception:
                warnings.warn(f'Failed to copy label {label} to {dst_lbl}')
        selected_rows.append({'image_path': str(resolved), 'distance_to_centroid': float(sorted_dists[local_idx]), 'selection_order': int(order)})

    # save outputs
    pd.DataFrame(selected_rows).to_csv(OUT_ROOT / 'selected_images.csv', index=False)
    write_json(OUT_ROOT / 'selection_summary.json', {'N_selected': len(selected_rows), 'copied': copied, 'missing_labels': missing_labels, 'unresolved': unresolved})

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
    with open(OUT_DATASET / 'data.yaml', 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(yaml_lines) + '\n')

    # Final print
    print('ShiftScore:', ShiftScore)
    print('N_total_shifted:', N_total)
    print('N_selected:', len(selected_rows))
    pct = 100.0 * len(selected_rows) / float(N_total) if N_total > 0 else 0.0
    print(f'Percentage selected: {pct:.2f}%')
    print('Images copied:', copied)
    print('Missing labels:', missing_labels)
    print('Unresolved meta images:', unresolved)


if __name__ == '__main__':
    main()

