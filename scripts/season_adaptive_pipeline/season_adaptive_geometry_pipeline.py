#!/usr/bin/env python3
"""
season_adaptive_geometry_pipeline.py

Unified bidirectional season-adaptive selection pipeline.

Parts:
- PART 1: Setup paths, models, outputs
- PART 2: Pseudo-label generation (model predictions -> keypoints -> geometry)
- PART 3: Geometry-FPS selection on normalized (total_len, ratio, angle)
- PART 4: Build final dataset copying original images + true labels
- PART 5: Reporting

Requirements: pathlib, shutil, math, numpy, pandas, torch, ultralytics
Deterministic behavior (seed=0)
"""
from pathlib import Path
import math
import shutil
import json
import warnings

import numpy as np
import pandas as pd
import torch

# Try import ultralytics
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# ---------------------- Configuration ----------------------
SEED = 0
np.random.seed(SEED)

ALPHA = 0.2

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEASON_A = '2024'
SEASON_B = '2025'

DATA_2024 = PROJECT_ROOT / 'datasets' / 'train_on_2024_all'
DATA_2025 = PROJECT_ROOT / 'datasets' / 'train_on_2025_all'

MODEL_2024 = PROJECT_ROOT / 'models' / '2024' / 'all-ponds' / 'weights' / 'best.pt'
MODEL_2025 = PROJECT_ROOT / 'models' / '2025' / 'all-ponds' / 'weights' / 'best.pt'

OUT_ROOT = PROJECT_ROOT / 'datasets' / 'season_adaptive_geometry'
OUT_ROOT.mkdir(parents=True, exist_ok=True)
OUT_24_TO_25 = OUT_ROOT / '2024_to_2025'
OUT_25_TO_24 = OUT_ROOT / '2025_to_2024'

SUBSET_PERCENT = 0.01
MIN_IMAGES = 10
TOP_CANDIDATE_FRAC = 0.40
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------------- Helpers ----------------------

def collect_images_for_season(season):
    if season == '2024':
        root = DATA_2024
    else:
        root = DATA_2025
    imgs = []
    for sub in (root / 'images', root / 'val' / 'images'):
        if not sub.exists():
            continue
        for p in sorted(sub.rglob('*')):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                imgs.append(p.resolve())
    # deduplicate
    imgs = list(dict.fromkeys(imgs))
    imgs.sort(key=lambda p: str(p))
    return imgs


def load_model(weights_path: Path):
    if YOLO is None:
        raise RuntimeError('ultralytics is required but not installed')
    if not weights_path.exists():
        raise FileNotFoundError(f'Model weights not found: {weights_path}')
    model = YOLO(str(weights_path))
    return model


def extract_keypoints_from_result(res):
    """Attempt to extract keypoints (x,y) from a single result object.
    Return list of keypoints for first detected object or None.
    """
    # multiple possible shapes depending on model
    # try attributes in order
    kp = None
    try:
        # ultralytics v8 pose uses res.keypoints
        if hasattr(res, 'keypoints') and res.keypoints is not None:
            kp_attr = res.keypoints
            # kp_attr may have .xy or be a tensor
            try:
                arr = kp_attr.xy if hasattr(kp_attr, 'xy') else kp_attr.cpu().numpy()
            except Exception:
                try:
                    arr = kp_attr.numpy()
                except Exception:
                    arr = None
            if arr is not None:
                # arr shape may be (N, K, 2)
                if arr.ndim == 3 and arr.shape[0] >= 1:
                    first = arr[0]
                    # convert to list of tuples
                    kp = [(float(v[0]), float(v[1])) for v in first]
                    return kp
        # try boxes.xy or masks - fallback none
    except Exception:
        kp = None
    # try res.keypoints.data or res.keypoints.xy
    try:
        if hasattr(res, 'keypoints'):
            obj = res.keypoints
            if hasattr(obj, 'data'):
                a = obj.data
                if isinstance(a, (list, tuple)):
                    a = a[0]
                if hasattr(a, 'numpy'):
                    arr = a.numpy()
                    if arr.ndim >= 2:
                        first = arr[0]
                        kp = [(float(v[0]), float(v[1])) for v in first]
                        return kp
    except Exception:
        pass
    # if not found, try annotations in res.boxes or res.masks - not reliable
    return None


def run_inference_and_compute_geometry(model, image_paths):
    """Run model on list of image Paths and extract geometric metrics.
    Returns list of dicts: image_path, total_len, carapace_len, ratio, angle
    """
    records = []
    device = DEVICE
    for p in image_paths:
        try:
            results = model.predict(str(p), device=device, verbose=False)
            res = results[0]
            kps = extract_keypoints_from_result(res)
            if kps is None or len(kps) < 4:
                # skip if no keypoints
                continue
            # detect if normalized (coords between 0 and 1)
            arr = np.array(kps).astype(float)
            if arr.max() <= 1.5:
                # need image size to convert to pixels
                from PIL import Image
                try:
                    with Image.open(p) as im:
                        W, H = im.size
                except Exception:
                    W, H = None, None
                if W is not None:
                    abs_kps = [(float(x * W), float(y * H)) for (x, y) in kps]
                else:
                    continue
            else:
                abs_kps = [(float(x), float(y)) for (x, y) in kps]
            # compute geometry
            car_len = math.hypot(abs_kps[1][0] - abs_kps[0][0], abs_kps[1][1] - abs_kps[0][1])
            tot_len = math.hypot(abs_kps[3][0] - abs_kps[2][0], abs_kps[3][1] - abs_kps[2][1])
            if not math.isfinite(car_len) or not math.isfinite(tot_len) or tot_len <= 0:
                continue
            ratio = car_len / tot_len
            dx = abs_kps[3][0] - abs_kps[2][0]
            dy = abs_kps[3][1] - abs_kps[2][1]
            angle = math.degrees(math.atan2(dy, dx))
            if angle >= 180.0:
                angle -= 360.0
            if angle < -180.0:
                angle += 360.0
            records.append({'image_path': str(p.resolve()), 'total_len': float(tot_len), 'carapace_len': float(car_len), 'ratio': float(ratio), 'angle': float(angle)})
        except Exception:
            # skip on any inference error
            continue
    return records


def normalize_features(df, keys):
    norms = {}
    for k in keys:
        arr = df[k].to_numpy(dtype=np.float64)
        mn = np.nanmin(arr) if arr.size > 0 else 0.0
        mx = np.nanmax(arr) if arr.size > 0 else 1.0
        if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
            norms[k] = (np.zeros_like(arr, dtype=np.float64), mn, mx)
        else:
            norms[k] = ((arr - mn) / (mx - mn), mn, mx)
    # return normalized matrix in order keys
    X = np.vstack([norms[k][0] for k in keys]).T
    return X, norms


def fps_on_features(X, n_select):
    # choose initial as farthest from centroid
    centroid = X.mean(axis=0)
    d = np.linalg.norm(X - centroid[None, :], axis=1)
    init = int(np.argmax(d))
    return fps_greedy_generic(X, n_select, initial_idx=init)


def fps_greedy_generic(X, n_select, initial_idx=0):
    M = X.shape[0]
    if n_select >= M:
        return list(range(M))
    selected = [int(initial_idx)]
    # compute squared distances
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


def copy_selected_to_dataset(selected_records, out_root):
    # deterministic 80/20 split
    n = len(selected_records)
    indices = np.arange(n)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(indices)
    n_train = int(math.floor(0.8 * n))
    train_set = set(perm[:n_train].tolist())

    imgs_train = out_root / 'images' / 'train'
    imgs_val = out_root / 'images' / 'val'
    lbls_train = out_root / 'labels' / 'train'
    lbls_val = out_root / 'labels' / 'val'
    for d in (imgs_train, imgs_val, lbls_train, lbls_val):
        d.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing_labels = 0
    for i, rec in enumerate(selected_records):
        src_img = Path(rec['image_path'])
        # find original label in season folders
        # Try same-stem under season folders
        stem = src_img.stem
        # Attempt to locate label
        label_found = None
        for root in (DATA_2024, DATA_2025):
            for lbl_root in (root / 'labels', root / 'val' / 'labels'):
                if lbl_root.exists():
                    candidate = lbl_root / (stem + '.txt')
                    if candidate.exists():
                        label_found = candidate
                        break
            if label_found is not None:
                break
        split = 'train' if i in train_set else 'val'
        dst_img = (imgs_train if split == 'train' else imgs_val) / src_img.name
        try:
            shutil.copy2(src_img, dst_img)
            copied += 1
        except Exception:
            pass
        if label_found is not None and label_found.exists():
            dst_lbl = (lbls_train if split == 'train' else lbls_val) / label_found.name
            try:
                shutil.copy2(label_found, dst_lbl)
            except Exception:
                missing_labels += 1
        else:
            missing_labels += 1
    # write data.yaml
    train_abs = str((out_root / 'images' / 'train').resolve())
    val_abs = str((out_root / 'images' / 'val').resolve())
    yaml_lines = [
        'nc: 1',
        "names: ['prawn']",
        'kpt_shape: [4, 3]',
        'flip_idx: [0, 1, 2, 3]',
        f'train: {train_abs}',
        f'val: {val_abs}',
    ]
    with open(out_root / 'data.yaml', 'w', encoding='utf-8') as f:
        f.write('\n'.join(yaml_lines) + '\n')
    return copied, missing_labels

# ---------------- Main pipeline ----------------

def process_direction(model_path, source_season, target_season, out_dir):
    print(f'Processing direction {source_season} -> {target_season}')
    # load model
    model = load_model(model_path)
    # collect target images (we will run inference on target season images)
    if target_season == '2025':
        imgs = collect_images_for_season('2025')
    else:
        imgs = collect_images_for_season('2024')
    print('Images to run inference on:', len(imgs))
    # run inference and compute geometry
    recs = run_inference_and_compute_geometry(model, imgs)
    if len(recs) == 0:
        print('No geometries extracted; skipping direction')
        return
    df = pd.DataFrame(recs)
    # compute normalized features
    X, norms = normalize_features(df, ['total_len', 'ratio', 'angle'])
    # candidate pool: top 40% most shifted (distance to centroid)
    centroid = np.mean(X, axis=0)
    d2c = np.linalg.norm(X - centroid[None, :], axis=1)
    order = np.argsort(-d2c)
    candidate_count = max(1, int(math.ceil(TOP_CANDIDATE_FRAC * len(df))))
    candidates_idx = order[:candidate_count]
    X_cand = X[candidates_idx]
    # select subset size
    N_total = len(df)
    # compute ShiftScore based on target vs source centroid distances
    # source centroid: compute using source season embeddings? here approximate using inference outputs
    # For simplicity compute mean distance in df (target) and use spread of target as denominator
    mean_target = float(np.mean(d2c))
    spread = float(np.mean(np.linalg.norm(X - centroid[None, :], axis=1)))
    ShiftScore = mean_target / (spread if spread > 0 else 1.0)
    excess_shift = max(0.0, ShiftScore - 1.0)
    N_raw = ALPHA * excess_shift * N_total if 'ALPHA' in globals() else 0.0
    N_min = max(1, int(math.ceil(0.01 * N_total)))
    N_max = max(1, int(math.floor(0.5 * N_total)))
    N_final = int(round(max(N_min, min(N_raw, N_max))))
    if N_final < MIN_IMAGES:
        N_final = MIN_IMAGES
    if N_final > X_cand.shape[0]:
        N_final = X_cand.shape[0]
    # FPS on candidates
    selected_local = fps_greedy_generic(X_cand, N_final, initial_idx=0)
    selected_indices = [candidates_idx[i] for i in selected_local]
    selected_records = df.iloc[selected_indices].to_dict(orient='records')
    # build dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    copied, missing = copy_selected_to_dataset(selected_records, out_dir)
    # reporting
    full_var = np.nanvar(df[['total_len', 'ratio', 'angle']].to_numpy(dtype=np.float64), axis=0)
    subset_var = np.nanvar(df.iloc[selected_indices][['total_len', 'ratio', 'angle']].to_numpy(dtype=np.float64), axis=0)
    min_cov = X[selected_indices].min(axis=0)
    max_cov = X[selected_indices].max(axis=0)
    print(f'Direction {source_season}->{target_season}: N_total={N_total}, N_selected={len(selected_indices)}')
    print('Full var:', full_var.tolist())
    print('Subset var:', subset_var.tolist())
    print('Coverage min:', min_cov.tolist(), 'max:', max_cov.tolist())
    print('Images copied:', copied, 'missing labels:', missing)
    # save records
    out_root = out_dir
    pd.DataFrame(selected_records).to_csv(out_root / 'selected_images.csv', index=False)
    with open(out_root / 'selection_summary.json', 'w', encoding='utf-8') as f:
        json.dump({'N_total': int(N_total), 'N_selected': int(len(selected_indices)), 'copied': int(copied), 'missing_labels': int(missing)}, f, indent=2)


def main():
    # process 2024 -> 2025
    process_direction(MODEL_2024, '2024', '2025', OUT_24_TO_25)
    # process 2025 -> 2024
    process_direction(MODEL_2025, '2025', '2024', OUT_25_TO_24)
    print('\nAll done.')


if __name__ == '__main__':
    main()
