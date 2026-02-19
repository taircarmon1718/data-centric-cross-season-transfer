#!/usr/bin/env python3
"""
select_uncertain_diverse_shifted.py

Uncertainty + diversity selection on shifted 2025 images only.
- Computes per-image uncertainty with YOLO (models/2024/all-ponds/weights/best.pt)
- Loads unified embeddings (scripts/shift_experiments/embeddings/all_embeddings.npy and all_meta.csv)
- Filters embeddings for dataset == '2025_shifted'
- Matches uncertainty by absolute image_path
- Keeps top 30% most uncertain images
- Runs k-center greedy selection on L2-normalized embeddings to select N images (configurable)
- Saves outputs under scripts/shift_experiments/active_selection/

Outputs:
- selected_images.csv (image_path, uncertainty_score, selected_rank)
- uncertainty_scores.csv (image_path, uncertainty_score)
- config.json

Deterministic (seed=0). Uses pathlib. No labels used in selection.
"""
from pathlib import Path
import argparse
import json
import math
import warnings
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

# Try importing ultralytics for YOLO inference
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# Config defaults
SEED = 0
DEFAULT_MODEL = Path("C:/Users/carmonta/Desktop/data-centric-cross-season-transfer/models/2024/all-ponds/weights/best.pt")
EMB_DIR = Path("C:/Users/carmonta/Desktop/data-centric-cross-season-transfer/scripts/shift_experiments/embeddings")
EMB_VEC = EMB_DIR / "all_embeddings.npy"
EMB_META = EMB_DIR / "all_meta.csv"

SHIFTED_IMG_DIRS = [
    Path("C:/Users/carmonta/Desktop/data-centric-cross-season-transfer/outputs/rep_analysis/core_set_selection/shifted_2025_experiment/shifted_2025_experiment/images/train"),
    Path("C:/Users/carmonta/Desktop/data-centric-cross-season-transfer/outputs/rep_analysis/core_set_selection/shifted_2025_experiment/shifted_2025_experiment/images/val"),
]

OUT_DIR = Path("C:/Users/carmonta/Desktop/data-centric-cross-season-transfer/scripts/shift_experiments/active_selection")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SELECTED_CSV = OUT_DIR / 'selected_images.csv'
UNC_CSV = OUT_DIR / 'uncertainty_scores.csv'
CONFIG_JSON = OUT_DIR / 'config.json'

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# Utilities

def collect_shifted_images():
    imgs = []
    seen = set()
    for d in SHIFTED_IMG_DIRS:
        if not d.exists():
            warnings.warn(f'Directory {d} not found — skipping')
            continue
        for p in sorted(d.rglob('*')):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                try:
                    rp = p.resolve()
                except Exception:
                    rp = p
                if str(rp) in seen:
                    continue
                seen.add(str(rp))
                imgs.append(rp)
    return imgs


def compute_uncertainty_for_images(image_paths, model_path, device=None, conf=0.001):
    if YOLO is None:
        raise RuntimeError('ultralytics YOLO not available; please install ultralytics')
    if not Path(model_path).exists():
        raise FileNotFoundError(f'Model weights not found: {model_path}')
    model = YOLO(str(model_path))
    if device is None:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model.predict accepts file paths
    records = []
    for p in tqdm(image_paths, desc='Running inference'):
        try:
            results = model.predict(str(p), conf=conf, device=device, verbose=False)
            res = results[0]
            # Try to extract confidences from boxes
            confs = None
            if hasattr(res, 'boxes') and hasattr(res.boxes, 'conf'):
                try:
                    confs = res.boxes.conf.cpu().numpy()
                except Exception:
                    try:
                        confs = np.array(res.boxes.conf)
                    except Exception:
                        confs = np.array([])
            if confs is None or confs.size == 0:
                uncertainty = 1.0
            else:
                mean_conf = float(np.mean(confs))
                uncertainty = float(1.0 - mean_conf)
        except Exception as e:
            # inference failed -> high uncertainty
            warnings.warn(f'Inference failed on {p}: {e}')
            uncertainty = 1.0
        records.append({'image_path': str(p.resolve()), 'uncertainty_score': float(uncertainty)})
    return pd.DataFrame.from_records(records)


def load_embeddings(meta_path, vec_path):
    if not Path(meta_path).exists() or not Path(vec_path).exists():
        raise FileNotFoundError('Embeddings or meta missing at provided paths')
    meta = pd.read_csv(meta_path)
    vecs = np.load(vec_path)
    if len(meta) != vecs.shape[0]:
        m = min(len(meta), vecs.shape[0])
        warnings.warn(f'Meta rows ({len(meta)}) != vec rows ({vecs.shape[0]}). Aligning to {m}.')
        meta = meta.iloc[:m].reset_index(drop=True)
        vecs = vecs[:m]
    # ensure absolute paths
    meta['image_path'] = meta['image_path'].astype(str).apply(lambda p: str(Path(p).resolve()))
    return meta, vecs


def l2_normalize_rows(X):
    X = X.astype(np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def k_center_greedy(emb, n_select, initial_idx=0):
    N = emb.shape[0]
    if n_select >= N:
        return list(range(N))
    selected = [int(initial_idx)]
    norms = np.sum(emb ** 2, axis=1, keepdims=True)
    min_sq = np.full(N, np.inf)
    x0 = emb[selected[0]]
    dots = emb.dot(x0)
    min_sq = np.minimum(min_sq, (norms.flatten() + float(np.sum(x0 * x0)) - 2.0 * dots))
    for _ in range(1, n_select):
        nxt = int(np.argmax(min_sq))
        selected.append(nxt)
        xnew = emb[nxt]
        dots = emb.dot(xnew)
        sq = (norms.flatten() + float(np.sum(xnew * xnew)) - 2.0 * dots)
        sq[sq < 0] = 0.0
        min_sq = np.minimum(min_sq, sq)
    return selected


def main():
    parser = argparse.ArgumentParser(description='Select uncertain + diverse shifted 2025 images')
    parser.add_argument('--model', type=str, default=str(DEFAULT_MODEL))
    parser.add_argument('--emb_meta', type=str, default=str(EMB_META))
    parser.add_argument('--emb_vec', type=str, default=str(EMB_VEC))
    parser.add_argument('--select_n', type=int, default=100, help='Number of images to select via FPS (default 100)')
    args = parser.parse_args()

    rng = np.random.RandomState(SEED)

    print('Collecting shifted images...')
    imgs = collect_shifted_images()
    print(f'Found {len(imgs)} shifted images')
    if len(imgs) == 0:
        print('No shifted images found; exiting')
        return

    # compute uncertainty
    print('Computing uncertainty scores...')
    unc_df = compute_uncertainty_for_images(imgs, args.model)
    unc_df.to_csv(UNC_CSV, index=False)

    # load embeddings
    print('Loading unified embeddings...')
    meta, vecs = load_embeddings(args.emb_meta, args.emb_vec)

    # filter embeddings for 2025_shifted
    mask = meta['dataset'].astype(str) == '2025_shifted'
    meta_shift = meta[mask].copy().reset_index(drop=True)
    vecs_shift = vecs[mask.values]
    print(f'Found {len(meta_shift)} shifted embeddings')

    # match by absolute image_path
    # ensure unc_df paths are absolute
    unc_df['image_path'] = unc_df['image_path'].astype(str).apply(lambda p: str(Path(p).resolve()))
    merged = pd.merge(unc_df, meta_shift, on='image_path', how='inner')
    if merged.empty:
        print('No matches between uncertainty results and shifted embeddings; exiting')
        return

    # keep top 30% most uncertain
    merged = merged.sort_values('uncertainty_score', ascending=False).reset_index(drop=True)
    top_k = max(1, int(math.ceil(0.30 * len(merged))))
    candidates = merged.iloc[:top_k].reset_index(drop=True)
    print(f'Candidates for FPS: {len(candidates)} (top 30%)')

    # build embeddings matrix for candidates
    # Need to get indices in meta_shift to fetch vecs_shift
    # meta_shift has same order as vecs_shift
    # create dictionary image_path->index
    idx_map = {str(p): i for i, p in enumerate(meta_shift['image_path'].astype(str))}
    cand_indices = [idx_map[str(p)] for p in candidates['image_path'].astype(str)]
    E = vecs_shift[cand_indices]
    E_n = l2_normalize_rows(E)

    n_select = args.select_n
    if n_select > E_n.shape[0]:
        n_select = E_n.shape[0]
    selected_local = k_center_greedy(E_n, n_select, initial_idx=0)

    # prepare outputs
    selected_rows = []
    for rank, local_idx in enumerate(selected_local, start=1):
        img_path = str(candidates.iloc[local_idx]['image_path'])
        uncert = float(candidates.iloc[local_idx]['uncertainty_score'])
        selected_rows.append({'image_path': img_path, 'uncertainty_score': uncert, 'selected_rank': rank})

    sel_df = pd.DataFrame(selected_rows)
    sel_df.to_csv(SELECTED_CSV, index=False)

    # save config
    cfg = {
        'seed': SEED,
        'model': args.model,
        'emb_meta': args.emb_meta,
        'emb_vec': args.emb_vec,
        'select_n': n_select,
        'candidates': len(candidates),
    }
    with open(CONFIG_JSON, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2)

    print('Saved uncertainty scores to', UNC_CSV)
    print('Saved selected images to', SELECTED_CSV)
    print('Saved config to', CONFIG_JSON)
    print('Selection complete. Selected count =', len(selected_rows))


if __name__ == '__main__':
    main()

