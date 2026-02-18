#!/usr/bin/env python3
"""
select_uncertain_diverse_2024.py

Uncertainty + diversity (k-center) selection for Season 2024 training images.

This mirrors the 2025 pipeline but:
- Collects images from datasets/train_on_all (train + val)
- Uses the 2025-trained model for uncertainty: models/2025/all-ponds/weights/best.pt
- Loads precomputed embeddings from scripts/representation_analysis/outputs_representation_2024/
  files: embeddings_meta_2024.csv, embeddings_vectors_2024.npy
- Saves outputs to scripts/active_selection_2024/

Usage:
    python scripts/active_selection_2024/select_uncertain_diverse_2024.py --k 100 --seed 0

Deterministic and does NOT recompute embeddings or use labels.
"""

from pathlib import Path
import argparse
import json
import time
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Use 2025-trained model to score 2024 images
MODEL_PATH = PROJECT_ROOT / 'models' / '2025' / 'all-ponds' / 'weights' / 'best.pt'
# Embeddings produced for 2024
EMB_DIR_2024 = PROJECT_ROOT / 'scripts' / 'representation_analysis' / 'outputs_representation_2024'
EMB_META = EMB_DIR_2024 / 'embeddings_meta_2024.csv'
EMB_VECT = EMB_DIR_2024 / 'embeddings_vectors_2024.npy'

CANDIDATE_DIRS = [
    PROJECT_ROOT / 'datasets' / 'train_on_all' / 'images',
    PROJECT_ROOT / 'datasets' / 'train_on_all' / 'val' / 'images',
]
OUT_DIR = PROJECT_ROOT / 'scripts' / 'active_selection_2024'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Functions

def collect_2024_images():
    IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    imgs = []
    for d in CANDIDATE_DIRS:
        if not d.exists():
            continue
        for p in sorted(d.rglob('*')):
            if p.suffix.lower() in IMAGE_EXTS and p.is_file():
                imgs.append(p.resolve())
    imgs = list(dict.fromkeys(imgs))
    return imgs


def compute_uncertainty(image_paths, model_path=MODEL_PATH, device=None, conf_th=0.001):
    if device is None:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    if YOLO is None:
        raise RuntimeError('ultralytics.YOLO is required for inference but not available')
    if not Path(model_path).exists():
        raise FileNotFoundError(f'Model weights not found at {model_path}')
    model = YOLO(str(model_path))
    records = []
    for p in tqdm(image_paths, desc='Inference for uncertainty'):
        img_str = str(p)
        try:
            results = model.predict(img_str, conf=conf_th, device=device, verbose=False)
            r = results[0]
            confs = None
            try:
                if getattr(r, 'boxes', None) is not None and getattr(r.boxes, 'conf', None) is not None:
                    confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes.conf, 'cpu') else np.array(r.boxes.conf)
                else:
                    confs = np.array([])
            except Exception:
                confs = np.array([])
            if confs.size == 0:
                uncertainty = 1.0
            else:
                mean_conf = float(np.mean(confs))
                uncertainty = float(1.0 - mean_conf)
        except Exception as e:
            print(f'Warning: inference failed for {img_str}: {e}')
            uncertainty = 1.0
        records.append({'image_path': str(p), 'basename': p.name, 'uncertainty_score': uncertainty})
    df = pd.DataFrame.from_records(records)
    return df


def load_embeddings(meta_path=EMB_META, vec_path=EMB_VECT):
    if not meta_path.exists() or not vec_path.exists():
        raise FileNotFoundError(f'Embedding files not found: {meta_path} or {vec_path}')
    try:
        meta = pd.read_csv(meta_path)
    except Exception:
        meta = pd.read_csv(meta_path, header=None)
    vecs = np.load(vec_path)
    if len(meta) != vecs.shape[0]:
        m = min(len(meta), vecs.shape[0])
        warnings.warn(f'Meta rows ({len(meta)}) != vectors rows ({vecs.shape[0]}). Aligning by min length={m}.')
        meta = meta.iloc[:m].reset_index(drop=True)
        vecs = vecs[:m]
    meta.columns = [c.strip() for c in meta.columns]
    if 'image_path' not in meta.columns:
        meta = meta.rename(columns={meta.columns[0]: 'image_path'})
    if 'season' not in meta.columns:
        if len(meta.columns) > 1:
            meta = meta.rename(columns={meta.columns[1]: 'season'})
        else:
            meta['season'] = ''
    if 'dataset_type' not in meta.columns:
        if 'split' in meta.columns:
            meta = meta.rename(columns={'split': 'dataset_type'})
        else:
            meta['dataset_type'] = ''
    meta['basename'] = meta['image_path'].astype(str).apply(lambda p: Path(str(p).replace('\\','/')).name)
    return meta, vecs


def k_center_greedy(embeddings, initial_index=0, k=100):
    N = embeddings.shape[0]
    if k >= N:
        return list(range(N))
    selected = [int(initial_index)]
    emb = embeddings
    norms = np.sum(emb ** 2, axis=1, keepdims=True)
    min_sq_dists = np.full(N, np.inf)
    x0 = emb[selected[0]]
    dots = emb.dot(x0)
    min_sq_dists = np.minimum(min_sq_dists, (norms.flatten() + np.sum(x0 ** 2) - 2.0 * dots))
    for _ in range(1, k):
        next_idx = int(np.argmax(min_sq_dists))
        selected.append(next_idx)
        xnew = emb[next_idx]
        dots = emb.dot(xnew)
        sq_dists_to_new = (norms.flatten() + np.sum(xnew ** 2) - 2.0 * dots)
        sq_dists_to_new[sq_dists_to_new < 0] = 0.0
        min_sq_dists = np.minimum(min_sq_dists, sq_dists_to_new)
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description='Uncertainty + diversity selection for 2024')
    parser.add_argument('--k', type=int, default=100, help='Number of images to select')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for determinism')
    parser.add_argument('--uncertain_frac', type=float, default=0.3, help='Fraction of most uncertain to keep before diversity')
    parser.add_argument('--model', type=str, default=str(MODEL_PATH))
    parser.add_argument('--emb_meta', type=str, default=str(EMB_META))
    parser.add_argument('--emb_vec', type=str, default=str(EMB_VECT))
    args = parser.parse_args(argv)
    np.random.seed(args.seed)
    imgs = collect_2024_images()
    total_pool = len(imgs)
    print(f'Total 2024 pool images found: {total_pool}')
    if total_pool == 0:
        print('No 2024 images found; exiting.')
        return
    print('Computing uncertainty scores (running model inference) ...')
    try:
        df_unc = compute_uncertainty(imgs, model_path=args.model)
    except Exception as e:
        print('ERROR during uncertainty computation:', e)
        return
    unc_csv = OUT_DIR / 'uncertainty_scores_2024.csv'
    df_unc.to_csv(unc_csv, index=False)
    print('Saved uncertainty scores to', unc_csv)
    df_sorted = df_unc.sort_values('uncertainty_score', ascending=False).reset_index(drop=True)
    uncertain_pool_size = max(1, int(math.ceil(len(df_sorted) * args.uncertain_frac)))
    df_uncertain = df_sorted.iloc[:uncertain_pool_size].reset_index(drop=True)
    print(f'Uncertain pool size (top {args.uncertain_frac*100:.1f}%): {len(df_uncertain)}')
    print('Loading embeddings ...')
    try:
        meta, vecs = load_embeddings(Path(args.emb_meta), Path(args.emb_vec))
    except Exception as e:
        print('ERROR loading embeddings:', e)
        return
    # filter meta to season==2024 and dataset_type==train
    meta_filtered_mask = (meta['season'].astype(str) == '2024') & (meta['dataset_type'].astype(str) == 'train')
    meta2024 = meta[meta_filtered_mask].copy().reset_index(drop=True)
    vecs2024 = vecs[meta_filtered_mask.values]
    print(f'Available embeddings for 2024 train: {len(meta2024)}')
    basename_to_indices = {}
    for idx, row in meta2024.reset_index().iterrows():
        basename_to_indices.setdefault(str(row['basename']), []).append(idx)
    emb_indices = []
    emb_paths = []
    emb_uncert = []
    skipped_no_emb = 0
    for _, r in df_uncertain.iterrows():
        b = r['basename']
        if b in basename_to_indices and len(basename_to_indices[b]) > 0:
            i = basename_to_indices[b][0]
            emb_indices.append(i)
            emb_paths.append(meta2024.iloc[i]['image_path'])
            emb_uncert.append(r['uncertainty_score'])
        else:
            skipped_no_emb += 1
    if skipped_no_emb > 0:
        print(f'Warning: {skipped_no_emb} uncertain images had no matching embedding and were skipped')
    if len(emb_indices) == 0:
        print('No uncertain images matched to embeddings; exiting.')
        return
    E = np.asarray(vecs2024)[emb_indices]
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    E_n = E / norms
    initial_idx = int(np.argmax(np.asarray(emb_uncert)))
    K = args.k
    if K > E_n.shape[0]:
        K = E_n.shape[0]
    print(f'Selecting K={K} images from uncertain pool of size {E_n.shape[0]}')
    selected_local_indices = k_center_greedy(E_n, initial_index=initial_idx, k=K)
    selected_records = []
    for rank, local_idx in enumerate(selected_local_indices, start=1):
        img_path = emb_paths[local_idx]
        uncert = float(emb_uncert[local_idx])
        selected_records.append({'image_path': img_path, 'uncertainty_score': uncert, 'selected_rank': rank})
    sel_df = pd.DataFrame.from_records(selected_records)
    sel_csv = OUT_DIR / 'selected_images_2024.csv'
    sel_df.to_csv(sel_csv, index=False)
    cfg = {
        'k': args.k,
        'seed': args.seed,
        'uncertain_frac': args.uncertain_frac,
        'model_path': args.model,
        'emb_meta': args.emb_meta,
        'emb_vec': args.emb_vec,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(OUT_DIR / 'config.json', 'w') as f:
        json.dump(cfg, f, indent=2)
    avg_uncert_selected = sel_df['uncertainty_score'].mean() if not sel_df.empty else float('nan')
    print('Selection summary:')
    print(f'  Total 2024 pool size: {total_pool}')
    print(f'  Uncertain pool size: {len(df_uncertain)}')
    print(f'  Final selected count: {len(sel_df)}')
    print(f'  Average uncertainty of selected set: {avg_uncert_selected:.4f}')
    print('Saved selected images to', sel_csv)


if __name__ == '__main__':
    main()

