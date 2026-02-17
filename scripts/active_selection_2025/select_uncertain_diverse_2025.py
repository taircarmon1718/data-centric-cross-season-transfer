#!/usr/bin/env python3
"""
select_uncertain_diverse_2025.py

Uncertainty + diversity (k-center) selection for Season 2025 training images.

Usage:
    python scripts/active_selection_2025/select_uncertain_diverse_2025.py --k 100 --seed 0

Requirements:
- ultralytics (for model inference)
- numpy, pandas, tqdm

Behavior:
- Collect images from datasets/train_on_2025_all/images and datasets/train_on_2025_all/val/images
- Run YOLO (models/2024/all-ponds/weights/best.pt) inference to compute per-image uncertainty = 1 - mean_confidence
  (if no detections -> uncertainty = 1.0)
- Keep top 30% most uncertain images (configurable fraction)
- From that uncertain pool, load precomputed YOLO embeddings and run k-center greedy selection to pick K images
- Save outputs under scripts/active_selection_2025/

Important: does NOT recompute embeddings, does NOT use labels, deterministic via seed.
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

# Try importing ultralytics YOLO; if missing, the script will still be syntactically fine but will error at runtime
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# Constants / Default paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / 'models' / '2024' / 'all-ponds' / 'weights' / 'best.pt'
EMB_META = PROJECT_ROOT / 'scripts' / 'representation_analysis' / 'outputs_repreasentation' / 'rep_analysis' / 'embeddings_meta.csv'
EMB_VECT = PROJECT_ROOT / 'scripts' / 'representation_analysis' / 'outputs_repreasentation' / 'rep_analysis' / 'embeddings_vectors.npy'

CANDIDATE_DIRS = [
    PROJECT_ROOT / 'datasets' / 'train_on_2025_all' / 'images',
    PROJECT_ROOT / 'datasets' / 'train_on_2025_all' / 'val' / 'images',
]
OUT_DIR = PROJECT_ROOT / 'scripts' / 'active_selection_2025'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Utilities

def collect_2025_images():
    """Collects candidate image paths (both train and val folders) under 2025 dataset."""
    IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    imgs = []
    for d in CANDIDATE_DIRS:
        if not d.exists():
            continue
        for p in sorted(d.rglob('*')):
            if p.suffix.lower() in IMAGE_EXTS and p.is_file():
                imgs.append(p.resolve())
    # deduplicate
    imgs = list(dict.fromkeys(imgs))
    return imgs


def compute_uncertainty(image_paths, model_path=MODEL_PATH, device=None, conf_th=0.001):
    """Run model inference and compute uncertainty = 1 - mean_confidence per image.
    Returns DataFrame with columns: image_path, basename, uncertainty_score
    """
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
            # run prediction with low confidence threshold to collect detections
            results = model.predict(img_str, conf=conf_th, device=device, verbose=False)
            r = results[0]
            # extract confidences
            confs = None
            try:
                # r.boxes.conf may be a tensor or list
                if getattr(r, 'boxes', None) is not None and getattr(r.boxes, 'conf', None) is not None:
                    confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes.conf, 'cpu') else np.array(r.boxes.conf)
                elif getattr(r, 'boxes', None) is not None and getattr(r.boxes, 'xyxy', None) is not None:
                    # fallback: no conf available
                    confs = np.array([])
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
            # on inference failure treat as high uncertainty but log
            print(f'Warning: inference failed for {img_str}: {e}')
            uncertainty = 1.0

        records.append({'image_path': str(p), 'basename': p.name, 'uncertainty_score': uncertainty})

    df = pd.DataFrame.from_records(records)
    return df


def load_embeddings(meta_path=EMB_META, vec_path=EMB_VECT):
    """Load embeddings and meta, align by min length, return (meta_df, vectors)
    """
    if not meta_path.exists() or not vec_path.exists():
        raise FileNotFoundError(f'Embedding files not found: {meta_path} or {vec_path}')
    # read meta
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
    # normalize column names and create basename
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
    # add basename for matching
    meta['basename'] = meta['image_path'].astype(str).apply(lambda p: Path(str(p).replace('\\','/')).name)
    return meta, vecs


def k_center_greedy(embeddings, initial_index=0, k=100):
    """Vectorized k-center greedy selection.
    embeddings: numpy array (N, D) L2-normalized or not (Euclidean used)
    initial_index: index in embeddings to start with
    returns list of selected indices in original embeddings array order
    """
    N = embeddings.shape[0]
    if k >= N:
        return list(range(N))

    selected = [int(initial_index)]
    # compute distances to selected set
    # use squared distances for speed
    emb = embeddings
    # precompute norms
    norms = np.sum(emb ** 2, axis=1, keepdims=True)  # (N,1)
    # compute initial min dists (squared)
    # d^2(i, S) = min_s (||xi - xs||^2) = min_s (norms[i] + norms[s] - 2 xi.xs)
    # initialize min_sq_dists with +inf
    min_sq_dists = np.full(N, np.inf)
    # update with initial
    x0 = emb[selected[0]]
    dots = emb.dot(x0)
    min_sq_dists = np.minimum(min_sq_dists, (norms.flatten() + np.sum(x0 ** 2) - 2.0 * dots))

    for _ in range(1, k):
        # pick farthest point (max min_sq_dists)
        next_idx = int(np.argmax(min_sq_dists))
        selected.append(next_idx)
        xnew = emb[next_idx]
        dots = emb.dot(xnew)
        sq_dists_to_new = (norms.flatten() + np.sum(xnew ** 2) - 2.0 * dots)
        # numerical safety
        sq_dists_to_new[sq_dists_to_new < 0] = 0.0
        # update min distances
        min_sq_dists = np.minimum(min_sq_dists, sq_dists_to_new)
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description='Uncertainty + diversity selection for 2025')
    parser.add_argument('--k', type=int, default=100, help='Number of images to select')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for determinism')
    parser.add_argument('--uncertain_frac', type=float, default=0.3, help='Fraction of most uncertain to keep before diversity')
    parser.add_argument('--model', type=str, default=str(MODEL_PATH))
    parser.add_argument('--emb_meta', type=str, default=str(EMB_META))
    parser.add_argument('--emb_vec', type=str, default=str(EMB_VECT))
    args = parser.parse_args(argv)

    np.random.seed(args.seed)

    # Step 0: collect candidate images
    imgs = collect_2025_images()
    total_pool = len(imgs)
    print(f'Total 2025 pool images found: {total_pool}')
    if total_pool == 0:
        print('No 2025 images found; exiting.')
        return

    # Step 1: compute uncertainty scores
    print('Computing uncertainty scores (running model inference) ...')
    try:
        df_unc = compute_uncertainty(imgs, model_path=args.model)
    except Exception as e:
        print('ERROR during uncertainty computation:', e)
        return

    # save uncertainty scores
    unc_csv = OUT_DIR / 'uncertainty_scores.csv'
    df_unc.to_csv(unc_csv, index=False)
    print('Saved uncertainty scores to', unc_csv)

    # Step 2: filter top uncertain_frac
    df_sorted = df_unc.sort_values('uncertainty_score', ascending=False).reset_index(drop=True)
    uncertain_pool_size = max(1, int(math.ceil(len(df_sorted) * args.uncertain_frac)))
    df_uncertain = df_sorted.iloc[:uncertain_pool_size].reset_index(drop=True)
    print(f'Uncertain pool size (top {args.uncertain_frac*100:.1f}%): {len(df_uncertain)}')

    # Step 3: diversity selection using embeddings
    print('Loading embeddings ...')
    try:
        meta, vecs = load_embeddings(Path(args.emb_meta), Path(args.emb_vec))
    except Exception as e:
        print('ERROR loading embeddings:', e)
        return

    # filter meta to season==2025 and dataset_type==train
    meta_filtered_mask = (meta['season'].astype(str) == '2025') & (meta['dataset_type'].astype(str) == 'train')
    meta2025 = meta[meta_filtered_mask].copy().reset_index(drop=True)
    vecs2025 = vecs[meta_filtered_mask.values]
    print(f'Available embeddings for 2025 train: {len(meta2025)}')

    # map uncertain images to embedding indices by basename
    basename_to_indices = {}
    for idx, row in meta2025.reset_index().iterrows():
        basename_to_indices.setdefault(str(row['basename']), []).append(idx)

    emb_indices = []
    emb_paths = []
    emb_uncert = []
    skipped_no_emb = 0
    for _, r in df_uncertain.iterrows():
        b = r['basename']
        if b in basename_to_indices and len(basename_to_indices[b]) > 0:
            i = basename_to_indices[b][0]  # deterministic pick first
            emb_indices.append(i)
            emb_paths.append(meta2025.iloc[i]['image_path'])
            emb_uncert.append(r['uncertainty_score'])
        else:
            skipped_no_emb += 1
    if skipped_no_emb > 0:
        print(f'Warning: {skipped_no_emb} uncertain images had no matching embedding and were skipped')

    if len(emb_indices) == 0:
        print('No uncertain images matched to embeddings; exiting.')
        return

    # construct embeddings matrix for selection
    E = np.asarray(vecs2025)[emb_indices]
    # L2 normalize rows
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    E_n = E / norms

    # choose initial index: the most uncertain among eligible (index 0 within E_n)
    # find argmax uncertainty
    initial_idx = int(np.argmax(np.asarray(emb_uncert)))
    K = args.k
    if K > E_n.shape[0]:
        K = E_n.shape[0]
    print(f'Selecting K={K} images from uncertain pool of size {E_n.shape[0]}')

    selected_local_indices = k_center_greedy(E_n, initial_index=initial_idx, k=K)

    # map back to global image paths and uncertainties
    selected_records = []
    for rank, local_idx in enumerate(selected_local_indices, start=1):
        img_path = emb_paths[local_idx]
        uncert = float(emb_uncert[local_idx])
        selected_records.append({'image_path': img_path, 'uncertainty_score': uncert, 'selected_rank': rank})

    sel_df = pd.DataFrame.from_records(selected_records)
    sel_csv = OUT_DIR / 'selected_images.csv'
    sel_df.to_csv(sel_csv, index=False)

    # save config
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

    # Print summary stats
    avg_uncert_selected = sel_df['uncertainty_score'].mean() if not sel_df.empty else float('nan')
    print('Selection summary:')
    print(f'  Total 2025 pool size: {total_pool}')
    print(f'  Uncertain pool size: {len(df_uncertain)}')
    print(f'  Final selected count: {len(sel_df)}')
    print(f'  Average uncertainty of selected set: {avg_uncert_selected:.4f}')
    print('Saved selected images to', sel_csv)


if __name__ == '__main__':
    main()

