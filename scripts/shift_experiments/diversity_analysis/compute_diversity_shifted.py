#!/usr/bin/env python3
"""
compute_diversity_shifted.py

Diversity analysis for shifted core-set experiments.

Scans datasets/shifted_2025_all_k/kXX for k sets, matches images to embeddings in
scripts/shift_experiments/embeddings/all_embeddings.npy and all_meta.csv (dataset=='2025_shifted'),
computes diversity metrics, saves CSV, JSON, and plots.

Requirements: pathlib, numpy, pandas, matplotlib, tqdm
Deterministic seed=0
"""
from pathlib import Path
import math
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Config
SEED = 0
np.random.seed(SEED)

PROJECT_ROOT = Path(r"C:\Users\carmonta\Desktop\data-centric-cross-season-transfer")
K_LIST = [1, 2, 5, 10, 20, 50]
K_STRS = [f'k{int(k):02d}' for k in K_LIST]

SHIFTED_DATASETS_ROOT = PROJECT_ROOT / 'datasets' / 'shifted_2025_all_k'
EMB_DIR = PROJECT_ROOT / 'scripts' / 'shift_experiments' / 'embeddings'
EMB_VEC = EMB_DIR / 'all_embeddings.npy'
EMB_META = EMB_DIR / 'all_meta.csv'
OUT_DIR = PROJECT_ROOT / 'scripts' / 'shift_experiments' / 'diversity_analysis'
PLOTS_DIR = OUT_DIR / 'plots'
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DIVERSITY_CSV = OUT_DIR / 'diversity_results.csv'
SUMMARY_JSON = OUT_DIR / 'diversity_summary.json'

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# Helpers

def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def pairwise_euclidean_stats(X: np.ndarray):
    """Compute pairwise Euclidean distances stats (upper triangle). Returns (avg, min, max).
       X shape (N,D).
    """
    N = X.shape[0]
    if N < 2:
        return float('nan'), float('nan'), float('nan')
    # use efficient computation: ||a-b||^2 = |a|^2 + |b|^2 - 2 a.b
    norms = np.sum(X * X, axis=1, keepdims=True)  # (N,1)
    dots = X @ X.T  # (N,N)
    sq = norms + norms.T - 2.0 * dots
    sq[sq < 0] = 0.0
    d = np.sqrt(sq)
    # extract upper triangle
    iu = np.triu_indices(N, k=1)
    vals = d[iu]
    return float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))


def cov_trace_and_det(X: np.ndarray):
    if X.shape[0] < 2:
        return float('nan'), float('nan')
    cov = np.cov(X, rowvar=False)
    trace = float(np.trace(cov))
    try:
        det = float(np.linalg.det(cov))
    except Exception:
        det = float('nan')
    return trace, det


def mean_std_centroid_dist(X: np.ndarray):
    if X.shape[0] == 0:
        return float('nan'), float('nan')
    centroid = np.mean(X, axis=0)
    dists = np.linalg.norm(X - centroid[None, :], axis=1)
    return float(np.mean(dists)), float(np.std(dists))


def collect_dataset_image_paths(k_str: str):
    root = SHIFTED_DATASETS_ROOT / k_str
    imgs = []
    for sub in (root / 'train' / 'images', root / 'val' / 'images'):
        if not sub.exists():
            continue
        for p in sorted(sub.rglob('*')):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                try:
                    imgs.append(str(p.resolve()))
                except Exception:
                    imgs.append(str(p))
    # deduplicate and sort deterministically
    imgs = sorted(list(dict.fromkeys(imgs)))
    return imgs


def main():
    # load embeddings meta and vectors
    if not EMB_META.exists() or not EMB_VEC.exists():
        print('Embedding files missing under', EMB_DIR)
        return
    meta = pd.read_csv(EMB_META)
    vecs = np.load(EMB_VEC)
    if len(meta) != vecs.shape[0]:
        m = min(len(meta), vecs.shape[0])
        warnings.warn('Meta rows and vec rows mismatch; aligning to min')
        meta = meta.iloc[:m].reset_index(drop=True)
        vecs = vecs[:m]
    # ensure absolute paths in meta
    meta['image_path'] = meta['image_path'].astype(str).apply(lambda p: str(Path(p).resolve()))

    # filter to shifted embeddings
    mask_shifted = meta['dataset'].astype(str) == '2025_shifted'
    meta_shifted = meta[mask_shifted].reset_index(drop=True)
    vecs_shifted = vecs[mask_shifted.values]
    # build mapping from image_path -> index in shifted arrays
    idx_map = {str(p): i for i, p in enumerate(meta_shifted['image_path'].astype(str))}

    results = []
    ks = []
    avg_pairwise_list = []
    trace_list = []

    for k in K_STRS:
        imgs = collect_dataset_image_paths(k)
        if len(imgs) == 0:
            print(f'k={k}: no images found; skipping')
            continue
        # match to embeddings
        emb_list = []
        missing = 0
        for img in imgs:
            img_abs = str(Path(img).resolve())
            if img_abs in idx_map:
                emb_list.append(vecs_shifted[idx_map[img_abs]])
            else:
                missing += 1
        if len(emb_list) == 0:
            print(f'k={k}: no matching embeddings found; skipping')
            continue
        X = np.vstack(emb_list).astype(np.float64)
        # L2 normalize rows
        Xn = l2_normalize_rows(X)
        # compute metrics
        avg_pairwise, min_pairwise, max_pairwise = pairwise_euclidean_stats(Xn)
        trace_cov, det_cov = cov_trace_and_det(Xn)
        mean_centroid_dist, std_centroid_dist = mean_std_centroid_dist(Xn)

        results.append({
            'k': k,
            'num_images': int(Xn.shape[0]),
            'avg_pairwise': avg_pairwise,
            'min_pairwise': min_pairwise,
            'max_pairwise': max_pairwise,
            'trace_cov': trace_cov,
            'det_cov': det_cov,
            'mean_centroid_dist': mean_centroid_dist,
            'std_centroid_dist': std_centroid_dist,
            'missing_embeddings': int(missing),
        })
        ks.append(int(k.replace('k', '')))
        avg_pairwise_list.append(avg_pairwise)
        trace_list.append(trace_cov)

    # save CSV
    df_res = pd.DataFrame(results)
    df_res.to_csv(DIVERSITY_CSV, index=False)

    # summary JSON
    summary = {}
    if not df_res.empty:
        # most diverse by avg_pairwise
        idx_max = df_res['avg_pairwise'].idxmax()
        idx_min = df_res['avg_pairwise'].idxmin()
        summary['most_diverse_k'] = df_res.loc[idx_max, 'k']
        summary['least_diverse_k'] = df_res.loc[idx_min, 'k']
        # correlations (pearson)
        xs = np.array([int(s.replace('k', '')) for s in df_res['k']])
        ys_avg = df_res['avg_pairwise'].to_numpy(dtype=np.float64)
        ys_trace = df_res['trace_cov'].to_numpy(dtype=np.float64)
        def safe_corr(a, b):
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() < 2:
                return float('nan')
            a2 = a[mask]
            b2 = b[mask]
            a2 = a2 - a2.mean()
            b2 = b2 - b2.mean()
            denom = math.sqrt((a2**2).sum() * (b2**2).sum())
            if denom == 0:
                return float('nan')
            return float((a2 * b2).sum() / denom)
        summary['corr_k_avg_pairwise'] = safe_corr(xs, ys_avg)
        summary['corr_k_trace_cov'] = safe_corr(xs, ys_trace)
    write_json = lambda p, obj: p.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    write_json(SUMMARY_JSON, summary)

    # plots: one figure per metric vs k, default matplotlib styling
    # prepare x values (numeric)
    if not df_res.empty:
        x = np.array([int(s.replace('k', '')) for s in df_res['k']])
        def plot_and_save(y, ylabel, fname):
            fig, ax = plt.subplots(figsize=(6,4))
            ax.plot(x, y, marker='o')
            ax.set_xlabel('k (%)')
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel + ' vs k')
            plt.tight_layout()
            outp = PLOTS_DIR / fname
            fig.savefig(outp, dpi=200)
            plt.close(fig)

        plot_and_save(df_res['avg_pairwise'].to_numpy(), 'avg_pairwise', 'avg_pairwise_vs_k.png')
        plot_and_save(df_res['min_pairwise'].to_numpy(), 'min_pairwise', 'min_pairwise_vs_k.png')
        plot_and_save(df_res['trace_cov'].to_numpy(), 'trace_cov', 'trace_cov_vs_k.png')
        # combined diversity metric: avg_pairwise * trace_cov (example)
        combined = df_res['avg_pairwise'].to_numpy() * df_res['trace_cov'].to_numpy()
        plot_and_save(combined, 'diversity_combined', 'diversity_vs_k.png')

    # print final summary
    print('Diversity analysis complete.')
    if 'most_diverse_k' in summary:
        print('Most diverse k:', summary['most_diverse_k'])
        print('Least diverse k:', summary['least_diverse_k'])
        print('Correlation k vs avg_pairwise:', summary['corr_k_avg_pairwise'])
        print('Correlation k vs trace_cov:', summary['corr_k_trace_cov'])


if __name__ == '__main__':
    main()

