#!/usr/bin/env python3
"""
analyze_shift_unified.py

Load unified embeddings and compute embedding-space statistics and cross-dataset metrics.
Saves results to scripts/shift_experiments/results/embedding_shift_metrics.json
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

PROJECT_ROOT = Path(r"C:\Users\carmonta\Desktop\data-centric-cross-season-transfer")
EMB_DIR = PROJECT_ROOT / 'scripts' / 'shift_experiments' / 'embeddings'
OUT_DIR = PROJECT_ROOT / 'scripts' / 'shift_experiments' / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)
EMB_VEC = EMB_DIR / 'all_embeddings.npy'
EMB_META = EMB_DIR / 'all_meta.csv'
OUT_JSON = OUT_DIR / 'embedding_shift_metrics.json'

# Kernel bandwidth for RBF MMD
RBF_GAMMA = None  # if None, will be set to 1 / median(pairwise_distance^2)


def load_embeddings():
    if not EMB_META.exists() or not EMB_VEC.exists():
        raise FileNotFoundError('Embeddings or meta not found under ' + str(EMB_DIR))
    meta = pd.read_csv(EMB_META)
    X = np.load(EMB_VEC)
    if len(meta) != X.shape[0]:
        m = min(len(meta), X.shape[0])
        warnings.warn('Meta rows and vectors mismatch; aligning to min length')
        meta = meta.iloc[:m].reset_index(drop=True)
        X = X[:m]
    return meta, X


def compute_within_stats(X):
    # centroid
    c = X.mean(axis=0)
    # trace of covariance
    cov = np.cov(X, rowvar=False)
    trace = float(np.trace(cov))
    # mean pairwise cosine similarity
    # cosine similarity = 1 - cosine_distance
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    cos_sim = (Xn @ Xn.T)
    # exclude diagonal
    n = X.shape[0]
    if n <= 1:
        mean_cos = float('nan')
    else:
        mean_cos = float((np.sum(cos_sim) - n) / (n * (n - 1)))
    # mean pairwise euclidean
    if n <= 1:
        mean_euc = float('nan')
    else:
        d = cdist(X, X, metric='euclidean')
        mean_euc = float((np.sum(d) - 0.0) / (n * (n - 1)))
    return {'centroid': c.tolist(), 'trace_cov': trace, 'mean_cosine': mean_cos, 'mean_euclidean': mean_euc}


def median_pairwise_sq_dist(A, B=None):
    if B is None:
        D = cdist(A, A, metric='euclidean')
    else:
        D = cdist(A, B, metric='euclidean')
    sq = D.flatten() ** 2
    return np.median(sq)


def rbf_kernel_matrix(A, B, gamma):
    D2 = cdist(A, B, metric='sqeuclidean')
    K = np.exp(-gamma * D2)
    return K


def compute_mmd_rbf(X, Y, gamma):
    nx = X.shape[0]
    ny = Y.shape[0]
    Kxx = rbf_kernel_matrix(X, X, gamma)
    Kyy = rbf_kernel_matrix(Y, Y, gamma)
    Kxy = rbf_kernel_matrix(X, Y, gamma)
    mmd = Kxx.sum() / (nx * nx) + Kyy.sum() / (ny * ny) - 2.0 * Kxy.sum() / (nx * ny)
    return float(mmd)


def main():
    meta, X = load_embeddings()
    # split
    mask24 = meta['dataset'].astype(str) == '2024'
    mask25c = meta['dataset'].astype(str) == '2025_clean'
    mask25s = meta['dataset'].astype(str) == '2025_shifted'
    X24 = X[mask24.values]
    X25c = X[mask25c.values]
    X25s = X[mask25s.values]

    results = {}
    results['counts'] = {'n_2024': int(X24.shape[0]), 'n_2025_clean': int(X25c.shape[0]), 'n_2025_shifted': int(X25s.shape[0])}

    if X24.shape[0] > 0:
        results['stats_2024'] = compute_within_stats(X24)
    else:
        results['stats_2024'] = None
    if X25c.shape[0] > 0:
        results['stats_2025_clean'] = compute_within_stats(X25c)
    else:
        results['stats_2025_clean'] = None
    if X25s.shape[0] > 0:
        results['stats_2025_shifted'] = compute_within_stats(X25s)
    else:
        results['stats_2025_shifted'] = None

    # centroid distances
    def centroid_dist(A, B):
        if A is None or B is None:
            return None
        a = np.array(A)
        b = np.array(B)
        return float(np.linalg.norm(a - b))

    c24 = results['stats_2024']['centroid'] if results['stats_2024'] else None
    c25c = results['stats_2025_clean']['centroid'] if results['stats_2025_clean'] else None
    c25s = results['stats_2025_shifted']['centroid'] if results['stats_2025_shifted'] else None
    results['centroid_distances'] = {
        '2024_vs_2025_clean': centroid_dist(c24, c25c),
        '2024_vs_2025_shifted': centroid_dist(c24, c25s),
        '2025_clean_vs_2025_shifted': centroid_dist(c25c, c25s),
    }

    # MMD computation
    # choose gamma = 1 / median_sq_dist of combined
    # combined of 2024 and target
    if X24.shape[0] > 0 and X25c.shape[0] > 0:
        med_sq = median_pairwise_sq_dist(np.vstack([X24, X25c]))
        gamma = 1.0 / med_sq if med_sq > 0 else 1.0
        mmd_24_c = compute_mmd_rbf(X24, X25c, gamma)
    else:
        mmd_24_c = None
    if X24.shape[0] > 0 and X25s.shape[0] > 0:
        med_sq = median_pairwise_sq_dist(np.vstack([X24, X25s]))
        gamma = 1.0 / med_sq if med_sq > 0 else 1.0
        mmd_24_s = compute_mmd_rbf(X24, X25s, gamma)
    else:
        mmd_24_s = None
    results['mmd'] = {'2024_vs_2025_clean': mmd_24_c, '2024_vs_2025_shifted': mmd_24_s}

    # save
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print('Saved metrics to', OUT_JSON)


if __name__ == '__main__':
    main()

