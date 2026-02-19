#!/usr/bin/env python3
"""
visualize_shift_space.py

Load unified embeddings and metadata and produce PCA and t-SNE plots
and cosine distance histograms. Saves PNG files under results/.
"""
from pathlib import Path
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist

PROJECT_ROOT = Path(r"C:\Users\carmonta\Desktop\data-centric-cross-season-transfer")
EMB_DIR = PROJECT_ROOT / 'scripts' / 'shift_experiments' / 'embeddings'
OUT_DIR = PROJECT_ROOT / 'scripts' / 'shift_experiments' / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)
EMB_VEC = EMB_DIR / 'all_embeddings.npy'
EMB_META = EMB_DIR / 'all_meta.csv'


def load_all():
    if not EMB_META.exists() or not EMB_VEC.exists():
        raise FileNotFoundError('Embeddings or meta missing')
    meta = pd.read_csv(EMB_META)
    X = np.load(EMB_VEC)
    if len(meta) != X.shape[0]:
        m = min(len(meta), X.shape[0])
        warnings.warn('Aligning meta and vectors to min length')
        meta = meta.iloc[:m].reset_index(drop=True)
        X = X[:m]
    return meta, X


def pca_plot(meta, X):
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(8, 6))
    # colors
    mapping = {'2024': 'blue', '2025_clean': 'green', '2025_shifted': 'red'}
    for label, color in mapping.items():
        mask = meta['dataset'].astype(str) == label
        if mask.sum() == 0:
            continue
        ax.scatter(Z[mask, 0], Z[mask, 1], s=6, c=color, label=label, alpha=0.7)
    ax.legend()
    ax.set_title('PCA 2D of unified embeddings')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    plt.tight_layout()
    outp = OUT_DIR / 'pca_plot.png'
    fig.savefig(outp, dpi=200)
    plt.close(fig)
    print('Saved PCA plot to', outp)


def tsne_plot(meta, X):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=0, init='pca')
    Z = tsne.fit_transform(X)
    fig, ax = plt.subplots(figsize=(8, 6))
    mapping = {'2024': 'blue', '2025_clean': 'green', '2025_shifted': 'red'}
    for label, color in mapping.items():
        mask = meta['dataset'].astype(str) == label
        if mask.sum() == 0:
            continue
        ax.scatter(Z[mask, 0], Z[mask, 1], s=6, c=color, label=label, alpha=0.7)
    ax.legend()
    ax.set_title('t-SNE 2D of unified embeddings')
    ax.set_xlabel('t-SNE1')
    ax.set_ylabel('t-SNE2')
    plt.tight_layout()
    outp = OUT_DIR / 'tsne_plot.png'
    fig.savefig(outp, dpi=200)
    plt.close(fig)
    print('Saved t-SNE plot to', outp)


def cosine_distance_histograms(meta, X):
    # compute cosine distances within datasets and cross 2024 vs shifted
    def mean_cosine_dists(A):
        if A.shape[0] < 2:
            return np.array([])
        # cosine distance = 1 - cosine similarity
        norms = np.linalg.norm(A, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        An = A / norms
        sims = An @ An.T
        idxs = np.triu_indices_from(sims, k=1)
        dists = 1.0 - sims[idxs]
        return dists

    mapping = {'2024': 'blue', '2025_clean': 'green', '2025_shifted': 'red'}
    datasets = {}
    for label in mapping.keys():
        mask = meta['dataset'].astype(str) == label
        datasets[label] = X[mask.values]

    # compute distances
    dists = {}
    for label, A in datasets.items():
        dists[label] = mean_cosine_dists(A)
    # cross 2024 vs shifted
    if datasets['2024'].shape[0] > 0 and datasets['2025_shifted'].shape[0] > 0:
        # pairwise cosine distances
        a = datasets['2024']
        b = datasets['2025_shifted']
        na = np.linalg.norm(a, axis=1, keepdims=True)
        nb = np.linalg.norm(b, axis=1, keepdims=True)
        na[na==0]=1.0
        nb[nb==0]=1.0
        an = a/na
        bn = b/nb
        sims = an @ bn.T
        dists_cross = (1.0 - sims).flatten()
    else:
        dists_cross = np.array([])

    # plot histograms
    fig, ax = plt.subplots(figsize=(8,6))
    bins = 100
    for label, arr in dists.items():
        if arr.size == 0:
            continue
        ax.hist(arr, bins=bins, alpha=0.6, label=f'within {label}', density=True)
    if dists_cross.size > 0:
        ax.hist(dists_cross, bins=bins, alpha=0.6, label='2024 vs 2025_shifted', density=True)
    ax.set_xlabel('Cosine distance')
    ax.set_ylabel('Density')
    ax.legend()
    ax.set_title('Cosine distance histograms')
    plt.tight_layout()
    outp = OUT_DIR / 'distance_histograms.png'
    fig.savefig(outp, dpi=200)
    plt.close(fig)
    print('Saved distance histograms to', outp)


def main():
    meta, X = load_all()
    pca_plot(meta, X)
    tsne_plot(meta, X)
    cosine_distance_histograms(meta, X)


if __name__ == '__main__':
    main()

