#!/usr/bin/env python3
"""
analyze_embedding_vs_error_from_summary_2025.py

Domain-shift analysis (DINO embeddings):
- Load DINO embeddings (meta + vectors)
- Split into 2024 TRAIN (reference) and 2025 TEST (target)
- L2-normalize embeddings
- Compute centroid over 2024 TRAIN
- For each 2025 TEST sample compute:
    - distance_to_2024_centroid
    - nearest_neighbor_distance (to 2024 TRAIN) computed vectorized
- Load evaluation summary, compute per-image MAE and detection_success
- Merge by basename and compute correlations for both distance metrics
- Save output CSV to outputs/analysis/dino_domain_shift_analysis_2025.csv
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import stats
import warnings

# Paths (adjusted to DINO outputs produced earlier)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMBED_META = PROJECT_ROOT / "scripts/representation_analysis/outputs_dino_full/dino_embeddings_full_meta.csv"
EMBED_VECT = PROJECT_ROOT / "scripts/representation_analysis/outputs_dino_full/dino_embeddings_full.npy"
SUMMARY_CSV = Path("/Users/taircarmon/Desktop/data-centric-cross-season-transfer/scripts/eval/outputs/2024_Full_models/test_on_2025/all-ponds_weights_best.pt/summary.csv")
OUT_DIR = PROJECT_ROOT / "outputs" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "dino_domain_shift_analysis_2025.csv"

# Helpers

def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def read_summary(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df


def compute_image_metrics(summary_df: pd.DataFrame) -> pd.DataFrame:
    # find err column
    err_col = None
    for c in summary_df.columns:
        if c.lower() in ("err_mm", "err", "err_mm_pred"):
            err_col = c
            break
    if err_col is None:
        numeric_cols = [c for c in summary_df.columns if pd.api.types.is_numeric_dtype(summary_df[c])]
        if len(numeric_cols) > 0:
            err_col = numeric_cols[0]

    img_col = None
    for c in summary_df.columns:
        if c.lower() in ('image', 'image_name') or c.lower().endswith('image'):
            img_col = c
            break
    if img_col is None:
        raise KeyError('Could not find image column in summary CSV')

    status_col = None
    for c in summary_df.columns:
        if c.lower() in ('status', 'result', 'status_str'):
            status_col = c
            break

    df = summary_df.copy()
    if err_col is not None:
        df['_abs_err'] = df[err_col].abs()
    else:
        df['_abs_err'] = np.nan
    df['_basename'] = df[img_col].astype(str).apply(lambda p: Path(p).name)

    grouped = df.groupby('_basename')
    records = []
    for name, g in grouped:
        mae_image = float(g['_abs_err'].dropna().mean()) if g['_abs_err'].notna().any() else np.nan
        if status_col is not None:
            det_succ = 1 if (g[status_col].astype(str) == 'OK').any() else 0
        else:
            det_succ = 1 if g['_abs_err'].notna().any() else 0
        mae_total = np.nan
        mae_carapace = np.nan
        if 'mode' in g.columns:
            try:
                total_rows = g[g['mode'].astype(str).str.lower().str.contains('body|total')]
                car_rows = g[g['mode'].astype(str).str.lower().str.contains('carapace')]
                if not total_rows.empty:
                    mae_total = float(total_rows['_abs_err'].dropna().mean()) if total_rows['_abs_err'].notna().any() else np.nan
                if not car_rows.empty:
                    mae_carapace = float(car_rows['_abs_err'].dropna().mean()) if car_rows['_abs_err'].notna().any() else np.nan
            except Exception:
                pass
        records.append({'image': name, 'MAE_image': mae_image, 'detection_success': int(det_succ), 'MAE_total': mae_total, 'MAE_carapace': mae_carapace})
    metrics = pd.DataFrame.from_records(records)
    return metrics


def load_embeddings_and_compute_distances(meta_path: Path, vec_path: Path):
    if not meta_path.exists() or not vec_path.exists():
        raise FileNotFoundError(f'Embeddings not found: {meta_path}, {vec_path}')
    try:
        meta = pd.read_csv(meta_path)
    except Exception:
        meta = pd.read_csv(meta_path, header=None)
    vecs = np.load(vec_path)
    if len(meta) != vecs.shape[0]:
        m = min(len(meta), vecs.shape[0])
        meta = meta.iloc[:m].reset_index(drop=True)
        vecs = vecs[:m]

    # normalize column names
    meta.columns = [c.strip() for c in meta.columns]
    if 'image_path' not in meta.columns:
        meta = meta.rename(columns={meta.columns[0]: 'image_path'})
    # ensure season, dataset_type, subtype exist; tolerate different column names
    if 'season' not in meta.columns:
        if len(meta.columns) > 1:
            meta = meta.rename(columns={meta.columns[1]: 'season'})
        else:
            meta['season'] = ''
    # dataset_type may be named 'dataset_type' or 'split' in previous scripts
    if 'dataset_type' not in meta.columns:
        if 'split' in meta.columns:
            meta = meta.rename(columns={'split': 'dataset_type'})
        else:
            meta['dataset_type'] = ''
    if 'subtype' not in meta.columns:
        # try 'pond' or create 'subtype' as 'all'
        if 'pond' in meta.columns:
            meta = meta.rename(columns={'pond': 'subtype'})
        else:
            meta['subtype'] = ''

    # create basename
    meta['basename'] = meta['image_path'].astype(str).apply(lambda p: Path(str(p).replace('\\','/')).name)

    # L2-normalize vectors
    vecs_n = l2_normalize_rows(np.asarray(vecs))

    # select 2024 TRAIN and 2025 TEST indices
    is_2024_train = (meta['season'].astype(str) == '2024') & (meta['dataset_type'].astype(str) == 'train')
    is_2025_test = (meta['season'].astype(str) == '2025') & (meta['dataset_type'].astype(str) == 'test')

    idx_2024 = np.where(is_2024_train.values)[0]
    idx_2025 = np.where(is_2025_test.values)[0]

    print(f'Number of 2024 TRAIN embeddings: {len(idx_2024)}')
    print(f'Number of 2025 TEST embeddings: {len(idx_2025)}')

    if len(idx_2024) == 0 or len(idx_2025) == 0:
        # still return meta and empty distances
        meta = meta.copy()
        meta['distance_to_2024_centroid'] = np.nan
        meta['nearest_neighbor_distance'] = np.nan
        return meta, vecs_n, idx_2024, idx_2025

    vecs_2024 = vecs_n[idx_2024, :]
    vecs_2025 = vecs_n[idx_2025, :]

    # compute centroid of 2024
    centroid_2024 = vecs_2024.mean(axis=0)

    # distance to centroid
    # d = ||x - c||
    # vectorized
    diff = vecs_2025 - centroid_2024[None, :]
    dist_centroid = np.linalg.norm(diff, axis=1)

    # nearest neighbor distance to 2024 TRAIN using vectorized formula
    # squared distances = |a|^2 + |b|^2 - 2 a.b
    a2 = np.sum(vecs_2025 ** 2, axis=1, keepdims=True)  # (m,1)
    b2 = np.sum(vecs_2024 ** 2, axis=1, keepdims=True)  # (n,1)
    # compute dot product (m x n)
    dots = vecs_2025.dot(vecs_2024.T)  # (m, n)
    # squared dists: a2 + b2.T - 2*dots
    sq_dists = a2 + b2.T - 2.0 * dots
    # numerical safety
    sq_dists[sq_dists < 0] = 0.0
    nn_dists = np.sqrt(np.min(sq_dists, axis=1))

    # assign distances back into meta aligned with idx_2025
    meta = meta.copy()
    meta['distance_to_2024_centroid'] = np.nan
    meta['nearest_neighbor_distance'] = np.nan
    meta.loc[meta.index[idx_2025], 'distance_to_2024_centroid'] = dist_centroid
    meta.loc[meta.index[idx_2025], 'nearest_neighbor_distance'] = nn_dists

    return meta, vecs_n, idx_2024, idx_2025


def compute_and_save():
    print('Loading summary CSV...')
    try:
        summary_df = read_summary(SUMMARY_CSV)
    except Exception as e:
        print('ERROR loading summary CSV:', e)
        sys.exit(1)

    print('Computing image-level metrics from summary...')
    metrics_df = compute_image_metrics(summary_df)

    print('Loading embeddings and computing distances...')
    try:
        emb_meta, vecs_n, idx_2024, idx_2025 = load_embeddings_and_compute_distances(EMBED_META, EMBED_VECT)
    except Exception as e:
        print('ERROR loading embeddings:', e)
        sys.exit(2)

    # keep only 2025 test rows with computed distances
    emb_2025 = emb_meta[(emb_meta['season'].astype(str) == '2025') & (emb_meta['dataset_type'].astype(str) == 'test')].copy()
    print(f'Found {len(emb_2025)} embedding rows for season 2025 (test)')

    # prepare merge: keep columns distance_to_2024_centroid and nearest_neighbor_distance
    emb_2025_small = emb_2025[['basename', 'distance_to_2024_centroid', 'nearest_neighbor_distance']].copy()

    # merge by basename
    merged = pd.merge(emb_2025_small, metrics_df, left_on='basename', right_on='image', how='inner')
    print(f'Merged size (images present in both embeddings and summary): {len(merged)}')

    if merged.empty:
        print('No overlapping images between embeddings (2025 test) and summary results — nothing to analyze')
        merged.to_csv(OUT_CSV, index=False)
        print('Wrote empty output to', OUT_CSV)
        return

    # compute correlations for both metrics
    results = {}
    def safe_corr(x, y, method='pearson'):
        try:
            if method == 'pearson':
                return stats.pearsonr(x, y)
            if method == 'spearman':
                return stats.spearmanr(x, y)
        except Exception:
            return (np.nan, np.nan)

    # drop NaNs for MAE
    merged_mae = merged.dropna(subset=['MAE_image'])

    if len(merged_mae) >= 3:
        pc_r, pc_p = safe_corr(merged_mae['distance_to_2024_centroid'], merged_mae['MAE_image'], 'pearson')
        ps_r, ps_p = safe_corr(merged_mae['distance_to_2024_centroid'], merged_mae['MAE_image'], 'spearman')
        nc_r, nc_p = safe_corr(merged_mae['nearest_neighbor_distance'], merged_mae['MAE_image'], 'pearson')
        ns_r, ns_p = safe_corr(merged_mae['nearest_neighbor_distance'], merged_mae['MAE_image'], 'spearman')
    else:
        pc_r = pc_p = ps_r = ps_p = nc_r = nc_p = ns_r = ns_p = np.nan

    # point-biserial for detection_success vs distances
    try:
        if merged['detection_success'].nunique() > 1:
            pbd_r, pbd_p = stats.pointbiserialr(merged['detection_success'], merged['distance_to_2024_centroid'])
        else:
            pbd_r = pbd_p = np.nan
    except Exception:
        pbd_r = pbd_p = np.nan
    try:
        if merged['detection_success'].nunique() > 1:
            pbn_r, pbn_p = stats.pointbiserialr(merged['detection_success'], merged['nearest_neighbor_distance'])
        else:
            pbn_r = pbn_p = np.nan
    except Exception:
        pbn_r = pbn_p = np.nan

    def interpret(r):
        if r is None or (isinstance(r, float) and np.isnan(r)):
            return 'n/a'
        ar = abs(r)
        if ar < 0.1:
            return 'no meaningful correlation'
        if ar < 0.3:
            return 'weak'
        return 'moderate or stronger'

    print('\nCorrelation results:')
    print('Distance to 2024 centroid vs MAE_image:')
    print(f'  Pearson r={pc_r:.4f}, p={pc_p:.3g} -> {interpret(pc_r)}')
    print(f'  Spearman rho={ps_r:.4f}, p={ps_p:.3g} -> {interpret(ps_r)}')
    print('Nearest neighbor distance vs MAE_image:')
    print(f'  Pearson r={nc_r:.4f}, p={nc_p:.3g} -> {interpret(nc_r)}')
    print(f'  Spearman rho={ns_r:.4f}, p={ns_p:.3g} -> {interpret(ns_r)}')
    print('Point-biserial (detection_success vs distance to centroid):')
    print(f'  r={pbd_r:.4f}, p={pbd_p:.3g} -> {interpret(pbd_r)}')
    print('Point-biserial (detection_success vs nearest neighbor):')
    print(f'  r={pbn_r:.4f}, p={pbn_p:.3g} -> {interpret(pbn_r)}')

    # Save detailed CSV
    out_df = merged[['image', 'basename', 'distance_to_2024_centroid', 'nearest_neighbor_distance', 'MAE_image', 'detection_success', 'MAE_total', 'MAE_carapace']].copy()
    out_df = out_df.rename(columns={'image':'image_name'})
    out_df.to_csv(OUT_CSV, index=False)
    print('Wrote output to', OUT_CSV)


if __name__ == '__main__':
    compute_and_save()
