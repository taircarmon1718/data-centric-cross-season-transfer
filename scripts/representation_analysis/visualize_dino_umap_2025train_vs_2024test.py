#!/usr/bin/env python3
"""
visualize_dino_umap_2025train_vs_2024test.py

UMAP visualization of DINO embeddings (opposite split):
- 2025 TRAIN vs 2024 TEST (colored by season)
- 2024 TEST colored by MAE (continuous)

Inputs:
- DINO embeddings meta: scripts/representation_analysis/outputs_dino_full/dino_embeddings_full_meta.csv
- DINO embeddings vectors: scripts/representation_analysis/outputs_dino_full/dino_embeddings_full.npy
- Evaluation summary CSV (MAE for 2024 test using 2025-trained model):
  /Users/taircarmon/Desktop/data-centric-cross-season-transfer/scripts/eval/outputs/2025_Full_models/test_on_2024/YOLOv11n_train_on_2025_all_pose_300ep/summary.csv

Outputs:
- outputs/analysis/umap_season_2025_train_vs_2024_test.png
- outputs/analysis/umap_2024_colored_by_MAE.png

This script intentionally mirrors the structure and robustness of
`visualize_dino_umap_2024_2025.py` but swaps the reference/target sets.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import warnings

# Optional: umap might not be installed in minimal env; will error at runtime if missing
try:
    import umap
except Exception:
    umap = None

# Deterministic
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMBED_META = PROJECT_ROOT / 'scripts' / 'representation_analysis' / 'outputs_dino_full' / 'dino_embeddings_full_meta.csv'
EMBED_VEC = PROJECT_ROOT / 'scripts' / 'representation_analysis' / 'outputs_dino_full' / 'dino_embeddings_full.npy'
# Summary CSV for 2025-trained model evaluated on 2024 test
SUMMARY_CSV = Path('/Users/taircarmon/Desktop/data-centric-cross-season-transfer/scripts/eval/outputs/2025_Full_models/test_on_2024/YOLOv11n_train_on_2025_all_pose_300ep/summary.csv')
OUT_DIR = PROJECT_ROOT / 'outputs' / 'analysis'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_SEASON = OUT_DIR / 'umap_season_2025_train_vs_2024_test.png'
OUT_MAE = OUT_DIR / 'umap_2024_colored_by_MAE.png'

# UMAP params (same as before)
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_RANDOM_STATE = RANDOM_STATE


def load_embeddings(meta_path: Path, vec_path: Path):
    if not meta_path.exists() or not vec_path.exists():
        raise FileNotFoundError(f"Embedding files not found: {meta_path} or {vec_path}")
    try:
        meta = pd.read_csv(meta_path)
    except Exception:
        meta = pd.read_csv(meta_path, header=None)
    vecs = np.load(vec_path)
    if len(meta) != vecs.shape[0]:
        m = min(len(meta), vecs.shape[0])
        warnings.warn(f"Meta rows ({len(meta)}) != vector rows ({vecs.shape[0]}). Aligning by min length={m}.")
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
    if 'subtype' not in meta.columns:
        if 'pond' in meta.columns:
            meta = meta.rename(columns={'pond': 'subtype'})
        else:
            meta['subtype'] = ''
    meta['basename'] = meta['image_path'].astype(str).apply(lambda p: Path(str(p).replace('\\','/')).name)
    return meta, vecs


def compute_image_metrics(summary_csv: Path):
    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")
    df = pd.read_csv(summary_csv)
    df.columns = [c.strip() for c in df.columns]
    err_col = None
    for c in df.columns:
        if c.lower() in ('err_mm', 'err', 'err_mm_pred'):
            err_col = c
            break
    if err_col is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        err_col = numeric_cols[0] if numeric_cols else None
    img_col = None
    for c in df.columns:
        if c.lower() in ('image', 'image_name') or c.lower().endswith('image'):
            img_col = c
            break
    if img_col is None:
        raise KeyError('Could not find image column in summary CSV')
    status_col = None
    for c in df.columns:
        if c.lower() in ('status', 'result', 'status_str'):
            status_col = c
            break
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
        records.append({'basename': name, 'MAE_image': mae_image, 'detection_success': int(det_succ)})
    metrics = pd.DataFrame.from_records(records)
    return metrics


def run_umap_and_plot(meta, vecs, metrics):
    if umap is None:
        raise RuntimeError('umap-learn not installed. Please pip install umap-learn')
    vecs_n = normalize(vecs, norm='l2')
    # Now opposite selection: 2025 TRAIN (reference) and 2024 TEST (target)
    is_2025_train = (meta['season'].astype(str) == '2025') & (meta['dataset_type'].astype(str) == 'train')
    is_2024_test = (meta['season'].astype(str) == '2024') & (meta['dataset_type'].astype(str) == 'test')
    idx_2025 = np.where(is_2025_train.values)[0]
    idx_2024 = np.where(is_2024_test.values)[0]
    print(f"Samples: 2025 train={len(idx_2025)}, 2024 test={len(idx_2024)}")
    if len(idx_2025) == 0 or len(idx_2024) == 0:
        raise RuntimeError('Not enough samples for UMAP: need both 2025 train and 2024 test')
    combined_idx = np.concatenate([idx_2025, idx_2024])
    combined_vecs = vecs_n[combined_idx]
    reducer = umap.UMAP(n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST, n_components=2, random_state=UMAP_RANDOM_STATE)
    embedding_2d = reducer.fit_transform(combined_vecs)
    coords_2025 = embedding_2d[:len(idx_2025)]
    coords_2024 = embedding_2d[len(idx_2025):]
    # plot season
    plt.figure(figsize=(10, 8))
    plt.scatter(coords_2025[:, 0], coords_2025[:, 1], s=6, c='green', label='2025 train', alpha=0.7)
    plt.scatter(coords_2024[:, 0], coords_2024[:, 1], s=6, c='purple', label='2024 test', alpha=0.7)
    plt.legend()
    plt.title('UMAP: 2025 Train vs 2024 Test (DINO embeddings)')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.tight_layout()
    plt.savefig(OUT_SEASON, dpi=200)
    plt.close()
    print('Saved season UMAP to', OUT_SEASON)
    # prepare 2024 coords and merge MAE
    meta_2024 = meta.iloc[idx_2024].copy().reset_index(drop=True)
    coords_2024_df = pd.DataFrame(coords_2024, columns=['umap1', 'umap2'])
    coords_2024_df['basename'] = meta_2024['basename'].values
    merged = pd.merge(coords_2024_df, metrics, on='basename', how='left')
    # plot MAE for 2024 test
    plt.figure(figsize=(10, 8))
    mae_vals = merged['MAE_image'].values
    nan_mask = np.isnan(mae_vals)
    if nan_mask.any():
        plt.scatter(merged.loc[nan_mask, 'umap1'], merged.loc[nan_mask, 'umap2'], s=6, c='lightgray', label='no MAE', alpha=0.6)
    valid_mask = ~nan_mask
    sc = plt.scatter(merged.loc[valid_mask, 'umap1'], merged.loc[valid_mask, 'umap2'], c=merged.loc[valid_mask, 'MAE_image'], s=8, cmap='viridis', alpha=0.9)
    plt.colorbar(sc, label='MAE (mm)')
    plt.title('UMAP: 2024 Test colored by MAE')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.tight_layout()
    plt.savefig(OUT_MAE, dpi=200)
    plt.close()
    print('Saved MAE UMAP to', OUT_MAE)
    merged_count = merged.shape[0]
    valid_mae_count = int(valid_mask.sum())
    print(f'Merged 2024 points with MAE (including NaNs): {merged_count}')
    print(f'2024 points with MAE available: {valid_mae_count}')


def main():
    try:
        meta, vecs = load_embeddings(EMBED_META, EMBED_VEC)
    except Exception as e:
        print('ERROR loading embeddings:', e)
        sys.exit(1)
    try:
        metrics = compute_image_metrics(SUMMARY_CSV)
    except Exception as e:
        print('ERROR loading summary CSV or computing metrics:', e)
        sys.exit(1)
    try:
        run_umap_and_plot(meta, vecs, metrics)
    except Exception as e:
        print('ERROR during UMAP or plotting:', e)
        sys.exit(1)


if __name__ == '__main__':
    main()
