#!/usr/bin/env python3
"""
analyze_embedding_vs_error_from_summary_2025.py

Compute whether embedding distance correlates with model error using an existing
model summary (CSV) and precomputed embeddings.

Inputs (hard-coded per prompt):
- Evaluation summary CSV (from check_on_2025 output):
  /Users/taircarmon/Desktop/data-centric-cross-season-transfer/scripts/eval/outputs/2024_Full_models/test_on_2025/all-ponds_weights_best.pt/summary.csv
- Embeddings metadata CSV:
  scripts/representation_analysis/outputs_repreasentation/rep_analysis/embeddings_meta.csv
- Embeddings vectors:
  scripts/representation_analysis/outputs_repreasentation/rep_analysis/embeddings_vectors.npy

Output:
- outputs/analysis/embedding_vs_error_from_summary_2025.csv

The script is deterministic and robust to small format differences in the meta CSV.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import stats
import warnings

# Paths (use absolute summary path provided in the prompt)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_CSV = Path("/Users/taircarmon/Desktop/data-centric-cross-season-transfer/scripts/eval/outputs/2024_Full_models/test_on_2025/all-ponds_weights_best.pt/summary.csv")
EMBED_META = PROJECT_ROOT / "scripts/representation_analysis/outputs_repreasentation/rep_analysis/embeddings_meta.csv"
EMBED_VECT = PROJECT_ROOT / "scripts/representation_analysis/outputs_repreasentation/rep_analysis/embeddings_vectors.npy"
OUT_DIR = PROJECT_ROOT / "outputs" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "embedding_vs_error_from_summary_2025.csv"

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
    # Expect columns: image, mode, err_mm, status (based on check_on_2025 outputs)
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    return df


def compute_image_metrics(summary_df: pd.DataFrame) -> pd.DataFrame:
    # Ensure expected columns exist; tolerate alternatives
    cols = summary_df.columns.str.lower()
    # find err column
    err_col = None
    for c in summary_df.columns:
        if c.lower() in ("err_mm", "err", "err_mm_pred"):
            err_col = c
            break
    if err_col is None:
        # try to find a numeric column that looks like error
        numeric_cols = [c for c in summary_df.columns if pd.api.types.is_numeric_dtype(summary_df[c])]
        if len(numeric_cols) > 0:
            err_col = numeric_cols[0]
    # find image col
    img_col = None
    for c in summary_df.columns:
        if c.lower() == 'image' or c.lower() == 'image_name' or c.lower().endswith('image'):
            img_col = c
            break
    if img_col is None:
        raise KeyError('Could not find image column in summary CSV')
    # find status col
    status_col = None
    for c in summary_df.columns:
        if c.lower() in ('status', 'result', 'status_str'):
            status_col = c
            break

    # compute abs(err_mm) if err_col present
    df = summary_df.copy()
    if err_col is not None:
        df['_abs_err'] = df[err_col].abs()
    else:
        df['_abs_err'] = np.nan

    # Group by image basename (summary likely uses basename)
    df['_basename'] = df[img_col].astype(str).apply(lambda p: Path(p).name)

    grouped = df.groupby('_basename')
    records = []
    for name, g in grouped:
        mae_image = float(g['_abs_err'].dropna().mean()) if g['_abs_err'].notna().any() else np.nan
        # detection_success if any status == 'OK'
        if status_col is not None:
            det_succ = 1 if (g[status_col].astype(str) == 'OK').any() else 0
        else:
            # fallback: if any pred_mm or err present treat as detected
            det_succ = 1 if g['_abs_err'].notna().any() else 0
        # optional by-mode MAE
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


def load_embeddings_and_distances(meta_path: Path, vec_path: Path) -> pd.DataFrame:
    if not meta_path.exists() or not vec_path.exists():
        raise FileNotFoundError(f'Embeddings not found: {meta_path}, {vec_path}')
    # try reading meta with header; some old files may be headerless
    try:
        meta = pd.read_csv(meta_path)
    except Exception:
        meta = pd.read_csv(meta_path, header=None)
    vecs = np.load(vec_path)
    # align by min length
    if len(meta) != vecs.shape[0]:
        minlen = min(len(meta), vecs.shape[0])
        meta = meta.iloc[:minlen].reset_index(drop=True)
        vecs = vecs[:minlen]
    # normalize meta columns
    meta_cols = [c.strip() for c in meta.columns]
    meta.columns = meta_cols
    # ensure image_path and season columns
    if 'image_path' not in meta.columns:
        # assume first col is image_path
        meta = meta.rename(columns={meta.columns[0]: 'image_path'})
    if 'season' not in meta.columns:
        # try second column
        if len(meta.columns) > 1:
            meta = meta.rename(columns={meta.columns[1]: 'season'})
        else:
            meta['season'] = ''
    # compute distances
    vecs_n = l2_normalize_rows(np.asarray(vecs))
    centroid = vecs_n.mean(axis=0)
    dists = np.linalg.norm(vecs_n - centroid, axis=1)
    meta = meta.copy()
    meta['distance_to_centroid'] = dists
    # create basename
    meta['basename'] = meta['image_path'].astype(str).apply(lambda p: Path(p.replace('\\','/')).name)
    return meta


def compute_and_save():
    print('Loading summary CSV...')
    try:
        summary_df = read_summary(SUMMARY_CSV)
    except Exception as e:
        print('ERROR loading summary CSV:', e)
        sys.exit(1)

    print('Computing image-level metrics from summary...')
    metrics_df = compute_image_metrics(summary_df)

    print('Loading embeddings metadata and vectors...')
    try:
        emb_meta = load_embeddings_and_distances(EMBED_META, EMBED_VECT)
    except Exception as e:
        print('ERROR loading embeddings:', e)
        sys.exit(2)

    # filter to season 2025
    emb_2025 = emb_meta[emb_meta['season'].astype(str).str.contains('2025')].copy()
    print(f'Found {len(emb_2025)} embedding rows for season 2025')

    # merge by basename
    merged = pd.merge(emb_2025[['basename', 'distance_to_centroid']], metrics_df, left_on='basename', right_on='image', how='inner')
    print(f'Merged size (images present in both embeddings and summary): {len(merged)}')

    if merged.empty:
        print('No overlapping images between embeddings (2025) and summary results — nothing to analyze')
        merged.to_csv(OUT_CSV, index=False)
        print('Wrote empty output to', OUT_CSV)
        return

    # correlation analyses
    out_results = []
    # Pearson
    try:
        pear_r, pear_p = stats.pearsonr(merged['distance_to_centroid'], merged['MAE_image'])
    except Exception:
        pear_r, pear_p = (float('nan'), float('nan'))
    # Spearman
    try:
        spe_r, spe_p = stats.spearmanr(merged['distance_to_centroid'], merged['MAE_image'])
    except Exception:
        spe_r, spe_p = (float('nan'), float('nan'))
    # Point-biserial: detection_success (binary) vs distance
    try:
        if merged['detection_success'].nunique() > 1:
            pb_r, pb_p = stats.pointbiserialr(merged['detection_success'], merged['distance_to_centroid'])
        else:
            pb_r, pb_p = (float('nan'), float('nan'))
    except Exception:
        pb_r, pb_p = (float('nan'), float('nan'))

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
    print(f'Pearson: r={pear_r:.4f}, p={pear_p:.3g} -> {interpret(pear_r)}')
    print(f'Spearman: rho={spe_r:.4f}, p={spe_p:.3g} -> {interpret(spe_r)}')
    print(f'Point-biserial (detection_success vs distance): r={pb_r:.4f}, p={pb_p:.3g} -> {interpret(pb_r)}')

    # Save CSV with required columns
    out_df = merged[['image', 'basename', 'distance_to_centroid', 'MAE_image', 'detection_success']].copy()
    out_df = out_df.rename(columns={'image': 'image_name'})
    out_df.to_csv(OUT_CSV, index=False)
    print('Wrote analysis CSV to', OUT_CSV)


if __name__ == '__main__':
    compute_and_save()

