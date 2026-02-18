#!/usr/bin/env python3
"""
run_season_shift_analysis.py

Full pipeline to analyze domain shift between Season 2024 and Season 2025.

Directory outputs (created under scripts/season_shift_analysis/):
 - visual_shift/       (brightness histograms, boxplots, per-image CSV)
 - geometric_shift/    (carapace/total length, bbox size, orientation CSV + plots)
 - embedding_shift/    (PCA scatter, centroid distances, embedding stats)
 - pond_analysis/      (if pond metadata exists)
 - summary.json        (overall numeric summary)

Notes:
- Uses the shared YOLO-trained-on-2024 embedding space located at:
  scripts/representation_analysis/outputs_repreasentation/rep_analysis/embeddings_meta.csv
  scripts/representation_analysis/outputs_repreasentation/rep_analysis/embeddings_vectors.npy
- Does NOT recompute embeddings.
- Robust to missing dirs/files; prints clear warnings.

Run:
    python scripts/season_shift_analysis/run_season_shift_analysis.py

"""

from pathlib import Path
import os
import sys
import json
import math
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings

# --- Config / Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = PROJECT_ROOT / 'scripts' / 'season_shift_analysis'
VISUAL_DIR = OUT_ROOT / 'visual_shift'
GEOM_DIR = OUT_ROOT / 'geometric_shift'
EMB_DIR = OUT_ROOT / 'embedding_shift'
POND_DIR = OUT_ROOT / 'pond_analysis'
for d in [OUT_ROOT, VISUAL_DIR, GEOM_DIR, EMB_DIR, POND_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# datasets
DS_2024_ROOT = PROJECT_ROOT / 'datasets' / 'train_on_all'
DS_2025_ROOT = PROJECT_ROOT / 'datasets' / 'train_on_2025_all'
DS_2024_IMG_DIRS = [DS_2024_ROOT / 'images', DS_2024_ROOT / 'val' / 'images']
DS_2025_IMG_DIRS = [DS_2025_ROOT / 'images', DS_2025_ROOT / 'val' / 'images']
DS_2024_LABEL_DIRS = [DS_2024_ROOT / 'labels', DS_2024_ROOT / 'val' / 'labels']
DS_2025_LABEL_DIRS = [DS_2025_ROOT / 'labels', DS_2025_ROOT / 'val' / 'labels']

# embeddings
EMB_META_PATH = PROJECT_ROOT / 'scripts' / 'representation_analysis' / 'outputs_repreasentation' / 'rep_analysis' / 'embeddings_meta.csv'
EMB_VEC_PATH = PROJECT_ROOT / 'scripts' / 'representation_analysis' / 'outputs_repreasentation' / 'rep_analysis' / 'embeddings_vectors.npy'

# plotting defaults
sns.set(style='whitegrid')
RND = 42
np.random.seed(RND)

# --- Utilities ---
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def collect_images_for_dirs(dirs):
    imgs = []
    for d in dirs:
        if not d.exists():
            warnings.warn(f"Directory {d} does not exist — skipping")
            continue
        for p in sorted(d.rglob('*')):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                imgs.append(p.resolve())
    return imgs


def compute_brightness_stats(image_paths):
    records = []
    hist_accum = np.zeros(256, dtype=float)
    for p in tqdm(image_paths, desc='brightness calc'):
        try:
            with Image.open(p) as im:
                im_l = im.convert('L')
                arr = np.asarray(im_l, dtype=np.uint8)
            mean = float(arr.mean())
            std = float(arr.std())
            hist, _ = np.histogram(arr.ravel(), bins=256, range=(0, 255))
            hist_accum += hist
            records.append({'image_path': str(p), 'basename': p.name, 'mean_brightness': mean, 'std_brightness': std})
        except Exception as e:
            warnings.warn(f"Failed reading image {p}: {e}")
    df = pd.DataFrame.from_records(records)
    return df, hist_accum


def save_hist_and_box(df_2024, hist_2024, df_2025, hist_2025):
    # overlay histograms (normalized)
    fig, ax = plt.subplots(figsize=(8, 5))
    h1 = hist_2024 / hist_2024.sum() if hist_2024.sum() > 0 else hist_2024
    h2 = hist_2025 / hist_2025.sum() if hist_2025.sum() > 0 else hist_2025
    ax.plot(np.arange(256), h1, label='2024', color='blue')
    ax.plot(np.arange(256), h2, label='2025', color='red')
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Probability')
    ax.set_title('Overlay intensity histogram: 2024 vs 2025')
    ax.legend()
    fig.savefig(VISUAL_DIR / 'intensity_histogram_overlay.png', dpi=200)
    plt.close(fig)

    # boxplots for mean brightness
    fig, ax = plt.subplots(figsize=(6, 5))
    data = [df_2024['mean_brightness'].dropna().values, df_2025['mean_brightness'].dropna().values]
    ax.boxplot(data, labels=['2024', '2025'])
    ax.set_ylabel('Mean brightness')
    ax.set_title('Mean brightness by season')
    fig.savefig(VISUAL_DIR / 'brightness_boxplot.png', dpi=200)
    plt.close(fig)

    # save per-image CSVs
    df_2024.to_csv(VISUAL_DIR / 'brightness_2024.csv', index=False)
    df_2025.to_csv(VISUAL_DIR / 'brightness_2025.csv', index=False)


def find_label_file_for_image(image_path: Path, label_dirs):
    # first try same relative path under labels
    for img_root in [d for d in label_dirs[0].parents if True]:
        pass
    # simple fallback: search by basename stem
    stem = image_path.stem
    for ld in label_dirs:
        if not ld.exists():
            continue
        # look for direct file <stem>.txt
        candidate = ld / (stem + '.txt')
        if candidate.exists():
            return candidate
        # search recursively
        for p in ld.rglob('*.txt'):
            if p.stem == stem:
                return p
    return None


def parse_pose_label(label_path: Path):
    """Parse YOLO-pose label: class cx cy w h kp1x kp1y kp2x kp2y ...
    Returns dict with keys: cx, cy, w, h, kpts (list of (x,y) floats)
    Values may be normalized (0-1) or absolute; caller must handle scaling using image size.
    """
    try:
        txt = label_path.read_text().strip()
        if not txt:
            return None
        parts = txt.split()
        # assume single object per file
        floats = [float(x) for x in parts]
        if len(floats) < 5:
            return None
        cx, cy, w, h = floats[1:5]
        kpts = []
        kp_floats = floats[5:]
        for i in range(0, len(kp_floats), 2):
            if i+1 < len(kp_floats):
                kpts.append((kp_floats[i], kp_floats[i+1]))
        return {'cx': cx, 'cy': cy, 'w': w, 'h': h, 'kpts': kpts}
    except Exception:
        return None


def compute_geometric_stats(image_dirs, label_dirs, season_name):
    records = []
    for img_path in tqdm(collect_images_for_dirs(image_dirs), desc=f'geom {season_name}'):
        label_path = find_label_file_for_image(img_path, label_dirs)
        if label_path is None:
            # skip but record missing
            records.append({'image_path': str(img_path), 'basename': img_path.name, 'carapace_px': np.nan, 'total_px': np.nan, 'bbox_area_px': np.nan, 'orientation_deg': np.nan, 'label_found': False})
            continue
        lbl = parse_pose_label(label_path)
        if lbl is None:
            records.append({'image_path': str(img_path), 'basename': img_path.name, 'carapace_px': np.nan, 'total_px': np.nan, 'bbox_area_px': np.nan, 'orientation_deg': np.nan, 'label_found': False})
            continue
        # get image size
        try:
            with Image.open(img_path) as im:
                W, H = im.size
        except Exception:
            W, H = None, None
        # interpret bbox
        cx, cy, w, h = lbl['cx'], lbl['cy'], lbl['w'], lbl['h']
        # detect normalization heuristically
        is_norm = all(0.0 <= v <= 1.2 for v in [cx, cy, w, h])
        if is_norm and W is not None:
            bx = cx * W
            by = cy * H
            bw = w * W
            bh = h * H
        else:
            bx = cx
            by = cy
            bw = w
            bh = h
        bbox_area = abs(bw * bh) if (not math.isnan(bw) and not math.isnan(bh)) else np.nan
        # keypoints
        kpts = lbl.get('kpts', [])
        car_px = np.nan
        tot_px = np.nan
        orient_deg = np.nan
        if len(kpts) >= 4:
            def to_abs(pt):
                x, y = pt
                if is_norm and W is not None:
                    return (x * W, y * H)
                return (x, y)
            kabs = [to_abs(k) for k in kpts[:4]]
            # CAR_IDXS = (0,1), TOT_IDXS = (2,3)
            car_px = float(np.hypot(kabs[1][0] - kabs[0][0], kabs[1][1] - kabs[0][1]))
            tot_px = float(np.hypot(kabs[3][0] - kabs[2][0], kabs[3][1] - kabs[2][1]))
            # orientation: angle of vector rostrum(2)->tail(3) in degrees
            dx = kabs[3][0] - kabs[2][0]
            dy = kabs[3][1] - kabs[2][1]
            orient_deg = float((math.degrees(math.atan2(dy, dx)) + 360) % 360)
        records.append({'image_path': str(img_path), 'basename': img_path.name, 'carapace_px': car_px, 'total_px': tot_px, 'bbox_area_px': bbox_area, 'orientation_deg': orient_deg, 'label_found': True})
    df = pd.DataFrame.from_records(records)
    df.to_csv(GEOM_DIR / f'geometric_{season_name}.csv', index=False)
    return df


def load_embeddings_meta_and_vectors(meta_path, vec_path):
    if not meta_path.exists() or not vec_path.exists():
        raise FileNotFoundError(f'Embeddings not found at: {meta_path} or {vec_path}')
    try:
        meta = pd.read_csv(meta_path)
    except Exception:
        meta = pd.read_csv(meta_path, header=None)
    vecs = np.load(vec_path)
    if len(meta) != vecs.shape[0]:
        m = min(len(meta), vecs.shape[0])
        warnings.warn(f"Meta rows ({len(meta)}) != vec rows ({vecs.shape[0]}). Aligning by min length={m}.")
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
    # ensure basename
    meta['basename'] = meta['image_path'].astype(str).apply(lambda p: Path(str(p).replace('\\','/')).name)
    return meta, vecs


def embedding_space_analysis(meta, vecs):
    # split
    mask24 = (meta['season'].astype(str) == '2024')
    mask25 = (meta['season'].astype(str) == '2025')
    idx24 = np.where(mask24.values)[0]
    idx25 = np.where(mask25.values)[0]
    results = {}
    if len(idx24) == 0 or len(idx25) == 0:
        warnings.warn('Not enough samples for embedding split analysis')
        return results
    X24 = vecs[idx24].astype(float)
    X25 = vecs[idx25].astype(float)
    # L2 normalize rows for distance computations
    def l2norm_rows(X):
        n = np.linalg.norm(X, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return X / n
    X24n = l2norm_rows(X24)
    X25n = l2norm_rows(X25)
    # PCA on combined
    X_all = np.vstack([X24n, X25n])
    pca = PCA(n_components=2, random_state=RND)
    pcs = pca.fit_transform(X_all)
    pcs24 = pcs[:len(X24n)]
    pcs25 = pcs[len(X24n):]
    # centroids
    c24 = X24n.mean(axis=0)
    c25 = X25n.mean(axis=0)
    centroid_distance = float(np.linalg.norm(c24 - c25))
    # intra-season variance (mean squared distance to centroid)
    intra24 = float(np.mean(np.sum((X24n - c24) ** 2, axis=1)))
    intra25 = float(np.mean(np.sum((X25n - c25) ** 2, axis=1)))
    # inter mean distance: mean pairwise distance between samples across seasons
    # compute pairwise using broadcasting (may be large)
    # use efficient formula: ||a-b||^2 = |a|^2 + |b|^2 - 2 a.b
    a2 = np.sum(X24n ** 2, axis=1, keepdims=True)
    b2 = np.sum(X25n ** 2, axis=1, keepdims=True)
    dots = X24n.dot(X25n.T)
    sq = a2 + b2.T - 2.0 * dots
    sq[sq < 0] = 0.0
    inter_mean = float(np.mean(np.sqrt(sq)))
    # Save PCA scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(pcs24[:, 0], pcs24[:, 1], s=6, c='blue', label='2024')
    ax.scatter(pcs25[:, 0], pcs25[:, 1], s=6, c='red', label='2025')
    ax.set_title('PCA (2D) of embeddings (shared space)')
    ax.legend()
    fig.savefig(EMB_DIR / 'pca_2024_vs_2025.png', dpi=200)
    plt.close(fig)
    # Save centroid distances and stats
    emb_stats = {
        'centroid_distance': centroid_distance,
        'intra24': intra24,
        'intra25': intra25,
        'inter_mean_distance': inter_mean,
        'n_2024': int(len(X24n)),
        'n_2025': int(len(X25n)),
    }
    with open(EMB_DIR / 'embedding_stats.json', 'w') as f:
        json.dump(emb_stats, f, indent=2)
    # Save per-point PCA coords + season
    df_pca = pd.DataFrame(np.vstack([pcs24, pcs25]), columns=['pc1', 'pc2'])
    df_pca['season'] = ['2024'] * len(pcs24) + ['2025'] * len(pcs25)
    df_pca.to_csv(EMB_DIR / 'pca_coords.csv', index=False)
    results.update(emb_stats)
    return results


def pond_level_analysis(meta, vecs):
    if 'pond' not in meta.columns and 'pond_type' not in [c.lower() for c in meta.columns]:
        print('No pond metadata found; skipping pond-level analysis')
        return None
    # try common column names
    pond_col = 'pond' if 'pond' in meta.columns else None
    if pond_col is None:
        for c in meta.columns:
            if c.lower() == 'pond_type' or c.lower() == 'pond':
                pond_col = c
                break
    if pond_col is None:
        print('No pond column discovered; skipping pond-level analysis')
        return None
    meta2 = meta.copy()
    meta2['pond'] = meta2[pond_col].astype(str)
    res_tables = {}
    for season in ['2024', '2025']:
        sub = meta2[meta2['season'].astype(str) == season]
        ponds = sub['pond'].unique()
        records = []
        for p in ponds:
            inds = sub[sub['pond'] == p].index.values
            if len(inds) == 0:
                continue
            vecs_sub = vecs[inds]
            vecs_sub_n = vecs_sub.astype(float)
            n = np.linalg.norm(vecs_sub_n, axis=1, keepdims=True)
            n[n == 0] = 1.0
            vecs_sub_n = vecs_sub_n / n
            centroid = vecs_sub_n.mean(axis=0)
            var = float(np.mean(np.sum((vecs_sub_n - centroid) ** 2, axis=1)))
            records.append({'pond': p, 'n': int(len(inds)), 'variance': var})
        dfp = pd.DataFrame.from_records(records)
        dfp.to_csv(POND_DIR / f'pond_stats_{season}.csv', index=False)
        res_tables[season] = dfp
    # compute distances between matching ponds across seasons
    # align pond names
    merged = None
    if '2024' in res_tables and '2025' in res_tables:
        df24 = res_tables['2024'].set_index('pond')
        df25 = res_tables['2025'].set_index('pond')
        common = set(df24.index).intersection(set(df25.index))
        pond_comp = []
        for p in sorted(common):
            pond_comp.append({'pond': p, 'var_2024': float(df24.loc[p, 'variance']), 'var_2025': float(df25.loc[p, 'variance'])})
        if pond_comp:
            pd.DataFrame.from_records(pond_comp).to_csv(POND_DIR / 'pond_variance_comparison.csv', index=False)
    return res_tables


def assemble_summary(visual_stats, geom_stats_diff, emb_stats):
    summary = {}
    # visual_stats: dataframes mean brightness per season
    bmean24 = float(visual_stats['mean_brightness_2024'])
    bmean25 = float(visual_stats['mean_brightness_2025'])
    summary['brightness_diff'] = bmean25 - bmean24
    # geom_stats_diff: dict with carapace mean difference
    summary['carapace_mean_diff_px'] = geom_stats_diff.get('carapace_mean_diff_px', None)
    # embedding centroid distance
    summary['embedding_centroid_distance'] = emb_stats.get('centroid_distance', None)
    # intra vs inter variance ratio
    intra24 = emb_stats.get('intra24', None)
    intra25 = emb_stats.get('intra25', None)
    inter = emb_stats.get('inter_mean_distance', None)
    if intra24 is not None and intra25 is not None and inter is not None and inter > 0:
        summary['intra_mean'] = 0.5 * (intra24 + intra25)
        summary['intra_inter_ratio'] = summary['intra_mean'] / inter
    else:
        summary['intra_mean'] = None
        summary['intra_inter_ratio'] = None
    # save JSON
    with open(OUT_ROOT / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    print('Starting season shift analysis...')
    # 1) Visual distribution shift
    imgs24 = collect_images_for_dirs(DS_2024_IMG_DIRS)
    imgs25 = collect_images_for_dirs(DS_2025_IMG_DIRS)
    print(f'Found {len(imgs24)} images for 2024, {len(imgs25)} for 2025')
    if len(imgs24) > 0:
        df24, hist24 = compute_brightness_stats(imgs24)
    else:
        df24, hist24 = pd.DataFrame(), np.zeros(256)
    if len(imgs25) > 0:
        df25, hist25 = compute_brightness_stats(imgs25)
    else:
        df25, hist25 = pd.DataFrame(), np.zeros(256)
    # compute means for summary
    visual_stats = {'mean_brightness_2024': float(df24['mean_brightness'].mean()) if not df24.empty else None,
                    'mean_brightness_2025': float(df25['mean_brightness'].mean()) if not df25.empty else None}
    save_hist_and_box(df24, hist24, df25, hist25)

    # 2) Geometric distribution shift
    geom24 = compute_geometric_stats(DS_2024_IMG_DIRS, DS_2024_LABEL_DIRS, '2024')
    geom25 = compute_geometric_stats(DS_2025_IMG_DIRS, DS_2025_LABEL_DIRS, '2025')
    # compute carapace mean diff (px) where available
    car_mean24 = float(geom24['carapace_px'].dropna().mean()) if not geom24.empty and geom24['carapace_px'].notna().any() else None
    car_mean25 = float(geom25['carapace_px'].dropna().mean()) if not geom25.empty and geom25['carapace_px'].notna().any() else None
    geom_stats_diff = {'carapace_mean_2024_px': car_mean24, 'carapace_mean_2025_px': car_mean25}
    if car_mean24 is not None and car_mean25 is not None:
        geom_stats_diff['carapace_mean_diff_px'] = car_mean25 - car_mean24
    else:
        geom_stats_diff['carapace_mean_diff_px'] = None

    # 3) Embedding shift
    try:
        meta, vecs = load_embeddings_meta_and_vectors(EMB_META_PATH, EMB_VEC_PATH)
        emb_stats = embedding_space_analysis(meta, vecs)
    except Exception as e:
        warnings.warn(f'Embedding analysis failed: {e}')
        emb_stats = {}

    # 4) Pond-level
    pond_tables = pond_level_analysis(meta, vecs) if 'meta' in locals() else None

    # assemble summary JSON and print final console summary
    summary = assemble_summary(visual_stats, geom_stats_diff, emb_stats)

    print('\n=== Final Summary ===')
    vb = summary.get('brightness_diff', None)
    if vb is None:
        print('Visual shift: insufficient data')
    else:
        coh = None
        # compute simple effect size using brightness pooled std
        try:
            pooled = np.sqrt(((df24['std_brightness'].dropna() ** 2).mean() + (df25['std_brightness'].dropna() ** 2).mean()) / 2.0)
            if pooled > 0:
                coh = vb / pooled
        except Exception:
            coh = None
        if coh is None:
            print(f'Brightness difference (2025 - 2024): {vb:.3f}')
        else:
            print(f'Brightness difference (2025 - 2024): {vb:.3f} (Cohen d ~ {coh:.3f})')

    car_diff = summary.get('carapace_mean_diff_px', None)
    if car_diff is None:
        print('Geometric shift (carapace): insufficient data')
    else:
        print(f'Carapace mean difference in pixels (2025 - 2024): {car_diff:.3f}')

    cent_dist = summary.get('embedding_centroid_distance', None)
    if cent_dist is None:
        print('Embedding shift: insufficient data')
    else:
        print(f'Embedding centroid distance (2024 vs 2025): {cent_dist:.6f}')
        intra_ratio = summary.get('intra_inter_ratio', None)
        if intra_ratio is not None:
            print(f'Intra-mean / Inter-mean ratio: {intra_ratio:.4f}')

    # high-level judgments
    print('\nJudgments:')
    if vb is not None and abs(vb) > 5.0:
        print(' - Visual shift: YES (mean brightness differs substantially)')
    else:
        print(' - Visual shift: unclear or small')
    if car_diff is not None and abs(car_diff) > 5.0:
        print(' - Geometric shift: YES (mean carapace length differs)')
    else:
        print(' - Geometric shift: unclear or small')
    if cent_dist is not None and cent_dist > 0.05:
        print(' - Embedding shift: YES (centroids separated)')
    else:
        print(' - Embedding shift: small or negligible')

    if pond_tables is not None:
        print('Pond-level tables saved under', POND_DIR)

    print('\nDone. All outputs are under', OUT_ROOT)


if __name__ == '__main__':
    main()

