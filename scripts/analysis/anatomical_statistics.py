#!/usr/bin/env python3
"""
anatomical_statistics.py

Analyze anatomical geometric statistics of prawn keypoints for Season 2024 and Season 2025,
compare distributions, run statistical tests, and generate plots.

Saves outputs to: outputs/anatomical_statistics/

Usage:
    python scripts/analysis/anatomical_statistics.py

Dependencies: numpy, pandas, scipy, matplotlib, seaborn, pathlib, tqdm
"""
from pathlib import Path
import math
import sys
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker
from tqdm import tqdm

# ----------------- Configuration -----------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED = 0
np.random.seed(SEED)

DATA_2024 = PROJECT_ROOT / 'datasets' / 'train_on_all'
DATA_2025 = PROJECT_ROOT / 'datasets' / 'train_on_2025_all'

LABEL_DIRS = {
    '2024': [DATA_2024 / 'labels', DATA_2024 / 'val' / 'labels'],
    '2025': [DATA_2025 / 'labels', DATA_2025 / 'val' / 'labels'],
}
IMAGE_DIRS = {
    '2024': [DATA_2024 / 'images', DATA_2024 / 'val' / 'images'],
    '2025': [DATA_2025 / 'images', DATA_2025 / 'val' / 'images'],
}

OUT_DIR = PROJECT_ROOT / 'outputs' / 'anatomical_statistics'
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# Keypoint indices: 0 carapace,1 eyes,2 rostrum,3 tail
KP_IND = {'carapace': 0, 'eyes': 1, 'rostrum': 2, 'tail': 3}

# ----------------- Helper functions -----------------

def find_image_for_label(label_path: Path, season):
    stem = label_path.stem
    for root in IMAGE_DIRS[season]:
        if not root.exists():
            continue
        # direct check
        for ext in IMAGE_EXTS:
            cand = root / (stem + ext)
            if cand.exists():
                return cand
        # recursive deterministic
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.stem == stem:
                return p
    return None


def parse_yolo_pose_label(label_path: Path):
    """Parse YOLO pose label first line and return list of keypoints (x,y) normalized or absolute.
    Returns dict: {'kpts':[ (x,y)... ], 'is_normalized':bool} or None on error.
    """
    try:
        txt = label_path.read_text().strip()
        if not txt:
            return None
        line = txt.splitlines()[0].strip()
        parts = line.split()
        vals = [float(x) for x in parts]
        if len(vals) < 5 + 8:
            return None
        kps = vals[5:5 + 8]
        kpts = [(kps[i], kps[i+1]) for i in range(0, len(kps), 2)]
        if len(kpts) < 4:
            return None
        flat = np.array(kps)
        is_norm = False
        try:
            if flat.max() <= 1.5:
                is_norm = True
        except Exception:
            is_norm = False
        return {'kpts': kpts[:4], 'is_normalized': is_norm}
    except Exception:
        return None


def image_size(path: Path):
    try:
        # use matplotlib to read image size without adding PIL explicit import
        import matplotlib.image as mpimg
        arr = mpimg.imread(str(path))
        if hasattr(arr, 'shape'):
            h, w = arr.shape[0], arr.shape[1]
            return int(w), int(h)
    except Exception:
        pass
    return None, None


def to_pixel_coords(kpts, is_norm, W, H):
    if is_norm:
        return [(float(x) * W, float(y) * H) for (x, y) in kpts]
    else:
        return [(float(x), float(y)) for (x, y) in kpts]


def compute_metrics_from_kpts(kpts_px):
    # kpts_px is list of 4 tuples: indices 0..3
    (x0, y0) = kpts_px[0]  # carapace
    (x1, y1) = kpts_px[1]  # eyes
    (x2, y2) = kpts_px[2]  # rostrum
    (x3, y3) = kpts_px[3]  # tail
    def dist(a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])
    d_rostrum_carapace = dist((x2,y2),(x0,y0))
    d_eyes_carapace = dist((x1,y1),(x0,y0))
    d_eyes_rostrum = dist((x1,y1),(x2,y2))
    d_rostrum_tail = dist((x2,y2),(x3,y3))
    ratio = d_rostrum_carapace / d_rostrum_tail if d_rostrum_tail > 0 else float('nan')
    # bend angle between vectors (carapace->rostrum) and (carapace->tail)
    v1 = (x2 - x0, y2 - y0)
    v2 = (x3 - x0, y3 - y0)
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])
    if n1 == 0 or n2 == 0:
        angle = float('nan')
    else:
        cosv = max(-1.0, min(1.0, dot / (n1 * n2)))
        angle = math.degrees(math.acos(cosv))
    return {
        'd_rostrum_carapace': d_rostrum_carapace,
        'd_eyes_carapace': d_eyes_carapace,
        'd_eyes_rostrum': d_eyes_rostrum,
        'd_rostrum_tail': d_rostrum_tail,
        'ratio': ratio,
        'bend_angle': angle,
    }


def aggregate_stats(arr: np.ndarray):
    arr_clean = arr[np.isfinite(arr)]
    if arr_clean.size == 0:
        return {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan, 'cv': np.nan}
    mean = float(np.mean(arr_clean))
    std = float(np.std(arr_clean, ddof=1)) if arr_clean.size > 1 else 0.0
    mn = float(np.min(arr_clean))
    mx = float(np.max(arr_clean))
    cv = float(std / mean) if mean != 0 else float('nan')
    return {'mean': mean, 'std': std, 'min': mn, 'max': mx, 'cv': cv}

# ----------------- Main analysis -----------------

def analyze_season(season_label_dirs, season_image_dirs):
    records = []
    for lbl_dir in season_label_dirs:
        if not lbl_dir.exists():
            continue
        label_files = sorted(lbl_dir.rglob('*.txt'))
        for lbl in tqdm(label_files, desc=f'Parsing labels in {lbl_dir}'):
            parsed = parse_yolo_pose_label(lbl)
            if parsed is None:
                continue
            img = find_image_for_label(lbl, '2024' if season_label_dirs == LABEL_DIRS['2024'] else '2025')
            if img is None or not img.exists():
                continue
            W, H = image_size(img)
            if W is None or H is None:
                continue
            kpts_px = to_pixel_coords(parsed['kpts'], parsed['is_normalized'], W, H)
            metrics = compute_metrics_from_kpts(kpts_px)
            metrics.update({'image': str(img.resolve()), 'label': str(lbl.resolve())})
            records.append(metrics)
    df = pd.DataFrame(records)
    return df


def compare_and_test(df24, df25, out_dir: Path):
    metrics = ['d_rostrum_carapace', 'd_eyes_carapace', 'd_eyes_rostrum', 'd_rostrum_tail', 'ratio', 'bend_angle']
    stats = {'metric': [], 'season': [], 'mean': [], 'std': [], 'min': [], 'max': [], 'cv': []}
    test_results = []
    for m in metrics:
        arr24 = df24[m].to_numpy(dtype=np.float64) if m in df24.columns else np.array([])
        arr25 = df25[m].to_numpy(dtype=np.float64) if m in df25.columns else np.array([])
        s24 = aggregate_stats(arr24)
        s25 = aggregate_stats(arr25)
        for season, s in (('2024', s24), ('2025', s25)):
            stats['metric'].append(m)
            stats['season'].append(season)
            stats['mean'].append(s['mean'])
            stats['std'].append(s['std'])
            stats['min'].append(s['min'])
            stats['max'].append(s['max'])
            stats['cv'].append(s['cv'])
        # t-test (Welch)
        try:
            tstat, pval_t = ttest_ind(arr24[np.isfinite(arr24)], arr25[np.isfinite(arr25)], equal_var=False, nan_policy='omit')
        except Exception:
            tstat, pval_t = float('nan'), float('nan')
        try:
            ksstat, pval_ks = ks_2samp(arr24[np.isfinite(arr24)], arr25[np.isfinite(arr25)])
        except Exception:
            ksstat, pval_ks = float('nan'), float('nan')
        test_results.append({'metric': m, 't_stat': float(tstat), 't_pvalue': float(pval_t), 'ks_stat': float(ksstat), 'ks_pvalue': float(pval_ks)})
        # plots for this metric
        out_hist = out_dir / f'{m}_hist.png'
        out_box = out_dir / f'{m}_box.png'
        out_kde = out_dir / f'{m}_kde.png'
        plt.figure(figsize=(8,5))
        if arr24.size>0:
            plt.hist(arr24[~np.isnan(arr24)], bins=60, alpha=0.5, density=True)
        if arr25.size>0:
            plt.hist(arr25[~np.isnan(arr25)], bins=60, alpha=0.5, density=True)
        plt.title(f'Histogram overlay: {m} (2024 vs 2025)')
        plt.xlabel(m)
        plt.ylabel('Density')
        plt.legend(['2024','2025'])
        plt.tight_layout()
        plt.savefig(out_hist, dpi=200)
        plt.close()

        # boxplot
        plt.figure(figsize=(6,5))
        data = []
        labels = []
        if arr24.size>0:
            data.append(arr24[~np.isnan(arr24)])
            labels.append('2024')
        if arr25.size>0:
            data.append(arr25[~np.isnan(arr25)])
            labels.append('2025')
        if data:
            plt.boxplot(data, labels=labels)
            plt.title(f'Boxplot: {m}')
            plt.ylabel(m)
            plt.tight_layout()
            plt.savefig(out_box, dpi=200)
            plt.close()

        # KDE using seaborn
        plt.figure(figsize=(8,5))
        try:
            if arr24.size>0:
                sns.kdeplot(arr24[~np.isnan(arr24)], fill=True)
            if arr25.size>0:
                sns.kdeplot(arr25[~np.isnan(arr25)], fill=True)
            plt.title(f'KDE: {m}')
            plt.xlabel(m)
            plt.ylabel('Density')
            plt.legend(['2024','2025'])
            plt.tight_layout()
            plt.savefig(out_kde, dpi=200)
            plt.close()
        except Exception:
            # seaborn KDE can fail for small samples; ignore
            pass

    df_stats = pd.DataFrame(stats)
    df_tests = pd.DataFrame(test_results)
    df_stats.to_csv(out_dir / 'summary_stats.csv', index=False)
    df_tests.to_csv(out_dir / 'stat_tests.csv', index=False)
    # combined JSON
    summary = {'stats': df_stats.to_dict(orient='records'), 'tests': df_tests.to_dict(orient='records')}
    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    return df_stats, df_tests


def main():
    print('Starting anatomical statistics analysis')
    df24 = pd.DataFrame()
    df25 = pd.DataFrame()
    # analyze 2024
    print('Analyzing Season 2024...')
    recs24 = []
    for lbl_dir in LABEL_DIRS['2024']:
        if not lbl_dir.exists():
            print('Skipping missing dir', lbl_dir)
            continue
        for lbl in tqdm(sorted(lbl_dir.rglob('*.txt')), desc='2024 labels'):
            parsed = parse_yolo_pose_label(lbl)
            if parsed is None:
                continue
            img = find_image_for_label(lbl, '2024')
            if img is None or not img.exists():
                continue
            W, H = image_size(img)
            if W is None:
                continue
            kpts_px = to_pixel_coords(parsed['kpts'], parsed['is_normalized'], W, H)
            metrics = compute_metrics_from_kpts(kpts_px)
            metrics.update({'image': str(img.resolve()), 'label': str(lbl.resolve())})
            recs24.append(metrics)
    if recs24:
        df24 = pd.DataFrame(recs24)

    # analyze 2025
    print('Analyzing Season 2025...')
    recs25 = []
    for lbl_dir in LABEL_DIRS['2025']:
        if not lbl_dir.exists():
            print('Skipping missing dir', lbl_dir)
            continue
        for lbl in tqdm(sorted(lbl_dir.rglob('*.txt')), desc='2025 labels'):
            parsed = parse_yolo_pose_label(lbl)
            if parsed is None:
                continue
            img = find_image_for_label(lbl, '2025')
            if img is None or not img.exists():
                continue
            W, H = image_size(img)
            if W is None:
                continue
            kpts_px = to_pixel_coords(parsed['kpts'], parsed['is_normalized'], W, H)
            metrics = compute_metrics_from_kpts(kpts_px)
            metrics.update({'image': str(img.resolve()), 'label': str(lbl.resolve())})
            recs25.append(metrics)
    if recs25:
        df25 = pd.DataFrame(recs25)

    # Save per-image metrics
    if not df24.empty:
        df24.to_csv(OUT_DIR / 'per_image_metrics_2024.csv', index=False)
    if not df25.empty:
        df25.to_csv(OUT_DIR / 'per_image_metrics_2025.csv', index=False)

    # Compare and test
    df_stats, df_tests = compare_and_test(df24, df25, OUT_DIR)

    # Print side-by-side stats
    if not df_stats.empty:
        print('\nSide-by-side statistics:')
        print(df_stats.pivot(index='metric', columns='season', values='mean'))

    print('\nStatistical test results saved to', OUT_DIR)
    print('Done.')


if __name__ == '__main__':
    main()

