#!/usr/bin/env python3
"""
season_morphology_variance.py

Compare morphological variance between Season 2024 and Season 2025.

- Detect PROJECT_ROOT automatically
- Load YOLO-pose labels (4 keypoints: carapace, eyes, rostrum, tail)
- Compute per-image: total length (rostrum-tail), carapace length (carapace-eyes), ratio, pose angle
- Aggregate statistics per season and plot distributions
- Perform KS tests between seasons for each metric
- Print clean comparison table and conclusions

Usage: python scripts/analysis/season_morphology_variance.py

Requirements: numpy, pandas, scipy, matplotlib, pathlib, tqdm
"""
from pathlib import Path
import math
import sys
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

# -------------------- Configuration --------------------
# Deterministic seed for any sampling (not required here but keep deterministic)
SEED = 0
np.random.seed(SEED)

# PROJECT_ROOT detection
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Data paths
DATA_2024 = PROJECT_ROOT / 'datasets' / 'train_on_2024_all'
DATA_2025 = PROJECT_ROOT / 'datasets' / 'train_on_2025_all'

LABEL_DIRS = {
    '2024': [DATA_2024 / 'labels', DATA_2024 / 'val' / 'labels'],
    '2025': [DATA_2025 / 'labels', DATA_2025 / 'val' / 'labels'],
}
IMAGE_DIRS = {
    '2024': [DATA_2024 / 'images', DATA_2024 / 'val' / 'images'],
    '2025': [DATA_2025 / 'images', DATA_2025 / 'val' / 'images'],
}

OUTPUT_DIR = PROJECT_ROOT / 'scripts' / 'analysis' / 'plots'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# -------------------- Helpers --------------------

def find_image_for_label(label_path: Path, image_roots):
    """Find corresponding image file for a label by matching stem in given image roots.
    Returns Path or None.
    """
    stem = label_path.stem
    # try same relative structure if possible
    for root in image_roots:
        if not root.exists():
            continue
        # try direct candidate with same relative path
        candidate = root / (label_path.name.replace('.txt', '.jpg'))
        if candidate.exists():
            return candidate
        # search by stem for allowed extensions
        for ext in IMAGE_EXTS:
            cand = root / (stem + ext)
            if cand.exists():
                return cand
        # recursive search deterministic
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.stem == stem:
                return p
    return None


def parse_label_file(label_path: Path, image_size=None):
    """Parse YOLO-pose label and return list of keypoints in absolute pixel coords if possible.
    Returns list of 4 (x,y) tuples or None if parsing fails or insufficient keypoints.
    image_size: (W,H) to convert normalized coords to pixels; if None and coords normalized we return normalized values.
    """
    try:
        text = label_path.read_text().strip()
        if not text:
            return None
        # take first line
        line = text.splitlines()[0].strip()
        parts = line.split()
        vals = [float(x) for x in parts]
        if len(vals) < 5 + 8:  # class + bbox(4) + 4kpts*2
            return None
        kps = vals[5:5 + 8]
        kpts = []
        for i in range(0, len(kps), 2):
            x = kps[i]
            y = kps[i + 1]
            kpts.append((x, y))
        if len(kpts) < 4:
            return None
        # detect normalized vs absolute: if max coordinate <= 1.5 treat as normalized
        flat = np.array(kps)
        if flat.max() <= 1.5:
            # normalized; convert using image_size if available
            if image_size is None:
                # return normalized values
                return kpts
            W, H = image_size
            abs_kpts = [(float(x * W), float(y * H)) for (x, y) in kpts]
            return abs_kpts
        else:
            # absolute coords
            return [(float(x), float(y)) for (x, y) in kpts]
    except Exception:
        return None


def compute_metrics_for_season(season_label_dirs, season_image_dirs):
    """Iterate labels and compute metrics arrays for season.
    Returns dict of lists: total_len, carapace_len, ratio, angle
    """
    totals = []
    carapaces = []
    ratios = []
    angles = []
    n_labels = 0
    n_skipped = 0
    for lbl_root in season_label_dirs:
        if not lbl_root.exists():
            continue
        for lbl in sorted(lbl_root.rglob('*.txt')):
            n_labels += 1
            # find image
            img_path = find_image_for_label(lbl, season_image_dirs)
            if img_path is None or not img_path.exists():
                n_skipped += 1
                continue
            # get image size
            try:
                from PIL import Image
                with Image.open(img_path) as im:
                    W, H = im.size
            except Exception:
                W, H = None, None
            kpts = parse_label_file(lbl, image_size=(W, H))
            if kpts is None or len(kpts) < 4:
                n_skipped += 1
                continue
            # ensure numeric
            try:
                (x0, y0) = kpts[0]
                (x1, y1) = kpts[1]
                (x2, y2) = kpts[2]
                (x3, y3) = kpts[3]
            except Exception:
                n_skipped += 1
                continue
            # compute lengths
            car = math.hypot(x1 - x0, y1 - y0)
            tot = math.hypot(x3 - x2, y3 - y2)
            if math.isfinite(car) and math.isfinite(tot) and tot > 0:
                ratio = car / tot
            else:
                ratio = float('nan')
            # angle of body axis rostrum(2) -> tail(3)
            dx = x3 - x2
            dy = y3 - y2
            angle = math.degrees(math.atan2(dy, dx))
            # normalize angle to [-180,180)
            if angle >= 180.0:
                angle -= 360.0
            if angle < -180.0:
                angle += 360.0
            totals.append(car if False else tot)  # tot is total length
            carapaces.append(car)
            ratios.append(ratio)
            angles.append(angle)
    results = {
        'n_labels_found': n_labels,
        'n_skipped': n_skipped,
        'total_length': np.array(totals, dtype=np.float64),
        'carapace_length': np.array(carapaces, dtype=np.float64),
        'ratio': np.array(ratios, dtype=np.float64),
        'angle': np.array(angles, dtype=np.float64),
    }
    return results


def compute_stats(arr: np.ndarray):
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {k: float('nan') for k in ['mean', 'std', 'var', 'min', 'max', 'iqr']}
    q75 = np.percentile(arr, 75)
    q25 = np.percentile(arr, 25)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        'var': float(np.var(arr, ddof=1)) if arr.size > 1 else 0.0,
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'iqr': float(q75 - q25),
    }

# -------------------- Main --------------------

def main():
    print('PROJECT_ROOT detected as', PROJECT_ROOT)
    print('Collecting and computing metrics for seasons...')

    res2024 = compute_metrics_for_season(LABEL_DIRS['2024'], IMAGE_DIRS['2024'])
    res2025 = compute_metrics_for_season(LABEL_DIRS['2025'], IMAGE_DIRS['2025'])

    # prepare stats
    stats = {}
    for name, res in (('2024', res2024), ('2025', res2025)):
        stats[name] = {
            'n_labels_found': int(res['n_labels_found']),
            'n_skipped': int(res['n_skipped']),
            'total_length_stats': compute_stats(res['total_length']),
            'carapace_length_stats': compute_stats(res['carapace_length']),
            'ratio_stats': compute_stats(res['ratio']),
            'angle_stats': compute_stats(res['angle']),
        }

    # KS tests
    ks_results = {}
    metrics = ['total_length', 'carapace_length', 'ratio', 'angle']
    for m in metrics:
        a = res2024[m]
        b = res2025[m]
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if a.size >= 2 and b.size >= 2:
            stat, pval = ks_2samp(a, b)
        else:
            stat, pval = float('nan'), float('nan')
        ks_results[m] = {'ks_stat': float(stat), 'p_value': float(pval)}

    # Save plots: three figures
    # Total length overlay
    def save_hist_overlay(arr1, arr2, label1, label2, title, outpath, bins=60):
        plt.figure(figsize=(8,6))
        plt.hist(arr1[np.isfinite(arr1)], bins=bins, alpha=0.5, label=label1, density=True)
        plt.hist(arr2[np.isfinite(arr2)], bins=bins, alpha=0.5, label=label2, density=True)
        plt.title(title)
        plt.xlabel(title)
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()

    save_hist_overlay(res2024['total_length'], res2025['total_length'], '2024', '2025', 'Total length', OUTPUT_DIR / 'total_length.png')
    save_hist_overlay(res2024['carapace_length']/ (res2024['total_length'] + 1e-12), res2025['carapace_length']/(res2025['total_length'] + 1e-12), '2024', '2025', 'Carapace/Total ratio', OUTPUT_DIR / 'ratio.png')
    save_hist_overlay(res2024['angle'], res2025['angle'], '2024', '2025', 'Pose angle (deg)', OUTPUT_DIR / 'angle.png')

    # Print comparison table
    rows = []
    for name in ('2024', '2025'):
        s = stats[name]
        row = {
            'season': name,
            'n_labels': s['n_labels_found'],
            'n_skipped': s['n_skipped'],
            'total_mean': s['total_length_stats']['mean'],
            'total_std': s['total_length_stats']['std'],
            'car_mean': s['carapace_length_stats']['mean'],
            'car_std': s['carapace_length_stats']['std'],
            'ratio_mean': s['ratio_stats']['mean'],
            'ratio_std': s['ratio_stats']['std'],
            'angle_mean': s['angle_stats']['mean'],
            'angle_std': s['angle_stats']['std'],
        }
        rows.append(row)
    df_table = pd.DataFrame(rows)

    print('\n=== Statistical comparison (KS test p-values) ===')
    for m in metrics:
        res = ks_results[m]
        print(f'{m}: ks_stat={res["ks_stat"]:.6f} p-value={res["p_value"]:.6g}')

    # Conclusions
    print('\n=== Conclusions ===')
    alpha = 0.05
    for m in metrics:
        p = ks_results[m]['p_value']
        if math.isnan(p):
            print(f'{m}: insufficient data for KS test')
        elif p < alpha:
            print(f'{m}: distributions differ significantly (p={p:.3g})')
        else:
            print(f'{m}: no significant difference (p={p:.3g})')
    # Variance comparison
    def higher_variance(name):
        return stats['2025'][name]['var'] > stats['2024'][name]['var']
    for metric_key, label in (('total_length_stats','Total length'), ('carapace_length_stats','Carapace length'), ('ratio_stats','Ratio'), ('angle_stats','Angle')):
        var24 = stats['2024'][metric_key]['var']
        var25 = stats['2025'][metric_key]['var']
        if math.isnan(var24) or math.isnan(var25):
            print(f'{label}: insufficient data')
        else:
            if var25 > var24:
                print(f'{label}: Season 2025 has higher variance ({var25:.4f} > {var24:.4f})')
            else:
                print(f'{label}: Season 2025 does not have higher variance ({var25:.4f} <= {var24:.4f})')

    # Print comparison table
    print('\n=== Comparison table ===')
    print(df_table.to_string(index=False))


if __name__ == '__main__':
    main()

