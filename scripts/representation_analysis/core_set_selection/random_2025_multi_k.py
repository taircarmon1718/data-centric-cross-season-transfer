"""
Generate multiple deterministic random core-set CSVs for Season 2025
by sampling k-percent of available images (train + val).

Saves per-k CSVs to:
  outputs/rep_analysis/core_set_selection/random_2025_kXX/core_set.csv

Behavior and constraints:
- Detects PROJECT_ROOT robustly.
- Scans only:
    datasets/train_on_2025_all/images
    datasets/train_on_2025_all/val/images
  for .jpg images.
- No metadata CSVs or embeddings are used.
- Deterministic sampling using fixed SEED.
- Uses pathlib and clean modular code.

CSV columns:
- image_path (relative to PROJECT_ROOT)
- split (train or val)

All comments are in English.
"""
from pathlib import Path
import numpy as np
import csv
import sys

# Configuration
K_LIST = [1, 2, 5, 10, 20, 50]
SEED = 0

# Project root detection (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_2025 = PROJECT_ROOT / 'datasets' / 'train_on_2025_all'
OUT_BASE = PROJECT_ROOT / 'outputs' / 'rep_analysis' / 'core_set_selection'

RNG = np.random.RandomState(SEED)


def collect_image_paths():
    """Collect .jpg image paths from train and val folders.

    Returns a list of tuples (relative_path_str, split) where split is 'train' or 'val'.
    Paths are relative to PROJECT_ROOT and deduplicated while preserving order.
    """
    paths = []
    # val first to preserve consistent ordering if user expects val before train
    val_dir = DATA_2025 / 'val' / 'images'
    train_dir = DATA_2025 / 'images'

    if val_dir.exists():
        for p in sorted(val_dir.rglob('*.jpg')):
            rel = p.relative_to(PROJECT_ROOT)
            paths.append((str(rel), 'val'))
    if train_dir.exists():
        for p in sorted(train_dir.rglob('*.jpg')):
            rel = p.relative_to(PROJECT_ROOT)
            paths.append((str(rel), 'train'))

    # dedupe while preserving order
    seen = set()
    uniq = []
    for p, s in paths:
        if p not in seen:
            uniq.append((p, s)); seen.add(p)
    return uniq


def sample_for_k(all_items, k_percent):
    """Deterministically sample floor(k_percent% * N) items from all_items.

    all_items: list of (path, split)
    k_percent: integer percentage (e.g. 5)
    Returns a list of selected items in stable sorted order.
    """
    N = len(all_items)
    n_k = int(np.floor(k_percent / 100.0 * N))
    if n_k <= 0:
        return []
    # create deterministic permutation and take first n_k
    idxs = np.arange(N)
    RNG.shuffle(idxs)
    chosen_idxs = np.sort(idxs[:n_k])  # sort to keep stable order in CSV
    selected = [all_items[i] for i in chosen_idxs]
    return selected


def format_k_dirname(k):
    """Return folder name like k01, k05, k20."""
    # zero pad to two digits
    return f"random_2025_k{int(k):02d}"


def save_core_csv(selected, out_csv_path):
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with out_csv_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'split'])
        for p, s in selected:
            writer.writerow([p, s])


def main():
    all_items = collect_image_paths()
    N = len(all_items)
    if N == 0:
        print(f"No images found under {DATA_2025}. Nothing to do.")
        return 1
    print(f"Found {N} unique .jpg images under {DATA_2025} (train+val)")

    for k in K_LIST:
        sel = sample_for_k(all_items, k)
        dirname = format_k_dirname(k)
        out_dir = OUT_BASE / dirname
        out_csv = out_dir / 'core_set.csv'
        save_core_csv(sel, out_csv)
        print(f"k={k}%: selected {len(sel)} / {N} -> {out_csv}")

    return 0


if __name__ == '__main__':
    sys.exit(main())

