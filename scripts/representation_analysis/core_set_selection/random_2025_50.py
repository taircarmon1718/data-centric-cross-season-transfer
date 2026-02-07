"""
Stratified random 50% sampling for Season 2025 images.

Saves:
  outputs/rep_analysis/core_set_selection/random_2025_50/core_set.csv

Behavior:
- Attempts to load metadata from outputs/rep_analysis/by_pond/embeddings_by_pond.csv (preferred)
- Fallback to outputs/rep_analysis/embeddings_meta.csv
- Fallback to scanning datasets/train_on_2025_all/images recursively
- Stratifies by 'pond' if available, otherwise single stratum
- Computes 50% quota per stratum using fair rounding (largest remainder)
- Deterministically samples with fixed seed (seed=0)
- Saves CSV with columns: image_path, season (if available), pond (if available)

Note on exact 50%: If total number is odd the script selects floor(N/2) images (deterministic). The per-stratum allocation uses largest-remainder method to reach the total quota.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import math
import csv

# Config
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / 'outputs' / 'rep_analysis' / 'core_set_selection' / 'random_2025_50'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / 'core_set.csv'
BY_POND_CSV = PROJECT_ROOT / 'outputs' / 'rep_analysis' / 'by_pond' / 'embeddings_by_pond.csv'
META_CSV = PROJECT_ROOT / 'outputs' / 'rep_analysis' / 'embeddings_meta.csv'
DATA_2025 = PROJECT_ROOT / 'datasets' / 'train_on_2025_all'
SEED = 0

rng = np.random.RandomState(SEED)


def load_by_pond():
    p = BY_POND_CSV
    if p.exists():
        try:
            df = pd.read_csv(p)
            # require image_path and season/pond optionally
            if 'image_path' not in df.columns:
                return None
            return df
        except Exception:
            return None
    return None


def load_meta():
    p = META_CSV
    if p.exists():
        try:
            df = pd.read_csv(p)
            if 'image_path' not in df.columns:
                return None
            return df
        except Exception:
            return None
    return None


def scan_dataset_images():
    img_dir = DATA_2025 / 'images'
    val_dir = DATA_2025 / 'val' / 'images'
    paths = []
    if val_dir.exists():
        for p in sorted(val_dir.rglob('*.jpg')):
            rel = p.relative_to(PROJECT_ROOT)
            paths.append(str(rel))
    if img_dir.exists():
        for p in sorted(img_dir.rglob('*.jpg')):
            rel = p.relative_to(PROJECT_ROOT)
            paths.append(str(rel))
    # dedupe maintain order
    seen = set()
    uniq = []
    for x in paths:
        if x not in seen:
            uniq.append(x); seen.add(x)
    df = pd.DataFrame({'image_path': uniq})
    df['season'] = '2025'
    df['pond'] = 'unknown'
    return df


def allocate_quota(counts, total_quota):
    # counts: dict pond->n
    # compute ideal quotas
    items = list(counts.items())
    ponds = [k for k,_ in items]
    ns = np.array([v for _,v in items], dtype=float)
    ideals = ns * 0.5
    floors = np.floor(ideals).astype(int)
    remainder = int(total_quota - floors.sum())
    if remainder <= 0:
        return dict(zip(ponds, floors.tolist()))
    fracs = ideals - floors
    order = np.argsort(-fracs)
    alloc = floors.copy()
    i = 0
    while remainder > 0 and i < len(order):
        alloc[order[i]] += 1
        remainder -= 1
        i += 1
        if i == len(order) and remainder > 0:
            # distribute remaining 1-by-1 from largest groups
            sizes_order = np.argsort(-ns)
            j = 0
            while remainder > 0:
                alloc[sizes_order[j % len(sizes_order)]] += 1
                remainder -= 1
                j += 1
    return dict(zip(ponds, alloc.tolist()))


def stratified_sample(df, seed=0):
    # df must contain 'image_path' and optionally 'pond' and 'season'
    df2 = df.copy()
    if 'pond' not in df2.columns:
        df2['pond'] = 'unknown'
    # filter season==2025 if present
    if 'season' in df2.columns:
        df2 = df2[df2['season'].astype(str) == '2025']
    # compute total and quota
    N = len(df2)
    if N == 0:
        raise ValueError('No 2025 images found')
    total_quota = N // 2  # floor if odd
    # counts per pond
    counts = df2['pond'].value_counts().to_dict()
    # allocate quotas per pond fairly
    quota_map = allocate_quota(counts, total_quota)
    selected_rows = []
    for pond, q in quota_map.items():
        group = df2[df2['pond'] == pond]
        if len(group) == 0 or q <= 0:
            continue
        # deterministic shuffle
        idxs = np.arange(len(group))
        rng.shuffle(idxs)
        take = idxs[:q]
        selected_rows.append(group.iloc[take])
    if not selected_rows:
        return pd.DataFrame(columns=df2.columns)
    sel_df = pd.concat(selected_rows, ignore_index=True)
    # ensure exact quota
    if len(sel_df) != total_quota:
        # if mismatch due to rounding, adjust by random sampling of leftover
        need = total_quota - len(sel_df)
        available = df2[~df2['image_path'].isin(sel_df['image_path'])]
        if need > 0 and len(available) > 0:
            avail_idxs = np.arange(len(available))
            rng.shuffle(avail_idxs)
            sel_extra = available.iloc[avail_idxs[:need]]
            sel_df = pd.concat([sel_df, sel_extra], ignore_index=True)
        elif need < 0:
            # reduce
            sel_df = sel_df.sample(n=total_quota, random_state=seed)
    # final deterministic sort by image_path
    sel_df = sel_df.sort_values('image_path').reset_index(drop=True)
    return sel_df


def main():
    df = load_by_pond()
    source = 'by_pond'
    if df is None:
        df = load_meta()
        source = 'meta'
    if df is None:
        df = scan_dataset_images()
        source = 'scan'
    print(f"Loaded metadata from: {source}; total rows={len(df)}")
    sel = stratified_sample(df, seed=SEED)
    print(f"Selected {len(sel)} images (target {len(df)//2})")
    # save CSV with pond/season if available
    outcols = ['image_path']
    if 'season' in sel.columns:
        outcols.append('season')
    if 'pond' in sel.columns:
        outcols.append('pond')
    sel.to_csv(OUT_CSV, columns=outcols, index=False)
    print(f"Saved random core-set CSV to: {OUT_CSV}")

if __name__ == '__main__':
    main()

