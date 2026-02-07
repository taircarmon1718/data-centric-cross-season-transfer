"""
Create adaptive core-sets for Season 2025 using Farthest-Point Sampling with elbow stopping.

Produces two runs (identical logic, different embedding spaces):
 - YOLO-only (uses outputs/rep_analysis/embeddings_vectors.npy + embeddings_meta.csv)
 - Fused (uses outputs/rep_analysis/geom_fusion_2024/fused_embeddings_2024.npy and outputs/rep_analysis/geom_fusion_2025/fused_embeddings_2025.npy and aligned_records_*.json)

Saves outputs under:
 outputs/rep_analysis/core_set_selection/yolo_elbow/
 outputs/rep_analysis/core_set_selection/fused_elbow/

Each contains:
 - core_set.csv (image_path, season, selection_order, distance_at_selection)
 - selection_log.csv (iteration, selected_image, R_t, Delta_t, stop_flag)
 - elbow_curve.png

Algorithm parameters (defaults):
 - seed_N = 10
 - knn_k = 5 (for computing distance to 2024 for seed selection)
 - eps = 0.01
 - K_window = 5
 - distance metric: Euclidean

Deterministic behavior ensured by deterministic tie-breaking and fixed seeds where relevant.

Usage:
 python scripts/representation_analysis/core_set_selection/create_elbow_core_sets.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import json
import math
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Config
PROJECT_ROOT = Path(__file__).resolve().parents[3]
print(f"[DEBUG] PROJECT_ROOT resolved to: {PROJECT_ROOT}")
OUT_BASE = PROJECT_ROOT / 'outputs' / 'rep_analysis' / 'core_set_selection'
OUT_BASE.mkdir(parents=True, exist_ok=True)
YOLO_EMB_PATH = PROJECT_ROOT / 'outputs' / 'rep_analysis' / 'embeddings_vectors.npy'
YOLO_META_CSV = PROJECT_ROOT / 'outputs' / 'rep_analysis' / 'embeddings_meta.csv'

FUSED_2024_EMB = PROJECT_ROOT / 'outputs' / 'rep_analysis' / 'geom_fusion_2024' / 'fused_embeddings_2024.npy'
FUSED_2024_RECS = PROJECT_ROOT / 'outputs' / 'rep_analysis' / 'geom_fusion_2024' / 'aligned_records_2024.json'
FUSED_2025_EMB = PROJECT_ROOT / 'outputs' / 'rep_analysis' / 'geom_fusion_2025' / 'fused_embeddings_2025.npy'
FUSED_2025_RECS = PROJECT_ROOT / 'outputs' / 'rep_analysis' / 'geom_fusion_2025' / 'aligned_records_2025.json'

# Parameters
SEED_N = 10
KNN_K = 5
EPS = 0.01
K_WINDOW = 5

# Utilities

def load_yolo_embeddings():
    if not YOLO_EMB_PATH.exists():
        raise FileNotFoundError(f"YOLO embeddings not found at {YOLO_EMB_PATH}")
    vecs = np.load(str(YOLO_EMB_PATH))
    records = None
    if YOLO_META_CSV.exists():
        try:
            df = pd.read_csv(str(YOLO_META_CSV))
            # ensure image_path present
            if 'image_path' in df.columns:
                records = df.to_dict(orient='records')
        except Exception:
            records = None
    if records is None:
        # synthesize basic records
        records = [{'image_path': f'idx_{i}'} for i in range(vecs.shape[0])]
    return records, vecs


def load_fused_embeddings():
    if not FUSED_2024_EMB.exists() or not FUSED_2025_EMB.exists():
        raise FileNotFoundError('Fused embeddings for 2024/2025 not found under geom_fusion directories')
    emb24 = np.load(str(FUSED_2024_EMB))
    emb25 = np.load(str(FUSED_2025_EMB))
    # load records
    rec24 = []
    rec25 = []
    if FUSED_2024_RECS.exists():
        with open(str(FUSED_2024_RECS), 'r') as fh:
            rec24 = json.load(fh)
    if FUSED_2025_RECS.exists():
        with open(str(FUSED_2025_RECS), 'r') as fh:
            rec25 = json.load(fh)
    if len(rec24) != emb24.shape[0]:
        # try to synthesize
        rec24 = [{'image_path': f'24_idx_{i}'} for i in range(emb24.shape[0])]
    if len(rec25) != emb25.shape[0]:
        rec25 = [{'image_path': f'25_idx_{i}'} for i in range(emb25.shape[0])]
    return rec24, emb24, rec25, emb25


def select_indices_by_season(records, season='2025'):
    idxs = []
    for i, r in enumerate(records):
        s = None
        if isinstance(r, dict):
            s = r.get('season') or r.get('year')
            ip = r.get('image_path') or r.get('image') or r.get('file')
        else:
            s = None
            ip = r
        if s is not None and str(s) == str(season):
            idxs.append(i)
        else:
            if ip and isinstance(ip, str) and '2025' in ip:
                idxs.append(i)
    return idxs


def compute_knn_mean_distance(X_ref, X_query, k=5):
    if X_ref.shape[0] == 0:
        return np.full((X_query.shape[0],), np.nan)
    nn = NearestNeighbors(n_neighbors=min(k, X_ref.shape[0]), metric='euclidean')
    nn.fit(X_ref)
    dists, _ = nn.kneighbors(X_query)
    mean_d = np.mean(dists, axis=1)
    return mean_d


def farthest_point_sampling_with_elbow(X24, X25, rec25, seed_n=10, knn_k=5, eps=0.01, K_window=5, out_dir=Path('.')):
    # X24: np.array MxD ; X25: NxD ; rec25: list of records aligned with X25
    N = X25.shape[0]
    if N == 0:
        raise ValueError('No 2025 candidates')
    # Step 1: compute d_to_2024 mean KNN distance
    d_to_24 = compute_knn_mean_distance(X24, X25, k=knn_k)
    # Seed: top-N most distant
    seed_n = min(seed_n, N)
    seed_order = np.argsort(-d_to_24)[:seed_n].tolist()
    selected = list(seed_order)  # local indices into X25
    selected_set = set(selected)
    # compute initial min distances to S
    # pairwise distances between X25 and selected set
    from sklearn.metrics import pairwise_distances
    if len(selected) > 0:
        D_sel = pairwise_distances(X25, X25[selected], metric='euclidean')
        min_dists = D_sel.min(axis=1)
    else:
        min_dists = np.full((N,), np.inf)
    # after seed, set min_dists for selected to 0
    for s in selected:
        min_dists[s] = 0.0
    R_prev = float(np.max(min_dists))
    R_history = [R_prev]
    log_rows = []
    # Log seed selections
    for i,s in enumerate(selected, start=1):
        log_rows.append({'iteration': i, 'selected_image': rec25[s].get('image_path') if isinstance(rec25[s], dict) else rec25[s], 'R_t': float(R_prev), 'Delta_t': None, 'stop_flag': False})
    iter_idx = len(selected)
    stagnation = 0
    # FPS loop
    while True:
        # pick unselected with maximum min_dists (tie-breaker: smallest index)
        candidates = [i for i in range(N) if i not in selected_set]
        if not candidates:
            break
        cand_dists = min_dists[candidates]
        # choose argmax
        best_pos_in_candidates = int(np.argmax(cand_dists))
        best_idx = candidates[best_pos_in_candidates]
        best_dist = float(cand_dists[best_pos_in_candidates])
        # Add
        selected.append(best_idx)
        selected_set.add(best_idx)
        iter_idx += 1
        # Update min_dists
        new_dists = pairwise_distances(X25, X25[[best_idx]], metric='euclidean').reshape(-1)
        min_dists = np.minimum(min_dists, new_dists)
        min_dists[list(selected_set)] = 0.0
        R_curr = float(np.max(min_dists))
        R_history.append(R_curr)
        # compute Delta relative improvement
        Delta = None
        if R_prev > 0:
            Delta = (R_prev - R_curr) / R_prev
        else:
            Delta = 0.0
        stop_flag = False
        if Delta < eps:
            stagnation += 1
        else:
            stagnation = 0
        if stagnation >= K_window:
            stop_flag = True
        log_rows.append({'iteration': iter_idx, 'selected_image': rec25[best_idx].get('image_path') if isinstance(rec25[best_idx], dict) else rec25[best_idx], 'R_t': R_curr, 'Delta_t': Delta, 'stop_flag': stop_flag})
        R_prev = R_curr
        if stop_flag:
            break
    # Build core_set.csv: order entries by selection order with selection_order starting at 1
    core_rows = []
    for order, idx in enumerate(selected, start=1):
        core_rows.append({'image_path': rec25[idx].get('image_path') if isinstance(rec25[idx], dict) else rec25[idx], 'season': '2025', 'selection_order': order, 'distance_at_selection': float(0.0)})
    # fill distance_at_selection with recorded min distance when selected: we can recompute distances to set at selection time roughly
    # For simplicity, recompute distances sequentially to get distance at selection
    selected_set2 = []
    min_dists2 = np.full((N,), np.inf)
    Dmat = None
    for s in selected:
        if not selected_set2:
            # first selection: distance to 2024 used? for seed we store d_to_24
            pass
        # distance at selection = min distance to previous selected_set2
        if len(selected_set2) == 0:
            dist_sel = float(d_to_24[s])
        else:
            # compute distances from s to selected_set2
            if Dmat is None:
                from sklearn.metrics import pairwise_distances
                Dmat = pairwise_distances(X25, X25, metric='euclidean')
            dist_sel = float(np.min(Dmat[s, selected_set2]))
        # append
        selected_set2.append(s)
        # find corresponding row in core_rows and set distance_at_selection
        for row in core_rows:
            if (row['selection_order'] == selected_set2.index(s)+1) and (row['image_path'] == (rec25[s].get('image_path') if isinstance(rec25[s], dict) else rec25[s])):
                row['distance_at_selection'] = dist_sel
                break
    return core_rows, log_rows, R_history


def run_yolo_elbow():
    out_dir = OUT_BASE / 'yolo_elbow'
    out_dir.mkdir(parents=True, exist_ok=True)
    recs, vecs = load_yolo_embeddings()
    # select 2024/2025 indices
    idx_2025 = [i for i,r in enumerate(recs) if (isinstance(r, dict) and (str(r.get('season'))=='2025' or ('2025' in str(r.get('image_path') or ''))))]
    idx_2024 = [i for i,r in enumerate(recs) if (isinstance(r, dict) and (str(r.get('season'))=='2024' or ('2024' in str(r.get('image_path') or ''))))]
    if len(idx_2025) == 0:
        # fallback: try image_path contains
        idx_2025 = [i for i,r in enumerate(recs) if ('2025' in str(r.get('image_path') if isinstance(r, dict) else r))]
    if len(idx_2024) == 0:
        idx_2024 = [i for i,r in enumerate(recs) if ('2024' in str(r.get('image_path') if isinstance(r, dict) else r))]
    X24 = vecs[np.array(idx_2024)]
    X25 = vecs[np.array(idx_2025)]
    recs25 = [recs[i] for i in idx_2025]
    core_rows, log_rows, R_history = farthest_point_sampling_with_elbow(X24, X25, recs25, seed_n=SEED_N, knn_k=KNN_K, eps=EPS, K_window=K_WINDOW, out_dir=out_dir)
    # save core_set
    core_df = pd.DataFrame(core_rows)
    core_df.to_csv(out_dir / 'core_set.csv', index=False)
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(out_dir / 'selection_log.csv', index=False)
    # plot elbow
    plt.figure(figsize=(6,4))
    plt.plot(range(len(R_history)), R_history, marker='o')
    plt.xlabel('|S| (iterations)')
    plt.ylabel('R_t (max min distance)')
    # find stopping point
    stop_idx = len(R_history)-1
    plt.axvline(stop_idx, color='red', linestyle='--')
    plt.savefig(out_dir / 'elbow_curve.png', dpi=150, bbox_inches='tight')
    print('YOLO core-set saved to', out_dir)


def run_fused_elbow():
    out_dir = OUT_BASE / 'fused_elbow'
    out_dir.mkdir(parents=True, exist_ok=True)
    rec24, emb24, rec25, emb25 = load_fused_embeddings()
    # Here rec25 correspond to emb25 order; select all (they should all be 2025)
    core_rows, log_rows, R_history = farthest_point_sampling_with_elbow(emb24, emb25, rec25, seed_n=SEED_N, knn_k=KNN_K, eps=EPS, K_window=K_WINDOW, out_dir=out_dir)
    core_df = pd.DataFrame(core_rows)
    core_df.to_csv(out_dir / 'core_set.csv', index=False)
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(out_dir / 'selection_log.csv', index=False)
    plt.figure(figsize=(6,4))
    plt.plot(range(len(R_history)), R_history, marker='o')
    plt.xlabel('|S| (iterations)')
    plt.ylabel('R_t (max min distance)')
    stop_idx = len(R_history)-1
    plt.axvline(stop_idx, color='red', linestyle='--')
    plt.savefig(out_dir / 'elbow_curve.png', dpi=150, bbox_inches='tight')
    print('Fused core-set saved to', out_dir)


def main():
    print('Running YOLO elbow core-set selection...')
    try:
        run_yolo_elbow()
    except Exception as e:
        print('YOLO run failed:', e)
    print('\nRunning fused elbow core-set selection...')
    try:
        run_fused_elbow()
    except Exception as e:
        print('Fused run failed:', e)

if __name__ == '__main__':
    main()
