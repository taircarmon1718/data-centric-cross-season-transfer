"""
Project Season 2025 images into the fused representation space defined by Season 2024.

Saves outputs to:
  outputs/rep_analysis/geom_fusion_2025/

Produces:
 - geom_embeddings_2025.npy
 - fused_embeddings_2025.npy
 - aligned_records_2025.json

Requirements enforced by script:
 - Do NOT modify outputs/rep_analysis/embeddings_vectors.npy
 - Load geom_stats_2024.npz from outputs/rep_analysis/geom_fusion_2024/
 - Use the same geometric descriptor and normalization as 2024 implementation
 - Drop images missing / invalid pose labels (consistently)
 - Preserve input YOLO embedding ordering when selecting 2025 samples

Usage:
  python scripts/representation_analysis/geom_fusion_2025/create_geom_fused_embeddings_2025.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import json
import math
import sys

# Try import helper loader
try:
    from utils.io import load_embeddings
except Exception:
    load_embeddings = None

PROJECT_ROOT = Path(__file__).resolve().parents[3]
YOLO_OUT_DIR = PROJECT_ROOT / 'outputs' / 'rep_analysis'
FUSION2024_DIR = YOLO_OUT_DIR / 'geom_fusion_2024'
FUSION2025_DIR = YOLO_OUT_DIR / 'geom_fusion_2025'
FUSION2025_DIR.mkdir(parents=True, exist_ok=True)
DATA2025 = PROJECT_ROOT / 'datasets' / 'train_on_2025_all'
LABELS_DIR = DATA2025 / 'labels'
VAL_LABELS_DIR = DATA2025 / 'val' / 'labels'

# Input files
YOLO_VEC_NPY = YOLO_OUT_DIR / 'embeddings_vectors.npy'
YOLO_META_CSV = YOLO_OUT_DIR / 'embeddings_meta.csv'
GEOM_STATS = FUSION2024_DIR / 'geom_stats_2024.npz'

# Outputs
GEOM_OUT = FUSION2025_DIR / 'geom_embeddings_2025.npy'
FUSED_OUT = FUSION2025_DIR / 'fused_embeddings_2025.npy'
RECORDS_OUT = FUSION2025_DIR / 'aligned_records_2025.json'

# Determinism
RNG = np.random.RandomState(0)

# --- utility functions (must match 2024 implementation) ---
def parse_pose_label(label_path: Path, expected_kpts=4):
    if not label_path.exists():
        return None
    try:
        with open(label_path, 'r') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                toks = line.split()
                floats = []
                for t in toks:
                    try:
                        floats.append(float(t))
                    except Exception:
                        pass
                if len(floats) == 0:
                    continue
                need = 2 * expected_kpts
                if len(floats) >= 5 + need:
                    kp_f = floats[-need:]
                elif len(floats) >= need:
                    kp_f = floats[-need:]
                else:
                    return None
                kpts = np.array(kp_f, dtype=float).reshape(-1, 2)
                if np.any(np.isnan(kpts)):
                    return None
                kpts = np.clip(kpts, 0.0, 1.0)
                return kpts
    except Exception:
        return None
    return None


def angle_between(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos = np.dot(v1, v2) / (n1 * n2)
    cos = max(-1.0, min(1.0, cos))
    return math.acos(cos)


def build_geom_descriptor(kpts):
    K = kpts.shape[0]
    if K < 2:
        return None
    vecs = kpts[1:] - kpts[:-1]
    seg_lengths = np.linalg.norm(vecs, axis=1)
    body_vec = kpts[-1] - kpts[0]
    body_len = np.linalg.norm(body_vec)
    if body_len <= 1e-8:
        body_len = np.maximum(1e-8, np.mean(seg_lengths) if seg_lengths.size>0 else 1.0)
    norm_seg = seg_lengths / body_len
    pair_dists = []
    for i in range(K):
        for j in range(i+1, K):
            d = np.linalg.norm(kpts[j] - kpts[i]) / body_len
            pair_dists.append(d)
    pair_dists = np.array(pair_dists)
    ang_cos = []
    for i in range(len(vecs)-1):
        v1 = vecs[i]
        v2 = vecs[i+1]
        a = angle_between(v1, v2)
        ang_cos.append(math.cos(a))
    ang_cos = np.array(ang_cos)
    gv = body_vec / (np.linalg.norm(body_vec) + 1e-12)
    gv_cos, gv_sin = float(gv[0]), float(gv[1])
    desc = np.concatenate([norm_seg, pair_dists, ang_cos, np.array([gv_cos, gv_sin])])
    return desc

# --- I/O helpers ---

def load_yolo_embeddings_and_records(out_dir: Path):
    if load_embeddings is not None:
        try:
            recs, vecs = load_embeddings(str(out_dir))
            return list(recs), np.asarray(vecs)
        except Exception as e:
            print(f"DEBUG: utils.io.load_embeddings failed: {e}")
    # fallback
    meta_csv = out_dir / 'embeddings_meta.csv'
    vec_npy = out_dir / 'embeddings_vectors.npy'
    if not vec_npy.exists():
        raise FileNotFoundError(f"YOLO embeddings not found at {vec_npy}")
    vecs = np.load(str(vec_npy))
    records = None
    if meta_csv.exists():
        try:
            df = pd.read_csv(str(meta_csv))
            if 'image_path' in df.columns:
                records = df.to_dict(orient='records')
        except Exception as e:
            print(f"Warning: failed to read {meta_csv}: {e}")
    if records is None:
        records = [{'image_path': f'idx_{i}'} for i in range(len(vecs))]
    return records, vecs

# --- Main processing ---

def main():
    print('Loading YOLO embeddings and records...')
    try:
        records, yolo_vecs = load_yolo_embeddings_and_records(YOLO_OUT_DIR)
    except Exception as e:
        print('ERROR: could not load YOLO embeddings:', e, file=sys.stderr)
        return
    records = list(records)
    yolo_vecs = np.asarray(yolo_vecs)
    print(f'Loaded {len(records)} records, embeddings shape {yolo_vecs.shape}')

    if not GEOM_STATS.exists():
        print(f'ERROR: geom stats not found at {GEOM_STATS}', file=sys.stderr)
        return
    stats = np.load(str(GEOM_STATS))
    mean = stats['mean']
    std = stats['std']
    std_adj = np.where(std < 1e-12, 1.0, std)

    # Select only 2025 records in the same ordering as records list
    idxs_2025 = []
    recs_2025 = []
    for i, r in enumerate(records):
        season = r.get('season') or r.get('year') or r.get('split')
        # robust check: season field may be '2025' or record image_path contains '2025' path
        if season is not None and str(season) == '2025':
            idxs_2025.append(i)
            recs_2025.append(r)
        else:
            # try image_path heuristic
            ip = r.get('image_path') or r.get('image') or r.get('file')
            if ip and '2025' in str(ip):
                idxs_2025.append(i)
                recs_2025.append(r)

    print(f'Found {len(idxs_2025)} Season 2025 records in YOLO embeddings (ordered).')

    geom_list = []
    kept_indices = []
    kept_records = []
    dropped = 0
    for local_pos, global_idx in enumerate(idxs_2025):
        rec = records[global_idx]
        ip = rec.get('image_path') or rec.get('image') or rec.get('file')
        if ip is None:
            dropped += 1
            continue
        p = Path(ip)
        if not p.is_absolute():
            p = PROJECT_ROOT / ip
        stem = p.stem
        label = LABELS_DIR / f"{stem}.txt"
        if not label.exists():
            label = VAL_LABELS_DIR / f"{stem}.txt"
        kpts = parse_pose_label(label, expected_kpts=4)
        if kpts is None:
            dropped += 1
            continue
        desc = build_geom_descriptor(kpts)
        if desc is None or np.any(np.isnan(desc)):
            dropped += 1
            continue
        geom_list.append(desc)
        kept_indices.append(global_idx)
        # store aligned record (keep original fields plus unified image path)
        aligned_rec = dict(rec)
        aligned_rec['image_path'] = str(p)
        kept_records.append(aligned_rec)

    print(f'Dropped {dropped} images; kept {len(geom_list)} 2025 images with valid pose labels.')
    if len(geom_list) == 0:
        print('No valid 2025 geometric descriptors computed; aborting.', file=sys.stderr)
        return

    geom_array = np.vstack(geom_list)
    # Normalize by mean/std from 2024 and L2 normalize rows
    geom_z = (geom_array - mean) / std_adj
    norms = np.linalg.norm(geom_z, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    geom_norm = geom_z / norms

    # Build corresponding yolo subset in same order
    yolo_subset = yolo_vecs[np.array(kept_indices)]
    if yolo_subset.shape[0] != geom_norm.shape[0]:
        print('ERROR: mismatch after filtering; aborting.', file=sys.stderr)
        print('yolo_subset.shape=', yolo_subset.shape, 'geom.shape=', geom_norm.shape, file=sys.stderr)
        return

    fused = np.concatenate([yolo_subset, geom_norm], axis=1)

    # Save outputs
    np.save(str(GEOM_OUT), geom_norm)
    np.save(str(FUSED_OUT), fused)
    with open(str(RECORDS_OUT), 'w') as fh:
        json.dump(kept_records, fh, indent=2)
    print('Saved outputs:')
    print(' -', GEOM_OUT)
    print(' -', FUSED_OUT)
    print(' -', RECORDS_OUT)

if __name__ == '__main__':
    main()

