"""
Create geometric pose embeddings aligned to existing YOLO embeddings and produce fused embeddings.

Saves outputs under: outputs/rep_analysis/geom_fusion_2024/

Behavior:
 - Loads YOLO embeddings and their records via utils.io.load_embeddings(out_dir)
 - Iterates images in that exact order, locates pose label under datasets/train_on_all/labels/<stem>.txt
 - Parses 4 keypoints per label (heuristic: uses last 8 floats in the label line)
 - Builds scale-invariant geometric descriptor per image
 - Drops images with missing or invalid pose labels (removes them from both embeddings)
 - Computes z-score normalization of geometric embeddings and optionally L2-normalizes
 - Concatenates YOLO embedding and geom embedding to produce fused embedding
 - Saves:
    - geom_embeddings_2024.npy
    - fused_embeddings_2024.npy
    - aligned_records_2024.json (list of records used, in order)
    - geom_stats_2024.npz (mean/std used for z-score)

Usage:
    python scripts/representation_analysis/geom_fusion_2024/create_geom_fused_embeddings.py

"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import math
import sys

# Try to import helper from repo
try:
    from utils.io import load_embeddings
except Exception:
    load_embeddings = None

# Constants / paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
YOLO_OUT_DIR = PROJECT_ROOT / 'outputs' / 'rep_analysis'
FUSION_OUT = YOLO_OUT_DIR / 'geom_fusion_2024'
FUSION_OUT.mkdir(parents=True, exist_ok=True)
DATA_2024 = PROJECT_ROOT / 'datasets' / 'train_on_all'
LABELS_DIR = DATA_2024 / 'labels'

# Expected output filenames
GEOM_NPY = FUSION_OUT / 'geom_embeddings_2024.npy'
FUSED_NPY = FUSION_OUT / 'fused_embeddings_2024.npy'
RECORDS_JSON = FUSION_OUT / 'aligned_records_2024.json'
STATS_NPZ = FUSION_OUT / 'geom_stats_2024.npz'

# Determinism
RNG = np.random.RandomState(0)

# Utilities
def safe_load_yolo_embeddings(out_dir: Path):
    """Return (records, vectors) loading via utils.io.load_embeddings when available, else try CSV/NPY combos."""
    if load_embeddings is not None:
        try:
            recs, vecs = load_embeddings(str(out_dir))
            return recs, np.asarray(vecs)
        except Exception as e:
            print(f"DEBUG: load_embeddings failed: {e}")
    # fallback attempts
    meta_csv = out_dir / 'embeddings_meta.csv'
    vec_npy = out_dir / 'embeddings_vectors.npy'
    if not vec_npy.exists():
        raise FileNotFoundError(f"Embeddings vectors not found at {vec_npy}")
    vectors = np.load(str(vec_npy))
    records = None
    if meta_csv.exists():
        try:
            df = pd.read_csv(str(meta_csv))
            # expect columns including image_path
            if 'image_path' in df.columns:
                records = df.to_dict(orient='records')
        except Exception as e:
            print(f"Warning: failed to read {meta_csv}: {e}")
    if records is None:
        # synthesize records as idx placeholders
        records = [{'image_path': f'idx_{i}'} for i in range(len(vectors))]
    return records, vectors


def parse_pose_label(label_path: Path, expected_kpts=4):
    """Parse YOLO-pose label file and return keypoints as numpy array shape (K,2) in normalized coords [0,1].
    Heuristic: read first non-empty line, parse floats, if >= 5+2*K tokens assume format [class cx cy w h kpx kpy ...], else take last 2*K floats as kpts.
    Returns None if parsing fails or coordinates are invalid.
    """
    if not label_path.exists():
        return None
    try:
        with open(label_path, 'r') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                toks = line.split()
                # parse floats after possibly discarding class
                floats = []
                for t in toks:
                    try:
                        floats.append(float(t))
                    except Exception:
                        pass
                if len(floats) == 0:
                    continue
                # heuristic: if len >= 5 + 2*K, take last 2*K as kpts, else if len >= 2*K take last 2*K
                need = 2 * expected_kpts
                if len(floats) >= 5 + need:
                    kp_f = floats[-need:]
                elif len(floats) >= need:
                    kp_f = floats[-need:]
                else:
                    # cannot parse
                    return None
                kpts = np.array(kp_f, dtype=float).reshape(-1, 2)
                # basic validity: coords between 0 and 1
                if np.any(np.isnan(kpts)):
                    return None
                # allow coords slightly outside [0,1] but clamp
                kpts = np.clip(kpts, 0.0, 1.0)
                return kpts
    except Exception:
        return None
    return None


def angle_between(v1, v2):
    # returns signed angle in radians between v1 and v2
    # handle zero vectors
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos = np.dot(v1, v2) / (n1 * n2)
    cos = max(-1.0, min(1.0, cos))
    return math.acos(cos)


def build_geom_descriptor(kpts):
    """Build a Procrustes-style body-frame shape descriptor for 4 keypoints.

    Steps:
      - Use rostrum (index 2) as origin; tail index 3 defines body axis.
      - Translate so rostrum at (0,0).
      - Compute body vector = tail - rostrum, length L. Require L > eps.
      - Scale coordinates by 1/L (scale invariance) and rotate so body axis aligns to +x.
      - After body-frame normalization stack 4x2 matrix, center is at rostrum (0,0).
      - Apply Frobenius norm normalization (divide matrix by its Frobenius norm). Do NOT reflect.
      - Flatten to 8D vector [x0,y0,x1,y1,...].
    Returns 1D numpy array length 8 or None on failure.
    """
    kpts = np.asarray(kpts, dtype=float)
    if kpts.shape[0] < 4 or kpts.shape[1] != 2:
        return None
    # Enforce keypoint order expectation: 0 carapace,1 eyes,2 rostrum,3 tail
    rostrum = kpts[2].astype(float)
    tail = kpts[3].astype(float)

    body_vec = tail - rostrum
    L = np.linalg.norm(body_vec)
    if L <= 1e-8:
        return None

    # translate so rostrum at origin
    translated = kpts - rostrum

    # rotate so body_vec aligns with +x
    angle = math.atan2(body_vec[1], body_vec[0])
    c = math.cos(-angle)
    s = math.sin(-angle)
    R = np.array([[c, -s], [s, c]])
    rotated = (R @ translated.T).T

    # scale by body length
    body_scaled = rotated / L

    # Now perform Procrustes-like normalization: divide by Frobenius norm (no reflection)
    F = np.linalg.norm(body_scaled)  # Frobenius norm
    if F <= 1e-12:
        return None
    shape = body_scaled / F

    # flatten in row-major order: x0,y0,x1,y1,...
    desc = shape.reshape(-1)
    if desc.shape[0] != 8:
        return None
    return desc


def main():
    print('Geom-fusion: loading YOLO embeddings and records...', flush=True)
    try:
        records, yolo_vecs = safe_load_yolo_embeddings(YOLO_OUT_DIR)
    except Exception as e:
        print('ERROR: could not load YOLO embeddings:', e, file=sys.stderr)
        return
    records = list(records)
    yolo_vecs = np.asarray(yolo_vecs)
    N, D = yolo_vecs.shape
    print(f'Loaded {N} YOLO embeddings (dim={D})')

    # iterate records in order, parse labels and build geom descriptors
    geom_list = []
    kept_records = []
    dropped = 0
    for i, rec in enumerate(records):
        img_path = rec.get('image_path') or rec.get('image') or rec.get('file')
        if img_path is None:
            print(f'Warning: record {i} missing image_path; dropping')
            dropped += 1
            continue
        # normalize path: if relative, assume relative to PROJECT_ROOT
        p = Path(img_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / img_path
        stem = p.stem
        label_path = LABELS_DIR / f"{stem}.txt"
        kpts = parse_pose_label(label_path, expected_kpts=4)
        if kpts is None:
            # try alternative: look under datasets/train_on_all/val/labels
            alt = DATA_2024 / 'val' / 'labels' / f"{stem}.txt"
            kpts = parse_pose_label(alt, expected_kpts=4)
        if kpts is None:
            # drop image globally
            dropped += 1
            continue
        desc = build_geom_descriptor(kpts)
        if desc is None or np.any(np.isnan(desc)):
            dropped += 1
            continue
        geom_list.append(desc)
        kept_records.append({'image_path': str(p), 'orig_record': rec})

    print(f'Dropped {dropped} images due to missing/invalid pose labels. Kept {len(geom_list)} images.', flush=True)

    if len(geom_list) == 0:
        print('No geometric embeddings computed; aborting.', file=sys.stderr)
        return

    geom_array = np.vstack(geom_list)
    # Ensure alignment: need to subset yolo_vecs to kept_records order
    # Build mapping from original records to indices: assume original records align with yolo order
    # We must preserve exact ordering and remove dropped entries from both embeddings
    # Approach: iterate original records and collect indices of kept ones in same order
    keep_mask = []
    kept_indices = []
    rec_to_index = {}
    # build mapping from normalized image_path to list of indices (handle duplicates)
    for idx, rec in enumerate(records):
        key = str(rec.get('image_path') or rec.get('image') or rec.get('file'))
        rec_to_index.setdefault(key, []).append(idx)
    used = set()
    for kr in kept_records:
        key = str(kr['orig_record'].get('image_path') or kr['orig_record'].get('image') or kr['orig_record'].get('file'))
        # consume next unused index
        lst = rec_to_index.get(key, [])
        found = None
        for x in lst:
            if x not in used:
                found = x
                used.add(x)
                break
        if found is None:
            # fallback: cannot map; abort
            print(f'ERROR: could not map kept record {key} back to embeddings ordering', file=sys.stderr)
            return
        kept_indices.append(found)
        keep_mask.append(True)

    # sort kept_indices + corresponding geom_array by the order of kept_indices (they are already in order of kept_records)
    # Build new yolo subset in the same order
    yolo_kept = yolo_vecs[np.array(kept_indices)]

    if yolo_kept.shape[0] != geom_array.shape[0]:
        print('ERROR: alignment mismatch between yolo and geom embeddings after filtering', file=sys.stderr)
        print('yolo_kept.shape =', yolo_kept.shape, 'geom.shape =', geom_array.shape, file=sys.stderr)
        return

    # --- New normalization: compute mean/std on raw shape descriptors (2024 reference)
    mean = np.mean(geom_array, axis=0)
    std = np.std(geom_array, axis=0)
    std_adj = np.where(std < 1e-12, 1.0, std)
    # z-score
    geom_z = (geom_array - mean) / std_adj
    # L2 normalize each row
    norms = np.linalg.norm(geom_z, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    geom_norm = geom_z / norms

    # Fuse
    fused = np.concatenate([yolo_kept, geom_norm], axis=1)

    # Save outputs under FUSION_OUT
    np.save(str(GEOM_NPY), geom_norm)
    np.save(str(FUSED_NPY), fused)
    # aligned_records: save kept_records with original record data and mapping to kept index
    aligned = []
    for i, kr in enumerate(kept_records):
        rec = kr['orig_record'].copy()
        rec['aligned_index'] = int(kept_indices[i])
        aligned.append(rec)
    with open(str(RECORDS_JSON), 'w') as fh:
        json.dump(aligned, fh, indent=2)
    # stats
    np.savez(str(STATS_NPZ), mean=mean, std=std_adj)

    print(f'Saved geom embeddings: {GEOM_NPY}')
    print(f'Saved fused embeddings: {FUSED_NPY}')
    print(f'Saved aligned records: {RECORDS_JSON}')
    print(f'Saved geom stats: {STATS_NPZ}')

if __name__ == '__main__':
    main()
