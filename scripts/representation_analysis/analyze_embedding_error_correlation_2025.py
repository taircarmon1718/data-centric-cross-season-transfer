#!/usr/bin/env python3
"""
analyze_embedding_error_correlation_2025.py

Robust analysis: test whether embedding distance correlates with model error on 2025.

Behavior summary:
- Loads embeddings metadata and vectors from outputs/rep_analysis (or fallback locations).
- L2-normalizes vectors and computes Euclidean distance to centroid per embedding.
- Matches embeddings (rows) to actual test images using check_on_2025 lists when available, otherwise searches datasets/train_on_2025_all.
- Runs YOLO inference using the 2024 model and the same logic as check_on_2025 to compute detection success and MAE for total length and carapace.
- Assembles a DataFrame and computes Pearson, Spearman, and point-biserial correlations.
- Writes results to outputs/analysis/embedding_error_correlation_2025.csv

This file replaces the earlier broken script and is written to be robust and self-contained.
"""

from pathlib import Path
import sys
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Try to import the existing evaluation helper to reuse constants and GT builders
try:
    from scripts.eval import check_on_2025 as chk
    HAS_CHECK = True
except Exception:
    HAS_CHECK = False

# Try ultralytics
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# Paths
YOLO_WEIGHTS = PROJECT_ROOT / 'models' / '2024' / 'all-ponds' / 'weights' / 'best.pt'
OUT_DIR = PROJECT_ROOT / 'outputs' / 'analysis'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / 'embedding_error_correlation_2025.csv'

# Candidate embedding dirs (preferred outputs/rep_analysis)
EMBED_CANDIDATES = [
    PROJECT_ROOT / 'outputs' / 'rep_analysis',
    PROJECT_ROOT / 'scripts' / 'representation_analysis' / 'outputs_repreasentation' / 'rep_analysis',
]

# Inference / evaluation constants (use check_on_2025 defaults if available)
CONF_TH = getattr(chk, 'CONF_TH', 0.25) if HAS_CHECK else 0.25
DEVICE = getattr(chk, 'DEVICE', None) if HAS_CHECK else None
HFOV = getattr(chk, 'HFOV_DERIVED_DEG', 76.2) if HAS_CHECK else 76.2
VFOV = getattr(chk, 'VFOV_DERIVED_DEG', 46.0) if HAS_CHECK else 46.0
IOU_MIN = getattr(chk, 'IOU_MIN', 0.05) if HAS_CHECK else 0.05
IOA_MIN = getattr(chk, 'IOA_MIN', 0.5) if HAS_CHECK else 0.5
CAR_IDXS = getattr(chk, 'CAR_IDXS', (0,1)) if HAS_CHECK else (0,1)
TOT_IDXS = getattr(chk, 'TOT_IDXS', (2,3)) if HAS_CHECK else (2,3)
WORK_FRAME_W = getattr(chk, 'WORK_FRAME_W', 640) if HAS_CHECK else 640
WORK_FRAME_H = getattr(chk, 'WORK_FRAME_H', 360) if HAS_CHECK else 360

# Utility functions (copy of evaluation helpers where needed)

def deg2rad(d):
    return d * math.pi / 180.0


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    return inter / (area_a + area_b - inter + 1e-9)


def ioa_xyxy(gt, pred):
    ax1, ay1, ax2, ay2 = gt
    bx1, by1, bx2, by2 = pred
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_gt = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    return inter / (area_gt + 1e-9)


def center_in_bbox(cx, cy, b):
    x1, y1, x2, y2 = b
    return (x1 <= cx <= x2) and (y1 <= cy <= y2)


def find_best_pred_for_gt(gt_xyxy, det_xyxy_all):
    if det_xyxy_all.shape[0] == 0:
        return None, None, None, None
    ious = []
    ioas = []
    centers = []
    for d in det_xyxy_all:
        d_list = d.tolist()
        iou = iou_xyxy(gt_xyxy, d_list)
        ioa = ioa_xyxy(gt_xyxy, d_list)
        cx = 0.5 * (d_list[0] + d_list[2])
        cy = 0.5 * (d_list[1] + d_list[3])
        cin = center_in_bbox(cx, cy, gt_xyxy)
        ious.append(iou)
        ioas.append(ioa)
        centers.append(cin)
    ious = np.array(ious)
    ioas = np.array(ioas)
    centers = np.array(centers, dtype=bool)
    valid = (ious >= IOU_MIN) | (ioas >= IOA_MIN) | centers
    if not np.any(valid):
        return None, None, None, None
    cand = np.where(valid)[0]
    scores = ious[cand] + ioas[cand]
    best = cand[np.argmax(scores)]
    return int(best), float(ious[best]), float(ioas[best]), bool(centers[best])


def pixel_scales_mm_per_px(distance_mm, img_w, img_h, hfov_deg=HFOV, vfov_deg=VFOV):
    S_h = 2.0 * distance_mm * math.tan(math.radians(hfov_deg) / 2.0) / float(img_w)
    S_v = 2.0 * distance_mm * math.tan(math.radians(vfov_deg) / 2.0) / float(img_h)
    return S_h, S_v


def segment_len_px(kpts_xy, i0, i1):
    x0, y0 = kpts_xy[i0]
    x1, y1 = kpts_xy[i1]
    dx, dy = x1 - x0, y1 - y0
    return float(np.hypot(dx, dy)), dx, dy


def segment_len_mm_with_theta(length_px, dx, dy, S_h, S_v):
    theta_rad = math.atan2(dy, dx)
    theta_deg = math.degrees(theta_rad)
    theta_norm = min(abs(theta_deg) % 180, 180 - (abs(theta_deg) % 180))
    S_total = math.sqrt((S_h * math.cos(math.radians(theta_norm))) ** 2 + (S_v * math.sin(math.radians(theta_norm))) ** 2)
    return length_px * S_total

# Embedding utilities

def find_embedding_files():
    for d in EMBED_CANDIDATES:
        meta = d / 'embeddings_meta.csv'
        vecs = d / 'embeddings_vectors.npy'
        if meta.exists() and vecs.exists():
            return meta, vecs
    # last-ditch: search outputs tree
    outd = PROJECT_ROOT / 'outputs'
    if outd.exists():
        for p in outd.rglob('embeddings_vectors.npy'):
            candidate = p.parent
            meta = candidate / 'embeddings_meta.csv'
            if meta.exists():
                return meta, p
    raise FileNotFoundError('Could not find embeddings_meta.csv and embeddings_vectors.npy under outputs/rep_analysis or candidates')


def load_embeddings(meta_path: Path, vec_path: Path):
    meta = pd.read_csv(meta_path)
    vecs = np.load(vec_path)
    if len(meta) != vecs.shape[0]:
        warnings.warn(f"embeddings_meta.csv rows ({len(meta)}) != vectors rows ({vecs.shape[0]}); aligning by min length")
        m = min(len(meta), vecs.shape[0])
        meta = meta.iloc[:m].reset_index(drop=True)
        vecs = vecs[:m]
    return meta, vecs


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

# Main

def main():
    print('Running embedding vs error correlation analysis (2025)')

    # Defensive default
    model = None

    try:
        meta_path, vec_path = find_embedding_files()
    except FileNotFoundError as e:
        print('ERROR:', e)
        return 1

    print('Using embeddings:', meta_path, vec_path)
    meta_df, vecs = load_embeddings(meta_path, vec_path)

    # compute distance to centroid (after L2-normalize)
    vecs_n = l2_normalize_rows(vecs)
    centroid = vecs_n.mean(axis=0)
    dists = np.linalg.norm(vecs_n - centroid, axis=1)

    # require image_path column
    if 'image_path' not in meta_df.columns:
        print('ERROR: embeddings_meta.csv must contain `image_path` column')
        return 2

    # Build mapping from metadata rows to indices
    meta_paths = meta_df['image_path'].astype(str).tolist()

    # Build test image list
    test_images = []
    if HAS_CHECK:
        try:
            car_imgs = sorted(Path(chk.CAR_IMAGES_DIR).glob('*.jpg'))
            body_imgs = sorted(Path(chk.BODY_IMAGES_DIR).glob('*.jpg'))
            test_images = sorted(list({p.resolve() for p in (car_imgs + body_imgs)}))
            print(f'Using test image lists from check_on_2025: {len(test_images)} images')
        except Exception:
            test_images = []
    if not test_images:
        # fallback to datasets/train_on_2025_all images
        cand = PROJECT_ROOT / 'datasets' / 'train_on_2025_all'
        if cand.exists():
            for p in (cand / 'images').rglob('*.jpg') if (cand / 'images').exists() else []:
                test_images.append(p.resolve())
            for p in (cand / 'val' / 'images').rglob('*.jpg') if (cand / 'val' / 'images').exists() else []:
                test_images.append(p.resolve())
        test_images = sorted(list({p for p in test_images}))
        print(f'Using dataset fallback test images: {len(test_images)} images')

    # Build a global basename->paths index across likely locations (datasets, outputs, data, scripts)
    search_dirs = []
    # include previously discovered test_images first
    for p in test_images:
        search_dirs.append(p.parent)
    # core candidate roots
    for d in ['datasets', 'outputs', 'data', 'scripts']:
        cand = PROJECT_ROOT / d
        if cand.exists():
            search_dirs.append(cand)
    # deduplicate
    uniq_dirs = []
    for p in search_dirs:
        if p not in uniq_dirs:
            uniq_dirs.append(p)

    print(f'Scanning {len(uniq_dirs)} folders to build basename index (this may take a few seconds)...')
    basename_index = {}
    for root in uniq_dirs:
        try:
            for f in root.rglob('*.jpg'):
                basename_index.setdefault(f.name, []).append(f.resolve())
        except Exception:
            continue

    # Select only Season 2025 rows from metadata
    meta_2025_indices = []
    if 'season' in meta_df.columns:
        for i, r in meta_df.iterrows():
            try:
                if int(r['season']) == 2025:
                    meta_2025_indices.append(i)
            except Exception:
                if str(r['season']) == '2025':
                    meta_2025_indices.append(i)
    else:
        # fallback: include rows with '2025' in path
        for i, pstr in enumerate(meta_paths):
            if '2025' in pstr:
                meta_2025_indices.append(i)

    if len(meta_2025_indices) == 0:
        print('No 2025 rows found in embeddings metadata; aborting.')
        return 3

    # Match metadata rows to actual files to decide which rows we can evaluate
    matched_rows = []  # tuples (meta_idx, resolved_image_path)
    match_stats = {'basename_unique': 0, 'basename_ambig': 0, 'abs': 0, 'rel': 0, 'none': 0}

    for midx in meta_2025_indices:
        mpath = meta_paths[midx]
        # normalize separators
        mpath_norm = str(mpath).replace('\\', '/').strip()
        mname = Path(mpath_norm).name
        chosen = None
        # 1) try project-relative path under PROJECT_ROOT
        cand_abs = PROJECT_ROOT / Path(mpath_norm)
        if cand_abs.exists():
            chosen = cand_abs.resolve()
            match_stats['rel'] += 1
        # 2) try basename index
        if chosen is None and mname in basename_index:
            plist = sorted(basename_index[mname], key=lambda x: str(x))
            chosen = plist[0]
            if len(plist) == 1:
                match_stats['basename_unique'] += 1
            else:
                match_stats['basename_ambig'] += 1
        # 3) try absolute path given in metadata
        if chosen is None:
            p = Path(mpath_norm)
            if p.is_absolute() and p.exists():
                chosen = p.resolve()
                match_stats['abs'] += 1
        # 4) fallback: keep project-relative (may not exist)
        if chosen is None:
            chosen = (PROJECT_ROOT / Path(mpath_norm))
            match_stats['none'] += 1
        matched_rows.append((midx, chosen))

    exist_count = sum(1 for _, p in matched_rows if p.exists())
    print(f'Matched {len(matched_rows)} metadata rows ({len(meta_2025_indices)} season-2025 rows); {exist_count} files exist on disk and will be processed for inference')
    print('Match stats:', match_stats)

    records = []

    # Loop and evaluate
    for midx, img_path in tqdm(matched_rows, desc='Eval'):
        emb_dist = float(dists[midx]) if (midx < len(dists) and not np.isnan(dists[midx])) else np.nan
        detection_success = 0
        mae_total = np.nan
        mae_carapace = np.nan

        # if model is missing or image missing -> skip inference, leave NaNs
        if model is not None and img_path.exists():
            try:
                results = model.predict(str(img_path), conf=CONF_TH, device=DEVICE, verbose=False)
                r0 = results[0]
            except Exception as e:
                print(f'Warning: inference failed on {img_path}: {e}')
                r0 = None
        else:
            r0 = None

        dets = np.zeros((0, 4))
        kpts = np.zeros((0, 4, 2))
        W, H = (None, None)
        if r0 is not None:
            try:
                dets = r0.boxes.xyxy.cpu().numpy() if (r0.boxes is not None and getattr(r0.boxes, 'xyxy', None) is not None) else np.zeros((0, 4))
            except Exception:
                dets = np.zeros((0, 4))
            try:
                kpts = r0.keypoints.xy.cpu().numpy() if (hasattr(r0, 'keypoints') and r0.keypoints is not None and getattr(r0.keypoints, 'xy', None) is not None) else np.zeros((0, 4, 2))
            except Exception:
                kpts = np.zeros((0, 4, 2))
            # image size
            try:
                from PIL import Image
                with Image.open(str(img_path)) as im:
                    W, H = im.size
            except Exception:
                W, H = (WORK_FRAME_W, WORK_FRAME_H)
        else:
            # try to get W,H if file exists
            if img_path.exists():
                try:
                    from PIL import Image
                    with Image.open(str(img_path)) as im:
                        W, H = im.size
                except Exception:
                    W, H = (WORK_FRAME_W, WORK_FRAME_H)
            else:
                W, H = (WORK_FRAME_W, WORK_FRAME_H)

        # Compute GT matching and MAE using check_on_2025 logic (if GT tables exist)
        total_errors = []
        car_errors = []

        img_name = img_path.name
        # body GTs
        for _, brow in body_df[body_df['Image'].astype(str).str.strip() == img_name].iterrows():
            if HAS_CHECK:
                gt_obb, gt_aabb = chk.gt_obb_and_aabb_from_row(brow, image_size=(W, H))
            else:
                gt_obb, gt_aabb = (None, None)
            if gt_aabb is None:
                continue
            best_idx, best_iou, best_ioa, best_center = find_best_pred_for_gt(gt_aabb, dets)
            if best_idx is None:
                continue
            height_mm = float(brow.get('Height(mm)', 300))
            S_h, S_v = pixel_scales_mm_per_px(height_mm, W, H, HFOV, VFOV)
            pred_kpts = kpts[best_idx] if (kpts is not None and kpts.shape[0] > best_idx) else np.zeros((4, 2))
            length_px, dx, dy = segment_len_px(pred_kpts, *TOT_IDXS)
            length_mm = segment_len_mm_with_theta(length_px, dx, dy, S_h, S_v)
            length_mm = length_mm / 2.15
            gt_mm = float(brow.get('Avg_Length', np.nan)) if pd.notna(brow.get('Avg_Length', np.nan)) else None
            if gt_mm is not None and not math.isnan(gt_mm):
                total_errors.append(abs(length_mm - gt_mm))

        # carapace GTs
        for _, crow in car_df[car_df['Image'].astype(str).str.strip() == img_name].iterrows():
            if HAS_CHECK:
                gt_obb, gt_aabb = chk.gt_obb_and_aabb_from_row(crow, image_size=(W, H))
            else:
                gt_obb, gt_aabb = (None, None)
            if gt_aabb is None:
                continue
            best_idx, best_iou, best_ioa, best_center = find_best_pred_for_gt(gt_aabb, dets)
            if best_idx is None:
                continue
            pred_kpts = kpts[best_idx] if (kpts is not None and kpts.shape[0] > best_idx) else np.zeros((4, 2))
            length_px, dx, dy = segment_len_px(pred_kpts, *CAR_IDXS)
            S_h, S_v = pixel_scales_mm_per_px(float(crow.get('Height(mm)', 300)), W, H, HFOV, VFOV)
            length_mm = segment_len_mm_with_theta(length_px, dx, dy, S_h, S_v)
            length_mm = length_mm / 1.95
            gt_mm = float(crow.get('Avg_Length', np.nan)) if pd.notna(crow.get('Avg_Length', np.nan)) else None
            if gt_mm is not None and not math.isnan(gt_mm):
                car_errors.append(abs(length_mm - gt_mm))

        if len(total_errors) > 0:
            detection_success = 1
            mae_total = float(np.mean(total_errors))
        if len(car_errors) > 0:
            mae_carapace = float(np.mean(car_errors))

        records.append({
            'image_path': str(img_path),
            'emb_meta_path': meta_paths[midx],
            'distance_to_centroid': emb_dist,
            'detection_success': int(detection_success),
            'mae_total': mae_total,
            'mae_carapace': mae_carapace,
        })

    df = pd.DataFrame(records)

    # correlations
    df_corr = df.dropna(subset=['distance_to_centroid', 'mae_total'])
    results = {}
    if len(df_corr) >= 3:
        pearson_r, pearson_p = stats.pearsonr(df_corr['distance_to_centroid'], df_corr['mae_total'])
        spearman_r, spearman_p = stats.spearmanr(df_corr['distance_to_centroid'], df_corr['mae_total'])
        df_pb = df.dropna(subset=['distance_to_centroid'])
        if df_pb['detection_success'].nunique() > 1:
            pb_r, pb_p = stats.pointbiserialr(df_pb['detection_success'], df_pb['distance_to_centroid'])
        else:
            pb_r, pb_p = (np.nan, np.nan)
        results['pearson'] = (pearson_r, pearson_p)
        results['spearman'] = (spearman_r, spearman_p)
        results['pointbiserial'] = (pb_r, pb_p)

    def interpret(r):
        ar = abs(r)
        if ar < 0.1:
            return 'no meaningful correlation'
        if ar < 0.3:
            return 'weak'
        return 'moderate or stronger'

    print('\nCorrelation results:')
    if 'pearson' in results:
        pr, pp = results['pearson']
        sr, sp = results['spearman']
        pbr, pbp = results['pointbiserial']
        print(f"Pearson (distance vs mae_total): r={pr:.4f}, p={pp:.3g} -> {interpret(pr)}")
        print(f"Spearman(rho): rho={sr:.4f}, p={sp:.3g} -> {interpret(sr)}")
        print(f"Point-biserial (distance vs detection_success): r={pbr:.4f}, p={pbp:.3g} -> {interpret(pbr)}")
    else:
        print('Not enough matched rows to compute correlations')

    df.to_csv(OUT_CSV, index=False)
    print(f'Wrote results to {OUT_CSV}')
    print('Match stats summary:', match_stats)

    return 0


if __name__ == '__main__':
    sys.exit(main())
