#!/usr/bin/env python3
# ============================================================
# batch_eval_dual_sets_MULTI_models_2025_COMBINED_ONLY_vCALIBRATED.py
# ============================================================
# Evaluates YOLO models on 2025 test set (all ponds combined)
# Includes full refraction-aware calibration:
#   - HFOV/VFOV-based mm-per-px conversion
#   - Feret-angle normalization (rightward policy)
#   - Scaling of FeretX/Y if frame != 640x360
#   - Stable OBB thickness derivation
# ============================================================

from pathlib import Path
import math, re, os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

# ========= Editable paths =========
CAR_IMAGES_DIR = Path("/Users/taircarmon/Desktop/Uni-Projects/prawn-size-project/data/images/test_2025_gamma/test_images_2025_gamma/ALL_IMAGES_640x360")
BODY_IMAGES_DIR = CAR_IMAGES_DIR
CAR_XLSX = Path("/Users/taircarmon/Desktop/Uni-Projects/prawn-size-project/data/excel/TEST_2025_Carapace_WITH_OBB_v2.xlsx")
BODY_XLSX = Path("/Users/taircarmon/Desktop/Uni-Projects/prawn-size-project/data/excel/TEST_2025_Body_WITH_OBB_v2.xlsx")
MODEL_ROOTS = [Path("/Users/taircarmon/Desktop/Uni-Projects/prawn-size-project/models/TF/tf_grids/B3_final")]
OUT_ROOT = Path("/Users/taircarmon/Desktop/Uni-Projects/prawn-size-project/scripts/eval/outputs_TF/final/B3_final/test_on_2025")

# ========= Settings =========
CONF_TH = 0.25
DEVICE = None
HFOV_DERIVED_DEG = 76.2
VFOV_DERIVED_DEG = 46.0
IOU_MIN = 0.05      # loose IoU threshold (as in 2024 script)
IOA_MIN = 0.50      # intersection over GT >= 50% also counts as detection

KEYPOINT_NAMES = ["carapace-start", "eyes", "rostrum", "tail"]
CAR_IDXS, TOT_IDXS = (0, 1), (2, 3)
WORK_FRAME_W, WORK_FRAME_H = 640, 360  # reference for FeretX/Y scaling

# ============================================================
# Helper functions (with calibration logic)
# ============================================================
def _normalize_angle_rightward(angle_deg: float) -> float:
    """Force angle orientation so that horizontal vector points rightward (u_x >= 0)."""
    a = ((angle_deg + 180.0) % 360.0) - 180.0
    if math.cos(math.radians(a)) < 0:
        a = ((a + 180.0) + 180.0) % 360.0 - 180.0
    return a

def _maybe_scale_to_image(x, y, W, H):
    """If FeretX/Y measured on 640×360 but actual frame differs, apply linear scaling."""
    if W <= 0 or H <= 0:
        return x, y
    sx, sy = float(W) / WORK_FRAME_W, float(H) / WORK_FRAME_H
    return x * sx, y * sy

def obb_corners_from_feret(start_x, start_y, feret_len_px, thickness_px, angle_deg):
    θ = math.radians(angle_deg)
    dx, dy = feret_len_px * math.cos(θ), feret_len_px * math.sin(θ)
    cx, cy = start_x + dx / 2.0, start_y + dy / 2.0
    u = np.array([math.cos(θ), math.sin(θ)])
    n = np.array([-math.sin(θ), math.cos(θ)])
    L2, T2 = feret_len_px / 2.0, max(1.0, thickness_px / 2.0)
    c = np.array([cx, cy])
    return np.stack([
        c + u * L2 + n * T2,
        c + u * L2 - n * T2,
        c - u * L2 - n * T2,
        c - u * L2 + n * T2
    ])

def aabb_from_corners(corners):
    x, y = corners[:, 0], corners[:, 1]
    return [float(x.min()), float(y.min()), float(x.max()), float(y.max())]

def gt_obb_and_aabb_from_row(row: pd.Series, image_size=None):
    """Builds GT OBB and AABB from Feret-based Excel columns with calibration."""
    req = ["FeretX", "FeretY", "Feret", "Width", "Height_img", "px_per_mm"]
    if not all(k in row and pd.notna(row[k]) for k in req):
        return None, None

    if "Angle_draw" in row and pd.notna(row["Angle_draw"]):
        ang = float(row["Angle_draw"])
    elif "Angle" in row and pd.notna(row["Angle"]):
        ang = -float(row["Angle"])
    else:
        return None, None
    ang = _normalize_angle_rightward(ang)

    feret_len_px = max(1.0, float(row["Feret"]) * float(row["px_per_mm"]))
    thick = max(1.0, min(float(row["Width"]), float(row["Height_img"])))

    x, y = float(row["FeretX"]), float(row["FeretY"])
    if image_size is not None:
        x, y = _maybe_scale_to_image(x, y, *image_size)

    corners = obb_corners_from_feret(x, y, feret_len_px, thick, ang)
    return corners, aabb_from_corners(corners)

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
    """Intersection over ground-truth bbox."""
    ax1, ay1, ax2, ay2 = gt
    bx1, by1, bx2, by2 = pred
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_gt = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    return inter / (area_gt + 1e-9)

def center_in_bbox(cx: float, cy: float, b):
    x1, y1, x2, y2 = b
    return (x1 <= cx <= x2) and (y1 <= cy <= y2)

def find_best_pred_for_gt(gt_xyxy, det_xyxy_all: np.ndarray):
    """
    For this single GT:
    - check ALL predictions.
    - valid if IoU>=IOU_MIN or IoA>=IOA_MIN or center inside GT.
    - return index + scores of best match, or (None, ..) if no match.
    """
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

def deg2rad(d):
    return d * math.pi / 180.0

def pixel_scales_mm_per_px(distance_mm, img_w, img_h,
                           hfov_deg=HFOV_DERIVED_DEG, vfov_deg=VFOV_DERIVED_DEG):
    S_h = 2.0 * distance_mm * math.tan(deg2rad(hfov_deg) / 2.0) / float(img_w)
    S_v = 2.0 * distance_mm * math.tan(deg2rad(vfov_deg) / 2.0) / float(img_h)
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
    S_total = math.sqrt(
        (S_h * math.cos(math.radians(theta_norm))) ** 2 +
        (S_v * math.sin(math.radians(theta_norm))) ** 2
    )
    return length_px * S_total

# ============================================================
# Visualization
# ============================================================
def draw_vis(img, det_xyxy, kpts_xy, mode, gt_obb=None, gt_mm=None, pred_mm=None, iou=None, pond="Combined"):
    x1, y1, x2, y2 = map(int, det_xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
    if gt_obb is not None:
        cv2.polylines(img, [gt_obb.astype(np.int32)], True, (255, 0, 0), 2)
    if kpts_xy is not None and len(kpts_xy) >= 4:
        for (x, y) in kpts_xy:
            cv2.circle(img, (int(x), int(y)), 3, (0, 200, 0), -1)

    if mode == "carapace":
        color_line = (0, 200, 0)
    else:
        color_line = (0, 255, 255)

    if gt_mm is not None and pred_mm is not None:
        info = [
            f"{mode.capitalize()} | GT {gt_mm:.1f} mm | Pred {pred_mm:.1f} mm",
            f"IoU={iou:.3f} | {pond}"
        ]
    else:
        info = [
            f"{mode.capitalize()} | GT {gt_mm} mm | Pred {pred_mm}",
            f"IoU={iou} | {pond}"
        ]

    for i, line in enumerate(info):
        cv2.putText(
            img,
            line,
            (x1, max(20, y1 - 10 + 20 * i)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

# ============================================================
# Per-image logic
# ============================================================
def process_image(img_path, mode, model, df, out_vis_dir, save_visual):
    recs = []
    try:
        df_img = df[df["Image"].astype(str).str.strip() == img_path.name]
        if df_img.empty:
            return []

        results = model.predict(str(img_path), conf=CONF_TH, device=DEVICE, verbose=False)
        r0 = results[0]
        dets = r0.boxes.xyxy.cpu().numpy() if (r0.boxes is not None and r0.boxes.xyxy is not None) else np.zeros((0, 4))
        kpts = r0.keypoints.xy.cpu().numpy() if (hasattr(r0, "keypoints") and r0.keypoints is not None and r0.keypoints.xy is not None) else np.zeros((0, 4, 2))

        with Image.open(str(img_path)) as im:
            W, H = im.size

        print(f"[DEBUG] {img_path.name} | GT={len(df_img)} | Pred={len(dets)}")

        for i, (_, row) in enumerate(df_img.iterrows(), 1):
            gt_obb, gt_aabb = gt_obb_and_aabb_from_row(row, (W, H))
            if gt_aabb is None:
                continue

            gt_mm = float(row.get("Avg_Length", np.nan)) if pd.notna(row.get("Avg_Length", np.nan)) else None

            # Matching logic with IoU/IoA/center criteria (like 2024)
            best_idx, best_iou, best_ioa, best_center = find_best_pred_for_gt(gt_aabb, dets)

            if best_idx is None:
                # GT exists but no prediction matched -> MISS
                recs.append({
                    "image": img_path.name,
                    "mode": mode,
                    "GT_index": i,
                    "gt_mm": gt_mm,
                    "pred_mm": None,
                    "err_mm": None,
                    "mape": None,
                    "IoU": None,
                    "status": "MISS_NO_MATCH"
                })
                continue

            det_xyxy = dets[best_idx]
            kpts_xy = kpts[best_idx] if len(kpts) > best_idx else np.zeros((4, 2))

            height_mm = float(row.get("Height(mm)", 300))
            S_h, S_v = pixel_scales_mm_per_px(height_mm, W, H)

            if mode == "carapace":
                length_px, dx, dy = segment_len_px(kpts_xy, *CAR_IDXS)
            else:
                length_px, dx, dy = segment_len_px(kpts_xy, *TOT_IDXS)

            length_mm = segment_len_mm_with_theta(length_px, dx, dy, S_h, S_v)

            # ======== Empirical division correction (kept as in original) ========
            if mode == "carapace":
                length_mm = length_mm / 1.95
            else:  # body / total
                length_mm = length_mm / 2.15
            # =====================================================================

            err = length_mm - gt_mm if gt_mm else None
            mape = abs(err) / max(gt_mm, 1e-9) * 100 if gt_mm else None

            recs.append({
                "image": img_path.name,
                "mode": mode,
                "GT_index": i,
                "gt_mm": gt_mm,
                "pred_mm": length_mm,
                "err_mm": err,
                "mape": mape,
                "IoU": best_iou,
                "status": "OK"
            })

            if save_visual and best_iou is not None and best_iou > 0.1:
                img = cv2.imread(str(img_path))
                draw_vis(img, det_xyxy, kpts_xy, mode, gt_obb, gt_mm, length_mm, best_iou)
                out_vis_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_vis_dir / f"{img_path.stem}_GT{i}_{mode}.jpg"), img)

    except Exception as e:
        recs.append({
            "image": img_path.name,
            "mode": mode,
            "GT_index": None,
            "gt_mm": None,
            "pred_mm": None,
            "err_mm": None,
            "mape": None,
            "IoU": None,
            "status": f"ERR: {e}"
        })
    return recs

# ============================================================
# Summary and main
# ============================================================
def compute_summary(df):
    # All GT samples per mode
    car_all = df[df["mode"] == "carapace"]
    bod_all = df[df["mode"] == "body"]

    # Only successfully detected (status == "OK")
    car_ok = car_all[car_all["status"] == "OK"]
    bod_ok = bod_all[bod_all["status"] == "OK"]

    # Detection rates
    det_rate_car = (len(car_ok) / len(car_all) * 100.0) if len(car_all) > 0 else np.nan
    det_rate_tot = (len(bod_ok) / len(bod_all) * 100.0) if len(bod_all) > 0 else np.nan

    # Error statistics (computed only on OK rows)
    mae_car = car_ok["err_mm"].abs().mean() if not car_ok.empty else np.nan
    mpe_car = car_ok["mape"].mean() if not car_ok.empty else np.nan
    mae_tot = bod_ok["err_mm"].abs().mean() if not bod_ok.empty else np.nan
    mpe_tot = bod_ok["mape"].mean() if not bod_ok.empty else np.nan

    return pd.DataFrame(
        {
            "Detection Rate – carapace (%)": [det_rate_car],
            "Detection Rate – total (%)": [det_rate_tot],
            "MAE carapace (mm)": [mae_car],
            "MPE carapace (%)": [mpe_car],
            "MAE total (mm)": [mae_tot],
            "MPE total (%)": [mpe_tot],
        },
        index=["Combined"],
    )

def discover_models(roots):
    found = {}
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.pt"):
            key = str(p.relative_to(root)).replace(os.sep, "_")
            found[key] = p
            print(f"[DEBUG] Found model: {key}")
    print(f"[INFO] Found {len(found)} models.")
    return found

def run_for_model(label, path, car_df, body_df, all_tables):
    print(f"\n[INFO] ===== Running model '{label}' =====")
    model = YOLO(str(path))
    out_dir = OUT_ROOT / label
    out_car, out_body = out_dir / "vis_carapace", out_dir / "vis_body"
    rows = []

    car_images = sorted(CAR_IMAGES_DIR.glob("*.jpg"))
    body_images = sorted(BODY_IMAGES_DIR.glob("*.jpg"))
    print(f"[INFO] Carapace images: {len(car_images)}")
    print(f"[INFO] Body images    : {len(body_images)}")

    for i, img in enumerate(car_images, 1):
        rows.extend(process_image(img, "carapace", model, car_df, out_car, i <= 5))
    for i, img in enumerate(body_images, 1):
        rows.extend(process_image(img, "body", model, body_df, out_body, i <= 5))

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "summary.csv", index=False)
    summary_df = compute_summary(df)
    summary_df.to_csv(out_dir / "combined_summary.csv")
    all_tables[label] = summary_df
    print(f"[INFO] Saved summaries to {out_dir}")

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    car_df, body_df = pd.read_excel(CAR_XLSX), pd.read_excel(BODY_XLSX)
    model_map = discover_models(MODEL_ROOTS)
    if not model_map:
        print("[WARN] No models found.")
        return

    all_tables = {}
    for lbl, path in model_map.items():
        run_for_model(lbl, path, car_df, body_df, all_tables)

    final_xlsx = OUT_ROOT / "all_models_combined_summaries.xlsx"
    with pd.ExcelWriter(final_xlsx, engine="xlsxwriter") as w:
        for name, tbl in all_tables.items():
            safe_name = re.sub(r"[^A-Za-z0-9_]", "_", name)[:31]
            tbl.to_excel(w, sheet_name=safe_name)
        combined = pd.concat(
            [t.assign(Model=n) for n, t in all_tables.items()],
            ignore_index=True
        )
        combined.to_excel(w, sheet_name="ALL_MODELS", index=False)
    print(f"\n✅ Done. Combined Excel written to: {final_xlsx}")

if __name__ == "__main__":
    main()
