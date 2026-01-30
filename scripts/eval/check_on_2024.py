#!/usr/bin/env python3
# ============================================================
# batch_eval_dual_sets_MULTI_models_final_alias_only_v2.py
# ============================================================
# Evaluates models from:
#   - models/2024  (all *.pt)
#   - models/2025
#   - models/TF
#
# On 2024 ORIGINAL SIZE images (carapace/body).
#
# Detection logic (IMPORTANT):
# - For each image & mode (carapace/body):
#     * Load its single GT bbox from Excel.
#     * Run YOLO once, get all predictions.
#     * GT is "detected" if ANY prediction satisfies:
#           IoU >= IOU_MIN  OR
#           IoA >= IOA_MIN  OR
#           center of pred inside GT.
# - Failure only if:
#     GT exists AND no prediction matches it.
#
# Outputs per model:
# - summary.csv  (per-image stats)
# - pond_summary.csv (per-pond metrics)
# - <model>_pond_summary.xlsx
# - failed_detections/ (visual debug: GT red, preds yellow)
# ============================================================

from pathlib import Path
import math
import json
import re
import ast
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

# ========= Editable paths =========
CAR_IMAGES_DIR = Path("/Users/taircarmon/Desktop/Uni-Projects/prawn-size-project/data/images/test_2024_originalSize/test_images_2024_orginalSize/carapace")
BODY_IMAGES_DIR = Path("/Users/taircarmon/Desktop/Uni-Projects/prawn-size-project/data/images/test_2024_originalSize/test_images_2024_orginalSize/body")

CAR_XLSX = Path("/Users/taircarmon/Desktop/Uni-Projects/prawn-size-project/data/excel/updated_filtered_data_with_lengths_carapace-all.xlsx")
BODY_XLSX = Path("/Users/taircarmon/Desktop/Uni-Projects/prawn-size-project/data/excel/updated_filtered_data_with_lengths_body-all.xlsx")

MODEL_ROOTS = [
    Path("../../models/TF/tf_grids/B3_final"),
    # Path("../../models/TF/tf_grids/B2"),
    # Path("../../models/2024"),
]

OUT_ROOT = Path("/Users/taircarmon/Desktop/Uni-Projects/prawn-size-project/scripts/eval/outputs_TF/final/B3_final/test_on_2024")

# ========= Camera / geometry params =========
HFOV_DERIVED_DEG = 76.2
VFOV_DERIVED_DEG = 46.0

# Original video frame size (for GT bbox normalization)
ORIG_FRAME_W = 5312
ORIG_FRAME_H = 2988

# ========= Inference / matching settings =========
CONF_TH = 0.25
IOU_MIN = 0.05      # loose IoU threshold
IOA_MIN = 0.50      # intersection over GT >= 50% also counts
DEVICE = None       # "cuda:0" or None

KEYPOINT_NAMES = ["carapace-start", "eyes", "rostrum", "tail"]
CAR_IDXS = (0, 1)
TOT_IDXS = (2, 3)

GX3_RE = re.compile(r"(GX\d+_\d+_\d+)")
GX2_RE = re.compile(r"(GX\d+_\d+)")

# ============================================================
# Pond mapping
# ============================================================
def normalize_pond_from_excels(raw) -> str:
    if raw is None:
        return "Unknown"
    s = str(raw).strip().lower()
    if s == "" or s == "nan":
        return "Unknown"
    if "right" in s:
        return "Circular 1"
    if "left" in s:
        return "Circular 2"
    if "square" in s or s == "car":
        return "Square"
    return "Unknown"

def pond_group_for_label(key: str, mode: str,
                         car_row: dict | None,
                         body_row: dict | None) -> str:
    primary_row = car_row if mode == "carapace" else body_row
    secondary_row = body_row if mode == "carapace" else car_row
    pond_primary = normalize_pond_from_excels(primary_row.get("Pond_Type")) if primary_row else "Unknown"
    if pond_primary != "Unknown":
        return pond_primary
    pond_secondary = normalize_pond_from_excels(secondary_row.get("Pond_Type")) if secondary_row else "Unknown"
    return pond_secondary

# ============================================================
# Label / Excel helpers
# ============================================================
def remap_model_kpts_to_true(kpts_xy_model: np.ndarray) -> np.ndarray:
    assert kpts_xy_model.shape[0] >= 4
    return kpts_xy_model.astype(float)

def extract_label_key(filename: str) -> str | None:
    m = GX3_RE.search(filename)
    if m:
        return m.group(1)
    m = GX2_RE.search(filename)
    return m.group(1) if m else None

def find_label_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if str(c).strip().lower() == "label":
            return c
    raise KeyError("Could not find a 'Label' column in the Excel file.")

def row_by_label(df: pd.DataFrame, key: str) -> dict | None:
    col = find_label_col(df)
    s = df[col].astype(str).str.strip()
    m = df[s == key]
    if not m.empty:
        return m.iloc[0].to_dict()
    m = df[s.str.contains(re.escape(key), na=False)]
    if not m.empty:
        return m.iloc[0].to_dict()
    parts = key.split("_")
    if len(parts) >= 2:
        short_key = "_".join(parts[:2])
        m = df[s == short_key]
        if not m.empty:
            return m.iloc[0].to_dict()
        m = df[s.str.contains(re.escape(short_key), na=False)]
        if not m.empty:
            return m.iloc[0].to_dict()
    return None

def parse_bbox(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        return [float(v) for v in value[:4]]
    if isinstance(value, str):
        v = value.strip()
        # try literal list/tuple
        try:
            arr = ast.literal_eval(v)
            if isinstance(arr, (list, tuple)) and len(arr) >= 4:
                return [float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])]
        except Exception:
            pass
        # try JSON
        try:
            obj = json.loads(v)
            if isinstance(obj, (list, tuple)) and len(obj) >= 4:
                return [float(obj[0]), float(obj[1]), float(obj[2]), float(obj[3])]
        except Exception:
            pass
    return None

def bbox_to_xyxy(b, img_w, img_h):
    if b is None:
        return None
    b = list(b[:4])
    is_norm = all(0.0 <= v <= 1.2 for v in b)
    if is_norm:
        scale = np.array([img_w, img_h, img_w, img_h], dtype=float)
        b = (np.array(b, dtype=float) * scale).tolist()
    x, y, w_or_x2, h_or_y2 = b
    if w_or_x2 > x and h_or_y2 > y:
        # already xyxy
        return [float(x), float(y), float(w_or_x2), float(h_or_y2)]
    else:
        # xywh
        return [float(x), float(y),
                float(x + w_or_x2),
                float(y + h_or_y2)]

def rescale_xyxy(bxyxy, src_w, src_h, dst_w, dst_h):
    if bxyxy is None:
        return None
    x1, y1, x2, y2 = bxyxy
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    x1r = max(0.0, min(dst_w, x1 * sx))
    y1r = max(0.0, min(dst_h, y1 * sy))
    x2r = max(0.0, min(dst_w, x2 * sx))
    y2r = max(0.0, min(dst_h, y2 * sy))
    return [x1r, y1r, x2r, y2r]

# ============================================================
# Geometry
# ============================================================
def deg2rad(d): return d * math.pi / 180.0

def pixel_scales_mm_per_px(distance_mm: float, img_w: int, img_h: int,
                           hfov_deg: float = HFOV_DERIVED_DEG,
                           vfov_deg: float = VFOV_DERIVED_DEG) -> tuple[float, float]:
    S_h = 2.0 * distance_mm * math.tan(deg2rad(hfov_deg) / 2.0) / float(img_w)
    S_v = 2.0 * distance_mm * math.tan(deg2rad(vfov_deg) / 2.0) / float(img_h)
    return S_h, S_v

def segment_len_px(kpts_xy: np.ndarray, i0: int, i1: int) -> tuple[float, float, float]:
    x0, y0 = kpts_xy[i0]
    x1, y1 = kpts_xy[i1]
    dx, dy = float(x1 - x0), float(y1 - y0)
    return float(np.hypot(dx, dy)), dx, dy

def segment_len_mm_with_theta(length_px: float, dx: float, dy: float,
                              S_h: float, S_v: float) -> float:
    theta_rad = math.atan2(dy, dx)
    theta_deg = math.degrees(theta_rad)
    theta_norm = min(abs(theta_deg) % 180, 180 - (abs(theta_deg) % 180))
    S_total = math.sqrt(
        (S_h * math.cos(math.radians(theta_norm))) ** 2
        + (S_v * math.sin(math.radians(theta_norm))) ** 2
    )
    return length_px * S_total

# ============================================================
# Matching helpers: IoU, IoA, center
# ============================================================
def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union

def ioa_xyxy(gt, pred):
    ax1, ay1, ax2, ay2 = gt
    bx1, by1, bx2, by2 = pred
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_gt = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    return inter / (area_gt + 1e-9)

def center_in_bbox(cx: float, cy: float, b: list[float]) -> bool:
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
        iou = iou_xyxy(gt_xyxy, d)
        ioa = ioa_xyxy(gt_xyxy, d)
        cx = 0.5 * (d[0] + d[2])
        cy = 0.5 * (d[1] + d[3])
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

# ============================================================
# Visualization
# ============================================================
def draw_vis(img_bgr, det_xyxy, kpts_xy, texts, mode: str,
             gt_xyxy=None, kp_names=None, gt_car_mm=None, gt_tot_mm=None):
    # pred bbox (yellow)
    if det_xyxy is not None:
        x1, y1, x2, y2 = map(int, det_xyxy)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # GT bbox (red)
    if gt_xyxy is not None:
        gx1, gy1, gx2, gy2 = map(int, gt_xyxy)
        cv2.rectangle(img_bgr, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)

    # keypoints
    if kpts_xy is not None:
        for i, (x, y) in enumerate(kpts_xy):
            name = kp_names[i] if (kp_names is not None and i < len(kp_names)) else str(i)
            color = (0, 200, 0) if i < 4 else (180, 180, 180)
            cv2.circle(img_bgr, (int(x), int(y)), 4, color, -1)
            cv2.putText(img_bgr, name, (int(x) + 6, int(y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2, cv2.LINE_AA)

    y0txt = 25
    for i, t in enumerate(texts):
        yy = y0txt + i * 20
        cv2.putText(img_bgr, t, (10, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)

# ============================================================
# Per-image evaluation
# ============================================================
def process_image(image_path: Path, mode: str, model: YOLO,
                  car_df: pd.DataFrame, body_df: pd.DataFrame,
                  out_vis_dir: Path, save_visual: bool):

    rec = {
        "image": image_path.name,
        "path": str(image_path),
        "mode": mode,
        "label": None,
        "status": "OK",
        "model_name": None,
        "pond_raw": None,
        "pond_group": None,
        "car_px": None,
        "car_mm": None,
        "gt_car_mm": None,
        "car_err_mm": None,
        "car_mape": None,
        "tot_px": None,
        "tot_mm": None,
        "gt_tot_mm": None,
        "tot_err_mm": None,
        "tot_mape": None,
        "IoU": None,
        "IoA": None,
        "center_in": None,
    }

    try:
        assert image_path.exists(), f"Image not found: {image_path}"
        key = extract_label_key(image_path.name)
        if key is None:
            raise ValueError("Could not extract label from filename")
        rec["label"] = key

        car_row = row_by_label(car_df, key)
        body_row = row_by_label(body_df, key)
        if car_row is None or body_row is None:
            raise ValueError("Row not found in one/both Excel sheets")

        pond_group = pond_group_for_label(key, mode, car_row, body_row)
        rec["pond_group"] = pond_group
        rec["pond_raw"] = (
            car_row.get("Pond_Type") if mode == "carapace"
            else body_row.get("Pond_Type")
        )

        # Choose GT row per mode
        gt_row = car_row if mode == "carapace" else body_row
        gt_bbox_raw = gt_row.get("BoundingBox_1", None)
        gt_bbox = parse_bbox(gt_bbox_raw)
        if gt_bbox is None:
            raise ValueError("No GT bbox parsed from Excel")

        with Image.open(str(image_path)) as im:
            W, H = im.size

        gt_xyxy_full = bbox_to_xyxy(gt_bbox, ORIG_FRAME_W, ORIG_FRAME_H)
        gt_xyxy = rescale_xyxy(gt_xyxy_full, ORIG_FRAME_W, ORIG_FRAME_H, W, H) if gt_xyxy_full else None
        if gt_xyxy is None:
            raise ValueError("Could not construct GT bbox in image space")

        # run YOLO
        results = model.predict(
            source=str(image_path),
            conf=CONF_TH,
            device=DEVICE,
            verbose=False,
            save=False
        )
        if not results:
            raise RuntimeError("No results returned by model")
        r0 = results[0]
        det_xyxy_all = (
            r0.boxes.xyxy.cpu().numpy()
            if (r0.boxes is not None and r0.boxes.xyxy is not None)
            else np.zeros((0, 4))
        )
        kpts_all = (
            r0.keypoints.xy.cpu().numpy()
            if (hasattr(r0, "keypoints") and r0.keypoints is not None and r0.keypoints.xy is not None)
            else np.zeros((0, 4, 2))
        )

        # MATCH: check ALL predictions against this GT
        best_idx, best_iou, best_ioa, best_center = find_best_pred_for_gt(gt_xyxy, det_xyxy_all)

        fail_dir = out_vis_dir.parent / "failed_detections"
        fail_dir.mkdir(parents=True, exist_ok=True)

        if best_idx is None:
            # True MISS: GT exists but no prediction matched
            rec["status"] = "MISS_NO_MATCH"

            # save fail visualization
            img_bgr = cv2.imread(str(image_path))
            if img_bgr is not None:
                # draw all preds in yellow
                for d in det_xyxy_all:
                    x1, y1, x2, y2 = map(int, d)
                    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)
                # draw GT in red
                gx1, gy1, gx2, gy2 = map(int, gt_xyxy)
                cv2.rectangle(img_bgr, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)
                cv2.putText(img_bgr, "GT (red), YOLO preds (yellow) - NO MATCH",
                            (max(10, gx1), max(20, gy1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 2, cv2.LINE_AA)
                out_fail = fail_dir / f"{image_path.stem}_{mode}_fail.jpg"
                cv2.imwrite(str(out_fail), img_bgr)

            return rec  # no length eval because אין דיטקציה

        # we have a matching prediction
        rec["IoU"] = best_iou
        rec["IoA"] = best_ioa
        rec["center_in"] = best_center

        det_xyxy = det_xyxy_all[best_idx]
        kpts_xy_model = kpts_all[best_idx] if kpts_all.shape[0] > best_idx else None
        kpts_xy = remap_model_kpts_to_true(kpts_xy_model) if (kpts_xy_model is not None and kpts_xy_model.shape[0] >= 4) else None

        img_bgr = None
        if save_visual:
            img_bgr = cv2.imread(str(image_path))

        # common GT lengths
        gt_car_mm = float(car_row["Avg_Length"]) if pd.notna(car_row.get("Avg_Length", np.nan)) else None
        gt_tot_mm = float(body_row["Avg_Length"]) if pd.notna(body_row.get("Avg_Length", np.nan)) else None
        car_height_mm = float(car_row["Height(mm)"])
        body_height_mm = float(body_row["Height(mm)"])

        # length eval by mode
        if mode == "carapace" and kpts_xy is not None:
            car_px, dx, dy = segment_len_px(kpts_xy, *CAR_IDXS)
            S_h, S_v = pixel_scales_mm_per_px(car_height_mm, W, H)
            car_mm = segment_len_mm_with_theta(car_px, dx, dy, S_h, S_v)
            rec["car_px"] = car_px
            rec["car_mm"] = car_mm
            rec["gt_car_mm"] = gt_car_mm
            if gt_car_mm is not None:
                err = car_mm - gt_car_mm
                rec["car_err_mm"] = err
                rec["car_mape"] = abs(err) / max(gt_car_mm, 1e-9) * 100.0

            if save_visual and img_bgr is not None:
                text_lines = [
                    f"Carapace: pred {car_mm:.1f} mm (px {car_px:.1f}) | GT {gt_car_mm:.1f} mm" if gt_car_mm is not None else
                    f"Carapace: pred {car_mm:.1f} mm (px {car_px:.1f})",
                    f"IoU={best_iou:.3f}, IoA={best_ioa:.3f}, center_in={best_center}",
                    f"{pond_group}"
                ]
                draw_vis(img_bgr, det_xyxy, kpts_xy, text_lines,
                         mode="carapace", gt_xyxy=gt_xyxy,
                         kp_names=KEYPOINT_NAMES,
                         gt_car_mm=gt_car_mm, gt_tot_mm=None)
                out_img = out_vis_dir / f"{image_path.stem}_{mode}_vis.jpg"
                out_vis_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_img), img_bgr)

        elif mode == "body" and kpts_xy is not None:
            tot_px, dx, dy = segment_len_px(kpts_xy, *TOT_IDXS)
            S_h, S_v = pixel_scales_mm_per_px(body_height_mm, W, H)
            tot_mm = segment_len_mm_with_theta(tot_px, dx, dy, S_h, S_v)
            rec["tot_px"] = tot_px
            rec["tot_mm"] = tot_mm
            rec["gt_tot_mm"] = gt_tot_mm
            if gt_tot_mm is not None:
                err = tot_mm - gt_tot_mm
                rec["tot_err_mm"] = err
                rec["tot_mape"] = abs(err) / max(gt_tot_mm, 1e-9) * 100.0

            if save_visual and img_bgr is not None:
                text_lines = [
                    f"Total: pred {tot_mm:.1f} mm (px {tot_px:.1f}) | GT {gt_tot_mm:.1f} mm" if gt_tot_mm is not None else
                    f"Total: pred {tot_mm:.1f} mm (px {tot_px:.1f})",
                    f"IoU={best_iou:.3f}, IoA={best_ioa:.3f}, center_in={best_center}",
                    f"{pond_group}"
                ]
                draw_vis(img_bgr, det_xyxy, kpts_xy, text_lines,
                         mode="body", gt_xyxy=gt_xyxy,
                         kp_names=KEYPOINT_NAMES,
                         gt_car_mm=None, gt_tot_mm=gt_tot_mm)
                out_img = out_vis_dir / f"{image_path.stem}_{mode}_vis.jpg"
                out_vis_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_img), img_bgr)

    except Exception as e:
        rec["status"] = f"ERR: {e}"

    return rec

# ============================================================
# Utilities
# ============================================================
def collect_images(folder: Path):
    assert folder.exists(), f"Images dir not found: {folder}"
    return sorted([p for p in folder.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])

def fmt_val(x, is_percent=False):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "N/A"
    if is_percent:
        return f"{x:.3f}%"
    return f"{x:.3f}mm"

def compute_pond_table(df: pd.DataFrame) -> pd.DataFrame:
    report_cols = ["Circular 1", "Circular 2", "Square", "Combined"]

    def mean_nonnull(series):
        s = pd.to_numeric(series, errors="coerce").dropna()
        return float(s.mean()) if not s.empty else None

    def pond_metrics(sub_df: pd.DataFrame):
        out = {}
        car = sub_df[sub_df["mode"] == "carapace"]
        bod = sub_df[sub_df["mode"] == "body"]

        # detection rate per GT (status=="OK")
        dr_car = (car["status"] == "OK").mean() * 100.0 if not car.empty else None
        dr_tot = (bod["status"] == "OK").mean() * 100.0 if not bod.empty else None

        car_ok = car[car["status"] == "OK"]
        bod_ok = bod[bod["status"] == "OK"]

        mae_car = mean_nonnull(car_ok["car_err_mm"].abs()) if not car_ok.empty else None
        mape_car = mean_nonnull(car_ok["car_mape"]) if not car_ok.empty else None

        mae_tot = mean_nonnull(bod_ok["tot_err_mm"].abs()) if not bod_ok.empty else None
        mape_tot = mean_nonnull(bod_ok["tot_mape"]) if not bod_ok.empty else None

        out["Detection Rate – carapace (%)"] = dr_car
        out["Detection Rate – total (%)"] = dr_tot
        out["MAE carapace (mm)"] = mae_car
        out["MPE carapace (%)"] = mape_car
        out["MAE total (mm)"] = mae_tot
        out["MPE total (%)"] = mape_tot
        return out

    valid_ponds = ["Circular 1", "Circular 2", "Square"]

    pond_groups = {
            "Circular 1": df[df["pond_group"] == "Circular 1"],
            "Circular 2": df[df["pond_group"] == "Circular 2"],
            "Square": df[df["pond_group"] == "Square"],
            "Combined": df[df["pond_group"].isin(valid_ponds)],  # <--- התיקון
        }

    raw = {name: pond_metrics(sub) for name, sub in pond_groups.items()}

    rows = [
        "Detection Rate – carapace (%)",
        "Detection Rate – total (%)",
        "MAE carapace (mm)",
        "MPE carapace (%)",
        "MAE total (mm)",
        "MPE total (%)",
    ]

    table = {col: [] for col in report_cols}
    for col in report_cols:
        m = raw[col]
        table[col].append(fmt_val(m["Detection Rate – carapace (%)"], is_percent=True))
        table[col].append(fmt_val(m["Detection Rate – total (%)"], is_percent=True))
        table[col].append(fmt_val(m["MAE carapace (mm)"]))
        table[col].append(fmt_val(m["MPE carapace (%)"], is_percent=True))
        table[col].append(fmt_val(m["MAE total (mm)"]))
        table[col].append(fmt_val(m["MPE total (%)"], is_percent=True))

    pond_summary_df = pd.DataFrame(table, index=rows, columns=report_cols)
    return pond_summary_df

# ============================================================
# Model discovery (same logic, with preferences)
# ============================================================
def make_model_label(root: Path, pt_path: Path) -> str:
    name = pt_path.name.lower()
    stem = pt_path.stem

    for suf in ["_best", "_last"]:
        if stem.lower().endswith(suf):
            stem = stem[:-len(suf)]

    parent = pt_path.parent.name
    if parent.lower() in {"weights", "runs", "train", "models"}:
        parent = pt_path.parent.parent.name

    if name in {"best.pt", "last.pt"}:
        return parent

    if stem and stem not in {"best", "last"}:
        if parent and parent not in stem and not stem.startswith(parent):
            return f"{parent}_{stem}"
        return stem

    return parent or stem

def discover_models_across_roots(roots: list[Path]) -> dict[str, Path]:
    candidates = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.pt"):
            if not p.is_file():
                continue
            label = make_model_label(root, p)
            candidates.append((label, p, root))

    if not candidates:
        print(f"[WARN] No .pt files found under roots: {', '.join(str(r) for r in roots)}")
        return {}

    grouped: dict[str, list[Path]] = {}
    def score(p: Path) -> tuple:
        n = p.name.lower()
        pref = 0
        if n.endswith("_best.pt") or n == "best.pt":
            pref = 4
        elif n.endswith("_last.pt") or n == "last.pt":
            pref = 2
        if "final_alias" in n:
            pref += 1
        return (pref, p.stat().st_mtime)

    for label, p, _root in candidates:
        grouped.setdefault(label, []).append(p)

    chosen: dict[str, Path] = {}
    for label, plist in grouped.items():
        plist_sorted = sorted(plist, key=score, reverse=True)
        chosen[label] = plist_sorted[0]
    return chosen

# ============================================================
# Per-model run
# ============================================================
def run_for_model(model_label: str, weights_path: Path,
                  car_df: pd.DataFrame, body_df: pd.DataFrame,
                  all_model_tables: dict):
    print(f"\n[INFO] ===== Running model '{model_label}' =====")
    print(f"[INFO] Using weights: {weights_path}")

    out_dir = OUT_ROOT / model_label
    out_dir.mkdir(parents=True, exist_ok=True)

    out_vis_car = out_dir / "vis_carapace"
    out_vis_body = out_dir / "vis_body"
    out_vis_car.mkdir(parents=True, exist_ok=True)
    out_vis_body.mkdir(parents=True, exist_ok=True)

    car_images = collect_images(CAR_IMAGES_DIR)
    body_images = collect_images(BODY_IMAGES_DIR)
    print(f"[INFO] Carapace images: {len(car_images)}")
    print(f"[INFO] Body images    : {len(body_images)}")

    model = YOLO(str(weights_path))

    rows = []

    for i, img_path in enumerate(car_images, 1):
        save_vis = (i <= 5)
        rec = process_image(img_path, "carapace", model,
                            car_df, body_df,
                            out_vis_car, save_vis)
        rec["model_name"] = model_label
        rows.append(rec)

    for i, img_path in enumerate(body_images, 1):
        save_vis = (i <= 5)
        rec = process_image(img_path, "body", model,
                            car_df, body_df,
                            out_vis_body, save_vis)
        rec["model_name"] = model_label
        rows.append(rec)

    df = pd.DataFrame(rows)
    summary_csv = out_dir / "summary.csv"
    df.to_csv(summary_csv, index=False)
    print(f"[INFO] Wrote per-image summary CSV: {summary_csv}")

    pond_summary_df = compute_pond_table(df)
    pond_csv = out_dir / "pond_summary.csv"
    pond_summary_df.to_csv(pond_csv)
    print(f"[INFO] Wrote pond summary CSV: {pond_csv}")

    per_model_xlsx = out_dir / f"{model_label}_pond_summary.xlsx"
    with pd.ExcelWriter(per_model_xlsx, engine="xlsxwriter") as writer:
        pond_summary_df.to_excel(writer, sheet_name="POND_SUMMARY", index=True)
    print(f"[INFO] Wrote per-model Excel: {per_model_xlsx}")

    all_model_tables[model_label] = pond_summary_df.copy()

# ============================================================
# Main
# ============================================================
def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    assert CAR_XLSX.exists() and BODY_XLSX.exists(), "Excel files not found."

    car_df = pd.read_excel(CAR_XLSX)
    body_df = pd.read_excel(BODY_XLSX)

    model_map = discover_models_across_roots(MODEL_ROOTS)
    if not model_map:
        return

    all_model_tables: dict[str, pd.DataFrame] = {}
    for model_label, weights_path in sorted(model_map.items()):
        run_for_model(model_label, weights_path, car_df, body_df, all_model_tables)

    final_xlsx = OUT_ROOT / "all_models_pond_summaries.xlsx"
    with pd.ExcelWriter(final_xlsx, engine="xlsxwriter") as writer:
        for mname, tbl in all_model_tables.items():
            sheet_name = mname[:31]
            tbl.to_excel(writer, sheet_name=sheet_name, index=True)

        stacked = []
        for mname, tbl in all_model_tables.items():
            t = tbl.copy()
            t.insert(0, "Metric", t.index)
            t.insert(0, "Model", mname)
            t.reset_index(drop=True, inplace=True)
            stacked.append(t)
        if stacked:
            combined_df = pd.concat(stacked, ignore_index=True)
            combined_df.to_excel(writer, sheet_name="ALL_MODELS", index=False)

    print(f"\n[INFO] Wrote final Excel with all models' pond tables: {final_xlsx}")
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
