#!/usr/bin/env python3
"""
visualize_single_prediction.py

Run a YOLO model on a single test image and visualize the predicted prawn length (mm).
This script reuses the same calibration logic used in check_on_2025.py.

Usage examples:
    python scripts/eval/visualize_single_prediction.py \
        --model models/2025/YOLOv11n_train_on_2025_all_pose_300ep_best.pt \
        --image data/images/test_2025_gamma/test_images_2025_gamma/ALL_IMAGES_640x360/some_image.jpg \
        --mode carapace --out out_vis.jpg --show

If the image exists in the project's Excel annotation files, the script will read Height(mm)
and Avg_Length from the corresponding sheet to compute accurate mm-per-px scaling and show GT.
Otherwise a default Height(mm)=300 will be assumed (same fallback as original script).
"""

from pathlib import Path
import math
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
import shutil

# -------------------- Configuration (defaults copied from original) --------------------
CONF_TH = 0.25
DEVICE = None
HFOV_DERIVED_DEG = 76.2
VFOV_DERIVED_DEG = 46.0
WORK_FRAME_W, WORK_FRAME_H = 640, 360
CAR_IDXS, TOT_IDXS = (0, 1), (2, 3)  # keypoint index pairs used for length computation
DEFAULT_HEIGHT_MM = 300.0

# Default Excel annotation files used by the project (will be used to look up Height(mm) and GT)
CAR_XLSX = Path("/Users/taircarmon/Desktop/Uni-Projects/prawn-size-project/data/excel/TEST_2025_Carapace_WITH_OBB_v2.xlsx")
BODY_XLSX = Path("/Users/taircarmon/Desktop/Uni-Projects/prawn-size-project/data/excel/TEST_2025_Body_WITH_OBB_v2.xlsx")

# paper-friendly colors (hex -> RGB -> BGR for OpenCV)
# #0072B2 (blue), #009E73 (green), #D55E00 (orange)
BOX_COLOR = (178, 114, 0)    # BGR for #0072B2
KP_COLOR = (120, 120, 120)   # grey for keypoints
CAR_LINE_COLOR = (115, 158, 0)  # BGR for #009E73 (carapace)
TOT_LINE_COLOR = (0, 94, 213)   # BGR for #D55E00 (total/body)
TEXT_COLOR = (255, 255, 255)

# -------------------- Helpers (reused logic) --------------------

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


# -------------------- Visualization helpers --------------------

def draw_prediction(img, det_xyxy, kpts_xy, mode, pred_mm, gt_mm=None, iou=None):
    # copy image to avoid modifying original
    out = img.copy()
    H, W = out.shape[:2]

    if det_xyxy is not None:
        xA, yA, xB, yB = map(int, det_xyxy)
        cv2.rectangle(out, (xA, yA), (xB, yB), BOX_COLOR, 2)

    if kpts_xy is not None and len(kpts_xy) >= 4:
        # draw all keypoints and a connecting line for the chosen segment
        for (x, y) in kpts_xy:
            cv2.circle(out, (int(x), int(y)), 4, KP_COLOR, -1)

        if mode == 'carapace':
            i0, i1 = CAR_IDXS
            color = CAR_LINE_COLOR
        else:
            i0, i1 = TOT_IDXS
            color = TOT_LINE_COLOR

        x0, y0 = kpts_xy[i0]
        x1_kp, y1_kp = kpts_xy[i1]
        cv2.line(out, (int(x0), int(y0)), (int(x1_kp), int(y1_kp)), color, 3)

    info_lines = [f"Mode: {mode}", f"Pred: {pred_mm:.1f} mm"]
    if gt_mm is not None:
        info_lines.insert(1, f"GT: {gt_mm:.1f} mm")
    if iou is not None:
        info_lines.append(f"IoU: {iou:.3f}")

    # Put info below the bbox (preferred). If there's no bbox or not enough space,
    # place above bbox or at bottom of image.
    line_h = 22
    if det_xyxy is not None:
        tx = xA
        start_y = yB + 10
        total_h = line_h * len(info_lines)
        if start_y + total_h > H - 5:
            # not enough space below - place above bbox
            start_y = yA - 10 - total_h
            if start_y < 5:
                start_y = 5
    else:
        tx = 10
        start_y = H - 10 - line_h * len(info_lines)
        if start_y < 5:
            start_y = 5

    # Draw text lines
    for i, line in enumerate(info_lines):
        y = int(start_y + i * line_h + 16)
        # text shadow for readability
        cv2.putText(out, line, (int(tx) + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(out, line, (int(tx), y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)

    return out


# -------------------- Main flow --------------------

def load_annotation_table(mode):
    if mode == 'carapace' and CAR_XLSX.exists():
        return pd.read_excel(CAR_XLSX)
    if mode == 'body' and BODY_XLSX.exists():
        return pd.read_excel(BODY_XLSX)
    return None


def find_annotation_for_image(df, image_name):
    if df is None:
        return None
    df_img = df[df['Image'].astype(str).str.strip() == image_name]
    if df_img.empty:
        return None
    # return first match
    return df_img.iloc[0]


def main():
    pass


# === Top-level runner (automatically run on script execution) ===
def run_visualization(model_path, img_path, mode, out_path=None, show=False, idx=0):
    """Core logic so it can be called programmatically. (Moved to top-level)"""
    model_path = Path(model_path)
    img_path = Path(img_path)

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        return

    # load annotations (if available)
    ann_tbl = load_annotation_table(mode)
    ann_row = find_annotation_for_image(ann_tbl, img_path.name)
    gt_mm = float(ann_row.get('Avg_Length')) if (ann_row is not None and pd.notna(ann_row.get('Avg_Length'))) else None
    height_mm = float(ann_row.get('Height(mm)')) if (ann_row is not None and pd.notna(ann_row.get('Height(mm)'))) else DEFAULT_HEIGHT_MM

    print(f"Using Height(mm)={height_mm} mm (GT length={gt_mm})")

    # load model and predict
    model = YOLO(str(model_path))
    results = model.predict(str(img_path), conf=CONF_TH, device=DEVICE, verbose=False)
    r0 = results[0]

    dets = r0.boxes.xyxy.cpu().numpy() if (r0.boxes is not None and r0.boxes.xyxy is not None) else np.zeros((0, 4))
    scores = r0.boxes.conf.cpu().numpy() if (r0.boxes is not None and r0.boxes.conf is not None) else np.zeros((dets.shape[0],))
    kpts = r0.keypoints.xy.cpu().numpy() if (hasattr(r0, 'keypoints') and r0.keypoints is not None and r0.keypoints.xy is not None) else np.zeros((0, 4, 2))

    if dets.shape[0] == 0:
        print("No detections found in image.")
        return

    # choose detection: by provided idx or highest confidence
    if idx == 0:
        # pick highest confidence
        best_idx = int(np.argmax(scores)) if scores.size > 0 else 0
    else:
        best_idx = idx - 1
        best_idx = max(0, min(best_idx, dets.shape[0] - 1))

    det_xyxy = dets[best_idx]
    kpts_xy = kpts[best_idx] if len(kpts) > best_idx else np.zeros((4, 2))

    # get image size
    with Image.open(str(img_path)) as im:
        W, H = im.size

    # compute pixel scales and predicted length (mm)
    S_h, S_v = pixel_scales_mm_per_px(height_mm, W, H)

    if mode == 'carapace':
        length_px, dx, dy = segment_len_px(kpts_xy, *CAR_IDXS)
    else:
        length_px, dx, dy = segment_len_px(kpts_xy, *TOT_IDXS)

    length_mm = segment_len_mm_with_theta(length_px, dx, dy, S_h, S_v)

    # empirical division correction as used in original script
    if mode == 'carapace':
        length_mm = length_mm / 1.95
    else:
        length_mm = length_mm / 2.15

    print(f"Predicted length: {length_mm:.2f} mm")

    # draw visualization
    img_cv = cv2.imread(str(img_path))
    vis = draw_prediction(img_cv, det_xyxy, kpts_xy, mode, length_mm, gt_mm)

    if out_path:
        outp = Path(out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(outp), vis)
        print(f"Saved visualization to {outp}")

    if show:
        cv2.imshow('Prediction', vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def automatic_run_on_random():
    """Automatically run both CL (carapace) and TL (total/body) visualizations on a random
    image from the test set. Saves two images with suffixes _CL and _TL to the outputs folder.
    Also copies the original image to the output folder as <image>_orig.jpg.
    This function is non-interactive (no GUI) and uses the hard-coded model path requested by the user.
    """
    import random

    MODEL_PATH = Path("models/2025/YOLOv11n_train_on_2025_all_pose_300ep_best.pt")
    TEST_IMAGES_DIR = Path("/Users/taircarmon/Desktop/prawn-size-project_SAFE/data/images/test_2025_gamma/test_images_2025_gamma/ALL_IMAGES_640x360")
    OUT_DIR = Path("scripts/eval/random_vis_outputs")
    SHOW = False

    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}; please check the path.")
        return
    if not TEST_IMAGES_DIR.exists():
        print(f"Test images directory not found: {TEST_IMAGES_DIR}")
        return

    imgs = sorted(TEST_IMAGES_DIR.glob("*.jpg"))
    if not imgs:
        print(f"No images found in {TEST_IMAGES_DIR}")
        return

    chosen = random.choice(imgs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # copy original image into outputs folder
    try:
        orig_out = OUT_DIR / f"{chosen.stem}_orig{chosen.suffix}"
        shutil.copy2(chosen, orig_out)
        print(f"Saved original image to: {orig_out}")
    except Exception as e:
        print(f"Warning: could not copy original image: {e}")

    # Run both modes: carapace (CL) and body/total (TL)
    modes = {'CL': 'carapace', 'TL': 'body'}

    print(f"Running model {MODEL_PATH} on random image: {chosen.name}")
    for suffix, mode in modes.items():
        out_path = OUT_DIR / f"{chosen.stem}_{suffix}_vis.jpg"
        print(f"  Mode={mode} -> saving to: {out_path}")
        try:
            run_visualization(MODEL_PATH, chosen, mode, out_path=str(out_path), show=SHOW)
        except Exception as e:
            print(f"  Error running mode {mode} on {chosen.name}: {e}")

    print(f"Finished. Visualizations saved to: {OUT_DIR}")


if __name__ == '__main__':
    automatic_run_on_random()
