#!/usr/bin/env python3
from pathlib import Path
import cv2
import numpy as np
import sys

# ===============================
# EXACT FILES
# ===============================

IMAGE_PATH = Path("/Users/taircarmon/Desktop/data-centric-cross-season-transfer/datasets/2024_smallDS_circulars/images/train/GX010065_8_236-jpg_gamma_jpg.rf.957e18c616758f829dbf1e77d7bfd1e0.jpg")
LABEL_PATH = Path("/Users/taircarmon/Desktop/data-centric-cross-season-transfer/datasets/2024_smallDS_circulars/labels/train/GX010065_8_236-jpg_gamma_jpg.rf.957e18c616758f829dbf1e77d7bfd1e0.txt")
OUT_PATH = Path("/Users/taircarmon/Desktop/data-centric-cross-season-transfer/outputs/kp_debug_vis.png")

EXPECTED_KPTS = 4


# ===============================
# Correct YOLO pose parsing
# ===============================

def parse_label(label_path: Path, expected_kpts=4):
    with open(label_path, "r") as f:
        line = f.readline().strip()

    parts = [float(x) for x in line.split()]

    # Skip class + bbox
    data = parts[5:]

    if len(data) != expected_kpts * 3:
        raise ValueError("Label format does not match expected keypoint count.")

    kpts = []
    for i in range(expected_kpts):
        x = data[i * 3 + 0]
        y = data[i * 3 + 1]
        v = data[i * 3 + 2]
        kpts.append((x, y, v))

    return np.array(kpts)


# ===============================
# Main
# ===============================

def main():
    if not IMAGE_PATH.exists():
        print("Image not found", file=sys.stderr)
        sys.exit(1)

    if not LABEL_PATH.exists():
        print("Label not found", file=sys.stderr)
        sys.exit(1)

    img = cv2.imread(str(IMAGE_PATH))
    h, w = img.shape[:2]

    print(f"Image size: {w} x {h}")

    kpts = parse_label(LABEL_PATH, EXPECTED_KPTS)
    print("Raw keypoints:")
    print(kpts)

    # Determine if normalized
    max_val = np.max(kpts[:, :2])
    is_normalized = max_val <= 1.0

    if is_normalized:
        print("Detected NORMALIZED keypoints.")
        kpts[:, 0] *= w
        kpts[:, 1] *= h
    else:
        print("Detected ABSOLUTE keypoints.")

    # Drawing
    for i, (x, y, v) in enumerate(kpts):

        if v == 0:
            print(f"kp {i}: not labeled")
            continue

        x_px = int(round(x))
        y_px = int(round(y))

        print(f"kp {i}: x={x_px}, y={y_px}, visibility={v}")

        cv2.circle(img, (x_px, y_px), 20, (0, 255, 0), -1)
        cv2.putText(img, str(i),
                    (x_px + 15, y_px + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_PATH), img)

    print(f"Saved visualization to: {OUT_PATH}")


if __name__ == "__main__":
    main()
