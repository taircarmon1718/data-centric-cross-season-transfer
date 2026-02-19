#!/usr/bin/env python3
"""
build_shifted_2025_experiment.py

Creates a shifted dataset of 300 randomly sampled images from
`datasets/train_on_2025_all` and generates random core-set CSVs for K_LIST.

Requirements: pathlib, numpy, cv2, pandas, tqdm

This script is deterministic (SEED = 0).
"""
from pathlib import Path
import argparse
import random
import shutil
import math
import csv
import sys
import json

import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm

# -------------------- Configuration --------------------
SEED = 0
K_LIST = [1, 2, 5, 10, 20, 50]
SAMPLE_N = 300

PROJECT_ROOT = Path(r"C:\Users\carmonta\Desktop\data-centric-cross-season-transfer")
SRC_ROOT = PROJECT_ROOT / "datasets" / "train_on_2025_all"
SRC_IMG_TRAIN = SRC_ROOT / "images"
SRC_IMG_VAL = SRC_ROOT / "val" / "images"
SRC_LABEL_TRAIN = SRC_ROOT / "labels"
SRC_LABEL_VAL = SRC_ROOT / "val" / "labels"

OUT_BASE = PROJECT_ROOT / "outputs" / "rep_analysis" / "core_set_selection"
SHIFTED_ROOT = OUT_BASE / "shifted_2025_experiment" / "shifted_2025_experiment"
IMAGES_TRAIN_OUT = SHIFTED_ROOT / "images" / "train"
IMAGES_VAL_OUT = SHIFTED_ROOT / "images" / "val"
LABELS_TRAIN_OUT = SHIFTED_ROOT / "labels" / "train"
LABELS_VAL_OUT = SHIFTED_ROOT / "labels" / "val"

CORESETS_BASE = OUT_BASE

IMAGE_EXT = ".jpg"

# -------------------- Utilities --------------------

def collect_jpg_images():
    """Collect .jpg images from train + val (deterministic sorted order)."""
    imgs = []
    for root, split in ((SRC_IMG_TRAIN, "train"), (SRC_IMG_VAL, "val")):
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.jpg")):
            if p.is_file():
                imgs.append({"path": p, "split": split})
    return imgs


def ensure_dirs():
    for d in (IMAGES_TRAIN_OUT, IMAGES_VAL_OUT, LABELS_TRAIN_OUT, LABELS_VAL_OUT):
        d.mkdir(parents=True, exist_ok=True)


def turbidity(img):
    # blend with white and add slight Gaussian blur
    h, w = img.shape[:2]
    alpha = np.random.uniform(0.15, 0.35)  # moderate haze
    white = np.full_like(img, 255)
    blended = cv2.addWeighted(img, 1 - alpha, white, alpha, 0)
    k = int(max(3, (min(h, w) // 200) * 3))
    if k % 2 == 0:
        k += 1
    blurred = cv2.GaussianBlur(blended, (k, k), 0)
    return blurred


def brightness_shift(img):
    # convert to HSV and scale V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    factor = np.random.uniform(0.7, 1.3)  # moderate
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def contrast_reduction(img):
    # reduce contrast by blending with mean color
    alpha = np.random.uniform(0.5, 0.85)  # less than 1 reduces contrast
    mean = np.full_like(img, int(np.mean(img)))
    out = cv2.addWeighted(img, alpha, mean, 1 - alpha, 0)
    return out


def green_color_cast(img):
    # increase green channel moderately
    out = img.copy().astype(np.int16)
    gain = np.random.randint(30, 80)
    out[:, :, 1] = np.clip(out[:, :, 1] + gain, 0, 255)
    return out.astype(np.uint8)


def motion_blur(img):
    # directional motion blur using kernel
    size = np.random.randint(7, 17)  # kernel size
    # random angle
    angle = np.random.uniform(0, 360)
    # create kernel: line at center
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0
    # rotate kernel
    M = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (size, size))
    kernel = kernel / np.sum(kernel)
    out = cv2.filter2D(img, -1, kernel)
    return out


def occlusion(img):
    h, w = img.shape[:2]
    area_frac = np.random.uniform(0.08, 0.25)  # cover 8-25% of area
    occl_w = int(w * np.sqrt(area_frac))
    occl_h = int(h * np.sqrt(area_frac))
    x = np.random.randint(0, max(1, w - occl_w))
    y = np.random.randint(0, max(1, h - occl_h))
    overlay = img.copy()
    alpha = np.random.uniform(0.35, 0.6)
    # semi-transparent rectangle (gray)
    cv2.rectangle(overlay, (x, y), (x + occl_w, y + occl_h), (128, 128, 128), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img


def apply_random_shift(img, rng):
    # choose one of six shifts
    choices = [turbidity, brightness_shift, contrast_reduction, green_color_cast, motion_blur, occlusion]
    idx = rng.randint(0, len(choices))
    func = choices[idx]
    return func(img), idx


def find_label_file(img_path):
    # try structured mapping
    p = Path(img_path)
    stem = p.stem
    # check corresponding train label
    candidate = SRC_LABEL_TRAIN / (p.name.replace('.jpg', '.txt'))
    if candidate.exists():
        return candidate
    candidate = SRC_LABEL_VAL / (p.name.replace('.jpg', '.txt'))
    if candidate.exists():
        return candidate
    # fallback: search by stem recursively deterministically
    for root in (SRC_LABEL_TRAIN, SRC_LABEL_VAL):
        if not root.exists():
            continue
        for f in sorted(root.rglob('*.txt')):
            if f.stem == stem:
                return f
    return None


def save_image_and_label(src_img_path, img_out_path, label_out_dir):
    # save image (BGR) using cv2.imwrite
    cv2.imwrite(str(img_out_path), img)

# -------------------- Main Execution --------------------

def run():
    rng = np.random.RandomState(SEED)

    # collect jpg images
    all_images = []
    for root, split in ((SRC_IMG_TRAIN, 'train'), (SRC_IMG_VAL, 'val')):
        if not root.exists():
            continue
        for p in sorted(root.rglob('*.jpg')):
            if p.is_file():
                all_images.append({'path': p, 'split': split})

    if len(all_images) < SAMPLE_N:
        print(f'ERROR: found only {len(all_images)} images (need {SAMPLE_N}). Aborting.')
        sys.exit(1)

    # deterministic sample of 300
    indices = rng.choice(len(all_images), size=SAMPLE_N, replace=False)
    sampled = [all_images[i] for i in indices]

    # ensure output dirs
    for d in (IMAGES_TRAIN_OUT, IMAGES_VAL_OUT, LABELS_TRAIN_OUT, LABELS_VAL_OUT):
        d.mkdir(parents=True, exist_ok=True)

    # Apply shifts and copy labels
    shift_counts = {i: 0 for i in range(6)}
    shifted_records = []  # list of dicts with image_path relative and split
    for item in tqdm(sampled, desc='Processing sampled images'):
        src_path = item['path']
        split = item['split']
        # read image via cv2
        img = cv2.imread(str(src_path))
        if img is None:
            print(f'Warning: failed to read {src_path}; skipping')
            continue
        shifted_img, shift_idx = apply_random_shift(img.copy(), rng)
        shift_counts[shift_idx] += 1
        # output paths
        if split == 'train':
            out_img_path = IMAGES_TRAIN_OUT / src_path.name
            out_lbl_dir = LABELS_TRAIN_OUT
        else:
            out_img_path = IMAGES_VAL_OUT / src_path.name
            out_lbl_dir = LABELS_VAL_OUT
        # save shifted image
        cv2.imwrite(str(out_img_path), shifted_img)
        # copy label
        label_src = find_label_file(src_path)
        if label_src is not None:
            shutil.copy2(label_src, out_lbl_dir / label_src.name)
        else:
            print(f'Warning: label not found for {src_path} (expected in labels); continuing without label')
        # record
        shifted_records.append({'image_path': str(out_img_path.relative_to(PROJECT_ROOT)), 'split': split})

    # Step 3: generate random k core sets from shifted images
    # Build list of relative paths
    shifted_rel_paths = [rec['image_path'] for rec in shifted_records]
    shifted_splits = [rec['split'] for rec in shifted_records]
    shifted_df = pd.DataFrame({'image_path': shifted_rel_paths, 'split': shifted_splits})

    k_counts = {}
    for k in K_LIST:
        n = int(math.floor(k / 100.0 * SAMPLE_N))
        n = max(1, n) if n > 0 else 0
        rng2 = np.random.RandomState(SEED + k)  # deterministic per k
        if n == 0:
            selected_idx = []
        else:
            selected_idx = rng2.choice(len(shifted_df), size=n, replace=False)
        selected = shifted_df.iloc[selected_idx]
        out_dir = CORESETS_BASE / f'shifted_2025_k{int(k):02d}'
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / 'core_set.csv'
        selected.to_csv(csv_path, index=False)
        k_counts[k] = len(selected)

    # Print summary
    print('\n=== Experiment Summary ===')
    print(f'Total sampled images: {len(shifted_records)} (expected {SAMPLE_N})')
    shift_names = ['turbidity', 'brightness', 'contrast', 'green_cast', 'motion_blur', 'occlusion']
    for i, name in enumerate(shift_names):
        print(f'  {name}: {shift_counts.get(i,0)}')
    for k in K_LIST:
        print(f'  k={k}% -> {k_counts.get(k,0)} images')
    print('Shifted dataset path:', SHIFTED_ROOT)


if __name__ == '__main__':
    run()

