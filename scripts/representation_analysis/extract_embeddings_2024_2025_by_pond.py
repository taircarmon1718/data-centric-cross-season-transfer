"""
Extract embeddings for 2024/2025 datasets split by pond type.
Saves a CSV with columns: image_path, season, pond, embedding_0, embedding_1, ...

Behavior:
- Deterministic (fixed random seeds, sorted file order)
- Robust to missing or corrupted files (logs and continues)
- Reuses existing model loading / feature-extraction from utils.model_utils

Usage:
    python scripts/representation_analysis/extract_embeddings_2024_2025_by_pond.py --model models/2024/all-ponds/weights/best.pt --out outputs/rep_analysis/by_pond

"""
import os
import sys
import argparse
import random
import math
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

# deterministic seeds
RANDOM_SEED = 12345
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

import torch

def set_torch_seed(seed: int = RANDOM_SEED):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_torch_seed()

# reuse model utility from this package
from utils.model_utils import load_yolo_backbone

# Dataset mapping: path -> (season, pond)
DATASET_MAP = {
    "datasets/train_on_circular_left": ("2024", "circular_left"),
    "datasets/train_on_circular_right": ("2024", "circular_right"),
    "datasets/train_on_square": ("2024", "square"),
    "datasets/train_on_2025_all": ("2025", "mixed"),
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect_files(dataset_map):
    """Recursively collect image file paths and associated metadata.
    Deterministic: returns a sorted list.
    """
    records = []
    for path, (season, pond) in dataset_map.items():
        if not os.path.isdir(path):
            # skip missing dirs but do not fail
            print(f"Warning: dataset directory `{path}` not found; skipping.")
            continue
        for root, _, files in os.walk(path):
            for fname in files:
                if os.path.splitext(fname)[1].lower() in IMAGE_EXTS:
                    full = os.path.join(root, fname)
                    records.append({"image_path": full, "season": season, "pond": pond})
    # sort deterministically by image_path
    records = sorted(records, key=lambda r: r["image_path"])
    return records


def extract_embeddings_2024_2025_by_pond(model_path: str,
                                         out_dir: str = "outputs/rep_analysis/by_pond",
                                         device: str = None,
                                         max_images: int = None):
    """Main pipeline to extract embeddings and save CSV with one column per embedding dimension.

    Parameters:
    - model_path: path to the trained YOLO Pose model (used by utils.model_utils.load_yolo_backbone)
    - out_dir: output directory to save CSV
    - device: "cuda" or "cpu" (auto-detected if None)
    - max_images: optionally limit number of processed images (for quick tests)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(out_dir, exist_ok=True)

    # collect image records
    records = collect_files(DATASET_MAP)
    if len(records) == 0:
        print("No images found in the configured dataset paths. Exiting.")
        return

    if max_images is not None:
        records = records[:max_images]

    print(f"Found {len(records)} images. Loading model from `{model_path}` on device={device} ...")
    backend = load_yolo_backbone(model_path, device=device)
    extract_fn = backend["extract_fn"]
    preprocess = backend["preprocess"]
    hook_handle = backend.get("hook_handle", None)

    embeddings_list = []
    meta_list = []

    # Process images one by one
    for rec in tqdm(records, desc="Extracting embeddings"):
        img_path = rec["image_path"]
        try:
            if not os.path.exists(img_path):
                print(f"Warning: image not found: {img_path} -- skipping")
                continue
            # load and preprocess
            img = Image.open(img_path).convert("RGB")
            tensor = preprocess(img)
            # extract
            emb = extract_fn(tensor)  # returns numpy array shape (1, C) or (B, C)
            if emb is None:
                print(f"Warning: extractor returned None for {img_path}; skipping")
                continue
            emb = np.asarray(emb).reshape(-1)
            embeddings_list.append(emb)
            meta_list.append({"image_path": img_path, "season": rec["season"], "pond": rec["pond"]})
        except Exception as e:
            # log and continue
            print(f"Error processing {img_path}: {e}")
            continue

    # remove hook if present
    if hook_handle is not None:
        try:
            hook_handle.remove()
        except Exception:
            pass

    if len(embeddings_list) == 0:
        print("No embeddings were extracted. Exiting.")
        return

    # Stack embeddings into array
    E = np.stack(embeddings_list, axis=0)  # (N, D)
    N, D = E.shape
    print(f"Extracted embeddings for {N} images with dimension {D}.")

    # Build DataFrame with embedding columns
    df_meta = pd.DataFrame(meta_list)
    # create embedding columns
    emb_cols = [f"embedding_{i}" for i in range(D)]
    df_emb = pd.DataFrame(E, columns=emb_cols)
    df_out = pd.concat([df_meta.reset_index(drop=True), df_emb.reset_index(drop=True)], axis=1)

    # Save CSV
    out_csv = os.path.join(out_dir, "embeddings_by_pond.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"Saved embeddings CSV to: {out_csv}")
    return out_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract embeddings for 2024/2025 by pond type")
    parser.add_argument("--model", type=str, default="models/2024/all-ponds/weights/best.pt", help="Path to YOLO-Pose checkpoint")
    parser.add_argument("--out", type=str, default="outputs/rep_analysis/by_pond", help="Output directory")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--max_images", type=int, default=None, help="Limit number of images processed (for testing)")
    args = parser.parse_args()
    extract_embeddings_2024_2025_by_pond(args.model, args.out, args.device, args.max_images)

