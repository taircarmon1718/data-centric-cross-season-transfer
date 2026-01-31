"""
Extract embeddings for 2024/2025 datasets split by pond type.

Outputs:
1) Efficient representation for reuse:
   - embeddings_vectors.npy
   - embeddings_meta.csv
2) Analysis-friendly CSV:
   - embeddings_2024_2025_by_pond.csv
"""

# --------------------------------------------------
# CRITICAL: must be set BEFORE importing torch
# --------------------------------------------------
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch

from utils.io import save_embeddings
from utils.model_utils import load_yolo_backbone


# --------------------------------------------------
# Deterministic setup
# --------------------------------------------------
RANDOM_SEED = 12345
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# --------------------------------------------------
# Project root
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)


# --------------------------------------------------
# Dataset mapping
# --------------------------------------------------
DATASET_MAP = {
    "datasets/train_on_circular_left":  ("2024", "circular_left"),
    "datasets/train_on_circular_right": ("2024", "circular_right"),
    "datasets/train_on_square":         ("2024", "square"),
    "datasets/train_on_2025_all":       ("2025", "mixed"),
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect_files(dataset_map):
    records = []
    for path, (season, pond) in dataset_map.items():
        if not os.path.isdir(path):
            print(f"Warning: dataset directory `{path}` not found; skipping.")
            continue

        for root, _, files in os.walk(path):
            for fname in files:
                if os.path.splitext(fname)[1].lower() in IMAGE_EXTS:
                    records.append({
                        "image_path": os.path.join(root, fname),
                        "season": season,
                        "pond": pond,
                    })

    return sorted(records, key=lambda r: r["image_path"])


def extract_embeddings_2024_2025_by_pond(
    model_path: str,
    out_dir: str,
    device: str = None,
    max_images: int = None,
):
    # --------------------------------------------------
    # Device handling (NO silent fallback)
    # --------------------------------------------------
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Using device:", device)

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available – environment issue")

    os.makedirs(out_dir, exist_ok=True)

    # --------------------------------------------------
    # Collect data
    # --------------------------------------------------
    records = collect_files(DATASET_MAP)
    if len(records) == 0:
        print("No images found. Exiting.")
        return

    if max_images is not None:
        records = records[:max_images]

    print(f"Found {len(records)} images. Loading model from `{model_path}`")

    backend = load_yolo_backbone(model_path, device=device)
    extract_fn = backend["extract_fn"]
    preprocess = backend["preprocess"]
    hook_handle = backend.get("hook_handle", None)

    embeddings = []
    meta = []

    # --------------------------------------------------
    # Embedding extraction
    # --------------------------------------------------
    for rec in tqdm(records, desc="Extracting embeddings"):
        img_path = rec["image_path"]
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = preprocess(img).to(device)

            with torch.no_grad():
                emb = extract_fn(tensor)

            if emb is None:
                continue

            emb = np.asarray(emb).reshape(-1)
            embeddings.append(emb)

            meta.append({
                "image_path": img_path,
                "season": rec["season"],
                "split": "all",      # required by save_embeddings
                "pond": rec["pond"],
            })

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    if hook_handle is not None:
        try:
            hook_handle.remove()
        except Exception:
            pass

    if len(embeddings) == 0:
        print("No embeddings extracted. Exiting.")
        return

    E = np.stack(embeddings, axis=0)
    N, D = E.shape
    print(f"Extracted embeddings: N={N}, D={D}")

    # --------------------------------------------------
    # Save efficient format (fast reload)
    # --------------------------------------------------
    save_embeddings(out_dir, meta, E)

    # --------------------------------------------------
    # Save CSV (analysis-friendly, unique name)
    # --------------------------------------------------
    df_meta = pd.DataFrame(meta)
    df_emb = pd.DataFrame(E, columns=[f"embedding_{i}" for i in range(D)])
    df_out = pd.concat([df_meta.reset_index(drop=True),
                        df_emb.reset_index(drop=True)], axis=1)

    out_csv = os.path.join(out_dir, "embeddings_2024_2025_by_pond.csv")
    df_out.to_csv(out_csv, index=False)

    print(f"Saved CSV to: {out_csv}")
    return out_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="models/2024/all-ponds/weights/best.pt",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/rep_analysis/by_pond",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    extract_embeddings_2024_2025_by_pond(
        args.model,
        args.out,
        args.device,
        args.max_images,
    )
