#!/usr/bin/env python3
"""
append_test_only_to_existing_dino_embeddings.py

Add ONLY missing test images (2024 + 2025) to existing DINO embedding files.

Does NOT recompute train.
Does NOT overwrite unless explicitly allowed.
Deterministic ordering.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path
import csv
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

# -----------------------------
# PATHS
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

EMB_DIR = PROJECT_ROOT / "scripts/representation_analysis/outputs_dino_full"
META_PATH = EMB_DIR / "dino_embeddings_full_meta.csv"
VECT_PATH = EMB_DIR / "dino_embeddings_full.npy"

# TEST PATHS (relative to project root)
TEST_2024_BODY = PROJECT_ROOT / "data/images/test_2024_originalSize/test_images_2024_orginalSize/body"
TEST_2024_CAR  = PROJECT_ROOT / "data/images/test_2024_originalSize/test_images_2024_orginalSize/carapace"
TEST_2025_ALL  = PROJECT_ROOT / "data/images/test_2025_gamma/test_images_2025_gamma/ALL_IMAGES_640x360"

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

# -----------------------------
# DINO MODEL
# -----------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
model.eval().to(device)

preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]),
])

# -----------------------------
# LOAD EXISTING
# -----------------------------

if not META_PATH.exists() or not VECT_PATH.exists():
    raise FileNotFoundError("Existing DINO embedding files not found.")

print("Loading existing embeddings...")
meta = []
with open(META_PATH, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        meta.append(row)

existing_paths = set([m["image_path"] for m in meta])

vectors = np.load(VECT_PATH)
print(f"Existing embeddings: {len(meta)} rows")

# -----------------------------
# COLLECT TEST IMAGES
# -----------------------------

def collect_test(folder, season, subtype):
    records = []
    if not folder.exists():
        print(f"Warning: {folder} not found.")
        return records
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() in IMAGE_EXTS:
            rel = p.relative_to(PROJECT_ROOT)
            records.append({
                "image_path": str(rel),
                "season": season,
                "dataset_type": "test",
                "subtype": subtype,
            })
    return records

test_records = []
test_records += collect_test(TEST_2024_BODY, "2024", "body")
test_records += collect_test(TEST_2024_CAR,  "2024", "carapace")
test_records += collect_test(TEST_2025_ALL,  "2025", "all")

print(f"Total test images found: {len(test_records)}")

# Filter only new ones
new_records = [r for r in test_records if r["image_path"] not in existing_paths]

print(f"New test images to embed: {len(new_records)}")

if len(new_records) == 0:
    print("Nothing to add. Exiting.")
    exit()

# -----------------------------
# EXTRACT NEW EMBEDDINGS
# -----------------------------

new_embeddings = []

for rec in tqdm(new_records, desc="Embedding new test images"):
    img_path = PROJECT_ROOT / rec["image_path"]
    try:
        img = Image.open(img_path).convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
        emb = out.squeeze(0).cpu().numpy()
        new_embeddings.append(emb)
    except Exception as e:
        print(f"Failed: {img_path} -> {e}")

new_embeddings = np.stack(new_embeddings, axis=0)

# -----------------------------
# APPEND
# -----------------------------

vectors_updated = np.vstack([vectors, new_embeddings])
meta_updated = meta + new_records

print(f"New total embeddings: {vectors_updated.shape[0]}")

# -----------------------------
# SAVE (overwrite safely)
# -----------------------------

np.save(VECT_PATH, vectors_updated)

with open(META_PATH, "w", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["image_path","season","dataset_type","subtype"])
    writer.writeheader()
    for row in meta_updated:
        writer.writerow(row)

print("Successfully appended test embeddings.")
