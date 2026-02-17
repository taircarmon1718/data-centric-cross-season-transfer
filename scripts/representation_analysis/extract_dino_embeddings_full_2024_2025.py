#!/usr/bin/env python3
"""
extract_dino_embeddings_full_2024_2025.py

Extract DINOv2 (ViT-S/14) CLS embeddings for 2024 and 2025 (train + test)
in the same style and ordering as existing YOLO embedding extractors.

Outputs saved to: scripts/representation_analysis/outputs_dino_full/
Files:
 - dino_embeddings_full_meta.csv (image_path, season, dataset_type, subtype)
 - dino_embeddings_full.npy

Behavior:
 - deterministic ordering
 - tqdm progress
 - skip broken images
 - use GPU if available
 - do not overwrite existing outputs (auto-increment filenames)
"""

from pathlib import Path
import os
import csv
import time
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T

# Deterministic seed
RANDOM_SEED = 12345
import random
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

# Model name/path
DINO_HUB = 'facebookresearch/dinov2'
DINO_MODEL = 'dinov2_vits14'  # ViT-S/14

# Input folders (as requested)
TRAIN_2024 = PROJECT_ROOT / 'datasets' / 'train_on_all'
TRAIN_2025 = PROJECT_ROOT / 'datasets' / 'train_on_2025_all'
TEST_2024_BODY = PROJECT_ROOT / 'data' / 'images' / 'test_2024_originalSize' / 'test_images_2024_orginalSize' / 'body'
TEST_2024_CAR = PROJECT_ROOT / 'data' / 'images' / 'test_2024_originalSize' / 'test_images_2024_orginalSize' / 'carapace'
TEST_2025_ALL = PROJECT_ROOT / 'data' / 'images' / 'test_2025_gamma' / 'test_images_2025_gamma' / 'ALL_IMAGES_640x360'

OUT_ROOT = PROJECT_ROOT / 'scripts' / 'representation_analysis' / 'outputs_dino_full'
OUT_ROOT.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# Preprocessing: shortest side -> 256, center crop 224, ImageNet norm
preprocess = T.Compose([
    T.Resize(256),           # resize shorter side to 256
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def collect_image_records():
    """Collect images from requested folders and return list of records.
    Each record: dict with image_path (relative), season, dataset_type, subtype
    Deterministic sorted order across (season, dataset_type, subtype, path).
    """
    records = []

    # helper to walk and collect
    def walk_folder(base: Path, season: str, dataset_type: str, subtype: str):
        if not base.exists():
            print(f"Warning: folder {base} not found; skipping {season}/{subtype}")
            return
        for p in sorted(base.rglob('*')):
            if p.suffix.lower() in IMAGE_EXTS and p.is_file():
                rel = p.relative_to(PROJECT_ROOT)
                records.append({
                    'image_path': str(rel),
                    'season': season,
                    'dataset_type': dataset_type,
                    'subtype': subtype,
                })

    # Training folders: for training sets, previous scripts scanned dataset roots (they included images/ and val/images) - we keep the same behavior: look for images under the dataset folder recursively
    def collect_train_dataset(root: Path, season: str):
        if not root.exists():
            print(f"Warning: train dataset folder {root} not found; skipping")
            return
        # collect recursively
        for p in sorted(root.rglob('*')):
            if p.suffix.lower() in IMAGE_EXTS and p.is_file():
                rel = p.relative_to(PROJECT_ROOT)
                records.append({
                    'image_path': str(rel),
                    'season': season,
                    'dataset_type': 'train',
                    'subtype': 'all',
                })

    # collect train
    collect_train_dataset(TRAIN_2024, '2024')
    collect_train_dataset(TRAIN_2025, '2025')

    # collect tests
    walk_folder(TEST_2024_BODY, '2024', 'test', 'body')
    walk_folder(TEST_2024_CAR, '2024', 'test', 'carapace')
    walk_folder(TEST_2025_ALL, '2025', 'test', 'all')

    # final deterministic sort
    records = sorted(records, key=lambda r: (r['season'], r['dataset_type'], r['subtype'], r['image_path']))
    return records


def unique_output_paths(base_name_meta: str, base_name_vec: str):
    """Return non-colliding meta csv path and npy path in OUT_ROOT by appending suffix if needed."""
    meta_path = OUT_ROOT / base_name_meta
    vec_path = OUT_ROOT / base_name_vec
    if not meta_path.exists() and not vec_path.exists():
        return meta_path, vec_path
    # append numeric suffix
    i = 1
    while True:
        m = OUT_ROOT / f"{meta_path.stem}_v{i}{meta_path.suffix}"
        v = OUT_ROOT / f"{vec_path.stem}_v{i}{vec_path.suffix}"
        if not m.exists() and not v.exists():
            return m, v
        i += 1


def extract_embeddings():
    records = collect_image_records()
    total_images = len(records)
    print(f'Total images found: {total_images}')
    if total_images == 0:
        print('No images to process; exiting.')
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Loading DINOv2 model via torch.hub ...')
    # load model
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.eval()
    model.to(device)

    embeddings = []
    processed_meta = []

    # extraction loop
    for rec in tqdm(records, desc='Extracting DINO embeddings'):
        p = PROJECT_ROOT / rec['image_path']
        try:
            with Image.open(p) as img:
                img = img.convert('RGB')
            tensor = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(tensor)
            # robustly handle outputs
            emb = None
            # if tensor returned
            if isinstance(out, torch.Tensor):
                if out.dim() == 2:
                    emb = out.squeeze(0).cpu().numpy()
                elif out.dim() == 3:
                    # assume shape [B, N, C] -> take CLS token at index 0
                    emb = out[:, 0, :].squeeze(0).cpu().numpy()
            elif isinstance(out, dict):
                # try common keys
                for key in ('cls', 'x', 'last_hidden_state', 'features'):
                    if key in out:
                        val = out[key]
                        if isinstance(val, torch.Tensor):
                            if val.dim() == 2:
                                emb = val.squeeze(0).cpu().numpy()
                            elif val.dim() == 3:
                                emb = val[:, 0, :].squeeze(0).cpu().numpy()
                        break
            elif isinstance(out, (list, tuple)):
                # pick first tensor-like entry
                for el in out:
                    if isinstance(el, torch.Tensor):
                        if el.dim() == 2:
                            emb = el.squeeze(0).cpu().numpy()
                            break
                        elif el.dim() == 3:
                            emb = el[:, 0, :].squeeze(0).cpu().numpy()
                            break

            if emb is None:
                # as a fallback, try calling model.forward_features if available
                if hasattr(model, 'forward_features'):
                    try:
                        feats = model.forward_features(tensor)
                        if isinstance(feats, torch.Tensor):
                            if feats.dim() == 3:
                                emb = feats[:, 0, :].squeeze(0).cpu().numpy()
                            elif feats.dim() == 2:
                                emb = feats.squeeze(0).cpu().numpy()
                    except Exception:
                        pass

            if emb is None:
                print(f'Warning: could not extract embedding for {p}; skipping')
                continue

            emb = np.asarray(emb).reshape(-1)
            embeddings.append(emb)
            processed_meta.append(rec)
        except Exception as e:
            print(f'Warning: failed to process {p}: {e}')
            continue

    # remove hooks if any (not used here but defensive)
    try:
        if hasattr(model, 'remove_hooks'):
            try:
                model.remove_hooks()
            except Exception:
                pass
    except Exception:
        pass

    if len(embeddings) == 0:
        print('No embeddings extracted.')
        return

    E = np.stack(embeddings, axis=0)
    N, D = E.shape
    print(f'Total embeddings extracted: {N}, dimension: {D}')

    # determine unique output paths
    meta_name = 'dino_embeddings_full_meta.csv'
    vec_name = 'dino_embeddings_full.npy'
    meta_path, vec_path = unique_output_paths(meta_name, vec_name)

    # save vectors
    np.save(vec_path, E)

    # save meta CSV with required columns
    with open(meta_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'season', 'dataset_type', 'subtype'])
        for m in processed_meta:
            writer.writerow([m['image_path'], m['season'], m['dataset_type'], m['subtype']])

    print('Saved embeddings and metadata:')
    print(' - meta:', meta_path)
    print(' - vectors:', vec_path)


if __name__ == '__main__':
    extract_embeddings()

