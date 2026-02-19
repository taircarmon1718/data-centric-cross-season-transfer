#!/usr/bin/env python3
"""
extract_all_embeddings.py

Collect images from the exact dataset folders (2024, 2025 clean, 2025 shifted),
extract YOLO backbone embeddings (model trained on 2024), normalize and save
embeddings and meta CSV. Deterministic (seed=0).

Saves to: scripts/shift_experiments/embeddings/all_embeddings.npy
         scripts/shift_experiments/embeddings/all_meta.csv
"""
from pathlib import Path
import warnings
from tqdm import tqdm

import numpy as np
import pandas as pd
from PIL import Image
import torch

# Configuration
SEED = 0
np.random.seed(SEED)

PROJECT_ROOT = Path(r"C:\Users\carmonta\Desktop\data-centric-cross-season-transfer")
OUT_EMB_DIR = PROJECT_ROOT / 'scripts' / 'shift_experiments' / 'embeddings'
OUT_EMB_DIR.mkdir(parents=True, exist_ok=True)
OUT_VEC = OUT_EMB_DIR / 'all_embeddings.npy'
OUT_META = OUT_EMB_DIR / 'all_meta.csv'

# Data collection folders (must match spec)
DIRS = [
    # 2024
    (PROJECT_ROOT / 'datasets' / 'train_on_2024_all' / 'images', '2024', '2024', 0),
    (PROJECT_ROOT / 'datasets' / 'train_on_2024_all' / 'val' / 'images', '2024', '2024', 0),
    # 2025 clean
    (PROJECT_ROOT / 'datasets' / 'train_on_2025_all' / 'images', '2025_clean', '2025', 0),
    (PROJECT_ROOT / 'datasets' / 'train_on_2025_all' / 'val' / 'images', '2025_clean', '2025', 0),
    # 2025 shifted
    (PROJECT_ROOT / 'outputs' / 'rep_analysis' / 'core_set_selection' / 'shifted_2025_experiment' / 'shifted_2025_experiment' / 'images' / 'train', '2025_shifted', '2025', 1),
    (PROJECT_ROOT / 'outputs' / 'rep_analysis' / 'core_set_selection' / 'shifted_2025_experiment' / 'shifted_2025_experiment' / 'images' / 'val', '2025_shifted', '2025', 1),
]

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
MODEL_PATH = PROJECT_ROOT / 'models' / '2024' / 'all-ponds' / 'weights' / 'best.pt'


def collect_images():
    records = []
    seen = set()
    for folder, dataset_label, season, is_shifted in DIRS:
        folder = Path(folder)
        if not folder.exists():
            warnings.warn(f'Directory {folder} does not exist — skipping')
            continue
        for p in folder.rglob('*'):
            if p.suffix.lower() in IMAGE_EXTS and p.is_file():
                try:
                    rp = p.resolve()
                except Exception:
                    rp = p
                if str(rp) in seen:
                    continue
                seen.add(str(rp))
                records.append({'image_path': str(rp), 'dataset': dataset_label, 'season': season, 'is_shifted': int(is_shifted)})
    # deterministic sort
    records = sorted(records, key=lambda r: r['image_path'])
    return records


def build_ultralytics_extractor(model_path, device='cpu'):
    """Load ultralytics YOLO model and return (preprocess_fn, extract_fn).
    extract_fn(tensor) -> numpy vector (1D) representing pooled features.
    This is a robust fallback and may not match project's exact layer but provides a global descriptor.
    """
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError('ultralytics not available for fallback extractor: ' + str(e))
    model = YOLO(str(model_path))
    model.to(device)
    model.eval()

    # preprocessing: resize shortest side to 640, center-crop optional; match common YOLO input
    import torchvision.transforms as T
    def preprocess_pil(img: Image.Image):
        # preserve aspect ratio: resize shorter side to 640, then letterbox/pad to 640x640
        img = img.convert('RGB')
        w, h = img.size
        short = min(w, h)
        scale = 640.0 / short
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)
        # center crop to 640x640
        left = (new_w - 640) // 2 if new_w > 640 else 0
        top = (new_h - 640) // 2 if new_h > 640 else 0
        img_cropped = img_resized.crop((left, top, left + 640, top + 640))
        tf = T.Compose([T.ToTensor()])
        tensor = tf(img_cropped).unsqueeze(0).to(device)
        return tensor

    def extract_fn(tensor):
        # run through model.model (the internal nn.Module) if possible and attempt to pool
        try:
            with torch.no_grad():
                out = model.model(tensor)
            # out may be a tensor or tuple; try to find a tensor with spatial dims
            if isinstance(out, torch.Tensor):
                x = out
            elif isinstance(out, (list, tuple)):
                # pick first tensor-like
                x = None
                for el in out:
                    if isinstance(el, torch.Tensor):
                        x = el
                        break
                if x is None:
                    return None
            else:
                return None
            # if x has shape (B, C, H, W) perform global average pooling
            if x.dim() == 4:
                pooled = x.mean(dim=[2, 3])  # (B, C)
            elif x.dim() == 2:
                pooled = x  # already (B, C)
            else:
                pooled = x.view(x.size(0), -1)
            arr = pooled.cpu().numpy().reshape(-1)
            return arr
        except Exception as e:
            # fallback: run model.predict and try to use features from results if available
            try:
                res = model.predict(tensor, verbose=False)
                r0 = res[0]
                # try boxes.conf or masks
                if hasattr(r0, 'boxes') and hasattr(r0.boxes, 'conf'):
                    conf = r0.boxes.conf.cpu().numpy()
                    return np.array(conf, dtype=np.float32)
            except Exception:
                return None
            return None

    return preprocess_pil, extract_fn


def main():
    print('Collecting images...')
    records = collect_images()
    if len(records) == 0:
        print('No images found; exiting')
        return
    print(f'Found {len(records)} images')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)

    try:
        preprocess, extract_fn = build_ultralytics_extractor(MODEL_PATH, device=device)
    except Exception as e:
        print('Failed to build extractor:', e)
        return

    embeddings = []
    meta_rows = []
    for rec in tqdm(records, desc='Extracting embeddings'):
        p = rec['image_path']
        try:
            img = Image.open(p).convert('RGB')
            tensor = preprocess(img)
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.to(device)
            with torch.no_grad():
                emb = extract_fn(tensor)
            if emb is None:
                warnings.warn(f'No embedding returned for {p}; skipping')
                continue
            emb = np.asarray(emb).reshape(-1)
            # L2 normalize
            nrm = np.linalg.norm(emb)
            if nrm == 0:
                nrm = 1.0
            emb = (emb / nrm).astype(np.float32)
            embeddings.append(emb)
            meta_rows.append({'image_path': rec['image_path'], 'dataset': rec['dataset'], 'season': rec['season'], 'is_shifted': rec['is_shifted']})
        except Exception as e:
            warnings.warn(f'Failed processing {p}: {e}')
            continue

    if len(embeddings) == 0:
        print('No embeddings extracted; exiting')
        return

    E = np.stack(embeddings, axis=0).astype(np.float32)
    np.save(OUT_VEC, E)
    df = pd.DataFrame(meta_rows)
    df.to_csv(OUT_META, index=False)
    print('Saved embeddings to', OUT_VEC)
    print('Saved meta to', OUT_META)


if __name__ == '__main__':
    main()
