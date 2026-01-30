import os
import numpy as np
from tqdm import tqdm
import argparse
from PIL import Image
import torch

from utils.io import collect_image_records, save_embeddings
from utils.model_utils import load_yolo_backbone
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

def main(model_path: str, out_dir: str = "outputs/rep_analysis", device: str = "cuda"):
    # map seasons -> splits -> paths
    root_map = {
        "2024": {
            "all": "datasets/train_on_all",
        },
        "2025": {
            "all": "datasets/train_on_2025_all",
        },
    }

    records = collect_image_records(root_map)
    if len(records) == 0:
        print("No images found; exiting.")
        return
    print(f"Found {len(records)} images. Loading model from `{model_path}` ...")
    backend = load_yolo_backbone(model_path, device=device)
    extract_fn = backend["extract_fn"]
    preprocess = backend["preprocess"]
    handle = backend.get("hook_handle", None)

    embeddings = []
    processed_records = []
    for rec in tqdm(records, desc="Embedding images"):
        try:
            img = Image.open(rec["image_path"]).convert("RGB")
            tensor = preprocess(img)
            emb = extract_fn(tensor)  # returns numpy array BxC
            if emb is None:
                continue
            emb = np.asarray(emb).reshape(-1)  # single vector
            embeddings.append(emb)
            processed_records.append(rec)
        except Exception as e:
            print(f"Warning: failed for `{rec['image_path']}`: {e}")
            continue
    if handle is not None:
        try:
            handle.remove()
        except Exception:
            pass
    if len(embeddings) == 0:
        print("No embeddings extracted.")
        return
    embeddings = np.stack(embeddings, axis=0)
    save_embeddings(out_dir, processed_records, embeddings)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/2024/all-ponds/weights/best.pt")
    parser.add_argument("--out", type=str, default="outputs/rep_analysis")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    main(args.model, args.out, args.device)

