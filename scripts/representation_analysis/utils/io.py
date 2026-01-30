import os
import csv
import warnings
import numpy as np
from typing import List, Dict


def collect_image_records(root_map: Dict[str, Dict[str, str]]):
    """
    root_map example:
    {
      "2024": {"train": "datasets/train_on_all/images", "val": "datasets/train_on_all/val/images"},
      "2025": {"train": "datasets/train_on_2025_all/images", "val": "datasets/train_on_2025_all/val/images"},
    }
    """
    records = []
    for season, splits in root_map.items():
        for split, path in splits.items():
            if not os.path.isdir(path):
                warnings.warn(f"Directory `{path}` does not exist — skipping.")
                continue
            for root, _, files in os.walk(path):
                for f in files:
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                        records.append({"image_path": os.path.join(root, f), "season": season, "split": split})
    return records


def save_embeddings(out_dir: str, records: List[Dict], embeddings: np.ndarray):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "embeddings_meta.csv")
    npy_path = os.path.join(out_dir, "embeddings_vectors.npy")
    # Save metadata
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_path", "season", "split"])
        for r in records:
            writer.writerow([r["image_path"], r["season"], r["split"]])
    # Save vectors
    np.save(npy_path, embeddings)
    print(f"Saved metadata to `{csv_path}` and vectors to `{npy_path}`.")


def load_embeddings(out_dir: str):
    import numpy as np
    csv_path = os.path.join(out_dir, "embeddings_meta.csv")
    npy_path = os.path.join(out_dir, "embeddings_vectors.npy")
    if not os.path.exists(csv_path) or not os.path.exists(npy_path):
        raise FileNotFoundError("Embeddings not found in " + out_dir)
    records = []
    with open(csv_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            records.append(row)
    vectors = np.load(npy_path)
    return records, vectors

