import os
import csv
import warnings
import numpy as np
from typing import List, Dict
from pathlib import Path


def collect_image_records(root_map: Dict[str, Dict[str, str]]):
    """
    root_map example:
    {
      "2024": {"train": "datasets/train_on_all/images", "val": "datasets/train_on_all/val/images"},
      "2025": {"train": "datasets/train_on_2025_all/images", "val": "datasets/train_on_2025_all/val/images"},
    }

    This function is flexible about the provided paths:
    - If `path` exists, it will walk it recursively and collect image files.
    - If `path` does not exist but has common alternates (e.g. `images`, `val/images`), those are tried.
    - If none of those exist, it will search the parent directory recursively for image files (so passing e.g. `datasets/train_on_all` will still find `datasets/train_on_all/images/*`).
    - If no images are found anywhere reasonable, it warns and skips that entry.
    """
    records = []
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    for season, splits in root_map.items():
        for split, path in splits.items():
            path = os.path.expanduser(path)
            search_roots = []

            # If exact path exists, prefer it
            if os.path.isdir(path):
                search_roots.append(path)
            # common alternates
            search_roots.extend([
                os.path.join(path, "images"),
                os.path.join(path, "val", "images"),
                os.path.join(path, "train", "images"),
                os.path.join(path, "images", "train"),
            ])
            # also try the parent directory (useful when user provided a dataset root)
            parent = os.path.dirname(path)
            if parent and os.path.isdir(parent):
                search_roots.append(parent)

            # as a last resort, try the path's parent via pathlib
            p = Path(path)
            if p.parent.exists():
                search_roots.append(str(p.parent))

            # de-duplicate and keep only existing directories
            tried = []
            existing_roots = []
            for r in search_roots:
                if r in tried:
                    continue
                tried.append(r)
                if os.path.isdir(r):
                    existing_roots.append(r)

            found_any_in_entry = False
            if not existing_roots:
                # nothing sensible to search; warn and continue
                warnings.warn(f"Directory `{path}` does not exist and no sensible alternates found — skipping.")
                continue

            # walk through existing roots and collect images
            for root_dir in existing_roots:
                for root, _, files in os.walk(root_dir):
                    for f in files:
                        if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                            records.append({"image_path": os.path.join(root, f), "season": season, "split": split})
                            found_any_in_entry = True
            if not found_any_in_entry:
                warnings.warn(f"No images found under provided roots for season={season}, split={split} (searched: {existing_roots}) — skipping.")

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

    # If both missing in provided out_dir, try to discover embeddings elsewhere in workspace
    if not os.path.exists(csv_path) and not os.path.exists(npy_path):
        # candidate locations to search
        candidates = [
            out_dir,
            os.path.join("outputs", "rep_analysis"),
            "outputs",
            ".",
        ]
        found_dir = None
        for cand in candidates:
            if not cand:
                continue
            c_csv = os.path.join(cand, "embeddings_meta.csv")
            c_npy = os.path.join(cand, "embeddings_vectors.npy")
            if os.path.exists(c_npy) or os.path.exists(c_csv):
                found_dir = cand
                break
        # if not found yet, walk outputs/ then workspace
        if found_dir is None and os.path.isdir("outputs"):
            for root, dirs, files in os.walk("outputs"):
                if "embeddings_vectors.npy" in files or "embeddings_meta.csv" in files:
                    found_dir = root
                    break
        if found_dir is None:
            # last resort: search entire workspace (could be slow)
            for root, dirs, files in os.walk("."):
                if "embeddings_vectors.npy" in files or "embeddings_meta.csv" in files:
                    found_dir = root
                    break
        if found_dir is not None:
            csv_path = os.path.join(found_dir, "embeddings_meta.csv")
            npy_path = os.path.join(found_dir, "embeddings_vectors.npy")
            print(f"Info: load_embeddings discovered embeddings in: {found_dir}")
        else:
            raise FileNotFoundError("Embeddings not found in " + out_dir)

    # If csv missing but npy present, synthesize minimal records
    if not os.path.exists(csv_path) and os.path.exists(npy_path):
        vectors = np.load(npy_path)
        records = []
        for i in range(vectors.shape[0]):
            records.append({"image_path": f"embedding_{i}", "season": "unknown", "split": "unknown"})
        return records, vectors
    # If csv present but npy missing -> error
    if os.path.exists(csv_path) and not os.path.exists(npy_path):
        raise FileNotFoundError("Embeddings vectors .npy not found in " + out_dir)
    # Both exist: normal flow
    records = []
    with open(csv_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            records.append(row)
    vectors = np.load(npy_path)
    return records, vectors
