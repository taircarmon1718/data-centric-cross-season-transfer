"""
Build Season-2025 core-sets using Fused embeddings (YOLO + geometric) and
Farthest Point Sampling (FPS).

Inputs (project-root relative):
  outputs/rep_analysis/geom_fusion_2025/fused_embeddings_2025.npy
  outputs/rep_analysis/geom_fusion_2025/aligned_records_2025.json

For each k in K_LIST the script will:
 - deterministically run FPS over the fused embeddings
 - select n_select = ceil(k% * N) samples
 - save selection logs and elbow plot under:
     outputs/rep_analysis/core_set_selection/fused_fps/kXX/
 - build a YOLO-style dataset under:
     datasets/train_on_2025_core_set_fused_fps_kXX/
   copying images and labels from datasets/train_on_2025_all/

Requirements and behavior:
 - Deterministic: numpy seed = 0 and deterministic tie-breaks
 - Preprocess embeddings with L2 row-normalization (float64)
 - FPS uses Euclidean distance and greedy farthest-point sampling
 - Uses pathlib, tqdm, matplotlib, csv
 - Does not modify original embeddings

Script location:
  scripts/representation_analysis/core_set_selection/build_2025_core_set_fused_fps.py
"""
from pathlib import Path
import numpy as np
import json
import csv
import math
import shutil
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configuration
K_LIST = [1, 2, 5, 10, 20, 50]
SEED = 0
PROJECT_ROOT = Path(__file__).resolve().parents[2]

FUSED_DIR = PROJECT_ROOT / 'outputs' / 'rep_analysis' / 'geom_fusion_2025'
FUSED_NPY = FUSED_DIR / 'fused_embeddings_2025.npy'
RECORDS_JSON = FUSED_DIR / 'aligned_records_2025.json'

SRC_DATA = PROJECT_ROOT / 'datasets' / 'train_on_2025_all'
SRC_LABELS_TRAIN = SRC_DATA / 'labels' / 'train'
SRC_LABELS_VAL = SRC_DATA / 'labels' / 'val'

OUT_LOG_BASE = PROJECT_ROOT / 'outputs' / 'rep_analysis' / 'core_set_selection' / 'fused_fps'
OUT_DS_PARENT = PROJECT_ROOT / 'datasets'

IMG_EXTS = ('.jpg', '.jpeg', '.png')

RNG = np.random.RandomState(SEED)


# ---------------------- IO ----------------------

def load_fused_embeddings(npy_path: Path, records_path: Path):
    if not npy_path.exists():
        raise FileNotFoundError(f"Fused embeddings not found: {npy_path}")
    if not records_path.exists():
        raise FileNotFoundError(f"Aligned records not found: {records_path}")
    embeddings = np.load(str(npy_path))
    with records_path.open('r') as fh:
        records = json.load(fh)
    if len(records) != embeddings.shape[0]:
        raise ValueError(f"Records length {len(records)} != embeddings rows {embeddings.shape[0]}")
    return records, embeddings


# ---------------------- Preprocessing ----------------------

def l2_normalize(vectors: np.ndarray):
    vectors = vectors.astype(np.float64, copy=False)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


# ---------------------- FPS ----------------------

def run_fps(vectors: np.ndarray, n_select: int):
    """Run deterministic Farthest Point Sampling on row-normalized vectors.

    Returns:
      selected_indices: list of ints (indices into vectors)
      selection_log: list of dicts with keys (iteration, selected_index, min_distance, max_distance, mean_distance)
    """
    M = vectors.shape[0]
    if n_select <= 0:
        return [], []
    if n_select >= M:
        # select all in original order
        selection_log = []
        for i in range(M):
            selection_log.append({
                'iteration': i + 1,
                'selected_index': int(i),
                'min_distance': 0.0,
                'max_distance': 0.0,
                'mean_distance': 0.0,
            })
        return list(range(M)), selection_log

    # deterministic initial seed
    first = int(RNG.randint(0, M))
    selected = [first]

    # compute initial min distances to the selected set (distance to first)
    diff = vectors - vectors[first]
    min_dists = np.linalg.norm(diff, axis=1)
    min_dists[first] = 0.0

    selection_log = []
    selection_log.append({
        'iteration': 1,
        'selected_index': int(first),
        'min_distance': float(min_dists.min()),
        'max_distance': float(min_dists.max()),
        'mean_distance': float(min_dists.mean()),
    })

    while len(selected) < n_select:
        max_val = float(min_dists.max())
        # tie-break: choose smallest index among those equal to max_val
        cand_idxs = np.where(np.isclose(min_dists, max_val))[0]
        if cand_idxs.size == 0:
            next_idx = int(np.argmax(min_dists))
        else:
            next_idx = int(cand_idxs.min())
        selected.append(next_idx)
        # update min_dists
        diff = vectors - vectors[next_idx]
        dists = np.linalg.norm(diff, axis=1)
        min_dists = np.minimum(min_dists, dists)
        min_dists[selected] = 0.0

        selection_log.append({
            'iteration': len(selected),
            'selected_index': int(next_idx),
            'min_distance': float(min_dists.min()),
            'max_distance': float(min_dists.max()),
            'mean_distance': float(min_dists.mean()),
        })

    return selected, selection_log


# ---------------------- Dataset builder ----------------------

def find_label_for_image(img_path: Path, labels_dir: Path):
    """Find label file for image under labels_dir. Return Path or None.

    Matching strategy:
      1. exact stem.txt
      2. prefix before first underscore
      3. any file starting with base prefix
    """
    if not labels_dir.exists():
        return None
    stem = img_path.stem
    cand = labels_dir / (stem + '.txt')
    if cand.exists():
        return cand
    if '_' in stem:
        base = stem.split('_')[0]
        cand2 = labels_dir / (base + '.txt')
        if cand2.exists():
            return cand2
    # fallback: search for files that start with base prefix
    base_prefix = stem.split('_')[0]
    for f in labels_dir.iterdir():
        if not f.is_file():
            continue
        if f.suffix == '.txt' and f.name.startswith(base_prefix):
            return f
    return None


def build_dataset(selected_records, out_dataset_dir: Path):
    """Create YOLO dataset structure and copy images/labels for selected_records.

    selected_records: list of record dicts (must include image_path)
    out_dataset_dir: path to dataset folder to create
    Returns (n_copied, n_missing_labels)
    """
    images_train = out_dataset_dir / 'images' / 'train'
    images_val = out_dataset_dir / 'images' / 'val'
    labels_train = out_dataset_dir / 'labels' / 'train'
    labels_val = out_dataset_dir / 'labels' / 'val'
    for p in [images_train, images_val, labels_train, labels_val]:
        p.mkdir(parents=True, exist_ok=True)

    n_copied = 0
    n_missing_labels = 0

    for rec in tqdm(selected_records, desc=f'Copying to {out_dataset_dir.name}'):
        img_rel = rec.get('image_path')
        if img_rel is None:
            continue
        img_path = Path(img_rel)
        if not img_path.is_absolute():
            img_path = PROJECT_ROOT / img_path
        if not img_path.exists():
            print(f"WARNING: source image not found: {img_path}")
            continue
        # detect split
        split = 'val' if 'val' in img_rel.split('/') else 'train'
        if split == 'val':
            dst_img = images_val / img_path.name
            dst_label_dir = labels_val
        else:
            dst_img = images_train / img_path.name
            dst_label_dir = labels_train
        shutil.copy2(img_path, dst_img)
        n_copied += 1
        # find label
        label = find_label_for_image(img_path, SRC_LABELS_TRAIN / split if split == 'train' else SRC_LABELS_VAL / split)
        # try both locations
        if label is None:
            label = find_label_for_image(img_path, SRC_LABELS_TRAIN)
        if label is None:
            label = find_label_for_image(img_path, SRC_LABELS_VAL)
        if label is None:
            n_missing_labels += 1
        else:
            dst_label = dst_label_dir / label.name
            shutil.copy2(label, dst_label)

    # write data.yaml
    data_yaml = {
        'train': str(images_train.resolve()),
        'val': str(images_val.resolve()),
        'nc': 1,
        'names': ['prawn']
    }
    try:
        import yaml
        with (out_dataset_dir / 'data.yaml').open('w') as fh:
            yaml.safe_dump(data_yaml, fh)
    except Exception:
        # fallback to simple write
        with (out_dataset_dir / 'data.yaml').open('w') as fh:
            fh.write(str(data_yaml))

    return n_copied, n_missing_labels


# ---------------------- Saving logs & plots ----------------------

def save_core_csv(out_dir: Path, k_label: str, selected_records, selection_order, distances_at_selection):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f'core_set_{k_label}.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'selection_order', 'distance_at_selection'])
        for order, (rec_idx, rec) in enumerate(zip(selection_order, selected_records), start=1):
            writer.writerow([rec.get('image_path'), order, distances_at_selection[order - 1]])
    return csv_path


def save_selection_log(out_dir: Path, k_label: str, selection_log):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f'selection_log_{k_label}.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iteration', 'selected_index', 'min_distance', 'max_distance', 'mean_distance'])
        for e in selection_log:
            writer.writerow([e['iteration'], e['selected_index'], e['min_distance'], e['max_distance'], e['mean_distance']])
    return csv_path


def save_elbow_plot(out_dir: Path, k_label: str, selection_log):
    out_dir.mkdir(parents=True, exist_ok=True)
    xs = [e['iteration'] for e in selection_log]
    ys = [e['max_distance'] for e in selection_log]
    plt.figure(figsize=(6,4))
    plt.plot(xs, ys, marker='o')
    plt.xlabel('iteration')
    plt.ylabel('max minimum distance')
    plt.title(f'Elbow curve ({k_label})')
    plt.grid(True)
    png_path = out_dir / f'elbow_curve_{k_label}.png'
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()
    return png_path


# ---------------------- Main ----------------------

def main():
    print('Fused-FPS core-set builder (Season 2025)')
    print(f'PROJECT_ROOT: {PROJECT_ROOT}')
    print(f'Loading fused embeddings: {FUSED_NPY}')

    records, fused = load_fused_embeddings(FUSED_NPY, RECORDS_JSON)
    N = fused.shape[0]
    print(f'Loaded {N} fused embeddings (shape={fused.shape})')

    # preprocess
    fused_norm = l2_normalize(fused)

    for k in K_LIST:
        n_select = int(math.ceil(k / 100.0 * N))
        if n_select <= 0:
            print(f'k={k}% -> n_select=0, skipping')
            continue
        k_label = f'k{int(k):02d}'
        print(f'\nProcessing k={k}% ({k_label}): selecting n={n_select} samples')

        selected_idxs, selection_log = run_fps(fused_norm, n_select)
        distances_at_selection = [selection_log[i]['max_distance'] for i in range(len(selection_log))]

        # assemble selected records in selection order
        selected_records = [records[i] for i in selected_idxs]

        # outputs
        out_log_dir = OUT_LOG_BASE / k_label
        out_log_dir.mkdir(parents=True, exist_ok=True)
        csv_path = save_core_csv(out_log_dir, k_label, selected_records, selected_idxs, distances_at_selection)
        log_path = save_selection_log(out_log_dir, k_label, selection_log)
        plot_path = save_elbow_plot(out_log_dir, k_label, selection_log)

        print(f'Saved core CSV: {csv_path}')
        print(f'Saved selection log: {log_path}')
        print(f'Saved elbow plot: {plot_path}')

        # build dataset
        out_ds_dir = OUT_DS_PARENT / f'train_on_2025_core_set_fused_fps_{k_label}'
        if out_ds_dir.exists():
            print(f'Output dataset {out_ds_dir} already exists; skipping dataset creation')
        else:
            n_copied, n_missing = build_dataset(selected_records, out_ds_dir)
            print(f'Dataset created at: {out_ds_dir} (images copied={n_copied}, missing_labels={n_missing})')

    print('\nAll k values processed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

