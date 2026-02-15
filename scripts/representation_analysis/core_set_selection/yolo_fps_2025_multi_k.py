#!/usr/bin/env python3
"""
Create Season-2025 core-sets using YOLO embeddings and Farthest Point Sampling (FPS).

Based on the structure of build_2025_core_set_dataset_multi_k.py but selects samples
using FPS over pretrained YOLO embeddings (no recomputation or modification).

Data sources (relative to PROJECT_ROOT):
- scripts/representation_analysis/outputs_repreasentation/rep_analysis/embeddings_meta.csv
- scripts/representation_analysis/outputs_repreasentation/rep_analysis/embeddings_vectors.npy

Outputs:
- datasets/train_on_2025_core_set_yolo_fps_kXX/ (YOLO dataset structure)
- outputs/rep_analysis/core_set_selection/yolo_fps/
    - core_set_kXX.csv
    - selection_log_kXX.csv
    - elbow_curve_kXX.png

Requirements satisfied:
- Deterministic (numpy seed = 0)
- Tie-breaking chooses smallest index
- Preserves original ordering of embeddings
- Uses pathlib, tqdm, clean modular functions
"""
from pathlib import Path
import csv
import sys
import math
import shutil
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configuration
K_LIST = [1, 2, 5, 10, 20, 50]
SEED = 0
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Embeddings paths (as requested)
EMBED_META = PROJECT_ROOT / 'scripts' / 'representation_analysis' / 'outputs_repreasentation' / 'rep_analysis' / 'embeddings_meta.csv'
EMBED_VECT = PROJECT_ROOT / 'scripts' / 'representation_analysis' / 'outputs_repreasentation' / 'rep_analysis' / 'embeddings_vectors.npy'

# Output locations
OUT_CORE_BASE = PROJECT_ROOT / 'outputs' / 'rep_analysis' / 'core_set_selection' / 'yolo_fps'
OUT_CORE_BASE.mkdir(parents=True, exist_ok=True)

# Dataset target prefix
DATASET_PREFIX = 'datasets' / 'train_on_2025_core_set_yolo_fps_'

# Source dataset for labels/images
SRC_DATA = PROJECT_ROOT / 'datasets' / 'train_on_2025_all'

# Helpers for label matching (copied/adapted from existing builder)
LABEL_EXT = '.txt'
IMG_EXTS = ('.jpg', '.jpeg', '.png')

rng = np.random.RandomState(SEED)


# ---------------------- I/O / Loading ----------------------
def load_embeddings(meta_path: Path, vec_path: Path):
    """Load metadata (CSV) and embedding vectors (npy).

    Returns (meta_rows, vectors) where meta_rows is a list of dicts in file order,
    and vectors is an (N, D) numpy array. Raises informative errors on mismatch.
    """
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {meta_path}")
    if not vec_path.exists():
        raise FileNotFoundError(f"Embedding vectors file not found: {vec_path}")

    # load meta
    meta = []
    with meta_path.open('r', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            meta.append(r)

    vectors = np.load(str(vec_path))

    if len(meta) != vectors.shape[0]:
        raise ValueError(f"Length mismatch: meta rows={len(meta)} vs vectors.shape[0]={vectors.shape[0]}")

    return meta, vectors


def filter_2025(meta_rows, vectors):
    """Filter meta and vectors to season == '2025'. Preserve original ordering.

    Returns (meta_2025, vectors_2025, indices_in_original)
    """
    meta2025 = []
    idxs = []
    for i, r in enumerate(meta_rows):
        # accept season value exactly equal to '2025' (string)
        season = r.get('season') if isinstance(r, dict) else None
        if season == '2025' or season == 2025:
            meta2025.append(r)
            idxs.append(i)
    if len(idxs) == 0:
        raise ValueError('No Season 2025 records found in metadata.')
    vec2025 = vectors[np.array(idxs, dtype=int)]
    return meta2025, vec2025, idxs


# ---------------------- Embedding preprocessing ----------------------
def l2_normalize(vectors: np.ndarray):
    """L2 normalize rows of vectors. Returns normalized copy."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


# ---------------------- FPS algorithm ----------------------
def run_fps(vectors: np.ndarray, n_select: int):
    """Run deterministic FPS over vectors (L2-normalized) using Euclidean distance.

    Returns selected_indices (list, length n_select) and selection_log (list of dicts).
    selected indices are indices into the provided vectors array (0..M-1).
    """
    M = vectors.shape[0]
    if n_select <= 0:
        return [], []
    if n_select >= M:
        # return all indices in original order
        selection_log = []
        for idx in range(M):
            selection_log.append({
                'iteration': idx + 1,
                'selected_index': int(idx),
                'min_distance': 0.0,
                'max_distance': 0.0,
                'mean_distance': 0.0,
            })
        return list(range(M)), selection_log

    # use deterministic initial seed
    first = int(rng.randint(0, M))

    selected = [first]
    # distances of each point to the closest selected point
    # initialize with distances to first
    # compute Euclidean distances
    diff = vectors - vectors[first]
    min_dists = np.linalg.norm(diff, axis=1)

    # set selected min_dist to 0
    min_dists[first] = 0.0

    selection_log = []
    # record stats for iteration 1
    selection_log.append({
        'iteration': 1,
        'selected_index': int(first),
        'min_distance': float(min_dists.min()),
        'max_distance': float(min_dists.max()),
        'mean_distance': float(min_dists.mean()),
    })

    # iterate
    while len(selected) < n_select:
        # choose argmax of min_dists; tie-break choose smallest index
        max_val = float(min_dists.max())
        # indices with max_val (use isclose to be robust)
        cand_idxs = np.where(np.isclose(min_dists, max_val))[0]
        if cand_idxs.size == 0:
            # numerical edge-case: fallback to argmax
            next_idx = int(np.argmax(min_dists))
        else:
            next_idx = int(cand_idxs.min())

        selected.append(next_idx)

        # update min_dists: compute distance to new selected point and take min
        diff = vectors - vectors[next_idx]
        dists = np.linalg.norm(diff, axis=1)
        # ensure selected indices have 0 distance
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


# ---------------------- Dataset building (copying) ----------------------
def find_label_for_image(img_path: Path, labels_dir: Path):
    """Find label file for given image under labels_dir. Return Path or None."""
    if not labels_dir.exists():
        return None
    name = img_path.stem
    cand = labels_dir / (name + LABEL_EXT)
    if cand.exists():
        return cand
    # try stripping after first underscore
    if '_' in name:
        base = name.split('_')[0]
        cand2 = labels_dir / (base + LABEL_EXT)
        if cand2.exists():
            return cand2
    # fallback: find file starting with base prefix
    base_prefix = name.split('_')[0]
    for f in labels_dir.iterdir():
        if not f.is_file():
            continue
        if f.name.endswith(LABEL_EXT) and f.name.startswith(base_prefix):
            return f
    return None


def build_dataset(selected_meta, out_dataset_dir: Path):
    """Create YOLO dataset structure and copy images and labels for selected_meta list.

    selected_meta: list of meta dicts, each must include 'image_path' (relative to PROJECT_ROOT)
    out_dataset_dir: Path where dataset will be created
    Returns (n_copied, n_missing_labels)
    """
    out_images_train = out_dataset_dir / 'images' / 'train'
    out_images_val = out_dataset_dir / 'images' / 'val'
    out_labels_train = out_dataset_dir / 'labels' / 'train'
    out_labels_val = out_dataset_dir / 'labels' / 'val'
    for d in [out_images_train, out_images_val, out_labels_train, out_labels_val]:
        d.mkdir(parents=True, exist_ok=True)

    n_copied = 0
    n_missing_labels = 0

    for r in tqdm(selected_meta, desc=f"Copying to {out_dataset_dir.name}"):
        img_rel = Path(r['image_path'])
        # resolve absolute path
        src_img = PROJECT_ROOT / img_rel
        if not src_img.exists():
            print(f"[WARN] source image not found: {src_img}; skipping")
            continue
        # determine split by checking whether 'val' is in the path fragments
        split = 'val' if 'val' in img_rel.parts else 'train'
        if split == 'val':
            dst_img = out_images_val / src_img.name
            dst_label_dir = out_labels_val
        else:
            dst_img = out_images_train / src_img.name
            dst_label_dir = out_labels_train
        shutil.copy2(src_img, dst_img)
        n_copied += 1

        # find label
        src_labels_dir = SRC_DATA / 'labels' / split
        label_file = find_label_for_image(src_img, src_labels_dir)
        if label_file is None:
            other_dir = SRC_DATA / 'labels' / ('val' if split == 'train' else 'train')
            label_file = find_label_for_image(src_img, other_dir)
        if label_file is None:
            n_missing_labels += 1
        else:
            dst_label = dst_label_dir / label_file.name
            shutil.copy2(label_file, dst_label)

    # write data.yaml for convenience
    data_yaml = {
        'train': str((out_dataset_dir / 'images' / 'train').resolve()),
        'val': str((out_dataset_dir / 'images' / 'val').resolve()),
        'nc': 1,
        'names': ['prawn']
    }
    with (out_dataset_dir / 'data.yaml').open('w') as f:
        import yaml
        yaml.safe_dump(data_yaml, f)

    return n_copied, n_missing_labels


# ---------------------- Saving logs & plots ----------------------
def save_core_csv(out_dir: Path, k_label: str, selected_meta, selection_order, distances_at_selection):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f'core_set_{k_label}.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'selection_order', 'distance_at_selection'])
        for order, (meta, idx) in enumerate(zip(selected_meta, selection_order), start=1):
            writer.writerow([meta['image_path'], order, distances_at_selection[order - 1]])
    return csv_path


def save_selection_log(out_dir: Path, k_label: str, selection_log):
    csv_path = out_dir / f'selection_log_{k_label}.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iteration', 'selected_index', 'min_distance', 'max_distance', 'mean_distance'])
        for entry in selection_log:
            writer.writerow([
                entry['iteration'], entry['selected_index'], entry['min_distance'], entry['max_distance'], entry['mean_distance']
            ])
    return csv_path


def save_elbow_plot(out_dir: Path, k_label: str, selection_log):
    xs = [e['iteration'] for e in selection_log]
    ys = [e['max_distance'] for e in selection_log]
    plt.figure(figsize=(6, 4))
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
    print('YOLO-FPS core-set builder (Season 2025)')
    print(f'PROJECT_ROOT: {PROJECT_ROOT}')
    print(f'Loading embeddings meta: {EMBED_META}')
    print(f'Loading embeddings vectors: {EMBED_VECT}')

    meta_rows, vectors = load_embeddings(EMBED_META, EMBED_VECT)

    meta2025, vec2025, orig_idxs = filter_2025(meta_rows, vectors)
    total_2025 = vec2025.shape[0]
    print(f'Found {total_2025} Season-2025 embeddings')

    # normalize
    vec2025 = l2_normalize(vec2025.astype(np.float64))

    # For mapping back to original meta rows, meta2025 is list aligned with vec2025

    for k in K_LIST:
        # compute n_select = ceil(k% * total_2025)
        n_select = int(math.ceil(k / 100.0 * total_2025))
        if n_select <= 0:
            print(f'k={k}% -> n_select=0, skipping')
            continue
        k_label = f'k{int(k):02d}'
        print(f'\nProcessing k={k}%, n_select={n_select} ({k_label})')

        # run fps
        selected_idxs, selection_log = run_fps(vec2025, n_select)

        # selection order distances: distance at selection is the max min distance at that iteration
        distances_at_selection = [selection_log[i]['max_distance'] for i in range(len(selection_log))]

        # build selected_meta list (in selection order)
        selected_meta = [meta2025[i] for i in selected_idxs]

        # outputs
        out_dir_k = OUT_CORE_BASE / k_label
        out_dir_k.mkdir(parents=True, exist_ok=True)

        core_csv = save_core_csv(out_dir_k, k_label, selected_meta, selected_idxs, distances_at_selection)
        log_csv = save_selection_log(out_dir_k, k_label, selection_log)
        plot_png = save_elbow_plot(out_dir_k, k_label, selection_log)

        print(f'Saved core CSV: {core_csv}')
        print(f'Saved selection log: {log_csv}')
        print(f'Saved elbow plot: {plot_png}')

        # build dataset folder using same structure as random builder
        dataset_out = PROJECT_ROOT / f'datasets/train_on_2025_core_set_yolo_fps_{k_label}'
        n_copied, n_missing = build_dataset(selected_meta, dataset_out)
        print(f'Dataset built at: {dataset_out} (images copied={n_copied}, missing_labels={n_missing})')

    print('\nAll k values processed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

