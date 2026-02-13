"""
Build YOLO-style datasets from generated core_set CSVs.

For each directory under outputs/rep_analysis/core_set_selection/ matching
random_2025_k*, read core_set.csv and construct a dataset at:
  datasets/train_on_2025_core_set_kXX/
with structure:
  images/train images/val labels/train labels/val

Behavior and constraints:
- PROJECT_ROOT detected robustly (same method as other scripts).
- No metadata CSVs or embeddings used.
- Copy images and corresponding label files deterministically.
- Support label filenames that may contain additional hashing or prefixes.
- Create data.yaml pointing to the new dataset paths.
- Print summary per dataset: images copied, missing labels, output path.
"""
from pathlib import Path
import shutil
import yaml
import csv
import sys
import re

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORESET_BASE = PROJECT_ROOT / 'outputs' / 'rep_analysis' / 'core_set_selection'
SRC_DATA = PROJECT_ROOT / 'datasets' / 'train_on_2025_all'
OUT_DATA_PARENT = PROJECT_ROOT / 'datasets'

LABEL_EXT = '.txt'
IMG_EXT = '.jpg'


def find_label_for_image(img_path: Path, labels_dir: Path):
    """Find a label file for the given image path under labels_dir.

    Returns Path or None.
    """
    if not labels_dir.exists():
        return None
    # image name
    name = img_path.stem  # without extension
    # candidate 1: exact match
    cand = labels_dir / (name + LABEL_EXT)
    if cand.exists():
        return cand
    # candidate 2: strip suffix after first underscore if present
    if '_' in name:
        base = name.split('_')[0]
        cand2 = labels_dir / (base + LABEL_EXT)
        if cand2.exists():
            return cand2
    # candidate 3: try regex matching first prefix
    # search for any file that starts with base
    base_prefix = name.split('_')[0]
    for f in labels_dir.iterdir():
        if not f.is_file():
            continue
        if f.name.endswith(LABEL_EXT) and f.name.startswith(base_prefix):
            return f
    return None


def read_core_csv(path: Path):
    rows = []
    with path.open('r', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def make_dataset_from_core(core_csv_path: Path, out_dataset_dir: Path):
    """Create a YOLO-style dataset folder from core_set.csv entries.

    Returns a tuple (n_images_copied, n_missing_labels)
    """
    rows = read_core_csv(core_csv_path)
    out_images_train = out_dataset_dir / 'images' / 'train'
    out_images_val = out_dataset_dir / 'images' / 'val'
    out_labels_train = out_dataset_dir / 'labels' / 'train'
    out_labels_val = out_dataset_dir / 'labels' / 'val'
    for d in [out_images_train, out_images_val, out_labels_train, out_labels_val]:
        d.mkdir(parents=True, exist_ok=True)

    n_copied = 0
    n_missing_labels = 0
    for r in rows:
        rel_path = Path(r['image_path'])
        split = r.get('split', 'train')
        src_img = PROJECT_ROOT / rel_path
        if not src_img.exists():
            print(f"WARNING: source image not found: {src_img}")
            continue
        # destination
        if split == 'val':
            dst_img = out_images_val / src_img.name
            dst_label_dir = out_labels_val
        else:
            dst_img = out_images_train / src_img.name
            dst_label_dir = out_labels_train
        # copy image
        shutil.copy2(src_img, dst_img)
        n_copied += 1
        # locate label in source dataset
        # original labels are assumed under datasets/train_on_2025_all/labels and val/labels
        src_labels_dir = SRC_DATA / 'labels' / split
        label_file = find_label_for_image(src_img, src_labels_dir)
        if label_file is None:
            # try val labels if split was train and vice versa
            other_dir = SRC_DATA / 'labels' / ('val' if split == 'train' else 'train')
            label_file = find_label_for_image(src_img, other_dir)
        if label_file is None:
            # record missing label
            n_missing_labels += 1
        else:
            dst_label = dst_label_dir / label_file.name
            shutil.copy2(label_file, dst_label)
    # create data.yaml
    data_yaml = {
        'train': str((out_dataset_dir / 'images' / 'train').resolve()),
        'val': str((out_dataset_dir / 'images' / 'val').resolve()),
        'nc': 1,
        'names': ['prawn']
    }
    with (out_dataset_dir / 'data.yaml').open('w') as f:
        yaml.safe_dump(data_yaml, f)
    return n_copied, n_missing_labels


def main():
    pattern = 'random_2025_k'
    candidates = [p for p in CORESET_BASE.iterdir() if p.is_dir() and p.name.startswith(pattern)]
    if not candidates:
        print(f"No core-set directories found under {CORESET_BASE} matching {pattern}*")
        return 1
    for c in sorted(candidates):
        core_csv = c / 'core_set.csv'
        if not core_csv.exists():
            print(f"Skipping {c}, no core_set.csv found")
            continue
        # determine k label from directory name
        klabel = c.name
        out_dataset_dir = OUT_DATA_PARENT / f'train_on_2025_core_set_{klabel}'
        if out_dataset_dir.exists():
            print(f"Output dataset already exists, skipping: {out_dataset_dir}")
            continue
        n_copied, n_missing = make_dataset_from_core(core_csv, out_dataset_dir)
        print(f"Built dataset {out_dataset_dir}: images copied={n_copied}, missing_labels={n_missing}")
    return 0


if __name__ == '__main__':
    sys.exit(main())

