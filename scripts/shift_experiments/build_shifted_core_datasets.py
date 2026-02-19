#!/usr/bin/env python3
"""
build_shifted_core_datasets.py

Build YOLO-style datasets from shifted experiment core-set CSVs.

For each k in K_LIST the script will:
 - read outputs/rep_analysis/core_set_selection/shifted_2025_kXX/core_set.csv
 - create datasets/train_on_2025_shifted_kXX/ with images/train images/val labels/train labels/val
 - copy images and matching labels from the shifted experiment source
 - write data.yaml with exact required content
 - print dataset path and counts

Usage: run from project root or anywhere; paths are resolved from PROJECT_ROOT.
"""
from pathlib import Path
import shutil
import csv
import sys
from typing import Tuple

# Configuration
PROJECT_ROOT = Path(r"C:\Users\carmonta\Desktop\data-centric-cross-season-transfer")
SHIFTED_SOURCE_ROOT = PROJECT_ROOT / "outputs" / "rep_analysis" / "core_set_selection" / "shifted_2025_experiment" / "shifted_2025_experiment"
CORESETS_BASE = PROJECT_ROOT / "outputs" / "rep_analysis" / "core_set_selection"
K_LIST = [1, 2, 5, 10, 20, 50]

# Helper functions

def read_core_csv(csv_path: Path):
    """Read core_set.csv expected to have columns image_path and split.
    Returns list of tuples (image_path_relative_str, split_str).
    """
    rows = []
    if not csv_path.exists():
        return rows
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            img = r.get('image_path') or r.get('image') or r.get('path')
            split = r.get('split') or r.get('dataset_type') or r.get('split')
            if img is None:
                continue
            split = (split or 'train').strip().lower()
            if split not in ('train', 'val'):
                # infer from path
                split = 'val' if 'val' in img.lower() else 'train'
            rows.append((img, split))
    return rows


def ensure_dataset_dirs(dataset_root: Path):
    for d in (
        dataset_root / 'images' / 'train',
        dataset_root / 'images' / 'val',
        dataset_root / 'labels' / 'train',
        dataset_root / 'labels' / 'val',
    ):
        d.mkdir(parents=True, exist_ok=True)


def copy_image_and_label(rel_img_path: str, split: str, dataset_root: Path) -> Tuple[bool, bool]:
    """Copy image and its matching label from shifted source to dataset_root preserving split.
    rel_img_path is relative to PROJECT_ROOT. Returns (copied_image, copied_label)
    """
    # Resolve source image path
    src_img = (PROJECT_ROOT / rel_img_path).resolve()
    if not src_img.exists():
        # try under SHIFTED_SOURCE_ROOT images folder
        candidate = None
        for root in (SHIFTED_SOURCE_ROOT / 'images' / 'train', SHIFTED_SOURCE_ROOT / 'images' / 'val'):
            cand = root / Path(rel_img_path).name
            if cand.exists():
                candidate = cand
                break
        if candidate is None:
            return False, False
        src_img = candidate
    # destination image path
    dst_img_dir = dataset_root / 'images' / split
    dst_img = dst_img_dir / src_img.name
    try:
        shutil.copy2(src_img, dst_img)
        copied_img = True
    except Exception:
        copied_img = False
    # find label: same stem with .txt under shifted labels
    stem = src_img.stem
    label_src = None
    for root in (SHIFTED_SOURCE_ROOT / 'labels' / split, SHIFTED_SOURCE_ROOT / 'labels' / ('train' if split == 'train' else 'val')):
        cand = root / (stem + '.txt')
        if cand.exists():
            label_src = cand
            break
    # fallback search
    if label_src is None:
        for root in (SHIFTED_SOURCE_ROOT / 'labels' / 'train', SHIFTED_SOURCE_ROOT / 'labels' / 'val'):
            if not root.exists():
                continue
            for p in root.rglob('*.txt'):
                if p.stem == stem:
                    label_src = p
                    break
            if label_src is not None:
                break
    copied_label = False
    if label_src is not None and label_src.exists():
        dst_lbl_dir = dataset_root / 'labels' / split
        dst_lbl = dst_lbl_dir / label_src.name
        try:
            shutil.copy2(label_src, dst_lbl)
            copied_label = True
        except Exception:
            copied_label = False
    return copied_img, copied_label


def write_data_yaml(dataset_root: Path):
    train_rel = 'images/train'
    val_rel = 'images/val'
    content_lines = [
        f"path: {str(dataset_root.resolve())}",
        f"train: {train_rel}",
        f"val: {val_rel}",
        f"nc: 1",
        "names: ['prawn']",
        "kpt_shape: [4, 3]",
        "flip_idx: [0, 1, 2, 3]",
    ]
    p = dataset_root / 'data.yaml'
    with open(p, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(content_lines) + '\n')
    return p


def build_dataset_for_k(k: int):
    k_str = f'k{int(k):02d}'
    csv_path = CORESETS_BASE / f'shifted_2025_k{k_str}' / 'core_set.csv'
    if not csv_path.exists():
        print(f'Warning: core set CSV not found for k={k}: {csv_path} (skipping)')
        return None
    rows = read_core_csv(csv_path)
    dataset_root = PROJECT_ROOT / f'datasets' / f'train_on_2025_shifted_{k_str}'
    ensure_dataset_dirs(dataset_root)
    train_count = 0
    val_count = 0
    for rel_img, split in rows:
        copied_img, copied_lbl = copy_image_and_label(rel_img, split, dataset_root)
        if copied_img:
            if split == 'train':
                train_count += 1
            else:
                val_count += 1
        else:
            # can't find image: try to see if the rel_img is already relative to SHIFTED_SOURCE_ROOT
            alt = (SHIFTED_SOURCE_ROOT / rel_img).resolve()
            if alt.exists():
                # copy from alt
                try:
                    dst = (dataset_root / 'images' / split) / alt.name
                    shutil.copy2(alt, dst)
                    if split == 'train':
                        train_count += 1
                    else:
                        val_count += 1
                except Exception:
                    pass
    # write data.yaml
    yaml_path = write_data_yaml(dataset_root)
    return dataset_root, train_count, val_count


def main():
    results = []
    for k in K_LIST:
        res = build_dataset_for_k(k)
        if res is None:
            continue
        dataset_root, train_count, val_count = res
        print(f'k={k:02d}: dataset={dataset_root} train={train_count} val={val_count}')
        results.append({'k': k, 'dataset': str(dataset_root.resolve()), 'train': train_count, 'val': val_count})
    # optional summary JSON
    summary_p = CORESETS_BASE / 'shifted_cores_summary.json'
    with open(summary_p, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print('\nSummary written to', summary_p)


if __name__ == '__main__':
    main()

