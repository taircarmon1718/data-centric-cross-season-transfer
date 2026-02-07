from pathlib import Path
import pandas as pd
import shutil
import os
import yaml

# ============================================================
# PATHS (project-root relative)
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DATASET = PROJECT_ROOT / "datasets" / "train_on_2025_all"
CORE_SELECTION_DIR = PROJECT_ROOT / "outputs" / "rep_analysis" / "core_set_selection"

# Map selection folder -> output dataset name suffix
SELECTION_MAP = {
    'yolo_elbow': 'yolo',
    'fused_elbow': 'fused'
}

# Candidate source roots to look for images (in order)
CAND_SRC_ROOTS = [
    SRC_DATASET,
    PROJECT_ROOT / 'datasets' / 'train_on_all',
    PROJECT_ROOT / 'data' / 'images',
    PROJECT_ROOT
]

# ============================================================
# Helpers
# ============================================================

def norm_path_str(s: str) -> str:
    if s is None:
        return ''
    return str(s).replace('\\', '/').strip()


def find_src_image(img_name: str, is_val: bool):
    """Try several candidate locations to find the source image and corresponding label path.
    Returns (src_img_path, src_lbl_path) or (None, None) if not found.
    """
    name = os.path.basename(img_name)
    candidates = []
    # prefer explicit split
    for root in CAND_SRC_ROOTS:
        if is_val:
            candidates.append(root / 'val' / 'images' / name)
            candidates.append(root / 'val' / 'images' / (name.replace('.jpg', '.jpeg')))
        candidates.append(root / 'images' / name)
    # also try direct absolute/relative path
    p = Path(img_name)
    if p.exists():
        cand_img = p
        cand_lbl = p.parent.parent / 'labels' / (p.name.replace('.jpg', '.txt')) if (p.parent.parent / 'labels').exists() else p.with_suffix('.txt')
        return cand_img, cand_lbl

    for c in candidates:
        if c.exists():
            # label path: look in same relative structure under labels
            # prefer corresponding labels directory near the found image
            lbl = None
            # try sibling labels folder (same parent parent)
            if 'images' in str(c.parent.name):
                # parent is images or images/<split>
                lbl_candidate = c.parent.parent / 'labels' / (c.name.replace('.jpg', '.txt'))
                if lbl_candidate.exists():
                    lbl = lbl_candidate
            # fallback: images/../labels
            fallback = c.with_suffix('.txt')
            if lbl is None and fallback.exists():
                lbl = fallback
            # final fallback: try root labels
            if lbl is None:
                for root in CAND_SRC_ROOTS:
                    candidate_lbl = root / 'labels' / (c.name.replace('.jpg', '.txt'))
                    if candidate_lbl.exists():
                        lbl = candidate_lbl
                        break
            return c, lbl
    return None, None


def write_data_yaml(src_yaml: Path, dst_yaml: Path, src_dataset: Path, dst_dataset: Path):
    # copy if exists and adjust paths; otherwise synthesize minimal yaml
    if src_yaml.exists():
        text = src_yaml.read_text(encoding='utf-8')
        text = text.replace(str(src_dataset).replace('\\','/'), str(dst_dataset).replace('\\','/'))
        dst_yaml.write_text(text, encoding='utf-8')
        return
    # synthesize
    data = {
        'train': str(dst_dataset / 'images' / 'train').replace('\\','/'),
        'val': str(dst_dataset / 'images' / 'val').replace('\\','/'),
        'nc': 1,
        'names': ['prawn']
    }
    with open(dst_yaml, 'w', encoding='utf-8') as fh:
        yaml.safe_dump(data, fh)


# ============================================================
# Main: iterate selection folders
# ============================================================
for sel_folder, suffix in SELECTION_MAP.items():
    sel_dir = CORE_SELECTION_DIR / sel_folder
    csv_path = sel_dir / 'core_set.csv'
    if not csv_path.exists():
        print(f"[WARN] selection CSV not found for '{sel_folder}' at {csv_path}; skipping")
        continue

    # Destination dataset
    DST_DATASET = PROJECT_ROOT / 'datasets' / f'train_on_2025_core_set_{suffix}'

    # create dirs
    for split in ['train', 'val']:
        (DST_DATASET / 'images' / split).mkdir(parents=True, exist_ok=True)
        (DST_DATASET / 'labels' / split).mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Building core-set dataset for '{sel_folder}' -> {DST_DATASET}")
    df = pd.read_csv(str(csv_path))

    # Accept both 'image_id' and 'image_path' columns
    img_col = 'image_path' if 'image_path' in df.columns else ('image_id' if 'image_id' in df.columns else None)
    if img_col is None:
        print(f"[ERROR] No image column found in {csv_path}. Expected 'image_path' or 'image_id'. Skipping.")
        continue

    missing_images = 0
    missing_labels = 0
    copied = 0

    for _, row in df.iterrows():
        raw = row[img_col]
        if pd.isna(raw):
            print(f"[WARN] empty image path in CSV row; skipping")
            missing_images += 1
            continue
        s = norm_path_str(str(raw))
        # determine split by presence of '/val/' or '/val/images'
        is_val = ('/val/' in s) or ('/val/images' in s) or ('/val\\' in str(raw))
        img_name = Path(s).name

        src_img, src_lbl = find_src_image(img_name, is_val)
        if src_img is None:
            # best-effort: also try with exact raw path relative to project
            candidate = PROJECT_ROOT / s.lstrip('/\\')
            if candidate.exists():
                src_img = candidate
                src_lbl = candidate.with_suffix('.txt')
        if src_img is None:
            print(f"[WARN] Missing image for CSV entry: {raw} (tried multiple locations)")
            missing_images += 1
            continue

        # determine destination paths
        split = 'val' if is_val else 'train'
        dst_img = DST_DATASET / 'images' / split / src_img.name
        dst_lbl = DST_DATASET / 'labels' / split / (src_lbl.name if src_lbl is not None else src_img.with_suffix('.txt').name)

        # copy image
        try:
            shutil.copy2(src_img, dst_img)
            copied += 1
        except Exception as e:
            print(f"[WARN] Failed to copy image {src_img} -> {dst_img}: {e}")
            missing_images += 1
            continue

        # copy label if exists
        if src_lbl is not None and src_lbl.exists():
            try:
                shutil.copy2(src_lbl, dst_lbl)
            except Exception as e:
                print(f"[WARN] Failed to copy label {src_lbl} -> {dst_lbl}: {e}")
                missing_labels += 1
        else:
            print(f"[WARN] Missing label for image {src_img}; expected {src_lbl}")
            missing_labels += 1

    # copy or synth data.yaml
    src_yaml = SRC_DATASET / 'data.yaml'
    dst_yaml = DST_DATASET / 'data.yaml'
    write_data_yaml(src_yaml, dst_yaml, SRC_DATASET, DST_DATASET)

    # summary
    print(f"✅ Core-set dataset build complete for '{sel_folder}'")
    print(f"   Source dataset: {SRC_DATASET}")
    print(f"   Target dataset: {DST_DATASET}")
    print(f"   Images copied: {copied}")
    print(f"   Missing images: {missing_images}")
    print(f"   Missing labels: {missing_labels}")

print('\nAll done.')
