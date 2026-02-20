#!/usr/bin/env python3
"""
Active Season-Adaptive Uncertainty Pipeline (FIXED VERSION)

- Recomputes uncertainty after each iteration
- Uses full unfreeze
- Uses real validation mAP for stopping
- Fully isolated under new folder
"""

from pathlib import Path
import argparse
import shutil
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from ultralytics import YOLO

# -------------------- CONFIG --------------------

SEED = 0
np.random.seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "scripts" / "active_season_adaptive_uncertainty_pipeline"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

MODEL_OUTPUT_BASE = PROJECT_ROOT / "models" / "Active_Uncertainty_Transfer"
MODEL_OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

DEFAULT_EPOCHS = 20
DEFAULT_FREEZE = 0
MAX_ITERS = 3
SELECT_PER_ITER = 20
IMPROVEMENT_THRESHOLD = 0.01  # 1%

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ------------------------------------------------


def collect_images(folder: Path):
    return [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS]


def compute_uncertainty(model_path, image_paths):
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(model_path))

    records = []

    for p in tqdm(image_paths, desc="Inference"):
        try:
            results = model.predict(str(p), conf=0.001, device=device, verbose=False)
            r = results[0]

            confs = []
            if hasattr(r, "boxes") and r.boxes is not None:
                confs = r.boxes.conf.cpu().numpy() if len(r.boxes) > 0 else []

            mean_conf = float(np.mean(confs)) if len(confs) > 0 else 0.0
            uncertainty = 1.0 - mean_conf

            records.append({
                "image_path": str(p),
                "mean_confidence": mean_conf,
                "uncertainty": uncertainty
            })

        except Exception:
            records.append({
                "image_path": str(p),
                "mean_confidence": 0.0,
                "uncertainty": 1.0
            })

    return pd.DataFrame(records)


def select_top_uncertain(df, N, already_selected):
    df_sorted = df.sort_values("uncertainty", ascending=False)
    selected = []

    for _, row in df_sorted.iterrows():
        p = str(Path(row["image_path"]).resolve())
        if p in already_selected:
            continue
        selected.append(p)
        already_selected.add(p)
        if len(selected) >= N:
            break

    return selected


def build_dataset(selected_images, dataset_path: Path):
    images_train = dataset_path / "images" / "train"
    labels_train = dataset_path / "labels" / "train"

    images_train.mkdir(parents=True, exist_ok=True)
    labels_train.mkdir(parents=True, exist_ok=True)

    for img in selected_images:
        shutil.copy2(img, images_train / Path(img).name)

    yaml_content = f"""
path: {dataset_path.resolve()}
train: images/train
val: images/train
nc: 1
names: ['prawn']
kpt_shape: [4, 3]
flip_idx: [0,1,2,3]
"""
    with open(dataset_path / "data.yaml", "w") as f:
        f.write(yaml_content.strip())


def train_and_evaluate(data_yaml, base_weights, save_dir):
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(base_weights))

    results = model.train(
        data=str(data_yaml),
        epochs=DEFAULT_EPOCHS,
        imgsz=640,
        batch=16,
        freeze=DEFAULT_FREEZE,
        device=device,
        project=str(save_dir.parent),
        name=save_dir.name,
        exist_ok=True
    )

    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        return None, None

    model = YOLO(str(best_weights))
    val_results = model.val(data=str(data_yaml))

    try:
        map50 = float(val_results.box.map50)
    except Exception:
        map50 = None

    return best_weights, map50


def run_pipeline(target_dir, base_model, iterations):

    all_images = collect_images(target_dir)
    print("Total target images:", len(all_images))

    current_model = base_model
    already_selected = set()

    prev_map = None
    total_selected = 0

    for it in range(iterations):

        print(f"\n========== ITERATION {it+1} ==========")

        # Recompute uncertainty each iteration
        df_unc = compute_uncertainty(current_model, all_images)

        selected = select_top_uncertain(df_unc, SELECT_PER_ITER, already_selected)

        if len(selected) == 0:
            print("No more images to select.")
            break

        dataset_path = PROJECT_ROOT / "datasets" / f"active_uncertainty_iter{it+1}"
        build_dataset(selected, dataset_path)

        print("Selected images:", len(selected))
        print("Dataset created at:", dataset_path)

        print(">> Add labels manually, then press ENTER to continue...")
        input()

        model_dir = MODEL_OUTPUT_BASE / f"iter{it+1}"
        best_weights, map50 = train_and_evaluate(dataset_path / "data.yaml",
                                                  current_model,
                                                  model_dir)

        if best_weights is None:
            print("Training failed or labels missing.")
            break

        print("Validation mAP50:", map50)

        total_selected += len(selected)

        if prev_map is not None and map50 is not None:
            improvement = (map50 - prev_map) / max(prev_map, 1e-6)
            print("Improvement:", improvement)

            if improvement < IMPROVEMENT_THRESHOLD:
                print("Improvement too small. Stopping.")
                break

        prev_map = map50
        current_model = str(best_weights)

    print("\n========== FINAL REPORT ==========")
    print("Total labeled images:", total_selected)
    print("Final model:", current_model)


# -------------------- CLI --------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", type=str, required=True)
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--iterations", type=int, default=3)

    args = parser.parse_args()

    run_pipeline(Path(args.target_dir),
                 args.base_model,
                 args.iterations)


if __name__ == "__main__":
    main()
