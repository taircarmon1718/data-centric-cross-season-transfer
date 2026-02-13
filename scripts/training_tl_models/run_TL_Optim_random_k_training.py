#!/usr/bin/env python3
"""
Batch TL-Optim training over random-k 2025 core-set datasets.

This script scans `datasets/` for folders named:
  train_on_2025_core_set_random_2025_k*

For each dataset found it runs a two-stage TL-Optim style training:
 - Stage 1: feature adaptation (freeze=8) using TL-Optim hyperparameters
 - Stage 2: end-to-end fine-tuning (freeze=0) as a short refinement

Outputs are saved under:
  models/TL_Optim_core_set/k_random/<klabel>/

Requirements satisfied:
 - PROJECT_ROOT detected automatically via pathlib
 - Uses the specified base weights for 2024->2025 transfer
 - Windows multiprocessing safe (if __name__ == '__main__')
 - Deterministic behavior where applicable
 - Clear logging and robust skipping on errors

Copy the file to: scripts/training_tl_models/run_TL_Optim_random_k_training.py
and run it with: python scripts/training_tl_models/run_TL_Optim_random_k_training.py
"""

# MUST set env vars BEFORE importing torch / ultralytics
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONWARNINGS"] = "ignore"

from pathlib import Path
import time
import sys
from typing import Optional

# Import ultralytics after env var setup
try:
    from ultralytics import YOLO
except Exception as e:
    print("ERROR: could not import ultralytics.YOLO. Ensure ultralytics is installed and available.")
    raise

# -------------------------
# Configuration
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = PROJECT_ROOT / "datasets"
DATASET_GLOB = "train_on_2025_core_set_random_2025_k*"

# Base weights for 2024 -> 2025 transfer (TL starting point)
BASE_WEIGHTS = PROJECT_ROOT / "models" / "TF" / "tf_grids" / "B2" / "from_2024to2025" / "B2_TL_2024to2025_k100pct_freeze4_S2_best.pt"

# Output base
OUT_BASE = PROJECT_ROOT / "models" / "TL_Optim_core_set" / "k_random"
OUT_BASE.mkdir(parents=True, exist_ok=True)

# TL-Optim (Stage 1) hyperparameters (as requested)
STAGE1_CFG = {
    "freeze": 8,
    "epochs": 150,
    "imgsz": 640,
    "batch": 16,
    "lr0": 0.01,
    "lrf": 0.01,
    "patience": 100,
    "deterministic": True,
    "optimizer": "auto",
    "device": 0,
    "val": True,
    "plots": True,
    "verbose": True,
    # avoid heavy multiprocessing issues
    "workers": 0,
}

# Stage 2 (short refinement) -- conservative settings
STAGE2_CFG = {
    "freeze": 0,
    "epochs": 30,
    "imgsz": 640,
    "batch": 8,
    "lr0": 4e-4,
    "lrf": 0.01,
    "patience": 30,
    "deterministic": True,
    "optimizer": "auto",
    "device": 0,
    "val": True,
    "plots": True,
    "verbose": True,
    "workers": 0,
}

# -------------------------
# Helpers
# -------------------------

def find_datasets(root: Path):
    """Return sorted list of dataset directories matching the glob pattern."""
    if not root.exists():
        return []
    return sorted([p for p in root.glob(DATASET_GLOB) if p.is_dir()])


def load_data_yaml(dataset_dir: Path) -> Optional[Path]:
    """Return path to data.yaml inside dataset_dir, or None if absent."""
    yaml_path = dataset_dir / "data.yaml"
    if yaml_path.exists():
        return yaml_path
    alt = dataset_dir / "data" / "data.yaml"
    if alt.exists():
        return alt
    return None


def stage1_feature_adaptation(data_yaml: Path, base_weights: Path, out_dir: Path, run_name: str) -> Optional[Path]:
    """Run Stage 1 (freeze=8) and return path to best.pt or None on failure."""
    print("\n" + "=" * 70)
    print(f"[STAGE 1] Feature Adaptation -> run_name={run_name}")
    print("=" * 70)
    try:
        model = YOLO(str(base_weights))
        model.train(
            data=str(data_yaml),
            epochs=STAGE1_CFG["epochs"],
            imgsz=STAGE1_CFG["imgsz"],
            batch=STAGE1_CFG["batch"],
            freeze=STAGE1_CFG["freeze"],
            lr0=STAGE1_CFG["lr0"],
            lrf=STAGE1_CFG["lrf"],
            optimizer=STAGE1_CFG["optimizer"],
            patience=STAGE1_CFG["patience"],
            deterministic=STAGE1_CFG["deterministic"],
            device=STAGE1_CFG["device"],
            name=run_name,
            project=str(out_dir),
            plots=STAGE1_CFG["plots"],
            verbose=STAGE1_CFG["verbose"],
            val=STAGE1_CFG["val"],
            workers=STAGE1_CFG["workers"],
        )
    except Exception as e:
        print(f"[ERROR] Stage 1 failed: {e}")
        return None

    best_ckpt = out_dir / run_name / "weights" / "best.pt"
    if not best_ckpt.exists():
        print(f"[WARN] Stage 1 completed but checkpoint not found at {best_ckpt}")
        return None
    return best_ckpt


def stage2_fine_tuning(stage1_ckpt: Path, data_yaml: Path, out_dir: Path, run_name: str) -> bool:
    """Run Stage 2 (unfreeze) starting from stage1_ckpt. Returns True on success."""
    print("\n" + "=" * 70)
    print(f"[STAGE 2] Fine-tuning -> base_ckpt={stage1_ckpt.name}")
    print("=" * 70)
    try:
        model = YOLO(str(stage1_ckpt))
        # Use a slightly different run name for stage2 to avoid overwriting stage1 logs
        run_name2 = f"{run_name}_stage2"
        model.train(
            data=str(data_yaml),
            epochs=STAGE2_CFG["epochs"],
            imgsz=STAGE2_CFG["imgsz"],
            batch=STAGE2_CFG["batch"],
            freeze=STAGE2_CFG["freeze"],
            lr0=STAGE2_CFG["lr0"],
            lrf=STAGE2_CFG["lrf"],
            optimizer=STAGE2_CFG["optimizer"],
            patience=STAGE2_CFG["patience"],
            deterministic=STAGE2_CFG["deterministic"],
            device=STAGE2_CFG["device"],
            name=run_name2,
            project=str(out_dir),
            plots=STAGE2_CFG["plots"],
            verbose=STAGE2_CFG["verbose"],
            val=STAGE2_CFG["val"],
            workers=STAGE2_CFG["workers"],
        )
    except Exception as e:
        print(f"[ERROR] Stage 2 failed: {e}")
        return False
    return True


# -------------------------
# Main loop
# -------------------------
def main():
    print("=" * 80)
    print("TL-Optim batch runner for random-k 2025 core-sets (two-stage)")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"Scanning datasets at: {DATASETS_ROOT} for pattern {DATASET_GLOB}")
    print("=" * 80)

    datasets = find_datasets(DATASETS_ROOT)
    if not datasets:
        print(f"No datasets found matching '{DATASET_GLOB}' under {DATASETS_ROOT}. Nothing to do.")
        return 1

    print(f"Found {len(datasets)} datasets:")
    for d in datasets:
        print("  -", d)

    if not BASE_WEIGHTS.exists():
        print(f"ERROR: base weights not found at expected path:\n  {BASE_WEIGHTS}\nPlease adjust BASE_WEIGHTS or add the file. Aborting.")
        return 2

    # deterministic ordering
    for dataset_dir in datasets:
        start = time.time()
        data_yaml = load_data_yaml(dataset_dir)
        if data_yaml is None:
            print(f"[SKIP] {dataset_dir}: data.yaml not found; skipping.")
            continue

        # short label extraction, e.g., k01
        name_parts = dataset_dir.name.split("_")
        short_label = None
        for p in reversed(name_parts):
            if p.startswith("k"):
                short_label = p
                break
        if short_label is None:
            short_label = dataset_dir.name

        out_dir_k = OUT_BASE / short_label
        out_dir_k.mkdir(parents=True, exist_ok=True)

        run_name = f"TL_Optim_2024to2025_random_{short_label}"

        print(f"\n[START] Dataset={dataset_dir.name} short_label={short_label}")
        print(f"run_name={run_name} out_dir={out_dir_k}")

        # Stage 1
        stage1_ckpt = stage1_feature_adaptation(data_yaml, BASE_WEIGHTS, out_dir_k, run_name)
        if stage1_ckpt is None:
            print(f"[FAIL] Stage1 failed or checkpoint missing for {dataset_dir}. Skipping this dataset.")
            continue

        # Stage 2
        ok2 = stage2_fine_tuning(stage1_ckpt, data_yaml, out_dir_k, run_name)
        elapsed = time.time() - start
        if ok2:
            print(f"[DONE] Completed training for {dataset_dir.name} in {elapsed:.1f}s")
        else:
            print(f"[WARN] Stage2 failed for {dataset_dir.name} after Stage1 succeeded (elapsed {elapsed:.1f}s)")

        # small sleep for nicer logging behavior
        time.sleep(0.5)

    print("\n✅ All datasets processed.")
    return 0


if __name__ == '__main__':
    sys.exit(main())

