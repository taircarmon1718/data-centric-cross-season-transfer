#!/usr/bin/env python3
# ============================================================
# run_B3_final_training.py
# ============================================================
# Purpose:
#   Train final B3 models (production-level) using full datasets.
#   Based on best configurations from B1–B2 (freeze=4).
#
# Notes:
#   - Compatible with Windows multiprocessing
#   - Includes both transfer directions:
#       (1) 2024 → 2025  (train on 2025)
#       (2) 2025 → 2024  (train on 2024)
# ============================================================

from ultralytics import YOLO
from pathlib import Path
import os

# ============================================================
# USER PATHS
# ============================================================

BASE = Path(r"C:\Users\carmonta\Desktop\Uni-Projects\prawn-size-project")

# Datasets
DATA_2025 = BASE / "datasets" / "train_on_2025_all" / "data.yaml"
DATA_2024 = BASE / "datasets" / "train_on_all" / "data.yaml"

# Best weights from B2 experiments
BASE_PT_2024to2025 = BASE / r"models\TF\tf_grids\B2\from_2024to2025\B2_TL_2024to2025_k100pct_freeze4_S2_best.pt"
BASE_PT_2025to2024 = BASE / r"models\TF\tf_grids\B2\from_2025to2024\B2_TL_2025to2024_k100pct_freeze4_S2_best.pt"

# Output directory for final models
OUT_DIR = BASE / "models" / "B3_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# TRAIN FUNCTION
# ============================================================

def train_model(name, base_pt, data_yaml, freeze_layers, epochs, project_dir):
    print("=" * 60)
    print(f"[INFO] Starting training for: {name}")
    print(f"[INFO] Using weights: {base_pt}")
    print(f"[INFO] Freeze layers: {freeze_layers}")
    print(f"[INFO] Data: {data_yaml}")
    print("=" * 60)

    model = YOLO(str(base_pt))

    # Training arguments
    args = dict(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=640,
        batch=16,
        freeze=freeze_layers,
        lr0=0.01,
        lrf=0.01,
        patience=100,
        deterministic=True,
        optimizer="auto",
        device=0,
        name=name,
        project=str(project_dir),
        plots=True,
        verbose=True,
        val=True
    )

    model.train(**args)


# ============================================================
# MAIN (safe multiprocessing for Windows)
# ============================================================

if __name__ == "__main__":
    os.environ["PYTHONWARNINGS"] = "ignore"

    # ---------- (1) 2024 → 2025 ----------
    train_model(
        name="B3_2024to2025_final_freeze4_full",
        base_pt=BASE_PT_2024to2025,
        data_yaml=DATA_2025,
        freeze_layers=4,
        epochs=150,
        project_dir=OUT_DIR
    )

    # ---------- (2) 2025 → 2024 ----------
    train_model(
        name="B3_2025to2024_final_freeze4_full",
        base_pt=BASE_PT_2025to2024,
        data_yaml=DATA_2024,
        freeze_layers=4,
        epochs=150,
        project_dir=OUT_DIR
    )

    print("\n✅ B3 Final Training Completed Successfully!")
