#!/usr/bin/env python3
# ============================================================
# B3_core_set_finetuning_TL_Optim.py
# ============================================================
# Purpose:
#   TL-Optim style transfer learning:
#   Stage 1: freeze=8  (feature adaptation)
#   Stage 2: freeze=0  (end-to-end fine-tuning)
#
#   Both stages run sequentially on the SAME 2025 core-set
# ============================================================

# ------------------------------------------------------------
# MUST set env vars BEFORE importing torch / ultralytics
# ------------------------------------------------------------
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONWARNINGS"] = "ignore"

from ultralytics import YOLO
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

BASE = Path(r"C:\Users\carmonta\Desktop\data-centric-cross-season-transfer")

# Core-set dataset (same images for both stages)
DATA_2025_CORE = BASE / "datasets" / "train_on_2025_core_set" / "data.yaml"

# Source model (trained on full Season 2024)
BASE_PT_2024 = BASE / "models" / "2024" / "all-ponds" / "weights" / "best.pt"

# Output directory
OUT_DIR = BASE / "models" / "TL_Optim_core_set"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# STAGE 1 — Feature Adaptation (freeze=8)
# ============================================================

def stage1_feature_adaptation():
    print("\n" + "=" * 70)
    print("[STAGE 1] Feature Adaptation (freeze=8)")
    print("=" * 70)

    model = YOLO(str(BASE_PT_2024))

    model.train(
        data=str(DATA_2025_CORE),
        epochs=30,              # short & stable
        imgsz=640,
        batch=4,
        freeze=8,               # TL-Optim key choice
        lr0=1e-3,
        lrf=0.01,
        optimizer="auto",
        patience=20,
        deterministic=True,
        device=0,
        name="TL_Optim_stage1_freeze8",
        project=str(OUT_DIR),
        plots=True,
        verbose=True,
        val=True,
        workers=0
    )

    # Return best checkpoint from Stage 1
    return OUT_DIR / "TL_Optim_stage1_freeze8" / "weights" / "best.pt"


# ============================================================
# STAGE 2 — End-to-End Fine-Tuning (freeze=0)
# ============================================================

def stage2_fine_tuning(stage1_ckpt: Path):
    print("\n" + "=" * 70)
    print("[STAGE 2] End-to-End Fine-Tuning (freeze=0)")
    print("=" * 70)
    print(f"[INFO] Initializing from: {stage1_ckpt}")

    model = YOLO(str(stage1_ckpt))

    model.train(
        data=str(DATA_2025_CORE),   # SAME images
        epochs=20,                  # shorter refinement
        imgsz=640,
        batch=4,
        freeze=0,                   # unfreeze all
        lr0=4e-4,                   # lower LR = safe refinement
        lrf=0.01,
        optimizer="auto",
        patience=15,
        deterministic=True,
        device=0,
        name="TL_Optim_stage2_unfreeze",
        project=str(OUT_DIR),
        plots=True,
        verbose=True,
        val=True,
        workers=0
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    stage1_best = stage1_feature_adaptation()
    stage2_fine_tuning(stage1_best)

    print("\n✅ TL-Optim core-set training completed successfully!")
