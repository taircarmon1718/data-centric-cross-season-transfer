#!/usr/bin/env python3
# ============================================================
# run_B2_staged_finetune_GRID.py
# ============================================================
# Runs full grid of Staged Fine-Tuning experiments:
#   - Progressive unfreezing (Stage-1 partial → Stage-2 full)
#   - Same source-season checkpoints as B1
#   - Fixed schedule lengths across k (data fractions)
# ============================================================

from pathlib import Path
import os, random, shutil
import numpy as np
import torch
from ultralytics import YOLO

# ---------------------------
# Global config
# ---------------------------
SEED = 0
DEVICE = "cuda"
OPTIMIZER = "AdamW"
IMGSZ = 640
BATCH = 8
USE_COSINE = True
PATIENCE = 10

EPOCHS_STAGE1 = 20
EPOCHS_STAGE2 = 20

# Learning rates (Stage 2 smaller)
ETA1_LR0, ETA1_LRF = 0.010, 0.010
ETA2_LR0, ETA2_LRF = 0.003, 0.003

FREEZE_DEPTHS = [10, 8, 4]
FRACTIONS = [1.0, 0.5, 0.25]
DIRECTIONS = ["2024to2025", "2025to2024"]

# ---------------------------
# Paths
# ---------------------------
HERE = Path(__file__).resolve().parent
PROJ = HERE.parent.parent
OUT_BASE = PROJ / "models/TF/tf_grids/B2"
OUT_BASE.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("WANDB_MODE", "disabled")


# ---------------------------
# Utils
# ---------------------------
def set_seed(s=0):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def freeze_first_N_blocks(model: YOLO, n_blocks: int):
    for p in model.model.parameters():
        p.requires_grad = False
    released = 0
    if hasattr(model.model, "model"):
        blocks = getattr(model.model, "model")
        for idx, m in enumerate(blocks):
            if idx >= n_blocks:
                for p in m.parameters():
                    p.requires_grad = True
                    released += 1
    print(f"[FREEZE] Stage-1: first {n_blocks} blocks frozen; released_params={released}")


def unfreeze_all(model: YOLO):
    for p in model.model.parameters():
        p.requires_grad = True
    print("[FREEZE] Stage-2: full fine-tuning (freeze=0)")


def copy_best(src_dir: Path, dst_path: Path):
    best = src_dir / "weights" / "best.pt"
    last = src_dir / "weights" / "last.pt"
    src = best if best.exists() else last
    if not src or not src.exists():
        print(f"[ERROR] No weights found under {src_dir}")
        return None
    shutil.copy2(src, dst_path)
    return dst_path


# ---------------------------
# Stage training routines
# ---------------------------
def train_stage1(model, data_yaml, out_root, run_name, freeze_depth, fraction):
    print(f"\n[INFO] ==== Stage-1 ({run_name}) freeze={freeze_depth} fraction={fraction} ====")
    freeze_first_N_blocks(model, freeze_depth)
    model.train(
        data=str(data_yaml),
        imgsz=IMGSZ,
        batch=BATCH,
        epochs=EPOCHS_STAGE1,
        patience=PATIENCE,
        lr0=ETA1_LR0,
        lrf=ETA1_LRF,
        cos_lr=USE_COSINE,
        seed=SEED,
        device=DEVICE,
        project=str(out_root),
        name=run_name + "_S1",
        workers=2,
        verbose=True,
        optimizer=OPTIMIZER,
        fraction=fraction,
    )
    return out_root / (run_name + "_S1")


def train_stage2(init_weights, data_yaml, out_root, run_name, fraction):
    print(f"\n[INFO] ==== Stage-2 ({run_name}) fine-tune full ====")
    model2 = YOLO(str(init_weights))
    unfreeze_all(model2)
    model2.train(
        data=str(data_yaml),
        imgsz=IMGSZ,
        batch=BATCH,
        epochs=EPOCHS_STAGE2,
        patience=PATIENCE,
        lr0=ETA2_LR0,
        lrf=ETA2_LRF,
        cos_lr=USE_COSINE,
        seed=SEED,
        device=DEVICE,
        project=str(out_root),
        name=run_name + "_S2",
        workers=2,
        verbose=True,
        optimizer=OPTIMIZER,
        fraction=fraction,
    )
    return out_root / (run_name + "_S2")


# ---------------------------
# Main
# ---------------------------
def run_B2_experiment(direction):
    if direction == "2024to2025":
        src = PROJ / "models/2024/all-ponds/weights/best.pt"
        data_yaml = PROJ / "datasets/prawn_2025_circ_small_v1/data.yaml"
    elif direction == "2025to2024":
        src = PROJ / "models/2025/YOLOv11n_train_on_2025_all_pose_300ep_best.pt"
        data_yaml = PROJ / "datasets/2024_smallDS_circulars/data.yaml"
    else:
        raise ValueError(direction)

    out_root = OUT_BASE / f"from_{direction}"
    out_root.mkdir(parents=True, exist_ok=True)

    for freeze_d in FREEZE_DEPTHS:
        for frac in FRACTIONS:
            run_prefix = f"B2_TL_{direction}_k{int(frac*100)}pct_freeze{freeze_d}"
            print("=" * 70)
            print(f"[RUN] {run_prefix}")
            print("=" * 70)

            model = YOLO(str(src))

            # Stage 1
            s1_dir = train_stage1(model, data_yaml, out_root, run_prefix, freeze_d, frac)
            s1_best = copy_best(s1_dir, out_root / f"{run_prefix}_S1_best.pt")
            if not s1_best:
                continue

            # Stage 2
            s2_dir = train_stage2(s1_best, data_yaml, out_root, run_prefix, frac)
            _ = copy_best(s2_dir, out_root / f"{run_prefix}_S2_best.pt")


def main():
    set_seed(SEED)
    for direction in DIRECTIONS:
        run_B2_experiment(direction)


if __name__ == "__main__":
    main()
