# ============================================================
# scripts/tl/run_B1_feature_extraction.py
# ============================================================
# Full grid: directions × freeze_schemes × fractions (k)
# B1 = Feature Extraction (freeze some backbone)
# ============================================================

from pathlib import Path
import shutil
import random
import numpy as np
import torch
from ultralytics import YOLO

# ---------------------- Global Config ----------------------
SEED = 0
EPOCHS   = 40
PATIENCE = 10
IMGSZ    = 640
BATCH    = 8
LR0      = 0.01
LRF      = 0.01
USE_COSINE = True
DEVICE     = "cpu"      # set to "0" or "cuda:0" if you have GPU
OPTIMIZER  = "AdamW"

# ---------------------- Experimental Grid ----------------------
FREEZE_SCHEMES = [10, 8, 4, 0]
FRACTIONS = [1.0, 0.5, 0.25]


# Define both directions
DIRECTIONS = [
    dict(
        src="models/2024/all-ponds/weights/best.pt",
        data="datasets/prawn_2025_circ_small_v1/data.yaml",
        prefix="B1_TL_2024to2025",
        out="models/TF/tf_grids/B1_final/from_2024_to_2025"
    ),
    dict(
        src="models/2025/YOLOv11n_train_on_2025_all_pose_300ep_best.pt",
        data="datasets/2024_smallDS_circulars/data.yaml",
        prefix="B1_TL_2025to2024",
        out="models/TF/tf_grids/B1_final/from_2025_to_2024"
    ),
]

# ============================================================
# Utils
# ============================================================
def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    try:
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def freeze_heads_only(model: YOLO):
    """Freeze everything except the pose head."""
    for _, p in model.model.named_parameters():
        p.requires_grad = False
    total = sum(1 for _ in model.model.parameters())
    released = 0

    if hasattr(model.model, "model"):
        blocks = getattr(model.model, "model")
        try:
            last = blocks[-1]
            for p in last.parameters():
                p.requires_grad = True
                released += 1
            print(f"[FREEZE] Released last module (assumed head). trainable={released}/{total}")
            return
        except Exception:
            pass

    head_terms = ("head", "detect", "pose", "kpt", "keypoint")
    for n, p in model.model.named_parameters():
        if any(t in n.lower() for t in head_terms):
            p.requires_grad = True
            released += 1
    print(f"[FREEZE] Released head params by name. trainable={released}/{total}")


def freeze_first_N_layers(model: YOLO, n_layers: int):
    """Freeze first n blocks (approx.)"""
    for _, p in model.model.named_parameters():
        p.requires_grad = False

    released = 0
    if hasattr(model.model, "model"):
        blocks = getattr(model.model, "model")
        try:
            for idx, m in enumerate(blocks):
                if idx >= n_layers:
                    for p in m.parameters():
                        p.requires_grad = True
                        released += 1
            print(f"[FREEZE] First {n_layers} blocks frozen. Released={released}")
            return
        except Exception:
            pass

    print(f"[WARN] Fallback freeze for {n_layers} layers used.")


def copy_best(src_dir: Path, dst_flat: Path):
    best_src = src_dir / "weights" / "best.pt"
    if best_src.exists():
        shutil.copy2(best_src, dst_flat)
        print(f"[INFO] Copied weights → {dst_flat}")
    else:
        print(f"[WARN] No best.pt found in {src_dir}")


# ============================================================
# Core run
# ============================================================
def run_one(direction, scheme, frac):
    set_seed(SEED)
    PROJ = Path(__file__).resolve().parents[2]
    src_w = PROJ / direction["src"]
    data_yaml = PROJ / direction["data"]
    out_root = PROJ / direction["out"]
    run_prefix = direction["prefix"]

    assert src_w.exists(), f"Missing source weights: {src_w}"
    assert data_yaml.exists(), f"Missing data.yaml: {data_yaml}"
    out_root.mkdir(parents=True, exist_ok=True)

    tag = str(scheme).replace(" ", "").replace("-", "")
    frac_tag = f"k{int(frac * 100)}pct"
    run_name = f"{run_prefix}_{frac_tag}_freeze{tag}"

    print(f"\n[INFO] ==== B1 run: {run_name} ====")
    print(f"[INFO] SRC: {src_w}")
    print(f"[INFO] DATA: {data_yaml}")
    print(f"[INFO] FREEZE={scheme} | FRACTION={frac}")

    model = YOLO(str(src_w))

    # --- Apply freezing ---
    if scheme == "heads-only":
        freeze_heads_only(model)
    elif isinstance(scheme, int):
        if scheme <= 0:
            for _, p in model.model.named_parameters():
                p.requires_grad = True
            print("[FREEZE] Full fine-tuning (no freeze)")
        else:
            freeze_first_N_layers(model, scheme)

    # --- Train ---
    model.train(
        data=str(data_yaml),
        imgsz=IMGSZ,
        batch=BATCH,
        epochs=EPOCHS,
        patience=PATIENCE,
        lr0=LR0,
        lrf=LRF,
        cos_lr=USE_COSINE,
        seed=SEED,
        device=DEVICE,
        project=str(out_root),
        name=run_name,
        workers=2,
        verbose=True,
        optimizer=OPTIMIZER,
        fraction=frac,
    )

    # --- Save best.pt flat ---
    copy_best(out_root / run_name, out_root / f"{run_name}_best.pt")


# ============================================================
# Main loop
# ============================================================
def main():
    total_runs = len(DIRECTIONS) * len(FREEZE_SCHEMES) * len(FRACTIONS)
    print(f"[INFO] Starting full B1 grid ({total_runs} runs total)")
    for direction in DIRECTIONS:
        for frac in FRACTIONS:
            for scheme in FREEZE_SCHEMES:
                run_one(direction, scheme, frac)
    print("\n[INFO] B1 grid finished. You can now run evaluation scripts for comparison.")


if __name__ == "__main__":
    main()
