#!/usr/bin/env python3
"""
run_k_sensitivity_experiment.py

Clean fixed-budget sensitivity experiment over K for geometry-aware adaptive selection.

For each K in [50, 100, 200, 400, 800], this script:
1) Runs selection using:
   - uncertainty = 1 - confidence
   - geometry features: length, orientation angle (atan2), bend angle
   - morphology deviation: abs(length - mean_length)
   - fused embedding + geometry space (PCA + z-score)
2) Applies fixed-budget strategy:
   - 40% top uncertainty
   - 40% top morphology deviation
   - 20% FPS diversity on fused space
3) Creates dataset:
   - dataset/images/train
   - dataset/labels/train
   - dataset/data.yaml
4) Trains YOLO with existing staged configuration.
5) Evaluates and saves metrics into results.json.

Outputs are stored ONLY under:
  k_sensitivity_experiment/
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path
import json
import csv
import random
import math
import shutil

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from sklearn.decomposition import PCA


# ---------------- Configuration ----------------
SEED = 0
K_VALUES = [50, 100, 200, 400, 800]
PCA_DIM = 64
ALPHA = 1.0

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = PROJECT_ROOT / "k_sensitivity_experiment"

# Reusing paths from the current adaptive pipeline (2024 -> 2025 setting)
SOURCE_MODEL = PROJECT_ROOT / "models" / "2024" / "all-ponds" / "weights" / "best.pt"
EMBED_META = PROJECT_ROOT / "scripts" / "representation_analysis" / "outputs_repreasentation" / "rep_analysis" / "embeddings_meta.csv"
EMBED_VEC = PROJECT_ROOT / "scripts" / "representation_analysis" / "outputs_repreasentation" / "rep_analysis" / "embeddings_vectors.npy"
TARGET_DATASET_ROOT = PROJECT_ROOT / "datasets" / "train_on_2025_all"
IMAGE_DIRS = [TARGET_DATASET_ROOT / "images", TARGET_DATASET_ROOT / "val" / "images"]
LABEL_DIRS = [TARGET_DATASET_ROOT / "labels", TARGET_DATASET_ROOT / "val" / "labels"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def set_deterministic(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)


def fps(points: np.ndarray, k: int) -> list[int]:
    if points.shape[0] == 0 or k <= 0:
        return []
    if k >= points.shape[0]:
        return list(range(points.shape[0]))

    centroid = points.mean(axis=0)
    dists = np.linalg.norm(points - centroid, axis=1)
    selected = [int(np.argmax(dists))]
    min_d = np.linalg.norm(points - points[selected[0]], axis=1)

    while len(selected) < k:
        idx = int(np.argmax(min_d))
        selected.append(idx)
        d = np.linalg.norm(points - points[idx], axis=1)
        min_d = np.minimum(min_d, d)
    return selected


def compute_bend_angle(kpts: np.ndarray) -> float:
    # kpt indices: 0 carapace, 2 rostrum, 3 tail
    car = kpts[0]
    ros = kpts[2]
    tail = kpts[3]
    v1 = ros - car
    v2 = tail - car
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 <= 1e-12 or n2 <= 1e-12:
        return 0.0
    cosv = np.dot(v1, v2) / (n1 * n2)
    cosv = np.clip(cosv, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))


def collect_images() -> list[Path]:
    out = []
    seen = set()
    for d in IMAGE_DIRS:
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                rp = p.resolve()
                if str(rp) in seen:
                    continue
                seen.add(str(rp))
                out.append(rp)
    return out


def load_embeddings(meta_csv: Path, vectors_npy: Path) -> tuple[pd.DataFrame, np.ndarray]:
    meta = pd.read_csv(meta_csv)
    vectors = np.load(vectors_npy)
    if len(meta) != vectors.shape[0]:
        m = min(len(meta), vectors.shape[0])
        meta = meta.iloc[:m].reset_index(drop=True)
        vectors = vectors[:m]

    cols = [str(c).strip() for c in meta.columns]
    meta.columns = cols
    if "image_path" not in meta.columns:
        meta = meta.rename(columns={meta.columns[0]: "image_path"})
    meta["basename"] = meta["image_path"].astype(str).apply(lambda p: Path(str(p).replace("\\", "/")).name)
    return meta, vectors


def run_error_analysis(model_path: Path, image_paths: list[Path]) -> list[dict]:
    """Run memory-safe inference for uncertainty + geometry extraction.

    Uses per-image prediction to avoid large CUDA allocations, and falls back to CPU
    for individual images if a CUDA OOM happens.
    """
    model = YOLO(str(model_path))
    device = 0 if torch.cuda.is_available() else "cpu"

    data = []
    total = len(image_paths)
    for i, img_path in enumerate(image_paths, start=1):
        if i == 1 or i % 100 == 0 or i == total:
            print(f"Inference progress: {i}/{total}")

        try:
            results = model.predict(
                source=str(img_path),
                imgsz=640,
                device=device,
                verbose=False,
                conf=0.001,
            )
        except torch.OutOfMemoryError:
            # Per-image fallback to CPU when GPU memory is tight.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                results = model.predict(
                    source=str(img_path),
                    imgsz=640,
                    device="cpu",
                    verbose=False,
                    conf=0.001,
                )
            except Exception:
                continue
        except Exception:
            continue

        if not results:
            continue
        r = results[0]
        try:
            if r.keypoints is None or r.keypoints.xy is None:
                continue
            kpts = r.keypoints.xy.cpu().numpy()
            confs = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None
            if len(kpts) == 0:
                continue
            k = kpts[0]
            if k.shape[0] < 4:
                continue
            if confs is None or len(confs) == 0:
                uncertainty = 1.0
            else:
                c = confs[0]
                uncertainty = float(1.0 - np.mean(c))
            length = float(np.linalg.norm(k[2] - k[0]))
            vec = k[2] - k[0]
            angle = float(np.arctan2(vec[1], vec[0]))
            bend = compute_bend_angle(k)
            data.append(
                {
                    "image_path": str(Path(r.path).resolve()),
                    "image": Path(r.path).name,
                    "uncertainty": uncertainty,
                    "length": length,
                    "angle": angle,
                    "bend": bend,
                }
            )
        except Exception:
            continue
    return data


def select_fixed_budget(analysis: list[dict], fused: np.ndarray, budget_k: int) -> list[int]:
    n = len(analysis)
    if n == 0:
        return []
    budget_k = min(int(budget_k), n)
    if budget_k <= 0:
        return []

    uncertainties = np.array([d["uncertainty"] for d in analysis], dtype=np.float64)
    lengths = np.array([d["length"] for d in analysis], dtype=np.float64)
    morpho_dev = np.abs(lengths - lengths.mean())

    k_unc = int(math.floor(0.4 * budget_k))
    k_morph = int(math.floor(0.4 * budget_k))
    k_div = budget_k - k_unc - k_morph

    rank_unc = np.argsort(-uncertainties).tolist()
    rank_morph = np.argsort(-morpho_dev).tolist()
    rank_div = fps(fused, k_div if k_div > 0 else 0)

    # Primary picks by quota
    candidates = rank_unc[:k_unc] + rank_morph[:k_morph] + rank_div

    # Unique keep-order
    selected = []
    seen = set()
    for idx in candidates:
        if idx not in seen:
            selected.append(idx)
            seen.add(idx)
        if len(selected) >= budget_k:
            return selected[:budget_k]

    # Refill deterministically to exact K
    refill_order = rank_unc + rank_morph + fps(fused, n)
    for idx in refill_order:
        if idx in seen:
            continue
        selected.append(idx)
        seen.add(idx)
        if len(selected) >= budget_k:
            break
    return selected[:budget_k]


def find_label_for_image(image_path: Path) -> Path | None:
    stem = image_path.stem
    for lbl_root in LABEL_DIRS:
        if not lbl_root.exists():
            continue
        direct = lbl_root / f"{stem}.txt"
        if direct.exists():
            return direct.resolve()
        for p in lbl_root.rglob("*.txt"):
            if p.stem == stem:
                return p.resolve()
    return None


def build_dataset(selected_paths: list[Path], dataset_root: Path) -> tuple[int, int]:
    images_train = dataset_root / "images" / "train"
    labels_train = dataset_root / "labels" / "train"
    images_train.mkdir(parents=True, exist_ok=True)
    labels_train.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing_labels = 0
    for img in selected_paths:
        lbl = find_label_for_image(img)
        if lbl is None or not lbl.exists():
            missing_labels += 1
            continue
        try:
            shutil.copy2(img, images_train / img.name)
            shutil.copy2(lbl, labels_train / lbl.name)
            copied += 1
        except Exception:
            continue

    yaml_content = f"""path: {dataset_root.resolve()}
train: images/train
val: images/train
nc: 1
names: ['prawn']
kpt_shape: [4,3]
flip_idx: [0,1,2,3]
"""
    (dataset_root / "data.yaml").write_text(yaml_content.strip() + "\n", encoding="utf-8")
    return copied, missing_labels


def train_model(base_weights: Path, yaml_path: Path, out_dir: Path) -> Path:
    device = 0 if torch.cuda.is_available() else "cpu"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _get_save_dir(train_result):
        if isinstance(train_result, dict) and "save_dir" in train_result:
            return Path(train_result["save_dir"])
        if hasattr(train_result, "save_dir"):
            return Path(train_result.save_dir)
        raise RuntimeError("Could not determine save_dir from YOLO.train() results.")

    model1 = YOLO(str(base_weights))
    results1 = model1.train(
        data=str(yaml_path),
        epochs=30,
        freeze=20,
        imgsz=640,
        batch=6,
        lr0=5e-5,
        mosaic=0.0,
        amp=False,
        workers=4,
        device=device,
        project=str(out_dir),
        name="stage1",
        exist_ok=True,
        verbose=False,
    )
    stage1_dir = _get_save_dir(results1)
    stage1_weights = stage1_dir / "weights" / "last.pt"
    if not stage1_weights.exists():
        alt = stage1_dir / "weights" / "best.pt"
        if alt.exists():
            stage1_weights = alt
        else:
            raise FileNotFoundError(f"Stage 1 weights not found in {stage1_dir / 'weights'}")

    model2 = YOLO(str(stage1_weights))
    results2 = model2.train(
        data=str(yaml_path),
        epochs=40,
        freeze=15,
        imgsz=640,
        batch=6,
        lr0=2e-5,
        mosaic=0.0,
        amp=False,
        workers=4,
        device=device,
        project=str(out_dir),
        name="stage2",
        exist_ok=True,
        verbose=False,
    )
    stage2_dir = _get_save_dir(results2)
    stage2_weights = stage2_dir / "weights" / "last.pt"
    if not stage2_weights.exists():
        alt = stage2_dir / "weights" / "best.pt"
        if alt.exists():
            stage2_weights = alt
        else:
            raise FileNotFoundError(f"Stage 2 weights not found in {stage2_dir / 'weights'}")

    model3 = YOLO(str(stage2_weights))
    results3 = model3.train(
        data=str(yaml_path),
        epochs=10,
        freeze=0,
        imgsz=640,
        batch=6,
        lr0=5e-6,
        cos_lr=True,
        mosaic=0.0,
        amp=False,
        workers=4,
        device=device,
        project=str(out_dir),
        name="stage3",
        exist_ok=True,
        verbose=False,
    )
    final_dir = _get_save_dir(results3)
    final_best = final_dir / "weights" / "best.pt"
    if not final_best.exists():
        alt = final_dir / "weights" / "last.pt"
        if alt.exists():
            final_best = alt
        else:
            raise FileNotFoundError(f"Final weights not found in {final_dir / 'weights'}")

    stable_best = out_dir / "best.pt"
    shutil.copy2(final_best, stable_best)
    return stable_best


def evaluate_model(model_path: Path, yaml_path: Path) -> dict:
    model = YOLO(str(model_path))
    val_res = model.val(data=str(yaml_path), verbose=False)
    metrics = {
        "mAP50": float(getattr(val_res.box, "map50", float("nan"))),
        "mAP50_95": float(getattr(val_res.box, "map", float("nan"))),
        "precision": float(getattr(val_res.box, "mp", float("nan"))),
        "recall": float(getattr(val_res.box, "mr", float("nan"))),
    }
    return metrics


def run_single_k(
    budget_k: int,
    analysis: list[dict],
    yolo_reduced: np.ndarray,
    experiment_root: Path,
) -> dict | None:
    k_dir = experiment_root / f"K_{budget_k}"
    if k_dir.exists():
        print(f"[K={budget_k}] Folder exists, skipping to avoid overwrite: {k_dir}")
        return None

    print(f"\n========== Running K={budget_k} ==========")
    (k_dir / "dataset").mkdir(parents=True, exist_ok=False)
    (k_dir / "model").mkdir(parents=True, exist_ok=False)

    lengths = np.array([d["length"] for d in analysis], dtype=np.float64)
    angles = np.array([d["angle"] for d in analysis], dtype=np.float64)
    bends = np.array([d["bend"] for d in analysis], dtype=np.float64)
    geo_block = np.stack([lengths, angles, bends], axis=1)
    fused = np.concatenate([zscore(yolo_reduced), ALPHA * zscore(geo_block)], axis=1)

    selected_idx = select_fixed_budget(analysis, fused, budget_k)
    selected_paths = [Path(analysis[i]["image_path"]) for i in selected_idx]
    print(f"[K={budget_k}] Requested={budget_k}, selected before label check={len(selected_paths)}")

    copied, missing_labels = build_dataset(selected_paths, k_dir / "dataset")
    if copied == 0:
        print(f"[K={budget_k}] No samples copied (all labels missing). Skipping training.")
        results = {
            "K": int(budget_k),
            "num_images": 0,
            "missing_labels": int(missing_labels),
            "mAP50": float("nan"),
            "mAP50_95": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
        }
        (k_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        return results

    yaml_path = k_dir / "dataset" / "data.yaml"
    best_weights = train_model(SOURCE_MODEL, yaml_path, k_dir / "model")
    eval_metrics = evaluate_model(best_weights, yaml_path)

    results = {
        "K": int(budget_k),
        "num_images": int(copied),
        "missing_labels": int(missing_labels),
        "mAP50": eval_metrics["mAP50"],
        "mAP50_95": eval_metrics["mAP50_95"],
        "precision": eval_metrics["precision"],
        "recall": eval_metrics["recall"],
    }
    (k_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[K={budget_k}] Done. num_images={copied}, mAP50={results['mAP50']:.4f}")
    return results


def write_summary(summary_rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["K", "num_images", "mAP50", "mAP50_95", "precision", "recall"],
        )
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(
                {
                    "K": r["K"],
                    "num_images": r["num_images"],
                    "mAP50": r["mAP50"],
                    "mAP50_95": r["mAP50_95"],
                    "precision": r["precision"],
                    "recall": r["recall"],
                }
            )


def main() -> None:
    set_deterministic(SEED)
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)

    if not SOURCE_MODEL.exists():
        raise FileNotFoundError(f"Missing source model: {SOURCE_MODEL}")
    if not EMBED_META.exists() or not EMBED_VEC.exists():
        raise FileNotFoundError(f"Missing embeddings: {EMBED_META} / {EMBED_VEC}")

    image_paths = collect_images()
    if len(image_paths) == 0:
        raise RuntimeError("No target images found.")
    print(f"Found {len(image_paths)} candidate images.")

    print("Running YOLO inference for uncertainty + geometry...")
    analysis = run_error_analysis(SOURCE_MODEL, image_paths)
    if len(analysis) == 0:
        raise RuntimeError("No valid inference outputs with keypoints found.")
    print(f"Valid inference records: {len(analysis)}")

    print("Loading and aligning embeddings...")
    meta, vectors = load_embeddings(EMBED_META, EMBED_VEC)
    emb_map = {}
    for i, b in enumerate(meta["basename"].astype(str).tolist()):
        if b not in emb_map:
            emb_map[b] = i

    filtered = []
    yolo_feats = []
    for d in analysis:
        idx = emb_map.get(d["image"])
        if idx is None:
            continue
        filtered.append(d)
        yolo_feats.append(vectors[idx])
    analysis = filtered
    if len(analysis) == 0:
        raise RuntimeError("No overlap between inference samples and embeddings.")

    yolo_feats = np.asarray(yolo_feats, dtype=np.float64)
    pca_dim = min(PCA_DIM, yolo_feats.shape[1], yolo_feats.shape[0])
    pca = PCA(n_components=pca_dim, random_state=SEED)
    yolo_reduced = pca.fit_transform(yolo_feats)
    print(f"Fused base prepared. Records after alignment: {len(analysis)}")
    print(f"PCA dim={pca_dim}, explained variance sum={np.sum(pca.explained_variance_ratio_):.4f}")

    summary_rows = []
    for k in K_VALUES:
        res = run_single_k(k, analysis, yolo_reduced, EXPERIMENT_ROOT)
        if res is not None:
            summary_rows.append(res)

    write_summary(summary_rows, EXPERIMENT_ROOT / "summary.csv")
    print(f"\nSummary saved to: {EXPERIMENT_ROOT / 'summary.csv'}")
    print("Experiment complete.")


if __name__ == "__main__":
    main()
