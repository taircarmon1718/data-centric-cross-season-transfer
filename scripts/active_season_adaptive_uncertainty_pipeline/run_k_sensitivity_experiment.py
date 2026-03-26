#!/usr/bin/env python3
"""
K-sensitivity version of run_adaptive_shift_pipeline.py.

Behavior matches adaptive pipeline for:
- feature construction
- embedding fusion
- training schedule
- dataset/YAML generation
- embedding-based filtering

Only intentional change:
- runs for multiple K values
- fixes selection union bug with deterministic exact-K selection

Outputs:
adaptive_k_experiment/
  2024_to_2025/K_*/{dataset,model}
  2025_to_2024/K_*/{dataset,model}
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import json
import csv
import random
import shutil

import numpy as np
import torch
from ultralytics import YOLO
from sklearn.decomposition import PCA


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = PROJECT_ROOT / "datasets"
EXPERIMENT_ROOT = PROJECT_ROOT / "adaptive_k_experiment"
EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)

SEED = 0
K_VALUES = [50, 100, 200, 400, 800]
PCA_DIM = 64
ALPHA = 1.0


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def zscore(x):
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)


def fps(points, k):
    if len(points) == 0:
        return []
    centroid = points.mean(axis=0)
    dists = np.linalg.norm(points - centroid, axis=1)
    selected = [int(np.argmax(dists))]
    min_d = np.linalg.norm(points - points[selected[0]], axis=1)
    while len(selected) < min(k, len(points)):
        idx = int(np.argmax(min_d))
        selected.append(idx)
        d = np.linalg.norm(points - points[idx], axis=1)
        min_d = np.minimum(min_d, d)
    return selected


def compute_bend_angle(k):
    car = k[0]
    ros = k[2]
    tail = k[3]
    v1 = ros - car
    v2 = tail - car
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cosv = np.dot(v1, v2) / (n1 * n2)
    cosv = np.clip(cosv, -1.0, 1.0)
    return np.degrees(np.arccos(cosv))


def load_embeddings(meta_csv, vectors_npy):
    meta = np.genfromtxt(meta_csv, delimiter=",", dtype=str, skip_header=1)
    image_names = meta[:, 0]
    vectors = np.load(vectors_npy)
    return image_names, vectors


def run_error_analysis(model_path, image_dir):
    model = YOLO(str(model_path))
    device = 0 if torch.cuda.is_available() else "cpu"
    results = model.predict(
        source=str(image_dir),
        imgsz=640,
        device=device,
        verbose=False,
        conf=0.001,
    )
    data = []
    for r in results:
        if r.keypoints is None:
            continue
        kpts = r.keypoints.xy.cpu().numpy()
        confs = r.keypoints.conf.cpu().numpy()
        if len(kpts) == 0:
            continue
        k = kpts[0]
        c = confs[0]
        uncertainty = 1.0 - np.mean(c)
        length = np.linalg.norm(k[2] - k[0])
        vec = k[2] - k[0]
        angle = np.arctan2(vec[1], vec[0])
        bend = compute_bend_angle(k)
        data.append(
            {
                "image": Path(r.path).name,
                "uncertainty": uncertainty,
                "length": length,
                "angle": angle,
                "bend": bend,
            }
        )
    return data


def select_exact_k(analysis, fused, k):
    k = min(int(k), len(analysis))
    uncertainties = np.array([d["uncertainty"] for d in analysis])
    lengths = np.array([d["length"] for d in analysis])
    morpho_dev = np.abs(lengths - np.mean(lengths))
    k_unc = int(k * 0.4)
    k_morph = int(k * 0.4)
    k_div = k - k_unc - k_morph
    idx_unc = np.argsort(-uncertainties)[:k_unc].tolist()
    idx_morph = np.argsort(-morpho_dev)[:k_morph].tolist()
    idx_div = fps(fused, k_div)
    ordered = idx_unc + idx_morph + idx_div
    selected = []
    seen = set()
    for idx in ordered:
        if idx in seen:
            continue
        selected.append(idx)
        seen.add(idx)
        if len(selected) >= k:
            return selected
    refill = np.argsort(-uncertainties).tolist() + np.argsort(-morpho_dev).tolist() + fps(fused, len(analysis))
    for idx in refill:
        if idx in seen:
            continue
        selected.append(idx)
        seen.add(idx)
        if len(selected) >= k:
            break
    return selected[:k]


def build_dataset(selected_names, image_dir, dataset_root, out_dataset_dir):
    if out_dataset_dir.exists():
        shutil.rmtree(out_dataset_dir)
    (out_dataset_dir / "images/train").mkdir(parents=True)
    (out_dataset_dir / "labels/train").mkdir(parents=True)
    copied = 0
    missing = 0
    for img_name in selected_names:
        stem = Path(img_name).stem
        copied_img = False
        for img in image_dir.rglob(img_name):
            shutil.copy2(img, out_dataset_dir / "images/train" / img.name)
            copied_img = True
            break
        copied_lbl = False
        for lbl in (dataset_root / "labels").rglob(stem + ".txt"):
            shutil.copy2(lbl, out_dataset_dir / "labels/train" / lbl.name)
            copied_lbl = True
            break
        if copied_img and copied_lbl:
            copied += 1
        else:
            missing += 1
    yaml_path = out_dataset_dir / "data.yaml"
    yaml_content = f"""path: {out_dataset_dir.resolve()}
train: images/train
val: images/train
nc: 1
names: ['prawn']
kpt_shape: [4,3]
flip_idx: [0,1,2,3]
"""
    yaml_path.write_text(yaml_content.strip())
    return yaml_path, copied, missing


def train_model(base_weights, yaml_path, out_dir):
    device = 0 if torch.cuda.is_available() else "cpu"

    model1 = YOLO(str(base_weights))
    r1 = model1.train(
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
        project=str(out_dir.parent),
        name=out_dir.name,
        exist_ok=True,
        verbose=False,
    )
    stage1_dir = Path(r1.save_dir)
    stage1 = stage1_dir / "weights" / "last.pt"
    if not stage1.exists():
        stage1 = stage1_dir / "weights" / "best.pt"

    model2 = YOLO(str(stage1))
    r2 = model2.train(
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
        project=str(out_dir.parent),
        name=out_dir.name,
        exist_ok=True,
        verbose=False,
    )
    stage2_dir = Path(r2.save_dir)
    stage2 = stage2_dir / "weights" / "last.pt"
    if not stage2.exists():
        stage2 = stage2_dir / "weights" / "best.pt"

    model3 = YOLO(str(stage2))
    r3 = model3.train(
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
        project=str(out_dir.parent),
        name=out_dir.name,
        exist_ok=True,
        verbose=False,
    )
    final_dir = Path(r3.save_dir)
    best = final_dir / "weights" / "best.pt"
    if not best.exists():
        best = final_dir / "weights" / "last.pt"
    return best


def run_direction(source_season, target_season, source_model, embed_meta_csv, embed_vectors_npy):
    print(f"\n========== {source_season} -> {target_season} ==========")
    dataset_root = DATASETS_ROOT / ("train_on_all" if target_season == "2024" else "train_on_2025_all")
    image_dir = dataset_root / "images"
    analysis = run_error_analysis(source_model, image_dir)
    embed_names, embed_vectors = load_embeddings(embed_meta_csv, embed_vectors_npy)
    embed_dict = {Path(name).name: vec for name, vec in zip(embed_names, embed_vectors)}
    filtered = []
    yolo_feats = []
    for d in analysis:
        if d["image"] in embed_dict:
            filtered.append(d)
            yolo_feats.append(embed_dict[d["image"]])
    analysis = filtered
    yolo_feats = np.array(yolo_feats)
    pca_dim = min(PCA_DIM, yolo_feats.shape[1], yolo_feats.shape[0])
    pca = PCA(n_components=pca_dim, random_state=SEED)
    yolo_reduced = pca.fit_transform(yolo_feats)
    uncertainties = np.array([d["uncertainty"] for d in analysis])
    lengths = np.array([d["length"] for d in analysis])
    angles = np.array([d["angle"] for d in analysis])
    bends = np.array([d["bend"] for d in analysis])
    mean_length = np.mean(lengths)
    _morpho_dev = np.abs(lengths - mean_length)
    geo_block = np.stack([lengths, angles, bends], axis=1)
    fused = np.concatenate([zscore(yolo_reduced), ALPHA * zscore(geo_block)], axis=1)

    direction_root = EXPERIMENT_ROOT / f"{source_season}_to_{target_season}"
    direction_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for k in K_VALUES:
        print(f"Running K={k} for {source_season}->{target_season}")
        selected_idx = select_exact_k(analysis, fused, k)
        selected = [analysis[i]["image"] for i in selected_idx]
        k_root = direction_root / f"K_{k}"
        dataset_out = k_root / "dataset"
        model_out = k_root / "model"
        model_out.mkdir(parents=True, exist_ok=True)
        yaml_path, copied, missing = build_dataset(selected, image_dir, dataset_root, dataset_out)
        best = train_model(source_model, yaml_path, model_out)
        result = {
            "K": int(k),
            "selected": int(len(selected)),
            "copied": int(copied),
            "missing": int(missing),
            "best_model": str(best),
        }
        (k_root / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        summary_rows.append(result)

    with open(direction_root / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["K", "selected", "copied", "missing", "best_model"])
        writer.writeheader()
        writer.writerows(summary_rows)


def main():
    set_deterministic(SEED)

    source_model_2024 = PROJECT_ROOT / "models/2024/all-ponds/weights/best.pt"
    source_model_2025 = PROJECT_ROOT / "models/2025/YOLOv11n_train_on_2025_all_pose_300ep_best.pt"
    embed_meta_2024 = PROJECT_ROOT / "scripts/representation_analysis/outputs_repreasentation/rep_analysis/embeddings_meta.csv"
    embed_vec_2024 = PROJECT_ROOT / "scripts/representation_analysis/outputs_repreasentation/rep_analysis/embeddings_vectors.npy"
    embed_meta_2025 = PROJECT_ROOT / "outputs/rep_analysis_2025_model/embeddings_meta.csv"
    embed_vec_2025 = PROJECT_ROOT / "outputs/rep_analysis_2025_model/embeddings_vectors.npy"

    run_direction("2024", "2025", source_model_2024, embed_meta_2024, embed_vec_2024)
    run_direction("2025", "2024", source_model_2025, embed_meta_2025, embed_vec_2025)
    print("Done.")


if __name__ == "__main__":
    main()
