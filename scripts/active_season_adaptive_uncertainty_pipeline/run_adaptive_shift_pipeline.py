#!/usr/bin/env python3
"""
Fused Geometry-Aware Adaptive Transfer
Uncertainty + Morphometric deviation + Fused Diversity
+ Full Debug + Proper YAML + Training
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import numpy as np
import shutil
import torch
from ultralytics import YOLO
from sklearn.decomposition import PCA


# =====================================================
# CONFIG
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASETS_ROOT = PROJECT_ROOT / "datasets"
MODELS_ROOT = PROJECT_ROOT / "models/Adaptive_Shift"
MODELS_ROOT.mkdir(parents=True, exist_ok=True)

TOTAL_BUDGET = 200
PCA_DIM = 64
ALPHA = 1.0


# =====================================================
# UTILS
# =====================================================

def zscore(x):
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)


def fps(points, k):
    if len(points) == 0:
        return []

    centroid = points.mean(axis=0)
    dists = np.linalg.norm(points - centroid, axis=1)
    selected = [np.argmax(dists)]
    min_d = np.linalg.norm(points - points[selected[0]], axis=1)

    while len(selected) < min(k, len(points)):
        idx = np.argmax(min_d)
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


# =====================================================
# LOAD EMBEDDINGS
# =====================================================

def load_embeddings(meta_csv, vectors_npy):
    meta = np.genfromtxt(meta_csv, delimiter=",", dtype=str, skip_header=1)
    image_names = meta[:, 0]
    vectors = np.load(vectors_npy)
    return image_names, vectors


# =====================================================
# INFERENCE
# =====================================================

def run_error_analysis(model_path, image_dir):

    model = YOLO(str(model_path))
    device = 0 if torch.cuda.is_available() else "cpu"

    results = model.predict(
        source=str(image_dir),
        imgsz=640,
        device=device,
        verbose=False,
        conf=0.001
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

        data.append({
            "image": Path(r.path).name,
            "uncertainty": uncertainty,
            "length": length,
            "angle": angle,
            "bend": bend
        })

    return data


# =====================================================
# TRAINING (UNCHANGED CONFIG)
# =====================================================

def train_model(base_weights, yaml_path, out_dir):

    device = 0 if torch.cuda.is_available() else "cpu"

    model1 = YOLO(str(base_weights))
    model1.train(
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
        verbose=False
    )

    stage1_weights = out_dir / "weights" / "last.pt"

    model2 = YOLO(str(stage1_weights))
    model2.train(
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
        verbose=False
    )

    stage2_weights = out_dir / "weights" / "last.pt"

    model3 = YOLO(str(stage2_weights))
    results = model3.train(
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
        verbose=False
    )

    return Path(results.save_dir) / "weights/best.pt"


# =====================================================
# CORE PIPELINE
# =====================================================

def run_adaptive(source_season,
                 target_season,
                 source_model,
                 embed_meta_csv,
                 embed_vectors_npy):

    print(f"\n========== {source_season} → {target_season} ==========")

    dataset_root = DATASETS_ROOT / (
        "train_on_all" if target_season == "2024"
        else "train_on_2025_all"
    )

    image_dir = dataset_root / "images"

    analysis = run_error_analysis(source_model, image_dir)

    embed_names, embed_vectors = load_embeddings(embed_meta_csv,
                                                 embed_vectors_npy)

    embed_dict = {
        Path(name).name: vec
        for name, vec in zip(embed_names, embed_vectors)
    }

    filtered = []
    yolo_feats = []

    for d in analysis:
        if d["image"] in embed_dict:
            filtered.append(d)
            yolo_feats.append(embed_dict[d["image"]])

    analysis = filtered
    yolo_feats = np.array(yolo_feats)

    print("\n--- EMBEDDING DEBUG ---")
    print("YOLO embedding shape:", yolo_feats.shape)

    pca = PCA(n_components=min(PCA_DIM, yolo_feats.shape[1]))
    yolo_reduced = pca.fit_transform(yolo_feats)

    print("Reduced embedding shape:", yolo_reduced.shape)
    print("Explained variance sum:",
          round(np.sum(pca.explained_variance_ratio_), 4))

    uncertainties = np.array([d["uncertainty"] for d in analysis])
    lengths = np.array([d["length"] for d in analysis])
    angles = np.array([d["angle"] for d in analysis])
    bends = np.array([d["bend"] for d in analysis])

    mean_length = np.mean(lengths)
    morpho_dev = np.abs(lengths - mean_length)

    geo_block = np.stack([lengths, angles, bends], axis=1)

    fused = np.concatenate([
        zscore(yolo_reduced),
        ALPHA * zscore(geo_block)
    ], axis=1)

    print("Fused embedding shape:", fused.shape)
    print("----------------------------------")

    k = TOTAL_BUDGET
    k_unc = int(k * 0.4)
    k_morph = int(k * 0.4)
    k_div = k - k_unc - k_morph

    idx_unc = np.argsort(-uncertainties)[:k_unc]
    idx_morph = np.argsort(-morpho_dev)[:k_morph]
    idx_div = fps(fused, k_div)

    selected_idx = list(set(idx_unc.tolist()) |
                        set(idx_morph.tolist()) |
                        set(idx_div))[:k]

    selected = [analysis[i]["image"] for i in selected_idx]

    print("Selected images:", len(selected))

    # =====================================================
    # CREATE DATASET + YAML (FIXED)
    # =====================================================

    out_dir = DATASETS_ROOT / f"adaptive_{source_season}_to_{target_season}"

    if out_dir.exists():
        shutil.rmtree(out_dir)

    (out_dir / "images/train").mkdir(parents=True)
    (out_dir / "labels/train").mkdir(parents=True)

    for img_name in selected:

        stem = Path(img_name).stem

        for img in image_dir.rglob(img_name):
            shutil.copy2(img, out_dir / "images/train" / img.name)

        for lbl in (dataset_root / "labels").rglob(stem + ".txt"):
            shutil.copy2(lbl, out_dir / "labels/train" / lbl.name)

    yaml_path = out_dir / "data.yaml"

    yaml_content = f"""path: {out_dir.resolve()}
train: images/train
val: images/train
nc: 1
names: ['prawn']
kpt_shape: [4,3]
flip_idx: [0,1,2,3]
"""

    yaml_path.write_text(yaml_content.strip())

    print("YAML created at:", yaml_path)

    # =====================================================
    # TRAIN
    # =====================================================

    trained_model = train_model(
        source_model,
        yaml_path,
        MODELS_ROOT / f"{source_season}_to_{target_season}"
    )

    print("Training finished.")
    print("====================================")


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    SOURCE_MODEL_2024 = PROJECT_ROOT / "models/2024/all-ponds/weights/best.pt"
    SOURCE_MODEL_2025 = PROJECT_ROOT / "models/2025/YOLOv11n_train_on_2025_all_pose_300ep_best.pt"

    EMBED_META_2024 = PROJECT_ROOT / "scripts/representation_analysis/outputs_repreasentation/rep_analysis/embeddings_meta.csv"
    EMBED_VEC_2024 = PROJECT_ROOT / "scripts/representation_analysis/outputs_repreasentation/rep_analysis/embeddings_vectors.npy"

    EMBED_META_2025 = PROJECT_ROOT / "outputs/rep_analysis_2025_model/embeddings_meta.csv"
    EMBED_VEC_2025 = PROJECT_ROOT / "outputs/rep_analysis_2025_model/embeddings_vectors.npy"

    run_adaptive("2024", "2025",
                 SOURCE_MODEL_2024,
                 EMBED_META_2024,
                 EMBED_VEC_2024)

    run_adaptive("2025", "2024",
                 SOURCE_MODEL_2025,
                 EMBED_META_2025,
                 EMBED_VEC_2025)