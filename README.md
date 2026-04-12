# Fixed-Budget Cross-Season Adaptation for Robotic Morphometric Monitoring of *Macrobrachium rosenbergii*

---

## Overview

Seasonal domain shift is a major challenge for robotic morphometric monitoring in commercial aquaculture. Models trained in one production season often degrade when deployed in subsequent seasons due to changes in illumination, turbidity, acquisition protocols, and pose variability.

This repository contains the full code and experiments for our fixed-budget cross-season adaptation framework, applied to a mobile keypoint-based morphometric monitoring system for the giant freshwater prawn *Macrobrachium rosenbergii* in commercial recirculating aquaculture systems (RAS).

---

## Key Findings

- **Seasonal transfer is directionally asymmetric**: S24 → S25 deployment remains stable, but the reverse direction (S25 → S24) causes severe detection collapse — detection rates drop below **62%** without adaptation.
- **The primary failure mode is localization instability** (missed detections), not regression drift. Missed detections remove individuals entirely from the measurement pipeline.
- Under a fixed annotation budget of **K = 200** images, our geometry-aware approach achieves results comparable to full transfer with **over 90% reduction in annotation effort**.
- Geometry-aware sampling **outperforms full transfer** in the challenging S25 → S24 direction while using only ~8% of the full training set.

---

## Method

### Measurements

Two body lengths are estimated from four predicted anatomical keypoints (rostrum tip, eye base, carapace posterior edge, telson):

- **Carapace Length (CL)**: carapace posterior edge → eye base
- **Total Length (TL)**: rostrum tip → telson

Pixel distances are converted to millimeters via refraction-aware camera calibration.

### Geometry-Aware Sample Selection

Rather than random sampling, target-season images are ranked using a fused representation that combines appearance embeddings with three interpretable geometric descriptors:

- **Carapace–rostrum length** — anatomical scale proxy
- **Carapace–rostrum orientation** — global body heading
- **Internal bend angle** — body curvature at the carapace vertex

The annotation set (size K) is constructed from three complementary criteria:
- **40%** highest keypoint prediction uncertainty
- **40%** largest morphological deviation from the mean carapace–rostrum length
- **20%** geometric diversity via Farthest Point Sampling (FPS) in fused embedding space

### Staged Fine-Tuning

A three-stage progressive unfreezing schedule balances representational stability and plasticity:

| Stage | Epochs | Freeze | LR | Purpose |
|-------|--------|--------|----|---------|
| 1 | 30 | 20 | 5×10⁻⁵ | Adapt high-level features, preserve geometry |
| 2 | 40 | 15 | 2×10⁻⁵ | Increase adaptation capacity gradually |
| 3 | 10 | 0 | 5×10⁻⁶ | Full alignment to target distribution |

---

## Results

### Detection Performance (K = 200)

| Method | Direction | CL Det. (%) | TL Det. (%) |
|--------|-----------|-------------|-------------|
| No Adaptation | S24 → S25 | 95.17 | 95.17 |
| Full Transfer | S24 → S25 | 96.55 | 96.55 |
| Random (200) | S24 → S25 | 94.71 | 94.71 |
| **Geometry-Aware (200)** | **S24 → S25** | **91.03** | **91.03** |
| No Adaptation | S25 → S24 | 61.86 | 67.52 |
| Full Transfer | S25 → S24 | 91.85 | 92.96 |
| Random (200) | S25 → S24 | 90.91 | 100.00 |
| **Geometry-Aware (200)** | **S25 → S24** | **95.09** | **100.00** |

### Morphometric Accuracy (K = 200)

| Method | Direction | MAE_CL (mm) | MAE_TL (mm) |
|--------|-----------|-------------|-------------|
| No Adaptation | S24 → S25 | 2.30 | 12.30 |
| Full Transfer | S24 → S25 | 2.31 | 5.42 |
| **Geometry-Aware (200)** | **S24 → S25** | **2.67** | **6.84** |
| No Adaptation | S25 → S24 | 4.56 | 16.01 |
| Full Transfer | S25 → S24 | 4.86 | 15.62 |
| **Geometry-Aware (200)** | **S25 → S24** | **4.78** | **12.70** |

---

## Repository Structure

```
data-centric-cross-season-transfer/
│
├── scripts/
│   ├── training_tl_models/          # B1, B2, B3 training scripts (freeze-depth grid)
│   │   ├── run_B1_feature_extraction.py
│   │   ├── run_B2_staged_finetune_GRID.py
│   │   ├── run_B3_final_training.py
│   │   └── run_TL_Optim_random_k_training.py
│   │
│   ├── eval/                        # Evaluation on S24 and S25 test sets
│   │   ├── check_on_2024.py
│   │   ├── check_on_2025.py
│   │   └── print_final_results.py
│   │
│   ├── representation_analysis/     # DINO embeddings + UMAP visualization
│   │   ├── extract_dino_embeddings_full_2024_2025.py
│   │   ├── visualize_dino_umap_2024_2025.py
│   │   ├── compute_knn_density.py
│   │   └── core_set_selection/      # Geometry-aware subset builders
│   │
│   ├── shift_experiments/           # Distribution shift quantification
│   │   ├── analyze_shift_unified.py
│   │   ├── visualize_shift_space.py
│   │   └── build_shifted_core_datasets.py
│   │
│   ├── active_season_adaptive_uncertainty_pipeline/
│   │   ├── run_k_sensitivity_experiment.py   # Budget sensitivity (K = 50–800)
│   │   ├── run_active_uncertainty_transfer.py
│   │   └── run_adaptive_shift_pipeline.py
│   │
│   ├── analysis/                    # Morphology variance, geometric analysis
│   ├── season_shift_analysis/       # Season-level domain gap characterization
│   ├── plots_for_paper/             # Figure generation scripts
│   ├── preprocess/                  # Data preprocessing utilities
│   └── R_scripts/                   # Statistical analysis
│
├── figs/                            # Paper figures and result plots
├── outputs/                         # Evaluation outputs (CSV, JSON)
├── adaptive_k_experiment/           # K-sensitivity experiment model runs
├── eval_k_sensitivity_results/      # K-sensitivity aggregated results
├── run_all_k_evaluations.py         # Sweeps eval across all K models
├── final_Excel.xlsx                 # Final aggregated results table
│
├── data/          (gitignored)      # Raw images + ImageJ annotations
├── datasets/      (gitignored)      # YOLO-format train/val datasets
└── models/        (gitignored)      # Trained model weights (.pt)
    ├── 2024/
    ├── 2025/
    └── TF/                          # Cross-season transfer models
```

---

## Dataset

Data is not included in this repository. The dataset comprises:

- **Season 2024 (S24)**: 2,508 annotated images across 3 ponds (2 circular, 1 rectangular)
- **Season 2025 (S25)**: 1,199 annotated images

Each image is annotated with 4 anatomical keypoints: rostrum tip, eye base, carapace posterior edge, and telson. Ground-truth length measurements were independently collected using ImageJ from calibrated underwater recordings.

Due to its size and associated constraints, the data is not publicly available but can be provided upon reasonable request.
