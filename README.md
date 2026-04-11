# Data-Centric Cross-Season Transfer for Prawn Size Measurement

A research pipeline for automated morphometric measurement of the freshwater prawn *Macrobrachium rosenbergii* from underwater images, with a focus on **cross-season domain adaptation** using data-centric methods.

---

## Overview

Underwater prawn images captured across different seasons (2024 and 2025) exhibit significant distribution shift in visual appearance, lighting, and pond conditions. This project investigates how to bridge that gap using **transfer learning** and **data-centric strategies** (core-set selection, active learning, representation-guided sampling) rather than architecture changes.

Two body measurements are predicted:
- **Carapace length** (carapace start → eyes)
- **Total body length** (rostrum → tail)

Predictions come from YOLO-based keypoint detection models, validated against ground-truth annotations (ImageJ + manual labeling). Pixel distances are converted to millimeters using refraction-aware camera calibration.

![Annotated Prawns](figs/anntoated%20prawns.png)

---

## Transfer Directions

All experiments run in both directions:
- **2024 → 2025**: model trained on 2024 data, adapted / evaluated on 2025 images
- **2025 → 2024**: model trained on 2025 data, adapted / evaluated on 2024 images

---

## Experimental Baselines

| Baseline | Description |
|----------|-------------|
| **B1** | Feature Extraction — freeze backbone layers, train only head across a grid of freeze depths × data fractions (k) |
| **B2** | Staged Fine-tuning — two-stage training with a grid search over freeze schemes and k values |
| **B3** | Full Production Training — final models trained at best configuration (freeze=4) on complete datasets |

---

## Repository Structure

```
data-centric-cross-season-transfer/
│
├── scripts/
│   ├── training_tl_models/          # B1, B2, B3 training scripts
│   │   ├── run_B1_feature_extraction.py
│   │   ├── run_B2_staged_finetune_GRID.py
│   │   ├── run_B3_final_training.py
│   │   ├── run_TL_Optim_random_k_training.py
│   │   └── clean_k_random_seeds.py
│   │
│   ├── eval/                        # Evaluation on 2024 and 2025 test sets
│   │   ├── check_on_2024.py
│   │   ├── check_on_2025.py
│   │   ├── print_final_results.py
│   │   └── visualize_single_prediction.py
│   │
│   ├── representation_analysis/     # DINO embeddings + UMAP visualization
│   │   ├── extract_embeddings.py
│   │   ├── extract_dino_embeddings_full_2024_2025.py
│   │   ├── visualize_dino_umap_2024_2025.py
│   │   ├── visualize_dino_umap_2025train_vs_2024test.py
│   │   ├── compute_knn_density.py
│   │   └── core_set_selection/      # Representation-guided core set builders
│   │
│   ├── shift_experiments/           # Distribution shift quantification and experiments
│   │   ├── extract_all_embeddings.py
│   │   ├── analyze_shift_unified.py
│   │   ├── visualize_shift_space.py
│   │   ├── build_shifted_core_datasets.py
│   │   └── active_selection/        # Uncertainty + diversity sampling
│   │
│   ├── active_season_adaptive_uncertainty_pipeline/
│   │   ├── run_k_sensitivity_experiment.py
│   │   ├── run_active_uncertainty_transfer.py
│   │   └── run_adaptive_shift_pipeline.py
│   │
│   ├── analysis/                    # Morphology variance and geometry analysis
│   ├── season_shift_analysis/       # Season-level domain gap analysis
│   ├── plots_for_paper/             # Figure generation scripts
│   ├── preprocess/                  # Data preprocessing utilities
│   ├── R_scripts/                   # R-based statistical analysis
│   └── results_summary.py          # Aggregates results into summary tables
│
├── figs/                            # Output figures and plots
├── outputs/                         # Evaluation outputs (CSV, JSON)
├── adaptive_k_experiment/           # K-sensitivity experiment runs
├── eval_k_sensitivity_results/      # Aggregated K-sensitivity evaluation results
│
├── run_all_k_evaluations.py         # Runs eval across all K-sensitivity models
├── final_Excel.xlsx                 # Final aggregated results
│
├── data/          (gitignored)      # Raw images and Excel annotations
├── datasets/      (gitignored)      # YOLO-format datasets (train/val splits)
└── models/        (gitignored)      # Trained model weights (.pt files)
    ├── 2024/
    ├── 2025/
    └── TF/                          # Transfer-learning model weights
```

---

## Getting Started

### Requirements

```bash
pip install ultralytics torch torchvision
pip install umap-learn scikit-learn pandas openpyxl matplotlib seaborn
```

### Training

Run B1 (feature extraction) grid:
```bash
python scripts/training_tl_models/run_B1_feature_extraction.py
```

Run B2 (staged fine-tuning) grid:
```bash
python scripts/training_tl_models/run_B2_staged_finetune_GRID.py
```

Run B3 (final production models):
```bash
python scripts/training_tl_models/run_B3_final_training.py
```

### Evaluation

Evaluate on 2025 test set:
```bash
python scripts/eval/check_on_2025.py --model <path/to/model.pt>
```

Run K-sensitivity sweep across all adaptive-k models:
```bash
python run_all_k_evaluations.py \
  --eval-script scripts/eval/check_on_2025.py \
  --model-arg --model
```

### Representation Analysis

Extract DINO embeddings and visualize season gap:
```bash
python scripts/representation_analysis/extract_dino_embeddings_full_2024_2025.py
python scripts/representation_analysis/visualize_dino_umap_2024_2025.py
```

---

## Key Results

Results are summarized in `final_Excel.xlsx` and figures are in `figs/`. Key metrics:
- **MAE** (Mean Absolute Error) in mm
- **MPE** (Mean Percentage Error)
- Evaluated per-pond and aggregated across the full test set

![Generalization Gap](figs/generalization_Gap.png)
![Detection Transfer vs Baseline](figs/comparison_detection_transfer_vs_baseline.png)

---

## Data

Data is not included in this repository (gitignored). The expected structure under `data/` is:
```
data/
├── images/          # Raw pond images (2024 and 2025)
├── excel/           # Ground-truth annotation files (ImageJ + manual)
├── kp_eval_2024_original/
└── kp_eval_2025_gamma/
```

Datasets (YOLO format) are built from the raw data using the preprocessing scripts in `scripts/preprocess/`.
