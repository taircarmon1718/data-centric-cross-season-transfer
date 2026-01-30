Representation analysis pipeline

Usage:
- Extract embeddings:
    python scripts/representation_analysis/extract_embeddings.py --model models/2024/all-ponds/weights/best.pt --out outputs/rep_analysis
- Visualize:
    python scripts/representation_analysis/visualize_embedding_space.py
- Compute kNN density:
    python scripts/representation_analysis/compute_knn_density.py

Notes:
- This pipeline will try to use `ultralytics` if available to load YOLO-Pose checkpoints, otherwise falls back to a best-effort torch model.
- Missing dataset dirs are skipped with a warning.

