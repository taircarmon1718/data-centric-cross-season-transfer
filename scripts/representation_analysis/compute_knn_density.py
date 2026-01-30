import numpy as np
import os
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from utils.io import load_embeddings


def compute_knn(out_dir: str = "outputs/rep_analysis", k: int = 10, save_suffix: str = "knn"):
    records, vectors = load_embeddings(out_dir)
    nbrs = NearestNeighbors(n_neighbors=k+1, metric="euclidean").fit(vectors)
    distances, indices = nbrs.kneighbors(vectors)
    # exclude self (first column)
    densities = 1.0 / (distances[:,1:].mean(axis=1) + 1e-12)
    df = pd.DataFrame(records)
    df["knn_density"] = densities
    csv_out = os.path.join(out_dir, f"embeddings_{save_suffix}.csv")
    df.to_csv(csv_out, index=False)
    print(f"Saved KNN densities to `{csv_out}`.")


if __name__ == "__main__":
    compute_knn()

