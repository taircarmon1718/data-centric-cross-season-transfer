import numpy as np
import os
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from utils.io import load_embeddings
import argparse


def find_embeddings_dir(preferred_dir: str = None):
    """Return a directory path containing embeddings_vectors.npy (or None)."""
    candidates = []
    if preferred_dir:
        candidates.append(preferred_dir)
    # also check common outputs location
    candidates.append("outputs/rep_analysis")
    # search entire outputs/ tree
    if os.path.isdir("outputs"):
        for root, dirs, files in os.walk("outputs"):
            if "embeddings_vectors.npy" in files:
                return root
    # check workspace current dir
    for root, dirs, files in os.walk("."):
        if "embeddings_vectors.npy" in files:
            return root
    # lastly check preferred_dir if provided
    if preferred_dir and os.path.isdir(preferred_dir):
        return preferred_dir
    return None


def compute_knn(out_dir: str = "outputs/rep_analysis", k: int = 10, save_suffix: str = "knn"):
    # Ensure we have an embeddings directory; try to locate if missing
    if not os.path.isdir(out_dir) or not (os.path.exists(os.path.join(out_dir, "embeddings_vectors.npy")) or os.path.exists(os.path.join(out_dir, "embeddings_meta.csv"))):
        found = find_embeddings_dir(out_dir)
        if found:
            print(f"Info: using discovered embeddings directory: {found}")
            out_dir = found
        else:
            raise FileNotFoundError(f"Could not find embeddings in `{out_dir}` nor in outputs/; run extract_embeddings first.")

    try:
        records, vectors = load_embeddings(out_dir)
    except FileNotFoundError as e:
        raise

    # Ensure vectors is 2D numpy array
    vectors = np.asarray(vectors)
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D array for vectors, got shape {vectors.shape}")

    nbrs = NearestNeighbors(n_neighbors=min(k+1, vectors.shape[0]), metric="euclidean").fit(vectors)
    distances, indices = nbrs.kneighbors(vectors)
    # exclude self (first column)
    if distances.shape[1] <= 1:
        densities = np.zeros(vectors.shape[0])
    else:
        densities = 1.0 / (distances[:,1:].mean(axis=1) + 1e-12)
    df = pd.DataFrame(records)
    df["knn_density"] = densities
    csv_out = os.path.join(out_dir, f"embeddings_{save_suffix}.csv")
    df.to_csv(csv_out, index=False)
    print(f"Saved KNN densities to `{csv_out}`.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="outputs/rep_analysis", help="Directory with embeddings (embeddings_vectors.npy and optional embeddings_meta.csv)")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--suffix", type=str, default="knn")
    args = parser.parse_args()
    compute_knn(args.out, args.k, args.suffix)
