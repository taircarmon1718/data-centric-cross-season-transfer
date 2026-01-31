"""
Compute distance-from-2024-manifold for Season 2025 embeddings.
Saves a CSV with an added `dist_to_2024` column for 2025 samples.

Usage: python scripts/representation_analysis/dist_to_2024.py --out outputs/rep_analysis --k 10 --agg mean
"""
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from utils.io import load_embeddings


def find_embeddings_dir(preferred_dir: str = None):
    """Return a directory path containing embeddings_vectors.npy (or None)."""
    if preferred_dir:
        # prefer the preferred dir if it contains embeddings
        if os.path.exists(os.path.join(preferred_dir, "embeddings_vectors.npy")) or os.path.exists(os.path.join(preferred_dir, "embeddings_meta.csv")):
            return preferred_dir
    # check common outputs location
    cand = "outputs/rep_analysis"
    if os.path.exists(os.path.join(cand, "embeddings_vectors.npy")) or os.path.exists(os.path.join(cand, "embeddings_meta.csv")):
        return cand
    # search entire outputs/ tree
    if os.path.isdir("outputs"):
        for root, dirs, files in os.walk("outputs"):
            if "embeddings_vectors.npy" in files:
                return root
    # search workspace for embeddings_vectors.npy
    for root, dirs, files in os.walk("."):
        if "embeddings_vectors.npy" in files:
            return root
    return None


def compute_dist_to_2024(out_dir: str = "outputs/rep_analysis", k: int = 10, agg: str = "mean", save_name: str = None):
    # If out_dir doesn't appear to contain embeddings, try to find them first
    csv_exists = os.path.exists(os.path.join(out_dir, "embeddings_meta.csv"))
    npy_exists = os.path.exists(os.path.join(out_dir, "embeddings_vectors.npy"))
    if not (csv_exists or npy_exists):
        found = find_embeddings_dir(out_dir)
        if found is None:
            raise FileNotFoundError(f"Embeddings not found in `{out_dir}` and none discovered in workspace. Run extract_embeddings first.")
        print(f"Info: using discovered embeddings directory: {found}")
        out_dir = found

    # Now load embeddings from the resolved out_dir
    records, vectors = load_embeddings(out_dir)
    vectors = np.asarray(vectors)
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape {vectors.shape}")

    # Build DataFrame from records
    df = pd.DataFrame(records)

    # If season column missing or contains 'unknown', try to infer from image_path heuristically
    def infer_season_from_path(path_str: str):
        p = str(path_str).lower()
        # common indicators for 2025
        indicators_2025 = ["2025", "train_on_2025", "2025_all", "test2025", "2025test", "2025_"]
        indicators_2024 = ["2024", "train_on_all", "train_on_2024", "2024test", "test2024", "2024_"]
        for tok in indicators_2025:
            if tok in p:
                return "2025"
        for tok in indicators_2024:
            if tok in p:
                return "2024"
        return None

    if "season" not in df.columns:
        # try to infer season for each record
        inferred = [infer_season_from_path(r.get("image_path", "")) for r in records]
        if any(x is not None for x in inferred):
            df["season"] = [x if x is not None else "unknown" for x in inferred]
            print("Info: inferred season labels for some records from image_path.")
        else:
            raise ValueError("Metadata must contain a 'season' column indicating 2024 or 2025, and inference failed.")
    else:
        # normalize existing season values and fill 'unknown'
        df["season"] = df["season"].astype(str).str.strip()
        # If any season values are 'unknown' or empty, attempt inference
        mask_unknown = df["season"].isin(["", "unknown", "None", "nan"]) | df["season"].isna()
        if mask_unknown.any():
            for idx in df[mask_unknown].index:
                s = infer_season_from_path(df.at[idx, "image_path"]) or "unknown"
                df.at[idx, "season"] = s
            if df["season"].isin(["unknown"]).any():
                print("Warning: some season values remain unknown after inference.")

    # Find indices for seasons
    idx_2024 = df.index[df["season"] == "2024"].tolist()
    idx_2025 = df.index[df["season"] == "2025"].tolist()

    # DEBUG prints
    print(f"DEBUG: resolved out_dir = {out_dir}")
    print(f"DEBUG: total samples = {len(df)}, #2024 = {len(idx_2024)}, #2025 = {len(idx_2025)}")

    if len(idx_2024) == 0:
        raise ValueError("No Season 2024 embeddings found; cannot compute distance to 2024 manifold.")
    if len(idx_2025) == 0:
        print("Warning: No Season 2025 embeddings found; nothing to compute.")

    X2024 = vectors[idx_2024]
    X2025 = vectors[idx_2025] if len(idx_2025) > 0 else np.zeros((0, vectors.shape[1]))

    n_neighbors = min(k, X2024.shape[0])
    if n_neighbors <= 0:
        raise ValueError("Not enough 2024 samples to compute neighbors (k=0 or fewer 2024 samples)")

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(X2024)
    if X2025.shape[0] > 0:
        distances, indices = nbrs.kneighbors(X2025)
        if agg == "mean":
            scores = distances.mean(axis=1)
        elif agg == "min":
            scores = distances.min(axis=1)
        else:
            raise ValueError("agg must be one of: 'mean', 'min'")
    else:
        scores = np.array([])

    # Add column to DataFrame; set NaN for non-2025 rows
    out_col = np.full(len(df), np.nan)
    for i, idx in enumerate(idx_2025):
        out_col[idx] = float(scores[i])
    df["dist_to_2024"] = out_col

    if save_name is None:
        save_name = os.path.join(out_dir, f"embeddings_2025_to_2024_knn_k{k}_{agg}.csv")
    # ensure directory exists
    save_dir = os.path.dirname(save_name)
    if save_dir and not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    print(f"DEBUG: saving CSV to {save_name}")
    df.to_csv(save_name, index=False)
    print(f"Saved distances CSV to: {save_name}")
    return save_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="outputs/rep_analysis", help="Directory with embeddings")
    parser.add_argument("--k", type=int, default=10, help="Number of 2024 neighbors to use")
    parser.add_argument("--agg", type=str, default="mean", choices=["mean", "min"], help="Aggregation of KNN distances")
    parser.add_argument("--save", type=str, default=None, help="Path to save resulting CSV (defaults to out dir)")
    args = parser.parse_args()
    compute_dist_to_2024(args.out, args.k, args.agg, args.save)
