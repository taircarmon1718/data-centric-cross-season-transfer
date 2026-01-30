import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils.io import load_embeddings


def visualize(out_dir: str = "outputs/rep_analysis", method: str = "pca", save_path: str = None):
    records, vectors = load_embeddings(out_dir)
    # simple normalization
    vs = vectors
    if method == "pca":
        proj = PCA(n_components=2).fit_transform(vs)
    else:
        proj = TSNE(n_components=2, init="pca", random_state=0).fit_transform(vs)
    # color by season
    seasons = [r["season"] for r in records]
    colors = {"2024":"tab:blue", "2025":"tab:orange"}
    plt.figure(figsize=(8,6))
    for s in sorted(set(seasons)):
        idx = [i for i,rr in enumerate(records) if rr["season"]==s]
        plt.scatter(proj[idx,0], proj[idx,1], label=s, alpha=0.7, c=colors.get(s, None))
    plt.legend()
    plt.title(f"Embedding projection ({method})")
    if save_path is None:
        save_path = os.path.join(out_dir, f"embedding_{method}.png")
    plt.savefig(save_path, dpi=150)
    print(f"Saved projection to `{save_path}`.")


if __name__ == "__main__":
    visualize()

