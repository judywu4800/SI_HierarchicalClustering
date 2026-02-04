import sys, os
sys.path.append(os.path.abspath('../src'))
import pandas as pd
import numpy as np
import seaborn as sns
from hierarchical_clustering_invariant import *
from palmerpenguins import load_penguins

def clusters_to_labels(clusters, n_samples):
    labels = np.empty(n_samples, dtype=int)
    for cid, cluster in enumerate(clusters):
        for idx in cluster.points:
            labels[idx] = cid
    return labels


def compute_stability(X, K, linkage="complete", tau=0.1, n_runs=100, random_state=None):
    rng = np.random.default_rng(random_state)
    labels_list = []

    # multiple runs with different seeds
    for _ in range(n_runs):
        seed = rng.integers(0, 2 ** 32 - 1)
        model = AgglomerativeClustering(X, tau=tau, n_clusters=K, linkage=linkage, random_state=seed)
        model.fit()
        clusters = model.K_clusters
        labels = clusters_to_labels(clusters, X.shape[0])
        labels_list.append(np.asarray(labels))

    labels_list = np.stack(labels_list, axis=0)  # shape (n_runs, n_samples)
    n_runs, n_samples = labels_list.shape

    cooccurrence = np.zeros((n_samples, n_samples))
    for r in range(n_runs):
        labels_r = labels_list[r]
        for k in np.unique(labels_r):
            idx = np.where(labels_r == k)[0]
            cooccurrence[np.ix_(idx, idx)] += 1
    cooccurrence /= n_runs  # convert to proportion

    # Per-sample average co-occurrence (how stably each point clusters with others)
    mean_cooccurrence_per_sample = np.mean(cooccurrence, axis=1)

    #sns.heatmap(cooccurrence, cmap="viridis")
    #plt.title("Pairwise Co-occurrence Matrix")
    #plt.show()

    return {
        "cooccurrence": cooccurrence,
        "mean_cooccurrence_per_sample": mean_cooccurrence_per_sample,
    }

if __name__ == '__main__':
    penguins_raw = load_penguins()
    penguins = penguins_raw[(penguins_raw["sex"] == "female") & (penguins_raw.notna().all(axis=1)) & (
        penguins_raw["year"].between(2007, 2008))]
    labels = penguins["species"]
    X = penguins[["flipper_length_mm", "bill_length_mm"]].to_numpy()
    result = compute_stability(X, 2, linkage="complete", tau=0.1, n_runs=500, random_state=0)
    coor = result["cooccurrence"]
    mean_coor = result["mean_cooccurrence_per_sample"]
    outdir = "../results/raw/penguins"
    os.makedirs(outdir, exist_ok=True)

    np.savetxt(os.path.join(outdir, "cooccurrence.csv"), coor, delimiter=",")