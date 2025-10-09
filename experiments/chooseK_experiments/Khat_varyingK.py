import sys, os
sys.path.append(os.path.abspath('../../src'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.datasets import make_blobs
from find_best_K import *
from joblib import Parallel, delayed
import random
from hierarchical_clustering_invariant import *
from scipy.cluster.hierarchy import linkage, fcluster


def wcss_from_labels(X, labels):
    """Compute WCSS given cluster labels."""
    wcss = 0
    for k in np.unique(labels):
        cluster_points = X[labels == k]
        center = cluster_points.mean(axis=0)
        wcss += ((cluster_points - center) ** 2).sum()
    return wcss


def compute_wcss_hclust(X, n_clusters, method="complete"):
    Z = linkage(X, method=method)
    labels = fcluster(Z, n_clusters, criterion="maxclust")
    return wcss_from_labels(X, labels)


def gap_statistic(X, K_max=10, B=50, method="complete", random_state=None):
    """
    Deterministic Gap Statistic (Tibshirani et al. 2001) using hierarchical clustering.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Input data
    K_max : int
        Max number of clusters to consider
    B : int
        Number of bootstrap reference datasets
    method : str
        Linkage method ('ward', 'complete', 'average', etc.)
    random_state : int or None

    Returns
    -------
    best_K : int
        Estimated number of clusters (hat{K})
    gaps : np.array
        Gap values for K = 1..K_max
    sk : np.array
        Standard error for each K
    """
    rng = np.random.RandomState(random_state)

    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    log_wcss = np.zeros(K_max)
    gap = np.zeros(K_max)
    sk = np.zeros(K_max)

    for k in range(1, K_max + 1):
        wk = compute_wcss_hclust(X, k, method=method)
        log_wcss[k - 1] = np.log(wk)
        wk_refs = np.zeros(B)
        for b in range(B):
            X_ref = rng.uniform(mins, maxs, size=X.shape)
            wk_refs[b] = compute_wcss_hclust(X_ref, k, method=method)
        log_wk_refs = np.log(wk_refs)

        gap[k - 1] = np.mean(log_wk_refs) - log_wcss[k - 1]
        sk[k - 1] = np.std(log_wk_refs) * np.sqrt(1 + 1 / B)

    for k in range(1, K_max):
        if gap[k - 1] >= gap[k] - sk[k]:
            return k, gap, sk
    return K_max, gap, sk


def one_replication(K_true, n=120, p=10, tau=0.1, total_alpha=0.05,
                    K_max=120, B=50, method="complete"):
    """Run one replication and return both K_hat_F and K_hat_gap for given K_true."""
    X, labels = make_blobs(n_samples=n, n_features=p, centers=K_true, cluster_std=1)
    # --- Proposed method---
    alpha_list = np.full(n - 1, total_alpha / (n - 1))
    K_hat_F, _, _ = find_best_K_F(X, tau=tau, alpha_list=alpha_list, total_alpha=total_alpha)

    # --- Gap test---
    K_hat_gap, _, _ = gap_statistic(X, K_max=K_max, B=B, method=method)

    return K_hat_F, K_hat_gap


def simulate_results(K_list, n_rep=100, n_jobs=-1, **kwargs):
    """
    Run simulations in parallel for different true K.
    Returns dict: {K_true: {"F": [...], "Gap": [...]}}
    """
    results = {}
    for K_true in K_list:
        pairs = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(one_replication)(K_true, **kwargs) for _ in range(n_rep)
        )
        k_hats_F, k_hats_gap = zip(*pairs)
        results[K_true] = {"Proposed Method": list(k_hats_F), "Gap Test": list(k_hats_gap)}
    return results


if __name__ == "__main__":
    random.seed(0)
    results_F = {}
    K_list = [1,3,5,7,9,11]
    #sd_list = [0.1, 0.5, 0.8, 1, 2, 5]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    output_dir = os.path.join(base_dir,"results", f"k_hat_boxplot_{timestamp}")
    os.makedirs("results", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)


    results = simulate_results(K_list, n_rep=100, n_jobs=-1)
    rows = []
    for k_true, result in results.items():
        for kf, kg in zip(result["Proposed Method"], result["Gap Test"]):
            rows.append((k_true, kf, kg))

    df = pd.DataFrame(rows, columns=["K_true", "K_hat_F", "K_hat_gap"])
    df["K_true"] = df["K_true"].astype(int)
    df.to_csv(os.path.join(output_dir, "k_hat_results.csv"), index=False)

    plt.figure(figsize=(8, 6))

    for k_true, group in df.groupby("K_true"):
        plt.boxplot(group["K_hat_F"], positions=[k_true], widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor="skyblue", color="blue"),
                    medianprops=dict(color="black"))
        plt.scatter([k_true]*len(group), group["K_hat_F"], color="black", alpha=0.3)

    lim = [min(df["K_true"].min(), df["K_hat_F"].min()),
           max(df["K_true"].max(), df["K_hat_F"].max())]
    plt.plot(lim, lim, "r--", label="y = x")

    plt.xlabel("True K")
    plt.ylabel("Estimated K̂")
    plt.title("Estimated vs True Clusters")
    plt.xlim(df['K_true'].min()-1,df["K_true"].max()+1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "k_hat_boxplot.png"))
    plt.close()

    # ---- Heatmap----
    max_khat = df["K_hat_F"].max()
    k_true_vals = sorted(df["K_true"].unique())
    k_hat_vals = range(1, max_khat + 1)

    freq_table = (
        df.groupby(["K_true", "K_hat_F"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=k_hat_vals, fill_value=0)
    ).T

    freq_table = freq_table.reindex(columns=k_true_vals, fill_value=0)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        freq_table,
        cmap="Blues",
        cbar_kws={'label': 'Frequency'},
        linewidths=0.5
    )
    plt.xlabel("True K")
    plt.ylabel("Estimated K̂")
    plt.title("Heatmap of Estimated vs True Clusters (Rectangular)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "k_hat_heatmap_rect.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    for k_true, group in df.groupby("K_true"):
        plt.boxplot(group["K_hat_F"], positions=[k_true], widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor="skyblue", color="blue"),
                    medianprops=dict(color="black"))
        plt.scatter([k_true] * len(group), group["K_hat_F"], color="black", alpha=0.3)

    lim = [min(df["K_true"].min(), df["K_hat_F"].min()),
           max(df["K_true"].max(), df["K_hat_F"].max())]
    plt.plot(lim, lim, "r--", label="y = x")

    plt.xlabel("True K")
    plt.ylabel("Estimated K̂ (F-test)")
    plt.title("Estimated vs True Clusters (F-test)")
    plt.xlim(df['K_true'].min() - 1, df["K_true"].max() + 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "k_hat_boxplot_F.png"))
    plt.close()

    # ---- Boxplot comparison ----
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="K_true", y="K_hat_F", data=df, color="skyblue", width=0.5)
    sns.boxplot(x="K_true", y="K_hat_gap", data=df, color="lightcoral", width=0.3)

    plt.xlabel("True K")
    plt.ylabel("Estimated K̂")
    plt.title("Comparison of Estimated Clusters: Proposed Method vs Gap Test")
    plt.legend(handles=[
        plt.Line2D([0], [0], color="skyblue", lw=8, label="Proposed Method"),
        plt.Line2D([0], [0], color="lightcoral", lw=8, label="Gap Test")
    ])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "k_hat_boxplot_Proposed_vs_Gap.png"))
    plt.close()

    # ---- Heatmap comparison ----
    freq_table_methods = (
        df.groupby(["K_hat_F", "K_hat_gap"])
        .size()
        .unstack(fill_value=0)
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        freq_table_methods,
        cmap="Blues",
        cbar_kws={'label': 'Frequency'},
        linewidths=0.5
    )
    plt.xlabel("K̂ (Gap Test)")
    plt.ylabel("K̂ (Proposed Method)")
    plt.title("Heatmap of K̂: Proposed Method vs Gap Test")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "k_hat_heatmap_Proposed_vs_Gap.png"))
    plt.close()

    # ---- Heatmaps: True K vs Estimated K for both methods ----
    max_khat = max(df["K_hat_F"].max(), df["K_hat_gap"].max())
    k_true_vals = sorted(df["K_true"].unique())
    k_hat_vals = range(1, max_khat + 1)

    # Proposed method frequency table
    freq_table_F = (
        df.groupby(["K_true", "K_hat_F"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=k_hat_vals, fill_value=0)
    ).T
    freq_table_F = freq_table_F.reindex(columns=k_true_vals, fill_value=0)

    # Gap test frequency table
    freq_table_Gap = (
        df.groupby(["K_true", "K_hat_gap"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=k_hat_vals, fill_value=0)
    ).T
    freq_table_Gap = freq_table_Gap.reindex(columns=k_true_vals, fill_value=0)

    # Plot side-by-side heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    sns.heatmap(
        freq_table_F, cmap="Blues", cbar_kws={'label': 'Frequency'},
        linewidths=0.5, ax=axes[0]
    )
    axes[0].set_title("Proposed Method")
    axes[0].set_xlabel("True K")
    axes[0].set_ylabel("Estimated K̂")

    sns.heatmap(
        freq_table_Gap, cmap="Blues", cbar_kws={'label': 'Frequency'},
        linewidths=0.5, ax=axes[1]
    )
    axes[1].set_title("Gap Test")
    axes[1].set_xlabel("True K")
    axes[1].set_ylabel("")  # no duplicate ylabel

    plt.suptitle("Heatmaps of Estimated vs True K", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "k_hat_heatmap_side_by_side.png"))
    plt.close()

