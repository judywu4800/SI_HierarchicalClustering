import sys, os
sys.path.append(os.path.abspath('../../src'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.datasets import make_blobs
from utils import generate_3cluster_data, generate_data_barbers
from joblib import Parallel, delayed
import random
from hierarchical_clustering_invariant import *
from scipy.cluster.hierarchy import linkage, fcluster
from find_best_K import find_best_K_chi

def generate_clustered_data(n=30, p=10, delta=3.0, sigma=1.0, random_state=None):

    rng = np.random.default_rng(random_state)
    n_per_cluster = n // 3

    mu = np.array([
        [-delta, 0],
        [delta, 0],
        [0, delta]
    ])

    X_signal = np.vstack([
        rng.normal(mu[k], sigma, size=(n_per_cluster, 2))
        for k in range(3)
    ])
    X_noise = rng.normal(0, 3, size=(n, p - 2))

    X = np.hstack([X_signal, X_noise])
    labels = np.repeat(np.arange(3), n_per_cluster)
    return X, labels


def generate_3cluster_data_varsize(n=50, p=2, delta=1.0, sigma=1.0,
                                   cluster_sizes=None, random_state=None, return_labels=True):
    rng = np.random.default_rng(random_state)

    # handle cluster sizes
    if cluster_sizes is None:
        if n % 3 != 0:
            raise ValueError("If cluster_sizes is None, n must be divisible by 3.")
        cluster_sizes = (n // 3, n // 3, n // 3)
    else:
        if len(cluster_sizes) != 3:
            raise ValueError("cluster_sizes must be a tuple/list of 3 integers.")
        if sum(cluster_sizes) != n:
            raise ValueError(f"sum(cluster_sizes)={sum(cluster_sizes)} must equal n={n}.")

    # define means (equilateral triangle in 2D, generalized in higher dimensions)
    mu1 = np.zeros(p)
    mu1[0] = -delta / 2

    mu2 = np.zeros(p)
    mu2[-1] = np.sqrt(3) * delta / 2

    mu3 = np.zeros(p)
    mu3[0] = delta / 2

    X1 = rng.normal(loc=mu1, scale=sigma, size=(cluster_sizes[0], p))
    X2 = rng.normal(loc=mu2, scale=sigma, size=(cluster_sizes[1], p))
    X3 = rng.normal(loc=mu3, scale=sigma, size=(cluster_sizes[2], p))

    X = np.vstack([X1, X2, X3])
    labels = np.array([0]*cluster_sizes[0] + [1]*cluster_sizes[1] + [2]*cluster_sizes[2])

    if return_labels:
        return X, labels
    else:
        return X
def find_best_K_F(X, tau, alpha_list, n_threshold=2, linkage="complete", total_alpha=0.05):
    n = np.shape(X)[0]
    if not np.isclose(np.sum(alpha_list), total_alpha):
        raise ValueError(
            f"Alpha list should sum up to total alpha = {total_alpha}, " 
            f"but got {np.sum(alpha_list):.4f} instead.")
    if len(alpha_list) != (n - 1):
        raise ValueError("The length of alpha_list should be equal to n - 1.")

    p_values = []
    alpha_seq = []
    K_hat = 1

    model = AgglomerativeClustering(X, tau=tau, n_clusters=1, linkage=linkage)
    model.fit()
    winning_nodes = list(model.existing_clusters_log.keys())  # to get all merges

    for t, winning_pair in enumerate(winning_nodes):
        node1, node2 = winning_pair
        node = node1.parent
        n1, n2 = len(node1.points), len(node2.points)

        if (1 / n1 + 1 / n2) >= n_threshold:
            alpha = np.min(alpha_list)  # conservative for smaller clusters
            idx = np.argmin(alpha_list)
            if (n1 + n2) == 2:
                pval = 1  # edge case
            else:
                pval, _, _ = model.merge_inference_F_grid(node, grid_width=180, ncoarse=50, ngrid=1000)
        else:
            alpha = np.max(alpha_list)  # more power for larger clusters
            idx = np.argmax(alpha_list)
            pval, _, _ = model.merge_inference_F_grid(node, grid_width=180, ncoarse=50, ngrid=1000)

        alpha_list = np.delete(alpha_list, idx)
        alpha_seq.append(alpha)
        p_values.append(pval)

        if pval < alpha:
            K_hat = n - t
            return (K_hat, p_values, alpha_seq, model)

    return (K_hat, p_values, alpha_seq, model)
def get_labels_at_K(model, K):
    for winning_pair, clusters in model.existing_clusters_log.items():
        if len(clusters) == K:
            labels = np.empty(model.n_samples, dtype=int)
            for cid, cluster in enumerate(clusters):
                for idx in cluster.points:
                    labels[idx] = cid
            return labels
    raise ValueError(f"No step found with {K} clusters")

def clusters_to_labels(clusters, n_samples):
    labels = np.empty(n_samples, dtype=int)
    for cid, cluster in enumerate(clusters):
        for idx in cluster.points:
            labels[idx] = cid
    return labels

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

def check_preserve(true_labels, est_labels):
    for k in np.unique(est_labels):
        if len(set(true_labels[est_labels == k])) > 1:
            return False
    return True

def gap_statistic(X, K_max=30, B=50, method="complete", true_labels = None, random_state=None):
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
            best_K = k
            break
    else:
        best_K = K_max

    # P(K_hat <= K^* & no cross cluster merge occurred)
    preserve = None
    if true_labels is not None:
        Z = linkage(X, method=method)
        #no_cross is only True when K_hat <=3 and the estimated clusters only contain points from same true clusters
        labels_est = fcluster(Z,max(best_K,3),criterion="maxclust")
        preserve = check_preserve(true_labels, labels_est)

    return best_K, gap, sk, preserve

# ------ Varying cluster separation------
def one_replication_std(delta,n=30, tau=0.1, total_alpha=0.05,
                    K_max=30, B=50, method="complete"):
    #X, labels = make_blobs(n_samples=n, n_features=p, centers=K_true, cluster_std=cl_sd)
    #X, labels = generate_data_barbers(n_each = 10, delta = delta, sigma=1)
    #X, labels = generate_3cluster_data(30,5,delta, 1)
    #X, labels = generate_3cluster_data_varsize(n= n, p=30,delta=delta, sigma = 1.0, cluster_sizes=[10,10,10])
    sigma = np.array([
        [1.0, 0.3],
        [0.3, 1.0]
    ])
    X, labels = generate_data_barbers(10, delta=delta, sigma=sigma)
    # --- Proposed method---
    alpha_list = np.full(n - 1, total_alpha / (n - 1))
    K_hat_F, _, _,model = find_best_K_chi(X, sigma = sigma, tau=tau, alpha_list=alpha_list, total_alpha=total_alpha)

    #labels_est = get_labels_at_K(model, max(K_hat_F, 3))
    #preserve = check_preserve(labels, labels_est)

    # --- Gap test---
    #K_hat_gap, _, _, no_cross_g = gap_statistic(X, K_max=K_max, B=B, method=method, true_labels=labels)
    K_hat_gap, _, _, _ = gap_statistic(X, K_max=K_max, B=B, method=method, true_labels=labels)
    return K_hat_F, K_hat_gap #, preserve, no_cross_g
def simulate_results_std(delta_list, n_rep=100, n_jobs=-1, **kwargs):
    """
    Run simulations in parallel for different true K.
    Returns dict: {K_true: {"F": [...], "Gap": [...]}}
    """
    results = {}
    for delta in delta_list:
        pairs = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(one_replication_std)(delta, **kwargs) for _ in range(n_rep)
        )
       # k_hats_F, k_hats_gap, no_cross, no_cross_g = zip(*pairs)
        k_hats_F, k_hats_gap = zip(*pairs)
        #results[delta] = {"Proposed Method": list(k_hats_F), "Gap Test": list(k_hats_gap), "Preserve": no_cross, "Preserve_Gap": no_cross_g}
        results[delta] = {"Proposed Method": list(k_hats_F), "Gap Test": list(k_hats_gap)}
    return results




if __name__ == "__main__":
    random.seed(0)
    results_F = {}
    #K_list = [1,3,5,7,9,11]
    delta_list = [6,8,10,12,14]
    n = 30
    p = 2
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    output_dir = os.path.join(base_dir, "results", f"k_hat_boxplot_{timestamp}_n={n}_p={p}")
    os.makedirs("results", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    results = simulate_results_std(delta_list, n_rep=100, n_jobs=-1)
    rows = []
    for sd, result in results.items():
        for kf, kg, pr,pg in zip(result["Proposed Method"], result["Gap Test"], result["Preserve"], result["Preserve_Gap"]):
            rows.append((sd, kf, kg, pr,pg))

    df = pd.DataFrame(rows, columns=["delta", "K_hat_F", "K_hat_gap", "Preserve", "Preserve_Gap"])
    df.to_csv(os.path.join(output_dir, "k_hat_results.csv"), index=False)

    '''
    result_corollary = (
        df.groupby("delta")
        .apply(lambda g: pd.Series({
            "P(K<=3| Preserve=True)": ((g["K_hat_F"] <= 3) & (g["Preserve"])).sum() / (g["Preserve"]).sum()
            if (g["Preserve"]).sum()>0 else np.nan,
            #"P(K<=3| Preserve=False)": ((g["K_hat_F"] <= 3) & (~g["Preserve"])).sum() / (~g["Preserve"]).sum() if (~g["Preserve"]).sum()>0 else np.nan,
            "P(K>3| Preserve=True)": ((g["K_hat_F"] > 3) & (g["Preserve"])).sum() / (g["Preserve"]).sum() if (g["Preserve"]).sum()>0 else np.nan,
            "P(Preserve = False)": (~g["Preserve"]).sum() / len(g),
            "P(K<=3| Preserve_gap=True)": ((g["K_hat_gap"] <= 3) & (g["Preserve_Gap"])).sum() / (g["Preserve_Gap"]).sum()
            if (g["Preserve_Gap"]).sum()>0 else np.nan,
            #"P(K<=3| Preserve_gap=False)": ((g["K_hat_gap"] <= 3) & (~g["Preserve_Gap"])).sum() / (~g["Preserve_Gap"]).sum() if (~g["Preserve_Gap"]).sum()>0 else np.nan,
            "P(Preserve_Gap= False)": (~g["Preserve_Gap"]).sum() / len(g)
        }))
        .reset_index()
    )

    result_corollary.to_csv(os.path.join(output_dir, "ratio_by_sd.csv"), index=False)    
    '''


    # ---- Boxplot comparison ----
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="delta", y="K_hat_F", data=df, color="skyblue", width=0.5)
    sns.boxplot(x="delta", y="K_hat_gap", data=df, color="lightcoral", width=0.3)
    plt.axhline(y=3, color="red", linestyle="--", linewidth=2)
    plt.xlabel("Cluster separation delta")
    plt.ylabel("Estimated K̂")
    plt.title("Comparison of Estimated Clusters: Proposed Method vs Gap Test")
    plt.legend(handles=[
        plt.Line2D([0], [0], color="skyblue", lw=8, label="Proposed Method"),
        plt.Line2D([0], [0], color="lightcoral", lw=8, label="Gap Test")
    ])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "k_hat_boxplot_Proposed_vs_Gap.png"))
    plt.close()

    # ----- Heatmap comparison -----
    k_hat_vals = range(1, 31)
    df["delta"] = df["delta"].astype(int)
    # Proposed Method
    freq_table_F = (
        df.groupby(["K_hat_F", "delta"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=k_hat_vals, columns=delta_list, fill_value=0)
    )

    # Gap Test
    freq_table_Gap = (
        df.groupby(["K_hat_gap", "delta"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=k_hat_vals, columns=delta_list, fill_value=0)
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 10), sharey=True)

    sns.heatmap(
        freq_table_F, cmap="Blues", cbar_kws={'label': 'Frequency'},
        linewidths=0.5, annot=True, fmt="d", ax=axes[0]
    )
    axes[0].set_title("Proposed Method")
    axes[0].set_xlabel("Cluster separation delta")
    axes[0].set_ylabel("Estimated K̂")

    sns.heatmap(
        freq_table_Gap, cmap="Blues", cbar_kws={'label': 'Frequency'},
        linewidths=0.5, annot=True, fmt="d", ax=axes[1]
    )
    axes[1].set_title("Gap Test")
    axes[1].set_xlabel("Cluster separation delta")
    axes[1].set_ylabel("")

    plt.suptitle("Heatmaps: Estimated K̂ vs Cluster separation", fontsize=14)
    plt.savefig(os.path.join(output_dir, "Heatmap_side_by_side.png"))
    plt.close()

    '''
# ---- Line plot: Compare P(K<=3 | Preserve=True) vs 1 - alpha ----
    alpha = 0.05
    plt.figure(figsize=(8, 6))

    plt.plot(
        result_corollary["delta"],
        result_corollary["P(K<=3| Preserve=True)"],
        marker="o", linewidth=2, label="Proposed Method"
    )

    plt.plot(
        result_corollary["delta"],
        result_corollary["P(K<=3| Preserve_gap=True)"],
        marker="s", linewidth=2, label="Gap Test"
    )

    plt.axhline(y=1 - alpha, color="red", linestyle="--", linewidth=2, label="1 - α (theoretical lower bound)")

    plt.xlabel("Cluster separation δ", fontsize=12)
    plt.ylabel("Conditional probability  P( K̂ ≤ 3 | Preserve=True )", fontsize=12)
    plt.title("Verification of Corollary 1 across cluster separations", fontsize=14)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "corollary_verification_lineplot.png"))
    plt.close()     
    '''
