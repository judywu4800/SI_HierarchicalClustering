import numpy as np
from sklearn.datasets import make_blobs
from hierarchical_clustering_invariant import AgglomerativeClustering
from utils import *
from find_best_K import *
import warnings
import matplotlib.pyplot as plt
from collections import Counter
import random
from datetime import datetime
import os

def _run_one_tau(tau, X, true_K, outdir, total_alpha=0.05,
                 equal_alpha=True, num_trials = 100):
    n = X.shape[0]
    if equal_alpha:
        alpha_list = np.full(n-1, total_alpha/(n-1))
    else:
        alpha_list = generate_alpha_list(n=n, total_alpha=total_alpha)

    Ks = []
    for _ in range(num_trials):
        K_hat, _, _ = find_best_K_F(X, tau=tau, alpha_list=alpha_list)
        Ks.append(K_hat)

    counter = Counter(Ks)
    min_K = 0
    max_K = 30
    full_counts = {k: counter.get(k, 0) for k in range(min_K, max_K + 1)}
    hist_df = pd.DataFrame({"K_value": list(full_counts.keys()),
                            "Count": list(full_counts.values())})
    #csv_path = os.path.join(outdir, f"hist_counts_tau={tau}_equal_{equal_alpha}.csv")
    #hist_df.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 6))
    plt.hist(Ks, bins=range(min_K, max_K + 2), density=True,
             alpha=0.5, edgecolor="black", label=r"$\hat{K}$")
    plt.axvline(x=true_K, color='red', linestyle='--', linewidth=2, label=f"True K = {true_K}")
    plt.xlabel("K")
    plt.ylabel("Density")
    plt.title(f"Histogram of K_hat (tau={tau})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    png_path = os.path.join(outdir, f"hist_tau={tau}_equal_{equal_alpha}.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()

    hist_df["tau"] = tau
    return hist_df


def get_histograms_parallel_tau(
    n, true_K, tau_list, outdir, total_alpha=0.05, num_trials=100,
    equal_alpha=True, cluster_std=0.5, n_jobs=-1):

    X, _ = make_blobs(n_samples=n, centers=true_K, cluster_std=cluster_std)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_one_tau)(
            tau, X, true_K, outdir, total_alpha=total_alpha,
            equal_alpha=True, num_trials = num_trials)
        for tau in tau_list
    )

    long_df = pd.concat(results, ignore_index=True)
    pivot = long_df.pivot_table(index="K_value", columns="tau", values="Count",
                                aggfunc="sum", fill_value=0).sort_index()

    max_indices = pivot.idxmax(axis=0)

    pivot.loc['Max Row'] = max_indices
    combined_csv = os.path.join(outdir, f"hist_counts_combined_equal_{equal_alpha}.csv")
    pivot.to_csv(combined_csv)

    return pivot, combined_csv
if __name__ == '__main__':
    n=30
    true_K=3
    tau_list = [0,0.005,0.01,0.05,0.1,0.5,1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("results", f"bestK_results_{timestamp}")
    os.makedirs(outdir, exist_ok=True)

    get_histograms_parallel_tau(n, true_K, tau_list, outdir, total_alpha=0.05, num_trials=100,equal_alpha=False)

