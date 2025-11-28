import sys, os
sys.path.append(os.path.abspath('../src'))
import glob
import pandas as pd
import numpy as np
import argparse
from palmerpenguins import load_penguins
from alg2_one_trial2 import gap_statistic
from find_best_K import find_best_K_F, generate_alpha_list_exp, get_labels_at_K, find_best_K_chi
from hierarchical_clustering_invariant import *

def split_data_fission(X, gamma, seed=None):
    rng = np.random.default_rng(seed)
    n, p = X.shape
    sigma2 = X.var(ddof=1)
    W = rng.normal(loc=0, scale=np.sqrt(gamma * sigma2), size=(n, p))
    U = X + W
    V = X - W / gamma
    return U, V
def split_data(X, p,  seed=None):
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)

    k = int(np.floor(p * n))
    idx1 = idx[:k]
    idx2 = idx[k:]

    return X[idx1], X[idx2]

def split_data_cluster(X, p, labels, seed=None):
    rng = np.random.default_rng(seed)
    labels = np.array(labels)

    idx1 = []
    idx2 = []

    for c in np.unique(labels):
        idx_c = np.where(labels == c)[0]
        idx_c = rng.permutation(idx_c)

        k = int(np.floor(p * len(idx_c)))
        idx1.extend(idx_c[:k])
        idx2.extend(idx_c[k:])

    idx1 = np.array(idx1)
    idx2 = np.array(idx2)

    return X[idx1], X[idx2]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--trial_id", type=int, default=None)
    args = parser.parse_args()

    penguins_raw = load_penguins()
    penguins = penguins_raw[(penguins_raw["sex"] == "female") & (penguins_raw.notna().all(axis=1)) & (
        penguins_raw["year"].between(2007, 2008))]
    labels = penguins["species"]
    X = penguins[["flipper_length_mm", "bill_length_mm"]].to_numpy()
    #n = X.shape[0]

    data_seed = int(args.gamma * 10000)
    #X_fit, X_inf = split_data_fission(X, args.gamma, data_seed)
    #X_fit, X_inf = split_data(X, args.gamma, data_seed)
    X_fit, X_inf = split_data_cluster(X, args.gamma, labels, data_seed)
    n = X_fit.shape[0]
    algo_seed = int(args.trial_id)

    total_alpha = 0.05
    alpha_list = generate_alpha_list_exp(n, total_alpha, decay_rate=0.5)

    K, _, _, _ = find_best_K_F(
        X_fit,
        tau=0.1,
        alpha_list=alpha_list,
        linkage="complete",
        total_alpha=total_alpha,
        n_threshold=int(0.4 * n),
        hard_threshold=int(0.05 * n),
        seed=algo_seed,
    )

    K_hat_gap = gap_statistic(X_fit, K_max=30, B=50)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    output_dir = os.path.join(base_dir, "results/raw/penguins_split_cluster")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f"gamma{args.gamma}_trial_{algo_seed}.csv")
    pd.DataFrame(
        {
            "gamma": [args.gamma],
            "trial_id": [algo_seed],
            "K_hat": [K],
            "K_hat_gap": [K_hat_gap],
        }
    ).to_csv(csv_path, index=False)

    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    if args.trial_id == 0:
        np.save(os.path.join(data_dir, f"gamma{args.gamma}_Xfit.npy"), X_fit)
        np.save(os.path.join(data_dir, f"gamma{args.gamma}_Xinf.npy"), X_inf)
