import os, sys, random
sys.path.append(os.path.abspath('../src'))
import numpy as np
import pandas as pd
from utils import *
from find_best_K import find_best_K_F, generate_alpha_list
from hierarchical_clustering_invariant import AgglomerativeClustering

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    output_dir = os.path.join(base_dir, "results/raw")
    os.makedirs(output_dir, exist_ok=True)

    n_each = 10
    delta = 8
    sigma = 1
    true_K = 3
    n = 30
    total_alpha = 0.05
    num_trials = 100

    # generate data
    X, y = generate_data_barbers(n_each, delta, sigma, true_mean=False, rng=rng)
    alpha_list = generate_alpha_list(n, total_alpha)

    Ks = []
    for rep in range(num_trials):
        print(f"Trial {rep+1}/{num_trials}...")
        child_rng = np.random.default_rng(rng.integers(1e9))
        K_hat, _, _, _ = find_best_K_F(X, tau=0.05, alpha_list=alpha_list, rng=child_rng)
        Ks.append(K_hat)

    Ks = np.array(Ks)
    np.savez(os.path.join(output_dir, "findK_results_tau005.npz"),
             X=X, y=y, Ks=Ks, true_K=true_K)
    print(f"Saved results to {os.path.join(output_dir, 'findK_results_tau005.npz')}")
