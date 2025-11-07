import os, sys, random
sys.path.append(os.path.abspath('../src'))
import numpy as np
import pandas as pd
from utils import *
from find_best_K import find_best_K_F, generate_alpha_list
from hierarchical_clustering_invariant import AgglomerativeClustering

if __name__ == "__main__":
    batch_id = int(os.environ.get("BATCH_ID", 0))        # e.g., from job array index
    num_batches = int(os.environ.get("NUM_BATCHES", 10)) # total number of batches
    reps_per_batch = int(os.environ.get("REPS_PER_BATCH", 10))  # trials per batch

    rng = np.random.default_rng(0)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    output_dir = os.path.join(base_dir, "results/raw")
    os.makedirs(output_dir, exist_ok=True)

    delta = 8
    sigma = 1
    true_K = 2
    n = 30
    n_each = n//true_K
    total_alpha = 0.05
    num_trials = 100

    # generate data
    X, y = generate_data_barbers(n_each, delta, sigma, n_clusters=true_K, true_mean=False, rng=rng)
    alpha_list = generate_alpha_list(n, total_alpha)

    Ks = []
    for rep in range(reps_per_batch):
        print(f"[Batch {batch_id}] Trial {rep + 1}/{reps_per_batch}...")
        child_rng = np.random.default_rng(rng.integers(1e9))
        K_hat, _, _, _ = find_best_K_F(X, tau=0.05, alpha_list=alpha_list, rng=child_rng)
        Ks.append(K_hat)

    Ks = np.array(Ks)
    output_file = os.path.join(output_dir, f"findK_results_tau005_K{true_K}_batch{batch_id}.npz")
    np.savez(output_file, X=X, y=y, Ks=Ks, true_K=true_K)
    print(f"Saved batch results to {output_file}")