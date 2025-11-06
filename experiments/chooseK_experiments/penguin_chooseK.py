import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import matplotlib.patches as mpatches
from find_best_K import find_best_K_F, generate_alpha_list, get_labels_at_K, find_best_K_chi
from hierarchical_clustering_invariant import *
from palmerpenguins import load_penguins
from joblib import Parallel, delayed
from collections import Counter
from datetime import datetime
import os




if __name__ == '__main__':
    master_rng = np.random.default_rng(0)
    penguins_raw = load_penguins()
    penguins = penguins_raw[(penguins_raw["sex"] == "female") & (penguins_raw.notna().all(axis=1)) & (
        penguins_raw["year"].between(2007, 2008))]
    labels = penguins["species"]
    X = penguins[["flipper_length_mm", "bill_length_mm"]].to_numpy()
    #ind = np.random.choice(X.shape[0], 12, replace=False)
    #X = X[ind, :]
    #n = 12
    n = X.shape[0]
    true_K = 3
    total_alpha = 0.05
    num_trials = 10
    Ks = []
    alpha_list = generate_alpha_list(n, 0.05)#np.full(n - 1, total_alpha / (n - 1))


    def run_one_trial(seed):
        rng = np.random.default_rng(seed)
        K, _, _, _ = find_best_K_F(X, tau=0.1, alpha_list=alpha_list, rng=rng)
        return K



    seeds = master_rng.integers(0, 1e9, size=num_trials)

    start_time = datetime.now()
    Ks = Parallel(n_jobs=-1, backend="loky", verbose=5)(delayed(run_one_trial)(seed) for seed in seeds)
    end_time = datetime.now()
    elapsed_minutes = (end_time - start_time).total_seconds() / 60
    print(f"Total runtime: {elapsed_minutes:.2f} minutes")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    output_dir = os.path.join(base_dir, "results/raw/penguins")
    os.makedirs(output_dir, exist_ok=True)

    counter = Counter(Ks)
    counter_df = pd.DataFrame(counter.items(), columns=["K_hat", "Count"])
    counter_df = counter_df.sort_values("K_hat")
    counter_df.to_csv(os.path.join(output_dir, "k_hat_counter.csv"), index=False)

    plt.figure(figsize=(8, 5))
    plt.hist(Ks, bins=20, density=True, alpha=0.5, color="blue", edgecolor="black", label="K hat")
    plt.axvline(x=true_K, color='red', linestyle='--', linewidth=2, label=f"True K = {true_K}")
    plt.xlabel("K")
    plt.ylabel("Density")
    plt.title("Histogram of K_hat for tau=0.05")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "penguin_K_hist.png"))

