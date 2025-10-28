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




if __name__ == '__main__':
    master_rng = np.random.default_rng(0)
    penguins_raw = load_penguins()
    penguins = penguins_raw[(penguins_raw["sex"] == "female") & (penguins_raw.notna().all(axis=1)) & (
        penguins_raw["year"].between(2007, 2008))]
    labels = penguins["species"]
    X = penguins[["flipper_length_mm", "bill_length_mm"]].to_numpy()
    random.seed(0)
    #n = 30
    n = X.shape[0]
    true_K = 3
    total_alpha = 0.05
    num_trials = 50
    Ks = []
    alpha_list = np.full(n - 1, total_alpha / (n - 1))
    #ind = np.random.choice(X.shape[0], 30, replace=False)
    #subset_mat = X[ind, :]
    #subset_labels = labels.iloc[ind]


    def run_one_trial(seed):
        rng = np.random.default_rng(seed)
        K, _, _, _ = find_best_K_F(X, tau=0.1, alpha_list=alpha_list)
        return K


    seeds = master_rng.integers(0, 1e9, size=num_trials)
    Ks = Parallel(n_jobs=-1, backend="loky")(delayed(run_one_trial)(seed) for seed in seeds)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    output_dir = os.path.join(base_dir, "results", f"penguin_choose_K_{timestamp}")
    os.makedirs("results", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(Ks, bins=20, density=True, alpha=0.5, color="blue", edgecolor="black", label="K hat")
    plt.axvline(x=true_K, color='red', linestyle='--', linewidth=2, label=f"True K = {true_K}")
    plt.xlabel("K")
    plt.ylabel("Density")
    plt.title("Histogram of K_hat for tau=0.05")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "penguin_K_hist.png"))

    counter = Counter(Ks)
    counter_df = pd.DataFrame(counter.items(), columns=["K_hat", "Count"])
    counter_df = counter_df.sort_values("K_hat")
    counter_df.to_csv(os.path.join(output_dir, "k_hat_counter.csv"), index=False)