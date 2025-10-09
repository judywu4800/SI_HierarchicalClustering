import sys, os
sys.path.append(os.path.abspath('../../src'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from utils import generate_null_data, check_p_value_uniformity
import random
from find_best_K import find_best_K_F, find_best_K_chi
from datetime import datetime
import os


def generate_alpha_list(n=30, total_alpha=0.05, seed=42):
    if n % 3 != 0:
        raise ValueError("n must be divisible by 3.")

    np.random.seed(seed)

    length = n - 1
    group_size = length // 3

    # Generate unnormalized values in each group
    large = np.random.uniform(0.003, 0.008, size=group_size)
    medium = np.random.uniform(0.001, 0.003, size=group_size)
    small = np.random.uniform(1e-5, 5e-4, size=length - 2 * group_size)

    # Combine and normalize
    alpha_raw = np.concatenate([large, medium, small])
    alpha_list = total_alpha * alpha_raw / np.sum(alpha_raw)

    return alpha_list
def run_single_trial(tau, n, p, sigma, total_alpha):
    mu = np.zeros(p)
    X_null = generate_null_data(n,p, mu,sigma)
    alpha_list = generate_alpha_list(n=n, total_alpha=total_alpha)
    #alpha_list = np.ones((n-1))/(n-1)*0.05

    K_hat, _, _ = find_best_K_F(X_null, tau=tau, alpha_list=alpha_list.copy(), total_alpha=total_alpha)
    #K_hat, _, _ = find_best_K_chi(X_null, tau=tau, alpha_list=alpha_list.copy(), total_alpha=total_alpha)
    return int(K_hat > 1), K_hat  # 1 if false rejection


if __name__ == "__main__":
    taus = [0, 0.01,0.05, 0.1,0.5,1]
    num_trials = 100
    n = 30
    p = 10
    sigma = 1
    total_alpha = 0.05

    results = []
    flat_k_hats = []

    for tau in taus:
        trial_results = Parallel(n_jobs=-1)(
            delayed(run_single_trial)(tau, n, p,sigma, total_alpha)
            for _ in range(num_trials)
        )

        errors, K_hats = zip(*trial_results)
        fwer = sum(errors) / num_trials

        tau_label = "naive" if tau == 0 else tau
        results.append({
            "tau": tau_label,
            "FWER": fwer,
            "num_trials": num_trials
        })

        flat_k_hats.extend([{"tau": tau_label, "K_hat": k} for k in K_hats])

    df_results = pd.DataFrame(results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    output_dir = os.path.join(base_dir, "results", f"results_fwer_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "fwer_results.csv")
    df_results.to_csv(output_file, index=False)

    df_k_hats = pd.DataFrame(flat_k_hats)
    k_hats_file = os.path.join(output_dir, "k_hats_flat.csv")
    df_k_hats.to_csv(k_hats_file, index=False)

    # Plotting
    labels = df_results["tau"].astype(str).tolist()
    x_positions = np.arange(len(labels))

    plt.figure(figsize=(8, 5))
    non_naive_mask = df_results["tau"] != "naive"
    plt.plot(
        x_positions[1:],
        df_results.loc[non_naive_mask, "FWER"],
        marker='o',
        color='blue',
        label="Empirical FWER"
    )
    plt.scatter(
        x_positions[0],
        df_results.loc[df_results["tau"] == "naive", "FWER"],
        color='orange',
        marker='s',
        s=100,
        label='Naive'
    )

    plt.xticks(x_positions, labels)
    plt.axhline(y=total_alpha, color='red', linestyle='--', label=f"Alpha = {total_alpha}")
    plt.xlabel("Tau")
    plt.ylabel("FWER")
    plt.title("FWER vs Tau (Null Data)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plot_file = os.path.join(output_dir, "fwer_plot.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()
