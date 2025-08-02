import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import random
from find_best_K import find_best_K_F
from datetime import datetime
import os

def generate_null_data(n, p, sigma=1):
    return np.random.normal(0, sigma, size=(n, p))
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
def run_single_trial(tau, n, p, total_alpha):
    X_null = generate_null_data(n=n, p=p)
    alpha_list = generate_alpha_list(n=n, total_alpha=total_alpha)
    K_hat, _, _ = find_best_K_F(X_null, tau=tau, alpha_list=alpha_list)
    return int(K_hat > 1)  # 1 if false rejection



if __name__ == "__main__":
    taus = [0,0.01, 0.05, 0.1, 0.5, 1.0]
    num_trials = 2000
    n = 30
    p = 10
    total_alpha = 0.05

    np.random.seed(0)
    random.seed(0)

    results = []

    for tau in taus:
        errors = Parallel(n_jobs=-1)(
            delayed(run_single_trial)(tau, n, p, total_alpha)
            for _ in range(num_trials)
        )
        fwer = sum(errors) / num_trials
        tau_label = "naive" if tau == 0 else tau
        results.append({
            "tau": tau_label,
            "FWER": fwer,
            "num_trials": num_trials
        })

    df_results = pd.DataFrame(results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"results_fwer_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "fwer_results.csv")
    df_results.to_csv(output_file, index=False)

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
    # naive 点
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
