import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from hierarchical_clustering import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.special import gamma
from sklearn import cluster
from joblib import Parallel, delayed
from datetime import datetime

def generate_3cluster_data(n=30, p=2, delta=1.0, sigma=1.0, random_state=None, return_labels=True):
    if n % 3 != 0:
        raise ValueError("n must be divisible by 3.")
    rng = np.random.default_rng(random_state)
    n_cluster = n // 3

    # Cluster 0 at -delta
    mu1 = np.zeros(p)
    mu1[0] = -delta/2

    # Cluster 1 at 0
    mu2 = np.zeros(p)
    mu2[-1] = np.sqrt(3) * delta / 2

    # Cluster 2 at +delta
    mu3 = np.zeros(p)
    mu3[0] = delta/2

    X1 = rng.normal(loc=mu1, scale=sigma, size=(n_cluster, p))
    X2 = rng.normal(loc=mu2, scale=sigma, size=(n_cluster, p))
    X3 = rng.normal(loc=mu3, scale=sigma, size=(n_cluster, p))

    X = np.vstack([X1, X2, X3])
    labels = np.array([0] * n_cluster + [1] * n_cluster + [2] * n_cluster)

    if return_labels:
        return X, labels
    else:
        return X



def single_tau_power(tau, n, p, sigma, delta, alpha, num_trials, max_attempts=50000):
    p_values = []
    recovery = 0
    trial_count = 0

    while len(p_values) < num_trials:
        trial_count += 1
        X, true_labels = generate_3cluster_data(n=n, p=p, delta=delta, sigma=sigma)
        model = AgglomerativeClustering(X, tau=tau, n_clusters=2, linkage="single")
        model.fit()

        winning_nodes = list(model.existing_clusters_log.keys())
        key = winning_nodes[-1]
        c1, c2 = key[0], key[1]
        c1_points = c1.points
        c2_points = c2.points

        c1_true_clusters = set(true_labels[c1_points])
        c2_true_clusters = set(true_labels[c2_points])

        if len(c1_true_clusters) == 1 and len(c2_true_clusters) == 1:
            recovery += 1
            node = c1.parent
            p_val, _, _ = model.merge_inference(node, grid_width=20, ncoarse=20, ngrid=1000, sd=sigma)
            p_values.append(p_val)

        if trial_count > max_attempts:
            print(f"Warning: Too few matching merges at tau={tau}")
            break

    power = np.mean(np.array(p_values) < alpha)
    recovery_prob = recovery / trial_count
    success = len(p_values) == num_trials
    return tau, power, recovery_prob, success

def check_power_multi_tau_parallel(n, p, sigma, tau_list, delta=10.0, alpha=0.05,
                                    num_trials=300, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs)(
        delayed(single_tau_power)(tau, n, p, sigma, delta, alpha, num_trials)
        for tau in tau_list
    )

    # Unpack results
    power_results_sel = {tau: power for tau, power, _, _ in results}
    recovery_results = {tau: rec for tau, _, rec, _ in results}
    full = [success for _, _, _, success in results]

    # Plotting (same as before)
    tau_vals = np.array(tau_list)
    power_vals = [power_results_sel[tau] for tau in tau_vals]
    recovery_vals = [recovery_results[tau] for tau in tau_vals]

    fig, ax1 = plt.subplots(figsize=(8, 6))

    color_power = 'tab:blue'
    ax1.set_xlabel("Tau (Randomization Level)")
    ax1.set_ylabel("Conditional Power", color=color_power)
    ax1.tick_params(axis='y', labelcolor=color_power)
    ax1.set_ylim(0, 1)

    if 0 in tau_vals:
        naive_idx = np.where(tau_vals == 0)[0][0]
        ax1.scatter(tau_vals[naive_idx], power_vals[naive_idx], color='orange', marker='s', s=100, label="Naive Power (τ=0)", zorder=5)

        tau_random = tau_vals[tau_vals != 0]
        power_random = [power_results_sel[t] for t in tau_random]
        ax1.plot(tau_random, power_random, marker='o', color=color_power, label="Randomized Power (τ>0)")
    else:
        ax1.plot(tau_vals, power_vals, marker='o', color=color_power, label="Conditional Power")

    ax2 = ax1.twinx()
    color_recovery = 'tab:red'
    ax2.set_ylabel("Recovery Probability", color=color_recovery)
    ax2.tick_params(axis='y', labelcolor=color_recovery)
    ax2.set_ylim(0, 1)

    if 0 in tau_vals:
        ax2.scatter(tau_vals[naive_idx], recovery_vals[naive_idx], color='darkorange', marker='D', s=100, label="Naive Recovery (τ=0)", zorder=5)
        recovery_random = [recovery_results[t] for t in tau_random]
        ax2.plot(tau_random, recovery_random, marker='s', linestyle='--', color=color_recovery, label="Randomized Recovery (τ>0)")
    else:
        ax2.plot(tau_vals, recovery_vals, marker='s', linestyle='--', color=color_recovery, label="Recovery Probability")

    plt.title("Conditional Power and Recovery Probability vs. Tau")
    fig.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.5)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    plt.legend(h1 + h2, l1 + l2, loc="upper right")
    plt.show()

    return power_results_sel, recovery_results, full


if __name__ == "__main__":
    import os

    n = 30
    p = 2
    sigma = 1
    tau_list = [0,0.1, 0.25, 0.5, 1,1.5,2,5,10]
    delta = 5.0
    layer = -1
    alpha = 0.05
    num_trials = 200
    n_jobs = -1


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.expanduser(f"power_results_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)


    power_results_sel, recovery_results, full = check_power_multi_tau_parallel(
        n=n, p=p, sigma=sigma, tau_list=tau_list,
        delta=delta, alpha=alpha, num_trials=num_trials,
        n_jobs=n_jobs
    )

    # Save results to CSV
    df = pd.DataFrame({
        "Tau": tau_list,
        "Conditional Power": [power_results_sel[t] for t in tau_list],
        "Recovery Probability": [recovery_results[t] for t in tau_list],
        "Successful Trials": full
    })
    df.to_csv(os.path.join(output_dir, "power_and_recovery.csv"), index=False)

    # Re-plot and save figure
    plt.figure(figsize=(8, 6))
    fig, ax1 = plt.subplots()

    tau_vals = np.array(tau_list)
    power_vals = [power_results_sel[t] for t in tau_vals]
    recovery_vals = [recovery_results[t] for t in tau_vals]

    color_power = 'tab:blue'
    ax1.set_xlabel("Tau (Randomization Level)")
    ax1.set_ylabel("Conditional Power", color=color_power)
    ax1.tick_params(axis='y', labelcolor=color_power)
    ax1.set_ylim(0, 1)

    if 0 in tau_vals:
        naive_idx = np.where(tau_vals == 0)[0][0]
        ax1.scatter(tau_vals[naive_idx], power_vals[naive_idx], color='orange', marker='s', s=100,
                    label="Power: Naive", zorder=5)

        tau_random = tau_vals[tau_vals != 0]
        power_random = [power_results_sel[t] for t in tau_random]
        ax1.plot(tau_random, power_random, marker='o', color=color_power, label="Power: Randomized")
    else:
        ax1.plot(tau_vals, power_vals, marker='o', color=color_power, label="Conditional Power")

    ax2 = ax1.twinx()
    color_recovery = 'tab:red'
    ax2.set_ylabel("Recovery Probability", color=color_recovery)
    ax2.tick_params(axis='y', labelcolor=color_recovery)
    ax2.set_ylim(0, 1)

    if 0 in tau_vals:
        ax2.scatter(tau_vals[naive_idx], recovery_vals[naive_idx], color='darkorange', marker='D', s=100,
                    label="Recovery: Naive", zorder=5)
        recovery_random = [recovery_results[t] for t in tau_random]
        ax2.plot(tau_random, recovery_random, marker='s', linestyle='--', color=color_recovery,
                 label="Recovery: Randomized")
    else:
        ax2.plot(tau_vals, recovery_vals, marker='s', linestyle='--', color=color_recovery,
                 label="Recovery Probability")

    plt.title("Conditional Power and Recovery Probability vs. Tau")
    fig.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.5)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    plt.legend(h1 + h2, l1 + l2, loc="upper right")

    fig.savefig(os.path.join(output_dir, "power_recovery_plot.png"))
    plt.close(fig)