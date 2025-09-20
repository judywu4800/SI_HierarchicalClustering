import numpy as np

from utils import *
from datetime import datetime
from pygam import LogisticGAM, s
import random


if __name__ == "__main__":
    import os
    random.seed(1)
    n = 30
    p = 10
    sigma = 1
    #tau=0.1
    tau_list = [0,0.01, 0.05, 0.1]
    #tau_list = [0.01]
    #deltas = [8,10]
    #deltas = [0,2,4,6,8,10]
    deltas = np.linspace(5,20,9)
    alpha = 0.05
    num_trials = 2000
    n_jobs = -1

    df_trials = check_power_es_multi_tau_delta_random_pair(n, p, sigma, tau_list, deltas, alpha, num_trials, n_jobs)


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"power_es_results_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    #reject_results, recovery_results, es_results, df_trials = check_power_multi_tau_delta_random_pair(
    #    n, p, sigma, tau_list, deltas, alpha, num_trials, n_jobs
    #)

    csv_path = os.path.join(output_dir, "reject_es.csv")
    df_trials.to_csv(csv_path, index=False)


    plt.figure(figsize=(10, 6))

    for tau, g in df_trials.groupby("tau"):
        X = g[["effect_size"]].values
        y = g["reject"].values

        gam = LogisticGAM(s(0) + s(1)).fit(X, y)

        grid = np.linspace(0.0, 3.0, 500)
        power_hat = gam.predict_mu(grid)

        plt.plot(grid, power_hat, lw=2, label=f"tau={tau}")

    plt.xlabel("Delta (effect size)")
    plt.ylabel("Power (Pr[Reject=1 | Δ])")
    plt.title("Power vs Delta for different tau")
    plt.legend(title="Tau")
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.savefig(os.path.join(output_dir, "power_vs_es_plot.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    all_es = df_trials["effect_size"].values
    bins = np.histogram_bin_edges(all_es, bins=30)

    for tau, g in df_trials.groupby("tau"):
        es = g["effect_size"].values
        plt.hist(es, bins=bins, density=True, histtype="step", linewidth=2.0,label=f"τ={tau}", alpha=0.9)

    plt.xlabel("Effect size")
    plt.ylabel("Density")
    plt.title("Distribution of effect sizes by tau")
    plt.legend(title="Tau", frameon=False, ncol=2)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "effectsize_hist_by_tau.png"), dpi=200)
    plt.close()
