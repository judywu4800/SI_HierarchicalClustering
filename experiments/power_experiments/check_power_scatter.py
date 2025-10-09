import sys, os
sys.path.append(os.path.abspath('../../src'))
from utils import *
from datetime import datetime

if __name__ == "__main__":
    import os

    n = 30
    p = 10
    sigma = 1
    tau_list = [0,0.01, 0.05, 0.1, 0.5, 1]
    delta = 8.0
    alpha = 0.05
    num_trials = 10000
    n_jobs = -1


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("../../results",f"power_results_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)


    power_results_sel, recovery_results, full = check_power_multi_tau_parallel_random_pair(
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