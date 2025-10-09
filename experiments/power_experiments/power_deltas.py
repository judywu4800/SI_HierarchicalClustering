import sys, os
sys.path.append(os.path.abspath('../../src'))
from utils import *
from datetime import datetime
import random

if __name__ == "__main__":
    import os
    random.seed(1)
    n = 30
    p = 10
    sigma = 1
    #tau=0.1
    tau_list = [0,0.01, 0.05, 0.1,0.5,1]
    #tau_list = [0.01]
    #deltas = [8,10]
    deltas = [0,2,4,6,8,10]
    alpha = 0.05
    num_trials = 500
    n_jobs = -1


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("../../results", f"power_delta_results_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    power_results, recovery_results, success_results = check_power_multi_tau_delta_random_pair(
        n, p, sigma, tau_list, deltas, alpha, num_trials, n_jobs
    )

    rows = []
    for tau in tau_list:
        for delta in deltas:
            rows.append({
                "Tau": tau,
                "Delta": delta,
                "Conditional Power": power_results[tau][delta],
                "Recovery Probability": recovery_results[tau][delta],
                "Successful Trials": success_results[tau][delta]
            })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "power_and_recovery.csv")
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 6))
    for tau in tau_list:
        label = "Naive" if tau == 0 else f"tau={tau}"
        plt.plot(deltas, [power_results[tau][d] for d in deltas],
                 marker='o', label=label)
    plt.xlabel("Delta")
    plt.ylabel("Power")
    plt.title("Power vs Delta for Multiple Tau Values")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, "combined_power_plot.png"))
    plt.close()
