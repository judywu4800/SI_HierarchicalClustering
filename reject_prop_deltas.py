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
    num_trials = 1000
    n_jobs = -1


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"rejprop_delta_results_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    prop_results, recovery_results, success_results = check_reject_prop_multi_tau_delta_random_pair(
        n, p, sigma, tau_list, deltas, alpha, num_trials, n_jobs
    )

    rows = []
    for tau in tau_list:
        for delta in deltas:
            rows.append({
                "Tau": tau,
                "Delta": delta,
                "Rejection Proportion": prop_results[tau][delta],
                "Recovery Probability": recovery_results[tau][delta],
                "Successful Trials": success_results[tau][delta]
            })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "rejprop_and_recovery.csv")
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 6))
    for tau in tau_list:
        label = "Naive" if tau == 0 else f"tau={tau}"
        plt.plot(deltas, [prop_results[tau][d] for d in deltas],
                 marker='o', label=label)
    plt.xlabel("Delta")
    plt.ylabel("Rejection Proportion")
    plt.title("Rejection Proportion vs Delta for Multiple Tau Values")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, "combined_rej_prop_plot.png"))
    plt.close()
