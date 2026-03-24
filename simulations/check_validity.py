import sys, os
def get_repo_root():
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )

REPO_ROOT = get_repo_root()
sys.path.append(os.path.join(REPO_ROOT, "src"))
from utils import *
from datetime import datetime
import random

if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--num_trials", type=int, required=True)
    args = parser.parse_args()

    n = 30
    p = 10
    sigma = 1.0
    K = args.K
    num_trials = args.num_trials
    tau_list = [0.025, 0.05, 0.1, 0.25, 0.5, 1, 5]
    linkage = "complete"
    n_jobs = -1

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(base_dir, "results/raw/validity")
    os.makedirs(output_dir, exist_ok=True)


    all_p_values, naive_p_values = check_p_value_uniformity_multi_tau_random_pair_parallel(
        n=n,
        p=p,
        sigma=sigma,
        K=K,
        tau_list=tau_list,
        linkage=linkage,
        num_trials=num_trials,
        n_jobs=n_jobs
    )

    df = pd.DataFrame({f"tau={tau}": all_p_values[tau] for tau in tau_list})
    df["naive"] = naive_p_values
    csv_path = os.path.join(output_dir, f"pval_validity_randomized_K{K}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")


