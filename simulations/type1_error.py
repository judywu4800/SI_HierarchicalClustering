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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--num_trials", type=int, default=200)
    parser.add_argument("--num_repeats", type=int, default=100)
    args = parser.parse_args()
    random.seed(0)
    np.random.seed(0)
    n = 30
    p = 10
    sigma = 1
    tau_list = [0,0.025,0.05,0.1,0.25, 0.5, 1, 5]
    K = args.K
    layer = -1
    alpha = 0.05
    num_trials = args.num_trials
    num_repeats = args.num_repeats
    n_jobs = -1

    df_results = check_type1_multi_tau_random_pair_parallel(n, p, sigma, tau_list, K,
                                                 alpha=alpha, num_trials=num_trials,
                                                 num_repeats=num_repeats, n_jobs=n_jobs)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(base_dir, "results/raw/validity")
    os.makedirs(output_dir, exist_ok=True)

    df_results.to_csv(os.path.join(output_dir, f"type1_error_randomized_K{K}.csv"), index=False)