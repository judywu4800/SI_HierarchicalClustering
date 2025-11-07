import sys, os
sys.path.append(os.path.abspath('../../src'))
from utils import *
from datetime import datetime
import random

if __name__ == "__main__":
    import os
    random.seed(0)
    np.random.seed(0)
    n = 30
    p = 10
    sigma = 1.0
    K = 2
    tau_list = [0.025,0.05,0.1,0.5,1,5]
    #tau_list = [0.01]
    layer = -1
    linkage = "complete"
    num_trials = 1000
    n_jobs = -1

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    output_dir = os.path.join(base_dir, "results/raw")
    os.makedirs(output_dir, exist_ok=True)

    all_p_values, naive_p_values = check_p_value_uniformity_multi_tau_random_pair_parallel(
        n, p, sigma, K, tau_list, linkage, num_trials, n_jobs
    )

    df = pd.DataFrame({f"tau={tau}": all_p_values[tau] for tau in tau_list})
    df["naive"] = naive_p_values
    df.to_csv(os.path.join(output_dir, f"pval_validity_randomized_K{K}.csv"), index=False)
