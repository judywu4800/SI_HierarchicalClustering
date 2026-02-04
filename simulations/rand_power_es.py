import sys, os
import numpy as np
import random
import pandas as pd
from utils import *

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../src"))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trials", type=int, default=2000)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--linkage", type=str, default="complete")
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)

    n = 30
    p = 2
    sigma = 1
    tau_list = 0.1
    deltas = np.linspace(1, 10, 9)
    alpha = 0.05
    num_trials = args.num_trials
    n_jobs = -1
    K = args.K
    linkage = args.linkage

    print("=" * 60)
    print(f"Running fig6_randomized.py for linkage = {linkage}, K = {K}")
    print(f"Current working dir: {os.getcwd()}")
    print(f"Source dir: {SRC_DIR}")
    print("=" * 60)

    df_trials = check_power_es_single_tau_fast(
        n, sigma, tau_list, deltas, alpha, num_trials,
        K=K, linkage=linkage, n_jobs=n_jobs
    )

    base_dir = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    output_dir = os.path.join(base_dir, "results/raw/fig4")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f"reject_effect_size_K{K}_{linkage}.csv")
    df_trials.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")
