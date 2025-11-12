import sys, os
import numpy as np
import random
import pandas as pd
from utils import *

# ---- safer, robust path handling ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../src"))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

if __name__ == "__main__":
    # ---- fixed seeds for reproducibility ----
    random.seed(0)
    np.random.seed(0)

    # ---- parameters ----
    n = 30
    p = 2
    sigma = 1
    tau_list = 0.1
    deltas = np.linspace(5, 20, 9)
    alpha = 0.05
    num_trials = 2
    K = 3
    n_jobs = -1

    # ---- parse linkage argument ----
    if len(sys.argv) > 1:
        linkage = sys.argv[1]
    else:
        linkage = os.getenv("LINKAGE", "complete")

    print("=" * 60)
    print(f"Running fig6_randomized.py for linkage = {linkage}")
    print(f"Current working dir: {os.getcwd()}")
    print(f"Source dir: {SRC_DIR}")
    print("=" * 60)

    # ---- run experiments ----
    df_trials = check_power_es_single_tau_fast(
        n, sigma, tau_list, deltas, alpha, num_trials,
        K=K, linkage=linkage, n_jobs=n_jobs
    )

    # ---- output ----
    base_dir = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
    output_dir = os.path.join(base_dir, "results/raw/fig6_es")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f"reject_effect_size_K{K}_{linkage}.csv")
    df_trials.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")
