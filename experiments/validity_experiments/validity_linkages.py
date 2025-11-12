import sys, os
sys.path.append(os.path.abspath('../../src'))
from utils import *
import random
import numpy as np
import pandas as pd
import warnings
import logging
from datetime import datetime
from multiprocessing import Pool

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("rpy2.rinterface_lib.callbacks").setLevel(logging.ERROR)


# =====================================================================
# --- Helper: Run trials for one linkage ---
# =====================================================================
def run_validity_for_linkage(n, p, sigma, K, tau, linkage, num_trials, n_jobs):
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Start linkage={linkage}, K={K}, tau={tau}")
    t0 = datetime.now()

    all_p_values, naive_p_values = check_p_value_uniformity_multi_tau_random_pair_parallel(
        n, p, sigma, K, [tau], linkage, num_trials, n_jobs
    )

    df = pd.DataFrame({f"tau={tau}": all_p_values[tau] for tau in [tau]})
    df["naive"] = naive_p_values

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    output_dir = os.path.join(base_dir, "results/raw/fig5_linkages")
    os.makedirs(output_dir, exist_ok=True)

    outpath = os.path.join(output_dir, f"pval_validity_randomized_K{K}_{linkage}.csv")
    df.to_csv(outpath, index=False)
    print(f"[{linkage}] Saved to {outpath} ({(datetime.now()-t0).total_seconds():.1f}s)")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--linkage", type=str, default=None,
                        help="Linkage type (e.g. complete, single, average, minimax)")
    parser.add_argument("--array_id", type=int, default=None,
                        help="Optional SLURM array index to select linkage")
    parser.add_argument("--num_trials", type=int, default=2000)
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)

    n = 30
    p = 10
    sigma = 1.0
    tau = 0.1
    n_jobs = -1

    linkage_list = ["complete", "average", "single", "minimax"]

    # Select linkage
    if args.linkage is not None:
        linkage = args.linkage
    elif args.array_id is not None:
        linkage = linkage_list[args.array_id % len(linkage_list)]
    else:
        raise ValueError("Must provide either --linkage or --array_id")

    # Run
    run_validity_for_linkage(
        n, p, sigma, args.K, tau,
        linkage, num_trials=args.num_trials, n_jobs=n_jobs
    )
