import sys, os
def get_repo_root():
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )

REPO_ROOT = get_repo_root()
sys.path.append(os.path.join(REPO_ROOT, "src"))
from utils import *
import random
import numpy as np
import pandas as pd
import warnings
import logging
import rpy2.robjects as ro





warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("rpy2.rinterface_lib.callbacks").setLevel(logging.ERROR)

# ---- Source R functions ----
#ro.r('source("/home/judydw/SI_HierarchicalClustering/src/r_functions.R")')
r_func_path = os.path.join(REPO_ROOT, "src", "r_functions.R")
ro.r(f'source("{r_func_path}")')

def run_randomized_pvals(n, p, sigma, K, tau, linkage, num_trials=1000, n_jobs=-1):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(base_dir, "results/raw/fig8_batch")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[Randomized] K={K}, linkage={linkage}, tau={tau}")
    all_p_values, naive_p_values = check_p_value_uniformity_multi_tau_random_pair_parallel(
        n=n,
        p=p,
        sigma=sigma,
        K=K,
        tau_list=[tau],
        linkage=linkage,
        num_trials=num_trials,
        n_jobs=n_jobs
    )
    df = pd.DataFrame({f"tau={tau}": all_p_values[tau]})
    df["naive"] = naive_p_values
    outpath = os.path.join(output_dir, f"pval_validity_randomized_K{K}_{linkage}.csv")
    df.to_csv(outpath, index=False)


def run_gao_barber_pvals(n, p, sigma, K, linkage, num_trials=1000):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(base_dir, "results/raw/fig8_batch")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[Gao/Barber] K={K}, linkage={linkage}")
    ro.globalenv['n_py'] = n
    ro.globalenv['p_py'] = p
    ro.globalenv['sigma_py'] = sigma
    ro.globalenv['K_py'] = K
    ro.globalenv['link_py'] = linkage
    ro.globalenv['num_trials_py'] = num_trials

    pvals_gao = ro.r("check_gao_uniformity_R(n_py, p_py, sigma_py, K_py, link_py, num_trials_py)")
    pvals_gao_c = ro.r("check_gao_clustered_uniformity_R(n_py, p_py, sigma_py, K_py, link_py, num_trials_py)")
    pvals_barber = ro.r("check_barber_uniformity_R(n_py, p_py, sigma_py, K_py, link_py, num_trials_py)")

    df = pd.DataFrame({
        "Gao (sigma_all)": np.array(pvals_gao),
        "Gao (sigma_clustered)": np.array(pvals_gao_c),
        "Barber": np.array(pvals_barber)
    })
    outpath = os.path.join(output_dir, f"pval_valid_gao&barber_K{K}_{linkage}.csv")
    df.to_csv(outpath, index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--linkage", type=str, required=True)
    parser.add_argument("--method", type=str, choices=["randomized", "gao_barber"], required=True)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--num_trials", type=int, default=1000)
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    n, p, sigma = 30, 10, 1.0

    if args.method == "randomized":
        run_randomized_pvals(n=n,
            p=p,
            sigma=sigma,
            K=args.K,
            tau=args.tau,
            linkage=args.linkage,
            num_trials=args.num_trials,
            n_jobs=args.n_jobs)
    else:
        run_gao_barber_pvals(n, p, sigma, args.K, args.linkage,
                             num_trials=args.num_trials)
