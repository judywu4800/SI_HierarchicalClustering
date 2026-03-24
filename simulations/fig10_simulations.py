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
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import default_converter, numpy2ri

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("rpy2.rinterface_lib.callbacks").setLevel(logging.ERROR)

# ---- Source R functions ----
r_func_path = os.path.join(REPO_ROOT, "src", "r_functions.R")
ro.r(f'source("{r_func_path}")')

def run_randomized_pvals(n, p, sigma, K, tau, linkage_list, num_trials=20, n_jobs=-1):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(base_dir, "results/raw/fig10")
    os.makedirs(output_dir, exist_ok=True)

    for linkage in linkage_list:
        print(f"\n[Randomized] K={K}, linkage={linkage}, tau={tau}")
        pvals, naive_pvals = check_p_value_uniformity_single_tau_parallel(
            n=n,
            p=p,
            sigma=sigma,
            K=K,
            tau=tau,
            linkage=linkage,
            num_trials=num_trials,
            n_jobs=n_jobs
        )

        df = pd.DataFrame({
            f"tau={tau}": pvals,
            "naive": naive_pvals
        })

        outpath = os.path.join(output_dir, f"pval_validity_randomized_K{K}_{linkage}.csv")
        df.to_csv(outpath, index=False)

def run_gao_barber_pvals(n, p, sigma, K, linkage_list, num_trials=20):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(base_dir, "results/raw/fig10")
    os.makedirs(output_dir, exist_ok=True)

    for linkage in linkage_list:
        print(f"\n[Gao/Barber] K={K}, linkage={linkage}")
        # call R once per linkage
        ro.globalenv['n_py'] = n
        ro.globalenv['p_py'] = p
        ro.globalenv['sigma_py'] = sigma
        ro.globalenv['K_py'] = K
        ro.globalenv['link_py'] = linkage
        ro.globalenv['num_trials_py'] = num_trials
        import numpy as np

        pvals_gao = ro.r("check_gao_uniformity_R(n_py, p_py, sigma_py, K_py, link_py, num_trials_py)")
        #pvals_gao_c = ro.r("check_gao_clustered_uniformity_R(n_py, p_py, sigma_py, K_py, link_py, num_trials_py)")
        pvals_barber = ro.r("check_barber_uniformity_R(n_py, p_py, sigma_py, K_py, link_py, num_trials_py)")
        max_len = max(len(pvals_gao), len(pvals_barber))

        def pad(arr, length):
            return np.pad(arr, (0, length - len(arr)), constant_values=np.nan)

        df = pd.DataFrame({
            "Gao (sigma_all)": pad(np.array(pvals_gao), max_len),
            #"Gao (sigma_clustered)": pad(np.array(pvals_gao_c), max_len),
            "Barber": pad(np.array(pvals_barber), max_len),
        })
        outpath = os.path.join(output_dir, f"pval_valid_gao&barber_K{K}_{linkage}.csv")
        df.to_csv(outpath, index=False)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--num_trials", type=int, default=2000)
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    n, p, sigma = 30, 10, 1.0
    num_trials = args.num_trials
    n_jobs = -1

    randomized_linkages = ["complete", "single", "average", "minimax"]
    gaobarber_linkages = ["complete", "single", "average"]

    Ks = [args.K] if args.K is not None else [2,3]

    for K in Ks:
        run_randomized_pvals(
            n=n,
            p=p,
            sigma=sigma,
            K=K,
            tau=0.1,
            linkage_list=randomized_linkages,
            num_trials=num_trials,
            n_jobs=n_jobs
        )

        run_gao_barber_pvals(n, p, sigma, K,
                             linkage_list=gaobarber_linkages,
                             num_trials=num_trials)
