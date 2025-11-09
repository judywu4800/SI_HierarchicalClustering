import sys, os
sys.path.append(os.path.abspath('../../src'))
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
ro.r('source("/Users/judydw/Documents/GitHub/SI_HierarchicalClustering/src/r_functions.R")')
#ro.r('source("/home/judydw/SI_HierarchicalClustering/src/r_functions.R")')


# ------------------------------------------------------------
#  Section 1: Randomized p-value uniformity (Python)
# ------------------------------------------------------------
def run_randomized_pvals(n, p, sigma, K, tau, linkage_list, num_trials=1000, n_jobs=-1):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    output_dir = os.path.join(base_dir, "results/raw/fig5")
    os.makedirs(output_dir, exist_ok=True)

    for linkage in linkage_list:
        print(f"\n[Randomized] K={K}, linkage={linkage}, tau={tau}")
        all_p_values, naive_p_values = check_p_value_uniformity_multi_tau_random_pair_parallel(
            n, p, sigma, K, [tau], linkage, num_trials, n_jobs
        )
        df = pd.DataFrame({f"tau={tau}": all_p_values[tau]})
        df["naive"] = naive_p_values
        outpath = os.path.join(output_dir, f"pval_validity_randomized_K{K}_{linkage}.csv")
        df.to_csv(outpath, index=False)

def run_gao_barber_pvals(n, p, sigma, K, linkage_list, num_trials=1000):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    output_dir = os.path.join(base_dir, "results/raw/fig5")
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
    random.seed(0)
    np.random.seed(0)
    n, p, sigma = 30, 10, 1.0
    num_trials = 1000
    n_jobs = -1

    randomized_linkages = ["complete", "single", "average", "minimax"]
    gaobarber_linkages = ["complete", "single", "average"]

    for K in [2, 3]:
        # ---- Randomized experiments
        run_randomized_pvals(n, p, sigma, K, tau=0.1, linkage_list=randomized_linkages,
                             num_trials=num_trials, n_jobs=n_jobs)

        # ---- Gao & Barber experiments
        run_gao_barber_pvals(n, p, sigma, K, linkage_list=gaobarber_linkages,
                             num_trials=num_trials)
