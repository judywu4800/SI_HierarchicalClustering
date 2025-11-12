import sys, os
sys.path.append(os.path.abspath('../../src'))
from utils import *
import random
import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import default_converter, numpy2ri
import warnings
import logging
from datetime import datetime

# =============================================================
# Setup
# =============================================================
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("rpy2.rinterface_lib.callbacks").setLevel(logging.ERROR)

# Source R functions (edit path for Great Lakes)
ro.r('source("/home/judydw/SI_HierarchicalClustering/src/r_functions.R")')
#ro.r('source("/Users/judydw/Documents/GitHub/SI_HierarchicalClustering/src/r_functions.R")')

# =============================================================
# Helper functions
# =============================================================
def compute_pval_gao(X, K, linkage, method="euclidean", seed=None):
    with localconverter(default_converter + numpy2ri.converter):
        ro.globalenv['X'] = ro.conversion.py2rpy(X)
    ro.globalenv['K_py'] = K
    ro.globalenv['link_py'] = linkage
    ro.globalenv['method_py'] = method
    if seed is not None:
        ro.globalenv['seed_py'] = int(seed)
        pval = ro.r("get_gao_pval(X, K_py, link_py, method_py, seed=seed_py)")
    else:
        pval = ro.r("get_gao_pval(X, K_py, link_py, method_py)")
    return pval


def compute_pval_gao_clustered(X, K, linkage, method="euclidean", seed=None):
    with localconverter(default_converter + numpy2ri.converter):
        ro.globalenv['X'] = ro.conversion.py2rpy(X)
    ro.globalenv['K_py'] = K
    ro.globalenv['link_py'] = linkage
    ro.globalenv['method_py'] = method
    if seed is not None:
        ro.globalenv['seed_py'] = int(seed)
        pval = ro.r("get_gao_pval_clustered(X, K_py, link_py, method_py, seed=seed_py)")
    else:
        pval = ro.r("get_gao_pval_clustered(X, K_py, link_py, method_py)")
    return pval


def compute_pval_barber(X, K, linkage, method="euclidean", seed=None):
    with localconverter(default_converter + numpy2ri.converter):
        ro.globalenv['X'] = ro.conversion.py2rpy(X)
    ro.globalenv['K_py'] = K
    ro.globalenv['link_py'] = linkage
    ro.globalenv['method_py'] = method
    if seed is not None:
        ro.globalenv['seed_py'] = int(seed)
        pval = ro.r("get_barber_pval(X, K_py, link_py, seed=seed_py)")
    else:
        pval = ro.r("get_barber_pval(X, K_py, link_py)")
    return pval


def check_gao_uniformity(n, p, sigma, K, linkage="complete", num_trials=500, base_seed=0):
    rng = np.random.default_rng(base_seed)
    mu = np.zeros(p)
    p_values = []
    for _ in range(num_trials):
        seed_r = int(rng.integers(0, 2 ** 31 - 1))
        X = generate_null_data(n, p, mu, sigma, rng=rng)
        pval = np.array(compute_pval_gao(X, K, linkage, seed_r))[0]
        p_values.append(pval)
    return p_values


def check_gao_clustered_uniformity(n, p, sigma, K, linkage="complete", num_trials=500, base_seed=0):
    rng = np.random.default_rng(base_seed)
    mu = np.zeros(p)
    p_values = []
    for _ in range(num_trials):
        seed_r = int(rng.integers(0, 2 ** 31 - 1))
        X = generate_null_data(n, p, mu, sigma, rng=rng)
        pval = np.array(compute_pval_gao_clustered(X, K, linkage, seed=seed_r))[0]
        p_values.append(pval)
    return p_values


def check_barber_uniformity(n, p, sigma, K, linkage="complete", num_trials=500, base_seed=0):
    rng = np.random.default_rng(base_seed)
    mu = np.zeros(p)
    p_values = []
    for _ in range(num_trials):
        seed_r = int(rng.integers(0, 2 ** 31 - 1))
        X = generate_null_data(n, p, mu, sigma, rng=rng)
        pval = np.array(compute_pval_barber(X, K, linkage, seed=seed_r))[0]
        p_values.append(pval)
    return p_values


# =============================================================
# Run for one linkage
# =============================================================
def run_validity_gao_barber_for_linkage(n, p, sigma, K, linkage, num_trials):
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Start linkage={linkage}, K={K}")
    t0 = datetime.now()

    pvals_gao = check_gao_uniformity(n, p, sigma, K, linkage, num_trials)
    pvals_gao_c = check_gao_clustered_uniformity(n, p, sigma, K, linkage, num_trials)
    pvals_barber = check_barber_uniformity(n, p, sigma, K, linkage, num_trials)

    pvals_result = {
        "Gao (sigma_all)": pvals_gao,
        "Gao (sigma_clustered)": pvals_gao_c,
        "Barber": pvals_barber,
    }
    df = pd.DataFrame.from_dict(pvals_result)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    output_dir = os.path.join(base_dir, "results/raw/fig5_linkages")
    os.makedirs(output_dir, exist_ok=True)

    outpath = os.path.join(output_dir, f"pval_valid_gao&barber_K{K}_{linkage}.csv")
    df.to_csv(outpath, index=False)

    print(f"[{linkage}] Saved to {outpath} ({(datetime.now()-t0).total_seconds():.1f}s)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--linkage", type=str, default=None)
    parser.add_argument("--array_id", type=int, default=None)
    parser.add_argument("--num_trials", type=int, default=2000)
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    n, p, sigma = 30, 10, 1.0
    linkage_list = ["complete", "average", "single"]

    if args.linkage is not None:
        linkage = args.linkage
    elif args.array_id is not None:
        linkage = linkage_list[args.array_id % len(linkage_list)]
    else:
        raise ValueError("Must provide either --linkage or --array_id")

    run_validity_gao_barber_for_linkage(
        n, p, sigma, args.K, linkage, num_trials=args.num_trials
    )
