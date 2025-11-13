import sys, os
sys.path.append(os.path.abspath('../../src'))
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import default_converter
from rpy2.robjects import numpy2ri, r
import warnings
import logging
from datetime import datetime
import os

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("rpy2.rinterface_lib.callbacks").setLevel(logging.ERROR)
#ro.r('source("/Users/judydw/Documents/GitHub/SI_HierarchicalClustering/src/r_functions.R")')
ro.r('source("/home/judydw/SI_HierarchicalClustering/src/r_functions.R")')
def compute_pval_gao(X, K, linkage, method = "euclidean", seed = None):
    #ro.r('source("/home/judydw/SI_HierarchicalClustering/src/r_functions.R")')
    #ro.r('source("/Users/judydw/Documents/GitHub/SI_HierarchicalClustering/src/r_functions.R")')
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

def compute_pval_gao_clustered(X, K, linkage, method = "euclidean", seed = None):
    #ro.r('source("/home/judydw/RAC_invariant/r_functions.R")')
    #ro.r('source("/Users/judydw/Documents/GitHub/SI_HierarchicalClustering/src/r_functions.R")')
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

def compute_pval_barber(X, K, method= "complete", seed = None):
    #ro.r('source("/home/judydw/RAC_invariant/r_functions.R")')
    #ro.r('source("/Users/judydw/Documents/GitHub/SI_HierarchicalClustering/src/r_functions.R")')
    with localconverter(default_converter + numpy2ri.converter):
        ro.globalenv['X'] = ro.conversion.py2rpy(X)

    ro.globalenv['K_py'] = K
    ro.globalenv['method_py'] = method
    if seed is not None:
        ro.globalenv['seed_py'] = int(seed)
        pval = ro.r("get_barber_pval(X, K_py, link_py,  seed=seed_py)")
    else:
        pval = ro.r("get_barber_pval(X, K_py, link_py)")
    return pval

def check_gao_uniformity(n, p, sigma, K, linkage = "complete", num_trials= 500, base_seed=0):
    rng = np.random.default_rng(base_seed)
    p_values = []
    mu = np.zeros(p)

    for _ in range(num_trials):
        seed_r = int(rng.integers(0, 2 ** 31 - 1))
        X = generate_null_data(n,p,mu,sigma, rng)
        pval = np.array(compute_pval_gao(X, K, linkage,seed_r))[0]
        p_values.append(pval)

    return p_values

def check_gao_clustered_uniformity(n, p, sigma, K, linkage = "complete", num_trials= 500, base_seed=0):
    rng = np.random.default_rng(base_seed)
    p_values = []
    mu = np.zeros(p)

    for _ in range(num_trials):
        seed_r = int(rng.integers(0, 2 ** 31 - 1))
        X = generate_null_data(n,p,mu,sigma, rng=rng)
        pval = np.array(compute_pval_gao_clustered(X, K, linkage, seed=seed_r))[0]
        p_values.append(pval)

    return p_values

def check_barber_uniformity(n, p, sigma, K, linkage = "complete", num_trials = 500, base_seed = 0):
    rng = np.random.default_rng(base_seed)
    p_values = []
    mu = np.zeros(p)

    for _ in range(num_trials):
        seed_r = int(rng.integers(0, 2 ** 31 - 1))
        X = generate_null_data(n, p, mu, sigma, rng=rng)
        pval = np.array(compute_pval_barber(X, K, linkage, seed=seed_r))[0]
        p_values.append(pval)

    return p_values

if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, required=True)
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    n = 30
    p = 10
    sigma = 1.0
    K = args.K
    linkage = "complete"
    num_trials = 2000

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    output_dir = os.path.join(base_dir, "results/raw")
    os.makedirs(output_dir, exist_ok=True)

    pvals_gao = check_gao_uniformity(n, p, sigma, K, linkage, num_trials)
    pvals_gao_c = check_gao_clustered_uniformity(n, p, sigma, K, linkage, num_trials)
    pvals_barber = check_barber_uniformity(n, p, sigma, K, linkage, num_trials)

    pvals_result = {"Gao (sigma_all)": pvals_gao, "Gao (sigma_clustered)": pvals_gao_c, "Barber": pvals_barber}
    pvals_df = pd.DataFrame.from_dict(pvals_result)

    pvals_df.to_csv(os.path.join(output_dir, f"pval_valid_gao&barber_K{K}.csv"), index=False)



