import random
import sys, os
sys.path.append(os.path.abspath('../../src'))
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import default_converter
from rpy2.robjects import numpy2ri, r
import warnings
import logging
from datetime import datetime
from multiprocessing import Pool
import numpy as np
import os


warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("rpy2.rinterface_lib.callbacks").setLevel(logging.ERROR)

def init_worker():
    ro.r('source("/home/judydw/SI_HierarchicalClustering/src/r_functions.R")')
    #ro.r('source("/Users/judydw/Documents/GitHub/SI_HierarchicalClustering/src/r_functions.R")')
def compute_pval_gao_es(X,true_means, sigma=1, K=3, linkage = "complete", seed=None):
    ro.r('source("/home/judydw/SI_HierarchicalClustering/src/r_functions.R")')
    #ro.r('source("/Users/judydw/Documents/GitHub/SI_HierarchicalClustering/src/r_functions.R")')
    with localconverter(default_converter + numpy2ri.converter):
        ro.globalenv['X'] = ro.conversion.py2rpy(X)
        ro.globalenv['true_mean_py'] = ro.conversion.py2rpy(true_means)
    ro.globalenv['K_py'] = K
    ro.globalenv['sigma_py'] = sigma
    ro.globalenv['link_py'] = linkage
    ro.globalenv['seed'] = int(seed)
    result = ro.r("get_gao_pval_es(X,true_mean_py,sigma_py, K_py, link_py, seed)")
    return result
def compute_power_Gao(n,sigma, K, delta, alpha,linkage, num_trials=10000, rng= None):
    if rng is None:
        rng = np.random.default_rng()
    n_each = n//K
    p_values = []
    effect_sizes = []
    size10s_all = []
    for _ in range(num_trials):
        trial_rng = np.random.default_rng(rng.integers(1e9))
        trial_seed = int(trial_rng.integers(1e9))
        X, true_label, true_means = generate_data_barbers(n_each, delta, sigma, n_clusters=K, true_mean = True, rng=trial_rng)
        pval, effect, size10 = np.array(compute_pval_gao_es(X,true_means, sigma=sigma, K=K, linkage=linkage, seed = trial_seed))

        p_values.append(float(pval))
        effect_sizes.append(float(effect))
        size10s_all.append(int(size10))
    #print(p_values)
    rejection = (np.array(p_values) < alpha).astype(int)
    return rejection, effect_sizes, size10s_all

def run_single_delta_gao(n,delta, K, linkage, base_seed=0):
    delta = float(delta)
    base_seed = int(base_seed)
    rng = np.random.default_rng(base_seed + int(delta * 1000))
    return compute_power_Gao(n,1, K, delta, 0.05, linkage,2000, rng=rng)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--linkage", type=str, required=True, choices=["single", "average","complete"])
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--num_trials", type=int, default=2000)
    args = parser.parse_args()


    random.seed(0)
    np.random.seed(0)
    
    n=30
    K = args.K
    linkage = args.linkage
    n_trials = args.num_trials

    deltas = np.linspace(5, 20, 9)
    delta_list = np.repeat(deltas, n_trials)
    alpha = 0.05
    num_workers = os.cpu_count()

    print(f"\n=== Running Gao ES linkage={linkage}, K={K}, num_trials={n_trials} ===")

    with Pool(processes=num_workers, initializer=init_worker) as pool:
        results_gao = pool.starmap(run_single_delta_gao, [(n, delta, K, linkage) for delta in deltas])

    rejections_all, effect_all, size10s_all = zip(*results_gao)
    rejections_all = np.concatenate(rejections_all, axis=0)
    effect_all = np.concatenate(effect_all, axis=0)
    size10s_all = np.concatenate(size10s_all, axis=0)

    df_gao = pd.DataFrame({
        "tau": ["NA"] * len(effect_all),
        "delta": delta_list,
        "effect_size": effect_all,
        "reject": rejections_all,
        "min_size>=10": size10s_all,
        "method": ["Gao"] * len(effect_all),
        "linkage": [linkage] * len(effect_all)
    })
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    output_dir = os.path.join(base_dir, "results/raw/fig6_es")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f"rejection_es_gao_K{K}_{linkage}.csv")
    df_gao.to_csv(csv_path, index=False)

    print(f"Saved results to {csv_path}")