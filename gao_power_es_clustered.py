import random

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
    ro.r('source("/home/judydw/RAC_invariant/r_functions.R")')
    #ro.r('source("/Users/judydw/Documents/GitHub/SI_HierarchicalClustering/r_functions.R")')
def compute_pval_gao_es(X,true_means, K=3, linkage = "complete"):
    ro.r('source("/home/judydw/RAC_invariant/r_functions.R")')
    #ro.r('source("/Users/judydw/Documents/GitHub/SI_HierarchicalClustering/r_functions.R")')
    with localconverter(default_converter + numpy2ri.converter):
        ro.globalenv['X'] = ro.conversion.py2rpy(X)
        ro.globalenv['true_mean_py'] = ro.conversion.py2rpy(true_means)
    ro.globalenv['K_py'] = K
    ro.globalenv['link_py'] = linkage
    result = ro.r("get_gao_pval_es_clustered(X,true_mean_py, K_py, link_py)")
    return result
def compute_power_Gao(sigma, delta, alpha, num_trials=10000):
    p_values = []
    effect_sizes = []
    for _ in range(num_trials):
        X, true_label, true_means = generate_data_barbers(10, delta, sigma, true_mean = True)
        pval, effect = np.array(compute_pval_gao_es(X,true_means, K=3, linkage="complete"))

        p_values.append(pval)
        effect_sizes.append(effect)

    rejection = (np.array(p_values) < alpha).astype(int)
    return rejection, effect_sizes

def run_single_delta_gao(delta):
    return compute_power_Gao(1, delta, 0.05, 2000)

if __name__ == "__main__":
    random.seed(1)
    n = 30
    p = 10
    sigma = 1
    n_trials = 2000
    deltas = np.linspace(5,20,9)
    delta_list = np.repeat(deltas,n_trials)
    #deltas = [0]
    alpha = 0.05


    num_workers = os.cpu_count()


    with Pool(processes=num_workers, initializer=init_worker) as pool:
        results_gao = pool.map(run_single_delta_gao, deltas)

    rejections_all, effect_all = zip(*results_gao)
    rejections_all = np.concatenate(rejections_all, axis=0)
    effect_all = np.concatenate(effect_all, axis=0)

    df_gao = pd.DataFrame({
        "tau": ["NA"] * len(effect_all),
        "delta":delta_list,
        "effect_size": effect_all,
        "reject": rejections_all,
        "method": ["Gao (sigma_clustered)"] * len(effect_all)
    })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"gao_es_clustered_results_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "rejection_and_effect_gao_clustered.csv")
    df_gao.to_csv(csv_path, index=False)


