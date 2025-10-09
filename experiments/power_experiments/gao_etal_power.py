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
    ro.r('source("/home/judydw/RAC_invariant/r_functions.R")')
    #ro.r('source("/Users/judydw/Documents/GitHub/SI_HierarchicalClustering/r_functions.R")')
def compute_pval_if_recovered_gao(X, cluster_true, linkage = "complete", K=3):
    ro.r('source("/home/judydw/RAC_invariant/r_functions.R")')
    #ro.r('source("/Users/judydw/Documents/GitHub/SI_HierarchicalClustering/r_functions.R")')
    with localconverter(default_converter + numpy2ri.converter):
        ro.globalenv['X'] = ro.conversion.py2rpy(X)
        ro.globalenv['true_cluster'] = ro.conversion.py2rpy(cluster_true)

    ro.globalenv['K_py'] = K
    ro.globalenv['link_py'] = linkage
    pval = ro.r("get_pval_if_recovered_gao(X, true_cluster, link_py, K_py)")
    return pval

def compute_power_Gao(n, sigma, delta, alpha, num_trials=10000):
    p_values = []
    recovery = 0
    for _ in range(num_trials):
        X, true_label = generate_data_barbers(10, delta, sigma)
        pval = np.array(compute_pval_if_recovered_gao(X, true_label, linkage = "complete", K=3))[0]
        if not np.isnan(pval):
            recovery += 1
            p_values.append(pval)
    power = 0
    if len(p_values) > 0:
        power = np.mean(np.array(p_values) < alpha)
    recovery_rate = recovery/num_trials
    return power, recovery_rate


def run_single_delta_gao(delta):
    return compute_power_Gao(30, 1, delta, 0.05, 5000)

if __name__ == "__main__":
    random.seed(1)
    n = 30
    p = 10
    sigma = 1
    deltas = [0,2,4,6,8,10]
    #deltas = [0]
    alpha = 0.05

    powers_gao = []
    recovery_list_gao = []
    num_workers = os.cpu_count()

    '''
    for delta in deltas:
        power, recovery = compute_power_Gao(n, sigma, delta, alpha, num_trials=500)
        powers_gao.append(power)
        recovery_list_gao.append(recovery)    
    '''

    with Pool(processes=num_workers, initializer=init_worker) as pool:
        results_gao = pool.map(run_single_delta_gao, deltas)

    powers_gao = [r[0] for r in results_gao]
    recovery_list_gao = [r[1] for r in results_gao]

    df_gao = pd.DataFrame({
        "Tau": ["NA"] * len(deltas),
        "Delta": deltas,
        "Power": powers_gao,
        "Recovery_Rate": recovery_list_gao,
        "Successful_Trials": ["NA"] * len(deltas),
        "Method": ["Gao"] * len(deltas)
    })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("../../results", f"power_gao_results_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "power_and_recovery_gao.csv")
    df_gao.to_csv(csv_path, index=False)


