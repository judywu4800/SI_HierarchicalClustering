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
def compute_pval_if_recovered_gao(X, cluster_true,alpha, linkage = "complete", K=3):
    ro.r('source("/home/judydw/RAC_invariant/r_functions.R")')
    #ro.r('source("/Users/judydw/Documents/GitHub/SI_HierarchicalClustering/r_functions.R")')
    with localconverter(default_converter + numpy2ri.converter):
        ro.globalenv['X'] = ro.conversion.py2rpy(X)
        ro.globalenv['true_cluster'] = ro.conversion.py2rpy(cluster_true)

    ro.globalenv['K_py'] = K
    ro.globalenv['alpha_py'] = alpha
    ro.globalenv['link_py'] = linkage
    result = ro.r("get_reject_prop_if_recovered_gao(X, true_cluster, alpha_py, link_py, K_py)")
    return result

def compute_power_Gao(n, sigma, delta, alpha, num_trials=10000):
    prop = []
    recovery = 0
    for _ in range(num_trials):
        X, true_label = generate_data_barbers(10, delta, sigma)
        result = compute_pval_if_recovered_gao(X, true_label, alpha, linkage="complete", K=3)
        recovered = np.array(result)[1]
        if recovered == 1:
            rej_prop = np.array(result)[0]
            recovery += 1
            prop.append(rej_prop)
    mean_prop = 0
    if len(prop) > 0:
        mean_prop = np.mean(prop)
    recovery_rate = recovery / num_trials
    return mean_prop, recovery_rate


def run_single_delta_gao(delta):
    return compute_power_Gao(30, 1, delta, 0.05, 500)

if __name__ == "__main__":
    random.seed(1)
    n = 30
    p = 10
    sigma = 1
    deltas = [0,2,4,6,8,10]
    #deltas = [10]
    alpha = 0.05

    prop_gao = []
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

    prop_gao = [r[0] for r in results_gao]
    recovery_list_gao = [r[1] for r in results_gao]

    df_gao = pd.DataFrame({
        "Tau": ["NA"] * len(deltas),
        "Delta": deltas,
        "Rejection Proportion": prop_gao,
        "Recovery_Rate": recovery_list_gao,
        "Successful_Trials": ["NA"] * len(deltas),
        "Method": ["Gao"] * len(deltas)
    })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    output_dir = os.path.join(base_dir, "results", f"rejprop_gao_results_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "rejprop_and_recovery_gao.csv")
    df_gao.to_csv(csv_path, index=False)


