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
def get_pval_if_recovered(X, true_label, alpha, K,linkage="complete"):
    ro.r('source("/home/judydw/RAC_invariant/r_functions.R")')
    #ro.r('source("/Users/judydw/Documents/GitHub/SI_HierarchicalClustering/r_functions.R")')
    with localconverter(default_converter + numpy2ri.converter):
        ro.globalenv['X'] = ro.conversion.py2rpy(X)
        ro.globalenv['true_cluster'] = ro.conversion.py2rpy(true_label)

    ro.globalenv['K_py'] = K
    ro.globalenv['alpha_py'] = alpha
    ro.globalenv['link_py'] = linkage
    result = ro.r("get_prop_if_recovered_barber(X, true_cluster, alpha_py, link_py, K_py)")
    return result

def compute_power_barbers(n, p, sigma, delta, alpha=0.05, num_trials=5000):
    prop = []
    recovery = 0
    for _ in range(num_trials):
        X, true_label = generate_data_barbers(10,delta,sigma)
        result = get_pval_if_recovered(X, true_label,alpha, linkage="complete", K=3)
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

def run_single_delta_barber(delta):
    return compute_power_barbers(30, 10, 1, delta, 0.05, 500)

if __name__ == "__main__":
    random.seed(1)
    n = 30
    #p = 10
    sigma = 1
    deltas = [0,2,4,6,8,10]
    #deltas = [10]
    alpha = 0.05

    powers_barber = []
    recovery_list_barber = []
    pval = {}
    num_workers = os.cpu_count()
    '''
    for delta in deltas:
        power, recovery, pvals = run_single_delta_barber(delta)
        powers_barber.append(power)
        recovery_list_barber.append(recovery)
        pval[delta] = pvals
    '''
    with Pool(processes=num_workers, initializer=init_worker) as pool:
        results_barber = pool.map(run_single_delta_barber, deltas)
    powers_barber = [r[0] for r in results_barber]
    recovery_list_barber = [r[1] for r in results_barber]

    df_barber = pd.DataFrame({
        "Tau": ["NA"] * len(deltas),
        "Delta": deltas,
        "Rejection Proportion": powers_barber,
        "Recovery_Rate": recovery_list_barber,
        "Successful_Trials": ["NA"] * len(deltas),
        "Method": ["Barber"] * len(deltas)
    })
    #df_pval = pd.DataFrame(pval)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"rejprop_barbers_results_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "rej_prop_and_recovery_barber.csv")
    df_barber.to_csv(csv_path, index=False)
    #csv_path2 = os.path.join(output_dir, "pval_barber.csv")
    #df_pval.to_csv(csv_path2, index=False)