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
def get_pval_if_recovered(X, true_label, K,linkage="complete"):
    ro.r('source("/home/judydw/RAC_invariant/r_functions.R")')
    #ro.r('source("/Users/judydw/Documents/GitHub/SI_HierarchicalClustering/r_functions.R")')
    with localconverter(default_converter + numpy2ri.converter):
        ro.globalenv['X'] = ro.conversion.py2rpy(X)
        ro.globalenv['true_cluster'] = ro.conversion.py2rpy(true_label)

    ro.globalenv['K_py'] = K
    ro.globalenv['link_py'] = linkage
    result = ro.r("get_pval_if_recovered_barber(X, true_cluster, link_py, K_py)")
    return result

def compute_power_barbers(n, p, sigma, delta, alpha=0.05, num_trials=5000):
    p_values = []
    recovery = 0
    for _ in range(num_trials):
        X, true_label = generate_data_barbers(10,delta,sigma)
        pval = np.array(get_pval_if_recovered(X, true_label, linkage="complete", K=3))[0]
        if pval <=1.0:
            recovery += 1
            p_values.append(pval)

    power = 0
    if len(p_values) > 0:
        power = np.mean(np.array(p_values) < alpha)
    recovery_rate = recovery / num_trials
    return power, recovery_rate, p_values

def run_single_delta_barber(delta):
    return compute_power_barbers(30, 10, 1, delta, 0.05, 5)

if __name__ == "__main__":
    random.seed(1)
    n = 30
    #p = 10
    sigma = 1
    #deltas = [0,2,4,6,8,10]
    deltas = [0]
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
        "Power": powers_barber,
        "Recovery_Rate": recovery_list_barber,
        "Successful_Trials": ["NA"] * len(deltas),
        "Method": ["Barber"] * len(deltas)
    })
    #df_pval = pd.DataFrame(pval)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"power_barbers_results_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "power_and_recovery_barber.csv")
    df_barber.to_csv(csv_path, index=False)
    #csv_path2 = os.path.join(output_dir, "pval_barber.csv")
    #df_pval.to_csv(csv_path2, index=False)