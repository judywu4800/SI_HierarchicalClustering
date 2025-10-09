import numpy as np
import sys, os
sys.path.append(os.path.abspath('../../src'))
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

def get_pval_if_recovered_barber(X, true_label, K,linkage="complete"):
    ro.r('source("/home/judydw/RAC_invariant/r_functions.R")')
    with localconverter(default_converter + numpy2ri.converter):
        ro.globalenv['X'] = ro.conversion.py2rpy(X)
        ro.globalenv['true_cluster'] = ro.conversion.py2rpy(true_label)

    ro.globalenv['K_py'] = K
    ro.globalenv['link_py'] = linkage
    pval = ro.r("get_pval_if_recovered_barber(X, true_cluster, link_py, K_py)")
    return pval

def compute_power_barbers(n, p, sigma, delta, alpha, num_trials=10000):
    p_values = []
    recovery = 0
    for _ in range(num_trials):
        X, true_label = generate_3cluster_data(n=n, p=p, delta=delta, sigma=sigma, random_state=None,
                                               return_labels=True)
        pval = get_pval_if_recovered_barber(X, true_label, linkage="complete", K=3)
        if not np.isnan(pval):
            recovery += 1
            p_values.append(pval)

    power = np.mean(np.array(p_values) < alpha)
    recovery_rate = recovery / num_trials
    return power, recovery_rate

def run_single_delta_barber(delta):
    return compute_power_barbers(30, 10, 1, delta, 0.05, 10000)


def compute_pval_if_recovered_gao(X, cluster_true, linkage = "complete", K=3):
    ro.r('source("/home/judydw/RAC_invariant/r_functions.R")')
    with localconverter(default_converter + numpy2ri.converter):
        ro.globalenv['X'] = ro.conversion.py2rpy(X)
        ro.globalenv['true_cluster'] = ro.conversion.py2rpy(cluster_true)

    ro.globalenv['K_py'] = K
    ro.globalenv['link_py'] = linkage
    pval = ro.r("get_pval_if_recovered_gao(X, true_cluster, link_py, K_py)")
    return pval

def compute_power_Gao(n, p, sigma, delta, alpha, num_trials=10000):
    p_values = []
    recovery = 0
    for _ in range(num_trials):
        X, true_label = generate_3cluster_data(n=n, p=p, delta=delta, sigma=sigma, random_state=None,
                                               return_labels=True)
        pval = compute_pval_if_recovered_gao(X, true_label, linkage = "complete", K=3)
        if not np.isnan(pval):
            recovery += 1
            p_values.append(pval)

    power = np.mean(np.array(p_values) < alpha)
    recovery_rate = recovery/num_trials
    return power, recovery_rate


def run_single_delta_gao(delta):
    return compute_power_Gao(30, 10, 1, delta, 0.05, 10000)


if __name__ == "__main__":
    n = 30
    p = 10
    sigma = 1
    deltas = [4,6,8,10,12]
    tau_list = [0, 0.01, 0.05, 0.1, 0.5, 1]
    alpha = 0.05
    num_trials = 10000
    n_jobs = -1

    powers_gao=[]
    recovery_list_gao=[]
    powers_barber = []
    recovery_list_barber = []
    num_workers = os.cpu_count()
    with Pool(processes=num_workers, initializer=init_worker) as pool:
        results_gao = pool.map(run_single_delta_gao, deltas)

    powers_gao = [r[0] for r in results_gao]
    recovery_list_gao = [r[1] for r in results_gao]

    with Pool(processes=num_workers, initializer=init_worker) as pool:
        results_barber = pool.map(run_single_delta_barber, deltas)
    powers_barber = [r[0] for r in results_barber]
    recovery_list_barber = [r[1] for r in results_barber]


    power_results, recovery_results, success_results = check_power_multi_tau_delta_random_pair(
        n, p, sigma, tau_list, deltas, alpha, num_trials, n_jobs
    )


    df_gao = pd.DataFrame({
        "Tau": ["NA"] * len(deltas),
        "Delta": deltas,
        "Power": powers_gao,
        "Recovery_Rate": recovery_list_gao,
        "Successful_Trials": ["NA"] * len(deltas),
        "Method": ["Gao"] * len(deltas)
    })

    df_barber = pd.DataFrame({
        "Tau": ["NA"] * len(deltas),
        "Delta": deltas,
        "Power": powers_barber,
        "Recovery_Rate": recovery_list_barber,
        "Successful_Trials": ["NA"] * len(deltas),
        "Method": ["Barber"] * len(deltas)
    })

    rows_randomized = []
    for tau in tau_list:
        for delta in deltas:
            rows_randomized.append({
                "Tau": tau,
                "Delta": delta,
                "Power": power_results[tau][delta],
                "Recovery_Rate": recovery_results[tau][delta],
                "Successful_Trials": success_results[tau][delta],
                "Method": "Randomized"
            })
    df_randomized = pd.DataFrame(rows_randomized)

    df_all = pd.concat([df_gao, df_barber, df_randomized], ignore_index=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("../../results", f"combined_power_results_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "power_comparison.csv")
    df_all.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 6))
    for tau in tau_list:
        label = "Naive" if tau == 0 else f"tau={tau}"
        plt.plot(deltas, [power_results[tau][d] for d in deltas],
                 marker='o', label=label)

    plt.plot(deltas, powers_gao, marker='^', linestyle='--', color='purple', label='Gao')

    plt.plot(deltas, powers_barber, marker='s', linestyle='--', color='green', label='Barber')

    plt.xlabel("Delta")
    plt.ylabel("Power")
    plt.title("Power vs Delta")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, "combined_power_plot.png"))
    plt.close()