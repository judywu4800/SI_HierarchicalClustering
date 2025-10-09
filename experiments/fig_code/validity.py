import sys, os
sys.path.append(os.path.abspath('../../src'))
from utils import *
from datetime import datetime
import random
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
def compute_pval_gao(X, K, linkage, method = "euclidean"):
    ro.r('source("/home/judydw/RAC_invariant/r_functions.R")')
    #ro.r('source("/Users/judydw/Documents/GitHub/SI_HierarchicalClustering/r_functions.R")')
    with localconverter(default_converter + numpy2ri.converter):
        ro.globalenv['X'] = ro.conversion.py2rpy(X)

    ro.globalenv['K_py'] = K
    ro.globalenv['link_py'] = linkage
    ro.globalenv['method_py'] = method
    pval = ro.r("get_gao_pval(X, K_py, link_py, method_py)")
    return pval

def compute_pval_gao_clustered(X, K, linkage, method = "euclidean"):
    ro.r('source("/home/judydw/RAC_invariant/r_functions.R")')
    #ro.r('source("/Users/judydw/Documents/GitHub/SI_HierarchicalClustering/r_functions.R")')
    with localconverter(default_converter + numpy2ri.converter):
        ro.globalenv['X'] = ro.conversion.py2rpy(X)

    ro.globalenv['K_py'] = K
    ro.globalenv['link_py'] = linkage
    ro.globalenv['method_py'] = method
    pval = ro.r("get_gao_pval_clustered(X, K_py, link_py, method_py)")
    return pval

def compute_pval_barber(X, K, method= "complete"):
    ro.r('source("/home/judydw/RAC_invariant/r_functions.R")')
    #ro.r('source("/Users/judydw/Documents/GitHub/SI_HierarchicalClustering/r_functions.R")')
    with localconverter(default_converter + numpy2ri.converter):
        ro.globalenv['X'] = ro.conversion.py2rpy(X)

    ro.globalenv['K_py'] = K
    ro.globalenv['method_py'] = method
    pval = ro.r("get_barber_pval(X, K_py, method_py)")
    return pval

def check_gao_uniformity(n, p, sigma, K, linkage = "complete", num_trials= 500):
    p_values = []
    mu = np.zeros(p)

    for _ in range(num_trials):
        X = generate_null_data(n,p,mu,sigma)
        pval = np.array(compute_pval_gao(X, K, linkage))[0]
        p_values.append(pval)

    return p_values

def check_gao_clustered_uniformity(n, p, sigma, K, linkage = "complete", num_trials= 500):
    p_values = []
    mu = np.zeros(p)

    for _ in range(num_trials):
        X = generate_null_data(n,p,mu,sigma)
        pval = np.array(compute_pval_gao_clustered(X, K, linkage))[0]
        p_values.append(pval)

    return p_values

def check_barber_uniformity(n, p, sigma, K, linkage = "complete", num_trials = 500):
    p_values = []
    mu = np.zeros(p)

    for _ in range(num_trials):
        X = generate_null_data(n, p, mu, sigma)
        pval = np.array(compute_pval_barber(X, K, linkage))[0]
        p_values.append(pval)

    return p_values

if __name__ == "__main__":
    import os
    random.seed(1)
    n = 30
    p = 10
    sigma = 1.0
    K = 3
    tau_list = [0.01, 0.05, 0.1, 0.5, 1, 5]
    layer = -1
    linkage = "complete"
    num_trials = 500
    n_jobs = -1

    output_dir_figs= os.path.join("figures")
    output_dir = os.path.join("results", f"pval_results")
    os.makedirs(output_dir, exist_ok=True)

    #RAC
    all_p_values, naive_p_values = check_p_value_uniformity_multi_tau_random_pair_parallel(
        n, p, sigma, K, tau_list, linkage, num_trials, n_jobs
    )

    df = pd.DataFrame({f"tau={tau}": all_p_values[tau] for tau in tau_list})
    df["naive"] = naive_p_values


    #Gao's with sigma_all
    pvals_gao = check_gao_uniformity(n, p, sigma, K, linkage, num_trials)
    # Gao's with sigma_clustered
    pvals_gao_c = check_gao_clustered_uniformity(n, p, sigma, K, linkage, num_trials)
    # Barber's
    pvals_barber = check_barber_uniformity(n, p, sigma, K, linkage, num_trials)

    pvals_result = {"Gao (sigma_all)": pvals_gao, "Gao (sigma_clustered)": pvals_gao_c, "Barber": pvals_barber}
    pvals_df = pd.DataFrame.from_dict(pvals_result)


    # Save results
    df.to_csv(os.path.join(output_dir, "pval_data_randomized.csv"), index=False)
    pvals_df.to_csv(os.path.join(output_dir, "pval_data_gao_barber.csv"), index=False)

    tau_cols = [c for c in df.columns if c.startswith('tau=')]
    naive = df['naive'].dropna().to_numpy()
    gao = pvals_df['Gao (sigma_all)']
    gao_c = pvals_df['Gao (sigma_clustered)']
    barber = pvals_df['Barber']

    def plot_ecdf(values, label):
        x = np.sort(values)
        y = np.arange(1, len(x) + 1) / len(x)
        plt.step(x, y, where='post', label=label)

    # Save ecdf plot
    plt.figure(figsize=(10, 6))
    for col in tau_cols:
        plot_ecdf(df[col].dropna().to_numpy(), f"Sel. ({col})")
    plot_ecdf(naive, "Naive")
    plot_ecdf(gao, "Gao et al(sigma_all)")
    plot_ecdf(gao_c, "Gao et al(sigma_clustered)")
    plot_ecdf(barber, "Yun and Barber")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Expected (Uniform)")
    plt.xlabel("P-value");
    plt.ylabel("ECDF")
    plt.title("ECDF of P-values")
    plt.legend();
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_figs, "ecdf_comparison.png"))
    plt.close()


