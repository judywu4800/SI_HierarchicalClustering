import numpy as np
import matplotlib.pyplot as plt
from utils import *
import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import default_converter
from rpy2.robjects import numpy2ri
import warnings
import logging
from datetime import datetime
import random

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("rpy2.rinterface_lib.callbacks").setLevel(logging.ERROR)

ro.r('source("r_functions.R")')
ro.r('source("Barbers.R")')

def get_Gao_pval(X, K, linkage="single"):
    with localconverter(default_converter + numpy2ri.converter):
        ro.globalenv['X'] = ro.conversion.py2rpy(X)

    ro.globalenv['K_py'] = K
    ro.globalenv['link_py'] = linkage
    pval = ro.r("baseline_pval(X, K_py, link_py)")[0]
    return pval

def single_repeat(tau, label, n, p, sigma, alpha, num_trials):
    mu = np.zeros(p)
    p_values = []

    while len(p_values) < num_trials:
        X = generate_null_data(n, p, mu, sigma)
        model = AgglomerativeClustering(X, tau=tau, n_clusters=1, linkage="single")
        model.fit()

        winning_nodes = list(model.existing_clusters_log.keys())
        key = winning_nodes[-1]
        value_list = model.existing_clusters_log[key]

        if len(value_list) < 2:
            continue

        try:
            sampled_nodes = random.sample(value_list, 2)
            node = sampled_nodes[0].parent
            p_val, _, _ = model.merge_inference_F(node, grid_width=50, ncoarse=20, ngrid=1000)
            if not np.isnan(p_val):
                p_values.append(p_val)
        except Exception:
            continue

    return {"Tau": tau, "Type": label, "Type I Error": np.mean(np.array(p_values) < alpha)}

def baseline_repeat(n, p, sigma, alpha, num_trials):
    mu = np.zeros(p)
    p_values = []

    while len(p_values) < num_trials:
        X = generate_null_data(n, p, mu, sigma)
        try:
            p_val = get_Gao_pval(X, 1, linkage="single")
            if not np.isnan(p_val):
                p_values.append(p_val)
        except Exception:
            continue

    return {"Tau": "Gao et al.", "Type": "Gao et al.", "Type I Error": np.mean(np.array(p_values) < alpha)}

def check_type1_multi_tau_parallel(n, p, sigma, tau_list, alpha=0.05, num_trials=200, num_repeats=10,
                                   n_jobs=-1, include_baseline=True):
    tasks = []
    for tau in tau_list:
        label = "Naive" if tau == 0 else "Randomized"
        for _ in range(num_repeats):
            tasks.append((tau, label, n, p, sigma, alpha, num_trials))
    results = Parallel(n_jobs=n_jobs)(
        delayed(single_repeat)(*task) for task in tasks
    )

    if include_baseline:
        baseline_results = []
        for _ in range(num_repeats):
            result = baseline_repeat(n, p, sigma, alpha, num_trials)
            baseline_results.append(result)
    else:
        baseline_results = []

    all_results = results + baseline_results
    df_results = pd.DataFrame(all_results)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_results, x="Tau", y="Type I Error", hue="Type")
    plt.axhline(y=alpha, linestyle='--', color='red', label=f"Significance level α = {alpha}")
    plt.title(f"Distribution of Type I Error Rates over {num_repeats} Repetitions")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()
    return df_results

if __name__ == "__main__":
    import os

    n = 30
    p = 10
    sigma = 1
    tau_list = [0, 0.1, 0.25, 0.5, 1, 1.5, 2, 5, 10]
    K = 1
    alpha = 0.05
    num_trials = 200
    num_repeats = 10
    n_jobs = -1

    df_results = check_type1_multi_tau_parallel(n, p, sigma, tau_list,
                                                alpha=alpha, num_trials=num_trials,
                                                num_repeats=num_repeats, n_jobs=n_jobs,
                                                include_baseline=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_type1_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    df_results.to_csv(os.path.join(output_dir, "type1_error_results.csv"), index=False)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_results, x="Type", y="Type I Error", palette="Set2")
    plt.axhline(y=alpha, linestyle='--', color='red', label=f"α = {alpha}")
    plt.title(f"Type I Error Rates over {num_repeats} Repetitions per Method")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "type1_error_boxplot.png"))
    plt.close()