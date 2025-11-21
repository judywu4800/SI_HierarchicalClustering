import sys, os
sys.path.append(os.path.abspath('../../src'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from utils import generate_null_data, check_p_value_uniformity
import random
from find_best_K import find_best_K_F, find_best_K_chi, generate_alpha_list
from datetime import datetime
import os


def run_single_trial(tau, n, p, sigma, total_alpha):
    mu = np.zeros(p)
    X_null = generate_null_data(n,p, mu,sigma)
    alpha_list = generate_alpha_list(n=n, total_alpha=total_alpha)
    #alpha_list = np.ones((n-1))/(n-1)*0.05

    K_hat, _, _, _ = find_best_K_F(X_null, tau=tau, alpha_list=alpha_list.copy(), total_alpha=total_alpha)
    #K_hat, _, _ = find_best_K_chi(X_null, tau=tau, alpha_list=alpha_list.copy(), total_alpha=total_alpha)
    return int(K_hat > 1), K_hat  # 1 if false rejection


if __name__ == "__main__":
    #taus = [0, 0.01,0.025,0.05, 0.1,0.5,1]
    taus = [0,0.05]
    num_trials = 100
    n = 30
    p = 10
    sigma = 1
    total_alpha = 0.05

    results = []
    flat_k_hats = []

    for tau in taus:
        trial_results = Parallel(n_jobs=-1)(
            delayed(run_single_trial)(tau, n, p,sigma, total_alpha)
            for _ in range(num_trials)
        )

        errors, K_hats = zip(*trial_results)
        fwer = sum(errors) / num_trials

        tau_label = "naive" if tau == 0 else tau
        results.append({
            "tau": tau_label,
            "FWER": fwer,
            "num_trials": num_trials
        })

        flat_k_hats.extend([{"tau": tau_label, "K_hat": k} for k in K_hats])

    df_results = pd.DataFrame(results)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    output_dir = os.path.join(base_dir, "results/raw")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "fwer_results.csv")
    df_results.to_csv(output_file, index=False)

    #df_k_hats = pd.DataFrame(flat_k_hats)
    #k_hats_file = os.path.join(output_dir, "k_hats_flat.csv")
    #df_k_hats.to_csv(k_hats_file, index=False)
