import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import matplotlib.patches as mpatches
from find_best_K import find_best_K_F, generate_alpha_list_exp, get_labels_at_K, find_best_K_chi
from hierarchical_clustering_invariant import *
from palmerpenguins import load_penguins
from alg2_one_trial2 import gap_statistic
from joblib import Parallel, delayed
from collections import Counter
from datetime import datetime
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--trial_id", type=int, default=0)
args = parser.parse_args()



if __name__ == '__main__':
    seed = int(args.trial_id)
    np.random.seed(seed)
    random.seed(seed)

    penguins_raw = load_penguins()
    penguins = penguins_raw[(penguins_raw["sex"] == "female") & (penguins_raw.notna().all(axis=1)) & (
        penguins_raw["year"].between(2007, 2008))]
    labels = penguins["species"]
    X = penguins[["flipper_length_mm", "bill_length_mm"]].to_numpy()
    n = X.shape[0]
    #ind = np.random.choice(X.shape[0], 30, replace=False)
    #X = X[ind, :]
    #n = 30
    true_K = 3
    total_alpha = 0.05
    Ks = []
    alpha_list = generate_alpha_list_exp(n, 0.05, decay_rate=0.5)

    print(f"Starting trial {seed} with n={n}")

    start_time = datetime.now()
    K, _, _, _ = find_best_K_F(X, tau=0.1, alpha_list=alpha_list, linkage = "complete",
                                     total_alpha=0.05, n_threshold=0.4*n, hard_threshold=0.1*n, seed = seed)
    K_hat_gap = gap_statistic(X,K_max=30, B=50)
    end_time = datetime.now()
    elapsed_minutes = (end_time - start_time).total_seconds() / 60

    print(f"Trial {seed} finished in {elapsed_minutes:.2f} minutes, K_hat = {K}")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    output_dir = os.path.join(base_dir, "results/raw/penguins")
    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(output_dir, f"K_trial_{seed}.csv")
    pd.DataFrame({"trial_id": [seed], "K_hat": [K], "K_hat_gap": [K_hat_gap], "elapsed_min": [elapsed_minutes]}).to_csv(out_path, index=False)

