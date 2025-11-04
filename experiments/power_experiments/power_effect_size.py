import sys, os
sys.path.append(os.path.abspath('../../src'))
import numpy as np
from utils import *
from datetime import datetime
from pygam import LogisticGAM, s
import random


if __name__ == "__main__":
    import os
    random.seed(0)
    np.random.seed(0)
    n = 30
    p = 10
    sigma = 1
    tau_list = [0,0.01,0.025, 0.05, 0.1]
    deltas = np.linspace(5,20,9)
    alpha = 0.05
    num_trials = 2000
    n_jobs = -1

    df_trials = check_power_es_multi_tau_delta_random_pair(n, p, sigma, tau_list, deltas, alpha, num_trials, n_jobs)



    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    output_dir = os.path.join(base_dir, "results/raw")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "reject_effect_size.csv")
    df_trials.to_csv(csv_path, index=False)

