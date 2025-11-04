import random
import sys, os
sys.path.append(os.path.abspath('../../src'))
import logging
import warnings
from multiprocessing import get_context
from datetime import datetime
import numpy as np
import pandas as pd

from utils import generate_null_data

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("rpy2.rinterface_lib.callbacks").setLevel(logging.ERROR)

import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import default_converter
from rpy2.robjects import numpy2ri

R_SCRIPT = "/home/judydw/SI_HierarchicalClustering/src/r_functions.R"
#R_SCRIPT = "/Users/judydw/Documents/GitHub/SI_HierarchicalClustering/src/r_functions.R"
_GAO_FUN = None

def init_worker_gao(r_script_path):
    global _GAO_FUN
    ro.r(f'source("{r_script_path}")')
    _GAO_FUN = ro.r['get_gao_pval_clustered']


def one_repeat_task_gao(args):
    (n, p, sigma, K, alpha, num_trials,
     linkage, metric, seed, label) = args
    pvals = []
    mu = np.zeros(p)
    rng = np.random.default_rng(seed)
    while len(pvals) < num_trials:
        X = generate_null_data(n, p, mu, sigma, rng=rng)
        with localconverter(default_converter + numpy2ri.converter):
            X_r = ro.conversion.py2rpy(X)
        try:
            pv = float(_GAO_FUN(X_r, K, linkage, seed=int(seed))[0])
        except Exception:
            continue

        if np.isfinite(pv) and (not np.isnan(pv)):
            pvals.append(pv)
    #print(pvals)
    type1 = float(np.mean(np.array(pvals) < alpha))
    return {"Method": "Gao", "Label": (label or "Gao (sigma_clustered)"), "Type I Error": type1}

def check_type1_pool_gao(n, p, sigma,
                         K=3, alpha=0.05,
                         num_trials=200, num_repeats=20,
                         linkage='complete', metric='euclidean',
                         n_jobs=32, seed_master=1,
                         r_script=R_SCRIPT, label="Gao (sigma_clustered)"):

    ss = np.random.SeedSequence(seed_master)
    child_seeds = ss.spawn(num_repeats)

    tasks = []
    for i in range(num_repeats):
        tasks.append((n, p, sigma, K, alpha, num_trials,
                      linkage, metric,
                      int(child_seeds[i].generate_state(1)[0]), label))

    ctx = get_context('spawn')
    with ctx.Pool(processes=n_jobs, initializer=init_worker_gao, initargs=(r_script,)) as pool:
        results = list(pool.imap_unordered(one_repeat_task_gao, tasks, chunksize=50))

    df = pd.DataFrame(results)
    return df

if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)
    n, p, sigma = 30, 10, 1.0
    K = 3
    alpha = 0.05

    num_trials  = 200
    num_repeats = 100
    n_jobs = 32

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    output_dir = os.path.join(base_dir, "results/raw")
    os.makedirs(output_dir, exist_ok=True)

    df_results = check_type1_pool_gao(
        n=n, p=p, sigma=sigma,
        K=K, alpha=alpha,
        num_trials=num_trials, num_repeats=num_repeats,
        linkage='complete', metric='euclidean',
        n_jobs=n_jobs, seed_master=1,
        r_script=R_SCRIPT, label="Gao (sigma_clustered)"
    )

    df_results.to_csv(os.path.join(output_dir, "type1_gao_clustered.csv"), index=False)
