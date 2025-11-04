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
    ro.r('source("/home/judydw/SI_HierarchicalClustering/src/r_functions.R")')
    #ro.r('source("/Users/judydw/Documents/GitHub/SI_HierarchicalClustering/src/r_functions.R")')
def compute_pval_barber_es(X,true_means, K=3, linkage = "complete",seed=None):
    ro.r('source("/home/judydw/SI_HierarchicalClustering/src/r_functions.R")')
    #ro.r('source("/Users/judydw/Documents/GitHub/SI_HierarchicalClustering/src/r_functions.R")')
    with localconverter(default_converter + numpy2ri.converter):
        ro.globalenv['X'] = ro.conversion.py2rpy(X)
        ro.globalenv['true_mean_py'] = ro.conversion.py2rpy(true_means)
    ro.globalenv['K_py'] = K
    ro.globalenv['link_py'] = linkage
    ro.globalenv['seed'] = int(seed)
    result = ro.r("get_barber_pval_es(X,true_mean_py, K_py, link_py,seed)")
    return result
def compute_power_barber(sigma, delta, alpha, num_trials=10000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    p_values = []
    effect_sizes = []
    for _ in range(num_trials):
        trial_rng = np.random.default_rng(rng.integers(1e9))
        trial_seed = int(trial_rng.integers(1e9))
        X, true_label, true_means = generate_data_barbers(10, delta, sigma, true_mean = True, rng=trial_rng)
        pval, effect = np.array(compute_pval_barber_es(X,true_means, K=3, linkage="complete", seed=trial_seed))

        p_values.append(pval)
        effect_sizes.append(effect)
    #print(p_values)
    rejection = (np.array(p_values) < alpha).astype(int)
    return rejection, effect_sizes

def run_single_delta_barber(delta,base_seed = 0):
    rng = np.random.default_rng(base_seed + int(delta * 1000))
    return compute_power_barber(1, delta, 0.05, 2, rng=rng)

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    n = 30
    p = 10
    sigma = 1
    n_trials = 2000
    deltas = np.linspace(5,20,9)
    delta_list = np.repeat(deltas,n_trials)
    #deltas = [0]
    alpha = 0.05


    num_workers = os.cpu_count()


    with Pool(processes=num_workers, initializer=init_worker) as pool:
        results_gao = pool.map(run_single_delta_barber, deltas)

    rejections_all, effect_all = zip(*results_gao)
    rejections_all = np.concatenate(rejections_all, axis=0)
    effect_all = np.concatenate(effect_all, axis=0)

    df_gao = pd.DataFrame({
        "tau": ["NA"] * len(effect_all),
        "delta":delta_list,
        "effect_size": effect_all,
        "reject": rejections_all,
        "method": ["Barber"] * len(effect_all)
    })

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    output_dir = os.path.join(base_dir, "results/raw")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "rejection_es_barber.csv")
    df_gao.to_csv(csv_path, index=False)


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
    ro.r('source("/home/judydw/SI_HierarchicalClustering/src/r_functions.R")')
    #ro.r('source("/Users/judydw/Documents/GitHub/SI_HierarchicalClustering/src/r_functions.R")')
def compute_pval_barber_es(X,true_means, K=3, linkage = "complete",seed=None):
    ro.r('source("/home/judydw/SI_HierarchicalClustering/src/r_functions.R")')
    #ro.r('source("/Users/judydw/Documents/GitHub/SI_HierarchicalClustering/src/r_functions.R")')
    with localconverter(default_converter + numpy2ri.converter):
        ro.globalenv['X'] = ro.conversion.py2rpy(X)
        ro.globalenv['true_mean_py'] = ro.conversion.py2rpy(true_means)
    ro.globalenv['K_py'] = K
    ro.globalenv['link_py'] = linkage
    ro.globalenv['seed'] = int(seed)
    result = ro.r("get_barber_pval_es(X,true_mean_py, K_py, link_py,seed)")
    return result
def compute_power_barber(sigma, delta, alpha, num_trials=10000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    p_values = []
    effect_sizes = []
    for _ in range(num_trials):
        trial_rng = np.random.default_rng(rng.integers(1e9))
        trial_seed = int(trial_rng.integers(1e9))
        X, true_label, true_means = generate_data_barbers(10, delta, sigma, true_mean = True, rng=trial_rng)
        pval, effect = np.array(compute_pval_barber_es(X,true_means, K=3, linkage="complete", seed=trial_seed))

        p_values.append(pval)
        effect_sizes.append(effect)
    #print(p_values)
    rejection = (np.array(p_values) < alpha).astype(int)
    return rejection, effect_sizes

def run_single_delta_barber(delta,base_seed = 0):
    rng = np.random.default_rng(base_seed + int(delta * 1000))
    return compute_power_barber(1, delta, 0.05, 2000, rng=rng)

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    base_seed = 0
    n = 30
    p = 10
    sigma = 1
    n_trials = 2000
    deltas = np.linspace(5,20,9)
    delta_list = np.repeat(deltas,n_trials)
    #deltas = [0]
    alpha = 0.05


    num_workers = os.cpu_count()


    with Pool(processes=num_workers, initializer=init_worker) as pool:
        results_gao = pool.map(run_single_delta_barber, [(delta, base_seed) for delta in deltas])#pool.map(run_single_delta_barber, deltas)

    rejections_all, effect_all = zip(*results_gao)
    rejections_all = np.concatenate(rejections_all, axis=0)
    effect_all = np.concatenate(effect_all, axis=0)

    df_gao = pd.DataFrame({
        "tau": ["NA"] * len(effect_all),
        "delta":delta_list,
        "effect_size": effect_all,
        "reject": rejections_all,
        "method": ["Barber"] * len(effect_all)
    })

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    output_dir = os.path.join(base_dir, "results/raw")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "rejection_es_barber.csv")
    df_gao.to_csv(csv_path, index=False)


