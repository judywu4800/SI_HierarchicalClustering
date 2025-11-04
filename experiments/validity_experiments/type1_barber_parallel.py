import sys, os
import logging
import warnings
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import default_converter
from rpy2.robjects import pandas2ri

# R script path
R_SCRIPT = "/home/judydw/SI_HierarchicalClustering/src/r_functions.R"
#R_SCRIPT = "/Users/judydw/Documents/GitHub/SI_HierarchicalClustering/src/r_functions.R"

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("rpy2.rinterface_lib.callbacks").setLevel(logging.ERROR)

if __name__ == "__main__":
    n, p, sigma = 30, 10, 1.0
    K = 3
    alpha = 0.05
    num_trials = 200
    num_repeats = 100
    n_jobs = 32
    seed_master = 0

    print("Loading R script...")
    ro.r(f'source("{R_SCRIPT}")')
    run_gao_type1_parallel = ro.r['run_barber_type1_parallel']

    print("Running Gao_clustered Type I Error simulation in R (parallel)...")
    with localconverter(default_converter + pandas2ri.converter):
        df_results = run_gao_type1_parallel(
            n, p, sigma, K, alpha,
            num_trials, num_repeats,
            "complete", n_jobs, seed_master
        )
        df_results = pd.DataFrame(df_results)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    output_dir = os.path.join(base_dir, "results/raw")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "type1_barber.csv")
    df_results.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")