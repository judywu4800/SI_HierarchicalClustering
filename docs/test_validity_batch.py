import sys, os
import random
def get_repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

REPO_ROOT = get_repo_root()
sys.path.append(os.path.join(REPO_ROOT, "src"))

from utils import *
import argparse
import numpy as np
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--trial_id", type=int, required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    n = 200
    p = 2
    sigma = 1.0
    K = args.K
    tau = 0.1
    linkage = "complete"
    n_jobs = 1

    seed = args.seed if args.seed is not None else (100000 * K + args.trial_id)
    np.random.seed(seed)
    random.seed(seed)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(base_dir, "results/raw/fig3_batch", f"K{K}")
    os.makedirs(output_dir, exist_ok=True)

    X = generate_null_data(n, p, np.zeros(p), sigma)
    model = AgglomerativeClustering(X, tau=tau, n_clusters=K, linkage=linkage)
    model.fit()
    winning_nodes = list(model.existing_clusters_log.keys())
    key = winning_nodes[-1]  # test for last merge for example
    node = key[0].parent
    pval, _ = model.merge_inference_F(node, limit=50)

    row = {"pval": float(pval) if pval is not None and np.isfinite(pval) else np.nan}

    df = pd.DataFrame([row])
    out_path = os.path.join(
        output_dir,
        f"pval_validity_randomized_K{K}_trial{args.trial_id:04d}.csv",
    )
    df.to_csv(out_path, index=False)
