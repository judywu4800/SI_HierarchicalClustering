import sys, os
sys.path.append(os.path.abspath('../../src'))
from hierarchical_clustering_invariant import AgglomerativeClustering
from hierarchical_clustering_adaptive_tau import AgglomerativeClustering_adaptivetau
import numpy as np
import pandas as pd
import argparse
from utils import generate_null_data,naive_p_value

def run_single_trial(n, p, sigma, K, epsilon, tau_list, linkage,n_trials =10, layer=-1, grid_width=120, ncoarse=20, ngrid=2000, seed=None):
    rng = np.random.default_rng(seed)
    mu = np.zeros(p)

    X = generate_null_data(n, p, mu, sigma,rng)
    model = AgglomerativeClustering_adaptivetau(X, epsilon, tau_list=tau_list, n_clusters=K, linkage=linkage, n_trials=n_trials, random_state=seed)
    model.choose_tau()
    model.fit_with_tau_star()
    tau_star = model.tau_star


    key = list(model.existing_clusters_log.keys())[layer]
    node = key[0].parent
    p_value, _, _ = model.merge_inference_F_grid(node, grid_width=grid_width, ncoarse=ncoarse, ngrid=ngrid)
    p_value_n = naive_p_value(X, K, layer, linkage)

    model_unadjusted = AgglomerativeClustering(X, n_clusters=K, linkage=linkage, tau=tau_star, random_state=seed)
    model_unadjusted.fit()
    winning_nodes = list(model_unadjusted.existing_clusters_log.keys())
    key = winning_nodes[layer]
    node = key[0].parent

    p_value_unadj, _, _ = model_unadjusted.merge_inference_F_grid(node, grid_width=150, ncoarse=20, ngrid=2000)

    return p_value, p_value_unadj, p_value_n


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial_id", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--p", type=int)
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--K", type=int)
    parser.add_argument("--epsilon", type=float)
    #parser.add_argument("--tau_list", nargs="+", type=float)
    #parser.add_argument("--linkage", type=str)
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    output_dir = os.path.join(base_dir, f"results/raw/validity_tau_epsilon{args.epsilon}")
    os.makedirs(output_dir, exist_ok=True)

    tau_list = np.linspace(0, 5, 20)
    #tau_list = [0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1, 2, 5]
    pval = run_single_trial(
        n=args.n,
        p=args.p,
        sigma=args.sigma,
        K=args.K,
        epsilon=args.epsilon,
        tau_list=tau_list,
        linkage="complete",
        seed=args.trial_id
    )

    with open(os.path.join(output_dir, f"trial_{args.trial_id}.txt"), "w") as f:
        f.write(str(pval))


