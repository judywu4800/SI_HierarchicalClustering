import random
import sys, os
sys.path.append(os.path.abspath('../src'))
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from hierarchical_clustering_invariant import AgglomerativeClustering
from utils import *
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

if __name__ == '__main__':
    rng_global = np.random.default_rng(42)
    n_each = 10
    sigma = 1
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(base_dir, "results/raw/fig7")
    os.makedirs(output_dir, exist_ok=True)

    delta_values = [2,4,6,8,10,12,14]
    taus = [0,0.1]
    n_runs = 500
    n_clusters = 2
    # Data collection
    results_wcss = []
    results_ari = []
    results_recovery = []

    for delta in delta_values:
        for i in range(n_runs):
            seed = rng_global.integers(1e9)
            rng = np.random.default_rng(seed)

            X, y = generate_data_barbers(n_each, delta, sigma, n_clusters=n_clusters,rng=rng)
            tss = np.sum((X - np.mean(X, axis=0)) ** 2)

            for tau in taus:
                if tau != 0:
                    clustering = AgglomerativeClustering(X, n_clusters=n_clusters, tau=tau, random_state=seed)
                    clustering.fit()
                    labels_pred = clustering.get_cluster_labels()
                    method = 'Randomized'
                else:
                    clustering = AgglomerativeClustering(X, n_clusters=n_clusters, tau=tau, random_state=seed)
                    clustering.fit()
                    labels_pred = clustering.get_cluster_labels()
                    method = 'Naive'

                # --- WCSS/TSS ---
                wcss = clustering.compute_wcss()
                results_wcss.append({'delta': delta,'Tau': tau, 'Method': method, 'WCSS/TSS': wcss / tss})

                # --- ARI ---
                ari = adjusted_rand_score(y, labels_pred)
                results_ari.append({'delta': delta,'Tau': tau, 'Method': method, 'ARI': ari})

                # --- Recovery probability ---
                recovered = all(len(set(y[c.points])) == 1 for c in clustering.K_clusters)

                results_recovery.append({
                    'delta': delta,
                    'Tau': tau,
                    'Method': method,
                    'Recovery': int(recovered)
                })

    # --- convert to DataFrames ---
    df_wcss = pd.DataFrame(results_wcss)
    df_ari = pd.DataFrame(results_ari)
    df_recovery = pd.DataFrame(results_recovery)
    df_wcss.to_csv(os.path.join(output_dir, f'df_wcss_deltas_K{n_clusters}_fig7.csv'))
    df_ari.to_csv(os.path.join(output_dir, f'df_ari_deltas_K{n_clusters}_fig7.csv'))
    df_recovery.to_csv(os.path.join(output_dir, f'df_recovery_deltas_K{n_clusters}_fig7.csv'))
