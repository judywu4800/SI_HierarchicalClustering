import random
import sys, os
sys.path.append(os.path.abspath('../../src'))
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
    random.seed(0)
    np.random.seed(0)
    n_each = 10
    delta = 6
    sigma = 1
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    output_dir = os.path.join(base_dir, "results/raw")
    os.makedirs(output_dir, exist_ok=True)

    tau_values = [0,0.05, 0.1,0.25,0.5,0.75,1,2,5]
    n_runs = 500
    n_clusters = 3
    # Data collection
    results_wcss = []
    results_ari = []
    results_recovery = []

    for i in range(n_runs):
        # generate data once per repetition
        X, y = generate_data_barbers(n_each, delta, sigma)
        tss = np.sum((X - np.mean(X, axis=0)) ** 2)

        for tau in tau_values:
            if tau != 0:
                clustering = AgglomerativeClustering(X, n_clusters=n_clusters, tau=tau, random_state=i)
                clustering.fit()
                labels_pred = clustering.get_cluster_labels()
                method = 'Randomized'
            else:
                clustering = AgglomerativeClustering(X, n_clusters=n_clusters, tau=tau, random_state=i)
                clustering.fit()
                labels_pred = clustering.get_cluster_labels()
                method = 'Naive'

            # --- WCSS/TSS ---
            wcss = clustering.compute_wcss()
            results_wcss.append({'Tau': tau, 'Method': method, 'WCSS/TSS': wcss / tss})

            # --- ARI ---
            ari = adjusted_rand_score(y, labels_pred)
            results_ari.append({'Tau': tau, 'Method': method, 'ARI': ari})

            # --- Recovery probability ---
            c1, c2 = clustering.K_clusters[0], clustering.K_clusters[1]  # reuse fitted
            c1_true = set(y[c1.points])
            c2_true = set(y[c2.points])
            recovered = (len(c1_true) == 1 and len(c2_true) == 1)
            results_recovery.append({'Tau': tau, 'Method': method, 'Recovery': int(recovered)})

    # --- convert to DataFrames ---
    df_wcss = pd.DataFrame(results_wcss)
    df_ari = pd.DataFrame(results_ari)
    df_recovery = pd.DataFrame(results_recovery)
    df_wcss.to_csv(os.path.join(output_dir, 'df_wcss.csv'))
    df_ari.to_csv(os.path.join(output_dir, 'df_ari.csv'))
    df_recovery.to_csv(os.path.join(output_dir, 'df_recovery.csv'))
