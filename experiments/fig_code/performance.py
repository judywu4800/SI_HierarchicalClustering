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
    n_each = 10
    delta = 8
    sigma = 1
    output_dir = os.path.join("figures")
    os.makedirs(output_dir, exist_ok=True)

    tau_values = [0,0.01,0.05, 0.1,0.5,1,5]
    n_runs = 500
    n_clusters = 3
    # Data collection
    results_wcss = []
    results_ari = []
    results_recovery = []

    for _ in range(n_runs):
        # generate data once per repetition
        X, y = generate_data_barbers(n_each, delta, sigma)
        tss = np.sum((X - np.mean(X, axis=0)) ** 2)

        for tau in tau_values:
            if tau != 0:
                clustering = AgglomerativeClustering(X, n_clusters=n_clusters, tau=tau)
                clustering.fit()
                labels_pred = clustering.get_cluster_labels()
                method = 'Randomized'
            else:
                clustering = AgglomerativeClustering(X, n_clusters=n_clusters, tau=tau)
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

    # --- aggregate recovery probability ---
    df_recovery = df_recovery.groupby(['Tau', 'Method'])['Recovery'].mean().reset_index()
    df_recovery.rename(columns={'Recovery': 'Recovery Probability'}, inplace=True)

    plt.figure(figsize=(5, 5))
    sns.boxplot(data=df_wcss, x='Tau', y='WCSS/TSS', hue='Method', palette='Set2')
    plt.title('Comparison of WCSS/TSS')
    plt.xlabel('Tau (Scale of Randomization)')
    plt.ylabel('WCSS/TSS')
    plt.legend(title='Method', loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, "wcss_tss.png"))
    plt.close()

    plt.figure(figsize=(5, 5))
    sns.boxplot(data=df_ari, x='Tau', y='ARI', hue='Method', palette='Set2')
    plt.title('ARI vs. Randomization Tau')
    plt.xlabel('Tau (Scale of Randomization)')
    plt.ylabel('Adjusted Rand Index (ARI)')
    plt.legend(title='Method', loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, "ari.png"))
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.plot(df_recovery['Tau'], df_recovery['Recovery Probability'], marker='o')
    plt.title('Recovery Probability vs. Randomization Tau')
    plt.xlabel('Tau (Scale of Randomization)')
    plt.ylabel('Recovery Probability')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, "recovery_probability.png"))
    plt.close()
    '''
        results = []
    

    for tau in tau_values:
        for _ in range(n_runs):
            X, y = generate_data_barbers(n_each, delta, sigma)
            tss = np.sum((X - np.mean(X, axis=0)) ** 2)
            if tau != 0:
                # Randomized hierarchical clustering
                randomized_clustering = AgglomerativeClustering(X, n_clusters=n_clusters, tau=tau)
                randomized_clustering.fit()
                randomized_score = randomized_clustering.compute_wcss()
                results.append({'Tau': tau, 'Method': 'Randomized', 'WCSS/TSS': randomized_score / tss})
            else:
                naive = cluster.AgglomerativeClustering(n_clusters=n_clusters)
                naive_labels = naive.fit_predict(X)
                naive_score = 0
                for k in range(n_clusters):
                    cluster_points = X[naive_labels == k]
                    if len(cluster_points) > 0:
                        center = np.mean(cluster_points, axis=0)
                        naive_score += np.sum((cluster_points - center) ** 2)
                results.append({'Tau': tau, 'Method': "Naive", 'WCSS/TSS': naive_score / tss})


    # Convert results to a DataFrame
    df = pd.DataFrame(results)

    # Create side-by-side boxplots
    plt.figure(figsize=(5, 5))
    sns.boxplot(data=df, x='Tau', y='WCSS/TSS', hue='Method', palette='Set2')
    plt.title('Comparison of WCSS/TSS')
    plt.xlabel('Tau (Scale of Randomization)')
    plt.ylabel('WCSS/TSS')
    plt.legend(title='Method',loc = 'lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, "wcss_tss.png"))
    plt.close()

    results = []

    for tau in tau_values:
        for _ in range(n_runs):
            X, y = generate_data_barbers(n_each,delta,sigma)
            if tau != 0:
                # Randomized hierarchical clustering
                randomized_clustering = AgglomerativeClustering(X, n_clusters=n_clusters, tau=tau)
                randomized_clustering.fit()
                labels_pred = randomized_clustering.get_cluster_labels()
                method = 'Randomized'
            else:
                naive = AgglomerativeClustering(X, n_clusters=n_clusters, tau=tau)
                naive.fit()
                labels_pred = naive.get_cluster_labels()
                method = 'Naive'

            ari = adjusted_rand_score(y, labels_pred)
            results.append({'Tau': tau, 'Method': method, 'ARI': ari})
    df_ari = pd.DataFrame(results)

    plt.figure(figsize=(5, 5))
    sns.boxplot(data=df_ari, x='Tau', y='ARI', hue='Method', palette='Set2')
    plt.title('ARI vs. Randomization Tau')
    plt.xlabel('Tau (Scale of Randomization)')
    plt.ylabel('Adjusted Rand Index (ARI)')
    plt.legend(title='Method', loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, "ari.png"))
    plt.close()


    for tau in tau_values:
        recovery = 0
        for _ in range(n_runs):
            X, y = generate_data_barbers(n_each, delta, sigma)
            if tau != 0:
                # Randomized hierarchical clustering
                randomized_clustering = AgglomerativeClustering(X, n_clusters=n_clusters, tau=tau)
                randomized_clustering.fit()
                c1 = randomized_clustering.K_clusters[0]
                c2 = randomized_clustering.K_clusters[1]
                c1_points = c1.points
                c2_points = c2.points

                c1_true_clusters = set(y[c1_points])
                c2_true_clusters = set(y[c2_points])

                if len(c1_true_clusters) == 1 and len(c2_true_clusters) == 1:
                    recovery += 1
                method = 'Randomized'
            else:
                naive = AgglomerativeClustering(X, n_clusters=n_clusters, tau=tau)
                naive.fit()
                labels_pred = naive.get_cluster_labels()
                c1 = naive.K_clusters[0]
                c2 = naive.K_clusters[1]
                c1_points = c1.points
                c2_points = c2.points

                c1_true_clusters = set(y[c1_points])
                c2_true_clusters = set(y[c2_points])

                if len(c1_true_clusters) == 1 and len(c2_true_clusters) == 1:
                    recovery += 1
                method = 'Naive'

        results.append({'Tau': tau, 'Method': method, 'Recovery Probability': recovery/n_runs})
    df_recovery = pd.DataFrame(results)

    plt.figure(figsize=(5, 5))
    plt.plot(df_recovery['Tau'], df_recovery['Recovery Probability'], marker='o')
    plt.title('Recovery Probability vs. Randomization Tau')
    plt.xlabel('Tau (Scale of Randomization)')
    plt.ylabel('Recovery Probability')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, "recovery_probability.png"))
    plt.close()
    '''
