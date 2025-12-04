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


if __name__ == "__main__":
    n_clusters = 2
    output_dir = os.path.join("../results/figures")
    df_wcss = pd.read_csv(f"../results/raw/df_wcss_deltas_K{n_clusters}_fig3.csv")
    df_ari =  pd.read_csv(f"../results/raw/df_ari_deltas_K{n_clusters}_fig3.csv")
    df_recovery =  pd.read_csv(f"../results/raw/df_recovery_deltas_K{n_clusters}_fig3.csv")

    # --- aggregate recovery probability ---
    df_recovery = (
        df_recovery
        .groupby(['delta', 'Tau', 'Method'], as_index=False)['Recovery']
        .mean()
        .rename(columns={'Recovery': 'Recovery Probability'})
    )

    distances = [2,4,6,8,10,12,14]

    custom_colors = ["#FF758F",  "#9CBE86"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.boxplot(data=df_wcss, x='delta', y='WCSS/TSS', hue='Method',
                palette=custom_colors, ax=axes[0])
    axes[0].set_title('Boxplot for WCSS/TSS', fontsize=14, fontweight='bold')
    axes[0].set_xlabel(r"$\delta$", fontsize=12)
    axes[0].set_ylabel("WCSS/TSS", fontsize=12)
    axes[0].tick_params(axis='x', labelsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    new_labels = ["RC(0)" if lbl == "Naive" else "RC(3)" for lbl in labels]
    axes[0].legend(handles, new_labels, title="Method",loc = "center right")

    sns.boxplot(data=df_ari, x='delta', y='ARI', hue='Method', palette=custom_colors, ax = axes[1])
    axes[1].set_title('Boxplot for ARI', fontsize=14, fontweight='bold')
    axes[1].set_xlabel(r"$\delta$", fontsize=12)
    axes[1].set_ylabel("ARI", fontsize=12)
    axes[1].tick_params(axis='x', labelsize=9)
    handles, labels = axes[1].get_legend_handles_labels()
    new_labels = ["RC(0)" if lbl == "Naive" else "RC(3)" for lbl in labels]
    axes[1].legend(handles, new_labels, title="Method",loc = "center right")



    '''
    methods = df_recovery['Method'].unique()
    colors = {"Naive": custom_colors[0], "Randomized": custom_colors[1]}

    for method in methods:
        subset = df_recovery[df_recovery['Method'] == method]
        axes[2].plot(
            subset['delta'],
            subset['Recovery Probability'],
            marker='o',
            linewidth=2,
            color=colors[method],
            label=method
        )

    axes[2].set_title("Line Plot for Recovery Probability", fontsize=14, fontweight="bold")
    axes[2].set_xticks(distances)
    axes[2].set_xticklabels(distances)
    axes[2].set_xlabel(r"$\delta$", fontsize=12)
    axes[2].set_ylabel("Recovery Probability", fontsize=12)
    axes[2].legend(title="Method",loc = 'lower right', fontsize=10)

    '''


    plt.tight_layout(pad=0.2, w_pad=0.3)
    plt.savefig(os.path.join(output_dir, f"figure3_K{n_clusters}.png"),
                dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close()

