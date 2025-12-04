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
    K=2
    output_dir = os.path.join("../results/figures")
    df_wcss = pd.read_csv(f"../results/raw/df_wcss_{K}.csv")
    df_ari =  pd.read_csv(f"../results/raw/df_ari_{K}.csv")
    df_recovery =  pd.read_csv(f"../results/raw/df_recovery_{K}.csv")

    # --- aggregate recovery probability ---
    df_recovery = df_recovery.groupby(['Tau', 'Method'])['Recovery'].mean().reset_index()
    df_recovery.rename(columns={'Recovery': 'Recovery Probability'}, inplace=True)

    subset_tau = [0,0.025,0.05,0.1,0.25,0.5,1,5]
    #labels = ["Naive","RAC(0.025)","RAC(0.05)", "RAC(0.1)", "RAC(0.25)", "RAC(0.5)", "RAC(1)","RAC(5)"]
    labels = ["RC(0)", "RC(1)", "RC(2)", "RC(3)", "RC(4)", "RC(5)", "RC(6)", "RC(7)"]
    rename_map = dict(zip(subset_tau, labels))

    wcss_sub = df_wcss[df_wcss['Tau'].isin(subset_tau)].copy()
    wcss_sub["method"] = wcss_sub["Tau"].map(rename_map)
    ari_sub = df_ari[df_ari['Tau'].isin(subset_tau)].copy()
    ari_sub["method"] = ari_sub["Tau"].map(rename_map)
    recovery_sub = df_recovery[df_recovery['Tau'].isin(subset_tau)].copy()
    rec_x_pos = range(len(subset_tau))

    custom_colors = ["#FF758F", "#BFE8A4", "#9CBE86", "#7B9669", "#5B704D", "#3D4C33",  "#222B1B"]
    custom_colors = ["#FF758F","#C4EAA7", "#A9D595", "#8DBE7E","#729869", "#587450", "#3F5237", "#252D1D"]
    #custom_colors = ['#cc5e72', '#99ba83', '#7d986b', '#627854', '#495a3e', '#313d29', '#1b2216']
    fig, axes = plt.subplots(1, 2, figsize=(12,4))

    sns.boxplot(data=wcss_sub, x='method', y='WCSS/TSS', hue='method',
                palette=custom_colors, ax=axes[0])
    axes[0].set_title('Boxplot for WCSS/TSS', fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Method", fontsize=12)
    axes[0].set_ylabel("WCSS/TSS", fontsize=12)
    axes[0].tick_params(axis='x', labelsize=12)

    sns.boxplot(data=ari_sub, x='method', y='ARI', hue='method', palette=custom_colors, ax = axes[1])
    axes[1].set_title('Boxplot for ARI', fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Method", fontsize=12)
    axes[1].set_ylabel("ARI", fontsize=12)
    axes[1].tick_params(axis='x', labelsize=12)

    '''
    axes[2].plot(rec_x_pos, recovery_sub['Recovery Probability'], marker='o', color="#222B1B", linewidth=2)
    axes[2].set_xticks(rec_x_pos)
    axes[2].set_xticklabels(labels)
    axes[2].tick_params(axis='x', labelsize=7)
    axes[2].set_xlabel("Method", fontsize=12)
    axes[2].set_ylabel("Recovery Probability", fontsize=12)
    axes[2].set_title("Line Plot for Recovery Probability", fontsize=14, fontweight='bold')
    '''



    plt.tight_layout(pad=0.2, w_pad=0.3)
    plt.savefig(os.path.join(output_dir, f"figure2_K{K}.png"),
                dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close()

