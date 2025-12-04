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
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
#import matplotlib as mpl
from matplotlib.colors import ListedColormap
import seaborn as sns
from find_best_K import find_best_K_F, generate_alpha_list
from collections import Counter
import glob

if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)
    rng = np.random.default_rng(0)
    tau = 0.1
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(BASE_DIR, "results", "figures")
    os.makedirs(output_dir, exist_ok=True)

    raw_dir = os.path.join(BASE_DIR, "results", "raw", "fig1")
    all_files = sorted(glob.glob(os.path.join(raw_dir, "findK_results_tau005_K*.npz")))
    Ks = []
    X = y = None
    true_K = 2

    for f in all_files:
        data = np.load(f)
        Ks.extend(data['Ks'])
        if X is None:  # only keep one representative dataset
            X = data['X']
            y = data['y']
            true_K = int(data['true_K'])
        data.close()

    Ks = np.array(Ks)
    #print(f"Loaded {len(all_files)} batch files, total {len(Ks)} trials combined.")

    #print(Ks)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))


    manual_colors = ["#ff924c", "#9579d9", "#7ab13f"]
    cluster_0 = y[0]
    cluster_10 = y[10]
    cluster_23 = y[23]

    color_dict = {
        cluster_0: "#ff924c",
        cluster_10: "#7ab13f",
        cluster_23: "#9579d9"
    }

    unique_labels = np.unique(y)
    remaining_clusters = [lab for lab in unique_labels if lab not in color_dict]
    remaining_colors = [c for c in manual_colors if c not in color_dict.values()]
    for lab, col in zip(remaining_clusters, remaining_colors):
        color_dict[lab] = col

    point_colors = [color_dict[label] for label in y]

    axes[0,0].scatter(X[:, 0], X[:, 1], s=50, c=point_colors)
    axes[0,0].set_title('(a) Data', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Feature 1', fontsize=12, fontweight='bold')
    axes[0,0].set_ylabel('Feature 2', fontsize=12, fontweight='bold')
    axes[0,0].tick_params(axis='x', labelsize=8)
    axes[0,0].tick_params(axis='y', labelsize=10)
    for i in range(len(X)):
        axes[0,0].annotate(str(i), (X[i, 0], X[i, 1]), fontsize=10, ha='center', va='center')

    for label_val in unique_labels:
        axes[0,0].scatter([], [], color=color_dict[label_val], label=f"Label {int(label_val)}")
    legend = axes[0,0].legend(
        title="Cluster",
        loc="lower right",
        fontsize=9,
        title_fontsize=11,
        frameon=False
    )
    legend.get_title().set_fontweight('bold')

    axes[1, 1].hist(Ks, bins=20, density=False, alpha=0.7,
                    color="#9579d9", edgecolor="black", label=r"$\hat{K}$")
    axes[1, 1].axvline(x=true_K, color='red', linestyle='--',
                       linewidth=2, label=fr"$K_{{\mathrm{{true}}}}$= {true_K}")
    axes[1, 1].set_xlabel("$\hat{K}$", fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel("Density", fontsize=12, fontweight='bold')
    axes[1, 1].set_title(r"(d) Histogram of $\hat{K}$", fontsize=14, fontweight='bold')
    #axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    axes[1, 1].legend(fontsize=10)

    model = cluster.AgglomerativeClustering(n_clusters=true_K, linkage='complete')
    labels_sklearn = model.fit_predict(X)
    Z = linkage(X, method='complete')

    target_points = [0, 10, 23]
    target_colors = ["#ff924c", "#7ab13f", "#9579d9"]
    cluster_color_map = {labels_sklearn[i]: c for i, c in zip(target_points, target_colors)}

    all_clusters = np.unique(labels_sklearn)
    unused_colors = [c for c in target_colors if c not in cluster_color_map.values()]
    for c in all_clusters:
        if c not in cluster_color_map:
            cluster_color_map[c] = unused_colors.pop(0) if unused_colors else "gray"


    def link_color_func(node_id):
        n_samples = len(labels_sklearn)
        if node_id < n_samples:
            return cluster_color_map[labels_sklearn[node_id]]
        else:
            left = int(Z[node_id - n_samples, 0])
            right = int(Z[node_id - n_samples, 1])
            left_color = link_color_func(left)
            right_color = link_color_func(right)
            return left_color if left_color == right_color else "gray"


    dendrogram(
        Z,
        color_threshold=0,
        above_threshold_color="gray",
        link_color_func=link_color_func,
        ax=axes[0,1],
    )
    axes[0,1].set_title("(b) Deterministic", fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel("Sample Index", fontsize=12, fontweight='bold')
    axes[0,1].set_ylabel("Distance", fontsize=12, fontweight='bold')
    axes[0,1].tick_params(axis='x', labelsize=8)
    axes[0,1].tick_params(axis='y', labelsize=10)
    axes[0,1].set_xticklabels(axes[0,1].get_xticklabels(), rotation=0)

    model_r = AgglomerativeClustering(X, tau=tau, n_clusters=true_K, linkage="complete", random_state=42)
    model_r.fit(dendrogram=True)
    #print(model_r.K_clusters)
    #print(model_r.existing_clusters_log)
    model_r.plot_dendrogram(ax=axes[1,0], show=False, save_fig=False, outdir=output_dir, manual_color=True)
    axes[1,0].set_title(f"(c) Randomized (tau={tau})", fontsize=14, fontweight='bold')


    plt.tight_layout(pad=0.2, w_pad=0.3)
    plt.savefig(os.path.join(output_dir, f"figure1_K{true_K}.png"),
                dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close()
