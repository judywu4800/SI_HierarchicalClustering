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

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    n_each = 10
    delta = 6
    sigma = 1
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(BASE_DIR, "results", "figures")
    os.makedirs(output_dir, exist_ok=True)

    X, y = generate_data_barbers(n_each, delta, sigma, true_mean=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    manual_colors =  ["#7ab13f","#ff924c",  "#9579d9"]
    unique_labels = np.unique(y)
    color_dict = {lab: manual_colors[i % len(manual_colors)] for i, lab in enumerate(unique_labels)}
    point_colors = [color_dict[label] for label in y]

    axes[0].scatter(X[:, 0], X[:, 1], s=50, c=point_colors)
    axes[0].set_title('Data', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Feature 1', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Feature 2', fontsize=12, fontweight='bold')
    axes[0].tick_params(axis='x', labelsize=8)
    axes[0].tick_params(axis='y', labelsize=10)
    for i in range(len(X)):
        axes[0].annotate(str(i), (X[i, 0], X[i, 1]), fontsize=10, ha='right', va='center')

    for label_val in unique_labels:
        axes[0].scatter([], [], color=color_dict[label_val], label=f"Label {int(label_val)}")
    legend = axes[0].legend(
        title="Cluster",
        loc="upper right",
        fontsize=10,
        title_fontsize=12,
        frameon=False
    )
    legend.get_title().set_fontweight('bold')

    model = cluster.AgglomerativeClustering(n_clusters=3, linkage='complete')
    labels_sklearn = model.fit_predict(X)
    Z = linkage(X, method='complete')
    manual_colors = ["#ff924c","#7ab13f",  "#9579d9"]
    unique_labels = np.unique(labels_sklearn)
    color_dict = {lab: manual_colors[i % len(manual_colors)] for i, lab in enumerate(unique_labels)}


    def link_color_func(node_id):
        n_samples = len(labels_sklearn)
        if node_id < n_samples:
            return color_dict[labels_sklearn[node_id]]
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
        ax=axes[1],
    )
    axes[1].set_title("Deterministic", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Sample Index", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Distance", fontsize=12, fontweight='bold')
    axes[1].tick_params(axis='x', labelsize=8)
    axes[1].tick_params(axis='y', labelsize=10)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

    model001 = AgglomerativeClustering(X, tau=0.01, n_clusters=3, linkage="complete", random_state=0)
    model001.fit(dendrogram=True)

    model001.plot_dendrogram(ax=axes[2], show=False, save_fig=False, outdir=output_dir, manual_color=True)
    axes[2].set_title("Randomized (tau=0.01)", fontsize=14, fontweight='bold')

    plt.tight_layout(pad=0.2, w_pad=0.3)
    plt.savefig(os.path.join(output_dir, "figure1.png"),
                dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close()