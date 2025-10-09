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
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    random.seed(0)
    n_each = 10
    delta = 3
    sigma = 1
    output_dir = os.path.join("figures")
    os.makedirs(output_dir, exist_ok=True)


    plt.figure(figsize=(5,5))
    X, y = generate_data_barbers(n_each, delta, sigma,true_mean=False)
    plt.scatter(X[:, 0], X[:, 1], s=50, c=y, cmap='viridis', label='Data Points')
    plt.title('Agglomerative Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster Label')
    for i in range(len(X)):
        plt.annotate(str(i), (X[i, 0], X[i, 1]), fontsize=12, ha='right')

    plt.savefig(os.path.join(output_dir, "data_visualization.png"))
    plt.close()

    # Non-randomized hierarchical clustering using sklearn.cluster
    model = cluster.AgglomerativeClustering(n_clusters=3, linkage='complete')
    labels_sklearn = model.fit_predict(X)

    Z = linkage(X, method='complete')
    plt.figure(figsize=(5, 5))
    dendrogram(Z)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    plt.savefig(os.path.join(output_dir, "dendro_0.png"))
    plt.close()

    # Randomized hierarchical clustering with tau=0.01
    model001 = AgglomerativeClustering(X, tau=0.01, n_clusters=3, linkage="complete")
    model001.fit(dendrogram=True)

    model001.plot_dendrogram(save_fig=True, outdir = output_dir)