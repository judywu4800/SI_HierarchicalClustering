import numpy as np
from hierarchical_clustering_invariant import AgglomerativeClustering
from utils import *
import warnings
import matplotlib.pyplot as plt

def generate_alpha_list(n=30, total_alpha=0.05, seed=0):
    np.random.seed(seed)
    length = n - 1
    group_size = length // 3
    large = np.random.uniform(0.1, 0.5, size=5)
    medium = np.random.uniform(0.0005, 0.002, size=5)
    small = np.random.uniform(1e-7, 5e-6, size=length - 10)
    #small = np.zeros(length - 10)
    alpha_raw = np.concatenate([large, medium, small])
    alpha_list = total_alpha * alpha_raw / np.sum(alpha_raw)
    return alpha_list

import numpy as np

import numpy as np

def generate_alpha_list_exp(n=30, total_alpha=0.05, decay_rate=1):
    length = n - 1
    i = np.arange(length)
    alpha_raw = np.exp(-decay_rate * i)
    alpha_list = total_alpha * alpha_raw / np.sum(alpha_raw)
    return alpha_list

def get_labels_at_K(model, K):
    for winning_pair, clusters in model.existing_clusters_log.items():
        if len(clusters) == K:
            labels = np.empty(model.n_samples, dtype=int)
            for cid, cluster in enumerate(clusters):
                for idx in cluster.points:
                    labels[idx] = cid
            return labels
    raise ValueError(f"No step found with {K} clusters")
def find_best_K_F(X, tau, alpha_list, n_threshold=25, linkage="complete", total_alpha=0.05, rng = None):
    n = np.shape(X)[0]
    if not np.isclose(np.sum(alpha_list), total_alpha):
        raise ValueError(
            f"Alpha list should sum up to total alpha = {total_alpha}, " f"but got {np.sum(alpha_list):.4f} instead.")
    if len(alpha_list) != (n - 1):
        raise ValueError("The length of alpha_list should be equal to n - 1.")

    if rng is None:
        rng = np.random.default_rng()
    elif isinstance(rng, (int, np.integer)):
        rng = np.random.default_rng(rng)

    p_values = []
    alpha_seq = []
    K_hat = 1

    model = AgglomerativeClustering(X, tau=tau, n_clusters=1, linkage=linkage, random_state=rng.integers(1e9))
    model.fit()
    winning_nodes = list(model.existing_clusters_log.keys())  # to get all merges
    for t in range(len(winning_nodes)):
        winning_pair = winning_nodes[t]
        node1 = winning_pair[0]
        node2 = winning_pair[1]
        node = node1.parent
        n1 = len(node1.points)
        n2 = len(node2.points)
        if min(n1, n2) <= n_threshold:
            alpha = np.min(alpha_list)  # more conservative for smaller clusters
            idx = np.argmin(alpha_list)

            if (n1 + n2) == 2:
                pval = 1  # F distribution method cannot handle the case when n1+n2=2
            else:
                pval, _, _ = model.merge_inference_F_grid(node, grid_width=250, ncoarse=30, ngrid=1000)

        else:
            alpha = np.max(alpha_list)  # More power for larger clusters
            idx = np.argmax(alpha_list)
            pval, _, _ = model.merge_inference_F_grid(node, grid_width=250, ncoarse=30, ngrid=1000)
        alpha_list = np.delete(alpha_list, idx)
        alpha_seq.append(alpha)
        p_values.append(pval)
        if pval < alpha:
            K_hat = n - t
            labels_est = get_labels_at_K(model, K_hat)
            return (K_hat, p_values, alpha_seq, labels_est)
        labels_est = np.zeros(n)
    return (K_hat, p_values, alpha_seq, labels_est)


def find_best_K_chi(X, tau, alpha_list, sigma = None, n_threshold=0.5, linkage="complete", total_alpha=0.05):
    n = np.shape(X)[0]
    if not np.isclose(np.sum(alpha_list), total_alpha):
        raise ValueError(
            f"Alpha list should sum up to total alpha = {total_alpha}, " f"but got {np.sum(alpha_list):.4f} instead.")
    if len(alpha_list) != (n - 1):
        raise ValueError("The length of alpha_list should be equal to n - 1.")

    p_values = []
    alpha_seq = []
    K_hat = 1

    model = AgglomerativeClustering(X, sigma, tau=tau, n_clusters=1, linkage=linkage)
    model.fit()
    winning_nodes = list(model.existing_clusters_log.keys())  # to get all merges
    for t in range(len(winning_nodes)):
        winning_pair = winning_nodes[t]
        node1 = winning_pair[0]
        node2 = winning_pair[1]
        node = node1.parent
        n1 = len(node1.points)
        n2 = len(node2.points)
        if (1 / n1 + 1 / n2) >= n_threshold:
            alpha = np.min(alpha_list)  # more conservative for smaller clusters
            idx = np.argmin(alpha_list)
        else:
            alpha = np.max(alpha_list)  # More power for larger clusters
            idx = np.argmax(alpha_list)
        pval, _, _ = model.merge_inference_chi(node, grid_width=180, ncoarse=20, ngrid=1000, Sigma=1)
        alpha_list = np.delete(alpha_list, idx)
        alpha_seq.append(alpha)
        p_values.append(pval)
        if pval < alpha:
            K_hat = n - t
            return (K_hat, p_values, alpha_seq)
    return (K_hat, p_values, alpha_seq)
