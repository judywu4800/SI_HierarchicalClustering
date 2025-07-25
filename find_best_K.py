import numpy as np
from hierarchical_clustering import AgglomerativeClustering
from utils import *
import warnings
import matplotlib.pyplot as plt

def generate_alpha_list(n=30, total_alpha=0.05, seed=42):
    if n % 3 != 0:
        raise ValueError("n must be divisible by 3.")

    np.random.seed(seed)

    length = n - 1
    group_size = length // 3

    # Generate unnormalized values in each group
    large = np.random.uniform(0.003, 0.008, size=group_size)
    medium = np.random.uniform(0.001, 0.003, size=group_size)
    small = np.random.uniform(1e-5, 5e-4, size=length - 2 * group_size)

    # Combine and normalize
    alpha_raw = np.concatenate([large, medium, small])
    alpha_list = total_alpha * alpha_raw / np.sum(alpha_raw)

    return alpha_list
def find_best_K_F(X, tau, alpha_list, n_threshold=2, linkage="complete", total_alpha=0.05):
    n = np.shape(X)[0]
    if not np.isclose(np.sum(alpha_list), total_alpha):
        raise ValueError(
            f"Alpha list should sum up to total alpha = {total_alpha}, " f"but got {np.sum(alpha_list):.4f} instead.")
    if len(alpha_list) != (n - 1):
        raise ValueError("The length of alpha_list should be equal to n - 1.")

    p_values = []
    alpha_seq = []
    K_hat = 1

    model = AgglomerativeClustering(X, tau=tau, n_clusters=1, linkage=linkage)
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

            if (n1 + n2) == 2:
                pval = 1  # F distribution method cannot handle the case when n1+n2=2
            else:
                pval, _, _ = model.merge_inference_F(node, grid_width=50, ncoarse=20, ngrid=1000)

        else:
            alpha = np.max(alpha_list)  # More power for larger clusters
            idx = np.argmax(alpha_list)
            pval, _, _ = model.merge_inference_F(node, grid_width=50, ncoarse=20, ngrid=1000)
        alpha_list = np.delete(alpha_list, idx)
        alpha_seq.append(alpha)
        p_values.append(pval)
        if pval < alpha:
            K_hat = n - t
            return (K_hat, p_values, alpha_seq)
    return (K_hat, p_values, alpha_seq)


def find_best_K_chi(X, tau, alpha_list, n_threshold=2, linkage="complete", total_alpha=0.05):
    n = np.shape(X)[0]
    if not np.isclose(np.sum(alpha_list), total_alpha):
        raise ValueError(
            f"Alpha list should sum up to total alpha = {total_alpha}, " f"but got {np.sum(alpha_list):.4f} instead.")
    if len(alpha_list) != (n - 1):
        raise ValueError("The length of alpha_list should be equal to n - 1.")

    p_values = []
    alpha_seq = []
    K_hat = 1

    model = AgglomerativeClustering(X, tau=tau, n_clusters=1, linkage=linkage)
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
            '''
            if (n1+n2) == 2:
                pval = 1 #F distribution method cannot handle the case when n1+n2=2
            else:
                pval,_,_ = model.merge_inference_F(node, grid_width=50, ncoarse=20, ngrid=1000)
            '''
        else:
            alpha = np.max(alpha_list)  # More power for larger clusters
            idx = np.argmax(alpha_list)
        pval, _, _ = model.merge_inference_chi(node, grid_width=50, ncoarse=20, ngrid=1000)
        alpha_list = np.delete(alpha_list, idx)
        alpha_seq.append(alpha)
        p_values.append(pval)
        if pval < alpha:
            K_hat = n - t
            return (K_hat, p_values, alpha_seq)
    return (K_hat, p_values, alpha_seq)
