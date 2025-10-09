import sys, os
sys.path.append(os.path.abspath('../../src'))
import numpy as np
from hierarchical_clustering_invariant import *
from utils import generate_3cluster_data, generate_data_barbers
from joblib import Parallel, delayed
import random
def find_best_K_F(X, tau, alpha_list, n_threshold=2, linkage="complete", total_alpha=0.05):
    n = np.shape(X)[0]
    if not np.isclose(np.sum(alpha_list), total_alpha):
        raise ValueError(
            f"Alpha list should sum up to total alpha = {total_alpha}, " 
            f"but got {np.sum(alpha_list):.4f} instead.")
    if len(alpha_list) != (n - 1):
        raise ValueError("The length of alpha_list should be equal to n - 1.")

    p_values = []
    alpha_seq = []
    K_hat = 1

    model = AgglomerativeClustering(X, tau=tau, n_clusters=1, linkage=linkage)
    model.fit()
    winning_nodes = list(model.existing_clusters_log.keys())  # to get all merges

    for t, winning_pair in enumerate(winning_nodes):
        node1, node2 = winning_pair
        node = node1.parent
        n1, n2 = len(node1.points), len(node2.points)

        if (1 / n1 + 1 / n2) >= n_threshold:
            alpha = np.min(alpha_list)  # conservative for smaller clusters
            idx = np.argmin(alpha_list)
            if (n1 + n2) == 2:
                pval = 1  # edge case
            else:
                pval, _, _ = model.merge_inference_F(node, grid_width=180, ncoarse=50, ngrid=1000)
        else:
            alpha = np.max(alpha_list)  # more power for larger clusters
            idx = np.argmax(alpha_list)
            pval, _, _ = model.merge_inference_F(node, grid_width=180, ncoarse=50, ngrid=1000)

        alpha_list = np.delete(alpha_list, idx)
        alpha_seq.append(alpha)
        p_values.append(pval)

        if pval < alpha:
            K_hat = n - t
            return (K_hat, p_values, alpha_seq, model)

    return (K_hat, p_values, alpha_seq, model)
def get_labels_at_K(model, K):
    for winning_pair, clusters in model.existing_clusters_log.items():
        if len(clusters) == K:
            labels = np.empty(model.n_samples, dtype=int)
            for cid, cluster in enumerate(clusters):
                for idx in cluster.points:
                    labels[idx] = cid
            return labels
    raise ValueError(f"No step found with {K} clusters")

def clusters_to_labels(clusters, n_samples):
    labels = np.empty(n_samples, dtype=int)
    for cid, cluster in enumerate(clusters):
        for idx in cluster.points:
            labels[idx] = cid
    return labels
def run_one_randomized_clustering(X, labels_true, K_true, tau=0.01, total_alpha=0.05, random_state=None):
    n = X.shape[0]
    alpha_list = np.full(n - 1, total_alpha / (n - 1))
    K_hat, _, _, model = find_best_K_F(X, tau=tau, alpha_list=alpha_list, total_alpha=total_alpha)
    preserve = True
    if K_hat > K_true:
        preserve = False

    labels_est = get_labels_at_K(model, K_true)

    for k in np.unique(labels_est):
        if len(np.unique(labels_true[labels_est == k])) > 1:
            preserve = False
            break

    return preserve, K_hat

def verify_corollary(B=200, n=200, K_star=3, alpha=0.05, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_one_randomized_clustering)(
            *generate_data_barbers(10, 8, 1),  # X, labels_true
            K_star,
            total_alpha=alpha,
            random_state=b
        )
        for b in range(B)
    )

    assumption_count = 0
    success_count = 0

    for preserve, K_hat in results:
        print(preserve, K_hat)

        if not preserve:
            continue
        assumption_count += 1
        if K_hat <= K_star:
            success_count += 1

   # if assumption_count == 0:
   #     print("The assumption is never satisfied.")
   #     return None

    proportion = success_count / assumption_count
    print(f"Assumption count: {assumption_count}/{B}")
    print(f"P(K_hat <= K*) ≈ {proportion:.3f}, lower bound: {1-alpha}")
    return proportion #, assumption_count, success_count

if __name__ == "__main__":
    random.seed(0)
    verify_corollary(B=1000, n=30, K_star=3, alpha=0.05, n_jobs=-1)

