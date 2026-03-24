import random
import sys, os
def get_repo_root():
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )

REPO_ROOT = get_repo_root()
sys.path.append(os.path.join(REPO_ROOT, "src"))
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from rand_hclust import *
from find_best_K import find_best_K_F, generate_alpha_list_exp
from scipy.cluster.hierarchy import linkage, fcluster, cut_tree
from sklearn.datasets import make_blobs

def generate_3cluster_data_varsize(n=50, p=2, delta=1.0, sigma=1.0,
                                   cluster_sizes=None, random_state=None, return_labels=True):
    rng = np.random.default_rng(random_state)

    # handle cluster sizes
    if cluster_sizes is None:
        if n % 3 != 0:
            raise ValueError("If cluster_sizes is None, n must be divisible by 3.")
        cluster_sizes = (n // 3, n // 3, n // 3)
    else:
        if len(cluster_sizes) != 3:
            raise ValueError("cluster_sizes must be a tuple/list of 3 integers.")
        if sum(cluster_sizes) != n:
            raise ValueError(f"sum(cluster_sizes)={sum(cluster_sizes)} must equal n={n}.")

    # define means (equilateral triangle in 2D, generalized in higher dimensions)
    mu1 = np.zeros(p)
    mu1[0] = -delta / 2

    mu2 = np.zeros(p)
    mu2[-1] = np.sqrt(3) * delta / 2

    mu3 = np.zeros(p)
    mu3[0] = delta / 2

    X1 = rng.normal(loc=mu1, scale=sigma, size=(cluster_sizes[0], p))
    X2 = rng.normal(loc=mu2, scale=sigma, size=(cluster_sizes[1], p))
    X3 = rng.normal(loc=mu3, scale=sigma, size=(cluster_sizes[2], p))

    X = np.vstack([X1, X2, X3])
    labels = np.array([0]*cluster_sizes[0] + [1]*cluster_sizes[1] + [2]*cluster_sizes[2])

    if return_labels:
        return X, labels
    else:
        return X
def generate_3cluster_data_varsize(n, p=2, delta=1.0, sigma=1.0, seed = None):
    n_each = n//3
    rng = np.random.default_rng(seed)
    cov = np.eye(p) * (sigma ** 2)
    mu1 = np.zeros(p); mu1[0] = -delta/2
    mu2 = np.zeros(p); mu2[-1] = np.sqrt(3)*delta/2
    mu3 = np.zeros(p); mu3[0] = delta/2
    X1 = multivariate_normal.rvs(mean=mu1, cov=cov, size=n_each, random_state=rng)
    X2 = multivariate_normal.rvs(mean=mu2, cov=cov, size=n_each, random_state=rng)
    X3 = multivariate_normal.rvs(mean=mu3, cov=cov, size=n_each, random_state=rng)
    X = np.vstack([X1, X2, X3])
    labels = np.array([0]*n_each + [1]*n_each + [2]*n_each)
    return X, labels

def generate_Kcluster_equal(n=120, p=2, K=3, delta=3.0, sigma=1.0, seed=None):
    rng = np.random.default_rng(seed)
    cov = np.eye(p) * (sigma ** 2)

    angles = np.linspace(0, 2*np.pi, K, endpoint=False)
    centers = np.stack([delta * np.cos(angles), delta * np.sin(angles)], axis=1)

    n_per = n // K
    remainder = n % K

    X_list = []
    labels = []

    for k in range(K):
        nk = n_per + (1 if k < remainder else 0)
        Xk = multivariate_normal.rvs(mean=centers[k], cov=cov, size=nk, random_state=rng)
        X_list.append(Xk)
        labels.extend([k] * nk)

    X = np.vstack(X_list)
    return X, np.array(labels), centers
def generate_Kcluster_custom(p=2, K=3, delta=3.0, sigma=1.0,
                             cluster_sizes=None, seed=None):
    rng = np.random.default_rng(seed)
    cov = np.eye(p) * (sigma ** 2)

    angles = np.linspace(0, 2*np.pi, K, endpoint=False)
    centers = np.stack([delta * np.cos(angles), delta * np.sin(angles)], axis=1)

    if cluster_sizes is None:
        n = 120
        n_per = n // K
        remainder = n % K
        cluster_sizes = [n_per + (1 if k < remainder else 0) for k in range(K)]

    X_list = []
    labels = []

    for k in range(K):
        nk = cluster_sizes[k]
        Xk = multivariate_normal.rvs(mean=centers[k], cov=cov,
                                     size=nk, random_state=rng)
        X_list.append(Xk)
        labels.extend([k] * nk)

    X = np.vstack(X_list)
    return X, np.array(labels), centers

def wcss_from_labels(X, labels):
    return sum(((X[labels==k] - X[labels==k].mean(axis=0))**2).sum() for k in np.unique(labels))
'''
def compute_wcss_hclust(X, n_clusters):
    Z = linkage(X, method="complete")
    labels = fcluster(Z, n_clusters, criterion="maxclust")
    return wcss_from_labels(X, labels)

def gap_statistic(X, K_max=30, B=50, eps = 1e-12):
    rng = np.random.RandomState(0)
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    log_wcss = np.zeros(K_max)
    gap = np.zeros(K_max)
    sk = np.zeros(K_max)
    for k in range(1, K_max + 1):
        wk = compute_wcss_hclust(X, k)
        log_wcss[k - 1] = np.log(wk)
        wk_refs = np.zeros(B)
        for b in range(B):
            X_ref = rng.uniform(mins, maxs, size=X.shape)
            wk_refs[b] = compute_wcss_hclust(X_ref, k)
        log_wk_refs = np.log(np.maximum(wk_refs, eps))
        gap[k-1] = np.mean(log_wk_refs) - log_wcss[k-1]
        sk[k-1] = np.std(log_wk_refs)*np.sqrt(1+1/B)
    for k in range(1, K_max):
        if gap[k-1] >= gap[k] - sk[k]:
            return k
    return K_max

'''

def compute_wcss_hclust(X, n_clusters, method="complete"):
    n = X.shape[0]
    if n_clusters <= 1:
        labels = np.zeros(n, dtype=int)
        return wcss_from_labels(X, labels)
    if n_clusters >= n:
        return 0.0
    Z = linkage(X, method=method)
    labels = cut_tree(Z, n_clusters=n_clusters).reshape(-1)
    return wcss_from_labels(X, labels)

def gap_statistic_full(X, K_max=30, B=50, method="complete", eps=1e-12, seed=0):
    n = X.shape[0]
    K_max = min(K_max, n - 1)

    rng = np.random.RandomState(seed)
    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    X_refs = [rng.uniform(mins, maxs, size=X.shape) for _ in range(B)]

    wk = np.zeros(K_max)
    log_wcss = np.zeros(K_max)
    gap = np.zeros(K_max)
    sk = np.zeros(K_max)

    for k in range(1, K_max + 1):
        wk[k - 1] = compute_wcss_hclust(X, k, method=method)
        log_wcss[k - 1] = np.log(max(wk[k - 1], eps))

        wk_refs = np.array([
            compute_wcss_hclust(X_ref, k, method=method) for X_ref in X_refs
        ])
        log_wk_refs = np.log(np.maximum(wk_refs, eps))

        gap[k - 1] = np.mean(log_wk_refs) - log_wcss[k - 1]
        sk[k - 1] = np.std(log_wk_refs, ddof=1) * np.sqrt(1 + 1 / B)

    K_hat = K_max
    for k in range(1, K_max):
        if gap[k - 1] >= gap[k] - sk[k]:
            K_hat = k
            break

    return {
        "K_hat": K_hat,
        "wk": wk,
        "gap": gap,
        "sk": sk,
        "log_wcss": log_wcss
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--trial", type=int, required=True)
    args = parser.parse_args()

    np.random.seed(0)
    random.seed(0)

    n=30

    #X, labels,_ = generate_Kcluster_equal(n=n, p=2, K=3, delta=args.delta, sigma=1.0, seed=int(1000*args.trial+args.delta+3))
    X, labels = generate_3cluster_data_varsize(n=n, p=2, delta=args.delta,sigma=1.0, seed=int(1000*args.trial+args.delta+3))
    alpha_list = generate_alpha_list_exp(n, 0.05, decay_rate=0.3)
    K_hat_F, p_values, alpha_seq, _ = find_best_K_F(X, tau=0.1, alpha_list=alpha_list,
                                     total_alpha=0.05, n_threshold=0.4*n, hard_threshold=0.1*n, seed=int(1000*args.trial+args.delta+3))
    K_hat_gap = gap_statistic_full(X,K_max=10, seed=int(1000*args.trial+args.delta+3))["K_hat"]

    out = pd.DataFrame([[args.delta, args.trial, K_hat_F, K_hat_gap]],
                       columns=["delta", "trial", "K_hat_F", "K_hat_gap"])

    os.makedirs(f"results/raw/fig5", exist_ok=True)
    out.to_csv(f"results/raw/fig5/delta{args.delta}_trial{args.trial}.csv",
               index=False)

    df = pd.DataFrame({
        "pval": p_values,
        "alpha": alpha_seq
    })
    df["reject"] = df["pval"] < df["alpha"]

    os.makedirs(f"results/raw/fig5/pvals", exist_ok=True)
    df.to_csv(f"results/raw/fig5/pvals/{args.delta}_trial{args.trial}.csv",)



