import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from hierarchical_clustering import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.special import gamma
from sklearn import cluster
from datetime import datetime
from joblib import Parallel, delayed

def generate_alternative_data(n, p, sigma, K, mean_range=5):
    X_parts = []
    n_per_cluster = n // K

    cluster_means = np.random.uniform(low=-mean_range, high=mean_range, size=(K, p))

    for i in range(K):
        X_cluster = np.random.normal(loc=cluster_means[i], scale=sigma, size=(n_per_cluster, p))
        X_parts.append(X_cluster)

    X = np.vstack(X_parts)
    return X


def generate_null_data(n, p, mu, sigma):
    mu = np.array(mu)  # Ensure mu is an array
    cov = (sigma ** 2) * np.eye(p)  # Covariance matrix
    return np.random.multivariate_normal(mu, cov, size=n)


def compute_nu(node, n):
    # return the projection direction from the given node
    G_1 = np.array(node.left.points)
    G_2 = np.array(node.right.points)
    n_G1 = len(G_1)
    n_G2 = len(G_2)

    nu = np.zeros(n)

    nu[G_1] += 1 / n_G1
    nu[G_2] -= 1 / n_G2
    return nu


def naive_p_value(X, node, sigma=1, grid_width=25, ngrid=1000):
    from scipy.stats import chi
    n = np.shape(X)[0]
    p = np.shape(X)[1]
    nu = compute_nu(node, n).reshape(-1, 1)
    # obs_target = np.linalg.norm(X.T@nu)/(np.linalg.norm(nu)*sigma)
    # Extract cluster indices
    left_indices = node.left.points
    right_indices = node.right.points

    # Compute cluster means
    left_mean = np.mean(X[left_indices, :], axis=0)
    right_mean = np.mean(X[right_indices, :], axis=0)

    # Compute the Euclidean distance between cluster means
    obs_target = np.linalg.norm(left_mean - right_mean)

    # Compute the scaling factor
    n_left = len(left_indices)
    n_right = len(right_indices)
    scale_factor = sigma * np.sqrt(1 / n_left + 1 / n_right)
    p_value = 1 - chi.cdf(obs_target / scale_factor, df=p)
    return p_value


def single_repeat(tau, label, n, p, sigma, K, layer, alpha, num_trials):
    mu = np.zeros(p)
    p_values = []

    for _ in range(num_trials):
        X = generate_null_data(n, p, mu, sigma)
        model = AgglomerativeClustering(X, tau=tau, n_clusters=K, linkage="single")
        model.fit()

        winning_nodes = list(model.existing_clusters_log.keys())
        key = winning_nodes[layer]
        node = key[0].parent

        p_val, _, _ = model.merge_inference(node, grid_width=20, ncoarse=20, ngrid=1000, sd=sigma)
        p_values.append(p_val)

    type_I_error = np.mean(np.array(p_values) < alpha)
    return {"Tau": tau, "Type": label, "Type I Error": type_I_error}


def check_type1_multi_tau_parallel(n, p, sigma, tau_list, K, layer, alpha=0.05, num_trials=200, num_repeats=10,
                                   n_jobs=-1):
    tasks = []
    for tau in tau_list:
        label = "Naive" if tau == 0 else "Randomized"
        for _ in range(num_repeats):
            tasks.append((tau, label, n, p, sigma, K, layer, alpha, num_trials))

    results = Parallel(n_jobs=n_jobs)(
        delayed(single_repeat)(*task) for task in tasks
    )

    df_results = pd.DataFrame(results)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_results, x="Tau", y="Type I Error", hue="Type")
    plt.axhline(y=alpha, linestyle='--', color='red', label=f"Significance level α = {alpha}")
    plt.title(f"Distribution of Type I Error Rates over {num_repeats} Repetitions (Layer {layer})")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()
    return df_results


if __name__ == "__main__":
    import os

    n = 30
    p = 10
    sigma = 1
    tau_list = [0,0.1, 0.25, 0.5, 1,1.5,2,5,10]
    K = 3
    layer = -1
    alpha = 0.05
    num_trials = 200
    num_repeats = 100
    n_jobs = -1

    df_results = check_type1_multi_tau_parallel(n, p, sigma, tau_list, K, layer,
                                                 alpha=alpha, num_trials=num_trials,
                                                 num_repeats=num_repeats, n_jobs=n_jobs)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_type1_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    df_results.to_csv(os.path.join(output_dir, "type1_error_results.csv"), index=False)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_results, x="Tau", y="Type I Error", hue="Type")
    plt.axhline(y=alpha, linestyle='--', color='red', label=f"Significance level α = {alpha}")
    plt.title(f"Distribution of Type I Error Rates over {num_repeats} Repetitions (Layer {layer})")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "type1_error_boxplot.png"))
    plt.close()