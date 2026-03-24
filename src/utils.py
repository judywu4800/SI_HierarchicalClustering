import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rand_hclust import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal,f
from joblib import Parallel, delayed
from itertools import combinations

# Data generation
def fun_gen_X(n, p, ss, delta=0):
    #ss： the real unknown sigma
    n_each = round(n / 2)
    mu1 = np.ones(p)*delta
    cov = ss * np.eye(p)
    X1 = multivariate_normal.rvs(mean=mu1, cov=cov, size=n_each)
    X2 = np.random.normal(loc=0, scale=np.sqrt(ss), size=(n_each, p))

    X = np.vstack([X1, X2])

    return X

def generate_null_data(n, p, mu=None, sigma=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if mu is None:
        mu = np.zeros(p)
    mu = np.asarray(mu)

    # handle sigma
    if np.isscalar(sigma):
        Sigma = (sigma ** 2) * np.eye(p)
    else:
        Sigma = np.asarray(sigma)
        if Sigma.shape != (p, p):
            raise ValueError(f"Covariance matrix must be {p}×{p}, got {Sigma.shape}.")

    if mu.shape[0] != p:
        raise ValueError(f"Mean vector length {mu.shape[0]} does not match p={p}.")

    X = rng.multivariate_normal(mean=mu, cov=Sigma, size=n)
    return X

def generate_3cluster_data(n=30, p=2, delta=1.0, sigma=1.0, random_state=None, return_labels=True):
    if n % 3 != 0:
        raise ValueError("n must be divisible by 3.")
    rng = np.random.default_rng(random_state)
    n_cluster = n // 3

    # Cluster 0 at -delta
    mu1 = np.zeros(p)
    mu1[0] = -delta/2

    # Cluster 1 at 0
    mu2 = np.zeros(p)
    mu2[-1] = np.sqrt(3) * delta / 2

    # Cluster 2 at +delta
    mu3 = np.zeros(p)
    mu3[0] = delta/2

    X1 = rng.normal(loc=mu1, scale=sigma, size=(n_cluster, p))
    X2 = rng.normal(loc=mu2, scale=sigma, size=(n_cluster, p))
    X3 = rng.normal(loc=mu3, scale=sigma, size=(n_cluster, p))

    X = np.vstack([X1, X2, X3])
    labels = np.array([0] * n_cluster + [1] * n_cluster + [2] * n_cluster)

    if return_labels:
        return X, labels
    else:
        return X
def generate_data_barbers(n_each, delta, sigma, n_clusters=3, true_mean=False, rng=None):
    if np.isscalar(sigma):
        cov = np.eye(2) * (sigma ** 2)
    else:
        cov = np.asarray(sigma)
        if cov.shape != (2, 2):
            raise ValueError(f"Expected 2x2 covariance matrix, got shape {cov.shape}")

    if rng is None:
        rng = np.random.default_rng()

    if n_clusters == 2:
        mus = [np.array([0, 0]),
               np.array([delta, 0])]
    elif n_clusters == 3:
        mus = [np.array([0, 0]),
               np.array([delta, 0]),
               np.array([delta / 2, np.sqrt(delta ** 2 - (delta ** 2) / 4)])]

    else:
        raise ValueError("n_clusters must be 2 or 3.")

    X_parts, labels_parts = [], []
    for i, mu in enumerate(mus, start=1):
        Xi = rng.multivariate_normal(mean=mu, cov=cov, size=n_each)
        X_parts.append(Xi)
        labels_parts.append(np.ones(n_each) * i)

    X = np.vstack(X_parts)
    labels = np.concatenate(labels_parts)

    mu = np.vstack(mus)[labels.astype(int) - 1]

    if true_mean:
        return X, labels, mu
    else:
        return X, labels


def compute_nu(node,n):
    # return the projection direction from the given node
    G_1= np.array(node.left.points)
    G_2 = np.array(node.right.points)
    n_G1 = len(G_1)
    n_G2 = len(G_2)

    nu = np.zeros(n)

    nu[G_1] += 1/n_G1
    nu[G_2] -= 1/n_G2
    return nu

def compute_nu_pair(c1,c2, n):
    # return the projection direction from the given node
    G_1 = np.array(c1.points)
    G_2 = np.array(c2.points)
    n_G1 = len(G_1)
    n_G2 = len(G_2)

    nu = np.zeros(n)

    nu[G_1] += 1 / n_G1
    nu[G_2] -= 1 / n_G2
    return nu

def create_indicator_diagonal_matrix(index_list, n):
    diag = np.zeros(n)
    diag[index_list] = 1
    return np.diag(diag), diag


def naive_p_value(X, K, layer, linkage):
    n = X.shape[0]
    p = X.shape[1]
    model = AgglomerativeClustering(X, tau=0, n_clusters=K, linkage=linkage) #fit non-randomized hierarchical clustering model
    model.fit()
    winning_nodes = list(model.existing_clusters_log.keys())
    key = winning_nodes[layer]
    node = key[0].parent
    nu = compute_nu(node, n).reshape(-1, 1)
    p_node_1 = node.left
    p_node_2 = node.right
    m = len(p_node_1.points) + len(p_node_2.points)
    c1 = model.K_clusters[0]
    c2 = model.K_clusters[1]
    P0 = nu @ nu.T / np.linalg.norm(nu) ** 2
    I1, one1 = create_indicator_diagonal_matrix(p_node_1.points, n)
    I2, one2 = create_indicator_diagonal_matrix(p_node_2.points, n)
    one1 = one1.reshape(-1, 1)
    one2 = one2.reshape(-1, 1)
    P1 = (I1 - one1 @ one1.T / len(p_node_1.points)) + (I2 - one2 @ one2.T / len(p_node_2.points))

    observed_target = (m - 2) * np.linalg.norm(P0 @ X, 'fro') ** 2 / np.linalg.norm(P1 @ X, 'fro') ** 2

    p_value = 1 - f.cdf(observed_target, dfn=p, dfd=(m - 2) * p)
    return p_value

def naive_p_value_random_pair(X, K, linkage):
    n = X.shape[0]
    p = X.shape[1]
    model = AgglomerativeClustering(X, tau=0, n_clusters=K, linkage=linkage) #fit non-randomized hierarchical clustering model
    model.fit()
    c1 = model.K_clusters[0]
    c2 = model.K_clusters[1]
    nu = compute_nu_pair(c1,c2,n).reshape(-1, 1)
    p_node_1 = c1
    p_node_2 = c2
    m = len(p_node_1.points) + len(p_node_2.points)
    if m <= 2:
        return np.nan
    P0 = nu @ nu.T / np.linalg.norm(nu) ** 2
    I1, one1 = create_indicator_diagonal_matrix(p_node_1.points, n)
    I2, one2 = create_indicator_diagonal_matrix(p_node_2.points, n)
    one1 = one1.reshape(-1, 1)
    one2 = one2.reshape(-1, 1)
    P1 = (I1 - one1 @ one1.T / len(p_node_1.points)) + (I2 - one2 @ one2.T / len(p_node_2.points))

    observed_target = (m - 2) * np.linalg.norm(P0 @ X, 'fro') ** 2 / np.linalg.norm(P1 @ X, 'fro') ** 2

    p_value = 1 - f.cdf(observed_target, dfn=p, dfd=(m - 2) * p)
    return p_value


def check_p_value_uniformity(n, p, sigma, K, tau, layer, linkage="complete", num_trials=1000):
    p_values = []
    p_values_n = []
    mu = np.zeros(p)

    while len(p_values_n) < num_trials:
        X = generate_null_data(n, p, mu, sigma)
        model = AgglomerativeClustering(X, tau=tau, n_clusters=K, linkage=linkage)
        model.fit()

        winning_nodes = list(model.existing_clusters_log.keys())
        key = winning_nodes[layer]
        node = key[0].parent
        c1 = model.K_clusters[0]
        c2 = model.K_clusters[1]
        #p_value, obs, sel_corrected = model.merge_inference_F_random_pair_grid(c1,c2, grid_width=180, ncoarse=20, ngrid=2000)
        p_value, obs = model.merge_inference_F_random_pair(c1,c2, limit = 100)
        p_value_n = naive_p_value(X, K, layer, linkage)
        if not (np.isnan(p_value) and np.isnan(p_value_n)):
            p_values.append(p_value)
            p_values_n.append(p_value_n)
    p_values = np.array(p_values)
    p_values_n = np.array(p_values_n)

    # Histogram for both p-values
    plt.figure(figsize=(8, 5))
    plt.hist(p_values, bins=20, density=True, alpha=0.5, color="blue", edgecolor="black",
             label="Selection-based p-value")
    plt.hist(p_values_n, bins=20, density=True, alpha=0.5, color="orange", edgecolor="black", label="Naive p-value")
    plt.axhline(1, color='red', linestyle='dashed', linewidth=2, label="Uniform(0,1)")
    plt.xlabel("P-value")
    plt.ylabel("Density")
    plt.title("Histogram of P-values Under the Null Hypothesis")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.ecdfplot(p_values, color="blue", label="Selection-based p-values")
    sns.ecdfplot(p_values_n, color="orange", label="Naive p-values")
    plt.plot([0, 1], [0, 1], linestyle="--", color="red", label="Expected (Uniform)")
    plt.xlabel("P-value")
    plt.ylabel("ECDF")
    plt.title("Empirical CDF of P-values")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

    # Q-Q Plot for both p-values
    plt.figure(figsize=(8, 5))
    sorted_p_values = np.sort(p_values)
    sorted_p_values_n = np.sort(p_values_n)
    theoretical_quantiles = np.linspace(0, 1, num_trials)

    plt.plot(theoretical_quantiles, sorted_p_values, marker='o', linestyle='', color="blue",
             label="Selection-based p-values")
    plt.plot(theoretical_quantiles, sorted_p_values_n, marker='o', linestyle='', color="orange", label="Naive p-values")
    plt.plot([0, 1], [0, 1], linestyle="--", color="red", label="Expected (Uniform)")
    plt.xlabel("Theoretical Uniform Quantiles")
    plt.ylabel("Empirical P-values")
    plt.title("Q-Q Plot: P-values vs. Uniform(0,1)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

def run_trial_random_pair(n, p, sigma, K, tau_list, linkage, random_state, method = "quad",limit=50, ngrid=None, ncoarse = None, grid_width=None):
    rng = np.random.default_rng(random_state)
    X = generate_null_data(n, p, np.zeros(p), sigma, rng=rng)
    trial_results = {}
    naive_val = naive_p_value_random_pair(X, K, linkage)
    trial_results['naive'] = naive_val

    for tau in tau_list:
        model = AgglomerativeClustering(X, tau=tau, n_clusters=K, linkage=linkage, random_state=random_state)
        model.fit()
        c1 = model.K_clusters[0]
        c2 = model.K_clusters[1]

        p_val, _= model.merge_inference_F_random_pair(c1,c2, method=method,limit=limit, ngrid=ngrid, ncoarse=ncoarse, grid_width=grid_width)
        trial_results[tau] = p_val

    return trial_results

def check_p_value_uniformity_multi_tau_random_pair_parallel(n, p, sigma, K, tau_list,
                                                linkage="complete", method="quad", limit=50, ngrid=None, ncoarse = None, grid_width=None,
                                                num_trials=1000, n_jobs=-1, seed=42):
    main_rng = np.random.default_rng(seed)
    rng_list = main_rng.spawn(num_trials)

    results = Parallel(n_jobs=n_jobs)(
        delayed(run_trial_random_pair)(
            n, p, sigma, K, tau_list, linkage,
            random_state=rng_list[i].integers(0, 2 ** 32 - 1),
            method=method,limit=limit, ngrid=ngrid, ncoarse=ncoarse, grid_width=grid_width
        )
        for i in range(num_trials)
    )

    all_p_values = {tau: [] for tau in tau_list}
    naive_p_values = []

    for res in results:
        naive_p_values.append(res['naive'])
        for tau in tau_list:
            all_p_values[tau].append(res[tau])

    for tau in tau_list:
        all_p_values[tau] = np.array(all_p_values[tau])
    naive_p_values = np.array(naive_p_values)

    return all_p_values, naive_p_values

def check_p_value_uniformity_single_tau_parallel(
        n, p, sigma, K, tau,
        linkage="complete",
        num_trials=1000,
        n_jobs=-1,
        seed=42):
    main_rng = np.random.default_rng(seed)
    rng_list = main_rng.spawn(num_trials)

    def one_trial(random_state):
        rng = np.random.default_rng(random_state)
        X = generate_null_data(n, p, np.zeros(p), sigma, rng=rng)

        res = {"naive": np.nan, "pval": np.nan}
        try:
            # naive p-value
            res["naive"] = naive_p_value_random_pair(X, K, linkage)

            # randomized clustering
            model = AgglomerativeClustering(
                X, tau=tau, n_clusters=K, linkage=linkage, random_state=random_state
            )
            model.fit()

            c1, c2 = model.K_clusters[0], model.K_clusters[1]
            if min(len(c1), len(c2)) <= 2:
                return res

            p_val, _ = model.merge_inference_F_random_pair(
                c1, c2, limit=50
            )
            res["pval"] = p_val
        except Exception as e:
            # skip failed trials safely
            res["pval"] = np.nan
        return res

    # --- run in parallel ---
    results = Parallel(n_jobs=n_jobs)(
        delayed(one_trial)(rng_list[i].integers(0, 2**32 - 1))
        for i in range(num_trials)
    )

    pvals = np.array([r["pval"] for r in results])
    naive_pvals = np.array([r["naive"] for r in results])


    return pvals, naive_pvals


def single_repeat_random_pair(tau, label, n, p, sigma, K, alpha, num_trials, random_state, method="quad", limit=50, ngrid=None, ncoarse = None, grid_width=None):
    rng = np.random.default_rng(random_state)
    mu = np.zeros(p)
    p_values = []
    while len(p_values)<num_trials:
        X = generate_null_data(n, p, mu, sigma, rng=rng)
        model = AgglomerativeClustering(X, tau=tau, n_clusters=K, linkage="complete", random_state=rng.integers(1e9))
        model.fit()

        c1 = model.K_clusters[0]
        c2 = model.K_clusters[1]

        p_val, _= model.merge_inference_F_random_pair(c1,c2, method=method,limit=limit, ngrid=ngrid, ncoarse=ncoarse, grid_width=grid_width)
        if not np.isnan(p_val):
            p_values.append(p_val)

    type_I_error = np.mean(np.array(p_values) < alpha)
    return {"Tau": tau, "Type": label, "Type I Error": type_I_error}


def check_type1_multi_tau_random_pair_parallel(n, p, sigma, tau_list, K, alpha=0.05,
                                               method="quad", limit=50, ngrid=None, ncoarse = None, grid_width=None,
                                               num_trials=200, num_repeats=10, n_jobs=-1, base_seed=0):
    tasks = []
    rng = np.random.default_rng(base_seed)
    for tau in tau_list:
        label = "Naive" if tau == 0 else "Randomized"
        for r in range(num_repeats):
            seed = rng.integers(1e9)
            tasks.append((tau, label, n, p, sigma, K, alpha, num_trials, seed, method, limit, ngrid, ncoarse, grid_width))

    results = Parallel(n_jobs=n_jobs)(
        delayed(single_repeat_random_pair)(*task) for task in tasks
    )

    df_results = pd.DataFrame(results)
    return df_results

def check_power_es_single_tau_fast(n, sigma, tau, deltas, alpha=0.05,
                                   num_trials=500, K=3, linkage="complete",
                                   n_jobs=-1, base_seed=0,
                                   method="quad", limit=50, ngrid=None, ncoarse = None, grid_width=None):
    """
    Efficient power/effect size simulation for a single tau value across multiple deltas.
    Parallelizes across trials for each delta.
    """

    rng = np.random.default_rng(base_seed)
    all_dfs = []

    def compute_es(true_mean,c1_points, c2_points, sigma, linkage):
        from scipy.spatial.distance import cdist
        X1 = true_mean[c1_points]
        X2 = true_mean[c2_points]
        dists = cdist(X1, X2, metric='euclidean')
        if linkage == "single":
            dist = np.min(dists)
        elif linkage == "complete":
            dist = np.max(dists)
        elif linkage == "average":
            dist = np.mean(dists)
        elif linkage == "centroid":
            mean1 = np.mean(X1, axis=0)
            mean2 = np.mean(X2, axis=0)
            dist = np.linalg.norm(mean1 - mean2)
        elif linkage == "minimax":
            all_points = np.vstack([X1, X2])
            pairwise = cdist(all_points, all_points, metric="euclidean")
            radii = np.max(pairwise, axis=1)
            dist = np.min(radii)
        else:
            raise ValueError(f"Unsupported linkage type: {linkage}")

        effect_size = dist / sigma
        return effect_size

    def one_trial(delta, seed):
        local_rng = np.random.default_rng(seed)
        X, true_labels, true_means = generate_data_barbers(
            n // K, delta, sigma, n_clusters=K, true_mean=True, rng=local_rng
        )
        model = AgglomerativeClustering(
            X, tau=tau, n_clusters=K, linkage=linkage,
            random_state=local_rng.integers(1e9)
        )
        model.fit()

        c1, c2 = model.K_clusters[0], model.K_clusters[1]
        c1_points, c2_points = c1.points, c2.points

        effect_size = compute_es(true_means, c1_points, c2_points, sigma, linkage)

        n1 = len(c1_points)
        n2 = len(c2_points)
        size10 =  int(min(n1, n2) >= 10)

        idx = np.concatenate([c1_points, c2_points])
        unique_labels = np.unique(true_labels[idx])
        non_alt = len(unique_labels) == 1

        p_val, _ = model.merge_inference_F_random_pair(
            c1, c2, method=method,limit=limit, ngrid=ngrid, ncoarse=ncoarse, grid_width=grid_width
        )
        reject = int(p_val < alpha)
        recovered = 0 if non_alt else 1

        return reject, effect_size, recovered, size10

    # -------------------------------
    # Loop over all deltas (parallel inside each)
    # -------------------------------
    for j, delta in enumerate(deltas):
        seeds = rng.integers(1e9, size=num_trials)
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(one_trial)(delta, s) for s in seeds
        )
        rejects, effects, recovs, size10s = zip(*results)
        df = pd.DataFrame({
            "tau": [tau] * num_trials,
            "delta": [delta] * num_trials,
            "effect_size": effects,
            "reject": rejects,
            "min_size>=10": size10s,
            "method": ["Randomized"] * num_trials
        })
        df["recovery_prob"] = np.mean(recovs)
        all_dfs.append(df)

    final_df = pd.concat(all_dfs, ignore_index=True)
    return final_df