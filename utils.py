import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hierarchical_clustering_invariant import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal,f
from joblib import Parallel, delayed
from itertools import combinations

def generate_data_barbers(n_each, delta, sigma, true_mean = False):
    cov = np.eye(2) * sigma**2
    mu1 = np.array([0, 0])
    mu2 = np.array([delta, 0])
    mu3 = np.array([delta / 2, np.sqrt(delta ** 2 - delta ** 2 / 4)])
    #mu1 = np.array([0, 0])
    #mu2 = np.array([delta, 0])
    #mu3 = np.array([2*delta, 0])

    X1 = multivariate_normal.rvs(mean=mu1, cov=cov, size=n_each)
    X2 = multivariate_normal.rvs(mean=mu2, cov=cov, size=n_each)
    X3 = multivariate_normal.rvs(mean=mu3, cov=cov, size=n_each)

    labels1 = np.ones(n_each)
    labels2 = np.ones(n_each) * 2
    labels3 = np.ones(n_each) * 3

    X = np.vstack([X1, X2, X3])
    labels = np.concatenate([labels1, labels2, labels3])
    cluster_means = np.vstack([mu1, mu2, mu3])
    mu = cluster_means[labels.astype(int) - 1]
    if true_mean:
        return X, labels, mu
    else:
        return X, labels

def fun_gen_X(n, p, ss, delta=0):
    #ss： the real unknown sigma
    n_each = round(n / 2)
    mu1 = np.ones(p)*delta
    cov = ss * np.eye(p)
    X1 = multivariate_normal.rvs(mean=mu1, cov=cov, size=n_each)
    X2 = np.random.normal(loc=0, scale=np.sqrt(ss), size=(n_each, p))

    X = np.vstack([X1, X2])

    return X

def generate_null_data(n, p, mu, sigma):
    mu = np.array(mu)  # Ensure mu is an array
    cov = (sigma ** 2) * np.eye(p)  # Covariance matrix
    return np.random.multivariate_normal(mu, cov, size=n)

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
        p_value, obs, sel_corrected = model.merge_inference_F(node, grid_width=30, ncoarse=20, ngrid=2000)
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

def check_p_value_uniformity_multi_tau(n, p, sigma, K, tau_list, layer, linkage="complete", num_trials=1000):
    all_p_values = {tau: [] for tau in tau_list}
    naive_p_values = []

    mu = np.zeros(p)

    for trial in range(num_trials):
        X = generate_null_data(n, p, mu, sigma)
        naive_val = naive_p_value(X, K, layer, linkage)
        naive_p_values.append(naive_val)

        for tau in tau_list:
            model = AgglomerativeClustering(X, tau=tau, n_clusters=K, linkage=linkage)
            model.fit()

            winning_nodes = list(model.existing_clusters_log.keys())
            key = winning_nodes[layer]
            node = key[0].parent

            p_val, _, _ = model.merge_inference_F(node, grid_width=5, ncoarse=20, ngrid=2000)
            all_p_values[tau].append(p_val)

    for tau in tau_list:
        all_p_values[tau] = np.array(all_p_values[tau])
    naive_p_values = np.array(naive_p_values)

    plt.figure(figsize=(10, 6))
    for tau in tau_list:
        plt.hist(all_p_values[tau], bins=20, density=True, alpha=0.4, label=f"Sel. (tau={tau})", edgecolor='black')
    plt.hist(naive_p_values, bins=20, density=True, alpha=0.4, label=f"Naive", edgecolor='gray', linestyle='dashed')
    plt.axhline(1, color='red', linestyle='dashed', linewidth=2, label="Uniform(0,1)")
    plt.xlabel("P-value")
    plt.ylabel("Density")
    plt.title("Histogram of P-values Under the Null")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

    plt.figure(figsize=(10, 6))
    for tau in tau_list:
        sns.ecdfplot(all_p_values[tau], label=f"Sel. (tau={tau})", linestyle="-")
    sns.ecdfplot(naive_p_values, label="Naive", linestyle="--")
    plt.plot([0, 1], [0, 1], linestyle="--", color="red", label="Expected (Uniform)")
    plt.xlabel("P-value")
    plt.ylabel("ECDF")
    plt.title("ECDF of P-values")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()


    plt.figure(figsize=(10, 6))
    theoretical_quantiles = np.linspace(0, 1, num_trials)
    for tau in tau_list:
        sorted_sel = np.sort(all_p_values[tau])
        plt.plot(theoretical_quantiles, sorted_sel, marker='o', linestyle='', label=f"Sel. (tau={tau})")
    sorted_naive = np.sort(naive_p_values)
    plt.plot(theoretical_quantiles, sorted_naive, marker='x', linestyle='', label="Naive")
    plt.plot([0, 1], [0, 1], linestyle="--", color="red", label="Expected (Uniform)")
    plt.xlabel("Theoretical Uniform Quantiles")
    plt.ylabel("Empirical P-values")
    plt.title("Q-Q Plot: P-values vs. Uniform(0,1)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

def run_trial(n, p, sigma, K, tau_list, layer, linkage):
    X = generate_null_data(n, p, np.zeros(p), sigma)
    trial_results = {}

    naive_val = naive_p_value(X, K, layer, linkage)
    trial_results['naive'] = naive_val

    for tau in tau_list:
        model = AgglomerativeClustering(X, tau=tau, n_clusters=K, linkage=linkage)
        model.fit()
        winning_nodes = list(model.existing_clusters_log.keys())
        key = winning_nodes[layer]
        node = key[0].parent
        p_val, _, _ = model.merge_inference_F(node, grid_width=5, ncoarse=20, ngrid=2000)
        trial_results[tau] = p_val

    return trial_results

def check_p_value_uniformity_multi_tau_parallel(n, p, sigma, K, tau_list, layer,
                                                linkage="complete", num_trials=1000, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_trial)(n, p, sigma, K, tau_list, layer, linkage)
        for _ in range(num_trials)
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

def run_trial_random_pair(n, p, sigma, K, tau_list, linkage):
    X = generate_null_data(n, p, np.zeros(p), sigma)
    trial_results = {}
    naive_val = naive_p_value_random_pair(X, K, linkage)
    trial_results['naive'] = naive_val

    for tau in tau_list:
        model = AgglomerativeClustering(X, tau=tau, n_clusters=K, linkage=linkage)
        model.fit()
        c1 = model.K_clusters[0]
        c2 = model.K_clusters[1]

        p_val, _, _ = model.merge_inference_F_random_pair(c1,c2, grid_width=10, ncoarse=20, ngrid=2000)
        trial_results[tau] = p_val

    return trial_results

def check_p_value_uniformity_multi_tau_random_pair_parallel(n, p, sigma, K, tau_list,
                                                linkage="complete", num_trials=1000, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_trial_random_pair)(n, p, sigma, K, tau_list, linkage)
        for _ in range(num_trials)
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
def single_repeat(tau, label, n, p, sigma, K, layer, alpha, num_trials):
    mu = np.zeros(p)
    p_values = []

    while len(p_values)<num_trials:
        X = generate_null_data(n, p, mu, sigma)
        model = AgglomerativeClustering(X, tau=tau, n_clusters=K, linkage="complete")
        model.fit()

        winning_nodes = list(model.existing_clusters_log.keys())
        key = winning_nodes[layer]
        node = key[0].parent

        p_val, _, _ = model.merge_inference_F(node, grid_width=5, ncoarse=20, ngrid=1000)
        if not np.isnan(p_val):
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
    plt.axhline(y=alpha, linestyle='--', color='red', label   =f"Significance level α = {alpha}")
    plt.title(f"Distribution of Type I Error Rates over {num_repeats} Repetitions")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()
    return df_results

def single_repeat_random_pair(tau, label, n, p, sigma, K, alpha, num_trials):
    mu = np.zeros(p)
    p_values = []

    while len(p_values)<num_trials:
        X = generate_null_data(n, p, mu, sigma)
        model = AgglomerativeClustering(X, tau=tau, n_clusters=K, linkage="complete")
        model.fit()

        c1 = model.K_clusters[0]
        c2 = model.K_clusters[1]

        p_val, _, _ = model.merge_inference_F_random_pair(c1,c2, grid_width=8, ncoarse=20, ngrid=2000)
        if not np.isnan(p_val):
            p_values.append(p_val)

    type_I_error = np.mean(np.array(p_values) < alpha)
    return {"Tau": tau, "Type": label, "Type I Error": type_I_error}


def check_type1_multi_tau_random_pair_parallel(n, p, sigma, tau_list, K, alpha=0.05, num_trials=200, num_repeats=10,
                                   n_jobs=-1):
    tasks = []
    for tau in tau_list:
        label = "Naive" if tau == 0 else "Randomized"
        for _ in range(num_repeats):
            tasks.append((tau, label, n, p, sigma, K, alpha, num_trials))

    results = Parallel(n_jobs=n_jobs)(
        delayed(single_repeat_random_pair)(*task) for task in tasks
    )

    df_results = pd.DataFrame(results)
    return df_results


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



def single_tau_power(tau, n, p, sigma, delta, alpha, num_trials=500, max_attempts=50000):
    p_values = []
    recovery = 0
    trial_count = 0

    for _ in range(num_trials):
    #while len(p_values) < num_trials:
        trial_count += 1
        X, true_labels = generate_3cluster_data(n=n, p=p, delta=delta, sigma=sigma)
        model = AgglomerativeClustering(X, tau=tau, n_clusters=2, linkage="complete")
        model.fit()

        winning_nodes = list(model.existing_clusters_log.keys())
        key = winning_nodes[-1]
        c1, c2 = key[0], key[1]
        c1_points = c1.points
        c2_points = c2.points

        c1_true_clusters = set(true_labels[c1_points])
        c2_true_clusters = set(true_labels[c2_points])

        if len(c1_true_clusters) == 1 and len(c2_true_clusters) == 1:
            recovery += 1
            node = c1.parent
            p_val, _, _ = model.merge_inference_F(node, grid_width=15, ncoarse=20, ngrid=1000)
            p_values.append(p_val)
        '''
        if trial_count > max_attempts:
            print(f"Warning: Too few matching merges at tau={tau}")
            break
        
        '''


    power = np.mean(np.array(p_values) < alpha)
    recovery_prob = recovery / num_trials
    success = len(p_values) == num_trials
    return tau, power, recovery_prob, success

def check_power_multi_tau_parallel(n, p, sigma, tau_list, delta=10.0, alpha=0.05,
                                    num_trials=500, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs)(
        delayed(single_tau_power)(tau, n, p, sigma, delta, alpha, num_trials)
        for tau in tau_list
    )

    power_results_sel = {tau: power for tau, power, _, _ in results}
    recovery_results = {tau: rec for tau, _, rec, _ in results}
    full = [success for _, _, _, success in results]

    tau_vals = np.array(tau_list)
    power_vals = [power_results_sel[tau] for tau in tau_vals]
    recovery_vals = [recovery_results[tau] for tau in tau_vals]

    fig, ax1 = plt.subplots(figsize=(8, 6))

    color_power = 'tab:blue'
    ax1.set_xlabel("Tau (Randomization Level)")
    ax1.set_ylabel("Conditional Power", color=color_power)
    ax1.tick_params(axis='y', labelcolor=color_power)
    ax1.set_ylim(0, 1)

    if 0 in tau_vals:
        naive_idx = np.where(tau_vals == 0)[0][0]
        ax1.scatter(tau_vals[naive_idx], power_vals[naive_idx], color='orange', marker='s', s=100, label="Naive Power (τ=0)", zorder=5)

        tau_random = tau_vals[tau_vals != 0]
        power_random = [power_results_sel[t] for t in tau_random]
        ax1.plot(tau_random, power_random, marker='o', color=color_power, label="Randomized Power (τ>0)")
    else:
        ax1.plot(tau_vals, power_vals, marker='o', color=color_power, label="Conditional Power")

    ax2 = ax1.twinx()
    color_recovery = 'tab:red'
    ax2.set_ylabel("Recovery Probability", color=color_recovery)
    ax2.tick_params(axis='y', labelcolor=color_recovery)
    ax2.set_ylim(0, 1)

    if 0 in tau_vals:
        ax2.scatter(tau_vals[naive_idx], recovery_vals[naive_idx], color='darkorange', marker='D', s=100, label="Naive Recovery (τ=0)", zorder=5)
        recovery_random = [recovery_results[t] for t in tau_random]
        ax2.plot(tau_random, recovery_random, marker='s', linestyle='--', color=color_recovery, label="Randomized Recovery (τ>0)")
    else:
        ax2.plot(tau_vals, recovery_vals, marker='s', linestyle='--', color=color_recovery, label="Recovery Probability")

    plt.title("Conditional Power and Recovery Probability vs. Tau")
    fig.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.5)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    plt.legend(h1 + h2, l1 + l2, loc="upper right")
    plt.show()

    return power_results_sel, recovery_results, full

def single_delta_power(delta, n, p, sigma, tau, alpha, num_trials=500, max_attempts=50000):
    p_values = []
    recovery = 0
    trial_count = 0

    while len(p_values) < num_trials:
        trial_count += 1
        X, true_labels = generate_3cluster_data(n=n, p=p, delta=delta, sigma=sigma)
        model = AgglomerativeClustering(X, tau=tau, n_clusters=2, linkage="complete")
        model.fit()

        winning_nodes = list(model.existing_clusters_log.keys())
        key = winning_nodes[-1]
        c1, c2 = key[0], key[1]
        c1_points = c1.points
        c2_points = c2.points

        c1_true_clusters = set(true_labels[c1_points])
        c2_true_clusters = set(true_labels[c2_points])

        if len(c1_true_clusters) == 1 and len(c2_true_clusters) == 1:
            recovery += 1
            node = c1.parent
            p_val, _, _ = model.merge_inference_F(node, grid_width=15, ncoarse=20, ngrid=1000)
            p_values.append(p_val)

        if trial_count > max_attempts:
            print(f"Warning: Too few matching merges at tau={tau}")
            break



    power = np.mean(np.array(p_values) < alpha)
    recovery_prob = recovery / num_trials
    success = len(p_values) == num_trials
    return power, recovery_prob, success

def compute_for_tau_delta(tau, delta, n, p, sigma, alpha, num_trials):
    power, recovery, success = single_delta_power(delta, n, p, sigma, tau, alpha, num_trials)
    return tau, delta, power, recovery, success
def check_power_multi_tau_delta(n, p, sigma, tau_list, delta_list, alpha=0.05,
                                 num_trials=500, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_for_tau_delta)(tau, delta, n, p, sigma, alpha, num_trials)
        for tau in tau_list
        for delta in delta_list
    )

    # Organize results into dictionary
    power_results = {tau: {} for tau in tau_list}
    recovery_results = {tau: {} for tau in tau_list}
    success_results = {tau: {} for tau in tau_list}

    for tau, delta, power, recovery, success in results:
        power_results[tau][delta] = power
        recovery_results[tau][delta] = recovery
        success_results[tau][delta] = success

    return power_results, recovery_results, success_results


def single_tau_power_random_pair(tau, n, p, sigma, delta, alpha, num_trials=500, max_attempts=50000):
    p_values = []
    recovery = 0
    trial_count = 0

    for _ in range(num_trials):
        # while len(p_values) < num_trials:
        trial_count += 1
        X, true_labels = generate_3cluster_data(n=n, p=p, delta=delta, sigma=sigma)
        model = AgglomerativeClustering(X, tau=tau, n_clusters=3, linkage="complete")
        model.fit()

        idx1, idx2 = np.random.choice(np.arange(3), size=2, replace=False)
        c1 = model.K_clusters[idx1]
        c2 = model.K_clusters[idx2]
        c1_points = c1.points
        c2_points = c2.points

        c1_true_clusters = set(true_labels[c1_points])
        c2_true_clusters = set(true_labels[c2_points])

        if len(c1_true_clusters) == 1 and len(c2_true_clusters) == 1:
            recovery += 1
            p_val, _, _ = model.merge_inference_F_random_pair(c1,c2, grid_width=15, ncoarse=20, ngrid=1000)
            p_values.append(p_val)
        '''
        if trial_count > max_attempts:
            print(f"Warning: Too few matching merges at tau={tau}")
            break

        '''

    power = np.mean(np.array(p_values) < alpha)
    recovery_prob = recovery / num_trials
    success = len(p_values) == num_trials
    return tau, power, recovery_prob, success


def check_power_multi_tau_parallel_random_pair(n, p, sigma, tau_list, delta=10.0, alpha=0.05,
                                   num_trials=500, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs)(
        delayed(single_tau_power_random_pair)(tau, n, p, sigma, delta, alpha, num_trials)
        for tau in tau_list
    )

    power_results_sel = {tau: power for tau, power, _, _ in results}
    recovery_results = {tau: rec for tau, _, rec, _ in results}
    full = [success for _, _, _, success in results]

    tau_vals = np.array(tau_list)
    power_vals = [power_results_sel[tau] for tau in tau_vals]
    recovery_vals = [recovery_results[tau] for tau in tau_vals]

    fig, ax1 = plt.subplots(figsize=(8, 6))

    color_power = 'tab:blue'
    ax1.set_xlabel("Tau (Randomization Level)")
    ax1.set_ylabel("Conditional Power", color=color_power)
    ax1.tick_params(axis='y', labelcolor=color_power)
    ax1.set_ylim(0, 1)

    if 0 in tau_vals:
        naive_idx = np.where(tau_vals == 0)[0][0]
        ax1.scatter(tau_vals[naive_idx], power_vals[naive_idx], color='orange', marker='s', s=100,
                    label="Naive Power (τ=0)", zorder=5)

        tau_random = tau_vals[tau_vals != 0]
        power_random = [power_results_sel[t] for t in tau_random]
        ax1.plot(tau_random, power_random, marker='o', color=color_power, label="Randomized Power (τ>0)")
    else:
        ax1.plot(tau_vals, power_vals, marker='o', color=color_power, label="Conditional Power")

    ax2 = ax1.twinx()
    color_recovery = 'tab:red'
    ax2.set_ylabel("Recovery Probability", color=color_recovery)
    ax2.tick_params(axis='y', labelcolor=color_recovery)
    ax2.set_ylim(0, 1)

    if 0 in tau_vals:
        ax2.scatter(tau_vals[naive_idx], recovery_vals[naive_idx], color='darkorange', marker='D', s=100,
                    label="Naive Recovery (τ=0)", zorder=5)
        recovery_random = [recovery_results[t] for t in tau_random]
        ax2.plot(tau_random, recovery_random, marker='s', linestyle='--', color=color_recovery,
                 label="Randomized Recovery (τ>0)")
    else:
        ax2.plot(tau_vals, recovery_vals, marker='s', linestyle='--', color=color_recovery,
                 label="Recovery Probability")

    plt.title("Conditional Power and Recovery Probability vs. Tau")
    fig.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.5)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    plt.legend(h1 + h2, l1 + l2, loc="upper right")
    plt.show()

    return power_results_sel, recovery_results, full


def single_delta_power_random_pair(delta, n, p, sigma, tau, alpha, num_trials=500, max_attempts=50000):
    p_values = []
    recovery = 0
    trial_count = 0

    for _ in range(num_trials):
        trial_count += 1
        X, true_labels = generate_data_barbers(10,delta,sigma)
        model = AgglomerativeClustering(X, tau=tau, n_clusters=3, linkage="complete")
        model.fit()

        #idx1, idx2 = np.random.choice(np.arange(3), size=2, replace=False)
        c1 = model.K_clusters[0]
        c2 = model.K_clusters[1]
        c1_points = c1.points
        c2_points = c2.points

        #c1_true_clusters = set(true_labels[c1_points])
        #c2_true_clusters = set(true_labels[c2_points])
        idx = np.concatenate([c1_points, c2_points])
        unique_labels = np.unique(true_labels[idx])
        non_alternative = len(unique_labels)== 1

        if not non_alternative:
            recovery += 1
            p_val, _, _ = model.merge_inference_F_random_pair(c1,c2, grid_width=20, ncoarse=20, ngrid=1000)
            p_values.append(p_val)

    power = np.mean(np.array(p_values) < alpha)
    recovery_prob = recovery / num_trials
    success = len(p_values) == num_trials
    return power, recovery_prob, success

def compute_for_tau_delta_random_pair(tau, delta, n, p, sigma, alpha, num_trials):
    #power, recovery, success = single_delta_power_random_pair(delta, n, p, sigma, tau, alpha, num_trials)
    power, recovery, effect_size = single_delta_power_random_pair(delta, n, p, sigma, tau, alpha, num_trials)
    return tau, delta, power, recovery, effect_size
def check_power_multi_tau_delta_random_pair(n, p, sigma, tau_list, delta, alpha=0.05,
                                 num_trials=500, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_for_tau_delta_random_pair)(tau, delta, n, p, sigma, alpha, num_trials)
        for tau in tau_list
    )

    power_results = {tau: power for tau,_, power, _, _ in results}
    recovery_results = {tau: rec for tau, _, _,rec, _ in results}
    effect_size_results = {tau: es for tau, _, _, _, es in results}

    return power_results, recovery_results, effect_size_results

def single_power_es_random_pair(delta, n, p, sigma, tau, alpha, num_trials=500):
    p_values = []
    effect_sizes = []
    recovery = 0
    trial_count = 0

    for _ in range(num_trials):
        trial_count += 1
        X, true_labels, true_means = generate_data_barbers(10,delta,sigma, true_mean=True)
        model = AgglomerativeClustering(X, tau=tau, n_clusters=3, linkage="complete")
        model.fit()

        #idx1, idx2 = np.random.choice(np.arange(3), size=2, replace=False)
        c1 = model.K_clusters[0]
        c2 = model.K_clusters[1]
        c1_points = c1.points
        c2_points = c2.points
        c12_points = np.concatenate([c1_points, c2_points])

        sigma2_all = np.sum(np.linalg.norm(X - np.mean(X, axis=0), axis=1) ** 2) / ((n - 1) * 2)
        mean_k1 = np.mean(true_means[c1_points, :], axis=0)
        mean_k2 = np.mean(true_means[c2_points, :], axis=0)
        mean_k12 = np.mean(true_means[c12_points, :], axis=0)
        #sigma2_cluster =np.sum(np.linalg.norm(X[c12_points,:] - mean_k12) ** 2) / ((len(c12_points) - 1))
        #sigma2_cluster = (np.sum(np.linalg.norm(X[c1_points,:] - mean_k1, axis=1) ** 2) + np.sum(np.linalg.norm(X[c2_points,:] - mean_k2, axis=1) ** 2))/ ((len(c1_points)+ len(c2_points) - 2))
        effect_size = np.linalg.norm(mean_k1 - mean_k2) / np.sqrt(sigma2_all)
        effect_sizes.append(effect_size)

        idx = np.concatenate([c1_points, c2_points])
        unique_labels = np.unique(true_labels[idx])
        non_alternative = len(unique_labels)== 1
        #if tau < 0.05:
        #    grid_width = 40
        p_val, _ = model.merge_inference_F_random_pair_grid(c1, c2, grid_width= 70, ncoarse=20, ngrid=1000)
        p_values.append(p_val)
        if not non_alternative:
            recovery += 1


    #power = np.mean(np.array(p_values) < alpha)
    reject = (np.array(p_values) < alpha).astype(int).tolist()
    recovery_prob = recovery / num_trials
    #success = len(p_values) == num_trials

    return reject, recovery_prob, effect_sizes

def compute_es_power_random_pair_delta_tau(delta,tau,n,p,sigma, alpha, num_trials=500):
    reject, recovery_prob, effect_sizes = single_power_es_random_pair(delta, n, p, sigma, tau, alpha, num_trials)
    return delta, tau, reject, effect_sizes,recovery_prob

def check_power_es_multi_tau_delta_random_pair(n, p, sigma, tau_list, deltas, alpha=0.05,
                                 num_trials=500, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_es_power_random_pair_delta_tau)(delta,tau, n, p, sigma, alpha, num_trials)
        for tau in tau_list
        for delta in deltas
    )
    all_dfs = []
    for rows in results:
        delta = rows[0]
        tau = rows[1]
        reject = rows[2]
        effect_size = rows[3]
        recovery_prob = rows[4]
        df = pd.DataFrame({
            "tau": [tau] * len(effect_size),
            "delta": [delta] * len(effect_size),
            "effect_size": effect_size,
            "reject": reject,
            "method": ["Randomized"] * len(effect_size)
            })
        all_dfs.append(df)
    final_df = pd.concat(all_dfs, ignore_index=True)

    return(final_df)

def single_es_random_pair(delta, n, p, sigma, tau, alpha, num_trials=500):
    p_values = []
    effect_sizes = []
    recovery = 0

    for _ in range(num_trials):
        X, true_labels, true_means = generate_data_barbers(10,delta,sigma, true_mean=True)
        model = AgglomerativeClustering(X, tau=tau, n_clusters=3, linkage="complete")
        model.fit()

        pairs = combinations(model.K_clusters, 2)

        for c1, c2 in pairs:
            c1_points = c1.points
            c2_points = c2.points
            c12_points = np.concatenate([c1_points, c2_points])

            p_val, _, _ = model.merge_inference_F_random_pair(c1,c2,grid_width= 50, ncoarse=20, ngrid=1000)
            p_values.append(p_val)


            #sigma2_all = np.sum(np.linalg.norm(X - np.mean(X, axis=0), axis=1) ** 2) / ((n - 1) * 2)
            mean_k1 = np.mean(true_means[c1_points, :], axis=0)
            mean_k2 = np.mean(true_means[c2_points, :], axis=0)
            mean_k12 = np.mean(true_means[c12_points, :], axis=0)
            sigma2_cluster =np.sum(np.linalg.norm(X[c12_points,:] - mean_k12) ** 2) / ((len(c12_points) - 1))
            #sigma2_cluster = (np.sum(np.linalg.norm(X[c1_points,:] - mean_k1, axis=1) ** 2) + np.sum(np.linalg.norm(X[c2_points,:] - mean_k2, axis=1) ** 2))/ ((len(c1_points)+ len(c2_points) - 2))
            effect_size = np.linalg.norm(mean_k1 - mean_k2) / np.sqrt(sigma2_cluster)
            effect_sizes.append(effect_size)


    #power = np.mean(np.array(p_values) < alpha)
    reject = (np.array(p_values) < alpha).astype(int).tolist()
    #recovery_prob = recovery / num_trials
    #success = len(p_values) == num_trials

    return reject, effect_sizes

def compute_es_random_pair_delta_tau(delta,tau,n,p,sigma, alpha, num_trials=500):
    reject, effect_sizes = single_es_random_pair(delta, n, p, sigma, tau, alpha, num_trials)
    return delta, tau, reject, effect_sizes

def check_es_multi_tau_delta_random_pair(n, p, sigma, tau_list, deltas, alpha=0.05,
                                 num_trials=500, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_es_random_pair_delta_tau)(delta,tau, n, p, sigma, alpha, num_trials)
        for tau in tau_list
        for delta in deltas
    )
    all_dfs = []
    for rows in results:
        delta = rows[0]
        tau = rows[1]
        reject = rows[2]
        effect_size = rows[3]
        df = pd.DataFrame({
            "tau": [tau] * len(effect_size),
            "delta": [delta] * len(effect_size),
            "effect_size": effect_size,
            "reject": reject
            })
        all_dfs.append(df)
    final_df = pd.concat(all_dfs, ignore_index=True)

    return(final_df)

def single_delta_reject_prop_random_pair(delta, n, p, sigma, tau, alpha, num_trials=500, max_attempts=50000):
    #p_values = []
    proportion = []
    recovery = 0
    trial_count = 0

    for _ in range(num_trials):
        trial_count += 1
        X, true_labels = generate_data_barbers(10,delta,sigma)
        model = AgglomerativeClustering(X, tau=tau, n_clusters=3, linkage="complete")
        model.fit()

        #idx1, idx2 = np.random.choice(np.arange(3), size=2, replace=False)
        pairs = combinations(model.K_clusters,2)
        #check whether under alternative
        non_alternative_pairs = []
        for c1, c2 in pairs:
            c1_points = c1.points
            c2_points = c2.points

            #c1_true_clusters = set(true_labels[c1_points])
            #c2_true_clusters = set(true_labels[c2_points])
            idx = np.concatenate([c1_points, c2_points])
            unique_labels = np.unique(true_labels[idx])
            non_alternative = len(unique_labels)== 1 #all points belong to one true cluster
            non_alternative_pairs.append(non_alternative)

        if sum(non_alternative_pairs) == 0: #all pairs under alternative
            recovery += 1
            pvals = []
            pairs = combinations(model.K_clusters, 2)
            for c1, c2 in pairs:
                p_val, _, _ = model.merge_inference_F_random_pair(c1,c2, grid_width=70, ncoarse=20, ngrid=1000)
                pvals.append(p_val)
            proportion.append(np.mean(np.array(pvals) < alpha))

    #power = np.mean(np.array(p_values) < alpha)
    mean_prop = np.mean(np.array(proportion))
    recovery_prob = recovery / num_trials
    #success = len(p_values) == num_trials
    success = 0
    return mean_prop, recovery_prob, success

def compute_reject_prop_for_tau_delta_random_pair(tau, delta, n, p, sigma, alpha, num_trials):
    mean_prop, recovery, success = single_delta_reject_prop_random_pair(delta, n, p, sigma, tau, alpha, num_trials)
    return tau, delta, mean_prop, recovery, success
def check_reject_prop_multi_tau_delta_random_pair(n, p, sigma, tau_list, delta_list, alpha=0.05,
                                 num_trials=500, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_reject_prop_for_tau_delta_random_pair)(tau, delta, n, p, sigma, alpha, num_trials)
        for tau in tau_list
        for delta in delta_list
    )

    # Organize results into dictionary
    prop_results = {tau: {} for tau in tau_list}
    recovery_results = {tau: {} for tau in tau_list}
    success_results = {tau: {} for tau in tau_list}

    for tau, delta, power, recovery, success in results:
        prop_results[tau][delta] = power
        recovery_results[tau][delta] = recovery
        success_results[tau][delta] = success

    return prop_results, recovery_results, success_results

