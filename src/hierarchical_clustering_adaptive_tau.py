import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.metrics import silhouette_score
from itertools import combinations
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.special import gamma, logsumexp
from scipy.stats import f, chi
from scipy.linalg import sqrtm
from scipy.integrate import cumulative_trapezoid
import os
from hierarchical_clustering_invariant import *


# Choose tau data-adaptively
class AgglomerativeClustering_adaptivetau:
    def __init__(self, X, epsilon, tau_list, sigma=None, n_clusters=2,n_trials = 1, affinity='euclidean', linkage='single', random_state=None):
        self.X = X
        self.sigma = sigma
        self.epsilon = epsilon
        self.n = np.shape(X)[0]
        self.p = np.shape(X)[1]
        self.tau_list = tau_list
        self.n_trials = n_trials
        self.cluster_nodes = None
        self.distance_matrix = None
        self.n_clusters = n_clusters  # Number of clusters to form
        self.affinity = affinity  # Distance metric
        self.linkage = linkage  # Linkage criteria
        self.root = None  # Root of the cluster hierarchy
        self.step = 0
        self.existing_clusters_log = {}
        # dictionary of all clusters that have ever existed to retrieve distance.
        # key: the winning clusters at the step.
        # item: all the existing clusters (before merge) at this step
        self.distance_log = {}
        # Dictionary saving all distances
        # key:
        # item:
        self.labels = []

        self.tau_t_log = []
        self.linkage_matrix = []
        # (n-1) x 4 matrix to draw dendrogram
        # id1, id2, randomized distance, # of points in the new cluster
        self.cluster_id_counter = self.n  # IDs for merged clusters start after sample indices
        self.node_to_id = {}

        self.random_state = random_state
        if random_state is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(random_state)

    def choose_tau(self):
        X = self.X
        rng = self.rng
        probs, U, scores = self.compute_tau_selection_prob(X)
        tau_star = rng.choice(self.tau_list, p=probs)
        #if np.any(~np.isfinite(probs)) or probs.sum() == 0:
            # uniform fallback
        #    probs = np.ones(len(self.tau_list)) / len(self.tau_list)
        self.tau_prob = probs #the selection probability using original data
        self.tau_star = tau_star
        self.scores = scores
        return tau_star
    def compute_tau_selection_prob(self, X):
        scores = self.compute_tau_scores(X) # -WCSS/TSS
        U = self.discrete_first_derivative(self.tau_list, scores)

        U_scaled = U / self.epsilon
        U_shift = U_scaled - np.max(U_scaled)
        exps = np.exp(U_shift)
        #probs = exps / np.sum(exps)
        #exps = np.exp(U/self.epsilon)
        s = exps.sum()
        probs = exps / s
        return probs, U, scores
    def compute_tau_scores(self, data=None):
        if data is None:
            X = self.X
        else:
            X = data
        tss = np.sum((X - np.mean(X, axis=0)) ** 2)
        scores = []

        rng = np.random.default_rng(self.random_state)
        for tau in self.tau_list:
            vals = []
            for rep in range(self.n_trials):
                seed = rng.integers(1e9)
                model = AgglomerativeClustering(
                    X,
                    sigma=self.sigma,
                    n_clusters=self.n_clusters,
                    tau=tau,
                    affinity=self.affinity,
                    linkage=self.linkage,
                    random_state=seed
                )
                model.fit()
                vals.append(model.compute_wcss() / tss)
            scores.append(np.median(vals)) #\Tilde(R) = median({R}_B)
        return np.array(scores, dtype=float)

    def discrete_first_derivative(self, taus, ys):
        taus = np.asarray(taus, dtype=float)
        ys = np.asarray(ys, dtype=float)

        n = len(taus)
        U = np.zeros(n, dtype=float)

        for j in range(1, n - 1):
            U[j] = ((ys[j+1]-ys[j])/ (taus[j+1]-taus[j]) - (ys[j]-ys[j-1])/ (taus[j]-taus[j-1]))/(taus[j+1]-taus[j-1])
            #U[j] = (ys[j - 1] - 2 * ys[j] + ys[j + 1])/(taus[j+1] - taus[j-1])
            #U[j] = (ys[j+1] - ys[j]) /(taus[j+1] - taus[j])

        return U



    def fit_with_tau_star(self):
        model = AgglomerativeClustering(self.X, sigma = self.sigma, n_clusters=self.n_clusters, tau=self.tau_star, affinity=self.affinity, linkage=self.linkage, random_state= self.random_state)
        model.fit()
        self.existing_clusters_log = model.existing_clusters_log
        self.distance_log = model.distance_log
        self.labels = model.labels


    def _calculate_linkage_distance(self, new_node, cluster, data=None):
        """Calculate the distance between clusters based on the chosen linkage method."""
        if data is None:
            # data = self.Z
            data = self.X

        if self.linkage == 'ward':
            return self._ward_distance(new_node, cluster, data)
        elif self.linkage == 'single':
            return self._single_linkage(new_node, cluster, data)
        elif self.linkage == 'complete':
            return self._complete_linkage(new_node, cluster, data)
        elif self.linkage == 'average':
            return self._average_linkage(new_node, cluster, data)
        elif self.linkage == 'weighted':
            return self._weighted_linkage(new_node, cluster, data)
        elif self.linkage == 'centroid':
            return self._centroid_linkage(new_node, cluster, data)
        elif self.linkage == 'median':
            return self._median_linkage(new_node, cluster, data)
        elif self.linkage == 'minimax':
            return self._minimax_linkage(new_node, cluster, data)
        else:
            raise ValueError("Unknown linkage method: {}".format(self.linkage))

    def _ward_distance(self, new_node, cluster, data=None):
        if data is None:
            # data = self.Z
            data = self.X

        data_new_node = data[new_node.points]
        data_cluster = data[cluster.points]
        centroid_new = np.mean(data_new_node, axis=0)
        centroid_cluster = np.mean(data_cluster, axis=0)

        # Calculate the number of points in each cluster
        size_new = len(new_node.points)
        size_cluster = len(cluster.points)

        # Calculate the squared distance between the centroids
        distance_between_centroids = np.sum((centroid_new - centroid_cluster) ** 2)

        # Calculate the Ward's distance: increase in variance after merging
        ward_distance = distance_between_centroids * (size_new * size_cluster) / (size_new + size_cluster)

        return float(ward_distance)

    def _single_linkage(self, new_node, cluster, data=None):
        if data is None:
            # data = self.Z
            data = self.X
        # Single linkage: Minimum distance between clusters
        data_new_node = data[new_node.points]
        data_cluster = data[cluster.points]
        distances = distance.cdist(data_new_node, data_cluster, metric=self.affinity)
        return float(np.min(distances))

    def _complete_linkage(self, new_node, cluster, data=None):
        # Complete linkage: Maximum distance between clusters
        if data is None:
            # data = self.Z
            data = self.X
        data_new_node = data[new_node.points]
        data_cluster = data[cluster.points]
        distances = distance.cdist(data_new_node, data_cluster, metric=self.affinity)
        return float(np.max(distances))

    def _average_linkage(self, new_node, cluster, data=None):
        if data is None:
            # data = self.Z
            data = self.X
        data_new_node = data[new_node.points]
        data_cluster = data[cluster.points]
        distances = distance.cdist(data_new_node, data_cluster, metric=self.affinity)
        return float(np.mean(distances))

    def _minimax_linkage(self, new_node, cluster, data=None):
        if data is None:
            data = self.X

        all_points_idx = np.concatenate([new_node.points, cluster.points])
        data_all = data[all_points_idx]
        pairwise = distance.cdist(data_all, data_all, metric=self.affinity)
        radii = np.max(pairwise, axis=1)
        minimax_distance = np.min(radii)

        return float(minimax_distance)

    def _weighted_linkage(self, new_node, cluster, data=None):
        if data is None:
            data = self.X

        # Ensure neither cluster is empty
        size_new = len(new_node.points)
        size_cluster = len(cluster.points)
        if size_new == 0 or size_cluster == 0:
            raise ValueError("One of the clusters is empty.")

        # Compute centroids
        data_new_node = data[new_node.points]
        data_cluster = data[cluster.points]
        centroid_new = np.mean(data_new_node, axis=0)
        centroid_cluster = np.mean(data_cluster, axis=0)

        dist = self._calculate_distance(centroid_new, centroid_cluster)
        return float(dist)

    def _centroid_linkage(self, new_node, cluster, data=None):
        if data is None:
            # data = self.Z
            data = self.X
        data_new_node = data[new_node.points]
        data_cluster = data[cluster.points]
        centroid_new = np.mean(data_new_node, axis=0)
        centroid_cluster = np.mean(data_cluster, axis=0)
        return self._calculate_distance(centroid_new, centroid_cluster)

    def _median_linkage(self, new_node, cluster, data=None):
        if data is None:
            # data = self.Z
            data = self.X
        data_new_node = data[new_node.points]
        data_cluster = data[cluster.points]
        median_new = np.median(data_new_node, axis=0)
        median_cluster = np.median(data_cluster, axis=0)
        return self._calculate_distance(median_new, median_cluster)

    def _calculate_distance(self, point1, point2):
        """Calculate the distance between two points based on the chosen affinity."""
        if self.affinity == 'euclidean':
            return float(np.linalg.norm(point1 - point2))
        else:
            raise ValueError("Unknown affinity: {}".format(self.affinity))

        ### Inference part:

    def get_all_winning_pairs(self):
        winning_pairs = []
        dictionary = self.existing_clusters_log
        for idx, key in enumerate(dictionary.keys()):
            winning_pairs.append(key)
        return winning_pairs

    def compute_nu(self, node):
        # return the projection direction from the given node
        G_1 = np.array(node.left.points)
        G_2 = np.array(node.right.points)
        n_G1 = len(G_1)
        n_G2 = len(G_2)

        nu = np.zeros(self.n)

        nu[G_1] += 1 / n_G1
        nu[G_2] -= 1 / n_G2
        return nu

    def _sel_correction_F(self, node, grid, P2, R0, R1, S):

        # node: a ClusterNode saving point, left, right, distance between merged, depth
        # grid: each value is a grid value

        def find_current_step(node1, node2):
            dictionary = self.existing_clusters_log
            for idx, key in enumerate(dictionary.keys()):
                if (key == (node1, node2)) or (key == (node2, node1)):
                    return idx + 1
            return -1

        # get the parent clusters of the given node
        p_node_1 = node.left
        p_node_2 = node.right
        m = len(p_node_1.points) + len(p_node_2.points)
        nu = self.compute_nu(node)
        nu_norm = np.linalg.norm(nu)
        # print("m: ",m)
        current_step = find_current_step(p_node_1, p_node_2)
        # print("current step: {}".format(current_step))
        all_winning_pairs = self.get_all_winning_pairs()
        # print("all winning pairs: {}".format(all_winning_pairs))

        cor_prob = np.zeros_like(grid)  # for each grid value, cor_prob[g] = \sum (p(\hat{s}^{(t)}|X(g)))
        G_w_1 = p_node_1  # G^{(t)}_1 and G^{(t)}_2
        G_w_2 = p_node_2
        s = current_step  # going from top level to the beginning
        corrections = np.zeros((len(grid), s))
        tau_correction = np.zeros(len(grid))
        while s > 0:
            # print("level: ", s)
            merged_pair = (G_w_1, G_w_2)
            # print("winning pair at this step: ", merged_pair)
            merged_pair_r = (G_w_2, G_w_1)
            # to get all the existing cluster at this step
            if merged_pair in self.existing_clusters_log.keys():
                clusters_s = self.existing_clusters_log[merged_pair]
            else:
                clusters_s = self.existing_clusters_log[merged_pair_r]

            for g_idx, g in enumerate(grid):
                # get the reconstructed X_grid from grid value
                # print("grid value: ", g)
                cor_scores = []  # the vector [p_1,....,p_d], first item is always the optimal
                Ds_grid = []
                X_grid = (np.sqrt((g) / (m - 2 + (g))) * R0 + np.sqrt((m - 2) / (m - 2 + (g))) * R1) * np.sqrt(
                    S) + P2 @ self.X

                ### Selection correction for tau
                if s == 2:
                    probs,_,_ = self.compute_tau_selection_prob(X_grid)
                    idx = np.where(self.tau_list == self.tau_star)[0][0]
                    logp_tau = np.log(probs[idx])
                    tau_correction[g_idx] = logp_tau

                ## Selection correction for clustering
                D_opt_grid = self._calculate_linkage_distance(G_w_1, G_w_2, X_grid)  # D(\hat{G}_1, \hat{G}_2; X_grid)
                Ds_grid.append(D_opt_grid)

                pairs = combinations(clusters_s, 2)
                for cluster1, cluster2 in pairs:
                    if not ((G_w_1 == cluster1 and G_w_2 == cluster2) or (G_w_2 == cluster1 and G_w_1 == cluster2)):
                        D_grid = self._calculate_linkage_distance(cluster1, cluster2, X_grid)
                        Ds_grid.append(D_grid)

                tau_t_grid = self.tau_star * np.mean(Ds_grid)
                cor_scores = [np.exp(-(1 / tau_t_grid) * D_grid) for D_grid in Ds_grid]
                cor_scores = (cor_scores / np.sum(cor_scores))  # normalization
                # cor_scores[0] = exp(-1\e*d(s_hat;X(u)))/ sum_s exp(-1/e*d(s;X(u))) = P(s_hat|X(u))
                cor_prob[g_idx] = np.log(cor_scores[0])
                # cor_prob[g_idx] += np.log(cor_scores[0])
                # print("cor_prob: ", cor_prob[g_idx])

            corrections[:, s - 1] += cor_prob

            if s > 1:
                winning_pair_s = all_winning_pairs[s - 2]  # get the winning pair of previous level
                G_w_1 = winning_pair_s[0]
                G_w_2 = winning_pair_s[1]

            s -= 1
        return np.array(corrections), np.array(tau_correction)

    def merge_inference_F(self, node, ngrid=10000, ncoarse=20, grid_width=15):
        def create_indicator_diagonal_matrix(index_list, n):
            diag = np.zeros(n)
            diag[index_list] = 1
            return np.diag(diag), diag

        if self.tau_star != 0:
            nu = self.compute_nu(node).reshape(-1, 1)
            p_node_1 = node.left
            p_node_2 = node.right
            m = len(p_node_1.points) + len(p_node_2.points)
            if m == 2:
                p_value = np.nan
                observed_target = np.nan
                sel_probs = np.nan

            else:
                P0 = nu @ nu.T / np.linalg.norm(nu) ** 2
                I1, one1 = create_indicator_diagonal_matrix(p_node_1.points, self.n)
                I2, one2 = create_indicator_diagonal_matrix(p_node_2.points, self.n)
                one1 = one1.reshape(-1, 1)
                one2 = one2.reshape(-1, 1)
                P1 = (I1 - one1 @ one1.T / len(p_node_1.points)) + (I2 - one2 @ one2.T / len(p_node_2.points))
                P2 = np.eye(self.n) - P0 - P1

                S = np.linalg.norm(P0 @ self.X, 'fro') ** 2 + np.linalg.norm(P1 @ self.X, 'fro') ** 2
                R0 = (P0 @ self.X) / np.linalg.norm(P0 @ self.X, 'fro')
                R1 = (P1 @ self.X) / np.linalg.norm(P1 @ self.X, 'fro')

                stat_grid = np.linspace(0.00001, grid_width, num=ngrid)
                observed_target = (m - 2) * np.linalg.norm(P0 @ self.X, 'fro') ** 2 / (
                            np.linalg.norm(P1 @ self.X, 'fro') ** 2)
                # print(np.linalg.norm(P0 @ self.X, 'fro') ** 2)
                # print(np.linalg.norm(P1 @ self.X, 'fro') ** 2)
                # print(observed_target)
                # print("Are they close?", np.allclose(self.X, (np.sqrt(observed_target/(m-2+observed_target)) * R0 + np.sqrt((m-2)/(m-2+observed_target)) * R1) *np.sqrt(S) + P2 @ self.X))
                # projection_error = np.linalg.norm((np.eye(self.n) - np.outer(nu, nu) / np.linalg.norm(nu) ** 2) @ nu)
                # print("Projection error (should be close to 0):", projection_error)
                # print("obs:",observed_target)
                if ncoarse is not None:
                    coarse_grid = np.linspace(0.00001, grid_width, ncoarse)
                    eval_grid = coarse_grid
                else:
                    eval_grid = stat_grid

                if ncoarse is None:
                    sel_probs, sel_probs_tau = self._sel_correction_F(node, stat_grid, P2, R0, R1, S)
                    p = self.p
                    log_prior = np.zeros(ngrid)
                    for g in range(ngrid):
                        log_prior[g] = f.logpdf(x=stat_grid[g], dfn=p, dfd=(m - 2) * p)
                    log_post = log_prior + sel_probs+ sel_probs_tau
                    posterior = np.exp(log_post)

                    sum = 0
                    num = 0
                    for g in range(ngrid):
                        sum += posterior[g]
                        if stat_grid[g] >= observed_target:
                            num += posterior[g]
                    p_value = num / sum
                else:
                    sel_probs_coarse, sel_probs_tau_coarse = self._sel_correction_F(node, eval_grid, P2, R0, R1, S)
                    step = sel_probs_coarse.shape[1]

                    grid = np.linspace(0.00001, grid_width, num=ngrid)
                    sel_probs = np.zeros(ngrid)
                    log_prior = np.zeros(ngrid)
                    p = self.p

                    '''
                    for g in range(ngrid):
                        log_prior[g] = f.logpdf(x=grid[g], dfn=p, dfd=(m - 2) * p)
                    for s in range(step):
                        approx_fn = interp1d(eval_grid,
                                         sel_probs_coarse[:,s],
                                         kind='quadratic',
                                         bounds_error=False,
                                         fill_value='extrapolate')
                        #for g in range(ngrid):
                            #sel_probs[g] += approx_fn(grid[g]) #selection probability
                    '''
                    log_prior = f.logpdf(x=grid, dfn=p, dfd=(m - 2) * p)

                    interpolation = np.array([
                        interp1d(eval_grid, sel_probs_coarse[:, s],
                                 kind='quadratic',
                                 bounds_error=False,
                                 fill_value='extrapolate')(grid)
                        for s in range(step)
                    ])
                    sel_probs = interpolation.sum(axis=0)
                    f_tau = interp1d(eval_grid, sel_probs_tau_coarse,
                                     kind='quadratic',
                                     bounds_error=False,
                                     fill_value='extrapolate')
                    sel_probs_tau_fine = f_tau(grid)


                    '''
                    approx_fn = interp1d(eval_grid,
                                         sel_probs_coarse,
                                         kind='quadratic',
                                         bounds_error=False,
                                         fill_value='extrapolate')

                                        for g in range(ngrid):
                        log_prior[g] = f.logpdf(x=grid[g], dfn=p, dfd=(m - 2) * p)
                        sel_probs[g] = approx_fn(grid[g])
                    '''
                    log_post = log_prior + sel_probs + sel_probs_tau_fine
                    posterior = np.exp(log_post)

                    posterior = posterior / np.max(posterior)
                    sum = 0
                    num = 0
                    for g in range(ngrid):
                        sum += posterior[g]
                        if grid[g] >= (observed_target):
                            num += posterior[g]
                    p_value = num / sum
        else:
            nu = self.compute_nu(node).reshape(-1, 1)
            p_node_1 = node.left
            p_node_2 = node.right
            m = len(p_node_1.points) + len(p_node_2.points)
            if m == 2:
                p_value = np.nan
                observed_target = np.nan
                sel_probs = np.nan
            else:
                P0 = nu @ nu.T / np.linalg.norm(nu) ** 2
                I1, one1 = create_indicator_diagonal_matrix(p_node_1.points, self.n)
                I2, one2 = create_indicator_diagonal_matrix(p_node_2.points, self.n)
                one1 = one1.reshape(-1, 1)
                one2 = one2.reshape(-1, 1)
                P1 = (I1 - one1 @ one1.T / len(p_node_1.points)) + (I2 - one2 @ one2.T / len(p_node_2.points))

                stat_grid = np.linspace(0.00001, grid_width, num=ngrid)
                observed_target = (m - 2) * np.linalg.norm(P0 @ self.X, 'fro') ** 2 / np.linalg.norm(P1 @ self.X,
                                                                                                     'fro') ** 2

                sel_probs = 0
                p = self.p
                posterior = np.zeros(ngrid)
                for g in range(ngrid):
                    posterior[g] = f.pdf(stat_grid[g], p, (m - 2) * p)

                sum = 0
                num = 0
                for g in range(ngrid):
                    sum += posterior[g]
                    if stat_grid[g] >= observed_target:
                        num += posterior[g]
                p_value = num / sum

        return (p_value, observed_target, sel_probs)
    def merge_inference_F_grid(self, node, ngrid=10000, ncoarse=20, grid_width=15):
        def get_fine_grid(cdf, grid, qlow=0.005, qhigh=0.995, buffer=3):
            low = np.interp(qlow, cdf, grid)
            high = np.interp(qhigh, cdf, grid)
            width = high - low
            low = max(grid.min(), low - buffer * width)
            high = min(grid.max(), high + buffer * width)
            print(low, high)
            return low, high

        def get_corrected_cdf(sel_probs, dfn, dfd, grid):
            sel_log = np.asarray(sel_probs).reshape(-1)
            log_prior = f.logpdf(grid, dfn, dfd)
            log_post = log_prior + sel_log
            dx = np.gradient(grid)
            unnorm = np.exp(log_post - log_post.max())
            Z = (unnorm * dx).sum() + 1e-300
            corr_pdf = unnorm / Z
            w = corr_pdf * dx
            cdf = np.cumsum(w)
            cdf /= cdf[-1]
            return cdf

        def create_indicator_diagonal_matrix(index_list, n):
            diag = np.zeros(n)
            diag[index_list] = 1
            return np.diag(diag), diag

        if self.tau_star != 0:
            nu = self.compute_nu(node).reshape(-1, 1)
            p_node_1 = node.left
            p_node_2 = node.right
            m = len(p_node_1.points) + len(p_node_2.points)
            if m == 2:
                p_value = np.nan
                observed_target = np.nan
                sel_probs = np.nan

            else:
                P0 = nu @ nu.T / np.linalg.norm(nu) ** 2
                I1, one1 = create_indicator_diagonal_matrix(p_node_1.points, self.n)
                I2, one2 = create_indicator_diagonal_matrix(p_node_2.points, self.n)
                one1 = one1.reshape(-1, 1)
                one2 = one2.reshape(-1, 1)
                P1 = (I1 - one1 @ one1.T / len(p_node_1.points)) + (I2 - one2 @ one2.T / len(p_node_2.points))
                P2 = np.eye(self.n) - P0 - P1

                S = np.linalg.norm(P0 @ self.X, 'fro') ** 2 + np.linalg.norm(P1 @ self.X, 'fro') ** 2
                R0 = (P0 @ self.X) / np.linalg.norm(P0 @ self.X, 'fro')
                R1 = (P1 @ self.X) / np.linalg.norm(P1 @ self.X, 'fro')

                stat_grid = np.linspace(0.00001, grid_width, num=ngrid)
                observed_target = (m - 2) * np.linalg.norm(P0 @ self.X, 'fro') ** 2 / (
                        np.linalg.norm(P1 @ self.X, 'fro') ** 2)
                if ncoarse is not None:
                    coarse_grid = np.linspace(0.00001, grid_width, ncoarse)
                    eval_grid = coarse_grid
                else:
                    eval_grid = stat_grid

                if ncoarse is None:
                    sel_probs, sel_probs_tau = self._sel_correction_F(node, stat_grid, P2, R0, R1, S)
                    p = self.p
                    log_prior = np.zeros(ngrid)
                    for g in range(ngrid):
                        log_prior[g] = f.logpdf(x=stat_grid[g], dfn=p, dfd=(m - 2) * p)
                    log_post = log_prior + sel_probs + sel_probs_tau
                    posterior = np.exp(log_post)

                    sum = 0
                    num = 0
                    for g in range(ngrid):
                        sum += posterior[g]
                        if stat_grid[g] >= observed_target:
                            num += posterior[g]
                    p_value = num / sum
                else:
                    grid = np.linspace(0.00001, grid_width, num=ngrid)
                    dfn, dfd = self.p, (m - 2) * self.p
                    sel_probs_coarse, sel_probs_tau_coarse = self._sel_correction_F(node, eval_grid, P2, R0, R1, S)
                    step = sel_probs_coarse.shape[1]


                    # interpolation to get correction on fine grid
                    interpolation = np.array([
                        interp1d(eval_grid, sel_probs_coarse[:, s],
                                 kind='quadratic',
                                 bounds_error=False,
                                 fill_value='extrapolate')(grid)
                        for s in range(step)
                    ])
                    sel_probs = interpolation.sum(axis=0)

                    f_tau = interp1d(eval_grid, sel_probs_tau_coarse,
                                     kind='quadratic',
                                     bounds_error=False,
                                     fill_value='extrapolate')
                    sel_probs_tau_fine = f_tau(grid)

                    sel_probs_total = sel_probs + sel_probs_tau_fine

                    # compute corrected cdf to get shorter grid
                    corr_cdf = get_corrected_cdf(sel_probs, dfn, dfd, grid)
                    low, high = get_fine_grid(corr_cdf, grid)
                    new_coarse_grid = np.linspace(low, high, ncoarse)
                    fine_grid = np.linspace(low, high, ngrid * 2)

                    sel_probs_coarse, sel_probs_tau_coarse = self._sel_correction_F(node, new_coarse_grid, P2, R0, R1, S)
                    interpolation = np.array([
                        interp1d(new_coarse_grid, sel_probs_coarse[:, s],
                                 kind='quadratic',
                                 bounds_error=False,
                                 fill_value='extrapolate')(fine_grid)
                        for s in range(step)
                    ])
                    print(sel_probs_tau_coarse)

                    f_tau = interp1d(new_coarse_grid, sel_probs_tau_coarse,
                                     kind='quadratic',
                                     bounds_error=False,
                                     fill_value='extrapolate')
                    sel_probs_tau_fine = f_tau(fine_grid)

                    log_prior = f.logpdf(x=fine_grid, dfn=dfn, dfd=dfd)
                    sel_probs = interpolation.sum(axis=0)
                    log_post = log_prior + sel_probs + sel_probs_tau_fine
                    # posterior = np.exp(log_post)

                    # posterior = posterior / np.max(posterior)
                    log_post_shift = log_post - np.max(log_post)
                    posterior = np.exp(log_post_shift)
                    posterior_sum = posterior.sum()
                    if posterior_sum == 0 or np.isnan(posterior_sum):
                        # fallback: use uniform distribution to avoid nan
                        posterior = np.ones_like(posterior) / len(posterior)
                    else:
                        posterior = posterior / posterior_sum
                    sum = 0
                    num = 0
                    for g in range(ngrid * 2):
                        sum += posterior[g]
                        if fine_grid[g] >= (observed_target):
                            num += posterior[g]
                    p_value = num / sum
        else:
            nu = self.compute_nu(node).reshape(-1, 1)
            p_node_1 = node.left
            p_node_2 = node.right
            m = len(p_node_1.points) + len(p_node_2.points)
            if m == 2:
                p_value = np.nan
                observed_target = np.nan
                sel_probs = np.nan
            else:
                P0 = nu @ nu.T / np.linalg.norm(nu) ** 2
                I1, one1 = create_indicator_diagonal_matrix(p_node_1.points, self.n)
                I2, one2 = create_indicator_diagonal_matrix(p_node_2.points, self.n)
                one1 = one1.reshape(-1, 1)
                one2 = one2.reshape(-1, 1)
                P1 = (I1 - one1 @ one1.T / len(p_node_1.points)) + (I2 - one2 @ one2.T / len(p_node_2.points))

                stat_grid = np.linspace(0.00001, grid_width, num=ngrid)
                observed_target = (m - 2) * np.linalg.norm(P0 @ self.X, 'fro') ** 2 / np.linalg.norm(P1 @ self.X,
                                                                                                     'fro') ** 2

                sel_probs = 0
                p = self.p
                posterior = np.zeros(ngrid)
                for g in range(ngrid):
                    posterior[g] = f.pdf(stat_grid[g], p, (m - 2) * p)

                sum = 0
                num = 0
                for g in range(ngrid):
                    sum += posterior[g]
                    if stat_grid[g] >= observed_target:
                        num += posterior[g]
                p_value = num / sum

        return (p_value, observed_target, sel_probs)


    def _sel_correction_F_random_pair(self, c1, c2, grid, P2, R0, R1, S):

        # node: a ClusterNode saving point, left, right, distance between merged, depth
        # grid: each value is a grid value

        def find_current_step(node1, node2):
            dictionary = self.existing_clusters_log
            for idx, key in enumerate(dictionary.keys()):
                if (key == (node1, node2)) or (key == (node2, node1)):
                    return idx + 1
            return -1

        # get the parent clusters of the given node
        p_node_1 = c1
        p_node_2 = c2

        winning_pair = list(self.existing_clusters_log.keys())[-1]
        m = len(p_node_1.points) + len(p_node_2.points)
        nu = self.compute_nu_pair(c1, c2)
        nu_norm = np.linalg.norm(nu)
        # print("m: ",m)
        current_step = find_current_step(winning_pair[0], winning_pair[1])
        # print("current step: {}".format(current_step))
        all_winning_pairs = self.get_all_winning_pairs()
        # print("all winning pairs: {}".format(all_winning_pairs))

        cor_prob = np.zeros_like(grid)  # for each grid value, cor_prob[g] = \sum (p(\hat{s}^{(t)}|X(g)))
        G_w_1 = winning_pair[0]  # G^{(t)}_1 and G^{(t)}_2
        G_w_2 = winning_pair[1]
        s = current_step  # going from top level to the beginning
        corrections = np.zeros((len(grid), s))
        while s > 0:
            # print("level: ", s)
            merged_pair = (G_w_1, G_w_2)
            # print("winning pair at this step: ", merged_pair)
            merged_pair_r = (G_w_2, G_w_1)
            # to get all the existing cluster at this step
            if merged_pair in self.existing_clusters_log.keys():
                clusters_s = self.existing_clusters_log[merged_pair]
            else:
                clusters_s = self.existing_clusters_log[merged_pair_r]

            for g_idx, g in enumerate(grid):
                # get the reconstructed X_grid from grid value
                # print("grid value: ", g)
                cor_scores = []  # the vector [p_1,....,p_d], first item is always the optimal
                Ds_grid = []
                X_grid = (np.sqrt((g) / (m - 2 + (g))) * R0 + np.sqrt((m - 2) / (m - 2 + (g))) * R1) * np.sqrt(
                    S) + P2 @ self.X
                D_opt_grid = self._calculate_linkage_distance(G_w_1, G_w_2, X_grid)  # D(\hat{G}_1, \hat{G}_2; X_grid)
                Ds_grid.append(D_opt_grid)

                pairs = combinations(clusters_s, 2)
                for cluster1, cluster2 in pairs:
                    if not ((G_w_1 == cluster1 and G_w_2 == cluster2) or (G_w_2 == cluster1 and G_w_1 == cluster2)):
                        D_grid = self._calculate_linkage_distance(cluster1, cluster2, X_grid)
                        Ds_grid.append(D_grid)

                tau_t_grid = self.tau_star * np.mean(Ds_grid)
                cor_scores = [np.exp(-(1 / tau_t_grid) * D_grid) for D_grid in Ds_grid]
                cor_scores = (cor_scores / np.sum(cor_scores))  # normalization
                # cor_scores[0] = exp(-1\e*d(s_hat;X(u)))/ sum_s exp(-1/e*d(s;X(u))) = P(s_hat|X(u))
                cor_prob[g_idx] = np.log(cor_scores[0])
                # cor_prob[g_idx] += np.log(cor_scores[0])
                # print("cor_prob: ", cor_prob[g_idx])

            corrections[:, s - 1] += cor_prob

            if s > 1:
                winning_pair_s = all_winning_pairs[s - 2]  # get the winning pair of previous level
                G_w_1 = winning_pair_s[0]
                G_w_2 = winning_pair_s[1]

            s -= 1
        return np.array(corrections)
        # return np.array(cor_prob)

    def merge_inference_F_random_pair_grid(self, c1, c2, ngrid=10000, ncoarse=20, grid_width=15):
        def get_fine_grid(cdf, grid, qlow=0.005, qhigh=0.995, buffer=5):
            low = np.interp(qlow, cdf, grid)
            high = np.interp(qhigh, cdf, grid)
            width = high - low
            low = max(grid.min(), low - buffer * width)
            high = min(grid.max(), high + buffer * width)
            print(low, high)
            return low, high

        def get_corrected_cdf(sel_probs, dfn, dfd, grid):
            sel_log = np.asarray(sel_probs).reshape(-1)
            log_prior = f.logpdf(grid, dfn, dfd)
            log_post = log_prior + sel_log
            dx = np.gradient(grid)
            unnorm = np.exp(log_post - log_post.max())
            Z = (unnorm * dx).sum() + 1e-300
            corr_pdf = unnorm / Z
            w = corr_pdf * dx
            cdf = np.cumsum(w)
            cdf /= cdf[-1]
            return cdf

        def create_indicator_diagonal_matrix(index_list, n):
            diag = np.zeros(n)
            diag[index_list] = 1
            return np.diag(diag), diag

        low = 0
        high = grid_width
        if self.tau != 0:
            nu = self.compute_nu_pair(c1, c2).reshape(-1, 1)
            p_node_1 = c1
            p_node_2 = c2
            m = len(p_node_1.points) + len(p_node_2.points)
            if m == 2:
                p_value = np.nan
                observed_target = np.nan
                sel_probs = np.nan

            else:
                P0 = nu @ nu.T / np.linalg.norm(nu) ** 2
                I1, one1 = create_indicator_diagonal_matrix(p_node_1.points, self.n)
                I2, one2 = create_indicator_diagonal_matrix(p_node_2.points, self.n)
                one1 = one1.reshape(-1, 1)
                one2 = one2.reshape(-1, 1)
                P1 = (I1 - one1 @ one1.T / len(p_node_1.points)) + (I2 - one2 @ one2.T / len(p_node_2.points))
                P2 = np.eye(self.n) - P0 - P1

                S = np.linalg.norm(P0 @ self.X, 'fro') ** 2 + np.linalg.norm(P1 @ self.X, 'fro') ** 2
                R0 = (P0 @ self.X) / np.linalg.norm(P0 @ self.X, 'fro')
                R1 = (P1 @ self.X) / np.linalg.norm(P1 @ self.X, 'fro')

                stat_grid = np.linspace(0.00001, grid_width, num=ngrid)
                observed_target = (m - 2) * np.linalg.norm(P0 @ self.X, 'fro') ** 2 / (
                        np.linalg.norm(P1 @ self.X, 'fro') ** 2)

                if ncoarse is not None:
                    coarse_grid = np.linspace(0.00001, grid_width, ncoarse)
                    eval_grid = coarse_grid
                else:
                    eval_grid = stat_grid

                if ncoarse is None:
                    sel_probs = self._sel_correction_F_random_pair(c1, c2, stat_grid, P2, R0, R1, S)
                    p = self.p
                    log_prior = np.zeros(ngrid)
                    for g in range(ngrid):
                        log_prior[g] = f.logpdf(x=stat_grid[g], dfn=p, dfd=(m - 2) * p)
                    log_post = log_prior + sel_probs
                    posterior = np.exp(log_post)

                    sum = 0
                    num = 0
                    for g in range(ngrid):
                        sum += posterior[g]
                        if stat_grid[g] >= observed_target:
                            num += posterior[g]
                    p_value = num / sum
                else:
                    grid = np.linspace(0.00001, grid_width, num=ngrid)
                    dfn, dfd = self.p, (m - 2) * self.p
                    sel_probs_coarse = self._sel_correction_F_random_pair(c1, c2, eval_grid, P2, R0, R1, S)
                    step = sel_probs_coarse.shape[1]

                    # interpolation to get correction on fine grid
                    interpolation = np.array([
                        interp1d(eval_grid, sel_probs_coarse[:, s],
                                 kind='quadratic',
                                 bounds_error=False,
                                 fill_value='extrapolate')(grid)
                        for s in range(step)
                    ])
                    sel_probs = interpolation.sum(axis=0)

                    # compute corrected cdf to get shorter grid
                    corr_cdf = get_corrected_cdf(sel_probs, dfn, dfd, grid)
                    low, high = get_fine_grid(corr_cdf, grid)
                    new_coarse_grid = np.linspace(low, high, ncoarse)
                    fine_grid = np.linspace(low, high, ngrid * 2)

                    sel_probs_coarse = self._sel_correction_F_random_pair(c1, c2, new_coarse_grid, P2, R0, R1, S)
                    interpolation = np.array([
                        interp1d(new_coarse_grid, sel_probs_coarse[:, s],
                                 kind='quadratic',
                                 bounds_error=False,
                                 fill_value='extrapolate')(fine_grid)
                        for s in range(step)
                    ])
                    log_prior = f.logpdf(x=fine_grid, dfn=dfn, dfd=dfd)
                    sel_probs = interpolation.sum(axis=0)
                    log_post = log_prior + sel_probs
                    posterior = np.exp(log_post)

                    posterior = posterior / np.max(posterior)
                    sum = 0
                    num = 0
                    for g in range(ngrid * 2):
                        sum += posterior[g]
                        if fine_grid[g] >= (observed_target):
                            num += posterior[g]
                    p_value = num / sum
        else:
            nu = self.compute_nu_pair(c1, c2).reshape(-1, 1)
            p_node_1 = c1
            p_node_2 = c2
            m = len(p_node_1.points) + len(p_node_2.points)
            if m == 2:
                p_value = np.nan
                observed_target = np.nan
                sel_probs = np.nan
            else:
                P0 = nu @ nu.T / np.linalg.norm(nu) ** 2
                I1, one1 = create_indicator_diagonal_matrix(p_node_1.points, self.n)
                I2, one2 = create_indicator_diagonal_matrix(p_node_2.points, self.n)
                one1 = one1.reshape(-1, 1)
                one2 = one2.reshape(-1, 1)
                P1 = (I1 - one1 @ one1.T / len(p_node_1.points)) + (I2 - one2 @ one2.T / len(p_node_2.points))

                stat_grid = np.linspace(0.00001, grid_width, num=ngrid)
                observed_target = (m - 2) * np.linalg.norm(P0 @ self.X, 'fro') ** 2 / np.linalg.norm(P1 @ self.X,
                                                                                                     'fro') ** 2

                sel_probs = 0
                p = self.p
                posterior = np.zeros(ngrid)
                for g in range(ngrid):
                    posterior[g] = f.pdf(stat_grid[g], p, (m - 2) * p)

                sum = 0
                num = 0
                for g in range(ngrid):
                    sum += posterior[g]
                    if stat_grid[g] >= observed_target:
                        num += posterior[g]
                p_value = num / sum

        return (p_value, observed_target, sel_probs)