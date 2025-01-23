import random
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import silhouette_score
from itertools import combinations


#TODO: Store all distances between clusters (now only stored the winning pair distance)
# (Done but possibly have more efficient way)
#TODO: Add randomization terms and store them
# (Saved in nested dictionary)
# For both tasks, not sure if it'd be too difficult to retrieve


#TODO: check the scale of randomization term (when =1, need tau = 100 to be mixed)
class ClusterNode:
    def __init__(self, points=None, left=None, right=None, distance=0, depth=0):
        self.points = points  # Points contained in this cluster
        self.left = left  # Left child node (merged cluster)
        self.right = right  # Right child node (merged cluster)
        self.distance = distance  # Distance between merged clusters
        self.depth = depth  # Depth of this node in the hierarchy

    def __repr__(self):
        return f"ClusterNode(points={self.points})"


class AgglomerativeClustering:
    def __init__(self, X, n_clusters=2, tau=1, affinity='euclidean', linkage='ward'):
        self.X = X
        self.n = np.shape(X)[0]
        self.tau = tau
        self.cluster_nodes = None
        self.distance_matrix = None
        self.n_clusters = n_clusters  # Number of clusters to form
        self.affinity = affinity  # Distance metric
        self.linkage = linkage  # Linkage criteria
        self.root = None  # Root of the cluster hierarchy
        self.step = 0
        # dictionary of all clusters that have ever existed to retrieve distance.
        # key: the winning clusters at the step. item: all the existing clusters at this step
        self.existing_clusters_log = {}
        self.distance_log = {}  # Dictionary saving all distances

        self.randomization_log = {}  # Dictionary saving all randomization terms

        self.labels = []

    def fit(self):
        self.n_samples = self.X.shape[0]
        self.cluster_nodes = [ClusterNode(points=[i]) for i in
                              range(self.n_samples)]  #initial step: each point as a cluster
        self.distance_matrix = self._compute_distance_matrix()  #initial

        while len(self.cluster_nodes) > self.n_clusters:
            current_clusters = self.cluster_nodes.copy()
            self.step += 1
            # Find the two closest clusters
            i, j = self._find_closest_clusters(self.distance_matrix)
            self.existing_clusters_log[(self.cluster_nodes[i], self.cluster_nodes[j])] = current_clusters.copy()
            self._merge_clusters(i, j, self.distance_matrix)

        self.root = self.cluster_nodes[0]  # Final merged cluster as root
        self.final_step = self.step

    def _compute_distance_matrix(self, data=None):
        """Compute the initial distance matrix for all points."""
        if data is None:
            data = self.X
        from scipy.spatial.distance import pdist, squareform
        distance_matrix = squareform(pdist(data, metric=self.affinity))
        for i in range(len(data)):
            for j in range(i + 1, len(data)):  # Only upper triangular part
                self.distance_log[(self.cluster_nodes[i], self.cluster_nodes[j])] = distance_matrix[i, j]
        return distance_matrix

    def _find_closest_clusters(self, distance_matrix):
        """Find the indices of the two closest clusters."""
        min_distance = np.inf
        closest_clusters = (-1, -1)

        for i in range(len(self.cluster_nodes)):
            for j in range(i + 1, len(self.cluster_nodes)):
                random_term = random.gauss(0, self.tau)
                randomized_distance = distance_matrix[i, j] + random_term
                cluster1, cluster2 = self.cluster_nodes[i], self.cluster_nodes[j]

                if self.step not in self.randomization_log:
                    self.randomization_log[self.step] = {}

                # Update the inner dictionary with (cluster1, cluster2) as the key
                if (cluster1, cluster2) not in self.randomization_log[self.step]:
                    self.randomization_log[self.step][(cluster1, cluster2)] = random_term

                #if (cluster1, cluster2) not in self.randomization_log:
                #    self.randomization_log[(cluster1, cluster2)] = {}
                #self.randomization_log[(cluster1, cluster2)][self.step] = random_term
                if randomized_distance < min_distance:
                    min_distance = randomized_distance
                    closest_clusters = (i, j)
        return closest_clusters

    def _merge_clusters(self, i, j, distance_matrix, data=None):
        """Merge two clusters and update the distance matrix."""
        if data is None:
            data = self.X

        # Merge clusters
        merged_points = self.cluster_nodes[i].points + self.cluster_nodes[j].points
        new_node = ClusterNode(points=merged_points, left=self.cluster_nodes[i], right=self.cluster_nodes[j],
                               distance=distance_matrix[i, j],
                               depth=max(self.cluster_nodes[i].depth, self.cluster_nodes[j].depth) + 1)

        self.cluster_nodes.append(new_node)
        # Update the distance matrix
        self.distance_matrix = self._update_distance_matrix(distance_matrix, new_node, i, j, data)

        # Remove the merged clusters from the list
        self.cluster_nodes.pop(max(i, j))  # Remove the higher index first
        self.cluster_nodes.pop(min(i, j))  # Then remove the lower index

    def _update_distance_matrix(self, distance_matrix, new_node, i, j, data=None):
        """
        Update the distance matrix after merging clusters.
        All other entries stay the same, only need to update the distance related to new node
        """
        if data is None:
            data = self.X

        new_size = distance_matrix.shape[0] + 1
        new_distance_matrix = np.zeros((new_size, new_size))

        new_distance_matrix[:new_size - 1, :new_size - 1] = distance_matrix

        # Compute new distances from the new node to all remaining clusters
        for k in range(len(self.cluster_nodes)):
            if k == i or k == j:
                continue
            # Compute distance between new_node and cluster k
            dist = self._calculate_linkage_distance(new_node, self.cluster_nodes[k], data)
            new_distance_matrix[new_size - 1, k] = dist
            new_distance_matrix[k, new_size - 1] = dist
            self.distance_log[(new_node, self.cluster_nodes[k])] = dist

        # Remove the old distances
        distance_matrix = np.delete(new_distance_matrix, (i, j), axis=0)
        distance_matrix = np.delete(distance_matrix, (i, j), axis=1)
        return distance_matrix

    def get_cluster_labels(self):
        """Extract cluster labels for each point."""
        labels = np.zeros(self.n_samples, dtype=int)
        for cluster_id, node in enumerate(self.cluster_nodes):
            for point in node.points:
                labels[point] = cluster_id
        self.labels = labels
        return labels

    def compute_silhouette_score(self):
        """Compute the Silhouette Score for the clustering."""
        labels = self.get_cluster_labels()
        score = silhouette_score(self.X, labels, metric=self.affinity)
        return score

    def compute_wcss(self):
        """Compute the Within-Cluster Sum of Squares (WCSS) for the clustering."""
        labels = self.get_cluster_labels()
        wcss = 0
        for cluster in set(labels):
            # Extract points belonging to the current cluster
            cluster_points = self.X[labels == cluster]
            # Calculate the centroid of the cluster
            centroid = cluster_points.mean(axis=0)
            # Calculate the sum of squared distances of points to the centroid
            wcss += ((cluster_points - centroid) ** 2).sum()
        return wcss

    def _calculate_linkage_distance(self, new_node, cluster, data=None):
        """Calculate the distance between clusters based on the chosen linkage method."""
        if data is None:
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
        else:
            raise ValueError("Unknown linkage method: {}".format(self.linkage))

    def _ward_distance(self, new_node, cluster, data=None):
        if data is None:
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
            data = self.X
        # Single linkage: Minimum distance between clusters
        data_new_node = data[new_node.points]
        data_cluster = data[cluster.points]
        distances = distance.cdist(data_new_node, data_cluster, metric=self.affinity)
        return float(np.min(distances))

    def _complete_linkage(self, new_node, cluster, data=None):
        # Complete linkage: Maximum distance between clusters
        if data is None:
            data = self.X
        data_new_node = data[new_node.points]
        data_cluster = data[cluster.points]
        distances = distance.cdist(data_new_node, data_cluster, metric=self.affinity)
        return float(np.max(distances))

    def _average_linkage(self, new_node, cluster, data=None):
        if data is None:
            data = self.X
        data_new_node = self.data[new_node.points]
        data_cluster = self.data[cluster.points]
        distances = distance.cdist(data_new_node, data_cluster, metric=self.affinity)
        return float(np.mean(distances))

    def _weighted_linkage(self, new_node, cluster, data=None):
        if data is None:
            data = self.X
        size_new = len(new_node.points)
        size_cluster = len(cluster.points)

        # Ensure neither cluster is empty
        if size_new == 0 or size_cluster == 0:
            raise ValueError("One of the clusters is empty.")

        # Extract points from X
        data_new_node = data[new_node.points]
        data_cluster = data[cluster.points]

        # Calculate the pairwise distances
        distances = distance.cdist(data_new_node, data_cluster, metric=self.affinity)

        # Calculate the total weighted distance
        total_weighted_distance = np.sum(distances)

        # The weighted linkage distance is averaged based on the sizes of the clusters
        weighted_distance = total_weighted_distance / (size_new * size_cluster)

        return float(weighted_distance)  # Return as float

    def _centroid_linkage(self, new_node, cluster, data=None):
        if data is None:
            data = self.X
        data_new_node = data[new_node.points]
        data_cluster = data[cluster.points]
        centroid_new = np.mean(data_new_node, axis=0)
        centroid_cluster = np.mean(data_cluster, axis=0)
        return self._calculate_distance(centroid_new, centroid_cluster)

    def _median_linkage(self, new_node, cluster, data=None):
        if data is None:
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
    def compute_reference_measure(self, grid, sd_rand=1):
        """
        Compute an approximate reference measure over a grid of values.
        :param grid: Array of grid points to evaluate the reference measure.
        :param sd_rand: Standard deviation of the randomization term.
        :return: Array of reference measures for each grid point.
        """
        ref_measure = np.zeros(len(grid))

        for g_idx, g in enumerate(grid):
            # Reinitialize the hierarchy to traverse
            node = self.root
            self._traverse_tree_for_reference(node, g, ref_measure, g_idx, sd_rand)

        return ref_measure

    def get_all_winning_pairs(self):
        winning_pairs = []
        dictionary = self.existing_clusters_log
        for idx, key in enumerate(dictionary.keys()):
            winning_pairs.append(key)
        return winning_pairs

    def _traverse_tree_for_reference(self, node, g, ref_measure, g_idx, sd_rand):
        """
        Recursively traverse the hierarchy, calculating contributions to the reference measure.
        """
        if node.left is None and node.right is None:
            # Leaf node, base case
            return

        # Compute contribution at the current node
        cluster_dist = node.distance
        randomization = self.randomization_log.get((node.left, node.right), {}).get(self.step, 0)
        contribution = -0.5 * (g - cluster_dist - randomization) ** 2 / (sd_rand ** 2)

        # Accumulate contribution
        ref_measure[g_idx] += contribution

        # Recurse into child nodes
        if node.left:
            self._traverse_tree_for_reference(node.left, g, ref_measure, g_idx, sd_rand)
        if node.right:
            self._traverse_tree_for_reference(node.right, g, ref_measure, g_idx, sd_rand)

    def _approx_log_reference(self, node, grid, nuisance,
                              contrast, sd=1, sd_rand=1):

        # node: a ClusterNode saving point, left, right, distance between merged, depth
        # grid: each value is a grid value of ||nu^TX||_2
        # contrast: nu^TX
        # nuisance: \pi_\nu X
        # X = nuisance + g/norm(nu) *nu * dir(contrast)

        def find_current_step(target_key):
            dictionary = self.existing_clusters_log
            for idx, key in enumerate(dictionary.keys()):
                if key == target_key:
                    return idx + 1
            return -1

        def compute_nu(node):
            # return the projection direction from the given node
            G_1 = np.array(node.left.points)
            G_2 = np.array(node.right.points)
            n_G1 = len(G_1)
            n_G2 = len(G_2)

            nu = np.zeros(self.n)

            nu[G_1] += 1 / n_G1
            nu[G_2] -= 1 / n_G2
            return nu

        def compute_dirT(w):
            #return dir(w)^T
            norm = np.linalg.norm(w)
            dir_w = (w / norm) if norm != 0 else np.zeros_like(w)
            return dir_w.T

        #get the parent clusters of the given node
        p_node_1 = node.left
        p_node_2 = node.right

        current_step = find_current_step((p_node_1, p_node_2))
        print("current step: {}".format(current_step))

        all_winning_pairs = self.get_all_winning_pairs()
        print("all winning pairs: {}".format(all_winning_pairs))

        # TODO: need to add a layer of step (from last step to the beginning)
        G_w_1 = p_node_1  #the top level winning pair
        G_w_2 = p_node_2
        s = current_step  #going from top level to the beginning
        while s > 0:
            merged_pair = (G_w_1, G_w_2)
            merged_pair_r = (G_w_2, G_w_1)
            # to get all the existing cluster at this step
            if merged_pair in self.existing_clusters_log.keys():
                clusters_s = self.existing_clusters_log[merged_pair]
            else:
                clusters_s = self.existing_clusters_log[merged_pair_r]
            pairs = combinations(clusters_s, 2)  # get all the possible pairs at the step

            for g_idx, g in enumerate(grid):  #g = ||\nu^T X||_2 ?
                # get the reconstructed X_grid from grid value
                nu = compute_nu(node).reshape(-1, 1)
                X_grid = nuisance + g / np.linalg.norm(nu) * nu @ compute_dirT(contrast).reshape(1, -1)
                rand_dict = self.randomization_log[current_step]

                if merged_pair in self.distance_log.keys():
                    D_opt_obs = self.distance_log[merged_pair]
                else:
                    D_opt_obs = self.distance_log[merged_pair_r]  #D(\hat{G}_1, \hat{G}_2; X)

                if merged_pair in rand_dict.keys():
                    randomization_opt = rand_dict[merged_pair]
                else:
                    randomization_opt = rand_dict[merged_pair_r]
                D_opt_grid = self._calculate_linkage_distance(G_w_1, G_w_2, X_grid)  #D(\hat{G}_1, \hat{G}_2; X_grid)

                implied_mean = []
                observed_opt = []

                for cluster1, cluster2 in pairs:
                    #print(f"Processing pair: {cluster1}, {cluster2}")

                    if not ((cluster1 == p_node_1 and cluster2 == p_node_2) or (
                            cluster2 == p_node_1 and cluster1 == p_node_2)):
                        pair = (cluster1, cluster2)
                        if pair in self.distance_log.keys():
                            D_obs = self.distance_log[pair]
                        else:
                            D_obs = self.distance_log[(cluster2, cluster1)]

                        if pair in rand_dict.keys():
                            randomization_obs = rand_dict[pair]
                        else:
                            randomization_obs = rand_dict[(cluster2, cluster1)]
                        D_grid = self._calculate_linkage_distance(cluster1, cluster2, X_grid)

                        observed_opt_s_i = D_opt_obs - D_obs + randomization_opt - randomization_obs
                        observed_opt.append(observed_opt_s_i)

                        implied_mean_s_i = D_opt_grid - D_grid
                        implied_mean.append(implied_mean_s_i)

                #print("implied_mean", implied_mean)
                break
                #print("observed_opt", observed_opt)
                # TODO: write the solve_barrier function for this alg

            winning_pair_s = all_winning_pairs[s - 2] #get the winning pair of previous level
            print(winning_pair_s)
            s = s-1

        # not sure why here the arrays are not appended
