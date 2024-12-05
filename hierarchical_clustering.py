import random
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import silhouette_score


#TODO: Store all distances between clusters (now only stored the winning pair distance)
# (Done but possibly have more efficient way)
#TODO: Add randomization terms and store them
# (Saved in nested dictionary)
# For both tasks, not sure if it'd be too difficult to retrieve


#TODO: check the scale of randomization term (when =1, need tau = 100 to be mixed)
class ClusterNode:
    def __init__(self, points=None, left=None, right=None, distance= 0, depth=0):
        self.points = points  # Points contained in this cluster
        self.left = left  # Left child node (merged cluster)
        self.right = right  # Right child node (merged cluster)
        self.distance = distance  # Distance between merged clusters
        self.depth = depth  # Depth of this node in the hierarchy
    def __repr__(self):
        return f"ClusterNode(points={self.points})"


class AgglomerativeClustering:
    def __init__(self, X, n_clusters=2, tau = 1, affinity='euclidean', linkage='ward'):
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
        self.distance_log = {} # Dictionary saving all distances

        self.randomization_log = {} # Dictionary saving all randomization terms

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

    def _compute_distance_matrix(self):
        """Compute the initial distance matrix for all points."""
        from scipy.spatial.distance import pdist, squareform
        distance_matrix = squareform(pdist(self.X, metric=self.affinity))
        for i in range(len(self.X)):
            for j in range(i + 1, len(self.X)):  # Only upper triangular part
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
                if (cluster1, cluster2) not in self.randomization_log:
                    self.randomization_log[(cluster1, cluster2)] = {}
                self.randomization_log[(cluster1, cluster2)][self.step] = random_term*np.sqrt(self.n)
                if randomized_distance < min_distance:
                    min_distance = randomized_distance
                    closest_clusters = (i, j)
        return closest_clusters

    def _merge_clusters(self, i, j, distance_matrix):
        """Merge two clusters and update the distance matrix."""
        # Merge clusters
        merged_points = self.cluster_nodes[i].points + self.cluster_nodes[j].points
        new_node = ClusterNode(points=merged_points, left=self.cluster_nodes[i], right=self.cluster_nodes[j],
                               distance=distance_matrix[i, j],
                               depth= max(self.cluster_nodes[i].depth, self.cluster_nodes[j].depth) + 1)

        self.cluster_nodes.append(new_node)
        # Update the distance matrix
        self.distance_matrix = self._update_distance_matrix(distance_matrix, new_node, i, j)

        # Remove the merged clusters from the list
        self.cluster_nodes.pop(max(i, j))  # Remove the higher index first
        self.cluster_nodes.pop(min(i, j))  # Then remove the lower index


    def _update_distance_matrix(self, distance_matrix, new_node, i, j):
        """
        Update the distance matrix after merging clusters.
        All other entries stay the same, only need to update the distance related to new node
        """

        new_size = distance_matrix.shape[0] + 1
        new_distance_matrix = np.zeros((new_size, new_size))

        new_distance_matrix[:new_size - 1, :new_size - 1] = distance_matrix

        # Compute new distances from the new node to all remaining clusters
        for k in range(len(self.cluster_nodes)):
            if k == i or k == j:
                continue
            # Compute distance between new_node and cluster k
            dist = self._calculate_linkage_distance(new_node, self.cluster_nodes[k])
            new_distance_matrix[new_size - 1, k] = dist
            new_distance_matrix[k, new_size - 1] = dist
            self.distance_log[(new_node,self.cluster_nodes[k])] = dist

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

    def _calculate_linkage_distance(self, new_node, cluster):
        """Calculate the distance between clusters based on the chosen linkage method."""
        if self.linkage == 'ward':
            return self._ward_distance(new_node, cluster)
        elif self.linkage == 'single':
            return self._single_linkage(new_node, cluster)
        elif self.linkage == 'complete':
            return self._complete_linkage(new_node, cluster)
        elif self.linkage == 'average':
            return self._average_linkage(new_node, cluster)
        elif self.linkage == 'weighted':
            return self._weighted_linkage(new_node, cluster)
        elif self.linkage == 'centroid':
            return self._centroid_linkage(new_node, cluster)
        elif self.linkage == 'median':
            return self._median_linkage(new_node, cluster)
        else:
            raise ValueError("Unknown linkage method: {}".format(self.linkage))

    def _ward_distance(self, new_node, cluster):
        X_new_node = self.X[new_node.points]
        X_cluster = self.X[cluster.points]
        centroid_new = np.mean(X_new_node, axis=0)
        centroid_cluster = np.mean(X_cluster, axis=0)

        # Calculate the number of points in each cluster
        size_new = len(new_node.points)
        size_cluster = len(cluster.points)

        # Calculate the squared distance between the centroids
        distance_between_centroids = np.sum((centroid_new - centroid_cluster) ** 2)

        # Calculate the Ward's distance: increase in variance after merging
        ward_distance = distance_between_centroids * (size_new * size_cluster) / (size_new + size_cluster)

        return float(ward_distance)

    def _single_linkage(self, new_node, cluster):
        # Single linkage: Minimum distance between clusters
        X_new_node = self.X[new_node.points]
        X_cluster = self.X[cluster.points]
        distances = distance.cdist(X_new_node, X_cluster, metric=self.affinity)
        return float(np.min(distances))

    def _complete_linkage(self, new_node, cluster):
        # Complete linkage: Maximum distance between clusters
        X_new_node = self.X[new_node.points]
        X_cluster = self.X[cluster.points]
        distances = distance.cdist(X_new_node, X_cluster, metric=self.affinity)
        return float(np.max(distances))

    def _average_linkage(self, new_node, cluster):
        X_new_node = self.X[new_node.points]
        X_cluster = self.X[cluster.points]
        distances = distance.cdist(X_new_node, X_cluster, metric=self.affinity)
        return float(np.mean(distances))

    def _weighted_linkage(self, new_node, cluster):
        size_new = len(new_node.points)
        size_cluster = len(cluster.points)

        # Ensure neither cluster is empty
        if size_new == 0 or size_cluster == 0:
            raise ValueError("One of the clusters is empty.")

        # Extract points from X
        X_new_node = self.X[new_node.points]
        X_cluster = self.X[cluster.points]

        # Calculate the pairwise distances
        distances = distance.cdist(X_new_node, X_cluster, metric=self.affinity)

        # Calculate the total weighted distance
        total_weighted_distance = np.sum(distances)

        # The weighted linkage distance is averaged based on the sizes of the clusters
        weighted_distance = total_weighted_distance / (size_new * size_cluster)

        return float(weighted_distance)  # Return as float

    def _centroid_linkage(self, new_node, cluster):
        X_new_node = self.X[new_node.points]
        X_cluster = self.X[cluster.points]
        centroid_new = np.mean(X_new_node, axis=0)
        centroid_cluster = np.mean(X_cluster, axis=0)
        return self._calculate_distance(centroid_new, centroid_cluster)

    def _median_linkage(self, new_node, cluster):
        X_new_node = self.X[new_node.points]
        X_cluster = self.X[cluster.points]
        median_new = np.median(X_new_node, axis=0)
        median_cluster = np.median(X_cluster, axis=0)
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