if (!require("devtools", quietly = TRUE)) install.packages("devtools", repos = "http://cran.us.r-project.org")
if (!require("fastcluster", quietly = TRUE)) install.packages("fastcluster", repos = "http://cran.us.r-project.org")
if (!require("clusterpval", quietly = TRUE)) {
  devtools::install_github("mosem/clusterpval")
}

library(devtools)
library(fastcluster)
library(clusterpval)

get_cluster_obs <- function(hcl, node) {
  if (node < 0) {
    return(-node)
  } else {
    children <- hcl$merge[node, ]
    return(c(get_cluster_obs(hcl, children[1]),
             get_cluster_obs(hcl, children[2])))
  }
}

get_last_pair <- function(hcl,K){
   clust_labels <- cutree(hcl, k = K+1)
   last_merge <- hcl$merge[nrow(hcl$merge) - K  + 1, ]
   cluster1 <- get_cluster_obs(hcl, last_merge[1])
   cluster2 <- get_cluster_obs(hcl, last_merge[2])
   label1 <- clust_labels[cluster1[1]]
   label2 <- clust_labels[cluster2[1]]
   return(c(label1,label2))
}

baseline_pval <- function(X,K, linkage, method = "euclidean"){
    hcl <- hclust(dist(X, method="euclidean")^2, method=linkage) 
    pval <- test_hier_clusters_exact(X, link=linkage, K=2, k1=1, k2=2, hcl=hcl)$pval
    return(pval)
}

