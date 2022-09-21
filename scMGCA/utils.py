import numpy as np
import scanpy as sc
import scipy.sparse
from sklearn.decomposition import PCA


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# Compute cluster centroids, which is the mean of all points in one cluster.
def computeCentroids(data, labels):
    n_clusters = len(np.unique(labels))
    return np.array([data[labels == i].mean(0) for i in range(n_clusters)])

def init_center(args, Y, adj_n, cluster_number):
    # Cluster center initialization
    from sklearn.cluster import KMeans, SpectralClustering
    if args.dataname in ["camp2","Quake_Smart-seq2_Lung","Muraro", "Adam", "Quake_10x_Limb_Muscle", "Quake_Smart-seq2_Heart", 
    "Young", "Plasschaert", "Quake_10x_Spleen", "Chen", "Tosches turtle"]:
        labels = SpectralClustering(n_clusters=cluster_number,affinity="precomputed", assign_labels="discretize", n_init=20).fit_predict(adj_n)
        centers = computeCentroids(Y, labels)
    else:
        cluster_model = KMeans(n_clusters=cluster_number, n_init=20)
        labels = cluster_model.fit_predict(Y)
        centers = cluster_model.cluster_centers_
    return centers

def find_resolution(adata_, n_clusters, random = 0): 
    adata = adata_.copy()
    obtained_clusters = -1
    iteration = 0
    resolutions = [0., 1000.]
    
    while obtained_clusters != n_clusters and iteration < 50:
        current_res = sum(resolutions)/2
        sc.tl.leiden(adata, resolution = current_res, random_state = random)
        labels = adata.obs['leiden']
        obtained_clusters = len(np.unique(labels))
        
        if obtained_clusters < n_clusters:
            resolutions[0] = current_res
        else:
            resolutions[1] = current_res
        
        iteration = iteration + 1
        
    return current_res

def densify(arr):
    if scipy.sparse.issparse(arr):
        return arr.toarray()
    return arr

def empty_safe(fn, dtype):
    def _fn(x):
        if x.size:
            return fn(x)
        return x.astype(dtype)
    return _fn

def dopca(X, dim=10):
    pcaten = PCA(n_components=dim)
    X_10 = pcaten.fit_transform(X)
    return X_10

decode = empty_safe(np.vectorize(lambda _x: _x.decode("utf-8")), str)
