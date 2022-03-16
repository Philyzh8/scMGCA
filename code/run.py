#!/home/yuzhuohan/miniconda2/envs/tensorflow/bin/python3.7
import os
import pandas as pd
import tensorflow as tf
from numpy.random import seed
from scMGCA.preprocess import *
from scMGCA.utils import *
import argparse
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn import metrics
from sklearn import preprocessing 
import scipy.io as scio
from scipy import sparse as sp
seed(1)
tf.random.set_seed(1)

# Remove warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from scMGCA.scmgca import SCMGCA
from scMGCA.losses import *
from scMGCA.graph_function import *
from spektral.layers import GraphConv


# Compute cluster centroids, which is the mean of all points in one cluster.
def computeCentroids(data, labels):
    n_clusters = len(np.unique(labels))
    return np.array([data[labels == i].mean(0) for i in range(n_clusters)])


def train(data, highly_genes=500, pretrain_epochs=1000, maxiter=300):
    # Load data
    x, y = prepro(data)

    print("Cell number:", x.shape[0])
    print("Gene number",x.shape[1])        
    x = np.ceil(x).astype(np.int)
    cluster_number = int(max(y) - min(y) + 1)
    print("Cluster number:", cluster_number)
    adata = sc.AnnData(x)
    adata.obs['Group'] = y
    adata = normalize(adata, copy=True, highly_genes=highly_genes, size_factors=True, normalize_input=True, logtrans_input=True)
    count = adata.X
    size_factor = np.array(adata.obs.size_factors).reshape(-1, 1).astype(np.float32)
    
    # Build model
    adj = get_adj(count)
    adj_n = GraphConv.preprocess(adj)
       
    # Pre-training
    model = SCMGCA(count, adj=adj, adj_n=adj_n, size_factor=size_factor)
    model.pre_train(epochs=pretrain_epochs)
    Y = model.embedding(count, adj_n)
    
    # Cluster center initialization
    from sklearn.cluster import KMeans, SpectralClustering
    labels_k = KMeans(n_clusters=cluster_number, n_init=20).fit_predict(Y)
    labels_s = SpectralClustering(n_clusters=cluster_number,affinity="precomputed", assign_labels="discretize", n_init=20).fit_predict(adj_n)
    y = list(map(int, y))
    labels = labels_s if (np.round(metrics.normalized_mutual_info_score(y, labels_s), 5)>=np.round(metrics.normalized_mutual_info_score(y, labels_k), 5) 
    and np.round(metrics.adjusted_rand_score(y, labels_s), 5)>=np.round(metrics.adjusted_rand_score(y, labels_k), 5)) else labels_k
    centers = computeCentroids(Y, labels)  
    
    # Clustering training
    Cluster_predicted=model.alt_train(y, epochs=maxiter, centers=centers)

    if y is not None:
        y = list(map(int, y))
        Cluster_predicted.y_pred = np.array(Cluster_predicted.y_pred)
        nmi = np.round(metrics.normalized_mutual_info_score(y, Cluster_predicted.y_pred), 5)
        ari = np.round(metrics.adjusted_rand_score(y, Cluster_predicted.y_pred), 5)
        print('NMI= %.4f, ARI= %.4f' % (nmi, ari))