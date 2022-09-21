import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import pandas as pd
import tensorflow as tf
from numpy.random import seed
from scMGCA.preprocess import *
from scMGCA.utils import *
import argparse
seed(1)
tf.random.set_seed(1)
import math
# Remove warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from scMGCA.scmgca import SCMGCAL
from scMGCA.losses import *
from scMGCA.graph_function import *
from spektral.layers import GraphConv
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--highly_genes", default = 1000, type=int)
    parser.add_argument("--pretrain_epochs", default = 200, type=int)
    parser.add_argument("--maxiter", default = 100, type=int)
    parser.add_argument("--batch", default = 2000, type=int)
    args = parser.parse_args()


    input_file = "./1M_brain_cells_10X.sparse.h5ad"
    adata = sc.read(input_file)
    sc.pp.filter_genes(adata, min_counts=20)
    sc.pp.filter_cells(adata, min_counts=200)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = args.highly_genes, subset=True)
    print(adata.shape)
    count=adata.X.todense()

    # Build graph
    adj, adj_n = get_adj_batch(count, args.batch)
    n_sample=count.shape[0]
    in_dim=count.shape[1]
    
    # Pre-training
    model = SCMGCAL(n_sample, in_dim, args.batch, hidden_dim=32, latent_dim=20,dec_dim=[32])
    model.pre_train(count, adj, adj_n, epochs=args.pretrain_epochs)
    latent_pre = model.embeddingBatch(count, adj_n)

    adata_latent = sc.AnnData(latent_pre)
    sc.pp.neighbors(adata_latent, n_neighbors = 15, use_rep="X")
    adata_latent = sc.tl.leiden(adata_latent, resolution = 0.2, copy = True)
    Y_pred_init = adata_latent.obs['leiden']
    labels = np.asarray(Y_pred_init, dtype=int)
    centers = computeCentroids(latent_pre, labels)
    
    # Clustering training
    Cluster_predicted=model.train(count, adj, adj_n, epochs=args.maxiter, centers=centers)
    
    latent = Cluster_predicted.latent
    prelabel = Cluster_predicted.y_pred
    

