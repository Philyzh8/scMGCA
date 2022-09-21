import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import pandas as pd
import tensorflow as tf
from numpy.random import seed
from scMGCA.preprocess import *
from scMGCA.utils import *
import argparse
from sklearn import metrics
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataname", default = "pancreas", type = str)
    parser.add_argument("--highly_genes", default = 2000, type=int)
    parser.add_argument("--pretrain_epochs", default = 1000, type=int)
    parser.add_argument("--maxiter", default = 300, type=int)
    args = parser.parse_args()

    adata = read_pancreas("./pancreas", cache=True)
    print("Cell number:", adata.X.shape[0])
    print("Gene number",adata.X.shape[1])   
    y = np.array(adata.obs['celltype'].values, dtype=str)
    cluster_number = len(np.unique(y))
    print("Cluster number:", cluster_number)
    count = normalize_batch(adata, batch_key = 'tech', n_high_var = args.highly_genes)  
    print(count.shape)

    # Build graph
    adj = get_adj(count)
    adj_n = GraphConv.preprocess(adj)
       
    # Pre-training
    model = SCMGCA(count, adj=adj, adj_n=adj_n, latent_dim=20, dec_dim=[128])
    model.pre_train(epochs=args.pretrain_epochs, W_a=0.6, lr=5e-4)
    latent_pre = model.embedding(count, adj_n)

    adata_latent = sc.AnnData(latent_pre)
    sc.pp.neighbors(adata_latent, n_neighbors = 15, use_rep="X")
    resolution = find_resolution(adata_latent, 8, 0)
    adata_latent = sc.tl.leiden(adata_latent, resolution = resolution, random_state = 0, copy = True)
    Y_pred_init = np.asarray(adata_latent.obs['leiden'], dtype=int)
    features = pd.DataFrame(adata_latent.X, index = np.arange(0, adata_latent.shape[0]))
    Group = pd.Series(Y_pred_init, index = np.arange(0, adata_latent.shape[0]), name="Group")
    Mergefeature = pd.concat([features, Group],axis=1)
    centers = np.asarray(Mergefeature.groupby("Group").mean())

    # Clustering training
    Cluster_predicted=model.train(epochs=args.maxiter, W_a=0.6, centers=centers)
    
    # latent representation
    latent = Cluster_predicted.latent

    y = list(map(str, y))
    Cluster_predicted.y_pred = np.array(Cluster_predicted.y_pred)
    nmi = metrics.normalized_mutual_info_score(y, Cluster_predicted.y_pred)
    ari = metrics.adjusted_rand_score(y, Cluster_predicted.y_pred)
    print('NMI= %.4f, ARI= %.4f' % (nmi, ari))

