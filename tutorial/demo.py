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

from scMGCA.scmgca import SCMGCA, SCMGCAL
from scMGCA.losses import *
from scMGCA.graph_function import *
from spektral.layers import GraphConv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataname", default = "Quake_10x_Bladder", type = str)
    parser.add_argument("--highly_genes", default = 500, type=int)
    parser.add_argument("--pretrain_epochs", default = 1000, type=int)
    parser.add_argument("--maxiter", default = 300, type=int)
    parser.add_argument("--gpu_option", default = "0")
    parser.add_argument("--batch", default = 2000, type=int)
    args = parser.parse_args()
    print(args.dataname)

    data = './dataset/'+args.dataname+'/data.h5'
    x, y = prepro(data)
    
    print("Cell number:", x.shape[0])
    print("Gene number",x.shape[1])        
    x = np.ceil(x).astype(np.int)
    cluster_number = len(np.unique(y))
    print("Cluster number:", cluster_number)
    adata = sc.AnnData(x)
    adata.obs['Group'] = y
    adata = normalize(adata, copy=True, highly_genes=args.highly_genes, size_factors=True, normalize_input=True, logtrans_input=True)
    count = adata.X
    
    # Build model
    adj = get_adj(count)
    adj_n = GraphConv.preprocess(adj)
    
    # Pre-training
    model = SCMGCA(count, adj=adj, adj_n=adj_n)
    model.pre_train(epochs=args.pretrain_epochs)
    latent_pre = model.embedding(count, adj_n)

    centers = init_center(args, latent_pre, adj_n, cluster_number)
    
    # Clustering training
    Cluster_predicted = model.train(epochs=args.maxiter, centers=centers)
    
    y = list(map(int, y))
    Cluster_predicted.y_pred = np.array(Cluster_predicted.y_pred)
    nmi = metrics.normalized_mutual_info_score(y, Cluster_predicted.y_pred)
    ari = metrics.adjusted_rand_score(y, Cluster_predicted.y_pred)
    print('NMI= %.4f, ARI= %.4f' % (nmi, ari))

