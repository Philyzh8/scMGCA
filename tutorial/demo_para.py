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

from scMGCA.scmgca_para import SCMGCA, SCMGCAL
from scMGCA.losses import *
from scMGCA.graph_function import *
from spektral.layers import GraphConv
from hyperopt import fmin, tpe, hp,space_eval,rand,Trials,partial,STATUS_OK

if __name__ == "__main__":

    para = {"neighbor":[10, 15, 20],"me":['euclidean', 'cosine', 'correlation', 'minkowski'],
         "s":[1, 2, 3],
         "hg":[300, 500, 1000, 1500, 2000],
         "W_a":[0.3, 0.6, 0.9],
         "W_x":[1, 1.5, 2],
         "W_c":[1, 1.5, 2],
         "ly":[[], [128], [256, 128], [512, 256, 128]]}

    space = {"neighbor":hp.choice("neighbor", (10, 15)),
         "me":hp.choice("me", ('euclidean', 'cosine', 'correlation', 'minkowski')),
         "s":hp.choice("s", (1, 2, 3)),
         "hg":hp.choice("hg", (300, 500, 1000, 1500, 2000)),
         "W_a":hp.choice("W_a", (0.3, 0.6, 0.9)),
         "W_x":hp.choice("W_x", (1, 1.5, 2)),
         "W_c":hp.choice("W_c", (1, 1.5, 2)),
         "ly":hp.choice("ly", ([], [128], [256, 128], [512, 256, 128])),
        }
    
    def hyperpara(argsDict):
        parser = argparse.ArgumentParser(description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("--dataname", default = "Pollen", type = str)
        parser.add_argument("--highly_genes", default = 500, type=int)
        parser.add_argument("--pretrain_epochs", default = 1000, type=int)
        parser.add_argument("--maxiter", default = 300, type=int)
        parser.add_argument("--gpu_option", default = "0")
        parser.add_argument("--k", default = 15, type=int)
        parser.add_argument("--s", default = 2, type=int)
        parser.add_argument("--metric", default = "euclidean", type=str)
        parser.add_argument("--W_a", default = 0.3, type=float)
        parser.add_argument("--W_x", default = 1, type=float)
        parser.add_argument("--W_c", default = 1.5, type=float)
        parser.add_argument("--layer", default = [128], type=list)
        args = parser.parse_args()
        
        args.k = argsDict["neighbor"]
        args.s = argsDict["s"]
        args.metric = argsDict["me"]
        args.highly_genes = argsDict["hg"]
        args.W_a = argsDict["W_a"]
        args.W_x = argsDict["W_x"]
        args.W_c = argsDict["W_c"]
        args.ly = argsDict["ly"]

        data = './dataset/'+args.dataname+'/data.h5'
        x, y = prepro(data)
              
        x = np.ceil(x).astype(np.int)
        cluster_number = len(np.unique(y))
        adata0 = sc.AnnData(x)
        adata0.obs['Group'] = y
        adata = normalize(adata0, copy=True, highly_genes=args.highly_genes, size_factors=True, normalize_input=True, logtrans_input=True)
        count = adata.X
        adj = get_adj(count, k=args.k, metric=args.metric, s=args.s)
        adj_n = GraphConv.preprocess(adj)
        # Pre-training
        model = SCMGCA(count, hidden_dim=args.layer, adj=adj, adj_n=adj_n)
        model.pre_train(epochs=args.pretrain_epochs, W_a=args.W_a, W_x=args.W_x)
        latent_pre = model.embedding(count, adj_n)

        centers = init_center(args, latent_pre, adj_n, cluster_number)
        
        # Clustering training
        Cluster_predicted=model.train(epochs=args.maxiter, centers=centers, W_a=args.W_a, W_x=args.W_x, W_c=args.W_c)
        
        y = list(map(int, y))
        Cluster_predicted.y_pred = np.array(Cluster_predicted.y_pred)
        nmi = metrics.normalized_mutual_info_score(y, Cluster_predicted.y_pred)
        ari = metrics.adjusted_rand_score(y, Cluster_predicted.y_pred)

        """
        NOTE:
        In fact, clustering is an unsupervised learning, so to do an algorithm based on hyperparameter search, an unsupervised objective function should be selected for optimization. 
        However, here we simply recommend a parameter to the user, so we use the NMI value as the objective function of optimization.
        """
        return nmi

    algo = partial(tpe.suggest,n_startup_jobs=1)
    best = fmin(hyperpara,space,algo=algo,max_evals=10)
    for key in para.keys():
        para[key]=para[key][best[key]]
    print(best)
    print(para)
    print(hyperpara(para))

    
  