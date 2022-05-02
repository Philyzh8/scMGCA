import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.neighbors import kneighbors_graph
from scMGCA.utils import dopca
import scanpy as sc
from anndata import AnnData

def get_adj(count, k=15, pca=50, mode="connectivity"):
    if pca:
        countp = dopca(count, dim=pca)
    else:
        countp = count
    A = kneighbors_graph(countp, k, mode=mode, metric="euclidean", include_self=True)
    adj = A.toarray()

    node = adj.shape[0]
    A = random_surf(adj, 2, 0.98)
    ppmi = PPMI_matrix(A)
    for i in range(node):
        ppmi[i] = ppmi[i]/(np.max(ppmi[i]))
    adj = ppmi
    return adj


def degree_power(A, k):
    degrees = np.power(np.array(A.sum(1)), k).flatten()
    degrees[np.isinf(degrees)] = 0.
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D


def norm_adj(A):
    normalized_D = degree_power(A, -0.5)
    output = normalized_D.dot(A).dot(normalized_D)
    return output


def scale_sim_matrix(mat):
    # Row-wise sacling of matrix
    mat = mat - np.diag(np.diag(mat))  # Make diag elements zero
    D_inv = np.diag(np.reciprocal(np.sum(mat, axis=0)))
    mat = np.dot(D_inv, mat)

    return mat


def random_surf(cosine_sim_matrix, num_hops, alpha):
    num_nodes = len(cosine_sim_matrix)
    adj_matrix = cosine_sim_matrix
    P0 = np.eye(num_nodes, dtype='float32')
    P = np.eye(num_nodes, dtype='float32')
    A = np.zeros((num_nodes, num_nodes), dtype='float32')

    for i in range(num_hops):
        P = (alpha * np.dot(P, adj_matrix)) + ((1 - alpha) * P0)
        A = A + P

    return A


# PPMI Matrix
def PPMI_matrix(A):
    num_nodes = len(A)
    row_sum = np.sum(A, axis=1).reshape(num_nodes, 1)
    col_sum = np.sum(A, axis=0).reshape(1, num_nodes)
    D = np.sum(col_sum)
    PPMI = np.log(np.divide(np.multiply(D, A), np.dot(row_sum, col_sum)))
    PPMI[np.isinf(PPMI)] = 0.0
    PPMI[np.isneginf(PPMI)] = 0.0
    PPMI[PPMI < 0.0] = 0.0

    return PPMI
