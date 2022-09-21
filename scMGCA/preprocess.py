from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import scMGCA.utils as utils
import os
import h5py
import scipy as sp
import numpy as np
import scanpy as sc
import pandas as pd
from scipy.sparse import issparse
from anndata import AnnData


def convert_string_to_encoding(string, vector_key): 
    return np.argwhere(vector_key == string)[0][0]

def convert_vector_to_encoding(vector):
    vector_key = np.unique(vector)
    vector_strings = list(vector)
    vector_num = [convert_string_to_encoding(string, vector_key) for string in vector_strings]
    return vector_num
def read_clean(data):
    assert isinstance(data, np.ndarray)
    if data.dtype.type is np.bytes_:
        data = utils.decode(data)
    if data.size == 1:
        data = data.flat[0]
    return data


def dict_from_group(group):
    assert isinstance(group, h5py.Group)
    d = utils.dotdict()
    for key in group:
        if isinstance(group[key], h5py.Group):
            value = dict_from_group(group[key])
        else:
            value = read_clean(group[key][...])
        d[key] = value
    return d


def read_data(filename, sparsify = False, skip_exprs = False):
    with h5py.File(filename, "r") as f:
        obs = pd.DataFrame(dict_from_group(f["obs"]), index = utils.decode(f["obs_names"][...]))
        var = pd.DataFrame(dict_from_group(f["var"]), index = utils.decode(f["var_names"][...]))
        uns = dict_from_group(f["uns"])
        if not skip_exprs:
            exprs_handle = f["exprs"]
            if isinstance(exprs_handle, h5py.Group):
                mat = sp.sparse.csr_matrix((exprs_handle["data"][...], exprs_handle["indices"][...],
                                               exprs_handle["indptr"][...]), shape = exprs_handle["shape"][...])
            else:
                mat = exprs_handle[...].astype(np.float32)
                if sparsify:
                    mat = sp.sparse.csr_matrix(mat)
        else:
            mat = sp.sparse.csr_matrix((obs.shape[0], var.shape[0]))
    return mat, obs, var, uns

def read_pancreas(path, cache=True):
    pathlist = os.listdir(path)
    adata = sc.read(os.path.join(path, pathlist[0]))
          
    for i in range(1,len(pathlist)):
        adata = adata.concatenate(sc.read(os.path.join(path, pathlist[i])))

    sc.pp.filter_cells(adata, min_genes = 200)
    sc.pp.filter_genes(adata, min_cells = 30)
    mito_genes = adata.var_names.str.startswith('mt-')
    adata.obs['percent_mito'] = np.sum(
        adata[:, mito_genes].X, axis=1).A1 / np.sum(adata.X, axis=1).A1
    adata.obs['n_counts'] = adata.X.sum(axis=1).A1
    
    notmito_genes = [not x for x in mito_genes]
    adata = adata[:,notmito_genes]
    del adata.obs['batch']
    print(adata)
    
    return adata


def prepro(filename):
    data_path = filename
    mat, obs, var, uns = read_data(data_path, sparsify=False, skip_exprs=False)
    if isinstance(mat, np.ndarray):
        X = np.array(mat)
    else:
        X = np.array(mat.toarray())
    cell_name = np.array(obs["cell_type1"])
    cell_type, cell_label = np.unique(cell_name, return_inverse=True)
    return X, cell_label

def normalize(adata, copy=True, highly_genes = None, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError
    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error
    if adata.X.size < 50e6: # check if adata.X is integer only if array is small
        if sp.sparse.issparse(adata.X):
            assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
        else:
            assert np.all(adata.X.astype(int) == adata.X), norm_error

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata
    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0
    if logtrans_input:
        sc.pp.log1p(adata)
    if highly_genes != None:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = highly_genes, subset=True)
    if normalize_input:
        sc.pp.scale(adata)
    return adata


def normalize_batch(adata, batch_key = None, n_high_var = 2000,
                     normalize_samples = True, log_normalize = True, 
                     normalize_features = True):

    n, p = adata.shape
    sparsemode = issparse(adata.X)
    
    if batch_key is not None:
        batch = list(adata.obs[batch_key])
        batch = convert_vector_to_encoding(batch)
        batch = np.asarray(batch)
        batch = batch.astype('float32')
    else:
        batch = np.ones((n,), dtype = 'float32')
        norm_by_batch = False
        
    sc.pp.filter_genes(adata, min_counts=1)
    sc.pp.filter_cells(adata, min_counts=1)
        
    count = adata.X.copy()
        
    if normalize_samples:
        out = sc.pp.normalize_total(adata, inplace = False)
        obs_ = adata.obs
        var_ = adata.var
        adata = None
        adata = AnnData(out['X'])
        adata.obs = obs_
        adata.var = var_
        
        size_factors = out['norm_factor'] / np.median(out['norm_factor'])
        out = None
    else:
        size_factors = np.ones((adata.shape[0], ))
        
    if not log_normalize:
        adata_ = adata.copy()
    
    sc.pp.log1p(adata)
    
    if n_high_var is not None:
        sc.pp.highly_variable_genes(adata, inplace = True, min_mean = 0.0125, max_mean = 3, min_disp = 0.5, 
                                          n_bins = 20, n_top_genes = n_high_var, batch_key = batch_key)
        
        hvg = adata.var['highly_variable'].values
        
        if not log_normalize:
            adata = adata_.copy()

    else:
        hvg = [True] * adata.shape[1]
        
    if normalize_features:
        batch_list = np.unique(batch)

        if sparsemode:
            adata.X = adata.X.toarray()

        for batch_ in batch_list:
            indices = [x == batch_ for x in batch]
            sub_adata = adata[indices]
            
            sc.pp.scale(sub_adata)
            adata[indices] = sub_adata.X
        
        adata.layers["normalized input"] = adata.X
        adata.X = count
        adata.var['Variance Type'] = [['LVG', 'HVG'][int(x)] for x in hvg]
            
    else:
        if sparsemode:   
            adata.layers["normalized input"] = adata.X.toarray()
        else:
            adata.layers["normalized input"] = adata.X
            
        adata.var['Variance Type'] = [['LVG', 'HVG'][int(x)] for x in hvg]
        
    if n_high_var is not None:
        del_keys = ['dispersions', 'dispersions_norm', 'highly_variable', 'highly_variable_intersection', 'highly_variable_nbatches', 'means']
        del_keys = [x for x in del_keys if x in adata.var.keys()]
        adata.var = adata.var.drop(del_keys, axis = 1)
            
    y = np.unique(batch)
    num_batch = len(y)
    
    adata.obs['size factors'] = size_factors.astype('float32')
    adata.obs['batch'] = batch
    adata.uns['num_batch'] = num_batch
    
    if sparsemode:
        adata.X = adata.X.toarray()
        
    count = adata.layers["normalized input"][:, adata.var['Variance Type'] == 'HVG']
    return count
