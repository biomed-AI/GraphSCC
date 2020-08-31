# -*- coding: utf-8 -*-
import os
import numpy as np
import scipy.sparse as sp
import h5py
import torch
from torch.utils.data import Dataset
import scanpy.api as sc
from preprocess import read_dataset, process_normalize
from pre_processing import pre_processing_single
from sklearn.preprocessing import scale, minmax_scale
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time
import umap
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import copy
import random
import seaborn as sns



def print_time(f):
    """Decorator of viewing function runtime.
    eg:
        ```py
        from print_time import print_time as pt
        @pt
        def work(...):
            print('work is running')
        word()
        # work is running
        # --> RUN TIME: <work> : 2.8371810913085938e-05
        ```
    """

    def fi(*args, **kwargs):
        s = time.time()
        res = f(*args, **kwargs)
        print('--> RUN TIME: <%s> : %s' % (f.__name__, time.time() - s))
        return res

    return fi




def load_graph(dataset, k=None, n=10, label=None):
    import os
    graph_path = os.getcwd()
    if k:
        path = graph_path + '/{}{}_graph.txt'.format(dataset, k)
    else:
        path =graph_path +  '/{}_graph.txt'.format(dataset)


    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    import os
    print("delete file: ", path)
    os.remove(path)

    return adj


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def anta_normalize(x, y):
    # preprocessing scRNA-seq read counts matrix
    y = y.astype(np.int32)
    adata = sc.AnnData(x)
    adata.obs['Group'] = y

    adata = read_dataset(adata,
                         transpose=False,
                         test_split=False,
                         copy=True)

    adata = process_normalize(adata,
                              size_factors=True,
                              normalize_input=True,
                              logtrans_input=True)

    print(adata.X.shape)

    x_sd = adata.X.std(0)
    x_sd_median = np.median(x_sd)
    print("median of gene sd: %.5f" % x_sd_median)

    x = adata.X.astype(np.float32)
    y = y.astype(np.int32)
    raw_data = adata.raw.X
    return x, y, adata.obs.size_factors, raw_data


class load_data_origin_data(Dataset):
    def __init__(self, dataset, load_type="csv", take_log=False, scaling=False):
        def load_txt():
            self.x = np.loadtxt('data/{}.txt'.format(dataset), dtype=float)
            self.y = np.loadtxt('data/{}_label.txt'.format(dataset), dtype=int)

        def load_h5():
            data_mat = h5py.File(dataset)
            self.x = np.array(data_mat['X'])
            self.y = np.array(data_mat['Y'])

        def load_csv():
            pre_process_paras = {'take_log': take_log, 'scaling': scaling}
            self.pre_process_paras = pre_process_paras
            print(pre_process_paras)
            dataset_list = pre_processing_single(dataset, pre_process_paras, type='csv')
            self.x = dataset_list[0]['gene_exp'].transpose().astype(np.float32)
            self.y = dataset_list[0]['cell_labels'].astype(np.int32)
            self.cluster_label = dataset_list[0]['cluster_labels'].astype(np.int32)

        if load_type == "csv":
            load_csv()
        elif load_type == "h5":
            load_h5()
        elif load_type == "txt":
            load_txt()



    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])),\
               torch.from_numpy(np.array(self.y[idx])),\
               torch.from_numpy(np.array(idx))

