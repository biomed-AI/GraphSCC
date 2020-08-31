import pandas as pd
import numpy as np
import h5py
from sklearn.metrics.pairwise import cosine_similarity as cos
from sklearn.metrics import pairwise_distances as pair
from sklearn.preprocessing import normalize
from pre_processing import pre_processing_single
from scipy.stats import spearmanr, pearsonr, kendalltau

from sklearn.metrics.pairwise import cosine_similarity


def construct_graph_kmean(file_name, features, pred_label, label, load_type='csv', topk=None, method='ncos'):
    import os
    graph_path = os.getcwd()
    if topk:
        fname =graph_path + '/{}{}_graph.txt'.format(file_name, topk)
    else:
        fname =graph_path + '/{}_graph.txt'.format(file_name)
    num = len(label)
    dist = None

    if method == 'spearmanr':
        dist = spearmanr(features, axis=1)[0]
    elif method == 'cos':
        features[features > 0] = 1
        dist = np.dot(features, features.T)
    elif method == 'ncos':
        dist = cosine_similarity(features, features)

    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)

    y_pred = pred_label
    f = open(fname, 'w')

    counter = 0
    total = 0

    for i, v in enumerate(inds):
        for vv in v:
            if vv == i:
                pass
            else:
                total += 1
                if label[vv] != label[i]:
                    counter += 1
                f.write('{} {}\n'.format(i, vv))

    f.close()

    print('error rate: {}'.format(counter / (total*1.0)))
    print('error rate: {}'.format(counter / (num*topk)))

    return round(counter / (total * 1.0), 4)



def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


