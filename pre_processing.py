# !/usr/bin/env python
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale, minmax_scale
from sklearn import preprocessing 



def read_csv(filename, take_log):
    """ Read TPM data of a dataset saved in csv format
    Format of the csv:
    first row: sample labels
    second row: cell labels
    third row: cluster labels from Seurat
    first column: gene symbols
    Args:
        filename: name of the csv file
        take_log: whether do log-transformation on input data
    Returns:
        dataset: a dict with keys 'gene_exp', 'gene_sym', 'sample_labels', 'cell_labels', 'cluster_labels'
    """
    dataset = {}
    df = pd.read_csv(filename, header=None)
    dat = df[df.columns[1:]].values
    dataset['sample_labels'] = dat[0, :].astype(int)
    dataset['cell_labels'] = dat[1, :].astype(int)
    dataset['cluster_labels'] = dat[2, :].astype(int)
    gene_sym = df[df.columns[0]].tolist()[3:]
    gene_exp = dat[3:, :]


    if take_log:
            gene_exp = np.log2(gene_exp + 1)
    dataset['gene_exp'] = gene_exp
    dataset['gene_sym'] = gene_sym
    return dataset



def read_txt(filename, take_log):
    dataset = {}
    df = pd.read_table(filename, header=None)
    dat = df[df.columns[1:]].values
    dataset['cell_labels'] = dat[8, 1:]
    gene_sym = df[df.columns[0]].tolist()[11:]
    gene_exp = dat[11:, 1:].astype(np.float32)
    if take_log:
        gene_exp = np.log2(gene_exp + 1)
    dataset['gene_exp'] = gene_exp
    dataset['gene_sym'] = gene_sym
    dataset['cell_labels'] = convert_strclass_to_numclass(dataset['cell_labels'])

    save_csv(gene_exp, gene_sym,  dataset['cell_labels'])

    return dataset


def pre_processing_single(dataset_file_list, pre_process_paras, type='csv'):
    """ pre-processing of multiple datasets
    Args:
        dataset_file_list: list of filenames of datasets
        pre_process_paras: dict, parameters for pre-processing
    Returns:
        dataset_list: list of datasets
    """
    # parameters
    take_log = pre_process_paras['take_log']
    scaling = pre_process_paras['scaling']
    dataset_list = []
    data_file = dataset_file_list

    if type == 'csv':
        dataset = read_csv(data_file, take_log)
    elif type == 'txt':
        dataset = read_txt(data_file, take_log)

    dataset['gene_exp'] = dataset['gene_exp'].astype(np.float)

    if scaling:  # scale to [0,1]
        minmax_scale(dataset['gene_exp'], feature_range=(0, 1), axis=1, copy=False)


    dataset_list.append(dataset)
    return dataset_list


