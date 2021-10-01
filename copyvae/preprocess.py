#! /usr/bin/env python3

import csv
import pandas as pd
import numpy as np
import logging
import anndata

logger = logging.getLogger(__name__)

def load_cortex_txt(path_to_file: str) -> anndata.AnnData:
    logger.info("Loading Cortex data from {}".format(path_to_file))
    rows = []
    gene_names = []
    with open(path_to_file, "r") as csvfile:
        data_reader = csv.reader(csvfile, delimiter="\t")
        for i, row in enumerate(data_reader):
            if i == 1:
                precise_clusters = np.asarray(row, dtype=str)[2:]
            if i == 8:
                clusters = np.asarray(row, dtype=str)[2:]
            if i >= 11:
                rows.append(row[1:])
                gene_names.append(row[0])
    cell_types, labels = np.unique(clusters, return_inverse=True)
    _, precise_labels = np.unique(precise_clusters, return_inverse=True)
    data = np.asarray(rows, dtype=int).T[1:]
    gene_names = np.asarray(gene_names, dtype=str)
    gene_indices = []
    extra_gene_indices = []
    gene_indices = np.concatenate(
                                [gene_indices, 
                                extra_gene_indices]).astype(np.int32)
    if gene_indices.size == 0:
        gene_indices = slice(None)

    data = data[:, gene_indices]
    gene_names = gene_names[gene_indices]
    data_df = pd.DataFrame(data, columns=gene_names)
    adata = anndata.AnnData(X=data_df)
    adata.obs["labels"] = labels
    adata.obs["precise_labels"] = precise_clusters
    adata.obs["cell_type"] = clusters
    logger.info("Finished loading Cortex data")
    return adata


def load_copykat_data(file):

    X = pd.read_csv(file, sep='\t', low_memory=False).T
    gene_names = X.columns[2:]
    clusters = np.asarray(X['cluster.pred'], dtype=str)
    cell_types, labels = np.unique(clusters, return_inverse=True)
    data_df = X.iloc[:, 2:]
    adata = anndata.AnnData(X=data_df)
    adata.obs["labels"] = labels
    adata.obs["cell_type"] = clusters

    return adata


def load_data(file):

    X = pd.read_csv(file, sep='\t', low_memory=False, index_col=0)
    clusters = np.asarray(X['cluster.pred'], dtype=str)
    cell_types, labels = np.unique(clusters, return_inverse=True)
    data_df = X.iloc[:,:-1]
    adata = anndata.AnnData(X=data_df)
    adata.obs["labels"] = labels
    adata.obs["cell_type"] = clusters

    return adata


def annotate_data(data, abs_pos):
    """ Pack data into anndata class

    Args:
        data: pandas DataFrame
    Returns:
        adata: anndata class
    """

    clusters = np.asarray(data['cluster.pred'], dtype=str)
    cell_types, labels = np.unique(clusters, return_inverse=True)
    data_df = data.iloc[:,:-1]
    adata = anndata.AnnData(X=data_df)
    adata.var['name'] = data.columns.values[1:]
    adata.var['abspos'] = abs_pos
    adata.obs["labels"] = labels
    adata.obs["cell_type"] = clusters

    return adata

### example
"""
from scvi.data._anndata import setup_anndata

data_path_scvi = 'scvi_data/'
data_path_kat = 'copykat_data/txt_files/'
adata = load_cortex_txt(data_path_scvi + 'expression_mRNA_17-Aug-2014.txt')
# copyKAT DCIS1
data = load_copykat_data(data_path_kat + 'GSM4476485_combined_UMIcount_CellTypes_DCIS1.txt')
setup_anndata(data, labels_key="labels")
"""
