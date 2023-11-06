#! /usr/bin/env python3

import csv
import pandas as pd
import numpy as np
import logging
import anndata
import scanpy as sc
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_h5ad_data(file) -> anndata.AnnData:

    adata = anndata.read_h5ad(file)
    adata.var['gene_ids'] = adata.var['gene_name']

    cy = pd.read_csv("../data/Macosko_cell_cycle_genes.txt",sep='\t')
    cy_genes = cy.values.ravel()
    cy_g = cy_genes[~pd.isnull(cy_genes)]

    # filter out cell cycle genes, mtRNA and HLA
    adata.var['if_cycle'] = adata.var['gene_ids'].isin(cy_g)
    adata.var['if_mt'] = adata.var['gene_ids'].str.startswith('MT')
    adata.var['if_hla'] = adata.var['gene_ids'].str.contains("HLA")
    adata_n = adata[:,(adata.var.if_cycle==False) &(adata.var.if_mt==False) & (adata.var.if_hla==False)].copy()

    sc.pp.normalize_total(adata_n)

    return adata_n


def filter_and_normalize_data(file_path: Path, cy_file_path: Path, normalize=True, alternative_feature_names=None) -> anndata.AnnData:
    """
    Load and preprocess anndata from a .h5ad file by filtering and optionally normalizing the data.

    Args:
        file_path (Path): Path to the .h5ad file.
        normalize (bool): Whether to perform total count normalization (default is True).
        alternative_feature_names (list or None): List of alternative feature names to search for.

    Returns:
        anndata.AnnData: Preprocessed AnnData object.
    """
    
    # Load the AnnData object from the .h5ad file
    adata = anndata.read_h5ad(file_path)

    # Check if the 'gene_name' feature exists in the file
    if 'gene_name' in adata.var:
        adata.var['gene_ids'] = adata.var['gene_name']
    elif alternative_feature_names:
            # Search for alternative feature names
            for alt_name in alternative_feature_names:
                if alt_name in adata.var:
                    adata.var['gene_ids'] = adata.var[alt_name]
                    break
            else:
                raise ValueError("None of the alternative feature names were found in the dataset.")
    else:
            raise ValueError("The provided file does not contain the 'gene_name' feature, and no alternative feature names were provided.")

    # Rename 'gene_name' to 'gene_ids' for consistency
    #adata.var['gene_ids'] = adata.var['gene_name']

    # Read cell cycle genes from a list
    #cy_file =  "../data/Macosko_cell_cycle_genes.txt"
    cy = pd.read_csv(cy_file_path, sep='\t')
    cy_genes = cy.values.ravel()
    cy_g = cy_genes[~pd.isnull(cy_genes)]

    # Filter out cell cycle genes, mitochondrial RNA (mtRNA), and HLA genes
    adata.var['if_cycle'] = adata.var['gene_ids'].isin(cy_g)
    adata.var['if_mt'] = adata.var['gene_ids'].str.startswith('MT')
    adata.var['if_hla'] = adata.var['gene_ids'].str.contains("HLA")
    adata_filtered = adata[:, ~((adata.var.if_cycle) | (adata.var.if_mt) | (adata.var.if_hla))].copy()

    # Perform total count normalization if requested
    if normalize:
        sc.pp.normalize_total(adata_filtered)

    return adata_filtered



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



