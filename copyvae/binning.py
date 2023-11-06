#! /usr/bin/env python3

import numpy as np
import pandas as pd
import scanpy as sc
import pybiomart


CHR_BASE_PAIRS = np.array([
    248956422,
    242193529,
    198295559,
    190214555,
    181538259,
    170805979,
    159345973,
    145138636,
    138394717,
    133797422,
    135086622,
    133275309,
    114364328,
    107043718,
    101991189,
    90338345,
    83257441,
    80373285,
    58617616,
    64444167,
    46709983,
    50818468,
    156040895])


def _build_gene_map(chr_pos=CHR_BASE_PAIRS):
    """ Build gene map from meta data

    """
    server = pybiomart.Server(host='http://www.ensembl.org')
    mart = server['ENSEMBL_MART_ENSEMBL']
    dataset = mart['hsapiens_gene_ensembl']
    attributes = [#'ensembl_transcript_id',
                  'external_gene_name', 
                  'ensembl_gene_id',
                  'chromosome_name',
                  'start_position',
                  'end_position']
    
    gene_info = dataset.query(attributes= attributes)
    gene_info.rename(columns={'Chromosome/scaffold name': 'chr'}, inplace=True)

    gene_info.loc[gene_info.chr == 'X', 'chr'] = '23'
    gene_info = gene_info[
        gene_info['chr'].isin(
            np.arange(1, 24).astype(str)
        )
    ]
    gene_info = gene_info.copy()
    gene_info.loc[:, 'chr'] = gene_info.chr.astype(int)
    gene_info = gene_info.sort_values(by=['chr', 'Gene start (bp)'])

    #gene_map = gene_info[gene_info['Gene type']=='protein_coding'].copy()
    gene_map = gene_info
    gene_map['abspos'] = gene_map['Gene start (bp)']
    for i in range(len(chr_pos)):
        gene_map.loc[gene_map['chr'] == i + 1, 'abspos'] += chr_pos[:i].sum()

    return gene_map


def build_gene_map(server_url, attributes, chr_pos=CHR_BASE_PAIRS):
    """
    Build a gene map from Ensembl metadata.

    Args:
        server_url (str): URL of the Ensembl server.
        attributes (list): List of attributes to retrieve.
        chr_pos (array-like): Array of chromosome base pair counts.

    Returns:
        pd.DataFrame: DataFrame containing the gene map.
    """
    
    # Connect to the Ensembl server
    server = pybiomart.Server(host=server_url)
    
    # Select the dataset and attributes
    mart = server['ENSEMBL_MART_ENSEMBL']
    dataset = mart['hsapiens_gene_ensembl']
    
    # Query for gene information
    gene_info = dataset.query(attributes=attributes)
    gene_info.rename(columns={'Chromosome/scaffold name': 'chr'}, inplace=True)
    
    # Filter chromosomes to only include 1-23
    gene_info.loc[gene_info.chr == 'X', 'chr'] = '23'
    valid_chromosomes = np.arange(1, 24).astype(str)
    gene_info = gene_info[gene_info['chr'].isin(valid_chromosomes)]
    
    # Convert 'chr' column to integer
    gene_info['chr'] = gene_info['chr'].astype(int)
    
    # Sort the gene information by chromosome and start position
    gene_info = gene_info.sort_values(by=['chr', 'Gene start (bp)'])
    
    # Calculate absolute positions
    gene_info['abspos'] = gene_info['Gene start (bp)']
    for i in range(len(chr_pos)):
        gene_info.loc[gene_info['chr'] == i + 1, 'abspos'] += chr_pos[:i].sum()
    
    return gene_info


def bin_genes_from_anndata(adata, bin_size, gene_map):
    """
    Perform gene binning for UMI counts.

    Args:
        adata (anndata object): Anndata object with UMI counts.
        bin_size (int): Number of genes per bin.
        gene_map (pd.DataFrame): Gene map with chromosome and absolute position information.

    Returns:
        data (anndata object): Anndata object with high expressed gene counts.
        chrom_list (list): List of chromosome boundary bins.
    """

    # Normalize UMI counts
    sc.pp.filter_genes(adata, min_cells=1)

    # Map chromosome and absolute position information
    map_chr = dict(gene_map[['Gene name', 'chr']].values)
    adata.var['chr'] = adata.var.gene_ids.map(map_chr)
    map_abs = dict(gene_map[['Gene name', 'abspos']].values)
    adata.var['abspos'] = adata.var.gene_ids.map(map_abs)

    # Filter out genes with missing chromosome or absolute position
    adata_clean = adata[:, adata.var.chr.notna() & adata.var.abspos.notna()].copy()
    adata_clean.var['chr'] = adata_clean.var['chr'].astype('int')
    adata_clean.var['abspos'] = adata_clean.var['abspos'].astype('int')

    # Sort genes by absolute position
    adata_clean = adata_clean[:, list(adata_clean.var.sort_values(by='abspos').index)].copy()

    # Calculate the number of genes exceeded for each chromosome
    n_exceeded = adata_clean.var['chr'].value_counts() % bin_size
    ind_list = []

    for chrom in n_exceeded.index:
        n = n_exceeded[chrom]
        ind = adata_clean.var[adata_clean.var['chr'] == chrom].sort_values(
            by=['n_cells', 'abspos'])[:n].index.values
        ind_list.append(ind)

    # Remove exceeded genes
    data = adata_clean[:, ~adata_clean.var.index.isin(np.concatenate(ind_list))].copy()

    # Find chromosome boundary bins
    bin_number = adata_clean.var['chr'].value_counts() // bin_size
    chrom_bound = bin_number.sort_index().cumsum()
    chrom_list = [(0, chrom_bound[1])]

    for i in range(2, 24):
        start_p = chrom_bound[i - 1]
        end_p = chrom_bound[i]
        chrom_list.append((start_p, end_p))

    return data, chrom_list


def _bin_genes_from_anndata(file, bin_size):
    """ Gene binning for UMI counts for 10X data

    Args:
        file: h5 file
        bin_size: number of genes per bin (int)
        gene_metadata: gene name map

    Returns:
        data: (anndata object) high expressed gene counts,
                        number of genes fully divided by bin_size
        chrom_list: list of chromosome boundry bins
    """

    #adata = sc.read_h5ad(file)
    adata=file
    #adata.var.drop_duplicates(subset=['gene_ids'],inplace =True)
    #adata = sc.read(file)
    gene_map = build_gene_map()

    # normalize UMI counts
    sc.pp.filter_genes(adata, min_cells=1)
    #sc.pp.normalize_total(adata, inplace=True)
    #adata.X = np.round(adata.X)

    # extract genes
    """gene_df = pd.merge(
        adata.var,
        gene_map,
        right_on=['Gene name'],#['Gene stable ID'],
        left_on=['gene_ids'],
        how='right').dropna()
    gene_df = gene_df.drop_duplicates(subset=['gene_ids'])
    adata_clean = adata[:, adata.var.gene_ids.isin(gene_df.gene_ids.values)].copy()

    # add position in genome
    adata_clean.var['chr'] = gene_df['chr'].values
    adata_clean.var['abspos'] = gene_df['abspos'].values"""

    map_chr = dict(gene_map[['Gene name', 'chr']].values)
    adata.var['chr'] = adata.var.gene_ids.map(map_chr)
    map_abs = dict(gene_map[['Gene name', 'abspos']].values)
    adata.var['abspos'] = adata.var.gene_ids.map(map_abs)
    adata_clean = adata[:, (adata.var.chr.notna() & adata.var.abspos.notna())].copy()
    adata_clean.var['chr'] =  adata_clean.var.chr.astype('int')
    adata_clean.var['abspos'] =  adata_clean.var.abspos.astype('int')
    adata_clean = adata_clean[:, list(adata_clean.var.sort_values(by = 'abspos'.split()).index)].copy()

    # filter out low expressed genes
    #sc.pp.filter_genes(adata_clean, min_cells=100)
    # remove exceeded genes
    n_exceeded = adata_clean.var['chr'].value_counts() % bin_size
    ind_list = []
    for chrom in n_exceeded.index:
        n = n_exceeded[chrom]
        ind = adata_clean.var[adata_clean.var['chr'] == chrom].sort_values(
                                by=['n_cells', 'abspos'])[:n].index.values
        ind_list.append(ind)
    data = adata_clean[:, ~adata_clean.var.index.isin(
                            np.concatenate(ind_list))]
    #with open('abs.npy', 'wb') as f:
    #    np.save(f, data.var['abspos'].values)

    # find chromosome boundry bins
    bin_number = adata_clean.var['chr'].value_counts() // bin_size
    chrom_bound = bin_number.sort_index().cumsum()
    chrom_list = [(0, chrom_bound[1])]
    for i in range(2, 24):
        start_p = chrom_bound[i - 1]
        end_p = chrom_bound[i]
        chrom_list.append((start_p, end_p))

    return data, chrom_list
