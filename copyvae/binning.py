#! /usr/bin/env python3

import numpy as np
import pandas as pd
from os import path

LOCAL_DIRECTORY = path.dirname(path.abspath(__file__))
GENE_META = path.join(LOCAL_DIRECTORY, '../data/mart_export.txt')

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


def build_gene_map(gene_metadata=GENE_META, chr_pos=CHR_BASE_PAIRS):
    """ Build gene map from meta data

    """
    gene_info = pd.read_csv(gene_metadata, sep='\t')
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

    gene_map = gene_info[gene_info['Gene type']=='protein_coding']
    gene_map['abspos']= gene_map['Gene start (bp)']
    for i in range(len(chr_pos)):
        gene_map.loc[gene_map['chr']==i+1, 'abspos'] += chr_pos[:i].sum()

    return gene_map


def bin_genes_umi(umi_counts, bin_size, gene_metadata=GENE_META):
    """ Gene binning for UMI counts

    Args:
        umi_counts: text file
        bin_size: number of genes per bin (int)
        gene_metadata: gene name map

    Returns:
        expressed_gene: high expressed gene counts,
                        number of genes fully divided by bin_size
    """

    umis = pd.read_csv(umi_counts, sep='\t', low_memory=False).reset_index()
    labels = umis.iloc[1, :]

    gene_map = build_gene_map(gene_metadata)

    # remove cell cycle genes
    umis = umis[~umis['index'].str.contains("HLA")]
    # gene name mapping
    sorted_gene = pd.merge(
        umis, gene_map, right_on=['Gene name'],
        left_on=['index'], how='inner'
    )

    # remove non-expressed genes
    ch = sorted_gene.iloc[:, 1:-7].astype('int32')
    nonz = np.count_nonzero(ch, axis=1)
    sorted_gene['expressed'] = nonz
    expressed_gene = sorted_gene[sorted_gene['expressed'] > 0]

    # bin and remove exceeded genes
    n_exceeded = expressed_gene.chr.value_counts() % bin_size
    for chrom in n_exceeded.index:
        n = n_exceeded[chrom]
        ind = expressed_gene[
            expressed_gene['chr'] == chrom
        ].sort_values(by=['expressed', 'Gene start (bp)']
                      )[:n].index
        expressed_gene = expressed_gene.drop(index=ind)

    abs_pos = expressed_gene['abspos'].values

    # extract chromosome boundary
    bin_number = expressed_gene.chr.value_counts() // bin_size
    chrom_bound = bin_number.sort_index().cumsum()
    chrom_list = [(0, chrom_bound[1])]
    for i in range(2, 24):
        start_p = chrom_bound[i - 1]
        end_p = chrom_bound[i]
        chrom_list.append((start_p, end_p))

    # clean and add labels
    expressed_gene.drop(columns=expressed_gene.columns[-8:],
                        axis=1, inplace=True)
    expressed_gene = pd.concat([expressed_gene.T, labels], axis=1)
    expressed_gene.rename(columns=expressed_gene.iloc[0], inplace=True)
    expressed_gene.drop(expressed_gene.index[0], inplace=True)
    expressed_gene = expressed_gene.sort_values(by='cluster.pred')
    expressed_gene.to_csv('bined_expressed_cell.csv', sep='\t')

    return expressed_gene, chrom_list, abs_pos
