#! /usr/bin/env python3

import numpy as np
import pandas as pd
from os import path

LOCAL_DIRECTORY = path.dirname(path.abspath(__file__))
GENE_MAP = path.join(LOCAL_DIRECTORY, '../data/mart_export.txt')


def bin_genes_umi(umi_counts, bin_size, gene_metadata=GENE_MAP):
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
    gene_info = pd.read_csv(gene_metadata, sep='\t')
    gene_info.rename(columns={'Chromosome/scaffold name': 'chr'}, inplace=True)

    # remove cell cycle genes
    umis = umis[~umis['index'].str.contains("HLA")]
    # gene name mapping
    gene_merge = pd.merge(
        umis, gene_info, right_on=['Gene name'],
        left_on=['index'], how='inner'
    )
    gene_merge.loc[gene_merge.chr == 'X', 'chr'] = '23'
    coding_gene = gene_merge[
        gene_merge['chr'].isin(
            np.arange(1, 24).astype(str)
        )
    ]
    coding_gene = coding_gene.copy()
    coding_gene.loc[:, 'chr'] = coding_gene.chr.astype(int)
    sorted_gene = coding_gene.sort_values(by=['chr', 'Gene start (bp)'])
    sorted_gene.to_csv('sorted_genes.csv', sep='\t')

    # remove non-expressed genes
    ch = sorted_gene.iloc[:, 1:-6].astype('int32')
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

    # extract chromosome boundary
    bin_number = expressed_gene.chr.value_counts() // bin_size
    chrom_bound = bin_number.sort_index().cumsum()
    chrom_list = [(0, chrom_bound[1])]
    for i in range(2, 24):
        start_p = chrom_bound[i - 1]
        end_p = chrom_bound[i]
        chrom_list.append((start_p, end_p))

    # clean and add labels
    expressed_gene.drop(columns=expressed_gene.columns[-7:],
                        axis=1, inplace=True)
    expressed_gene = pd.concat([expressed_gene.T, labels], axis=1)
    expressed_gene.rename(columns=expressed_gene.iloc[0], inplace=True)
    expressed_gene.drop(expressed_gene.index[0], inplace=True)
    expressed_gene = expressed_gene.sort_values(by='cluster.pred')
    expressed_gene.to_csv('bined_expressed_cell.csv', sep='\t')

    return expressed_gene, chrom_list
