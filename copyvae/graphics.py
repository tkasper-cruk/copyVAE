#!/usr/bin/env python3

import matplotlib.pylab as plt
import seaborn as sns
import scanpy as sc

def draw_heatmap(data_arr, file_name):

    plt.figure(figsize=(17,100),dpi=120)
    fig = sns.heatmap(data_arr, cbar_kws={"shrink": 0.2},
                            cmap=sns.blend_palette(['#7CAACD','white', '#CB6527'],
                            as_cmap=True)).get_figure()
    fig.savefig('figures/{}.png'.format(file_name))

    return None


def draw_umap(adata, feature, file_name):

    sc.pp.neighbors(adata, use_rep=feature)
    sc.tl.umap(adata, min_dist=0.3)
    sc.pl.umap(
                adata,
                color=["cell_type"],
                frameon=False,
                save='{}.png'.format(file_name)
                )

    return None
