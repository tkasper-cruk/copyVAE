#! /usr/bin/env python3

import argparse
import numpy as np
import tensorflow as tf

from copyvae.binning import bin_genes_from_text, bin_genes_from_anndata
from copyvae.vae import CopyVAE, train_vae, zinb_pos, nb_pos
from copyvae.clustering import find_clones_gmm
from copyvae.segmentation import bin_to_segment
from copyvae.cell_tools import Clone
from copyvae.graphics import draw_umap, draw_heatmap, plot_breakpoints


def run_pipeline(umi_counts, is_anndata=False):
    """ Main pipeline

    Args:
        umi_counts: umi file
        is_anndata: set to True when using 10X data
    Params:
        max_cp: maximum copy number
        bin_size: number of genes per bin
        intermediate_dim: number of intermediate dimensions for vae
        latent_dim: number of latent dimensions for vae
        batch_size: batch size for training
        epochs = number of epochs training
    """

    max_cp = 6
    bin_size = 25
    intermediate_dim = 128
    latent_dim = 10
    batch_size = 128
    epochs = 250

    # assign genes to bins
    if is_anndata:
        adata, chroms= bin_genes_from_anndata(umi_counts, bin_size)
        # TODO add cell_type for 10X data
        adata.obs["cell_type"] = 1
        x_train = adata.X.todense()
    else:
        adata, chroms= bin_genes_from_text(umi_counts, bin_size)
        x_train = adata.X

    # train model
    # TODO remove try except after debugging vae training
    for attempt in range(5):
        try:
            model = CopyVAE(x_train.shape[-1],
                    intermediate_dim,
                    latent_dim,
                    bin_size=bin_size,
                    max_cp=max_cp)
            copy_vae = train_vae(model, x_train, batch_size, epochs)
        except BaseException:
            tf.keras.backend.clear_session()
            continue
        else:
            break

    # get copy number and latent output
    """
    z_mean, _, z = copy_vae.encoder.predict(x_train)
    reconstruction, gene_cn, _ = copy_vae.decoder(z)
  
    recon = - zinb_pos(x_train, reconstruction)
    #recon = - nb_pos(x_train[379:], reconstruction)
    print("infer:")
    print(np.sum(recon))

    gtcp = np.load('gtcp.npy')
    mu = model.decoder.k_layer(gtcp)
    reconstruction[0] = mu

    recon = - zinb_pos(x_train, reconstruction)
    #recon = - nb_pos(x_train[379:], reconstruction)
    print("gound truth:")
    print(np.sum(recon))

    """
    # split into batch to avoid OOM
    input_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    input_dataset = input_dataset.batch(batch_size)
    for step, x in enumerate(input_dataset):
        if step == 0:
            z_mean, _, z = copy_vae.encoder.predict(x)
            reconstruction, gene_cn, _ = copy_vae.decoder(z)
        else:
            z_mean_old = z_mean
            gene_cn_old = gene_cn
            z_mean, _, z = copy_vae.encoder.predict(x)
            reconstruction, gene_cn, _ = copy_vae.decoder(z)
            z_mean = tf.concat([z_mean_old, z_mean], 0)
            gene_cn = tf.concat([gene_cn_old, gene_cn], 0)

    adata.obsm['latent'] = z_mean
    #draw_umap(adata, 'latent', '_latent')
    #draw_heatmap(gene_cn,'gene_copies')
    with open('copy.npy', 'wb') as f:
        np.save(f, gene_cn)

    # compute bin copy number
    gn = x_train.shape[1]
    bin_number = gn // bin_size
    tmp_arr = np.split(gene_cn, bin_number, axis=1)
    tmp_arr = np.stack(tmp_arr, axis=1)
    bin_cn = np.median(tmp_arr, axis=2)
    #draw_heatmap(bin_cn,'bin_copies')
    with open('median_cp.npy', 'wb') as f:
        np.save(f, bin_cn)

    # seperate tumour cells from normal
    tumour_mask = find_clones_gmm(z_mean, x_train)
    cells = bin_cn[tumour_mask]

    clone_size = np.shape(cells)[0]
    t_clone = Clone(1,
                    clone_size,
                    bin_size,
                    cell_gene_cn=gene_cn[tumour_mask],
                    cell_bin_cn=bin_cn[tumour_mask],
                    chrom_bound=chroms
                    )

    # call clone breakpoints
    t_clone.call_breakpoints()
    #print(t_clone.breakpoints)
    #cp_arr = np.mean(cells, axis=0)
    #plot_breakpoints(cp_arr, t_clone.breakpoints, 'bp_plot')

    # generate clone profile
    t_clone.generate_profile()
    clone_seg = t_clone.segment_cn
    #print(clone_seg)
    clone_gene_cn = np.repeat(clone_seg, bin_size)
    with open('clone_gene_cn.npy', 'wb') as f:
        np.save(f, clone_gene_cn)

    # generate consensus segment profile
    bp_arr = t_clone.breakpoints
    seg_profile = bin_to_segment(bin_cn, bp_arr)
    with open('segments.npy', 'wb') as f:
        np.save(f, seg_profile)
    #draw_heatmap(seg_profile, "tumour_seg")
    #"""
    return None


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help="input UMI")
    parser.add_argument('-g', '--gpu', type=int, help="GPU id")

    args = parser.parse_args()
    file = args.input

    if args.gpu:
        dvc = '/device:GPU:{}'.format(args.gpu)
    else:
        dvc = '/device:GPU:0'

    with tf.device(dvc):
        run_pipeline(file)


if __name__ == "__main__":
    main()
