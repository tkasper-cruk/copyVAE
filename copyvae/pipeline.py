#! /usr/bin/env python3

import argparse
import numpy as np
import tensorflow as tf
import anndata
from tensorflow.keras.optimizers import Adam
from tqdm.keras import TqdmCallback

#from copyvae.preprocess import load_copykat_data
#from copyvae.binning import bin_genes_from_text, bin_genes_from_anndata
from copyvae.vae import CopyVAE
from copyvae.clustering import find_clones_gmm
from copyvae.segmentation import generate_clone_profile
#from copyvae.cell_tools import Clone
from copyvae.graphics import draw_umap, draw_heatmap, plot_breakpoints


def run_pipeline(umi_counts, is_anndata):
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

    bin_size = 25
    max_cp = 15
    intermediate_dim = 128
    latent_dim = 15 #20
    batch_size = 128
    epochs = 200 #400

    #data = load_copykat_data(umi_counts)
    #sc.pp.highly_variable_genes(data,n_top_genes=5000, flavor='seurat_v3', subset=True)
    data = anndata.read_h5ad(umi_counts)
    try: 
        x = data.X.todense()
    except AttributeError:
        x = data.X
    x_bin = x.reshape(-1,bin_size).copy().mean(axis=1).reshape(x.shape[0],-1)

    # train model step 1
    train_dataset1 = tf.data.Dataset.from_tensor_slices(x_bin)
    train_dataset1 = train_dataset1.shuffle(buffer_size=1024).batch(batch_size)
    
    clus_model = CopyVAE(x_bin.shape[-1],
                            intermediate_dim,
                            latent_dim,
                            max_cp)
    clus_model.compile(optimizer=Adam(learning_rate=1e-3,epsilon=0.01))
    clus_model.fit(train_dataset1, epochs=epochs, verbose=0, callbacks=[TqdmCallback()])
    
    # clustering
    m,v,z = clus_model.z_encoder(x_bin)
    pred_label = find_clones_gmm(z, n_clones=2)
    data.obs["pred"] = pred_label.astype('str')

    # find normal cells
    mask_list = []
    min_std = np.inf
    for clus in np.unique(pred_label):
        mask = (pred_label==clus)
        mask_list.append(mask)
        clone_masked = x_bin[mask]
        ex_std = clone_masked.std(axis=1).sum()
        if ex_std < min_std:
            norm_mask = mask
            min_std = ex_std
    confident_norm_x = x_bin[norm_mask]
    """
    norm_mask = (pred_label).astype(bool)
    clone_masked = x_bin[norm_mask]
    clone_unmasked = x_bin[~norm_mask]
    if clone_masked.std(axis=1).sum() > clone_unmasked.std(axis=1).sum():
        norm_mask = (1 - pred_label).astype(bool)
    confident_norm_x = x_bin[norm_mask]
    """
    
    # normalise according to baseline
    baseline = np.median(confident_norm_x, axis=0) #median --> mean, exclude lower .2
    baseline[baseline == 0] = 1
    norm_x = x_bin / baseline * 2
    #norm_x[x_bin <= 0.2] = x_bin[x_bin <= 0.2]
    norm_x[norm_mask] = 2.
    norm_x = tf.convert_to_tensor(norm_x,dtype='float')

    # train model step 2
    train_dataset2 = tf.data.Dataset.from_tensor_slices(norm_x)
    train_dataset2 = train_dataset2.shuffle(buffer_size=1024).batch(batch_size)

    copyvae = CopyVAE(norm_x.shape[-1],
                            intermediate_dim,
                            latent_dim,
                            max_cp)
    copyvae.compile(optimizer=Adam(learning_rate=1e-3,epsilon=0.01))
    copyvae.fit(train_dataset2, epochs=epochs, verbose=0, callbacks=[TqdmCallback()])

    # get copy number profile
    _, _, latent_z = copyvae.z_encoder(norm_x)
    copy_bin = copyvae.encoder(latent_z)

    data.obsm['latent'] = z
    #draw_umap(data, 'latent', '_latent')
    data.obsm['copy_number'] = copy_bin
    #draw_umap(data, 'copy_number', '_copy_number')
    #draw_heatmap(copy_bin,'bin_copies')
    with open('copy.npy', 'wb') as f:
        np.save(f, copy_bin)

    # seperate tumour cells from normal
    nor_cp = np.median(copy_bin[norm_mask],axis=0)
    nor_cp = np.repeat(nor_cp, bin_size)
    tum_cp = np.median(copy_bin[~norm_mask],axis=0)
    tum_cp = np.repeat(tum_cp, bin_size)
    
    # generate clone profile
    chrom_list = np.load('chrom_list.npy')
    clone_cn = copy_bin[~norm_mask]
    segment_cn, breakpoints = generate_clone_profile(clone_cn, chrom_list, eta=6)
    clone_arr = clone_cn.numpy()
    st = 0
    for i in range(1,len(breakpoints)):
        ed = breakpoints[i]
        m = np.mean(clone_arr[:,st:ed],axis=1)
        clone_arr[:, st:ed] = np.repeat(
                                        np.reshape(m,(1, m.size)).T,
                                        ed-st,
                                        axis=1)
        st = ed
    m = np.mean(clone_arr[:,st:],axis=1)
    clone_arr[:, st:] = np.repeat(
                                    np.reshape(m,(1, m.size)).T,
                                    np.shape(clone_arr)[1]-st,
                                    axis=1)
    with open('single_cell_profile.npy', 'wb') as f:
        np.save(f, clone_arr)
    final_prof = np.repeat(segment_cn, bin_size)
    with open('clone_profile.npy', 'wb') as f:
        np.save(f, final_prof)
    with open('breakpoints.npy', 'wb') as f:
        np.save(f, breakpoints)

    return None


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help="input UMI")
    parser.add_argument('-a', nargs='?', const=True, default=False, help="flag for 10X data")
    parser.add_argument('-g', '--gpu', type=int, help="GPU id")

    args = parser.parse_args()
    file = args.input
    is_10x = args.a

    if args.gpu:
        dvc = '/device:GPU:{}'.format(args.gpu)
    else:
        dvc = '/device:GPU:0'

    with tf.device(dvc):
        run_pipeline(file, is_anndata=is_10x)


if __name__ == "__main__":
    main()
