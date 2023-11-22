#! /usr/bin/env python3

import argparse
import numpy as np
import tensorflow as tf
import anndata
from tensorflow.keras.optimizers import Adam
from tqdm.keras import TqdmCallback

from copyvae.preprocess import filter_and_normalize_data
from copyvae.binning import build_gene_map, bin_genes_from_anndata
from copyvae.vae import CopyVAE
from copyvae.clustering import find_clones_kmeans, find_normal_cluster
from copyvae.segmentation import generate_clone_profile
#from copyvae.graphics import draw_umap, draw_heatmap, plot_breakpoints

def train_neural_network(data, max_cp, intermediate_dim, latent_dim, batch_size, epochs, lr=1e-3, eps=0.01):
    """
    Train a neural network on the provided data.

    Args:
        data (np.ndarray): Data array with shape (n_samples, n_features).
        intermediate_dim (int): Dimension of the intermediate layer in the neural network.
        latent_dim (int): Dimension of the latent space in the neural network.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        lr (float): learning rate.
        eps (float): epsilon.

    Returns:
        tf.keras.models.Model: Trained neural network.
    """
    
    # Create a TensorFlow dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(data)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # Create the neural network model (replace with your model)
    model = CopyVAE(data.shape[-1], intermediate_dim, latent_dim, max_cp)

    # Compile the model with an optimizer
    optimizer = Adam(learning_rate=lr, epsilon=eps)
    model.compile(optimizer=optimizer)

    # Fit the model with the training dataset
    model.fit(train_dataset, epochs=epochs, verbose=0, callbacks=[TqdmCallback()])

    return model


def compute_pseudo_copy(x_bin, norm_mask):
    """
    Compute the pseudo-copy values for a cells.

    Args:
        x_bin (np.ndarray): Data array with shape (n_samples, n_features).
        norm_mask (np.ndarray): Boolean mask for the normal cells

    Returns:
        pseudo_cp: Pseudo-copy values.
    """

    confident_norm_x = x_bin[norm_mask]
    print(np.shape(confident_norm_x))
    baseline = np.median(confident_norm_x, axis=0)
    baseline[baseline == 0] = 1
    pseudo_cp = x_bin / baseline * 2
    #pseudo_cp[x_bin <= 0.2] = x_bin[x_bin <= 0.2]
    pseudo_cp[norm_mask] = 2.

    # Convert to TensorFlow tensor
    pseudo_cp = tf.convert_to_tensor(pseudo_cp, dtype='float')

    return pseudo_cp



def split_data_by_cluster(data, pred_label, ncell_index):
    """
    Split data by unique values in pred_label excluding normal cell index.

    Args:
        data (np.ndarray): Data array with shape (n_samples, n_features).
        pred_label (np.ndarray): Cluster labels for each sample.
        ncell_index (int): Index to exclude from splitting.

    Returns:
        dict: Dictionary where keys are unique labels and values are corresponding subarrays.
    """
    unique_labels = np.unique(pred_label)

    # Exclude ncell_index from unique labels
    unique_labels = unique_labels[unique_labels != ncell_index]

    split_data = {}

    for label in unique_labels:
        mask = (pred_label == label)
        split_data[label] = data[mask]

    return split_data


def perform_segmentation(clone_cn, chrom_list, eta):
    """
    Perform segmentation on copy number data.

    Args:
        clone_cn (tf.Tensor): Copy number data.
        chrom_list (list): List of chromosome boundary bins.
        eta (int): Eta value for segmentation.

    Returns:
        segment_cn
        breakpoints
        cell_cn : Single cell copy number profile.
    """

    # Generate clone profile and breakpoints
    segment_cn, breakpoints = generate_clone_profile(clone_cn, chrom_list, eta)
    
    # Generate single cell copy profile
    cell_cn = clone_cn.numpy()
    st = 0
    for i in range(1, len(breakpoints)):
        ed = breakpoints[i]
        m = np.mean(cell_cn[:, st:ed], axis=1)
        cell_cn[:, st:ed] = np.repeat(
                                        np.reshape(m, (1, m.size)).T,
                                        ed - st,
                                        axis=1)
        st = ed

    m = np.mean(cell_cn[:, st:], axis=1)
    cell_cn[:, st:] = np.repeat(
                                    np.reshape(m, (1, m.size)).T,
                                    np.shape(cell_cn)[1] - st,
                                    axis=1)

    return segment_cn, breakpoints, cell_cn


def run_pipeline(umi_counts, cell_cycle_gene_list):
    """ Main pipeline

    Args:
        umi_counts: umi file
        cell_cycle_gene_list: cell cycle gene file
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
    number_of_clones = 2

    # preprocess data
    feature_names=['gene_ids','gene_symbol']
    adata = filter_and_normalize_data(umi_counts,
                                      cell_cycle_gene_list,
                                      alternative_feature_names=feature_names)
    server_url = 'http://www.ensembl.org'
    attributes = ['external_gene_name',
                  'ensembl_gene_id',
                  'chromosome_name',
                  'start_position',
                  'end_position']
    gene_map = build_gene_map(server_url, attributes)
    data, chrom_list = bin_genes_from_anndata(adata, bin_size, gene_map)
    with open('chrom_list.npy', 'wb') as f:
        np.save(f, chrom_list)
    print('finished preprocess')

    # train cluster NNs
    try: 
        x = data.X.todense()
    except AttributeError:
        x = data.X
    x_bin = x.reshape(-1,bin_size).copy().mean(axis=1).reshape(x.shape[0],-1)

    # train model step 1
    clus_model = train_neural_network(x_bin,
                                      max_cp,
                                      intermediate_dim,
                                      latent_dim,
                                      batch_size,
                                      epochs)
    
    # clustering
    print('Idetifying normal cells...')
    m,v,z = clus_model.z_encoder(x_bin)
    pred_label = find_clones_kmeans(z, n_clones=number_of_clones)
    #data.obs["pred"] = pred_label.astype('str')

    # find normal cells
    cluster_auto_corr, ncell_index = find_normal_cluster(x_bin, pred_label)
    norm_mask = (pred_label == ncell_index)
    norm_x = compute_pseudo_copy(x_bin, norm_mask)

    # train model step 2
    print('Training CopyVAE...')
    copyvae = train_neural_network(norm_x,
                                   max_cp,
                                   intermediate_dim,
                                   latent_dim,
                                   batch_size,
                                   epochs,
                                   eps=0.02)

    # get copy number profile
    _, _, latent_z = copyvae.z_encoder(norm_x)
    copy_bin = copyvae.encoder([norm_x,latent_z])

    data.obsm['latent'] = z
    #draw_umap(data, 'latent', '_latent')
    data.obsm['copy_number'] = copy_bin
    #draw_umap(data, 'copy_number', '_copy_number')
    #draw_heatmap(copy_bin,'bin_copies')
    with open('copy.npy', 'wb') as f:
        np.save(f, copy_bin)

    # seperate tumour cells from normal
    cluster_dict = split_data_by_cluster(copy_bin, pred_label, ncell_index)
    
    # generate clone profile
    print('Segmantation')
    #chrom_list = np.load('chrom_list.npy')
    for label, cluster_data in cluster_dict.items():
        segment_cn, breakpoints, sc_cn = perform_segmentation(cluster_data, chrom_list, eta=6)
        filename = f"clone_{label}_single_cell_profile.npy"
        with open(filename, 'wb') as f:
            np.save(f, sc_cn)
        final_prof = np.repeat(segment_cn, bin_size)
        filename = f"clone_{label}_profile.npy"
        with open(filename, 'wb') as f:
            np.save(f, final_prof)
        filename = f"clone_{label}_breakpoints.npy"
        with open(filename, 'wb') as f:
            np.save(f, breakpoints)

    return None


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help="input UMI")
    parser.add_argument('cell_cycle_genes', help="path of list of cell cycle genes")
    parser.add_argument('-g', '--gpu', type=int, help="GPU id")

    args = parser.parse_args()
    file = args.input
    cc_genes = args.cell_cycle_genes

    if args.gpu:
        dvc = '/device:GPU:{}'.format(args.gpu)
    else:
        dvc = '/device:GPU:0'

    with tf.device(dvc):
        run_pipeline(file, cc_genes)


if __name__ == "__main__":
    main()
