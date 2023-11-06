#! /usr/bin/env python3

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture


def find_clones_gmm(latent_data, expression_data=None, n_clones=2):
    """ GMM find tumour clone

    Args:
        latent_data: latent feature array (cell x latent dim)
        expression_data: copy number array (cell x gene or cell x bin)
        n_clones: number of clones
    Returns:
        pred_label: clone prediction
    """

    # fit GMM
    gmm = GaussianMixture(n_components=n_clones, n_init=5, max_iter=100000)
    gmm.fit(latent_data)
    pred_label = gmm.predict(latent_data)

    return pred_label


def find_clones_kmeans(data, n_clones=2):
    """ kmeans clustering method for finding clones """

    kmeans = KMeans(n_clusters=n_clones, random_state=0).fit(data)
    prediction = kmeans.labels_

    return prediction


def find_clones_dbscan(data, min_members=1):
    """ DBSCAN clustering """

    clustering = DBSCAN(min_samples=min_members).fit(data)
    prediction = clustering.labels_

    return prediction

def auto_corr(cluster_data):
    """
    Calculate the autocorrelation of cluster data.

    Args:
        cluster_data (np.ndarray): Data array with shape (n_samples, n_features).

    Returns:
        autocorrelation: Autocorrelation value.
    """

    n_samples = cluster_data.shape[0]
    res = 0
    count = 0

    for i in range(n_samples):
        for j in range(i, n_samples):
            res += np.sum(cluster_data[i,:] * cluster_data[j,:])
            count += 1
    res = res / count
    autocorrelation = res / np.var(cluster_data,axis=1).mean()

    return autocorrelation


def find_normal_cluster(x_bin, pred_label, n_clusters=2):
    """
    Calculate autocorrelation for clusters and identify the normal cell cluster.

    Args:
        x_bin (np.ndarray): Data array with shape (n_samples, n_features).
        pred_label (np.ndarray): Cluster labels for each sample.
        n_clusters (int): Number of clusters.

    Returns:
        np.ndarray: Array of cluster autocorrelations.
        int: Index of the normal cell cluster.
    """

    cluster_auto_corr = np.zeros([n_clusters])

    for i in range(n_clusters):
        cluster_mean = x_bin[pred_label == i, :].mean(axis=0)
        cluster_x = x_bin[pred_label == i, :]

        cluster_auto_corr[i] = auto_corr(cluster_x - cluster_mean)

    normal_index = np.argmax(cluster_auto_corr)

    return cluster_auto_corr, normal_index