#! /usr/bin/env python3

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture


def find_clones_gmm(latent_data, expression_data, n_clones=2):
    """ GMM find tumour clone

    Args:
        latent_data: latent feature array (cell x latent dim)
        expression_data: copy number array (cell x gene or cell x bin)
        n_clones: number of clones
    Returns:
        mask: mask for tumour cells
    """

    # fit GMM
    gmm = GaussianMixture(n_components=n_clones, n_init=5, max_iter=100000)
    gmm.fit(latent_data)
    pred_label = gmm.predict(latent_data)

    # define tumour clones
    mask = (pred_label).astype(bool)
    clone_masked = expression_data[mask]
    clone_unmasked = expression_data[~mask]
    if clone_masked.mean(axis=1).std() > clone_unmasked.mean(axis=1).std():
        return mask
    else:
        mask = (1 - pred_label).astype(bool)
        return mask


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
