#! /usr/bin/env python3

#import numpy as np
from sklearn.cluster import KMeans

def find_clones(data, n_clones=2):
    """ Clustering method for finding clones """

    kmeans = KMeans(n_clusters=n_clones, random_state=0).fit(data)
    prediction = kmeans.labels_

    return prediction


#d = np.load('./data/latent.npy')
#pred = find_clones(d)
#bincp_array = np.load('median_cp.npy')
#mask = (1-pred).astype(bool)
#cells = bincp_array[mask]
