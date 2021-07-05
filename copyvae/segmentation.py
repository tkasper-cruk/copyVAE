#!/usr/bin/env python3

import numpy as np
import scipy.special as sc
from copyvae.graphics import draw_heatmap


def log_likelihood(x_array, lam):

    n = len(x_array)
    log_gamma = sc.gammaln(x_array + 1)
    ll = - n * lam + np.log(lam) * np.sum(x_array) - np.sum(log_gamma)

    return ll


def cost_func(x_array):
    """ Loss Function: -2 x LL """
    cost = 0
    for lam in range(1,7):
        cost = cost + -2 * log_likelihood(x_array, lam)

    return cost


def opt_partition(bincp_array, beta):
    """ Optimal Partitioning """

    f = - beta
    f_dict = {0:f}
    breakpoints = []
    n = len(bincp_array)

    for j in range(1, n+1):
        local_f = {}

        for i in range(j):
            v = f_dict[i] + cost_func(bincp_array[i+1:j]) + beta
            local_f[i] = v
        f_dict[j] = min(local_f.values())
        bp = min(local_f, key=local_f.get)
        breakpoints.append(bp)
    breakpoints = sorted(list(set(breakpoints)))

    return breakpoints



def pelt_algorithm(bincp_array, beta, k=0):
    """ PELT Method by Killick et al"""

    f = - beta
    f_dict = {0:f}
    breakpoints = []
    n = len(bincp_array)
    r_set = [0]

    for j in range(1, n+1):
        local_f = {}

        for i in r_set:
            v = f_dict[i] + cost_func(bincp_array[i+1:j]) + beta
            local_f[i] = v

        f_dict[j] = min(local_f.values())
        bp = min(local_f, key=local_f.get)
        breakpoints.append(bp)

        ## update R{}
        r_set.append(j)
        new_r_set = []
        for t in r_set:
            left = f_dict[t] + cost_func(bincp_array[t+1:j]) + k
            right = f_dict[j]
            if left <= right:
                new_r_set.append(t)
        r_set = new_r_set

    breakpoints = sorted(list(set(breakpoints)))
    return breakpoints


def incremental_mean(x, mu_pre, n):

    mu = (x + (n - 1) * mu_pre)/n
    return mu


def incremental_var(x, mu_pre, var_pre, n):

    var = (n - 2) / (n - 1) * var_pre + (x - mu_pre)**2 / n
    return var


def variance_loss(x, mu_pre, var_pre, n):

    loss = - var_pre / (n - 1) + (x - mu_pre)**2 / n
    return loss


def merge_bins(bincp_array):

    seg_array = []
    seg_length = 1
    for i in range(1, len(bincp_array)):
        if bincp_array[i] == bincp_array[i-1]:
            seg_length = seg_length + 1
        else:
            seg_array.append((seg_length, bincp_array[i-1]))
            seg_length = 1
    seg_array.append((seg_length, bincp_array[i]))

    return seg_array


def merge_segments(bincp_array, break_loss=.01):

    #bp_array = [0]
    segcp_array = np.array([])

    mean_pre = -1
    var_pre = 0.0
    seg_list = merge_bins(bincp_array)

    for i in range(len(seg_list)):
        bin_n, bin_cp = seg_list[i]

        # first bins in the segment
        if mean_pre == -1:
            mean_pre = bin_cp
            n = bin_n
            continue
        mean = mean_pre
        var = var_pre
        # compute mean and variance for per mini seg
        for j in range(bin_n):
            var = incremental_var(bin_cp, mean, var, n+j+1)
            mean = incremental_mean(bin_cp, mean, n+j+1)
        var_loss = var - var_pre

        # close the current segment
        if var_loss > break_loss:
            seg_len = n + bin_n
            segcp_array = np.concatenate((segcp_array, [mean] * seg_len))
            n = 0
            mean_pre = -1
            var_pre = 0.0
        else:
            mean_pre = mean
            var_pre = var
            n = n + bin_n
    segcp_array = np.concatenate((segcp_array, [mean] * n))

    return segcp_array



import time

bincp_array = np.load("../data/median_cp.npy")
"""
resli = []

for row in bincp_array:
    cell_array = row
    result = merge_segments(cell_array, .1)
    resli.append(result)

res = np.stack(resli, axis=0)
draw_heatmap(res,"segment.1")
"""
cell = bincp_array[700]
t0 = time.time()
bps0 = opt_partition(cell, beta=25)
t1 = time.time()
bps1 = pelt_algorithm(cell, beta=25)
t2 = time.time()
print(t1-t0)
print(t2-t1)
print(bps0)
print(bps1)
