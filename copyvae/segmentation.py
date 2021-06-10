#!/usr/bin/env python3

import numpy as np
from copyvae.graphics import draw_heatmap

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



bincp_array = np.load("median_cp.npy")
resli = []

for row in bincp_array:
    cell_array = row
    result = merge_segments(cell_array, .005)
    resli.append(result)

res = np.stack(resli, axis=0)
draw_heatmap(res,"segment.005")
