#!/usr/bin/env python3

import numpy as np
import scipy.special as sc
from scipy.stats import poisson
from copyvae.graphics import draw_heatmap, plot_breakpoints

def compute_ll_matrix(x_array):
    """ Compute log-likelihood for iid Poisson random variables """

    ll_list = []
    for lam in range(1,7):
        likelihood = poisson.logpmf(x_array, lam)
        ll = np.sum(likelihood, axis=0)
        ll_list.append(ll)

    return np.array(ll_list)


def value_func(x_array):
    """ Cost function: -2 x LL """

    max_cp_ll = x_array.sum(axis=1).max()
    cost = -2 * max_cp_ll

    return cost


def log_likelihood(x_array, lam):
    """ Compute log-likelihood for Poisson random variables"""

    n = len(x_array)
    log_gamma = sc.gammaln(x_array + 1)
    ll = - n * lam + np.log(lam) * np.sum(x_array) - np.sum(log_gamma)

    return ll


def log_likelihood_multi(x_array, lam):
    """ Compute log-likelihood for Poisson random variables (2D array) """

    n = np.shape(x_array)[1]
    log_gamma = sc.gammaln(x_array + 1)
    ll = - n * lam + np.log(lam) * np.sum(x_array, axis = 1) \
                                    - np.sum(log_gamma, axis = 1)
    return np.sum(ll)


def cost_func(x_array, multi = False):
    """ Loss Function: -2 x LL """

    cost = 0
    if multi:
        for lam in range(1,7):
            cost = cost - 2 * log_likelihood_multi(x_array, lam)
        #cost = np.sum(cost)

    else:
        for lam in range(1,7):
            cost = cost - 2 * log_likelihood(x_array, lam)

    return cost


def opt_partition(x_array, beta):
    """ Optimal Partitioning

    Args:
        x_array: observations
        beta: penalty constant

    Returns:
        a list of breakpoints
    """

    f = - beta
    f_dict = {0:f}
    breakpoints = []
    n = len(x_array)

    for j in range(1, n+1):
        local_f = {}

        for i in range(j):
            v = f_dict[i] + cost_func(x_array[i+1:j]) + beta
            local_f[i] = v
        f_dict[j] = min(local_f.values())
        bp = min(local_f, key=local_f.get)
        breakpoints.append(bp)
    breakpoints = sorted(list(set(breakpoints)))

    return np.array(breakpoints)



def pelt_algorithm(x_array, beta, k=0):
    """ PELT Method

    Segmentation method by Killick et al

    Args:
        x_array: observations
        beta: penalty constant
        k: constant

    Returns:
        a list of breakpoints
    """

    f = - beta
    f_dict = {0:f}
    breakpoints = []
    n = len(x_array)
    r_set = [0]

    for j in range(1, n+1):
        local_f = {}

        for i in r_set:
            v = f_dict[i] + cost_func(x_array[i+1:j]) + beta
            local_f[i] = v

        f_dict[j] = min(local_f.values())
        bp = min(local_f, key=local_f.get)
        breakpoints.append(bp)

        ## update R{}
        r_set.append(j)
        new_r_set = []
        for t in r_set:
            left = f_dict[t] + cost_func(x_array[t+1:j]) + k
            right = f_dict[j]
            if left <= right:
                new_r_set.append(t)
        r_set = new_r_set

    breakpoints = sorted(list(set(breakpoints)))
    return np.array(breakpoints)


def pelt_multi(x_array, beta, k=0):
    """ PELT for multiple samples

    Args:
        x_array: observations (sample x position)
        beta: penalty constant
        k: constant

    Returns:
        a list of breakpoints
    """

    f = - beta
    f_dict = {0:f}
    breakpoints = []
    n = np.shape(x_array)[1]
    r_set = [0]
    ll_array = compute_ll_matrix(x_array)

    for j in range(1, n+1):
        local_f = {}

        for i in r_set:
            v = f_dict[i] + value_func(ll_array[:,i+1:j]) + beta
            local_f[i] = v

        f_dict[j] = min(local_f.values())
        bp = min(local_f, key=local_f.get)
        breakpoints.append(bp)

        ## update R{}
        r_set.append(j)
        new_r_set = []
        for t in r_set:
            left = f_dict[t] + value_func(ll_array[:,t+1:j]) + k
            right = f_dict[j]
            if left <= right:
                new_r_set.append(t)
        r_set = new_r_set

    breakpoints = sorted(list(set(breakpoints)))
    return np.array(breakpoints)


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



from copyvae.clustering import find_clones

bincp_array = np.load('median_cp.npy')
d= np.load('latent.npy')

pred = find_clones(d)
mask = (1-pred).astype(bool)
cells = bincp_array[mask]

#resli = []
#for row in bincp_array:
#    cell_array = row
#    result = merge_segments(cell_array, .1)
#    resli.append(result)
#res = np.stack(resli, axis=0)
#draw_heatmap(res,"segment.1")

chroms = [(0, 69), (69, 115), (115, 154), (154, 179), (179, 210),
            (210, 243), (243, 275), (275, 299), (299, 325), (325, 350),
            (350, 388), (388, 423), (423, 435), (435, 457), (457, 477),
            (477, 507), (507, 547), (547, 557), (557, 607), (607, 626),
            (626, 633), (633, 649), (649, 672)
            ]

k = np.shape(cells)[0]
n = np.shape(cells)[1]
beta = 0.5 * k * np.log(n)
bp_arr = np.array([])

for tup in chroms:
    start_bin = tup[0]
    end_bin = tup[1]
    chrom_bps = pelt_multi(cells[:, start_bin:end_bin], beta)
    bps = chrom_bps + start_bin
    bp_arr = np.concatenate((bp_arr, bps))


print(bp_arr)
cp_arr = np.mean(cells, axis=0)
plot_breakpoints(cp_arr, bp_arr, 'bp_plot')
