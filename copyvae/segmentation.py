#! /usr/bin/env python3

import numpy as np
from scipy.stats import poisson, norm


def compute_ll_matrix(x_array, max_cp=25):
    """ Compute log-likelihood for iid random variables """

    ll_list = []
    for lam in range(1,max_cp):
        #likelihood = poisson.logpmf(x_array, lam)
        likelihood = norm.logpdf(x_array, loc=lam, scale=0.5)
        ll = np.sum(likelihood, axis=0)
        ll_list.append(ll)

    return np.array(ll_list)


def value_func(x_array):
    """ Cost function: -2 x LL """

    max_cp_ll = x_array.sum(axis=1).max()
    cost = -2 * max_cp_ll

    return cost


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


def generate_clone_profile(bin_cp, chrom_list, eta=1):
    """ clone copy number profile

    Args:
        bin_cp: observations (sample x position)
        chrom_list: a list of chromosome boundary
        eta: weight on beta in PELT alg

    Returns:
        segment_cn: clone copy numbers
        bp_arr: breakpoints
    """

    k = np.shape(bin_cp)[0]
    n = np.shape(bin_cp)[1]
    beta = eta * k * np.log(n)
    bp_arr = np.array([])

    for tup in chrom_list:
            start_bin = tup[0]
            end_bin = tup[1]
            chrom_bps = pelt_multi(
                                    bin_cp[:, start_bin:end_bin], 
                                    beta
                                    )
            bps = chrom_bps + start_bin
            bp_arr = np.concatenate((bp_arr, bps))
    bp_arr = bp_arr.astype(int)
    print("{} breakpoints are detected.".format(len(bp_arr)))
    seg_array = np.zeros_like(bin_cp)

    for cell in range(k):
            for i in range(len(bp_arr) - 1):
                start_p = bp_arr[i]
                end_p = bp_arr[i + 1]
                seg_array[cell, start_p:end_p] = np.median(
                    bin_cp[cell, start_p:end_p])
            seg_array[cell, end_p:] = np.median(
                            bin_cp[cell, end_p:]
                            )              
    segment_cn = np.mean(seg_array, axis=0)

    return segment_cn, bp_arr
