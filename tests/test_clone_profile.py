#! /usr/bin/env python3

import pandas as pd
import numpy as np

CHR_BASE_PAIRS = [
    248956422,
    242193529,
    198295559,
    190214555,
    181538259,
    170805979,
    159345973,
    145138636,
    138394717,
    133797422,
    135086622,
    133275309,
    114364328,
    107043718,
    101991189,
    90338345,
    83257441,
    80373285,
    58617616,
    64444167,
    46709983,
    50818468,
    156040895]


def construct_position_df(base_per_region=180000):
    """ Generate position dataframe for given resolution

    Args:
        base_per_region: genome resolution
    Returns:
        dataframe containing chromosome name, position in chromosome,
        and absolute position
    """

    chr_arr = np.array([])
    abs_arr = np.array([])
    name_arr = np.array([])
    chr_pos = np.array(CHR_BASE_PAIRS)

    for i in range(len(chr_pos)):
        chrpos = np.arange(0, chr_pos[i], base_per_region)
        abspos = chrpos + chr_pos[:i].sum()
        chr_name = np.ones_like(chrpos) + i
        chr_arr = np.concatenate((chr_arr, chrpos), axis=0)
        abs_arr = np.concatenate((abs_arr, abspos), axis=0)
        name_arr = np.concatenate((name_arr, chr_name), axis=0)

    narr = np.stack((name_arr, chr_arr, abs_arr), axis=-1).astype(int)
    df = pd.DataFrame(data=narr, columns=['chrom', 'chrompos', 'abspos'])
    #df = df.iloc[5:].reset_index(drop=True)
    return df
