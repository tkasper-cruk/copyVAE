#! /usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from copyvae.binning import CHR_BASE_PAIRS


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


def test_dcis_inference(clone_profile, abs_position, gt_profile):
    """ Test DCIS1 data inference

    Args:
        clone_profile: inferred clone profile
        abs_position: absolute gene position in the genome
        gt_profile: ground truth clone profile
    Returns:
        distance: Euclidean distance
    """

    # process groud truth
    gt = pd.read_csv(gt_profile, sep='\t')
    gt['abspos'] = gt['abspos'].astype('int')

    #gt['copy'] = round(2**gt['med.DNA']*2)
    gt['copy'] = 2**gt['med.DNA'] * 2

    # estimate copy of positions in ground truth
    clone = {'abspos': abs_position, 'cp_inf': clone_profile}
    df = pd.DataFrame(data=clone)
    compdf = pd.merge_asof(
        gt.sort_values('abspos'),
        df.sort_values('abspos'),
        on='abspos')

    distance = np.linalg.norm(compdf['copy'].values - compdf['cp_inf'].values)

    compdf.loc[compdf['copy'] > 6, 'copy'] = 6.0
    gtcp = compdf['copy'].values
    infcp = compdf['cp_inf'].values
    plt.figure(figsize=(17, 5), dpi=120)
    plt.plot(infcp)
    plt.plot(gtcp)
    plt.savefig('dcis1.png')

    return distance


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help="clone profile")
    parser.add_argument('-a', '--abs', help="absolute position")
    parser.add_argument('-gt', '--gt', help="ground truth")

    args = parser.parse_args()
    clone_profile = np.load(args.input)
    abs_pos = np.load(args.abs)
    gt_profile = args.gt

    dis = test_dcis_inference(clone_profile, abs_pos, gt_profile)
    print('Euclidean:')
    print(dis)


if __name__ == "__main__":
    main()
