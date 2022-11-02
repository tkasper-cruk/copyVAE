#! /usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from scipy.spatial import distance
from scipy.stats import pearsonr
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
        distance1: Euclidean distance
        distance2: Cosine similarity
        distance3: Pearson correlation coefficient
        distance4: Manhattan distance
    """

    # process groud truth
    gt = pd.read_csv(gt_profile, sep='\t')
    gt['abspos'] = gt['abspos'].astype('int')

    gt['copy'] = gt['med.DNA']
    #gt['copy'] = 2**gt['med.DNA'] * 2

    # estimate copy of positions in ground truth
    clone = {'abspos': abs_position, 'cp_inf': clone_profile}
    df = pd.DataFrame(data=clone)
    compdf = pd.merge_asof(
        gt.sort_values('abspos'),
        df.sort_values('abspos'),
        on='abspos')

    #distance1 = np.linalg.norm(compdf['copy'].values - compdf['cp_inf'].values)
    distance1 = distance.euclidean(compdf['copy'].values, compdf['cp_inf'].values)
    distance2 = distance.cosine(compdf['copy'].values, compdf['cp_inf'].values)
    distance3 = pearsonr(compdf['copy'].values, compdf['cp_inf'].values)
    distance4 = distance.cityblock(compdf['copy'].values, compdf['cp_inf'].values)

    print('Euclidean distance: {}'.format(distance1))
    print('Cosine distance: {}'.format(distance2))
    print('Pearson correlation: {}'.format(distance3[0]))
    print('Manhattan distance: {}'.format(distance4))

    gtcp = compdf['copy'].values
    infcp = compdf['cp_inf'].values
    plt.figure(figsize=(17, 5), dpi=120)
    plt.plot(infcp,label='inference')
    plt.plot(gtcp, label='GT')
    plt.legend()
    plt.savefig('dcis1.png')

    return distance1, distance2, distance3, distance4


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help="clone profile")
    parser.add_argument('-a', '--abs', help="absolute position")
    parser.add_argument('-gt', '--gt', help="ground truth")

    args = parser.parse_args()
    clone_profile = np.load(args.input)
    abs_pos = np.load(args.abs)
    gt_profile = args.gt

    d1, d2, d3, d4 = test_dcis_inference(clone_profile, abs_pos, gt_profile)


if __name__ == "__main__":
    main()
