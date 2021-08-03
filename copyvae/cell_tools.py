#! /usr/bin/env python3

import numpy as np
from scipy import stats
from dataclasses import dataclass


@dataclass
class Clone:
    """ Class for cell clones """

    id: int
    clone_size: int
    bin_size: int
    cell_gene_cn: np.ndarray = np.array([])
    cell_bin_cn: np.ndarray = np.array([])
    breakpoints: np.ndarray = np.array([])
    segment_cn: np.ndarray = np.array([])

    def generate_profile(self):
        """ Generate clone profile """

        seg_array = np.zeros_like(self.cell_bin_cn)

        for cell in range(self.clone_size):
            for i in range(len(self.breakpoints) - 1):
                start_p = self.breakpoints[i]
                end_p = self.breakpoints[i + 1]
                seg_array[cell, start_p:end_p] = stats.mode(
                    self.cell_gene_cn[cell,
                                    start_p * self.bin_size:end_p * self.bin_size]
                                    ).mode
            seg_array[cell, end_p:] = stats.mode(
                            self.cell_gene_cn[cell, end_p * self.bin_size:]
                            ).mode
        self.segment_cn = np.mean(seg_array, axis=0)
