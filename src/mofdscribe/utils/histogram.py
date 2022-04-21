# -*- coding: utf-8 -*-
import math

import numpy as np
from scipy.stats import gaussian_kde


def get_rdf(array, lower_lim, upper_lim, bin_size, num_sites, volume, normalized: bool = True):
    dist_hist, dist_bins = np.histogram(
        array,
        bins=np.arange(lower_lim, upper_lim + bin_size, bin_size),
        density=False,
    )

    if normalized:
        shell_vol = 4.0 / 3.0 * math.pi * (np.power(dist_bins[1:], 3) - np.power(dist_bins[:-1], 3))
        number_density = num_sites / volume
        rdf = dist_hist / shell_vol / number_density
        return rdf

    return dist_hist


def smear_histogram(histogram, bw, lower_lim, upper_lim):
    kernel = gaussian_kde(histogram, bw_method=bw)
    x = np.linspace(lower_lim, upper_lim, len(histogram))
    y = kernel(x)
    return y
