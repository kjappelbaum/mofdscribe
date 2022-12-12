# -*- coding: utf-8 -*-
"""Helpers for computing histograms."""
import math
from typing import Optional

import numpy as np
from scipy.stats import gaussian_kde


def get_rdf(
    array: np.array,
    lower_lim: float,
    upper_lim: float,
    bin_size: float,
    num_sites: Optional[int] = None,
    volume: Optional[float] = None,
    normalized: bool = True,
    density: bool = False,
) -> np.array:
    """Compute the RDF

    Args:
        array (np.array): distances
        lower_lim (float): lower limit of the RDF
        upper_lim (float): upper limit of the RDF
        bin_size (float): size of the bins
        num_sites (int, optional): number of sites in the cell, used for normalization
        volume (float, optional): volume of the cell, used for normalization
        normalized (bool): If True, normalize the RDF. Defaults to True.
        density (bool): If True, return the density of the RDF. Defaults to False.

    Returns:
        np.array: RDF
    """
    dist_hist, dist_bins = np.histogram(
        array,
        bins=np.arange(lower_lim, upper_lim + bin_size, bin_size),
        density=density,
    )

    if normalized:
        shell_vol = 4.0 / 3.0 * math.pi * (np.power(dist_bins[1:], 3) - np.power(dist_bins[:-1], 3))
        number_density = num_sites / volume
        rdf = dist_hist / shell_vol / number_density
        return rdf

    return dist_hist.astype(np.float64)


def smear_histogram(histogram: np.array, bw: float, lower_lim: float, upper_lim: float) -> np.array:
    """Use a gaussian kernel to smooth the histogram.

    Args:
        histogram (np.array): histogram to be smoothed
        bw (float): bandwidth of the gaussian kernel
        lower_lim (float): lower limit of the RDF
        upper_lim (float): upper limit of the RDF

    Returns:
        np.array: smoothed histogram
    """
    kernel = gaussian_kde(histogram, bw_method=bw)
    x = np.linspace(lower_lim, upper_lim, len(histogram))
    y = kernel(x)
    return y
