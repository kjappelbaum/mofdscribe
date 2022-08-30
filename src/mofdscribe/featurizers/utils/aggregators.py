# -*- coding: utf-8 -*-
"""Functions that can be used to aggregate values/lists of values"""

import numpy as np
from scipy.stats import gmean, hmean
from scipy.stats.mstats import gmean as mstast_gmean
from scipy.stats.mstats import hmean as mstast_hmean


def ma_percentile(values, percentile):
    mdata = np.ma.filled(values.astype(float), np.nan)
    return np.nanpercentile(mdata, percentile)


def trimean(values: np.typing.ArrayLike) -> float:
    r"""Calculate the trimean of the values:

    .. math::

        TM={\frac {Q_{1}+2Q_{2}+Q_{3}}{4}}

    Args:
        values (np.typing.ArrayLike): values to compute the trimean for

    Returns:
        float: trimean
    """
    q1 = np.percentile(values, 25)
    q2 = np.percentile(values, 50)
    q3 = np.percentile(values, 75)
    return (q1 + 2 * q2 + q3) / 4


def masked_mad(values):
    diff = values - np.ma.median(values)
    return np.ma.median(np.ma.abs(diff))


def mad(values):
    diff = values - np.median(values)
    return np.median(np.abs(diff))


def masked_trimean(values):
    q1 = ma_percentile(values, 25)
    q2 = ma_percentile(values, 50)
    q3 = ma_percentile(values, 75)
    return (q1 + 2 * q2 + q3) / 4


AGGREGATORS = {
    "sum": lambda x: x[0] + x[1],
    "avg": lambda x: (x[0] + x[1]) / 2,
    "max": lambda x: max(x),
    "min": lambda x: min(x),
    "first": lambda x: x[0],
    "last": lambda x: x[1],
    "product": lambda x: x[0] * x[1],
    "diff": lambda x: abs(x[1] - x[0]),
}


ARRAY_AGGREGATORS = {
    "sum": lambda x, **kwargs: np.sum(x, **kwargs),
    "avg": lambda x, **kwargs: np.mean(x, **kwargs),
    "max": lambda x, **kwargs: np.max(x, **kwargs),
    "min": lambda x, **kwargs: np.min(x, **kwargs),
    "std": lambda x, **kwargs: np.std(x, **kwargs),
    "range": lambda x, **kwargs: np.max(x, **kwargs) - np.min(x, **kwargs),
    "mean": lambda x, **kwargs: np.mean(x, **kwargs),
    "median": lambda x, **kwargs: np.median(x, **kwargs),
    "geomean": lambda x, **kwargs: gmean(x, **kwargs),
    "harmean": lambda x, **kwargs: hmean(x, **kwargs),
    "mad": lambda x, **kwargs: mad(x, **kwargs),
    "trimean": lambda x, **kwargs: trimean(x, **kwargs),
    "inf": lambda x, **kwargs: np.linalg.norm(x, ord=np.inf, **kwargs),
    "manhattan": lambda x, **kwargs: np.linalg.norm(x, ord=1, **kwargs),
}


MA_ARRAY_AGGREGATORS = {
    "sum": lambda x, **kwargs: np.ma.sum(x, **kwargs),
    "avg": lambda x, **kwargs: np.ma.mean(x, **kwargs),
    "max": lambda x, **kwargs: np.ma.max(x, **kwargs),
    "min": lambda x, **kwargs: np.ma.min(x, **kwargs),
    "std": lambda x, **kwargs: np.ma.std(x, **kwargs),
    "range": lambda x, **kwargs: np.ma.max(x, **kwargs) - np.ma.min(x, **kwargs),
    "mean": lambda x, **kwargs: np.ma.mean(x, **kwargs),
    "median": lambda x, **kwargs: np.ma.median(x, **kwargs),
    "geomean": lambda x, **kwargs: mstast_gmean(x, **kwargs),
    "harmean": lambda x, **kwargs: mstast_hmean(x, **kwargs),
    "mad": lambda x, **kwargs: masked_mad(x, **kwargs),
    "trimean": lambda x, **kwargs: masked_trimean(x, **kwargs),
    "inf": lambda x, **kwargs: np.linalg.norm(x, ord=np.inf, **kwargs),
    "manhattan": lambda x, **kwargs: np.linalg.norm(x, ord=1, **kwargs),
}
