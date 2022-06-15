# -*- coding: utf-8 -*-
"""Functions that can be used to aggregate values/lists of values"""
import numpy as np

AGGREGATORS = {
    'sum': lambda x: x[0] + x[1],
    'avg': lambda x: (x[0] + x[1]) / 2,
    'max': lambda x: max(x),
    'min': lambda x: min(x),
    'first': lambda x: x[0],
    'last': lambda x: x[1],
    'product': lambda x: x[0] * x[1],
    'diff': lambda x: abs(x[1] - x[0]),
}


ARRAY_AGGREGATORS = {
    'sum': lambda x, **kwargs: np.sum(x, **kwargs),
    'avg': lambda x, **kwargs: np.mean(x, **kwargs),
    'max': lambda x, **kwargs: np.max(x, **kwargs),
    'min': lambda x, **kwargs: np.min(x, **kwargs),
    'std': lambda x, **kwargs: np.std(x, **kwargs),
    'range': lambda x, **kwargs: np.max(x, **kwargs) - np.min(x, **kwargs),
    'mean': lambda x, **kwargs: np.mean(x, **kwargs),
    'median': lambda x, **kwargs: np.median(x, **kwargs),
}


MA_ARRAY_AGGREGATORS = {
    'sum': lambda x, **kwargs: np.ma.sum(x, **kwargs),
    'avg': lambda x, **kwargs: np.ma.mean(x, **kwargs),
    'max': lambda x, **kwargs: np.ma.max(x, **kwargs),
    'min': lambda x, **kwargs: np.ma.min(x, **kwargs),
    'std': lambda x, **kwargs: np.ma.std(x, **kwargs),
    'range': lambda x, **kwargs: np.ma.max(x, **kwargs) - np.ma.min(x, **kwargs),
    'mean': lambda x, **kwargs: np.ma.mean(x, **kwargs),
    'median': lambda x, **kwargs: np.ma.median(x, **kwargs),
}
