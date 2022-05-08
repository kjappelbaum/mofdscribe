# -*- coding: utf-8 -*-
import numpy as np

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
    "sum": lambda x: sum(x),
    "avg": lambda x: sum(x) / len(x),
    "max": lambda x: max(x),
    "min": lambda x: min(x),
    "std": lambda x: np.std(x),
    "range": lambda x: max(x) - min(x),
    "mean": lambda x: np.mean(x),
    "median": lambda x: np.median(x),
}


MA_ARRAY_AGGREGATORS = {
    "sum": lambda x: np.ma.sum(x),
    "max": lambda x: np.ma.max(x),
    "min": lambda x: np.ma.min(x),
    "std": lambda x: np.ma.std(x),
    "range": lambda x: np.ma.max(x) - np.ma.min(x),
    "mean": lambda x: np.ma.mean(x),
    "avg": lambda x: np.ma.mean(x), 
    "median": lambda x: np.ma.median(x),
}
