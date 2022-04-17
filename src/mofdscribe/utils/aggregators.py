import math

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
}
