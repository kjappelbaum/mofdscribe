# -*- coding: utf-8 -*-
"""
Cached graphs and compression.

Compression code taken from https://www.kaggle.com/code/nickycan/compress-70-of-dataset.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core import IStructure
from structuregraph_helpers.create import get_structure_graph
from structuregraph_helpers.hash import (
    decorated_graph_hash,
    decorated_scaffold_hash,
    undecorated_graph_hash,
    undecorated_scaffold_hash,
)


@lru_cache(maxsize=None)
def get_structure_graph_cached(structure: IStructure) -> StructureGraph:
    return get_structure_graph(structure)


@lru_cache(maxsize=None)
def get_decorated_graph_hash_cached(structure: IStructure) -> str:
    return decorated_graph_hash(get_structure_graph_cached(structure))


@lru_cache(maxsize=None)
def get_undecorated_graph_hash_cached(structure: IStructure) -> str:
    return undecorated_graph_hash(get_structure_graph_cached(structure))


@lru_cache(maxsize=None)
def get_decorated_scaffold_hash_cached(structure: IStructure) -> str:
    return decorated_scaffold_hash(get_structure_graph_cached(structure))


@lru_cache(maxsize=None)
def get_undecorated_scaffold_hash_cached(structure: IStructure) -> str:
    return undecorated_scaffold_hash(get_structure_graph_cached(structure))


_INT8_MIN = np.iinfo(np.int8).min
_INT8_MAX = np.iinfo(np.int8).max
_INT16_MIN = np.iinfo(np.int16).min
_INT16_MAX = np.iinfo(np.int16).max
_INT32_MIN = np.iinfo(np.int32).min
_INT32_MAX = np.iinfo(np.int32).max

_FLOAT16_MIN = np.finfo(np.float16).min
_FLOAT16_MAX = np.finfo(np.float16).max
_FLOAT32_MIN = np.finfo(np.float32).min
_FLOAT32_MAX = np.finfo(np.float32).max


def compress_dataset(data: pd.DataFrame) -> None:
    """Compress the dataset.

    Args:
        data (pd.DataFrame): The dataset to compress.
    """
    for col in data.columns:
        col_dtype = data[col][:100].dtype

        if col_dtype != "object":
            col_series = data[col]
            col_min = col_series.min()
            col_max = col_series.max()

            if col_dtype == "float64":
                if (col_min > _FLOAT16_MIN) and (col_max < _FLOAT16_MAX):
                    data[col] = data[col].astype(np.float16)

                elif (col_min > _FLOAT32_MIN) and (col_max < _FLOAT32_MAX):
                    data[col] = data[col].astype(np.float32)

                else:
                    pass

            if col_dtype == "int64":
                if (col_min > _INT8_MIN / 2) and (col_max < _INT8_MAX / 2):
                    data[col] = data[col].astype(np.int8)
                elif (col_min > _INT16_MIN) and (col_max < _INT16_MAX):
                    data[col] = data[col].astype(np.int16)
                elif (col_min > _INT32_MIN) and (col_max < _INT32_MAX):
                    data[col] = data[col].astype(np.int32)
                else:
                    pass
