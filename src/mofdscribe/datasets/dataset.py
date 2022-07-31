# -*- coding: utf-8 -*-
"""Base class for datasets."""
from typing import Iterable, List, Optional, Tuple

import numpy as np
from loguru import logger
from mofchecker import MOFChecker
from pymatgen.core import IStructure, Structure

from .utils import (
    get_decorated_graph_hash_cached,
    get_decorated_scaffold_hash_cached,
    get_undecorated_graph_hash_cached,
    get_undecorated_scaffold_hash_cached,
)


class StructureDataset:
    """Base class for datasets."""

    def __init__(self):
        """Initialize a dataset."""
        self._structures = None
        self._target = None

        self._years = None
        self._labels = None
        self._decorated_graph_hashes = None
        self._undecorated_graph_hashes = None
        self._decorated_scaffold_hashes = None
        self._undecorated_scaffold_hashes = None
        self._densities = None

    @property
    def available_labels(self) -> Tuple[str]:
        raise NotImplementedError()

    def __len__(self):
        """Return the number of structures."""
        return len(self._structures)

    def get_labels(
        self, idx: Iterable[int], labelnames: Optional[Iterable[str]] = None
    ) -> np.ndarray:
        raise NotImplementedError()

    def get_years(self, idx: int) -> int:
        if self._years is None:
            raise ValueError("Years are not available.")
        return self._years.iloc[idx].values

    # ToDo: think about how we can cache this in memory
    def get_structures(self, idx: Iterable[int]) -> Iterable[Structure]:
        return (IStructure.from_file(self._structures[i]) for i in idx)

    def get_filenames(self, idx: Iterable[int]) -> List[Structure]:
        return [self._structures[i] for i in idx]

    # ToDo: parallelize hash computation
    def get_decorated_graph_hashes(self, idx: Iterable[int]) -> str:
        if self._decorated_graph_hashes is None:
            logger.info("Computing hashes, this can take a while.")
            hashes = [get_decorated_graph_hash_cached(self._structures[i]) for i in idx]
            return hashes
        return self._decorated_graph_hashes.iloc[idx]

    def get_undecorated_graph_hashes(self, idx: Iterable[int]) -> str:
        if self._undecorated_graph_hashes is None:
            logger.info("Computing hashes, this can take a while.")
            hashes = [get_undecorated_graph_hash_cached(self._structures[i]) for i in idx]
            return hashes
        return self._undecorated_graph_hashes.iloc[idx]

    def get_decorated_scaffold_hashes(self, idx: Iterable[int]) -> str:
        if self._decorated_graph_hashes is None:
            logger.info("Computing hashes, this can take a while.")
            hashes = [get_decorated_scaffold_hash_cached(self._structures[i]) for i in idx]
            return hashes
        return self._decorated_scaffold_hashes.iloc[idx]

    def get_undecorated_scaffold_hashes(self, idx: Iterable[int]) -> str:
        if self._undecorated_scaffold_hashes is None:
            logger.info("Computing hashes, this can take a while.")
            hashes = [get_undecorated_graph_hash_cached(self._structures[i]) for i in idx]
            return hashes
        return self._undecorated_scaffold_hashes.iloc[idx]

    def get_densities(self, idx: Iterable[int]) -> np.ndarray:
        if self._densities is None:
            return np.array([s.density for s in self.get_structures(idx)])
        return self._densities.iloc[idx]

    # ToDo: think how this should behave.
    def select(self, indices: Iterable[int], labels: Optional[Iterable[str]] = None):
        return indices

    @property
    def citations(self) -> Tuple[str]:
        raise NotImplementedError()
