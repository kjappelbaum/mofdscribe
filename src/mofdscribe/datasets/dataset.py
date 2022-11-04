# -*- coding: utf-8 -*-
"""Base class for datasets."""
from typing import Iterable, List, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from loguru import logger
from pymatgen.core import IStructure, Structure

from mofdscribe.datasets.utils import (
    get_decorated_graph_hash_cached,
    get_decorated_scaffold_hash_cached,
    get_undecorated_graph_hash_cached,
)

__all__ = ["AbstractStructureDataset"]


class AbstractStructureDataset:
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

    def get_subset(self, indices: Iterable[int]) -> "AbstractStructureDataset":
        raise NotImplementedError()

    @property
    def available_info(self) -> List[str]:
        raise NotImplementedError()

    @property
    def available_features(self) -> List[str]:
        raise NotImplementedError()

    @property
    def available_labels(self) -> List[str]:
        raise NotImplementedError()

    def __len__(self):
        """Return the number of structures."""
        return len(self._structures)

    def __iter__(self):
        """Iterate over the structures."""
        return self.get_structures(range(len(self)))

    def get_labels(
        self, idx: Iterable[int], labelnames: Optional[Iterable[str]] = None
    ) -> np.ndarray:
        raise NotImplementedError()

    def get_years(self, idx: int) -> int:
        if self._years is None:
            raise ValueError("Years are not available.")
        return self._years[idx]

    # ToDo: think about how we can cache this in memory
    def get_structures(self, idx: Iterable[int]) -> Iterable[Structure]:
        return Parallel(n_jobs=-1)(delayed(IStructure.from_file)(self._structures[i]) for i in idx)

    def get_filenames(self, idx: Iterable[int]) -> List[Structure]:
        return [self._structures[i] for i in idx]

    # ToDo: parallelize hash computation
    def get_decorated_graph_hashes(self, idx: Iterable[int]) -> str:
        if self._decorated_graph_hashes is None:
            logger.info("Computing hashes, this can take a while.")
            # fixme: this is wrong, we need to compute it for all and not only for the indexed ones
            hashes = np.array(
                Parallel(n_jobs=-1)(
                    delayed(get_decorated_graph_hash_cached)(s)
                    for s in self.get_structures(range(len(self)))
                )
            )
            self._decorated_graph_hashes = hashes
        return self._decorated_graph_hashes[idx]

    def get_undecorated_graph_hashes(self, idx: Iterable[int]) -> str:
        if self._undecorated_graph_hashes is None:
            logger.info("Computing hashes, this can take a while.")
            hashes = np.array(
                Parallel(n_jobs=-1)(
                    delayed(get_undecorated_graph_hash_cached)(s)
                    for s in self.get_structures(range(len(self)))
                )
            )
            self._undecorated_graph_hashes = hashes
        return self._undecorated_graph_hashes

    def get_decorated_scaffold_hashes(self, idx: Iterable[int]) -> str:
        if self._decorated_graph_hashes is None:
            logger.info("Computing hashes, this can take a while.")
            hashes = np.array(
                Parallel(n_jobs=-1)(
                    delayed(get_decorated_scaffold_hash_cached)(s)
                    for s in self.get_structures(range(len(self)))
                )
            )
            self._decorated_scaffold_hashes = hashes

        return self._decorated_scaffold_hashes[idx]

    def get_undecorated_scaffold_hashes(self, idx: Iterable[int]) -> str:
        if self._undecorated_scaffold_hashes is None:
            logger.info("Computing hashes, this can take a while.")
            hashes = np.array(
                Parallel(n_jobs=-1)(
                    delayed(get_undecorated_graph_hash_cached)(s)
                    for s in self.get_structures(range(len(self)))
                )
            )
            self._undecorated_scaffold_hashes = hashes
        return self._undecorated_scaffold_hashes[idx]

    def get_densities(self, idx: Iterable[int]) -> np.ndarray:
        if self._densities is None:

            def get_denstiy(s):
                return s.density

            self._densities = np.array(
                Parallel(n_jobs=-1)(
                    delayed(get_denstiy)(s) for s in self.get_structures(range(len(self)))
                )
            )
        return self._densities[idx]

    # ToDo: think how this should behave.
    def select(self, indices: Iterable[int], labels: Optional[Iterable[str]] = None):
        return indices

    def show_structure(self, index):
        import nglview as nv

        structure = list(self.get_structures(index))[0]
        return nv.show_pymatgen(structure)

    @property
    def citations(self) -> Tuple[str]:
        raise NotImplementedError()
