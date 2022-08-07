"""Base featurizer for MOF structure based featurizers."""
from multiprocessing import Pool
from typing import Union

import numpy as np
from loguru import logger
from matminer.featurizers.base import BaseFeaturizer, MultipleFeaturizer
from pymatgen.core import IStructure, Structure


class MOFBaseFeaturizer(BaseFeaturizer):
    """Base featurizer for MOF structure based featurizers."""

    def __init__(self, primitive: bool = False) -> None:
        """
        Args:
            primitive (bool): If True, use the primitive cell of the structure.
                Defaults to False.
        """
        self.primitive = primitive

    def _get_primitive(self, structure: Union[Structure, IStructure]) -> Structure:
        return structure.get_primitive_structure()

    def featurize(self, structure: Union[Structure, IStructure]) -> np.ndarray:
        """Compute the descriptor for a given structure.

        Args:
            structure (Union[Structure, IStructure]): Structure to compute the descriptor for.

        Returns:
            A numpy array containing the descriptor.
        """
        if self.primitive:
            structure = self._get_primitive(structure)
        return self._featurize(structure)


class MOFBaseMultipleFeaturizer(MultipleFeaturizer):
    """Base multiplefeaturizer for MOF structure based featurizers.

    .. warning::

        This MultipleFeaturizer only works with featurizers that only
        accept single structures as input.

    .. warning::

        This MultipleFeaturizer cannot efficiently be used if you want
        to use the primitive for some featurizers and not for others.
        In this case, you're better off using two separate MultipleFeaturizers.
    """

    def __init__(self, featurizers, iterate_over_entries=True, primitive=True):
        self.featurizers = featurizers
        # unset the primitive on the individual featurizers
        for featurizer in self.featurizers:
            featurizer.primitive = False
        self.iterate_over_entries = iterate_over_entries
        self.primitive = primitive

    def _get_primitive(self, structure: Union[Structure, IStructure]) -> Structure:
        return structure.get_primitive_structure()

    def featurize(self, structure: Union[Structure, IStructure]) -> np.ndarray:
        if self.primitive:
            logger.debug("Getting primitive cell")
            structure = self._get_primitive(structure)
        return np.ndarray([feature for f in self.featurizers for feature in f.featurize(structure)])

    def _get_primitive_many(self, structures):
        if self.n_jobs == 1:
            return [self._get_primitive(s) for s in structures]
        else:
            with Pool(self.n_jobs, maxtasksperchild=1) as p:
                res = p.map(self._get_primitive, structures, chunksize=self.chunksize)
                return res

    def featurize_many(self, entries, ignore_errors=False, return_errors=False, pbar=True):
        if self.iterate_over_entries:
            return super().featurize_many(
                entries,
                ignore_errors=ignore_errors,
                return_errors=return_errors,
                pbar=pbar,
            )
        else:
            logger.debug("Precomputing primitive cells")
            entries = self._get_primitive_many(entries)
            logger.debug("Featurizing the primitive cells")
            features = [
                f.featurize_many(
                    entries,
                    ignore_errors=ignore_errors,
                    return_errors=return_errors,
                    pbar=pbar,
                )
                for f in self.featurizers
            ]
            return [sum(x, []) for x in zip(*features)]

    def featurize_wrapper(self, x, return_errors=False, ignore_errors=False):
        if self.iterate_over_entries:
            return [
                feature
                for f in self.featurizers
                for feature in f.featurize_wrapper(
                    x, return_errors=return_errors, ignore_errors=ignore_errors
                )
            ]
        else:
            return super().featurize_wrapper(
                x, return_errors=return_errors, ignore_errors=ignore_errors
            )
