"""Base featurizer for MOF structure based featurizers.

The main purpose of these classes is currently that they handle
conversion of the structures to primtive cells.
This can have computational benefits and also make the use of some
aggregations such as "sum" more meaningful.

However, since they support this functionality, they are less flexible
than the "original" matminer :code:`BaseFeaturizer` and :code:`MultipleFeaturizer`.
In practice, this means that they only accept one pymatgen structure object
or and iterable of pymatgen structure objects.
"""
from abc import abstractmethod
from functools import partial
from multiprocessing import Pool
from typing import Union

import numpy as np
import pandas as pd
from loguru import logger
from matminer.featurizers.base import BaseFeaturizer, MultipleFeaturizer
from pymatgen.core import IMolecule, IStructure, Molecule, Structure
from tqdm.auto import tqdm


def get_primitive(structure: Union[IStructure, Structure, Molecule, IMolecule]) -> Structure:
    """Get the primitive cell of a structure.

    We use this wrapper because we want to have a passtrough
    for molecules, which some of our featurizers also accept.

    Args:
        structure: Structure object.

    Returns:
        Structure object.
    """
    if isinstance(structure, (Structure, IStructure)):
        return structure.get_primitive_structure()
    else:
        return structure


class MOFBaseFeaturizer(BaseFeaturizer):
    """Base featurizer for MOF structure based featurizers.

    .. note::

        If you implement a new :code:`MOFBaseFeaturizer`,
        you need to implement a :code:`_featurize` method.
        If you implement :code:`featurize` directly,
        you would override the conversion to primitive cell.

    .. warning::

        If you implement a new :code:`MOFBaseFeaturizer`,
        and your featurizer needs to be fitted, keep in mind
        to also call the :code:`_get_primitive` method.
    """

    def __init__(self, primitive: bool = False) -> None:
        """
        Construct a MOFBaseFeaturizer.

        Args:
            primitive (bool): If True, use the primitive cell of the structure.
                Defaults to False.
        """
        self.primitive = primitive

    def _get_primitive_many(self, structures):
        if self.n_jobs == 1:
            return [self._get_primitive(s) for s in structures]
        else:
            with Pool(self.n_jobs, maxtasksperchild=1) as p:
                res = p.map(self._get_primitive, structures, chunksize=self.chunksize)
                return res

    def _get_primitive(self, structure: Union[Structure, IStructure]) -> Structure:
        logger.debug("Getting primitive cell for structure in MOFBaseFeaturizer")
        return get_primitive(structure)

    @abstractmethod
    def _featurize(self, structure: Union[Structure, IStructure]) -> np.ndarray:
        raise NotImplementedError("_featurize must be implemented in a subclass")

    def fit(self, structures):
        if not isinstance(structures, (list, tuple)):
            structures = [structures]
        if self.primitive:
            structures = self._get_primitive_many(structures)
        self._fit(structures)

    def featurize(self, structure: Union[Structure, IStructure]) -> np.ndarray:
        """Compute the descriptor for a given structure.

        Args:
            structure (Union[Structure, IStructure]): Structure to compute the descriptor for.

        Returns:
            A numpy array containing the descriptor.
        """
        logger.debug("Featurizing structure in MOFBaseFeaturizer")
        if self.primitive:
            logger.debug("Getting primitive cell for structure in MOFBaseFeaturizer")
            structure = get_primitive(structure)
        return self._featurize(structure)

    def featurize_many(self, entries, ignore_errors=False, return_errors=False, pbar=True):
        """Featurize a list of entries.

        If `featurize` takes multiple inputs, supply inputs as a list of tuples.

        Featurize_many supports entries as a list, tuple, numpy array,
        Pandas Series, or Pandas DataFrame.

        Args:
            entries (list-like object): A list of entries to be featurized.
            ignore_errors (bool): Returns NaN for entries where exceptions are
                thrown if True. If False, exceptions are thrown as normal.
            return_errors (bool): If True, returns the feature list as
                determined by ignore_errors with traceback strings added
                as an extra 'feature'. Entries which featurize without
                exceptions have this extra feature set to NaN.
            pbar (bool): Show a progress bar for featurization if True.

        Returns:
            (list) features for each entry.

        Raises:
            Exception: If entries is not a list-like object.
            ValueError: If return_errors is set and ignore_errors is True.
        """
        if return_errors and not ignore_errors:
            raise ValueError("Please set ignore_errors to True to use" " return_errors.")

        # Check inputs
        if not isinstance(entries, (tuple, list, np.ndarray, pd.Series, pd.DataFrame)):
            raise Exception("'entries' must be a list-like object")

        # Special case: Empty list
        if len(entries) == 0:
            return []

        # If the featurize function only has a single arg, zip the inputs
        if isinstance(entries, pd.DataFrame):
            entries = entries.values
        elif isinstance(entries, pd.Series) or not isinstance(
            entries[0], (tuple, list, np.ndarray)
        ):
            entries = zip(entries)

        # Add a progress bar
        if pbar:
            # list() required, tqdm has issues with memory if generator given
            entries = tqdm(list(entries), desc=self.__class__.__name__)

        # Run the actual featurization
        if self.n_jobs == 1:
            return np.array(
                [
                    self.featurize_wrapper(
                        x, ignore_errors=ignore_errors, return_errors=return_errors
                    )
                    for x in entries
                ]
            )
        else:
            with Pool(self.n_jobs, maxtasksperchild=1) as p:
                func = partial(
                    self.featurize_wrapper,
                    return_errors=return_errors,
                    ignore_errors=ignore_errors,
                )
                res = p.map(func, entries, chunksize=self.chunksize)
                return np.array(res)


class MOFMultipleFeaturizer(MultipleFeaturizer):
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
        """
        Construct a MOFMultipleFeaturizer.

        Args:
            featurizers (list): List of featurizers to use.
            iterate_over_entries (bool): If True, featurize each entry in the list.
                If False, featurize each structure in the list. Defaults to True.
            primitive (bool): If True, use the primitive cell of the structure.
                Defaults to True.
        """
        # unset the primitive on the individual featurizers
        for featurizer in featurizers:
            featurizer.primitive = False

        self.featurizers = featurizers
        self.iterate_over_entries = iterate_over_entries
        self.primitive_multiple = primitive

    def _get_primitive(self, structure: Union[Structure, IStructure]) -> Structure:
        return get_primitive(structure)

    def featurize(self, structure: Union[Structure, IStructure]) -> np.ndarray:
        logger.debug(f"Featurizing structure in MOFMultipleFeaturizer, {self.primitive_multiple}")
        if self.primitive_multiple:
            logger.debug("Getting primitive cell")
            structure = self._get_primitive(structure)
        return np.array([feature for f in self.featurizers for feature in f.featurize(structure)])

    def _get_primitive_many(self, structures):
        if self.n_jobs == 1:
            return [self._get_primitive(s) for s in structures]
        else:
            with Pool(self.n_jobs, maxtasksperchild=1) as p:
                res = p.map(self._get_primitive, structures, chunksize=self.chunksize)
                return res

    def featurize_many(self, entries, ignore_errors=False, return_errors=False, pbar=True):
        logger.debug("Precomputing primitive cells")
        if self.primitive_multiple:
            entries = self._get_primitive_many(entries)
        if self.iterate_over_entries:
            return np.array(
                super().featurize_many(
                    entries,
                    ignore_errors=ignore_errors,
                    return_errors=return_errors,
                    pbar=pbar,
                )
            )
        else:
            features = [
                f.featurize_many(
                    entries,
                    ignore_errors=ignore_errors,
                    return_errors=return_errors,
                    pbar=pbar,
                )
                for f in self.featurizers
            ]

            return np.hstack(features)

    def featurize_wrapper(self, x, return_errors=False, ignore_errors=False):
        if self.iterate_over_entries:
            return np.array(
                [
                    feature
                    for f in self.featurizers
                    for feature in f.featurize_wrapper(
                        x, return_errors=return_errors, ignore_errors=ignore_errors
                    )
                ]
            )
        else:
            return super().featurize_wrapper(
                x, return_errors=return_errors, ignore_errors=ignore_errors
            )
