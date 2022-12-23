# -*- coding: utf-8 -*-
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
from typing import Collection, Union

import numpy as np
from matminer.featurizers.base import BaseFeaturizer, MultipleFeaturizer
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core import IMolecule, IStructure, Molecule, Structure

from mofdscribe.mof import MOF


class MOFBaseFeaturizer(BaseFeaturizer):
    """Base featurizer for MOF structure based featurizers.

    .. note::

        If you implement a new :code:`MOFBaseFeaturizer`,
        you need to implement a :code:`_featurize` method.
    """

    @abstractmethod
    def _featurize(
        self, structure_object: Union[Structure, IStructure, Molecule, IMolecule, StructureGraph]
    ) -> np.ndarray:
        raise NotImplementedError("_featurize must be implemented in a subclass")

    @abstractmethod
    def featurize(self, mof: MOF) -> np.ndarray:
        raise NotImplementedError("featurize must be implemented in a subclass")

    # ToDo: make make them abstractmethods in a "FittabelMOFBaseFeaturizer" class
    def _fit(
        self, structure_object: Union[Structure, IStructure, Molecule, IMolecule, StructureGraph]
    ):
        raise NotImplementedError("_fit is not implemented for MOFBaseFeaturizer")

    def fit(self, mofs: Union[MOF, Collection[MOF]]):
        raise NotImplementedError("fit is not implemented for MOFBaseFeaturizer")


class MOFBaseSiteFeaturizer(BaseFeaturizer):
    """Base featurizer for MOF site based featurizers.

    .. note::

        If you implement a new :code:`MOFBaseSiteFeaturizer`,
        you need to implement a :code:`_featurize` method.
    """

    @abstractmethod
    def _featurize(
        self,
        structure_object: Union[Structure, IStructure, Molecule, IMolecule, StructureGraph],
        idx: int,
    ) -> np.ndarray:
        raise NotImplementedError("_featurize must be implemented in a subclass")

    @abstractmethod
    def featurize(self, mof: MOF, idx: int) -> np.ndarray:
        raise NotImplementedError("featurize must be implemented in a subclass")

    # ToDo: make make them abstractmethods in a "FittabelMOFBaseFeaturizer" class
    def _fit(
        self,
        structure_object: Union[Structure, IStructure, Molecule, IMolecule, StructureGraph],
        sites: Union[int, Collection[int]],
    ):
        raise NotImplementedError("_fit is not implemented for MOFBaseSiteFeaturizer")

    def fit(self, mofs: Union[MOF, Collection[MOF]], sites: Union[int, Collection[int]]):
        raise NotImplementedError("fit is not implemented for MOFBaseSiteFeaturizer")


# Probably, collect the featurizers into different groups depending on the type of input,
# i.e. graphs, molecules, structures, etc. and then have a multiple featurizer for each group.
# and this one just calls the appropriate multiple featurizer depending on the type of input.
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

    def __init__(self, featurizers):
        """
        Construct a MOFMultipleFeaturizer.

        Args:
            featurizers (list): List of featurizers to use.
        """
        self.iterate_over_entries = True
        self.featurizers = featurizers

    def featurize(self, mof: MOF) -> np.ndarray:
        return np.array([feature for f in self.featurizers for feature in f.featurize(mof)])

    def featurize_many(self, entries, ignore_errors=False, return_errors=False, pbar=True):
        return np.array(
            super().featurize_many(
                entries,
                ignore_errors=ignore_errors,
                return_errors=return_errors,
                pbar=pbar,
            )
        )

    def featurize_wrapper(self, x, return_errors=False, ignore_errors=False):

        return np.array(
            [
                feature
                for f in self.featurizers
                for feature in f.featurize_wrapper(
                    x, return_errors=return_errors, ignore_errors=ignore_errors
                )
            ]
        )
