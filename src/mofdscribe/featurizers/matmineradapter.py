# -*- coding: utf-8 -*-
"""Use matminer featurizers in mofdscribe."""
from typing import List

import numpy as np
from matminer.featurizers.base import BaseFeaturizer
from pymatgen.core.structure import Structure

from mofdscribe.featurizers.base import MOFBaseFeaturizer


class MatminerAdapter(MOFBaseFeaturizer):
    """
    MatminerAdapter is a wrapper for matminer featurizers to be used in mofdscribe.

    That is, you can then pass :py:class:`mofdscribe.mof.MOF` objects to the
    featurizer.

    .. note::

        If your matminer featurizer needs an expensive
        computation, e.g., the structure graph, if cannot reuse the
        one computed on the ``MOF`` object.

    Example:

        >>> from mofdscribe.featurizers.matmineradapter import MatminerAdapter
        >>> from matminer.featurizers.structure import DensityFeatures
        >>> from mofdscribe.mof import MOF
        >>> from pymatgen.core.structure import Structure
        >>> hkust_structure = Structure.from_file("tests/data/hkust-1.cif")
        >>> matminer_featurizer = DensityFeatures()
        >>> adapter = MatminerAdapter(matminer_featurizer)
        >>> features = adapter.featurize(MOF(hkust_structure))
        >>> original_features = matminer_featurizer.featurize(hkust_structure)
        >>> np.allclose(features, original_features)
        True
    """

    def __init__(self, matminer_featurizer: BaseFeaturizer):
        """Construct a MatminerAdapter.

        Args:
            matminer_featurizer (BaseFeaturizer): A matminer featurizer.
        """
        self.matminer_featurizer = matminer_featurizer

    def _featurize(self, structure: Structure):
        """Call the featurize method of the matminer featurizer."""
        return self.matminer_featurizer.featurize(structure)

    def featurize(self, mof: "MOF") -> np.ndarray:
        """Call the featurize method of the matminer featurizer using a MOF as input.

        Args:
            mof (MOF): A MOF object.

        Returns:
            np.ndarray: The features.
        """
        return self._featurize(mof.structure)

    def _fit(self, structures: List[Structure]) -> np.ndarray:
        """Call the fit method of the matminer featurizer."""
        self.matminer_featurizer.fit(structures)

    def fit(self, mofs: List["MOF"]):
        """Call the fit method of the matminer featurizer using a list of MOFs as input.

        Args:
            mofs (List[MOF]): A list of MOF objects.
        """
        self._fit([Structure.from_sites(mof.structure.sites) for mof in mofs])

    def feature_labels(self):
        """Call the feature_labels method of the matminer featurizer."""
        return self.matminer_featurizer.feature_labels()

    def citations(self):
        """Call the citations method of the matminer featurizer."""
        return self.matminer_featurizer.citations()

    def implementors(self):
        """Call the implementors method of the matminer featurizer."""
        return self.matminer_featurizer.implementors()
