# -*- coding: utf-8 -*-
"""Use matminer featurizers in mofdscribe."""
from typing import List

from pymatgen.core.structure import Structure

from mofdscribe.featurizers.base import MOFBaseFeaturizer


class MatminerAdapter(MOFBaseFeaturizer):
    def __init__(self, matminer_featurizer):
        self.matminer_featurizer = matminer_featurizer

    def _featurize(self, structure: Structure):
        return self.matminer_featurizer.featurize(structure)

    def featurize(self, mof):
        return self._featurize(mof.structure)

    def _fit(self, structures: List[Structure]):
        self.matminer_featurizer.fit(structures)

    def fit(self, mofs):
        self._fit([mof.structure for mof in mofs])

    def feature_labels(self):
        return self.matminer_featurizer.feature_labels()

    def citations(self):
        return self.matminer_featurizer.citations()

    def implementors(self):
        return self.matminer_featurizer.implementors()
