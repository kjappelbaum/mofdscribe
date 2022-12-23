# -*- coding: utf-8 -*-
"""Test the MatminerAdapter."""
import numpy as np
from matminer.featurizers.site import SOAP
from matminer.featurizers.structure import DensityFeatures, SiteStatsFingerprint
from pymatgen.core.structure import Structure

from mofdscribe.featurizers.matmineradapter import MatminerAdapter
from mofdscribe.mof import MOF


def test_matminer_adapter(hkust_structure):
    matminer_featurizer = DensityFeatures()
    adapter = MatminerAdapter(matminer_featurizer)
    features = adapter.featurize(MOF(hkust_structure))
    original_features = matminer_featurizer.featurize(hkust_structure)

    assert np.allclose(features, original_features)

    assert adapter.feature_labels() == matminer_featurizer.feature_labels()
    assert adapter.citations() == matminer_featurizer.citations()
    assert adapter.implementors() == matminer_featurizer.implementors()


def test_matminer_adapter_with_fit(hkust_structure):
    original_featurizer = SiteStatsFingerprint(SOAP(6, 8, 8, 0.4, False, "gto", False))
    hkust_structure = Structure.from_sites(hkust_structure.sites)
    original_featurizer.fit([hkust_structure])
    original_features = original_featurizer.featurize(hkust_structure)

    matminer_featurizer = SiteStatsFingerprint(SOAP(6, 8, 8, 0.4, False, "gto", False))
    adapter = MatminerAdapter(matminer_featurizer)
    adapter.fit([MOF(hkust_structure)])
    features = adapter.featurize(MOF(hkust_structure))

    assert np.allclose(features, original_features)
