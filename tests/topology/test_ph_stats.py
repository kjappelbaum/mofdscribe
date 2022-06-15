# -*- coding: utf-8 -*-
"""Test the PH Stats featurizer"""
from mofdscribe.topology.ph_stats import PHStats

from ..helpers import is_jsonable


def test_ph_stats(hkust_structure, irmof_structure, cof_structure):
    """Ensure we get the correct number of features and that they are different for different structures"""
    for structure in [hkust_structure, irmof_structure, cof_structure]:
        featurizer = PHStats()
        features = featurizer.featurize(structure)
        feature_labels = featurizer.feature_labels()
        assert len(features) == len(feature_labels)

    hkust_feats = featurizer.featurize(hkust_structure)
    irmof_feats = featurizer.featurize(irmof_structure)
    assert (hkust_feats != irmof_feats).any()
    assert is_jsonable(dict(zip(featurizer.feature_labels(), features)))
    assert features.ndim == 1
