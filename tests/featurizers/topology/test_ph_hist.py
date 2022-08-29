# -*- coding: utf-8 -*-
"""Test the PH Histogram featurizer."""
import pytest

from mofdscribe.featurizers.topology.ph_hist import PHHist

from ..helpers import is_jsonable


def test_ph_stats(hkust_structure, irmof_structure, cof_structure, hkust_la_structure):
    """Ensure we get the correct number of features and that they are different for different structures."""
    for structure in [hkust_structure, irmof_structure, cof_structure]:
        featurizer = PHHist()
        features = featurizer.featurize(structure)
        feature_labels = featurizer.feature_labels()
        assert len(features) == len(feature_labels)

    hkust_feats = featurizer.featurize(hkust_structure)
    irmof_feats = featurizer.featurize(irmof_structure)

    assert (hkust_feats != irmof_feats).any()
    assert is_jsonable(dict(zip(featurizer.feature_labels(), features)))
    assert features.ndim == 1

    # now, try that encoding chemistry with varying atomic radii works
    # we do this by computing the PH histograms for structures with same geometry
    # and connectivity, but different atoms
    featurizer = PHHist(atom_types=None, alpha_weight="atomic_radius_calculated")
    hkust_feats = featurizer.featurize(hkust_structure)
    features_hkust_la = featurizer.featurize(hkust_la_structure)
    assert hkust_feats.shape == features_hkust_la.shape
    assert (hkust_feats != features_hkust_la).any()

    # now, to be sure do not encode the same thing with the atomic radius
    featurizer = PHHist(atom_types=None, alpha_weight=None)
    hkust_feats = featurizer.featurize(hkust_structure)
    features_hkust_la = featurizer.featurize(hkust_la_structure)
    assert hkust_feats.shape == features_hkust_la.shape
    assert hkust_feats == pytest.approx(features_hkust_la, rel=1e-2)
