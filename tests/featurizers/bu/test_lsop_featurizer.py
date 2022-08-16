# -*- coding: utf-8 -*-
"""Test that the fragment LSOP featurizer works."""

from mofdscribe.featurizers.bu.lsop_featurizer import LSOP


def test_lsop_featurizer(triangle_structure):
    """Test that the LSOP featurizer works."""
    featurizer = LSOP(types=["tri_plan", "sq_pyr", "cn"])
    features = featurizer.featurize(triangle_structure)
    assert features.shape == (3,)
    assert len(features) == len(featurizer.feature_labels())
    assert features[0] > 0.5
    assert features[1] < 0.1
    assert features[2] == 3
