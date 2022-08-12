# -*- coding: utf-8 -*-
"""Test the pairwise distance stats featurizer."""
from mofdscribe.featurizers.bu.distance_stats_featurizer import PairwiseDistanceStats


def test_pairwise_distance_stats_featurizer(molecule, linker_molecule, triangle_structure):
    """Test the pairwise distance stats featurizer."""
    featurizer = PairwiseDistanceStats()
    feats = featurizer.featurize(molecule)
    assert len(feats) == 4
    for f in feats:
        assert f > 0

    feats = featurizer.featurize(linker_molecule)
    assert len(feats) == 4
    for f in feats:
        assert f > 0
    assert len(feats) == len(featurizer.feature_labels())

    feats = featurizer.featurize(triangle_structure)
    assert len(feats) == 4
    for f in feats:
        assert f > 0
