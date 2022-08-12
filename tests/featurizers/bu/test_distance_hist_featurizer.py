# -*- coding: utf-8 -*-
"""Test the pairwise distance histogram featurizer."""
from mofdscribe.featurizers.bu.distance_hist_featurizer import PairwiseDistanceHist


def test_pairwise_distance_hist_featurizer(molecule, linker_molecule, triangle_structure):
    """Test the pairwise distance hist featurizer."""
    for mol in (molecule, linker_molecule, triangle_structure):
        featurizer = PairwiseDistanceHist()
        feats = featurizer.featurize(mol)
        assert len(feats) == 30
        for f in feats:
            assert f >= 0
        assert feats.sum() >= 1
        assert len(feats) == len(featurizer.feature_labels())
