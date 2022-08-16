# -*- coding: utf-8 -*-
from mofdscribe.featurizers.bu.compositionstats_featurizer import CompositionStats


def test_composition_stats_featurizer(molecule, linker_molecule, triangle_structure):
    """Test the composition stats featurizer."""
    for mol in (molecule, linker_molecule, triangle_structure):
        featurizer = CompositionStats()
        feats = featurizer.featurize(mol)
        assert len(feats) == 8
        for f in feats:
            assert f >= 0
        assert len(feats) == len(featurizer.feature_labels())
