# -*- coding: utf-8 -*-
"""Test the SBU featurizer."""

from mofdscribe.sbu.sbu_featurizer import MOFBBs, SBUFeaturizer
from mofdscribe.topology import PHStats


def test_sbu_featurizer(molecule_graph, molecule):
    """Test that we can call SBU featurizers with pymatgen molecules."""
    featurizer = SBUFeaturizer(PHStats(no_supercell=True))

    mofbbs = MOFBBs(nodes=None, linkers=[molecule])
    features = featurizer.featurize(mofbbs)
    assert features.shape == (768,)
    assert features[0] > 0
    assert features[0] < 2
