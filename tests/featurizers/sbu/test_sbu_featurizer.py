# -*- coding: utf-8 -*-
"""Test the SBU featurizer."""

from mofdscribe.featurizers.sbu.sbu_featurizer import MOFBBs, SBUFeaturizer
from mofdscribe.featurizers.topology import PHStats
from matminer.featurizers.site import SOAP
from matminer.featurizers.structure import SiteStatsFingerprint
from pymatgen.core import Structure


def test_sbu_featurizer(hkust_structure, molecule):
    """Test that we can call SBU featurizers with pymatgen molecules."""
    featurizer = SBUFeaturizer(PHStats(no_supercell=True))

    mofbbs = MOFBBs(nodes=None, linkers=[molecule])
    features = featurizer.featurize(mofbbs=mofbbs)
    assert features.shape == (768,)
    assert features[0] > 0
    assert features[0] < 2

    featurizer = SBUFeaturizer(PHStats(no_supercell=True))
    features = featurizer.featurize(structure=hkust_structure)
    assert features.shape == (768,)
    assert features[0] > 0
    assert features[0] < 2


def test_sbu_featurizer_with_matminer_featurizer(hkust_structure):
    """Test that we can call SBU featurizers with matminer molecules."""
    base_feat = SiteStatsFingerprint(SOAP.from_preset("formation_energy"))
    hkust_structure = Structure.from_sites(hkust_structure.sites)
    base_feat.fit([hkust_structure])
    featurizer = SBUFeaturizer(base_feat, aggregations=("mean",))
    features = featurizer.featurize(structure=hkust_structure)
    assert features.shape == (9504 * 2,)
    assert features[0] > 0
    assert features[0] < 2

    linker_feats = [f for f in featurizer.feature_labels() if "linker" in f]
    assert len(linker_feats) == 9504
