# -*- coding: utf-8 -*-
"""Test the BU featurizer."""

import numpy as np
from matminer.featurizers.site import SOAP
from matminer.featurizers.structure import SiteStatsFingerprint
from pymatgen.core import Structure

from mofdscribe.featurizers.bu.bu_featurizer import BUFeaturizer, MOFBBs
from mofdscribe.featurizers.topology import PHStats


def test_bu_featurizer(hkust_structure, molecule):
    """Test that we can call BU featurizers with pymatgen molecules."""
    featurizer = BUFeaturizer(PHStats(no_supercell=True))

    mofbbs = MOFBBs(nodes=None, linkers=[molecule])
    features = featurizer.featurize(mofbbs=mofbbs)
    assert features.shape == (768,)
    assert features[600] >= 0
    assert features[600] < 2

    featurizer = BUFeaturizer(PHStats(no_supercell=True))
    features = featurizer.featurize(structure=hkust_structure)
    assert features.shape == (768,)
    assert features[0] > 0
    assert features[0] < 2


def test_bu_featurizer_with_matminer_featurizer(hkust_structure, hkust_linker_structure):
    """Test that we can call BU featurizers with matminer molecules."""
    # we disable the periodic keyword to be able to compare with the molecules
    base_feat = SiteStatsFingerprint(SOAP(6, 8, 8, 0.4, False, "gto", False))
    hkust_structure = Structure.from_sites(hkust_structure.sites)
    base_feat.fit([hkust_structure])
    featurizer = BUFeaturizer(base_feat, aggregations=("mean",))
    features = featurizer.featurize(structure=hkust_structure)
    assert features.shape == (2592 * 2,)
    assert features[0] >= 0
    assert features[0] < 2

    linker_feats = [f for f in featurizer.feature_labels() if "linker" in f]
    assert len(linker_feats) == 2592

    # test that our fit method works
    featurizer = BUFeaturizer(
        SiteStatsFingerprint(SOAP(6, 8, 8, 0.4, False, "gto", False)), aggregations=("mean",)
    )
    featurizer.fit([hkust_structure])
    features_direct_fit = featurizer.featurize(structure=hkust_structure)
    assert np.allclose(features, features_direct_fit, rtol=0.01)

    # test that the linker features are actually the ones we get when
    # we featurize the linker
    linker_feats = featurizer._featurizer.featurize(hkust_linker_structure)
    linker_feature_mask = [i for i, f in enumerate(featurizer.feature_labels()) if "linker" in f]
    assert np.allclose(features[linker_feature_mask], linker_feats, rtol=0.01, equal_nan=True)
