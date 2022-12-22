# -*- coding: utf-8 -*-
"""Test the BU featurizer."""

import numpy as np
from matminer.featurizers.site import SOAP
from matminer.featurizers.structure import SiteStatsFingerprint
from pymatgen.core import Structure

from mofdscribe.featurizers.bu.bu_featurizer import BUFeaturizer, MOFBBs
from mofdscribe.featurizers.matmineradapter import MatminerAdapter
from mofdscribe.featurizers.topology import PHStats
from mofdscribe.mof import MOF


def test_bu_featurizer(hkust_structure, molecule):
    """Test that we can call BU featurizers with pymatgen molecules."""
    featurizer = BUFeaturizer(PHStats(no_supercell=True))

    mofbbs = MOFBBs(nodes=None, linkers=[molecule])
    features = featurizer.featurize(mofbbs=mofbbs)
    assert features.shape == (768,)
    assert features[600] >= 0
    assert features[600] < 2

    featurizer = BUFeaturizer(PHStats(no_supercell=True))
    features = featurizer.featurize(mof=MOF(hkust_structure))
    assert features.shape == (768,)
    assert features[0] > 0
    assert features[0] < 2


def test_bu_featurizer_with_matminer_featurizer(hkust_structure, hkust_linker_structure):
    """Test that we can call BU featurizers with matminer molecules."""
    # we disable the periodic keyword to be able to compare with the molecules
    base_feat = MatminerAdapter(SiteStatsFingerprint(SOAP(4, 4, 4, 0.1, False, "gto", False)))
    hkust_structure = Structure.from_sites(hkust_structure.sites)
    featurizer = BUFeaturizer(base_feat, aggregations=("mean",))
    featurizer.fit([MOF(hkust_structure)])
    features = featurizer.featurize(mof=MOF(hkust_structure))
    assert features.shape == (400 * 2,)
    assert features[0] >= 0
    assert features[0] < 2

    linker_feats = [f for f in featurizer.feature_labels() if "linker" in f]
    assert len(linker_feats) == 400

    # test that our fit method works
    featurizer = BUFeaturizer(
        MatminerAdapter(SiteStatsFingerprint(SOAP(4, 4, 4, 0.1, False, "gto", False))),
        aggregations=("mean",),
    )
    featurizer.fit([MOF(hkust_structure)])
    features_direct_fit = featurizer.featurize(mof=MOF(hkust_structure))
    assert np.allclose(features, features_direct_fit, rtol=0.01)

    # test that the linker features are actually the ones we get when
    # we featurize the linker
    linker_feats = featurizer._featurizer._featurize(hkust_linker_structure)
    linker_feature_mask = [i for i, f in enumerate(featurizer.feature_labels()) if "linker" in f]
    assert len(features[linker_feature_mask]) == len(linker_feats)
    # todo: check where potential differences come from
    assert np.allclose(features[linker_feature_mask], linker_feats, rtol=0.005, atol=1e-3)
