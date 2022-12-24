# -*- coding: utf-8 -*-
"""Test the BU featurizer."""

import numpy as np
import pandas as pd
from matminer.featurizers.site import SOAP
from matminer.featurizers.structure import SiteStatsFingerprint
from pymatgen.core import Structure

from mofdscribe.featurizers.bu.bu_featurizer import (
    BindingSitesFeaturizer,
    BranchingSitesFeaturizer,
    BUFeaturizer,
    MOFBBs,
)
from mofdscribe.featurizers.bu.lsop_featurizer import LSOP
from mofdscribe.featurizers.graph.dimensionality import Dimensionality
from mofdscribe.featurizers.matmineradapter import MatminerAdapter
from mofdscribe.featurizers.topology import PHStats
from mofdscribe.mof import MOF


def test_bu_featurizer(hkust_structure, molecule, mof74_structure):
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

    # test if it works with graph featurizers
    featurizer = BUFeaturizer(Dimensionality(), aggregations=("mean",))
    features = featurizer.featurize(mof=MOF(hkust_structure))
    assert features.shape == (2,)
    assert features[0] == 0
    assert features[1] == 0

    # make sure we find the rod node
    featurizer = BUFeaturizer(Dimensionality(), aggregations=("mean",))
    mof4 = MOF(mof74_structure, fragmentation_kwargs={"check_dimensionality": False})
    features = featurizer.featurize(mof=mof4)
    assert features.shape == (2,)
    assert features[0] == 1
    assert features[1] == 0


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


def test_binding_site_featurizer(hkust_structure):
    """Make sure we can featurize on the 'binding' substructure and that the features make some sense."""
    featurizer = BindingSitesFeaturizer(LSOP())
    feats = featurizer.featurize(mof=MOF(hkust_structure))
    labels = featurizer.feature_labels()
    for label in labels:
        assert "BindingSite" in label
    assert len(feats) == len(labels)

    df = pd.DataFrame(data=[feats], columns=labels)

    assert df["BindingSitesFeaturizer_node_mean_lsop_cn"].values[0] == 8.0

    assert df["BindingSitesFeaturizer_node_mean_lsop_cuboct_max"].values[0] > 0.95

    assert df["BindingSitesFeaturizer_linker_mean_lsop_cn"].values[0] == 6.0

    assert df["BindingSitesFeaturizer_linker_mean_lsop_hex_plan_max"].values[0] > 0.4


def test_branching_site_featurizer(hkust_structure):
    """Make sure we can featurize on the 'branching' substructure and that the features make some sense."""
    featurizer = BranchingSitesFeaturizer(LSOP())
    feats = featurizer.featurize(mof=MOF(hkust_structure))
    labels = featurizer.feature_labels()
    for label in labels:
        assert "BranchingSite" in label
    assert len(feats) == len(labels)

    df = pd.DataFrame(data=[feats], columns=labels)

    assert df["BranchingSitesFeaturizer_node_mean_lsop_cn"].values[0] == 4.0
    assert df["BranchingSitesFeaturizer_linker_mean_lsop_cn"].values[0] == 3.0

    assert df["BranchingSitesFeaturizer_node_mean_lsop_sq_plan_max"].values[0] > 0.9

    assert df["BranchingSitesFeaturizer_linker_mean_lsop_tri_plan"].values[0] > 0.98
