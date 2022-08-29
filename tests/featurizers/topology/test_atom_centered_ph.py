# -*- coding: utf-8 -*-
"""Test the atom centred PH featurizer."""
import numpy as np
import pytest

from mofdscribe.featurizers.topology.atom_centered_ph import AtomCenteredPH, AtomCenteredPHSite

from ..helpers import is_jsonable


def test_atom_centered_ph_site(hkust_structure, irmof_structure, cof_structure):
    """Ensure we get the correct number of features and that they are different for different sites."""
    for i, structure in enumerate([hkust_structure, irmof_structure, cof_structure]):
        featurizer = AtomCenteredPHSite()
        features = featurizer.featurize(structure, 0)
        feature_labels = featurizer.feature_labels()
        assert len(features) == len(feature_labels)
        features_1 = featurizer.featurize(structure, 1)
        assert len(features_1) == len(feature_labels)
        if i < 2:
            # The metals should be equivalent
            assert np.allclose(features, features_1)
            features_not_metal = featurizer.featurize(structure, -1)
            assert np.abs(features - features_not_metal).sum() > 0
    assert is_jsonable(dict(zip(featurizer.feature_labels(), features)))
    assert features.ndim == 1


def test_atom_centered_ph(hkust_structure, irmof_structure, hkust_la_structure):
    """Ensure we get the correct number of features and that they are different for different structures."""
    for structure in [hkust_structure]:
        featurizer = AtomCenteredPH()
        features = featurizer.featurize(structure)
        feature_labels = featurizer.feature_labels()
        assert len(features) == len(feature_labels)

    features_hkust = featurizer.featurize(hkust_structure)
    features_irmof = featurizer.featurize(irmof_structure)
    assert (features_hkust != features_irmof).any()
    assert is_jsonable(dict(zip(featurizer.feature_labels(), features)))
    assert features.ndim == 1

    # now, try that encoding chemistry with varying atomic radii works
    # we do this by computing the PH histograms for structures with same geometry
    # and connectivity, but different atoms
    featurizer = AtomCenteredPH(atom_types=None, alpha_weight="van_der_waals_radius")
    features_hkust = featurizer.featurize(hkust_structure)
    features_hkust_la = featurizer.featurize(hkust_la_structure)
    assert features_hkust.shape == features_hkust_la.shape
    assert (features_hkust != features_hkust_la).any()

    # # now, to be sure do not encode the same thing with the atomic radius
    featurizer = AtomCenteredPH(atom_types=None, alpha_weight=None)
    features_hkust = featurizer.featurize(hkust_structure)
    features_hkust_la = featurizer.featurize(hkust_la_structure)
    assert features_hkust.shape == features_hkust_la.shape
    assert features_hkust == pytest.approx(features_hkust_la, rel=1e-2)
