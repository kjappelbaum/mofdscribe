# -*- coding: utf-8 -*-
import numpy as np

from mofdscribe.topology.atom_centered_ph import AtomCenteredPHSite, AtomCenteredPH


def test_atom_centered_ph_site(hkust_structure, irmof_structure):
    for structure in [hkust_structure, irmof_structure]:
        featurizer = AtomCenteredPHSite()
        features = featurizer.featurize(structure, 0)
        feature_labels = featurizer.feature_labels()
        assert len(features) == len(feature_labels)
        features_1 = featurizer.featurize(structure, 1)
        assert len(features_1) == len(feature_labels)
        # The metals should be equivalent
        assert (features == features_1).all()
        features_not_metal = featurizer.featurize(structure, -1)
        assert np.abs(features - features_not_metal).sum() > 0


def test_atom_centered_ph(hkust_structure, irmof_structure):
    for structure in [hkust_structure]:
        featurizer = AtomCenteredPH()
        features = featurizer.featurize(structure)
        feature_labels = featurizer.feature_labels()
        assert len(features) == len(feature_labels)

    features_hkust = featurizer.featurize(hkust_structure)
    features_irmof = featurizer.featurize(irmof_structure)
    assert (features_hkust != features_irmof).any()
