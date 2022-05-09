from mofdscribe.topology.atom_centered_ph import AtomCenteredPHSite
import numpy as np


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
