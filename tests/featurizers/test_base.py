import numpy as np
import pandas as pd
import pytest
from matminer.featurizers.structure import DensityFeatures

from mofdscribe.featurizers.base import MOFMultipleFeaturizer
from mofdscribe.featurizers.chemistry import APRDF, RACS


@pytest.mark.parametrize("primitive", [True, False])
def test_mofmultiplefeaturizer(hkust_structure, irmof_structure, primitive):
    """Test that the calls work. However, I currently do not know of a good
    way of testing if and how often get_primitive is called
    (other than looking at the logs)"""
    structures = [hkust_structure, irmof_structure]
    featurizer = MOFMultipleFeaturizer([RACS(), DensityFeatures()], primitive=primitive)
    for featurizer_ in featurizer.featurizers:
        assert featurizer_.primitive is False

    # make sure the simple call works
    features = featurizer.featurize(irmof_structure)
    assert (
        len(features)
        == len(featurizer.feature_labels())
        == len(RACS().feature_labels()) + len(DensityFeatures().feature_labels())
    )

    # make sure the multiple calls work
    features_many_1 = featurizer.featurize_many(structures)
    features_many_1_labels = featurizer.feature_labels()
    assert features_many_1.ndim == 2

    featurizer = MOFMultipleFeaturizer(
        [RACS(), DensityFeatures()], iterate_over_entries=False, primitive=primitive
    )
    features_many_2 = featurizer.featurize_many(structures)
    features_many_2_labels = featurizer.feature_labels()
    assert features_many_2.ndim == 2

    print(features_many_2.shape, features_many_1.shape)

    features_many_1_df = pd.DataFrame(features_many_1, columns=features_many_1_labels)
    features_many_2_df = pd.DataFrame(features_many_2, columns=features_many_2_labels)

    # independent of the way we call the featurizer, the features should be the same
    values_1 = features_many_1_df.values
    values_2 = features_many_2_df[features_many_1_df.columns].values
    assert np.allclose(values_1, values_2, rtol=1e-2)
