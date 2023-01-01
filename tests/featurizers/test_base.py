from matminer.featurizers.structure import DensityFeatures

from mofdscribe.featurizers.base import MOFMultipleFeaturizer
from mofdscribe.featurizers.chemistry import RACS
from mofdscribe.featurizers.matmineradapter import MatminerAdapter
from mofdscribe.mof import MOF


def test_mofmultiplefeaturizer(hkust_structure, irmof_structure):
    """Test that the calls work. However, I currently do not know of a good
    way of testing if and how often get_primitive is called
    (other than looking at the logs)"""
    featurizer = MOFMultipleFeaturizer([RACS(), MatminerAdapter(DensityFeatures())])
    # make sure the simple call works
    features = featurizer.featurize(MOF(irmof_structure))
    assert (
        len(features)
        == len(featurizer.feature_labels())
        == len(RACS().feature_labels()) + len(DensityFeatures().feature_labels())
    )

    # make sure the multiple calls work
    features_many_1 = featurizer.featurize_many([MOF(irmof_structure), MOF(hkust_structure)])
    assert features_many_1.ndim == 2
