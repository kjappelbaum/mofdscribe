from matminer.featurizers.structure.sites import SiteStatsFingerprint

from mofdscribe.featurizers.hostguest import HostGuestFeaturizer
from mofdscribe.mof import MOF


def test_host_guest_featurizer(floating_structure):
    """Test the HostGuestFeaturizer."""
    featurizer = HostGuestFeaturizer(
        featurizer=SiteStatsFingerprint.from_preset("SOAP_formation_energy"),
        aggregations=("mean",),
    )
    featurizer.fit([MOF(floating_structure)])
    features = featurizer.featurize(MOF(floating_structure))
    labels = featurizer._featurizer.feature_labels()
    assert len(features) == 2 * len(labels)
