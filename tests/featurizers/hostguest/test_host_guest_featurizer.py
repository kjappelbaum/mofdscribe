from matminer.featurizers.structure.sites import SiteStatsFingerprint

from mofdscribe.featurizers.hostguest import HostGuestFeaturizer


def test_host_guest_featurizer(floating_structure):
    """Test the HostGuestFeaturizer."""
    featurizer = HostGuestFeaturizer(
        featurizer=SiteStatsFingerprint.from_preset("SOAP_formation_energy"),
        aggregations=("mean",),
    )
    featurizer.fit([floating_structure])
    features = featurizer.featurize(floating_structure)
    labels = featurizer._featurizer.feature_labels()
    assert len(features) == 2 * len(labels)
