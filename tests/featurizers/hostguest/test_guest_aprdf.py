from mofdscribe.featurizers.hostguest.guest_aprdf import GuestCenteredAPRDF


def test_guest_centered_aprdf(floating_structure):
    """Test the GuestCenteredAPRDF."""
    featurizer = GuestCenteredAPRDF()

    features = featurizer.featurize(floating_structure)
    assert len(features) == (20 - 2) / 0.25 * 2 * 3
    assert len(featurizer.feature_labels()) == (20 - 2) / 0.25 * 2 * 3
