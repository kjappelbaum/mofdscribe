from mofdscribe.topology.ph_image import PHImage


def test_phimage(hkust_structure, irmof_structure):
    phi = PHImage()
    for structure in [hkust_structure, irmof_structure]:
        features = phi.featurize(structure)
        assert len(features) == 20 * 20 * 4 * 3

    assert len(phi.feature_labels()) == 20 * 20 * 4 * 3
