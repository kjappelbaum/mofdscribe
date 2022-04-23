from mofdscribe.chemistry.partialchargehistogram import PartialChargeHistogram


def test_partialchargehistogram(hkust_structure, irmof_structure):
    pch = PartialChargeHistogram()
    assert len(pch.feature_labels()) == 16

    for structure in [hkust_structure, irmof_structure]:
        features = pch.featurize(structure)
        assert len(features) == 16
