from mofdscribe.chemistry.henryhoa import HenryHOA


def test_henryhoa(hkust_structure, irmof_structure):
    for structure in [hkust_structure, irmof_structure]:
        featurizer = HenryHOA()
