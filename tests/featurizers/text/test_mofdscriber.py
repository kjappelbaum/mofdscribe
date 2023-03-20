from mofdscribe.featurizers.text.mofdscriber import MOFDescriber


def test_mofdscriber(hkust_structure, hkust_structure_graph):
    describer = MOFDescriber()
    result = describer._featurize(hkust_structure, hkust_structure_graph)
    assert "Cu" in result
    assert "surface area" in result
