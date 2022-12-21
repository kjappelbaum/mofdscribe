from mofdscribe.featurizers.graph.dimensionality import Dimensionality
from mofdscribe.featurizers.utils.structure_graph import get_structure_graph


def test_dimensionality(hkust_structure):
    dim = Dimensionality()
    sg = get_structure_graph(hkust_structure, "vesta")
    assert dim.featurize(hkust_structure) == 3
    assert dim.featurize(sg) == 3
