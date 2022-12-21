from pymatgen.analysis.graphs import StructureGraph

from mofdscribe.featurizers.graph.graphfeaturizer import get_structure_graph


def test_get_structure_graph(hkust_structure):
    sg = get_structure_graph(hkust_structure)
    assert isinstance(sg, StructureGraph)
    assert len(sg) == len(hkust_structure)
