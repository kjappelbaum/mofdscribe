# -*- coding: utf-8 -*-
from mofdscribe.featurizers.graph.dimensionality import Dimensionality
from mofdscribe.featurizers.utils.structure_graph import get_structure_graph
from mofdscribe.mof import MOF


def test_dimensionality(hkust_structure):
    dim = Dimensionality()
    sg = get_structure_graph(hkust_structure, "vesta")
    assert dim.featurize(MOF(hkust_structure))[0] == 3
    assert dim._featurize(sg)[0] == 3
