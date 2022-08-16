# -*- coding: utf-8 -*-
"""Test that we can convert pymatgen Molecules to RDkit molecules."""
from mofdscribe.featurizers.bu.utils import create_rdkit_mol_from_mol_graph


def test_conversion_to_rdkit(molecule_graph):
    """Test that we can convert pymatgen Molecules to RDkit molecules."""
    rdkit_mol = create_rdkit_mol_from_mol_graph(molecule_graph)
    assert rdkit_mol.GetNumAtoms() == molecule_graph.graph.number_of_nodes()
