# -*- coding: utf-8 -*-
"""Test that the nconf20 featurizer works on molecules."""

from rdkit.Chem import AllChem

from mofdscribe.featurizers.bu.nconf20_featurizer import NConf20, _n_conf20


def test_nconf20_featurizer(linker_molecule):
    """Test that the nconf20 featurizer works on molecules."""
    # first test that it works directly with RDKit molecules
    # structure (latanoprost) is from Fig 1 in https://www.nature.com/articles/s41597-022-01288-4
    mol = AllChem.MolFromSmiles(
        r"CC(C)OC(=O)CCC/C=C\C[C@H]1[C@H](C[C@H]([C@@H]1CC[C@H](CCC2=CC=CC=C2)O)O)O"
    )
    nconf20 = _n_conf20(mol)
    assert nconf20[0] > 50

    featurizer = NConf20()
    feats = featurizer.featurize(linker_molecule)
    assert len(feats) == 1
    assert feats[0] == 0

    assert len(featurizer.citations()) == 3
