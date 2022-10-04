from rdkit import Chem

from mofdscribe.featurizers.bu.smarts_matches import number_smart_matches


def test_number_smart_matches():
    """Test that we get the correct number of matches."""
    mol = Chem.MolFromSmiles("C1=CC=CC=C1")
    smarts = ["n1nnnc1"]
    assert number_smart_matches(mol, smarts) == 0

    mol = Chem.MolFromSmiles("CCCP=O")
    smarts = ["[C,S,P]=O"]
    assert number_smart_matches(mol, smarts) == 1

