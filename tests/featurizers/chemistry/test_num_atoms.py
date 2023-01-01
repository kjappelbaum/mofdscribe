from mofdscribe.featurizers.chemistry import NumAtoms
from mofdscribe.mof import MOF


def test_num_atoms(hkust_structure):
    mof = MOF(hkust_structure)
    num_atoms = NumAtoms()
    assert num_atoms.featurize(mof) == 624
