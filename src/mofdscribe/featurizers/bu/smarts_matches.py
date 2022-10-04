from typing import Iterable

from rdkit import Chem


def number_smart_matches(mol, smarts: Iterable[str]) -> int: 
    """Count the number of SMARTS matches in a molecule.
    This can be useful if we have some prior knowledge 
    about which substructures might be interesting/relevant.

    Args:
        mol (rdkit.Chem.rdchem.Mol): RDKit molecule.
        smarts (Iterable[str]): SMARTS patterns to match.

    Returns:
        int: Number of SMARTS matches.
    """
    s = ",".join("$(" + s + ")" for s in smarts)
    smarts_mol = Chem.MolFromSmarts("[" + s + "]")
    return len(mol.GetSubstructMatches(smarts_mol))