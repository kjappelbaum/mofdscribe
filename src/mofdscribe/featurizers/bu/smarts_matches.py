"""Featurize a molecule using SMARTS matches."""
from functools import partial
from typing import Iterable, List, Optional

from rdkit import Chem

from mofdscribe.featurizers.bu.rdkitadaptor import RDKitAdaptor


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


class SmartsMatchCounter(RDKitAdaptor):
    """Count the number of SMARTS matches in a molecule.

    This can be useful if we have some prior knowledge
    about which substructures might be interesting/relevant.
    For instance, you might want to count the number of
    carboxylic acid groups in a molecule.
    """

    # ToDo: Perhaps normalize by the length of the SMARTS substructure
    def __init__(self, smarts: Iterable[str], feature_labels: Optional[Iterable[str]]) -> None:
        """Construct a new SmartsMatchCounter.

        Args:
            smarts (Iterable[str]): SMARTS patterns to match.
            feature_labels (str, optional): Feature labels.
                If None, the SMARTS patterns are concatenated to a labels.
        """
        featurizer = partial(number_smart_matches, smarts=smarts)
        if feature_labels is None:
            smarts_string = "_".join(smarts)
            feature_labels = [f"smarts_{smarts_string}"]
        super().__init__(featurizer, feature_labels)


class AcidGroupCounter(SmartsMatchCounter):
    """Count the number of acidic groups in a molecule.

    SMARTS patterns are taken from the Mordred package.
    """

    def __init__(self) -> None:
        """Construct a new AcidGroupCounter."""
        smarts = ["[O;H1]-[C,S,P]=O", "[*;-;!$(*~[*;+])]", "[NH](S(=O)=O)C(F)(F)F", "n1nnnc1"]
        super().__init__(smarts, feature_labels=["acid_groups"])

    def citations(self) -> List[str]:
        return super().citations() + [
            "@article{Moriwaki_2018,"
            "doi = {10.1186/s13321-018-0258-y},"
            "url = {https://doi.org/10.1186%2Fs13321-018-0258-y},"
            "year = 2018,"
            "month = {feb},"
            "publisher = {Springer Science and Business Media {LLC}},"
            "volume = {10},"
            "number = {1},"
            "author = {Hirotomo Moriwaki and Yu-Shi Tian and Norihito Kawashita and Tatsuya Takagi},"
            "title = {Mordred: a molecular descriptor calculator},"
            "journal = {J Cheminform}"
            "}"
        ]

    def implementors(self) -> List[str]:
        return super().implementors() + ["Moriwaki H, Tian Y-S, Kawashita N, Takagi T"]


class BaseGroupCounter(SmartsMatchCounter):
    """Count the number of basic groups in a molecule.

    SMARTS pattern taken from the Mordred package
    """

    def __init__(self) -> None:
        """Construct a new BaseGroupCounter."""
        smarts = [
            "[NH2]-[CX4]",
            "[NH](-[CX4])-[CX4]",
            "N(-[CX4])(-[CX4])-[CX4]",
            "[*;+;!$(*~[*;-])]",
            "N=C-N",
            "N-C=N",
        ]
        super().__init__(smarts, feature_labels=["base_groups"])

    def citations(self) -> List[str]:
        return super().citations() + [
            "@article{Moriwaki_2018,"
            "doi = {10.1186/s13321-018-0258-y},"
            "url = {https://doi.org/10.1186%2Fs13321-018-0258-y},"
            "year = 2018,"
            "month = {feb},"
            "publisher = {Springer Science and Business Media {LLC}},"
            "volume = {10},"
            "number = {1},"
            "author = {Hirotomo Moriwaki and Yu-Shi Tian and Norihito Kawashita and Tatsuya Takagi},"
            "title = {Mordred: a molecular descriptor calculator},"
            "journal = {J Cheminform}"
            "}"
        ]

    def implementors(self) -> List[str]:
        return super().implementors() + ["Moriwaki H, Tian Y-S, Kawashita N, Takagi T"]
