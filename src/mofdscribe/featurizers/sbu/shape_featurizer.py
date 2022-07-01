"""Implement some shape featurizers from RDKit using the RDKitAdaptor."""
from typing import List

from rdkit.Chem.Descriptors3D import NPR1 as NPR1_rdkit
from rdkit.Chem.Descriptors3D import NPR2 as NPR2_rdkit
from rdkit.Chem.Descriptors3D import PMI1 as PMI1_rdkit
from rdkit.Chem.Descriptors3D import PMI2 as PMI2_rdkit
from rdkit.Chem.Descriptors3D import PMI3 as PMI3_rdkit
from rdkit.Chem.Descriptors3D import Asphericity as Asphericity_rdkit
from rdkit.Chem.Descriptors3D import Eccentricity as Eccentricity_rdkit
from rdkit.Chem.Descriptors3D import InertialShapeFactor as InertialShapeFactor_rdkit
from rdkit.Chem.Descriptors3D import RadiusOfGyration as RadiusOfGyration_rdkit
from rdkit.Chem.Descriptors3D import SpherocityIndex as SpherocityIndex_rdkit

from .rdkitadaptor import RDKitAdaptor


def _rod_likeness(mol):
    """Compute the ROD-likeness of a molecule."""
    return NPR2_rdkit(mol) - NPR1_rdkit(mol)


def _disk_likeness(mol):
    """Compute the disk-likeness of a molecule."""
    return 2 - 2 * NPR2_rdkit(mol)


def _sphericity(mol):
    """Compute the sphericity of a molecule."""
    return NPR1_rdkit(mol) + NPR2_rdkit(mol) - 1


class Asphericity(RDKitAdaptor):
    """Featurizer for the RDKit Asphericity descriptor."""

    def __init__(self):
        super().__init__(Asphericity_rdkit, ["asphericity"])

    def citations(self) -> List[str]:
        return self.super().citations() + [
            "@incollection{Todeschini2008,"
            "doi = {10.1002/9783527618279.ch37},"
            "url = {https://doi.org/10.1002/9783527618279.ch37},"
            "year = {2008},"
            "month = may,"
            "publisher = {Wiley-{VCH} Verlag {GmbH}},"
            "pages = {1004--1033},"
            "author = {Roberto Todeschini and Viviana Consonni},"
            "title = {Descriptors from Molecular Geometry},"
            "booktitle = {Handbook of Chemoinformatics}"
            "}"
        ]


class Eccentricity(RDKitAdaptor):
    """Featurizer for the RDKit Eccentricity descriptor."""

    def __init__(self):
        super().__init__(Eccentricity_rdkit, ["eccentricity"])

    def citations(self) -> List[str]:
        return self.super().citations() + [
            "@incollection{Todeschini2008,"
            "doi = {10.1002/9783527618279.ch37},"
            "url = {https://doi.org/10.1002/9783527618279.ch37},"
            "year = {2008},"
            "month = may,"
            "publisher = {Wiley-{VCH} Verlag {GmbH}},"
            "pages = {1004--1033},"
            "author = {Roberto Todeschini and Viviana Consonni},"
            "title = {Descriptors from Molecular Geometry},"
            "booktitle = {Handbook of Chemoinformatics}"
            "}"
        ]


class InertialShapeFactor(RDKitAdaptor):
    """Featurizer for the RDKit InertialShapeFactor descriptor."""

    def __init__(self):
        super().__init__(InertialShapeFactor_rdkit, ["inertial_shape_factor"])

    def citations(self) -> List[str]:
        return self.super().citations() + [
            "@incollection{Todeschini2008,"
            "doi = {10.1002/9783527618279.ch37},"
            "url = {https://doi.org/10.1002/9783527618279.ch37},"
            "year = {2008},"
            "month = may,"
            "publisher = {Wiley-{VCH} Verlag {GmbH}},"
            "pages = {1004--1033},"
            "author = {Roberto Todeschini and Viviana Consonni},"
            "title = {Descriptors from Molecular Geometry},"
            "booktitle = {Handbook of Chemoinformatics}"
            "}"
        ]


class NPR1(RDKitAdaptor):
    """Featurizer for the RDKit NPR1 descriptor."""

    def __init__(self):
        super().__init__(NPR1_rdkit, ["npr1"])

    def citations(self) -> List[str]:
        return self.super().citations() + [
            "@article{Sauer2003"
            "doi = {10.1021/ci025599w}"
            "url = {https://doi.org/10.1021/ci025599w}"
            "year = {2003}"
            "month = mar"
            "publisher = {American Chemical Society ({ACS})}"
            "volume = {43}"
            "number = {3}"
            "pages = {987--1003}"
            "author = {Wolfgang H. B. Sauer and Matthias K. Schwarz}"
            "title = {Molecular Shape Diversity of Combinatorial Libraries:{\hspace{0.167em}} A Prerequisite for Broad Bioactivity}"
            "journal = {Journal of Chemical Information and Computer Sciences"
            ""
        ]


class NPR2(RDKitAdaptor):
    """Featurizer for the RDKit NPR2 descriptor."""

    def __init__(self):
        super().__init__(NPR2_rdkit, ["npr2"])

    def citations(self) -> List[str]:
        return self.super().citations() + [
            "@article{Sauer2003"
            "doi = {10.1021/ci025599w}"
            "url = {https://doi.org/10.1021/ci025599w}"
            "year = {2003}"
            "month = mar"
            "publisher = {American Chemical Society ({ACS})}"
            "volume = {43}"
            "number = {3}"
            "pages = {987--1003}"
            "author = {Wolfgang H. B. Sauer and Matthias K. Schwarz}"
            "title = {Molecular Shape Diversity of Combinatorial Libraries:{\hspace{0.167em}} A Prerequisite for Broad Bioactivity}"
            "journal = {Journal of Chemical Information and Computer Sciences"
            ""
        ]


class PMI1(RDKitAdaptor):
    """Featurizer for the RDKit PMI1 descriptor."""

    def __init__(self):
        super().__init__(PMI1_rdkit, ["pmi1"])

    def citations(self) -> List[str]:
        return super().citations()


class PMI2(RDKitAdaptor):
    """Featurizer for the RDKit PMI2 descriptor."""

    def __init__(self):
        super().__init__(PMI2_rdkit, ["pmi2"])

    def citations(self) -> List[str]:
        return super().citations()


class PMI3(RDKitAdaptor):
    """Featurizer for the RDKit PMI3 descriptor."""

    def __init__(self):
        super().__init__(PMI3_rdkit, ["pmi3"])

    def citations(self) -> List[str]:
        return super().citations()


class RadiusOfGyration(RDKitAdaptor):
    """Featurizer for the RDKit RadiusOfGyration descriptor."""

    def __init__(self):
        super().__init__(RadiusOfGyration_rdkit, ["radius_of_gyration"])

    def citations(self) -> List[str]:
        return self.super().citations() + [
            "@incollection{Todeschini2008,"
            "doi = {10.1002/9783527618279.ch37},"
            "url = {https://doi.org/10.1002/9783527618279.ch37},"
            "year = {2008},"
            "month = may,"
            "publisher = {Wiley-{VCH} Verlag {GmbH}},"
            "pages = {1004--1033},"
            "author = {Roberto Todeschini and Viviana Consonni},"
            "title = {Descriptors from Molecular Geometry},"
            "booktitle = {Handbook of Chemoinformatics}"
            "}"
        ]


class SpherocityIndex(RDKitAdaptor):
    """Featurizer for the RDKit Spherocity Index descriptor."""

    def __init__(self):
        super().__init__(SpherocityIndex_rdkit, ["spherocity"])

    def citations(self) -> List[str]:
        return self.super().citations() + [
            "@incollection{Todeschini2008,"
            "doi = {10.1002/9783527618279.ch37},"
            "url = {https://doi.org/10.1002/9783527618279.ch37},"
            "year = {2008},"
            "month = may,"
            "publisher = {Wiley-{VCH} Verlag {GmbH}},"
            "pages = {1004--1033},"
            "author = {Roberto Todeschini and Viviana Consonni},"
            "title = {Descriptors from Molecular Geometry},"
            "booktitle = {Handbook of Chemoinformatics}"
            "}"
        ]


class RodLikeness(RDKitAdaptor):
    """Featurizer for the RDKit Rod Likeness descriptor."""

    def __init__(self):
        super().__init__(_rod_likeness, ["rod_likeness"])

    def citations(self) -> List[str]:
        return self.super().citations() + [
            "@article{Wirth2013,"
            "doi = {10.1007/s10822-013-9659-1},"
            "url = {https://doi.org/10.1007/s10822-013-9659-1},"
            "year = {2013},"
            "month = jun,"
            "publisher = {Springer Science and Business Media {LLC}},"
            "volume = {27},"
            "number = {6},"
            "pages = {511--524},"
            "author = {Matthias Wirth and Andrea Volkamer and Vincent Zoete and Friedrich Rippmann and Olivier Michielin and Matthias Rarey and Wolfgang H. B. Sauer},"
            "title = {Protein pocket and ligand shape comparison and its application in virtual screening},"
            "journal = {Journal of Computer-Aided Molecular Design}"
            "}"
        ]


class DiskLikeness(RDKitAdaptor):
    """Featurizer for the RDKit Disk Likeness descriptor."""

    def __init__(self):
        super().__init__(_disk_likeness, ["disk_likeness"])

    def citations(self) -> List[str]:
        return self.super().citations() + [
            "@article{Wirth2013,"
            "doi = {10.1007/s10822-013-9659-1},"
            "url = {https://doi.org/10.1007/s10822-013-9659-1},"
            "year = {2013},"
            "month = jun,"
            "publisher = {Springer Science and Business Media {LLC}},"
            "volume = {27},"
            "number = {6},"
            "pages = {511--524},"
            "author = {Matthias Wirth and Andrea Volkamer and Vincent Zoete and Friedrich Rippmann and Olivier Michielin and Matthias Rarey and Wolfgang H. B. Sauer},"
            "title = {Protein pocket and ligand shape comparison and its application in virtual screening},"
            "journal = {Journal of Computer-Aided Molecular Design}"
            "}"
        ]


class SphereLikeness(RDKitAdaptor):
    """Featurizer for the RDKit Sphere Likeness descriptor."""

    def __init__(self):
        super().__init__(_sphericity, ["sphere_likeness"])

    def citations(self) -> List[str]:
        return self.super().citations() + [
            "@article{Wirth2013,"
            "doi = {10.1007/s10822-013-9659-1},"
            "url = {https://doi.org/10.1007/s10822-013-9659-1},"
            "year = {2013},"
            "month = jun,"
            "publisher = {Springer Science and Business Media {LLC}},"
            "volume = {27},"
            "number = {6},"
            "pages = {511--524},"
            "author = {Matthias Wirth and Andrea Volkamer and Vincent Zoete and Friedrich Rippmann and Olivier Michielin and Matthias Rarey and Wolfgang H. B. Sauer},"
            "title = {Protein pocket and ligand shape comparison and its application in virtual screening},"
            "journal = {Journal of Computer-Aided Molecular Design}"
            "}"
        ]
