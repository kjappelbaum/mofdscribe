"""Implement some shape featurizers from RDKit using the RDKitAdaptor."""
from typing import List
from rdkit.Chem.Descriptors3D import (
    Asphericity as Asphericity_rdkit,
    Eccentricity as Eccentricity_rdkit,
    InertialShapeFactor as InertialShapeFactor_rdkit,
    NPR1 as NPR1_rdkit,
    NPR2 as NPR2_rdkit,
    PMI1 as PMI1_rdkit,
    PMI2 as PMI2_rdkit,
    PMI3 as PMI3_rdkit,
    RadiusOfGyration as RadiusOfGyration_rdkit,
    SpherocityIndex as SpherocityIndex_rdkit,
)

from .rdkitadaptor import RDKitAdaptor


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
