# -*- coding: utf-8 -*-
"""Describe MOF structures in natural language."""

from collections import Counter
from typing import Dict, Optional, Union

from moffragmentor import MOF as MOFFragmentorMOF
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core import IStructure, Structure
from robocrys import StructureCondenser, StructureDescriber

from mofdscribe.featurizers.base import BaseFeaturizer, MOFMultipleFeaturizer
from mofdscribe.featurizers.pore import AccessibleVolume, PoreDiameters, SurfaceArea
from mofdscribe.featurizers.utils.structure_graph import get_sg

_pore_formatters = {
    "lis": lambda x: "largest included sphere {:.2f} A".format(x),
    "density_0.1": lambda x: "density {:.2f} g/cm3".format(x),
    "asa_m2g_0.1": lambda x: "surface area {:.2f} m2/g".format(x),
    "av_cm3g_0.1": lambda x: "accessible volume {:.2f} cm3/g".format(x),
}


class MOFDescriber(BaseFeaturizer):
    """Describe a metal-organic framework in natural language.

    Uses robocrystallographer [Robocrys]_ as well as MOF-specific descriptions.

    References:
        .. [Robocrys] Ganose, A., & Jain, A. (2019).
            Robocrystallographer: Automated crystal structure text descriptions and analysis.
            MRS Communications, 9(3), 874-881.
            https://doi.org/10.1557/mrc.2019.94
    """

    def __init__(
        self,
        condenser_kwargs: Optional[Dict] = None,
        describer_kwargs: Optional[Dict] = None,
        incorporate_smiles: bool = True,
        descibe_pores: bool = True,
    ) -> None:
        """Construct an instance of the MOFDescriber.

        Args:
            condenser_kwargs (Dict): Arguments to pass to the
                StructureCondenser.
            describer_kwargs (Dict): Arguments to pass to the
                StructureDescriber
            incorporate_smiles (bool): If True, describe building blocks.
            describe_pores (bool): If True, add description of the geometry
                of the MOF pores.
        """
        describer_defaults = {"describe_oxidation_states": False, "describe_bond_lengths": True}
        self.condenser_kwargs = condenser_kwargs or {}
        self.describer_kwargs = {**describer_defaults, **(describer_kwargs or {})}
        self.incorporate_smiles = incorporate_smiles
        self.descibe_pores = descibe_pores

    def _get_bb_description(self, structure: Structure, structure_graph: StructureGraph) -> str:
        moffragmentor_mof = MOFFragmentorMOF(structure, structure_graph)
        fragments = moffragmentor_mof.fragment()
        linker_counter = Counter(fragments.linkers.smiles)
        metal_counter = Counter(fragments.nodes.smiles)

        linker_smiles = " ,".join("{} {}".format(v, k) for k, v in linker_counter.items())
        metal_smiles = " ,".join("{} {}".format(v, k) for k, v in metal_counter.items())
        return "Linkers: {}. Metal clusters: {}.".format(linker_smiles, metal_smiles)

    def _get_pore_description(self, structure):
        pore_featurizer = MOFMultipleFeaturizer(
            [PoreDiameters(), SurfaceArea(), AccessibleVolume()]
        )

        features = pore_featurizer.featurize(structure)
        feature_names = pore_featurizer.feature_labels()

        d = dict(zip(feature_names, features))
        return "The MOF has " + ", ".join([v(d[k]) for k, v in _pore_formatters.items()])

    def _get_robocrys_description(self, structure):
        sc = StructureCondenser(**self.condenser_kwargs)
        sd = StructureDescriber(**self.describer_kwargs)
        structure = Structure.from_sites(structure.sites)
        condensed_structure = sc.condense_structure(structure)
        return sd.describe(condensed_structure)

    def _featurize(self, structure: Structure, structure_graph: StructureGraph):
        description = self._get_robocrys_description(structure)
        if self.incorporate_smiles:
            description += " " + self._get_bb_description(structure, structure_graph)
        if self.descibe_pores:
            description += " " + self._get_pore_description(structure)
        return description

    def featurize(self, structure: Union[Structure, IStructure]):
        return self._featurize(structure, get_sg(structure))

    def feature_labels(self):
        return ["description"]

    def citations(self):
        return [
            "@article{Ganose_2019,"
            "    doi = {10.1557/mrc.2019.94},"
            "    url = {https://doi.org/10.1557%2Fmrc.2019.94},"
            "    year = 2019,"
            "    month = {sep},"
            "    publisher = {Springer Science and Business Media {LLC}},"
            "    volume = {9},"
            "    number = {3},"
            "    pages = {874--881},"
            "    author = {Alex M. Ganose and Anubhav Jain},"
            "    title = {Robocrystallographer: automated crystal structure text descriptions and analysis},"
            "    journal = {MRS Communications} Communications}"
            "}"
        ]

    def implementors(self):
        return ["Kevin Maik Jablonka"]
