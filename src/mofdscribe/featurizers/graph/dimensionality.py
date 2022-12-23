# -*- coding: utf-8 -*-
"""Compute the bond-topological dimensionality."""
from typing import List

import numpy as np
from pymatgen.analysis.dimensionality import get_dimensionality_larsen
from pymatgen.analysis.graphs import StructureGraph

from mofdscribe.featurizers.graph.graphfeaturizer import GraphFeaturizer
from mofdscribe.mof import MOF


class Dimensionality(GraphFeaturizer):
    def __init__(self) -> None:
        """Construct a new Dimensionality featurizer."""
        super().__init__()

    def featurize(self, mof: MOF) -> np.ndarray:
        return self._featurize(mof.structure_graph)

    def _featurize(self, structure_graph: StructureGraph) -> np.ndarray:
        return get_dimensionality_larsen(structure_graph)

    def feature_labels(self) -> List[str]:
        return ["dimensionality"]

    def citations(self) -> List[str]:
        return [
            "@article{Larsen_2019,"
            "doi = {10.1103/physrevmaterials.3.034003},"
            "url = {https://doi.org/10.1103%2Fphysrevmaterials.3.034003},"
            "year = 2019,"
            "month = {mar},"
            "publisher = {American Physical Society ({APS})},"
            "volume = {3},"
            "number = {3},"
            "author = {Peter Mahler Larsen and Mohnish Pandey and Mikkel Strange and Karsten Wedel Jacobsen},"
            "title = {Definition of a scoring parameter to identify low-dimensional materials components},"
            "journal = {Phys. Rev. Materials}"
            "}"
        ]

    def implementors(self) -> List[str]:
        return ["Kevin Maik Jablonka"]
