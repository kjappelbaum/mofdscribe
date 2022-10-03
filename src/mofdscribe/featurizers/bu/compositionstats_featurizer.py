# -*- coding: utf-8 -*-
"""Describe the chemical composition of structures."""
from collections import defaultdict
from typing import List, Tuple, Union

import numpy as np
from element_coder import encode
from matminer.featurizers.base import BaseFeaturizer
from pymatgen.core import IMolecule, IStructure, Molecule, Structure

from mofdscribe.featurizers.utils.aggregators import ARRAY_AGGREGATORS
from mofdscribe.featurizers.utils.extend import (
    operates_on_imolecule,
    operates_on_istructure,
    operates_on_molecule,
    operates_on_structure,
)


@operates_on_molecule
@operates_on_imolecule
@operates_on_istructure
@operates_on_structure
class CompositionStats(BaseFeaturizer):
    """
    Describe the composition of molecules by computing statistics of their compositions.

    The featurizer will encode the element on all sites in the structure
    using user-defined encodings. Then it aggregates those encodings using
    user-defined encodings (e.g. min, max, min).
    """

    def __init__(
        self,
        encodings: Tuple[str] = ("mod_pettifor", "X"),
        aggregations: Tuple[str] = ("mean", "std", "max", "min"),
    ) -> None:
        """Initialize a CompositionStats featurizer.

        Args:
            encodings (Tuple[str]): Encoding used for the elements.
                Can be one of :py:obj:`element_coder.data.coding_data._PROPERTY_KEYS`.
                Defaults to ("mod_pettifor", "X").
            aggregations (Tuple[str]): Statistic to compute over the element encodings.
                Can be one of :py:obj:`mofdscribe.featurizers.utils.aggregators.ARRAY_AGGREGATORS`.
                Defaults to ("mean", "std", "max", "min").
        """
        self.aggregations = aggregations
        self.encodings = encodings

    def feature_labels(self) -> List[str]:
        feature_labels = []

        for encoding in self.encodings:
            for agg in self.aggregations:
                feature_labels.append(f"composition_stats_{encoding}_{agg}")

        return feature_labels

    def featurize(self, molecule: Union[Molecule, IMolecule, Structure, IStructure]) -> np.ndarray:
        encodings = defaultdict(list)
        for encoding in self.encodings:
            for site in molecule.sites:
                encodings[encoding] = encode(site.specie, encoding)

        features = []
        for encoding in self.encodings:
            for agg in self.aggregations:
                features.append(ARRAY_AGGREGATORS[agg](encodings[encoding]))

        return np.array(features)

    def implementors(self) -> List[str]:
        return ["Kevin Maik Jablonka"]

    def citations(self) -> List[str]:
        return []
