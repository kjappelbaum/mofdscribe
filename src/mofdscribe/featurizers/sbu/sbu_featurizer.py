# -*- coding: utf-8 -*-
"""Compute features on the SBUs and then aggregate them."""
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
from matminer.featurizers.base import BaseFeaturizer
from pydantic import BaseModel
from pymatgen.core import IMolecule, IStructure, Molecule, Structure

from mofdscribe.featurizers.utils import nan_array
from mofdscribe.featurizers.utils.aggregators import ARRAY_AGGREGATORS


class MOFBBs(BaseModel):
    """Container for MOF building blocks."""

    nodes: Optional[Iterable[Union[Structure, Molecule, IStructure, IMolecule]]]
    linkers: Optional[Iterable[Union[Structure, Molecule, IStructure, IMolecule]]]


class SBUFeaturizer(BaseFeaturizer):
    """
    Compute features on the SBUs and then aggregate them.

    This can be useful if you want to compute some features on the SBUs
    and them aggregrate them to obtain one fixed-length feature vector for the MOF.

    .. warning::
        Note that, currently. not all featurizers can operate on both
        Structures and Molecules.
        If you want to include featurizers that can only operate on one type
        (e.g. :py:obj:`~mofdscribe.sbu.rdkitadaptor.RDKitAdaptor`
        and :py:obj:`~mofdscribe.chemistry.amd.AMD`)
        then you need to create two separate MOFBBs and SBUFeaturizer objects.

    Examples:
        >>> from mofdscribe.sbu import SBUFeaturizer, MOFBBs
        >>> from mofdscribe.sbu.rdkitadaptor import RDKitAdaptor
        >>> from rdkit.Chem.Descriptors3D import Asphericity
        >>> from pymatgen.core import Molecule
        >> from pymatgen.io.babel import BabelMolAdaptor
        >>> base_featurizer = RDKitAdaptor(featurizer=Asphericity, feature_labels=["asphericity"])
        >>> sbu_featurizer = SBUFeaturizer(featurizer=base_featurizer, aggregations=("mean", "std", "min", "max"))
        >>> sbu_featurizer.featurize(
                mofbbs=MOFBBs(nodes=[BabelMolAdaptor.from_string(
                    "[CH-]1C=CC=C1.[CH-]1C=CC=C1.[Fe+2]", "smi").pymatgen_mol],
                linkers=[BabelMolAdaptor.from_string("CCCC", "smi").pymatgen_mol]))

    """

    def __init__(
        self, featurizer: BaseFeaturizer, aggregations: Tuple[str] = ("mean", "std", "min", "max")
    ) -> None:
        """
        Construct a new SBUFeaturizer.

        Args:
            featurizer (BaseFeaturizer): The featurizer to use.
            aggregations (Tuple[str]): The aggregations to use.
                Must be one of :py:obj:`ARRAY_AGGREGATORS`.
        """
        self._featurizer = featurizer
        self._aggregations = aggregations

    def feature_labels(self) -> List[str]:
        labels = []
        base_labels = self._featurizer.feature_labels()
        for bb in ["node", "linker"]:
            for aggregation in self._aggregations:
                for label in base_labels:
                    labels.append(f"{bb}_{aggregation}_{label}")
        return labels

    def featurize(
        self,
        structure: Optional[Union[Structure, IStructure]] = None,
        mofbbs: Optional[MOFBBs] = None,
    ) -> np.ndarray:
        """
        Compute features on the SBUs and then aggregate them.

        If you provide a structure, we will fragment the MOF into SBUs.
        If you already have precomputed fragements or only want to consider a subset
        of the SBUs, you can provide them manually via the `mofbbs` argument.

        Args:
            structure (Union[Structure, IStructure], optional): The structure to featurize.
            mofbbs (MOFBBs, optional): The MOF fragments (nodes and linkers).

        Returns:
            A numpy array of features.
        """
        num_features = len(self._featurizer.feature_labels())
        if mofbbs.linkers is not None:
            linker_feats = [self._featurizer.featurize(linker) for linker in mofbbs.linkers]
        else:
            linker_feats = [nan_array(num_features)]
        if mofbbs.nodes is not None:
            node_feats = [self._featurizer.featurize(node) for node in mofbbs.nodes]
        else:
            node_feats = [nan_array(num_features)]

        aggregated_linker_feats = []
        for aggregation in self._aggregations:
            aggregated_linker_feats.extend(ARRAY_AGGREGATORS[aggregation](linker_feats, axis=0))
        aggregated_linker_feats = np.array(aggregated_linker_feats)

        aggregated_node_feats = []
        for aggregation in self._aggregations:
            aggregated_node_feats.extend(ARRAY_AGGREGATORS[aggregation](node_feats, axis=0))
        aggregated_node_feats = np.array(aggregated_node_feats)

        return np.concatenate((aggregated_linker_feats, aggregated_node_feats))

    def citations(self) -> List[str]:
        return self._featurizer.citations()

    def implementors(self) -> List[str]:
        return self._featurizer.implementors()
