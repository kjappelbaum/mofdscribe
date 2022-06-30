# -*- coding: utf-8 -*-
"""Compute features on the SBUs and then aggregate them."""
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
from matminer.featurizers.base import BaseFeaturizer
from pydantic import BaseModel
from pymatgen.core import IMolecule, IStructure, Molecule, Structure

from mofdscribe.featurizers.utils import nan_array
from mofdscribe.featurizers.utils.aggregators import ARRAY_AGGREGATORS

from .utils import boxed_molecule


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
        Note that, currently, not all featurizers can operate on both
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
                Currently, we do not support `MultipleFeaturizer`s.
                Please, instead, use multiple SBUFeaturizers.
                If you use a featurizer that is not implemented in mofdscribe
                (e.g. a matminer featurizer), you need to wrap using a method
                that describes on which data objects the featurizer can operate on.
                If you do not do this, we default to assuming that it operates on structures.
            aggregations (Tuple[str]): The aggregations to use.
                Must be one of :py:obj:`ARRAY_AGGREGATORS`.

        ToDo:
            - Support `MultipleFeaturizer`s (should be ok, if we recursively call the operates_on method).

        """
        self._featurizer = featurizer
        self._aggregations = aggregations
        try:
            _operates_on = featurizer.operates_on()
            if (Structure in _operates_on) and (Molecule in _operates_on):
                self._operates_on = "both"
            elif Structure in _operates_on:
                self._operates_on = "structure"
            elif Molecule in _operates_on:
                self._operates_on = "molecule"

        except AttributeError:
            self._operates_on = "structure"

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

        If you manually provide the `mofbbs`,  we will convert molecules to structures
        where possible.

        Args:
            structure (Union[Structure, IStructure], optional): The structure to featurize.
            mofbbs (MOFBBs, optional): The MOF fragments (nodes and linkers).

        Returns:
            A numpy array of features.

        Raises:
            ValueError: If neither `structure` nor `mofbbs` are provided.
            ValueError: If structures are provided, but the selected featurizer
                operates on molecules
            RuntimeError: If an unexpected combination of types and
                `operates_on` is provided.

        ToDo:
            - Perhaps use type dispatch instead of different keyword arguments.
              (we can use fastcore or code it ourselves)
            - We can also directly pass the graph to the featurizer if it want to work
                on the graph.
        """
        # if i know what the featurizer wants, I can always cast to a structure
        num_features = len(self._featurizer.feature_labels())
        if structure is None and mofbbs is None:
            raise ValueError("You must provide a structure or mofbbs.")

        if structure is not None:
            from moffragmentor import MOF

            mof = MOF.from_structure(structure)
            fragments = mof.fragment()

            if self._operates_on == "both" or self._operates_on == "molecule":
                linkers = [linker.molecule for linker in fragments.linkers]
                nodes = [node.molecule for node in fragments.nodes]
            else:
                linkers = [boxed_molecule(linker.molecule) for linker in fragments.linkers]
                nodes = [boxed_molecule(node.molecule) for node in fragments.nodes]

        if mofbbs is not None:

            linkers = list(mofbbs.linkers) if mofbbs.linkers is not None else []
            nodes = list(mofbbs.nodes) if mofbbs.nodes is not None else []
            types = [type(node) for node in nodes] + [type(linker) for linker in linkers]

            if not len(set(types)) == 1:
                raise ValueError("All nodes and linkers must be of the same type.")

            this_type = types[0]
            if (this_type == Structure or this_type == IStructure) and (
                self._operates_on == "both" or self._operates_on == "structure"
            ):
                # this is the simple case, we do not need to convert to molecules
                pass
            elif (this_type == Molecule or this_type == IMolecule) and (
                self._operates_on == "both" or self._operates_on == "molecule"
            ):
                # again simple case, we do not need to convert to structures

                pass
            elif (this_type == Molecule or this_type == IMolecule) and (
                self._operates_on == "structure"
            ):
                # we need to convert to structures
                nodes = [boxed_molecule(node) for node in nodes]
                linkers = [boxed_molecule(linker) for linker in linkers]
            elif (this_type == Structure or this_type == IStructure) and (
                self._operates_on == "molecule"
            ):
                raise ValueError(
                    "You provided structures for a featurizer that operates on molecules. "
                    / "Cannot automatically convert to molecules from structures."
                )
            else:
                raise RuntimeError("Unexpected type of nodes or linkers.")
        linker_feats = [self._featurizer.featurize(linker) for linker in linkers]
        if not linker_feats:
            linker_feats = [nan_array(num_features)]

        node_feats = [self._featurizer.featurize(node) for node in nodes]
        if not node_feats:
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
