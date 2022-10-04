# -*- coding: utf-8 -*-
"""Use RDkit featurizers on pymatgen molecules."""

from typing import Callable, Iterable, List, Union

import numpy as np
from loguru import logger
from matminer.featurizers.base import BaseFeaturizer
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.core import Molecule
from structuregraph_helpers.create import get_local_env_method

from .utils import create_rdkit_mol_from_mol_graph
from ..utils.extend import operates_on_imolecule, operates_on_molecule


@operates_on_molecule
@operates_on_imolecule
class RDKitAdaptor(BaseFeaturizer):
    """
    Use any featurizer that can operate on RDkit molecules on pymatgen molecules.

    For this, we convert the pymatgen molecule to an RDkit molecule,
    using the coordinates of the pymatgen molecule as the coordinates of
    a conformer.
    """

    def __init__(
        self,
        featurizer: Callable,
        feature_labels: Iterable[str],
        local_env_strategy: str = "vesta",
        force_sanitize: bool = True,
    ) -> None:
        """Constuct a new RDKitAdaptor.

        Args:
            featurizer (Callable): Function that takes an RDKit molecule and returns
                some features (int, float, or list or array of them).
            feature_labels (Iterable[str]): Names of features. Must be the same length as
                the number of features returned by the featurizer.
            local_env_strategy (str): If the `featurize` method is called with a `Molecule`
                object, this determines the local environment strategy to use to convert
                the molecule to a MoleculeGraph.
                Defaults to "vesta".
            force_sanitize (bool): If True, the RDKit molecule will be sanitized
        """
        self._featurizer = featurizer
        self._feature_labels = list(feature_labels)
        self._local_env_strategy = local_env_strategy
        self._force_sanitize = force_sanitize

    def feature_labels(self) -> List[str]:
        return self._feature_labels

    def featurize(self, molecule: Union[Molecule, MoleculeGraph]) -> np.ndarray:
        """
        Call the RDKit featurizer on the molecule.

        If the input molecule is a Molecule, we convert it to a MoleculeGraph
        using the local environment strategy specified in the constructor.

        Args:
            molecule: A pymatgen Molecule or MoleculeGraph object.

        Returns:
            A numpy array of features.
        """
        if isinstance(molecule, MoleculeGraph):
            molecule_graph = molecule
        else:
            molecule_graph = MoleculeGraph.with_local_env_strategy(
                molecule, get_local_env_method(self._local_env_strategy)
            )
        rdkit_mol = create_rdkit_mol_from_mol_graph(
            molecule_graph, force_sanitize=self._force_sanitize
        )
        feats = self._featurizer(rdkit_mol)
        if isinstance(feats, (list, tuple, np.ndarray)):
            return np.asarray(feats)
        elif isinstance(feats, (float, int)):
            return np.array([feats])
        else:
            logger.warning("Featurizer returned an unsupported type: {}".format(type(feats)))
            return feats

    def citations(self) -> List[str]:
        return [
            "@misc{https://doi.org/10.5281/zenodo.591637,"
            "doi = {10.5281/ZENODO.591637},"
            "url = {https://zenodo.org/record/591637},"
            "author = {Landrum,  Greg and Tosco,  Paolo and Kelley,"
            " Brian and {Ric} and {Sriniker} and {Gedeck} and Vianello,  "
            "Riccardo and {NadineSchneider} and Kawashima,"
            " Eisuke and Dalke,  Andrew and N,  Dan and Cosgrove,"
            " David and Jones,  Gareth and Cole,  Brian and Swain,"
            "  Matt and Turk,  Samo and {AlexanderSavelyev} and Vaucher,  Alain"
            " and Wójcikowski,  Maciej and {Ichiru Take} and Probst,  Daniel "
            "and Ujihara,  Kazuya and Scalfani,  Vincent F. and Godin,  Guillaume"
            " and Pahl,  Axel and {Francois Berenger} and {JLVarjo} "
            "and {Strets123} and {JP} and {DoliathGavid}},"
            "    title = {rdkit/rdkit: 2022_03_3 (Q1 2022) Release},"
            "    publisher = {Zenodo},"
            "    year = {2022},"
            "    copyright = {Open Access}"
            " }"
        ]

    def implementors(self) -> List[str]:
        return ["Kevin Maik Jablonka", "Greg Landrum and RDKit authors"]
