# -*- coding: utf-8 -*-
"""Measure the RMSD between a building block and topological prototypes."""
import json
import os
from collections import OrderedDict
from typing import List, Tuple, Union

import numpy as np
from loguru import logger
from matminer.featurizers.base import BaseFeaturizer
from pymatgen.core import IMolecule, IStructure, Molecule, Structure
from superpose3d import Superpose3D

from ..utils.aggregators import ARRAY_AGGREGATORS

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(THIS_DIR, "prototype_env.json"), "r") as handle:
    STRUCTURE_ENVS = json.loads(handle.read())

ALL_AVAILABLE_TOPOS = tuple(STRUCTURE_ENVS.keys())

__all__ = ("BUMatch",)


def match_bb(
    bb: Union[Structure, IStructure, Molecule, IMolecule],
    prototype: str,
    aggregations: Tuple[str],
    allow_rescale: bool = True,
    match: str = "auto",
    skip_non_fitting_if_possible: bool = True,
    mismatch_fill_value: float = 10_000,
) -> float:
    """
    Compute the RMSD between a building block and a prototype.

    Args:
        bb (Union[Structure, IStructure, Molecule, IMolecule]): The building block to compare.
        prototype (str): The prototype to compare against.
        aggregations (Tuple[str]): The aggregations to use.
        allow_rescale (bool): Whether to scale the RMSD by the number of atoms.
            Defaults to True.
        match (str): The type of matching to use. Defaults to 'auto'.
        skip_non_fitting_if_possible (bool): Whether to skip RMSDs of
            building blocks that do not match due to mismatching coordination numbers.
        mismatch_fill_value (float): The value to fill in for mismatching
            coordination numbers. Defaults to 10_000.

    Returns:
        The RMSD between the two structures.
    """
    if match != "auto":
        is_node = 1 if match == "node" else -1
    else:
        cn = len(bb)
        is_node = -1 if cn == 2 else 1
    logger.debug(f"Matching {bb} to {prototype} with {aggregations}. Is node: {is_node}")
    coords_this = bb.cart_coords
    keys_to_match = [k for k in STRUCTURE_ENVS[prototype].keys() if int(k) * (is_node) >= 0]
    logger.debug(f"Matching {keys_to_match}")
    rmsds_fitting = []
    rmsds_non_fitting = []
    for key in keys_to_match:
        reference_coordinates = STRUCTURE_ENVS[prototype][key]
        reference_coordinates = np.array(reference_coordinates)

        if len(reference_coordinates) == len(coords_this):
            rmsd, _, _, _ = Superpose3D(
                reference_coordinates, coords_this, allow_rescale=allow_rescale
            )
            rmsds_fitting.append(rmsd)
        else:
            rmsds_non_fitting.append(mismatch_fill_value)
    logger.debug(f"Fitting RMSDs: {rmsds_fitting}")
    logger.debug(f"Non-fitting RMSDs: {rmsds_non_fitting}")
    rmsds = None
    if (len(rmsds_fitting) > 0) & skip_non_fitting_if_possible:
        rmsds = rmsds_fitting
    else:
        rmsds = rmsds_non_fitting + rmsds_fitting
    aggregation_results = OrderedDict()

    for aggregation in aggregations:
        aggregation_results[f"{prototype}_{aggregation}"] = ARRAY_AGGREGATORS[aggregation](rmsds)

    return aggregation_results


class BUMatch(BaseFeaturizer):
    """MOFs are assembled from building blocks on a net.

    The "ideal" vertex "structures" of the net can fit better or
    worse with the "shape" of the actual building blocks.
    This featurizer attempts to quantify this mismatch.

    .. note::

        The edge match values will all be quite close to zero
        and hence not that meaningful (two points always form a line,
        there is not much room for mismatch unless the length of the line
        [which is ignored by default with `allow_rescale=True`]...).
        In pratice, you should consider treating them seperate
        from the vertex match values.

    Examples:
        >>> from mofdscribe.bu import BUMatch
        >>> from pymatgen.core import Structure
        >>> s = Structure.from_file("tests/test_files/bu_test_1.cif")
        >>> bu_match = BUMatch(topos=["tbo", "pcu"], aggregations=["mean", "min"])
        >>> bu_match.featurize(s)
    """

    def __init__(
        self,
        allow_rescale: bool = True,
        mismatch_fill_value: float = 1_000,
        return_only_best: bool = True,
        aggregations: Tuple[str] = ("max", "min", "mean", "std"),
        topos: Tuple[str] = ALL_AVAILABLE_TOPOS,
        match: str = "auto",
        skip_non_fitting_if_possible: bool = True,
    ) -> None:
        """Create a new BUMatch featurizer.

        Args:
            allow_rescale (bool): If True, allow to multiple coordinates of structure
                with scalar to better match the reference structure.
                Defaults to True.
            mismatch_fill_value (float): Value use to fill entries for which the RMSD
                computation cannot be perform due to a mismatch in coordination numbers.
                Defaults to 1_000.
            return_only_best (bool): If True, do not compute statistics but only return the minimum
                RMSD. Defaults to True.
            aggregations (Tuple[str]): Functions to use to aggregate RMSD of
                the different possible positions. Defaults to ("max", "min", "mean", "std").
            topos (Tuple[str]): RCSR codes to consider for matching.
                Defaults to ALL_AVAILABLE_TOPOS.
            match (str): BB to consider for matching. Must be one of "auto", "edge" or "node".
                If the mode is "auto" it assumes that the number of sites in the input building
                block is equal to the number of connection vertices. Hence, if there are more
                than two sites, it will match the node. If there are only two sites, it will match
                the edge.
                Defaults to "auto".
            skip_non_fitting_if_possible (bool): If True, do not compute RMSD for
                non-compatible BBs. Defaults to True.
        """
        self.allow_rescale = allow_rescale
        self.mismatch_fill_value = mismatch_fill_value
        self.topos = topos
        self.return_only_best = return_only_best
        if not return_only_best and aggregations is None:
            logger.error("If return_only_best is False, aggregations must be set.")
        self.aggregations = aggregations
        if self.return_only_best:
            self.aggregations = ("min",)

        self.match = match
        self.skip_non_fitting_if_possible = skip_non_fitting_if_possible

    def _get_feature_labels(self):
        labels = []
        for topo in self.topos:
            if self.return_only_best:
                labels.append(f"bumatch_{self.allow_rescale}_{topo}")
            else:
                for aggregation in self.aggregations:
                    labels.append(f"bumatch_{self.allow_rescale}_{topo}_{aggregation}")

        return labels

    def feature_labels(self) -> List[str]:
        return self._get_feature_labels()

    def featurize(self, s: Union[Structure, IStructure, Molecule, IMolecule]) -> np.ndarray:
        """Structure is here spanned by the connecting points of a BU."""
        features = []
        for topo in self.topos:
            feats = match_bb(
                s,
                topo,
                self.aggregations,
                self.allow_rescale,
                self.match,
                self.skip_non_fitting_if_possible,
                self.mismatch_fill_value,
            )
            features.extend(feats.values())

        return np.array(features)

    def citations(self):
        return ["Kevin Maik Jablonka and Berend Smit, TBA."]

    def implementors(self):
        return ["Kevin Maik Jablonka"]
