# -*- coding: utf-8 -*-
"""Featurizer that computes 3D voxelgrids."""
from typing import List, Tuple, Union

import numpy as np
from element_coder import encode
from pymatgen.core import Element, IStructure, Structure

from mofdscribe.featurizers.base import MOFBaseFeaturizer

from ._voxelgrid import VoxelGrid as VGBase


def make_supercell(
    coords: np.ndarray,
    elements: List[int],
    lattice: Tuple[float, float, float],
    size: float,
    min_size: float = -5,
) -> np.ndarray:
    """Generate cubic supercell of a given size.

    Args:
        coords (np.ndarray): matrix of xyz coordinates of the system
        elements (List[int]): atomic numbers of every site
        lattice (Tuple[float, float, float]): lattice constants of the system
        size (float): dimension size of cubic cell, e.g., 10x10x10
        min_size (float): minimum axes size to keep negative xyz
            coordinates from the original cell. Defaults to -5.

    Returns:
        new_cell: supercell array
    """
    a, b, c = lattice
    elements = np.array(elements).reshape(-1, 1)
    xyz_periodic_copies = [coords]
    element_periodic_copies = [elements]

    min_range = -3  # we aren't going in the minimum direction too much, so can make this small
    max_range = 20  # make this large enough, but can modify if wanting an even larger cell

    for x in range(-min_range, max_range):
        for y in range(0, max_range):
            for z in range(0, max_range):
                if x == y == z == 0:
                    continue
                add_vector = x * a + y * b + z * c
                xyz_periodic_copies.append(coords + add_vector)
                element_periodic_copies.append(elements)

    # Combine into one array
    xyz_periodic_total = np.vstack(xyz_periodic_copies)
    elements_all = np.vstack(element_periodic_copies)

    # Filter out all atoms outside of the cubic box
    filter_mask_a = np.max(xyz_periodic_total[:, :3], axis=1) < size
    new_cell = xyz_periodic_total[filter_mask_a]

    filter_mask_b = np.min(new_cell[:, :3], axis=1) > min_size
    new_cell = new_cell[filter_mask_b]

    elements_all = elements_all[filter_mask_a]
    elements_all = elements_all[filter_mask_b]

    return new_cell, elements_all.flatten()


def compute_properties(numbers: np.array, properties: Tuple[str]) -> np.array:
    property_lookup = {}

    unique_numbers = np.unique(numbers)

    for number in unique_numbers:
        element = Element.from_Z(number)
        prop_vec = []
        for prop in properties:
            prop_vec.append(encode(element.symbol, prop))

        property_lookup[number] = prop_vec

    property_array = []
    for number in numbers:
        property_array.append(property_lookup[number])

    return np.array(property_array)


# ToDo: Potentially, we could also do the substruture-based approach for chemistry.
# Not sure though how useful it would be as it would lead to quite large feature vectors.
class VoxelGrid(MOFBaseFeaturizer):
    """
    Describe the structure using a voxel grid.

    For this, we first compute a supercell, the "voxelize" the point cloud.

    For setting the value of the voxels, different options are available:
    Geometry Aggregations: - `binary`: 1 if the voxel is occupied, 0 otherwise -
    `density`: the number of atoms in the voxel / total number of atoms - `TDF`:
    truncated distance function. Value between 0 and 1 indicating the distance
    between the voxel's center and the closest point. 1 on the surface, 0 on
    voxels further than 2 * voxel side.

    Properties: Alternatively/additionally one can use the average of any
    available properties of pymatgen Element objects.
    """

    def __init__(
        self,
        min_size: float = 30,
        n_x: int = 25,
        n_y: int = 25,
        n_z: int = 25,
        geometry_aggregations: Tuple[str] = ("binary",),
        properties: Tuple[str, int] = ("X", "electron_affinity"),
        flatten: bool = True,
        regular_bounding_box: bool = True,
        primitive: bool = False,
    ):
        """Initialize a VoxelGrid featurizer.

        Args:
            min_size (float): Minimum supercell size in Angstrom.
                Defaults to 30.
            n_x (int): Number of bins in x direction
                (Hung et al used 30 and 60 at a cell size of 60). Defaults to 25.
            n_y (int): Number of bins in x direction
                (Hung et al used 30 and 60 at a cell size of 60). Defaults to 25.
            n_z (int): Number of bins in x direction
                (Hung et al used 30 and 60 at a cell size of 60). Defaults to 25.
            geometry_aggregations (Union[Tuple["density" | "binary" | "TDF"], None]):
                Mode for encoding the occupation of voxels.
                * binary: 0 for empty voxels, 1 for occupied.
                * density: number of points inside voxel / total number of points.
                * TDF: Truncated Distance Function. Value between 0 and 1 indicating the distance,
                between the voxel's center and the closest point. 1 on the surface,
                0 on voxels further than 2 * voxel side.
                Defaults to ("binary",).
            properties (Union[Tuple[str, int], None]): Properties used for calculation of the AP-RDF.
                All properties of `pymatgen.core.Species` are available. Defaults to ("X", "electron_affinity").
            flatten (bool): It true, flatten the 3D voxelgrid to 1D array. Defaults to True.
            regular_bounding_box (bool): If True, the bounding box of the point cloud will be adjusted
                in order to have all the dimensions of equal length.
                Defaults to True.
            primitive (bool): If True, the structure is reduced to its primitive
                form before the descriptor is computed. Defaults to True.
        """
        self.min_size = min_size
        self.n_x = n_x
        self.n_y = n_y
        self.n_z = n_z
        self._num_voxels = n_x * n_y * n_z
        self.properties = properties
        self.geometry_aggregations = geometry_aggregations
        self.flatten = flatten
        self.regular_bounding_box = regular_bounding_box
        super().__init__(primitive=primitive)

    def _featurize(self, structure: Union[IStructure, Structure]) -> np.ndarray:
        coords, numbers = make_supercell(
            structure.cart_coords, structure.atomic_numbers, structure.lattice.matrix, self.min_size
        )

        if self.properties is not None:
            properties = compute_properties(numbers, self.properties)
        else:
            properties = None

        vg = VGBase(
            coords,
            properties,
            self.n_x,
            self.n_y,
            self.n_z,
            regular_bounding_box=self.regular_bounding_box,
        )
        vg.compute()
        features = []
        if self.geometry_aggregations is not None:
            for agg in self.geometry_aggregations:
                features.append(
                    vg.get_feature_vector(
                        agg,
                        flatten=self.flatten,
                    )
                )

        if self.properties is not None:
            for i, _ in enumerate(self.properties):
                features.append(
                    vg.get_feature_vector(
                        i,
                        flatten=self.flatten,
                    )
                )

        if self.flatten:
            return np.stack(features).flatten()
        return features

    def _get_feature_labels(self):
        feature_labels = []
        for geometry_aggregation in self.geometry_aggregations:
            for voxel in range(self._num_voxels):
                feature_labels.append(f"voxelgrid_{geometry_aggregation}_{voxel}")
        for prop in self.properties:
            for voxel in range(self._num_voxels):
                feature_labels.append(f"voxelgrid_{prop}_{voxel}")

        return feature_labels

    def feature_labels(self) -> List[str]:
        return self._get_feature_labels()

    def citations(self) -> List[str]:
        return [
            "@article{Hung2022,"
            "doi = {10.1021/acs.jpcc.1c09649},"
            "url = {https://doi.org/10.1021/acs.jpcc.1c09649},"
            "year = {2022},"
            "month = jan,"
            "publisher = {American Chemical Society ({ACS})},"
            "volume = {126},"
            "number = {5},"
            "pages = {2813--2822},"
            "author = {Ting-Hsiang Hung and Zhi-Xun Xu and Dun-Yen Kang "
            "and Li-Chiang Lin},"
            "title = {Chemistry-Encoded Convolutional Neural Networks for "
            "Predicting Gaseous Adsorption in Porous Materials},"
            "journal = {The Journal of Physical Chemistry C}"
            "}",
            "@article{Cho2021,"
            "doi = {10.1021/acs.jpclett.1c00293},"
            "url = {https://doi.org/10.1021/acs.jpclett.1c00293},"
            "year = {2021},"
            "month = mar,"
            "publisher = {American Chemical Society ({ACS})},"
            "volume = {12},"
            "number = {9},"
            "pages = {2279--2285},"
            "author = {Eun Hyun Cho and Li-Chiang Lin},"
            "title = {Nanoporous Material Recognition via 3D Convolutional "
            "Neural Networks: Prediction of Adsorption Properties},"
            "journal = {The Journal of Physical Chemistry Letters}"
            "}",
        ]

    def implementors(self) -> List[str]:
        return ["Kevin Maik Jablonka"]
