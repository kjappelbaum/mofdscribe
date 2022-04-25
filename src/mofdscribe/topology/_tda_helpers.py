# -*- coding: utf-8 -*-
from collections import defaultdict
from typing import List, Tuple, Union

import numpy as np
from moltda.construct_pd import construct_pds
from moltda.read_file import make_supercell
from moltda.vectorize_pds import diagrams_to_arrays, get_images
from pymatgen.core import Structure
from loguru import logger
from mofdscribe.utils.substructures import filter_element


# @np_cache
def construct_pds_cached(coords):
    return construct_pds(coords)


# ToDo: only do this for selected elements
# ToDo: only do this for all if we want
def get_persistent_images_for_structure(
    structure: Structure,
    elements: List[List[str]],
    compute_for_all_elements: bool = True,
    min_size: int = 20,
    spread: float = 0.2,
    weighting: str = "identity",
    pixels: Tuple[int] = (50, 50),
    maxB: int = 18,
    maxP: int = 18,
    minB: int = 0,
) -> dict:
    """
    Get the persistent images for a structure.
    Args:
        structure (Structure): input structure
        elements (List[List[str]]): list of elements to compute for
        compute_for_all_elements (bool): compute for all elements
        min_size (int): minimum size of the cell for construction of persistent images
        spread (float): spread of kernel for construction of persistent images
        weighting (str): weighting scheme for construction of persistent images
        pixels (Tuple[int]): size of the image in pixels
        maxB (int): maximum birth time for construction of persistent images
        maxP (int): maximum persistence time for construction of persistent images
    Returns:
        persistent_images (dict): dictionary of persistent images and their barcode representations
    """

    element_images = defaultdict(dict)
    specs = []
    for mB, mP in zip(maxB, maxP):
        specs.append({"minBD": 0, "maxB": mB, "maxP": mP})
    for element in elements:
        try:
            filtered_structure = filter_element(structure, element)
            coords = make_supercell(
                filtered_structure.cart_coords, filtered_structure.lattice.matrix, min_size
            )
            pds = construct_pds_cached(coords)
            pd = diagrams_to_arrays(pds)

            images = get_images(
                pd,
                spread=spread,
                weighting=weighting,
                pixels=pixels,
                specs=specs,
            )
        except ValueError as e:
            images = np.zeros((0, pixels[0], pixels[1]))
            pd = np.zeros((0, maxP + 1))

        # ToDo: make sure that we have the correct length
        element_images["image"][element] = images
        element_images["array"][element] = pd

    if compute_for_all_elements:
        coords = make_supercell(structure.cart_coords, structure.lattice.matrix, min_size)
        pd = diagrams_to_arrays(construct_pds_cached(coords))

        images = get_images(pd, spread=spread, weighting=weighting, pixels=pixels, specs=specs)
        element_images["image"]["all"] = images
        element_images["array"]["all"] = pd

    return element_images


def get_min_max_from_dia(dia, birth_persistence: bool = True):
    if len(dia) == 0:
        return [0, 0, 0, 0]
    d = np.array([[x["birth"], x["death"]] for x in dia])

    if birth_persistence:
        # convert to birth - persistence
        d[:, 1] -= d[:, 0]
    d = np.ma.masked_invalid(d)
    return [d[:, 0].min(), d[:, 0].max(), d[:, 1].min(), d[:, 1].max()]


def diagrams_to_bd_arrays(dgms):
    """Convert persistence diagram objects to persistence diagram arrays."""
    dgm_arrays = {}
    for dim, dgm in enumerate(dgms):
        if len(dgm) == 0:
            dgm_arrays[f"dim{dim}"] = np.zeros((0, 2))
        else:
            arr = np.array([[np.sqrt(p.birth), np.sqrt(p.death)] for p in dgm])

            mask = np.isfinite(arr).all(axis=1)

            arr = arr[mask]
            dgm_arrays[f"dim{dim}"] = arr
    return dgm_arrays


def get_diagrams_for_structure(
    structure,
    elements: List[List[str]],
    compute_for_all_elements: bool = True,
    min_size: int = 20,
):
    keys = [f"dim{i}" for i in range(3)]
    element_dias = defaultdict(dict)
    for element in elements:
        try:
            filtered_structure = filter_element(structure, element)
            coords = make_supercell(
                filtered_structure.cart_coords, filtered_structure.lattice.matrix, min_size
            )
            pds = construct_pds_cached(coords)
            arrays = diagrams_to_bd_arrays(pds)
        except Exception:
            arrays = {key: np.zeros((0, 2)) for key in keys}
        if not len(arrays) == 4:
            for key in keys:
                if key not in arrays:
                    arrays[key] = np.zeros((0, 2))
        element_dias[element] = arrays

    if compute_for_all_elements:
        coords = make_supercell(structure.cart_coords, structure.lattice.matrix, min_size)
        pds = construct_pds_cached(coords)
        arrays = diagrams_to_bd_arrays(pds)
        element_dias["all"] = arrays
        if not len(arrays) == 4:
            for key in keys:
                if key not in arrays:
                    arrays[key] = np.zeros((0, 2))
    assert len(element_dias) == len(elements) + int(compute_for_all_elements)
    return element_dias


def get_persistence_image_limits_for_structure(
    structure: Structure,
    elements: List[List[str]],
    compute_for_all_elements: bool = True,
    min_size: int = 20,
) -> dict:
    limits = defaultdict(list)
    for element in elements:
        try:
            filtered_structure = filter_element(structure, element)
            coords = make_supercell(
                filtered_structure.cart_coords, filtered_structure.lattice.matrix, min_size
            )
            pds = construct_pds(coords)
            pd = diagrams_to_arrays(pds)
            for k, v in pd.items():
                limits[k].append(get_min_max_from_dia(v))
        except ValueError as e:
            pass

    if compute_for_all_elements:
        coords = make_supercell(structure.cart_coords, structure.lattice.matrix, min_size)
        pd = diagrams_to_arrays(construct_pds(coords))
        for k, v in pd.items():
            limits[k].append(get_min_max_from_dia(v))
    return limits
