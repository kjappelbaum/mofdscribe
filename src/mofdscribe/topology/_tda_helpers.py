# -*- coding: utf-8 -*-
"""Utlities for working with persistence diagrams."""
from collections import defaultdict
from typing import List, Tuple, Union

import numpy as np
from moltda.construct_pd import construct_pds
from moltda.read_file import make_supercell
from moltda.vectorize_pds import diagrams_to_arrays, get_images
from pymatgen.core import Structure
from pymatgen.transformations.advanced_transformations import CubicSupercellTransformation

from mofdscribe.utils.aggregators import MA_ARRAY_AGGREGATORS
from mofdscribe.utils.substructures import filter_element


# @np_cache
def construct_pds_cached(coords, periodic=False):
    return construct_pds(coords, periodic=periodic)


# ToDo: only do this for selected elements
# ToDo: only do this for all if we want
def get_persistent_images_for_structure(
    structure: Structure,
    elements: List[List[str]],
    compute_for_all_elements: bool = True,
    min_size: int = 20,
    spread: float = 0.2,
    weighting: str = 'identity',
    pixels: Tuple[int] = (50, 50),
    maxB: int = 18,
    maxP: int = 18,
    minB: int = 0,
    periodic: bool = False,
) -> dict:
    """
    Get the persistent images for a structure.

    Args:
        structure (Structure): input structure elements (List[List[str]]): list
            of elements to compute for compute_for_all_elements (bool): compute for
            all elements min_size (int): minimum size of the cell for construction
            of persistent images spread (float): spread of kernel for construction
            of persistent images weighting (str): weighting scheme for construction
            of persistent images pixels (Tuple[int]): size of the image in pixels
        maxB (int): maximum birth time for construction of persistent images
        maxP (int): maximum persistence time for construction of persistent
            images periodic (bool): if True (experimental!), use the periodic
            Euclidean distance

    Returns:
        persistent_images (dict): dictionary of persistent images and their
        barcode representations
    """

    element_images = defaultdict(dict)
    specs = []
    for mB, mP in zip(maxB, maxP):
        specs.append({'minBD': 0, 'maxB': mB, 'maxP': mP})
    for element in elements:
        try:
            filtered_structure = filter_element(structure, element)
            if periodic:
                sc = CubicSupercellTransformation(min_size=min_size).apply_transformation(
                    filtered_structure
                )
                coords = sc.frac_coords
            else:
                coords = make_supercell(
                    filtered_structure.cart_coords, filtered_structure.lattice.matrix, min_size
                )
            pds = construct_pds_cached(coords, periodic=periodic)
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
            images[:] = np.nan
            pd = np.zeros((0, maxP + 1))
            pd[:] = np.nan

        # ToDo: make sure that we have the correct length
        element_images['image'][element] = images
        element_images['array'][element] = pd

    if compute_for_all_elements:
        if periodic:
            sc = CubicSupercellTransformation(min_size=min_size).apply_transformation(structure)
            coords = sc.frac_coords
        else:
            coords = make_supercell(structure.cart_coords, structure.lattice.matrix, min_size)
        pd = diagrams_to_arrays(construct_pds_cached(coords))

        images = get_images(pd, spread=spread, weighting=weighting, pixels=pixels, specs=specs)
        element_images['image']['all'] = images
        element_images['array']['all'] = pd

    return element_images


def get_min_max_from_dia(dia, birth_persistence: bool = True):
    if len(dia) == 0:
        return [0, 0, 0, 0]
    d = np.array([[x['birth'], x['death']] for x in dia])

    if birth_persistence:
        # convert to birth - persistence
        d[:, 1] -= d[:, 0]
    d = np.ma.masked_invalid(d)
    return [d[:, 0].min(), d[:, 0].max(), d[:, 1].min(), d[:, 1].max()]


def diagrams_to_bd_arrays(dgms):
    """Convert persistence diagram objects to persistence diagram arrays."""
    dgm_arrays = {}
    for dim, dgm in enumerate(dgms):

        if dgm:
            arr = np.array(
                [[np.sqrt(dgm[i].birth), np.sqrt(dgm[i].death)] for i in range(len(dgm))]
            )

            mask = np.isfinite(arr).all(axis=1)

            arr = arr[mask]
            dgm_arrays[f'dim{dim}'] = arr

        else:
            dgm_arrays[f'dim{dim}'] = np.zeros((0, 2))

    return dgm_arrays


def get_diagrams_for_structure(
    structure,
    elements: List[List[str]],
    compute_for_all_elements: bool = True,
    min_size: int = 20,
    periodic: bool = False,
):
    keys = [f'dim{i}' for i in range(3)]
    element_dias = defaultdict(dict)
    nan_array = np.zeros((0, 2))
    nan_array[:] = np.nan
    for element in elements:
        try:
            filtered_structure = filter_element(structure, element)
            if periodic:
                sc = CubicSupercellTransformation(min_size=min_size).apply_transformation(
                    filtered_structure
                )
                coords = sc.frac_coords
            else:
                coords = make_supercell(
                    filtered_structure.cart_coords, filtered_structure.lattice.matrix, min_size
                )
            pds = construct_pds_cached(coords)
            arrays = diagrams_to_bd_arrays(pds)
        except Exception:
            arrays = {key: nan_array for key in keys}
        if not len(arrays) == 4:
            for key in keys:
                if key not in arrays:
                    arrays[key] = nan_array
        element_dias[element] = arrays

    if compute_for_all_elements:
        if periodic:
            sc = CubicSupercellTransformation(min_size=min_size).apply_transformation(structure)
            coords = sc.frac_coords
        else:
            coords = make_supercell(structure.cart_coords, structure.lattice.matrix, min_size)
        pds = construct_pds_cached(coords)
        arrays = diagrams_to_bd_arrays(pds)
        element_dias['all'] = arrays
        if len(arrays) != 4:
            for key in keys:
                if key not in arrays:
                    arrays[key] = nan_array
    if len(element_dias) != len(elements) + int(compute_for_all_elements):
        raise ValueError('Something went wrong with the diagram extraction.')
    return element_dias


def get_persistence_image_limits_for_structure(
    structure: Structure,
    elements: List[List[str]],
    compute_for_all_elements: bool = True,
    min_size: int = 20,
    periodic: bool = False,
) -> dict:
    limits = defaultdict(list)
    for element in elements:
        try:
            filtered_structure = filter_element(structure, element)

            if periodic:
                sc = CubicSupercellTransformation(min_size=min_size).apply_transformation(
                    filtered_structure
                )
                coords = sc.frac_coords
            else:
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
        if periodic:
            sc = CubicSupercellTransformation(min_size=min_size).apply_transformation(structure)
            coords = sc.frac_coords
        else:
            coords = make_supercell(structure.cart_coords, structure.lattice.matrix, min_size)
        pd = diagrams_to_arrays(construct_pds(coords))
        for k, v in pd.items():
            limits[k].append(get_min_max_from_dia(v))
    return limits


def persistent_diagram_stats(
    diagram: np.ndarray, aggregrations: Tuple[str], nanfiller: float = 0
) -> dict:
    """
    Compute statistics for a persistence diagram.

    Args:
        diagram (np.ndarray): The persistence diagram.
        aggregrations (Tuple[str]): The name of the aggregations to compute.
        nanfiller (float): The value to use when a nan is encountered.

    Returns:
        dict: nested dictionary with the following structure:
            {'persistence_parameter': {'statistic': value}}
        where persistence_parameter is one of ['birth', 'death', 'persistence']
    """
    stats = {
        'birth': {},
        'death': {},
        'persistence': {},
    }

    try:
        d = np.array([[x['birth'], x['death'], x['death'] - x['birth']] for x in diagram])
    except IndexError:
        d = np.array([[x[0], x[1], x[1] - x[0]] for x in diagram])
    d = np.ma.masked_invalid(d)

    for aggregation in aggregrations:
        agg_func = MA_ARRAY_AGGREGATORS[aggregation]
        for i, key in enumerate(['birth', 'death', 'persistence']):
            try:
                stats[key][aggregation] = agg_func(d[:, i])
            except IndexError:
                stats[key][aggregation] = nanfiller
    return stats
