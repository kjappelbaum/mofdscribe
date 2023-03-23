# -*- coding: utf-8 -*-
"""Utlities for working with persistence diagrams."""
from collections import defaultdict
from typing import Collection, Dict, List, Optional, Tuple

import numpy as np
from element_coder import encode_many
from loguru import logger
from pymatgen.core import Structure
from pymatgen.transformations.advanced_transformations import CubicSupercellTransformation
from tqdm import tqdm

from mofdscribe.featurizers.utils import flat
from mofdscribe.featurizers.utils.aggregators import MA_ARRAY_AGGREGATORS
from mofdscribe.featurizers.utils.substructures import filter_element


# @np_cache
def construct_pds_cached(coords, periodic=False, weights: Optional[Collection] = None):
    from moleculetda.construct_pd import construct_pds

    return construct_pds(coords, periodic=periodic, weights=weights)


def _get_homology_generators(
    filtration, persistence: Optional["dionysus._dionysus.ReducedMatrix"] = None
) -> dict:
    import dionysus as d
    from moleculetda.construct_pd import get_persistence

    if persistence is None:
        persistence = get_persistence(filtration)

    homology_generators = defaultdict(lambda: defaultdict(list))

    for i, c in tqdm(enumerate(persistence), total=len(persistence)):
        try:
            dim = len(list(filtration[i])) - 1

            death = filtration[i].data
            points_a = list(filtration[i])
            points_b = [list(filtration[x.index]) for x in c]
            data_b = [filtration[x.index].data for x in c]
            birth = data_b[-1]

            all_points = points_a + points_b
            all_points = list(set(flat(all_points)))
            if birth < death:
                homology_generators[dim][(birth, death)].append(all_points)
        except Exception as e:
            pass

    return homology_generators


def make_supercell(
    coords: np.ndarray,
    lattice: List[np.array],
    size: float,
    elements: Optional[List[str]] = None,
    min_size: float = -5,
) -> np.ndarray:
    """
    Generate cubic supercell of a given size.
    Args:
        coords (np.ndarray): matrix of xyz coordinates of the system
        lattice (Tuple[np.array]): lattice vectors of the system
        elements (List[str]): list of elements in the system.
            If None, will create a list of 'X' of the same length as coords
        size (float): dimension size of cubic cell, e.g., 10x10x10
        min_size (float): minimum axes size to keep negative xyz coordinates from the original cell
    Returns:
        new_cell: supercell array
    """

    # handle potential weights that we want to carry over but not change
    a, b, c = lattice

    xyz_periodic_copies = []
    element_copies = []

    # xyz_periodic_copies.append(coords)
    # element_copies.append(np.array(elements).reshape(-1,1))
    min_range = -3  # we aren't going in the minimum direction too much, so can make this small
    max_range = 20  # make this large enough, but can modify if wanting an even larger cell

    if elements is None:
        elements = ["X"] * len(coords)

    for x in range(-min_range, max_range):
        for y in range(0, max_range):
            for z in range(0, max_range):
                if x == y == z == 0:
                    continue
                add_vector = x * a + y * b + z * c
                xyz_periodic_copies.append(coords + add_vector)
                assert len(elements) == len(
                    coords
                ), f"Elements and coordinates are not the same length. Found {len(coords)} coordinates and {len(elements)} elements."
                element_copies.append(np.array(elements).reshape(-1, 1))

    # Combine into one array
    xyz_periodic_total = np.vstack(xyz_periodic_copies)

    element_periodic_total = np.vstack(element_copies)
    assert len(xyz_periodic_total) == len(
        element_periodic_total
    ), f"Elements and coordinates are not the same length. Found {len(xyz_periodic_total)} coordinates and {len(element_periodic_total)} elements."
    # Filter out all atoms outside of the cubic box
    filter_a = np.max(xyz_periodic_total, axis=1) < size
    new_cell = xyz_periodic_total[filter_a]
    filter_b = np.min(new_cell[:], axis=1) > min_size
    new_cell = new_cell[filter_b]
    new_elements = element_periodic_total[filter_a][filter_b]

    return new_cell, new_elements.flatten()


def _coords_for_structure(
    structure: Structure,
    min_size: int = 50,
    periodic: bool = False,
    no_supercell: bool = False,
    weighting: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if no_supercell:
        if weighting is not None:
            weighting = encode_many([str(s.symbol) for s in structure.species], weighting)
        return structure.cart_coords, weighting

    else:
        if periodic:
            transformed_s = CubicSupercellTransformation(
                min_length=min_size, force_90_degrees=True
            ).apply_transformation(structure)
            if weighting is not None:
                weighting = encode_many([str(s.symbol) for s in transformed_s.species], weighting)
            return transformed_s.cart_coords, weighting
        else:
            if weighting is not None:
                weighting_arr = np.array(
                    encode_many([str(s.symbol) for s in structure.species], weighting)
                )
                # we can add the weighing as additional column for the cooords
                coords_w_weight, elements = make_supercell(
                    np.hstack([structure.cart_coords, weighting_arr.reshape(-1, 1)]),
                    structure.lattice.matrix,
                    min_size,
                )
                return coords_w_weight[:, :-1], coords_w_weight[:, -1], elements
            else:
                sc, elements = make_supercell(
                    structure.cart_coords, structure.lattice.matrix, min_size
                )
                return (
                    sc,
                    None,
                    elements,
                )


def _pd_arrays_from_coords(
    coords, periodic: bool = False, bd_arrays: bool = False, weights: Optional[np.ndarray] = None
):
    from moleculetda.vectorize_pds import diagrams_to_arrays

    pds = construct_pds_cached(coords, periodic=periodic, weights=weights)
    if bd_arrays:
        pd = diagrams_to_bd_arrays(pds)
    else:
        pd = diagrams_to_arrays(pds)

    return pd


def get_images(
    pd,
    spread: float = 0.2,
    weighting: str = "identity",
    pixels: List[int] = (50, 50),
    specs: List[dict] = None,
    dimensions: Collection[int] = (0, 1, 2),
):
    from moleculetda.vectorize_pds import pd_vectorization

    images = []
    for dim in dimensions:
        dgm = pd[f"dim{dim}"]
        images.append(
            pd_vectorization(
                dgm, spread=spread, weighting=weighting, pixels=pixels, specs=specs[dim]
            )
        )
    return images


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
    max_b: int = 18,
    max_p: int = 18,
    periodic: bool = False,
    no_supercell: bool = False,
    alpha_weighting: Optional[str] = None,
) -> dict:
    """
    Get the persistent images for a structure.

    Args:
        structure (Structure): input structure
        elements (List[List[str]]): list of elements to compute for
        compute_for_all_elements (bool): compute for all elements
        min_size (int): minimum size of the cell for construction of persistent images
        spread (float): spread of kernel for construction
            of persistent images
        weighting (str): weighting scheme for construction
            of persistent images
        pixels (Tuple[int]): size of the image in pixels
        max_b (int): maximum birth time for construction of persistent images
        max_p (int): maximum persistence time for construction of persistent
            images
        periodic (bool): if True (experimental!), use the periodic
            Euclidean distance
        no_supercell (bool): if True, then supercell expansion is not performed.
            The preceeding min_size argument is then ignored.
            Defaults to False.
        alpha_weighting (str, optional): if given use weighted alpha shapes,
            e.g., `atomic_radius_calculated` or `van_der_waals_radius`.
            Defaults to None.

    Returns:
        persistent_images (dict): dictionary of persistent images and their
            barcode representations
    """

    element_images: Dict[dict] = defaultdict(dict)
    specs = []
    for mb, mp in zip(max_b, max_p):
        specs.append({"minBD": 0, "maxB": mb, "maxP": mp})
    for element in elements:
        try:
            filtered_structure = filter_element(structure, element)
            coords, _weights, _elements = _coords_for_structure(
                filtered_structure,
                min_size=min_size,
                periodic=periodic,
                no_supercell=no_supercell,
                weighting=alpha_weighting,
            )
            pd = _pd_arrays_from_coords(coords, periodic=periodic)

            images = get_images(
                pd,
                spread=spread,
                weighting=weighting,
                pixels=pixels,
                specs=specs,
                dimensions=(0, 1, 2),
            )
        except Exception:
            logger.exception(f"Error computing persistent images for {element}")
            images = {}
            for dim in [0, 1, 2]:
                im = np.zeros((pixels[0], pixels[1]))
                im[:] = np.nan
                images[dim] = im
            pd = np.zeros((0, max(max_p) + 1))
            pd[:] = np.nan

        # ToDo: make sure that we have the correct length
        element_images["image"][element] = images
        element_images["array"][element] = pd

    if compute_for_all_elements:
        try:
            coords, weights, _elements = _coords_for_structure(
                structure,
                min_size=min_size,
                periodic=periodic,
                no_supercell=no_supercell,
                weighting=alpha_weighting,
            )
            pd = _pd_arrays_from_coords(coords, periodic=periodic)

            images = get_images(
                pd,
                spread=spread,
                weighting=weighting,
                pixels=pixels,
                specs=specs,
                dimensions=(0, 1, 2),
            )
            element_images["image"]["all"] = images
            element_images["array"]["all"] = pd
        except Exception:
            logger.exception(f"Error computing persistent images for all elements")
            images = {}
            for dim in [0, 1, 2]:
                im = np.zeros((pixels[0], pixels[1]))
                im[:] = np.nan
                images[dim] = im
            pd = np.zeros((0, max(max_p) + 1))
            pd[:] = np.nan

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
        if dgm:
            arr = np.array(
                [[np.sqrt(dgm[i].birth), np.sqrt(dgm[i].death)] for i in range(len(dgm))]
            )

            mask = np.isfinite(arr).all(axis=1)

            arr = arr[mask]
            dgm_arrays[f"dim{dim}"] = arr

        else:
            dgm_arrays[f"dim{dim}"] = np.zeros((0, 2))

    return dgm_arrays


def get_diagrams_for_structure(
    structure,
    elements: List[List[str]],
    compute_for_all_elements: bool = True,
    min_size: int = 20,
    periodic: bool = False,
    no_supercell: bool = False,
    alpha_weighting: Optional[str] = None,
):
    keys = [f"dim{i}" for i in range(3)]
    element_dias = defaultdict(dict)
    nan_array = np.zeros((0, 2))
    nan_array[:] = np.nan
    for element in elements:
        try:
            filtered_structure = filter_element(structure, element)
            coords, weights, _elements = _coords_for_structure(
                filtered_structure,
                min_size=min_size,
                periodic=periodic,
                no_supercell=no_supercell,
                weighting=alpha_weighting,
            )
            arrays = _pd_arrays_from_coords(
                coords, periodic=periodic, bd_arrays=True, weights=weights
            )
        except Exception:
            logger.exception(f"Error for element {element}")
            arrays = {key: nan_array for key in keys}
        if not len(arrays) == 4:
            for key in keys:
                if key not in arrays:
                    arrays[key] = nan_array
        element_dias[element] = arrays

    if compute_for_all_elements:
        coords, weights, _elements = _coords_for_structure(
            structure,
            min_size=min_size,
            periodic=periodic,
            no_supercell=no_supercell,
            weighting=alpha_weighting,
        )
        arrays = _pd_arrays_from_coords(coords, periodic=periodic, bd_arrays=True, weights=weights)
        element_dias["all"] = arrays
        if len(arrays) != 4:
            for key in keys:
                if key not in arrays:
                    arrays[key] = nan_array
    if len(element_dias) != len(elements) + int(compute_for_all_elements):
        raise ValueError("Something went wrong with the diagram extraction.")
    return element_dias


def get_persistence_image_limits_for_structure(
    structure: Structure,
    elements: List[List[str]],
    compute_for_all_elements: bool = True,
    min_size: int = 20,
    periodic: bool = False,
    no_supercell: bool = False,
    alpha_weighting: Optional[str] = None,
) -> dict:
    limits = defaultdict(list)
    for element in elements:
        try:
            filtered_structure = filter_element(structure, element)

            coords, weights, _elements = _coords_for_structure(
                filtered_structure,
                min_size=min_size,
                periodic=periodic,
                no_supercell=no_supercell,
                weighting=alpha_weighting,
            )
            pd = _pd_arrays_from_coords(coords, periodic=periodic, weights=weights)
            for k, v in pd.items():
                limits[k].append(get_min_max_from_dia(v))
        except ValueError:
            logger.exception("Could not extract diagrams for element %s", element)
            pass

    if compute_for_all_elements:
        coords, weights, _elements = _coords_for_structure(
            structure,
            min_size=min_size,
            periodic=periodic,
            no_supercell=no_supercell,
            weighting=alpha_weighting,
        )
        pd = _pd_arrays_from_coords(coords, periodic=periodic, weights=weights)
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
        "birth": {},
        "death": {},
        "persistence": {},
    }

    try:
        d = np.array([[x["birth"], x["death"], x["death"] - x["birth"]] for x in diagram])
    except IndexError:
        d = np.array([[x[0], x[1], x[1] - x[0]] for x in diagram])
    d = np.ma.masked_invalid(d)

    for aggregation in aggregrations:
        agg_func = MA_ARRAY_AGGREGATORS[aggregation]
        for i, key in enumerate(["birth", "death", "persistence"]):
            try:
                stats[key][aggregation] = agg_func(d[:, i])
            except IndexError:
                stats[key][aggregation] = nanfiller
    return stats
