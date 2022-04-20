from collections import defaultdict
from pathlib import Path
from typing import List, Union
import numpy as np
from moltda.construct_pd import construct_pds
from moltda.io import dump_json
from moltda.read_file import make_supercell
from moltda.vectorize_pds import diagrams_to_arrays, get_images, pd_vectorization
from pymatgen.core import Structure

from mofdscribe.utils.substructures import elements_in_structure, filter_element

# specs =         "maxB": maxB,
#     "maxP": maxP,
#     "minBD":

# ToDo: only do this for selected elements
# ToDo: only do this for all if we want
def get_persistent_images_for_structure(
    structure: Structure,
    elements: List[List[str]],
    compute_for_all_elements: bool = True,
    min_size: int = 20,
    spread: float = 0.2,
    weighting: str = "identity",
    pixels: List[int] = [50, 50],
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
        pixels (List[int]): size of the image in pixels
    Returns:
        persistent_images (dict): dictionary of persistent images and their barcode representations
    """

    element_images = defaultdict(dict)
    for element in elements:
        try:
            filtered_structure = filter_element(structure, element)
            coords = make_supercell(
                filtered_structure.cart_coords, filtered_structure.lattice.matrix, min_size
            )
            pd = diagrams_to_arrays(construct_pds(coords))

            images = get_images(pd, spread=spread, weighting=weighting, pixels=pixels)
        except ValueError:
            images = np.zeros((0, pixels[0], pixels[1]))

        element_images["image"][element] = images
        element_images["array"][element] = pd

    if compute_for_all_elements:
        coords = make_supercell(structure.cart_coords, structure.lattice.matrix, min_size)
        pd = diagrams_to_arrays(construct_pds(coords))

        images = get_images(pd, spread=spread, weighting=weighting, pixels=pixels)
        element_images["image"]["all"] = images
        element_images["array"]["all"] = pd

    return element_images
