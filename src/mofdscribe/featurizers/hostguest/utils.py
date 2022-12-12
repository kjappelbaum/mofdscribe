# -*- coding: utf-8 -*-
"""Helper functions for the host-guest featurizers."""
from typing import List, Optional, Union

from pydantic import BaseModel
from pymatgen.core import IMolecule, IStructure, Molecule, Structure
from structuregraph_helpers.subgraph import get_subgraphs_as_molecules

from mofdscribe.featurizers.bu.utils import boxed_molecule
from mofdscribe.featurizers.utils.structure_graph import get_structure_graph


def remove_guests_from_structure(structure, guest_indices: List[List[int]]) -> Structure:
    """
    Remove guests from a structure.

    Args:
        structure (Structure): The structure to remove guests from.
        guest_indices (List[List[int]]): The indices of the guests.

    Returns:
        Structure: The structure without guests.
    """
    flattened_guest_indices = sum(guest_indices, [])
    ok_sites = []
    for i, site in enumerate(structure):
        if i not in flattened_guest_indices:
            ok_sites.append(site)

    return Structure.from_sites(ok_sites)


class HostGuest(BaseModel):
    """Container for host and guests."""

    host: Union[Structure, IStructure]
    guests: Optional[List[Union[Structure, Molecule, IStructure, IMolecule]]]


def _extract_host_guest(
    structure: Optional[Union[Structure, IStructure]] = None,
    host_guest: Optional[HostGuest] = None,
    operates_on: str = "structure",
    remove_guests: bool = True,
    local_env_method: str = "vesta",
):
    if structure is None and host_guest is None:
        raise ValueError("You must provide a structure or host_guest.")

    if structure is not None:
        if not isinstance(structure, (IStructure)):
            structure = IStructure.from_sites(structure.sites)
        structure_graph = get_structure_graph(structure, local_env_method)
        mols, _mol_graphs, mol_indices, _centers, _coordinates = get_subgraphs_as_molecules(
            structure_graph
        )
        if remove_guests:
            host = remove_guests_from_structure(structure_graph.structure, mol_indices)
        else:
            host = structure_graph.structure

        if operates_on == "structure":
            mols = [boxed_molecule(mol) for mol in mols]

    else:
        host = host_guest.host
        mols = host_guest.guests
        if operates_on == "structure" and isinstance(mols[0], (Molecule, IMolecule)):
            mols = [boxed_molecule(mol) for mol in mols]

    return HostGuest(host=host, guests=mols)
