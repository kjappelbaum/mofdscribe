# -*- coding: utf-8 -*-
"""Code taken from the SI for 10.1021/acs.jcim.6b00565"""
from collections import OrderedDict
from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from .rdkitadaptor import RDKitAdaptor


# ToDo: Allow to set some of the options, e.g. for UFF
def _generate_conformers(mol, num_confs):
    # Add H atoms to skeleton
    molecule = Chem.AddHs(mol)
    conformer_integers = []
    # Embed and optimise the conformers
    conformers = AllChem.EmbedMultipleConfs(molecule, num_confs, pruneRmsThresh=0.5, numThreads=5)

    optimised_and_energies = AllChem.MMFFOptimizeMoleculeConfs(
        molecule, maxIters=600, numThreads=5, nonBondedThresh=100.0
    )
    energy_dict_with_key_was_id = {}
    final_conformers_to_use = {}
    # Only keep the conformers which were successfully fully optimised

    for conformer in conformers:
        optimised, energy = optimised_and_energies[conformer]
        if optimised == 0:
            energy_dict_with_key_was_id[conformer] = energy
            conformer_integers.append(conformer)
    # Keep the lowest energy conformer
    lowestenergy = min(energy_dict_with_key_was_id.values())
    for key, value in energy_dict_with_key_was_id.items():
        if value == lowestenergy:
            lowest_energy_conformer_id = key
    final_conformers_to_use[lowest_energy_conformer_id] = lowestenergy

    # Remove H atoms to speed up substructure matching
    molecule = AllChem.RemoveHs(molecule)
    # Find all substructure matches of the molecule with itself,
    # to account for symmetry
    matches = molecule.GetSubstructMatches(molecule, uniquify=False)
    maps = [list(enumerate(match)) for match in matches]
    # Loop over conformers other than the lowest energy one
    for conformer_id, _ in energy_dict_with_key_was_id.items():
        okay_to_add = True
        for finalconformer_id in final_conformers_to_use:
            rms = AllChem.GetBestRMS(molecule, molecule, finalconformer_id, conformer_id, maps)
            if rms < 1.0:
                okay_to_add = False
                break

        if okay_to_add:
            final_conformers_to_use[conformer_id] = energy_dict_with_key_was_id[conformer_id]

    sorted_dictionary = OrderedDict(sorted(final_conformers_to_use.items(), key=lambda t: t[1]))
    energies = list(sorted_dictionary.values())

    return energies


def _calc_nconf20(energy_list):
    energy_descriptor = 0

    relative_energies = np.array(energy_list) - energy_list[0]

    for energy in relative_energies[1:]:
        if 0 <= energy < 20:
            energy_descriptor += 1

    return energy_descriptor


def _n_conf20(mol):
    try:
        energy_list = _generate_conformers(mol, 100)
        descriptor = _calc_nconf20(energy_list)
        return np.array([descriptor])
    except Exception:
        return np.array([np.nan])


class NConf20(RDKitAdaptor):
    """
    Compute the nConf20 descriptor for a molecule.

    This descriptor attempts to capture the flexibility of molecules
    by sampling the conformational space. The descriptor is a count
    of the "accessible" conformers (based on relative conformer energies up to 20 kcal/mol,
    the lowest energy conformer is not counted).
    Conformers are generated using the RDKit conformer generator.

    ... warning::
        Part of the featurization is a geometry optimization using the MMFF
        force field. This will naturally fail for some molecules, and for all metal clusters.
        In these cases, the descriptor will be set to NaN.
    """

    def __init__(self):
        """Construct a new nConf20 featurizer."""
        super().__init__(featurizer=_n_conf20, feature_labels=["n_conf20"])

    def citations(self) -> List[str]:
        return super().citations() + [
            "@article{Wicker2016,"
            "doi = {10.1021/acs.jcim.6b00565},"
            "url = {https://doi.org/10.1021/acs.jcim.6b00565},"
            "year = {2016},"
            "month = dec,"
            "publisher = {American Chemical Society ({ACS})},"
            "volume = {56},"
            "number = {12},"
            "pages = {2347--2352},"
            "author = {Jerome G. P. Wicker and Richard I. Cooper},"
            "title = {Beyond Rotatable Bond Counts: Capturing 3D Conformational Flexibility in a Single Descriptor},"
            "journal = {Journal of Chemical Information and Modeling}"
            "}",
            "@article{tosco2014bringing,"
            "title={Bringing the MMFF force field to the RDKit: implementation and validation},"
            "author={Tosco, Paolo and Stiefl, Nikolaus and Landrum, Gregory},"
            "journal={Journal of cheminformatics},"
            "volume={6},"
            "number={1},"
            "pages={1--4},"
            "year={2014},"
            "publisher={Springer}"
            "}",
        ]

    def implementors(self) -> List[str]:
        return super().implementors() + ["Jerome G. P. Wicker and Richard I. Cooper"]
