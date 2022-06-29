# -*- coding: utf-8 -*-
"""Helper functions for obtaining the correct number of replicas"""
from math import ceil, cos, fabs, hypot, pi, sin
from typing import Tuple

import numpy as np
from pymatgen.core import Structure


def resize_unit_cell(structure: Structure, threshold: float) -> Tuple[int, int, int]:
    """Return the multiplication factors for the cell vectors.

    In every direction: min(perpendicular_width) > threshold.

    Args:
        structure (Structure): pymatgen Structure
        threshold (float): threshold for the minimum width of the cell

    Returns:
        Tuple[int, int, int]: Multiplication factors for the cell vectors.
    """
    deg2rad = pi / 180.0

    a_len = structure.lattice.a
    b_len = structure.lattice.b
    c_len = structure.lattice.c

    alpha = structure.lattice.alpha * deg2rad
    beta = structure.lattice.beta * deg2rad
    gamma = structure.lattice.gamma * deg2rad

    # Computing triangular cell matrix
    vol = np.sqrt(
        1
        - cos(alpha) ** 2
        - cos(beta) ** 2
        - cos(gamma) ** 2
        + 2 * cos(alpha) * cos(beta) * cos(gamma)
    )
    cell = np.zeros((3, 3))
    cell[0, :] = [a_len, 0, 0]
    cell[1, :] = [b_len * cos(gamma), b_len * sin(gamma), 0]
    cell[2, :] = [
        c_len * cos(beta),
        c_len * (cos(alpha) - cos(beta) * cos(gamma)) / (sin(gamma)),
        c_len * vol / sin(gamma),
    ]
    cell = np.array(cell)

    # Computing perpendicular widths, as implemented in Raspa
    # for the check (simplified for triangular cell matrix)
    axc1 = cell[0, 0] * cell[2, 2]
    axc2 = -cell[0, 0] * cell[2, 1]
    bxc1 = cell[1, 1] * cell[2, 2]
    bxc2 = -cell[1, 0] * cell[2, 2]
    bxc3 = cell[1, 0] * cell[2, 1] - cell[1, 1] * cell[2, 0]
    det = fabs(cell[0, 0] * cell[1, 1] * cell[2, 2])
    perpwidth = np.zeros(3)
    perpwidth[0] = det / hypot(bxc1, bxc2, bxc3)
    perpwidth[1] = det / hypot(axc1, axc2)
    perpwidth[2] = cell[2, 2]

    # prevent from crashing if threshold value is zero
    thr = max(0.001, threshold)

    return (
        int(ceil(thr / perpwidth[0])),
        int(ceil(thr / perpwidth[1])),
        int(ceil(thr / perpwidth[2])),
    )
