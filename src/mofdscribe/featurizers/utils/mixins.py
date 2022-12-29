# -*- coding: utf-8 -*-
"""Mixin classes for featurizers."""
import numpy as np


class GetGridMixin:
    """Mixin class for getting a linearly spaced grid."""

    def _get_grid(self, lower_bound, upper_bound, bin_size):
        return np.arange(lower_bound, upper_bound, bin_size)
