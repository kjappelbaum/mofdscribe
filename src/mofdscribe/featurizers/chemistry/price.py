# -*- coding: utf-8 -*-
"""Feature MOFs using economic descriptors."""
import operator
import os
from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import pandas as pd

from mofdscribe.featurizers.base import MOFBaseFeaturizer

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

_ELEMENT_PRICES = pd.read_csv(os.path.join(_THIS_DIR, "data", "elementprices.csv"), comment="#")


def _get_element_prices():
    """Return element prices dictionary."""
    prices_dict = {}

    for _i, row in _ELEMENT_PRICES.iterrows():
        # clean 'price' column
        price = row["Price [USD/kg]"]
        if isinstance(price, str) and "–" in price:
            prices = price.split("–")
            price = (float(prices[0]) + float(prices[1])) / 2
        elif price in ["Not traded.", "No reliable price available."]:
            price = np.nan
        elif price == "Negative price.":
            price = 0

        prices_dict[row["Symbol"]] = float(price)

    return prices_dict


def _get_element_abundance():
    """Return element abundance dictionary."""
    abundance_dict = {}

    for _i, row in _ELEMENT_PRICES.iterrows():
        abundance = row["Abundance in Earth's crust [mg/kg]"]

        if isinstance(abundance, str):
            abundance = abundance.strip("[i]")

            abundance = abundance.strip("~")
            abundance = abundance.strip("≤")
            abundance = abundance.strip()

        abundance_dict[row["Symbol"]] = float(abundance) * 1e-6

    return abundance_dict


_ELEMENT_PRICES_DICT = _get_element_prices()


def _gravimetric_price(element_price_fractions_kg, structure):
    return sum(element_price_fractions_kg.values())


def _volumetric_price(element_price_fractions_kg, structure):
    kg_price = _gravimetric_price(element_price_fractions_kg, structure)
    density = structure.density
    return kg_price * density  # 1 g / cm3 = 1kg / L


def _price_per_atom(element_price_fractions_kg, structure):
    return (
        _gravimetric_price(element_price_fractions_kg, structure)
        * (structure.composition.weight)
        / structure.num_sites
    )


class PriceLowerBound(MOFBaseFeaturizer):
    """Compute a lower bound for the price based on the element prices.

    Obviously, this featurization does well in highlighting rare (and expensive)
    metals.
    It can be useful as additional information in a screening but might
    also be an interesting way to describe the composition of a material
    with one number.

    Price data from `Wikipedia (2020-09-08) <https://en.wikipedia.org/wiki/Prices_of_chemical_elements>`_.

    The units of the projections are:

    - gravimetric: USD / kg
    - volumetric: USD / L
    - per atom: USD / atom

    The implementation is based on some code from `Leopold Talirz <https://ltalirz.github.io/>`_.
    """

    _PROJECTIONS = {
        "gravimetric": _gravimetric_price,
        "volumetric": _volumetric_price,
        "per_atom": _price_per_atom,
    }

    def __init__(
        self, projections: Tuple[str] = ("gravimetric", "volumetric"), primitive: bool = False
    ):
        """Initialize the PriceLowerBound featurizer.

        Args:
            projections (Tuple[str]): List of projections to use.
                Possible values are "gravimetric" and "volumetric".
                Default is ("gravimetric", "volumetric").
            primitive (bool): If True, the structure is reduced to its primitive
                form before the descriptor is computed. Defaults to False.
        """
        self.projections = projections
        self.element_masses = _get_element_abundance()

        self.projections = projections
        super().__init__(primitive=primitive)

    def feature_labels(self) -> List[str]:
        labels = []
        for projection in self.projections:
            labels.append(f"price_lower_bound_{projection}")
        return labels

    def _featurize(self, structure) -> np.ndarray:
        element_masses = {}

        for site in structure.sites:
            element_masses[site.specie.symbol] = site.specie.atomic_mass

        total_mass = sum(element_masses.values())
        # mapping from symbol to price / 1 kg of MOF
        tuples = [
            (symbol, mass / total_mass * _ELEMENT_PRICES_DICT[symbol])
            for symbol, mass in element_masses.items()
        ]
        element_price_fractions_kg = OrderedDict(
            sorted(tuples, key=operator.itemgetter(1), reverse=True)
        )

        features = []
        for projection in self.projections:
            features.append(self._PROJECTIONS[projection](element_price_fractions_kg, structure))

        return np.array(features)

    def citations(self) -> List[str]:
        return []

    def implementors(self) -> List[str]:
        return ["Leopold Talirz", "Kevin Maik Jablonka"]
