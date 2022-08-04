# -*- coding: utf-8 -*-
import pytest

from mofdscribe.featurizers.chemistry.price import PriceLowerBound


def test_price_lower_bound(hkust_structure):
    """Comparing with the original implementation."""
    pricer = PriceLowerBound()
    feats = pricer.featurize(hkust_structure)
    assert len(feats) == 2
    assert feats[0] == pytest.approx(4.176635436396251)
    assert feats[1] == pytest.approx(3.671662426852288)

    pricer = PriceLowerBound(("per_atom",))
    feats = pricer.featurize(hkust_structure)
    assert len(feats) == 1
    assert feats[0] == pytest.approx(64.77758513112447)
