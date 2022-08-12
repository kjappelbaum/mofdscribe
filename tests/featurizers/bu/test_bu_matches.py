# -*- coding: utf-8 -*-
"""Test the BB mistmatch measure module."""
import json

from pymatgen.core import Structure

from mofdscribe.featurizers.bu.bu_matches import BUMatch, match_bb

HKUST_node = '{"@module": "pymatgen.core.structure", "@class": "Structure", "charge": null, "lattice": {"matrix": [[5.552462033529595, 0.0, 3.399902428374611e-16], [-3.399902428374611e-16, 5.552462033529595, 3.399902428374611e-16], [0.0, 0.0, 5.552462033529595]], "a": 5.552462033529595, "b": 5.552462033529595, "c": 5.552462033529595, "alpha": 90.0, "beta": 90.0, "gamma": 90.0, "volume": 171.1814863041015}, "sites": [{"species": [{"element": "C", "occu": 1}], "abc": [1.40908140438495, 0.9631095120880301, 0.32736234647327106], "xyz": [7.823870999999998, 5.3476289999999995, 1.8176669999999995], "label": "C", "properties": {}}, {"species": [{"element": "C", "occu": 1}], "abc": [1.4090814043849502, 0.9631095120880301, -0.32736234647327106], "xyz": [7.823870999999999, 5.3476289999999995, -1.8176669999999977], "label": "C", "properties": {}}, {"species": [{"element": "C", "occu": 1}], "abc": [0.9631095120880304, 1.40908140438495, -0.32736234647327106], "xyz": [5.347629000000001, 7.823870999999998, -1.8176669999999977], "label": "C", "properties": {}}, {"species": [{"element": "C", "occu": 1}], "abc": [0.9631095120880301, 1.40908140438495, 0.32736234647327106], "xyz": [5.347628999999999, 7.823870999999998, 1.8176669999999995], "label": "C", "properties": {}}], "@version": "2021.3.9"}'  # noqa: E501


def test_match_bb():
    """Ensure that the basic unit of BB matching is working."""
    s = Structure.from_dict(json.loads(HKUST_node))
    results = match_bb(s, "tbo", aggregations=("min", "max", "mean", "std"))
    assert len(results) == 4
    assert results["tbo_min"] < 4
    assert results["tbo_max"] == results["tbo_min"]

    results = match_bb(s, "pcu", aggregations=("min", "max", "mean", "std"))
    assert len(results) == 4
    assert results["pcu_min"] == 10_000


def test_bu_match():
    """Ensure that our BU matching returns reasonable results for HKUST."""
    bu_matcher = BUMatch()
    s = Structure.from_dict(json.loads(HKUST_node))
    feats = bu_matcher.featurize(s)
    feature_labels = bu_matcher.feature_labels()
    matching = feats[feats < 1_000]
    assert len(matching) > 4
    assert len(matching) < len(feats)
    assert len(feats) == len(feature_labels)
