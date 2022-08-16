# -*- coding: utf-8 -*-
"""Test the geometric properties featurizers."""

from pytest import approx

from mofdscribe.featurizers.pore.geometric_properties import (
    AccessibleVolume,
    PoreDiameters,
    PoreSizeDistribution,
    RayTracingHistogram,
    SurfaceArea,
    _parse_res_zeopp,
    _parse_sa_zeopp,
    _parse_volpo_zeopp,
)

from ..helpers import is_jsonable

SA_SAMPLE_OUTPUT = """@ EDI.sa Unitcell_volume: 307.484   Density: 1.62239

ASA_A^2: 60.7713 ASA_m^2/cm^3: 1976.4 ASA_m^2/g: 1218.21

NASA_A^2: 0 NASA_m^2/cm^3: 0 NASA_m^2/g: 0"""

VOLPO_SAMPLE_OUTPUT = """@ EDI.vol Unitcell_volume: 307.484   Density: 1.62239

AV_A^3: 22.6493 AV_Volume_fraction: 0.07366 AV_cm^3/g: 0.0454022

NAV_A^3: 0 NAV_Volume_fraction: 0 NAV_cm^3/g: 0"""


RES_SAMPLE_OUTPUT = "output_file.res    1.70107 0.95106  1.64805"


def test_parse_sa():
    """Ensure that the parser works as expected."""
    parsed = _parse_sa_zeopp(SA_SAMPLE_OUTPUT)
    assert parsed["unitcell_volume"] == approx(307.484, 0.1)
    assert parsed["density"] == approx(1.62239, 0.1)
    assert parsed["asa_a2"] == approx(60.7713, 0.1)
    assert parsed["asa_m2cm3"] == approx(1976.4, 0.1)
    assert parsed["asa_m2g"] == approx(1218.21, 0.1)
    assert parsed["nasa_a2"] == 0
    assert parsed["nasa_m2cm3"] == 0
    assert parsed["nasa_m2g"] == 0


def test_parse_res():
    """Ensure that the parser works as expected."""
    res = _parse_res_zeopp(RES_SAMPLE_OUTPUT)
    assert res == {
        "lis": 1.70107,  # largest included sphere
        "lifs": 0.95106,  # largest free sphere
        "lifsp": 1.64805,  # largest included sphere along free sphere path
    }


def test_parse_volpo_zeopp():
    """Ensure that the parser works as expected."""
    res = _parse_volpo_zeopp(VOLPO_SAMPLE_OUTPUT)
    assert res["unitcell_volume"] == approx(307.484, 0.1)
    assert res["density"] == approx(1.62239, 0.1)
    # assert res["av_a3"] == approx(22.6493, 0.1)
    assert res["av_volume_fraction"] == approx(0.07366, 0.1)
    assert res["av_cm3g"] == approx(0.0454022, 0.1)
    assert res["nav_a3"] == 0
    assert res["nav_volume_fraction"] == 0
    assert res["nav_cm3g"] == 0


def test_pore_diameters(hkust_structure):
    """Ensure that the featurizer works as expected."""
    pd = PoreDiameters()
    assert pd.feature_labels() == ["lis", "lifs", "lifsp"]
    result = pd.featurize(hkust_structure)
    assert len(result) == 3
    assert result[0] == approx(13.21425, 0.1)
    assert result[1] == approx(6.66829, 0.1)
    assert result[2] == approx(13.21425, 0.1)
    assert is_jsonable(dict(zip(pd.feature_labels(), result)))
    assert result.ndim == 1


def test_surface_area(hkust_structure):
    """Ensure that the featurizer works as expected."""
    sa = SurfaceArea()
    expected_labels = [
        "uc_volume",
        "density",
        "asa_a2",
        "asa_m2cm3",
        "asa_m2g",
        "nasa_a2",
        "nasa_m2cm3",
        "nasa_m2g",
    ]

    for el, fl in zip(expected_labels, sa.feature_labels()):
        assert el in fl

    result = sa.featurize(hkust_structure)
    assert len(result) == 8

    assert result[0] == approx(4570.21, 0.1)
    assert result[1] == approx(8.79097e-01, 0.1)
    # assert result[2] == approx(5.13510e03, 0.1)
    assert result[3] == approx(2.80901e03, 0.1)
    assert result[4] == approx(3.19533e03, 0.1)
    assert result[5] == 0
    assert result[6] == 0
    assert result[7] == 0
    assert is_jsonable(dict(zip(sa.feature_labels(), result)))
    assert result.ndim == 1


def test_accessible_volume(hkust_structure):
    """Ensure that the featurizer works as expected."""
    av = AccessibleVolume()
    expected_labels = [
        "uc_volume",
        "density",
        "av_a2",
        "av_volume_fraction",
        "av_cm3g",
        "nav_a3",
        "nav_volume_fraction",
        "nav_cm3g",
    ]

    for el, fl in zip(expected_labels, av.feature_labels()):
        assert el in fl

    result = av.featurize(hkust_structure)
    assert len(result) == 8
    assert result[0] == approx(4570.21, 0.1)
    assert result[1] == approx(8.79097e-01, 0.1)
    assert result[3] == approx(7.40000e-01, 0.2)
    assert result[4] == approx(8.41773e-01, 0.2)
    assert result[5] == 0
    assert result[6] == 0
    assert result[7] == 0
    assert is_jsonable(dict(zip(av.feature_labels(), result)))
    assert result.ndim == 1


def test_raytracing_histogram(hkust_structure):
    """Ensure that the featurizer works as expected."""
    rth = RayTracingHistogram()
    assert len(rth.feature_labels()) == 1000
    assert len(rth.citations()) == 2
    features = rth.featurize(hkust_structure)
    assert len(features) == 1000
    assert features[0] >= 1.0
    assert is_jsonable(dict(zip(rth.feature_labels(), features)))
    assert features.ndim == 1


def test_psd(hkust_structure):
    """Ensure that the featurizer works as expected."""
    psd = PoreSizeDistribution()
    assert len(psd.feature_labels()) == 1000
    assert len(psd.citations()) == 2
    features = psd.featurize(hkust_structure)
    assert len(features) == 1000
    assert features[0] == 0
    assert is_jsonable(dict(zip(psd.feature_labels(), features)))
    assert features.ndim == 1
