from mofdscribe.pore.geometric_properties import (
    _parse_res_zeopp,
    _parse_sa_zeopp,
    PoreDiameters,
    SurfaceArea,
)

SA_SAMPLE_OUTPUT = """@ EDI.sa Unitcell_volume: 307.484   Density: 1.62239   

ASA_A^2: 60.7713 ASA_m^2/cm^3: 1976.4 ASA_m^2/g: 1218.21 

NASA_A^2: 0 NASA_m^2/cm^3: 0 NASA_m^2/g: 0"""


RES_SAMPLE_OUTPUT = "output_file.res    1.70107 0.95106  1.64805"


def test_parse_sa():
    parsed = _parse_sa_zeopp(SA_SAMPLE_OUTPUT)
    assert parsed["unitcell_volume"] == 307.484
    assert parsed["density"] == 1.62239
    assert parsed["asa_a2"] == 60.7713
    assert parsed["asa_m2cm3"] == 1976.4
    assert parsed["asa_m2g"] == 1218.21
    assert parsed["nasa_a2"] == 0
    assert parsed["nasa_m2cm3"] == 0
    assert parsed["nasa_m2g"] == 0


def test_parse_res():
    res = _parse_res_zeopp(RES_SAMPLE_OUTPUT)
    assert res == {
        "lis": 1.70107,  # largest included sphere
        "lifs": 0.95106,  # largest free sphere
        "lifsp": 1.64805,  # largest included sphere along free sphere path
    }


def test_pore_diameters(hkust_structure):
    pd = PoreDiameters()
    assert pd.feature_labels() == ["lis", "lifs", "lifsp"]
    result = pd.featurize(hkust_structure)
    assert len(result) == 3
    assert result[0] == 13.21425
    assert result[1] == 6.66829
    assert result[2] == 13.21425


def test_surface_area(hkust_structure):
    sa = SurfaceArea()
    assert sa.feature_labels() == [
        "uc_volume",
        "density",
        "asa_a2",
        "asa_m2cm3",
        "asa_m2g",
        "nasa_a2",
        "nasa_m2cm3",
        "nasa_m2g",
    ]
    result = sa.featurize(hkust_structure)
    print(result)
    assert len(result) == 8

    assert result[0] == 1.82808e04
    assert result[1] == 8.79097e-01
    assert result[2] == 5.13510e03
    assert result[3] == 2.80901e03
    assert result[4] == 3.19533e03
    assert result[5] == 0
    assert result[6] == 0
    assert result[7] == 0
