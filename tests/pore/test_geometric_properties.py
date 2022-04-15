from mofdscribe.pore.geometric_properties import _parse_res_zeopp, _parse_sa_zeopp

SA_SAMPLE_OUTPUT = """@ EDI.sa Unitcell_volume: 307.484   Density: 1.62239   

ASA_A^2: 60.7713 ASA_m^2/cm^3: 1976.4 ASA_m^2/g: 1218.21 

NASA_A^2: 0 NASA_m^2/cm^3: 0 NASA_m^2/g: 0"""


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
