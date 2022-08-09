from mofdscribe.bench.watermark import get_watermark


def test_get_watermark():
    watermark = get_watermark()
    assert isinstance(watermark, dict)
    assert watermark["packages"] is not None
    assert watermark["system"] is not None
