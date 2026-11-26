"""Unit tests for eo_brain.agent.geocoder with Nominatim mocked (no network)."""

from types import SimpleNamespace

import pytest
from geopy.exc import GeocoderServiceError, GeocoderTimedOut

from eo_brain.agent import geocoder
from eo_brain.agent.validators import estimate_area_km2


class FakeNominatim:
    def __init__(self, responses):
        self.responses = list(responses)
        self.queries = []

    def geocode(self, query, **kwargs):
        self.queries.append((query, kwargs))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _location(lat, lon, *, bbox=None, osm_type="city", address="Somewhere"):
    raw = {"type": osm_type}
    if bbox is not None:
        raw["boundingbox"] = bbox
    return SimpleNamespace(latitude=lat, longitude=lon, raw=raw, address=address)


@pytest.fixture
def fake(monkeypatch):
    geocoder._geocode_cached.cache_clear()

    def install(*responses):
        f = FakeNominatim(responses)
        monkeypatch.setattr(geocoder, "_GEOCODER", f)
        return f

    yield install
    geocoder._geocode_cached.cache_clear()


def test_uses_nominatim_bbox(fake):
    fake(_location(43.77, 11.25, bbox=["43.72", "43.83", "11.15", "11.33"], address="Firenze, Italia"))
    res = geocoder.geocode("Firenze")
    assert res.source == "nominatim_bbox"
    assert res.bbox == (11.15, 43.72, 11.33, 43.83)  # [S, N, W, E] -> (W, S, E, N)
    assert res.name == "Firenze, Italia"
    assert res.to_list() == [11.15, 43.72, 11.33, 43.83]


def test_request_parameters(fake):
    f = fake(_location(43.77, 11.25, bbox=["43.72", "43.83", "11.15", "11.33"]))
    geocoder.geocode("Firenze")
    query, kwargs = f.queries[0]
    assert query == "Firenze"
    assert kwargs == {"exactly_one": True, "language": "en", "addressdetails": True}


def test_synthetic_bbox_without_boundingbox(fake):
    fake(_location(45.0, 10.0, osm_type="city"))
    res = geocoder.geocode("somewhere")
    assert res.source == "synthetic"
    assert res.bbox == pytest.approx((9.95, 44.95, 10.05, 45.05))


def test_region_types_get_wider_padding(fake):
    fake(_location(45.0, 10.0, osm_type="river"))
    res = geocoder.geocode("Po")
    assert res.bbox == pytest.approx((9.85, 44.85, 10.15, 45.15))


def test_unparseable_boundingbox_falls_back(fake):
    fake(_location(45.0, 10.0, bbox=["a", "b", "c", "d"]))
    assert geocoder.geocode("x").source == "synthetic"


def test_degenerate_boundingbox_falls_back(fake):
    fake(_location(45.0, 10.0, bbox=["45.0", "45.0", "10.0", "10.0"]))
    res = geocoder.geocode("point")
    assert res.source == "synthetic"
    assert res.bbox[0] < res.bbox[2] and res.bbox[1] < res.bbox[3]


def test_oversized_bbox_is_clamped(fake):
    fake(_location(45.0, 5.0, bbox=["40.0", "50.0", "0.0", "10.0"]))
    res = geocoder.geocode("big region")
    assert estimate_area_km2(list(res.bbox)) <= 50000


def test_no_result_returns_none_and_is_cached(fake):
    f = fake(None)
    assert geocoder.geocode("nowhere") is None
    assert geocoder.geocode("nowhere") is None
    assert len(f.queries) == 1


def test_success_is_cached(fake):
    f = fake(_location(43.77, 11.25, bbox=["43.72", "43.83", "11.15", "11.33"]))
    geocoder.geocode("Firenze")
    geocoder.geocode("Firenze")
    assert len(f.queries) == 1


@pytest.mark.parametrize("exc", [GeocoderTimedOut("slow"), GeocoderServiceError("503")])
def test_transient_errors_return_none_and_are_not_cached(fake, exc):
    f = fake(exc, _location(43.77, 11.25, bbox=["43.72", "43.83", "11.15", "11.33"]))
    assert geocoder.geocode("Firenze") is None
    assert geocoder.geocode("Firenze") is not None  # retried, not served from cache
    assert len(f.queries) == 2


def test_geocode_or_fail_raises_value_error(fake):
    fake(None)
    with pytest.raises(ValueError, match="Could not geocode 'nowhere'"):
        geocoder.geocode_or_fail("nowhere")
