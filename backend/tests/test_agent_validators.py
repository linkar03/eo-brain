"""Unit tests for eo_brain.agent.validators (deterministic guardrails)."""

import logging

import pytest

from eo_brain.agent.validators import (
    ValidationError,
    clamp_geocoder_bbox,
    estimate_area_km2,
    resolve_algorithm,
    resolve_resolution,
    validate_algorithm_compatibility,
    validate_bbox,
    validate_location,
    validate_temporal_range,
)

FLORENCE = [11.15, 43.72, 11.33, 43.83]


class TestResolveAlgorithm:
    @pytest.mark.parametrize("name", ["ndvi", "ndwi", "mndwi", "lisa", "flood"])
    def test_canonical_names(self, name):
        assert resolve_algorithm(name) == name

    def test_case_and_whitespace_insensitive(self):
        assert resolve_algorithm("  NDVI ") == "ndvi"

    @pytest.mark.parametrize(
        "alias,expected",
        [
            ("vegetation", "ndvi"), ("vegetazione", "ndvi"), ("verde", "ndvi"),
            ("water", "ndwi"), ("acqua", "ndwi"),
            ("alluvione", "flood"), ("flooding", "flood"), ("esondazione", "flood"),
            ("drought", "lisa"), ("siccita", "lisa"), ("anomaly", "lisa"), ("anomalia", "lisa"),
        ],
    )
    def test_aliases_it_en(self, alias, expected):
        assert resolve_algorithm(alias) == expected

    def test_unknown_lists_available(self):
        with pytest.raises(ValidationError) as exc:
            resolve_algorithm("traffic")
        msg = str(exc.value)
        assert "traffic" in msg
        for name in ["flood", "lisa", "mndwi", "ndvi", "ndwi"]:
            assert name in msg


class TestValidateLocation:
    def test_strips(self):
        assert validate_location("  Firenze ") == "Firenze"

    @pytest.mark.parametrize("bad", ["", "   ", None, 42])
    def test_empty_or_not_text(self, bad):
        with pytest.raises(ValidationError, match="empty"):
            validate_location(bad)

    def test_too_long(self):
        with pytest.raises(ValidationError, match="too long"):
            validate_location("x" * 201)


class TestTemporalRange:
    @pytest.mark.parametrize("days", [1, 30, 365])
    def test_bounds_inclusive(self, days):
        assert validate_temporal_range(days) == days

    @pytest.mark.parametrize("days", [0, -5])
    def test_below_minimum(self, days):
        with pytest.raises(ValidationError, match=">= 1"):
            validate_temporal_range(days)

    @pytest.mark.parametrize("days", [366, 730, 1825])
    def test_above_maximum_is_rejected_not_clamped(self, days):
        with pytest.raises(ValidationError, match="exceeds maximum"):
            validate_temporal_range(days)

    @pytest.mark.parametrize("raw,expected", [(90.0, 90), ("90", 90), (" 7 ", 7)])
    def test_whole_numbers_in_other_types(self, raw, expected):
        assert validate_temporal_range(raw) == expected

    @pytest.mark.parametrize("raw", [90.5, "three months", True, None, [90]])
    def test_non_whole_numbers_rejected(self, raw):
        with pytest.raises(ValidationError, match="whole number"):
            validate_temporal_range(raw)


class TestBBox:
    def test_valid_passthrough(self):
        assert validate_bbox(list(FLORENCE)) == pytest.approx(FLORENCE)

    def test_wrong_length(self):
        with pytest.raises(ValidationError, match="exactly 4"):
            validate_bbox([1, 2, 3])

    def test_latitude_out_of_range(self):
        with pytest.raises(ValidationError, match="Latitude out of range"):
            validate_bbox([10, -95, 11, 45])

    def test_swapped_latitudes(self):
        with pytest.raises(ValidationError, match="swapped"):
            validate_bbox([11.15, 43.83, 11.33, 43.72])

    def test_antimeridian_rejected(self):
        with pytest.raises(ValidationError, match="Anti-meridian"):
            validate_bbox([179.5, -17.0, -179.5, -16.0])

    def test_longitude_normalised(self):
        west, south, east, north = validate_bbox([371.15, 43.72, 371.33, 43.83])
        assert west == pytest.approx(11.15)
        assert east == pytest.approx(11.33)

    def test_too_narrow(self):
        with pytest.raises(ValidationError, match="too narrow"):
            validate_bbox([11.0, 43.0, 11.0005, 43.1])

    def test_too_small(self):
        with pytest.raises(ValidationError, match="Area too small"):
            validate_bbox([11.0, 43.0, 11.002, 43.002])

    def test_too_large(self):
        with pytest.raises(ValidationError, match="Area too large"):
            validate_bbox([0.0, 40.0, 10.0, 50.0])

    def test_utm_crossing_only_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="eo_brain.agent.validators"):
            assert validate_bbox([11.9, 43.0, 12.1, 43.2])
        assert "crosses UTM zones" in caplog.text


class TestArea:
    def test_one_degree_at_equator(self):
        assert estimate_area_km2([0, 0, 1, 1]) == pytest.approx(111 * 111, rel=1e-3)

    def test_shrinks_with_latitude(self):
        assert estimate_area_km2([0, 60, 1, 61]) < estimate_area_km2([0, 0, 1, 1]) / 1.9


class TestResolution:
    def test_small_area_keeps_10m(self):
        assert resolve_resolution([11.20, 43.75, 11.25, 43.80]) == 10

    def test_city_scale_coarsens(self):
        assert resolve_resolution(FLORENCE) == 20  # ~176 km2 -> 200 km2 tier

    def test_requested_coarser_than_tier_wins(self):
        assert resolve_resolution([11.20, 43.75, 11.25, 43.80], requested=60) == 60

    def test_regional_scale(self):
        assert resolve_resolution([10.0, 43.0, 11.0, 44.0]) == 100

    def test_beyond_last_tier_falls_back(self):
        assert resolve_resolution([0.0, 40.0, 10.0, 50.0]) == 60


class TestCompatibility:
    def test_mndwi_ok_on_sentinel2(self):
        assert validate_algorithm_compatibility("mndwi") == "mndwi"

    def test_mndwi_rejected_on_unknown_provider(self):
        with pytest.raises(ValidationError, match="SWIR"):
            validate_algorithm_compatibility("mndwi", provider="other")

    def test_other_algorithms_pass(self):
        assert validate_algorithm_compatibility("ndvi", provider="other") == "ndvi"


class TestClampGeocoderBBox:
    def test_small_bbox_unchanged(self):
        assert clamp_geocoder_bbox(tuple(FLORENCE)) == FLORENCE

    def test_large_bbox_shrunk_around_centroid(self):
        big = (0.0, 40.0, 10.0, 50.0)
        w, s, e, n = clamp_geocoder_bbox(big, max_area_km2=1000)
        assert estimate_area_km2([w, s, e, n]) == pytest.approx(950, rel=0.02)
        assert (w + e) / 2 == pytest.approx(5.0)
        assert (s + n) / 2 == pytest.approx(45.0)

    def test_clamped_bbox_passes_validation(self):
        clamped = clamp_geocoder_bbox((12.37, 36.64, 15.65, 38.32))  # Sicily, > 50000 km2
        assert validate_bbox(clamped)
