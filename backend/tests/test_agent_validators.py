"""Unit tests for eo_brain.agent.validators (deterministic guardrails)."""

import logging
from datetime import datetime, timedelta, timezone

import pytest

from eo_brain.agent.validators import (
    NeedsUserError,
    ValidationError,
    clamp_geocoder_bbox,
    estimate_area_km2,
    resolve_algorithm,
    resolve_period,
    resolve_resolution,
    validate_algorithm_compatibility,
    validate_bbox,
    validate_location,
    validate_temporal_range,
)

FLORENCE = [11.15, 43.72, 11.33, 43.83]
NOW = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
UTC = timezone.utc


def period(days_back=None, start=None, end=None, now=NOW):
    return resolve_period(days_back, start, end, now, default_days_back=30)


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


class TestResolvePeriod:
    """The window of a launch: days_back (look-back) or start_date + end_date (calendar)."""

    # -- look-back form: exactly the old window
    def test_default_window(self):
        p = period()
        assert (p.start, p.end, p.days_back) == (NOW - timedelta(days=30), NOW, 30)

    def test_days_back_window(self):
        p = period(90)
        assert (p.start, p.end, p.days_back) == (NOW - timedelta(days=90), NOW, 90)

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_empty_dates_are_not_given(self, blank):
        p = period(7, blank, blank)
        assert (p.start, p.days_back) == (NOW - timedelta(days=7), 7)

    def test_days_back_limits_still_apply(self):
        with pytest.raises(NeedsUserError, match="exceeds maximum"):
            period(730)
        with pytest.raises(ValidationError, match="whole number"):
            period("three months")

    # -- calendar form
    def test_past_period_runs_from_midnight_to_the_end_of_its_last_day(self):
        p = period(start="2025-08-01", end="2025-09-30")
        assert p.start == datetime(2025, 8, 1, tzinfo=UTC)
        assert p.end == datetime(2025, 9, 30, 23, 59, 59, 999999, tzinfo=UTC)
        assert p.days_back is None
        assert (str(p.start)[:10], str(p.end)[:10]) == ("2025-08-01", "2025-09-30")

    def test_period_ending_today_ends_at_the_clock_reading(self):
        p = period(start="2025-12-01", end="2026-01-15")
        assert (p.start, p.end) == (datetime(2025, 12, 1, tzinfo=UTC), NOW)

    def test_single_day(self):
        p = period(start="2025-11-03", end="2025-11-03")
        assert p.start == datetime(2025, 11, 3, tzinfo=UTC)
        assert p.end == datetime(2025, 11, 3, 23, 59, 59, 999999, tzinfo=UTC)
        today = period(start="2026-01-15", end="2026-01-15")
        assert (today.start, today.end) == (datetime(2026, 1, 15, tzinfo=UTC), NOW)

    def test_dates_are_stripped(self):
        assert period(start=" 2025-08-01 ", end="2025-08-31\n").start_date.isoformat() == "2025-08-01"

    @pytest.mark.parametrize(
        "start,end",
        [("2025-01-01", "2026-01-01"),    # 365 days from start to end
         ("2024-01-01", "2024-12-31"),    # a leap calendar year
         ("2025-01-01", "2025-12-31")],
        ids=["span-365", "leap-year", "calendar-year"],
    )
    def test_up_to_365_days_from_start_to_end(self, start, end):
        assert period(start=start, end=end).start_date.isoformat() == start

    def test_366_days_is_refused(self):
        with pytest.raises(NeedsUserError) as exc:
            period(start="2024-12-31", end="2026-01-01")
        assert "2024-12-31 to 2026-01-01 spans 366 days" in str(exc.value)
        assert "maximum of 365 days" in str(exc.value)

    def test_both_forms_together(self):
        with pytest.raises(ValidationError, match="not both") as exc:
            period(30, "2025-08-01", "2025-09-30")
        assert not exc.value.needs_user

    def test_start_without_end_says_today(self):
        with pytest.raises(ValidationError, match=r"end_date is missing.*\(2026-01-15\)") as exc:
            period(start="2025-12-01")
        assert not exc.value.needs_user

    @pytest.mark.parametrize("end", ["2025-09-30", "not a date"])
    def test_end_without_start_needs_the_user(self, end):
        # Checked before the format: any start would be invented.
        with pytest.raises(NeedsUserError, match="no start date"):
            period(start="  ", end=end)

    @pytest.mark.parametrize(
        "bad",
        ["2025-8-1", "01/08/2025", "2025-08-01T00:00:00Z", 20250801, True,
         "２０２５-08-01"],   # full-width digits: \d would accept them
        ids=["no-padding", "dd/mm/yyyy", "datetime", "int", "bool", "non-ascii-digits"],
    )
    @pytest.mark.parametrize("which", ["start_date", "end_date"])
    def test_malformed_date(self, bad, which):
        args = {"start": "2025-08-01", "end": "2025-08-31"}
        args["start" if which == "start_date" else "end"] = bad
        with pytest.raises(ValidationError) as exc:
            period(**args)
        msg = str(exc.value)
        assert not exc.value.needs_user
        assert msg.startswith(which) and "YYYY-MM-DD" in msg and repr(bad) in msg

    def test_day_that_does_not_exist(self):
        with pytest.raises(ValidationError, match="end_date '2025-09-31' is not a real calendar day") as exc:
            period(start="2025-09-01", end="2025-09-31")
        assert "YYYY-MM-DD" in str(exc.value) and not exc.value.needs_user

    def test_start_after_end(self):
        with pytest.raises(ValidationError, match="start_date 2025-09-30 is after end_date 2025-08-01") as exc:
            period(start="2025-09-30", end="2025-08-01")
        assert not exc.value.needs_user

    def test_start_in_the_future_needs_the_user(self):
        with pytest.raises(NeedsUserError, match=r"starts in the future \(today is 2026-01-15\)"):
            period(start="2026-01-16", end="2026-01-31")

    def test_end_in_the_future_is_model_fixable(self):
        with pytest.raises(ValidationError, match=r"end_date 2026-02-28 is in the future \(today is 2026-01-15\)") as exc:
            period(start="2025-12-01", end="2026-02-28")
        assert "today's date (2026-01-15)" in str(exc.value) and not exc.value.needs_user

    def test_order_too_long_wins_over_end_in_the_future(self):
        with pytest.raises(NeedsUserError, match="spans 545 days"):
            period(start="2025-01-01", end="2026-06-30")

    def test_order_reversed_wins_over_future_start(self):
        with pytest.raises(ValidationError, match="is after end_date") as exc:
            period(start="2026-03-01", end="2026-02-01")
        assert not exc.value.needs_user

    def test_today_is_the_utc_date(self):
        local = datetime(2026, 1, 16, 0, 30, tzinfo=timezone(timedelta(hours=2)))  # 2026-01-15 22:30 UTC
        with pytest.raises(ValidationError, match=r"today is 2026-01-15"):
            period(start="2026-01-10", end="2026-01-16", now=local)
        assert period(start="2026-01-10", end="2026-01-15", now=local).end == local


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


class TestRejectionKinds:
    """
    Which rejections only the user can resolve (NeedsUserError: the agent then
    refuses any further job-creating call in the request) and which the model
    may fix with a corrected retry (plain ValidationError).
    """

    @pytest.mark.parametrize(
        "check",
        [
            lambda: validate_temporal_range(366),
            lambda: validate_temporal_range(730),
            lambda: validate_location(""),
            lambda: validate_bbox([179.5, -17.0, -179.5, -16.0]),   # anti-meridian
            lambda: validate_bbox([11.0, 43.0, 11.0005, 43.1]),     # too narrow
            lambda: validate_bbox([11.0, 43.0, 11.002, 43.002]),    # too small
            lambda: validate_bbox([0.0, 40.0, 10.0, 50.0]),         # too large
            lambda: validate_algorithm_compatibility("mndwi", provider="other"),
            lambda: period(start="2027-03-01", end="2027-03-31"),
            lambda: period(start="2024-01-01", end="2025-12-31"),
            lambda: period(end="2025-09-30"),
        ],
        ids=["days-366", "days-730", "empty-location", "antimeridian", "too-narrow",
             "too-small", "too-large", "mndwi-provider", "period-in-future", "period-over-365",
             "end-without-start"],
    )
    def test_needs_user(self, check):
        with pytest.raises(NeedsUserError) as exc:
            check()
        assert exc.value.needs_user is True
        assert isinstance(exc.value, ValidationError)  # existing except clauses still catch it

    @pytest.mark.parametrize(
        "check",
        [
            lambda: resolve_algorithm("vegetation health"),
            lambda: validate_temporal_range(0),
            lambda: validate_temporal_range(90.5),
            lambda: validate_temporal_range("three months"),
            lambda: validate_location("x" * 201),
            lambda: validate_bbox([1, 2, 3]),
            lambda: validate_bbox([10, -95, 11, 45]),
            lambda: validate_bbox([11.15, 43.83, 11.33, 43.72]),
            lambda: period(start="01/08/2025", end="2025-09-30"),
            lambda: period(start="2025-09-01", end="2025-09-31"),
            lambda: period(start="2025-09-30", end="2025-08-01"),
            lambda: period(start="2025-12-01", end="2026-02-28"),
            lambda: period(start="2025-12-01"),
            lambda: period(60, "2025-08-01", "2025-09-30"),
        ],
        ids=["unknown-alias", "days-0", "days-fraction", "days-text", "location-too-long",
             "bbox-length", "bbox-latitude", "bbox-swapped", "date-format", "date-not-a-day",
             "start-after-end", "end-in-future", "start-without-end", "dates-and-days-back"],
    )
    def test_model_fixable(self, check):
        with pytest.raises(ValidationError) as exc:
            check()
        assert not isinstance(exc.value, NeedsUserError)
        assert exc.value.needs_user is False


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
