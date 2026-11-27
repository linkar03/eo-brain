"""
eo_brain/agent/validators.py - Guardrails for agent tool inputs.

Every tool parameter passes through validation before touching
the DB or processing pipeline. Errors are human-readable so
Gemini can relay them to the user as actionable suggestions.

Every rejection is one of two kinds, and the agent loop treats them
differently (see ``brain.Agent._execute_tool_call``):

* ``ValidationError``  the model can fix it on its own: an alias it
  spelled wrong, a number or date in the wrong format, a place name to
  correct. A corrected retry in the same request is allowed.
* ``NeedsUserError``   only the user can resolve it: the request is past
  a limit (window too long, a period in the future, area too large or
  too small, ...). Any value the model could retry with would be a
  different request from the one the user made, so once one is raised no
  further job-creating call runs in that request.
"""

from __future__ import annotations

import math
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, List, Optional, Tuple

from eo_brain.algorithms.base import ALGORITHMS as ALGO_REGISTRY

logger = logging.getLogger(__name__)

_ALLOWED_ALGORITHMS = frozenset(ALGO_REGISTRY.keys())

_MAX_AREA_KM2 = 50000  # effectively unlimited for demo
_MIN_AREA_KM2 = 0.5
_MAX_DAYS_BACK = 365
_MIN_DAYS_BACK = 1
_MAX_LOCATION_CHARS = 200

# Resolution tiers based on area size.
# Aggressive downscaling for demo speed -- at typical zoom levels
# the visual difference between 10m and 20m is imperceptible,
# but the processing time difference is 4x.
_RESOLUTION_TIERS = [
    (50, 10),
    (200, 20),
    (600, 30),
    (2500, 60),
    (10000, 100),   # regional scale
    (50000, 100),   # country scale
]

_SWIR_ALGORITHMS = {"mndwi", "flood"}

_ALGO_BAND_REQUIREMENTS = {
    "ndvi": {"nir", "red"},
    "ndwi": {"green", "nir"},
    "mndwi": {"green", "swir16"},
    "flood": {"green", "nir"},
    "lisa": {"nir", "red"},
}

_ALGO_ALIASES = {
    "vegetation": "ndvi", "vegetazione": "ndvi", "health": "ndvi",
    "salute": "ndvi", "greenness": "ndvi", "verde": "ndvi",
    "water": "ndwi", "acqua": "ndwi",
    "flood": "flood", "alluvione": "flood", "flooding": "flood",
    "inondazione": "flood", "esondazione": "flood",
    "drought": "lisa", "stress": "lisa", "siccita": "lisa",
    "anomaly": "lisa", "anomalia": "lisa", "outlier": "lisa",
    "mndwi": "mndwi",
}


class ValidationError(Exception):
    """
    Raised with a user-friendly message the agent can relay.

    This base class is the kind the MODEL can fix with a corrected call.
    """

    needs_user = False


class NeedsUserError(ValidationError):
    """
    A rejection only the USER can resolve: the request itself is outside
    what the system does. The model must relay it and ask, not retry with
    other values.
    """

    needs_user = True


def resolve_algorithm(raw: str) -> str:
    key = raw.strip().lower()
    if key in _ALLOWED_ALGORITHMS:
        return key
    resolved = _ALGO_ALIASES.get(key)
    if resolved:
        return resolved
    # Model-fixable (a misspelt or unlisted alias), so a plain ValidationError.
    # Code cannot tell that from a request for something no analysis measures
    # (traffic, weather): not substituting a supported analysis for those is a
    # system-prompt rule (brain.SYSTEM_PROMPT), not an enforced one.
    raise ValidationError(
        f"Unknown algorithm '{raw}'. Available: {', '.join(sorted(_ALLOWED_ALGORITHMS))}. "
        f"You can also say: vegetation, water, flood, drought."
    )


def estimate_area_km2(bbox: List[float]) -> float:
    west, south, east, north = bbox
    lat_mid = (south + north) / 2
    width_km = (east - west) * 111 * math.cos(math.radians(lat_mid))
    height_km = (north - south) * 111
    return abs(width_km * height_km)


def _lon_to_utm_zone(lon: float) -> int:
    return int((lon + 180) / 6) + 1


def validate_bbox(bbox: List[float], warn: bool = True) -> List[float]:
    """
    Full bbox validation with actionable error messages.

    The bbox comes from the geocoder, never from the model. Two kinds:
    * malformed geocoder output (wrong length, latitude out of range,
      south >= north): plain ValidationError, treated like a failed
      geocoding (the model may try a corrected place name). The real
      geocoder cannot produce these: it always returns 4 values, clamps
      latitudes and replaces zero-area boxes with a padded one;
    * the extent of the area the user named is outside the limits
      (anti-meridian, too narrow, too small, too large): NeedsUserError,
      because any retry would analyse a different area.
    """
    if len(bbox) != 4:
        raise ValidationError("BBox must have exactly 4 values: [west, south, east, north].")

    west, south, east, north = bbox

    if not (-90 <= south <= 90 and -90 <= north <= 90):
        raise ValidationError(
            f"Latitude out of range: south={south}, north={north}. Must be [-90, 90]."
        )

    west = (west + 180) % 360 - 180
    east = (east + 180) % 360 - 180
    bbox = [west, south, east, north]

    if south >= north:
        raise ValidationError(
            f"Invalid bbox: south ({south}) >= north ({north}). Coordinates may be swapped."
        )
    if west >= east:
        raise NeedsUserError(
            f"Invalid bbox: west ({west}) >= east ({east}). "
            f"Anti-meridian crossing is not supported -- narrow the area."
        )

    if (east - west) < 0.001 or (north - south) < 0.001:
        raise NeedsUserError(
            "BBox is too narrow (< ~100m). The geocoder may have returned a point. "
            "Try a more specific area name or widen the search."
        )

    area = estimate_area_km2(bbox)
    if area > _MAX_AREA_KM2:
        # Reachable only with a geocoder that does not clamp: both the real
        # geocoder and the eval stub shrink oversized boxes first
        # (clamp_geocoder_bbox), which substitutes a smaller area silently.
        raise NeedsUserError(
            f"Area too large: {area:.0f} km2 (limit: {_MAX_AREA_KM2} km2). "
            f"Try a smaller region or a specific city/district name."
        )
    if area < _MIN_AREA_KM2:
        raise NeedsUserError(
            f"Area too small: {area:.2f} km2 (minimum: {_MIN_AREA_KM2} km2). "
            f"Try zooming out or using a broader location name."
        )

    zone_west = _lon_to_utm_zone(west)
    zone_east = _lon_to_utm_zone(east)
    if warn and zone_west != zone_east:
        logger.warning(
            f"BBox crosses UTM zones {zone_west}-{zone_east}. "
            f"Edge distortion may occur. Consider splitting into two analyses."
        )

    return bbox


def validate_location(raw) -> str:
    """
    Location must be a non-empty place name of reasonable length.

    Empty -> NeedsUserError: almost always the user named no place, and any
    place the model filled in would be invented. (If the model merely dropped
    a place the user did name, this blocks a legitimate retry; the user then
    has to repeat the request.) Too long -> ValidationError: the model pasted
    more than the place name and can shorten it.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise NeedsUserError("Location is empty. Name a place, e.g. 'Florence, Italy'.")
    location = raw.strip()
    if len(location) > _MAX_LOCATION_CHARS:
        raise ValidationError(
            f"Location name is too long ({len(location)} chars, max {_MAX_LOCATION_CHARS})."
        )
    return location


def _as_whole_days(days_back) -> int:
    """
    Model SDKs may deliver an INTEGER argument as float (90.0) or string
    ("90"). Accept whole numbers in any of those forms, reject the rest.
    """
    if isinstance(days_back, bool):
        raise ValidationError("days_back must be a whole number of days.")
    if isinstance(days_back, int):
        return days_back
    if isinstance(days_back, float) and days_back.is_integer():
        return int(days_back)
    if isinstance(days_back, str) and days_back.strip().lstrip("-").isdigit():
        return int(days_back.strip())
    raise ValidationError(f"days_back must be a whole number of days, got {days_back!r}.")


def validate_temporal_range(days_back) -> int:
    """
    Not a whole number, or < 1 -> ValidationError: a mapping mistake by the
    model (e.g. "today" passed as 0), which it can correct. Over the maximum
    -> NeedsUserError: the user asked for a longer window than the system
    supports, and a shorter one is a different request.
    """
    days_back = _as_whole_days(days_back)
    if days_back < _MIN_DAYS_BACK:
        raise ValidationError(f"days_back must be >= {_MIN_DAYS_BACK}.")
    if days_back > _MAX_DAYS_BACK:
        raise NeedsUserError(
            f"days_back={days_back} exceeds maximum ({_MAX_DAYS_BACK} days / ~1 year). "
            f"Satellite composites degrade over very long periods."
        )
    return days_back


# -- The period of an analysis: a look-back window or calendar dates --

# ASCII digits only: \d would also accept other scripts' digits.
_ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


@dataclass(frozen=True)
class Period:
    """
    The UTC window a job runs on. ``days_back`` is set for the look-back
    form and None for the calendar form (start_date + end_date).
    """

    start: datetime
    end: datetime
    days_back: Optional[int] = None

    @property
    def start_date(self) -> date:
        return as_utc(self.start).date()

    @property
    def end_date(self) -> date:
        return as_utc(self.end).date()


def as_utc(now: datetime) -> datetime:
    """The clock reading in UTC (a naive reading is taken as UTC)."""
    return now.astimezone(timezone.utc) if now.tzinfo else now.replace(tzinfo=timezone.utc)


def date_given(raw: Any) -> bool:
    """None, "" and whitespace mean the date argument was not given."""
    return raw is not None and not (isinstance(raw, str) and not raw.strip())


def parse_date_arg(name: str, raw: Any) -> date:
    """
    One date argument: exactly YYYY-MM-DD and a real calendar day.
    Anything else is a model mistake (ValidationError).
    """
    text = raw.strip() if isinstance(raw, str) else None
    if text is None or not _ISO_DATE.fullmatch(text):
        raise ValidationError(
            f"{name} must be a date in YYYY-MM-DD form, e.g. 2025-08-01; got {raw!r}."
        )
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise ValidationError(
            f"{name} {text!r} is not a real calendar day. Pass an existing date in YYYY-MM-DD form."
        ) from None


def resolve_period(days_back: Any, start_date: Any, end_date: Any, now: datetime,
                   *, default_days_back: int) -> Period:
    """
    The window of one launch, from the request's clock reading ``now``.

    Without dates it is the look-back form: ``days_back`` (or the default)
    days ending at ``now``. With dates, the checks run in this order, and
    each says which kind it is:

    * dates together with days_back, or a start without an end: the model
      mixed the forms up (ValidationError);
    * an end without a start: any start would be invented (NeedsUserError);
    * a malformed date, or start after end (ValidationError);
    * a period that starts in the future (NeedsUserError): no data exists;
    * more than ``_MAX_DAYS_BACK`` days from start to end (NeedsUserError).
      Checked before the end date, so a period that is both too long and
      runs into the future is refused as too long, not sent back to the model;
    * an end in the future after a past start (ValidationError): ending the
      period today still covers every day of it that has happened.

    The window runs from 00:00 UTC of start_date to the end of end_date,
    or to ``now`` when the period ends today. Dates are UTC calendar days.
    """
    if not date_given(start_date) and not date_given(end_date):
        days = validate_temporal_range(default_days_back if days_back is None else days_back)
        return Period(start=now - timedelta(days=days), end=now, days_back=days)

    utc_now = as_utc(now)
    today = utc_now.date()
    if days_back is not None:
        raise ValidationError(
            "Pass either days_back or start_date and end_date, not both. Use start_date and "
            "end_date when the user names calendar dates, months, seasons or years; use days_back "
            "for 'the last N days, weeks, months or years'."
        )
    if not date_given(end_date):
        raise ValidationError(
            "start_date and end_date go together: end_date is missing. For a period that runs "
            f"up to now, end_date is today's date ({today})."
        )
    if not date_given(start_date):
        raise NeedsUserError(
            f"The period has an end date ({str(end_date).strip()}) but no start date. "
            "Ask the user which date the period starts from."
        )

    start = parse_date_arg("start_date", start_date)
    end = parse_date_arg("end_date", end_date)
    if start > end:
        raise ValidationError(
            f"start_date {start} is after end_date {end}. Pass the earlier date as start_date."
        )
    if start > today:
        raise NeedsUserError(
            f"The period {start} to {end} starts in the future (today is {today}). "
            "Satellite data exists only for past dates."
        )
    span = (end - start).days
    if span > _MAX_DAYS_BACK:
        raise NeedsUserError(
            f"The period {start} to {end} spans {span} days, over the maximum of {_MAX_DAYS_BACK} "
            "days (~1 year; a full calendar year is allowed). Satellite composites degrade over "
            "very long periods."
        )
    if end > today:
        raise ValidationError(
            f"end_date {end} is in the future (today is {today}). For a period that runs up "
            f"to now, end_date is today's date ({today})."
        )

    window_start = datetime.combine(start, time.min, tzinfo=timezone.utc)
    window_end = min(datetime.combine(end, time.max, tzinfo=timezone.utc), utc_now)
    return Period(start=window_start, end=window_end)


def resolve_resolution(bbox: List[float], requested: int = 10) -> int:
    """
    Auto-tune resolution based on area size for optimal speed/quality tradeoff.
    """
    area = estimate_area_km2(bbox)
    for threshold, resolution in _RESOLUTION_TIERS:
        if area <= threshold:
            if requested > resolution:
                return requested
            if resolution > requested:
                logger.info(
                    f"Area {area:.0f} km2: resolution {requested}m -> {resolution}m "
                    f"(tier threshold {threshold} km2)"
                )
            return resolution
    return 60


def validate_algorithm_compatibility(algorithm: str, provider: str = "element84-s2") -> str:
    # NeedsUserError: running NDWI instead is a different analysis. Unreachable
    # today (the agent always uses element84-s2).
    if algorithm == "mndwi" and provider not in ("element84-s2", "element84-landsat"):
        raise NeedsUserError(
            "MNDWI requires SWIR bands which may not be available with this provider. "
            "Try NDWI instead (uses NIR, available on all providers)."
        )
    return algorithm


def clamp_geocoder_bbox(
    bbox: Tuple[float, float, float, float],
    max_area_km2: float = _MAX_AREA_KM2,
) -> List[float]:
    """
    Nominatim sometimes returns province-level bboxes for cities.
    If the bbox is too large, shrink it toward the centroid.

    Uses 95% of max_area as target to leave headroom for float
    precision — prevents the clamped bbox from landing exactly on
    the validation boundary and failing the > check.
    """
    west, south, east, north = bbox
    area = estimate_area_km2([west, south, east, north])

    if area <= max_area_km2:
        return [west, south, east, north]

    # Target 95% of limit so validation never rejects on float edge cases
    target_area = max_area_km2 * 0.95
    ratio = math.sqrt(target_area / area)
    cx = (west + east) / 2
    cy = (south + north) / 2
    hw = (east - west) / 2 * ratio
    hh = (north - south) / 2 * ratio

    clamped = [cx - hw, cy - hh, cx + hw, cy + hh]
    logger.info(
        f"Geocoder bbox clamped: {area:.0f} km2 -> {estimate_area_km2(clamped):.0f} km2"
    )
    return clamped