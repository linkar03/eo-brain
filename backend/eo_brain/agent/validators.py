"""
eo_brain/agent/validators.py - Guardrails for agent tool inputs.

Every tool parameter passes through validation before touching
the DB or processing pipeline. Errors are human-readable so
Gemini can relay them to the user as actionable suggestions.
"""

from __future__ import annotations

import math
import logging
from typing import List, Tuple

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
    """Raised with a user-friendly message the agent can relay."""
    pass


def resolve_algorithm(raw: str) -> str:
    key = raw.strip().lower()
    if key in _ALLOWED_ALGORITHMS:
        return key
    resolved = _ALGO_ALIASES.get(key)
    if resolved:
        return resolved
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


def validate_bbox(bbox: List[float]) -> List[float]:
    """Full bbox validation with actionable error messages."""
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
        raise ValidationError(
            f"Invalid bbox: west ({west}) >= east ({east}). "
            f"Anti-meridian crossing is not supported -- narrow the area."
        )

    if (east - west) < 0.001 or (north - south) < 0.001:
        raise ValidationError(
            "BBox is too narrow (< ~100m). The geocoder may have returned a point. "
            "Try a more specific area name or widen the search."
        )

    area = estimate_area_km2(bbox)
    if area > _MAX_AREA_KM2:
        raise ValidationError(
            f"Area too large: {area:.0f} km2 (limit: {_MAX_AREA_KM2} km2). "
            f"Try a smaller region or a specific city/district name."
        )
    if area < _MIN_AREA_KM2:
        raise ValidationError(
            f"Area too small: {area:.2f} km2 (minimum: {_MIN_AREA_KM2} km2). "
            f"Try zooming out or using a broader location name."
        )

    zone_west = _lon_to_utm_zone(west)
    zone_east = _lon_to_utm_zone(east)
    if zone_west != zone_east:
        logger.warning(
            f"BBox crosses UTM zones {zone_west}-{zone_east}. "
            f"Edge distortion may occur. Consider splitting into two analyses."
        )

    return bbox


def validate_location(raw) -> str:
    """Location must be a non-empty place name of reasonable length."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValidationError("Location is empty. Name a place, e.g. 'Florence, Italy'.")
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
    days_back = _as_whole_days(days_back)
    if days_back < _MIN_DAYS_BACK:
        raise ValidationError(f"days_back must be >= {_MIN_DAYS_BACK}.")
    if days_back > _MAX_DAYS_BACK:
        raise ValidationError(
            f"days_back={days_back} exceeds maximum ({_MAX_DAYS_BACK} days / ~1 year). "
            f"Satellite composites degrade over very long periods."
        )
    return days_back


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
    if algorithm == "mndwi" and provider not in ("element84-s2", "element84-landsat"):
        raise ValidationError(
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