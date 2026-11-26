"""
eo_brain/agent/geocoder.py - Production geocoding with bbox generation.

Uses Nominatim (OSM) via geopy. No API key required.
Falls back to viewbox from Nominatim when available, otherwise
generates a synthetic bbox from the centroid + adaptive padding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

from eo_brain.agent.validators import clamp_geocoder_bbox

logger = logging.getLogger(__name__)

_GEOCODER = Nominatim(
    user_agent="eo-brain/1.0",
    timeout=10,
)

# Padding in degrees applied to centroid when Nominatim doesn't return a bbox.
# Rough heuristic: 0.05 deg ~ 5km at mid-latitudes.
_DEFAULT_PADDING_DEG = 0.05

# For named regions (rivers, valleys), we want wider coverage
_REGION_PADDING_DEG = 0.15

_REGION_TYPES = {"river", "waterway", "valley", "mountain_range", "state", "county"}


@dataclass(frozen=True)
class GeocodingResult:
    name: str
    lat: float
    lon: float
    bbox: Tuple[float, float, float, float]  # (west, south, east, north)
    source: str  # "nominatim_bbox" | "synthetic"
    raw_type: Optional[str] = None

    def to_list(self) -> list[float]:
        return list(self.bbox)


def _clamp_bbox(west: float, south: float, east: float, north: float):
    """Ensure bbox stays within valid WGS84 bounds."""
    return (
        max(west, -180.0),
        max(south, -90.0),
        min(east, 180.0),
        min(north, 90.0),
    )


def _synthetic_bbox(
    lat: float, lon: float, padding: float
) -> Tuple[float, float, float, float]:
    return _clamp_bbox(
        lon - padding,
        lat - padding,
        lon + padding,
        lat + padding,
    )


class _TransientGeocoderError(Exception):
    """Raised inside the cached function so lru_cache does not store it."""


def geocode(query: str) -> Optional[GeocodingResult]:
    """
    Resolve a place name to a GeocodingResult with bbox.

    Cached in-process (LRU). Returns None on failure rather than raising,
    so the agent can report a clean error to the user. Timeouts and service
    errors are not cached, so a transient outage does not poison the cache.
    """
    try:
        return _geocode_cached(query)
    except _TransientGeocoderError:
        return None


@lru_cache(maxsize=256)
def _geocode_cached(query: str) -> Optional[GeocodingResult]:
    try:
        location = _GEOCODER.geocode(
            query,
            exactly_one=True,
            language="en",
            addressdetails=True,
        )
    except (GeocoderTimedOut, GeocoderServiceError) as exc:
        logger.warning(f"Geocoding failed for '{query}': {exc}")
        raise _TransientGeocoderError(str(exc)) from exc

    if location is None:
        logger.info(f"No geocoding result for '{query}'")
        return None

    lat, lon = location.latitude, location.longitude
    raw = location.raw or {}
    osm_type = raw.get("type", "")

    # Nominatim often returns a 'boundingbox' as [south, north, west, east] (strings)
    raw_bbox = raw.get("boundingbox")
    if raw_bbox and len(raw_bbox) == 4:
        try:
            s, n, w, e = (float(v) for v in raw_bbox)
            bbox = _clamp_bbox(w, s, e, n)
            source = "nominatim_bbox"
        except (ValueError, TypeError):
            bbox = _synthetic_bbox(lat, lon, _DEFAULT_PADDING_DEG)
            source = "synthetic"
    else:
        padding = _REGION_PADDING_DEG if osm_type in _REGION_TYPES else _DEFAULT_PADDING_DEG
        bbox = _synthetic_bbox(lat, lon, padding)
        source = "synthetic"

    # Sanity: bbox must have nonzero area
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        bbox = _synthetic_bbox(lat, lon, _DEFAULT_PADDING_DEG)
        source = "synthetic"

    # Clamp oversized bboxes (Nominatim returns province-level for some cities)
    clamped = clamp_geocoder_bbox(bbox)
    bbox = (clamped[0], clamped[1], clamped[2], clamped[3])

    return GeocodingResult(
        name=location.address or query,
        lat=lat,
        lon=lon,
        bbox=bbox,
        source=source,
        raw_type=osm_type or None,
    )


def geocode_or_fail(query: str) -> GeocodingResult:
    """Same as geocode() but raises on failure. Use in tool context."""
    result = geocode(query)
    if result is None:
        raise ValueError(
            f"Could not geocode '{query}'. Try a more specific place name "
            f"(e.g. 'Rome, Italy' instead of 'Roma')."
        )
    return result