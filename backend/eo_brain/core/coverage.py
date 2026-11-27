"""
eo_brain/core/coverage.py - How much of the requested area a result covers.

Coverage is the share of the requested bounding box (WGS84, as the user asked
for it) whose pixels have a value in the result layer, i.e. at least one
cloud-free observation passed the pixel mask. The uncovered rest is split in
two: pixels that the loaded scenes saw but the mask removed on every day
(clouds, their shadows, snow, ...), and pixels outside every loaded scene's
footprint. The block is stored with the job (``result.coverage``), shown in the
job's completion message and exposed to the data agent (``agent.analyses``), so
"why is only part of the area covered" has an answer from data.

A job whose coverage is below ``LOW_COVERAGE_PCT`` completes, but is flagged
``low_coverage`` and its message says so, with the reason and what to try.
80 %: below it a fifth or more of the area is blank, the map shows it as large
holes and any area figure (hectares at alert, mean index) describes a
different, smaller area than the one asked for. Above it the gaps are mostly
what the mask removes even in a good month (cloud remnants and their shadows,
snow, water and shadowed slopes it is unsure about); a higher threshold would
fire on most mountain analyses and stop meaning anything. A job with no
covered pixel at all fails instead.

``inside_bbox_mask`` needs pyproj and ``usable_pixels_per_slice`` xarray; the
rest is plain numpy and unit-tested offline.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from eo_brain.core.scenes import DEFAULT_SELECTION

LOW_COVERAGE_PCT = 80.0
_CLOUDY = f"{DEFAULT_SELECTION.loose_cloud_bound:g}%"

# The result layer whose NaNs mark "no cloud-free observation". The discrete
# outputs (flood and lisa severity) write 0 for both "none" and "no data", so
# coverage is read from the continuous index they are computed from.
COVERAGE_LAYER = {
    "ndvi": "ndvi",
    "ndwi": "ndwi",
    "mndwi": "mndwi",
    "flood": "water_index",
    "lisa": "ndvi",
}


def coverage_layer(algorithm: str, result_vars: Sequence[str]) -> str:
    name = COVERAGE_LAYER.get(algorithm)
    if name and name in result_vars:
        return name
    return list(result_vars)[0]


def inside_bbox_mask(xs: np.ndarray, ys: np.ndarray, crs: str, bbox: Sequence[float]) -> np.ndarray:
    """(len(ys), len(xs)) bool: pixel centres (in ``crs``) that fall inside the WGS84 ``bbox``."""
    from pyproj import Transformer

    west, south, east, north = bbox
    to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    gx, gy = np.meshgrid(np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64))
    lon, lat = to_wgs84.transform(gx, gy)
    return (lon >= west) & (lon <= east) & (lat >= south) & (lat <= north)


def usable_pixels_per_slice(ds, provider, inside: np.ndarray, masked: bool) -> Tuple[Any, Any]:
    """
    Lazy ``(per_slice, has_data)`` for the loaded dataset ``ds``.

    ``per_slice``: (time,) count of usable pixels inside the area per time
    slice (the slice has data there and, when the cloud mask is applied, the
    mask keeps it). ``has_data``: (y, x) bool, some loaded slice has data there.
    Computed together with the result (``dask.compute``) so the mask band is
    read once.
    """
    import xarray as xr

    mask_name = getattr(provider, "cloud_mask_band", None)
    band_name = mask_name if mask_name in ds else next(iter(ds.data_vars))
    band = ds[band_name]
    nodata = band.attrs.get("nodata")
    data = band != nodata if nodata is not None else xr.ones_like(band, dtype=bool)
    usable = data & provider.compute_cloud_mask(band) if (masked and band_name == mask_name) else data
    inside_da = xr.DataArray(inside, dims=("y", "x"), coords={"y": ds.y, "x": ds.x})
    per_slice = (usable & inside_da).sum(dim=("y", "x"))
    has_data = data.any(dim="time") if "time" in data.dims else data
    return per_slice, has_data


def _pct(part: int, whole: int) -> float:
    return round(100.0 * part / whole, 1) if whole else 0.0


def summarize_coverage(
    valid: np.ndarray,
    inside: np.ndarray,
    has_data: np.ndarray,
    slice_dates: Sequence[date],
    slice_valid_px: Sequence[int],
    selection: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    The ``coverage`` block stored with a job.

    ``valid``: (H, W) bool, pixels with a value in the result layer.
    ``inside``: (H, W) bool, pixels of the requested area.
    ``has_data``: (H, W) bool, pixels some loaded scene has data for.
    ``slice_dates`` / ``slice_valid_px``: per time slice loaded, its date and
    its number of usable (cloud-free) pixels inside the area. A slice is one
    acquisition day, or part of one when that day's scenes have different
    radiometry rules (core.datacube.radiometry_groups): days are counted once.
    ``selection``: ``SceneSelection.to_dict()`` of the scenes loaded.
    """
    area_px = int(inside.sum())
    covered_px = int((valid & inside).sum())
    no_data_px = int((inside & ~has_data & ~valid).sum())
    masked_px = area_px - covered_px - no_data_px
    # Rounded down, so "100%" means every pixel and a 79.96% job is not shown as 80%.
    pct = (1000 * covered_px // area_px) / 10.0 if area_px else 0.0
    days = sorted({d for d, n in zip(slice_dates, slice_valid_px) if int(n) > 0})
    sel = selection or {}
    return {
        "coverage_pct": pct,
        "low_coverage": pct < LOW_COVERAGE_PCT,
        "low_coverage_below_pct": LOW_COVERAGE_PCT,
        "uncovered_masked_pct": _pct(masked_px, area_px),
        "uncovered_no_data_pct": _pct(no_data_px, area_px),
        "area_px": area_px,
        "covered_px": covered_px,
        "scenes_found": sel.get("found"),
        "scenes_too_cloudy": sel.get("too_cloudy"),
        "scenes_checked": sel.get("screened") if sel.get("method") == "pixel_mask" else None,
        "scenes_with_clear_pixels": sel.get("with_clear"),
        "scenes_used": len(sel.get("scene_ids") or []) if sel else None,
        "acquisition_days_loaded": len(set(slice_dates)),
        "acquisition_days_with_data": len(days),
        "first_date_with_data": days[0].isoformat() if days else None,
        "last_date_with_data": days[-1].isoformat() if days else None,
        "selection": sel or None,
    }


def _days_phrase(cov: Dict[str, Any]) -> str:
    n = cov["acquisition_days_with_data"]
    first, last = cov["first_date_with_data"], cov["last_date_with_data"]
    if n == 0:
        return "no acquisition day"
    if n == 1:
        return f"1 acquisition day, {first}"
    return f"{n} acquisition days, {first} to {last}"


def _scenes_phrase(cov: Dict[str, Any]) -> Optional[str]:
    found, clear, checked = cov.get("scenes_found"), cov.get("scenes_with_clear_pixels"), cov.get("scenes_checked")
    if found is None or clear is None or checked is None:
        return None
    too_cloudy = cov.get("scenes_too_cloudy") or 0
    if too_cloudy:
        return (f"of the {found} scenes in the window, {too_cloudy} were at least {_CLOUDY} cloudy "
                f"and {clear} of the other {checked} had cloud-free pixels over the area")
    return f"{clear} of the {found} scenes in the window had cloud-free pixels over the area"


def _reason(cov: Dict[str, Any]) -> str:
    parts = []
    if cov["uncovered_masked_pct"] > 0:
        parts.append(f"{cov['uncovered_masked_pct']:g}% of the area was masked out (cloud, shadow, snow) "
                     f"on every day loaded")
    if cov["uncovered_no_data_pct"] > 0:
        parts.append(f"{cov['uncovered_no_data_pct']:g}% was outside the footprint of the scenes loaded")
    reason = "; ".join(parts) or "no cloud-free view of the rest of the area"
    scenes = _scenes_phrase(cov)
    return f"{reason} ({scenes})" if scenes else reason


def completion_message(cov: Dict[str, Any]) -> str:
    """User-facing message of a completed job: always the coverage; for low coverage, why and what to try."""
    pct = cov["coverage_pct"]
    days = _days_phrase(cov)
    if not cov["low_coverage"]:
        return f"Completed: cloud-free data covers {pct:g}% of the requested area ({days})."
    return (f"Completed with partial coverage: cloud-free data covers only {pct:g}% of the requested "
            f"area ({days}). Reason: {_reason(cov)}. A longer time window may cover more of it.")


def no_data_message(selection: Dict[str, Any]) -> str:
    """Failure message when no pixel of the area has a cloud-free view."""
    found = selection.get("found")
    too_cloudy = selection.get("too_cloudy") or 0
    if found is None:
        detail = "none of the scenes in the window has a cloud-free view of it"
    elif too_cloudy == found:
        detail = f"all {found} scenes in the window are at least {_CLOUDY} cloudy"
    else:
        detail = f"none of the {found} scenes in the window has a cloud-free view of it"
    return f"No cloud-free pixels over the requested area: {detail}. Try a longer time window."
