"""
Raster to vector conversion for anomaly features.

Extracts polygons from severity rasters (LISA, Flood) and prepares
them for PostGIS storage. All metric calculations (area, perimeter)
are done in the native UTM projection before reprojecting to WGS84.

A polygon is a zone: a connected patch (4-connected pixels) of one class.
For flood the class is the severity; for LISA it is the severity and the
Moran quadrant together, so a same-severity patch holding both high (HL)
and low (LH) outliers becomes one zone per quadrant and every zone has
exactly one quadrant. Patches with fewer than ``min_area_pixels`` pixels are
not mapped; how many there were and their area are part of the result. With
``area_mask``, only the pixels inside it (the requested area) are mapped.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from uuid import UUID
import logging

import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping
from shapely.ops import transform as shapely_transform
import pyproj

from eo_brain.algorithms.base import OUTLIER_QUADRANTS
from eo_brain.db.vector_repo import vector_repo

logger = logging.getLogger(__name__)

# A zone's class value in the label raster: severity * _QUADRANT_BASE + quadrant.
_QUADRANT_BASE = 10


@dataclass(frozen=True)
class VectorizeConfig:
    """Configuration for vectorization process."""

    # Minimum zone size in pixels (smaller patches are noise and are not mapped)
    min_area_pixels: int = 4

    # Simplification tolerance as fraction of pixel size
    simplify_tolerance_factor: float = 0.5

    # Only extract features with severity >= this value
    min_severity: int = 1

    # Maximum features to extract (safety limit)
    max_features: int = 50000


@dataclass
class VectorizeResult:
    """Result of vectorization process."""
    feature_count: int
    total_area_m2: float
    by_severity: Dict[int, int]  # severity -> count
    skipped_small: int
    skipped_invalid: int
    skipped_small_area_m2: float = 0.0   # total area of the patches under the minimum size
    min_area_pixels: int = 0             # minimum zone size used, in pixels
    min_area_m2: float = 0.0             # the same in square metres
    max_features_reached: bool = False   # mapping stopped at config.max_features

    def stats(self) -> Dict[str, Any]:
        """The job's ``result.vector_stats`` block; agent.analyses reads the zone-size keys."""
        return {
            "feature_count": self.feature_count,
            "total_area_m2": self.total_area_m2,
            "by_severity": self.by_severity,
            "min_zone_pixels": self.min_area_pixels,
            "min_zone_area_m2": self.min_area_m2,
            "skipped_small": self.skipped_small,
            "skipped_small_area_m2": self.skipped_small_area_m2,
            "skipped_invalid": self.skipped_invalid,
            "max_features_reached": self.max_features_reached,
        }


def _get_transformer(src_crs: str, dst_crs: str = "EPSG:4326"):
    """Create a pyproj transformer for coordinate conversion."""
    return pyproj.Transformer.from_crs(
        src_crs, dst_crs, always_xy=True
    ).transform


def _extract_polygons(
    data: np.ndarray,
    affine_transform: rasterio.Affine,
    src_crs: str,
    config: VectorizeConfig,
    quadrant: Optional[np.ndarray] = None,
    area_mask: Optional[np.ndarray] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Extract polygons from raster data.

    ``data`` is the severity raster. With ``quadrant`` (same shape), a zone is
    a patch of one severity AND one quadrant. With ``area_mask`` (same shape,
    bool), pixels outside it are not mapped.

    Yields dicts with geometry in UTM (for metric calculations)
    and metadata; status "small" and "invalid" for the patches left out, and a
    final status "limit" if ``config.max_features`` stopped the extraction.
    """
    # Get pixel size for area threshold
    pixel_size_x = abs(affine_transform.a)
    pixel_size_y = abs(affine_transform.e)
    pixel_area = pixel_size_x * pixel_size_y
    min_area_m2 = pixel_area * config.min_area_pixels

    # Simplification tolerance in meters (native CRS units)
    simplify_tolerance = min(pixel_size_x, pixel_size_y) * config.simplify_tolerance_factor

    logger.info(f"Extracting polygons: min_area={min_area_m2:.0f}m², simplify={simplify_tolerance:.1f}m")

    # Mask for valid severity values
    severity_raster = np.asarray(data)
    mask = severity_raster >= config.min_severity
    if area_mask is not None:
        mask &= np.asarray(area_mask, dtype=bool)

    if not mask.any():
        logger.warning("No features above minimum severity threshold")
        return

    # One label per class: connected pixels of one label form one zone
    labels = severity_raster.astype(np.int32)
    if quadrant is not None:
        labels = labels * _QUADRANT_BASE + np.asarray(quadrant).astype(np.int32)

    feature_count = 0

    for geom_dict, value in shapes(labels, mask=mask, transform=affine_transform):
        if feature_count >= config.max_features:
            logger.warning(f"Reached max features limit ({config.max_features})")
            yield {"status": "limit"}
            break

        value = int(value)
        severity, zone_quadrant = (value // _QUADRANT_BASE, value % _QUADRANT_BASE) if quadrant is not None \
            else (value, None)
        if severity < config.min_severity:
            continue

        # Create shapely geometry (in UTM coordinates)
        geom_utm = shape(geom_dict)

        if not geom_utm.is_valid:
            geom_utm = geom_utm.buffer(0)  # Fix invalid geometries
            if not geom_utm.is_valid:
                yield {"status": "invalid"}
                continue

        # Filter by size, in pixels (the area is a whole number of pixels: no
        # float comparison decides whether a patch of exactly the minimum is kept)
        area_m2 = geom_utm.area
        if round(area_m2 / pixel_area) < config.min_area_pixels:
            yield {"status": "small", "area": area_m2, "severity": severity}
            continue

        # Simplify in UTM (where tolerance has consistent meaning)
        geom_simplified = geom_utm.simplify(simplify_tolerance, preserve_topology=True)

        # Calculate perimeter in meters
        perimeter_m = geom_simplified.length

        # Centroid in UTM
        centroid_utm = geom_simplified.centroid

        feature_count += 1

        yield {
            "status": "ok",
            "severity": severity,
            "quadrant": zone_quadrant,
            "geom_utm": geom_simplified,
            "centroid_utm": centroid_utm,
            "area_m2": area_m2,
            "perimeter_m": perimeter_m,
            "src_crs": src_crs
        }


def _check_layers(
    data: np.ndarray,
    algorithm: str,
    quadrant_data: Optional[np.ndarray],
    area_mask: Optional[np.ndarray],
    config: VectorizeConfig,
) -> None:
    """Refuse a LISA raster without its quadrants, misaligned layers, and zones that cannot exist."""
    if algorithm == "lisa" and quadrant_data is None:
        raise ValueError("LISA zones need the Moran quadrant raster (quadrant_data)")
    if algorithm != "lisa" and quadrant_data is not None:
        raise ValueError(f"{algorithm} has no Moran quadrant")
    for name, layer in (("quadrant_data", quadrant_data), ("area_mask", area_mask)):
        if layer is not None and np.shape(layer) != np.shape(data):
            raise ValueError(f"{name} {np.shape(layer)} and the severity raster {np.shape(data)} differ in shape")
    if quadrant_data is not None:
        mapped = np.asarray(data) >= config.min_severity
        if area_mask is not None:
            mapped &= np.asarray(area_mask, dtype=bool)
        quadrants = np.unique(np.asarray(quadrant_data)[mapped])
        allowed = [int(q) for q in OUTLIER_QUADRANTS]
        if not np.isin(quadrants, allowed).all():
            raise ValueError(f"LISA pixels with a severity have quadrants {quadrants.tolist()}, expected {allowed}")


def _vectorize(
    data: np.ndarray,
    affine: rasterio.Affine,
    src_crs: str,
    job_id: UUID,
    algorithm: str,
    quadrant_data: Optional[np.ndarray],
    area_mask: Optional[np.ndarray],
    config: VectorizeConfig,
) -> VectorizeResult:
    """Extract the zones of ``data``, insert them into PostGIS and count what was left out."""
    _check_layers(data, algorithm, quadrant_data, area_mask, config)

    # Prepare transformer for final reprojection to WGS84
    to_wgs84 = _get_transformer(src_crs, "EPSG:4326")

    features_to_insert: List[Dict[str, Any]] = []
    stats = {
        "by_severity": {1: 0, 2: 0, 3: 0},
        "total_area_m2": 0.0,
        "skipped_small": 0,
        "skipped_small_area_m2": 0.0,
        "skipped_invalid": 0,
        "max_features_reached": False,
    }

    for result in _extract_polygons(data, affine, src_crs, config, quadrant_data, area_mask):
        if result["status"] == "small":
            stats["skipped_small"] += 1
            stats["skipped_small_area_m2"] += result["area"]
            continue
        elif result["status"] == "invalid":
            stats["skipped_invalid"] += 1
            continue
        elif result["status"] == "limit":
            stats["max_features_reached"] = True
            continue

        # Reproject to WGS84 for storage
        geom_wgs84 = shapely_transform(to_wgs84, result["geom_utm"])
        centroid_wgs84 = shapely_transform(to_wgs84, result["centroid_utm"])

        severity = result["severity"]

        features_to_insert.append({
            "job_id": str(job_id),
            "algorithm": algorithm,
            "severity": severity,
            "quadrant": result["quadrant"],
            "geometry_wkt": geom_wgs84.wkt,
            "centroid_wkt": centroid_wgs84.wkt,
            "area_m2": result["area_m2"],
            "perimeter_m": result["perimeter_m"]
        })

        stats["by_severity"][severity] = stats["by_severity"].get(severity, 0) + 1
        stats["total_area_m2"] += result["area_m2"]

    # Bulk insert
    if features_to_insert:
        vector_repo.bulk_insert(features_to_insert)

    pixel_area = abs(affine.a) * abs(affine.e)
    return VectorizeResult(
        feature_count=len(features_to_insert),
        total_area_m2=stats["total_area_m2"],
        by_severity=stats["by_severity"],
        skipped_small=stats["skipped_small"],
        skipped_invalid=stats["skipped_invalid"],
        skipped_small_area_m2=stats["skipped_small_area_m2"],
        min_area_pixels=config.min_area_pixels,
        min_area_m2=config.min_area_pixels * pixel_area,
        max_features_reached=stats["max_features_reached"],
    )


def vectorize_cog(
    cog_path: str,
    job_id: UUID,
    algorithm: str,
    quadrant_data: Optional[np.ndarray] = None,
    config: Optional[VectorizeConfig] = None,
    area_mask: Optional[np.ndarray] = None,
) -> VectorizeResult:
    """
    Extract vector features from a COG and insert into PostGIS.

    Args:
        cog_path: Path to Cloud Optimized GeoTIFF (severity raster)
        job_id: UUID of parent job
        algorithm: Algorithm name ('lisa' or 'flood')
        quadrant_data: Moran quadrant raster on the COG's grid; required for LISA,
            whose zones are split by quadrant (HL vs LH)
        config: Vectorization configuration
        area_mask: Optional bool raster on the COG's grid; only pixels inside
            it are mapped (the requested area)

    Returns:
        VectorizeResult with statistics
    """
    config = config or VectorizeConfig()

    cog_path = Path(cog_path)
    if not cog_path.exists():
        raise FileNotFoundError(f"COG not found: {cog_path}")

    logger.info(f"Vectorizing {cog_path.name} for job {job_id}")

    with rasterio.open(cog_path) as src:
        data = src.read(1)
        src_crs = str(src.crs)
        affine = src.transform

    result = _vectorize(data, affine, src_crs, job_id, algorithm, quadrant_data, area_mask, config)

    logger.info(
        f"Vectorization complete: {result.feature_count} features, "
        f"{result.total_area_m2/10000:.2f} ha total, "
        f"skipped {result.skipped_small} small ({result.skipped_small_area_m2/10000:.2f} ha), "
        f"{result.skipped_invalid} invalid"
    )

    return result


def vectorize_from_dataset(
    severity_data: np.ndarray,
    quadrant_data: Optional[np.ndarray],
    transform: rasterio.Affine,
    crs: str,
    job_id: UUID,
    algorithm: str,
    config: Optional[VectorizeConfig] = None,
    area_mask: Optional[np.ndarray] = None,
) -> VectorizeResult:
    """
    Vectorize directly from numpy arrays (for in-memory processing).

    Useful when you want to vectorize without writing COG first. Same rules
    as ``vectorize_cog``.
    """
    config = config or VectorizeConfig()
    return _vectorize(severity_data, transform, crs, job_id, algorithm, quadrant_data, area_mask, config)
