"""
Raster to vector conversion for anomaly features.

Extracts polygons from severity rasters (LISA, Flood) and prepares
them for PostGIS storage. All metric calculations (area, perimeter)
are done in the native UTM projection before reprojecting to WGS84.
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

from eo_brain.db.vector_repo import vector_repo

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VectorizeConfig:
    """Configuration for vectorization process."""
    
    # Minimum area in pixels (features smaller than this are noise)
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


def _get_transformer(src_crs: str, dst_crs: str = "EPSG:4326"):
    """Create a pyproj transformer for coordinate conversion."""
    return pyproj.Transformer.from_crs(
        src_crs, dst_crs, always_xy=True
    ).transform


def _extract_polygons(
    data: np.ndarray,
    affine_transform: rasterio.Affine,
    src_crs: str,
    config: VectorizeConfig
) -> Iterator[Dict[str, Any]]:
    """
    Extract polygons from raster data.
    
    Yields dicts with geometry in UTM (for metric calculations)
    and metadata.
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
    mask = data >= config.min_severity
    
    if not mask.any():
        logger.warning("No features above minimum severity threshold")
        return
    
    feature_count = 0
    
    for geom_dict, value in shapes(data, mask=mask, transform=affine_transform):
        if feature_count >= config.max_features:
            logger.warning(f"Reached max features limit ({config.max_features})")
            break
        
        severity = int(value)
        if severity < config.min_severity:
            continue
        
        # Create shapely geometry (in UTM coordinates)
        geom_utm = shape(geom_dict)
        
        if not geom_utm.is_valid:
            geom_utm = geom_utm.buffer(0)  # Fix invalid geometries
            if not geom_utm.is_valid:
                yield {"status": "invalid"}
                continue
        
        # Filter by area (in UTM = square meters)
        area_m2 = geom_utm.area
        if area_m2 < min_area_m2:
            yield {"status": "small", "area": area_m2}
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
            "geom_utm": geom_simplified,
            "centroid_utm": centroid_utm,
            "area_m2": area_m2,
            "perimeter_m": perimeter_m,
            "src_crs": src_crs
        }


def vectorize_cog(
    cog_path: str,
    job_id: UUID,
    algorithm: str,
    quadrant_data: Optional[np.ndarray] = None,
    config: Optional[VectorizeConfig] = None
) -> VectorizeResult:
    """
    Extract vector features from a COG and insert into PostGIS.
    
    Args:
        cog_path: Path to Cloud Optimized GeoTIFF (severity raster)
        job_id: UUID of parent job
        algorithm: Algorithm name ('lisa' or 'flood')
        quadrant_data: Optional quadrant raster for LISA (to tag HL vs LH)
        config: Vectorization configuration
    
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
    
    # Prepare transformer for final reprojection to WGS84
    to_wgs84 = _get_transformer(src_crs, "EPSG:4326")
    
    features_to_insert: List[Dict[str, Any]] = []
    stats = {
        "by_severity": {1: 0, 2: 0, 3: 0},
        "total_area_m2": 0.0,
        "skipped_small": 0,
        "skipped_invalid": 0
    }
    
    for result in _extract_polygons(data, affine, src_crs, config):
        if result["status"] == "small":
            stats["skipped_small"] += 1
            continue
        elif result["status"] == "invalid":
            stats["skipped_invalid"] += 1
            continue
        
        # Reproject to WGS84 for storage
        geom_wgs84 = shapely_transform(to_wgs84, result["geom_utm"])
        centroid_wgs84 = shapely_transform(to_wgs84, result["centroid_utm"])
        
        severity = result["severity"]
        
        # For LISA, try to get quadrant from quadrant raster
        quadrant = None
        if algorithm == "lisa" and quadrant_data is not None:
            # Sample quadrant at centroid location
            # This is approximate but good enough for labeling
            try:
                centroid_utm = result["centroid_utm"]
                col = int((centroid_utm.x - affine.c) / affine.a)
                row = int((centroid_utm.y - affine.f) / affine.e)
                if 0 <= row < quadrant_data.shape[0] and 0 <= col < quadrant_data.shape[1]:
                    quadrant = int(quadrant_data[row, col])
            except (IndexError, ValueError):
                pass
        
        features_to_insert.append({
            "job_id": str(job_id),
            "algorithm": algorithm,
            "severity": severity,
            "quadrant": quadrant,
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
    
    result = VectorizeResult(
        feature_count=len(features_to_insert),
        total_area_m2=stats["total_area_m2"],
        by_severity=stats["by_severity"],
        skipped_small=stats["skipped_small"],
        skipped_invalid=stats["skipped_invalid"]
    )
    
    logger.info(
        f"Vectorization complete: {result.feature_count} features, "
        f"{result.total_area_m2/10000:.2f} ha total, "
        f"skipped {result.skipped_small} small, {result.skipped_invalid} invalid"
    )
    
    return result


def vectorize_from_dataset(
    severity_data: np.ndarray,
    quadrant_data: Optional[np.ndarray],
    transform: rasterio.Affine,
    crs: str,
    job_id: UUID,
    algorithm: str,
    config: Optional[VectorizeConfig] = None
) -> VectorizeResult:
    """
    Vectorize directly from numpy arrays (for in-memory processing).
    
    Useful when you want to vectorize without writing COG first.
    """
    config = config or VectorizeConfig()
    
    to_wgs84 = _get_transformer(crs, "EPSG:4326")
    
    features_to_insert: List[Dict[str, Any]] = []
    stats = {
        "by_severity": {},
        "total_area_m2": 0.0,
        "skipped_small": 0,
        "skipped_invalid": 0
    }
    
    for result in _extract_polygons(severity_data, transform, crs, config):
        if result["status"] == "small":
            stats["skipped_small"] += 1
            continue
        elif result["status"] == "invalid":
            stats["skipped_invalid"] += 1
            continue
        
        geom_wgs84 = shapely_transform(to_wgs84, result["geom_utm"])
        centroid_wgs84 = shapely_transform(to_wgs84, result["centroid_utm"])
        
        severity = result["severity"]
        quadrant = None
        
        if algorithm == "lisa" and quadrant_data is not None:
            try:
                c_utm = result["centroid_utm"]
                col = int((c_utm.x - transform.c) / transform.a)
                row = int((c_utm.y - transform.f) / transform.e)
                if 0 <= row < quadrant_data.shape[0] and 0 <= col < quadrant_data.shape[1]:
                    quadrant = int(quadrant_data[row, col])
            except (IndexError, ValueError):
                pass
        
        features_to_insert.append({
            "job_id": str(job_id),
            "algorithm": algorithm,
            "severity": severity,
            "quadrant": quadrant,
            "geometry_wkt": geom_wgs84.wkt,
            "centroid_wkt": centroid_wgs84.wkt,
            "area_m2": result["area_m2"],
            "perimeter_m": result["perimeter_m"]
        })
        
        stats["by_severity"][severity] = stats["by_severity"].get(severity, 0) + 1
        stats["total_area_m2"] += result["area_m2"]
    
    if features_to_insert:
        vector_repo.bulk_insert(features_to_insert)
    
    return VectorizeResult(
        feature_count=len(features_to_insert),
        total_area_m2=stats["total_area_m2"],
        by_severity=stats["by_severity"],
        skipped_small=stats["skipped_small"],
        skipped_invalid=stats["skipped_invalid"]
    )