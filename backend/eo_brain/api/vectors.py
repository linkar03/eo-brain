"""
Vector tile endpoints for serving MVT (Mapbox Vector Tiles).

Provides endpoints for:
- MVT tiles for map rendering
- GeoJSON export for debugging/download
- Feature statistics
"""

from uuid import UUID
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Response

from eo_brain.db.vector_repo import vector_repo
from eo_brain.db.database import job_repo

router = APIRouter(prefix="/vectors", tags=["vectors"])


@router.get("/{job_id}/{z}/{x}/{y}.pbf")
async def get_mvt_tile(
    job_id: str,
    z: int,
    x: int,
    y: int,
    layer_name: str = Query(default="alerts", description="MVT layer name")
):
    """
    Serve MVT (Mapbox Vector Tile) for vector features.
    
    Tiles contain polygonized anomaly regions with properties:
    - id: feature ID
    - severity: 1 (watch), 2 (suspect), 3 (alert)
    - quadrant: LISA quadrant (3=HL, 4=LH) or null for flood
    - algorithm: 'lisa' or 'flood'
    - area_m2: area in square meters
    
    Standard MVT format, compatible with Mapbox GL, MapLibre, Leaflet+plugins.
    """
    try:
        uuid = UUID(job_id)
    except ValueError:
        raise HTTPException(400, "Invalid job ID format")
    
    # Verify job exists
    job = job_repo.get(uuid)
    if not job:
        raise HTTPException(404, "Job not found")
    
    # Validate tile coordinates
    max_tile = 2 ** z
    if not (0 <= x < max_tile and 0 <= y < max_tile):
        raise HTTPException(400, f"Invalid tile coordinates for zoom {z}")
    
    if z > 18:
        raise HTTPException(400, "Zoom level too high (max 18)")
    
    mvt_bytes = vector_repo.get_mvt_tile(uuid, z, x, y, layer_name)
    
    if mvt_bytes is None:
        # Return empty tile (valid MVT with no features)
        return Response(
            content=b"",
            media_type="application/x-protobuf",
            headers={
                "Content-Length": "0",
                "Cache-Control": "public, max-age=3600"
            }
        )
    
    return Response(
        content=mvt_bytes,
        media_type="application/x-protobuf",
        headers={
            "Cache-Control": "public, max-age=3600",
            "Access-Control-Allow-Origin": "*"
        }
    )


@router.get("/{job_id}.geojson")
async def get_geojson(
    job_id: str,
    min_severity: int = Query(default=1, ge=1, le=3, description="Minimum severity to include")
):
    """
    Get all features for a job as GeoJSON FeatureCollection.
    
    Useful for:
    - Debugging and inspection
    - Export/download
    - Small datasets where MVT overhead isn't worth it
    
    For large feature sets, use MVT tiles instead.
    """
    try:
        uuid = UUID(job_id)
    except ValueError:
        raise HTTPException(400, "Invalid job ID format")
    
    job = job_repo.get(uuid)
    if not job:
        raise HTTPException(404, "Job not found")
    
    geojson = vector_repo.get_features_geojson(uuid, min_severity)
    
    return geojson


@router.get("/{job_id}/stats")
async def get_feature_stats(job_id: str):
    """
    Get aggregated statistics for job features.
    
    Returns:
    - total_features: total number of polygons
    - total_area_m2: sum of all polygon areas
    - by_severity: breakdown by severity level
    """
    try:
        uuid = UUID(job_id)
    except ValueError:
        raise HTTPException(400, "Invalid job ID format")
    
    job = job_repo.get(uuid)
    if not job:
        raise HTTPException(404, "Job not found")
    
    stats = vector_repo.get_stats(uuid)
    
    return {
        "job_id": job_id,
        "algorithm": job.algorithm,
        **stats
    }


@router.delete("/{job_id}")
async def delete_features(job_id: str):
    """
    Delete all vector features for a job.
    
    Note: Features are also deleted automatically when the parent job
    is deleted (CASCADE).
    """
    try:
        uuid = UUID(job_id)
    except ValueError:
        raise HTTPException(400, "Invalid job ID format")
    
    count = vector_repo.delete_by_job(uuid)
    
    return {"deleted": count}


@router.get("/{job_id}/count")
async def count_features(job_id: str):
    """Get count of features for a job."""
    try:
        uuid = UUID(job_id)
    except ValueError:
        raise HTTPException(400, "Invalid job ID format")
    
    count = vector_repo.count_by_job(uuid)
    
    return {"job_id": job_id, "feature_count": count}