"""
Vector feature repository for PostGIS operations.

Handles bulk insert of polygonized features and MVT tile generation.
Uses raw psycopg2 for performance (no ORM overhead).
"""

from typing import List, Dict, Any, Optional
from uuid import UUID
import logging

from psycopg2.extras import execute_values
from eo_brain.db.database import get_connection

logger = logging.getLogger(__name__)


class VectorRepository:
    """Repository for alert_features table operations."""
    
    def bulk_insert(self, features: List[Dict[str, Any]]) -> int:
        """
        Bulk insert features using execute_values for performance.
        
        Each feature dict must contain:
            - job_id: UUID
            - algorithm: str
            - severity: int (1-3)
            - geometry_wkt: str (WKT in EPSG:4326)
            - centroid_wkt: str (WKT in EPSG:4326)
            - area_m2: float
            - perimeter_m: float (optional)
            - quadrant: int (optional, LISA only)
        
        Returns number of inserted rows.
        """
        if not features:
            return 0
        
        sql = """
            INSERT INTO alert_features 
                (job_id, algorithm, severity, quadrant, geometry, centroid, area_m2, perimeter_m)
            VALUES %s
        """
        
        # Template for execute_values
        template = """(
            %(job_id)s,
            %(algorithm)s,
            %(severity)s,
            %(quadrant)s,
            ST_GeomFromText(%(geometry_wkt)s, 4326),
            ST_GeomFromText(%(centroid_wkt)s, 4326),
            %(area_m2)s,
            %(perimeter_m)s
        )"""
        
        with get_connection() as conn:
            with conn.cursor() as cur:
                execute_values(
                    cur, sql, features,
                    template=template,
                    page_size=1000
                )
                count = cur.rowcount
        
        logger.info(f"Inserted {count} vector features")
        return count
    
    def get_mvt_tile(
        self, 
        job_id: UUID, 
        z: int, 
        x: int, 
        y: int,
        layer_name: str = "alerts"
    ) -> Optional[bytes]:
        """
        Generate MVT tile for given coordinates.
        
        Uses ST_AsMVT with proper index utilization:
        - Transforms tile envelope to 4326 (not geometry column)
        - This allows GIST index on geometry to be used
        
        Returns binary MVT data or None if no features in tile.
        """
        sql = """
            WITH 
            tile_bounds AS (
                SELECT ST_Transform(ST_TileEnvelope(%(z)s, %(x)s, %(y)s), 4326) AS geom_4326,
                       ST_TileEnvelope(%(z)s, %(x)s, %(y)s) AS geom_3857
            ),
            tile_features AS (
                SELECT 
                    f.id,
                    f.severity,
                    f.quadrant,
                    f.algorithm,
                    f.area_m2,
                    ST_AsMVTGeom(
                        ST_Transform(f.geometry, 3857),
                        t.geom_3857,
                        4096,
                        64,
                        true
                    ) AS geom
                FROM alert_features f, tile_bounds t
                WHERE f.job_id = %(job_id)s
                  AND ST_Intersects(f.geometry, t.geom_4326)
            )
            SELECT ST_AsMVT(tile_features, %(layer_name)s, 4096, 'geom')
            FROM tile_features
            WHERE geom IS NOT NULL;
        """
        
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, {
                    "job_id": str(job_id),
                    "z": z,
                    "x": x,
                    "y": y,
                    "layer_name": layer_name
                })
                result = cur.fetchone()
                
                if result and result[0]:
                    mvt_bytes = bytes(result[0])
                    return mvt_bytes if len(mvt_bytes) > 0 else None
                return None
    
    def get_features_geojson(
        self, 
        job_id: UUID,
        severity_min: int = 1
    ) -> Dict[str, Any]:
        """
        Get all features for a job as GeoJSON FeatureCollection.
        Useful for small result sets or debugging.
        """
        sql = """
            SELECT json_build_object(
                'type', 'FeatureCollection',
                'features', COALESCE(json_agg(
                    json_build_object(
                        'type', 'Feature',
                        'id', id,
                        'geometry', ST_AsGeoJSON(geometry)::json,
                        'properties', json_build_object(
                            'severity', severity,
                            'quadrant', quadrant,
                            'algorithm', algorithm,
                            'area_m2', area_m2
                        )
                    )
                ), '[]'::json)
            )
            FROM alert_features
            WHERE job_id = %s AND severity >= %s;
        """
        
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (str(job_id), severity_min))
                result = cur.fetchone()
                return result[0] if result else {"type": "FeatureCollection", "features": []}
    
    def get_stats(self, job_id: UUID) -> Dict[str, Any]:
        """Get aggregated statistics for job features."""
        sql = """
            SELECT 
                algorithm,
                severity,
                COUNT(*) as count,
                SUM(area_m2) as total_area_m2,
                AVG(area_m2) as avg_area_m2
            FROM alert_features
            WHERE job_id = %s
            GROUP BY algorithm, severity
            ORDER BY severity DESC;
        """
        
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (str(job_id),))
                rows = cur.fetchall()
                
                stats = {
                    "total_features": 0,
                    "total_area_m2": 0,
                    "by_severity": {}
                }
                
                for algo, sev, count, total_area, avg_area in rows:
                    stats["total_features"] += count
                    stats["total_area_m2"] += total_area or 0
                    stats["by_severity"][sev] = {
                        "count": count,
                        "total_area_m2": total_area,
                        "avg_area_m2": avg_area
                    }
                
                return stats
    
    def delete_by_job(self, job_id: UUID) -> int:
        """Delete all features for a job. Returns deleted count."""
        sql = "DELETE FROM alert_features WHERE job_id = %s"
        
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (str(job_id),))
                return cur.rowcount
    
    def count_by_job(self, job_id: UUID) -> int:
        """Count features for a job."""
        sql = "SELECT COUNT(*) FROM alert_features WHERE job_id = %s"
        
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (str(job_id),))
                result = cur.fetchone()
                return result[0] if result else 0


# Singleton instance
vector_repo = VectorRepository()