"""Database connection pool and job repository."""

import os
import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, List
from uuid import UUID

import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor, Json
from pydantic import BaseModel

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/eo_brain"
)

# Pool size tuned for: analysis workers + tile serving + API handlers
_POOL_MIN = int(os.getenv("DB_POOL_MIN", "2"))
_POOL_MAX = int(os.getenv("DB_POOL_MAX", "20"))

_pool: Optional[ThreadedConnectionPool] = None


def _get_pool() -> ThreadedConnectionPool:
    """Lazy-init singleton pool. Thread-safe by psycopg2 internals."""
    global _pool
    if _pool is None or _pool.closed:
        _pool = ThreadedConnectionPool(_POOL_MIN, _POOL_MAX, DATABASE_URL)
        logger.info(f"DB pool initialized (min={_POOL_MIN}, max={_POOL_MAX})")
    return _pool


def close_pool():
    """Call on application shutdown."""
    global _pool
    if _pool and not _pool.closed:
        _pool.closeall()
        logger.info("DB pool closed")


@contextmanager
def get_connection():
    """Borrow a connection from the pool, auto-return on exit."""
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def _bbox_to_wkt(bbox: List[float]) -> str:
    """Convert [west, south, east, north] to WKT POLYGON."""
    w, s, e, n = bbox
    return f"POLYGON(({w} {s}, {e} {s}, {e} {n}, {w} {n}, {w} {s}))"


def _row_to_job(row: dict) -> "JobRecord":
    bbox_wkt = row.pop('bbox_text', None)
    if bbox_wkt:
        coords = bbox_wkt.replace('POLYGON((', '').replace('))', '').split(',')
        p1 = coords[0].split()
        p3 = coords[2].split()
        row['bbox'] = [float(p1[0]), float(p1[1]), float(p3[0]), float(p3[1])]
    else:
        row['bbox'] = []
    return JobRecord(**row)


class JobRecord(BaseModel):
    id: UUID
    status: str
    algorithm: str
    provider: str
    bbox: List[float]
    start_date: datetime
    end_date: datetime
    params: dict
    progress: int
    message: Optional[str]
    result: Optional[dict]
    cog_path: Optional[str]
    output_crs: Optional[str]
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime]


class JobRepository:

    def create(
        self,
        algorithm: str,
        provider: str,
        bbox: List[float],
        start_date: datetime,
        end_date: datetime,
        params: dict = None
    ) -> JobRecord:
        sql = """
            INSERT INTO jobs (algorithm, provider, bbox, start_date, end_date, params)
            VALUES (%s, %s, ST_GeomFromText(%s, 4326), %s, %s, %s)
            RETURNING *, ST_AsText(bbox) as bbox_text
        """
        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (
                    algorithm, provider, _bbox_to_wkt(bbox),
                    start_date, end_date, Json(params or {})
                ))
                return _row_to_job(dict(cur.fetchone()))

    def get(self, job_id: UUID) -> Optional[JobRecord]:
        sql = "SELECT *, ST_AsText(bbox) as bbox_text FROM jobs WHERE id = %s"
        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (str(job_id),))
                row = cur.fetchone()
                return _row_to_job(dict(row)) if row else None

    def update_progress(
        self,
        job_id: UUID,
        progress: int,
        message: str = None,
        status: str = None
    ):
        sql = """
            UPDATE jobs
            SET progress = %s, message = %s, status = COALESCE(%s, status)
            WHERE id = %s
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (progress, message, status, str(job_id)))

    def complete(
        self,
        job_id: UUID,
        result: dict,
        cog_path: str,
        output_crs: str,
        message: Optional[str] = None,
    ):
        """Mark the job completed; ``message`` replaces the last progress message."""
        sql = """
            UPDATE jobs
            SET status = 'completed', progress = 100, result = %s,
                cog_path = %s, output_crs = %s, completed_at = NOW(),
                message = COALESCE(%s, message)
            WHERE id = %s
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (Json(result), cog_path, output_crs, message, str(job_id)))

    def fail(self, job_id: UUID, message: str):
        sql = """
            UPDATE jobs
            SET status = 'failed', message = %s, completed_at = NOW()
            WHERE id = %s
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (message, str(job_id)))

    def list_recent(self, limit: int = 20) -> List[JobRecord]:
        sql = """
            SELECT *, ST_AsText(bbox) as bbox_text
            FROM jobs
            ORDER BY created_at DESC
            LIMIT %s
        """
        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (limit,))
                return [_row_to_job(dict(row)) for row in cur.fetchall()]

    def delete(self, job_id: UUID) -> bool:
        sql = "DELETE FROM jobs WHERE id = %s"
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (str(job_id),))
                return cur.rowcount > 0


job_repo = JobRepository()