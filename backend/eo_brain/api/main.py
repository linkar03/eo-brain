"""
EO Brain API - FastAPI Application

Satellite intelligence platform with streaming pipeline architecture.
Uses Dask for distributed computing to handle large areas without OOM.
"""

import os
import sys
import asyncio
import logging
import uuid as uuid_mod
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import numpy as np
import rioxarray

# Configure logging FIRST, before any getLogger calls
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    force=True,  # Override any prior basicConfig from double-import
)
logger = logging.getLogger(__name__)

from eo_brain.core.datacube import (
    DataCube, BBox, TimeRange, SearchParams, LoadConfig, LoadResult, PROVIDERS
)
from eo_brain.algorithms.base import get_algorithm, ALGORITHMS
from eo_brain.db.database import job_repo, get_connection, close_pool
from eo_brain.api.tiles import router as tiles_router, tile_query_for_algorithm
from eo_brain.api.vectors import router as vectors_router
from eo_brain.api.cog import save_severity_cog, cleanup_job_cogs, save_ndvi_cog, save_water_cog
from eo_brain.algorithms.vectorize import vectorize_cog, VectorizeConfig
from eo_brain.db.vector_repo import vector_repo

# Lazy agent loader to avoid double-import issues with uvicorn --reload.
# Instead of importing at module level (which runs twice and can leave
# _AGENT_AVAILABLE in an inconsistent state), we import on first use.
_agent_chat_fn = None
_agent_checked = False


def _get_agent_chat():
    """Lazy-load agent module. Safe to call multiple times."""
    global _agent_chat_fn, _agent_checked
    if _agent_checked:
        return _agent_chat_fn
    _agent_checked = True
    try:
        from eo_brain.agent.brain import chat as _chat
        _agent_chat_fn = _chat
        logger.info("Agent module loaded successfully")
    except Exception as e:
        logger.warning(f"Agent module unavailable: {e}", exc_info=True)
        _agent_chat_fn = None
    return _agent_chat_fn


class AnalysisError(Exception):
    pass

class DataNotFoundError(AnalysisError):
    pass

class ProviderError(AnalysisError):
    pass

class ComputeError(AnalysisError):
    pass


class Settings(BaseModel):
    default_provider: str = os.getenv("STAC_PROVIDER", "element84-s2")
    output_dir: str = os.getenv("OUTPUT_DIR", "./output")
    max_area_km2: float = float(os.getenv("MAX_AREA_KM2", "50000"))
    analysis_timeout_sec: int = int(os.getenv("ANALYSIS_TIMEOUT_SEC", "600"))
    api_base_url: str = os.getenv("API_BASE_URL", "http://localhost:8000")


settings = Settings()
_running_tasks: Dict[UUID, asyncio.Task] = {}

CONTINUOUS_OUTPUT_ALGORITHMS = {"ndvi", "ndwi", "mndwi"}
DISCRETE_OUTPUT_ALGORITHMS = {"lisa", "flood"}

ALGORITHM_BANDS = {
    "ndvi": ["red", "nir"],
    "lisa": ["red", "nir"],
    "ndwi": ["green", "nir"],
    "mndwi": ["green", "swir16"],
    "flood": ["green", "nir", "swir16"],
}


def get_required_bands(algorithm: str, provider) -> List[str]:
    base_bands = ALGORITHM_BANDS.get(algorithm, ["red", "nir"])
    if provider.cloud_mask_band and provider.cloud_mask_band not in base_bands:
        return base_bands + [provider.cloud_mask_band]
    return base_bands


def reset_stalled_jobs():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE jobs 
                SET status = 'failed', 
                    message = 'Server restarted while job was processing',
                    completed_at = NOW()
                WHERE status = 'processing'
                RETURNING id
            """)
            stalled = cur.fetchall()
            if stalled:
                logger.warning(f"Reset {len(stalled)} stalled jobs")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting EO Brain API (provider: {settings.default_provider})")
    os.makedirs(settings.output_dir, exist_ok=True)
    reset_stalled_jobs()

    # Pre-warm STAC client
    try:
        warmup_cube = DataCube(provider=settings.default_provider)
        _ = warmup_cube.client
        logger.info(f"STAC client pre-warmed for {settings.default_provider}")
    except Exception as e:
        logger.warning(f"STAC pre-warm failed (non-fatal): {e}")

    # Pre-warm agent (triggers lazy import, logs result)
    _get_agent_chat()

    yield
    for job_id, task in _running_tasks.items():
        if not task.done():
            task.cancel()
    if _running_tasks:
        await asyncio.gather(*_running_tasks.values(), return_exceptions=True)
    close_pool()
    logger.info("Shutdown complete")


app = FastAPI(
    title="EO Brain - Satellite Intelligence Platform",
    version="1.3.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(tiles_router)
app.include_router(vectors_router)


def estimate_bbox_area_km2(bbox: List[float]) -> float:
    west, south, east, north = bbox
    lat_mid = (south + north) / 2
    width_km = (east - west) * 111 * np.cos(np.radians(lat_mid))
    height_km = (north - south) * 111
    return abs(width_km * height_km)


# --- Request / Response Models ---

class AnalysisRequest(BaseModel):
    bbox: List[float] = Field(..., min_length=4, max_length=4)
    start_date: datetime
    end_date: datetime
    algorithm: str
    provider: Optional[str] = None
    resolution_m: int = Field(default=10, ge=10, le=100)
    apply_cloud_mask: bool = True
    params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('bbox')
    @classmethod
    def validate_bbox_geometry(cls, v):
        west, south, east, north = v
        west = (west + 180) % 360 - 180
        east = (east + 180) % 360 - 180
        v = [west, south, east, north]
        if south >= north:
            raise ValueError("south must be < north")
        if not (-90 <= south <= 90 and -90 <= north <= 90):
            raise ValueError("latitude must be in [-90, 90]")
        # The limit advertised by GET / (MAX_AREA_KM2) is enforced here, so an
        # oversized request is rejected with 422 before a job row is created.
        area = estimate_bbox_area_km2(v)
        if area > settings.max_area_km2:
            raise ValueError(
                f"area too large: {area:.0f} km2 (limit: {settings.max_area_km2:.0f} km2)"
            )
        return v

    @field_validator('end_date')
    @classmethod
    def validate_date_range(cls, v, info):
        start = info.data.get('start_date')
        if start and v <= start:
            raise ValueError("end_date must be after start_date")
        return v


class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    message: Optional[str]
    result: Optional[Dict[str, Any]]
    tile_url: Optional[str] = None
    vector_url: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None


class QueryPayload(BaseModel):
    """Read-only query run by the data agent: the executed SQL and its rows."""
    sql: str
    columns: List[str]
    rows: List[List[Any]]
    row_count: int
    truncated: bool


class ChatResponse(BaseModel):
    text: str
    job_id: Optional[str] = None
    session_id: str
    bbox: Optional[List[float]] = None
    query: Optional[QueryPayload] = None


# --- Shared Helpers ---

def _build_tile_url(job_id: str, algorithm: str) -> str:
    # Colormap and rescale come from the tile server's per-algorithm defaults,
    # the same ones it applies to a request without query parameters.
    base_url = f"{settings.api_base_url}/tiles/{job_id}/{{z}}/{{x}}/{{y}}.png"
    return f"{base_url}?{tile_query_for_algorithm(algorithm)}"


def _calculate_stats_from_cog(cog_path: str) -> Dict[str, Any]:
    try:
        with rioxarray.open_rasterio(cog_path) as ds:
            valid = ds.where(ds != ds.rio.nodata)
            if valid.count() == 0:
                return {}
            stats = {
                "min": float(valid.min().values),
                "max": float(valid.max().values),
                "mean": float(valid.mean().values),
            }
            if stats["max"] is not None and stats["max"] <= 10:
                data = valid.values.flatten()
                data = data[~np.isnan(data)]
                unique, counts = np.unique(data, return_counts=True)
                for val, count in zip(unique, counts):
                    stats[f"count_{int(val)}"] = int(count)
            return {"band_1": stats}
    except Exception as e:
        logger.error(f"Failed to calculate stats from COG: {e}")
        return {}


# --- Analysis Pipeline ---

def _optimal_chunks(resolution_m: int) -> dict:
    if resolution_m <= 10:
        return {"x": 2048, "y": 2048, "time": 1}
    elif resolution_m <= 30:
        return {"x": 1024, "y": 1024, "time": 1}
    return {"x": 512, "y": 512, "time": 1}


def _run_analysis_sync(job_id: UUID, request: AnalysisRequest, provider_name: str):
    """
    Synchronous analysis worker. Runs in thread pool executor.
    """
    try:
        job_repo.update_progress(job_id, 5, "Initializing analysis engine...", "processing")

        bbox = BBox.from_tuple(tuple(request.bbox))
        cube = DataCube(provider=provider_name)

        config = LoadConfig(
            resolution_m=request.resolution_m,
            chunks=_optimal_chunks(request.resolution_m),
        )

        bands = get_required_bands(request.algorithm, cube.provider)
        logger.info(f"Job {job_id}: Loading bands {bands} at {request.resolution_m}m for {request.algorithm}")

        job_repo.update_progress(job_id, 10, "Searching satellite catalog...")

        params = SearchParams(
            bbox=bbox,
            time_range=TimeRange(start=request.start_date, end=request.end_date)
        ).with_query(**{"eo:cloud_cover": {"lt": 30}})

        job_repo.update_progress(job_id, 15, "Connecting to data provider...")

        try:
            load_result: LoadResult = cube.load(params, bands=bands, config=config)
        except Exception as e:
            raise ProviderError(f"STAC provider error: {e}") from e

        if not load_result.data.data_vars:
            raise DataNotFoundError(
                "No satellite data found. Try a wider date range or different area."
            )

        job_repo.update_progress(job_id, 30, "Satellite data loaded, preparing...")

        ds = load_result.data
        if request.apply_cloud_mask:
            job_repo.update_progress(job_id, 35, "Filtering clouds...")
            ds = load_result.apply_cloud_mask()

        job_repo.update_progress(job_id, 40, f"Computing {request.algorithm.upper()} analysis...")

        try:
            algo = get_algorithm(request.algorithm)
            result = algo.run(ds, resolution_m=request.resolution_m, **request.params)
        except ValueError as e:
            raise ComputeError(str(e)) from e

        job_repo.update_progress(job_id, 55, "Materializing results...")

        result.data = result.data.compute()

        job_repo.update_progress(job_id, 65, "Generating map tiles...")

        try:
            if request.algorithm in {"ndwi", "mndwi"}:
                cog_path = save_water_cog(
                    ds=result.data, job_id=str(job_id), crs=load_result.crs,
                    bbox=request.bbox, index_name=request.algorithm
                )
            elif request.algorithm == "ndvi":
                cog_path = save_ndvi_cog(
                    ds=result.data, job_id=str(job_id), crs=load_result.crs,
                    bbox=request.bbox
                )
            elif request.algorithm == "flood":
                cog_path = save_severity_cog(
                    ds=result.data, job_id=str(job_id), crs=load_result.crs,
                    bbox=request.bbox, algorithm=request.algorithm,
                    severity_var="flood_severity"
                )
            elif request.algorithm == "lisa":
                cog_path = save_severity_cog(
                    ds=result.data, job_id=str(job_id), crs=load_result.crs,
                    bbox=request.bbox, algorithm=request.algorithm,
                    severity_var="lisa_severity"
                )
            else:
                cog_path = save_severity_cog(
                    ds=result.data, job_id=str(job_id), crs=load_result.crs,
                    bbox=request.bbox, algorithm=request.algorithm
                )
        except MemoryError:
            raise ComputeError("Out of memory during COG streaming. Try reducing area size.")
        except Exception as e:
            raise ComputeError(f"COG generation failed: {e}") from e

        job_repo.update_progress(job_id, 80, "Computing statistics...")

        summary = _calculate_stats_from_cog(cog_path)

        job_repo.update_progress(job_id, 85, "Extracting vector features...")

        vector_stats = None
        if request.algorithm in DISCRETE_OUTPUT_ALGORITHMS:
            try:
                vec_result = vectorize_cog(
                    cog_path=cog_path, job_id=job_id,
                    algorithm=request.algorithm,
                    config=VectorizeConfig(min_area_pixels=4, simplify_tolerance_factor=0.5)
                )
                vector_stats = {
                    "feature_count": vec_result.feature_count,
                    "total_area_m2": vec_result.total_area_m2,
                    "by_severity": vec_result.by_severity
                }
            except Exception as e:
                logger.warning(f"Job {job_id}: vectorization failed (non-fatal): {e}")
                vector_stats = {"error": str(e)}

        job_repo.update_progress(job_id, 95, "Preparing map layer...")

        tile_url = _build_tile_url(str(job_id), request.algorithm)

        vector_url = None
        if request.algorithm in DISCRETE_OUTPUT_ALGORITHMS:
            vector_url = f"{settings.api_base_url}/vectors/{job_id}/{{z}}/{{x}}/{{y}}.pbf"

        job_repo.complete(
            job_id,
            result={
                "summary": summary, "bbox": request.bbox,
                "metadata": result.metadata, "crs": load_result.crs,
                "tile_url": tile_url, "vector_url": vector_url,
                "vector_stats": vector_stats
            },
            cog_path=cog_path, output_crs=load_result.crs
        )
        logger.info(f"Job {job_id} completed successfully")

    except DataNotFoundError as e:
        logger.warning(f"Job {job_id} - no data: {e}")
        job_repo.fail(job_id, str(e))
    except ProviderError as e:
        logger.error(f"Job {job_id} - provider error: {e}")
        job_repo.fail(job_id, f"Data provider error: {e}")
    except ComputeError as e:
        logger.error(f"Job {job_id} - compute error: {e}")
        job_repo.fail(job_id, str(e))
    except asyncio.CancelledError:
        job_repo.fail(job_id, "Job cancelled")
        raise
    except Exception as e:
        logger.exception(f"Job {job_id} unexpected failure")
        job_repo.fail(job_id, f"Internal error: {type(e).__name__}")


async def _run_analysis_async(job_id: UUID, request: AnalysisRequest, provider_name: str):
    loop = asyncio.get_running_loop()
    try:
        await asyncio.wait_for(
            loop.run_in_executor(None, _run_analysis_sync, job_id, request, provider_name),
            timeout=settings.analysis_timeout_sec
        )
    except asyncio.TimeoutError:
        logger.error(f"Job {job_id} timed out after {settings.analysis_timeout_sec}s")
        job_repo.fail(job_id, f"Analysis timed out after {settings.analysis_timeout_sec}s")
    except asyncio.CancelledError:
        job_repo.fail(job_id, "Job cancelled")
    finally:
        _running_tasks.pop(job_id, None)


def _schedule_job(job_id: UUID, request: AnalysisRequest, provider_name: str):
    task = asyncio.create_task(_run_analysis_async(job_id, request, provider_name))
    _running_tasks[job_id] = task


# === API Endpoints ===

@app.get("/")
async def root():
    return {
        "service": "EO Brain - Satellite Intelligence Platform",
        "version": "1.3.0",
        "providers": list(PROVIDERS.keys()),
        "algorithms": {
            name: {
                "description": cls.description,
                "output_type": "continuous" if name in CONTINUOUS_OUTPUT_ALGORITHMS else "discrete",
                "required_bands": ALGORITHM_BANDS.get(name, [])
            }
            for name, cls in ALGORITHMS.items()
        },
        "limits": {"max_area_km2": settings.max_area_km2}
    }


@app.get("/health")
async def health():
    agent_fn = _get_agent_chat()
    return {
        "status": "healthy",
        "running_jobs": len(_running_tasks),
        "agent_available": agent_fn is not None,
    }


@app.get("/providers")
async def list_providers():
    return {
        name: {"catalog_url": cls.catalog_url, "collections": cls.default_collections}
        for name, cls in PROVIDERS.items()
    }


@app.get("/algorithms")
async def list_algorithms():
    return {
        name: {
            "description": cls.description,
            "output_type": "continuous" if name in CONTINUOUS_OUTPUT_ALGORITHMS else "discrete",
            "required_bands": ALGORITHM_BANDS.get(name, [])
        }
        for name, cls in ALGORITHMS.items()
    }


@app.post("/analyze", response_model=JobResponse)
async def submit_analysis(request: AnalysisRequest):
    provider_name = request.provider or settings.default_provider
    if provider_name not in PROVIDERS:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Unknown provider: {provider_name}")
    if request.algorithm not in ALGORITHMS:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Unknown algorithm: {request.algorithm}")

    job = job_repo.create(
        algorithm=request.algorithm, provider=provider_name,
        bbox=request.bbox, start_date=request.start_date,
        end_date=request.end_date, params=request.params
    )
    _schedule_job(job.id, request, provider_name)
    return JobResponse(job_id=str(job.id), status="pending", message="Job queued")


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    try:
        uid = UUID(job_id)
    except ValueError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid job ID format")

    job = job_repo.get(uid)
    if not job:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")

    tile_url = job.result.get("tile_url") if job.result else None
    vector_url = job.result.get("vector_url") if job.result else None

    return JobStatusResponse(
        job_id=str(job.id), status=job.status, progress=job.progress,
        message=job.message, result=job.result, tile_url=tile_url,
        vector_url=vector_url, created_at=job.created_at,
        completed_at=job.completed_at
    )


@app.get("/jobs")
async def list_jobs(limit: int = Query(default=20, le=100)):
    jobs = job_repo.list_recent(limit)
    return [
        {"job_id": str(j.id), "status": j.status, "algorithm": j.algorithm,
         "progress": j.progress, "created_at": j.created_at.isoformat()}
        for j in jobs
    ]


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    try:
        uid = UUID(job_id)
    except ValueError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid job ID format")

    if uid in _running_tasks:
        _running_tasks[uid].cancel()

    cleanup_job_cogs(job_id)
    try:
        vector_repo.delete_by_job(uid)
    except Exception as e:
        logger.warning(f"Failed to delete vector features for {job_id}: {e}")

    if not job_repo.delete(uid):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    return {"deleted": True}


# --- Agent Chat Endpoint ---

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    Agent chat. Runs Gemini call in a thread executor so it doesn't
    block the event loop (job polling, tile serving stay responsive).
    """
    session_id = req.session_id or str(uuid_mod.uuid4())

    agent_chat = _get_agent_chat()

    if agent_chat is None:
        logger.warning("Chat request received but agent is unavailable")
        return ChatResponse(
            text="AI agent is not available. Check server logs for details.",
            job_id=None,
            session_id=session_id,
        )

    logger.info(f"Chat request: session={session_id[:8]}..., message='{req.message[:80]}'")

    try:
        result = await asyncio.to_thread(
            agent_chat, message=req.message, session_id=session_id
        )
    except Exception as e:
        logger.error(f"Agent chat failed: {e}", exc_info=True)
        return ChatResponse(
            text=f"Agent error: {type(e).__name__}: {e}",
            job_id=None,
            session_id=session_id,
        )

    logger.info(f"Chat result: job_id={result.get('job_id')}, text_len={len(result.get('text', ''))}")

    if result.get("job_id"):
        try:
            uid = UUID(result["job_id"])
            job = job_repo.get(uid)
            if job and job.status == "pending":
                provider_name = job.provider or settings.default_provider
                analysis_req = AnalysisRequest(
                    bbox=job.bbox,
                    start_date=job.start_date,
                    end_date=job.end_date,
                    algorithm=job.algorithm,
                    provider=provider_name,
                    resolution_m=job.params.get("resolution_m", 10),
                    apply_cloud_mask=job.params.get("apply_cloud_mask", True),
                    params={k: v for k, v in job.params.items()
                            if k not in ("resolution_m", "apply_cloud_mask", "geocoded_name")},
                )
                _schedule_job(uid, analysis_req, provider_name)
                logger.info(f"Scheduled job {uid} for processing")
            else:
                logger.warning(f"Job {uid} not found or not pending (status={job.status if job else 'N/A'})")
        except Exception as e:
            logger.error(f"Failed to schedule agent job: {e}", exc_info=True)
            # Mark job failed so frontend stops polling
            try:
                job_repo.fail(uid, f"Scheduling error: {e}")
            except Exception:
                pass

    return ChatResponse(
        text=result["text"],
        job_id=result.get("job_id"),
        session_id=session_id,
        bbox=result.get("bbox"),
        query=result.get("query"),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("eo_brain.api.main:app", host="0.0.0.0", port=8000, reload=True)