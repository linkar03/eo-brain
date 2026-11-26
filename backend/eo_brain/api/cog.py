"""
COG (Cloud Optimized GeoTIFF) generation utilities.

Uses GDAL's native COG driver (GDAL >= 3.1) for single-pass writing,
eliminating the temp-file + cog_translate two-pass overhead.

Supports:
- Discrete maps (Severity classification: 0-3, uint8)
- Continuous maps (NDVI, NDWI, MNDWI, float32)
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import numpy as np
import xarray as xr
import rioxarray  # noqa: F401

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output"))

NODATA_UINT8 = 255
NODATA_FLOAT32 = np.nan
BLOCK_SIZE = 256

# Overview resampling per data type
_RESAMPLING = {"discrete": "nearest", "continuous": "bilinear"}


def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _prepare_array(arr: xr.DataArray) -> xr.DataArray:
    """Remove time/band dims if present, ensure 2D spatial array."""
    if "time" in arr.dims:
        arr = arr.isel(time=0)
    if "band" in arr.dims:
        arr = arr.isel(band=0)
    return arr


def _write_cog(
    da: xr.DataArray,
    output_path: Path,
    dtype: str,
    nodata: Union[int, float],
    overview_resampling: str = "nearest",
):
    """
    Single-pass COG write using GDAL's native COG driver.
    No temp files, no cog_translate round-trip.
    """
    logger.info(f"Writing COG: {output_path.name} (dtype={dtype})")

    da.rio.to_raster(
        str(output_path),
        driver="COG",
        dtype=dtype,
        compress="lzw",
        blocksize=BLOCK_SIZE,
        overview_resampling=overview_resampling,
        windowed=True,
        lock=True,
        compute=True,
    )

    file_size = output_path.stat().st_size / 1024
    logger.info(f"Generated COG: {output_path.name} ({file_size:.1f} KB)")


def _save_cog(
    ds: xr.Dataset,
    job_id: str,
    crs: str,
    var_name: str,
    is_continuous: bool,
    suffix: str = "",
) -> str:
    """
    Extract a variable from the dataset and save as COG.
    Returns the output file path.
    """
    ensure_output_dir()

    if var_name not in ds.data_vars:
        available = list(ds.data_vars)
        raise ValueError(f"Variable '{var_name}' not found. Available: {available}")

    data = _prepare_array(ds[var_name])

    if is_continuous:
        dtype, nodata = "float32", NODATA_FLOAT32
        resampling = _RESAMPLING["continuous"]
        data = data.astype(dtype)
        data = xr.where(np.isinf(data), np.nan, data)
    else:
        dtype, nodata = "uint8", NODATA_UINT8
        resampling = _RESAMPLING["discrete"]
        data = np.abs(data).clip(0, 254).astype(dtype)

    if data.rio.crs is None:
        data = data.rio.write_crs(crs)
    data = data.rio.write_nodata(nodata)

    filename = f"{job_id}_{suffix}.tif" if suffix else f"{job_id}.tif"
    output_path = OUTPUT_DIR / filename

    _write_cog(data, output_path, dtype, nodata, resampling)
    return str(output_path)


def compute_stats(da: xr.DataArray) -> Dict[str, Any]:
    """
    Compute stats from an in-memory DataArray before writing to disk.
    Avoids re-reading the COG after generation.
    """
    da = _prepare_array(da)

    try:
        valid = da.where(da != da.rio.nodata if da.rio.nodata is not None else True)
        count = int(valid.count().values)
        if count == 0:
            return {}

        stats = {
            "min": float(valid.min().values),
            "max": float(valid.max().values),
            "mean": float(valid.mean().values),
        }

        # Class counts for discrete outputs (low max value = classification)
        if stats["max"] is not None and stats["max"] <= 10:
            flat = valid.values.flatten()
            flat = flat[~np.isnan(flat)]
            unique, counts = np.unique(flat, return_counts=True)
            for val, cnt in zip(unique, counts):
                stats[f"count_{int(val)}"] = int(cnt)

        return {"band_1": stats}
    except Exception as e:
        logger.error(f"Stats computation failed: {e}")
        return {}


# Public API (backward compatible wrappers)

def save_severity_cog(
    ds: xr.Dataset,
    job_id: str,
    crs: str,
    bbox: tuple,
    algorithm: str,
    severity_var: Optional[str] = None,
) -> str:
    if not severity_var:
        candidates = ["lisa_severity", "yoy_severity", "flood_severity", "max_severity"]
        severity_var = next(
            (v for v in candidates if v in ds.data_vars), None
        )
        if not severity_var:
            raise ValueError(f"No severity variable found. Available: {list(ds.data_vars)}")

    return _save_cog(ds, job_id, crs, var_name=severity_var, is_continuous=False)


def save_ndvi_cog(
    ds: xr.Dataset, job_id: str, crs: str, bbox: tuple,
) -> str:
    return _save_cog(ds, job_id, crs, var_name="ndvi", is_continuous=True)


def save_water_cog(
    ds: xr.Dataset, job_id: str, crs: str, bbox: tuple, index_name: str = "ndwi",
) -> str:
    return _save_cog(ds, job_id, crs, var_name=index_name, is_continuous=True)


def save_multiband_cog(
    ds: xr.Dataset,
    job_id: str,
    crs: str,
    bands: List[str],
    suffix: str = "bands",
) -> str:
    """Save multiple bands as a single multi-band COG."""
    ensure_output_dir()

    available_bands = [b for b in bands if b in ds.data_vars]
    if not available_bands:
        raise ValueError(f"None of requested bands {bands} found")

    band_arrays = [_prepare_array(ds[b]) for b in available_bands]
    stacked = xr.concat(band_arrays, dim="band")
    stacked = stacked.assign_coords(band=available_bands)

    if stacked.rio.crs is None:
        stacked = stacked.rio.write_crs(crs)

    output_path = OUTPUT_DIR / f"{job_id}_{suffix}.tif"

    logger.info(f"Writing multiband COG: {output_path.name}")
    stacked.rio.to_raster(
        str(output_path),
        driver="COG",
        compress="lzw",
        blocksize=BLOCK_SIZE,
        windowed=True,
        lock=True,
        compute=True,
    )

    logger.info(f"Multi-band COG: {output_path}")
    return str(output_path)


def cleanup_job_cogs(job_id: str):
    """Remove all COG files for a job."""
    for f in OUTPUT_DIR.glob(f"{job_id}*.tif"):
        try:
            f.unlink()
            logger.info(f"Deleted: {f}")
        except Exception as e:
            logger.warning(f"Failed to delete {f}: {e}")