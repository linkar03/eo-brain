"""
Tile server endpoints using rio-tiler.

Colormaps are pre-built as NumPy LUTs (256x4 uint8) for O(1) rendering.
Discrete maps use direct index lookup, continuous maps normalize then LUT.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict
from io import BytesIO

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Response
from PIL import Image

from rio_tiler.io import Reader
from rio_tiler.errors import TileOutsideBounds

router = APIRouter(prefix="/tiles", tags=["tiles"])

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output"))


def _build_lut(mapping: Dict[int, Tuple[int, int, int, int]]) -> np.ndarray:
    """Convert a {value: (r,g,b,a)} dict to a (256, 4) LUT array."""
    lut = np.zeros((256, 4), dtype=np.uint8)
    for val, color in mapping.items():
        if 0 <= val < 256:
            lut[val] = color
    return lut


def _build_ndvi_lut() -> np.ndarray:
    """Brown -> yellow -> green gradient for vegetation."""
    lut = np.zeros((256, 4), dtype=np.uint8)

    # Control points: (t_start, t_end, rgb_start, rgb_end)
    segments = [
        (0.00, 0.20, (139, 90, 43),  (139, 90, 43)),
        (0.20, 0.35, (139, 90, 43),  (210, 180, 100)),
        (0.35, 0.50, (210, 180, 100), (180, 220, 80)),
        (0.50, 0.65, (180, 220, 80),  (100, 200, 100)),
        (0.65, 0.80, (100, 200, 100), (34, 139, 34)),
        (0.80, 1.00, (34, 139, 34),   (0, 100, 0)),
    ]

    for i in range(256):
        t = i / 255.0
        for t0, t1, c0, c1 in segments:
            if t0 <= t < t1 or (t1 == 1.0 and t == 1.0):
                p = (t - t0) / (t1 - t0) if t1 > t0 else 0
                r = int(c0[0] + (c1[0] - c0[0]) * p)
                g = int(c0[1] + (c1[1] - c0[1]) * p)
                b = int(c0[2] + (c1[2] - c0[2]) * p)
                lut[i] = (r, g, b, 220 if i > 10 else 0)
                break

    lut[0] = (0, 0, 0, 0)
    return lut


def _build_water_lut() -> np.ndarray:
    """Brown (dry) -> white (neutral) -> blue (water) gradient."""
    lut = np.zeros((256, 4), dtype=np.uint8)

    segments = [
        (0.00, 0.30, (139, 90, 43),   (210, 180, 140), (200, 200)),
        (0.30, 0.45, (210, 180, 140),  (200, 200, 200), (180, 180)),
        (0.45, 0.50, (200, 200, 200),  (240, 240, 255), (150, 150)),
        (0.50, 0.55, (240, 240, 255),  (200, 220, 255), (150, 200)),
        (0.55, 0.70, (200, 220, 255),  (65, 150, 255),  (200, 230)),
        (0.70, 0.85, (65, 150, 255),   (30, 105, 225),  (240, 240)),
        (0.85, 1.00, (30, 105, 225),   (0, 60, 180),    (255, 255)),
    ]

    for i in range(256):
        t = i / 255.0
        for t0, t1, c0, c1, (a0, a1) in segments:
            if t0 <= t < t1 or (t1 == 1.0 and t == 1.0):
                p = (t - t0) / (t1 - t0) if t1 > t0 else 0
                r = int(c0[0] + (c1[0] - c0[0]) * p)
                g = int(c0[1] + (c1[1] - c0[1]) * p)
                b = int(c0[2] + (c1[2] - c0[2]) * p)
                a = int(a0 + (a1 - a0) * p)
                lut[i] = (r, g, b, a)
                break

    lut[0] = (0, 0, 0, 0)
    return lut


# Pre-computed LUTs (module-level, built once at import)
_SEVERITY_LUT = _build_lut({
    0: (0, 0, 0, 0),
    1: (255, 235, 59, 200),
    2: (255, 152, 0, 220),
    3: (244, 67, 54, 240),
})

_FLOOD_LUT = _build_lut({
    0: (0, 0, 0, 0),
    1: (173, 216, 230, 180),
    2: (65, 105, 225, 220),
    3: (0, 0, 139, 255),
})

_NDVI_LUT = _build_ndvi_lut()
_WATER_LUT = _build_water_lut()


# Colormap registry: name -> (lut, default_rescale, is_discrete)
_COLORMAPS = {
    "severity": (_SEVERITY_LUT, None, True),
    "flood":    (_FLOOD_LUT, None, True),
    "ndvi":     (_NDVI_LUT, (-0.1, 0.8), False),
    "water":    (_WATER_LUT, (-0.3, 0.8), False),
}


def _get_colormap(name: str):
    """Lookup colormap config, fallback to NDVI."""
    return _COLORMAPS.get(name.lower(), (_NDVI_LUT, (-0.2, 0.8), False))


def _apply_discrete(data: np.ndarray, mask: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Single LUT lookup for discrete severity values."""
    indices = data.astype(np.uint8)
    rgba = lut[indices]
    rgba[mask == 0, 3] = 0
    return rgba


def _apply_continuous(
    data: np.ndarray,
    mask: np.ndarray,
    vmin: float,
    vmax: float,
    lut: np.ndarray
) -> np.ndarray:
    """Normalize to 1-255 range, then single LUT lookup. 0 = nodata."""
    range_val = max(vmax - vmin, 1e-10)
    normalized = np.clip((data - vmin) / range_val, 0, 1)
    scaled = (normalized * 254 + 1).astype(np.uint8)
    scaled[mask == 0] = 0
    scaled[np.isnan(data)] = 0
    return lut[scaled]


def _render_png(rgba: np.ndarray, tilesize: int) -> bytes:
    """RGBA array -> PNG bytes, with optional downscale to 256."""
    img = Image.fromarray(rgba, mode="RGBA")
    if tilesize != 256:
        img = img.resize((256, 256), Image.Resampling.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


_EMPTY_TILE = _render_png(np.zeros((256, 256, 4), dtype=np.uint8), 256)

_TILE_HEADERS = {
    "Cache-Control": "public, max-age=3600",
    "Access-Control-Allow-Origin": "*",
}


def _cog_path(job_id: str) -> Path:
    return OUTPUT_DIR / f"{job_id}.tif"


@router.get("/{job_id}/info")
async def get_tile_info(job_id: str):
    path = _cog_path(job_id)
    if not path.exists():
        raise HTTPException(404, f"COG not found for job {job_id}")

    try:
        with Reader(str(path)) as src:
            info = src.info()
            return {
                "bounds": info.bounds,
                "crs": str(info.crs),
                "width": info.width,
                "height": info.height,
                "minzoom": info.minzoom,
                "maxzoom": info.maxzoom,
                "dtype": str(info.dtype),
            }
    except Exception as e:
        raise HTTPException(500, f"Error reading COG: {e}")


@router.get("/{job_id}/bounds")
async def get_bounds(job_id: str):
    path = _cog_path(job_id)
    if not path.exists():
        raise HTTPException(404, f"COG not found for job {job_id}")

    try:
        with Reader(str(path)) as src:
            bounds = src.geographic_bounds
            return {
                "bounds": list(bounds),
                "center": [(bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2]
            }
    except Exception as e:
        raise HTTPException(500, f"Error reading bounds: {e}")


@router.get("/{job_id}/{z}/{x}/{y}.png")
async def get_tile(
    job_id: str,
    z: int,
    x: int,
    y: int,
    colormap: str = Query(default="severity"),
    band: int = Query(default=1),
    rescale: Optional[str] = Query(default=None),
    tilesize: int = Query(default=512, ge=256, le=1024),
):
    path = _cog_path(job_id)
    if not path.exists():
        raise HTTPException(404, f"COG not found for job {job_id}")

    try:
        lut, default_rescale, is_discrete = _get_colormap(colormap)

        with Reader(str(path)) as src:
            resampling = "nearest" if is_discrete else "bilinear"
            tile = src.tile(
                x, y, z,
                indexes=band,
                tilesize=tilesize,
                buffer=0 if is_discrete else 2,
                resampling_method=resampling,
            )

            data = tile.data[0]
            mask = tile.mask

            if is_discrete:
                rgba = _apply_discrete(data, mask, lut)
            else:
                if rescale:
                    try:
                        vmin, vmax = map(float, rescale.split(","))
                    except ValueError:
                        vmin, vmax = default_rescale or (-0.2, 0.8)
                else:
                    vmin, vmax = default_rescale or (-0.2, 0.8)

                rgba = _apply_continuous(data, mask, vmin, vmax, lut)

            content = _render_png(rgba, tilesize)
            return Response(content=content, media_type="image/png", headers=_TILE_HEADERS)

    except TileOutsideBounds:
        return Response(content=_EMPTY_TILE, media_type="image/png")
    except Exception as e:
        raise HTTPException(500, f"Tile generation error: {e}")


@router.get("/{job_id}/preview.png")
async def get_preview(
    job_id: str,
    width: int = Query(default=800, le=2048),
    height: int = Query(default=800, le=2048),
    colormap: str = Query(default="severity"),
    rescale: Optional[str] = Query(default=None),
):
    path = _cog_path(job_id)
    if not path.exists():
        raise HTTPException(404, f"COG not found for job {job_id}")

    try:
        lut, default_rescale, is_discrete = _get_colormap(colormap)

        with Reader(str(path)) as src:
            resampling = "nearest" if is_discrete else "bilinear"
            preview = src.preview(
                max_size=max(width, height),
                resampling_method=resampling,
            )

            data = preview.data[0]
            mask = preview.mask

            if is_discrete:
                rgba = _apply_discrete(data, mask, lut)
            else:
                if rescale:
                    vmin, vmax = map(float, rescale.split(","))
                else:
                    vmin, vmax = default_rescale or (-0.2, 0.8)
                rgba = _apply_continuous(data, mask, vmin, vmax, lut)

            img = Image.fromarray(rgba, mode="RGBA")
            buf = BytesIO()
            img.save(buf, format="PNG")
            return Response(content=buf.getvalue(), media_type="image/png")

    except Exception as e:
        raise HTTPException(500, f"Preview generation error: {e}")