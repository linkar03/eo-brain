"""
Tile server endpoints using rio-tiler (>= 7.0, see requirements.txt).

Colormaps are pre-built as NumPy LUTs (256x4 uint8) for O(1) rendering.
Discrete maps use direct index lookup, continuous maps normalize then LUT.

A request that names no colormap is rendered with the defaults of the
algorithm that produced the COG (``ALGORITHM_COLORMAP``; the algorithm is
stored in the COG as the ``EO_BRAIN_ALGORITHM`` tag by ``api.cog``). The
tile URL a job returns (``api.main._build_tile_url``) spells out the same
defaults, so both render identically.

The endpoints are plain ``def`` on purpose: FastAPI runs them in its thread
pool, so blocking raster reads do not stall the event loop.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict
from io import BytesIO

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Response
from PIL import Image

from rasterio.crs import CRS
from rio_tiler.io import Reader
from rio_tiler.errors import TileOutsideBounds
from rio_tiler.models import ImageData

from eo_brain.api.cog import ALGORITHM_TAG  # the algorithm that produced the COG

router = APIRouter(prefix="/tiles", tags=["tiles"])

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output"))

WGS84 = CRS.from_epsg(4326)

# Continuous outputs are rendered with bilinear resampling and a 2 px buffer
# (cropped again before encoding); discrete ones with nearest and no buffer.
_CONTINUOUS_BUFFER = 2


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

# Default colormap per algorithm (its rescale range is the colormap's default).
ALGORITHM_COLORMAP: Dict[str, str] = {
    "ndvi": "ndvi",
    "ndwi": "water",
    "mndwi": "water",
    "flood": "flood",
    "lisa": "severity",
}


def default_rendering(
    algorithm: Optional[str], dtype: Optional[str] = None
) -> Tuple[str, Optional[Tuple[float, float]]]:
    """
    Colormap name and rescale range for a request that names no colormap.

    Known algorithm -> its colormap. Otherwise (a COG without the algorithm
    tag) the data type decides: floating point -> "ndvi", integer -> "severity".
    """
    name = ALGORITHM_COLORMAP.get(algorithm or "")
    if name is None:
        is_float = dtype is not None and np.issubdtype(np.dtype(dtype), np.floating)
        name = "ndvi" if is_float else "severity"
    return name, _COLORMAPS[name][1]


def tile_query_for_algorithm(algorithm: str) -> str:
    """Query string of the tile URL returned for a job (same defaults as above)."""
    name, rescale = default_rendering(algorithm)
    query = f"colormap={name}"
    if rescale:
        query += f"&rescale={rescale[0]:g},{rescale[1]:g}"
    return query + "&tilesize=512"


def _get_colormap(name: str):
    """Lookup colormap config, fallback to NDVI."""
    return _COLORMAPS.get(name.lower(), (_NDVI_LUT, (-0.2, 0.8), False))


def _parse_rescale(rescale: Optional[str]) -> Optional[Tuple[float, float]]:
    """'vmin,vmax' -> (vmin, vmax); None if not given. Malformed -> HTTP 422."""
    if rescale is None:
        return None
    try:
        vmin, vmax = (float(v) for v in rescale.split(","))
    except ValueError:
        raise HTTPException(422, f"rescale must be 'min,max', got {rescale!r}")
    return vmin, vmax


def _resolve_rendering(src: Reader, colormap: Optional[str], rescale: Optional[str]):
    """(lut, (vmin, vmax), is_discrete) for this request and COG."""
    user_range = _parse_rescale(rescale)
    if colormap is None:
        algorithm = src.dataset.tags().get(ALGORITHM_TAG)
        colormap, _ = default_rendering(algorithm, src.dataset.dtypes[0])
    lut, default_rescale, is_discrete = _get_colormap(colormap)
    vrange = user_range or default_rescale or (-0.2, 0.8)
    return lut, vrange, is_discrete


def _first_band(img: ImageData) -> Tuple[np.ndarray, np.ndarray]:
    """
    First band and its validity (True = data).

    Read from the masked array (``ImageData.array``). Since rio-tiler 9,
    ``ImageData.mask`` holds the data type's min/max instead of 0/255, so for a
    float32 raster ``mask == 0`` never finds the nodata pixels.
    """
    band = img.array[0]
    return np.ma.getdata(band), ~np.ma.getmaskarray(band)


def _apply_discrete(data: np.ndarray, valid: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Single LUT lookup for discrete severity values."""
    indices = np.nan_to_num(data, nan=0).astype(np.uint8)
    rgba = lut[indices]
    rgba[~valid, 3] = 0
    return rgba


def _apply_continuous(
    data: np.ndarray,
    valid: np.ndarray,
    vmin: float,
    vmax: float,
    lut: np.ndarray
) -> np.ndarray:
    """Normalize to 1-255 range, then single LUT lookup. 0 = nodata."""
    range_val = max(vmax - vmin, 1e-10)
    with np.errstate(invalid="ignore"):
        normalized = np.clip((data - vmin) / range_val, 0, 1)
    nodata = ~valid | np.isnan(data)
    scaled = (np.nan_to_num(normalized, nan=0) * 254 + 1).astype(np.uint8)
    scaled[nodata] = 0
    return lut[scaled]


def _colorize(img: ImageData, lut: np.ndarray, vrange, is_discrete: bool) -> np.ndarray:
    data, valid = _first_band(img)
    if is_discrete:
        return _apply_discrete(data, valid, lut)
    return _apply_continuous(data, valid, vrange[0], vrange[1], lut)


def _render_png(rgba: np.ndarray, size: Optional[int] = 256, discrete: bool = False) -> bytes:
    """RGBA array -> PNG bytes, resized to size x size (None = keep the size)."""
    img = Image.fromarray(rgba)
    if size is not None and img.size != (size, size):
        resample = Image.Resampling.NEAREST if discrete else Image.Resampling.LANCZOS
        img = img.resize((size, size), resample)
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


_EMPTY_TILE = _render_png(np.zeros((256, 256, 4), dtype=np.uint8))

_TILE_HEADERS = {
    "Cache-Control": "public, max-age=3600",
    "Access-Control-Allow-Origin": "*",
}


def _cog_path(job_id: str) -> Path:
    return OUTPUT_DIR / f"{job_id}.tif"


@router.get("/{job_id}/info")
def get_tile_info(job_id: str):
    """Raster metadata. ``bounds`` are in the raster's own CRS (``crs``)."""
    path = _cog_path(job_id)
    if not path.exists():
        raise HTTPException(404, f"COG not found for job {job_id}")

    try:
        with Reader(str(path)) as src:
            return {
                "bounds": list(src.bounds),
                "crs": src.crs.to_string(),
                "width": src.dataset.width,
                "height": src.dataset.height,
                "minzoom": src.minzoom,
                "maxzoom": src.maxzoom,
                "dtype": src.dataset.dtypes[0],
                "algorithm": src.dataset.tags().get(ALGORITHM_TAG),
            }
    except Exception as e:
        raise HTTPException(500, f"Error reading COG: {e}")


@router.get("/{job_id}/bounds")
def get_bounds(job_id: str):
    """Bounds in WGS84 (EPSG:4326) longitude/latitude, and their centre."""
    path = _cog_path(job_id)
    if not path.exists():
        raise HTTPException(404, f"COG not found for job {job_id}")

    try:
        with Reader(str(path)) as src:
            bounds = src.get_geographic_bounds(WGS84)
            return {
                "bounds": list(bounds),
                "center": [(bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2]
            }
    except Exception as e:
        raise HTTPException(500, f"Error reading bounds: {e}")


@router.get("/{job_id}/{z}/{x}/{y}.png")
def get_tile(
    job_id: str,
    z: int,
    x: int,
    y: int,
    colormap: Optional[str] = Query(default=None, description="Default: the producing algorithm's colormap"),
    band: int = Query(default=1),
    rescale: Optional[str] = Query(default=None, description="'min,max'; default: the colormap's range"),
    tilesize: int = Query(default=512, ge=256, le=1024),
):
    path = _cog_path(job_id)
    if not path.exists():
        raise HTTPException(404, f"COG not found for job {job_id}")

    try:
        with Reader(str(path)) as src:
            lut, vrange, is_discrete = _resolve_rendering(src, colormap, rescale)
            buffer = 0 if is_discrete else _CONTINUOUS_BUFFER
            tile = src.tile(
                x, y, z,
                indexes=band,
                tilesize=tilesize,
                buffer=buffer,
                resampling_method="nearest" if is_discrete else "bilinear",
            )
            rgba = _colorize(tile, lut, vrange, is_discrete)
            if buffer:
                rgba = rgba[buffer:-buffer, buffer:-buffer]
            content = _render_png(rgba, 256, discrete=is_discrete)
            return Response(content=content, media_type="image/png", headers=_TILE_HEADERS)

    except TileOutsideBounds:
        return Response(content=_EMPTY_TILE, media_type="image/png", headers=_TILE_HEADERS)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Tile generation error: {e}")


@router.get("/{job_id}/preview.png")
def get_preview(
    job_id: str,
    width: int = Query(default=800, ge=16, le=2048),
    height: int = Query(default=800, ge=16, le=2048),
    colormap: Optional[str] = Query(default=None, description="Default: the producing algorithm's colormap"),
    rescale: Optional[str] = Query(default=None, description="'min,max'; default: the colormap's range"),
):
    path = _cog_path(job_id)
    if not path.exists():
        raise HTTPException(404, f"COG not found for job {job_id}")

    try:
        with Reader(str(path)) as src:
            lut, vrange, is_discrete = _resolve_rendering(src, colormap, rescale)
            preview = src.preview(
                max_size=max(width, height),
                resampling_method="nearest" if is_discrete else "bilinear",
            )
            rgba = _colorize(preview, lut, vrange, is_discrete)
            return Response(content=_render_png(rgba, None), media_type="image/png")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Preview generation error: {e}")
