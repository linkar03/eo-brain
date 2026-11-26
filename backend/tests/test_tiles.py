"""
tests/test_tiles.py - Tile server endpoints against real COGs

The COGs are written by the pipeline's own writer (eo_brain.api.cog) into a
temp dir and served by the tile router through FastAPI's TestClient, so the
rio-tiler, rasterio and Pillow versions that are installed are the ones under
test. No network, no database.

Validates:
- /info and /bounds against the installed rio-tiler API
- default rendering (no query parameters) is the producing algorithm's
  colormap and is never blank; it equals the tile URL a job returns
- nodata stays transparent; discrete classes keep their exact colours
"""

from io import BytesIO

import numpy as np
import pytest
import xarray as xr
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
import morecantile
from pyproj import Transformer

from eo_brain.api import cog, tiles

# Synthetic grid: UTM 32N (near Florence), 10 m pixels, 400 x 300.
CRS = "EPSG:32632"
WEST, NORTH, RES = 680000.0, 4850000.0, 10.0
NX, NY = 400, 300


def _grid(values: np.ndarray, name: str) -> xr.Dataset:
    x = WEST + RES * np.arange(NX) + RES / 2
    y = NORTH - RES * np.arange(NY) - RES / 2
    return xr.Dataset({name: xr.DataArray(values, dims=("y", "x"), coords={"y": y, "x": x})})


def _ndvi_values() -> np.ndarray:
    data = np.linspace(-0.2, 0.9, NX, dtype=np.float32)[None, :].repeat(NY, axis=0)
    data[:20, :20] = np.nan  # nodata corner (north-west)
    return data


def _severity_values() -> np.ndarray:
    data = np.zeros((NY, NX), dtype=np.int8)
    data[100:150, 100:150] = 1
    data[100:150, 200:250] = 2
    data[100:150, 300:350] = 3
    return data


@pytest.fixture
def cogs(tmp_path, monkeypatch):
    """Write one COG per kind into tmp_path and point both modules at it."""
    monkeypatch.setattr(cog, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(tiles, "OUTPUT_DIR", tmp_path)

    cog.save_ndvi_cog(_grid(_ndvi_values(), "ndvi"), "ndvi-job", CRS, bbox=None)
    cog.save_severity_cog(
        _grid(_severity_values(), "lisa_severity"), "lisa-job", CRS, bbox=None,
        algorithm="lisa", severity_var="lisa_severity",
    )
    # A float COG without the algorithm tag (as written before the tag existed)
    cog._save_cog(_grid(_ndvi_values(), "ndvi"), "untagged-job", CRS,
                  var_name="ndvi", is_continuous=True)
    return tmp_path


@pytest.fixture
def client(cogs):
    app = FastAPI()
    app.include_router(tiles.router)
    return TestClient(app)


def _png(response) -> np.ndarray:
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/png"
    return np.asarray(Image.open(BytesIO(response.content)).convert("RGBA"))


def _visible_share(rgba: np.ndarray) -> float:
    return float((rgba[..., 3] > 0).mean())


def _distinct_colours(rgba: np.ndarray) -> int:
    visible = rgba[rgba[..., 3] > 0]
    return len(np.unique(visible.reshape(-1, 4), axis=0))


_TO_WGS84 = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)


def _tile_at(client, job_id: str, row: int, col: int, zoom_above_max: int) -> morecantile.Tile:
    """Web-Mercator tile containing raster pixel (row, col), at maxzoom + k."""
    maxzoom = client.get(f"/tiles/{job_id}/info").json()["maxzoom"]
    lon, lat = _TO_WGS84.transform(WEST + RES * (col + 0.5), NORTH - RES * (row + 0.5))
    return morecantile.tms.get("WebMercatorQuad").tile(lon, lat, maxzoom + zoom_above_max)


def _centre_tile(client, job_id: str) -> morecantile.Tile:
    """A tile well inside the raster (about 0.9 km wide at this latitude)."""
    return _tile_at(client, job_id, NY // 2, NX // 2, 2)


class TestInfoAndBounds:
    def test_info(self, client):
        r = client.get("/tiles/ndvi-job/info")
        assert r.status_code == 200, r.text
        info = r.json()
        assert info["crs"] == CRS
        assert info["width"] == NX and info["height"] == NY
        assert info["bounds"] == pytest.approx(
            [WEST, NORTH - NY * RES, WEST + NX * RES, NORTH], abs=1e-6
        )
        assert isinstance(info["minzoom"], int) and isinstance(info["maxzoom"], int)
        assert 0 <= info["minzoom"] <= info["maxzoom"] <= 24
        assert info["dtype"] == "float32"
        assert info["algorithm"] == "ndvi"

    def test_info_discrete(self, client):
        info = client.get("/tiles/lisa-job/info").json()
        assert info["dtype"] == "uint8"
        assert info["algorithm"] == "lisa"

    def test_bounds_are_wgs84(self, client):
        r = client.get("/tiles/ndvi-job/bounds")
        assert r.status_code == 200, r.text
        west, south, east, north = r.json()["bounds"]
        # 4 km x 3 km box east of Florence
        assert 11.2 < west < east < 11.35
        assert 43.7 < south < north < 43.8
        lon, lat = r.json()["center"]
        assert west < lon < east and south < lat < north

    def test_missing_cog_is_404(self, client):
        assert client.get("/tiles/no-such-job/info").status_code == 404
        assert client.get("/tiles/no-such-job/bounds").status_code == 404
        assert client.get("/tiles/no-such-job/preview.png").status_code == 404


class TestDefaultRendering:
    """A request without query parameters must produce a visible image."""

    def test_ndvi_preview_default_is_visible(self, client):
        rgba = _png(client.get("/tiles/ndvi-job/preview.png"))
        assert rgba.shape[:2] == (NY, NX)
        assert _visible_share(rgba) > 0.6
        assert _distinct_colours(rgba) > 20  # a gradient, not one colour
        # the NaN corner stays transparent
        assert (rgba[:20, :20, 3] == 0).all()

    def test_ndvi_tile_default_is_visible(self, client):
        t = _centre_tile(client, "ndvi-job")
        rgba = _png(client.get(f"/tiles/ndvi-job/{t.z}/{t.x}/{t.y}.png"))
        assert rgba.shape == (256, 256, 4)
        assert _visible_share(rgba) > 0.9
        assert _distinct_colours(rgba) > 20

    def test_default_equals_job_tile_url(self, client):
        """The tile URL a job returns spells out the defaults: same image."""
        t = _centre_tile(client, "ndvi-job")
        path = f"/tiles/ndvi-job/{t.z}/{t.x}/{t.y}.png"
        default = client.get(path)
        explicit = client.get(f"{path}?{tiles.tile_query_for_algorithm('ndvi')}")
        assert default.content == explicit.content

    def test_severity_preview_default_keeps_class_colours(self, client):
        rgba = _png(client.get("/tiles/lisa-job/preview.png"))
        assert rgba.shape[:2] == (NY, NX)
        assert tuple(rgba[125, 125]) == (255, 235, 59, 200)   # WATCH
        assert tuple(rgba[125, 225]) == (255, 152, 0, 220)    # SUSPECT
        assert tuple(rgba[125, 325]) == (244, 67, 54, 240)    # ALERT
        assert rgba[10, 10, 3] == 0                           # NONE is transparent
        assert 0 < _visible_share(rgba) < 0.2

    def test_severity_tile_default_is_visible(self, client):
        t = _tile_at(client, "lisa-job", 125, 325, 1)  # inside the ALERT block
        rgba = _png(client.get(f"/tiles/lisa-job/{t.z}/{t.x}/{t.y}.png"))
        visible = rgba[rgba[..., 3] > 0]
        assert len(visible) > 0
        # nearest resampling: only the class colours, no blended edges
        allowed = {(255, 235, 59, 200), (255, 152, 0, 220), (244, 67, 54, 240)}
        assert {tuple(c) for c in np.unique(visible, axis=0)} <= allowed

    def test_untagged_float_cog_falls_back_to_continuous(self, client):
        assert client.get("/tiles/untagged-job/info").json()["algorithm"] is None
        rgba = _png(client.get("/tiles/untagged-job/preview.png"))
        assert _visible_share(rgba) > 0.6
        assert _distinct_colours(rgba) > 20

    def test_explicit_severity_on_float_is_still_honoured(self, client):
        """An explicit colormap wins over the default (here: a mismatched one)."""
        rgba = _png(client.get("/tiles/ndvi-job/preview.png?colormap=severity"))
        assert _visible_share(rgba) < 0.5


class TestRequestHandling:
    def test_malformed_rescale_is_422(self, client):
        assert client.get("/tiles/ndvi-job/preview.png?rescale=abc").status_code == 422
        t = _centre_tile(client, "ndvi-job")
        r = client.get(f"/tiles/ndvi-job/{t.z}/{t.x}/{t.y}.png?rescale=1")
        assert r.status_code == 422

    def test_tile_outside_bounds_is_empty(self, client):
        rgba = _png(client.get("/tiles/ndvi-job/13/0/0.png"))
        assert rgba.shape == (256, 256, 4)
        assert (rgba[..., 3] == 0).all()

    @pytest.mark.parametrize("algorithm, query", [
        ("ndvi", "colormap=ndvi&rescale=-0.1,0.8&tilesize=512"),
        ("ndwi", "colormap=water&rescale=-0.3,0.8&tilesize=512"),
        ("mndwi", "colormap=water&rescale=-0.3,0.8&tilesize=512"),
        ("flood", "colormap=flood&tilesize=512"),
        ("lisa", "colormap=severity&tilesize=512"),
    ])
    def test_job_tile_query_unchanged(self, algorithm, query):
        """The tile URLs jobs return are the same as before the refactor."""
        assert tiles.tile_query_for_algorithm(algorithm) == query
