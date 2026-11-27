"""
The analysis worker (api.main._run_analysis_sync) end to end on a synthetic
Sentinel-2 cube, for every algorithm. The catalogue search, the scene choice,
the STAC load, the vectoriser's database write and the job table are stubbed;
the algorithms, the cloud mask, the coverage, the value distribution, the COG
writing and the statistics run for real (no network, no database).

Needs the full requirements.txt (rioxarray / GDAL), like tests/test_api.py.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import dask.array as dsk
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from eo_brain.api import cog as cog_mod
from eo_brain.api import main
from eo_brain.core import radiometry as rad
from eo_brain.core.datacube import LoadResult, get_provider
from eo_brain.core.distribution import BIN_WIDTH, bin_counts

CRS = "EPSG:32632"
RES = 20
# a small box near Florence; the loaded grid runs three pixels past it on every side
BBOX = [11.20, 43.75, 11.215, 43.76]
ALGORITHMS = ["ndvi", "ndwi", "mndwi", "flood", "lisa"]


def _grid():
    from pyproj import Transformer

    to_utm = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    xs, ys = to_utm.transform([BBOX[0], BBOX[2], BBOX[0], BBOX[2]], [BBOX[1], BBOX[1], BBOX[3], BBOX[3]])
    x0, x1 = (np.floor(min(xs) / RES) - 3) * RES, (np.ceil(max(xs) / RES) + 3) * RES
    y0, y1 = (np.floor(min(ys) / RES) - 3) * RES, (np.ceil(max(ys) / RES) + 3) * RES
    x = np.arange(x0, x1, RES) + RES / 2
    y = np.arange(y1, y0, -RES) - RES / 2
    return x, y


def synthetic_cube() -> xr.Dataset:
    """Two acquisition days of DNs (offset removed: reflectance = DN * 1e-4) and an SCL band."""
    x, y = _grid()
    rng = np.random.default_rng(7)
    shape = (2, y.size, x.size)

    def dn(lo, hi):
        return (rng.uniform(lo, hi, shape) * 1e4).astype(np.uint16)

    scl = np.full(shape, 4, np.uint8)                     # vegetation: kept by the mask
    scl[0, :6, :] = 9                                     # a cloud on day 1 only: day 2 fills it
    scl[:, -15:, :20] = 9                                 # cloudy on both days: no value there
    bands = {"red": dn(0.02, 0.20), "nir": dn(0.10, 0.50), "green": dn(0.03, 0.15),
             "swir16": dn(0.05, 0.30), "scl": scl}
    time = pd.to_datetime(["2025-06-04T10:20:00", "2025-06-14T10:20:00"])
    ds = xr.Dataset(
        {name: (("time", "y", "x"), dsk.from_array(arr, chunks=(1, 16, 16))) for name, arr in bands.items()},
        coords={"time": time, "y": y, "x": x},
    )
    for name in ds.data_vars:
        ds[name].attrs["nodata"] = 0
    return rad.attach_radiometry(ds, [rad.S2_L2A_OFFSET_REMOVED] * 2)


class FakeCube:
    def __init__(self, provider):
        self.provider = get_provider(provider)

    def search(self, params):
        return [SimpleNamespace(id="scene-a"), SimpleNamespace(id="scene-b")]

    def load_items(self, items, bbox, bands, config):
        assert config.resolution_m == RES
        return LoadResult(data=synthetic_cube()[bands], provider=self.provider, crs=CRS, bbox=bbox,
                          slice_rules=[rad.S2_L2A_OFFSET_REMOVED] * 2)


class RecordingRepo:
    def __init__(self):
        self.completed, self.failed = [], []

    def update_progress(self, *args, **kwargs):
        pass

    def complete(self, job_id, result, cog_path, output_crs, message=None, value_bins=None):
        self.completed.append(SimpleNamespace(result=result, message=message, value_bins=value_bins))

    def fail(self, job_id, message):
        self.failed.append(message)


@pytest.fixture
def repo(monkeypatch, tmp_path):
    rec = RecordingRepo()
    selection = SimpleNamespace(
        items=["scene-a", "scene-b"],
        to_dict=lambda: {"method": "pixel_mask", "found": 2, "too_cloudy": 0, "screened": 2,
                         "with_clear": 2, "scene_ids": ["scene-a", "scene-b"]},
    )
    monkeypatch.setattr(main, "job_repo", rec)
    monkeypatch.setattr(main, "DataCube", FakeCube)
    monkeypatch.setattr(main, "split_by_radiometry", lambda provider, items: (items, []))
    monkeypatch.setattr(main, "select_scenes", lambda *a, **k: selection)
    monkeypatch.setattr(main, "vectorize_cog", lambda **k: SimpleNamespace(
        feature_count=0, total_area_m2=0.0, by_severity={}))
    monkeypatch.setattr(cog_mod, "OUTPUT_DIR", tmp_path)
    return rec


def run(algorithm):
    request = main.AnalysisRequest(
        bbox=BBOX, algorithm=algorithm, resolution_m=RES,
        start_date=datetime(2025, 6, 1, tzinfo=timezone.utc), end_date=datetime(2025, 6, 30, tzinfo=timezone.utc),
    )
    main._run_analysis_sync(uuid4(), request, "element84-s2")


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_every_algorithm_completes(repo, algorithm):
    run(algorithm)
    assert repo.failed == []
    [done] = repo.completed
    cov = done.result["coverage"]
    # the box, not the whole grid; and the corner cloudy on both days has no value
    assert 0 < cov["covered_px"] < cov["area_px"] < len(_grid()[0]) * len(_grid()[1])
    assert 90 < cov["coverage_pct"] < 99
    if algorithm in main.CONTINUOUS_OUTPUT_ALGORITHMS:
        block = done.result["value_distribution"]
        assert "error" not in block, block
        assert sum(b.pixel_count for b in done.value_bins) == cov["covered_px"] == block["pixels"]
        assert block["index"] == algorithm and block["pixel_area_m2"] == RES * RES
        assert block["area_m2"] == pytest.approx(sum(b.area_m2 for b in done.value_bins))
        assert -1 <= block["min"] <= block["mean"] <= block["max"] <= 1
        first, last = done.value_bins[0], done.value_bins[-1]
        assert first.value_from <= block["min"] < first.value_to + 1e-9
        assert last.value_from <= block["max"] <= last.value_to
        assert all(b.value_to - b.value_from == pytest.approx(BIN_WIDTH) for b in done.value_bins)
    else:
        assert done.value_bins is None and done.result["value_distribution"] is None


def test_the_intervals_are_those_of_the_result_layer_inside_the_box(repo):
    """Recompute from the cube with the same index: the worker bins the covered pixels, no others."""
    run("ndvi")
    [done] = repo.completed
    ds = synthetic_cube()
    masked = ds.where(ds["scl"].isin([4, 5, 6]))
    red, nir = (rad.band_to_reflectance(masked[b]) for b in ("red", "nir"))
    ndvi = ((nir - red) / (nir + red)).median(dim="time", skipna=True).values
    from eo_brain.core.coverage import inside_bbox_mask

    inside = inside_bbox_mask(ds.x.values, ds.y.values, CRS, BBOX)
    counts = bin_counts(ndvi.astype(np.float32), inside)
    assert {b.bin_index: b.pixel_count for b in done.value_bins} == {
        int(k): int(n) for k, n in enumerate(counts) if n}
    assert int(counts.sum()) == int((np.isfinite(ndvi) & inside).sum()) == done.result["coverage"]["covered_px"]


def test_a_distribution_that_cannot_be_computed_does_not_fail_the_job(repo, monkeypatch):
    monkeypatch.setattr(main, "crs_in_metres", lambda crs: False)
    run("ndwi")
    assert repo.failed == []
    [done] = repo.completed
    assert done.value_bins is None
    assert "not projected in metres" in done.result["value_distribution"]["error"]
    assert done.message.startswith("Completed")
