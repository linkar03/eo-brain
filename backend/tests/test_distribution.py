"""
Offline tests for eo_brain.core.distribution: the 0.01-wide value intervals a
continuous-index job records (table index_value_bins, view
agent.index_value_distribution). Pure numpy; the CRS check needs pyproj and is
skipped without it.
"""

import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from eo_brain.core import distribution as dist_mod
from eo_brain.core.coverage import summarize_coverage
from eo_brain.core.distribution import (
    BIN_WIDTH, N_BINS, ValueBin, bin_counts, bin_edge, bin_indices, grid_pixel_area_m2,
    index_distribution, index_value_bins, value_stats,
)


def bins_of(values, inside=None, area=1.0):
    values = np.asarray(values)
    inside = np.ones(values.shape, bool) if inside is None else np.asarray(inside, bool)
    return index_value_bins(values, inside, area)


def as_tuples(bins):
    return [(b.bin_index, b.value_from, b.value_to, b.pixel_count) for b in bins]


# ---------------------------------------------------------------- intervals

def test_known_values_land_in_known_intervals():
    values = np.array([[-1.0, -0.995, 0.0, 0.004],
                       [0.649, 0.65, 0.999, 1.0]])
    assert as_tuples(bins_of(values)) == [
        (0, -1.0, -0.99, 2),
        (100, 0.0, 0.01, 2),
        (164, 0.64, 0.65, 1),
        (165, 0.65, 0.66, 1),
        (199, 0.99, 1.0, 2),
    ]


def test_the_edges_are_two_decimal_numbers():
    assert N_BINS == 200 and BIN_WIDTH == 0.01
    assert [bin_edge(k) for k in (0, 1, 99, 100, 165, 199, 200)] == [-1.0, -0.99, -0.01, 0.0, 0.65, 0.99, 1.0]
    for k in range(N_BINS):
        assert round(bin_edge(k + 1) - bin_edge(k), 10) == 0.01
        assert bin_edge(k) == round(bin_edge(k), 2)


def test_the_lower_edge_is_inclusive_also_for_float32_values():
    # float32(0.65) is 0.64999998: it still belongs to the interval that starts at 0.65,
    # so "value_from >= 0.65" counts the pixels a user would call 0.65.
    v = np.array([0.65, 0.0, -0.3, 0.2, 0.6499], dtype=np.float32)
    assert v[0] < np.float64(0.65)
    assert bin_indices(v).tolist() == [165, 100, 70, 120, 164]


def test_one_is_in_the_last_interval():
    [b] = bins_of([1.0])
    assert (b.bin_index, b.value_from, b.value_to) == (199, 0.99, 1.0)


def test_values_outside_minus_one_to_one_are_clipped_into_the_end_intervals():
    assert as_tuples(bins_of([-1.2, -1.0000001, 1.0000001, 1.3])) == [(0, -1.0, -0.99, 2), (199, 0.99, 1.0, 2)]


def test_nan_infinite_and_out_of_area_pixels_are_not_counted():
    values = np.array([[0.5, np.nan, np.inf], [-np.inf, 0.3, 0.7]])
    inside = np.array([[True, True, True], [True, True, False]])
    assert as_tuples(bins_of(values, inside)) == [(130, 0.3, 0.31, 1), (150, 0.5, 0.51, 1)]


def test_empty_input_gives_no_interval_and_no_stats():
    assert bins_of(np.array([])) == []
    assert bins_of(np.full((2, 2), np.nan)) == []
    assert bins_of(np.ones((2, 2)) * 0.5, np.zeros((2, 2), bool)) == []
    assert value_stats(np.full((2, 2), np.nan), np.ones((2, 2), bool)) is None


def test_only_non_empty_intervals_are_kept_in_value_order():
    bins = bins_of([0.9, -0.5, 0.9, 0.1])
    assert [b.bin_index for b in bins] == sorted(b.bin_index for b in bins)
    assert all(b.pixel_count > 0 for b in bins)


def test_counting_in_chunks_gives_the_same_counts(monkeypatch):
    rng = np.random.default_rng(7)
    values = rng.uniform(-1, 1, size=(37, 41))
    whole = bin_counts(values, np.ones(values.shape, bool))
    monkeypatch.setattr(dist_mod, "_CHUNK", 100)
    assert (bin_counts(values, np.ones(values.shape, bool)) == whole).all()


def test_a_threshold_sum_equals_a_direct_count():
    rng = np.random.default_rng(3)
    # Three decimals plus 0.0005: no value is near an interval edge.
    values = np.round(rng.uniform(-1, 0.999, size=(60, 60)), 3) + 0.0005
    bins = bins_of(values)
    for t in (-0.2, 0.0, 0.3, 0.65):
        above = sum(b.pixel_count for b in bins if b.value_from >= t)
        assert above == int((values >= t).sum()), t
    between = sum(b.pixel_count for b in bins if b.value_from >= 0.2 and b.value_to <= 0.5)
    assert between == int(((values >= 0.2) & (values < 0.5)).sum())


def test_shape_mismatch_and_bad_pixel_area_are_refused():
    with pytest.raises(ValueError, match="shape"):
        index_value_bins(np.zeros((2, 2)), np.ones((2, 3), bool), 1.0)
    for area in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="pixel area"):
            index_value_bins(np.zeros((2, 2)), np.ones((2, 2), bool), area)


# ---------------------------------------------------------------- the coverage invariant

def test_the_intervals_count_exactly_the_covered_pixels_of_the_coverage_block():
    rng = np.random.default_rng(11)
    values = rng.uniform(-1, 1, size=(50, 70)).astype(np.float32)
    values[rng.random(values.shape) < 0.3] = np.nan            # cloud-masked on every day
    inside = np.zeros(values.shape, bool)
    inside[5:45, 3:60] = True                                    # the requested box in the grid
    has_data = np.ones(values.shape, bool)
    coverage = summarize_coverage(np.isfinite(values), inside, has_data, [date(2026, 1, 9)], [1], {})
    dist = index_distribution("ndvi", values, inside, 400.0, coverage["covered_px"])
    assert dist.pixels == coverage["covered_px"] == sum(b.pixel_count for b in dist.bins)
    assert 0 < dist.pixels < int(inside.sum())


def test_a_distribution_of_other_pixels_is_refused():
    values = np.array([[0.1, 0.2], [np.nan, 0.4]])
    inside = np.ones((2, 2), bool)
    with pytest.raises(ValueError, match="the intervals count 3 pixels, the coverage counts 4"):
        index_distribution("ndvi", values, inside, 100.0, 4)


def test_stats_describe_the_same_pixels_as_the_intervals():
    values = np.array([[0.2, 0.4, 0.9], [np.nan, -0.6, 0.8]])
    inside = np.array([[True, True, False], [True, True, False]])
    dist = index_distribution("ndwi", values, inside, 100.0, 3)
    assert dist.stats == {"min": -0.6, "max": 0.4, "mean": pytest.approx(0.0)}
    s = dist.summary()
    assert s["index"] == "ndwi" and s["pixels"] == 3 and s["bin_width"] == 0.01
    assert (s["min"], s["max"]) == (-0.6, 0.4) and s["mean"] == pytest.approx(0.0)
    assert s["pixel_area_m2"] == 100.0 and s["area_m2"] == 300.0


# ---------------------------------------------------------------- area from the grid

def test_pixel_area_comes_from_the_grid_spacing():
    xs = 700_000.0 + 20.0 * np.arange(5) + 10.0      # pixel centres, 20 m apart
    ys = 4_800_000.0 - 20.0 * np.arange(3) - 10.0    # north-up: y decreases
    assert grid_pixel_area_m2(xs, ys) == 400.0
    assert grid_pixel_area_m2(60.0 * np.arange(4), 60.0 * np.arange(2)) == 3600.0


def test_the_area_of_an_interval_is_its_pixels_times_the_grid_pixel_area():
    area = grid_pixel_area_m2(60.0 * np.arange(3), -60.0 * np.arange(2))
    bins = index_value_bins(np.array([[0.7, 0.71, 0.705], [0.1, np.nan, 0.7]]), np.ones((2, 3), bool), area)
    assert [(b.value_from, b.pixel_count, b.area_m2) for b in bins] == [
        (0.1, 1, 3600.0), (0.7, 3, 10_800.0), (0.71, 1, 3600.0),
    ]


@pytest.mark.parametrize(
    "xs,ys,fragment",
    [
        ([700_000.0], [1.0, 2.0], "1 x coordinate"),
        ([0.0, 10.0], [], "0 y coordinate"),
        ([0.0, 10.0, 25.0], [0.0, 10.0], "x coordinates are not a regular grid"),
        ([0.0, 0.0], [0.0, 10.0], "x coordinates are not a regular grid"),
    ],
)
def test_an_irregular_or_degenerate_grid_is_refused(xs, ys, fragment):
    with pytest.raises(ValueError, match=fragment):
        grid_pixel_area_m2(xs, ys)


def test_utm_is_in_metres_and_geographic_is_not():
    pytest.importorskip("pyproj")
    assert dist_mod.crs_in_metres("EPSG:32632")
    assert not dist_mod.crs_in_metres("EPSG:4326")


def test_value_bin_is_a_plain_record():
    b = ValueBin(bin_index=165, value_from=0.65, value_to=0.66, pixel_count=2, area_m2=800.0)
    assert (b.value_from, b.value_to) == (bin_edge(165), bin_edge(166))


# ---------------------------------------------------------------- the synthetic seed

SEED = Path(__file__).resolve().parents[1] / "evals" / "fixtures" / "synthetic_seed.sql"
_BIN_ROW = re.compile(r"\('(?P<job>[0-9a-f-]{36})',\s*(?P<k>\d+),\s*(?P<lo>-?\d\.\d\d),\s*(?P<hi>-?\d\.\d\d),"
                      r"\s*(?P<n>\d+),\s*(?P<area>[\d.]+)\)")
_JOB_RESULT = re.compile(r"\('(?P<job>[0-9a-f-]{36})', 'completed', '(?P<algo>\w+)', [^\n]*\n[^\n]*\n[^\n]*\n"
                         r"   '(?P<params>\{[^']*\})',\n(?:[^\n]*\n){3}   '(?P<result>\{\"coverage\": [^']*)'\)")


def test_the_seeded_intervals_are_what_the_pipeline_would_record():
    """The DB tests and the SQL eval read these rows; check them here without a database."""
    text = SEED.read_text(encoding="utf-8")
    bins: dict = {}
    for m in _BIN_ROW.finditer(text.split("INSERT INTO index_value_bins", 1)[1]):
        bins.setdefault(m["job"], []).append(m)
    jobs = {m["job"]: m for m in _JOB_RESULT.finditer(text)}
    assert len(jobs) == 9 and len(bins) == 5
    for job, rows in bins.items():
        result = json.loads(jobs[job]["result"])
        res = json.loads(jobs[job]["params"])["resolution_m"]
        assert jobs[job]["algo"] in {"ndvi", "ndwi", "mndwi"}
        ks = [int(r["k"]) for r in rows]
        assert ks == sorted(set(ks)), job
        for r in rows:
            k = int(r["k"])
            assert (float(r["lo"]), float(r["hi"])) == (bin_edge(k), bin_edge(k + 1)), (job, k)
            assert float(r["area"]) == int(r["n"]) * res * res, (job, k)
        pixels = sum(int(r["n"]) for r in rows)
        block = result["value_distribution"]
        assert pixels == result["coverage"]["covered_px"] == block["pixels"], job
        assert block["pixel_area_m2"] == res * res and block["area_m2"] == pixels * res * res
        assert block["index"] == jobs[job]["algo"] and block["bin_width"] == BIN_WIDTH
        assert float(rows[0]["lo"]) <= block["min"] < float(rows[0]["hi"]), job
        assert float(rows[-1]["lo"]) <= block["max"] < float(rows[-1]["hi"]), job
        mid_mean = sum((float(r["lo"]) + BIN_WIDTH / 2) * int(r["n"]) for r in rows) / pixels
        assert abs(block["mean"] - mid_mean) <= BIN_WIDTH / 2, job
    # a continuous analysis run before distributions were recorded: coverage, no intervals
    old = [j for j, m in jobs.items() if m["algo"] in {"ndvi", "ndwi", "mndwi"} and j not in bins]
    assert len(old) == 1 and "value_distribution" not in json.loads(jobs[old[0]]["result"])
