"""
Unit tests of eo_brain/core/severity_areas.py: the exact area of each class of
a flood or lisa result over the covered pixels (what api.main stores in
severity_class_areas and agent.severity_areas reads). Pure numpy, plus the LISA
algorithm itself for the class invariants and a consistency check of the
synthetic seed's class rows. No database, no raster stack.
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from eo_brain.algorithms.base import OUTLIER_QUADRANTS, LISAAlgorithm, MoranQuadrant, Severity
from eo_brain.core import severity_areas as mod
from eo_brain.core.severity_areas import SeverityClass, severity_areas, severity_classes

SEED = Path(__file__).resolve().parents[1] / "evals" / "fixtures" / "synthetic_seed.sql"
HL, LH = int(MoranQuadrant.HL), int(MoranQuadrant.LH)


def _flood():
    """4 x 5 grid: a NaN pixel (not covered) and a column outside the box."""
    severity = np.array([[0, 1, 1, 3, 3],
                         [0, 2, 0, 3, 3],
                         [0, 0, 0, 0, 2],
                         [1, 0, 0, 0, 3]], dtype=np.int8)
    values = np.where(severity > 0, 0.3, -0.4).astype(np.float32)
    values[1, 2] = np.nan
    severity[1, 2] = 0                       # the algorithm gives NaN pixels severity 0
    inside = np.ones(severity.shape, bool)
    inside[:, 4] = False                     # the grid margin past the box
    return severity, values, inside


def _lisa():
    severity = np.array([[0, 0, 1, 1],
                         [0, 3, 3, 0],
                         [2, 0, 0, 0]], dtype=np.int8)
    quadrant = np.array([[0, 1, HL, LH],
                         [2, HL, LH, 4],
                         [LH, 3, 1, 0]], dtype=np.int8)
    ndvi = np.full(severity.shape, 0.6, np.float32)
    ndvi[2, 3] = np.nan
    inside = np.ones(severity.shape, bool)
    return severity, quadrant, ndvi, inside


def _counts(areas):
    return {(c.severity, c.quadrant): c.pixel_count for c in areas.classes}


# ---------------------------------------------------------------- classes

def test_the_classes_of_each_algorithm():
    assert severity_classes("flood") == [(0, 0), (1, 0), (2, 0), (3, 0)]
    assert severity_classes("lisa") == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
                                        (1, 3), (1, 4), (2, 3), (2, 4), (3, 3), (3, 4)]
    assert {q for s, q in severity_classes("lisa") if s > 0} == {int(q) for q in OUTLIER_QUADRANTS}
    with pytest.raises(ValueError, match="no severity classes for algorithm 'ndvi'"):
        severity_classes("ndvi")


def test_vegetated_is_a_lisa_quadrant():
    assert SeverityClass("lisa", 0, 0, 5, 500.0).vegetated is False
    assert SeverityClass("lisa", 0, 2, 5, 500.0).vegetated is True
    assert SeverityClass("lisa", 3, HL, 5, 500.0).vegetated is True
    assert SeverityClass("flood", 2, 0, 5, 500.0).vegetated is None


# ---------------------------------------------------------------- counting

def test_flood_counts_only_the_covered_pixels_and_keeps_empty_classes():
    severity, values, inside = _flood()
    severity[0, 1] = severity[0, 2] = severity[3, 0] = 0      # no watch pixel at all
    covered_px = int((np.isfinite(values) & inside).sum())    # 20 - 1 NaN - 4 outside = 15
    areas = severity_areas("flood", severity, None, values, inside, 100.0, covered_px)
    assert covered_px == 15
    assert _counts(areas) == {(0, 0): 12, (1, 0): 0, (2, 0): 1, (3, 0): 2}
    assert [(c.severity, c.quadrant) for c in areas.classes] == severity_classes("flood")
    assert all(c.algorithm == "flood" and c.area_m2 == c.pixel_count * 100.0 for c in areas.classes)
    assert areas.pixels == covered_px and areas.pixel_area_m2 == 100.0


def test_lisa_counts_every_quadrant_of_severity_0_and_the_outliers():
    severity, quadrant, ndvi, inside = _lisa()
    areas = severity_areas("lisa", severity, quadrant, ndvi, inside, 400.0, 11)
    assert _counts(areas) == {
        (0, 0): 1, (0, 1): 2, (0, 2): 1, (0, 3): 1, (0, 4): 1,
        (1, 3): 1, (1, 4): 1, (2, 3): 0, (2, 4): 1, (3, 3): 1, (3, 4): 1,
    }
    assert sum(c.area_m2 for c in areas.classes) == 11 * 400.0
    assert sum(c.pixel_count for c in areas.classes if c.vegetated) == 10


def test_summary_block():
    severity, quadrant, ndvi, inside = _lisa()
    block = severity_areas("lisa", severity, quadrant, ndvi, inside, 400.0, 11).summary()
    assert block == {
        "algorithm": "lisa", "pixels": 11, "pixel_area_m2": 400.0, "area_m2": 4400.0,
        "pixels_by_severity": {"0": 6, "1": 2, "2": 1, "3": 2}, "vegetated_pixels": 10,
    }
    severity, values, inside = _flood()
    flood = severity_areas("flood", severity, None, values, inside, 100.0, 15).summary()
    assert "vegetated_pixels" not in flood
    assert flood["pixels_by_severity"] == {"0": 9, "1": 3, "2": 1, "3": 2}
    json.dumps(block), json.dumps(flood)  # stored in jobs.result as JSON


def test_counting_in_chunks_gives_the_same_counts(monkeypatch):
    rng = np.random.default_rng(3)
    severity = rng.integers(0, 4, (37, 53)).astype(np.int8)
    values = rng.uniform(-1, 1, severity.shape).astype(np.float32)
    values[rng.random(severity.shape) < 0.1] = np.nan
    inside = rng.random(severity.shape) < 0.9
    covered = int((np.isfinite(values) & inside).sum())
    whole = severity_areas("flood", severity, None, values, inside, 25.0, covered)
    monkeypatch.setattr(mod, "_CHUNK", 7)
    chunked = severity_areas("flood", severity, None, values, inside, 25.0, covered)
    assert whole.classes == chunked.classes


def test_the_lisa_algorithm_only_produces_the_classes_recorded():
    """Severity only on HL / LH, quadrant 0 exactly where LISA did not classify: no class is refused."""
    rng = np.random.default_rng(11)
    ndvi = rng.uniform(0.0, 0.9, (60, 60)).astype(np.float32)
    ndvi[:5, :5] = np.nan
    out = LISAAlgorithm().run(xr.Dataset({"ndvi": (("y", "x"), ndvi)}), neighborhood_m=50, resolution_m=10).data
    severity, quadrant = out["lisa_severity"].values, out["lisa_quadrant"].values
    assert set(np.unique(quadrant[severity > 0])) == {int(q) for q in OUTLIER_QUADRANTS}
    classified = (ndvi >= 0.3) & np.isfinite(out["spatial_lag"].values)
    assert ((quadrant != 0) == classified).all()
    inside = np.ones(ndvi.shape, bool)
    areas = severity_areas("lisa", severity, quadrant, ndvi, inside, 100.0, int(np.isfinite(ndvi).sum()))
    assert sum(c.pixel_count for c in areas.classes if c.severity > 0) == int((severity > 0).sum()) > 0


# ---------------------------------------------------------------- refusals

def test_a_count_that_differs_from_the_coverage_is_refused():
    severity, values, inside = _flood()
    with pytest.raises(ValueError, match="the classes count 15 pixels, the coverage counts 16"):
        severity_areas("flood", severity, None, values, inside, 100.0, 16)


@pytest.mark.parametrize("bad, message", [
    (4, r"severity outside 0 to 3 on covered pixels: \[4\]"),
    (-1, r"severity outside 0 to 3 on covered pixels: \[-1\]"),
])
def test_an_unknown_severity_is_refused(bad, message):
    severity, values, inside = _flood()
    severity[0, 0] = bad
    with pytest.raises(ValueError, match=message):
        severity_areas("flood", severity, None, values, inside, 100.0, 15)


def test_values_outside_the_covered_pixels_are_not_checked():
    severity, values, inside = _flood()
    severity[:, 4] = 9                         # outside the box: never counted, never refused
    assert severity_areas("flood", severity, None, values, inside, 100.0, 15).pixels == 15


def test_a_lisa_severity_on_a_cluster_quadrant_is_refused():
    severity, quadrant, ndvi, inside = _lisa()
    quadrant[1, 1] = int(MoranQuadrant.HH)
    with pytest.raises(ValueError, match=r"impossible \(severity, quadrant\) classes \[\(3, 1\)\]"):
        severity_areas("lisa", severity, quadrant, ndvi, inside, 400.0, 11)


def test_an_unknown_quadrant_is_refused():
    severity, quadrant, ndvi, inside = _lisa()
    quadrant[0, 0] = 5
    with pytest.raises(ValueError, match="Moran quadrant outside 0 to 4"):
        severity_areas("lisa", severity, quadrant, ndvi, inside, 400.0, 11)


def test_the_quadrant_layer_is_required_for_lisa_only():
    severity, quadrant, ndvi, inside = _lisa()
    with pytest.raises(ValueError, match="lisa needs its Moran quadrant layer"):
        severity_areas("lisa", severity, None, ndvi, inside, 400.0, 11)
    severity, values, inside = _flood()
    with pytest.raises(ValueError, match="flood has none"):
        severity_areas("flood", severity, np.zeros_like(severity), values, inside, 100.0, 15)


@pytest.mark.parametrize("area", [0.0, -100.0, float("nan"), float("inf")])
def test_the_pixel_area_must_be_a_positive_number(area):
    severity, values, inside = _flood()
    with pytest.raises(ValueError, match="pixel area must be a positive number"):
        severity_areas("flood", severity, None, values, inside, area, 15)


def test_layers_of_different_shapes_are_refused():
    severity, values, inside = _flood()
    with pytest.raises(ValueError, match="layers differ in shape"):
        severity_areas("flood", severity, None, values[:, :4], inside, 100.0, 15)


def test_other_algorithms_are_refused():
    severity, values, inside = _flood()
    with pytest.raises(ValueError, match="no severity classes for algorithm 'ndwi'"):
        severity_areas("ndwi", severity, None, values, inside, 100.0, 15)


# ---------------------------------------------------------------- synthetic seed

def _seed():
    """{job: result} and {job: {(severity, quadrant): (algorithm, pixels, area_m2)}} from the seed file."""
    text = SEED.read_text(encoding="utf-8")
    results, job = {}, None
    for line in text.splitlines():
        start = re.match(r"\s*\('(0{8}-0000-4000-a000-0{9}\d{3})', '(?:completed|failed|pending)'", line)
        if start:
            job = start.group(1)
        found = re.search(r"'(\{\"coverage\".*\})'\)[,;]$", line)
        if found and job:
            results[job] = json.loads(found.group(1))
    block = text.split("INSERT INTO severity_class_areas", 1)[1]
    classes = {}
    for jid, alg, s, q, px, area in re.findall(
            r"\('([0-9a-f-]{36})', '(\w+)',\s*(\d), (\d),\s*(\d+),\s*([\d.]+)\)", block):
        classes.setdefault(jid, {})[(int(s), int(q))] = (alg, int(px), float(area))
    features = {}
    for jid, area in re.findall(r"\(\s*\d+, '([0-9a-f-]{36})', '\w+',\s*\d, [^,]+,\s*[\d.]+, [\d.]+,\s*([\d.]+),", text):
        features[jid] = features.get(jid, 0.0) + float(area)
    return results, classes, features


def test_the_seed_class_rows_are_what_the_pipeline_would_write():
    results, classes, features = _seed()
    assert set(classes) == {"00000000-0000-4000-a000-000000000002", "00000000-0000-4000-a000-000000000003"}
    for job, rows in classes.items():
        result = results[job]
        [algorithm] = {alg for alg, _, _ in rows.values()}
        assert list(rows) == severity_classes(algorithm), job        # every class, in order
        pixel_area = result["severity_areas"]["pixel_area_m2"]
        assert all(area == px * pixel_area for _, px, area in rows.values()), job
        pixels = sum(px for _, px, _ in rows.values())
        assert pixels == result["coverage"]["covered_px"] == result["severity_areas"]["pixels"], job
        by_severity = {str(int(s)): sum(px for (sev, _), (_, px, _) in rows.items() if sev == s) for s in Severity}
        assert result["severity_areas"]["pixels_by_severity"] == by_severity, job
        if algorithm == "lisa":
            vegetated = sum(px for (_, q), (_, px, _) in rows.items() if q != 0)
            assert result["severity_areas"]["vegetated_pixels"] == vegetated
        # the zones and the patches left out hold exactly the area of severity 1 to 3
        stats = result["vector_stats"]
        anomalous = sum(area for (s, _), (_, _, area) in rows.items() if s > 0)
        assert features[job] == stats["total_area_m2"], job
        assert features[job] + stats["skipped_small_area_m2"] == pytest.approx(anomalous), job
        assert stats["min_zone_area_m2"] == stats["min_zone_pixels"] * pixel_area
        # a patch left out is smaller than the minimum zone
        assert stats["skipped_small_area_m2"] < stats["skipped_small"] * stats["min_zone_area_m2"]
    # 001 is the flood analysis from before: no class rows, no zone size
    old = results["00000000-0000-4000-a000-000000000001"]
    assert "severity_areas" not in old and "min_zone_pixels" not in old["vector_stats"]
