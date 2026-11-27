"""
The class areas and the zones of the flood and lisa results through the
analysis worker (api.main._run_analysis_sync), on the synthetic Sentinel-2
cube of tests/test_pipeline_distribution.py. The algorithms, the coverage,
the class areas, the COG and the vectoriser run for real; the catalogue,
the scene choice, the STAC load, the job table and the vectoriser's database
write are stubbed (no network, no database).

The zones are checked against an independent labelling of the same layers
(scipy.ndimage.label, 4-connected, per severity and, for lisa, per quadrant,
inside the requested box): every zone of at least the minimum size is mapped
with its class, every smaller patch is counted as left out, and per class the
zones and the small patches together hold exactly the class area.

Needs the full requirements.txt (rioxarray / GDAL), like tests/test_api.py.
"""

from collections import defaultdict

import numpy as np
import pytest
from scipy import ndimage

from eo_brain.algorithms import vectorize as vectorize_mod
from eo_brain.algorithms.base import OUTLIER_QUADRANTS
from eo_brain.api import cog as cog_mod
from eo_brain.api import main
from eo_brain.core.severity_areas import severity_classes
from tests.test_pipeline_distribution import RES, FakeCube, RecordingRepo, run

PIXEL_AREA = RES * RES
DISCRETE = ["flood", "lisa"]


@pytest.fixture
def worker(monkeypatch, tmp_path):
    """The worker with the real vectoriser; records the job, the inserted zones and the layers used."""
    rec = RecordingRepo()
    rec.inserted, rec.layers = [], {}
    selection = type("Selection", (), {
        "items": ["scene-a", "scene-b"],
        "to_dict": staticmethod(lambda: {"method": "pixel_mask", "found": 2, "too_cloudy": 0, "screened": 2,
                                         "with_clear": 2, "scene_ids": ["scene-a", "scene-b"]}),
    })()
    monkeypatch.setattr(main, "job_repo", rec)
    monkeypatch.setattr(main, "DataCube", FakeCube)
    monkeypatch.setattr(main, "split_by_radiometry", lambda provider, items: (items, []))
    monkeypatch.setattr(main, "select_scenes", lambda *a, **k: selection)
    monkeypatch.setattr(vectorize_mod.vector_repo, "bulk_insert", lambda features: rec.inserted.extend(features))
    monkeypatch.setattr(cog_mod, "OUTPUT_DIR", tmp_path)

    real = main._severity_areas

    def spy(job_id, algorithm, data, layer, inside, crs, coverage):
        rec.layers = {"data": data, "layer": layer, "inside": inside}
        return real(job_id, algorithm, data, layer, inside, crs, coverage)

    monkeypatch.setattr(main, "_severity_areas", spy)
    return rec


def _layers(rec, algorithm):
    data, inside = rec.layers["data"], rec.layers["inside"]
    severity = data[main.SEVERITY_LAYER[algorithm]].values.astype(int)
    quadrant = data[main.QUADRANT_LAYER[algorithm]].values.astype(int) if algorithm in main.QUADRANT_LAYER \
        else np.zeros_like(severity)
    covered = np.isfinite(rec.layers["layer"].values) & inside
    return severity, quadrant, covered, inside


def _components(severity, quadrant, inside):
    """{(severity, quadrant): [pixel count of each 4-connected patch]} inside the box, severity >= 1."""
    sizes = defaultdict(list)
    for s in (1, 2, 3):
        for q in np.unique(quadrant[(severity == s) & inside]):
            labels, n = ndimage.label((severity == s) & (quadrant == q) & inside)
            sizes[(s, int(q))] = sorted(np.bincount(labels.ravel())[1:].tolist()) if n else []
    return sizes


@pytest.mark.parametrize("algorithm", DISCRETE)
def test_class_areas_count_every_covered_pixel_once(worker, algorithm):
    run(algorithm)
    assert worker.failed == []
    [done] = worker.completed
    cov, block = done.result["coverage"], done.result["severity_areas"]
    assert "error" not in block, block
    assert [(c.severity, c.quadrant) for c in done.severity_classes] == severity_classes(algorithm)
    assert sum(c.pixel_count for c in done.severity_classes) == cov["covered_px"] == block["pixels"]
    assert all(c.area_m2 == c.pixel_count * PIXEL_AREA for c in done.severity_classes)
    assert block["pixel_area_m2"] == PIXEL_AREA and block["area_m2"] == cov["covered_px"] * PIXEL_AREA

    # the same counts from the layers the worker used, without core.severity_areas
    severity, quadrant, covered, _ = _layers(worker, algorithm)
    for c in done.severity_classes:
        assert c.pixel_count == int(((severity == c.severity) & (quadrant == c.quadrant) & covered).sum()), c
    assert sum(block["pixels_by_severity"].values()) == cov["covered_px"]
    if algorithm == "lisa":
        assert block["vegetated_pixels"] == int(((quadrant != 0) & covered).sum())
        # the synthetic cube does produce anomalies of both kinds
        assert {c.quadrant for c in done.severity_classes if c.severity > 0 and c.pixel_count} == set(OUTLIER_QUADRANTS)
    else:
        assert "vegetated_pixels" not in block
    assert sum(c.pixel_count for c in done.severity_classes if c.severity > 0) > 0


@pytest.mark.parametrize("algorithm", DISCRETE)
def test_zones_are_the_patches_of_one_class_inside_the_box(worker, algorithm):
    run(algorithm)
    [done] = worker.completed
    stats = done.result["vector_stats"]
    assert "error" not in stats, stats
    severity, quadrant, _, inside = _layers(worker, algorithm)
    # not a vacuous clip: the grid margin outside the box holds severities too
    assert ((severity >= 1) & ~inside).any()

    expected = _components(severity, quadrant, inside)
    minimum = main.MIN_ZONE_PIXELS
    mapped = defaultdict(list)
    for f in worker.inserted:
        mapped[(f["severity"], f["quadrant"] if algorithm == "lisa" else 0)].append(round(f["area_m2"] / PIXEL_AREA))
        assert f["area_m2"] == pytest.approx(round(f["area_m2"] / PIXEL_AREA) * PIXEL_AREA)
    for key, sizes in expected.items():
        assert sorted(mapped.get(key, [])) == [n for n in sizes if n >= minimum], key
    assert set(mapped) <= set(expected)

    small = [n for sizes in expected.values() for n in sizes if n < minimum]
    assert stats["skipped_small"] == len(small) > 0
    assert stats["skipped_small_area_m2"] == pytest.approx(sum(small) * PIXEL_AREA)
    assert stats["min_zone_pixels"] == minimum and stats["min_zone_area_m2"] == minimum * PIXEL_AREA
    assert stats["feature_count"] == len(worker.inserted) > 0 and stats["max_features_reached"] is False
    assert stats["by_severity"] == {s: sum(1 for f in worker.inserted if f["severity"] == s) for s in (1, 2, 3)}

    # per class: the zones and the patches left out hold exactly the class area
    by_class = {(c.severity, c.quadrant): c.area_m2 for c in done.severity_classes}
    for key, sizes in expected.items():
        assert sum(sizes) * PIXEL_AREA == by_class[key], key
    zones = sum(f["area_m2"] for f in worker.inserted)
    anomalous = sum(a for (s, _), a in by_class.items() if s > 0)
    assert zones + stats["skipped_small_area_m2"] == pytest.approx(anomalous)
    assert zones < anomalous


def test_every_lisa_zone_has_one_outlier_quadrant(worker):
    """Fails if api.main stops passing the quadrant layer: the vectoriser refuses LISA without it."""
    run("lisa")
    [done] = worker.completed
    assert "error" not in done.result["vector_stats"], done.result["vector_stats"]
    assert worker.inserted
    assert all(f["quadrant"] in OUTLIER_QUADRANTS for f in worker.inserted)
    assert {f["quadrant"] for f in worker.inserted} == set(OUTLIER_QUADRANTS)


def test_flood_zones_have_no_quadrant(worker):
    run("flood")
    assert worker.inserted and all(f["quadrant"] is None for f in worker.inserted)


def test_class_areas_that_cannot_be_computed_do_not_fail_the_job(worker, monkeypatch):
    monkeypatch.setattr(main, "crs_in_metres", lambda crs: False)
    run("lisa")
    assert worker.failed == []
    [done] = worker.completed
    assert done.severity_classes is None
    assert "not projected in metres" in done.result["severity_areas"]["error"]
    assert done.message.startswith("Completed")


def test_a_vectoriser_failure_does_not_fail_the_job_or_the_class_areas(worker, monkeypatch):
    monkeypatch.setattr(main, "QUADRANT_LAYER", {})  # main no longer hands the quadrants over
    run("lisa")
    assert worker.failed == []
    [done] = worker.completed
    assert "quadrant" in done.result["vector_stats"]["error"]
    assert worker.inserted == []
    # without the quadrant layer the class areas are refused too, rather than recorded wrong
    assert done.severity_classes is None and "error" in done.result["severity_areas"]
