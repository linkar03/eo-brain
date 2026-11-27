"""
Scene selection (core.scenes) and coverage reporting (core.coverage).

Offline and pure numpy: no catalogue, no raster reads. The screening reads
(select_scenes / _screen) were measured against the real catalogue; see the
README, "Scene selection and coverage".
"""

import json
import re
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from eo_brain.core import coverage as cov_mod
from eo_brain.core.coverage import (
    LOW_COVERAGE_PCT,
    completion_message,
    coverage_layer,
    no_data_message,
    summarize_coverage,
)
from eo_brain.core.scenes import (
    DEFAULT_SELECTION,
    EVERY_VIEW_SELECTION,
    SceneSelection,
    fuse_days,
    greedy_select,
    obs_weights,
    pick_candidates,
    screen_resolution,
    select_scenes,
    selection_for,
    solar_day,
)

SEED = Path(__file__).resolve().parents[1] / "evals" / "fixtures" / "synthetic_seed.sql"


def item(id_, cloud, day=1, tile="MGRS-32TQP", hour=10):
    return SimpleNamespace(
        id=id_,
        datetime=datetime(2026, 11, day, hour, 0, tzinfo=timezone.utc),
        properties={"eo:cloud_cover": cloud, "grid:code": tile},
    )


# ---------------------------------------------------------------- candidates

def test_candidates_drop_only_the_nearly_overcast_and_rank_by_cloud():
    items = [item("a", 60.0), item("b", 19.8), item("c", 99.1), item("d", 94.9)]
    assert [i.id for i in pick_candidates(items)] == ["b", "a", "d"]


def test_candidates_are_capped_per_tile_keeping_the_clearest():
    cfg = replace(DEFAULT_SELECTION, max_per_tile=2)
    items = [item(f"q{i}", 10.0 * i, day=i + 1) for i in range(5)] + [item("p", 80.0, tile="MGRS-32TPP")]
    assert [i.id for i in pick_candidates(items, cfg)] == ["q0", "q1", "p"]


def test_missing_cloud_cover_ranks_in_the_middle():
    no_cc = SimpleNamespace(id="x", datetime=datetime(2026, 11, 1, tzinfo=timezone.utc), properties={})
    assert [i.id for i in pick_candidates([item("a", 60.0), no_cc, item("b", 40.0)])] == ["b", "x", "a"]


# ---------------------------------------------------------------- greedy pick

def test_obs_weights_halve_up_to_the_target_then_stop():
    assert obs_weights(3).tolist() == [1.0, 0.5, 0.25, 0.0]


def test_greedy_prefers_new_ground_over_a_second_view():
    inside = np.ones((1, 4), bool)
    clear = np.array([
        [[1, 1, 0, 0]],   # west half
        [[1, 1, 0, 0]],   # west half again, clearer catalogue figure
        [[0, 0, 1, 1]],   # east half
    ], bool)
    chosen, counts = greedy_select(clear, inside, cloud=[50.0, 10.0, 90.0])
    assert chosen[:2] == [1, 2]          # tie on the west half goes to the lower cloud; east next
    assert (counts > 0).all()


def test_greedy_stops_when_no_unit_adds_enough():
    inside = np.ones((20, 20), bool)
    clear = np.zeros((2, 20, 20), bool)
    clear[0] = True
    clear[1, 0, :2] = True              # 0.5 % of the area, as a second view: weighted gain 0.25 %
    chosen, _ = greedy_select(clear, inside, cloud=[0.0, 0.0])
    assert chosen == [0]


def test_greedy_respects_the_read_budget_and_the_scene_cap():
    inside = np.ones((2, 2), bool)
    clear = np.ones((5, 2, 2), bool)
    cfg = replace(DEFAULT_SELECTION, target_clear_obs=5, max_area_reads=2.5)
    chosen, _ = greedy_select(clear, inside, [0.0] * 5, cfg, cost=[1.0] * 5)
    assert len(chosen) == 2
    cfg = replace(DEFAULT_SELECTION, target_clear_obs=5, max_scenes=3)
    chosen, _ = greedy_select(clear, inside, [0.0] * 5, cfg, n_scenes=[2, 2, 1, 1, 1])
    assert sum([2, 2, 1, 1, 1][i] for i in chosen) <= 3


def test_a_max_composite_keeps_every_clear_day_a_median_stops_at_the_target():
    # Eight fully clear days: a median is served by three views per pixel, but a
    # per-pixel maximum (flood) must keep the one wet day, whichever it is.
    inside = np.ones((4, 4), bool)
    clear = np.ones((8, 4, 4), bool)
    median_pick, counts = greedy_select(clear, inside, [0.0] * 8, DEFAULT_SELECTION)
    assert len(median_pick) == DEFAULT_SELECTION.target_clear_obs
    assert counts.max() == DEFAULT_SELECTION.target_clear_obs
    every_pick, counts = greedy_select(clear, inside, [0.0] * 8, EVERY_VIEW_SELECTION)
    assert sorted(every_pick) == list(range(8))
    assert counts.min() == 8


def test_a_max_composite_still_respects_the_budgets_and_the_minimum_gain():
    inside = np.ones((10, 20), bool)                    # 200 pixels: the minimum gain is 1 pixel
    clear = np.ones((12, 10, 20), bool)
    clear[11] = False                                   # a fully cloudy day adds nothing
    chosen, _ = greedy_select(clear, inside, [0.0] * 12, EVERY_VIEW_SELECTION)
    assert len(chosen) == 10                            # max_area_reads 10, each day costs 1
    assert 11 not in chosen
    chosen, _ = greedy_select(clear, inside, [0.0] * 12, replace(EVERY_VIEW_SELECTION, max_scenes=4))
    assert len(chosen) == 4


@pytest.mark.parametrize("algorithm, params, expected", [
    ("flood", {}, True),
    ("flood", None, True),
    ("ndvi", {}, False),
    ("ndvi", {"composite_method": "max"}, True),
    ("ndwi", {"composite_method": "median"}, False),
    ("lisa", {}, False),
])
def test_selection_for_picks_every_view_for_a_maximum_composite(algorithm, params, expected):
    cfg = selection_for(algorithm, params)
    assert cfg.every_view_counts is expected
    assert cfg.max_area_reads == DEFAULT_SELECTION.max_area_reads      # same budgets either way
    assert cfg.max_scenes == DEFAULT_SELECTION.max_scenes


def test_greedy_spreads_near_ties_in_time():
    inside = np.ones((1, 10), bool)
    clear = np.ones((5, 1, 10), bool)
    clear[2, 0, 0] = False              # mid-window day: 10 % fewer clear pixels, still a near-tie
    when = [1, 2, 15, 29, 30]
    chosen, _ = greedy_select(clear, inside, [0.0] * 5, cost=[1.0] * 5, when=when)
    # first the clearest (index order on ties), then the far end, then the middle
    assert chosen[:3] == [0, 4, 2]


def test_greedy_with_no_clear_pixel_picks_nothing():
    chosen, counts = greedy_select(np.zeros((3, 4, 4), bool), np.ones((4, 4), bool), [0.0] * 3)
    assert chosen == [] and counts.sum() == 0


# ---------------------------------------------------------------- days

def test_same_day_tiles_are_one_observation_and_redundant_tiles_are_dropped():
    inside = np.ones((1, 4), bool)
    has = np.array([[[1, 1, 1, 1]], [[1, 1, 1, 1]], [[1, 1, 0, 0]]], bool)
    clear = np.array([[[1, 1, 0, 0]], [[1, 1, 0, 0]], [[1, 1, 0, 0]]], bool)
    d1, d2 = date(2026, 11, 3), date(2026, 11, 5)
    days = fuse_days([d1, d1, d2], [(1, "a"), (2, "b"), (3, "c")], clear, has, inside, [10.0, 5.0, 50.0])
    assert [d.day for d in days] == [d1, d2]
    assert len(days[0].members) == 1     # the second tile of the same pass adds no data
    assert days[0].cost == pytest.approx(1.0)
    assert days[1].cost == pytest.approx(0.5)


def test_fused_day_takes_the_earliest_valid_pixel_like_the_loader():
    inside = np.ones((1, 3), bool)
    has = np.array([[[1, 1, 0]], [[0, 1, 1]]], bool)
    clear = np.array([[[1, 0, 0]], [[0, 1, 1]]], bool)   # px 1: earlier scene cloudy, later clear
    d = date(2026, 11, 5)
    [day] = fuse_days([d, d], [(2, "b"), (1, "a")], clear[::-1], has[::-1], inside, [50.0, 50.0])
    assert sorted(day.members) == [0, 1]
    assert day.clear.tolist() == [[True, False, True]]   # px 1 comes from the earlier, cloudy scene


def test_solar_day_shifts_by_longitude():
    it = item("a", 0.0, day=5, hour=23)
    assert solar_day(it, 20.0) == date(2026, 11, 5) + timedelta(days=1)   # 23:00 UTC + 1 h 20
    assert solar_day(it, 12.0) == date(2026, 11, 5)


def test_screen_resolution_is_coarse_and_in_ten_metre_steps():
    assert screen_resolution((33_000, 41_000), 60, 256) == 170
    assert screen_resolution((2_000, 2_000), 10, 256) == 10


def test_select_without_mask_band_falls_back_to_the_cloud_ranking():
    provider = SimpleNamespace(cloud_mask_band=None)
    items = [item(f"s{i}", float(i)) for i in range(20)]
    sel = select_scenes(provider, items, None, "EPSG:32632", 60)
    assert sel.method == "cloud_rank"
    assert sel.scene_ids == [f"s{i}" for i in range(int(DEFAULT_SELECTION.max_area_reads))]


def test_select_with_only_overcast_scenes_returns_nothing():
    sel = select_scenes(SimpleNamespace(cloud_mask_band="scl"), [item("a", 99.0), item("b", 97.0)], None, "x", 60)
    assert sel.items == [] and sel.too_cloudy == 2
    assert "all 2 scenes in the window are at least 95% cloudy" in no_data_message(sel.to_dict())


# ---------------------------------------------------------------- coverage

def _cov(valid, has, inside=None, **sel):
    valid = np.asarray(valid, bool)
    inside = np.ones(valid.shape, bool) if inside is None else np.asarray(inside, bool)
    selection = {"found": 27, "too_cloudy": 12, "screened": 15, "with_clear": 14, "method": "pixel_mask",
                 "scene_ids": ["a", "b"], **sel}
    return summarize_coverage(valid, inside, np.asarray(has, bool), [date(2026, 10, 29), date(2026, 11, 3)],
                              [5, 0], selection)


def test_coverage_splits_the_uncovered_area_by_cause():
    c = _cov(valid=[[1, 0, 0, 0]], has=[[1, 1, 0, 0]])
    assert (c["coverage_pct"], c["uncovered_masked_pct"], c["uncovered_no_data_pct"]) == (25.0, 25.0, 50.0)
    assert c["low_coverage"] is True
    assert c["acquisition_days_with_data"] == 1 and c["first_date_with_data"] == "2026-10-29"
    assert c["scenes_used"] == 2 and c["scenes_checked"] == 15


def test_coverage_ignores_pixels_outside_the_requested_box():
    c = _cov(valid=[[1, 1, 0, 0]], has=[[1, 1, 1, 1]], inside=[[1, 1, 0, 0]])
    assert c["coverage_pct"] == 100.0 and c["low_coverage"] is False


def test_coverage_is_rounded_down_so_the_threshold_is_not_reached_by_rounding():
    valid = np.zeros(10_000, bool)
    valid[:7_999] = True                # 79.99 %
    c = summarize_coverage(valid, np.ones(10_000, bool), np.ones(10_000, bool), [], [], {})
    assert c["coverage_pct"] == 79.9 and c["low_coverage"] is True
    assert LOW_COVERAGE_PCT == 80.0


def test_completion_message_always_states_the_coverage():
    c = _cov(valid=[[1, 1, 1, 1]], has=[[1, 1, 1, 1]])
    assert completion_message(c) == (
        "Completed: cloud-free data covers 100% of the requested area (1 acquisition day, 2026-10-29).")


def test_low_coverage_message_gives_the_reason_and_the_remedy():
    c = _cov(valid=[[1, 0, 0, 0]], has=[[1, 1, 0, 0]])
    msg = completion_message(c)
    assert msg.startswith("Completed with partial coverage: cloud-free data covers only 25%")
    assert "25% of the area was masked out (cloud, shadow, snow) on every day loaded" in msg
    assert "50% was outside the footprint of the scenes loaded" in msg
    assert "of the 27 scenes in the window, 12 were at least 95% cloudy and 14 of the other 15" in msg
    assert msg.endswith("A longer time window may cover more of it.")


def test_low_coverage_message_without_screening_does_not_invent_scene_counts():
    c = _cov(valid=[[1, 0, 0, 0]], has=[[1, 1, 1, 1]], method="cloud_rank", with_clear=None, screened=0)
    msg = completion_message(c)
    assert "scenes in the window" not in msg and "75% of the area was masked out" in msg


def test_coverage_layer_uses_the_continuous_index_for_discrete_outputs():
    assert coverage_layer("flood", ["water_index", "flood_severity"]) == "water_index"
    assert coverage_layer("lisa", ["ndvi", "lisa_severity"]) == "ndvi"
    assert coverage_layer("other", ["x", "y"]) == "x"


def test_selection_dict_has_no_items():
    d = SceneSelection(items=[object()], found=1, candidates=1).to_dict()
    assert "items" not in d and d["found"] == 1


def test_seed_messages_are_what_the_pipeline_would_write():
    """The synthetic seed's completed jobs carry the message completion_message() gives their coverage."""
    rows = re.findall(r"   100,\n   '((?:[^']|'')*)',\n   '[^']*', '[^']*',\n   '(\{\"coverage\": [^']*)'\)",
                      SEED.read_text(encoding="utf-8"))
    assert len(rows) == 4
    for message, result in rows:
        assert message.replace("''", "'") == completion_message(json.loads(result)["coverage"])
    assert sum(json.loads(r)["coverage"]["low_coverage"] for _, r in rows) == 1


def test_inside_mask_follows_the_wgs84_box():
    pytest.importorskip("pyproj")
    # UTM 32N around 11.8E 43.7N; the box covers the west half of the grid only
    xs = np.array([717_000.0, 750_000.0])
    ys = np.array([4_840_000.0])
    mask = cov_mod.inside_bbox_mask(xs, ys, "EPSG:32632", (11.5, 43.5, 11.95, 43.9))
    assert mask.tolist() == [[True, False]]
