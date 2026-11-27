"""
eo_brain/core/scenes.py - Which scenes an analysis loads.

The pipeline used to ask the catalogue only for scenes with a tile-level
``eo:cloud_cover`` under 30 %. That number describes a whole 110 km tile, so a
cloudy month could leave one usable scene for an area that other scenes saw
clearly, and the result covered a sliver of it. The pixel mask (SCL) already
removes clouds pixel by pixel and the composite is a median, so the scene-level
number is now only a ranking and a loose bound:

1. Search the window with no cloud filter (``found``).
2. Candidates: drop scenes whose tile-level cloud cover is at least
   ``loose_cloud_bound`` (95 %); keep the ``max_per_tile`` clearest scenes of each
   tile (MGRS tile for Sentinel-2).
3. Screen: read only the pixel-mask band of each candidate, at a coarse
   resolution (at most ``screen_max_px`` pixels on the longer side), over the
   requested box: where it has data, where it is clear.
4. Group by acquisition day, as the loader does (``groupby="solar_day"``: the
   scenes of one day are fused into one time slice, the earliest valid pixel
   wins). A day is one observation for the median, however many tiles it
   spans. Within a day, keep only the scenes that add data over the area:
   overlapping tiles of the same pass hold the same pixels, reading both is
   wasted work.
5. Pick days greedily: add the day that adds the most clear pixels where the
   area has no clear view yet (weight 1), then where it has fewer than
   ``target_clear_obs`` (weights 1/2, 1/4, ...), until no day adds at least
   ``min_gain_share`` of the area, or the next day would exceed the budget.
   Days whose gain is within ``tie_share`` (10 %) of the best are near-ties: the one
   farthest in time from the days already picked wins, so a clear month is
   sampled across the window rather than in one week. For a per-pixel maximum
   over time (``flood``) every clear view counts the same, with no target, so
   every day that adds clear pixels is kept within the budgets
   (``selection_for``).

Only the scenes of the picked days are then loaded at full resolution, and the
per-pixel mask is applied as before. The work is bounded twice: the picked
scenes may together hold data for at most ``max_area_reads`` times the
requested area (each scene costs the share of the area its footprint covers,
so an area spanning several tiles gets more, smaller scenes for the same I/O),
and at most ``max_scenes`` scenes are opened. The full-resolution grid is
already bounded by the resolution tiers, so memory per time slice is bounded
too, and the median sees at most one slice per picked day.

``pick_candidates``, ``fuse_days`` and ``greedy_select`` are pure (numpy / item
properties) and unit-tested offline; ``select_scenes`` adds the screening reads
(odc-stac).
"""

from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SelectionConfig:
    max_area_reads: float = 10.0
    max_scenes: int = 40
    max_per_tile: int = 16
    loose_cloud_bound: float = 95.0
    target_clear_obs: int = 3
    min_gain_share: float = 0.005
    tie_share: float = 0.1
    screen_max_px: int = 256
    screen_workers: int = 8
    # Every clear view counts the same, with no target: for a per-pixel
    # maximum over time, where the one wet day matters (see selection_for).
    every_view_counts: bool = False


DEFAULT_SELECTION = SelectionConfig()
EVERY_VIEW_SELECTION = SelectionConfig(every_view_counts=True)


def selection_for(algorithm: str, params: Optional[Dict[str, Any]] = None) -> SelectionConfig:
    """
    The selection for an analysis. A median composite needs a few clear views
    per pixel, so the default stops adding days once most pixels have
    ``target_clear_obs``. ``flood`` (and any composite_method="max") keeps the
    wettest view of each pixel: a flood seen on one day only must not be
    dropped, so every day that adds clear pixels is kept, within the budgets.
    """
    if algorithm == "flood" or (params or {}).get("composite_method") == "max":
        return EVERY_VIEW_SELECTION
    return DEFAULT_SELECTION


@dataclass
class SceneSelection:
    items: List[Any]
    found: int
    candidates: int
    too_cloudy: int = 0                   # found, but at or above loose_cloud_bound: not checked
    screened: int = 0
    with_clear: Optional[int] = None      # screened scenes with a clear pixel over the area
    method: str = "pixel_mask"            # "pixel_mask" | "cloud_rank" (screening unavailable)
    screen_resolution_m: Optional[int] = None
    expected_coverage_pct: Optional[float] = None  # share of the area clear on >= 1 picked day, at screening resolution
    area_reads: Optional[float] = None    # sum over loaded scenes of the share of the area they have data for
    days_picked: List[str] = field(default_factory=list)
    note: Optional[str] = None
    scene_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("items")
        return d


# ---------------------------------------------------------------- item properties

def scene_cloud(item) -> float:
    """Tile-level cloud cover in percent; 50 when the catalogue gives none."""
    try:
        return float(item.properties.get("eo:cloud_cover", 50.0))
    except (TypeError, ValueError):
        return 50.0


def tile_key(item) -> str:
    p = item.properties
    if p.get("grid:code"):
        return str(p["grid:code"])
    if p.get("s2:mgrs_tile"):
        return str(p["s2:mgrs_tile"])
    if p.get("landsat:wrs_path") is not None:
        return f"WRS-{p.get('landsat:wrs_path')}-{p.get('landsat:wrs_row')}"
    return ""


def solar_day(item, lon: float) -> date:
    """Acquisition day in local solar time at ``lon``, as odc-stac's ``groupby="solar_day"``."""
    return (item.datetime + timedelta(hours=lon / 15.0)).date()


def _rank_key(item) -> Tuple[float, float, str]:
    """Clearest first; then the most recent; then the id, for a stable order."""
    ts = item.datetime.timestamp() if item.datetime else 0.0
    return (scene_cloud(item), -ts, item.id)


def pick_candidates(items: Sequence[Any], cfg: SelectionConfig = DEFAULT_SELECTION) -> List[Any]:
    """Scenes under the loose cloud bound, at most ``max_per_tile`` per tile, clearest first."""
    by_tile: Dict[str, List[Any]] = {}
    for it in items:
        if scene_cloud(it) < cfg.loose_cloud_bound:
            by_tile.setdefault(tile_key(it), []).append(it)
    kept = [it for group in by_tile.values() for it in sorted(group, key=_rank_key)[: cfg.max_per_tile]]
    return sorted(kept, key=_rank_key)


def rank_by_cloud(items: Sequence[Any], n: int) -> List[Any]:
    return sorted(items, key=_rank_key)[:n]


def _fallback_count(cfg: SelectionConfig) -> int:
    """Scenes to load when they cannot be screened: each assumed to cover the whole area."""
    return max(1, min(cfg.max_scenes, int(cfg.max_area_reads)))


# ---------------------------------------------------------------- days

@dataclass
class Day:
    day: date
    members: List[int]        # candidate indices kept for this day, in the loader's fusion order
    clear: np.ndarray         # (H, W) bool: clear after fusion
    cost: float               # sum of the members' data share of the area
    cloud: float              # lowest tile-level cloud cover among the members


def fuse_days(
    days: Sequence[date],
    order_keys: Sequence[Any],
    clear: np.ndarray,
    has_data: np.ndarray,
    inside: np.ndarray,
    cloud: Sequence[float],
    cfg: SelectionConfig = DEFAULT_SELECTION,
) -> List[Day]:
    """
    One ``Day`` per acquisition day, as the loader will fuse it.

    ``days[i]`` / ``order_keys[i]``: scene i's solar day and its fusion order
    within the day (odc-stac sorts by time, then id; the earliest valid pixel
    wins). Within a day, scenes are kept, most data first, only if they add
    data over at least ``min_gain_share`` of the area.
    """
    area = int(inside.sum())
    min_new = cfg.min_gain_share * area
    by_day: Dict[date, List[int]] = {}
    for i, d in enumerate(days):
        by_day.setdefault(d, []).append(i)

    out: List[Day] = []
    for d in sorted(by_day):
        idx = by_day[d]
        data_in = {i: int((has_data[i] & inside).sum()) for i in idx}
        clear_in = {i: int((clear[i] & inside).sum()) for i in idx}
        covered = np.zeros(inside.shape, dtype=bool)
        kept: List[int] = []
        for i in sorted(idx, key=lambda i: (-data_in[i], -clear_in[i], cloud[i], i)):
            new = int((has_data[i] & inside & ~covered).sum())
            if new > 0 and (new >= min_new or not kept):
                kept.append(i)
                covered |= has_data[i]
        if not kept:
            continue
        kept.sort(key=lambda i: order_keys[i])
        fused_clear = np.zeros(inside.shape, dtype=bool)
        filled = np.zeros(inside.shape, dtype=bool)
        for i in kept:
            take = has_data[i] & ~filled
            fused_clear[take] = clear[i][take]
            filled |= has_data[i]
        out.append(Day(
            day=d, members=kept, clear=fused_clear,
            cost=sum(data_in[i] for i in kept) / area if area else 0.0,
            cloud=min(cloud[i] for i in kept),
        ))
    return out


# ---------------------------------------------------------------- greedy pick

def obs_weights(target_clear_obs: int) -> np.ndarray:
    """Weight of one more clear view for a pixel that has c views: 1/2**c below the target, then 0."""
    return np.array([0.5 ** c for c in range(target_clear_obs)] + [0.0], dtype=np.float64)


def greedy_select(
    clear: np.ndarray,
    inside: np.ndarray,
    cloud: Sequence[float],
    cfg: SelectionConfig = DEFAULT_SELECTION,
    cost: Optional[Sequence[float]] = None,
    when: Optional[Sequence[float]] = None,
    n_scenes: Optional[Sequence[int]] = None,
) -> Tuple[List[int], np.ndarray]:
    """
    Indices of the picked units (days), in pick order, and the clear-view count per pixel.

    ``clear`` is (n, H, W) bool, ``inside`` (H, W) bool: the pixels of the
    requested area. ``cost``: each unit's share of the area that it reads
    (default 1.0); the picked costs add up to at most ``cfg.max_area_reads``.
    ``n_scenes``: scenes each unit opens (default 1), at most ``cfg.max_scenes``
    in total. ``when``: each unit's time (any unit, e.g. days); among near-ties
    the unit farthest from the picked ones wins. Remaining ties go to the lower
    ``cloud``, then the lower index.
    """
    n = clear.shape[0]
    counts = np.zeros(inside.shape, dtype=np.int32)
    area = int(inside.sum())
    if n == 0 or area == 0:
        return [], counts
    costs = [1.0] * n if cost is None else [float(c) for c in cost]
    scenes = [1] * n if n_scenes is None else [int(s) for s in n_scenes]
    usable = (clear & inside[None, :, :]).reshape(n, -1)
    if cfg.every_view_counts:
        lut, k = np.ones(1, dtype=np.float64), 0      # weight 1 whatever the count
    else:
        lut, k = obs_weights(cfg.target_clear_obs), cfg.target_clear_obs
    min_gain = cfg.min_gain_share * area
    budget, scene_budget = cfg.max_area_reads, cfg.max_scenes
    remaining = list(range(n))
    chosen: List[int] = []
    while True:
        remaining = [i for i in remaining if costs[i] <= budget + 1e-9 and scenes[i] <= scene_budget]
        if not remaining:
            break
        w = lut[np.minimum(counts.reshape(-1), k)]
        gains = usable[remaining].astype(np.float64) @ w
        best_gain = float(gains.max())
        if best_gain <= 0 or best_gain < min_gain:
            break

        def spread(i: int) -> float:
            if when is None or not chosen:
                return 0.0
            return min(abs(when[i] - when[c]) for c in chosen)

        near = [j for j in range(len(remaining)) if gains[j] >= best_gain * (1.0 - cfg.tie_share)]
        best_pos = min(near, key=lambda j: (-spread(remaining[j]), cloud[remaining[j]], remaining[j]))
        best = remaining.pop(best_pos)
        chosen.append(best)
        budget -= costs[best]
        scene_budget -= scenes[best]
        counts += usable[best].reshape(inside.shape)
    return chosen, counts


def screen_resolution(bbox_extent_m: Tuple[float, float], resolution_m: int, max_px: int) -> int:
    """Coarse resolution for the mask screening: at most ``max_px`` on the longer side, in 10 m steps."""
    longest = max(bbox_extent_m)
    res = max(float(resolution_m), longest / max_px)
    return int(math.ceil(res / 10.0) * 10)


# ---------------------------------------------------------------- with the catalogue

def select_scenes(
    provider,
    items: Sequence[Any],
    bbox,
    crs: str,
    resolution_m: int,
    cfg: SelectionConfig = DEFAULT_SELECTION,
) -> SceneSelection:
    """Pick the scenes to load for ``bbox`` (see the module docstring)."""
    found = len(items)
    candidates = pick_candidates(items, cfg)
    too_cloudy = sum(1 for it in items if scene_cloud(it) >= cfg.loose_cloud_bound)
    sel = SceneSelection(items=[], found=found, candidates=len(candidates), too_cloudy=too_cloudy)
    if not candidates:
        sel.note = f"all {found} scenes are at least {cfg.loose_cloud_bound:g}% cloudy"
        return sel

    def fallback(note: str) -> SceneSelection:
        chosen = rank_by_cloud(candidates, _fallback_count(cfg))
        sel.items, sel.method, sel.note = chosen, "cloud_rank", note
        sel.scene_ids = [it.id for it in chosen]
        return sel

    mask_band = getattr(provider, "cloud_mask_band", None)
    if not mask_band:
        return fallback("provider has no pixel mask band")
    try:
        clear, has_data, inside, res = _screen(provider, candidates, bbox, crs, resolution_m, mask_band, cfg)
    except Exception as e:  # network, missing asset, ...: fall back to the ranking
        logger.warning(f"Scene screening failed, ranking by cloud cover instead: {e}")
        return fallback(f"mask screening failed: {type(e).__name__}")

    area = int(inside.sum())
    lon = (bbox.to_tuple()[0] + bbox.to_tuple()[2]) / 2.0
    cloud = [scene_cloud(it) for it in candidates]
    days = fuse_days(
        [solar_day(it, lon) for it in candidates],
        [(it.datetime, it.id) for it in candidates],
        clear, has_data, inside, cloud, cfg,
    )
    picked, counts = greedy_select(
        np.stack([d.clear for d in days]) if days else np.zeros((0,) + inside.shape, dtype=bool),
        inside,
        [d.cloud for d in days],
        cfg,
        cost=[d.cost for d in days],
        when=[d.day.toordinal() for d in days],
        n_scenes=[len(d.members) for d in days],
    )
    chosen_days = sorted((days[i] for i in picked), key=lambda d: d.day)
    members = [i for d in chosen_days for i in d.members]
    sel.screened = len(candidates)
    sel.with_clear = int(sum(bool((c & inside).any()) for c in clear))
    sel.screen_resolution_m = res
    sel.expected_coverage_pct = round(100.0 * float(((counts > 0) & inside).sum()) / area, 1) if area else 0.0
    sel.area_reads = round(sum(days[i].cost for i in picked), 2)
    sel.days_picked = [d.day.isoformat() for d in chosen_days]
    sel.items = [candidates[i] for i in members]
    sel.scene_ids = [it.id for it in sel.items]
    logger.info(
        f"Scene selection: {found} found, {len(candidates)} candidates, {sel.with_clear} with clear pixels, "
        f"{len(days)} days, {len(picked)} days / {len(sel.items)} scenes picked, "
        f"expected coverage {sel.expected_coverage_pct}% at {res} m"
    )
    return sel


def _screen(provider, candidates, bbox, crs, resolution_m, mask_band, cfg):
    """(clear, has_data (n, H, W), inside (H, W), resolution) from each candidate's mask band."""
    from odc.stac import load as odc_load

    from eo_brain.core.coverage import inside_bbox_mask
    from eo_brain.core.datacube import configure_cloud_reads

    configure_cloud_reads()
    asset = provider.get_asset_mapping().get(mask_band, mask_band)
    # The mask band's declared nodata, as DataCube.load_items uses it (some
    # catalogues, e.g. Planetary Computer, declare none).
    stac_cfg: Dict[str, Any] = {"warnings": "ignore"}
    if getattr(provider, "mask_nodata", None) is not None:
        stac_cfg["assets"] = {asset: {"data_type": provider.mask_data_type, "nodata": provider.mask_nodata}}
    w, s, e, n = bbox.to_tuple()
    lat = math.radians((s + n) / 2)
    extent = ((e - w) * 111_320 * math.cos(lat), (n - s) * 110_540)
    res = screen_resolution(extent, resolution_m, cfg.screen_max_px)

    def read(item):
        ds = odc_load(
            [item], bands=[asset], bbox=bbox.to_tuple(), crs=crs, resolution=res,
            groupby="id", stac_cfg={"*": stac_cfg},
        )
        band = ds[asset].isel(time=0)
        nodata = band.attrs.get("nodata")
        has_data = (band.values != nodata) if nodata is not None else np.ones(band.shape, dtype=bool)
        return provider.compute_cloud_mask(band).values.astype(bool) & has_data, has_data, band

    with ThreadPoolExecutor(max_workers=cfg.screen_workers) as pool:
        out = list(pool.map(read, candidates))

    shapes = {c.shape for c, _, _ in out}
    if len(shapes) != 1:
        raise ValueError(f"screening grids differ: {sorted(shapes)}")
    band0 = out[0][2]
    inside = inside_bbox_mask(band0.x.values, band0.y.values, crs, bbox.to_tuple())
    clear = np.stack([c for c, _, _ in out])
    has_data = np.stack([d for _, d, _ in out])
    return clear, has_data, inside, res
