"""
eo_brain/core/distribution.py - How the values of a continuous index are spread
over the covered part of the requested area.

For an analysis whose result is a continuous index (``ndvi``, ``ndwi``,
``mndwi``), the pipeline counts the covered pixels in value intervals 0.01 wide
over [-1, 1] and stores one row per non-empty interval (table
``index_value_bins``, view ``agent.index_value_distribution``). The area above,
below or between any thresholds is then a plain sum: "how many hectares have
NDVI of at least 0.65" is ``SUM(area_ha) ... WHERE value_from >= 0.65``.

Which pixels: exactly those the coverage figures count as covered
(``core.coverage.summarize_coverage``): a finite value in the result layer and
a pixel centre inside the requested WGS84 box. So the pixel counts of a job's
intervals add up to ``coverage["covered_px"]``; ``index_distribution`` checks it.

Intervals: interval k (0 to 199) holds the values v with
-1 + k/100 <= v < -1 + (k+1)/100; the last one also holds 1.0. A value is
placed after rounding (v + 1) * 100 to 4 decimals, so a value less than 5e-7
below an edge counts as on it: 0.65 is 0.64999998 in float32 and still lands
in the interval that starts at 0.65. Values outside [-1, 1] (the indices are
clipped to it, so only numerical noise) go to the end intervals.

The area of a pixel comes from the result grid (``grid_pixel_area_m2``, the
spacing of its coordinates in a projected CRS in metres), not from the
requested resolution.

Pure numpy, except ``crs_in_metres`` (pyproj, imported when called).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

VALUE_MIN = -1.0
VALUE_MAX = 1.0
BINS_PER_UNIT = 100                                   # intervals 0.01 wide
BIN_WIDTH = 1.0 / BINS_PER_UNIT
N_BINS = int(round((VALUE_MAX - VALUE_MIN) * BINS_PER_UNIT))   # 200
EDGE_DECIMALS = 4      # (v + 1) * 100 is rounded to this many decimals before flooring
_CHUNK = 1 << 20       # values binned per step, to bound the float64 temporaries


@dataclass(frozen=True)
class ValueBin:
    """One non-empty interval: value_from <= v < value_to (the last one includes 1.0)."""

    bin_index: int
    value_from: float          # two decimals, -1.0 to 0.99
    value_to: float            # value_from + 0.01
    pixel_count: int
    area_m2: float


def bin_edge(k: int) -> float:
    """Lower edge of interval ``k`` (two decimals); ``bin_edge(N_BINS)`` is 1.0."""
    return round(VALUE_MIN + k / BINS_PER_UNIT, 2)


def bin_indices(values: Any) -> np.ndarray:
    """Interval index (0 to N_BINS - 1) of each value. NaN is not allowed here: filter first."""
    k = np.array(values, dtype=np.float64)       # a copy: modified in place below
    k -= VALUE_MIN
    k *= BINS_PER_UNIT
    np.round(k, EDGE_DECIMALS, out=k)
    np.floor(k, out=k)
    np.clip(k, 0, N_BINS - 1, out=k)
    return k.astype(np.int64)


def _covered_values(values: Any, inside: Any) -> np.ndarray:
    """The values coverage counts as covered: finite, and inside the requested area."""
    values = np.asarray(values)
    inside = np.asarray(inside, dtype=bool)
    if values.shape != inside.shape:
        raise ValueError(f"values {values.shape} and area mask {inside.shape} differ in shape")
    return values[np.isfinite(values) & inside]


def _counts_of(covered: np.ndarray) -> np.ndarray:
    counts = np.zeros(N_BINS, dtype=np.int64)
    for start in range(0, covered.size, _CHUNK):
        counts += np.bincount(bin_indices(covered[start:start + _CHUNK]), minlength=N_BINS)
    return counts


def bin_counts(values: Any, inside: Any) -> np.ndarray:
    """(N_BINS,) pixel count of each interval, over the covered pixels."""
    return _counts_of(_covered_values(values, inside))


def _bins_of(covered: np.ndarray, pixel_area_m2: float) -> List[ValueBin]:
    if not (np.isfinite(pixel_area_m2) and pixel_area_m2 > 0):
        raise ValueError(f"pixel area must be a positive number of square metres, got {pixel_area_m2!r}")
    return [
        ValueBin(bin_index=k, value_from=bin_edge(k), value_to=bin_edge(k + 1),
                 pixel_count=int(n), area_m2=float(n) * float(pixel_area_m2))
        for k, n in enumerate(_counts_of(covered)) if n
    ]


def index_value_bins(values: Any, inside: Any, pixel_area_m2: float) -> List[ValueBin]:
    """The non-empty intervals of the covered pixels, in value order."""
    return _bins_of(_covered_values(values, inside), pixel_area_m2)


def _stats_of(covered: np.ndarray) -> Optional[Dict[str, float]]:
    if covered.size == 0:
        return None
    return {
        "min": float(covered.min()),
        "max": float(covered.max()),
        "mean": float(covered.mean(dtype=np.float64)),
    }


def value_stats(values: Any, inside: Any) -> Optional[Dict[str, float]]:
    """Min, max and mean of the covered pixels (the same pixels as the intervals), or None if there are none."""
    return _stats_of(_covered_values(values, inside))


def _spacing(coords: Any, name: str) -> float:
    c = np.asarray(coords, dtype=np.float64)
    if c.ndim != 1 or c.size < 2:
        raise ValueError(f"cannot derive the pixel size from {c.size} {name} coordinate(s)")
    steps = np.abs(np.diff(c))
    step = float(np.median(steps))
    if not step > 0 or float(np.max(np.abs(steps - step))) > 1e-6 * step:
        raise ValueError(f"the {name} coordinates are not a regular grid")
    return step


def grid_pixel_area_m2(xs: Sequence[float], ys: Sequence[float]) -> float:
    """
    Area of one pixel of a regular grid, from the spacing of its pixel-centre
    coordinates. The grid's CRS must be projected in metres (``crs_in_metres``).
    """
    return _spacing(xs, "x") * _spacing(ys, "y")


def crs_in_metres(crs: Any) -> bool:
    """True when ``crs`` is a projected CRS whose horizontal axes are in metres (UTM is)."""
    from pyproj import CRS

    c = CRS.from_user_input(crs)
    return bool(c.is_projected) and all(
        (a.unit_name or "").lower() in ("metre", "meter") for a in c.axis_info[:2]
    )


@dataclass
class IndexDistribution:
    """What the pipeline records for one continuous-index job."""

    index: str
    pixel_area_m2: float
    bins: List[ValueBin] = field(default_factory=list)
    stats: Optional[Dict[str, float]] = None

    @property
    def pixels(self) -> int:
        return sum(b.pixel_count for b in self.bins)

    def summary(self) -> Dict[str, Any]:
        """The job's ``result.value_distribution`` block (agent.analyses reads min, max, mean)."""
        stats = self.stats or {}
        return {
            "index": self.index,
            "pixels": self.pixels,
            "pixel_area_m2": self.pixel_area_m2,
            "area_m2": self.pixels * self.pixel_area_m2,
            "bin_width": BIN_WIDTH,
            "min": stats.get("min"),
            "max": stats.get("max"),
            "mean": stats.get("mean"),
        }


def index_distribution(
    index: str, values: Any, inside: Any, pixel_area_m2: float, covered_px: int,
) -> IndexDistribution:
    """
    Intervals and min / max / mean of the covered pixels. ``covered_px`` is
    ``coverage["covered_px"]`` of the same job: the intervals must count exactly
    those pixels, or this raises instead of recording a distribution of other pixels.
    The covered values are selected once (one copy, 4 bytes per covered pixel for
    float32) and shared by the intervals and the statistics.
    """
    covered = _covered_values(values, inside)
    dist = IndexDistribution(
        index=index, pixel_area_m2=float(pixel_area_m2),
        bins=_bins_of(covered, pixel_area_m2), stats=_stats_of(covered),
    )
    if dist.pixels != int(covered_px):
        raise ValueError(
            f"the intervals count {dist.pixels} pixels, the coverage counts {int(covered_px)}"
        )
    return dist
