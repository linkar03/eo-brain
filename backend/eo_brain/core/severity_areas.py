"""
eo_brain/core/severity_areas.py - Exact area of each class of a discrete result
(flood, lisa) over the covered part of the requested area.

The alert polygons of a flood or lisa analysis (table alert_features) are mapped
zones: connected patches of one class with at least a minimum number of pixels
(algorithms.vectorize). Patches under that size are not mapped, so the zones do
not add up to the area of a class. These counts do: for every class, the
pipeline counts the covered pixels and stores one row per class, also for a
class that holds no pixel (table severity_class_areas, view agent.severity_areas).
A sum over a class is then 0 when the class holds no pixel, and it has no row
only when no areas were recorded.

Classes (``severity_classes``):

* flood: severity 0 to 3 (Severity), with quadrant 0: flood has no Moran quadrant.
* lisa: severity 0 with every Moran quadrant 0 to 4, where 0 marks the covered
  pixels LISA did not classify (not vegetated, or no vegetated pixel in the
  neighbourhood) and 1 to 4 the classified ones that are not anomalies; and
  severity 1 to 3 with quadrant HL or LH (OUTLIER_QUADRANTS): LISA gives a
  severity to spatial outliers only. A severity above 0 with any other quadrant
  is refused.

Which pixels: exactly those the coverage figures count as covered
(``core.coverage.summarize_coverage``): a finite value in the coverage layer
(flood: water_index; lisa: ndvi) and a pixel centre inside the requested WGS84
box. So the pixel counts of a job's classes add up to ``coverage["covered_px"]``;
``severity_areas`` checks it. The area of a pixel comes from the result grid
(``core.distribution.grid_pixel_area_m2``).

Pure numpy (the class codes come from algorithms.base).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from eo_brain.algorithms.base import OUTLIER_QUADRANTS, MoranQuadrant, Severity

ALGORITHMS = ("flood", "lisa")
_N_QUADRANTS = max(MoranQuadrant) + 1          # codes 0 to 4
_CHUNK = 1 << 20                                # pixels classified per step, to bound the temporaries


def severity_classes(algorithm: str) -> List[Tuple[int, int]]:
    """Every (severity, quadrant) class of ``algorithm``, in the order they are stored."""
    if algorithm == "flood":
        return [(int(s), int(MoranQuadrant.NONE)) for s in Severity]
    if algorithm == "lisa":
        return [(int(Severity.NONE), int(q)) for q in MoranQuadrant] + [
            (int(s), int(q)) for s in Severity if s is not Severity.NONE for q in OUTLIER_QUADRANTS
        ]
    raise ValueError(f"no severity classes for algorithm {algorithm!r}: expected one of {ALGORITHMS}")


@dataclass(frozen=True)
class SeverityClass:
    """One class of one job: its covered pixels and their area (both 0 for an empty class)."""

    algorithm: str
    severity: int
    quadrant: int              # MoranQuadrant code; 0 for flood, and for lisa pixels not classified
    pixel_count: int
    area_m2: float

    @property
    def vegetated(self) -> Optional[bool]:
        """lisa: the pixel was classified (vegetated, with vegetated neighbours); None for flood."""
        return self.quadrant != MoranQuadrant.NONE if self.algorithm == "lisa" else None


@dataclass
class SeverityAreas:
    """What the pipeline records for one flood or lisa job."""

    algorithm: str
    pixel_area_m2: float
    classes: List[SeverityClass] = field(default_factory=list)

    @property
    def pixels(self) -> int:
        return sum(c.pixel_count for c in self.classes)

    def summary(self) -> Dict[str, Any]:
        """The job's ``result.severity_areas`` block (the rows themselves go to severity_class_areas)."""
        by_severity = {str(int(s)): 0 for s in Severity}
        for c in self.classes:
            by_severity[str(c.severity)] += c.pixel_count
        block: Dict[str, Any] = {
            "algorithm": self.algorithm,
            "pixels": self.pixels,
            "pixel_area_m2": self.pixel_area_m2,
            "area_m2": self.pixels * self.pixel_area_m2,
            "pixels_by_severity": by_severity,
        }
        if self.algorithm == "lisa":
            block["vegetated_pixels"] = sum(c.pixel_count for c in self.classes if c.vegetated)
        return block


def _class_counts(algorithm: str, severity: np.ndarray, quadrant: Optional[np.ndarray],
                  covered: np.ndarray) -> np.ndarray:
    """(4, 5) pixel count per severity and quadrant over the covered pixels; refuses impossible classes."""
    sev_flat, cov_flat = severity.ravel(), covered.ravel()
    quad_flat = quadrant.ravel() if quadrant is not None else None
    counts = np.zeros(len(Severity) * _N_QUADRANTS, dtype=np.int64)
    for start in range(0, sev_flat.size, _CHUNK):
        part = slice(start, start + _CHUNK)
        keep = cov_flat[part]
        sev = sev_flat[part][keep].astype(np.int64)
        if sev.size and (sev.min() < Severity.NONE or sev.max() > Severity.ALERT):
            bad = np.unique(sev[(sev < Severity.NONE) | (sev > Severity.ALERT)])
            raise ValueError(f"severity outside 0 to 3 on covered pixels: {bad.tolist()}")
        if quad_flat is None:
            codes = sev * _N_QUADRANTS
        else:
            quad = quad_flat[part][keep].astype(np.int64)
            if quad.size and (quad.min() < 0 or quad.max() >= _N_QUADRANTS):
                raise ValueError("Moran quadrant outside 0 to 4 on covered pixels")
            codes = sev * _N_QUADRANTS + quad
        counts += np.bincount(codes, minlength=counts.size)
    table = counts.reshape(len(Severity), _N_QUADRANTS)
    allowed = np.zeros_like(table, dtype=bool)
    for s, q in severity_classes(algorithm):
        allowed[s, q] = True
    if (table[~allowed] > 0).any():
        bad = [(int(s), int(q)) for s, q in zip(*np.nonzero((table > 0) & ~allowed))]
        raise ValueError(f"{algorithm}: covered pixels in impossible (severity, quadrant) classes {bad}")
    return table


def severity_areas(
    algorithm: str,
    severity: Any,
    quadrant: Any,
    coverage_values: Any,
    inside: Any,
    pixel_area_m2: float,
    covered_px: int,
) -> SeverityAreas:
    """
    Pixel count and area of every class of a flood or lisa result, over the
    covered pixels: a finite value in ``coverage_values`` (the coverage layer,
    core.coverage.COVERAGE_LAYER) and ``inside`` the requested box.

    ``severity`` is the result's severity layer (0 to 3); ``quadrant`` its Moran
    quadrant layer for lisa, None for flood. ``covered_px`` is
    ``coverage["covered_px"]`` of the same job: the classes must count exactly
    those pixels, or this raises instead of recording areas of other pixels.
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(f"no severity classes for algorithm {algorithm!r}: expected one of {ALGORITHMS}")
    if (algorithm == "lisa") != (quadrant is not None):
        raise ValueError("lisa needs its Moran quadrant layer; flood has none")
    if not (np.isfinite(pixel_area_m2) and pixel_area_m2 > 0):
        raise ValueError(f"pixel area must be a positive number of square metres, got {pixel_area_m2!r}")
    severity = np.asarray(severity)
    values = np.asarray(coverage_values)
    inside = np.asarray(inside, dtype=bool)
    arrays = {"severity": severity, "coverage layer": values, "area mask": inside}
    if quadrant is not None:
        quadrant = np.asarray(quadrant)
        arrays["quadrant"] = quadrant
    shapes = {name: a.shape for name, a in arrays.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"layers differ in shape: {shapes}")

    table = _class_counts(algorithm, severity, quadrant, np.isfinite(values) & inside)
    area = float(pixel_area_m2)
    result = SeverityAreas(
        algorithm=algorithm, pixel_area_m2=area,
        classes=[
            SeverityClass(algorithm=algorithm, severity=s, quadrant=q,
                          pixel_count=int(table[s, q]), area_m2=float(table[s, q]) * area)
            for s, q in severity_classes(algorithm)
        ],
    )
    if result.pixels != int(covered_px):
        raise ValueError(f"the classes count {result.pixels} pixels, the coverage counts {int(covered_px)}")
    return result
