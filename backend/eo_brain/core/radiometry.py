"""
eo_brain/core/radiometry.py - Digital numbers to surface reflectance, scene by scene.

Every index (NDVI, NDWI, MNDWI, flood, LISA) is computed from surface
reflectance:

    reflectance = DN * scale + offset        where DN != nodata

``scale``, ``offset`` and ``nodata`` are not properties of a band or of a
provider but of each scene: one Sentinel-2 composite can mix products of
different processing baselines, and one provider can deliver the same product
with or without ESA's offset. So each provider declares a rule per scene
(``STACProvider.radiometry(item)``), the loader keeps scenes with different
rules in different time slices (``DataCube.load_items``) and attaches each
slice's rule to the data (coordinates ``radiometry_*`` along ``time``), and the
one conversion below is applied to every acquisition before any compositing.

The rules and their sources (README, "Radiometry"):

* Sentinel-2 L2A as ESA distributes it (Planetary Computer, Copernicus Data
  Space). From processing baseline 04.00 (25 January 2022) the DNs carry
  ``BOA_ADD_OFFSET`` = -1000: reflectance = (DN - 1000) / 10000; before 04.00,
  DN / 10000. DN 0 is NO_DATA. The baseline comes from the item
  (``s2:processing_baseline``, ``processing:version``, or the ``Nxxyy`` field of
  the product name), never from the acquisition date: reprocessed products of
  old dates have baseline 05.xx.
* Sentinel-2 L2A on Earth Search (``sentinel-2-l2a``). Earth Search removes the
  offset when it converts the products to COGs and says so per item with
  ``earthsearch:boa_offset_applied``. Baseline < 04.00, or >= 04.00 with the
  flag true: DN / 10000. The ``raster:bands`` offset of the assets (-0.1) is
  NOT used: it is -0.1 whether or not the offset was removed, and applying it
  subtracts the offset twice (Element84/earth-search #9, #71; checked on
  pixels). Baseline >= 04.00 with the flag false or missing is undetermined:
  flag-false scenes have been reported both with and without the offset
  (Element84/earth-search #66, #71), so the metadata does not say which.
* Landsat Collection 2 Level-2 surface reflectance (USGS): DN * 0.0000275 - 0.2,
  fill 0.

A scene whose rule cannot be determined is not loaded (``split_by_radiometry``)
and the job result lists it: a wrong offset moves every reflectance by 0.1, which
changes an NDVI median from 0.81 to 1.00, and a median composite would carry it
into the result with no trace. Refusing costs coverage, which the coverage block
measures and reports.

Non-positive reflectance. With an offset rule, DNs below the offset give
reflectance <= 0 (ESA introduced the offset to keep these negative values over
very dark surfaces: water, deep shadow). They are raised to one DN step above
zero (``scale``), the floor Earth Search applies when it removes the offset
(checked on pixels: where Planetary Computer has DN < 1000, Earth Search has
DN 1). They are not masked: a dark water pixel is exactly what the water
indices look for, and masking it would remove water from flood maps. The
floor also keeps ratios finite and makes the two providers agree. Reflectance
above 1 (bright cloud remnants, snow) is kept as it is.

Pure numpy / xarray: no STAC or raster dependency, unit-tested offline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Coordinates (along ``time``) that carry each acquisition's rule.
RULE = "radiometry_rule"
SCALE = "radiometry_scale"
OFFSET = "radiometry_offset"
NODATA = "radiometry_nodata"

# ``attrs["units"]`` of a band that already holds surface reflectance (0-1).
REFLECTANCE_UNITS = "surface_reflectance"

NON_POSITIVE_POLICY = (
    "reflectance <= 0 after the offset is raised to one DN step (the rule's scale); "
    "values above 1 are kept"
)

SOURCES = {
    "esa_l2a": "https://sentiwiki.copernicus.eu/web/s2-products",
    "esa_pb04": "https://sentinels.copernicus.eu/-/copernicus-sentinel-2-major-products-upgrade-upcoming",
    "esa_offset_value": "https://forum.step.esa.int/t/info-introduction-of-additional-radiometric-offset-in-pb04-00-products/35431",
    "earth_search_flag": "https://github.com/Element84/earth-search/issues/9",
    "earth_search_flag_false": "https://github.com/Element84/earth-search/issues/66",
    "earth_search_anomalies": "https://github.com/Element84/earth-search/issues/71",
    "planetary_computer": "https://github.com/microsoft/PlanetaryComputerExamples/blob/main/datasets/sentinel-2-l2a/baseline-change.ipynb",
    "usgs_c2_l2": "https://www.usgs.gov/faqs/how-do-i-use-scale-factor-landsat-level-2-science-products",
}


class RadiometryUndetermined(ValueError):
    """The rule turning a scene's (or a band's) DNs into reflectance is not known."""


@dataclass(frozen=True)
class Radiometry:
    """reflectance = DN * scale + offset, where DN != nodata."""

    rule: str
    scale: float
    offset: float
    nodata: int
    description: str

    @property
    def key(self) -> str:
        """Scenes with the same key can share a time slice (fused before conversion)."""
        return f"{self.rule}|{self.scale!r}|{self.offset!r}|{self.nodata}"

    def as_dict(self) -> Dict[str, Any]:
        return {"rule": self.rule, "scale": self.scale, "offset": self.offset,
                "nodata": self.nodata, "description": self.description}


S2_L2A_DN = Radiometry(
    "s2-l2a-pre-pb04", 1e-4, 0.0, 0,
    "Sentinel-2 L2A, processing baseline before 04.00: reflectance = DN / 10000",
)
S2_L2A_BOA_OFFSET = Radiometry(
    "s2-l2a-boa-offset", 1e-4, -0.1, 0,
    "Sentinel-2 L2A, processing baseline 04.00 or later, as ESA distributes it: "
    "reflectance = (DN - 1000) / 10000",
)
S2_L2A_OFFSET_REMOVED = Radiometry(
    "s2-l2a-offset-removed", 1e-4, 0.0, 0,
    "Sentinel-2 L2A, processing baseline 04.00 or later, offset already removed by Earth Search "
    "(earthsearch:boa_offset_applied true): reflectance = DN / 10000",
)
LANDSAT_C2_L2_SR = Radiometry(
    "landsat-c2-l2-sr", 2.75e-5, -0.2, 0,
    "Landsat Collection 2 Level-2 surface reflectance (USGS): reflectance = DN * 0.0000275 - 0.2",
)

PB_OFFSET_FROM = (4, 0)
_PRODUCT_BASELINE = re.compile(r"_N(\d{2})(\d{2})_")


# ---------------------------------------------------------------- rules per scene

def _props(item) -> Dict[str, Any]:
    return getattr(item, "properties", None) or {}


def s2_processing_baseline(item) -> Optional[Tuple[int, int]]:
    """(major, minor) of a Sentinel-2 product's processing baseline, or None if the item does not say."""
    p = _props(item)
    for key in ("s2:processing_baseline", "processing:version"):
        value = p.get(key)
        if value is None:
            continue
        m = re.fullmatch(r"\s*(\d{1,2})\.(\d{1,2})\s*", str(value))
        if m:
            return int(m.group(1)), int(m.group(2))
    for text in (p.get("s2:product_uri"), getattr(item, "id", None)):
        m = _PRODUCT_BASELINE.search(str(text or ""))
        if m:
            return int(m.group(1)), int(m.group(2))
    return None


def _baseline_text(pb: Tuple[int, int]) -> str:
    return f"{pb[0]:02d}.{pb[1]:02d}"


def sentinel2_esa(item) -> Radiometry:
    """Sentinel-2 L2A with ESA's DNs (Planetary Computer, Copernicus Data Space)."""
    pb = s2_processing_baseline(item)
    if pb is None:
        raise RadiometryUndetermined(
            "no processing baseline in the item (s2:processing_baseline, processing:version "
            "or the product name), so whether its DNs carry the -1000 offset is unknown"
        )
    return S2_L2A_DN if pb < PB_OFFSET_FROM else S2_L2A_BOA_OFFSET


def sentinel2_earth_search(item) -> Radiometry:
    """Sentinel-2 L2A on Earth Search (``sentinel-2-l2a``): the baseline, then Earth Search's flag."""
    pb = s2_processing_baseline(item)
    if pb is None:
        raise RadiometryUndetermined(
            "no processing baseline in the item, so whether its DNs carry the -1000 offset is unknown"
        )
    if pb < PB_OFFSET_FROM:
        return S2_L2A_DN
    flag = _props(item).get("earthsearch:boa_offset_applied")
    if flag is True:
        return S2_L2A_OFFSET_REMOVED
    state = "missing" if flag is None else repr(flag)
    raise RadiometryUndetermined(
        f"processing baseline {_baseline_text(pb)} with earthsearch:boa_offset_applied {state}: "
        "Earth Search scenes with this flag false have been found both with and without the offset "
        "(Element84/earth-search #66, #71), so the metadata does not say whether "
        "reflectance is DN / 10000 or (DN - 1000) / 10000"
    )


def landsat_c2_l2(item) -> Radiometry:
    """Landsat Collection 2 Level-2 surface reflectance."""
    p = _props(item)
    number, correction = str(p.get("landsat:collection_number", "")), str(p.get("landsat:correction", ""))
    if number == "02" and correction.startswith("L2"):
        return LANDSAT_C2_L2_SR
    raise RadiometryUndetermined(
        f"not a Collection 2 Level-2 product (landsat:collection_number {number or 'missing'}, "
        f"landsat:correction {correction or 'missing'})"
    )


def split_by_radiometry(provider, items: Iterable[Any]) -> Tuple[List[Any], List[Dict[str, str]]]:
    """(scenes whose rule is known, [{id, reason}] for the others)."""
    usable, refused = [], []
    for item in items:
        try:
            provider.radiometry(item)
        except RadiometryUndetermined as e:
            refused.append({"id": str(getattr(item, "id", "?")), "reason": str(e)})
        else:
            usable.append(item)
    return usable, refused


# ---------------------------------------------------------------- time slices

def radiometry_groups(items: Sequence[Any], scene_rules: Dict[str, Radiometry], lon: float):
    """
    The odc-stac ``groupby`` callable for ``items`` and, in time-slice order,
    the rule of each slice.

    Scenes of one solar day are fused into one slice before any conversion
    (odc-stac pastes them, first valid pixel wins), so they must share a rule:
    the key is (solar day, rule). When every day's scenes share a rule, which
    is the normal case, the slices are exactly those of ``groupby="solar_day"``;
    a day whose scenes have different rules becomes one slice per rule.
    odc-stac sorts the items by key and groups equal keys, so the i-th slice
    has the i-th smallest key.
    """
    from eo_brain.core.scenes import solar_day

    keys = {item.id: (solar_day(item, lon), scene_rules[item.id].key) for item in items}

    def group(item, parsed=None, idx=None):
        return keys[item.id]

    by_key = {r.key: r for r in scene_rules.values()}
    return group, [by_key[k[1]] for k in sorted(set(keys.values()))]


def attach_radiometry(ds, slice_rules: Sequence[Radiometry]):
    """Each time slice's rule as coordinates along ``time`` (read by ``band_to_reflectance``)."""
    if "time" not in ds.dims or ds.sizes["time"] != len(slice_rules):
        raise ValueError(f"{len(slice_rules)} radiometry slices for a dataset with dims {dict(ds.sizes)}")
    return ds.assign_coords({
        RULE: ("time", [r.rule for r in slice_rules]),
        SCALE: ("time", np.array([r.scale for r in slice_rules], dtype=np.float64)),
        OFFSET: ("time", np.array([r.offset for r in slice_rules], dtype=np.float64)),
        NODATA: ("time", np.array([r.nodata for r in slice_rules], dtype=np.int64)),
    })


# ---------------------------------------------------------------- conversion

def to_reflectance(dn, scale, offset, nodata):
    """
    Surface reflectance (float32) from DNs: nodata (and NaN, e.g. cloud-masked)
    -> NaN; DN * scale + offset; values <= 0 raised to ``scale`` (one DN step).
    ``dn`` is a numpy array or an xarray DataArray (lazy stays lazy);
    ``scale``/``offset``/``nodata`` are scalars or arrays that broadcast
    against it (one value per time slice).

    Computed as (DN + offset / scale) * scale: the offset is applied in DN
    units (-1000 for Sentinel-2, exact in float32), so the same scene gives
    the same float32 reflectance whether its DNs carry the offset or not, and
    dark pixels do not lose precision to a float32 subtraction of 0.1.
    """
    scale64, offset64 = _f64(scale), _f64(offset)
    shift32 = (offset64 / scale64).astype(np.float32)
    scale32 = scale64.astype(np.float32)
    values = dn.astype(np.float32) if hasattr(dn, "astype") else np.asarray(dn, np.float32)
    valid = (values != nodata) & ~_isnan(values)
    refl = (values + shift32) * scale32
    refl = _where(refl > 0, refl, scale32)
    return _where(valid, refl, np.float32(np.nan))


def band_to_reflectance(band):
    """
    ``to_reflectance`` with the rule the loader attached to ``band`` (coordinates
    ``radiometry_scale``/``_offset``/``_nodata`` along time). A band with no
    declared rule is refused: guessing a scale is what this module replaces.
    """
    missing = [c for c in (SCALE, OFFSET, NODATA) if c not in band.coords]
    if missing:
        raise RadiometryUndetermined(
            f"band {band.name!r} has no declared radiometry (coordinates {', '.join(missing)} missing): "
            "load it through DataCube.load_items, or mark it as reflectance "
            f"(attrs['units'] = '{REFLECTANCE_UNITS}')"
        )
    out = to_reflectance(band, band.coords[SCALE], band.coords[OFFSET], band.coords[NODATA])
    out.attrs = {**band.attrs, "units": REFLECTANCE_UNITS}
    out.attrs.pop("nodata", None)
    return out


def _f64(v):
    if hasattr(v, "astype"):
        return v.astype(np.float64)
    return np.float64(v)


def _isnan(values):
    return values.isnull() if hasattr(values, "isnull") else np.isnan(values)


def _where(cond, a, b):
    if hasattr(cond, "dims"):
        import xarray as xr
        return xr.where(cond, a, b, keep_attrs=True)
    return np.where(cond, a, b).astype(np.float32)


# ---------------------------------------------------------------- audit block

def summarize(
    provider_name: str,
    scene_rules: Dict[str, Radiometry],
    slice_rules: Sequence[str],
    refused: Sequence[Dict[str, str]] = (),
    max_listed: int = 20,
) -> Dict[str, Any]:
    """
    The ``radiometry`` block stored with a job, next to ``coverage``.

    ``scene_rules``: rule of each scene loaded (by id); ``slice_rules``: the rule
    (``Radiometry.rule``) of each time slice; ``refused``: scenes left out
    because their rule could not be determined (``split_by_radiometry``).
    """
    by_rule: Dict[str, Dict[str, Any]] = {}
    for rad in scene_rules.values():
        entry = by_rule.setdefault(rad.rule, {**rad.as_dict(), "scenes": 0, "acquisition_slices": 0})
        entry["scenes"] += 1
    for rule in slice_rules:
        if rule in by_rule:
            by_rule[rule]["acquisition_slices"] += 1
    return {
        "provider": provider_name,
        "quantity": "surface reflectance (0-1)",
        "rules": sorted(by_rule.values(), key=lambda r: r["rule"]),
        "non_positive": NON_POSITIVE_POLICY,
        "asset_scale_offset_used": False,
        "scenes_refused": len(refused),
        "refused": [dict(r) for r in list(refused)[:max_listed]],
    }


def refused_note(refused: Sequence[Dict[str, str]]) -> Optional[str]:
    """One sentence for the job message when scenes were left out, or None."""
    if not refused:
        return None
    n = len(refused)
    scenes = "scene was" if n == 1 else "scenes were"
    return (f"{n} {scenes} left out because how to convert their pixel values to reflectance "
            f"could not be determined from their metadata ({refused[0]['reason']}).")


def no_usable_scene_message(refused: Sequence[Dict[str, str]]) -> str:
    return (f"No usable scene: the radiometry of all {len(refused)} scenes in the window could not be "
            f"determined ({refused[0]['reason']}). Try another time window or provider.")
