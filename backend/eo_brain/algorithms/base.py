"""
eo_brain/algorithms/base.py - EO Brain Analysis Algorithms

All index computations auto-normalize input bands to [0,1] reflectance,
handling Sentinel-2 scaled integers transparently. Operations remain
lazy (Dask-compatible) until the caller triggers .compute().
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import IntEnum
import logging

import xarray as xr
import numpy as np
import dask.array as da
from scipy import ndimage

logger = logging.getLogger(__name__)


class Severity(IntEnum):
    NONE = 0
    WATCH = 1
    SUSPECT = 2
    ALERT = 3


class MoranQuadrant(IntEnum):
    NONE = 0
    HH = 1  # cluster: high surrounded by high
    LL = 2  # cluster: low surrounded by low
    HL = 3  # outlier: high surrounded by low
    LH = 4  # outlier: low surrounded by high


@dataclass(frozen=True)
class LISAConfig:
    neighborhood_m: int = 150
    resolution_m: int = 10
    bio_gate: float = 0.3
    z_watch: float = 0.8
    z_suspect: float = 1.0
    z_alert: float = 1.5
    z_min_significance: float = 0.5

    def __post_init__(self):
        if not (self.z_watch < self.z_suspect < self.z_alert):
            raise ValueError("Thresholds must be ordered: z_watch < z_suspect < z_alert")
        if not 0 <= self.bio_gate <= 1:
            raise ValueError("bio_gate must be in [0, 1]")


@dataclass(frozen=True)
class FloodConfig:
    water_threshold: float = 0.0
    flood_threshold: float = 0.2
    severe_flood: float = 0.35

    def __post_init__(self):
        if not (self.water_threshold < self.flood_threshold < self.severe_flood):
            raise ValueError("Thresholds must be ordered: water < flood < severe")


@dataclass
class AlgorithmResult:
    data: xr.Dataset
    metadata: Dict[str, Any] = field(default_factory=dict)


class Algorithm(ABC):
    """Base class with shared lazy operations for spectral indices."""

    name: str = "base"
    description: str = ""
    required_bands: List[str] = []

    @abstractmethod
    def run(self, ds: xr.Dataset, **params) -> AlgorithmResult:
        pass

    @staticmethod
    def _to_reflectance(band: xr.DataArray) -> xr.DataArray:
        """
        Normalize to float32 reflectance [0, 1].
        Sentinel-2 L2A from Element84 is stored as uint16 (DN * 10000).
        Dtype check is free and keeps the Dask graph lazy.
        """
        if np.issubdtype(band.dtype, np.integer):
            return (band.astype(np.float32) / 10000.0).clip(0, 1)
        return band.astype(np.float32)

    @staticmethod
    def _ensure_2d(arr: xr.DataArray, method: str = "median") -> xr.DataArray:
        if "time" not in arr.dims:
            return arr

        reducers = {"median": "median", "mean": "mean", "max": "max"}
        if method not in reducers:
            raise ValueError(f"Unknown reduction: {method}")
        return getattr(arr, reducers[method])(dim="time", skipna=True)

    @staticmethod
    def _safe_normalized_diff(a: xr.DataArray, b: xr.DataArray) -> xr.DataArray:
        """
        (a - b) / (a + b) with NaN where denominator is near-zero.
        Shared by NDVI, NDWI, MNDWI — avoids repeating the guard logic.
        """
        denom = a + b
        result = xr.where(denom > 1e-10, (a - b) / denom, np.nan)
        return result.clip(-1, 1)

    @classmethod
    def _compute_ndvi(cls, ds: xr.Dataset) -> xr.DataArray:
        if "ndvi" in ds:
            return ds["ndvi"]
        if "nir" not in ds or "red" not in ds:
            raise ValueError(f"NDVI requires 'nir' and 'red'. Available: {list(ds.data_vars)}")

        nir = cls._to_reflectance(ds["nir"])
        red = cls._to_reflectance(ds["red"])
        return cls._safe_normalized_diff(nir, red)

    @classmethod
    def _compute_ndwi(cls, ds: xr.Dataset) -> xr.DataArray:
        if "ndwi" in ds:
            return ds["ndwi"]
        if "green" not in ds or "nir" not in ds:
            raise ValueError(f"NDWI requires 'green' and 'nir'. Available: {list(ds.data_vars)}")

        green = cls._to_reflectance(ds["green"])
        nir = cls._to_reflectance(ds["nir"])
        return cls._safe_normalized_diff(green, nir)

    @classmethod
    def _compute_mndwi(cls, ds: xr.Dataset) -> Optional[xr.DataArray]:
        if "mndwi" in ds:
            return ds["mndwi"]
        if "green" not in ds:
            return None

        swir = None
        for band_name in ("swir16", "swir22", "swir"):
            if band_name in ds:
                swir = ds[band_name]
                break
        if swir is None:
            return None

        green = cls._to_reflectance(ds["green"])
        swir = cls._to_reflectance(swir)
        return cls._safe_normalized_diff(green, swir)

    @staticmethod
    def _safe_convolve(arr: xr.DataArray, kernel: np.ndarray, depth: int) -> xr.DataArray:
        """Convolution with reflect padding to avoid edge artifacts."""
        if not isinstance(arr.data, da.Array):
            result = ndimage.convolve(arr.values, kernel, mode='reflect')
            return xr.DataArray(result, coords=arr.coords, dims=arr.dims)

        def _conv(block):
            return ndimage.convolve(block, kernel, mode='reflect')

        result = da.map_overlap(
            arr.data, _conv,
            depth={0: depth, 1: depth},
            boundary='reflect',
            dtype=arr.dtype,
            meta=np.array((), dtype=arr.dtype)
        )
        return xr.DataArray(result, coords=arr.coords, dims=arr.dims)


class LISAAlgorithm(Algorithm):
    """
    Local Indicators of Spatial Association (Anselin 1995).
    Detects spatial outliers (Q3=HL, Q4=LH) via Local Moran's I.

    Only vegetated pixels (NDVI >= bio_gate) enter the statistics: the global
    mean/std and each pixel's spatial lag are computed over vegetated pixels
    only, so an outlier means "unusual relative to the surrounding vegetation".
    Non-vegetated and nodata pixels get z_score/spatial_lag NaN and are never
    classified.
    """

    name = "lisa"
    description = "Spatial Outlier Detection (Local Moran's I)"
    required_bands = ["red", "nir"]

    def run(self, ds: xr.Dataset, **kwargs) -> AlgorithmResult:
        cfg = LISAConfig(
            neighborhood_m=kwargs.get('neighborhood_m', LISAConfig.neighborhood_m),
            resolution_m=kwargs.get('resolution_m', LISAConfig.resolution_m),
            bio_gate=kwargs.get('bio_gate', LISAConfig.bio_gate),
            z_watch=kwargs.get('z_watch', LISAConfig.z_watch),
            z_suspect=kwargs.get('z_suspect', LISAConfig.z_suspect),
            z_alert=kwargs.get('z_alert', LISAConfig.z_alert),
        )

        ndvi = self._compute_ndvi(ds)
        ndvi = self._ensure_2d(ndvi, method="median")

        # Bio gate FIRST: only vegetated pixels (NDVI >= bio_gate) take part.
        # A non-vegetated pixel (bare soil, water, built-up; also NaN) is
        # neither an outlier nor part of a vegetated pixel's context. Gating
        # only the output, after the statistics, let bare ground pull the
        # global mean down and fill the neighbourhood, so a vegetated pixel
        # surrounded by bare ground came out as a "high outlier".
        veg = ndvi >= cfg.bio_gate
        ndvi_veg = ndvi.where(veg)

        global_mean = ndvi_veg.mean(skipna=True)
        global_std = ndvi_veg.std(skipna=True)

        # Floor std to prevent division-by-zero in homogeneous areas
        std_safe = xr.where(global_std > 0.01, global_std, 0.01)
        z_score = (ndvi_veg - global_mean) / std_safe

        # Excluded pixels contribute 0 to the convolution sums...
        z_clean = z_score.fillna(0)

        neighborhood_px = max(3, cfg.neighborhood_m // cfg.resolution_m)
        kernel_size = neighborhood_px * 2 + 1

        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
        kernel[neighborhood_px, neighborhood_px] = 0
        kernel /= kernel.sum()

        # ...and the lag is the mean z over the VEGETATED neighbours only
        # (normalised convolution). No vegetated neighbour -> no lag -> the
        # pixel is not classified.
        lag_sum = self._safe_convolve(z_clean, kernel, depth=neighborhood_px)
        veg_share = self._safe_convolve(veg.astype(np.float32), kernel, depth=neighborhood_px)
        spatial_lag = lag_sum / veg_share.where(veg_share > 1e-6)
        classified = veg & spatial_lag.notnull()

        # Moran quadrant classification
        z_pos = z_clean > 0
        lag_pos = spatial_lag > 0

        quadrant = xr.full_like(z_score, MoranQuadrant.NONE, dtype=np.int8)
        quadrant = xr.where(z_pos & lag_pos, MoranQuadrant.HH, quadrant)
        quadrant = xr.where(~z_pos & ~lag_pos, MoranQuadrant.LL, quadrant)
        quadrant = xr.where(z_pos & ~lag_pos, MoranQuadrant.HL, quadrant)
        quadrant = xr.where(~z_pos & lag_pos, MoranQuadrant.LH, quadrant)
        quadrant = xr.where(classified, quadrant, MoranQuadrant.NONE)

        is_outlier = (quadrant == MoranQuadrant.HL) | (quadrant == MoranQuadrant.LH)

        abs_z = np.abs(z_clean)
        severity = xr.full_like(z_score, Severity.NONE, dtype=np.int8)

        classifiable = classified & is_outlier & (abs_z >= cfg.z_min_significance)
        severity = xr.where(classifiable & (abs_z >= cfg.z_watch), Severity.WATCH, severity)
        severity = xr.where(classifiable & (abs_z >= cfg.z_suspect), Severity.SUSPECT, severity)
        severity = xr.where(classifiable & (abs_z >= cfg.z_alert), Severity.ALERT, severity)

        return AlgorithmResult(
            data=xr.Dataset({
                "ndvi": ndvi,
                "z_score": z_score,
                "spatial_lag": spatial_lag,
                "lisa_quadrant": quadrant,
                "lisa_severity": severity,
            }),
            metadata={
                "algorithm": self.name,
                "neighborhood_m": cfg.neighborhood_m,
                "neighborhood_px": neighborhood_px,
                "bio_gate": cfg.bio_gate,
                "thresholds": {
                    "z_watch": cfg.z_watch,
                    "z_suspect": cfg.z_suspect,
                    "z_alert": cfg.z_alert,
                }
            }
        )


class NDVIAlgorithm(Algorithm):
    name = "ndvi"
    description = "NDVI Composite Map"
    required_bands = ["red", "nir"]

    def run(self, ds: xr.Dataset, **kwargs) -> AlgorithmResult:
        method = kwargs.get("composite_method", "median")
        ndvi = self._compute_ndvi(ds)
        ndvi = self._ensure_2d(ndvi, method=method)
        return AlgorithmResult(
            data=xr.Dataset({"ndvi": ndvi}),
            metadata={"algorithm": self.name, "composite_method": method}
        )


class NDWIAlgorithm(Algorithm):
    name = "ndwi"
    description = "Water Index (NDWI)"
    required_bands = ["green", "nir"]

    def run(self, ds: xr.Dataset, **kwargs) -> AlgorithmResult:
        method = kwargs.get("composite_method", "median")
        ndwi = self._compute_ndwi(ds)
        ndwi = self._ensure_2d(ndwi, method=method)
        return AlgorithmResult(
            data=xr.Dataset({"ndwi": ndwi}),
            metadata={"algorithm": self.name, "composite_method": method}
        )


class MNDWIAlgorithm(Algorithm):
    name = "mndwi"
    description = "Modified Water Index (MNDWI)"
    required_bands = ["green", "swir16"]

    def run(self, ds: xr.Dataset, **kwargs) -> AlgorithmResult:
        method = kwargs.get("composite_method", "median")
        mndwi = self._compute_mndwi(ds)
        if mndwi is None:
            raise ValueError("MNDWI requires SWIR band (swir16/swir22)")
        mndwi = self._ensure_2d(mndwi, method=method)
        return AlgorithmResult(
            data=xr.Dataset({"mndwi": mndwi}),
            metadata={"algorithm": self.name, "composite_method": method}
        )


class FloodDetectionAlgorithm(Algorithm):
    """
    Flood extent mapping. Prefers MNDWI (better urban discrimination),
    falls back to NDWI when SWIR unavailable.
    """

    name = "flood"
    description = "Flood Detection with Severity Classification"
    required_bands = ["green", "nir"]

    def run(self, ds: xr.Dataset, **kwargs) -> AlgorithmResult:
        cfg = FloodConfig(
            water_threshold=kwargs.get('water_threshold', FloodConfig.water_threshold),
            flood_threshold=kwargs.get('flood_threshold', FloodConfig.flood_threshold),
            severe_flood=kwargs.get('severe_flood', FloodConfig.severe_flood),
        )

        mndwi = self._compute_mndwi(ds)
        has_mndwi = mndwi is not None

        if has_mndwi:
            water_index = self._ensure_2d(mndwi, method="max")
            index_name = "mndwi"
        else:
            logger.info("SWIR unavailable, using NDWI for flood detection")
            ndwi = self._compute_ndwi(ds)
            water_index = self._ensure_2d(ndwi, method="max")
            index_name = "ndwi"

        severity = xr.full_like(water_index, Severity.NONE, dtype=np.int8)
        severity = xr.where(water_index > cfg.water_threshold, Severity.WATCH, severity)
        severity = xr.where(water_index > cfg.flood_threshold, Severity.SUSPECT, severity)
        severity = xr.where(water_index > cfg.severe_flood, Severity.ALERT, severity)

        # NaN pixels stay at NONE (no false positives from missing data)
        severity = xr.where(np.isnan(water_index), Severity.NONE, severity)

        water_mask = (water_index > cfg.water_threshold).astype(np.int8)
        water_mask = xr.where(np.isnan(water_index), 0, water_mask)

        result_vars = {
            "water_index": water_index,
            "flood_severity": severity,
            "water_mask": water_mask,
            index_name: water_index,
        }

        return AlgorithmResult(
            data=xr.Dataset(result_vars),
            metadata={
                "algorithm": self.name,
                "index_used": index_name,
                "has_swir": has_mndwi,
                "thresholds": {
                    "water": cfg.water_threshold,
                    "flood": cfg.flood_threshold,
                    "severe": cfg.severe_flood,
                }
            }
        )


ALGORITHMS: Dict[str, type] = {
    "lisa": LISAAlgorithm,
    "ndvi": NDVIAlgorithm,
    "ndwi": NDWIAlgorithm,
    "mndwi": MNDWIAlgorithm,
    "flood": FloodDetectionAlgorithm,
}


def get_algorithm(name: str) -> Algorithm:
    if name not in ALGORITHMS:
        available = ", ".join(ALGORITHMS.keys())
        raise ValueError(f"Unknown algorithm: '{name}'. Available: {available}")
    return ALGORITHMS[name]()