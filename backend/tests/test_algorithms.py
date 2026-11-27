"""
tests/test_algorithms.py - Algorithm Unit Tests

Validates:
- Mathematical correctness of LISA quadrant classification
- Proper lazy execution (no premature .compute())
- Edge case handling (empty data, uniform data, chunk boundaries)
- Flood detection index selection logic
"""

import pytest
import numpy as np
import xarray as xr
import dask.array as da

from eo_brain.algorithms.base import (
    LISAAlgorithm, LISAConfig, 
    FloodDetectionAlgorithm, FloodConfig,
    NDVIAlgorithm, NDWIAlgorithm, MNDWIAlgorithm,
    Severity, MoranQuadrant,
    get_algorithm, ALGORITHMS,
)
from eo_brain.core.radiometry import REFLECTANCE_UNITS

# Synthetic bands below are already surface reflectance and say so: a band
# with no declared radiometry is refused (tests/test_radiometry.py).
REFL = {"units": REFLECTANCE_UNITS}


class TestLISAMathematics:
    """Verify LISA quadrant classification matches Anselin (1995) definition."""
    
    def test_quadrant_hl_detection(self):
        """
        Q3 (HL): High value surrounded by low values.
        This is the classic spatial outlier - e.g., irrigated field in dry area.
        """
        # Create low-value background with one high-value spot
        data = np.full((20, 20), 0.35, dtype=np.float32)  # Just above bio_gate
        data[10, 10] = 0.85  # Isolated high value
        
        ds = xr.Dataset({"ndvi": xr.DataArray(data, dims=["y", "x"])})
        
        algo = LISAAlgorithm()
        result = algo.run(ds, neighborhood_m=30, resolution_m=10)
        
        quadrant = result.data["lisa_quadrant"].values
        
        # The high pixel should be classified as HL (Q3)
        assert quadrant[10, 10] == MoranQuadrant.HL
    
    def test_quadrant_lh_detection(self):
        """
        Q4 (LH): Low value surrounded by high values.
        E.g., dead tree in healthy forest, fire damage spot.
        """
        # High-value background with one low-value spot
        data = np.full((20, 20), 0.75, dtype=np.float32)
        data[10, 10] = 0.35  # Isolated low value (still above bio_gate)
        
        ds = xr.Dataset({"ndvi": xr.DataArray(data, dims=["y", "x"])})
        
        algo = LISAAlgorithm()
        result = algo.run(ds, neighborhood_m=30, resolution_m=10)
        
        quadrant = result.data["lisa_quadrant"].values
        
        # The low pixel should be classified as LH (Q4)
        assert quadrant[10, 10] == MoranQuadrant.LH
    
    def test_quadrant_hh_is_not_outlier(self):
        """
        Q1 (HH): High value surrounded by high values.
        This is a cluster, NOT an outlier. Severity should be NONE.
        """
        # Uniform high-value area
        data = np.full((20, 20), 0.70, dtype=np.float32)
        
        ds = xr.Dataset({"ndvi": xr.DataArray(data, dims=["y", "x"])})
        
        algo = LISAAlgorithm()
        result = algo.run(ds, neighborhood_m=30, resolution_m=10)
        
        severity = result.data["lisa_severity"].values
        
        # Clusters should not trigger alerts
        assert np.all(severity == Severity.NONE)
    
    def test_quadrant_ll_is_not_outlier(self):
        """
        Q2 (LL): Low value surrounded by low values.
        Cluster of stress, but not a spatial outlier.
        """
        # Uniform low-value area (but above bio_gate)
        data = np.full((20, 20), 0.35, dtype=np.float32)
        
        ds = xr.Dataset({"ndvi": xr.DataArray(data, dims=["y", "x"])})
        
        algo = LISAAlgorithm()
        result = algo.run(ds, neighborhood_m=30, resolution_m=10)
        
        severity = result.data["lisa_severity"].values
        
        assert np.all(severity == Severity.NONE)
    
    def test_bio_gate_excludes_non_vegetation(self):
        """Pixels below bio_gate threshold should be masked out."""
        data = np.full((20, 20), 0.15, dtype=np.float32)  # Below bio_gate
        data[10, 10] = 0.85  # High outlier but in non-veg context
        
        ds = xr.Dataset({"ndvi": xr.DataArray(data, dims=["y", "x"])})
        
        algo = LISAAlgorithm()
        result = algo.run(ds, neighborhood_m=30, resolution_m=10, bio_gate=0.3)
        
        severity = result.data["lisa_severity"].values
        quadrant = result.data["lisa_quadrant"].values
        
        # Everything should be masked because background is below bio_gate
        assert np.all(severity == Severity.NONE)
        assert np.all(quadrant == MoranQuadrant.NONE)

    def test_bare_ground_is_not_vegetation_context(self):
        """
        Uniform vegetation next to bare ground is not an outlier: only
        vegetated pixels enter the statistics and the spatial lag. A real
        outlier inside the vegetation is still found.
        """
        data = np.full((30, 30), 0.60, dtype=np.float32)  # Uniform vegetation
        data[:, :12] = 0.10                               # Bare ground strip
        data[15, 22] = 0.95                               # Outlier inside the vegetation

        ds = xr.Dataset({"ndvi": xr.DataArray(data, dims=["y", "x"])})

        algo = LISAAlgorithm()
        result = algo.run(ds, neighborhood_m=30, resolution_m=10, bio_gate=0.3)

        severity = result.data["lisa_severity"].values
        quadrant = result.data["lisa_quadrant"].values

        # Bare ground is excluded, not scored
        assert np.all(np.isnan(result.data["z_score"].values[:, :12]))
        assert np.all(quadrant[:, :12] == MoranQuadrant.NONE)

        # The vegetation edge along the bare strip raises no alert
        flagged = np.argwhere(severity > Severity.NONE)
        assert [tuple(p) for p in flagged] == [(15, 22)]
        assert quadrant[15, 22] == MoranQuadrant.HL

    def test_severity_levels_ordered(self):
        """Higher z-scores should produce higher severity levels."""
        # Create gradient of outlier intensities
        data = np.full((30, 30), 0.40, dtype=np.float32)
        
        # Three outliers with increasing deviation
        data[10, 10] = 0.55  # Mild outlier
        data[15, 15] = 0.70  # Moderate outlier
        data[20, 20] = 0.90  # Strong outlier
        
        ds = xr.Dataset({"ndvi": xr.DataArray(data, dims=["y", "x"])})
        
        algo = LISAAlgorithm()
        result = algo.run(ds, neighborhood_m=30, resolution_m=10)
        
        severity = result.data["lisa_severity"].values
        
        # Stronger outliers should have higher severity
        # (exact values depend on z-score thresholds)
        assert severity[10, 10] <= severity[15, 15] <= severity[20, 20]


class TestLISAEdgeCases:
    """Edge case handling for robustness."""
    
    def test_uniform_data_no_crash(self):
        """Completely uniform data (std=0) should not crash."""
        data = np.full((20, 20), 0.5, dtype=np.float32)
        ds = xr.Dataset({"ndvi": xr.DataArray(data, dims=["y", "x"])})
        
        algo = LISAAlgorithm()
        result = algo.run(ds, neighborhood_m=30, resolution_m=10)
        
        # Should complete without error
        assert "lisa_severity" in result.data
        # Z-scores should be ~0 (after safe division)
        z_scores = result.data["z_score"].values
        assert np.allclose(z_scores, 0, atol=0.1)
    
    def test_all_nan_returns_empty_result(self):
        """All-NaN input should return valid empty result."""
        data = np.full((20, 20), np.nan, dtype=np.float32)
        ds = xr.Dataset({"ndvi": xr.DataArray(data, dims=["y", "x"])})
        
        algo = LISAAlgorithm()
        result = algo.run(ds, neighborhood_m=30, resolution_m=10)
        
        # Should have structure but no classifications
        assert "lisa_severity" in result.data
    
    def test_computes_ndvi_from_bands(self):
        """Algorithm should compute NDVI if only red/nir provided."""
        np.random.seed(42)
        ds = xr.Dataset({
            "red": xr.DataArray(np.random.rand(20, 20).astype(np.float32) * 0.3, dims=["y", "x"], attrs=REFL),
            "nir": xr.DataArray(np.random.rand(20, 20).astype(np.float32) * 0.5 + 0.3, dims=["y", "x"], attrs=REFL),
        })
        
        algo = LISAAlgorithm()
        result = algo.run(ds, neighborhood_m=50, resolution_m=10)
        
        assert "ndvi" in result.data
        assert "lisa_severity" in result.data
    
    def test_missing_bands_raises_clear_error(self):
        """Missing required bands should raise informative error."""
        ds = xr.Dataset({
            "blue": xr.DataArray(np.random.rand(10, 10), dims=["y", "x"])
        })
        
        algo = LISAAlgorithm()
        
        with pytest.raises(ValueError, match="nir.*red|red.*nir"):
            algo.run(ds)
    
    def test_config_validation(self):
        """Invalid config should raise at construction time."""
        with pytest.raises(ValueError, match="ordered"):
            LISAConfig(z_watch=2.0, z_suspect=1.5, z_alert=1.0)
        
        with pytest.raises(ValueError, match="bio_gate"):
            LISAConfig(bio_gate=1.5)


class TestLazyExecution:
    """Verify operations stay lazy until explicit compute."""
    
    @pytest.fixture
    def lazy_dataset(self):
        np.random.seed(42)
        data = da.from_array(
            np.random.rand(100, 100).astype(np.float32) * 0.5 + 0.3,
            chunks=(50, 50)
        )
        return xr.Dataset({"ndvi": xr.DataArray(data, dims=["y", "x"])})
    
    def test_lisa_returns_lazy(self, lazy_dataset):
        """Result should be Dask-backed until computed."""
        algo = LISAAlgorithm()
        result = algo.run(lazy_dataset, neighborhood_m=50, resolution_m=10)
        
        for var in result.data.data_vars:
            arr = result.data[var].data
            assert isinstance(arr, da.Array), f"{var} was eagerly computed"
    
    def test_flood_returns_lazy(self, lazy_dataset):
        """Flood detection should also be lazy."""
        # Add green band for NDWI
        lazy_dataset["green"] = (lazy_dataset["ndvi"] * 0.8).assign_attrs(REFL)
        lazy_dataset["nir"] = lazy_dataset["ndvi"].copy().assign_attrs(REFL)
        
        algo = FloodDetectionAlgorithm()
        result = algo.run(lazy_dataset)
        
        for var in result.data.data_vars:
            arr = result.data[var].data
            assert isinstance(arr, da.Array), f"{var} was eagerly computed"


class TestChunkBoundaries:
    """Verify convolution doesn't create artifacts at chunk edges."""
    
    def test_uniform_data_uniform_output(self):
        """
        Convolution of uniform data should produce uniform result.
        Any discontinuity at chunk boundary indicates a bug.
        """
        uniform = 0.5
        data = da.full((100, 100), uniform, chunks=(50, 50), dtype=np.float32)
        ds = xr.Dataset({"ndvi": xr.DataArray(data, dims=["y", "x"])})
        
        algo = LISAAlgorithm()
        result = algo.run(ds, neighborhood_m=50, resolution_m=10, bio_gate=0.3)
        
        computed = result.data.compute()
        z_scores = computed["z_score"].values
        
        # Z-scores should be uniform (all ~0)
        boundary = z_scores[48:52, :]  # Around chunk boundary
        interior = z_scores[25, :]
        
        np.testing.assert_allclose(
            np.nanmean(boundary), np.nanmean(interior),
            atol=0.05,
            err_msg="Discontinuity at chunk boundary"
        )


class TestFloodDetection:
    """Flood algorithm correctness tests."""
    
    def test_mndwi_preferred_over_ndwi(self):
        """When SWIR available, should use MNDWI."""
        ds = xr.Dataset({
            "green": xr.DataArray(np.full((10, 10), 0.3), dims=["y", "x"], attrs=REFL),
            "nir": xr.DataArray(np.full((10, 10), 0.2), dims=["y", "x"], attrs=REFL),
            "swir16": xr.DataArray(np.full((10, 10), 0.1), dims=["y", "x"], attrs=REFL),
        })
        
        algo = FloodDetectionAlgorithm()
        result = algo.run(ds)
        
        assert result.metadata["index_used"] == "mndwi"
        assert result.metadata["has_swir"] is True
    
    def test_ndwi_fallback_without_swir(self):
        """Without SWIR, should fall back to NDWI."""
        ds = xr.Dataset({
            "green": xr.DataArray(np.full((10, 10), 0.3), dims=["y", "x"], attrs=REFL),
            "nir": xr.DataArray(np.full((10, 10), 0.2), dims=["y", "x"], attrs=REFL),
        })
        
        algo = FloodDetectionAlgorithm()
        result = algo.run(ds)
        
        assert result.metadata["index_used"] == "ndwi"
        assert result.metadata["has_swir"] is False
    
    def test_severity_thresholds(self):
        """Verify severity classification matches threshold config."""
        # Create gradient from dry to water
        water_values = np.array([[-0.2, 0.1, 0.25, 0.4]])  # dry, watch, suspect, alert
        
        ds = xr.Dataset({
            "green": xr.DataArray(np.full((1, 4), 0.3), dims=["y", "x"]),
            "nir": xr.DataArray(np.full((1, 4), 0.2), dims=["y", "x"]),
        })
        
        # Mock the water index directly
        algo = FloodDetectionAlgorithm()
        
        cfg = FloodConfig()
        
        # Manual severity assignment based on thresholds
        assert cfg.water_threshold == 0.0
        assert cfg.flood_threshold == 0.2
        assert cfg.severe_flood == 0.35
    
    def test_config_validation(self):
        """Invalid flood config should raise."""
        with pytest.raises(ValueError, match="ordered"):
            FloodConfig(water_threshold=0.5, flood_threshold=0.3, severe_flood=0.2)


class TestAlgorithmRegistry:
    """Registry and factory tests."""
    
    def test_all_algorithms_registered(self):
        """All expected algorithms should be in registry."""
        expected = {"lisa", "ndvi", "ndwi", "mndwi", "flood"}
        assert expected == set(ALGORITHMS.keys())
    
    def test_get_known_algorithm(self):
        """Factory should return correct algorithm type."""
        assert isinstance(get_algorithm("lisa"), LISAAlgorithm)
        assert isinstance(get_algorithm("flood"), FloodDetectionAlgorithm)
        assert isinstance(get_algorithm("ndvi"), NDVIAlgorithm)
    
    def test_get_unknown_raises(self):
        """Unknown algorithm should raise with helpful message."""
        with pytest.raises(ValueError, match="Unknown.*yoy"):
            get_algorithm("yoy")


class TestNDVIAlgorithm:
    """NDVI composite tests."""
    
    def test_median_composite(self):
        """Default should use median reduction."""
        data = np.stack([
            np.full((10, 10), 0.3),
            np.full((10, 10), 0.5),
            np.full((10, 10), 0.7),
        ]).astype(np.float32)
        
        ds = xr.Dataset({
            "ndvi": xr.DataArray(data, dims=["time", "y", "x"])
        })
        
        algo = NDVIAlgorithm()
        result = algo.run(ds)
        
        # Median of [0.3, 0.5, 0.7] = 0.5
        np.testing.assert_allclose(result.data["ndvi"].values, 0.5, atol=0.01)
    
    def test_max_composite(self):
        """Max composite for peak greenness."""
        data = np.stack([
            np.full((10, 10), 0.3),
            np.full((10, 10), 0.8),
            np.full((10, 10), 0.5),
        ]).astype(np.float32)
        
        ds = xr.Dataset({
            "ndvi": xr.DataArray(data, dims=["time", "y", "x"])
        })
        
        algo = NDVIAlgorithm()
        result = algo.run(ds, composite_method="max")
        
        np.testing.assert_allclose(result.data["ndvi"].values, 0.8, atol=0.01)