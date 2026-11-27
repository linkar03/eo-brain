"""
tests/test_vectorize.py - Vectorization Unit Tests

Validates:
- Polygon extraction from severity rasters
- Area calculation in correct CRS (meters, not degrees)
- Geometry simplification
- Small feature filtering
"""

import pytest
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import shape
from unittest.mock import patch, MagicMock
from uuid import uuid4

from eo_brain.algorithms.vectorize import (
    VectorizeConfig,
    VectorizeResult,
    _extract_polygons,
    _get_transformer,
)


class TestPolygonExtraction:
    """Test polygon extraction from rasters."""
    
    @pytest.fixture
    def sample_severity_raster(self):
        """Create a simple severity raster with known shapes."""
        # 100x100 raster at 10m resolution
        data = np.zeros((100, 100), dtype=np.uint8)
        
        # Add a severity=3 square (20x20 pixels = 200x200m = 40,000 m²)
        data[40:60, 40:60] = 3
        
        # Add a severity=2 rectangle (10x30 pixels = 100x300m = 30,000 m²)
        data[10:20, 10:40] = 2
        
        # Add small noise (2x2 pixels = 400 m², should be filtered)
        data[80:82, 80:82] = 1
        
        return data
    
    @pytest.fixture
    def utm_transform(self):
        """Affine transform for UTM coordinates (10m pixels)."""
        # Simulated UTM zone 33N, origin at (300000, 4600000)
        return from_bounds(300000, 4500000, 301000, 4501000, 100, 100)
    
    def test_extract_large_features(self, sample_severity_raster, utm_transform):
        """Features above min_area should be extracted."""
        config = VectorizeConfig(min_area_pixels=4, min_severity=1)
        
        features = list(_extract_polygons(
            sample_severity_raster,
            utm_transform,
            "EPSG:32633",
            config
        ))
        
        ok_features = [f for f in features if f["status"] == "ok"]
        
        # Should have 2 large features (sev=3 and sev=2), small one filtered
        assert len(ok_features) >= 2
        
        # Check severity values
        severities = {f["severity"] for f in ok_features}
        assert 3 in severities
        assert 2 in severities
    
    def test_filter_small_features(self, sample_severity_raster, utm_transform):
        """Features below min_area should be filtered."""
        # With 10m pixels, 4 pixels = 400 m²
        # The 2x2 noise region is exactly 400 m², so set threshold higher
        config = VectorizeConfig(min_area_pixels=5, min_severity=1)
        
        features = list(_extract_polygons(
            sample_severity_raster,
            utm_transform,
            "EPSG:32633",
            config
        ))
        
        small_count = sum(1 for f in features if f["status"] == "small")
        
        # The 2x2 region should be filtered
        assert small_count >= 1
    
    def test_area_in_meters(self, sample_severity_raster, utm_transform):
        """Area should be calculated in square meters (UTM), not degrees."""
        config = VectorizeConfig(min_area_pixels=4, min_severity=3)
        
        features = list(_extract_polygons(
            sample_severity_raster,
            utm_transform,
            "EPSG:32633",
            config
        ))
        
        ok_features = [f for f in features if f["status"] == "ok"]
        assert len(ok_features) >= 1
        
        # The 20x20 pixel square at 10m resolution = 40,000 m²
        # Allow some tolerance for simplification
        large_feature = max(ok_features, key=lambda f: f["area_m2"])
        assert 35000 < large_feature["area_m2"] < 45000, \
            f"Area should be ~40,000 m², got {large_feature['area_m2']}"
    
    def test_min_severity_filter(self, sample_severity_raster, utm_transform):
        """Only features >= min_severity should be extracted."""
        config = VectorizeConfig(min_area_pixels=1, min_severity=3)
        
        features = list(_extract_polygons(
            sample_severity_raster,
            utm_transform,
            "EPSG:32633",
            config
        ))
        
        ok_features = [f for f in features if f["status"] == "ok"]
        
        # Only severity=3 features should be extracted
        for f in ok_features:
            assert f["severity"] == 3


class TestCoordinateTransformation:
    """Test CRS transformations."""
    
    def test_transformer_creation(self):
        """Transformer should be created for valid CRS."""
        transform_fn = _get_transformer("EPSG:32633", "EPSG:4326")
        
        # Transform a point in UTM zone 33N
        x, y = transform_fn(500000, 4500000)
        
        # Should be in Italy region (roughly)
        assert 10 < x < 20, f"Longitude {x} out of expected range"
        assert 40 < y < 45, f"Latitude {y} out of expected range"
    
    def test_inverse_transform(self):
        """Round-trip transformation should preserve location."""
        to_wgs84 = _get_transformer("EPSG:32633", "EPSG:4326")
        to_utm = _get_transformer("EPSG:4326", "EPSG:32633")
        
        original = (500000, 4500000)
        wgs84 = to_wgs84(*original)
        back = to_utm(*wgs84)
        
        assert abs(back[0] - original[0]) < 0.01
        assert abs(back[1] - original[1]) < 0.01


class TestVectorizeConfig:
    """Test configuration validation."""
    
    def test_default_config(self):
        cfg = VectorizeConfig()
        assert cfg.min_area_pixels == 4
        assert cfg.simplify_tolerance_factor == 0.5
        assert cfg.min_severity == 1
    
    def test_custom_config(self):
        cfg = VectorizeConfig(
            min_area_pixels=10,
            simplify_tolerance_factor=0.8,
            min_severity=2
        )
        assert cfg.min_area_pixels == 10
        assert cfg.min_severity == 2


class TestVectorizeResult:
    """Test result dataclass."""
    
    def test_result_structure(self):
        result = VectorizeResult(
            feature_count=42,
            total_area_m2=150000.0,
            by_severity={1: 10, 2: 20, 3: 12},
            skipped_small=5,
            skipped_invalid=2
        )
        
        assert result.feature_count == 42
        assert result.total_area_m2 == 150000.0
        assert result.by_severity[3] == 12
        assert result.skipped_small == 5


class TestEdgeCases:
    """Edge case handling."""
    
    def test_empty_raster(self):
        """Raster with all zeros should produce no features."""
        data = np.zeros((50, 50), dtype=np.uint8)
        transform = from_bounds(0, 0, 500, 500, 50, 50)
        config = VectorizeConfig()
        
        features = list(_extract_polygons(data, transform, "EPSG:32633", config))
        
        ok_features = [f for f in features if f["status"] == "ok"]
        assert len(ok_features) == 0
    
    def test_single_pixel(self):
        """Single pixel should be filtered as too small."""
        data = np.zeros((50, 50), dtype=np.uint8)
        data[25, 25] = 3  # Single pixel
        
        transform = from_bounds(0, 0, 500, 500, 50, 50)
        config = VectorizeConfig(min_area_pixels=2)
        
        features = list(_extract_polygons(data, transform, "EPSG:32633", config))
        
        ok_features = [f for f in features if f["status"] == "ok"]
        assert len(ok_features) == 0
    
    def test_max_features_limit(self):
        """Should respect max_features limit."""
        # Create many small regions
        data = np.zeros((100, 100), dtype=np.uint8)
        for i in range(0, 100, 5):
            for j in range(0, 100, 5):
                data[i:i+3, j:j+3] = 3  # 3x3 squares
        
        transform = from_bounds(0, 0, 1000, 1000, 100, 100)
        config = VectorizeConfig(min_area_pixels=1, max_features=10)
        
        features = list(_extract_polygons(data, transform, "EPSG:32633", config))
        
        ok_features = [f for f in features if f["status"] == "ok"]
        assert len(ok_features) <= 10

class TestZoneClasses:
    """A zone is a 4-connected patch of one class, inside the area mask."""

    @pytest.fixture
    def transform(self):
        return from_bounds(300000, 4500000, 300100, 4500100, 10, 10)   # 10 m pixels, 100 m2

    def test_a_lisa_patch_of_both_quadrants_becomes_one_zone_per_quadrant(self, transform):
        severity = np.zeros((10, 10), dtype=np.int8)
        severity[2:6, 2:8] = 3                       # one severity-3 patch of 24 pixels...
        quadrant = np.zeros_like(severity)
        quadrant[2:6, 2:5] = 3                       # ...12 of them HL
        quadrant[2:6, 5:8] = 4                       # ...12 of them LH
        config = VectorizeConfig(min_area_pixels=4)
        ok = [f for f in _extract_polygons(severity, transform, "EPSG:32633", config, quadrant) if f["status"] == "ok"]
        assert sorted((f["severity"], f["quadrant"], round(f["area_m2"])) for f in ok) == [(3, 3, 1200), (3, 4, 1200)]
        # without the quadrant raster it is the single zone of the old behaviour, with no quadrant
        ok = [f for f in _extract_polygons(severity, transform, "EPSG:32633", config) if f["status"] == "ok"]
        assert [(f["severity"], f["quadrant"], round(f["area_m2"])) for f in ok] == [(3, None, 2400)]

    def test_the_area_mask_clips_zones_and_drops_those_outside(self, transform):
        severity = np.zeros((10, 10), dtype=np.int8)
        severity[0:4, 0:4] = 2                       # half outside the mask
        severity[6:9, 6:9] = 3                       # all outside
        inside = np.zeros((10, 10), dtype=bool)
        inside[2:, :6] = True
        out = list(_extract_polygons(severity, transform, "EPSG:32633", VectorizeConfig(min_area_pixels=4),
                                     area_mask=inside))
        assert [(f["status"], f["severity"], round(f["area_m2"])) for f in out] == [("ok", 2, 800)]

    def test_a_patch_of_the_minimum_size_is_kept_and_a_smaller_one_reported(self, transform):
        severity = np.zeros((10, 10), dtype=np.int8)
        severity[0:2, 0:2] = 1                       # exactly 4 pixels
        severity[5, 5:8] = 2                         # 3 pixels
        out = list(_extract_polygons(severity, transform, "EPSG:32633", VectorizeConfig(min_area_pixels=4)))
        assert sorted((f["status"], f["severity"]) for f in out) == [("ok", 1), ("small", 2)]
        [small] = [f for f in out if f["status"] == "small"]
        assert small["area"] == pytest.approx(300.0)

    def test_the_limit_is_reported(self):
        data = np.zeros((20, 20), dtype=np.uint8)
        data[::2, ::2] = 3                           # 100 single pixels
        out = list(_extract_polygons(data, from_bounds(0, 0, 200, 200, 20, 20), "EPSG:32633",
                                     VectorizeConfig(min_area_pixels=1, max_features=5)))
        assert sum(f["status"] == "ok" for f in out) == 5 and out[-1] == {"status": "limit"}


class TestVectorize:
    """vectorize_from_dataset (the same core as vectorize_cog) with the database write stubbed."""

    @pytest.fixture
    def inserted(self, monkeypatch):
        from eo_brain.algorithms import vectorize as vectorize_mod

        rows = []
        monkeypatch.setattr(vectorize_mod.vector_repo, "bulk_insert", lambda features: rows.extend(features))
        return rows

    @pytest.fixture
    def lisa(self):
        severity = np.zeros((12, 12), dtype=np.int8)
        quadrant = np.ones((12, 12), dtype=np.int8)        # HH background, no severity
        severity[1:4, 1:4], quadrant[1:4, 1:4] = 3, 3      # HL zone, 9 px
        severity[6:8, 6:9], quadrant[6:8, 6:9] = 1, 4      # LH zone, 6 px
        severity[10, 10], quadrant[10, 10] = 2, 4          # LH patch, 1 px: left out
        severity[0, 11], quadrant[0, 11] = 3, 3            # outside the mask
        inside = np.ones((12, 12), dtype=bool)
        inside[0, 11] = False
        return severity, quadrant, inside, from_bounds(300000, 4500000, 300120, 4500120, 12, 12)

    def test_zones_rows_and_stats(self, inserted, lisa):
        from eo_brain.algorithms.vectorize import vectorize_from_dataset

        severity, quadrant, inside, transform = lisa
        job = uuid4()
        result = vectorize_from_dataset(severity, quadrant, transform, "EPSG:32633", job, "lisa",
                                        VectorizeConfig(min_area_pixels=4), area_mask=inside)
        assert sorted((r["severity"], r["quadrant"], round(r["area_m2"])) for r in inserted) == [(1, 4, 600), (3, 3, 900)]
        assert {r["job_id"] for r in inserted} == {str(job)} and {r["algorithm"] for r in inserted} == {"lisa"}
        assert result.stats() == {
            "feature_count": 2, "total_area_m2": pytest.approx(1500.0), "by_severity": {1: 1, 2: 0, 3: 1},
            "min_zone_pixels": 4, "min_zone_area_m2": 400.0, "skipped_small": 1,
            "skipped_small_area_m2": pytest.approx(100.0), "skipped_invalid": 0, "max_features_reached": False,
        }
        # zones + small patches = every pixel of severity 1 to 3 inside the mask
        assert result.total_area_m2 + result.skipped_small_area_m2 == pytest.approx(((severity > 0) & inside).sum() * 100)

    def test_flood_zones_have_no_quadrant(self, inserted):
        from eo_brain.algorithms.vectorize import vectorize_from_dataset

        severity = np.zeros((10, 10), dtype=np.int8)
        severity[2:6, 2:6] = 2
        vectorize_from_dataset(severity, None, from_bounds(0, 0, 100, 100, 10, 10), "EPSG:32633", uuid4(), "flood")
        assert [r["quadrant"] for r in inserted] == [None]

    @pytest.mark.parametrize("case, message", [
        ("lisa without quadrants", "LISA zones need the Moran quadrant raster"),
        ("flood with quadrants", "flood has no Moran quadrant"),
        ("severity on a cluster", r"quadrants \[1, 3, 4\], expected \[3, 4\]"),
        ("misaligned mask", "area_mask .* differ in shape"),
    ])
    def test_impossible_inputs_are_refused_before_any_write(self, inserted, lisa, case, message):
        from eo_brain.algorithms.vectorize import vectorize_from_dataset

        severity, quadrant, inside, transform = lisa
        algorithm = "lisa"
        if case == "lisa without quadrants":
            quadrant = None
        elif case == "flood with quadrants":
            algorithm = "flood"
        elif case == "severity on a cluster":
            severity = severity.copy()
            severity[5, 0] = 2                       # an HH pixel with a severity
        else:
            inside = inside[:, :-1]
        with pytest.raises(ValueError, match=message):
            vectorize_from_dataset(severity, quadrant, transform, "EPSG:32633", uuid4(), algorithm,
                                   area_mask=inside)
        assert inserted == []
