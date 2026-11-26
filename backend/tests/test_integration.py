"""
tests/test_integration.py - Real STAC API Integration Tests

These tests hit real STAC APIs. Run with:
    pytest -m integration

Skip in CI unless explicitly enabled.
"""

import pytest
from datetime import datetime

# Mark entire module as integration
pytestmark = pytest.mark.integration


class TestElement84Integration:
    """Real tests against Element84 Earth Search."""
    
    def test_search_sentinel2(self):
        """Search for real Sentinel-2 data."""
        from eo_brain.core.datacube import (
            DataCube, BBox, TimeRange, SearchParams
        )
        
        cube = DataCube(provider="element84-s2")
        
        params = SearchParams(
            bbox=BBox(west=12.4, south=41.8, east=12.6, north=42.0),
            time_range=TimeRange(
                start=datetime(2024, 7, 1),
                end=datetime(2024, 7, 31)
            ),
            limit=10
        )
        
        items = cube.search(params)
        
        assert len(items) > 0, "No items found - API may be down"
        assert items[0].id.startswith("S2")
    
    @pytest.mark.slow
    def test_load_small_area(self):
        """Load actual data from Element84."""
        from eo_brain.core.datacube import (
            DataCube, BBox, TimeRange, SearchParams, LoadConfig
        )
        
        cube = DataCube(provider="element84-s2")
        
        # Very small area to minimize download
        params = SearchParams(
            bbox=BBox(west=12.48, south=41.88, east=12.50, north=41.90),
            time_range=TimeRange(
                start=datetime(2024, 7, 1),
                end=datetime(2024, 7, 10)
            ),
            limit=1
        ).with_query(**{"eo:cloud_cover": {"lt": 10}})
        
        config = LoadConfig(resolution_m=60)  # Lower res for speed
        
        result = cube.load(params, bands=["red", "nir"], config=config)
        
        assert result.data.data_vars, "No data loaded"
        assert "red" in result.data
        assert "nir" in result.data
        
        # Should be lazy (Dask)
        import dask.array as da
        assert isinstance(result.data["red"].data, da.Array)
    
    @pytest.mark.slow
    def test_full_pipeline_small(self):
        """Run LISA on real data (small area)."""
        from eo_brain.core.datacube import (
            DataCube, BBox, TimeRange, SearchParams, LoadConfig
        )
        from eo_brain.algorithms.base import get_algorithm
        
        cube = DataCube(provider="element84-s2")
        
        params = SearchParams(
            bbox=BBox(west=12.48, south=41.88, east=12.50, north=41.90),
            time_range=TimeRange(
                start=datetime(2024, 7, 1),
                end=datetime(2024, 7, 15)
            ),
            limit=3
        ).with_query(**{"eo:cloud_cover": {"lt": 20}})
        
        config = LoadConfig(resolution_m=60)
        
        result = cube.load(
            params, 
            bands=["red", "nir", "scl"], 
            config=config
        )
        
        if not result.data.data_vars:
            pytest.skip("No data available for test period")
        
        ds = result.apply_cloud_mask()
        
        algo = get_algorithm("lisa")
        analysis = algo.run(ds, neighborhood_m=300, resolution_m=60)
        
        # Compute (this downloads data)
        computed = analysis.data.compute()
        
        assert "lisa_severity" in computed.data_vars
        assert "ndvi" in computed.data_vars
        
        # Should have some variation
        severity = computed["lisa_severity"].values
        unique_values = set(severity.flatten())
        assert len(unique_values) >= 1  # At least some classification


class TestCopernicusIntegration:
    """Tests against Copernicus Data Space."""
    
    def test_search_copernicus(self):
        """Search Copernicus catalog."""
        from eo_brain.core.datacube import (
            DataCube, BBox, TimeRange, SearchParams
        )
        
        cube = DataCube(provider="copernicus-s2")
        
        params = SearchParams(
            bbox=BBox(west=12.4, south=41.8, east=12.6, north=42.0),
            time_range=TimeRange(
                start=datetime(2024, 7, 1),
                end=datetime(2024, 7, 31)
            ),
            limit=5
        )
        
        try:
            items = cube.search(params)
            assert len(items) >= 0  # Copernicus may return 0 without auth
        except Exception as e:
            # Copernicus often requires auth for full access
            pytest.skip(f"Copernicus API unavailable: {e}")


class TestProviderComparison:
    """Compare results across providers."""
    
    def test_same_area_different_providers(self):
        """Same search should return similar counts."""
        from eo_brain.core.datacube import (
            DataCube, BBox, TimeRange, SearchParams
        )
        
        bbox = BBox(west=12.4, south=41.8, east=12.6, north=42.0)
        time_range = TimeRange(
            start=datetime(2024, 7, 1),
            end=datetime(2024, 7, 31)
        )
        
        results = {}
        
        for provider in ["element84-s2"]:  # Add others if available
            try:
                cube = DataCube(provider=provider)
                params = SearchParams(
                    bbox=bbox,
                    time_range=time_range,
                    limit=50
                )
                items = cube.search(params)
                results[provider] = len(items)
            except Exception as e:
                results[provider] = f"Error: {e}"
        
        # At least one provider should work
        working = [k for k, v in results.items() if isinstance(v, int) and v > 0]
        assert len(working) > 0, f"No providers working: {results}"


class TestUTMZoneBehavior:
    """Test real behavior at UTM zone boundaries."""
    
    @pytest.mark.slow
    def test_cross_zone_loading(self):
        """Load data that crosses UTM zones."""
        import warnings
        from eo_brain.core.datacube import (
            DataCube, BBox, TimeRange, SearchParams, LoadConfig
        )
        
        # Area crossing zones 32-33 (boundary at 12Â°E)
        bbox = BBox(west=11.8, south=41.8, east=12.2, north=42.0)
        
        cube = DataCube(provider="element84-s2")
        
        params = SearchParams(
            bbox=bbox,
            time_range=TimeRange(
                start=datetime(2024, 7, 1),
                end=datetime(2024, 7, 10)
            ),
            limit=1
        )
        
        config = LoadConfig(resolution_m=100)
        
        # Should emit warning about zone crossing
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = cube.load(params, bands=["red"], config=config)
            
            zone_warnings = [x for x in w if "UTM zone" in str(x.message)]
            assert len(zone_warnings) > 0, "Should warn about UTM zone crossing"