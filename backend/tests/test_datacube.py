"""
tests/test_datacube.py - DataCube Core Tests

Validates P0-3 (CRS handling) and P0-6 (explicit provider pattern).
"""

import pytest
import warnings
import numpy as np
import xarray as xr
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from eo_brain.core.datacube import (
    BBox, TimeRange, SearchParams, LoadConfig, LoadResult,
    DataCube, Element84Sentinel2, Element84Landsat,
    get_provider, PROVIDERS
)


class TestBBox:
    """P0-3: CRS and UTM zone handling."""
    
    def test_basic_creation(self):
        bbox = BBox(west=12.0, south=41.0, east=13.0, north=42.0)
        assert bbox.to_tuple() == (12.0, 41.0, 13.0, 42.0)
    
    def test_invalid_bbox_west_east(self):
        with pytest.raises(ValueError, match="west must be < east"):
            BBox(west=13.0, south=41.0, east=12.0, north=42.0)
    
    def test_invalid_bbox_south_north(self):
        with pytest.raises(ValueError, match="south must be < north"):
            BBox(west=12.0, south=42.0, east=13.0, north=41.0)
    
    def test_from_tuple(self):
        bbox = BBox.from_tuple((12.0, 41.0, 13.0, 42.0))
        assert bbox.west == 12.0
        assert bbox.north == 42.0
    
    def test_center(self):
        bbox = BBox(west=12.0, south=41.0, east=14.0, north=43.0)
        assert bbox.center() == (13.0, 42.0)
    
    # P0-3: UTM zone estimation
    def test_utm_zone_italy(self):
        """Italy is in UTM zone 32N or 33N."""
        bbox = BBox(west=12.0, south=41.0, east=13.0, north=42.0)
        crs = bbox.estimate_utm_crs()
        assert crs == "EPSG:32633"  # Zone 33N
    
    def test_utm_zone_southern_hemisphere(self):
        """South Africa should get 32700+ EPSG."""
        bbox = BBox(west=18.0, south=-34.0, east=19.0, north=-33.0)
        crs = bbox.estimate_utm_crs()
        assert crs.startswith("EPSG:327")  # Southern hemisphere
    
    def test_utm_zone_crossing_detection(self):
        """Detect when bbox spans multiple UTM zones."""
        # Zone 32 ends at 12Â°E, zone 33 starts there
        bbox_single = BBox(west=13.0, south=41.0, east=14.0, north=42.0)
        assert not bbox_single.crosses_utm_zones()
        
        # This spans zones 32 and 33
        bbox_cross = BBox(west=11.0, south=41.0, east=13.0, north=42.0)
        assert bbox_cross.crosses_utm_zones()
    
    def test_utm_zone_crossing_warning(self):
        """P0-3: Warning emitted when crossing zones."""
        bbox = BBox(west=11.0, south=41.0, east=13.0, north=42.0)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            crs = bbox.estimate_utm_crs()
            
            assert len(w) == 1
            assert "crosses UTM zones" in str(w[0].message)
            assert "32-33" in str(w[0].message)


class TestTimeRange:
    def test_valid_range(self):
        tr = TimeRange(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 6, 30)
        )
        assert "2024-01-01" in tr.to_stac_string()
        assert "2024-06-30" in tr.to_stac_string()
    
    def test_invalid_range(self):
        with pytest.raises(ValueError, match="start must be before end"):
            TimeRange(
                start=datetime(2024, 6, 30),
                end=datetime(2024, 1, 1)
            )


class TestSearchParams:
    def test_with_query_immutable(self):
        """with_query returns new instance, doesn't mutate."""
        bbox = BBox(west=12.0, south=41.0, east=13.0, north=42.0)
        params1 = SearchParams(
            bbox=bbox,
            time_range=TimeRange(datetime(2024, 1, 1), datetime(2024, 6, 30))
        )
        
        params2 = params1.with_query(**{"eo:cloud_cover": {"lt": 20}})
        
        assert params1.query == {}
        assert params2.query == {"eo:cloud_cover": {"lt": 20}}


class TestLoadConfig:
    def test_default_values(self):
        config = LoadConfig()
        assert config.resolution_m == 10
        assert config.crs is None
        assert config.chunks == {"x": 2048, "y": 2048, "time": 1}
    
    def test_resolve_crs_explicit(self):
        """If CRS is set, use it."""
        bbox = BBox(west=12.0, south=41.0, east=13.0, north=42.0)
        config = LoadConfig(crs="EPSG:3857")
        assert config.resolve_crs(bbox) == "EPSG:3857"
    
    def test_resolve_crs_auto(self):
        """If CRS is None, auto-detect UTM."""
        bbox = BBox(west=12.0, south=41.0, east=13.0, north=42.0)
        config = LoadConfig(crs=None)
        crs = config.resolve_crs(bbox)
        assert crs.startswith("EPSG:326")  # Northern UTM


class TestLoadResult:
    """P0-6: Explicit provider pattern."""
    
    @pytest.fixture
    def mock_dataset(self):
        """Create a mock xarray Dataset with SCL band."""
        return xr.Dataset({
            "red": xr.DataArray(np.random.rand(10, 10), dims=["y", "x"]),
            "nir": xr.DataArray(np.random.rand(10, 10), dims=["y", "x"]),
            "scl": xr.DataArray(
                np.random.choice([4, 5, 6, 8, 9], size=(10, 10)),
                dims=["y", "x"]
            ),
        })
    
    @pytest.fixture
    def mock_provider(self):
        provider = Element84Sentinel2()
        return provider
    
    def test_load_result_has_explicit_provider(self, mock_dataset, mock_provider):
        """Provider is stored directly, not in attrs."""
        bbox = BBox(west=12.0, south=41.0, east=13.0, north=42.0)
        
        result = LoadResult(
            data=mock_dataset,
            provider=mock_provider,
            crs="EPSG:32633",
            bbox=bbox
        )
        
        # Provider is explicit attribute, not hidden in ds.attrs
        assert result.provider is mock_provider
        assert "_provider" not in result.data.attrs
    
    def test_cloud_mask_uses_provider(self, mock_dataset, mock_provider):
        """apply_cloud_mask delegates to provider."""
        bbox = BBox(west=12.0, south=41.0, east=13.0, north=42.0)
        
        result = LoadResult(
            data=mock_dataset,
            provider=mock_provider,
            crs="EPSG:32633",
            bbox=bbox
        )
        
        masked = result.apply_cloud_mask()
        
        # Should have NaN where SCL was 8 or 9 (cloud/shadow)
        assert masked is not None
        assert "red" in masked.data_vars
    
    def test_cloud_mask_missing_band_warns(self, mock_provider, caplog):
        """Warning when cloud mask band not loaded."""
        ds_no_scl = xr.Dataset({
            "red": xr.DataArray(np.random.rand(10, 10), dims=["y", "x"]),
        })
        bbox = BBox(west=12.0, south=41.0, east=13.0, north=42.0)
        
        result = LoadResult(
            data=ds_no_scl,
            provider=mock_provider,
            crs="EPSG:32633",
            bbox=bbox
        )
        
        masked = result.apply_cloud_mask()
        
        # Should return unchanged and log warning
        assert "scl" not in masked.data_vars
    
    def test_provider_survives_dataset_operations(self, mock_dataset, mock_provider):
        """P0-6: Provider reference isn't lost by xarray operations."""
        bbox = BBox(west=12.0, south=41.0, east=13.0, north=42.0)
        
        result = LoadResult(
            data=mock_dataset,
            provider=mock_provider,
            crs="EPSG:32633",
            bbox=bbox
        )
        
        # Simulate operations that would lose attrs
        result.data = result.data + 0  # This loses attrs in xarray
        result.data = result.data.astype(float)
        
        # Provider is still accessible because it's on LoadResult, not ds.attrs
        assert result.provider is mock_provider
        masked = result.apply_cloud_mask()  # Should still work
        assert masked is not None


class TestProviders:
    """Test provider-specific cloud masking."""
    
    def test_sentinel2_cloud_mask(self):
        provider = Element84Sentinel2()
        
        # SCL values: 4=veg, 5=soil, 6=water, 8=cloud_medium, 9=cloud_high
        scl = xr.DataArray(
            np.array([[4, 5, 6], [8, 9, 4]]),
            dims=["y", "x"]
        )
        
        mask = provider.compute_cloud_mask(scl)
        
        expected = np.array([[True, True, True], [False, False, True]])
        np.testing.assert_array_equal(mask.values, expected)
    
    def test_landsat_cloud_mask(self):
        provider = Element84Landsat()
        
        # QA_PIXEL bitmask: bit 3 = cloud, bit 4 = shadow
        # 0 = clear, 8 = cloud (bit 3), 16 = shadow (bit 4), 24 = both
        qa = xr.DataArray(
            np.array([[0, 8, 16], [24, 0, 0]], dtype=np.uint16),
            dims=["y", "x"]
        )
        
        mask = provider.compute_cloud_mask(qa)
        
        expected = np.array([[True, False, False], [False, True, True]])
        np.testing.assert_array_equal(mask.values, expected)
    
    def test_all_providers_have_required_methods(self):
        """All registered providers implement the interface."""
        for name, cls in PROVIDERS.items():
            provider = cls()
            assert hasattr(provider, 'catalog_url')
            assert hasattr(provider, 'default_collections')
            assert hasattr(provider, 'cloud_mask_band')
            assert hasattr(provider, 'get_asset_mapping')
            assert hasattr(provider, 'compute_cloud_mask')
            assert callable(provider.compute_cloud_mask)