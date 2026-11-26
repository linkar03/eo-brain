"""
tests/conftest.py - Shared Fixtures

Provides reusable test fixtures for all test modules.
"""

import pytest
import numpy as np
import xarray as xr
import dask.array as da
from datetime import datetime
from unittest.mock import MagicMock, patch


@pytest.fixture
def sample_bbox():
    """Standard test bounding box (Rome area)."""
    from eo_brain.core.datacube import BBox
    return BBox(west=12.4, south=41.8, east=12.6, north=42.0)


@pytest.fixture
def cross_zone_bbox():
    """BBox that crosses UTM zones 32-33."""
    from eo_brain.core.datacube import BBox
    return BBox(west=11.0, south=41.0, east=13.0, north=42.0)


@pytest.fixture
def sample_time_range():
    """Standard test time range."""
    from eo_brain.core.datacube import TimeRange
    return TimeRange(
        start=datetime(2024, 5, 1),
        end=datetime(2024, 9, 30)
    )


@pytest.fixture
def numpy_dataset():
    """
    Small numpy-backed dataset for fast unit tests.
    Contains red, nir, scl bands.
    """
    np.random.seed(42)
    size = (50, 50)
    
    return xr.Dataset({
        "red": xr.DataArray(
            np.random.rand(*size).astype(np.float32) * 0.3,
            dims=["y", "x"]
        ),
        "nir": xr.DataArray(
            np.random.rand(*size).astype(np.float32) * 0.5 + 0.3,
            dims=["y", "x"]
        ),
        "scl": xr.DataArray(
            np.random.choice([4, 5, 6, 8, 9], size=size).astype(np.uint8),
            dims=["y", "x"]
        ),
    })


@pytest.fixture
def dask_dataset():
    """
    Chunked dask-backed dataset for testing lazy execution.
    """
    np.random.seed(42)
    size = (100, 100)
    chunks = (50, 50)
    
    red = da.from_array(
        np.random.rand(*size).astype(np.float32) * 0.3,
        chunks=chunks
    )
    nir = da.from_array(
        np.random.rand(*size).astype(np.float32) * 0.5 + 0.3,
        chunks=chunks
    )
    scl = da.from_array(
        np.random.choice([4, 5, 6, 8, 9], size=size).astype(np.uint8),
        chunks=chunks
    )
    
    return xr.Dataset({
        "red": xr.DataArray(red, dims=["y", "x"]),
        "nir": xr.DataArray(nir, dims=["y", "x"]),
        "scl": xr.DataArray(scl, dims=["y", "x"]),
    })


@pytest.fixture
def ndvi_dataset():
    """Dataset with pre-computed NDVI (for algorithm tests)."""
    np.random.seed(42)
    size = (50, 50)
    
    # Generate realistic NDVI pattern with some high-value anomalies
    base = np.random.rand(*size).astype(np.float32) * 0.3 + 0.3  # 0.3-0.6
    
    # Add some "irrigated" high-NDVI spots
    base[20:25, 20:25] = 0.85  # High NDVI cluster
    base[40:42, 10:12] = 0.80  # Isolated high spot
    
    return xr.Dataset({
        "ndvi": xr.DataArray(base, dims=["y", "x"])
    })


@pytest.fixture
def time_series_dataset():
    """Dataset with time dimension for temporal analysis."""
    np.random.seed(42)
    
    # 5 time steps, 50x50 spatial
    times = [datetime(2024, m, 15) for m in range(5, 10)]
    data = np.random.rand(5, 50, 50).astype(np.float32) * 0.4 + 0.3
    
    return xr.Dataset({
        "ndvi": xr.DataArray(
            data,
            dims=["time", "y", "x"],
            coords={"time": times}
        )
    })


@pytest.fixture
def mock_stac_items():
    """Mock STAC items for search tests."""
    items = []
    for i in range(5):
        item = MagicMock()
        item.id = f"S2A_TEST_{i}"
        item.datetime = datetime(2024, 6, 15 + i)
        item.bbox = [12.4, 41.8, 12.6, 42.0]
        item.properties = {"eo:cloud_cover": 5.0 + i}
        item.assets = {
            "red": MagicMock(href="https://example.com/red.tif"),
            "nir": MagicMock(href="https://example.com/nir.tif"),
            "scl": MagicMock(href="https://example.com/scl.tif"),
        }
        items.append(item)
    return items


@pytest.fixture
def mock_datacube(mock_stac_items, numpy_dataset):
    """
    Mocked DataCube that returns test data without network calls.
    """
    from eo_brain.core.datacube import DataCube, LoadResult, Element84Sentinel2
    
    with patch.object(DataCube, 'search', return_value=mock_stac_items):
        with patch('odc.stac.load', return_value=numpy_dataset):
            cube = DataCube(provider="element84-s2")
            yield cube


# Pytest configuration
def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests requiring external services"
    )
    config.addinivalue_line(
        "markers", "db: needs a PostGIS superuser URL in TEST_DATABASE_URL (skipped otherwise)"
    )


# Skip slow tests by default in CI
def pytest_collection_modifyitems(config, items):
    if config.getoption("-m"):
        return  # User specified markers, respect their choice
    
    skip_slow = pytest.mark.skip(reason="Slow test - run with -m slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)