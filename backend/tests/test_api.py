"""
tests/test_api.py - API Integration Tests

Validates:
- Request validation (bbox, dates, algorithms)
- Error handling and response codes
- Async execution doesn't block event loop
"""

import pytest
import asyncio
import time
from datetime import datetime
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

# Mock external dependencies before importing app
import sys
sys.modules['pystac_client'] = MagicMock()
sys.modules['odc.stac'] = MagicMock()

from eo_brain.api.main import app, settings


class TestAPIEndpoints:
    """Basic endpoint smoke tests."""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "service" in data
        assert "providers" in data
        assert "algorithms" in data
        assert "lisa" in data["algorithms"]
        assert "flood" in data["algorithms"]
        # YoY should NOT be present
        assert "yoy" not in data["algorithms"]
    
    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_list_providers(self, client):
        response = client.get("/providers")
        assert response.status_code == 200
        
        data = response.json()
        assert "element84-s2" in data
    
    def test_list_algorithms(self, client):
        response = client.get("/algorithms")
        assert response.status_code == 200
        
        data = response.json()
        assert "lisa" in data
        assert "flood" in data
        assert "ndvi" in data
        assert "ndwi" in data
        assert "mndwi" in data
        
        # Verify output type classification
        assert data["lisa"]["output_type"] == "discrete"
        assert data["flood"]["output_type"] == "discrete"
        assert data["ndvi"]["output_type"] == "continuous"
        assert data["ndwi"]["output_type"] == "continuous"


class TestRequestValidation:
    """Input validation tests."""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture
    def valid_request(self):
        return {
            "bbox": [12.0, 41.0, 12.5, 41.5],
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2024-06-30T00:00:00",
            "algorithm": "lisa"
        }
    
    def test_invalid_bbox_length(self, client):
        response = client.post("/analyze", json={
            "bbox": [12.0, 41.0, 13.0],  # Missing 4th element
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2024-06-30T00:00:00",
            "algorithm": "lisa"
        })
        assert response.status_code == 422
    
    def test_invalid_bbox_south_north(self, client):
        response = client.post("/analyze", json={
            "bbox": [12.0, 42.0, 13.0, 41.0],  # south > north
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2024-06-30T00:00:00",
            "algorithm": "lisa"
        })
        assert response.status_code == 422
        assert "south" in response.json()["detail"][0]["msg"].lower()
    
    def test_unknown_provider(self, client, valid_request):
        valid_request["provider"] = "nonexistent_provider"
        response = client.post("/analyze", json=valid_request)
        assert response.status_code == 400
        assert "Unknown provider" in response.json()["detail"]
    
    def test_unknown_algorithm(self, client, valid_request):
        valid_request["algorithm"] = "nonexistent_algo"
        response = client.post("/analyze", json=valid_request)
        assert response.status_code == 400
        assert "Unknown algorithm" in response.json()["detail"]
    
    def test_removed_algorithm_yoy(self, client, valid_request):
        """YoY algorithm should no longer exist."""
        valid_request["algorithm"] = "yoy"
        response = client.post("/analyze", json=valid_request)
        assert response.status_code == 400
        assert "Unknown algorithm" in response.json()["detail"]
    
    def test_area_too_large(self, client):
        response = client.post("/analyze", json={
            "bbox": [10.0, 40.0, 15.0, 45.0],  # ~200,000 km²
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2024-06-30T00:00:00",
            "algorithm": "lisa"
        })
        assert response.status_code == 422
        # Area validation happens in Pydantic
    
    def test_end_date_before_start_date(self, client):
        response = client.post("/analyze", json={
            "bbox": [12.0, 41.0, 12.5, 41.5],
            "start_date": "2024-06-30T00:00:00",
            "end_date": "2024-01-01T00:00:00",
            "algorithm": "lisa"
        })
        assert response.status_code == 422
        assert "end_date" in str(response.json()).lower()


class TestJobManagement:
    """Job CRUD operations."""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_invalid_job_id_format(self, client):
        response = client.get("/jobs/not-a-uuid")
        assert response.status_code == 400
        assert "Invalid job ID" in response.json()["detail"]
    
    def test_job_not_found(self, client):
        response = client.get("/jobs/00000000-0000-0000-0000-000000000000")
        assert response.status_code == 404
    
    def test_delete_invalid_job_id(self, client):
        response = client.delete("/jobs/invalid")
        assert response.status_code == 400


class TestAsyncExecution:
    """Verify async pattern doesn't block."""
    
    @pytest.mark.asyncio
    async def test_health_responsive_during_load(self):
        """
        Health endpoint should respond immediately even if
        analysis jobs are running.
        """
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Health check should be fast regardless of server load
            start = time.time()
            response = await client.get("/health")
            elapsed = time.time() - start
            
            assert response.status_code == 200
            assert elapsed < 0.5  # Should be nearly instant
    
    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self):
        """Multiple concurrent requests shouldn't block each other."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            start = time.time()
            
            tasks = [client.get("/health") for _ in range(10)]
            responses = await asyncio.gather(*tasks)
            
            elapsed = time.time() - start
            
            assert all(r.status_code == 200 for r in responses)
            # 10 sequential requests would take much longer
            assert elapsed < 2.0


class TestAlgorithmMetadata:
    """Verify algorithm info is correctly exposed."""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_lisa_metadata(self, client):
        response = client.get("/algorithms")
        data = response.json()
        
        lisa = data["lisa"]
        assert lisa["output_type"] == "discrete"
        assert "red" in lisa["required_bands"]
        assert "nir" in lisa["required_bands"]
    
    def test_flood_metadata(self, client):
        response = client.get("/algorithms")
        data = response.json()
        
        flood = data["flood"]
        assert flood["output_type"] == "discrete"
        assert "green" in flood["required_bands"]
        assert "nir" in flood["required_bands"]
    
    def test_water_indices_metadata(self, client):
        response = client.get("/algorithms")
        data = response.json()
        
        assert data["ndwi"]["output_type"] == "continuous"
        assert data["mndwi"]["output_type"] == "continuous"
        assert "swir16" in data["mndwi"]["required_bands"]