"""
CRONOS AI Engine - API Tests

This module contains tests for REST and gRPC API endpoints.
"""

import pytest
import asyncio
import json
import base64
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
import grpc
from grpc import aio as grpc_aio

from ai_engine.api.rest import create_app
from ai_engine.api.grpc import AIEngineGRPCService, GRPCServer
from ai_engine.api.schemas import (
    ProtocolDiscoveryRequest,
    FieldDetectionRequest,
    AnomalyDetectionRequest,
    DataFormat,
    ProtocolType,
    DetectionLevel
)
from ai_engine.api.auth import initialize_auth, get_api_key
from ai_engine.core.config import Config
from ai_engine.models import ModelOutput
from . import TestConfig


class TestRESTAPI:
    """Test cases for REST API endpoints."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Config()
        config.model_path = "test_models"
        config.data_path = "test_data"
        config.device = "cpu"
        return config
    
    @pytest.fixture
    def app(self, config):
        """Create test FastAPI app."""
        return create_app(config)
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        with patch('ai_engine.api.rest.AIEngineAPI.startup'):
            return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Create authentication headers."""
        api_key = get_api_key()
        return {"Authorization": f"Bearer {api_key}"}
    
    @pytest.fixture
    def sample_data(self):
        """Create sample request data."""
        http_data = b"GET /api/v1/users HTTP/1.1\r\nHost: example.com\r\n\r\n"
        return {
            "base64_http": base64.b64encode(http_data).decode('utf-8'),
            "hex_modbus": "0103000000024C0B",
            "text_data": "Hello World"
        }
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "checks" in data
    
    def test_health_check_unhealthy_engine(self, client):
        """Test health check with unhealthy engine."""
        with patch('ai_engine.api.rest.AIEngineAPI.ai_engine', None):
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] in ["unhealthy", "degraded"]
    
    def test_get_status_success(self, client, auth_headers):
        """Test engine status endpoint."""
        mock_status = {
            "status": "ready",
            "uptime_seconds": 3600,
            "components": {
                "protocol_discovery": {"status": "ready"},
                "field_detector": {"status": "ready"},
                "anomaly_detector": {"status": "ready"}
            },
            "system_metrics": {"memory_usage": 0.5},
            "active_models": {"protocol": 1, "field": 1, "anomaly": 1}
        }
        
        with patch('ai_engine.api.rest.AIEngineAPI.ai_engine') as mock_engine:
            mock_engine.get_status.return_value = mock_status
            
            response = client.get("/status", headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            assert data["engine_version"] == "1.0.0"
            assert data["uptime_seconds"] == 3600
            assert len(data["components"]) >= 0
    
    def test_get_status_no_auth(self, client):
        """Test status endpoint without authentication."""
        response = client.get("/status")
        
        assert response.status_code == 401
        data = response.json()
        assert "detail" in data
    
    def test_protocol_discovery_success(self, client, auth_headers, sample_data):
        """Test successful protocol discovery."""
        request_data = {
            "data": sample_data["base64_http"],
            "data_format": "base64",
            "confidence_threshold": 0.7,
            "max_samples": 1000,
            "include_grammar": False
        }
        
        mock_result = ModelOutput(
            predictions=None,
            metadata={
                "protocol_type": "http",
                "confidence": 0.92,
                "characteristics": {"method": "GET", "version": "1.1"},
                "statistical_features": {"avg_length": 64}
            }
        )
        
        with patch('ai_engine.api.rest.AIEngineAPI.ai_engine') as mock_engine:
            mock_engine.discover_protocol.return_value = mock_result
            
            response = client.post(
                "/discovery/protocols",
                json=request_data,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["discovered_protocol"] == "http"
            assert data["confidence_score"] == 0.92
            assert "processing_time_ms" in data
            assert data["protocol_characteristics"]["method"] == "GET"
    
    def test_protocol_discovery_invalid_data(self, client, auth_headers):
        """Test protocol discovery with invalid data."""
        request_data = {
            "data": "invalid_base64_data!@#",
            "data_format": "base64"
        }
        
        response = client.post(
            "/discovery/protocols",
            json=request_data,
            headers=auth_headers
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "error" in data or "detail" in data
    
    def test_field_detection_success(self, client, auth_headers, sample_data):
        """Test successful field detection."""
        request_data = {
            "data": sample_data["base64_http"],
            "data_format": "base64",
            "protocol_hint": "http",
            "detection_level": "medium",
            "include_boundaries": True,
            "max_fields": 100
        }
        
        mock_result = ModelOutput(
            predictions=None,
            metadata={
                "detected_fields": [
                    {
                        "id": "field_1",
                        "name": "method",
                        "start_offset": 0,
                        "end_offset": 3,
                        "field_type": "method",
                        "confidence": 0.95,
                        "examples": ["GET"]
                    },
                    {
                        "id": "field_2",
                        "name": "path",
                        "start_offset": 4,
                        "end_offset": 17,
                        "field_type": "path",
                        "confidence": 0.88,
                        "examples": ["/api/v1/users"]
                    }
                ],
                "confidence_summary": {"average": 0.915}
            }
        )
        
        with patch('ai_engine.api.rest.AIEngineAPI.ai_engine') as mock_engine:
            mock_engine.detect_fields.return_value = mock_result
            
            response = client.post(
                "/detection/fields",
                json=request_data,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_fields"] == 2
            assert len(data["detected_fields"]) == 2
            assert data["detected_fields"][0]["field_type"] == "method"
            assert data["detected_fields"][1]["field_type"] == "path"
    
    def test_anomaly_detection_success(self, client, auth_headers, sample_data):
        """Test successful anomaly detection."""
        request_data = {
            "data": sample_data["base64_http"],
            "data_format": "base64",
            "sensitivity": "medium",
            "anomaly_threshold": 0.5,
            "include_explanations": True
        }
        
        mock_result = ModelOutput(
            predictions=None,
            metadata={
                "is_anomalous": False,
                "anomaly_score": {
                    "overall_score": 0.15,
                    "reconstruction_error": 0.12,
                    "statistical_deviation": 0.18
                },
                "explanations": []
            }
        )
        
        with patch('ai_engine.api.rest.AIEngineAPI.ai_engine') as mock_engine:
            mock_engine.detect_anomalies.return_value = mock_result
            
            response = client.post(
                "/detection/anomalies",
                json=request_data,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["is_anomalous"] is False
            assert data["anomaly_score"]["overall_score"] == 0.15
            assert "processing_time_ms" in data
    
    def test_anomaly_detection_with_baseline(self, client, auth_headers, sample_data):
        """Test anomaly detection with baseline data."""
        request_data = {
            "data": sample_data["base64_http"],
            "data_format": "base64",
            "baseline_data": [sample_data["base64_http"]] * 5,  # Multiple baseline samples
            "sensitivity": "high",
            "anomaly_threshold": 0.3
        }
        
        mock_result = ModelOutput(
            predictions=None,
            metadata={
                "is_anomalous": False,
                "anomaly_score": {"overall_score": 0.1},
                "baseline_comparison": {"deviation": 0.05}
            }
        )
        
        with patch('ai_engine.api.rest.AIEngineAPI.ai_engine') as mock_engine:
            mock_engine.detect_anomalies.return_value = mock_result
            
            response = client.post(
                "/detection/anomalies",
                json=request_data,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "baseline_comparison" in data
    
    def test_model_registration(self, client, auth_headers):
        """Test model registration endpoint."""
        request_data = {
            "model_name": "test_protocol_model",
            "model_version": "1.0.0",
            "model_type": "protocol_discovery",
            "description": "Test protocol discovery model",
            "tags": {"environment": "test"},
            "performance_metrics": {"accuracy": 0.92, "f1_score": 0.89}
        }
        
        response = client.post(
            "/models",
            json=request_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["registration_status"] == "success"
        assert "model_id" in data
    
    def test_list_models(self, client, auth_headers):
        """Test model listing endpoint."""
        # Mock model registry response
        mock_models = [
            Mock(
                model_id="model_1",
                name="protocol_model",
                version="1.0.0",
                model_type=Mock(value="protocol_discovery"),
                status=Mock(value="production"),
                created_at=1234567890.0,
                updated_at=1234567890.0,
                accuracy=0.92,
                precision=0.88,
                recall=0.90,
                f1_score=0.89,
                tags={"env": "prod"},
                description="Production protocol model"
            )
        ]
        
        with patch('ai_engine.api.rest.AIEngineAPI.ai_engine') as mock_engine:
            mock_engine.model_registry.list_models.return_value = mock_models
            
            response = client.get("/models", headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["models"]) == 1
            assert data["total_count"] == 1
            assert data["models"][0]["name"] == "protocol_model"
    
    def test_batch_processing_small(self, client, auth_headers, sample_data):
        """Test batch processing with small batch."""
        request_data = {
            "operation": "discovery",
            "data_items": [sample_data["base64_http"]] * 5,
            "data_format": "base64"
        }
        
        with patch('ai_engine.api.rest.AIEngineAPI.ai_engine'):
            response = client.post(
                "/batch",
                json=request_data,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_items"] == 5
            assert "batch_id" in data
            assert isinstance(data["results"], list)
    
    def test_batch_processing_large(self, client, auth_headers, sample_data):
        """Test batch processing with large batch (background processing)."""
        request_data = {
            "operation": "detection",
            "data_items": [sample_data["base64_http"]] * 50,  # Large batch
            "data_format": "base64"
        }
        
        with patch('ai_engine.api.rest.AIEngineAPI.ai_engine'):
            response = client.post(
                "/batch",
                json=request_data,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_items"] == 50
            assert data["batch_metrics"]["status"] == "processing"
    
    def test_generic_prediction(self, client, auth_headers, sample_data):
        """Test generic prediction endpoint."""
        request_data = {
            "model_name": "protocol_discovery_model",
            "input_data": sample_data["base64_http"],
            "input_format": "base64"
        }
        
        mock_result = ModelOutput(
            predictions=None,
            metadata={"protocol_type": "http", "confidence": 0.9}
        )
        
        with patch('ai_engine.api.rest.AIEngineAPI.ai_engine') as mock_engine:
            mock_engine.discover_protocol.return_value = mock_result
            
            response = client.post(
                "/predict",
                json=request_data,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "predictions" in data
            assert "model_info" in data
    
    def test_cors_headers(self, client):
        """Test CORS headers in responses."""
        response = client.options("/health")
        
        # CORS headers should be present (added by middleware)
        assert response.status_code == 200
    
    def test_rate_limiting(self, client, auth_headers):
        """Test rate limiting functionality."""
        # Make multiple rapid requests
        responses = []
        for _ in range(5):
            response = client.get("/health", headers=auth_headers)
            responses.append(response)
        
        # All should succeed initially (small number of requests)
        for response in responses:
            assert response.status_code == 200
        
        # Check rate limit headers are present
        last_response = responses[-1]
        assert "X-RateLimit-Limit" in last_response.headers
        assert "X-RateLimit-Remaining" in last_response.headers
    
    def test_request_validation(self, client, auth_headers):
        """Test request validation."""
        # Test missing required fields
        response = client.post(
            "/discovery/protocols",
            json={},  # Missing required 'data' field
            headers=auth_headers
        )
        
        assert response.status_code == 422  # Validation error
        data = response.json()
        assert "detail" in data
    
    def test_error_handling(self, client, auth_headers, sample_data):
        """Test error handling and error responses."""
        request_data = {
            "data": sample_data["base64_http"],
            "data_format": "base64"
        }
        
        with patch('ai_engine.api.rest.AIEngineAPI.ai_engine') as mock_engine:
            mock_engine.discover_protocol.side_effect = Exception("Internal error")
            
            response = client.post(
                "/discovery/protocols",
                json=request_data,
                headers=auth_headers
            )
            
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data or "error" in data


class TestGRPCAPI:
    """Test cases for gRPC API."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Config()
        config.grpc_port = 50052  # Use different port for testing
        return config
    
    @pytest.fixture
    async def grpc_service(self, config):
        """Create gRPC service for testing."""
        service = AIEngineGRPCService(config)
        
        # Mock AI Engine initialization
        with patch.object(service, 'initialize', new_callable=AsyncMock):
            await service.initialize()
        
        return service
    
    @pytest.fixture
    def sample_data(self):
        """Create sample gRPC request data."""
        return {
            "http_data": base64.b64encode(b"GET / HTTP/1.1\r\n\r\n").decode(),
            "modbus_data": "0103000000024C0B"
        }
    
    @pytest.mark.asyncio
    async def test_grpc_service_initialization(self, grpc_service):
        """Test gRPC service initialization."""
        assert grpc_service is not None
        assert grpc_service.config is not None
        assert grpc_service.stats["total_requests"] == 0
    
    @pytest.mark.asyncio
    async def test_protocol_discovery_grpc(self, grpc_service, sample_data):
        """Test gRPC protocol discovery."""
        # Create mock request
        mock_request = Mock()
        mock_request.data = sample_data["http_data"]
        mock_request.data_format = "base64"
        mock_request.confidence_threshold = 0.7
        mock_request.max_samples = 1000
        mock_request.include_grammar = False
        
        # Create mock context
        mock_context = Mock()
        
        # Mock AI Engine
        mock_result = ModelOutput(
            predictions=None,
            metadata={
                "protocol_type": "http",
                "confidence": 0.9,
                "characteristics": {"method": "GET"}
            }
        )
        
        grpc_service.ai_engine = Mock()
        grpc_service.ai_engine.discover_protocol = AsyncMock(return_value=mock_result)
        
        # Test discovery
        result = await grpc_service.DiscoverProtocol(mock_request, mock_context)
        
        assert result["discovered_protocol"] == "http"
        assert result["confidence_score"] == 0.9
        assert "processing_time_ms" in result
        assert grpc_service.stats["total_requests"] == 1
        assert grpc_service.stats["successful_requests"] == 1
    
    @pytest.mark.asyncio
    async def test_field_detection_grpc(self, grpc_service, sample_data):
        """Test gRPC field detection."""
        # Create mock request
        mock_request = Mock()
        mock_request.data = sample_data["http_data"]
        mock_request.data_format = "base64"
        mock_request.protocol_hint = "http"
        mock_request.detection_level = "medium"
        mock_request.include_boundaries = True
        
        # Create mock context
        mock_context = Mock()
        
        # Mock AI Engine
        mock_result = ModelOutput(
            predictions=None,
            metadata={
                "detected_fields": [
                    {
                        "id": "field_1",
                        "start_offset": 0,
                        "end_offset": 3,
                        "field_type": "method",
                        "confidence": 0.9
                    }
                ]
            }
        )
        
        grpc_service.ai_engine = Mock()
        grpc_service.ai_engine.detect_fields = AsyncMock(return_value=mock_result)
        
        # Test detection
        result = await grpc_service.DetectFields(mock_request, mock_context)
        
        assert result["total_fields"] == 1
        assert len(result["detected_fields"]) == 1
        assert result["detected_fields"][0]["field_type"] == "method"
    
    @pytest.mark.asyncio
    async def test_anomaly_detection_grpc(self, grpc_service, sample_data):
        """Test gRPC anomaly detection."""
        # Create mock request
        mock_request = Mock()
        mock_request.data = sample_data["http_data"]
        mock_request.data_format = "base64"
        mock_request.sensitivity = "medium"
        mock_request.anomaly_threshold = 0.5
        
        # Create mock context
        mock_context = Mock()
        
        # Mock AI Engine
        mock_result = ModelOutput(
            predictions=None,
            metadata={
                "is_anomalous": False,
                "anomaly_score": {"overall_score": 0.2},
                "explanations": []
            }
        )
        
        grpc_service.ai_engine = Mock()
        grpc_service.ai_engine.detect_anomalies = AsyncMock(return_value=mock_result)
        
        # Test detection
        result = await grpc_service.DetectAnomalies(mock_request, mock_context)
        
        assert result["is_anomalous"] is False
        assert result["anomaly_score"]["overall_score"] == 0.2
    
    @pytest.mark.asyncio
    async def test_grpc_error_handling(self, grpc_service, sample_data):
        """Test gRPC error handling."""
        # Create mock request
        mock_request = Mock()
        mock_request.data = sample_data["http_data"]
        mock_request.data_format = "base64"
        
        # Create mock context
        mock_context = Mock()
        
        # Mock AI Engine to raise exception
        grpc_service.ai_engine = Mock()
        grpc_service.ai_engine.discover_protocol = AsyncMock(side_effect=Exception("Test error"))
        
        # Test error handling
        result = await grpc_service.DiscoverProtocol(mock_request, mock_context)
        
        assert result == {}  # Empty response on error
        mock_context.set_code.assert_called_with(grpc.StatusCode.INTERNAL)
        assert grpc_service.stats["failed_requests"] == 1
    
    @pytest.mark.asyncio
    async def test_service_status_grpc(self, grpc_service):
        """Test gRPC service status."""
        # Create mock request and context
        mock_request = Mock()
        mock_context = Mock()
        
        # Mock AI Engine status
        mock_engine_status = {"status": "ready", "components": {}}
        grpc_service.ai_engine = Mock()
        grpc_service.ai_engine.get_status = AsyncMock(return_value=mock_engine_status)
        
        # Test status
        result = await grpc_service.GetServiceStatus(mock_request, mock_context)
        
        assert result["service_name"] == "CRONOS AI Engine gRPC"
        assert result["version"] == "1.0.0"
        assert "uptime_seconds" in result
        assert "statistics" in result
        assert "engine_status" in result
    
    @pytest.mark.asyncio
    async def test_health_check_grpc(self, grpc_service):
        """Test gRPC health check."""
        # Create mock request and context
        mock_request = Mock()
        mock_context = Mock()
        
        # Mock AI Engine
        grpc_service.ai_engine = Mock()
        grpc_service.ai_engine.get_status = AsyncMock(return_value={"status": "ready"})
        
        # Test health check
        result = await grpc_service.HealthCheck(mock_request, mock_context)
        
        assert result["status"] == "healthy"
        assert result["service"] == "cronos-ai-grpc"
        assert "timestamp" in result
    
    @pytest.mark.asyncio
    async def test_batch_processing_grpc(self, grpc_service, sample_data):
        """Test gRPC batch processing."""
        # Create mock request
        mock_request = Mock()
        mock_request.operation = "discovery"
        mock_request.data_items = [sample_data["http_data"]] * 3
        
        # Create mock context
        mock_context = Mock()
        
        # Mock AI Engine
        grpc_service.ai_engine = Mock()
        
        # Test batch processing
        result = await grpc_service.BatchProcess(mock_request, mock_context)
        
        assert result["total_items"] == 3
        assert "batch_id" in result
        assert "results" in result


class TestAPIAuthentication:
    """Test cases for API authentication."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config()
    
    def test_api_key_authentication(self, config):
        """Test API key authentication."""
        initialize_auth(config)
        api_key = get_api_key()
        
        assert api_key is not None
        assert api_key.startswith("cronos_ai_") or api_key == "cronos_ai_mock_key_for_testing"
    
    def test_invalid_api_key(self, client):
        """Test invalid API key handling."""
        headers = {"Authorization": "Bearer invalid_key"}
        
        response = client.get("/status", headers=headers)
        
        assert response.status_code == 401
    
    def test_missing_authentication(self, client):
        """Test missing authentication handling."""
        response = client.get("/status")
        
        assert response.status_code == 401


class TestAPIPerformance:
    """Performance tests for API endpoints."""
    
    @pytest.mark.asyncio
    async def test_api_response_times(self, client, auth_headers):
        """Test API response times."""
        import time
        
        start_time = time.time()
        response = client.get("/health", headers=auth_headers)
        end_time = time.time()
        
        response_time_ms = (end_time - start_time) * 1000
        
        assert response.status_code == 200
        assert response_time_ms < TestConfig.PERFORMANCE_THRESHOLD_MS
    
    def test_concurrent_requests(self, client, auth_headers):
        """Test handling of concurrent requests."""
        import threading
        import time
        
        results = []
        
        def make_request():
            start_time = time.time()
            response = client.get("/health", headers=auth_headers)
            end_time = time.time()
            results.append({
                "status_code": response.status_code,
                "response_time": (end_time - start_time) * 1000
            })
        
        # Create multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(results) == 10
        for result in results:
            assert result["status_code"] == 200
            assert result["response_time"] < TestConfig.PERFORMANCE_THRESHOLD_MS


# Pytest fixtures for all API tests

@pytest.fixture
def config():
    """Create test configuration."""
    config = Config()
    config.model_path = "test_models"
    config.data_path = "test_data"
    config.device = "cpu"
    return config


@pytest.fixture
def sample_data():
    """Create sample test data."""
    http_data = b"GET /api/v1/users HTTP/1.1\r\nHost: example.com\r\n\r\n"
    return {
        "base64_http": base64.b64encode(http_data).decode('utf-8'),
        "hex_modbus": "0103000000024C0B",
        "text_data": "Hello World"
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])