"""
Comprehensive tests for ai_engine/api/grpc.py

Tests cover:
- gRPC service initialization
- Protocol discovery (single and streaming)
- Field detection (single and streaming)
- Anomaly detection (single and streaming)
- Batch processing
- Service status and health checks
- Data encoding/decoding (base64, hex, text)
- Error handling
- Server lifecycle management
- Client utilities
"""

import pytest
import asyncio
import base64
import time
from unittest.mock import Mock, AsyncMock, MagicMock, patch, call
from typing import Dict, Any
import grpc

from ai_engine.api.grpc import (
    AIEngineGRPCService,
    GRPCServer,
    AIEngineGRPCClient,
    run_grpc_server,
)
from ai_engine.core.config import Config
from ai_engine.core.exceptions import AIEngineException


# Fixtures

@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = Mock(spec=Config)
    config.grpc_port = 50051
    config.grpc_max_workers = 10
    config.grpc_max_message_length = 10 * 1024 * 1024
    return config


@pytest.fixture
def mock_ai_engine():
    """Create mock AI engine."""
    engine = AsyncMock()
    engine._initialized = True
    engine.get_model_info.return_value = {
        "model_version": "1.0.0",
        "status": "loaded"
    }
    return engine


@pytest.fixture
def grpc_service(mock_config):
    """Create gRPC service instance."""
    service = AIEngineGRPCService(mock_config)
    return service


@pytest.fixture
def mock_grpc_context():
    """Create mock gRPC context."""
    context = Mock()
    context.set_code = Mock()
    context.set_details = Mock()
    return context


# Service Initialization Tests

def test_service_initialization(mock_config):
    """Test gRPC service initialization."""
    service = AIEngineGRPCService(mock_config)

    assert service.config == mock_config
    assert service.ai_engine is None
    assert service.server is None
    assert service.stats["total_requests"] == 0
    assert service.stats["successful_requests"] == 0
    assert service.stats["failed_requests"] == 0


@pytest.mark.asyncio
async def test_service_initialize_success(grpc_service, mock_ai_engine):
    """Test successful service initialization."""
    with patch("ai_engine.api.grpc.CronosAIEngine", return_value=mock_ai_engine):
        await grpc_service.initialize()

        assert grpc_service.ai_engine is not None
        mock_ai_engine.initialize.assert_called_once()


@pytest.mark.asyncio
async def test_service_initialize_failure(grpc_service):
    """Test service initialization failure."""
    with patch("ai_engine.api.grpc.CronosAIEngine", side_effect=Exception("Init failed")):
        with pytest.raises(Exception, match="Init failed"):
            await grpc_service.initialize()


# Data Encoding/Decoding Tests

def test_decode_data_base64(grpc_service):
    """Test base64 data decoding."""
    data = b"Hello, World!"
    encoded = base64.b64encode(data).decode()

    decoded = grpc_service._decode_data(encoded, "base64")
    assert decoded == data


def test_decode_data_hex(grpc_service):
    """Test hex data decoding."""
    data = b"Hello"
    hex_str = data.hex()

    decoded = grpc_service._decode_data(hex_str, "hex")
    assert decoded == data


def test_decode_data_hex_with_spaces(grpc_service):
    """Test hex data decoding with spaces."""
    hex_str = "48 65 6c 6c 6f"

    decoded = grpc_service._decode_data(hex_str, "hex")
    assert decoded == b"Hello"


def test_decode_data_text(grpc_service):
    """Test text data decoding."""
    text = "Hello, World!"

    decoded = grpc_service._decode_data(text, "text")
    assert decoded == text.encode("utf-8")


def test_decode_data_default(grpc_service):
    """Test default data decoding."""
    text = "Hello"

    decoded = grpc_service._decode_data(text, "unknown_format")
    assert decoded == text.encode("latin-1")


def test_decode_data_invalid_base64(grpc_service):
    """Test decoding invalid base64."""
    invalid_base64 = "!!!invalid!!!"

    with pytest.raises(ValueError, match="Data decoding failed"):
        grpc_service._decode_data(invalid_base64, "base64")


def test_decode_data_invalid_hex(grpc_service):
    """Test decoding invalid hex."""
    invalid_hex = "GHIJKL"

    with pytest.raises(ValueError, match="Data decoding failed"):
        grpc_service._decode_data(invalid_hex, "hex")


# Protocol Discovery Tests

@pytest.mark.asyncio
async def test_discover_protocol_success(grpc_service, mock_ai_engine, mock_grpc_context):
    """Test successful protocol discovery."""
    grpc_service.ai_engine = mock_ai_engine

    # Mock discovery result
    mock_ai_engine.discover_protocol.return_value = {
        "protocol_type": "http",
        "confidence": 0.95,
        "structure": {"method": "GET"},
        "grammar": "HTTP/1.1",
        "metadata": {},
        "processing_time": 0.15
    }

    # Create mock request
    request = Mock()
    request.data = base64.b64encode(b"GET / HTTP/1.1").decode()
    request.data_format = "base64"
    request.expected_protocol = "http"
    request.confidence_threshold = 0.7

    response = await grpc_service.DiscoverProtocol(request, mock_grpc_context)

    assert response["discovered_protocol"] == "http"
    assert response["confidence_score"] == 0.95
    assert response["structure"] == {"method": "GET"}
    assert response["processing_time_ms"] > 0

    assert grpc_service.stats["successful_requests"] == 1
    assert grpc_service.stats["total_requests"] == 1


@pytest.mark.asyncio
async def test_discover_protocol_engine_not_initialized(grpc_service, mock_grpc_context):
    """Test protocol discovery without initialized engine."""
    grpc_service.ai_engine = None

    request = Mock()
    request.data = base64.b64encode(b"test").decode()
    request.data_format = "base64"

    response = await grpc_service.DiscoverProtocol(request, mock_grpc_context)

    assert response == {}
    mock_grpc_context.set_code.assert_called_with(grpc.StatusCode.UNAVAILABLE)


@pytest.mark.asyncio
async def test_discover_protocol_failure(grpc_service, mock_ai_engine, mock_grpc_context):
    """Test protocol discovery failure."""
    grpc_service.ai_engine = mock_ai_engine
    mock_ai_engine.discover_protocol.side_effect = Exception("Discovery failed")

    request = Mock()
    request.data = base64.b64encode(b"test").decode()
    request.data_format = "base64"

    response = await grpc_service.DiscoverProtocol(request, mock_grpc_context)

    assert response == {}
    assert grpc_service.stats["failed_requests"] == 1
    mock_grpc_context.set_code.assert_called_with(grpc.StatusCode.INTERNAL)


@pytest.mark.asyncio
async def test_discover_protocol_stream(grpc_service, mock_ai_engine, mock_grpc_context):
    """Test streaming protocol discovery."""
    grpc_service.ai_engine = mock_ai_engine

    mock_ai_engine.discover_protocol.return_value = {
        "protocol_type": "http",
        "confidence": 0.95,
        "structure": {},
        "metadata": {},
        "processing_time": 0.1
    }

    # Create mock request iterator
    requests = []
    for i in range(3):
        req = Mock()
        req.data = base64.b64encode(f"request {i}".encode()).decode()
        req.data_format = "base64"
        requests.append(req)

    async def request_iterator():
        for req in requests:
            yield req

    # Collect responses
    responses = []
    async for response in grpc_service.DiscoverProtocolStream(request_iterator(), mock_grpc_context):
        responses.append(response)

    assert len(responses) == 3
    assert all(r["discovered_protocol"] == "http" for r in responses)


# Field Detection Tests

@pytest.mark.asyncio
async def test_detect_fields_success(grpc_service, mock_ai_engine, mock_grpc_context):
    """Test successful field detection."""
    grpc_service.ai_engine = mock_ai_engine

    mock_ai_engine.detect_fields.return_value = [
        {
            "id": "field1",
            "name": "header",
            "start": 0,
            "end": 10,
            "type": "string",
            "confidence": 0.9,
            "semantic_type": "metadata",
            "encoding": "utf-8",
            "examples": ["example1"]
        }
    ]

    request = Mock()
    request.data = base64.b64encode(b"test data").decode()
    request.data_format = "base64"
    request.protocol_hint = "http"

    response = await grpc_service.DetectFields(request, mock_grpc_context)

    assert response["total_fields"] == 1
    assert len(response["detected_fields"]) == 1
    assert response["detected_fields"][0]["field_name"] == "header"


@pytest.mark.asyncio
async def test_detect_fields_empty_result(grpc_service, mock_ai_engine, mock_grpc_context):
    """Test field detection with no fields found."""
    grpc_service.ai_engine = mock_ai_engine
    mock_ai_engine.detect_fields.return_value = []

    request = Mock()
    request.data = base64.b64encode(b"test").decode()
    request.data_format = "base64"

    response = await grpc_service.DetectFields(request, mock_grpc_context)

    assert response["total_fields"] == 0
    assert response["detected_fields"] == []


@pytest.mark.asyncio
async def test_detect_fields_stream(grpc_service, mock_ai_engine, mock_grpc_context):
    """Test streaming field detection."""
    grpc_service.ai_engine = mock_ai_engine
    mock_ai_engine.detect_fields.return_value = [{"id": "field1", "name": "test"}]

    requests = [Mock(data=base64.b64encode(b"test").decode(), data_format="base64") for _ in range(2)]

    async def request_iterator():
        for req in requests:
            yield req

    responses = []
    async for response in grpc_service.DetectFieldsStream(request_iterator(), mock_grpc_context):
        responses.append(response)

    assert len(responses) == 2


# Anomaly Detection Tests

@pytest.mark.asyncio
async def test_detect_anomalies_success(grpc_service, mock_ai_engine, mock_grpc_context):
    """Test successful anomaly detection."""
    grpc_service.ai_engine = mock_ai_engine

    mock_ai_engine.detect_anomaly.return_value = {
        "is_anomaly": True,
        "anomaly_score": 0.85,
        "confidence": 0.9,
        "explanation": "Unusual pattern detected",
        "detector_scores": {"vae": 0.8},
        "context": {},
        "processing_time": 0.2
    }

    request = Mock()
    request.data = base64.b64encode(b"anomalous data").decode()
    request.data_format = "base64"
    request.protocol_context = "http"
    request.sensitivity = "high"
    request.baseline_data = []
    request.anomaly_threshold = 0.5
    request.include_explanations = True

    response = await grpc_service.DetectAnomalies(request, mock_grpc_context)

    assert response["is_anomalous"] is True
    assert response["anomaly_score"]["score"] == 0.85
    assert len(response["anomaly_explanations"]) > 0


@pytest.mark.asyncio
async def test_detect_anomalies_with_baseline(grpc_service, mock_ai_engine, mock_grpc_context):
    """Test anomaly detection with baseline data."""
    grpc_service.ai_engine = mock_ai_engine

    mock_ai_engine.detect_anomaly.return_value = {
        "is_anomaly": False,
        "anomaly_score": 0.3,
        "confidence": 0.7,
        "processing_time": 0.15
    }

    request = Mock()
    request.data = base64.b64encode(b"test").decode()
    request.data_format = "base64"
    request.baseline_data = [base64.b64encode(b"baseline1").decode()]

    response = await grpc_service.DetectAnomalies(request, mock_grpc_context)

    assert response["is_anomalous"] is False


@pytest.mark.asyncio
async def test_detect_anomalies_stream(grpc_service, mock_ai_engine, mock_grpc_context):
    """Test streaming anomaly detection."""
    grpc_service.ai_engine = mock_ai_engine

    mock_ai_engine.detect_anomaly.return_value = {
        "is_anomaly": False,
        "anomaly_score": 0.2,
        "confidence": 0.8,
        "processing_time": 0.1
    }

    requests = [Mock(data=base64.b64encode(b"test").decode(), data_format="base64", baseline_data=[]) for _ in range(3)]

    async def request_iterator():
        for req in requests:
            yield req

    responses = []
    async for response in grpc_service.DetectAnomaliesStream(request_iterator(), mock_grpc_context):
        responses.append(response)

    assert len(responses) == 3


# Service Status Tests

@pytest.mark.asyncio
async def test_get_service_status_success(grpc_service, mock_ai_engine, mock_grpc_context):
    """Test getting service status."""
    grpc_service.ai_engine = mock_ai_engine
    grpc_service.stats["successful_requests"] = 100
    grpc_service.stats["failed_requests"] = 5
    grpc_service.stats["total_requests"] = 105
    grpc_service.stats["total_processing_time_ms"] = 5000

    response = await grpc_service.GetServiceStatus(Mock(), mock_grpc_context)

    assert response["service_name"] == "CRONOS AI Engine gRPC"
    assert response["version"] == "1.0.0"
    assert response["uptime_seconds"] > 0
    assert response["statistics"]["total_requests"] == 105
    assert response["statistics"]["successful_requests"] == 100
    assert response["statistics"]["failed_requests"] == 5
    assert 0 <= response["statistics"]["success_rate"] <= 1
    assert response["statistics"]["average_processing_time_ms"] == 50.0


@pytest.mark.asyncio
async def test_get_service_status_no_requests(grpc_service, mock_ai_engine, mock_grpc_context):
    """Test service status with no requests."""
    grpc_service.ai_engine = mock_ai_engine

    response = await grpc_service.GetServiceStatus(Mock(), mock_grpc_context)

    assert response["statistics"]["average_processing_time_ms"] == 0.0


@pytest.mark.asyncio
async def test_get_service_status_failure(grpc_service, mock_grpc_context):
    """Test service status retrieval failure."""
    grpc_service.ai_engine = Mock()
    grpc_service.ai_engine.get_model_info.side_effect = Exception("Status failed")

    response = await grpc_service.GetServiceStatus(Mock(), mock_grpc_context)

    assert response == {}
    mock_grpc_context.set_code.assert_called_with(grpc.StatusCode.INTERNAL)


# Health Check Tests

@pytest.mark.asyncio
async def test_health_check_healthy(grpc_service, mock_ai_engine, mock_grpc_context):
    """Test health check when service is healthy."""
    grpc_service.ai_engine = mock_ai_engine

    response = await grpc_service.HealthCheck(Mock(), mock_grpc_context)

    assert response["status"] == "healthy"
    assert response["service"] == "cronos-ai-grpc"
    assert "timestamp" in response


@pytest.mark.asyncio
async def test_health_check_unhealthy_no_engine(grpc_service, mock_grpc_context):
    """Test health check when engine is not initialized."""
    grpc_service.ai_engine = None

    response = await grpc_service.HealthCheck(Mock(), mock_grpc_context)

    assert response["status"] == "unhealthy"


@pytest.mark.asyncio
async def test_health_check_degraded(grpc_service, mock_grpc_context):
    """Test health check when engine is degraded."""
    mock_engine = Mock()
    mock_engine._initialized = False
    grpc_service.ai_engine = mock_engine

    response = await grpc_service.HealthCheck(Mock(), mock_grpc_context)

    assert response["status"] == "degraded"


@pytest.mark.asyncio
async def test_health_check_exception(grpc_service, mock_grpc_context):
    """Test health check with exception."""
    grpc_service.ai_engine = Mock()
    # Cause an exception by accessing a non-existent attribute
    type(grpc_service.ai_engine)._initialized = property(lambda self: 1/0)

    response = await grpc_service.HealthCheck(Mock(), mock_grpc_context)

    assert response["status"] == "unhealthy"
    assert "error" in response


# Batch Processing Tests

@pytest.mark.asyncio
async def test_batch_process_discovery(grpc_service, mock_ai_engine, mock_grpc_context):
    """Test batch processing for protocol discovery."""
    grpc_service.ai_engine = mock_ai_engine

    mock_ai_engine.discover_protocol.return_value = {
        "protocol_type": "http",
        "confidence": 0.9
    }

    request = Mock()
    request.operation = "discovery"
    request.data_items = [
        base64.b64encode(b"data1").decode(),
        base64.b64encode(b"data2").decode(),
    ]

    response = await grpc_service.BatchProcess(request, mock_grpc_context)

    assert response["total_items"] == 2
    assert response["successful_items"] == 2
    assert response["failed_items"] == 0
    assert len(response["results"]) == 2


@pytest.mark.asyncio
async def test_batch_process_detection(grpc_service, mock_ai_engine, mock_grpc_context):
    """Test batch processing for field detection."""
    grpc_service.ai_engine = mock_ai_engine
    mock_ai_engine.detect_fields.return_value = []

    request = Mock()
    request.operation = "detection"
    request.data_items = [base64.b64encode(b"data1").decode()]

    response = await grpc_service.BatchProcess(request, mock_grpc_context)

    assert response["total_items"] == 1
    assert response["successful_items"] == 1


@pytest.mark.asyncio
async def test_batch_process_anomaly(grpc_service, mock_ai_engine, mock_grpc_context):
    """Test batch processing for anomaly detection."""
    grpc_service.ai_engine = mock_ai_engine
    mock_ai_engine.detect_anomaly.return_value = {"is_anomaly": False}

    request = Mock()
    request.operation = "anomaly"
    request.data_items = [base64.b64encode(b"data1").decode()]

    response = await grpc_service.BatchProcess(request, mock_grpc_context)

    assert response["total_items"] == 1


@pytest.mark.asyncio
async def test_batch_process_with_failures(grpc_service, mock_ai_engine, mock_grpc_context):
    """Test batch processing with some failures."""
    grpc_service.ai_engine = mock_ai_engine

    # First succeeds, second fails
    mock_ai_engine.discover_protocol.side_effect = [
        {"protocol_type": "http"},
        Exception("Processing failed")
    ]

    request = Mock()
    request.operation = "discovery"
    request.data_items = [
        base64.b64encode(b"data1").decode(),
        base64.b64encode(b"data2").decode(),
    ]

    response = await grpc_service.BatchProcess(request, mock_grpc_context)

    assert response["total_items"] == 2
    assert response["successful_items"] == 1
    assert response["failed_items"] == 1
    assert response["results"][0]["success"] is True
    assert response["results"][1]["success"] is False
    assert "error" in response["results"][1]


@pytest.mark.asyncio
async def test_batch_process_unknown_operation(grpc_service, mock_ai_engine, mock_grpc_context):
    """Test batch processing with unknown operation."""
    grpc_service.ai_engine = mock_ai_engine

    request = Mock()
    request.operation = "unknown_operation"
    request.data_items = [base64.b64encode(b"data1").decode()]

    response = await grpc_service.BatchProcess(request, mock_grpc_context)

    assert response["failed_items"] == 1


@pytest.mark.asyncio
async def test_batch_process_empty_items(grpc_service, mock_ai_engine, mock_grpc_context):
    """Test batch processing with empty items."""
    grpc_service.ai_engine = mock_ai_engine

    request = Mock()
    request.operation = "discovery"
    request.data_items = []

    response = await grpc_service.BatchProcess(request, mock_grpc_context)

    assert response["total_items"] == 0
    assert response["successful_items"] == 0


# gRPC Server Tests

@pytest.mark.asyncio
async def test_grpc_server_initialization(mock_config):
    """Test gRPC server initialization."""
    server = GRPCServer(mock_config)

    assert server.config == mock_config
    assert server.port == 50051
    assert server.max_workers == 10


@pytest.mark.asyncio
async def test_grpc_server_start(mock_config):
    """Test gRPC server start."""
    server = GRPCServer(mock_config)

    with patch.object(server, "service", new_callable=AsyncMock):
        with patch("ai_engine.api.grpc.grpc.aio.server") as mock_server_create:
            mock_grpc_server = AsyncMock()
            mock_server_create.return_value = mock_grpc_server

            # Mock AIEngineGRPCService
            with patch("ai_engine.api.grpc.AIEngineGRPCService"):
                # Start in background (don't wait for termination)
                start_task = asyncio.create_task(server.start())

                # Give it a moment to start
                await asyncio.sleep(0.1)

                # Verify server was created and started
                assert server.server is not None

                # Cancel the task to stop waiting
                start_task.cancel()
                try:
                    await start_task
                except asyncio.CancelledError:
                    pass


@pytest.mark.asyncio
async def test_grpc_server_stop(mock_config):
    """Test gRPC server stop."""
    server = GRPCServer(mock_config)

    mock_grpc_server = AsyncMock()
    server.server = mock_grpc_server

    await server.stop(grace_period=2)

    mock_grpc_server.stop.assert_called_once_with(grace=2)


@pytest.mark.asyncio
async def test_grpc_server_wait_for_termination(mock_config):
    """Test waiting for server termination."""
    server = GRPCServer(mock_config)

    mock_grpc_server = AsyncMock()
    server.server = mock_grpc_server

    await server.wait_for_termination()

    mock_grpc_server.wait_for_termination.assert_called_once()


# gRPC Client Tests

def test_grpc_client_initialization():
    """Test gRPC client initialization."""
    client = AIEngineGRPCClient("localhost:50051")

    assert client.server_address == "localhost:50051"
    assert client.channel is None
    assert client.stub is None


@pytest.mark.asyncio
async def test_grpc_client_connect():
    """Test client connection."""
    client = AIEngineGRPCClient("localhost:50051")

    with patch("ai_engine.api.grpc.grpc.aio.insecure_channel") as mock_channel_create:
        mock_channel = AsyncMock()
        mock_channel_create.return_value = mock_channel

        await client.connect()

        assert client.channel is not None
        mock_channel_create.assert_called_once_with("localhost:50051")


@pytest.mark.asyncio
async def test_grpc_client_disconnect():
    """Test client disconnection."""
    client = AIEngineGRPCClient()
    client.channel = AsyncMock()

    await client.disconnect()

    client.channel.close.assert_called_once()


@pytest.mark.asyncio
async def test_grpc_client_health_check():
    """Test client health check."""
    client = AIEngineGRPCClient()

    result = await client.health_check()

    assert result["status"] == "healthy"
    assert "timestamp" in result


@pytest.mark.asyncio
async def test_grpc_client_discover_protocol():
    """Test client protocol discovery."""
    client = AIEngineGRPCClient()

    result = await client.discover_protocol(b"test data")

    assert "discovered_protocol" in result
    assert "confidence_score" in result


@pytest.mark.asyncio
async def test_grpc_client_detect_fields():
    """Test client field detection."""
    client = AIEngineGRPCClient()

    result = await client.detect_fields(b"test data")

    assert "detected_fields" in result
    assert "total_fields" in result


@pytest.mark.asyncio
async def test_grpc_client_detect_anomalies():
    """Test client anomaly detection."""
    client = AIEngineGRPCClient()

    result = await client.detect_anomalies(b"test data")

    assert "is_anomalous" in result
    assert "anomaly_score" in result


# Integration Tests

@pytest.mark.asyncio
async def test_full_grpc_workflow(mock_config, mock_ai_engine):
    """Test complete gRPC workflow."""
    # Create service
    service = AIEngineGRPCService(mock_config)

    with patch("ai_engine.api.grpc.CronosAIEngine", return_value=mock_ai_engine):
        await service.initialize()

    # Mock context
    context = Mock()
    context.set_code = Mock()
    context.set_details = Mock()

    # Mock AI engine responses
    mock_ai_engine.discover_protocol.return_value = {
        "protocol_type": "http",
        "confidence": 0.9,
        "structure": {},
        "metadata": {},
        "processing_time": 0.1
    }

    # Test protocol discovery
    request = Mock()
    request.data = base64.b64encode(b"GET / HTTP/1.1").decode()
    request.data_format = "base64"

    response = await service.DiscoverProtocol(request, context)
    assert response["discovered_protocol"] == "http"

    # Check statistics
    assert service.stats["total_requests"] == 1
    assert service.stats["successful_requests"] == 1


@pytest.mark.asyncio
async def test_concurrent_requests(grpc_service, mock_ai_engine, mock_grpc_context):
    """Test handling concurrent requests."""
    grpc_service.ai_engine = mock_ai_engine

    mock_ai_engine.discover_protocol.return_value = {
        "protocol_type": "http",
        "confidence": 0.9,
        "structure": {},
        "metadata": {},
        "processing_time": 0.1
    }

    # Create multiple requests
    requests = []
    for i in range(10):
        req = Mock()
        req.data = base64.b64encode(f"request {i}".encode()).decode()
        req.data_format = "base64"
        requests.append(req)

    # Process concurrently
    results = await asyncio.gather(*[
        grpc_service.DiscoverProtocol(req, mock_grpc_context) for req in requests
    ])

    assert len(results) == 10
    assert all(r["discovered_protocol"] == "http" for r in results)
    assert grpc_service.stats["successful_requests"] == 10


@pytest.mark.asyncio
async def test_error_propagation(grpc_service, mock_ai_engine, mock_grpc_context):
    """Test error propagation through gRPC layers."""
    grpc_service.ai_engine = mock_ai_engine

    # Simulate different types of errors
    errors = [
        ValueError("Invalid input"),
        RuntimeError("Processing error"),
        Exception("Unknown error")
    ]

    for error in errors:
        mock_ai_engine.discover_protocol.side_effect = error

        request = Mock()
        request.data = base64.b64encode(b"test").decode()
        request.data_format = "base64"

        response = await grpc_service.DiscoverProtocol(request, mock_grpc_context)

        assert response == {}
        mock_grpc_context.set_code.assert_called_with(grpc.StatusCode.INTERNAL)


# Server Runner Tests

@pytest.mark.asyncio
async def test_run_grpc_server():
    """Test gRPC server runner."""
    mock_config = Mock(spec=Config)

    with patch("ai_engine.api.grpc.GRPCServer") as MockServerClass:
        mock_server = AsyncMock()
        MockServerClass.return_value = mock_server

        # Run server in background for a short time
        server_task = asyncio.create_task(run_grpc_server(mock_config))

        # Give it time to start
        await asyncio.sleep(0.1)

        # Cancel to stop
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

        mock_server.start.assert_called_once()
