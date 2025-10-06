"""
Comprehensive tests for ai_engine/api/grpc.py - gRPC Service
"""

import pytest
import time
import asyncio
import base64
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import grpc

from ai_engine.api.grpc import (
    AIEngineGRPCService,
    GRPCServer,
    AIEngineGRPCClient,
    run_grpc_server,
)
from ai_engine.core.config import Config


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
    engine.initialize = AsyncMock()
    engine.discover_protocol = AsyncMock(return_value={
        "protocol_type": "http",
        "confidence": 0.95,
        "structure": {},
        "processing_time": 0.1
    })
    engine.detect_fields = AsyncMock(return_value=[
        {"id": "field1", "name": "test_field", "start": 0, "end": 10}
    ])
    engine.detect_anomaly = AsyncMock(return_value={
        "is_anomaly": False,
        "anomaly_score": 0.2,
        "confidence": 0.9,
        "processing_time": 0.15
    })
    engine.get_model_info = Mock(return_value={"status": "ready"})
    engine._initialized = True
    return engine


class TestAIEngineGRPCService:
    """Test suite for AIEngineGRPCService."""

    def test_initialization(self, mock_config):
        """Test service initialization."""
        service = AIEngineGRPCService(mock_config)
        
        assert service.config == mock_config
        assert service.ai_engine is None
        assert service.server is None
        assert service.stats["total_requests"] == 0

    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_config, mock_ai_engine):
        """Test successful service initialization."""
        service = AIEngineGRPCService(mock_config)
        
        with patch("ai_engine.api.grpc.CronosAIEngine", return_value=mock_ai_engine):
            await service.initialize()
            
            assert service.ai_engine is not None
            mock_ai_engine.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_failure(self, mock_config):
        """Test service initialization failure."""
        service = AIEngineGRPCService(mock_config)
        
        with patch("ai_engine.api.grpc.CronosAIEngine") as mock_engine_class:
            mock_engine_class.side_effect = Exception("Init failed")
            
            with pytest.raises(Exception, match="Init failed"):
                await service.initialize()

    @pytest.mark.asyncio
    async def test_discover_protocol_success(self, mock_config, mock_ai_engine):
        """Test successful protocol discovery."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = mock_ai_engine
        
        request = Mock()
        request.data = base64.b64encode(b"test data").decode()
        request.data_format = "base64"
        request.expected_protocol = "http"
        request.confidence_threshold = 0.7
        
        context = Mock()
        
        response = await service.DiscoverProtocol(request, context)
        
        assert response["discovered_protocol"] == "http"
        assert response["confidence_score"] == 0.95
        assert service.stats["successful_requests"] == 1

    @pytest.mark.asyncio
    async def test_discover_protocol_engine_not_initialized(self, mock_config):
        """Test protocol discovery with uninitialized engine."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = None
        
        request = Mock()
        context = Mock()
        
        response = await service.DiscoverProtocol(request, context)
        
        assert response == {}
        context.set_code.assert_called_with(grpc.StatusCode.UNAVAILABLE)

    @pytest.mark.asyncio
    async def test_discover_protocol_exception(self, mock_config, mock_ai_engine):
        """Test protocol discovery with exception."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = mock_ai_engine
        mock_ai_engine.discover_protocol.side_effect = Exception("Discovery failed")
        
        request = Mock()
        request.data = base64.b64encode(b"test data").decode()
        request.data_format = "base64"
        
        context = Mock()
        
        response = await service.DiscoverProtocol(request, context)
        
        assert response == {}
        assert service.stats["failed_requests"] == 1
        context.set_code.assert_called_with(grpc.StatusCode.INTERNAL)

    @pytest.mark.asyncio
    async def test_discover_protocol_stream(self, mock_config, mock_ai_engine):
        """Test streaming protocol discovery."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = mock_ai_engine
        
        async def request_iterator():
            for i in range(3):
                request = Mock()
                request.data = base64.b64encode(f"test data {i}".encode()).decode()
                request.data_format = "base64"
                yield request
        
        context = Mock()
        
        responses = []
        async for response in service.DiscoverProtocolStream(request_iterator(), context):
            responses.append(response)
        
        assert len(responses) == 3

    @pytest.mark.asyncio
    async def test_detect_fields_success(self, mock_config, mock_ai_engine):
        """Test successful field detection."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = mock_ai_engine
        
        request = Mock()
        request.data = base64.b64encode(b"test data").decode()
        request.data_format = "base64"
        request.protocol_hint = "http"
        
        context = Mock()
        
        response = await service.DetectFields(request, context)
        
        assert "detected_fields" in response
        assert response["total_fields"] == 1
        assert service.stats["successful_requests"] == 1

    @pytest.mark.asyncio
    async def test_detect_fields_stream(self, mock_config, mock_ai_engine):
        """Test streaming field detection."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = mock_ai_engine
        
        async def request_iterator():
            for i in range(2):
                request = Mock()
                request.data = base64.b64encode(f"test data {i}".encode()).decode()
                request.data_format = "base64"
                yield request
        
        context = Mock()
        
        responses = []
        async for response in service.DetectFieldsStream(request_iterator(), context):
            responses.append(response)
        
        assert len(responses) == 2

    @pytest.mark.asyncio
    async def test_detect_anomalies_success(self, mock_config, mock_ai_engine):
        """Test successful anomaly detection."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = mock_ai_engine
        
        request = Mock()
        request.data = base64.b64encode(b"test data").decode()
        request.data_format = "base64"
        request.baseline_data = []
        request.protocol_context = "http"
        request.sensitivity = "medium"
        
        context = Mock()
        
        response = await service.DetectAnomalies(request, context)
        
        assert "is_anomalous" in response
        assert response["is_anomalous"] is False
        assert service.stats["successful_requests"] == 1

    @pytest.mark.asyncio
    async def test_detect_anomalies_with_baseline(self, mock_config, mock_ai_engine):
        """Test anomaly detection with baseline data."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = mock_ai_engine
        
        request = Mock()
        request.data = base64.b64encode(b"test data").decode()
        request.data_format = "base64"
        request.baseline_data = [base64.b64encode(b"baseline1").decode()]
        
        context = Mock()
        
        response = await service.DetectAnomalies(request, context)
        
        assert "is_anomalous" in response

    @pytest.mark.asyncio
    async def test_detect_anomalies_stream(self, mock_config, mock_ai_engine):
        """Test streaming anomaly detection."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = mock_ai_engine
        
        async def request_iterator():
            for i in range(2):
                request = Mock()
                request.data = base64.b64encode(f"test data {i}".encode()).decode()
                request.data_format = "base64"
                request.baseline_data = []
                yield request
        
        context = Mock()
        
        responses = []
        async for response in service.DetectAnomaliesStream(request_iterator(), context):
            responses.append(response)
        
        assert len(responses) == 2

    @pytest.mark.asyncio
    async def test_get_service_status(self, mock_config, mock_ai_engine):
        """Test getting service status."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = mock_ai_engine
        service.stats["successful_requests"] = 10
        service.stats["total_requests"] = 12
        service.stats["total_processing_time_ms"] = 1000.0
        
        request = Mock()
        context = Mock()
        
        response = await service.GetServiceStatus(request, context)
        
        assert response["service_name"] == "CRONOS AI Engine gRPC"
        assert response["statistics"]["total_requests"] == 12
        assert response["statistics"]["successful_requests"] == 10

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, mock_config, mock_ai_engine):
        """Test health check when healthy."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = mock_ai_engine
        
        request = Mock()
        context = Mock()
        
        response = await service.HealthCheck(request, context)
        
        assert response["status"] == "healthy"
        assert response["service"] == "cronos-ai-grpc"

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, mock_config):
        """Test health check when unhealthy."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = None
        
        request = Mock()
        context = Mock()
        
        response = await service.HealthCheck(request, context)
        
        assert response["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_health_check_degraded(self, mock_config, mock_ai_engine):
        """Test health check when degraded."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = mock_ai_engine
        mock_ai_engine._initialized = False
        
        request = Mock()
        context = Mock()
        
        response = await service.HealthCheck(request, context)
        
        assert response["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_batch_process_discovery(self, mock_config, mock_ai_engine):
        """Test batch processing for discovery."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = mock_ai_engine
        
        request = Mock()
        request.operation = "discovery"
        request.data_items = [
            base64.b64encode(b"data1").decode(),
            base64.b64encode(b"data2").decode(),
        ]
        
        context = Mock()
        
        response = await service.BatchProcess(request, context)
        
        assert response["total_items"] == 2
        assert response["successful_items"] == 2

    @pytest.mark.asyncio
    async def test_batch_process_detection(self, mock_config, mock_ai_engine):
        """Test batch processing for detection."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = mock_ai_engine
        
        request = Mock()
        request.operation = "detection"
        request.data_items = [base64.b64encode(b"data1").decode()]
        
        context = Mock()
        
        response = await service.BatchProcess(request, context)
        
        assert response["total_items"] == 1

    @pytest.mark.asyncio
    async def test_batch_process_anomaly(self, mock_config, mock_ai_engine):
        """Test batch processing for anomaly detection."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = mock_ai_engine
        
        request = Mock()
        request.operation = "anomaly"
        request.data_items = [base64.b64encode(b"data1").decode()]
        
        context = Mock()
        
        response = await service.BatchProcess(request, context)
        
        assert response["total_items"] == 1

    @pytest.mark.asyncio
    async def test_batch_process_with_failures(self, mock_config, mock_ai_engine):
        """Test batch processing with some failures."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = mock_ai_engine
        
        # Make one call fail
        call_count = [0]
        original_discover = mock_ai_engine.discover_protocol
        
        async def failing_discover(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("Discovery failed")
            return await original_discover(*args, **kwargs)
        
        mock_ai_engine.discover_protocol = failing_discover
        
        request = Mock()
        request.operation = "discovery"
        request.data_items = [
            base64.b64encode(b"data1").decode(),
            base64.b64encode(b"data2").decode(),
        ]
        
        context = Mock()
        
        response = await service.BatchProcess(request, context)
        
        assert response["successful_items"] == 1
        assert response["failed_items"] == 1

    def test_decode_data_base64(self, mock_config):
        """Test data decoding from base64."""
        service = AIEngineGRPCService(mock_config)
        
        data = base64.b64encode(b"test data").decode()
        decoded = service._decode_data(data, "base64")
        
        assert decoded == b"test data"

    def test_decode_data_hex(self, mock_config):
        """Test data decoding from hex."""
        service = AIEngineGRPCService(mock_config)
        
        data = "74657374"  # "test" in hex
        decoded = service._decode_data(data, "hex")
        
        assert decoded == b"test"

    def test_decode_data_text(self, mock_config):
        """Test data decoding from text."""
        service = AIEngineGRPCService(mock_config)
        
        data = "test data"
        decoded = service._decode_data(data, "text")
        
        assert decoded == b"test data"

    def test_decode_data_invalid(self, mock_config):
        """Test data decoding with invalid format."""
        service = AIEngineGRPCService(mock_config)
        
        with pytest.raises(ValueError, match="Data decoding failed"):
            service._decode_data("invalid base64!", "base64")


class TestGRPCServer:
    """Test suite for GRPCServer."""

    def test_initialization(self, mock_config):
        """Test server initialization."""
        server = GRPCServer(mock_config)
        
        assert server.config == mock_config
        assert server.port == 50051
        assert server.max_workers == 10

    @pytest.mark.asyncio
    async def test_start_server(self, mock_config):
        """Test starting the server."""
        server = GRPCServer(mock_config)
        
        with patch("ai_engine.api.grpc.AIEngineGRPCService") as mock_service_class:
            mock_service = AsyncMock()
            mock_service.initialize = AsyncMock()
            mock_service_class.return_value = mock_service
            
            with patch("ai_engine.api.grpc.grpc.aio.server") as mock_grpc_server:
                mock_grpc_instance = AsyncMock()
                mock_grpc_instance.add_insecure_port = Mock()
                mock_grpc_instance.start = AsyncMock()
                mock_grpc_server.return_value = mock_grpc_instance
                
                # Start in background
                task = asyncio.create_task(server.start())
                await asyncio.sleep(0.1)
                
                # Cancel the task
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    @pytest.mark.asyncio
    async def test_stop_server(self, mock_config):
        """Test stopping the server."""
        server = GRPCServer(mock_config)
        server.server = AsyncMock()
        server.server.stop = AsyncMock()
        
        await server.stop()
        
        server.server.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_wait_for_termination(self, mock_config):
        """Test waiting for server termination."""
        server = GRPCServer(mock_config)
        server.server = AsyncMock()
        server.server.wait_for_termination = AsyncMock()
        
        await server.wait_for_termination()
        
        server.server.wait_for_termination.assert_called_once()


class TestAIEngineGRPCClient:
    """Test suite for AIEngineGRPCClient."""

    def test_initialization(self):
        """Test client initialization."""
        client = AIEngineGRPCClient("localhost:50051")
        
        assert client.server_address == "localhost:50051"
        assert client.channel is None

    @pytest.mark.asyncio
    async def test_connect(self):
        """Test client connection."""
        client = AIEngineGRPCClient("localhost:50051")
        
        with patch("ai_engine.api.grpc.grpc.aio.insecure_channel") as mock_channel:
            mock_channel_instance = AsyncMock()
            mock_channel.return_value = mock_channel_instance
            
            with patch.object(client, "health_check", return_value={"status": "healthy"}):
                await client.connect()
                
                assert client.channel is not None

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test client disconnection."""
        client = AIEngineGRPCClient("localhost:50051")
        client.channel = AsyncMock()
        client.channel.close = AsyncMock()
        
        await client.disconnect()
        
        client.channel.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test client health check."""
        client = AIEngineGRPCClient("localhost:50051")
        
        result = await client.health_check()
        
        assert result["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_discover_protocol(self):
        """Test client protocol discovery."""
        client = AIEngineGRPCClient("localhost:50051")
        
        result = await client.discover_protocol(b"test data")
        
        assert "discovered_protocol" in result

    @pytest.mark.asyncio
    async def test_detect_fields(self):
        """Test client field detection."""
        client = AIEngineGRPCClient("localhost:50051")
        
        result = await client.detect_fields(b"test data")
        
        assert "detected_fields" in result

    @pytest.mark.asyncio
    async def test_detect_anomalies(self):
        """Test client anomaly detection."""
        client = AIEngineGRPCClient("localhost:50051")
        
        result = await client.detect_anomalies(b"test data")
        
        assert "is_anomalous" in result


class TestRunGRPCServer:
    """Test suite for run_grpc_server function."""

    @pytest.mark.asyncio
    async def test_run_grpc_server(self, mock_config):
        """Test running gRPC server."""
        with patch("ai_engine.api.grpc.GRPCServer") as mock_server_class:
            mock_server = AsyncMock()
            mock_server.start = AsyncMock()
            mock_server.wait_for_termination = AsyncMock()
            mock_server_class.return_value = mock_server
            
            # Run in background and cancel
            task = asyncio.create_task(run_grpc_server(mock_config))
            await asyncio.sleep(0.1)
            task.cancel()
            
            try:
                await task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_run_grpc_server_keyboard_interrupt(self, mock_config):
        """Test gRPC server with keyboard interrupt."""
        with patch("ai_engine.api.grpc.GRPCServer") as mock_server_class:
            mock_server = AsyncMock()
            mock_server.start = AsyncMock()
            mock_server.wait_for_termination = AsyncMock(side_effect=KeyboardInterrupt())
            mock_server.stop = AsyncMock()
            mock_server_class.return_value = mock_server
            
            await run_grpc_server(mock_config)
            
            mock_server.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_grpc_server_exception(self, mock_config):
        """Test gRPC server with exception."""
        with patch("ai_engine.api.grpc.GRPCServer") as mock_server_class:
            mock_server = AsyncMock()
            mock_server.start = AsyncMock()
            mock_server.wait_for_termination = AsyncMock(side_effect=Exception("Server error"))
            mock_server.stop = AsyncMock()
            mock_server_class.return_value = mock_server
            
            with pytest.raises(Exception, match="Server error"):
                await run_grpc_server(mock_config)
            
            mock_server.stop.assert_called_once()


class TestGRPCServiceEdgeCases:
    """Test edge cases for gRPC service."""

    @pytest.mark.asyncio
    async def test_discover_protocol_with_all_metadata(self, mock_config, mock_ai_engine):
        """Test protocol discovery with all metadata fields."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = mock_ai_engine
        
        request = Mock()
        request.data = base64.b64encode(b"test data").decode()
        request.data_format = "base64"
        request.expected_protocol = "http"
        request.confidence_threshold = 0.8
        request.max_samples = 500
        request.include_grammar = True
        
        context = Mock()
        
        response = await service.DiscoverProtocol(request, context)
        
        assert "discovered_protocol" in response

    @pytest.mark.asyncio
    async def test_batch_process_unknown_operation(self, mock_config, mock_ai_engine):
        """Test batch processing with unknown operation."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = mock_ai_engine
        
        request = Mock()
        request.operation = "unknown_operation"
        request.data_items = [base64.b64encode(b"data1").decode()]
        
        context = Mock()
        
        response = await service.BatchProcess(request, context)
        
        assert response["failed_items"] == 1

    @pytest.mark.asyncio
    async def test_get_service_status_exception(self, mock_config):
        """Test service status with exception."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = Mock()
        service.ai_engine.get_model_info.side_effect = Exception("Status error")
        
        request = Mock()
        context = Mock()
        
        response = await service.GetServiceStatus(request, context)
        
        assert response == {}
        context.set_code.assert_called_with(grpc.StatusCode.INTERNAL)

    @pytest.mark.asyncio
    async def test_health_check_exception(self, mock_config):
        """Test health check with exception."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = Mock()
        
        # Force an exception
        with patch.object(service, 'ai_engine', side_effect=Exception("Health check error")):
            request = Mock()
            context = Mock()
            
            response = await service.HealthCheck(request, context)
            
            assert response["status"] == "unhealthy"
            assert "error" in response

    def test_decode_data_hex_with_spaces(self, mock_config):
        """Test hex decoding with spaces."""
        service = AIEngineGRPCService(mock_config)
        
        data = "74 65 73 74"  # "test" in hex with spaces
        decoded = service._decode_data(data, "hex")
        
        assert decoded == b"test"

    @pytest.mark.asyncio
    async def test_detect_anomalies_with_string_baseline(self, mock_config, mock_ai_engine):
        """Test anomaly detection with string baseline data."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = mock_ai_engine
        
        request = Mock()
        request.data = base64.b64encode(b"test data").decode()
        request.data_format = "base64"
        request.baseline_data = base64.b64encode(b"baseline").decode()
        
        context = Mock()
        
        response = await service.DetectAnomalies(request, context)
        
        assert "is_anomalous" in response
