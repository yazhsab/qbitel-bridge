"""
CRONOS AI Engine - Core Component Tests

This module contains unit tests for core AI Engine components.
"""

import pytest
import asyncio
import torch
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import json

from ai_engine.core.config import Config
from ai_engine.core.engine import AIEngine
from ai_engine.core.exceptions import AIEngineException, ValidationException
from ai_engine.models import ModelInput, ModelOutput
from . import TestConfig as BaseTestConfig


class TestConfig(BaseTestConfig):
    """Test configuration for core tests."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Config()
        config.model_path = "test_models"
        config.data_path = "test_data"
        config.device = "cpu"
        return config

    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        return {
            "http_data": b"GET /api/v1/users HTTP/1.1\r\nHost: example.com\r\n\r\n",
            "modbus_data": bytes([0x01, 0x03, 0x00, 0x00, 0x00, 0x02, 0xC4, 0x0B]),
            "hl7_data": b"MSH|^~\\&|SENDER|HOSPITAL|RECEIVER|CLINIC|20230101120000||ADT^A08|123456|P|2.5\r",
            "anomalous_data": np.random.bytes(256),
        }


class TestAIEngine:
    """Test cases for AI Engine core functionality."""

    @pytest.mark.asyncio
    async def test_engine_initialization(self, config):
        """Test AI Engine initialization."""
        engine = AIEngine(config)

        assert engine.config == config
        assert engine.state == "initialized"
        assert engine.protocol_discovery is None
        assert engine.field_detector is None
        assert engine.anomaly_detector is None
        assert engine.model_registry is None

    @pytest.mark.asyncio
    async def test_engine_initialize_success(self, config):
        """Test successful engine initialization."""
        engine = AIEngine(config)

        with patch.object(
            engine, "_initialize_components", new_callable=AsyncMock
        ) as mock_init:
            mock_init.return_value = None

            await engine.initialize()

            assert engine.state == "ready"
            mock_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_engine_initialize_failure(self, config):
        """Test engine initialization failure handling."""
        engine = AIEngine(config)

        with patch.object(
            engine, "_initialize_components", new_callable=AsyncMock
        ) as mock_init:
            mock_init.side_effect = Exception("Initialization failed")

            with pytest.raises(
                AIEngineException, match="Failed to initialize AI Engine"
            ):
                await engine.initialize()

            assert engine.state == "failed"

    @pytest.mark.asyncio
    async def test_protocol_discovery_success(self, config, sample_data):
        """Test successful protocol discovery."""
        engine = AIEngine(config)

        # Mock components
        mock_discovery = AsyncMock()
        mock_discovery.discover_protocol.return_value = ModelOutput(
            predictions=torch.tensor([1.0, 0.0, 0.0, 0.0]),  # HTTP protocol
            metadata={
                "protocol_type": "http",
                "confidence": 0.95,
                "characteristics": {"method": "GET", "version": "1.1"},
            },
        )
        engine.protocol_discovery = mock_discovery
        engine.state = "ready"

        # Create input
        model_input = ModelInput(data=sample_data["http_data"])

        # Test discovery
        result = await engine.discover_protocol(model_input)

        assert isinstance(result, ModelOutput)
        assert result.metadata["protocol_type"] == "http"
        assert result.metadata["confidence"] == 0.95
        mock_discovery.discover_protocol.assert_called_once_with(model_input)

    @pytest.mark.asyncio
    async def test_protocol_discovery_engine_not_ready(self, config, sample_data):
        """Test protocol discovery when engine is not ready."""
        engine = AIEngine(config)
        # Don't set state to ready

        model_input = ModelInput(data=sample_data["http_data"])

        with pytest.raises(AIEngineException, match="AI Engine not ready"):
            await engine.discover_protocol(model_input)

    @pytest.mark.asyncio
    async def test_field_detection_success(self, config, sample_data):
        """Test successful field detection."""
        engine = AIEngine(config)

        # Mock field detector
        mock_detector = AsyncMock()
        mock_detector.detect_fields.return_value = ModelOutput(
            predictions=torch.tensor([[1, 0], [0, 1], [1, 0]]),  # IOB tags
            metadata={
                "detected_fields": [
                    {
                        "id": "field_1",
                        "start_offset": 0,
                        "end_offset": 3,
                        "field_type": "method",
                        "confidence": 0.92,
                    }
                ]
            },
        )
        engine.field_detector = mock_detector
        engine.state = "ready"

        # Create input
        model_input = ModelInput(data=sample_data["http_data"])

        # Test detection
        result = await engine.detect_fields(model_input)

        assert isinstance(result, ModelOutput)
        assert len(result.metadata["detected_fields"]) == 1
        assert result.metadata["detected_fields"][0]["field_type"] == "method"
        mock_detector.detect_fields.assert_called_once_with(model_input)

    @pytest.mark.asyncio
    async def test_anomaly_detection_success(self, config, sample_data):
        """Test successful anomaly detection."""
        engine = AIEngine(config)

        # Mock anomaly detector
        mock_detector = AsyncMock()
        mock_detector.detect_anomalies.return_value = ModelOutput(
            predictions=torch.tensor([0.15]),  # Low anomaly score
            metadata={
                "is_anomalous": False,
                "anomaly_score": {
                    "overall_score": 0.15,
                    "reconstruction_error": 0.12,
                    "statistical_deviation": 0.18,
                },
            },
        )
        engine.anomaly_detector = mock_detector
        engine.state = "ready"

        # Create input
        model_input = ModelInput(data=sample_data["http_data"])

        # Test detection
        result = await engine.detect_anomalies(model_input)

        assert isinstance(result, ModelOutput)
        assert result.metadata["is_anomalous"] is False
        assert result.metadata["anomaly_score"]["overall_score"] == 0.15
        mock_detector.detect_anomalies.assert_called_once_with(model_input)

    @pytest.mark.asyncio
    async def test_get_status(self, config):
        """Test engine status reporting."""
        engine = AIEngine(config)
        engine.state = "ready"

        # Mock components
        engine.protocol_discovery = Mock()
        engine.field_detector = Mock()
        engine.anomaly_detector = Mock()
        engine.model_registry = Mock()

        status = await engine.get_status()

        assert status["status"] == "ready"
        assert "uptime_seconds" in status
        assert "components" in status
        assert "system_metrics" in status

    @pytest.mark.asyncio
    async def test_cleanup(self, config):
        """Test engine cleanup."""
        engine = AIEngine(config)
        engine.state = "ready"

        # Mock components with cleanup methods
        mock_registry = AsyncMock()
        mock_registry.cleanup = AsyncMock()
        engine.model_registry = mock_registry

        await engine.cleanup()

        assert engine.state == "stopped"
        mock_registry.cleanup.assert_called_once()


class TestModelInput:
    """Test cases for ModelInput class."""

    def test_model_input_creation_with_bytes(self):
        """Test ModelInput creation with bytes data."""
        data = b"test data"
        model_input = ModelInput(data=data)

        assert model_input.data == data
        assert model_input.metadata == {}

    def test_model_input_creation_with_metadata(self):
        """Test ModelInput creation with metadata."""
        data = b"test data"
        metadata = {"protocol_hint": "http", "confidence_threshold": 0.8}

        model_input = ModelInput(data=data, metadata=metadata)

        assert model_input.data == data
        assert model_input.metadata == metadata

    def test_to_tensor_with_bytes(self):
        """Test tensor conversion with bytes data."""
        data = b"hello"
        model_input = ModelInput(data=data)

        tensor = model_input.to_tensor()

        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float
        assert len(tensor) == len(data)

    def test_to_tensor_with_numpy_array(self):
        """Test tensor conversion with numpy array."""
        data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        model_input = ModelInput(data=data)

        tensor = model_input.to_tensor()

        assert isinstance(tensor, torch.Tensor)
        assert torch.equal(tensor, torch.from_numpy(data))

    def test_to_tensor_with_string(self):
        """Test tensor conversion with string data."""
        data = "hello world"
        model_input = ModelInput(data=data)

        tensor = model_input.to_tensor()

        assert isinstance(tensor, torch.Tensor)
        assert len(tensor) == len(data.encode("utf-8"))


class TestModelOutput:
    """Test cases for ModelOutput class."""

    def test_model_output_creation(self):
        """Test ModelOutput creation."""
        predictions = torch.tensor([0.8, 0.2])
        confidence = torch.tensor([0.9, 0.1])
        metadata = {"model": "test_model"}

        output = ModelOutput(
            predictions=predictions, confidence=confidence, metadata=metadata
        )

        assert torch.equal(output.predictions, predictions)
        assert torch.equal(output.confidence, confidence)
        assert output.metadata == metadata

    def test_to_dict(self):
        """Test ModelOutput dictionary conversion."""
        predictions = torch.tensor([0.8, 0.2])
        confidence = torch.tensor([0.9])

        output = ModelOutput(predictions=predictions, confidence=confidence)
        result_dict = output.to_dict()

        assert "predictions" in result_dict
        assert "confidence" in result_dict
        assert isinstance(result_dict["predictions"], list)
        assert isinstance(result_dict["confidence"], list)

    def test_to_dict_with_dict_predictions(self):
        """Test ModelOutput dictionary conversion with dict predictions."""
        predictions = {
            "class_probs": torch.tensor([0.8, 0.2]),
            "features": torch.tensor([1.0, 2.0, 3.0]),
        }

        output = ModelOutput(predictions=predictions)
        result_dict = output.to_dict()

        assert isinstance(result_dict["predictions"], dict)
        assert "class_probs" in result_dict["predictions"]
        assert "features" in result_dict["predictions"]


class TestConfigValidation:
    """Test cases for configuration validation."""

    def test_default_config_creation(self):
        """Test default configuration creation."""
        config = Config()

        assert hasattr(config, "device")
        assert hasattr(config, "model_path")
        assert hasattr(config, "data_path")

    def test_config_with_custom_values(self):
        """Test configuration with custom values."""
        config = Config()
        config.device = "cuda"
        config.model_path = "/custom/models"
        config.batch_size = 64

        assert config.device == "cuda"
        assert config.model_path == "/custom/models"
        assert config.batch_size == 64

    def test_config_validation(self):
        """Test configuration validation."""
        config = Config()

        # Test invalid device
        with pytest.raises(ValueError):
            config.validate_device("invalid_device")

        # Test valid device
        assert config.validate_device("cpu") == "cpu"
        assert config.validate_device("cuda") == "cuda"


class TestExceptionHandling:
    """Test cases for exception handling."""

    def test_ai_engine_exception(self):
        """Test AIEngineException creation and handling."""
        message = "Test AI Engine error"
        exception = AIEngineException(message)

        assert str(exception) == message
        assert isinstance(exception, Exception)

    def test_validation_exception(self):
        """Test ValidationException creation and handling."""
        message = "Validation failed"
        exception = ValidationException(message)

        assert str(exception) == message
        assert isinstance(exception, AIEngineException)

    @pytest.mark.asyncio
    async def test_exception_propagation(self, config):
        """Test exception propagation in AI Engine."""
        engine = AIEngine(config)

        with patch.object(engine, "protocol_discovery") as mock_discovery:
            mock_discovery.discover_protocol.side_effect = RuntimeError("Model error")
            engine.state = "ready"

            model_input = ModelInput(data=b"test")

            with pytest.raises(AIEngineException):
                await engine.discover_protocol(model_input)


class TestPerformance:
    """Performance tests for core components."""

    @pytest.mark.asyncio
    async def test_engine_initialization_time(self, config):
        """Test engine initialization performance."""
        import time

        engine = AIEngine(config)

        with patch.object(engine, "_initialize_components", new_callable=AsyncMock):
            start_time = time.time()
            await engine.initialize()
            end_time = time.time()

            initialization_time = (end_time - start_time) * 1000  # Convert to ms
            assert initialization_time < TestConfig.PERFORMANCE_THRESHOLD_MS

    @pytest.mark.asyncio
    async def test_protocol_discovery_latency(self, config, sample_data):
        """Test protocol discovery latency."""
        import time

        engine = AIEngine(config)
        engine.state = "ready"

        # Mock fast discovery
        mock_discovery = AsyncMock()
        mock_discovery.discover_protocol.return_value = ModelOutput(
            predictions=torch.tensor([1.0]),
            metadata={"protocol_type": "http", "confidence": 0.9},
        )
        engine.protocol_discovery = mock_discovery

        model_input = ModelInput(data=sample_data["http_data"])

        start_time = time.time()
        result = await engine.discover_protocol(model_input)
        end_time = time.time()

        latency = (end_time - start_time) * 1000  # Convert to ms
        assert latency < TestConfig.PERFORMANCE_THRESHOLD_MS
        assert result.processing_time_ms is not None


class TestIntegration:
    """Integration tests for AI Engine components."""

    @pytest.mark.asyncio
    async def test_full_pipeline_integration(self, config, sample_data):
        """Test full AI Engine pipeline integration."""
        engine = AIEngine(config)

        # Mock all components
        mock_discovery = AsyncMock()
        mock_field_detector = AsyncMock()
        mock_anomaly_detector = AsyncMock()

        mock_discovery.discover_protocol.return_value = ModelOutput(
            predictions=torch.tensor([1.0]),
            metadata={"protocol_type": "http", "confidence": 0.9},
        )

        mock_field_detector.detect_fields.return_value = ModelOutput(
            predictions=torch.tensor([[1, 0]]),
            metadata={
                "detected_fields": [
                    {
                        "id": "1",
                        "start_offset": 0,
                        "end_offset": 3,
                        "field_type": "method",
                        "confidence": 0.9,
                    }
                ]
            },
        )

        mock_anomaly_detector.detect_anomalies.return_value = ModelOutput(
            predictions=torch.tensor([0.1]),
            metadata={"is_anomalous": False, "anomaly_score": {"overall_score": 0.1}},
        )

        engine.protocol_discovery = mock_discovery
        engine.field_detector = mock_field_detector
        engine.anomaly_detector = mock_anomaly_detector
        engine.state = "ready"

        # Test full pipeline
        model_input = ModelInput(data=sample_data["http_data"])

        # Run all three main functions
        protocol_result = await engine.discover_protocol(model_input)
        field_result = await engine.detect_fields(model_input)
        anomaly_result = await engine.detect_anomalies(model_input)

        # Verify results
        assert protocol_result.metadata["protocol_type"] == "http"
        assert len(field_result.metadata["detected_fields"]) == 1
        assert anomaly_result.metadata["is_anomalous"] is False

        # Verify all components were called
        mock_discovery.discover_protocol.assert_called_once()
        mock_field_detector.detect_fields.assert_called_once()
        mock_anomaly_detector.detect_anomalies.assert_called_once()


# Pytest fixtures and setup


@pytest.fixture
def config():
    """Create test configuration."""
    config = Config()
    config.model_path = "test_models"
    config.data_path = "test_data"
    config.device = "cpu"
    config.batch_size = 16
    return config


@pytest.fixture
def sample_data():
    """Create sample test data."""
    return {
        "http_data": b"GET /api/v1/users HTTP/1.1\r\nHost: example.com\r\n\r\n",
        "modbus_data": bytes([0x01, 0x03, 0x00, 0x00, 0x00, 0x02, 0xC4, 0x0B]),
        "hl7_data": b"MSH|^~\\&|SENDER|HOSPITAL|RECEIVER|CLINIC|20230101120000||ADT^A08|123456|P|2.5\r",
        "anomalous_data": np.random.bytes(256),
    }


@pytest.fixture
def temp_directory():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
