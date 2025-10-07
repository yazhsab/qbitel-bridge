"""
CRONOS AI Engine - Comprehensive APM Integration Tests

Complete test suite for Application Performance Monitoring integration
with Elastic APM, Datadog APM, and APM Manager functionality.
"""

import pytest
import asyncio
import time
import json
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
from typing import Dict, List, Any, Optional

from ai_engine.monitoring.apm_integration import (
    APMTransaction,
    TransactionType,
    ElasticAPMIntegration,
    DatadogAPMIntegration,
    APMManager,
    initialize_apm,
    get_apm_manager,
)
from ai_engine.core.config import Config
from ai_engine.core.exceptions import ObservabilityException


class TestAPMTransaction:
    """Test APMTransaction dataclass."""

    def test_apm_transaction_creation(self):
        """Test creating APMTransaction instance."""
        transaction = APMTransaction(
            transaction_id="txn-123",
            name="test_transaction",
            type=TransactionType.REQUEST,
            start_time=time.time(),
            context={"user_id": "user123", "request_id": "req456"}
        )
        
        assert transaction.transaction_id == "txn-123"
        assert transaction.name == "test_transaction"
        assert transaction.type == TransactionType.REQUEST
        assert transaction.context == {"user_id": "user123", "request_id": "req456"}
        assert transaction.result == "success"
        assert transaction.end_time is None
        assert transaction.duration_ms is None
        assert len(transaction.errors) == 0
        assert len(transaction.custom_metrics) == 0

    def test_apm_transaction_finish(self):
        """Test finishing APM transaction."""
        start_time = time.time()
        transaction = APMTransaction(
            transaction_id="txn-456",
            name="test_transaction",
            type=TransactionType.REQUEST,
            start_time=start_time
        )
        
        # Wait a bit to ensure duration > 0
        time.sleep(0.01)
        
        transaction.finish("success")
        
        assert transaction.result == "success"
        assert transaction.end_time is not None
        assert transaction.end_time > start_time
        assert transaction.duration_ms is not None
        assert transaction.duration_ms > 0

    def test_apm_transaction_finish_with_error(self):
        """Test finishing APM transaction with error result."""
        transaction = APMTransaction(
            transaction_id="txn-789",
            name="error_transaction",
            type=TransactionType.REQUEST,
            start_time=time.time()
        )
        
        transaction.finish("error")
        
        assert transaction.result == "error"

    def test_apm_transaction_add_error(self):
        """Test adding error to APM transaction."""
        transaction = APMTransaction(
            transaction_id="txn-error",
            name="error_transaction",
            type=TransactionType.REQUEST,
            start_time=time.time()
        )
        
        error = ValueError("Test error message")
        transaction.add_error(error, handled=True)
        
        assert len(transaction.errors) == 1
        error_data = transaction.errors[0]
        assert error_data["type"] == "ValueError"
        assert error_data["message"] == "Test error message"
        assert error_data["handled"] is True
        assert "timestamp" in error_data

    def test_apm_transaction_add_unhandled_error(self):
        """Test adding unhandled error to APM transaction."""
        transaction = APMTransaction(
            transaction_id="txn-unhandled",
            name="unhandled_error_transaction",
            type=TransactionType.REQUEST,
            start_time=time.time()
        )
        
        error = RuntimeError("Unhandled error")
        transaction.add_error(error, handled=False)
        
        assert len(transaction.errors) == 1
        error_data = transaction.errors[0]
        assert error_data["handled"] is False

    def test_apm_transaction_set_custom_metric(self):
        """Test setting custom metric on APM transaction."""
        transaction = APMTransaction(
            transaction_id="txn-metric",
            name="metric_transaction",
            type=TransactionType.REQUEST,
            start_time=time.time()
        )
        
        transaction.set_custom_metric("response_time", 150.5)
        transaction.set_custom_metric("memory_usage", 1024.0)
        
        assert transaction.custom_metrics["response_time"] == 150.5
        assert transaction.custom_metrics["memory_usage"] == 1024.0

    def test_apm_transaction_multiple_errors(self):
        """Test adding multiple errors to APM transaction."""
        transaction = APMTransaction(
            transaction_id="txn-multi-error",
            name="multi_error_transaction",
            type=TransactionType.REQUEST,
            start_time=time.time()
        )
        
        transaction.add_error(ValueError("First error"))
        transaction.add_error(RuntimeError("Second error"))
        transaction.add_error(KeyError("Third error"))
        
        assert len(transaction.errors) == 3
        assert transaction.errors[0]["type"] == "ValueError"
        assert transaction.errors[1]["type"] == "RuntimeError"
        assert transaction.errors[2]["type"] == "KeyError"


class TestElasticAPMIntegration:
    """Test ElasticAPMIntegration functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock(spec=Config)
        config.elastic_apm_server_url = "http://localhost:8200"
        config.elastic_apm_secret_token = "test_token"
        config.service_name = "test-service"
        config.environment = Mock()
        config.environment.value = "test"
        return config

    @pytest.fixture
    def elastic_apm(self, mock_config):
        """Create ElasticAPMIntegration instance."""
        return ElasticAPMIntegration(mock_config)

    def test_elastic_apm_initialization(self, elastic_apm, mock_config):
        """Test ElasticAPMIntegration initialization."""
        assert elastic_apm.apm_server_url == "http://localhost:8200"
        assert elastic_apm.apm_secret_token == "test_token"
        assert elastic_apm.service_name == "test-service"
        assert elastic_apm.environment == "test"
        assert elastic_apm._batch_size == 50
        assert elastic_apm._flush_interval == 10
        assert elastic_apm.transactions_sent == 0
        assert elastic_apm.errors_sent == 0

    @pytest.mark.asyncio
    async def test_elastic_apm_initialize(self, elastic_apm):
        """Test ElasticAPMIntegration initialization."""
        with patch('aiohttp.ClientSession') as mock_session:
            await elastic_apm.initialize()
            
            assert elastic_apm._session is not None
            assert elastic_apm._flush_task is not None
            mock_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_elastic_apm_shutdown(self, elastic_apm):
        """Test ElasticAPMIntegration shutdown."""
        # Mock session and flush task
        elastic_apm._session = AsyncMock()
        elastic_apm._flush_task = AsyncMock()
        elastic_apm._flush_task.cancel = Mock()
        
        await elastic_apm.shutdown()
        
        elastic_apm._flush_task.cancel.assert_called_once()
        elastic_apm._session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_elastic_apm_send_transaction(self, elastic_apm):
        """Test sending transaction to Elastic APM."""
        transaction = APMTransaction(
            transaction_id="txn-123",
            name="test_transaction",
            type=TransactionType.REQUEST,
            start_time=time.time()
        )
        transaction.finish("success")
        
        with patch.object(elastic_apm, '_flush_batch') as mock_flush:
            await elastic_apm.send_transaction(transaction)
            
            assert len(elastic_apm._transaction_queue) == 1
            assert elastic_apm._transaction_queue[0] == transaction

    @pytest.mark.asyncio
    async def test_elastic_apm_send_transaction_batch_flush(self, elastic_apm):
        """Test sending transaction triggers batch flush."""
        # Fill queue to batch size
        for i in range(elastic_apm._batch_size):
            transaction = APMTransaction(
                transaction_id=f"txn-{i}",
                name=f"transaction_{i}",
                type=TransactionType.REQUEST,
                start_time=time.time()
            )
            transaction.finish("success")
            elastic_apm._transaction_queue.append(transaction)
        
        with patch.object(elastic_apm, '_flush_batch') as mock_flush:
            # Add one more transaction to trigger flush
            extra_transaction = APMTransaction(
                transaction_id="txn-extra",
                name="extra_transaction",
                type=TransactionType.REQUEST,
                start_time=time.time()
            )
            extra_transaction.finish("success")
            
            await elastic_apm.send_transaction(extra_transaction)
            
            mock_flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_elastic_apm_flush(self, elastic_apm):
        """Test flushing transactions."""
        # Add some transactions
        for i in range(3):
            transaction = APMTransaction(
                transaction_id=f"txn-{i}",
                name=f"transaction_{i}",
                type=TransactionType.REQUEST,
                start_time=time.time()
            )
            transaction.finish("success")
            elastic_apm._transaction_queue.append(transaction)
        
        with patch.object(elastic_apm, '_flush_batch') as mock_flush:
            await elastic_apm.flush()
            mock_flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_elastic_apm_flush_loop(self, elastic_apm):
        """Test background flush loop."""
        elastic_apm._session = AsyncMock()
        elastic_apm._transaction_queue.append(Mock())
        
        with patch.object(elastic_apm, '_flush_batch') as mock_flush:
            # Start flush loop
            task = asyncio.create_task(elastic_apm._flush_loop())
            
            # Wait a bit for the loop to run
            await asyncio.sleep(0.1)
            
            # Cancel the task
            task.cancel()
            
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            # Should have called flush at least once
            mock_flush.assert_called()

    @pytest.mark.asyncio
    async def test_elastic_apm_flush_batch_success(self, elastic_apm):
        """Test successful batch flush to Elastic APM."""
        # Create test transactions
        transactions = []
        for i in range(3):
            transaction = APMTransaction(
                transaction_id=f"txn-{i}",
                name=f"transaction_{i}",
                type=TransactionType.REQUEST,
                start_time=time.time()
            )
            transaction.finish("success")
            transactions.append(transaction)
        
        elastic_apm._transaction_queue.extend(transactions)
        elastic_apm._session = AsyncMock()
        
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 202
        elastic_apm._session.post.return_value.__aenter__.return_value = mock_response
        
        await elastic_apm._flush_batch()
        
        assert len(elastic_apm._transaction_queue) == 0
        assert elastic_apm.transactions_sent == 3
        elastic_apm._session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_elastic_apm_flush_batch_with_errors(self, elastic_apm):
        """Test batch flush with transaction errors."""
        # Create transaction with error
        transaction = APMTransaction(
            transaction_id="txn-error",
            name="error_transaction",
            type=TransactionType.REQUEST,
            start_time=time.time()
        )
        transaction.add_error(ValueError("Test error"))
        transaction.finish("error")
        
        elastic_apm._transaction_queue.append(transaction)
        elastic_apm._session = AsyncMock()
        
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 202
        elastic_apm._session.post.return_value.__aenter__.return_value = mock_response
        
        await elastic_apm._flush_batch()
        
        assert elastic_apm.transactions_sent == 1
        assert elastic_apm.errors_sent == 1

    @pytest.mark.asyncio
    async def test_elastic_apm_flush_batch_failure(self, elastic_apm):
        """Test batch flush failure handling."""
        transaction = APMTransaction(
            transaction_id="txn-fail",
            name="fail_transaction",
            type=TransactionType.REQUEST,
            start_time=time.time()
        )
        transaction.finish("success")
        
        elastic_apm._transaction_queue.append(transaction)
        elastic_apm._session = AsyncMock()
        
        # Mock failed response
        mock_response = AsyncMock()
        mock_response.status = 500
        elastic_apm._session.post.return_value.__aenter__.return_value = mock_response
        
        # Should not raise exception
        await elastic_apm._flush_batch()
        
        elastic_apm._session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_elastic_apm_flush_batch_exception(self, elastic_apm):
        """Test batch flush exception handling."""
        transaction = APMTransaction(
            transaction_id="txn-exception",
            name="exception_transaction",
            type=TransactionType.REQUEST,
            start_time=time.time()
        )
        transaction.finish("success")
        
        elastic_apm._transaction_queue.append(transaction)
        elastic_apm._session = AsyncMock()
        
        # Mock exception
        elastic_apm._session.post.side_effect = Exception("Network error")
        
        # Should not raise exception
        await elastic_apm._flush_batch()
        
        elastic_apm._session.post.assert_called_once()


class TestDatadogAPMIntegration:
    """Test DatadogAPMIntegration functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock(spec=Config)
        config.datadog_agent_url = "http://localhost:8126"
        config.datadog_api_key = "test_dd_key"
        config.service_name = "test-service"
        config.environment = Mock()
        config.environment.value = "test"
        return config

    @pytest.fixture
    def datadog_apm(self, mock_config):
        """Create DatadogAPMIntegration instance."""
        return DatadogAPMIntegration(mock_config)

    def test_datadog_apm_initialization(self, datadog_apm, mock_config):
        """Test DatadogAPMIntegration initialization."""
        assert datadog_apm.dd_agent_url == "http://localhost:8126"
        assert datadog_apm.dd_api_key == "test_dd_key"
        assert datadog_apm.service_name == "test-service"
        assert datadog_apm.environment == "test"
        assert datadog_apm._batch_size == 50
        assert datadog_apm._flush_interval == 10
        assert datadog_apm.transactions_sent == 0

    @pytest.mark.asyncio
    async def test_datadog_apm_initialize(self, datadog_apm):
        """Test DatadogAPMIntegration initialization."""
        with patch('aiohttp.ClientSession') as mock_session:
            await datadog_apm.initialize()
            
            assert datadog_apm._session is not None
            assert datadog_apm._flush_task is not None
            mock_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_datadog_apm_shutdown(self, datadog_apm):
        """Test DatadogAPMIntegration shutdown."""
        # Mock session and flush task
        datadog_apm._session = AsyncMock()
        datadog_apm._flush_task = AsyncMock()
        datadog_apm._flush_task.cancel = Mock()
        
        await datadog_apm.shutdown()
        
        datadog_apm._flush_task.cancel.assert_called_once()
        datadog_apm._session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_datadog_apm_send_transaction(self, datadog_apm):
        """Test sending transaction to Datadog APM."""
        transaction = APMTransaction(
            transaction_id="txn-123",
            name="test_transaction",
            type=TransactionType.REQUEST,
            start_time=time.time()
        )
        transaction.finish("success")
        
        with patch.object(datadog_apm, '_flush_batch') as mock_flush:
            await datadog_apm.send_transaction(transaction)
            
            assert len(datadog_apm._transaction_queue) == 1
            assert datadog_apm._transaction_queue[0] == transaction

    @pytest.mark.asyncio
    async def test_datadog_apm_flush_batch_success(self, datadog_apm):
        """Test successful batch flush to Datadog."""
        # Create test transactions
        transactions = []
        for i in range(3):
            transaction = APMTransaction(
                transaction_id=f"txn-{i}",
                name=f"transaction_{i}",
                type=TransactionType.REQUEST,
                start_time=time.time()
            )
            transaction.finish("success")
            transactions.append(transaction)
        
        datadog_apm._transaction_queue.extend(transactions)
        datadog_apm._session = AsyncMock()
        
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 200
        datadog_apm._session.put.return_value.__aenter__.return_value = mock_response
        
        await datadog_apm._flush_batch()
        
        assert len(datadog_apm._transaction_queue) == 0
        assert datadog_apm.transactions_sent == 3
        datadog_apm._session.put.assert_called_once()

    @pytest.mark.asyncio
    async def test_datadog_apm_flush_batch_failure(self, datadog_apm):
        """Test batch flush failure handling."""
        transaction = APMTransaction(
            transaction_id="txn-fail",
            name="fail_transaction",
            type=TransactionType.REQUEST,
            start_time=time.time()
        )
        transaction.finish("success")
        
        datadog_apm._transaction_queue.append(transaction)
        datadog_apm._session = AsyncMock()
        
        # Mock failed response
        mock_response = AsyncMock()
        mock_response.status = 500
        datadog_apm._session.put.return_value.__aenter__.return_value = mock_response
        
        # Should not raise exception
        await datadog_apm._flush_batch()
        
        datadog_apm._session.put.assert_called_once()


class TestAPMManager:
    """Test APMManager functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock(spec=Config)
        config.enable_elastic_apm = True
        config.enable_datadog_apm = True
        config.service_name = "test-service"
        config.environment = Mock()
        config.environment.value = "test"
        return config

    @pytest.fixture
    def apm_manager(self, mock_config):
        """Create APMManager instance."""
        with patch('ai_engine.monitoring.apm_integration.ElasticAPMIntegration'), \
             patch('ai_engine.monitoring.apm_integration.DatadogAPMIntegration'):
            return APMManager(mock_config)

    def test_apm_manager_initialization(self, apm_manager, mock_config):
        """Test APMManager initialization."""
        assert len(apm_manager.integrations) == 2
        assert len(apm_manager._active_transactions) == 0
        assert apm_manager._metrics_task is None

    @pytest.mark.asyncio
    async def test_apm_manager_initialize(self, apm_manager):
        """Test APMManager initialization."""
        # Mock integrations
        for integration in apm_manager.integrations:
            integration.initialize = AsyncMock()
        
        with patch('asyncio.create_task') as mock_create_task:
            await apm_manager.initialize()
            
            # Should initialize all integrations
            for integration in apm_manager.integrations:
                integration.initialize.assert_called_once()
            
            # Should start metrics collection
            mock_create_task.assert_called()

    @pytest.mark.asyncio
    async def test_apm_manager_shutdown(self, apm_manager):
        """Test APMManager shutdown."""
        # Mock metrics task
        apm_manager._metrics_task = AsyncMock()
        apm_manager._metrics_task.cancel = Mock()
        
        # Mock integrations
        for integration in apm_manager.integrations:
            integration.shutdown = AsyncMock()
        
        await apm_manager.shutdown()
        
        # Should cancel metrics task
        apm_manager._metrics_task.cancel.assert_called_once()
        
        # Should shutdown all integrations
        for integration in apm_manager.integrations:
            integration.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_apm_manager_transaction_context(self, apm_manager):
        """Test APM transaction context manager."""
        # Mock integrations
        for integration in apm_manager.integrations:
            integration.send_transaction = AsyncMock()
        
        async with apm_manager.transaction("test_operation", TransactionType.REQUEST) as txn:
            assert txn.name == "test_operation"
            assert txn.type == TransactionType.REQUEST
            assert txn.transaction_id in apm_manager._active_transactions
            
            # Add some context
            txn.set_custom_metric("test_metric", 123.45)
        
        # Should send transaction to all integrations
        for integration in apm_manager.integrations:
            integration.send_transaction.assert_called_once()
        
        # Should remove from active transactions
        assert len(apm_manager._active_transactions) == 0

    @pytest.mark.asyncio
    async def test_apm_manager_transaction_with_error(self, apm_manager):
        """Test APM transaction context manager with error."""
        # Mock integrations
        for integration in apm_manager.integrations:
            integration.send_transaction = AsyncMock()
        
        with pytest.raises(ValueError):
            async with apm_manager.transaction("error_operation", TransactionType.REQUEST) as txn:
                assert txn.name == "error_operation"
                raise ValueError("Test error")
        
        # Should send transaction with error to all integrations
        for integration in apm_manager.integrations:
            integration.send_transaction.assert_called_once()
        
        # Should remove from active transactions
        assert len(apm_manager._active_transactions) == 0

    @pytest.mark.asyncio
    async def test_apm_manager_transaction_with_context(self, apm_manager):
        """Test APM transaction context manager with context."""
        # Mock integrations
        for integration in apm_manager.integrations:
            integration.send_transaction = AsyncMock()
        
        context = {"user_id": "user123", "request_id": "req456"}
        
        async with apm_manager.transaction("context_operation", TransactionType.REQUEST, context) as txn:
            assert txn.context == context
        
        # Should send transaction to all integrations
        for integration in apm_manager.integrations:
            integration.send_transaction.assert_called_once()

    @pytest.mark.asyncio
    async def test_apm_manager_collect_system_metrics(self, apm_manager):
        """Test system metrics collection."""
        with patch('psutil.cpu_percent') as mock_cpu, \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk, \
             patch('psutil.net_io_counters') as mock_net:
            
            # Mock system metrics
            mock_cpu.return_value = 25.5
            mock_memory.return_value = Mock(percent=60.0, available=1024*1024*1024)
            mock_disk.return_value = Mock(percent=45.0, free=1024*1024*1024*1024)
            mock_net.return_value = Mock(bytes_sent=1000, bytes_recv=2000)
            
            await apm_manager._collect_system_metrics()
            
            assert "cpu_percent" in apm_manager._system_metrics
            assert "memory_percent" in apm_manager._system_metrics
            assert "disk_percent" in apm_manager._system_metrics
            assert "network_bytes_sent" in apm_manager._system_metrics
            assert "network_bytes_recv" in apm_manager._system_metrics

    def test_apm_manager_get_system_metrics(self, apm_manager):
        """Test getting system metrics."""
        # Set some test metrics
        apm_manager._system_metrics = {
            "cpu_percent": 25.5,
            "memory_percent": 60.0,
            "disk_percent": 45.0
        }
        
        metrics = apm_manager.get_system_metrics()
        
        assert metrics["cpu_percent"] == 25.5
        assert metrics["memory_percent"] == 60.0
        assert metrics["disk_percent"] == 45.0

    def test_apm_manager_get_statistics(self, apm_manager):
        """Test getting APM statistics."""
        # Mock integration statistics
        apm_manager.integrations[0].transactions_sent = 100
        apm_manager.integrations[0].errors_sent = 5
        apm_manager.integrations[1].transactions_sent = 150
        
        apm_manager._system_metrics = {"cpu_percent": 25.5}
        
        stats = apm_manager.get_statistics()
        
        assert stats["active_transactions"] == 0
        assert len(stats["integrations"]) == 2
        assert stats["system_metrics"]["cpu_percent"] == 25.5


class TestAPMIntegration:
    """Integration tests for APM components."""

    @pytest.mark.asyncio
    async def test_initialize_apm_global(self):
        """Test global APM initialization."""
        mock_config = Mock(spec=Config)
        mock_config.enable_elastic_apm = True
        mock_config.enable_datadog_apm = False
        
        with patch('ai_engine.monitoring.apm_integration.APMManager') as mock_manager_class:
            mock_manager = AsyncMock()
            mock_manager_class.return_value = mock_manager
            
            manager = await initialize_apm(mock_config)
            
            assert manager == mock_manager
            mock_manager.initialize.assert_called_once()

    def test_get_apm_manager(self):
        """Test getting global APM manager."""
        # Test when no manager is set
        from ai_engine.monitoring.apm_integration import _apm_manager
        original_manager = _apm_manager
        
        try:
            # Clear global manager
            import ai_engine.monitoring.apm_integration
            ai_engine.monitoring.apm_integration._apm_manager = None
            
            manager = get_apm_manager()
            assert manager is None
            
            # Set a mock manager
            mock_manager = Mock()
            ai_engine.monitoring.apm_integration._apm_manager = mock_manager
            
            manager = get_apm_manager()
            assert manager == mock_manager
            
        finally:
            # Restore original manager
            ai_engine.monitoring.apm_integration._apm_manager = original_manager

    @pytest.mark.asyncio
    async def test_apm_transaction_lifecycle(self):
        """Test complete APM transaction lifecycle."""
        mock_config = Mock(spec=Config)
        mock_config.enable_elastic_apm = True
        mock_config.enable_datadog_apm = True
        mock_config.service_name = "test-service"
        mock_config.environment = Mock()
        mock_config.environment.value = "test"
        
        with patch('ai_engine.monitoring.apm_integration.ElasticAPMIntegration') as mock_elastic, \
             patch('ai_engine.monitoring.apm_integration.DatadogAPMIntegration') as mock_datadog:
            
            # Mock integrations
            elastic_integration = AsyncMock()
            datadog_integration = AsyncMock()
            mock_elastic.return_value = elastic_integration
            mock_datadog.return_value = datadog_integration
            
            manager = APMManager(mock_config)
            
            # Test transaction
            async with manager.transaction("test_operation", TransactionType.REQUEST) as txn:
                txn.set_custom_metric("response_time", 150.0)
                txn.add_error(ValueError("Test error"), handled=True)
            
            # Should send to both integrations
            elastic_integration.send_transaction.assert_called_once()
            datadog_integration.send_transaction.assert_called_once()

    @pytest.mark.asyncio
    async def test_apm_concurrent_transactions(self):
        """Test concurrent APM transactions."""
        mock_config = Mock(spec=Config)
        mock_config.enable_elastic_apm = True
        mock_config.enable_datadog_apm = False
        mock_config.service_name = "test-service"
        mock_config.environment = Mock()
        mock_config.environment.value = "test"
        
        with patch('ai_engine.monitoring.apm_integration.ElasticAPMIntegration') as mock_elastic:
            elastic_integration = AsyncMock()
            mock_elastic.return_value = elastic_integration
            
            manager = APMManager(mock_config)
            
            # Create multiple concurrent transactions
            async def create_transaction(transaction_id):
                async with manager.transaction(f"operation_{transaction_id}", TransactionType.REQUEST) as txn:
                    txn.set_custom_metric("transaction_id", transaction_id)
                    await asyncio.sleep(0.01)  # Simulate work
            
            # Run concurrent transactions
            tasks = [create_transaction(i) for i in range(10)]
            await asyncio.gather(*tasks)
            
            # Should have sent 10 transactions
            assert elastic_integration.send_transaction.call_count == 10

    def test_apm_transaction_types(self):
        """Test different APM transaction types."""
        transaction_types = [
            TransactionType.REQUEST,
            TransactionType.BACKGROUND,
            TransactionType.SCHEDULED,
            TransactionType.MESSAGE
        ]
        
        for txn_type in transaction_types:
            transaction = APMTransaction(
                transaction_id=f"txn-{txn_type.value}",
                name=f"test_{txn_type.value}",
                type=txn_type,
                start_time=time.time()
            )
            
            assert transaction.type == txn_type
            assert transaction.name == f"test_{txn_type.value}"

    @pytest.mark.asyncio
    async def test_apm_error_handling(self):
        """Test APM error handling scenarios."""
        mock_config = Mock(spec=Config)
        mock_config.enable_elastic_apm = True
        mock_config.enable_datadog_apm = False
        mock_config.service_name = "test-service"
        mock_config.environment = Mock()
        mock_config.environment.value = "test"
        
        with patch('ai_engine.monitoring.apm_integration.ElasticAPMIntegration') as mock_elastic:
            elastic_integration = AsyncMock()
            elastic_integration.send_transaction.side_effect = Exception("APM error")
            mock_elastic.return_value = elastic_integration
            
            manager = APMManager(mock_config)
            
            # Should not raise exception even if APM fails
            async with manager.transaction("error_operation", TransactionType.REQUEST) as txn:
                txn.set_custom_metric("test", 123.0)
            
            # Should have attempted to send
            elastic_integration.send_transaction.assert_called_once()

    def test_apm_transaction_serialization(self):
        """Test APM transaction data serialization."""
        transaction = APMTransaction(
            transaction_id="txn-serialization",
            name="serialization_test",
            type=TransactionType.REQUEST,
            start_time=time.time(),
            context={"key": "value", "number": 123}
        )
        transaction.finish("success")
        transaction.set_custom_metric("metric1", 456.789)
        transaction.add_error(ValueError("Test error"))
        
        # Test that transaction can be converted to dict-like structure
        transaction_data = {
            "transaction_id": transaction.transaction_id,
            "name": transaction.name,
            "type": transaction.type.value,
            "duration_ms": transaction.duration_ms,
            "result": transaction.result,
            "context": transaction.context,
            "custom_metrics": transaction.custom_metrics,
            "errors": transaction.errors
        }
        
        assert transaction_data["transaction_id"] == "txn-serialization"
        assert transaction_data["name"] == "serialization_test"
        assert transaction_data["type"] == "request"
        assert transaction_data["result"] == "success"
        assert transaction_data["context"]["key"] == "value"
        assert transaction_data["custom_metrics"]["metric1"] == 456.789
        assert len(transaction_data["errors"]) == 1
