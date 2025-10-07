"""
CRONOS AI Engine - Config Service Comprehensive Tests

Comprehensive test suite for the centralized configuration service.
"""

import pytest
import asyncio
import json
import tempfile
import os
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, List, Any, Optional

from ai_engine.core.config_service import (
    ConfigService,
    ConfigVersion,
    ConfigChangeEvent,
    ConfigWatcher,
    EtcdConfigBackend,
    FileConfigBackend,
    MemoryConfigBackend,
    ConfigValidationError,
    ConfigNotFoundException,
    ConfigServiceConfig,
)


class TestConfigVersion:
    """Test ConfigVersion dataclass."""

    def test_config_version_creation(self):
        """Test creating ConfigVersion instance."""
        version = ConfigVersion(
            version_id="v1.0.0",
            timestamp=datetime.now(),
            author="test_user",
            description="Test version",
            config_data={"key": "value"},
            checksum="abc123"
        )
        
        assert version.version_id == "v1.0.0"
        assert version.author == "test_user"
        assert version.description == "Test version"
        assert version.config_data == {"key": "value"}
        assert version.checksum == "abc123"

    def test_config_version_checksum_calculation(self):
        """Test checksum calculation."""
        config_data = {"key": "value", "nested": {"inner": "data"}}
        version = ConfigVersion(
            version_id="v1.0.0",
            timestamp=datetime.now(),
            author="test_user",
            description="Test version",
            config_data=config_data,
            checksum=""
        )
        
        # Checksum should be calculated automatically
        assert version.checksum != ""

    def test_config_version_serialization(self):
        """Test ConfigVersion serialization."""
        version = ConfigVersion(
            version_id="v1.0.0",
            timestamp=datetime.now(),
            author="test_user",
            description="Test version",
            config_data={"key": "value"},
            checksum="abc123"
        )
        
        serialized = version.to_dict()
        assert serialized["version_id"] == "v1.0.0"
        assert serialized["author"] == "test_user"
        assert serialized["config_data"] == {"key": "value"}

    def test_config_version_deserialization(self):
        """Test ConfigVersion deserialization."""
        data = {
            "version_id": "v1.0.0",
            "timestamp": datetime.now().isoformat(),
            "author": "test_user",
            "description": "Test version",
            "config_data": {"key": "value"},
            "checksum": "abc123"
        }
        
        version = ConfigVersion.from_dict(data)
        assert version.version_id == "v1.0.0"
        assert version.author == "test_user"
        assert version.config_data == {"key": "value"}


class TestConfigChangeEvent:
    """Test ConfigChangeEvent dataclass."""

    def test_config_change_event_creation(self):
        """Test creating ConfigChangeEvent instance."""
        event = ConfigChangeEvent(
            event_id="event123",
            timestamp=datetime.now(),
            event_type="UPDATE",
            key="test.key",
            old_value="old_value",
            new_value="new_value",
            source="test_source"
        )
        
        assert event.event_id == "event123"
        assert event.event_type == "UPDATE"
        assert event.key == "test.key"
        assert event.old_value == "old_value"
        assert event.new_value == "new_value"
        assert event.source == "test_source"

    def test_config_change_event_types(self):
        """Test different event types."""
        event_types = ["CREATE", "UPDATE", "DELETE", "RESTORE"]
        
        for event_type in event_types:
            event = ConfigChangeEvent(
                event_id=f"event_{event_type}",
                timestamp=datetime.now(),
                event_type=event_type,
                key="test.key",
                old_value=None,
                new_value="new_value",
                source="test_source"
            )
            
            assert event.event_type == event_type


class TestConfigWatcher:
    """Test ConfigWatcher functionality."""

    @pytest.fixture
    def config_watcher(self):
        """Create ConfigWatcher instance."""
        return ConfigWatcher()

    def test_config_watcher_initialization(self, config_watcher):
        """Test ConfigWatcher initialization."""
        assert config_watcher is not None
        assert config_watcher.callbacks == []

    def test_add_callback(self, config_watcher):
        """Test adding callback."""
        callback = Mock()
        config_watcher.add_callback(callback)
        
        assert callback in config_watcher.callbacks

    def test_remove_callback(self, config_watcher):
        """Test removing callback."""
        callback = Mock()
        config_watcher.add_callback(callback)
        config_watcher.remove_callback(callback)
        
        assert callback not in config_watcher.callbacks

    def test_notify_callbacks(self, config_watcher):
        """Test notifying callbacks."""
        callback1 = Mock()
        callback2 = Mock()
        config_watcher.add_callback(callback1)
        config_watcher.add_callback(callback2)
        
        event = ConfigChangeEvent(
            event_id="test_event",
            timestamp=datetime.now(),
            event_type="UPDATE",
            key="test.key",
            old_value="old",
            new_value="new",
            source="test"
        )
        
        config_watcher.notify_callbacks(event)
        
        callback1.assert_called_once_with(event)
        callback2.assert_called_once_with(event)

    def test_clear_callbacks(self, config_watcher):
        """Test clearing all callbacks."""
        callback1 = Mock()
        callback2 = Mock()
        config_watcher.add_callback(callback1)
        config_watcher.add_callback(callback2)
        
        config_watcher.clear_callbacks()
        
        assert config_watcher.callbacks == []


class TestMemoryConfigBackend:
    """Test MemoryConfigBackend implementation."""

    @pytest.fixture
    def memory_backend(self):
        """Create MemoryConfigBackend instance."""
        return MemoryConfigBackend()

    def test_memory_backend_initialization(self, memory_backend):
        """Test MemoryConfigBackend initialization."""
        assert memory_backend is not None
        assert memory_backend.config_data == {}

    def test_set_config(self, memory_backend):
        """Test setting configuration."""
        memory_backend.set_config("test.key", "test_value")
        
        assert memory_backend.config_data["test.key"] == "test_value"

    def test_get_config(self, memory_backend):
        """Test getting configuration."""
        memory_backend.set_config("test.key", "test_value")
        
        value = memory_backend.get_config("test.key")
        assert value == "test_value"

    def test_get_config_not_found(self, memory_backend):
        """Test getting non-existent configuration."""
        with pytest.raises(ConfigNotFoundException):
            memory_backend.get_config("nonexistent.key")

    def test_delete_config(self, memory_backend):
        """Test deleting configuration."""
        memory_backend.set_config("test.key", "test_value")
        memory_backend.delete_config("test.key")
        
        with pytest.raises(ConfigNotFoundException):
            memory_backend.get_config("test.key")

    def test_list_configs(self, memory_backend):
        """Test listing configurations."""
        memory_backend.set_config("key1", "value1")
        memory_backend.set_config("key2", "value2")
        
        configs = memory_backend.list_configs()
        assert "key1" in configs
        assert "key2" in configs

    def test_clear_configs(self, memory_backend):
        """Test clearing all configurations."""
        memory_backend.set_config("key1", "value1")
        memory_backend.set_config("key2", "value2")
        
        memory_backend.clear_configs()
        
        assert memory_backend.config_data == {}


class TestFileConfigBackend:
    """Test FileConfigBackend implementation."""

    @pytest.fixture
    def temp_config_file(self):
        """Create temporary config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"test.key": "test_value"}, f)
            return f.name

    @pytest.fixture
    def file_backend(self, temp_config_file):
        """Create FileConfigBackend instance."""
        return FileConfigBackend(temp_config_file)

    def test_file_backend_initialization(self, file_backend, temp_config_file):
        """Test FileConfigBackend initialization."""
        assert file_backend is not None
        assert file_backend.config_file == temp_config_file

    def test_load_config_from_file(self, file_backend):
        """Test loading configuration from file."""
        value = file_backend.get_config("test.key")
        assert value == "test_value"

    def test_save_config_to_file(self, file_backend):
        """Test saving configuration to file."""
        file_backend.set_config("new.key", "new_value")
        
        # Reload from file
        new_backend = FileConfigBackend(file_backend.config_file)
        value = new_backend.get_config("new.key")
        assert value == "new_value"

    def test_file_backend_persistence(self, temp_config_file):
        """Test file backend persistence."""
        backend1 = FileConfigBackend(temp_config_file)
        backend1.set_config("persistent.key", "persistent_value")
        
        backend2 = FileConfigBackend(temp_config_file)
        value = backend2.get_config("persistent.key")
        assert value == "persistent_value"

    def test_file_backend_with_nonexistent_file(self):
        """Test FileConfigBackend with non-existent file."""
        backend = FileConfigBackend("nonexistent.json")
        
        # Should start with empty config
        with pytest.raises(ConfigNotFoundException):
            backend.get_config("any.key")

    def test_file_backend_invalid_json(self):
        """Test FileConfigBackend with invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            invalid_file = f.name
        
        try:
            backend = FileConfigBackend(invalid_file)
            # Should handle invalid JSON gracefully
            assert backend.config_data == {}
        finally:
            os.unlink(invalid_file)


class TestEtcdConfigBackend:
    """Test EtcdConfigBackend implementation."""

    @pytest.fixture
    def mock_etcd_client(self):
        """Create mock etcd client."""
        client = Mock()
        client.get = Mock()
        client.put = Mock()
        client.delete = Mock()
        client.get_prefix = Mock()
        return client

    @pytest.fixture
    def etcd_backend(self, mock_etcd_client):
        """Create EtcdConfigBackend with mock client."""
        with patch('ai_engine.core.config_service.etcd3.client', return_value=mock_etcd_client):
            return EtcdConfigBackend(host="localhost", port=2379)

    def test_etcd_backend_initialization(self, etcd_backend):
        """Test EtcdConfigBackend initialization."""
        assert etcd_backend is not None
        assert etcd_backend.host == "localhost"
        assert etcd_backend.port == 2379

    def test_etcd_backend_set_config(self, etcd_backend, mock_etcd_client):
        """Test setting configuration in etcd."""
        etcd_backend.set_config("test.key", "test_value")
        
        mock_etcd_client.put.assert_called_once_with("test.key", "test_value")

    def test_etcd_backend_get_config(self, etcd_backend, mock_etcd_client):
        """Test getting configuration from etcd."""
        mock_response = Mock()
        mock_response.value = b"test_value"
        mock_etcd_client.get.return_value = mock_response
        
        value = etcd_backend.get_config("test.key")
        assert value == "test_value"

    def test_etcd_backend_get_config_not_found(self, etcd_backend, mock_etcd_client):
        """Test getting non-existent configuration from etcd."""
        mock_etcd_client.get.return_value = None
        
        with pytest.raises(ConfigNotFoundException):
            etcd_backend.get_config("nonexistent.key")

    def test_etcd_backend_delete_config(self, etcd_backend, mock_etcd_client):
        """Test deleting configuration from etcd."""
        etcd_backend.delete_config("test.key")
        
        mock_etcd_client.delete.assert_called_once_with("test.key")

    def test_etcd_backend_list_configs(self, etcd_backend, mock_etcd_client):
        """Test listing configurations from etcd."""
        mock_responses = [
            Mock(key=b"key1", value=b"value1"),
            Mock(key=b"key2", value=b"value2")
        ]
        mock_etcd_client.get_prefix.return_value = mock_responses
        
        configs = etcd_backend.list_configs()
        assert "key1" in configs
        assert "key2" in configs

    def test_etcd_backend_connection_error(self):
        """Test EtcdConfigBackend with connection error."""
        with patch('ai_engine.core.config_service.etcd3.client', side_effect=Exception("Connection failed")):
            with pytest.raises(Exception):
                EtcdConfigBackend(host="invalid", port=2379)


class TestConfigService:
    """Test ConfigService main functionality."""

    @pytest.fixture
    def mock_backend(self):
        """Create mock backend."""
        backend = Mock()
        backend.get_config = Mock()
        backend.set_config = Mock()
        backend.delete_config = Mock()
        backend.list_configs = Mock(return_value={})
        return backend

    @pytest.fixture
    def config_service(self, mock_backend):
        """Create ConfigService instance."""
        config = ConfigServiceConfig(
            backend_type="memory",
            cache_ttl=300,
            validation_enabled=True
        )
        return ConfigService(config, backend=mock_backend)

    def test_config_service_initialization(self, config_service):
        """Test ConfigService initialization."""
        assert config_service is not None
        assert config_service.config is not None
        assert config_service.backend is not None
        assert config_service.cache is not None
        assert config_service.watcher is not None

    def test_get_config_success(self, config_service, mock_backend):
        """Test successful config retrieval."""
        mock_backend.get_config.return_value = "test_value"
        
        value = config_service.get_config("test.key")
        
        assert value == "test_value"
        mock_backend.get_config.assert_called_once_with("test.key")

    def test_get_config_cached(self, config_service, mock_backend):
        """Test config retrieval from cache."""
        mock_backend.get_config.return_value = "test_value"
        
        # First call - from backend
        value1 = config_service.get_config("test.key")
        
        # Second call - from cache
        value2 = config_service.get_config("test.key")
        
        assert value1 == value2 == "test_value"
        # Backend should only be called once due to caching
        mock_backend.get_config.assert_called_once_with("test.key")

    def test_get_config_not_found(self, config_service, mock_backend):
        """Test config retrieval when not found."""
        mock_backend.get_config.side_effect = ConfigNotFoundException("Key not found")
        
        with pytest.raises(ConfigNotFoundException):
            config_service.get_config("nonexistent.key")

    def test_set_config_success(self, config_service, mock_backend):
        """Test successful config setting."""
        config_service.set_config("test.key", "test_value")
        
        mock_backend.set_config.assert_called_once_with("test.key", "test_value")

    def test_set_config_with_validation(self, config_service, mock_backend):
        """Test config setting with validation."""
        # Add validation rule
        config_service.add_validation_rule("test.key", lambda x: isinstance(x, str))
        
        config_service.set_config("test.key", "valid_string")
        
        mock_backend.set_config.assert_called_once_with("test.key", "valid_string")

    def test_set_config_validation_failure(self, config_service, mock_backend):
        """Test config setting with validation failure."""
        # Add validation rule
        config_service.add_validation_rule("test.key", lambda x: isinstance(x, str))
        
        with pytest.raises(ConfigValidationError):
            config_service.set_config("test.key", 123)  # Invalid type
        
        mock_backend.set_config.assert_not_called()

    def test_delete_config_success(self, config_service, mock_backend):
        """Test successful config deletion."""
        config_service.delete_config("test.key")
        
        mock_backend.delete_config.assert_called_once_with("test.key")

    def test_list_configs(self, config_service, mock_backend):
        """Test listing configurations."""
        mock_backend.list_configs.return_value = {"key1": "value1", "key2": "value2"}
        
        configs = config_service.list_configs()
        
        assert configs == {"key1": "value1", "key2": "value2"}
        mock_backend.list_configs.assert_called_once()

    def test_add_validation_rule(self, config_service):
        """Test adding validation rule."""
        rule = lambda x: isinstance(x, int)
        config_service.add_validation_rule("test.key", rule)
        
        assert "test.key" in config_service.validation_rules
        assert config_service.validation_rules["test.key"] == rule

    def test_remove_validation_rule(self, config_service):
        """Test removing validation rule."""
        rule = lambda x: isinstance(x, int)
        config_service.add_validation_rule("test.key", rule)
        config_service.remove_validation_rule("test.key")
        
        assert "test.key" not in config_service.validation_rules

    def test_validate_config(self, config_service):
        """Test config validation."""
        config_service.add_validation_rule("test.key", lambda x: isinstance(x, str))
        
        # Valid config
        assert config_service.validate_config("test.key", "valid_string") is True
        
        # Invalid config
        with pytest.raises(ConfigValidationError):
            config_service.validate_config("test.key", 123)

    def test_cache_invalidation(self, config_service, mock_backend):
        """Test cache invalidation."""
        mock_backend.get_config.return_value = "test_value"
        
        # First call - cached
        config_service.get_config("test.key")
        
        # Invalidate cache
        config_service.invalidate_cache("test.key")
        
        # Second call - should hit backend again
        config_service.get_config("test.key")
        
        assert mock_backend.get_config.call_count == 2

    def test_cache_clear(self, config_service, mock_backend):
        """Test cache clearing."""
        mock_backend.get_config.return_value = "test_value"
        
        # Cache some values
        config_service.get_config("key1")
        config_service.get_config("key2")
        
        # Clear cache
        config_service.clear_cache()
        
        # Should hit backend again
        config_service.get_config("key1")
        
        assert mock_backend.get_config.call_count == 3

    def test_watch_config_changes(self, config_service):
        """Test watching config changes."""
        callback = Mock()
        config_service.watch_config_changes(callback)
        
        assert callback in config_service.watcher.callbacks

    def test_unwatch_config_changes(self, config_service):
        """Test unwatching config changes."""
        callback = Mock()
        config_service.watch_config_changes(callback)
        config_service.unwatch_config_changes(callback)
        
        assert callback not in config_service.watcher.callbacks

    def test_config_change_notification(self, config_service, mock_backend):
        """Test config change notification."""
        callback = Mock()
        config_service.watch_config_changes(callback)
        
        # Set config should trigger notification
        config_service.set_config("test.key", "new_value")
        
        # Verify callback was called
        callback.assert_called_once()
        event = callback.call_args[0][0]
        assert event.key == "test.key"
        assert event.new_value == "new_value"

    def test_bulk_config_operations(self, config_service, mock_backend):
        """Test bulk config operations."""
        configs = {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3"
        }
        
        config_service.set_configs(configs)
        
        assert mock_backend.set_config.call_count == 3

    def test_config_versioning(self, config_service, mock_backend):
        """Test config versioning."""
        mock_backend.get_config.return_value = "test_value"
        
        # Set initial config
        config_service.set_config("test.key", "value1")
        
        # Create version
        version = config_service.create_version("v1.0.0", "Initial version")
        
        assert version is not None
        assert version.version_id == "v1.0.0"
        assert version.description == "Initial version"

    def test_config_rollback(self, config_service, mock_backend):
        """Test config rollback."""
        # Set initial config
        config_service.set_config("test.key", "value1")
        
        # Create version
        version = config_service.create_version("v1.0.0", "Initial version")
        
        # Change config
        config_service.set_config("test.key", "value2")
        
        # Rollback to version
        config_service.rollback_to_version("v1.0.0")
        
        # Verify rollback
        value = config_service.get_config("test.key")
        assert value == "value1"

    def test_config_export_import(self, config_service, mock_backend):
        """Test config export and import."""
        mock_backend.list_configs.return_value = {"key1": "value1", "key2": "value2"}
        
        # Export config
        exported = config_service.export_config()
        
        assert "key1" in exported
        assert "key2" in exported
        
        # Import config
        new_configs = {"key3": "value3", "key4": "value4"}
        config_service.import_config(new_configs)
        
        assert mock_backend.set_config.call_count == 2

    def test_config_service_with_different_backends(self):
        """Test ConfigService with different backend types."""
        # Test with memory backend
        config = ConfigServiceConfig(backend_type="memory")
        memory_service = ConfigService(config)
        assert isinstance(memory_service.backend, MemoryConfigBackend)
        
        # Test with file backend
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            config_file = f.name
        
        try:
            config = ConfigServiceConfig(backend_type="file", config_file=config_file)
            file_service = ConfigService(config)
            assert isinstance(file_service.backend, FileConfigBackend)
        finally:
            os.unlink(config_file)

    def test_config_service_error_handling(self, config_service, mock_backend):
        """Test config service error handling."""
        mock_backend.get_config.side_effect = Exception("Backend error")
        
        with pytest.raises(Exception):
            config_service.get_config("test.key")

    def test_config_service_concurrent_access(self, config_service, mock_backend):
        """Test config service concurrent access."""
        mock_backend.get_config.return_value = "test_value"
        
        def get_config():
            return config_service.get_config("test.key")
        
        # Run concurrent operations
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(get_config) for _ in range(10)]
            results = [future.result() for future in futures]
        
        # All should return the same value
        assert all(result == "test_value" for result in results)

    def test_config_service_cleanup(self, config_service):
        """Test config service cleanup."""
        # Add some data
        config_service.set_config("test.key", "test_value")
        
        # Cleanup
        config_service.cleanup()
        
        # Should clear cache and stop watchers
        assert len(config_service.cache) == 0
        assert len(config_service.watcher.callbacks) == 0
