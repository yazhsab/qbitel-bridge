"""
Test suite for config_service.py with mocked dependencies.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import asyncio
from typing import Dict, Any

from ai_engine.core.config_service import ConfigService


class TestConfigService:
    """Test ConfigService with mocked dependencies."""

    @pytest.fixture
    def mock_etcd3(self):
        """Mock etcd3 client."""
        with patch('ai_engine.core.config_service.etcd3') as mock_etcd3:
            mock_client = Mock()
            mock_client.get = Mock(return_value=(b'{"test": "value"}', None))
            mock_client.put = Mock(return_value=True)
            mock_client.delete = Mock(return_value=True)
            mock_client.watch_prefix = Mock()
            mock_etcd3.client.return_value = mock_client
            yield mock_etcd3

    @pytest.fixture
    def config_service(self, mock_etcd3):
        """Create ConfigService instance."""
        return ConfigService(
            etcd_host="localhost",
            etcd_port=2379,
            namespace="test"
        )

    def test_initialization(self, config_service):
        """Test ConfigService initialization."""
        assert config_service.etcd_host == "localhost"
        assert config_service.etcd_port == 2379
        assert config_service.namespace == "test"
        assert config_service.etcd_client is None
        assert config_service.config_cache == {}
        assert config_service.watchers == {}

    def test_get_key_path(self, config_service):
        """Test key path generation."""
        key_path = config_service._get_key_path("test_key")
        assert key_path == "/test/test_key"

    def test_get_key_path_with_namespace(self, config_service):
        """Test key path generation with namespace."""
        config_service.namespace = "production"
        key_path = config_service._get_key_path("database_url")
        assert key_path == "/production/database_url"

    @pytest.mark.asyncio
    async def test_initialize_success(self, config_service, mock_etcd3):
        """Test successful initialization."""
        await config_service.initialize()
        
        assert config_service.etcd_client is not None
        assert config_service.initialized is True

    @pytest.mark.asyncio
    async def test_initialize_failure(self, config_service):
        """Test initialization failure."""
        with patch('ai_engine.core.config_service.etcd3.client', side_effect=Exception("Connection failed")):
            with pytest.raises(Exception, match="Failed to initialize config service"):
                await config_service.initialize()

    @pytest.mark.asyncio
    async def test_get_config_success(self, config_service, mock_etcd3):
        """Test successful config retrieval."""
        await config_service.initialize()
        
        result = await config_service.get_config("test_key")
        
        assert result == {"test": "value"}

    @pytest.mark.asyncio
    async def test_get_config_not_found(self, config_service, mock_etcd3):
        """Test config retrieval when key not found."""
        mock_etcd3.client.return_value.get.return_value = (None, None)
        await config_service.initialize()
        
        result = await config_service.get_config("non_existent_key")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_get_config_cached(self, config_service, mock_etcd3):
        """Test config retrieval from cache."""
        await config_service.initialize()
        config_service.config_cache["test_key"] = {"cached": "value"}
        
        result = await config_service.get_config("test_key")
        
        assert result == {"cached": "value"}
        # Should not call etcd client
        mock_etcd3.client.return_value.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_set_config_success(self, config_service, mock_etcd3):
        """Test successful config setting."""
        await config_service.initialize()
        
        result = await config_service.set_config("test_key", {"new": "value"})
        
        assert result is True
        mock_etcd3.client.return_value.put.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_config_failure(self, config_service, mock_etcd3):
        """Test config setting failure."""
        mock_etcd3.client.return_value.put.side_effect = Exception("Put failed")
        await config_service.initialize()
        
        result = await config_service.set_config("test_key", {"new": "value"})
        
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_config_success(self, config_service, mock_etcd3):
        """Test successful config deletion."""
        await config_service.initialize()
        
        result = await config_service.delete_config("test_key")
        
        assert result is True
        mock_etcd3.client.return_value.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_config_failure(self, config_service, mock_etcd3):
        """Test config deletion failure."""
        mock_etcd3.client.return_value.delete.side_effect = Exception("Delete failed")
        await config_service.initialize()
        
        result = await config_service.delete_config("test_key")
        
        assert result is False

    @pytest.mark.asyncio
    async def test_watch_config_success(self, config_service, mock_etcd3):
        """Test successful config watching."""
        await config_service.initialize()
        
        callback_called = False
        
        def test_callback(key, value):
            nonlocal callback_called
            callback_called = True
        
        result = await config_service.watch_config("test_key", test_callback)
        
        assert result is True
        assert "test_key" in config_service.watchers

    @pytest.mark.asyncio
    async def test_watch_config_failure(self, config_service, mock_etcd3):
        """Test config watching failure."""
        mock_etcd3.client.return_value.watch_prefix.side_effect = Exception("Watch failed")
        await config_service.initialize()
        
        def test_callback(key, value):
            pass
        
        result = await config_service.watch_config("test_key", test_callback)
        
        assert result is False

    @pytest.mark.asyncio
    async def test_stop_watching(self, config_service, mock_etcd3):
        """Test stopping config watching."""
        await config_service.initialize()
        
        # Add a watcher
        config_service.watchers["test_key"] = Mock()
        
        result = await config_service.stop_watching("test_key")
        
        assert result is True
        assert "test_key" not in config_service.watchers

    @pytest.mark.asyncio
    async def test_stop_watching_not_found(self, config_service, mock_etcd3):
        """Test stopping non-existent watcher."""
        await config_service.initialize()
        
        result = await config_service.stop_watching("non_existent_key")
        
        assert result is False

    @pytest.mark.asyncio
    async def test_get_all_configs(self, config_service, mock_etcd3):
        """Test getting all configs."""
        await config_service.initialize()
        
        # Mock the get_prefix method
        mock_etcd3.client.return_value.get_prefix.return_value = [
            (b'/test/key1', b'{"value1": "data1"}'),
            (b'/test/key2', b'{"value2": "data2"}')
        ]
        
        result = await config_service.get_all_configs()
        
        assert len(result) == 2
        assert "key1" in result
        assert "key2" in result

    @pytest.mark.asyncio
    async def test_clear_cache(self, config_service, mock_etcd3):
        """Test clearing config cache."""
        await config_service.initialize()
        
        # Add some cached configs
        config_service.config_cache["key1"] = {"cached": "value1"}
        config_service.config_cache["key2"] = {"cached": "value2"}
        
        config_service.clear_cache()
        
        assert config_service.config_cache == {}

    @pytest.mark.asyncio
    async def test_close(self, config_service, mock_etcd3):
        """Test closing config service."""
        await config_service.initialize()
        
        # Add some watchers
        config_service.watchers["key1"] = Mock()
        config_service.watchers["key2"] = Mock()
        
        await config_service.close()
        
        assert config_service.etcd_client is None
        assert config_service.watchers == {}
        assert config_service.initialized is False

    def test_config_validation(self, config_service):
        """Test config validation."""
        # Valid config
        valid_config = {"database_url": "postgresql://localhost/test"}
        assert config_service._validate_config(valid_config) is True
        
        # Invalid config (not a dict)
        invalid_config = "not a dict"
        assert config_service._validate_config(invalid_config) is False
        
        # Invalid config (None)
        assert config_service._validate_config(None) is False

    def test_key_validation(self, config_service):
        """Test key validation."""
        # Valid key
        assert config_service._validate_key("valid_key") is True
        assert config_service._validate_key("key_with_underscores") is True
        assert config_service._validate_key("key-with-dashes") is True
        
        # Invalid key
        assert config_service._validate_key("") is False
        assert config_service._validate_key(None) is False
        assert config_service._validate_key("key with spaces") is False
        assert config_service._validate_key("key/with/slashes") is False
