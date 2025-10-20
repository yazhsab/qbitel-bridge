"""
Basic test suite for config_service.py with mocked etcd3 dependency.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import asyncio
from typing import Dict, Any

# Mock etcd3 before importing config_service
with patch.dict("sys.modules", {"etcd3": MagicMock()}):
    from ai_engine.core.config_service import ConfigurationService


class TestConfigurationService:
    """Test ConfigurationService with mocked dependencies."""

    def test_initialization(self):
        """Test ConfigurationService initialization."""
        service = ConfigurationService(
            etcd_host="localhost", etcd_port=2379, namespace="test"
        )

        assert service.etcd_host == "localhost"
        assert service.etcd_port == 2379
        assert service.namespace == "test"
        assert service.etcd_client is None
        assert service.config_cache == {}
        assert service.watchers == {}

    def test_initialization_with_defaults(self):
        """Test ConfigurationService initialization with defaults."""
        service = ConfigurationService()

        assert service.etcd_host == "localhost"
        assert service.etcd_port == 2379
        assert service.namespace == "default"
        assert service.etcd_client is None
        assert service.config_cache == {}
        assert service.watchers == {}

    def test_get_key_path(self):
        """Test key path generation."""
        service = ConfigurationService(namespace="test")

        key_path = service._get_key_path("test_key")
        assert key_path == "/test/test_key"

        # Test with different namespace
        service.namespace = "production"
        key_path = service._get_key_path("database_url")
        assert key_path == "/production/database_url"

    def test_config_validation(self):
        """Test config validation."""
        service = ConfigurationService()

        # Valid config
        valid_config = {"database_url": "postgresql://localhost/test"}
        assert service._validate_config(valid_config) is True

        # Invalid config (not a dict)
        invalid_config = "not a dict"
        assert service._validate_config(invalid_config) is False

        # Invalid config (None)
        assert service._validate_config(None) is False

    def test_key_validation(self):
        """Test key validation."""
        service = ConfigurationService()

        # Valid key
        assert service._validate_key("valid_key") is True
        assert service._validate_key("key_with_underscores") is True
        assert service._validate_key("key-with-dashes") is True

        # Invalid key
        assert service._validate_key("") is False
        assert service._validate_key(None) is False
        assert service._validate_key("key with spaces") is False
        assert service._validate_key("key/with/slashes") is False

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful initialization."""
        with patch("ai_engine.core.config_service.etcd3.client") as mock_client:
            mock_etcd_client = Mock()
            mock_client.return_value = mock_etcd_client

            service = ConfigurationService()
            await service.initialize()

            assert service.etcd_client is not None
            assert service.initialized is True

    @pytest.mark.asyncio
    async def test_initialize_failure(self):
        """Test initialization failure."""
        with patch(
            "ai_engine.core.config_service.etcd3.client",
            side_effect=Exception("Connection failed"),
        ):
            service = ConfigurationService()

            with pytest.raises(Exception, match="Failed to initialize config service"):
                await service.initialize()

    @pytest.mark.asyncio
    async def test_get_config_success(self):
        """Test successful config retrieval."""
        with patch("ai_engine.core.config_service.etcd3.client") as mock_client:
            mock_etcd_client = Mock()
            mock_etcd_client.get.return_value = (b'{"test": "value"}', None)
            mock_client.return_value = mock_etcd_client

            service = ConfigurationService()
            await service.initialize()

            result = await service.get_config("test_key")

            assert result == {"test": "value"}

    @pytest.mark.asyncio
    async def test_get_config_not_found(self):
        """Test config retrieval when key not found."""
        with patch("ai_engine.core.config_service.etcd3.client") as mock_client:
            mock_etcd_client = Mock()
            mock_etcd_client.get.return_value = (None, None)
            mock_client.return_value = mock_etcd_client

            service = ConfigurationService()
            await service.initialize()

            result = await service.get_config("non_existent_key")

            assert result is None

    @pytest.mark.asyncio
    async def test_get_config_cached(self):
        """Test config retrieval from cache."""
        with patch("ai_engine.core.config_service.etcd3.client") as mock_client:
            mock_etcd_client = Mock()
            mock_client.return_value = mock_etcd_client

            service = ConfigurationService()
            await service.initialize()
            service.config_cache["test_key"] = {"cached": "value"}

            result = await service.get_config("test_key")

            assert result == {"cached": "value"}
            # Should not call etcd client
            mock_etcd_client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_set_config_success(self):
        """Test successful config setting."""
        with patch("ai_engine.core.config_service.etcd3.client") as mock_client:
            mock_etcd_client = Mock()
            mock_etcd_client.put.return_value = True
            mock_client.return_value = mock_etcd_client

            service = ConfigurationService()
            await service.initialize()

            result = await service.set_config("test_key", {"new": "value"})

            assert result is True
            mock_etcd_client.put.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_config_failure(self):
        """Test config setting failure."""
        with patch("ai_engine.core.config_service.etcd3.client") as mock_client:
            mock_etcd_client = Mock()
            mock_etcd_client.put.side_effect = Exception("Put failed")
            mock_client.return_value = mock_etcd_client

            service = ConfigurationService()
            await service.initialize()

            result = await service.set_config("test_key", {"new": "value"})

            assert result is False

    @pytest.mark.asyncio
    async def test_delete_config_success(self):
        """Test successful config deletion."""
        with patch("ai_engine.core.config_service.etcd3.client") as mock_client:
            mock_etcd_client = Mock()
            mock_etcd_client.delete.return_value = True
            mock_client.return_value = mock_etcd_client

            service = ConfigurationService()
            await service.initialize()

            result = await service.delete_config("test_key")

            assert result is True
            mock_etcd_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_config_failure(self):
        """Test config deletion failure."""
        with patch("ai_engine.core.config_service.etcd3.client") as mock_client:
            mock_etcd_client = Mock()
            mock_etcd_client.delete.side_effect = Exception("Delete failed")
            mock_client.return_value = mock_etcd_client

            service = ConfigurationService()
            await service.initialize()

            result = await service.delete_config("test_key")

            assert result is False

    @pytest.mark.asyncio
    async def test_watch_config_success(self):
        """Test successful config watching."""
        with patch("ai_engine.core.config_service.etcd3.client") as mock_client:
            mock_etcd_client = Mock()
            mock_etcd_client.watch_prefix.return_value = []
            mock_client.return_value = mock_etcd_client

            service = ConfigurationService()
            await service.initialize()

            callback_called = False

            def test_callback(key, value):
                nonlocal callback_called
                callback_called = True

            result = await service.watch_config("test_key", test_callback)

            assert result is True
            assert "test_key" in service.watchers

    @pytest.mark.asyncio
    async def test_watch_config_failure(self):
        """Test config watching failure."""
        with patch("ai_engine.core.config_service.etcd3.client") as mock_client:
            mock_etcd_client = Mock()
            mock_etcd_client.watch_prefix.side_effect = Exception("Watch failed")
            mock_client.return_value = mock_etcd_client

            service = ConfigurationService()
            await service.initialize()

            def test_callback(key, value):
                pass

            result = await service.watch_config("test_key", test_callback)

            assert result is False

    @pytest.mark.asyncio
    async def test_stop_watching(self):
        """Test stopping config watching."""
        with patch("ai_engine.core.config_service.etcd3.client") as mock_client:
            mock_etcd_client = Mock()
            mock_client.return_value = mock_etcd_client

            service = ConfigurationService()
            await service.initialize()

            # Add a watcher
            service.watchers["test_key"] = Mock()

            result = await service.stop_watching("test_key")

            assert result is True
            assert "test_key" not in service.watchers

    @pytest.mark.asyncio
    async def test_stop_watching_not_found(self):
        """Test stopping non-existent watcher."""
        with patch("ai_engine.core.config_service.etcd3.client") as mock_client:
            mock_etcd_client = Mock()
            mock_client.return_value = mock_etcd_client

            service = ConfigurationService()
            await service.initialize()

            result = await service.stop_watching("non_existent_key")

            assert result is False

    @pytest.mark.asyncio
    async def test_get_all_configs(self):
        """Test getting all configs."""
        with patch("ai_engine.core.config_service.etcd3.client") as mock_client:
            mock_etcd_client = Mock()
            mock_etcd_client.get_prefix.return_value = [
                (b"/test/key1", b'{"value1": "data1"}'),
                (b"/test/key2", b'{"value2": "data2"}'),
            ]
            mock_client.return_value = mock_etcd_client

            service = ConfigurationService()
            await service.initialize()

            result = await service.get_all_configs()

            assert len(result) == 2
            assert "key1" in result
            assert "key2" in result

    @pytest.mark.asyncio
    async def test_clear_cache(self):
        """Test clearing config cache."""
        with patch("ai_engine.core.config_service.etcd3.client") as mock_client:
            mock_etcd_client = Mock()
            mock_client.return_value = mock_etcd_client

            service = ConfigurationService()
            await service.initialize()

            # Add some cached configs
            service.config_cache["key1"] = {"cached": "value1"}
            service.config_cache["key2"] = {"cached": "value2"}

            service.clear_cache()

            assert service.config_cache == {}

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing config service."""
        with patch("ai_engine.core.config_service.etcd3.client") as mock_client:
            mock_etcd_client = Mock()
            mock_client.return_value = mock_etcd_client

            service = ConfigurationService()
            await service.initialize()

            # Add some watchers
            service.watchers["key1"] = Mock()
            service.watchers["key2"] = Mock()

            await service.close()

            assert service.etcd_client is None
            assert service.watchers == {}
            assert service.initialized is False
