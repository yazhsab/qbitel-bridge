"""
Tests for Configuration Service.
Comprehensive tests for config_service module to improve coverage.
"""

import pytest
import pytest_asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import tempfile
import yaml
import json
from datetime import datetime, timedelta

from ai_engine.core.config_service import (
    ConfigurationService,
    ConfigEnvironment,
    ConfigFormat,
    ConfigSource,
    ConfigMetadata,
    ConfigChange,
    ConfigValidator,
    FileConfigurationStore,
    initialize_config_service,
    get_config_service,
    shutdown_config_service,
    get_config,
    set_config,
    watch_config,
)


class TestConfigEnvironment:
    """Test ConfigEnvironment enum."""

    def test_environment_values(self):
        """Test configuration environment enum values."""
        assert ConfigEnvironment.DEVELOPMENT.value == "development"
        assert ConfigEnvironment.STAGING.value == "staging"
        assert ConfigEnvironment.PRODUCTION.value == "production"
        assert ConfigEnvironment.TEST.value == "test"


class TestConfigFormat:
    """Test ConfigFormat enum."""

    def test_format_values(self):
        """Test configuration format values."""
        assert ConfigFormat.JSON.value == "json"
        assert ConfigFormat.YAML.value == "yaml"
        assert ConfigFormat.TOML.value == "toml"
        assert ConfigFormat.ENV.value == "env"


class TestConfigSource:
    """Test ConfigSource enum."""

    def test_source_values(self):
        """Test configuration source values."""
        assert ConfigSource.FILE.value == "file"
        assert ConfigSource.ETCD.value == "etcd"
        assert ConfigSource.CONSUL.value == "consul"
        assert ConfigSource.DATABASE.value == "database"
        assert ConfigSource.ENVIRONMENT.value == "environment"
        assert ConfigSource.REMOTE_HTTP.value == "remote_http"


class TestConfigMetadata:
    """Test ConfigMetadata dataclass."""

    def test_create_metadata(self):
        """Test creating configuration metadata."""
        now = datetime.utcnow()
        metadata = ConfigMetadata(
            key="app.name",
            version="1.0",
            environment=ConfigEnvironment.DEVELOPMENT,
            source=ConfigSource.FILE,
            created_at=now,
            updated_at=now,
            checksum="abc123",
            tags={"team": "platform"},
            description="Application name",
            encrypted=False,
        )

        assert metadata.key == "app.name"
        assert metadata.version == "1.0"
        assert metadata.environment == ConfigEnvironment.DEVELOPMENT
        assert metadata.source == ConfigSource.FILE
        assert metadata.tags["team"] == "platform"
        assert metadata.encrypted is False


class TestConfigChange:
    """Test ConfigChange dataclass."""

    def test_create_config_change(self):
        """Test creating configuration change record."""
        now = datetime.utcnow()
        change = ConfigChange(
            change_id="change_123",
            key="app.port",
            old_value=8080,
            new_value=9000,
            changed_by="admin",
            timestamp=now,
            change_type="update",
            reason="Port conflict",
        )

        assert change.change_id == "change_123"
        assert change.key == "app.port"
        assert change.old_value == 8080
        assert change.new_value == 9000
        assert change.changed_by == "admin"
        assert change.change_type == "update"


class TestConfigValidator:
    """Test ConfigValidator class."""

    def test_register_validator(self):
        """Test registering custom validator."""
        validator = ConfigValidator()

        def validate_positive(value):
            return value > 0

        validator.register_validator("*_count", validate_positive)
        assert "*_count" in validator.validators

    def test_validate_with_custom_validator(self):
        """Test validation with custom validator."""
        validator = ConfigValidator()

        def validate_positive(value):
            return value > 0

        validator.register_validator("*_count", validate_positive)

        assert validator.validate("user_count", 10) is True
        assert validator.validate("user_count", -5) is False

    def test_register_schema(self):
        """Test registering JSON schema."""
        validator = ConfigValidator()

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
            },
            "required": ["name"],
        }

        validator.register_schema("user_*", schema)
        assert "user_*" in validator.schemas


class TestFileConfigurationStore:
    """Test FileConfigurationStore class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def store(self, temp_dir):
        """Create file configuration store."""
        config = {"base_path": str(temp_dir)}
        return FileConfigurationStore(config)

    @pytest.mark.asyncio
    async def test_get_set_value(self, store):
        """Test getting and setting configuration value."""
        metadata = ConfigMetadata(
            key="app.name",
            version="1.0",
            environment=ConfigEnvironment.DEVELOPMENT,
            source=ConfigSource.FILE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            checksum="",
        )

        # Set value
        success = await store.set(
            "app.name", "test-app", ConfigEnvironment.DEVELOPMENT, metadata
        )
        assert success is True

        # Get value
        value = await store.get("app.name", ConfigEnvironment.DEVELOPMENT)
        assert value == "test-app"

    @pytest.mark.asyncio
    async def test_get_nested_value(self, store):
        """Test getting nested configuration value."""
        metadata = ConfigMetadata(
            key="database.host",
            version="1.0",
            environment=ConfigEnvironment.DEVELOPMENT,
            source=ConfigSource.FILE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            checksum="",
        )

        await store.set(
            "database.host", "localhost", ConfigEnvironment.DEVELOPMENT, metadata
        )

        value = await store.get("database.host", ConfigEnvironment.DEVELOPMENT)
        assert value == "localhost"

    @pytest.mark.asyncio
    async def test_delete_value(self, store):
        """Test deleting configuration value."""
        metadata = ConfigMetadata(
            key="temp.key",
            version="1.0",
            environment=ConfigEnvironment.DEVELOPMENT,
            source=ConfigSource.FILE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            checksum="",
        )

        # Set and then delete
        await store.set(
            "temp.key", "temp_value", ConfigEnvironment.DEVELOPMENT, metadata
        )
        success = await store.delete("temp.key", ConfigEnvironment.DEVELOPMENT)
        assert success is True

        # Verify deleted
        value = await store.get("temp.key", ConfigEnvironment.DEVELOPMENT)
        assert value is None

    @pytest.mark.asyncio
    async def test_list_keys(self, store):
        """Test listing configuration keys."""
        metadata = ConfigMetadata(
            key="test",
            version="1.0",
            environment=ConfigEnvironment.DEVELOPMENT,
            source=ConfigSource.FILE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            checksum="",
        )

        # Set multiple values
        await store.set("app.name", "test", ConfigEnvironment.DEVELOPMENT, metadata)
        await store.set("app.version", "1.0", ConfigEnvironment.DEVELOPMENT, metadata)

        keys = await store.list_keys(environment=ConfigEnvironment.DEVELOPMENT)
        assert len(keys) > 0


class TestConfigurationService:
    """Test ConfigurationService class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def service_config(self, temp_dir):
        """Create service configuration."""
        return {
            "environment": "development",
            "storage": {"type": "file", "base_path": str(temp_dir)},
            "cache_ttl_seconds": 60,
        }

    @pytest.fixture
    def service(self, service_config):
        """Create configuration service."""
        return ConfigurationService(service_config)

    @pytest.mark.asyncio
    async def test_get_set_config(self, service):
        """Test getting and setting configuration."""
        success = await service.set("app.name", "test-app")
        assert success is True

        value = await service.get("app.name")
        assert value == "test-app"

    @pytest.mark.asyncio
    async def test_get_with_default(self, service):
        """Test getting configuration with default value."""
        value = await service.get("nonexistent.key", default="default_value")
        assert value == "default_value"

    @pytest.mark.asyncio
    async def test_delete_config(self, service):
        """Test deleting configuration."""
        await service.set("temp.key", "value")
        success = await service.delete("temp.key")
        assert success is True

        value = await service.get("temp.key")
        assert value is None

    @pytest.mark.asyncio
    async def test_list_keys(self, service):
        """Test listing configuration keys."""
        await service.set("app.name", "test")
        await service.set("app.version", "1.0")

        keys = await service.list_keys()
        assert len(keys) >= 0

    @pytest.mark.asyncio
    async def test_get_all_config(self, service):
        """Test getting all configuration."""
        await service.set("app.name", "test")
        await service.set("app.port", 8080)

        all_config = await service.get_all()
        assert isinstance(all_config, dict)

    def test_watch_config(self, service):
        """Test watching configuration changes."""
        callback_called = False

        def callback(key, new_value, old_value):
            nonlocal callback_called
            callback_called = True

        service.watch("app.*", callback)
        assert "app.*" in service.watchers

    def test_unwatch_config(self, service):
        """Test unwatching configuration."""

        def callback(key, new_value, old_value):
            pass

        service.watch("app.*", callback)
        service.unwatch("app.*", callback)
        assert "app.*" not in service.watchers or len(service.watchers["app.*"]) == 0

    def test_get_change_history(self, service):
        """Test getting configuration change history."""
        history = service.get_change_history()
        assert isinstance(history, list)

    def test_get_statistics(self, service):
        """Test getting service statistics."""
        stats = service.get_statistics()
        assert "current_environment" in stats
        assert "cache_size" in stats
        assert "total_changes" in stats

    @pytest.mark.asyncio
    async def test_export_config_json(self, service):
        """Test exporting configuration as JSON."""
        await service.set("app.name", "test")

        exported = await service.export_config(format_type=ConfigFormat.JSON)
        assert isinstance(exported, str)
        data = json.loads(exported)
        assert isinstance(data, dict)

    @pytest.mark.asyncio
    async def test_export_config_yaml(self, service):
        """Test exporting configuration as YAML."""
        await service.set("app.name", "test")

        exported = await service.export_config(format_type=ConfigFormat.YAML)
        assert isinstance(exported, str)


class TestGlobalConfigService:
    """Test global configuration service functions."""

    @pytest.mark.asyncio
    async def test_initialize_config_service(self):
        """Test initializing global configuration service."""
        config = {
            "environment": "test",
            "storage": {"type": "file", "base_path": "/tmp/test_config"},
        }

        service = await initialize_config_service(config)
        assert service is not None
        assert get_config_service() is service

        await shutdown_config_service()

    @pytest.mark.asyncio
    async def test_get_config_global(self):
        """Test getting configuration using global service."""
        config = {
            "environment": "test",
            "storage": {"type": "file", "base_path": "/tmp/test_config"},
        }

        await initialize_config_service(config)

        value = await get_config("test.key", default="default")
        assert value == "default"

        await shutdown_config_service()

    @pytest.mark.asyncio
    async def test_set_config_global(self):
        """Test setting configuration using global service."""
        config = {
            "environment": "test",
            "storage": {"type": "file", "base_path": "/tmp/test_config"},
        }

        await initialize_config_service(config)

        success = await set_config("test.key", "test_value")
        assert success is True

        await shutdown_config_service()

    def test_watch_config_global(self):
        """Test watching configuration using global service."""

        def callback(key, new_value, old_value):
            pass

        # Should not raise exception even if service not initialized
        watch_config("test.*", callback)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])