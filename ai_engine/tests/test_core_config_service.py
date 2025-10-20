"""
Comprehensive tests for ai_engine.core.config_service module.

This module provides configuration service functionality with etcd3 integration,
environment variable loading, and configuration validation.
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path
import yaml
import json

from ai_engine.core.config_service import (
    ConfigurationService,
    ConfigValidationError,
    ConfigLoadError,
    ConfigSaveError,
    EnvironmentConfigLoader,
    EtcdConfigBackend,
    FileConfigBackend,
    ConfigWatcher,
)


class TestConfigurationService:
    """Test ConfigurationService class."""

    @pytest.fixture
    def mock_etcd_client(self):
        """Mock etcd3 client."""
        mock_client = AsyncMock()
        mock_client.get.return_value = (b'{"test": "value"}', None)
        mock_client.put.return_value = True
        mock_client.delete.return_value = True
        return mock_client

    @pytest.fixture
    def config_service(self, mock_etcd_client):
        """Create ConfigurationService instance with mocked dependencies."""
        with patch("ai_engine.core.config_service.etcd3") as mock_etcd3:
            mock_etcd3.client.return_value = mock_etcd_client
            service = ConfigurationService(
                backend_type="etcd",
                etcd_host="localhost",
                etcd_port=2379,
                etcd_prefix="/cronos/config",
            )
            return service

    @pytest.mark.asyncio
    async def test_initialization_etcd_backend(self, mock_etcd_client):
        """Test ConfigurationService initialization with etcd backend."""
        with patch("ai_engine.core.config_service.etcd3") as mock_etcd3:
            mock_etcd3.client.return_value = mock_etcd_client

            service = ConfigurationService(
                backend_type="etcd",
                etcd_host="localhost",
                etcd_port=2379,
                etcd_prefix="/cronos/config",
            )

            assert service.backend_type == "etcd"
            assert service.etcd_host == "localhost"
            assert service.etcd_port == 2379
            assert service.etcd_prefix == "/cronos/config"
            mock_etcd3.client.assert_called_once_with(host="localhost", port=2379)

    @pytest.mark.asyncio
    async def test_initialization_file_backend(self):
        """Test ConfigurationService initialization with file backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = ConfigurationService(
                backend_type="file", config_file=os.path.join(tmpdir, "config.yaml")
            )

            assert service.backend_type == "file"
            assert service.config_file == os.path.join(tmpdir, "config.yaml")

    @pytest.mark.asyncio
    async def test_initialization_invalid_backend(self):
        """Test ConfigurationService initialization with invalid backend."""
        with pytest.raises(ConfigValidationError, match="Invalid backend type"):
            ConfigurationService(backend_type="invalid")

    @pytest.mark.asyncio
    async def test_get_config_success(self, config_service, mock_etcd_client):
        """Test successful config retrieval."""
        mock_etcd_client.get.return_value = (
            b'{"database": {"host": "localhost"}}',
            None,
        )

        result = await config_service.get_config("database.host")

        assert result == "localhost"
        mock_etcd_client.get.assert_called_once_with("/cronos/config/database/host")

    @pytest.mark.asyncio
    async def test_get_config_not_found(self, config_service, mock_etcd_client):
        """Test config retrieval when key not found."""
        mock_etcd_client.get.return_value = (None, None)

        result = await config_service.get_config("nonexistent.key")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_config_with_default(self, config_service, mock_etcd_client):
        """Test config retrieval with default value."""
        mock_etcd_client.get.return_value = (None, None)

        result = await config_service.get_config(
            "nonexistent.key", default="default_value"
        )

        assert result == "default_value"

    @pytest.mark.asyncio
    async def test_set_config_success(self, config_service, mock_etcd_client):
        """Test successful config setting."""
        mock_etcd_client.put.return_value = True

        result = await config_service.set_config("database.host", "newhost")

        assert result is True
        mock_etcd_client.put.assert_called_once_with(
            "/cronos/config/database/host", "newhost"
        )

    @pytest.mark.asyncio
    async def test_set_config_failure(self, config_service, mock_etcd_client):
        """Test config setting failure."""
        mock_etcd_client.put.return_value = False

        with pytest.raises(ConfigSaveError, match="Failed to save config"):
            await config_service.set_config("database.host", "newhost")

    @pytest.mark.asyncio
    async def test_delete_config_success(self, config_service, mock_etcd_client):
        """Test successful config deletion."""
        mock_etcd_client.delete.return_value = True

        result = await config_service.delete_config("database.host")

        assert result is True
        mock_etcd_client.delete.assert_called_once_with("/cronos/config/database/host")

    @pytest.mark.asyncio
    async def test_delete_config_failure(self, config_service, mock_etcd_client):
        """Test config deletion failure."""
        mock_etcd_client.delete.return_value = False

        with pytest.raises(ConfigSaveError, match="Failed to delete config"):
            await config_service.delete_config("database.host")

    @pytest.mark.asyncio
    async def test_list_configs(self, config_service, mock_etcd_client):
        """Test listing all configs."""
        mock_etcd_client.get_prefix.return_value = [
            (b'{"host": "localhost"}', None),
            (b'{"port": 5432}', None),
        ]

        result = await config_service.list_configs("database")

        assert len(result) == 2
        assert result[0] == {"host": "localhost"}
        assert result[1] == {"port": 5432}
        mock_etcd_client.get_prefix.assert_called_once_with("/cronos/config/database")

    @pytest.mark.asyncio
    async def test_watch_config(self, config_service, mock_etcd_client):
        """Test config watching functionality."""
        mock_etcd_client.watch_prefix.return_value = [
            (b'{"new_value": "updated"}', None)
        ]

        callback_called = False

        def callback(key, value):
            nonlocal callback_called
            callback_called = True
            assert key == "database.host"
            assert value == {"new_value": "updated"}

        await config_service.watch_config("database.host", callback)

        # Simulate watch event
        mock_etcd_client.watch_prefix.assert_called_once_with(
            "/cronos/config/database/host"
        )

    @pytest.mark.asyncio
    async def test_validate_config_structure(self, config_service):
        """Test config structure validation."""
        valid_config = {
            "database": {"host": "localhost", "port": 5432},
            "api": {"host": "0.0.0.0", "port": 8080},
        }

        # Should not raise exception
        config_service.validate_config_structure(valid_config)

    @pytest.mark.asyncio
    async def test_validate_config_structure_invalid(self, config_service):
        """Test config structure validation with invalid config."""
        invalid_config = {"database": "invalid_structure"}  # Should be dict

        with pytest.raises(ConfigValidationError, match="Invalid config structure"):
            config_service.validate_config_structure(invalid_config)

    @pytest.mark.asyncio
    async def test_export_config(self, config_service, mock_etcd_client):
        """Test config export functionality."""
        mock_etcd_client.get_prefix.return_value = [
            (b'{"host": "localhost"}', None),
            (b'{"port": 5432}', None),
        ]

        result = await config_service.export_config()

        assert isinstance(result, dict)
        mock_etcd_client.get_prefix.assert_called_once_with("/cronos/config")

    @pytest.mark.asyncio
    async def test_import_config(self, config_service, mock_etcd_client):
        """Test config import functionality."""
        config_data = {"database": {"host": "localhost", "port": 5432}}
        mock_etcd_client.put.return_value = True

        result = await config_service.import_config(config_data)

        assert result is True
        # Should call put for each config key
        assert mock_etcd_client.put.call_count >= 2

    @pytest.mark.asyncio
    async def test_import_config_validation_error(self, config_service):
        """Test config import with validation error."""
        invalid_config = "invalid_config"

        with pytest.raises(ConfigValidationError, match="Config must be a dictionary"):
            await config_service.import_config(invalid_config)

    @pytest.mark.asyncio
    async def test_backup_config(self, config_service, mock_etcd_client):
        """Test config backup functionality."""
        mock_etcd_client.get_prefix.return_value = [(b'{"host": "localhost"}', None)]

        with tempfile.TemporaryDirectory() as tmpdir:
            backup_file = os.path.join(tmpdir, "backup.json")
            result = await config_service.backup_config(backup_file)

            assert result is True
            assert os.path.exists(backup_file)

            # Verify backup content
            with open(backup_file, "r") as f:
                backup_data = json.load(f)
                assert isinstance(backup_data, dict)

    @pytest.mark.asyncio
    async def test_restore_config(self, config_service, mock_etcd_client):
        """Test config restore functionality."""
        config_data = {"database": {"host": "localhost"}}
        mock_etcd_client.put.return_value = True

        with tempfile.TemporaryDirectory() as tmpdir:
            backup_file = os.path.join(tmpdir, "backup.json")
            with open(backup_file, "w") as f:
                json.dump(config_data, f)

            result = await config_service.restore_config(backup_file)

            assert result is True
            mock_etcd_client.put.assert_called()

    @pytest.mark.asyncio
    async def test_restore_config_file_not_found(self, config_service):
        """Test config restore with non-existent file."""
        with pytest.raises(ConfigLoadError, match="Backup file not found"):
            await config_service.restore_config("nonexistent.json")

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, config_service, mock_etcd_client):
        """Test health check when service is healthy."""
        mock_etcd_client.status.return_value = {"health": "true"}

        result = await config_service.health_check()

        assert result["status"] == "healthy"
        assert result["backend"] == "etcd"

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, config_service, mock_etcd_client):
        """Test health check when service is unhealthy."""
        mock_etcd_client.status.side_effect = Exception("Connection failed")

        result = await config_service.health_check()

        assert result["status"] == "unhealthy"
        assert "error" in result

    @pytest.mark.asyncio
    async def test_close_connection(self, config_service, mock_etcd_client):
        """Test closing etcd connection."""
        await config_service.close()

        mock_etcd_client.close.assert_called_once()


class TestEnvironmentConfigLoader:
    """Test EnvironmentConfigLoader class."""

    def test_load_environment_variables(self):
        """Test loading environment variables."""
        with patch.dict(
            os.environ,
            {
                "CRONOS_DATABASE_HOST": "localhost",
                "CRONOS_DATABASE_PORT": "5432",
                "CRONOS_API_HOST": "0.0.0.0",
            },
        ):
            loader = EnvironmentConfigLoader(prefix="CRONOS_")
            config = loader.load()

            assert config["database"]["host"] == "localhost"
            assert config["database"]["port"] == "5432"
            assert config["api"]["host"] == "0.0.0.0"

    def test_load_environment_variables_with_type_conversion(self):
        """Test loading environment variables with type conversion."""
        with patch.dict(
            os.environ,
            {
                "CRONOS_DATABASE_PORT": "5432",
                "CRONOS_DEBUG": "true",
                "CRONOS_TIMEOUT": "30.5",
            },
        ):
            loader = EnvironmentConfigLoader(prefix="CRONOS_")
            config = loader.load()

            assert config["database"]["port"] == 5432
            assert config["debug"] is True
            assert config["timeout"] == 30.5

    def test_load_environment_variables_nested(self):
        """Test loading nested environment variables."""
        with patch.dict(
            os.environ,
            {
                "CRONOS_DATABASE_POSTGRES_HOST": "localhost",
                "CRONOS_DATABASE_POSTGRES_PORT": "5432",
                "CRONOS_DATABASE_REDIS_HOST": "redis-host",
            },
        ):
            loader = EnvironmentConfigLoader(prefix="CRONOS_")
            config = loader.load()

            assert config["database"]["postgres"]["host"] == "localhost"
            assert config["database"]["postgres"]["port"] == 5432
            assert config["database"]["redis"]["host"] == "redis-host"


class TestFileConfigBackend:
    """Test FileConfigBackend class."""

    @pytest.fixture
    def temp_config_file(self):
        """Create temporary config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_data = {
                "database": {"host": "localhost", "port": 5432},
                "api": {"host": "0.0.0.0", "port": 8080},
            }
            yaml.dump(config_data, f)
            return f.name

    def test_load_config_from_file(self, temp_config_file):
        """Test loading config from file."""
        backend = FileConfigBackend(temp_config_file)
        config = backend.load()

        assert config["database"]["host"] == "localhost"
        assert config["database"]["port"] == 5432
        assert config["api"]["host"] == "0.0.0.0"
        assert config["api"]["port"] == 8080

    def test_save_config_to_file(self, temp_config_file):
        """Test saving config to file."""
        backend = FileConfigBackend(temp_config_file)
        new_config = {"database": {"host": "newhost", "port": 3306}}

        backend.save(new_config)

        # Reload and verify
        loaded_config = backend.load()
        assert loaded_config["database"]["host"] == "newhost"
        assert loaded_config["database"]["port"] == 3306

    def test_load_config_file_not_found(self):
        """Test loading config from non-existent file."""
        backend = FileConfigBackend("nonexistent.yaml")

        with pytest.raises(ConfigLoadError, match="Config file not found"):
            backend.load()

    def test_save_config_invalid_data(self, temp_config_file):
        """Test saving invalid config data."""
        backend = FileConfigBackend(temp_config_file)

        with pytest.raises(ConfigSaveError, match="Invalid config data"):
            backend.save("invalid_data")

    def teardown_method(self):
        """Clean up temporary files."""
        import glob

        for file in glob.glob("/tmp/tmp*"):
            try:
                os.unlink(file)
            except OSError:
                pass


class TestConfigWatcher:
    """Test ConfigWatcher class."""

    @pytest.mark.asyncio
    async def test_watch_config_changes(self):
        """Test watching for config changes."""
        callback_called = False
        callback_data = None

        def callback(key, value):
            nonlocal callback_called, callback_data
            callback_called = True
            callback_data = (key, value)

        watcher = ConfigWatcher()

        # Mock file system events
        with patch(
            "ai_engine.core.config_service.watchdog.observers.Observer"
        ) as mock_observer:
            mock_observer_instance = Mock()
            mock_observer.return_value = mock_observer_instance

            await watcher.start_watching("/tmp", callback)

            # Simulate file change event
            from watchdog.events import FileModifiedEvent

            event = FileModifiedEvent("/tmp/config.yaml")
            watcher._on_file_changed(event)

            assert callback_called is True
            assert callback_data[0] == "/tmp/config.yaml"

    @pytest.mark.asyncio
    async def test_stop_watching(self):
        """Test stopping config watching."""
        watcher = ConfigWatcher()

        with patch(
            "ai_engine.core.config_service.watchdog.observers.Observer"
        ) as mock_observer:
            mock_observer_instance = Mock()
            mock_observer.return_value = mock_observer_instance

            await watcher.start_watching("/tmp", lambda k, v: None)
            await watcher.stop_watching()

            mock_observer_instance.stop.assert_called_once()
            mock_observer_instance.join.assert_called_once()


class TestConfigValidationError:
    """Test ConfigValidationError exception."""

    def test_config_validation_error_creation(self):
        """Test ConfigValidationError creation."""
        error = ConfigValidationError("Invalid config", field="database.host")

        assert str(error) == "Invalid config"
        assert error.field == "database.host"

    def test_config_validation_error_with_context(self):
        """Test ConfigValidationError with context."""
        error = ConfigValidationError(
            "Invalid config",
            field="database.host",
            context={"current_value": "invalid", "expected_type": "string"},
        )

        assert error.field == "database.host"
        assert error.context["current_value"] == "invalid"
        assert error.context["expected_type"] == "string"


class TestConfigLoadError:
    """Test ConfigLoadError exception."""

    def test_config_load_error_creation(self):
        """Test ConfigLoadError creation."""
        error = ConfigLoadError("Failed to load config", file_path="/tmp/config.yaml")

        assert str(error) == "Failed to load config"
        assert error.file_path == "/tmp/config.yaml"


class TestConfigSaveError:
    """Test ConfigSaveError exception."""

    def test_config_save_error_creation(self):
        """Test ConfigSaveError creation."""
        error = ConfigSaveError("Failed to save config", file_path="/tmp/config.yaml")

        assert str(error) == "Failed to save config"
        assert error.file_path == "/tmp/config.yaml"


class TestConfigurationServiceIntegration:
    """Integration tests for ConfigurationService."""

    @pytest.mark.asyncio
    async def test_full_config_lifecycle(self):
        """Test complete config lifecycle: create, read, update, delete."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = os.path.join(tmpdir, "config.yaml")

            # Initialize service with file backend
            service = ConfigurationService(backend_type="file", config_file=config_file)

            # Set initial config
            initial_config = {
                "database": {"host": "localhost", "port": 5432},
                "api": {"host": "0.0.0.0", "port": 8080},
            }

            await service.import_config(initial_config)

            # Read config
            host = await service.get_config("database.host")
            assert host == "localhost"

            # Update config
            await service.set_config("database.host", "newhost")
            updated_host = await service.get_config("database.host")
            assert updated_host == "newhost"

            # List configs
            database_configs = await service.list_configs("database")
            assert len(database_configs) >= 1

            # Delete config
            await service.delete_config("database.host")
            deleted_host = await service.get_config("database.host")
            assert deleted_host is None

    @pytest.mark.asyncio
    async def test_config_backup_and_restore(self):
        """Test config backup and restore functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = os.path.join(tmpdir, "config.yaml")
            backup_file = os.path.join(tmpdir, "backup.json")

            service = ConfigurationService(backend_type="file", config_file=config_file)

            # Set initial config
            initial_config = {
                "database": {"host": "localhost", "port": 5432},
                "api": {"host": "0.0.0.0", "port": 8080},
            }
            await service.import_config(initial_config)

            # Backup config
            backup_result = await service.backup_config(backup_file)
            assert backup_result is True
            assert os.path.exists(backup_file)

            # Modify config
            await service.set_config("database.host", "modified")

            # Restore from backup
            restore_result = await service.restore_config(backup_file)
            assert restore_result is True

            # Verify restore
            restored_host = await service.get_config("database.host")
            assert restored_host == "localhost"

    @pytest.mark.asyncio
    async def test_environment_variable_override(self):
        """Test environment variable configuration override."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = os.path.join(tmpdir, "config.yaml")

            # Set environment variables
            with patch.dict(
                os.environ,
                {"CRONOS_DATABASE_HOST": "env-host", "CRONOS_API_PORT": "9000"},
            ):
                service = ConfigurationService(
                    backend_type="file",
                    config_file=config_file,
                    enable_env_override=True,
                    env_prefix="CRONOS_",
                )

                # Set base config
                base_config = {
                    "database": {"host": "file-host", "port": 5432},
                    "api": {"host": "0.0.0.0", "port": 8080},
                }
                await service.import_config(base_config)

                # Environment variables should override file config
                db_host = await service.get_config("database.host")
                api_port = await service.get_config("api.port")

                assert db_host == "env-host"
                assert api_port == 9000
