"""
Tests for configuration management.
"""

import pytest
import os
import tempfile
import json
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from ai_engine.core.config_manager import (
    ConfigManager,
    MainConfig,
    Environment,
    DatabaseConfig,
    RedisConfig,
    AIEngineConfig,
    SecurityConfig,
    ConfigValidationError,
    ConfigLoadError,
    get_config_manager,
    load_config,
)


class TestEnvironmentEnum:
    """Test Environment enumeration."""

    def test_environment_values(self):
        """Test environment enum values."""
        assert Environment.DEVELOPMENT.value == "development"
        assert Environment.TESTING.value == "testing"
        assert Environment.STAGING.value == "staging"
        assert Environment.PRODUCTION.value == "production"


class TestDatabaseConfig:
    """Test DatabaseConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DatabaseConfig()
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "cronos_ai"
        assert config.username == "cronos_user"
        assert config.connection_pool_size == 10

    def test_password_from_env(self):
        """Test password loading from environment variable."""
        os.environ["DATABASE_PASSWORD"] = "test_password"

        config = DatabaseConfig()
        assert config.password == "test_password"

        del os.environ["DATABASE_PASSWORD"]

    def test_custom_values(self):
        """Test custom configuration values."""
        config = DatabaseConfig(
            host="db.example.com",
            port=5433,
            database="custom_db",
            password="custom_pass"
        )

        assert config.host == "db.example.com"
        assert config.port == 5433
        assert config.database == "custom_db"
        assert config.password == "custom_pass"


class TestRedisConfig:
    """Test RedisConfig dataclass."""

    def test_default_values(self):
        """Test default Redis configuration."""
        config = RedisConfig()
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.database == 0
        assert config.ssl is False

    def test_password_from_env(self):
        """Test password loading from environment."""
        os.environ["REDIS_PASSWORD"] = "redis_pass"

        config = RedisConfig()
        assert config.password == "redis_pass"

        del os.environ["REDIS_PASSWORD"]


class TestAIEngineConfig:
    """Test AIEngineConfig dataclass."""

    def test_default_values(self):
        """Test default AI engine configuration."""
        config = AIEngineConfig()
        assert config.use_gpu is True
        assert config.batch_size == 32
        assert config.max_sequence_length == 512
        assert config.learning_rate == 0.001

    def test_model_cache_dir_creation(self):
        """Test that model cache directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "models")
            config = AIEngineConfig(model_cache_dir=cache_dir)

            assert os.path.exists(cache_dir)
            assert os.path.isdir(cache_dir)


class TestSecurityConfig:
    """Test SecurityConfig dataclass."""

    def test_default_values(self):
        """Test default security configuration."""
        config = SecurityConfig()
        assert config.api_key_enabled is True
        assert config.jwt_enabled is False
        assert config.tls_enabled is True
        assert config.audit_logging_enabled is True

    def test_jwt_secret_generation(self):
        """Test automatic JWT secret generation."""
        config = SecurityConfig(jwt_enabled=True)
        assert config.jwt_secret != ""
        assert len(config.jwt_secret) > 20

    def test_encryption_key_generation(self):
        """Test automatic encryption key generation."""
        config = SecurityConfig()
        assert config.encryption_key != ""
        assert len(config.encryption_key) > 20


class TestMainConfig:
    """Test MainConfig dataclass."""

    def test_default_values(self):
        """Test default main configuration."""
        config = MainConfig()
        assert config.environment == Environment.DEVELOPMENT
        assert config.debug is False
        assert config.api_port == 8080
        assert isinstance(config.database, DatabaseConfig)
        assert isinstance(config.redis, RedisConfig)
        assert isinstance(config.ai_engine, AIEngineConfig)
        assert isinstance(config.security, SecurityConfig)

    def test_production_adjustments(self):
        """Test production environment adjustments."""
        config = MainConfig(environment=Environment.PRODUCTION)
        assert config.debug is False
        assert config.security.api_key_enabled is True
        assert config.security.tls_enabled is True
        assert config.monitoring.alerting_enabled is True

    def test_development_adjustments(self):
        """Test development environment adjustments."""
        config = MainConfig(environment=Environment.DEVELOPMENT)
        assert config.debug is True
        assert config.security.api_key_enabled is False
        assert config.security.tls_enabled is False


class TestConfigManager:
    """Test ConfigManager class."""

    def test_initialization(self):
        """Test config manager initialization."""
        manager = ConfigManager()
        assert manager.config_dir == Path.cwd() / "config"
        assert manager._config is None

    def test_custom_config_dir(self):
        """Test custom configuration directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ConfigManager(config_dir=tmpdir)
            assert manager.config_dir == Path(tmpdir)

    def test_load_from_dict(self):
        """Test loading configuration from dictionary."""
        config_dict = {
            "environment": "testing",
            "debug": True,
            "api_port": 9000,
            "database": {
                "host": "testdb.example.com",
                "port": 5433
            }
        }

        manager = ConfigManager()
        config = manager.load_config(config_dict=config_dict)

        assert config.environment == Environment.TESTING
        assert config.debug is True
        assert config.api_port == 9000
        assert config.database.host == "testdb.example.com"
        assert config.database.port == 5433

    def test_load_from_yaml_file(self):
        """Test loading configuration from YAML file."""
        config_data = {
            "environment": "staging",
            "api_port": 8888,
            "database": {"host": "staging-db.example.com"}
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.yaml"
            with open(config_file, "w") as f:
                yaml.dump(config_data, f)

            manager = ConfigManager(config_dir=tmpdir)
            config = manager.load_config(config_file=str(config_file))

            assert config.environment == Environment.STAGING
            assert config.api_port == 8888
            assert config.database.host == "staging-db.example.com"

    def test_load_from_json_file(self):
        """Test loading configuration from JSON file."""
        config_data = {
            "environment": "production",
            "api_port": 7777
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"
            with open(config_file, "w") as f:
                json.dump(config_data, f)

            manager = ConfigManager(config_dir=tmpdir)
            config = manager.load_config(config_file=str(config_file))

            assert config.environment == Environment.PRODUCTION
            assert config.api_port == 7777

    def test_file_not_found_error(self):
        """Test error when config file not found."""
        manager = ConfigManager()

        with pytest.raises(ConfigLoadError, match="not found"):
            manager.load_config(config_file="/nonexistent/config.yaml")

    def test_unsupported_format_error(self):
        """Test error with unsupported file format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.txt"
            config_file.write_text("invalid content")

            manager = ConfigManager(config_dir=tmpdir)

            with pytest.raises(ConfigLoadError, match="Unsupported"):
                manager.load_config(config_file=str(config_file))

    def test_env_variable_overrides(self):
        """Test environment variable overrides."""
        os.environ["DATABASE_HOST"] = "override-db.example.com"
        os.environ["API_PORT"] = "9999"
        os.environ["DEBUG"] = "true"

        try:
            config_dict = {"environment": "development"}
            manager = ConfigManager()
            config = manager.load_config(config_dict=config_dict)

            assert config.database.host == "override-db.example.com"
            assert config.api_port == 9999
            assert config.debug is True
        finally:
            del os.environ["DATABASE_HOST"]
            del os.environ["API_PORT"]
            del os.environ["DEBUG"]

    def test_validation_invalid_port(self):
        """Test validation fails for invalid port."""
        config_dict = {"api_port": 70000}  # Invalid port
        manager = ConfigManager()

        with pytest.raises(ConfigValidationError, match="Invalid API port"):
            manager.load_config(config_dict=config_dict)

    def test_validation_production_auth(self):
        """Test production requires authentication."""
        config_dict = {
            "environment": "production",
            "security": {
                "api_key_enabled": False,
                "jwt_enabled": False
            }
        }
        manager = ConfigManager()

        with pytest.raises(ConfigValidationError, match="Authentication must be enabled"):
            manager.load_config(config_dict=config_dict)

    def test_get_config(self):
        """Test getting current configuration."""
        manager = ConfigManager()
        config_dict = {"environment": "testing"}
        manager.load_config(config_dict=config_dict)

        config = manager.get_config()
        assert config.environment == Environment.TESTING

    def test_get_config_auto_loads(self):
        """Test get_config auto-loads if not loaded."""
        manager = ConfigManager()
        config = manager.get_config()
        assert config is not None

    def test_to_dict(self):
        """Test configuration to dictionary conversion."""
        manager = ConfigManager()
        config_dict = {"environment": "testing", "api_port": 8080}
        manager.load_config(config_dict=config_dict)

        result = manager.to_dict()
        assert result["environment"] == "testing"
        assert result["api_port"] == 8080
        assert "database" in result
        assert "security" in result

    def test_save_config_yaml(self):
        """Test saving configuration to YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ConfigManager()
            config_dict = {"environment": "testing"}
            manager.load_config(config_dict=config_dict)

            output_file = Path(tmpdir) / "output.yaml"
            manager.save_config(str(output_file), format="yaml")

            assert output_file.exists()

            # Load and verify
            with open(output_file) as f:
                loaded = yaml.safe_load(f)
            assert loaded["environment"] == "testing"

    def test_save_config_json(self):
        """Test saving configuration to JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ConfigManager()
            config_dict = {"environment": "staging"}
            manager.load_config(config_dict=config_dict)

            output_file = Path(tmpdir) / "output.json"
            manager.save_config(str(output_file), format="json")

            assert output_file.exists()

            # Load and verify
            with open(output_file) as f:
                loaded = json.load(f)
            assert loaded["environment"] == "staging"

    def test_config_change_detection(self):
        """Test configuration change detection."""
        manager = ConfigManager()
        config1_dict = {"api_port": 8080}
        manager.load_config(config_dict=config1_dict)

        old_hash = manager._config_hash

        # Reload with different config
        config2_dict = {"api_port": 9090}
        manager.load_config(config_dict=config2_dict)

        assert manager._config_hash != old_hash

    def test_config_watchers(self):
        """Test configuration change watchers."""
        manager = ConfigManager()

        watcher_called = []

        def watcher(config):
            watcher_called.append(config)

        manager.add_config_watcher(watcher)
        manager.load_config(config_dict={"environment": "testing"})

        assert len(watcher_called) == 1
        assert watcher_called[0].environment == Environment.TESTING

    def test_temporary_config_context(self):
        """Test temporary configuration override."""
        manager = ConfigManager()
        manager.load_config(config_dict={"api_port": 8080})

        original_port = manager.get_config().api_port

        with manager.temporary_config({"api_port": 9999}):
            assert manager.get_config().api_port == 9999

        # Should revert after context
        assert manager.get_config().api_port == original_port

    def test_auto_discover_config_file(self):
        """Test auto-discovery of configuration file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "cronos_ai.yaml"
            config_data = {"environment": "testing"}
            with open(config_file, "w") as f:
                yaml.dump(config_data, f)

            manager = ConfigManager(config_dir=tmpdir)
            config = manager.load_config()

            assert config.environment == Environment.TESTING

    def test_environment_specific_config(self):
        """Test environment-specific configuration file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["CRONOS_ENVIRONMENT"] = "production"

            try:
                config_file = Path(tmpdir) / "cronos_ai.production.yaml"
                config_data = {"api_port": 7777}
                with open(config_file, "w") as f:
                    yaml.dump(config_data, f)

                manager = ConfigManager(config_dir=tmpdir)
                config = manager.load_config()

                assert config.api_port == 7777
            finally:
                del os.environ["CRONOS_ENVIRONMENT"]


class TestGlobalConfigFunctions:
    """Test global configuration functions."""

    def test_get_config_manager_singleton(self):
        """Test config manager singleton pattern."""
        manager1 = get_config_manager()
        manager2 = get_config_manager()

        assert manager1 is manager2

    def test_load_config_function(self):
        """Test global load_config function."""
        config_dict = {"environment": "testing"}
        config = load_config(config_dict=config_dict)

        assert config.environment == Environment.TESTING


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
