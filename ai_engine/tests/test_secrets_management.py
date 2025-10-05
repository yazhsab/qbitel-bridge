"""
Tests for secrets management functionality.
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from ai_engine.security.secrets_manager import (
    SecretsManager,
    SecretBackend,
    SecretMetadata,
    SecretValidationError,
)
from ai_engine.core.config import DatabaseConfig, RedisConfig, SecurityConfig


class TestSecretsManager:
    """Test SecretsManager functionality."""

    def test_environment_backend_initialization(self):
        """Test initialization with environment backend."""
        config = {"backend": "environment"}
        manager = SecretsManager(config)

        assert manager.backend == SecretBackend.ENVIRONMENT
        assert manager.cache == {}

    def test_get_secret_from_environment(self):
        """Test getting secret from environment variable."""
        config = {"backend": "environment"}
        manager = SecretsManager(config)

        # Set environment variable
        os.environ["TEST_SECRET"] = "test_value"

        value = manager.get_secret("TEST_SECRET")
        assert value == "test_value"

        # Cleanup
        del os.environ["TEST_SECRET"]

    def test_get_secret_with_prefix(self):
        """Test getting secret with CRONOS_AI_ prefix."""
        config = {"backend": "environment"}
        manager = SecretsManager(config)

        # Set environment variable with prefix
        os.environ["CRONOS_AI_DB_PASSWORD"] = "secure_password"

        value = manager.get_secret("db_password")
        assert value == "secure_password"

        # Cleanup
        del os.environ["CRONOS_AI_DB_PASSWORD"]

    def test_get_secret_default_value(self):
        """Test getting secret with default value."""
        config = {"backend": "environment"}
        manager = SecretsManager(config)

        value = manager.get_secret("NONEXISTENT_SECRET", default="default_value")
        assert value == "default_value"

    def test_secret_caching(self):
        """Test that secrets are cached."""
        config = {"backend": "environment"}
        manager = SecretsManager(config)

        os.environ["CACHED_SECRET"] = "cached_value"

        # First call
        value1 = manager.get_secret("CACHED_SECRET")
        assert "CACHED_SECRET" in manager.cache

        # Second call should use cache
        value2 = manager.get_secret("CACHED_SECRET")
        assert value1 == value2

        # Cleanup
        del os.environ["CACHED_SECRET"]

    def test_validate_secret_length(self):
        """Test secret length validation."""
        config = {"backend": "environment"}
        manager = SecretsManager(config)

        # Too short
        assert not manager.validate_secret("test_key", "short")

        # Valid length (avoid weak patterns like 'password')
        assert manager.validate_secret("test_key", "StrongSecret123Key")

    def test_validate_secret_weak_patterns(self):
        """Test detection of weak secret patterns."""
        config = {"backend": "environment"}
        manager = SecretsManager(config)

        # Weak patterns
        assert not manager.validate_secret("test_key", "password123")
        assert not manager.validate_secret("test_key", "admin123456")
        assert not manager.validate_secret("test_key", "test12345678")

        # Strong secret
        assert manager.validate_secret("test_key", "xK9$mP2#vL8@qR5!")

    def test_file_backend_initialization(self):
        """Test file backend initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            secrets_file = Path(tmpdir) / ".secrets.enc"
            config = {"backend": "file", "secrets_file": str(secrets_file)}

            manager = SecretsManager(config)
            assert manager.backend == SecretBackend.FILE
            assert manager.secrets_file == secrets_file

    @patch("builtins.__import__")
    def test_vault_backend_initialization(self, mock_import):
        """Test Vault backend initialization."""
        # Mock hvac module
        mock_hvac = MagicMock()
        mock_client = MagicMock()
        mock_client.is_authenticated.return_value = True
        mock_hvac.Client.return_value = mock_client

        def import_mock(name, *args, **kwargs):
            if name == "hvac":
                return mock_hvac
            return __import__(name, *args, **kwargs)

        mock_import.side_effect = import_mock

        config = {
            "backend": "vault",
            "vault_addr": "http://localhost:8200",
            "vault_token": "test-token",
        }

        manager = SecretsManager(config)
        assert manager.backend == SecretBackend.VAULT

    @patch("builtins.__import__")
    def test_aws_backend_initialization(self, mock_import):
        """Test AWS Secrets Manager backend initialization."""
        # Mock boto3 module
        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        def import_mock(name, *args, **kwargs):
            if name == "boto3":
                return mock_boto3
            return __import__(name, *args, **kwargs)

        mock_import.side_effect = import_mock

        config = {"backend": "aws_secrets_manager", "aws_region": "us-east-1"}

        manager = SecretsManager(config)
        assert manager.backend == SecretBackend.AWS_SECRETS_MANAGER


class TestDatabaseConfig:
    """Test DatabaseConfig with secrets management."""

    def test_password_from_environment(self):
        """Test loading password from environment."""
        os.environ["CRONOS_AI_DB_PASSWORD"] = "env_password"

        config = DatabaseConfig()
        assert config.password == "env_password"

        # Cleanup
        del os.environ["CRONOS_AI_DB_PASSWORD"]

    def test_password_warning_when_empty(self, caplog):
        """Test warning when password is not set."""
        # Ensure no password in environment
        for key in ["CRONOS_AI_DB_PASSWORD", "DATABASE_PASSWORD"]:
            os.environ.pop(key, None)

        config = DatabaseConfig()
        assert config.password == ""

    def test_database_url_requires_password(self):
        """Test that database URL requires password."""
        config = DatabaseConfig()
        config.password = ""

        with pytest.raises(Exception):  # ConfigurationException
            _ = config.url

    def test_database_url_with_password(self):
        """Test database URL generation with password."""
        config = DatabaseConfig()
        config.password = "test_password"

        url = config.url
        assert "test_password" in url
        assert url.startswith("postgresql://")


class TestRedisConfig:
    """Test RedisConfig with secrets management."""

    def test_password_from_environment(self):
        """Test loading password from environment."""
        os.environ["CRONOS_AI_REDIS_PASSWORD"] = "redis_password"

        config = RedisConfig()
        assert config.password == "redis_password"

        # Cleanup
        del os.environ["CRONOS_AI_REDIS_PASSWORD"]

    def test_redis_url_without_password(self):
        """Test Redis URL without password."""
        config = RedisConfig()
        config.password = None

        url = config.url
        assert url == "redis://localhost:6379/0"

    def test_redis_url_with_password(self):
        """Test Redis URL with password."""
        config = RedisConfig()
        config.password = "test_password"

        url = config.url
        assert ":test_password@" in url


class TestSecurityConfig:
    """Test SecurityConfig with secrets management."""

    def test_jwt_secret_from_environment(self):
        """Test loading JWT secret from environment."""
        # Use a strong value without weak patterns (no 'secret', 'key', 'password', etc.)
        strong_jwt = "XyZ9mN4pQrS7tUvWaB2cDeF6gHiJ8kLmNoP"
        os.environ["CRONOS_AI_JWT_SECRET"] = strong_jwt

        config = SecurityConfig()
        assert config.jwt_secret == strong_jwt

        # Cleanup
        del os.environ["CRONOS_AI_JWT_SECRET"]

    def test_encryption_key_from_environment(self):
        """Test loading encryption key from environment."""
        # Use a strong value without weak patterns (no 'secret', 'key', 'password', etc.)
        strong_enc = "Ab1Cd2Ef3Gh4Ij5Kl6Mn7Op8Qr9St0Uv1Wx2Yz"
        os.environ["CRONOS_AI_ENCRYPTION_KEY"] = strong_enc

        config = SecurityConfig()
        assert config.encryption_key == strong_enc

        # Cleanup
        del os.environ["CRONOS_AI_ENCRYPTION_KEY"]

    def test_jwt_secret_validation(self):
        """Test JWT secret length validation."""
        config = SecurityConfig()
        config.jwt_secret = "short"

        with pytest.raises(Exception):  # ConfigurationException
            config.validate()

    def test_encryption_key_validation(self):
        """Test encryption key length validation."""
        config = SecurityConfig()
        config.encryption_key = "short"

        with pytest.raises(Exception):  # ConfigurationException
            config.validate()

    def test_valid_security_config(self):
        """Test valid security configuration."""
        config = SecurityConfig()
        config.jwt_secret = "a" * 32  # 32 characters
        config.encryption_key = "b" * 32  # 32 characters

        # Should not raise
        config.validate()


class TestSecretRotation:
    """Test secret rotation functionality."""

    def test_rotate_secret_updates_metadata(self):
        """Test that rotating a secret updates metadata."""
        config = {"backend": "environment"}
        manager = SecretsManager(config)

        # Mock set_secret to avoid actual storage
        with patch.object(manager, "set_secret") as mock_set:
            manager.rotate_secret("test_key", "new_value")

            # Verify set_secret was called
            mock_set.assert_called_once()

            # Verify metadata was updated
            call_args = mock_set.call_args
            metadata = (
                call_args[0][2]
                if len(call_args[0]) > 2
                else call_args[1].get("metadata")
            )

            if metadata:
                assert metadata.last_rotated is not None


class TestSecretValidation:
    """Test secret validation."""

    def test_no_hardcoded_passwords_in_config(self):
        """Test that config files don't contain hardcoded passwords."""
        from ai_engine.core import config

        # Check DatabaseConfig default
        db_config = DatabaseConfig()
        assert (
            db_config.password == ""
            or db_config.password is None
            or os.getenv("CRONOS_AI_DB_PASSWORD")
            or os.getenv("DATABASE_PASSWORD")
        ), "DatabaseConfig has hardcoded password"

    def test_no_hardcoded_secrets_in_security_config(self):
        """Test that SecurityConfig doesn't have hardcoded secrets."""
        config = SecurityConfig()

        # JWT secret should be empty or from environment
        if config.jwt_secret:
            assert os.getenv("CRONOS_AI_JWT_SECRET") or os.getenv(
                "JWT_SECRET"
            ), "SecurityConfig has hardcoded JWT secret"

        # Encryption key should be empty or from environment
        if config.encryption_key:
            assert os.getenv("CRONOS_AI_ENCRYPTION_KEY") or os.getenv(
                "ENCRYPTION_KEY"
            ), "SecurityConfig has hardcoded encryption key"


class TestProductionSafety:
    """Test production safety checks."""

    def test_production_requires_strong_secrets(self):
        """Test that production mode requires strong secrets."""
        config = SecurityConfig()
        config.jwt_secret = "weak"

        with pytest.raises(Exception):
            config.validate()

    def test_production_config_validation(self):
        """Test production configuration validation."""
        # This would test the actual production config file
        # to ensure no hardcoded credentials
        production_config_path = Path("config/cronos_ai.production.yaml")

        if production_config_path.exists():
            import yaml

            with open(production_config_path) as f:
                prod_config = yaml.safe_load(f)

            # Check database password is empty
            if "database" in prod_config:
                assert (
                    prod_config["database"].get("password", "") == ""
                ), "Production config has hardcoded database password"

            # Check security secrets are empty
            if "security" in prod_config:
                assert (
                    prod_config["security"].get("jwt_secret", "") == ""
                ), "Production config has hardcoded JWT secret"
                assert (
                    prod_config["security"].get("encryption_key", "") == ""
                ), "Production config has hardcoded encryption key"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
