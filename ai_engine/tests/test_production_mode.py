"""
QBITEL - Production Mode Detection Tests

Comprehensive test suite for production mode detection and validation
across all configuration classes.
"""

import os
import pytest
from unittest.mock import patch
import secrets

from ai_engine.core.config import (
    DatabaseConfig,
    RedisConfig,
    SecurityConfig,
    Config,
    ConfigurationException,
    Environment,
)


class TestProductionModeDetection:
    """Test production mode detection logic."""

    def test_production_mode_detected_from_qbitel_environment(self):
        """Test production mode detection from QBITEL_AI_ENVIRONMENT."""
        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "production"}, clear=True):
            config = DatabaseConfig.__new__(DatabaseConfig)
            assert config._is_production_mode() is True

    def test_production_mode_detected_from_environment(self):
        """Test production mode detection from ENVIRONMENT."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}, clear=True):
            config = DatabaseConfig.__new__(DatabaseConfig)
            assert config._is_production_mode() is True

    def test_production_mode_detected_from_prod_alias(self):
        """Test production mode detection from 'prod' alias."""
        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "prod"}, clear=True):
            config = DatabaseConfig.__new__(DatabaseConfig)
            assert config._is_production_mode() is True

    def test_production_mode_case_insensitive(self):
        """Test production mode detection is case-insensitive."""
        for env_value in ["PRODUCTION", "Production", "PROD", "Prod"]:
            with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": env_value}, clear=True):
                config = DatabaseConfig.__new__(DatabaseConfig)
                assert config._is_production_mode() is True

    def test_development_mode_default(self):
        """Test development mode is default when no environment set."""
        with patch.dict(os.environ, {}, clear=True):
            config = DatabaseConfig.__new__(DatabaseConfig)
            assert config._is_production_mode() is False

    def test_development_mode_explicit(self):
        """Test development mode when explicitly set."""
        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "development"}, clear=True):
            config = DatabaseConfig.__new__(DatabaseConfig)
            assert config._is_production_mode() is False

    def test_staging_mode_not_production(self):
        """Test staging mode is not treated as production."""
        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "staging"}, clear=True):
            config = DatabaseConfig.__new__(DatabaseConfig)
            assert config._is_production_mode() is False

    def test_testing_mode_not_production(self):
        """Test testing mode is not treated as production."""
        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "testing"}, clear=True):
            config = DatabaseConfig.__new__(DatabaseConfig)
            assert config._is_production_mode() is False

    def test_qbitel_environment_takes_priority(self):
        """Test QBITEL_AI_ENVIRONMENT takes priority over ENVIRONMENT."""
        with patch.dict(
            os.environ,
            {"QBITEL_AI_ENVIRONMENT": "production", "ENVIRONMENT": "development"},
        ):
            config = DatabaseConfig.__new__(DatabaseConfig)
            assert config._is_production_mode() is True


class TestProductionDatabaseRequirements:
    """Test production mode requirements for database configuration."""

    def test_production_requires_database_password(self):
        """Test production mode requires database password."""
        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "production"}, clear=True):
            with pytest.raises(ConfigurationException) as exc_info:
                DatabaseConfig()
            assert "Database password not configured" in str(exc_info.value)
            assert "REQUIRED" in str(exc_info.value)

    def test_production_rejects_short_database_password(self):
        """Test production mode rejects short database passwords."""
        with patch.dict(
            os.environ,
            {"QBITEL_AI_ENVIRONMENT": "production", "QBITEL_AI_DB_PASSWORD": "short"},
        ):
            with pytest.raises(ConfigurationException) as exc_info:
                DatabaseConfig()
            assert "too short" in str(exc_info.value).lower()

    def test_production_rejects_weak_database_password(self):
        """Test production mode rejects weak database passwords."""
        with patch.dict(
            os.environ,
            {
                "QBITEL_AI_ENVIRONMENT": "production",
                "QBITEL_AI_DB_PASSWORD": "password123456789",
            },
        ):
            with pytest.raises(ConfigurationException) as exc_info:
                DatabaseConfig()
            assert "weak patterns" in str(exc_info.value).lower()

    def test_production_accepts_strong_database_password(self):
        """Test production mode accepts strong database passwords."""
        strong_password = secrets.token_urlsafe(32)
        with patch.dict(
            os.environ,
            {
                "QBITEL_AI_ENVIRONMENT": "production",
                "QBITEL_AI_DB_PASSWORD": strong_password,
            },
        ):
            config = DatabaseConfig()
            assert config.password == strong_password

    def test_development_allows_missing_database_password(self):
        """Test development mode allows missing database password."""
        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "development"}, clear=True):
            config = DatabaseConfig()
            assert config.password == ""

    def test_development_warns_on_weak_database_password(self):
        """Test development mode warns but allows weak database passwords."""
        with patch.dict(
            os.environ,
            {
                "QBITEL_AI_ENVIRONMENT": "development",
                "QBITEL_AI_DB_PASSWORD": "weak123",
            },
        ):
            # Should not raise, just warn
            config = DatabaseConfig()
            assert config.password == "weak123"


class TestProductionRedisRequirements:
    """Test production mode requirements for Redis configuration."""

    def test_production_requires_redis_password(self):
        """Test production mode requires Redis password."""
        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "production"}, clear=True):
            with pytest.raises(ConfigurationException) as exc_info:
                RedisConfig()
            assert "Redis password not configured" in str(exc_info.value)
            assert "PRODUCTION mode" in str(exc_info.value)
            assert "CRITICAL security risk" in str(exc_info.value)

    def test_production_rejects_short_redis_password(self):
        """Test production mode rejects short Redis passwords."""
        with patch.dict(
            os.environ,
            {
                "QBITEL_AI_ENVIRONMENT": "production",
                "QBITEL_AI_REDIS_PASSWORD": "short",
            },
        ):
            with pytest.raises(ConfigurationException) as exc_info:
                RedisConfig()
            assert "too short" in str(exc_info.value).lower()

    def test_production_rejects_weak_redis_password(self):
        """Test production mode rejects weak Redis passwords."""
        with patch.dict(
            os.environ,
            {
                "QBITEL_AI_ENVIRONMENT": "production",
                "QBITEL_AI_REDIS_PASSWORD": "redis123456789abc",
            },
        ):
            with pytest.raises(ConfigurationException) as exc_info:
                RedisConfig()
            assert "weak patterns" in str(exc_info.value).lower()

    def test_production_accepts_strong_redis_password(self):
        """Test production mode accepts strong Redis passwords."""
        strong_password = secrets.token_urlsafe(32)
        with patch.dict(
            os.environ,
            {
                "QBITEL_AI_ENVIRONMENT": "production",
                "QBITEL_AI_REDIS_PASSWORD": strong_password,
            },
        ):
            config = RedisConfig()
            assert config.password == strong_password

    def test_development_allows_missing_redis_password(self):
        """Test development mode allows missing Redis password."""
        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "development"}, clear=True):
            config = RedisConfig()
            assert config.password is None


class TestProductionSecurityRequirements:
    """Test production mode requirements for security configuration."""

    def test_production_requires_jwt_secret(self):
        """Test production mode requires JWT secret."""
        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "production"}, clear=True):
            with pytest.raises(ConfigurationException) as exc_info:
                SecurityConfig()
            assert "JWT secret not configured" in str(exc_info.value)

    def test_production_requires_encryption_key_when_enabled(self):
        """Test production mode requires encryption key when encryption enabled."""
        jwt_secret = secrets.token_urlsafe(48)
        with patch.dict(
            os.environ,
            {"QBITEL_AI_ENVIRONMENT": "production", "QBITEL_AI_JWT_SECRET": jwt_secret},
            clear=True,
        ):
            with pytest.raises(ConfigurationException) as exc_info:
                SecurityConfig(enable_encryption=True)
            assert "Encryption key not configured" in str(exc_info.value)

    def test_production_forbids_cors_wildcard(self):
        """Test production mode forbids CORS wildcard."""
        jwt_secret = secrets.token_urlsafe(48)
        encryption_key = secrets.token_urlsafe(48)

        with patch.dict(
            os.environ,
            {
                "QBITEL_AI_ENVIRONMENT": "production",
                "QBITEL_AI_JWT_SECRET": jwt_secret,
                "QBITEL_AI_ENCRYPTION_KEY": encryption_key,
            },
        ):
            with pytest.raises(ConfigurationException) as exc_info:
                SecurityConfig(cors_origins=["*"])
            assert "CORS wildcard" in str(exc_info.value)
            assert "FORBIDDEN" in str(exc_info.value)

    def test_production_allows_specific_cors_origins(self):
        """Test production mode allows specific CORS origins."""
        jwt_secret = secrets.token_urlsafe(48)
        encryption_key = secrets.token_urlsafe(48)

        with patch.dict(
            os.environ,
            {
                "QBITEL_AI_ENVIRONMENT": "production",
                "QBITEL_AI_JWT_SECRET": jwt_secret,
                "QBITEL_AI_ENCRYPTION_KEY": encryption_key,
            },
        ):
            config = SecurityConfig(
                cors_origins=[
                    "https://app.example.com",
                    "https://dashboard.example.com",
                ]
            )
            assert "*" not in config.cors_origins

    def test_production_rejects_weak_jwt_secret(self):
        """Test production mode rejects weak JWT secrets."""
        with patch.dict(
            os.environ,
            {
                "QBITEL_AI_ENVIRONMENT": "production",
                "QBITEL_AI_JWT_SECRET": "secret123456789012345678901234567890",
                "QBITEL_AI_ENCRYPTION_KEY": secrets.token_urlsafe(48),
            },
        ):
            with pytest.raises(ConfigurationException) as exc_info:
                SecurityConfig()
            assert "weak patterns" in str(exc_info.value).lower()

    def test_production_rejects_low_entropy_secrets(self):
        """Test production mode rejects low entropy secrets."""
        with patch.dict(
            os.environ,
            {
                "QBITEL_AI_ENVIRONMENT": "production",
                "QBITEL_AI_JWT_SECRET": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "QBITEL_AI_ENCRYPTION_KEY": secrets.token_urlsafe(48),
            },
        ):
            with pytest.raises(ConfigurationException) as exc_info:
                SecurityConfig()
            assert "entropy" in str(exc_info.value).lower()

    def test_development_allows_cors_wildcard(self):
        """Test development mode allows CORS wildcard."""
        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "development"}, clear=True):
            # Should not raise
            config = SecurityConfig(cors_origins=["*"])
            assert "*" in config.cors_origins

    def test_development_allows_missing_secrets(self):
        """Test development mode allows missing secrets with warnings."""
        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "development"}, clear=True):
            # Should not raise, just warn
            config = SecurityConfig()
            assert config.jwt_secret is None


class TestProductionConfigValidation:
    """Test production mode validation in main Config class."""

    def test_production_config_requires_all_secrets(self):
        """Test production Config requires all secrets."""
        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "production"}, clear=True):
            with pytest.raises(ConfigurationException):
                Config.load_from_env()

    def test_production_config_with_all_secrets_succeeds(self):
        """Test production Config succeeds with all required secrets."""
        db_password = secrets.token_urlsafe(32)
        redis_password = secrets.token_urlsafe(32)
        jwt_secret = secrets.token_urlsafe(48)
        encryption_key = secrets.token_urlsafe(48)

        with patch.dict(
            os.environ,
            {
                "QBITEL_AI_ENVIRONMENT": "production",
                "QBITEL_AI_DB_PASSWORD": db_password,
                "QBITEL_AI_REDIS_PASSWORD": redis_password,
                "QBITEL_AI_JWT_SECRET": jwt_secret,
                "QBITEL_AI_ENCRYPTION_KEY": encryption_key,
            },
        ):
            # Create config manually to avoid CORS wildcard issue
            from ai_engine.core.config import (
                DatabaseConfig,
                RedisConfig,
                SecurityConfig,
            )

            db_config = DatabaseConfig()
            redis_config = RedisConfig()
            security_config = SecurityConfig(cors_origins=["https://app.example.com"])

            assert db_config.password == db_password
            assert redis_config.password == redis_password
            assert security_config.jwt_secret == jwt_secret
            assert security_config.encryption_key == encryption_key

    def test_development_config_allows_defaults(self):
        """Test development Config allows default values."""
        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "development"}, clear=True):
            config = Config.load_from_env()
            assert config.environment == Environment.DEVELOPMENT


class TestProductionErrorMessages:
    """Test production mode error messages are clear and actionable."""

    def test_database_password_error_is_actionable(self):
        """Test database password error provides clear remediation."""
        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "production"}, clear=True):
            with pytest.raises(ConfigurationException) as exc_info:
                DatabaseConfig()
            error_msg = str(exc_info.value)
            assert "REQUIRED" in error_msg
            assert "QBITEL_AI_DB_PASSWORD" in error_msg
            assert "Generate a secure password" in error_msg
            assert "python -c" in error_msg
            assert "secrets.token_urlsafe" in error_msg

    def test_redis_password_error_is_actionable(self):
        """Test Redis password error provides clear remediation."""
        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "production"}, clear=True):
            with pytest.raises(ConfigurationException) as exc_info:
                RedisConfig()
            error_msg = str(exc_info.value)
            assert "REQUIRED" in error_msg
            assert "QBITEL_AI_REDIS_PASSWORD" in error_msg
            assert "CRITICAL security risk" in error_msg

    def test_jwt_secret_error_is_actionable(self):
        """Test JWT secret error provides clear remediation."""
        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "production"}, clear=True):
            with pytest.raises(ConfigurationException) as exc_info:
                SecurityConfig()
            error_msg = str(exc_info.value)
            assert "REQUIRED" in error_msg
            assert "QBITEL_AI_JWT_SECRET" in error_msg
            assert "Generate a secure JWT secret" in error_msg

    def test_cors_wildcard_error_is_actionable(self):
        """Test CORS wildcard error provides clear remediation."""
        jwt_secret = secrets.token_urlsafe(48)
        encryption_key = secrets.token_urlsafe(48)

        with patch.dict(
            os.environ,
            {
                "QBITEL_AI_ENVIRONMENT": "production",
                "QBITEL_AI_JWT_SECRET": jwt_secret,
                "QBITEL_AI_ENCRYPTION_KEY": encryption_key,
            },
        ):
            with pytest.raises(ConfigurationException) as exc_info:
                SecurityConfig(cors_origins=["*"])
            error_msg = str(exc_info.value)
            assert "FORBIDDEN" in error_msg
            assert "Specify explicit allowed origins" in error_msg
            assert "critical security vulnerability" in error_msg


class TestProductionModeConsistency:
    """Test production mode detection is consistent across all config classes."""

    def test_all_configs_use_same_production_detection(self):
        """Test all config classes detect production mode consistently."""
        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "production"}, clear=True):
            db_config = DatabaseConfig.__new__(DatabaseConfig)
            redis_config = RedisConfig.__new__(RedisConfig)
            security_config = SecurityConfig.__new__(SecurityConfig)

            assert db_config._is_production_mode() is True
            assert redis_config._is_production_mode() is True
            assert security_config._is_production_mode() is True

    def test_all_configs_respect_environment_priority(self):
        """Test all configs respect QBITEL_AI_ENVIRONMENT priority."""
        with patch.dict(
            os.environ,
            {"QBITEL_AI_ENVIRONMENT": "production", "ENVIRONMENT": "development"},
        ):
            db_config = DatabaseConfig.__new__(DatabaseConfig)
            redis_config = RedisConfig.__new__(RedisConfig)
            security_config = SecurityConfig.__new__(SecurityConfig)

            assert db_config._is_production_mode() is True
            assert redis_config._is_production_mode() is True
            assert security_config._is_production_mode() is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
