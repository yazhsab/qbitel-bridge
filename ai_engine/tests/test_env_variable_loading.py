"""
CRONOS AI - Environment Variable Loading Tests

Comprehensive test suite for production-ready environment variable loading
and validation across all configuration classes.
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
    ConfigurationException
)


class TestDatabaseConfigEnvLoading:
    """Test DatabaseConfig environment variable loading and validation."""
    
    def test_load_password_from_cronos_ai_prefix(self):
        """Test loading password from CRONOS_AI_DB_PASSWORD."""
        password = secrets.token_urlsafe(32)
        with patch.dict(os.environ, {'CRONOS_AI_DB_PASSWORD': password}):
            config = DatabaseConfig()
            assert config.password == password
    
    def test_load_password_from_generic_prefix(self):
        """Test loading password from DATABASE_PASSWORD."""
        password = secrets.token_urlsafe(32)
        with patch.dict(os.environ, {'DATABASE_PASSWORD': password}, clear=True):
            config = DatabaseConfig()
            assert config.password == password
    
    def test_cronos_ai_prefix_takes_priority(self):
        """Test that CRONOS_AI_DB_PASSWORD takes priority."""
        cronos_password = secrets.token_urlsafe(32)
        generic_password = secrets.token_urlsafe(32)
        with patch.dict(os.environ, {
            'CRONOS_AI_DB_PASSWORD': cronos_password,
            'DATABASE_PASSWORD': generic_password
        }):
            config = DatabaseConfig()
            assert config.password == cronos_password
    
    def test_production_mode_requires_password(self):
        """Test that production mode requires a password."""
        with patch.dict(os.environ, {'CRONOS_AI_ENVIRONMENT': 'production'}, clear=True):
            with pytest.raises(ConfigurationException) as exc_info:
                DatabaseConfig()
            assert "Database password not configured" in str(exc_info.value)
    
    def test_development_mode_allows_missing_password(self):
        """Test that development mode allows missing password with warning."""
        with patch.dict(os.environ, {'CRONOS_AI_ENVIRONMENT': 'development'}, clear=True):
            config = DatabaseConfig()
            assert config.password == ''
    
    def test_password_minimum_length_validation(self):
        """Test password minimum length validation."""
        short_password = "short123"
        with patch.dict(os.environ, {
            'CRONOS_AI_DB_PASSWORD': short_password,
            'CRONOS_AI_ENVIRONMENT': 'production'
        }):
            with pytest.raises(ConfigurationException) as exc_info:
                DatabaseConfig()
            assert "too short" in str(exc_info.value).lower()
    
    def test_password_weak_pattern_detection(self):
        """Test detection of weak password patterns."""
        weak_password = "password123456789"  # Contains 'password'
        with patch.dict(os.environ, {
            'CRONOS_AI_DB_PASSWORD': weak_password,
            'CRONOS_AI_ENVIRONMENT': 'production'
        }):
            with pytest.raises(ConfigurationException) as exc_info:
                DatabaseConfig()
            assert "weak patterns" in str(exc_info.value).lower()
    
    def test_password_sequential_chars_detection(self):
        """Test detection of sequential characters."""
        sequential_password = "abcdefghijklmnop"
        with patch.dict(os.environ, {
            'CRONOS_AI_DB_PASSWORD': sequential_password,
            'CRONOS_AI_ENVIRONMENT': 'production'
        }):
            with pytest.raises(ConfigurationException) as exc_info:
                DatabaseConfig()
            assert "sequential" in str(exc_info.value).lower()
    
    def test_password_repeated_chars_detection(self):
        """Test detection of repeated characters."""
        repeated_password = "aaaaaaaaaaaaaaaa"
        with patch.dict(os.environ, {
            'CRONOS_AI_DB_PASSWORD': repeated_password,
            'CRONOS_AI_ENVIRONMENT': 'production'
        }):
            with pytest.raises(ConfigurationException) as exc_info:
                DatabaseConfig()
            assert "repeated" in str(exc_info.value).lower()
    
    def test_strong_password_passes_validation(self):
        """Test that a strong password passes all validations."""
        strong_password = secrets.token_urlsafe(32)
        with patch.dict(os.environ, {
            'CRONOS_AI_DB_PASSWORD': strong_password,
            'CRONOS_AI_ENVIRONMENT': 'production'
        }):
            config = DatabaseConfig()
            assert config.password == strong_password
    
    def test_database_url_generation(self):
        """Test database URL generation."""
        password = secrets.token_urlsafe(32)
        with patch.dict(os.environ, {'CRONOS_AI_DB_PASSWORD': password}):
            config = DatabaseConfig()
            url = config.url
            assert "postgresql://" in url
            assert password in url
            assert config.username in url
            assert config.host in url


class TestRedisConfigEnvLoading:
    """Test RedisConfig environment variable loading and validation."""
    
    def test_load_password_from_cronos_ai_prefix(self):
        """Test loading password from CRONOS_AI_REDIS_PASSWORD."""
        password = secrets.token_urlsafe(32)
        with patch.dict(os.environ, {'CRONOS_AI_REDIS_PASSWORD': password}):
            config = RedisConfig()
            assert config.password == password
    
    def test_load_password_from_generic_prefix(self):
        """Test loading password from REDIS_PASSWORD."""
        password = secrets.token_urlsafe(32)
        with patch.dict(os.environ, {'REDIS_PASSWORD': password}, clear=True):
            config = RedisConfig()
            assert config.password == password
    
    def test_production_mode_requires_password(self):
        """Test that production mode requires Redis authentication."""
        with patch.dict(os.environ, {'CRONOS_AI_ENVIRONMENT': 'production'}, clear=True):
            with pytest.raises(ConfigurationException) as exc_info:
                RedisConfig()
            assert "Redis password not configured" in str(exc_info.value)
            assert "CRITICAL security risk" in str(exc_info.value)
    
    def test_development_mode_allows_missing_password(self):
        """Test that development mode allows missing password."""
        with patch.dict(os.environ, {'CRONOS_AI_ENVIRONMENT': 'development'}, clear=True):
            config = RedisConfig()
            assert config.password is None
    
    def test_password_minimum_length_validation(self):
        """Test password minimum length validation."""
        short_password = "short123"
        with patch.dict(os.environ, {
            'CRONOS_AI_REDIS_PASSWORD': short_password,
            'CRONOS_AI_ENVIRONMENT': 'production'
        }):
            with pytest.raises(ConfigurationException) as exc_info:
                RedisConfig()
            assert "too short" in str(exc_info.value).lower()
    
    def test_password_weak_pattern_detection(self):
        """Test detection of weak password patterns."""
        weak_password = "redis123456789abc"  # Contains 'redis'
        with patch.dict(os.environ, {
            'CRONOS_AI_REDIS_PASSWORD': weak_password,
            'CRONOS_AI_ENVIRONMENT': 'production'
        }):
            with pytest.raises(ConfigurationException) as exc_info:
                RedisConfig()
            assert "weak patterns" in str(exc_info.value).lower()
    
    def test_strong_password_passes_validation(self):
        """Test that a strong password passes all validations."""
        strong_password = secrets.token_urlsafe(32)
        with patch.dict(os.environ, {
            'CRONOS_AI_REDIS_PASSWORD': strong_password,
            'CRONOS_AI_ENVIRONMENT': 'production'
        }):
            config = RedisConfig()
            assert config.password == strong_password
    
    def test_redis_url_generation_with_auth(self):
        """Test Redis URL generation with authentication."""
        password = secrets.token_urlsafe(32)
        with patch.dict(os.environ, {'CRONOS_AI_REDIS_PASSWORD': password}):
            config = RedisConfig()
            url = config.url
            assert "redis://" in url
            assert password in url
    
    def test_redis_url_generation_without_auth(self):
        """Test Redis URL generation without authentication."""
        with patch.dict(os.environ, {'CRONOS_AI_ENVIRONMENT': 'development'}, clear=True):
            config = RedisConfig()
            url = config.url
            assert "redis://" in url
            assert "@" not in url  # No auth in URL


class TestSecurityConfigEnvLoading:
    """Test SecurityConfig environment variable loading and validation."""
    
    def test_load_jwt_secret_from_cronos_ai_prefix(self):
        """Test loading JWT secret from CRONOS_AI_JWT_SECRET."""
        jwt_secret = secrets.token_urlsafe(48)
        with patch.dict(os.environ, {'CRONOS_AI_JWT_SECRET': jwt_secret}):
            config = SecurityConfig()
            assert config.jwt_secret == jwt_secret
    
    def test_load_jwt_secret_from_generic_prefix(self):
        """Test loading JWT secret from JWT_SECRET."""
        jwt_secret = secrets.token_urlsafe(48)
        with patch.dict(os.environ, {'JWT_SECRET': jwt_secret}, clear=True):
            config = SecurityConfig()
            assert config.jwt_secret == jwt_secret
    
    def test_load_encryption_key_from_cronos_ai_prefix(self):
        """Test loading encryption key from CRONOS_AI_ENCRYPTION_KEY."""
        encryption_key = secrets.token_urlsafe(48)
        with patch.dict(os.environ, {'CRONOS_AI_ENCRYPTION_KEY': encryption_key}):
            config = SecurityConfig()
            assert config.encryption_key == encryption_key
    
    def test_load_encryption_key_from_generic_prefix(self):
        """Test loading encryption key from ENCRYPTION_KEY."""
        encryption_key = secrets.token_urlsafe(48)
        with patch.dict(os.environ, {'ENCRYPTION_KEY': encryption_key}, clear=True):
            config = SecurityConfig()
            assert config.encryption_key == encryption_key
    
    def test_load_api_key_from_cronos_ai_prefix(self):
        """Test loading API key from CRONOS_AI_API_KEY."""
        api_key = f"cronos_{secrets.token_urlsafe(32)}"
        with patch.dict(os.environ, {'CRONOS_AI_API_KEY': api_key}):
            config = SecurityConfig()
            assert config.api_key == api_key
    
    def test_load_api_key_from_generic_prefix(self):
        """Test loading API key from API_KEY."""
        api_key = f"cronos_{secrets.token_urlsafe(32)}"
        with patch.dict(os.environ, {'API_KEY': api_key}, clear=True):
            config = SecurityConfig()
            assert config.api_key == api_key
    
    def test_production_mode_requires_jwt_secret(self):
        """Test that production mode requires JWT secret."""
        with patch.dict(os.environ, {'CRONOS_AI_ENVIRONMENT': 'production'}, clear=True):
            with pytest.raises(ConfigurationException) as exc_info:
                SecurityConfig()
            assert "JWT secret not configured" in str(exc_info.value)
    
    def test_production_mode_requires_encryption_key_when_enabled(self):
        """Test that production mode requires encryption key when encryption is enabled."""
        jwt_secret = secrets.token_urlsafe(48)
        with patch.dict(os.environ, {
            'CRONOS_AI_ENVIRONMENT': 'production',
            'CRONOS_AI_JWT_SECRET': jwt_secret
        }, clear=True):
            with pytest.raises(ConfigurationException) as exc_info:
                SecurityConfig(enable_encryption=True)
            assert "Encryption key not configured" in str(exc_info.value)
    
    def test_jwt_secret_minimum_length_validation(self):
        """Test JWT secret minimum length validation."""
        short_secret = "short123"
        encryption_key = secrets.token_urlsafe(48)
        with patch.dict(os.environ, {
            'CRONOS_AI_JWT_SECRET': short_secret,
            'CRONOS_AI_ENCRYPTION_KEY': encryption_key,
            'CRONOS_AI_ENVIRONMENT': 'production'
        }):
            with pytest.raises(ConfigurationException) as exc_info:
                SecurityConfig()
            assert "too short" in str(exc_info.value).lower()
    
    def test_jwt_secret_weak_pattern_detection(self):
        """Test detection of weak JWT secret patterns."""
        weak_secret = "secret123456789012345678901234567890"  # Contains 'secret'
        encryption_key = secrets.token_urlsafe(48)
        with patch.dict(os.environ, {
            'CRONOS_AI_JWT_SECRET': weak_secret,
            'CRONOS_AI_ENCRYPTION_KEY': encryption_key,
            'CRONOS_AI_ENVIRONMENT': 'production'
        }):
            with pytest.raises(ConfigurationException) as exc_info:
                SecurityConfig()
            assert "weak patterns" in str(exc_info.value).lower()
    
    def test_jwt_secret_low_entropy_detection(self):
        """Test detection of low entropy in JWT secret."""
        low_entropy_secret = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        encryption_key = secrets.token_urlsafe(48)
        with patch.dict(os.environ, {
            'CRONOS_AI_JWT_SECRET': low_entropy_secret,
            'CRONOS_AI_ENCRYPTION_KEY': encryption_key,
            'CRONOS_AI_ENVIRONMENT': 'production'
        }):
            with pytest.raises(ConfigurationException) as exc_info:
                SecurityConfig()
            assert "entropy" in str(exc_info.value).lower()
    
    def test_api_key_minimum_length_validation(self):
        """Test API key minimum length validation."""
        short_key = "short123"
        with patch.dict(os.environ, {
            'CRONOS_AI_JWT_SECRET': secrets.token_urlsafe(48),
            'CRONOS_AI_ENCRYPTION_KEY': secrets.token_urlsafe(48),
            'CRONOS_AI_API_KEY': short_key,
            'CRONOS_AI_ENVIRONMENT': 'production'
        }):
            with pytest.raises(ConfigurationException) as exc_info:
                SecurityConfig()
            assert "too short" in str(exc_info.value).lower()
    
    def test_api_key_format_validation(self):
        """Test API key format validation."""
        # API key without digits
        no_digits_key = "abcdefghijklmnopqrstuvwxyzabcdefgh"
        with patch.dict(os.environ, {
            'CRONOS_AI_JWT_SECRET': secrets.token_urlsafe(48),
            'CRONOS_AI_ENCRYPTION_KEY': secrets.token_urlsafe(48),
            'CRONOS_AI_API_KEY': no_digits_key,
            'CRONOS_AI_ENVIRONMENT': 'production'
        }):
            with pytest.raises(ConfigurationException) as exc_info:
                SecurityConfig()
            assert "should contain digits" in str(exc_info.value).lower()
    
    def test_strong_secrets_pass_validation(self):
        """Test that strong secrets pass all validations."""
        jwt_secret = secrets.token_urlsafe(48)
        encryption_key = secrets.token_urlsafe(48)
        api_key = f"cronos_{secrets.token_urlsafe(32)}"
        
        with patch.dict(os.environ, {
            'CRONOS_AI_JWT_SECRET': jwt_secret,
            'CRONOS_AI_ENCRYPTION_KEY': encryption_key,
            'CRONOS_AI_API_KEY': api_key,
            'CRONOS_AI_ENVIRONMENT': 'production'
        }):
            config = SecurityConfig(cors_origins=["https://app.example.com"])
            assert config.jwt_secret == jwt_secret
            assert config.encryption_key == encryption_key
            assert config.api_key == api_key
    
    def test_cors_wildcard_not_allowed_in_production(self):
        """Test that CORS wildcard is not allowed in production."""
        jwt_secret = secrets.token_urlsafe(48)
        encryption_key = secrets.token_urlsafe(48)
        
        with patch.dict(os.environ, {
            'CRONOS_AI_JWT_SECRET': jwt_secret,
            'CRONOS_AI_ENCRYPTION_KEY': encryption_key,
            'CRONOS_AI_ENVIRONMENT': 'production'
        }):
            with pytest.raises(ConfigurationException) as exc_info:
                SecurityConfig(cors_origins=["*"])
            assert "CORS wildcard" in str(exc_info.value)


class TestConfigIntegration:
    """Test full Config class integration with environment variables."""
    
    def test_full_config_from_environment(self):
        """Test loading full configuration from environment variables."""
        db_password = secrets.token_urlsafe(32)
        redis_password = secrets.token_urlsafe(32)
        jwt_secret = secrets.token_urlsafe(48)
        encryption_key = secrets.token_urlsafe(48)
        
        with patch.dict(os.environ, {
            'CRONOS_AI_ENVIRONMENT': 'production',
            'CRONOS_AI_DB_PASSWORD': db_password,
            'CRONOS_AI_REDIS_PASSWORD': redis_password,
            'CRONOS_AI_JWT_SECRET': jwt_secret,
            'CRONOS_AI_ENCRYPTION_KEY': encryption_key
        }):
            # Create config with explicit CORS origins for production
            from ai_engine.core.config import SecurityConfig as SecConfig
            security_config = SecConfig(cors_origins=["https://app.example.com"])
            
            config = Config(security=security_config)
            
            assert config.database.password == db_password
            assert config.redis.password == redis_password
            assert config.security.jwt_secret == jwt_secret
            assert config.security.encryption_key == encryption_key
    
    def test_production_mode_validation(self):
        """Test that production mode enforces all security requirements."""
        with patch.dict(os.environ, {'CRONOS_AI_ENVIRONMENT': 'production'}, clear=True):
            with pytest.raises(ConfigurationException):
                Config.load_from_env()
    
    def test_development_mode_allows_defaults(self):
        """Test that development mode allows default values."""
        with patch.dict(os.environ, {'CRONOS_AI_ENVIRONMENT': 'development'}, clear=True):
            config = Config.load_from_env()
            assert config.environment.value == 'development'


class TestErrorMessages:
    """Test that error messages are clear and actionable."""
    
    def test_database_password_error_includes_remediation(self):
        """Test that database password error includes remediation steps."""
        with patch.dict(os.environ, {'CRONOS_AI_ENVIRONMENT': 'production'}, clear=True):
            with pytest.raises(ConfigurationException) as exc_info:
                DatabaseConfig()
            error_msg = str(exc_info.value)
            assert "REQUIRED" in error_msg
            assert "CRONOS_AI_DB_PASSWORD" in error_msg
            assert "Generate a secure password" in error_msg
            assert "python -c" in error_msg
    
    def test_redis_password_error_includes_remediation(self):
        """Test that Redis password error includes remediation steps."""
        with patch.dict(os.environ, {'CRONOS_AI_ENVIRONMENT': 'production'}, clear=True):
            with pytest.raises(ConfigurationException) as exc_info:
                RedisConfig()
            error_msg = str(exc_info.value)
            assert "REQUIRED" in error_msg
            assert "CRONOS_AI_REDIS_PASSWORD" in error_msg
            assert "Generate a secure password" in error_msg
    
    def test_jwt_secret_error_includes_remediation(self):
        """Test that JWT secret error includes remediation steps."""
        with patch.dict(os.environ, {'CRONOS_AI_ENVIRONMENT': 'production'}, clear=True):
            with pytest.raises(ConfigurationException) as exc_info:
                SecurityConfig()
            error_msg = str(exc_info.value)
            assert "REQUIRED" in error_msg
            assert "CRONOS_AI_JWT_SECRET" in error_msg
            assert "Generate a secure JWT secret" in error_msg
            assert "secrets.token_urlsafe" in error_msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])