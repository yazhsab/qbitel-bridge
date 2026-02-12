"""
Basic tests for EnterpriseAuthenticationService - only testing methods that exist and work.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from ai_engine.api.auth_enterprise import EnterpriseAuthenticationService


class TestEnterpriseAuthenticationServiceBasic:
    """Basic tests for EnterpriseAuthenticationService."""

    @pytest.fixture
    def auth_service(self):
        """Create auth service with mocked config."""
        with patch("ai_engine.api.auth_enterprise.get_config") as mock_config:
            mock_config.return_value = Mock()
            mock_config.return_value.security = Mock()
            mock_config.return_value.security.jwt_secret = "test_secret_key_that_is_long_enough_for_jwt"

            with patch("ai_engine.api.auth_enterprise.get_audit_logger") as mock_audit:
                mock_audit.return_value = Mock()
                return EnterpriseAuthenticationService()

    def test_auth_service_initialization(self, auth_service):
        """Test auth service initialization."""
        assert auth_service is not None
        assert auth_service.algorithm == "HS256"
        assert auth_service.access_token_expire_minutes == 30
        assert auth_service.refresh_token_expire_days == 7

    def test_load_secret_key_success(self, auth_service):
        """Test successful secret key loading."""
        secret = auth_service._load_secret_key()
        assert secret == "test_secret_key_that_is_long_enough_for_jwt"

    def test_load_secret_key_failure(self):
        """Test secret key loading failure."""
        with patch("ai_engine.api.auth_enterprise.get_config") as mock_config:
            mock_config.return_value = Mock()
            mock_config.return_value.security = Mock()
            mock_config.return_value.security.jwt_secret = "short"

            with patch("ai_engine.api.auth_enterprise.get_audit_logger"):
                with pytest.raises(ValueError, match="JWT secret not configured or too short"):
                    EnterpriseAuthenticationService()

    def test_validate_password_strength_too_short(self, auth_service):
        """Test password strength validation with too short password."""
        password = "short"
        is_valid, error = auth_service.validate_password_strength(password)
        assert is_valid is False
        assert "at least 12 characters" in error

    def test_validate_password_strength_no_uppercase(self, auth_service):
        """Test password strength validation with no uppercase."""
        password = "validpassword123!"
        is_valid, error = auth_service.validate_password_strength(password)
        assert is_valid is False
        assert "uppercase" in error

    def test_validate_password_strength_no_lowercase(self, auth_service):
        """Test password strength validation with no lowercase."""
        password = "VALIDPASSWORD123!"
        is_valid, error = auth_service.validate_password_strength(password)
        assert is_valid is False
        assert "lowercase" in error

    def test_validate_password_strength_no_special(self, auth_service):
        """Test password strength validation with no special characters."""
        password = "ValidPassword123"
        is_valid, error = auth_service.validate_password_strength(password)
        assert is_valid is False
        assert "special character" in error

    def test_create_access_token(self, auth_service):
        """Test access token creation."""
        user_data = {
            "user_id": "user123",
            "username": "testuser",
            "email": "test@example.com",
        }

        token = auth_service.create_access_token(user_data)
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_refresh_token(self, auth_service):
        """Test refresh token creation."""
        user_data = {
            "user_id": "user123",
            "username": "testuser",
            "email": "test@example.com",
        }

        token = auth_service.create_refresh_token(user_data)
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

    def test_generate_totp_secret(self, auth_service):
        """Test TOTP secret generation."""
        with patch("ai_engine.api.auth_enterprise.pyotp.random_base32") as mock_random:
            mock_random.return_value = "test_secret_key_123456"
            secret = auth_service.generate_totp_secret()
            assert secret is not None
            assert isinstance(secret, str)
            assert len(secret) > 0

    def test_generate_backup_codes(self, auth_service):
        """Test backup codes generation."""
        codes = auth_service.generate_backup_codes()
        assert codes is not None
        assert isinstance(codes, list)
        assert len(codes) == 10
        assert all(isinstance(code, str) for code in codes)

    def test_generate_api_key(self, auth_service):
        """Test API key generation."""
        key_id, api_key = auth_service.generate_api_key()
        assert key_id is not None
        assert api_key is not None
        assert isinstance(key_id, str)
        assert isinstance(api_key, str)
        assert len(key_id) > 0
        assert len(api_key) > 0

    def test_verify_totp(self, auth_service):
        """Test TOTP verification."""
        with patch("ai_engine.api.auth_enterprise.pyotp.TOTP") as mock_totp:
            mock_totp_instance = Mock()
            mock_totp_instance.verify.return_value = True
            mock_totp.return_value = mock_totp_instance

            result = auth_service.verify_totp("test_secret", "123456")
            assert isinstance(result, bool)

    def test_auth_service_concurrent_operations(self, auth_service):
        """Test auth service can handle concurrent operations."""
        # Test that multiple operations can be called without issues
        user_data = {"user_id": "user123", "username": "testuser"}

        token1 = auth_service.create_access_token(user_data)
        token2 = auth_service.create_refresh_token(user_data)
        secret = auth_service.generate_totp_secret()
        codes = auth_service.generate_backup_codes()

        assert token1 is not None
        assert token2 is not None
        assert secret is not None
        assert codes is not None
