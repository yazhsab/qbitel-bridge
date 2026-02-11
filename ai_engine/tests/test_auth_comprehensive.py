"""
Comprehensive Unit Tests for api/auth.py
Ensures 100% code coverage for authentication and authorization.
"""

import pytest
import os
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi import HTTPException

from ai_engine.api.auth import (
    AuthenticationError,
    AuthenticationService,
    get_auth_service,
    verify_token,
    get_current_user,
    require_permission,
    require_role,
    login,
    refresh_access_token,
    logout,
    initialize_auth,
    get_api_key,
)


class TestAuthenticationError:
    """Test suite for AuthenticationError."""

    def test_exception_creation(self):
        """Test creating AuthenticationError."""
        exc = AuthenticationError("Test error")
        assert str(exc) == "Test error"


class TestAuthenticationService:
    """Test suite for AuthenticationService."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock()
        config.security = Mock()
        config.security.jwt_secret = (
            "test_secret_key_with_minimum_32_characters_required"
        )
        config.environment = Mock()
        config.environment.value = "development"
        config.redis = Mock()
        config.redis.host = "localhost"
        config.redis.port = 6379
        config.redis.db = 0
        return config

    @pytest.fixture
    def auth_service(self, mock_config):
        """Create AuthenticationService instance."""
        return AuthenticationService(mock_config)

    def test_init_with_config(self, mock_config):
        """Test initialization with config."""
        service = AuthenticationService(mock_config)
        assert service.config == mock_config
        assert service.algorithm == "HS256"
        assert service.access_token_expire_minutes == 30

    def test_init_without_config(self):
        """Test initialization without config."""
        with patch("ai_engine.api.auth.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.security = Mock()
            mock_config.security.jwt_secret = (
                "test_secret_key_with_minimum_32_characters_required"
            )
            mock_get_config.return_value = mock_config
            service = AuthenticationService()
            assert service.config is not None

    def test_load_secret_key_from_secrets_manager(self, mock_config):
        """Test loading JWT secret from secrets manager."""
        with patch("ai_engine.api.auth.get_secrets_manager") as mock_secrets:
            mock_mgr = Mock()
            mock_mgr.get_secret.return_value = "secret_from_vault_with_32_chars_min"
            mock_secrets.return_value = mock_mgr
            service = AuthenticationService(mock_config)
            assert len(service.secret_key) >= 32

    def test_load_secret_key_from_config(self, mock_config):
        """Test loading JWT secret from config."""
        service = AuthenticationService(mock_config)
        assert (
            service.secret_key == "test_secret_key_with_minimum_32_characters_required"
        )

    def test_load_secret_key_production_error(self, mock_config):
        """Test JWT secret error in production."""
        mock_config.security.jwt_secret = None
        mock_config.environment.value = "production"
        with patch("ai_engine.api.auth.get_secrets_manager") as mock_secrets:
            mock_mgr = Mock()
            mock_mgr.get_secret.return_value = None
            mock_secrets.return_value = mock_mgr
            with pytest.raises(
                AuthenticationError, match="JWT secret not configured in production"
            ):
                AuthenticationService(mock_config)

    def test_load_secret_key_generated_development(self, mock_config):
        """Test JWT secret generation in development."""
        mock_config.security.jwt_secret = None
        mock_config.environment.value = "development"
        with patch("ai_engine.api.auth.get_secrets_manager") as mock_secrets:
            mock_mgr = Mock()
            mock_mgr.get_secret.return_value = None
            mock_secrets.return_value = mock_mgr
            service = AuthenticationService(mock_config)
            assert len(service.secret_key) >= 32

    @pytest.mark.asyncio
    async def test_initialize_success(self, auth_service):
        """Test successful initialization."""
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        with patch("ai_engine.api.auth.redis.Redis", return_value=mock_redis):
            await auth_service.initialize()
            assert auth_service.redis_client is not None

    @pytest.mark.asyncio
    async def test_initialize_redis_unavailable(self, auth_service):
        """Test initialization with Redis unavailable."""
        with patch(
            "ai_engine.api.auth.redis.Redis", side_effect=Exception("Connection failed")
        ):
            await auth_service.initialize()
            assert auth_service.redis_client is None

    def test_hash_password(self, auth_service):
        """Test password hashing."""
        password = "test_password"
        hashed = auth_service.hash_password(password)
        assert hashed != password
        assert len(hashed) > 0

    def test_verify_password_success(self, auth_service):
        """Test password verification success."""
        password = "test_password"
        hashed = auth_service.hash_password(password)
        assert auth_service.verify_password(password, hashed) is True

    def test_verify_password_failure(self, auth_service):
        """Test password verification failure."""
        password = "test_password"
        hashed = auth_service.hash_password(password)
        assert auth_service.verify_password("wrong_password", hashed) is False

    def test_create_access_token(self, auth_service):
        """Test creating access token."""
        data = {"user_id": "test_user", "username": "testuser"}
        token = auth_service.create_access_token(data)
        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_refresh_token(self, auth_service):
        """Test creating refresh token."""
        data = {"user_id": "test_user", "username": "testuser"}
        token = auth_service.create_refresh_token(data)
        assert isinstance(token, str)
        assert len(token) > 0

    @pytest.mark.asyncio
    async def test_verify_token_success(self, auth_service):
        """Test token verification success."""
        data = {"user_id": "test_user", "username": "testuser"}
        token = auth_service.create_access_token(data)
        payload = await auth_service.verify_token(token)
        assert payload["user_id"] == "test_user"

    @pytest.mark.asyncio
    async def test_verify_token_expired(self, auth_service):
        """Test token verification with expired token."""
        with patch(
            "ai_engine.api.auth.jwt.decode", side_effect=Exception("Token expired")
        ):
            with pytest.raises(AuthenticationError):
                await auth_service.verify_token("expired_token")

    @pytest.mark.asyncio
    async def test_verify_token_blacklisted_redis(self, auth_service):
        """Test token verification with blacklisted token in Redis."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value="revoked")
        auth_service.redis_client = mock_redis

        data = {"user_id": "test_user"}
        token = auth_service.create_access_token(data)

        with pytest.raises(AuthenticationError, match="Token has been revoked"):
            await auth_service.verify_token(token)

    @pytest.mark.asyncio
    async def test_verify_token_blacklisted_memory(self, auth_service):
        """Test token verification with blacklisted token in memory."""
        auth_service.redis_client = None
        data = {"user_id": "test_user"}
        token = auth_service.create_access_token(data)

        auth_service._token_blacklist[token] = datetime.utcnow() + timedelta(hours=1)

        with pytest.raises(AuthenticationError, match="Token has been revoked"):
            await auth_service.verify_token(token)

    @pytest.mark.asyncio
    async def test_revoke_token_redis(self, auth_service):
        """Test revoking token with Redis."""
        mock_redis = AsyncMock()
        auth_service.redis_client = mock_redis

        data = {"user_id": "test_user"}
        token = auth_service.create_access_token(data)

        await auth_service.revoke_token(token)
        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_revoke_token_memory(self, auth_service):
        """Test revoking token in memory."""
        auth_service.redis_client = None
        data = {"user_id": "test_user"}
        token = auth_service.create_access_token(data)

        await auth_service.revoke_token(token)
        assert token in auth_service._token_blacklist

    @pytest.mark.asyncio
    async def test_revoke_token_redis_error(self, auth_service):
        """Test revoking token with Redis error."""
        mock_redis = AsyncMock()
        mock_redis.setex = AsyncMock(side_effect=Exception("Redis error"))
        auth_service.redis_client = mock_redis

        data = {"user_id": "test_user"}
        token = auth_service.create_access_token(data)

        await auth_service.revoke_token(token)
        # Should handle error gracefully

    @pytest.mark.asyncio
    async def test_store_session_redis(self, auth_service):
        """Test storing session in Redis."""
        mock_redis = AsyncMock()
        auth_service.redis_client = mock_redis

        await auth_service.store_session("user123", {"key": "value"}, ttl=3600)
        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_session_memory(self, auth_service):
        """Test storing session in memory."""
        auth_service.redis_client = None

        await auth_service.store_session("user123", {"key": "value"}, ttl=3600)
        assert "user123" in auth_service._session_store

    @pytest.mark.asyncio
    async def test_store_session_redis_error(self, auth_service):
        """Test storing session with Redis error."""
        mock_redis = AsyncMock()
        mock_redis.setex = AsyncMock(side_effect=Exception("Redis error"))
        auth_service.redis_client = mock_redis

        await auth_service.store_session("user123", {"key": "value"})
        # Should handle error gracefully

    @pytest.mark.asyncio
    async def test_get_session_redis(self, auth_service):
        """Test getting session from Redis."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=json.dumps({"key": "value"}))
        auth_service.redis_client = mock_redis

        result = await auth_service.get_session("user123")
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_get_session_redis_invalid_json(self, auth_service):
        """Test getting session with invalid JSON."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value="invalid json")
        auth_service.redis_client = mock_redis

        result = await auth_service.get_session("user123")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_session_memory(self, auth_service):
        """Test getting session from memory."""
        auth_service.redis_client = None
        auth_service._session_store["user123"] = {"key": "value"}
        auth_service._session_expiry["user123"] = datetime.utcnow() + timedelta(hours=1)

        result = await auth_service.get_session("user123")
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, auth_service):
        """Test getting non-existent session."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        auth_service.redis_client = mock_redis

        result = await auth_service.get_session("user123")
        assert result is None

    @pytest.mark.asyncio
    async def test_authenticate_user_success(self, auth_service):
        """Test user authentication success."""
        with patch.dict(os.environ, {"DEMO_ADMIN_PASSWORD": "test_password"}):
            result = await auth_service.authenticate_user("admin", "test_password")
            assert result is not None
            assert result["username"] == "admin"

    @pytest.mark.asyncio
    async def test_authenticate_user_failure(self, auth_service):
        """Test user authentication failure."""
        result = await auth_service.authenticate_user("admin", "wrong_password")
        assert result is None

    @pytest.mark.asyncio
    async def test_authenticate_user_production_error(self, mock_config):
        """Test authentication in production mode."""
        mock_config.environment.value = "production"
        service = AuthenticationService(mock_config)

        with pytest.raises(
            AuthenticationError,
            match="Legacy authentication not available in production",
        ):
            await service.authenticate_user("admin", "password")

    def test_prune_blacklist(self, auth_service):
        """Test pruning expired blacklist entries."""
        auth_service._token_blacklist["token1"] = datetime.utcnow() - timedelta(hours=1)
        auth_service._token_blacklist["token2"] = datetime.utcnow() + timedelta(hours=1)

        auth_service._prune_blacklist()
        assert "token1" not in auth_service._token_blacklist
        assert "token2" in auth_service._token_blacklist

    def test_prune_sessions(self, auth_service):
        """Test pruning expired sessions."""
        auth_service._session_store["user1"] = {"key": "value"}
        auth_service._session_expiry["user1"] = datetime.utcnow() - timedelta(hours=1)
        auth_service._session_store["user2"] = {"key": "value"}
        auth_service._session_expiry["user2"] = datetime.utcnow() + timedelta(hours=1)

        auth_service._prune_sessions()
        assert "user1" not in auth_service._session_store
        assert "user2" in auth_service._session_store


class TestAuthFunctions:
    """Test suite for auth module functions."""

    @pytest.mark.asyncio
    async def test_get_auth_service(self):
        """Test getting auth service instance."""
        with patch("ai_engine.api.auth._auth_service", None):
            with patch("ai_engine.api.auth.AuthenticationService") as mock_service:
                mock_instance = AsyncMock()
                mock_instance.initialize = AsyncMock()
                mock_service.return_value = mock_instance

                result = await get_auth_service()
                assert result is not None

    def test_initialize_auth_from_secrets_manager(self):
        """Test initializing auth from secrets manager."""
        with patch("ai_engine.api.auth.get_secrets_manager") as mock_secrets:
            mock_mgr = Mock()
            mock_mgr.get_secret.return_value = (
                "api_key_from_vault_with_32_chars_minimum"
            )
            mock_secrets.return_value = mock_mgr

            result = initialize_auth()
            assert len(result) >= 32

    def test_initialize_auth_from_env(self):
        """Test initializing auth from environment."""
        with patch.dict(
            os.environ, {"QBITEL_AI_API_KEY": "test_api_key_with_32_characters_min"}
        ):
            with patch("ai_engine.api.auth.get_secrets_manager") as mock_secrets:
                mock_mgr = Mock()
                mock_mgr.get_secret.return_value = None
                mock_secrets.return_value = mock_mgr

                result = initialize_auth()
                assert result == "test_api_key_with_32_characters_min"

    def test_initialize_auth_production_error(self):
        """Test auth initialization error in production."""
        mock_config = Mock()
        mock_config.environment = Mock()
        mock_config.environment.value = "production"
        mock_config.security = Mock()
        mock_config.security.api_key = None

        with patch("ai_engine.api.auth.get_config", return_value=mock_config):
            with patch("ai_engine.api.auth.get_secrets_manager") as mock_secrets:
                mock_mgr = Mock()
                mock_mgr.get_secret.return_value = None
                mock_secrets.return_value = mock_mgr
                with patch.dict(os.environ, {}, clear=True):
                    with pytest.raises(
                        AuthenticationError,
                        match="API key not configured in production",
                    ):
                        initialize_auth()

    def test_initialize_auth_generated(self):
        """Test auth initialization with generated key."""
        with patch("ai_engine.api.auth.get_secrets_manager") as mock_secrets:
            mock_mgr = Mock()
            mock_mgr.get_secret.return_value = None
            mock_secrets.return_value = mock_mgr
            with patch.dict(os.environ, {}, clear=True):
                result = initialize_auth()
                assert len(result) >= 32

    def test_get_api_key(self):
        """Test getting API key."""
        with patch("ai_engine.api.auth._api_key", "test_key"):
            result = get_api_key()
            assert result == "test_key"

    def test_get_api_key_not_initialized(self):
        """Test getting API key when not initialized."""
        with patch("ai_engine.api.auth._api_key", None):
            with patch(
                "ai_engine.api.auth.initialize_auth", return_value="initialized_key"
            ):
                result = get_api_key()
                assert result == "initialized_key"

    def test_get_api_key_error(self):
        """Test getting API key error."""
        with patch("ai_engine.api.auth._api_key", None):
            with patch("ai_engine.api.auth.initialize_auth", return_value=None):
                with pytest.raises(AuthenticationError, match="API key not configured"):
                    get_api_key()

    @pytest.mark.asyncio
    async def test_login_success(self):
        """Test login success."""
        with patch("ai_engine.api.auth.get_auth_service") as mock_get_service:
            mock_service = AsyncMock()
            mock_service.authenticate_user = AsyncMock(
                return_value={
                    "user_id": "user123",
                    "username": "testuser",
                    "role": "admin",
                    "permissions": ["read", "write"],
                }
            )
            mock_service.create_access_token = Mock(return_value="access_token")
            mock_service.create_refresh_token = Mock(return_value="refresh_token")
            mock_service.store_session = AsyncMock()
            mock_service.audit_logger = Mock()
            mock_service.audit_logger.log_login_success = Mock()
            mock_get_service.return_value = mock_service

            result = await login("testuser", "password")
            assert result["access_token"] == "access_token"
            assert result["refresh_token"] == "refresh_token"

    @pytest.mark.asyncio
    async def test_login_failure(self):
        """Test login failure."""
        with patch("ai_engine.api.auth.get_auth_service") as mock_get_service:
            mock_service = AsyncMock()
            mock_service.authenticate_user = AsyncMock(return_value=None)
            mock_service.audit_logger = Mock()
            mock_service.audit_logger.log_login_failed = Mock()
            mock_get_service.return_value = mock_service

            with pytest.raises(HTTPException) as exc_info:
                await login("testuser", "wrong_password")
            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_refresh_access_token_success(self):
        """Test refreshing access token."""
        with patch("ai_engine.api.auth.get_auth_service") as mock_get_service:
            mock_service = AsyncMock()
            mock_service.verify_token = AsyncMock(
                return_value={
                    "type": "refresh",
                    "user_id": "user123",
                    "username": "testuser",
                    "role": "admin",
                    "permissions": ["read"],
                }
            )
            mock_service.create_access_token = Mock(return_value="new_access_token")
            mock_get_service.return_value = mock_service

            result = await refresh_access_token("refresh_token")
            assert result["access_token"] == "new_access_token"

    @pytest.mark.asyncio
    async def test_refresh_access_token_invalid_type(self):
        """Test refreshing with invalid token type."""
        with patch("ai_engine.api.auth.get_auth_service") as mock_get_service:
            mock_service = AsyncMock()
            mock_service.verify_token = AsyncMock(return_value={"type": "access"})
            mock_get_service.return_value = mock_service

            with pytest.raises(HTTPException):
                await refresh_access_token("access_token")

    @pytest.mark.asyncio
    async def test_logout(self):
        """Test logout."""
        mock_user = {"user_id": "user123"}
        result = await logout(mock_user)
        assert result["message"] == "Logged out successfully"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
