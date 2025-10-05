"""
Tests for CRONOS AI Engine Authentication and Authorization
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
import jwt
import json
from typing import Dict, Any

from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

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
from ai_engine.core.config import Config


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = Mock(spec=Config)
    config.security = Mock()
    config.security.jwt_secret = "test_secret_key_at_least_32_characters_long"
    config.security.api_key = None
    config.environment = Mock()
    config.environment.value = "development"
    config.redis = Mock()
    config.redis.host = "localhost"
    config.redis.port = 6379
    config.redis.db = 0
    return config


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    redis_client = AsyncMock()
    redis_client.ping = AsyncMock()
    redis_client.get = AsyncMock(return_value=None)
    redis_client.setex = AsyncMock()
    return redis_client


@pytest.fixture
def auth_service(mock_config):
    """Create an authentication service instance."""
    return AuthenticationService(mock_config)


class TestAuthenticationError:
    """Tests for AuthenticationError exception."""

    def test_authentication_error(self):
        """Test AuthenticationError creation."""
        error = AuthenticationError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)


class TestAuthenticationService:
    """Tests for AuthenticationService."""

    def test_initialization(self, mock_config):
        """Test service initialization."""
        service = AuthenticationService(mock_config)
        
        assert service.config == mock_config
        assert service.algorithm == "HS256"
        assert service.access_token_expire_minutes == 30
        assert service.refresh_token_expire_days == 7
        assert len(service.secret_key) >= 32

    def test_load_secret_key_from_config(self, mock_config):
        """Test loading secret key from configuration."""
        mock_config.security.jwt_secret = "test_secret_key_at_least_32_characters_long"
        
        service = AuthenticationService(mock_config)
        
        assert service.secret_key == "test_secret_key_at_least_32_characters_long"

    def test_load_secret_key_from_secrets_manager(self, mock_config):
        """Test loading secret key from secrets manager."""
        with patch('ai_engine.api.auth.get_secrets_manager') as mock_secrets_mgr:
            mock_mgr = Mock()
            mock_mgr.get_secret.return_value = "secret_from_vault_at_least_32_chars"
            mock_secrets_mgr.return_value = mock_mgr
            
            service = AuthenticationService(mock_config)
            
            assert service.secret_key == "secret_from_vault_at_least_32_chars"

    def test_load_secret_key_production_error(self):
        """Test secret key loading fails in production without key."""
        config = Mock()
        config.security = Mock()
        config.security.jwt_secret = None
        config.environment = Mock()
        config.environment.value = "production"
        
        with patch('ai_engine.api.auth.get_secrets_manager') as mock_secrets_mgr:
            mock_mgr = Mock()
            mock_mgr.get_secret.return_value = None
            mock_secrets_mgr.return_value = mock_mgr
            
            with pytest.raises(AuthenticationError, match="JWT secret not configured"):
                AuthenticationService(config)

    def test_load_secret_key_generates_ephemeral(self, mock_config):
        """Test ephemeral secret key generation in development."""
        mock_config.security.jwt_secret = None
        mock_config.environment.value = "development"
        
        with patch('ai_engine.api.auth.get_secrets_manager') as mock_secrets_mgr:
            mock_mgr = Mock()
            mock_mgr.get_secret.return_value = None
            mock_secrets_mgr.return_value = mock_mgr
            
            service = AuthenticationService(mock_config)
            
            assert len(service.secret_key) >= 32

    @pytest.mark.asyncio
    async def test_initialize_with_redis(self, auth_service, mock_redis):
        """Test service initialization with Redis."""
        with patch('ai_engine.api.auth.redis.Redis', return_value=mock_redis):
            await auth_service.initialize()
            
            assert auth_service.redis_client is not None
            mock_redis.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_without_redis(self, auth_service):
        """Test service initialization without Redis."""
        with patch('ai_engine.api.auth.redis.Redis', side_effect=Exception("Redis unavailable")):
            await auth_service.initialize()
            
            assert auth_service.redis_client is None

    def test_hash_password(self, auth_service):
        """Test password hashing."""
        password = "test_password_123"
        hashed = auth_service.hash_password(password)
        
        assert hashed != password
        assert len(hashed) > 0
        assert hashed.startswith("$2b$")

    def test_verify_password_success(self, auth_service):
        """Test successful password verification."""
        password = "test_password_123"
        hashed = auth_service.hash_password(password)
        
        assert auth_service.verify_password(password, hashed) is True

    def test_verify_password_failure(self, auth_service):
        """Test failed password verification."""
        password = "test_password_123"
        wrong_password = "wrong_password"
        hashed = auth_service.hash_password(password)
        
        assert auth_service.verify_password(wrong_password, hashed) is False

    def test_create_access_token(self, auth_service):
        """Test access token creation."""
        data = {"user_id": "user123", "username": "testuser"}
        token = auth_service.create_access_token(data)
        
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Decode and verify
        payload = jwt.decode(token, auth_service.secret_key, algorithms=[auth_service.algorithm])
        assert payload["user_id"] == "user123"
        assert payload["type"] == "access"

    def test_create_refresh_token(self, auth_service):
        """Test refresh token creation."""
        data = {"user_id": "user123", "username": "testuser"}
        token = auth_service.create_refresh_token(data)
        
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Decode and verify
        payload = jwt.decode(token, auth_service.secret_key, algorithms=[auth_service.algorithm])
        assert payload["user_id"] == "user123"
        assert payload["type"] == "refresh"

    @pytest.mark.asyncio
    async def test_verify_token_success(self, auth_service):
        """Test successful token verification."""
        data = {"user_id": "user123", "username": "testuser"}
        token = auth_service.create_access_token(data)
        
        payload = await auth_service.verify_token(token)
        
        assert payload["user_id"] == "user123"
        assert payload["username"] == "testuser"

    @pytest.mark.asyncio
    async def test_verify_token_expired(self, auth_service):
        """Test expired token verification."""
        data = {"user_id": "user123"}
        # Create token that expires immediately
        expire = datetime.utcnow() - timedelta(seconds=1)
        data.update({"exp": expire})
        token = jwt.encode(data, auth_service.secret_key, algorithm=auth_service.algorithm)
        
        with pytest.raises(AuthenticationError, match="Token has expired"):
            await auth_service.verify_token(token)

    @pytest.mark.asyncio
    async def test_verify_token_invalid(self, auth_service):
        """Test invalid token verification."""
        with pytest.raises(AuthenticationError, match="Invalid token"):
            await auth_service.verify_token("invalid_token")

    @pytest.mark.asyncio
    async def test_verify_token_blacklisted_redis(self, auth_service, mock_redis):
        """Test blacklisted token verification with Redis."""
        auth_service.redis_client = mock_redis
        mock_redis.get.return_value = "revoked"
        
        data = {"user_id": "user123"}
        token = auth_service.create_access_token(data)
        
        with pytest.raises(AuthenticationError, match="Token has been revoked"):
            await auth_service.verify_token(token)

    @pytest.mark.asyncio
    async def test_verify_token_blacklisted_memory(self, auth_service):
        """Test blacklisted token verification with in-memory store."""
        data = {"user_id": "user123"}
        token = auth_service.create_access_token(data)
        
        # Blacklist the token
        auth_service._token_blacklist[token] = datetime.utcnow() + timedelta(hours=1)
        
        with pytest.raises(AuthenticationError, match="Token has been revoked"):
            await auth_service.verify_token(token)

    @pytest.mark.asyncio
    async def test_revoke_token_redis(self, auth_service, mock_redis):
        """Test token revocation with Redis."""
        auth_service.redis_client = mock_redis
        
        data = {"user_id": "user123"}
        token = auth_service.create_access_token(data)
        
        await auth_service.revoke_token(token)
        
        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_revoke_token_memory(self, auth_service):
        """Test token revocation with in-memory store."""
        data = {"user_id": "user123"}
        token = auth_service.create_access_token(data)
        
        await auth_service.revoke_token(token)
        
        assert token in auth_service._token_blacklist

    @pytest.mark.asyncio
    async def test_store_session_redis(self, auth_service, mock_redis):
        """Test session storage with Redis."""
        auth_service.redis_client = mock_redis
        
        session_data = {"login_time": "2024-01-01T00:00:00"}
        await auth_service.store_session("user123", session_data, ttl=3600)
        
        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_session_memory(self, auth_service):
        """Test session storage with in-memory store."""
        session_data = {"login_time": "2024-01-01T00:00:00"}
        await auth_service.store_session("user123", session_data, ttl=3600)
        
        assert "user123" in auth_service._session_store
        assert auth_service._session_store["user123"] == session_data

    @pytest.mark.asyncio
    async def test_get_session_redis(self, auth_service, mock_redis):
        """Test session retrieval with Redis."""
        auth_service.redis_client = mock_redis
        session_data = {"login_time": "2024-01-01T00:00:00"}
        mock_redis.get.return_value = json.dumps(session_data)
        
        result = await auth_service.get_session("user123")
        
        assert result == session_data

    @pytest.mark.asyncio
    async def test_get_session_memory(self, auth_service):
        """Test session retrieval with in-memory store."""
        session_data = {"login_time": "2024-01-01T00:00:00"}
        await auth_service.store_session("user123", session_data)
        
        result = await auth_service.get_session("user123")
        
        assert result == session_data

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, auth_service):
        """Test session retrieval when not found."""
        result = await auth_service.get_session("nonexistent")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_authenticate_user_success(self, auth_service):
        """Test successful user authentication."""
        # This uses demo users in development mode
        user = await auth_service.authenticate_user("admin", "DemoOnly_NotForProduction_123!")
        
        assert user is not None
        assert user["username"] == "admin"
        assert user["role"] == "administrator"

    @pytest.mark.asyncio
    async def test_authenticate_user_failure(self, auth_service):
        """Test failed user authentication."""
        user = await auth_service.authenticate_user("admin", "wrong_password")
        
        assert user is None

    @pytest.mark.asyncio
    async def test_authenticate_user_production_error(self):
        """Test authentication fails in production mode."""
        config = Mock()
        config.security = Mock()
        config.security.jwt_secret = "test_secret_key_at_least_32_characters_long"
        config.environment = Mock()
        config.environment.value = "production"
        config.redis = Mock()
        
        service = AuthenticationService(config)
        
        with pytest.raises(AuthenticationError, match="Legacy authentication not available"):
            await service.authenticate_user("admin", "password")

    def test_prune_blacklist(self, auth_service):
        """Test blacklist pruning."""
        # Add expired token
        auth_service._token_blacklist["expired_token"] = datetime.utcnow() - timedelta(hours=1)
        # Add valid token
        auth_service._token_blacklist["valid_token"] = datetime.utcnow() + timedelta(hours=1)
        
        auth_service._prune_blacklist()
        
        assert "expired_token" not in auth_service._token_blacklist
        assert "valid_token" in auth_service._token_blacklist

    def test_prune_sessions(self, auth_service):
        """Test session pruning."""
        # Add expired session
        auth_service._session_store["expired_user"] = {}
        auth_service._session_expiry["expired_user"] = datetime.utcnow() - timedelta(hours=1)
        # Add valid session
        auth_service._session_store["valid_user"] = {}
        auth_service._session_expiry["valid_user"] = datetime.utcnow() + timedelta(hours=1)
        
        auth_service._prune_sessions()
        
        assert "expired_user" not in auth_service._session_store
        assert "valid_user" in auth_service._session_store


class TestGetAuthService:
    """Tests for get_auth_service function."""

    @pytest.mark.asyncio
    async def test_get_auth_service_creates_instance(self):
        """Test get_auth_service creates instance."""
        # Reset global instance
        import ai_engine.api.auth as auth_module
        auth_module._auth_service = None
        
        with patch('ai_engine.api.auth.AuthenticationService') as mock_service_class:
            mock_service = AsyncMock()
            mock_service.initialize = AsyncMock()
            mock_service_class.return_value = mock_service
            
            result = await get_auth_service()
            
            assert result == mock_service
            mock_service.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_auth_service_reuses_instance(self):
        """Test get_auth_service reuses existing instance."""
        import ai_engine.api.auth as auth_module
        
        mock_service = AsyncMock()
        auth_module._auth_service = mock_service
        
        result = await get_auth_service()
        
        assert result == mock_service


class TestVerifyToken:
    """Tests for verify_token dependency."""

    @pytest.mark.asyncio
    async def test_verify_token_success(self):
        """Test successful token verification."""
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="valid_token"
        )
        
        with patch('ai_engine.api.auth.get_auth_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.verify_token = AsyncMock(return_value={
                "user_id": "user123",
                "username": "testuser",
            })
            mock_get_service.return_value = mock_service
            
            result = await verify_token(credentials)
            
            assert result["user_id"] == "user123"

    @pytest.mark.asyncio
    async def test_verify_token_api_key(self):
        """Test verification with API key."""
        with patch('ai_engine.api.auth.get_api_key', return_value="test_api_key"):
            credentials = HTTPAuthorizationCredentials(
                scheme="Bearer",
                credentials="test_api_key"
            )
            
            result = await verify_token(credentials)
            
            assert result["user_id"] == "api-key-user"
            assert result["role"] == "system"

    @pytest.mark.asyncio
    async def test_verify_token_no_credentials(self):
        """Test verification without credentials."""
        with pytest.raises(HTTPException) as exc_info:
            await verify_token(None)
        
        assert exc_info.value.status_code == 401
        assert "Not authenticated" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_verify_token_authentication_error(self):
        """Test verification with authentication error."""
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="invalid_token"
        )
        
        with patch('ai_engine.api.auth.get_auth_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.verify_token = AsyncMock(side_effect=AuthenticationError("Invalid token"))
            mock_get_service.return_value = mock_service
            
            with pytest.raises(HTTPException) as exc_info:
                await verify_token(credentials)
            
            assert exc_info.value.status_code == 401


class TestGetCurrentUser:
    """Tests for get_current_user dependency."""

    @pytest.mark.asyncio
    async def test_get_current_user_success(self):
        """Test successful current user retrieval."""
        token_payload = {
            "user_id": "user123",
            "username": "testuser",
            "role": "analyst",
            "permissions": ["read", "write"],
        }
        
        with patch('ai_engine.api.auth.get_auth_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_session = AsyncMock(return_value={"session_data": "test"})
            mock_get_service.return_value = mock_service
            
            result = await get_current_user(token_payload)
            
            assert result["user_id"] == "user123"
            assert result["username"] == "testuser"
            assert result["session_data"] == {"session_data": "test"}

    @pytest.mark.asyncio
    async def test_get_current_user_no_user_id(self):
        """Test current user retrieval without user_id."""
        token_payload = {"username": "testuser"}
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(token_payload)
        
        assert exc_info.value.status_code == 401
        assert "Invalid token payload" in str(exc_info.value.detail)


class TestRequirePermission:
    """Tests for require_permission dependency."""

    @pytest.mark.asyncio
    async def test_require_permission_success(self):
        """Test successful permission check."""
        current_user = {
            "user_id": "user123",
            "permissions": ["read", "write", "admin"],
        }
        
        check_func = require_permission("read")
        result = await check_func(current_user)
        
        assert result == current_user

    @pytest.mark.asyncio
    async def test_require_permission_failure(self):
        """Test failed permission check."""
        current_user = {
            "user_id": "user123",
            "permissions": ["read"],
        }
        
        check_func = require_permission("admin")
        
        with pytest.raises(HTTPException) as exc_info:
            await check_func(current_user)
        
        assert exc_info.value.status_code == 403
        assert "Permission 'admin' required" in str(exc_info.value.detail)


class TestRequireRole:
    """Tests for require_role dependency."""

    @pytest.mark.asyncio
    async def test_require_role_success(self):
        """Test successful role check."""
        current_user = {
            "user_id": "user123",
            "role": "administrator",
        }
        
        check_func = require_role("administrator")
        result = await check_func(current_user)
        
        assert result == current_user

    @pytest.mark.asyncio
    async def test_require_role_failure(self):
        """Test failed role check."""
        current_user = {
            "user_id": "user123",
            "role": "analyst",
        }
        
        check_func = require_role("administrator")
        
        with pytest.raises(HTTPException) as exc_info:
            await check_func(current_user)
        
        assert exc_info.value.status_code == 403
        assert "Role 'administrator' required" in str(exc_info.value.detail)


class TestLogin:
    """Tests for login function."""

    @pytest.mark.asyncio
    async def test_login_success(self):
        """Test successful login."""
        with patch('ai_engine.api.auth.get_auth_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.authenticate_user = AsyncMock(return_value={
                "user_id": "user123",
                "username": "testuser",
                "role": "analyst",
                "permissions": ["read", "write"],
            })
            mock_service.create_access_token = Mock(return_value="access_token")
            mock_service.create_refresh_token = Mock(return_value="refresh_token")
            mock_service.store_session = AsyncMock()
            mock_service.audit_logger = Mock()
            mock_service.audit_logger.log_login_success = Mock()
            mock_get_service.return_value = mock_service
            
            result = await login("testuser", "password")
            
            assert result["access_token"] == "access_token"
            assert result["refresh_token"] == "refresh_token"
            assert result["token_type"] == "bearer"
            assert result["user"]["username"] == "testuser"

    @pytest.mark.asyncio
    async def test_login_failure(self):
        """Test failed login."""
        with patch('ai_engine.api.auth.get_auth_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.authenticate_user = AsyncMock(return_value=None)
            mock_service.audit_logger = Mock()
            mock_service.audit_logger.log_login_failed = Mock()
            mock_get_service.return_value = mock_service
            
            with pytest.raises(HTTPException) as exc_info:
                await login("testuser", "wrong_password")
            
            assert exc_info.value.status_code == 401
            assert "Invalid credentials" in str(exc_info.value.detail)


class TestRefreshAccessToken:
    """Tests for refresh_access_token function."""

    @pytest.mark.asyncio
    async def test_refresh_token_success(self):
        """Test successful token refresh."""
        with patch('ai_engine.api.auth.get_auth_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.verify_token = AsyncMock(return_value={
                "user_id": "user123",
                "username": "testuser",
                "role": "analyst",
                "permissions": ["read"],
                "type": "refresh",
            })
            mock_service.create_access_token = Mock(return_value="new_access_token")
            mock_get_service.return_value = mock_service
            
            result = await refresh_access_token("refresh_token")
            
            assert result["access_token"] == "new_access_token"
            assert result["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_refresh_token_invalid_type(self):
        """Test token refresh with invalid token type."""
        with patch('ai_engine.api.auth.get_auth_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.verify_token = AsyncMock(return_value={
                "user_id": "user123",
                "type": "access",  # Wrong type
            })
            mock_get_service.return_value = mock_service
            
            with pytest.raises(HTTPException) as exc_info:
                await refresh_access_token("access_token")
            
            assert exc_info.value.status_code == 401


class TestLogout:
    """Tests for logout function."""

    @pytest.mark.asyncio
    async def test_logout(self):
        """Test logout."""
        current_user = {"user_id": "user123"}
        
        result = await logout(current_user)
        
        assert result["message"] == "Logged out successfully"


class TestInitializeAuth:
    """Tests for initialize_auth function."""

    def test_initialize_auth_from_secrets_manager(self):
        """Test API key initialization from secrets manager."""
        with patch('ai_engine.api.auth.get_secrets_manager') as mock_secrets_mgr:
            mock_mgr = Mock()
            mock_mgr.get_secret.return_value = "api_key_from_vault_at_least_32_chars"
            mock_secrets_mgr.return_value = mock_mgr
            
            api_key = initialize_auth()
            
            assert api_key == "api_key_from_vault_at_least_32_chars"

    def test_initialize_auth_from_env(self):
        """Test API key initialization from environment."""
        with patch('ai_engine.api.auth.get_secrets_manager') as mock_secrets_mgr:
            mock_mgr = Mock()
            mock_mgr.get_secret.return_value = None
            mock_secrets_mgr.return_value = mock_mgr
            
            with patch.dict('os.environ', {'CRONOS_AI_API_KEY': 'env_api_key_at_least_32_characters'}):
                api_key = initialize_auth()
                
                assert api_key == "env_api_key_at_least_32_characters"

    def test_initialize_auth_generates_key(self):
        """Test API key generation in development."""
        with patch('ai_engine.api.auth.get_secrets_manager') as mock_secrets_mgr:
            mock_mgr = Mock()
            mock_mgr.get_secret.return_value = None
            mock_secrets_mgr.return_value = mock_mgr
            
            with patch.dict('os.environ', {}, clear=True):
                api_key = initialize_auth()
                
                assert api_key is not None
                assert len(api_key) >= 32

    def test_initialize_auth_production_error(self):
        """Test API key initialization fails in production."""
        config = Mock()
        config.security = Mock()
        config.security.api_key = None
        config.environment = Mock()
        config.environment.value = "production"
        
        with patch('ai_engine.api.auth.get_secrets_manager') as mock_secrets_mgr:
            mock_mgr = Mock()
            mock_mgr.get_secret.return_value = None
            mock_secrets_mgr.return_value = mock_mgr
            
            with patch.dict('os.environ', {}, clear=True):
                with pytest.raises(AuthenticationError, match="API key not configured"):
                    initialize_auth(config)


class TestGetApiKey:
    """Tests for get_api_key function."""

    def test_get_api_key_success(self):
        """Test successful API key retrieval."""
        import ai_engine.api.auth as auth_module
        auth_module._api_key = "test_api_key"
        
        api_key = get_api_key()
        
        assert api_key == "test_api_key"

    def test_get_api_key_initializes(self):
        """Test API key retrieval initializes if not set."""
        import ai_engine.api.auth as auth_module
        auth_module._api_key = None
        
        with patch('ai_engine.api.auth.initialize_auth', return_value="initialized_key"):
            api_key = get_api_key()
            
            assert api_key == "initialized_key"

    def test_get_api_key_not_configured(self):
        """Test API key retrieval when not configured."""
        import ai_engine.api.auth as auth_module
        auth_module._api_key = None
        
        with patch('ai_engine.api.auth.initialize_auth', return_value=None):
            with pytest.raises(AuthenticationError, match="API key not configured"):
                get_api_key()