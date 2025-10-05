"""
Comprehensive tests for ai_engine/api/auth.py

Tests cover:
- JWT token creation (access and refresh)
- JWT token verification and expiration
- Token revocation and blacklisting
- Session management (Redis and in-memory)
- Password hashing and verification
- User authentication
- API key initialization and validation
- Secret key loading from multiple sources
- Permission and role-based access control
- Login and logout flows
- Token refresh
- Error handling
"""

import pytest
import asyncio
import os
import jwt
import secrets
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, MagicMock, patch, call
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from ai_engine.api.auth import (
    AuthenticationService,
    AuthenticationError,
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
from ai_engine.core.config import get_config


# Fixtures

@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = Mock()
    config.environment = Mock()
    config.environment.value = "development"
    config.security = Mock()
    config.security.jwt_secret = "test_secret_key_32chars_long!!"
    config.security.api_key = None
    config.redis = Mock()
    config.redis.host = "localhost"
    config.redis.port = 6379
    config.redis.db = 0
    return config


@pytest.fixture
def auth_service(mock_config):
    """Create AuthenticationService instance."""
    with patch("ai_engine.api.auth.get_config", return_value=mock_config):
        with patch("ai_engine.api.auth.get_secrets_manager"):
            with patch("ai_engine.api.auth.get_audit_logger"):
                service = AuthenticationService(mock_config)
                return service


@pytest.fixture
async def initialized_auth_service(auth_service):
    """Create initialized AuthenticationService."""
    with patch("ai_engine.api.auth.redis.Redis") as mock_redis_class:
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis_class.return_value = mock_redis

        await auth_service.initialize()
        return auth_service


# Initialization Tests

def test_service_initialization(mock_config):
    """Test authentication service initialization."""
    with patch("ai_engine.api.auth.get_config", return_value=mock_config):
        with patch("ai_engine.api.auth.get_secrets_manager"):
            with patch("ai_engine.api.auth.get_audit_logger"):
                service = AuthenticationService(mock_config)

                assert service.config == mock_config
                assert service.secret_key is not None
                assert len(service.secret_key) >= 32
                assert service.algorithm == "HS256"
                assert service.access_token_expire_minutes == 30
                assert service.refresh_token_expire_days == 7


def test_load_secret_key_from_secrets_manager():
    """Test loading JWT secret from secrets manager."""
    mock_config = Mock()
    mock_config.environment = Mock()
    mock_config.environment.value = "development"
    mock_config.security = Mock()

    mock_secrets_mgr = Mock()
    mock_secrets_mgr.get_secret.return_value = "secret_from_manager_that_is_at_least_32_chars_long"

    with patch("ai_engine.api.auth.get_config", return_value=mock_config):
        with patch("ai_engine.api.auth.get_secrets_manager", return_value=mock_secrets_mgr):
            with patch("ai_engine.api.auth.get_audit_logger"):
                service = AuthenticationService(mock_config)

                assert service.secret_key == "secret_from_manager_that_is_at_least_32_chars_long"


def test_load_secret_key_from_config():
    """Test loading JWT secret from config."""
    mock_config = Mock()
    mock_config.environment = Mock()
    mock_config.environment.value = "development"
    mock_config.security = Mock()
    mock_config.security.jwt_secret = "secret_from_config_that_is_at_least_32_characters_long"

    with patch("ai_engine.api.auth.get_config", return_value=mock_config):
        with patch("ai_engine.api.auth.get_secrets_manager") as mock_sm:
            mock_sm.return_value.get_secret.return_value = None
            with patch("ai_engine.api.auth.get_audit_logger"):
                service = AuthenticationService(mock_config)

                assert service.secret_key == "secret_from_config_that_is_at_least_32_characters_long"


def test_load_secret_key_production_missing():
    """Test missing JWT secret in production raises error."""
    mock_config = Mock()
    mock_config.environment = Mock()
    mock_config.environment.value = "production"
    mock_config.security = Mock()
    mock_config.security.jwt_secret = None

    with patch("ai_engine.api.auth.get_config", return_value=mock_config):
        with patch("ai_engine.api.auth.get_secrets_manager") as mock_sm:
            mock_sm.return_value.get_secret.return_value = None
            with patch("ai_engine.api.auth.get_audit_logger"):
                with pytest.raises(AuthenticationError, match="JWT secret not configured"):
                    AuthenticationService(mock_config)


def test_load_secret_key_development_generated():
    """Test ephemeral secret generation in development."""
    mock_config = Mock()
    mock_config.environment = Mock()
    mock_config.environment.value = "development"
    mock_config.security = Mock()
    mock_config.security.jwt_secret = None

    with patch("ai_engine.api.auth.get_config", return_value=mock_config):
        with patch("ai_engine.api.auth.get_secrets_manager") as mock_sm:
            mock_sm.return_value.get_secret.return_value = None
            with patch("ai_engine.api.auth.get_audit_logger"):
                service = AuthenticationService(mock_config)

                # Should have generated a secret
                assert service.secret_key is not None
                assert len(service.secret_key) >= 32


@pytest.mark.asyncio
async def test_initialize_with_redis(auth_service):
    """Test initialization with Redis."""
    with patch("ai_engine.api.auth.redis.Redis") as mock_redis_class:
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis_class.return_value = mock_redis

        await auth_service.initialize()

        assert auth_service.redis_client is not None
        mock_redis.ping.assert_called_once()


@pytest.mark.asyncio
async def test_initialize_redis_failure_fallback(auth_service):
    """Test fallback to in-memory when Redis fails."""
    with patch("ai_engine.api.auth.redis.Redis") as mock_redis_class:
        mock_redis = AsyncMock()
        mock_redis.ping.side_effect = Exception("Redis connection failed")
        mock_redis_class.return_value = mock_redis

        await auth_service.initialize()

        # Should fallback to in-memory
        assert auth_service.redis_client is None


# Password Management Tests

def test_hash_password(auth_service):
    """Test password hashing."""
    password = "test_password_123"

    # Mock passlib to avoid bcrypt backend issues in tests
    with patch("ai_engine.api.auth.pwd_context.hash") as mock_hash:
        mock_hash.return_value = "$2b$12$mockedhashvalue"
        hashed = auth_service.hash_password(password)

        assert hashed == "$2b$12$mockedhashvalue"
        mock_hash.assert_called_once_with(password)


def test_verify_password_success(auth_service):
    """Test password verification success."""
    password = "test_password_123"
    hashed = "$2b$12$mockedhash"

    with patch("ai_engine.api.auth.pwd_context.verify") as mock_verify:
        mock_verify.return_value = True
        result = auth_service.verify_password(password, hashed)

        assert result is True
        mock_verify.assert_called_once_with(password, hashed)


def test_verify_password_failure(auth_service):
    """Test password verification failure."""
    password = "test_password_123"
    hashed = "$2b$12$mockedhash"

    with patch("ai_engine.api.auth.pwd_context.verify") as mock_verify:
        mock_verify.return_value = False
        result = auth_service.verify_password("wrong_password", hashed)

        assert result is False


# JWT Token Tests

def test_create_access_token(auth_service):
    """Test access token creation."""
    data = {"user_id": "user123", "username": "testuser"}
    token = auth_service.create_access_token(data)

    assert isinstance(token, str)
    assert len(token) > 0

    # Decode and verify
    payload = jwt.decode(token, auth_service.secret_key, algorithms=[auth_service.algorithm])
    assert payload["user_id"] == "user123"
    assert payload["type"] == "access"
    assert "exp" in payload


def test_create_refresh_token(auth_service):
    """Test refresh token creation."""
    data = {"user_id": "user123"}
    token = auth_service.create_refresh_token(data)

    assert isinstance(token, str)

    payload = jwt.decode(token, auth_service.secret_key, algorithms=[auth_service.algorithm])
    assert payload["user_id"] == "user123"
    assert payload["type"] == "refresh"
    assert "exp" in payload


def test_token_expiration_times(auth_service):
    """Test token expiration times."""
    data = {"user_id": "user123"}

    access_token = auth_service.create_access_token(data)
    refresh_token = auth_service.create_refresh_token(data)

    access_payload = jwt.decode(access_token, auth_service.secret_key, algorithms=[auth_service.algorithm])
    refresh_payload = jwt.decode(refresh_token, auth_service.secret_key, algorithms=[auth_service.algorithm])

    access_exp = datetime.utcfromtimestamp(access_payload["exp"])
    refresh_exp = datetime.utcfromtimestamp(refresh_payload["exp"])

    # Refresh token should expire much later than access token
    assert (refresh_exp - access_exp).days >= 6


@pytest.mark.asyncio
async def test_verify_token_success(auth_service):
    """Test token verification success."""
    data = {"user_id": "user123", "username": "testuser"}
    token = auth_service.create_access_token(data)

    payload = await auth_service.verify_token(token)

    assert payload["user_id"] == "user123"
    assert payload["type"] == "access"


@pytest.mark.asyncio
async def test_verify_token_expired():
    """Test token verification with expired token."""
    mock_config = Mock()
    mock_config.environment = Mock()
    mock_config.environment.value = "development"
    mock_config.security = Mock()
    mock_config.security.jwt_secret = "test_secret_key_that_is_at_least_32_characters_long"

    with patch("ai_engine.api.auth.get_config", return_value=mock_config):
        with patch("ai_engine.api.auth.get_secrets_manager"):
            with patch("ai_engine.api.auth.get_audit_logger"):
                service = AuthenticationService(mock_config)

                # Create expired token
                data = {"user_id": "user123"}
                expire = datetime.utcnow() - timedelta(minutes=10)
                data.update({"exp": expire, "type": "access"})

                token = jwt.encode(data, service.secret_key, algorithm=service.algorithm)

                with pytest.raises(AuthenticationError, match="Token has expired"):
                    await service.verify_token(token)


@pytest.mark.asyncio
async def test_verify_token_invalid(auth_service):
    """Test token verification with invalid token."""
    with pytest.raises(AuthenticationError, match="Invalid token"):
        await auth_service.verify_token("invalid.token.here")


@pytest.mark.asyncio
async def test_verify_token_wrong_secret(auth_service):
    """Test token verification with wrong secret."""
    # Create token with different secret
    wrong_secret = "wrong_secret_key_that_is_different_32_chars"
    data = {"user_id": "user123", "type": "access"}
    token = jwt.encode(data, wrong_secret, algorithm="HS256")

    with pytest.raises(AuthenticationError, match="Invalid token"):
        await auth_service.verify_token(token)


# Token Revocation Tests

@pytest.mark.asyncio
async def test_revoke_token_with_redis(initialized_auth_service):
    """Test token revocation with Redis."""
    data = {"user_id": "user123"}
    token = initialized_auth_service.create_access_token(data)

    await initialized_auth_service.revoke_token(token)

    # Token should be blacklisted
    initialized_auth_service.redis_client.setex.assert_called()


@pytest.mark.asyncio
async def test_revoke_token_in_memory(auth_service):
    """Test token revocation with in-memory store."""
    data = {"user_id": "user123"}
    token = auth_service.create_access_token(data)

    await auth_service.revoke_token(token)

    # Token should be in blacklist
    assert token in auth_service._token_blacklist


@pytest.mark.asyncio
async def test_verify_revoked_token_in_memory(auth_service):
    """Test verification of revoked token (in-memory)."""
    data = {"user_id": "user123"}
    token = auth_service.create_access_token(data)

    # Revoke token
    await auth_service.revoke_token(token)

    # Try to verify
    with pytest.raises(AuthenticationError, match="Token has been revoked"):
        await auth_service.verify_token(token)


@pytest.mark.asyncio
async def test_verify_revoked_token_with_redis(initialized_auth_service):
    """Test verification of revoked token (Redis)."""
    data = {"user_id": "user123"}
    token = initialized_auth_service.create_access_token(data)

    # Mock Redis to return blacklisted status
    initialized_auth_service.redis_client.get = AsyncMock(return_value="revoked")

    with pytest.raises(AuthenticationError, match="Token has been revoked"):
        await initialized_auth_service.verify_token(token)


# Session Management Tests

@pytest.mark.asyncio
async def test_store_session_with_redis(initialized_auth_service):
    """Test session storage with Redis."""
    session_data = {"login_time": "2024-01-01", "ip": "192.168.1.1"}

    await initialized_auth_service.store_session("user123", session_data, ttl=3600)

    initialized_auth_service.redis_client.setex.assert_called()


@pytest.mark.asyncio
async def test_store_session_in_memory(auth_service):
    """Test session storage in-memory."""
    session_data = {"login_time": "2024-01-01"}

    await auth_service.store_session("user123", session_data, ttl=3600)

    assert "user123" in auth_service._session_store
    assert auth_service._session_store["user123"] == session_data


@pytest.mark.asyncio
async def test_get_session_with_redis(initialized_auth_service):
    """Test session retrieval with Redis."""
    session_data = {"login_time": "2024-01-01"}

    import json
    initialized_auth_service.redis_client.get = AsyncMock(
        return_value=json.dumps(session_data)
    )

    result = await initialized_auth_service.get_session("user123")

    assert result == session_data


@pytest.mark.asyncio
async def test_get_session_in_memory(auth_service):
    """Test session retrieval in-memory."""
    session_data = {"login_time": "2024-01-01"}

    await auth_service.store_session("user123", session_data)

    result = await auth_service.get_session("user123")

    assert result == session_data


@pytest.mark.asyncio
async def test_get_session_not_found(auth_service):
    """Test getting non-existent session."""
    result = await auth_service.get_session("nonexistent")

    assert result is None


@pytest.mark.asyncio
async def test_session_expiry_pruning(auth_service):
    """Test expired session pruning."""
    # Store session with short TTL
    await auth_service.store_session("user123", {"data": "test"}, ttl=1)

    # Set expiry to past
    auth_service._session_expiry["user123"] = datetime.utcnow() - timedelta(seconds=10)

    # Trigger pruning by getting session
    result = await auth_service.get_session("user123")

    # Should have been pruned
    assert result is None
    assert "user123" not in auth_service._session_store


# User Authentication Tests

@pytest.mark.asyncio
async def test_authenticate_user_success(auth_service):
    """Test successful user authentication (dev mode)."""
    # In development mode with demo users
    user = await auth_service.authenticate_user("admin", "DemoOnly_NotForProduction_123!")

    assert user is not None
    assert user["username"] == "admin"
    assert user["role"] == "administrator"
    assert "protocol_discovery" in user["permissions"]


@pytest.mark.asyncio
async def test_authenticate_user_invalid_credentials(auth_service):
    """Test authentication with invalid credentials."""
    user = await auth_service.authenticate_user("admin", "wrong_password")

    assert user is None


@pytest.mark.asyncio
async def test_authenticate_user_nonexistent_user(auth_service):
    """Test authentication with non-existent user."""
    user = await auth_service.authenticate_user("nonexistent", "password")

    assert user is None


@pytest.mark.asyncio
async def test_authenticate_user_production_mode():
    """Test authentication in production mode raises error."""
    mock_config = Mock()
    mock_config.environment = Mock()
    mock_config.environment.value = "production"
    mock_config.security = Mock()
    mock_config.security.jwt_secret = "production_secret_key_that_is_at_least_32_characters"

    with patch("ai_engine.api.auth.get_config", return_value=mock_config):
        with patch("ai_engine.api.auth.get_secrets_manager"):
            with patch("ai_engine.api.auth.get_audit_logger"):
                service = AuthenticationService(mock_config)

                with pytest.raises(AuthenticationError, match="Legacy authentication not available in production"):
                    await service.authenticate_user("admin", "password")


# Blacklist Pruning Tests

def test_prune_blacklist(auth_service):
    """Test blacklist pruning."""
    # Add expired tokens
    auth_service._token_blacklist["token1"] = datetime.utcnow() - timedelta(hours=1)
    auth_service._token_blacklist["token2"] = datetime.utcnow() + timedelta(hours=1)

    auth_service._prune_blacklist()

    # Expired token should be removed
    assert "token1" not in auth_service._token_blacklist
    # Valid token should remain
    assert "token2" in auth_service._token_blacklist


def test_prune_sessions(auth_service):
    """Test session pruning."""
    # Add expired sessions
    auth_service._session_store["user1"] = {"data": "test"}
    auth_service._session_expiry["user1"] = datetime.utcnow() - timedelta(hours=1)

    auth_service._session_store["user2"] = {"data": "test"}
    auth_service._session_expiry["user2"] = datetime.utcnow() + timedelta(hours=1)

    auth_service._prune_sessions()

    # Expired session should be removed
    assert "user1" not in auth_service._session_store
    # Valid session should remain
    assert "user2" in auth_service._session_store


# Global Service Instance Tests

@pytest.mark.asyncio
async def test_get_auth_service_singleton():
    """Test auth service singleton pattern."""
    with patch("ai_engine.api.auth.AuthenticationService") as MockService:
        mock_instance = AsyncMock()
        MockService.return_value = mock_instance

        service1 = await get_auth_service()
        service2 = await get_auth_service()

        # Should return same instance
        assert service1 is service2


# Permission and Role Tests

@pytest.mark.asyncio
async def test_require_permission_allowed():
    """Test permission check allows authorized user."""
    check = require_permission("protocol_discovery")

    user = {
        "user_id": "user123",
        "permissions": ["protocol_discovery", "copilot_access"]
    }

    result = await check(user)
    assert result == user


@pytest.mark.asyncio
async def test_require_permission_denied():
    """Test permission check denies unauthorized user."""
    check = require_permission("admin_access")

    user = {
        "user_id": "user123",
        "permissions": ["protocol_discovery"]
    }

    with pytest.raises(HTTPException) as exc_info:
        await check(user)

    assert exc_info.value.status_code == 403
    assert "admin_access" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_require_role_allowed():
    """Test role check allows authorized user."""
    check = require_role("administrator")

    user = {
        "user_id": "user123",
        "role": "administrator"
    }

    result = await check(user)
    assert result == user


@pytest.mark.asyncio
async def test_require_role_denied():
    """Test role check denies unauthorized user."""
    check = require_role("administrator")

    user = {
        "user_id": "user123",
        "role": "user"
    }

    with pytest.raises(HTTPException) as exc_info:
        await check(user)

    assert exc_info.value.status_code == 403


# Login/Logout Flow Tests

@pytest.mark.asyncio
async def test_login_success():
    """Test successful login."""
    with patch("ai_engine.api.auth.get_auth_service") as mock_get_service:
        mock_service = AsyncMock()
        mock_service.authenticate_user.return_value = {
            "user_id": "user123",
            "username": "testuser",
            "role": "user",
            "permissions": ["protocol_discovery"]
        }
        mock_service.create_access_token.return_value = "access_token"
        mock_service.create_refresh_token.return_value = "refresh_token"
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
async def test_login_invalid_credentials():
    """Test login with invalid credentials."""
    with patch("ai_engine.api.auth.get_auth_service") as mock_get_service:
        mock_service = AsyncMock()
        mock_service.authenticate_user.return_value = None
        mock_service.audit_logger = Mock()
        mock_service.audit_logger.log_login_failed = Mock()

        mock_get_service.return_value = mock_service

        with pytest.raises(HTTPException) as exc_info:
            await login("testuser", "wrong_password")

        assert exc_info.value.status_code == 401
        assert "Invalid credentials" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_refresh_access_token_success():
    """Test access token refresh."""
    with patch("ai_engine.api.auth.get_auth_service") as mock_get_service:
        mock_service = AsyncMock()
        mock_service.verify_token.return_value = {
            "user_id": "user123",
            "username": "testuser",
            "role": "user",
            "permissions": [],
            "type": "refresh"
        }
        mock_service.create_access_token.return_value = "new_access_token"

        mock_get_service.return_value = mock_service

        result = await refresh_access_token("refresh_token")

        assert result["access_token"] == "new_access_token"
        assert result["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_refresh_access_token_invalid_type():
    """Test refresh with access token instead of refresh token."""
    with patch("ai_engine.api.auth.get_auth_service") as mock_get_service:
        mock_service = AsyncMock()
        mock_service.verify_token.return_value = {
            "user_id": "user123",
            "type": "access"  # Wrong type
        }

        mock_get_service.return_value = mock_service

        with pytest.raises(HTTPException) as exc_info:
            await refresh_access_token("access_token")

        assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_logout():
    """Test logout."""
    user = {"user_id": "user123"}

    result = await logout(user)

    assert "message" in result


# API Key Management Tests

def test_initialize_auth_from_secrets_manager():
    """Test API key initialization from secrets manager."""
    mock_config = Mock()
    mock_config.environment = Mock()
    mock_config.environment.value = "development"
    mock_config.security = Mock()
    mock_config.security.api_key = None

    mock_secrets_mgr = Mock()
    mock_secrets_mgr.get_secret.return_value = "api_key_from_secrets_manager_32_chars_long"

    with patch("ai_engine.api.auth.get_config", return_value=mock_config):
        with patch("ai_engine.api.auth.get_secrets_manager", return_value=mock_secrets_mgr):
            api_key = initialize_auth(mock_config)

            assert api_key == "api_key_from_secrets_manager_32_chars_long"


def test_initialize_auth_from_env():
    """Test API key initialization from environment."""
    mock_config = Mock()
    mock_config.environment = Mock()
    mock_config.environment.value = "development"
    mock_config.security = Mock()
    mock_config.security.api_key = None

    with patch("ai_engine.api.auth.get_config", return_value=mock_config):
        with patch("ai_engine.api.auth.get_secrets_manager") as mock_sm:
            mock_sm.return_value.get_secret.return_value = None
            with patch.dict(os.environ, {"CRONOS_AI_API_KEY": "env_api_key_that_is_32_characters_long"}):
                api_key = initialize_auth(mock_config)

                assert api_key == "env_api_key_that_is_32_characters_long"


def test_initialize_auth_production_missing():
    """Test API key initialization in production without key."""
    mock_config = Mock()
    mock_config.environment = Mock()
    mock_config.environment.value = "production"
    mock_config.security = Mock()
    mock_config.security.api_key = None

    with patch("ai_engine.api.auth.get_config", return_value=mock_config):
        with patch("ai_engine.api.auth.get_secrets_manager") as mock_sm:
            mock_sm.return_value.get_secret.return_value = None
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(AuthenticationError, match="API key not configured"):
                    initialize_auth(mock_config)


def test_initialize_auth_development_generated():
    """Test API key generation in development."""
    mock_config = Mock()
    mock_config.environment = Mock()
    mock_config.environment.value = "development"
    mock_config.security = Mock()
    mock_config.security.api_key = None

    with patch("ai_engine.api.auth.get_config", return_value=mock_config):
        with patch("ai_engine.api.auth.get_secrets_manager") as mock_sm:
            mock_sm.return_value.get_secret.return_value = None
            with patch.dict(os.environ, {}, clear=True):
                api_key = initialize_auth(mock_config)

                # Should have generated a key
                assert api_key is not None
                assert api_key.startswith("cronos_ai_")


def test_get_api_key():
    """Test getting API key."""
    with patch("ai_engine.api.auth.initialize_auth", return_value="test_api_key"):
        api_key = get_api_key()

        assert api_key == "test_api_key"


def test_get_api_key_not_configured():
    """Test getting API key when not configured."""
    with patch("ai_engine.api.auth.initialize_auth", return_value=None):
        with pytest.raises(AuthenticationError, match="API key not configured"):
            get_api_key()


# Integration Tests

@pytest.mark.asyncio
async def test_full_auth_workflow():
    """Test complete authentication workflow."""
    mock_config = Mock()
    mock_config.environment = Mock()
    mock_config.environment.value = "development"
    mock_config.security = Mock()
    mock_config.security.jwt_secret = "test_secret_key_32_characters_long_here"

    with patch("ai_engine.api.auth.get_config", return_value=mock_config):
        with patch("ai_engine.api.auth.get_secrets_manager"):
            with patch("ai_engine.api.auth.get_audit_logger"):
                service = AuthenticationService(mock_config)

                # 1. Authenticate user
                user = await service.authenticate_user("admin", "DemoOnly_NotForProduction_123!")
                assert user is not None

                # 2. Create tokens
                access_token = service.create_access_token({
                    "user_id": user["user_id"],
                    "username": user["username"]
                })
                refresh_token = service.create_refresh_token({"user_id": user["user_id"]})

                # 3. Verify access token
                payload = await service.verify_token(access_token)
                assert payload["user_id"] == user["user_id"]

                # 4. Store session
                await service.store_session(user["user_id"], {"login_time": "2024-01-01"})

                # 5. Get session
                session = await service.get_session(user["user_id"])
                assert session is not None

                # 6. Revoke token
                await service.revoke_token(access_token)

                # 7. Verify revoked token fails
                with pytest.raises(AuthenticationError):
                    await service.verify_token(access_token)
