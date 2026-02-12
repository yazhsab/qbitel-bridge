"""
Tests for authentication and authorization.
Covers AuthenticationService, JWT tokens, password hashing, and session management.
"""

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import jwt

from ai_engine.api.auth import (
    AuthenticationService,
    AuthenticationError,
    pwd_context,
)
from ai_engine.core.config import Config


class TestPasswordHashing:
    """Test password hashing functionality."""

    @pytest.fixture
    def auth_service(self):
        """Create auth service instance."""
        with patch("ai_engine.api.auth.get_secrets_manager"):
            with patch("ai_engine.api.auth.get_audit_logger"):
                config = Config()
                config.security.jwt_secret = "test_secret_32chars_minimum!!"
                return AuthenticationService(config)

    def test_hash_password(self, auth_service):
        """Test password hashing."""
        password = "test_password_123"
        hashed = auth_service.hash_password(password)

        assert hashed != password
        assert len(hashed) > 50  # Bcrypt hashes are long
        assert hashed.startswith("$2b$")  # Bcrypt format

    def test_verify_password_correct(self, auth_service):
        """Test password verification with correct password."""
        password = "correct_password"
        hashed = auth_service.hash_password(password)

        assert auth_service.verify_password(password, hashed) is True

    def test_verify_password_incorrect(self, auth_service):
        """Test password verification with incorrect password."""
        password = "correct_password"
        hashed = auth_service.hash_password(password)

        assert auth_service.verify_password("wrong_password", hashed) is False

    def test_password_hash_uniqueness(self, auth_service):
        """Test that same password produces different hashes (salt)."""
        password = "same_password"
        hash1 = auth_service.hash_password(password)
        hash2 = auth_service.hash_password(password)

        assert hash1 != hash2  # Different salts
        assert auth_service.verify_password(password, hash1)
        assert auth_service.verify_password(password, hash2)


class TestJWTTokens:
    """Test JWT token creation and verification."""

    @pytest.fixture
    def auth_service(self):
        """Create auth service with known secret."""
        with patch("ai_engine.api.auth.get_secrets_manager"):
            with patch("ai_engine.api.auth.get_audit_logger"):
                config = Config()
                config.security.jwt_secret = "jwt_test_secret_32_chars_min!!"
                service = AuthenticationService(config)
                service.redis_client = None  # Use in-memory for tests
                return service

    def test_create_access_token(self, auth_service):
        """Test creating access token."""
        data = {"sub": "user123", "role": "admin"}
        token = auth_service.create_access_token(data)

        assert isinstance(token, str)
        assert len(token) > 50

        # Verify token structure
        decoded = jwt.decode(token, auth_service.secret_key, algorithms=[auth_service.algorithm])
        assert decoded["sub"] == "user123"
        assert decoded["role"] == "admin"
        assert decoded["type"] == "access"
        assert "exp" in decoded

    def test_create_refresh_token(self, auth_service):
        """Test creating refresh token."""
        data = {"sub": "user456"}
        token = auth_service.create_refresh_token(data)

        assert isinstance(token, str)

        decoded = jwt.decode(token, auth_service.secret_key, algorithms=[auth_service.algorithm])
        assert decoded["sub"] == "user456"
        assert decoded["type"] == "refresh"

    @pytest.mark.asyncio
    async def test_verify_valid_token(self, auth_service):
        """Test verifying valid token."""
        data = {"sub": "user789", "username": "testuser"}
        token = auth_service.create_access_token(data)

        payload = await auth_service.verify_token(token)

        assert payload["sub"] == "user789"
        assert payload["username"] == "testuser"
        assert payload["type"] == "access"

    @pytest.mark.asyncio
    async def test_verify_expired_token(self, auth_service):
        """Test verifying expired token."""
        data = {"sub": "user123"}
        # Create token that's already expired
        auth_service.access_token_expire_minutes = -1

        token = auth_service.create_access_token(data)

        with pytest.raises(AuthenticationError, match="expired"):
            await auth_service.verify_token(token)

    @pytest.mark.asyncio
    async def test_verify_invalid_token(self, auth_service):
        """Test verifying invalid token."""
        invalid_token = "invalid.jwt.token"

        with pytest.raises(AuthenticationError, match="Invalid token"):
            await auth_service.verify_token(invalid_token)

    @pytest.mark.asyncio
    async def test_verify_token_wrong_secret(self, auth_service):
        """Test verifying token with wrong secret."""
        data = {"sub": "user123"}
        token = auth_service.create_access_token(data)

        # Change secret
        auth_service.secret_key = "different_secret_key_for_testing_purposes_min_32"

        with pytest.raises(AuthenticationError, match="Invalid token"):
            await auth_service.verify_token(token)


class TestTokenRevocation:
    """Test token revocation/blacklisting."""

    @pytest.fixture
    def auth_service(self):
        """Create auth service."""
        with patch("ai_engine.api.auth.get_secrets_manager"):
            with patch("ai_engine.api.auth.get_audit_logger"):
                config = Config()
                config.security.jwt_secret = "revoke_test_secret_32chars!!"
                service = AuthenticationService(config)
                service.redis_client = None
                return service

    @pytest.mark.asyncio
    async def test_revoke_token_in_memory(self, auth_service):
        """Test revoking token using in-memory blacklist."""
        data = {"sub": "user123"}
        token = auth_service.create_access_token(data)

        # Token should be valid initially
        payload = await auth_service.verify_token(token)
        assert payload["sub"] == "user123"

        # Revoke token
        await auth_service.revoke_token(token)

        # Token should now be rejected
        with pytest.raises(AuthenticationError, match="revoked"):
            await auth_service.verify_token(token)

    @pytest.mark.asyncio
    async def test_revoke_token_with_redis(self, auth_service):
        """Test revoking token using Redis."""
        # Mock Redis client
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.setex = AsyncMock()
        auth_service.redis_client = mock_redis

        data = {"sub": "user123"}
        token = auth_service.create_access_token(data)

        await auth_service.revoke_token(token)

        # Should have called Redis setex
        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_blacklist_pruning(self, auth_service):
        """Test automatic blacklist pruning."""
        # Add expired entries
        old_expiry = datetime.utcnow() - timedelta(hours=1)
        auth_service._token_blacklist["old_token"] = old_expiry

        # Add valid entry
        future_expiry = datetime.utcnow() + timedelta(hours=1)
        auth_service._token_blacklist["valid_token"] = future_expiry

        # Prune should remove old entries
        auth_service._prune_blacklist()

        assert "old_token" not in auth_service._token_blacklist
        assert "valid_token" in auth_service._token_blacklist


class TestSessionManagement:
    """Test session management."""

    @pytest.fixture
    def auth_service(self):
        """Create auth service."""
        with patch("ai_engine.api.auth.get_secrets_manager"):
            with patch("ai_engine.api.auth.get_audit_logger"):
                config = Config()
                config.security.jwt_secret = "session_test_secret_32chars!!"
                service = AuthenticationService(config)
                service.redis_client = None
                return service

    @pytest.mark.asyncio
    async def test_store_session_in_memory(self, auth_service):
        """Test storing session in memory."""
        user_id = "user123"
        session_data = {"username": "testuser", "role": "admin"}

        await auth_service.store_session(user_id, session_data)

        # Check session was stored
        assert user_id in auth_service._session_store
        assert auth_service._session_store[user_id] == session_data

    @pytest.mark.asyncio
    async def test_get_session_in_memory(self, auth_service):
        """Test retrieving session from memory."""
        user_id = "user456"
        session_data = {"username": "anotheruser", "permissions": ["read", "write"]}

        await auth_service.store_session(user_id, session_data)
        retrieved = await auth_service.get_session(user_id)

        assert retrieved == session_data

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, auth_service):
        """Test retrieving non-existent session."""
        result = await auth_service.get_session("nonexistent_user")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_session(self, auth_service):
        """Test deleting session."""
        user_id = "user789"
        session_data = {"username": "tempuser"}

        await auth_service.store_session(user_id, session_data)
        assert user_id in auth_service._session_store

        await auth_service.delete_session(user_id)

        assert user_id not in auth_service._session_store

    @pytest.mark.asyncio
    async def test_session_expiry(self, auth_service):
        """Test session expiry tracking."""
        user_id = "user_expiry"
        session_data = {"username": "expiryuser"}
        ttl = 1  # 1 second TTL

        await auth_service.store_session(user_id, session_data, ttl)

        # Session should exist immediately
        assert user_id in auth_service._session_expiry

        # Wait for expiry
        await asyncio.sleep(1.5)

        # Prune expired sessions
        auth_service._prune_sessions()

        # Session should be removed
        assert user_id not in auth_service._session_store


class TestAuthenticationService:
    """Test AuthenticationService class."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        config = Config()
        config.security.jwt_secret = "auth_service_test_secret_32!!"
        config.redis.host = "localhost"
        config.redis.port = 6379
        config.redis.db = 0
        return config

    def test_service_initialization(self, config):
        """Test service initialization."""
        with patch("ai_engine.api.auth.get_secrets_manager"):
            with patch("ai_engine.api.auth.get_audit_logger"):
                service = AuthenticationService(config)

                assert service.config == config
                assert service.algorithm == "HS256"
                assert service.access_token_expire_minutes == 30
                assert service.refresh_token_expire_days == 7

    @pytest.mark.asyncio
    async def test_initialize_with_redis(self, config):
        """Test initialization with Redis."""
        with patch("ai_engine.api.auth.get_secrets_manager"):
            with patch("ai_engine.api.auth.get_audit_logger"):
                with patch("redis.asyncio.Redis") as mock_redis_class:
                    mock_redis = AsyncMock()
                    mock_redis.ping = AsyncMock()
                    mock_redis_class.return_value = mock_redis

                    service = AuthenticationService(config)
                    await service.initialize()

                    mock_redis.ping.assert_called_once()
                    assert service.redis_client is not None

    @pytest.mark.asyncio
    async def test_initialize_redis_failure(self, config):
        """Test initialization with Redis failure."""
        with patch("ai_engine.api.auth.get_secrets_manager"):
            with patch("ai_engine.api.auth.get_audit_logger"):
                with patch("redis.asyncio.Redis") as mock_redis_class:
                    mock_redis = AsyncMock()
                    mock_redis.ping = AsyncMock(side_effect=Exception("Connection failed"))
                    mock_redis_class.return_value = mock_redis

                    service = AuthenticationService(config)
                    await service.initialize()

                    # Should fall back to in-memory
                    assert service.redis_client is None


class TestSecretKeyLoading:
    """Test JWT secret key loading strategies."""

    def test_load_from_secrets_manager(self):
        """Test loading secret from secrets manager."""
        mock_secrets_mgr = Mock()
        mock_secrets_mgr.get_secret = Mock(return_value="secret_from_manager_32!!")

        with patch("ai_engine.api.auth.get_secrets_manager", return_value=mock_secrets_mgr):
            with patch("ai_engine.api.auth.get_audit_logger"):
                config = Config()
                service = AuthenticationService(config)

                assert len(service.secret_key) >= 32

    def test_load_from_config(self):
        """Test loading secret from configuration."""
        config = Config()
        config.security.jwt_secret = "config_secret_test_32chars!!"

        with patch("ai_engine.api.auth.get_secrets_manager") as mock_sm:
            mock_sm.return_value.get_secret = Mock(return_value=None)
            with patch("ai_engine.api.auth.get_audit_logger"):
                service = AuthenticationService(config)

                assert service.secret_key == "config_secret_test_32chars!!"

    def test_generate_ephemeral_development(self):
        """Test ephemeral secret generation for development."""
        config = Config()
        config.environment = Mock(value="development")

        with patch("ai_engine.api.auth.get_secrets_manager") as mock_sm:
            mock_sm.return_value.get_secret = Mock(return_value=None)
            with patch("ai_engine.api.auth.get_audit_logger"):
                service = AuthenticationService(config)

                assert service.secret_key is not None
                assert len(service.secret_key) >= 32

    def test_production_requires_secret(self):
        """Test production mode requires configured secret."""
        config = Config()
        config.environment = Mock(value="production")
        config.security.jwt_secret = ""

        with patch("ai_engine.api.auth.get_secrets_manager") as mock_sm:
            mock_sm.return_value.get_secret = Mock(return_value=None)
            with patch("ai_engine.api.auth.get_audit_logger"):
                with pytest.raises(AuthenticationError, match="not configured in production"):
                    AuthenticationService(config)


class TestTokenDataEncoding:
    """Test encoding various data types in tokens."""

    @pytest.fixture
    def auth_service(self):
        """Create auth service."""
        with patch("ai_engine.api.auth.get_secrets_manager"):
            with patch("ai_engine.api.auth.get_audit_logger"):
                config = Config()
                config.security.jwt_secret = "encoding_test_secret_32!!"
                return AuthenticationService(config)

    def test_encode_complex_data(self, auth_service):
        """Test encoding complex data structures."""
        data = {
            "sub": "user123",
            "username": "testuser",
            "roles": ["admin", "user"],
            "permissions": {"read": True, "write": True, "delete": False},
            "metadata": {"last_login": "2024-01-01", "login_count": 42},
        }

        token = auth_service.create_access_token(data)

        decoded = jwt.decode(token, auth_service.secret_key, algorithms=[auth_service.algorithm])

        assert decoded["sub"] == "user123"
        assert decoded["roles"] == ["admin", "user"]
        assert decoded["permissions"]["read"] is True

    def test_encode_minimal_data(self, auth_service):
        """Test encoding minimal data."""
        data = {"sub": "user_minimal"}
        token = auth_service.create_access_token(data)

        decoded = jwt.decode(token, auth_service.secret_key, algorithms=[auth_service.algorithm])

        assert decoded["sub"] == "user_minimal"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def auth_service(self):
        """Create auth service."""
        with patch("ai_engine.api.auth.get_secrets_manager"):
            with patch("ai_engine.api.auth.get_audit_logger"):
                config = Config()
                config.security.jwt_secret = "edge_case_test_secret_32!!"
                service = AuthenticationService(config)
                service.redis_client = None
                return service

    def test_hash_empty_password(self, auth_service):
        """Test hashing empty password."""
        hashed = auth_service.hash_password("")
        assert hashed != ""
        assert auth_service.verify_password("", hashed)

    def test_hash_very_long_password(self, auth_service):
        """Test hashing very long password."""
        long_password = "a" * 1000
        hashed = auth_service.hash_password(long_password)
        assert auth_service.verify_password(long_password, hashed)

    @pytest.mark.asyncio
    async def test_verify_token_empty_blacklist(self, auth_service):
        """Test verifying token with empty blacklist."""
        data = {"sub": "user123"}
        token = auth_service.create_access_token(data)

        payload = await auth_service.verify_token(token)
        assert payload["sub"] == "user123"

    @pytest.mark.asyncio
    async def test_store_empty_session(self, auth_service):
        """Test storing empty session data."""
        await auth_service.store_session("user123", {})

        session = await auth_service.get_session("user123")
        assert session == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
