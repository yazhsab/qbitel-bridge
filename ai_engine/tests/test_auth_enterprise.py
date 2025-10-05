"""
Tests for ai_engine/api/auth_enterprise.py - Enterprise Authentication
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi import HTTPException


class TestEnterpriseAuthenticationService:
    """Test suite for EnterpriseAuthenticationService."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = Mock()
        config.security = Mock()
        config.security.jwt_secret = "test_secret_key_with_at_least_32_characters_long"
        return config

    @pytest.fixture
    def auth_service(self, mock_config):
        """Create authentication service instance."""
        from ai_engine.api.auth_enterprise import EnterpriseAuthenticationService

        return EnterpriseAuthenticationService(mock_config)

    def test_service_initialization(self, auth_service):
        """Test service initialization."""
        assert auth_service.algorithm == "HS256"
        assert auth_service.access_token_expire_minutes == 30
        assert auth_service.refresh_token_expire_days == 7
        assert auth_service.mfa_required_for_admin is True

    def test_load_secret_key_success(self, mock_config):
        """Test loading JWT secret key."""
        from ai_engine.api.auth_enterprise import EnterpriseAuthenticationService

        service = EnterpriseAuthenticationService(mock_config)
        assert len(service.secret_key) >= 32

    def test_load_secret_key_failure(self):
        """Test loading JWT secret key failure."""
        from ai_engine.api.auth_enterprise import EnterpriseAuthenticationService

        config = Mock()
        config.security = Mock()
        config.security.jwt_secret = "short"

        with pytest.raises(ValueError, match="JWT secret not configured"):
            EnterpriseAuthenticationService(config)

    def test_hash_password(self, auth_service):
        """Test password hashing."""
        password = "TestPassword123!"
        hashed = auth_service.hash_password(password)

        assert hashed != password
        assert len(hashed) > 0

    def test_verify_password(self, auth_service):
        """Test password verification."""
        password = "TestPassword123!"
        hashed = auth_service.hash_password(password)

        assert auth_service.verify_password(password, hashed) is True
        assert auth_service.verify_password("WrongPassword", hashed) is False

    def test_validate_password_strength_valid(self, auth_service):
        """Test password strength validation with valid password."""
        password = "ValidPass123!@#"
        is_valid, error = auth_service.validate_password_strength(password)

        assert is_valid is True
        assert error is None

    def test_validate_password_strength_too_short(self, auth_service):
        """Test password strength validation - too short."""
        password = "Short1!"
        is_valid, error = auth_service.validate_password_strength(password)

        assert is_valid is False
        assert "at least 12 characters" in error

    def test_validate_password_strength_no_uppercase(self, auth_service):
        """Test password strength validation - no uppercase."""
        password = "lowercase123!"
        is_valid, error = auth_service.validate_password_strength(password)

        assert is_valid is False
        assert "uppercase letter" in error

    def test_validate_password_strength_no_lowercase(self, auth_service):
        """Test password strength validation - no lowercase."""
        password = "UPPERCASE123!"
        is_valid, error = auth_service.validate_password_strength(password)

        assert is_valid is False
        assert "lowercase letter" in error

    def test_validate_password_strength_no_digit(self, auth_service):
        """Test password strength validation - no digit."""
        password = "NoDigitsHere!"
        is_valid, error = auth_service.validate_password_strength(password)

        assert is_valid is False
        assert "digit" in error

    def test_validate_password_strength_no_special(self, auth_service):
        """Test password strength validation - no special character."""
        password = "NoSpecial123"
        is_valid, error = auth_service.validate_password_strength(password)

        assert is_valid is False
        assert "special character" in error

    def test_validate_password_strength_common_pattern(self, auth_service):
        """Test password strength validation - common pattern."""
        password = "Password123!"
        is_valid, error = auth_service.validate_password_strength(password)

        assert is_valid is False
        assert "common patterns" in error

    @pytest.mark.asyncio
    async def test_change_password_success(self, auth_service):
        """Test successful password change."""
        from ai_engine.models.database import User

        mock_db = AsyncMock()
        user = User(
            id="user123",
            username="testuser",
            password_hash=auth_service.hash_password("OldPassword123!"),
        )

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = user
        mock_db.execute.return_value = mock_result

        result = await auth_service.change_password(
            mock_db, "user123", "OldPassword123!", "NewPassword456!"
        )

        assert result is True
        assert user.must_change_password is False

    @pytest.mark.asyncio
    async def test_change_password_user_not_found(self, auth_service):
        """Test password change with user not found."""
        mock_db = AsyncMock()
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        with pytest.raises(HTTPException, match="User not found"):
            await auth_service.change_password(
                mock_db, "user123", "OldPassword123!", "NewPassword456!"
            )

    @pytest.mark.asyncio
    async def test_change_password_invalid_old_password(self, auth_service):
        """Test password change with invalid old password."""
        from ai_engine.models.database import User

        mock_db = AsyncMock()
        user = User(
            id="user123",
            username="testuser",
            password_hash=auth_service.hash_password("OldPassword123!"),
        )

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = user
        mock_db.execute.return_value = mock_result

        with pytest.raises(HTTPException, match="Invalid old password"):
            await auth_service.change_password(
                mock_db, "user123", "WrongPassword!", "NewPassword456!"
            )

    def test_generate_totp_secret(self, auth_service):
        """Test TOTP secret generation."""
        secret = auth_service.generate_totp_secret()

        assert len(secret) > 0
        assert isinstance(secret, str)

    def test_generate_totp_qr_code(self, auth_service):
        """Test TOTP QR code generation."""
        secret = auth_service.generate_totp_secret()
        qr_code = auth_service.generate_totp_qr_code("testuser", secret)

        assert len(qr_code) > 0
        assert isinstance(qr_code, str)

    def test_verify_totp(self, auth_service):
        """Test TOTP verification."""
        import pyotp

        secret = auth_service.generate_totp_secret()
        totp = pyotp.TOTP(secret)
        token = totp.now()

        assert auth_service.verify_totp(secret, token) is True
        assert auth_service.verify_totp(secret, "000000") is False

    def test_generate_backup_codes(self, auth_service):
        """Test backup codes generation."""
        codes = auth_service.generate_backup_codes(count=10)

        assert len(codes) == 10
        assert all(isinstance(code, str) for code in codes)
        assert all(len(code) == 8 for code in codes)  # 4 bytes hex = 8 chars

    @pytest.mark.asyncio
    async def test_enable_mfa_totp(self, auth_service):
        """Test enabling MFA with TOTP."""
        from ai_engine.models.database import User, MFAMethod

        mock_db = AsyncMock()
        user = User(id="user123", username="testuser")

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = user
        mock_db.execute.return_value = mock_result

        result = await auth_service.enable_mfa(mock_db, "user123", MFAMethod.TOTP)

        assert "qr_code" in result
        assert "secret" in result
        assert "backup_codes" in result
        assert user.mfa_enabled is True

    @pytest.mark.asyncio
    async def test_enable_mfa_user_not_found(self, auth_service):
        """Test enabling MFA with user not found."""
        from ai_engine.models.database import MFAMethod

        mock_db = AsyncMock()
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        with pytest.raises(HTTPException, match="User not found"):
            await auth_service.enable_mfa(mock_db, "user123", MFAMethod.TOTP)

    @pytest.mark.asyncio
    async def test_verify_mfa_token_success(self, auth_service):
        """Test MFA token verification success."""
        from ai_engine.models.database import User, MFAMethod
        import pyotp

        secret = auth_service.generate_totp_secret()
        totp = pyotp.TOTP(secret)
        token = totp.now()

        mock_db = AsyncMock()
        user = User(
            id="user123",
            username="testuser",
            mfa_enabled=True,
            mfa_method=MFAMethod.TOTP,
            mfa_secret=secret,
        )

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = user
        mock_db.execute.return_value = mock_result

        result = await auth_service.verify_mfa_token(mock_db, "user123", token)

        assert result is True

    def test_generate_api_key(self, auth_service):
        """Test API key generation."""
        api_key, key_hash = auth_service.generate_api_key()

        assert api_key.startswith("cronos_")
        assert len(key_hash) == 64  # SHA256 hex digest

    @pytest.mark.asyncio
    async def test_create_api_key(self, auth_service):
        """Test API key creation."""
        mock_db = AsyncMock()

        result = await auth_service.create_api_key(
            mock_db,
            user_id="user123",
            name="Test API Key",
            description="Test description",
            expires_days=30,
        )

        assert "api_key" in result
        assert "key_id" in result
        assert "key_prefix" in result
        assert result["api_key"].startswith("cronos_")

    @pytest.mark.asyncio
    async def test_revoke_api_key(self, auth_service):
        """Test API key revocation."""
        from ai_engine.models.database import APIKey, APIKeyStatus

        mock_db = AsyncMock()
        api_key = APIKey(
            id="key123",
            key_hash="hash",
            key_prefix="cronos_",
            name="Test Key",
            user_id="user123",
            status=APIKeyStatus.ACTIVE,
        )

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = api_key
        mock_db.execute.return_value = mock_result

        result = await auth_service.revoke_api_key(
            mock_db, "key123", "admin", "Security review"
        )

        assert result is True
        assert api_key.status == APIKeyStatus.REVOKED

    @pytest.mark.asyncio
    async def test_verify_api_key_success(self, auth_service):
        """Test API key verification success."""
        from ai_engine.models.database import APIKey, User, APIKeyStatus
        import hashlib

        api_key = "cronos_test_key"
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        mock_db = AsyncMock()
        api_key_record = APIKey(
            id="key123", key_hash=key_hash, status=APIKeyStatus.ACTIVE, usage_count=0
        )
        user = User(id="user123", username="testuser", is_active=True)

        mock_result = Mock()
        mock_result.first.return_value = (api_key_record, user)
        mock_db.execute.return_value = mock_result

        result = await auth_service.verify_api_key(mock_db, api_key)

        assert result == user

    def test_create_access_token(self, auth_service):
        """Test access token creation."""
        data = {"user_id": "user123", "username": "testuser"}
        token = auth_service.create_access_token(data)

        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_refresh_token(self, auth_service):
        """Test refresh token creation."""
        data = {"user_id": "user123"}
        token = auth_service.create_refresh_token(data)

        assert isinstance(token, str)
        assert len(token) > 0

    @pytest.mark.asyncio
    async def test_verify_token_success(self, auth_service):
        """Test token verification success."""
        data = {"user_id": "user123"}
        token = auth_service.create_access_token(data)

        payload = await auth_service.verify_token(token)

        assert payload["user_id"] == "user123"
        assert "exp" in payload

    @pytest.mark.asyncio
    async def test_verify_token_expired(self, auth_service):
        """Test token verification with expired token."""
        import jwt
        from datetime import datetime, timedelta

        data = {"user_id": "user123", "exp": datetime.utcnow() - timedelta(hours=1)}
        token = jwt.encode(
            data, auth_service.secret_key, algorithm=auth_service.algorithm
        )

        with pytest.raises(HTTPException, match="Token has expired"):
            await auth_service.verify_token(token)

    @pytest.mark.asyncio
    async def test_create_session(self, auth_service):
        """Test session creation."""
        mock_db = AsyncMock()

        session = await auth_service.create_session(
            mock_db,
            user_id="user123",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
        )

        assert session.user_id == "user123"
        assert session.ip_address == "192.168.1.1"

    @pytest.mark.asyncio
    async def test_revoke_session(self, auth_service):
        """Test session revocation."""
        from ai_engine.models.database import UserSession

        mock_db = AsyncMock()
        session = UserSession(session_token="token123", user_id="user123")

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = session
        mock_db.execute.return_value = mock_result

        result = await auth_service.revoke_session(mock_db, "token123")

        assert result is True
        assert session.revoked_at is not None

    @pytest.mark.asyncio
    async def test_authenticate_user_success(self, auth_service):
        """Test successful user authentication."""
        from ai_engine.models.database import User

        password = "TestPassword123!"
        mock_db = AsyncMock()
        user = User(
            id="user123",
            username="testuser",
            password_hash=auth_service.hash_password(password),
            is_active=True,
            failed_login_attempts=0,
            mfa_enabled=False,
        )

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = user
        mock_db.execute.return_value = mock_result

        result = await auth_service.authenticate_user(mock_db, "testuser", password)

        assert result == user
        assert user.failed_login_attempts == 0

    @pytest.mark.asyncio
    async def test_authenticate_user_not_found(self, auth_service):
        """Test authentication with user not found."""
        mock_db = AsyncMock()
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        result = await auth_service.authenticate_user(
            mock_db, "nonexistent", "password"
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_authenticate_user_account_locked(self, auth_service):
        """Test authentication with locked account."""
        from ai_engine.models.database import User

        mock_db = AsyncMock()
        user = User(
            id="user123",
            username="testuser",
            is_active=True,
            account_locked_until=datetime.utcnow() + timedelta(hours=1),
        )

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = user
        mock_db.execute.return_value = mock_result

        with pytest.raises(HTTPException, match="Account locked"):
            await auth_service.authenticate_user(mock_db, "testuser", "password")

    @pytest.mark.asyncio
    async def test_authenticate_user_invalid_password(self, auth_service):
        """Test authentication with invalid password."""
        from ai_engine.models.database import User

        mock_db = AsyncMock()
        user = User(
            id="user123",
            username="testuser",
            password_hash=auth_service.hash_password("CorrectPassword123!"),
            is_active=True,
            failed_login_attempts=0,
        )

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = user
        mock_db.execute.return_value = mock_result

        result = await auth_service.authenticate_user(
            mock_db, "testuser", "WrongPassword"
        )

        assert result is None
        assert user.failed_login_attempts == 1

    def test_requires_mfa_admin_user(self, auth_service):
        """Test MFA requirement for admin user."""
        from ai_engine.models.database import User, UserRole

        user = User(
            id="user123",
            username="admin",
            role=UserRole.ADMINISTRATOR,
            created_at=datetime.utcnow() - timedelta(days=30),
        )

        assert auth_service._requires_mfa(user) is True

    def test_requires_mfa_grace_period(self, auth_service):
        """Test MFA requirement during grace period."""
        from ai_engine.models.database import User, UserRole

        user = User(
            id="user123",
            username="admin",
            role=UserRole.ADMINISTRATOR,
            created_at=datetime.utcnow() - timedelta(days=3),
        )

        assert auth_service._requires_mfa(user) is False

    def test_is_privileged_user(self, auth_service):
        """Test privileged user check."""
        from ai_engine.models.database import User, UserRole

        admin_user = User(id="user1", role=UserRole.ADMINISTRATOR)
        regular_user = User(id="user2", role=UserRole.USER)

        assert auth_service._is_privileged_user(admin_user) is True
        assert auth_service._is_privileged_user(regular_user) is False

    @pytest.mark.asyncio
    async def test_authenticate_user_with_mfa_required(self, auth_service):
        """Test authentication when MFA is required but not provided."""
        from ai_engine.models.database import User, UserRole

        password = "TestPassword123!"
        mock_db = AsyncMock()
        user = User(
            id="user123",
            username="admin",
            password_hash=auth_service.hash_password(password),
            is_active=True,
            failed_login_attempts=0,
            mfa_enabled=True,
            role=UserRole.ADMINISTRATOR,
            created_at=datetime.utcnow() - timedelta(days=30),
        )

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = user
        mock_db.execute.return_value = mock_result

        with pytest.raises(HTTPException, match="MFA token required"):
            await auth_service.authenticate_user(
                mock_db, "admin", password, mfa_token=None
            )

    @pytest.mark.asyncio
    async def test_authenticate_user_with_invalid_mfa(self, auth_service):
        """Test authentication with invalid MFA token."""
        from ai_engine.models.database import User, UserRole, MFAMethod

        password = "TestPassword123!"
        secret = auth_service.generate_totp_secret()

        mock_db = AsyncMock()
        user = User(
            id="user123",
            username="admin",
            password_hash=auth_service.hash_password(password),
            is_active=True,
            failed_login_attempts=0,
            mfa_enabled=True,
            mfa_method=MFAMethod.TOTP,
            mfa_secret=secret,
            role=UserRole.ADMINISTRATOR,
            created_at=datetime.utcnow() - timedelta(days=30),
        )

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = user
        mock_db.execute.return_value = mock_result

        result = await auth_service.authenticate_user(
            mock_db, "admin", password, mfa_token="000000"
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_authenticate_user_account_lockout(self, auth_service):
        """Test account lockout after failed attempts."""
        from ai_engine.models.database import User

        mock_db = AsyncMock()
        user = User(
            id="user123",
            username="testuser",
            password_hash=auth_service.hash_password("CorrectPassword123!"),
            is_active=True,
            failed_login_attempts=4,
        )

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = user
        mock_db.execute.return_value = mock_result

        # This should trigger lockout
        result = await auth_service.authenticate_user(
            mock_db, "testuser", "WrongPassword"
        )

        assert result is None
        assert user.failed_login_attempts == 5
        assert user.account_locked_until is not None

    @pytest.mark.asyncio
    async def test_verify_api_key_expired(self, auth_service):
        """Test API key verification with expired key."""
        from ai_engine.models.database import APIKey, User, APIKeyStatus
        import hashlib

        api_key = "cronos_test_key"
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        mock_db = AsyncMock()
        api_key_record = APIKey(
            id="key123",
            key_hash=key_hash,
            status=APIKeyStatus.ACTIVE,
            expires_at=datetime.utcnow() - timedelta(days=1),
        )
        user = User(id="user123", username="testuser", is_active=True)

        mock_result = Mock()
        mock_result.first.return_value = (api_key_record, user)
        mock_db.execute.return_value = mock_result

        result = await auth_service.verify_api_key(mock_db, api_key)

        assert result is None
        assert api_key_record.status == APIKeyStatus.EXPIRED

    @pytest.mark.asyncio
    async def test_verify_api_key_not_found(self, auth_service):
        """Test API key verification with non-existent key."""
        mock_db = AsyncMock()
        mock_result = Mock()
        mock_result.first.return_value = None
        mock_db.execute.return_value = mock_result

        result = await auth_service.verify_api_key(mock_db, "invalid_key")

        assert result is None

    @pytest.mark.asyncio
    async def test_revoke_api_key_not_found(self, auth_service):
        """Test revoking non-existent API key."""
        mock_db = AsyncMock()
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        with pytest.raises(HTTPException, match="API key not found"):
            await auth_service.revoke_api_key(mock_db, "key123", "admin")

    @pytest.mark.asyncio
    async def test_revoke_session_not_found(self, auth_service):
        """Test revoking non-existent session."""
        mock_db = AsyncMock()
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        result = await auth_service.revoke_session(mock_db, "invalid_token")

        assert result is False

    @pytest.mark.asyncio
    async def test_verify_mfa_token_not_enabled(self, auth_service):
        """Test MFA verification when MFA is not enabled."""
        from ai_engine.models.database import User

        mock_db = AsyncMock()
        user = User(id="user123", username="testuser", mfa_enabled=False)

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = user
        mock_db.execute.return_value = mock_result

        result = await auth_service.verify_mfa_token(mock_db, "user123", "123456")

        assert result is False

    @pytest.mark.asyncio
    async def test_enable_mfa_unsupported_method(self, auth_service):
        """Test enabling MFA with unsupported method."""
        from ai_engine.models.database import User

        mock_db = AsyncMock()
        user = User(id="user123", username="testuser")

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = user
        mock_db.execute.return_value = mock_result

        # Create a mock MFA method that's not TOTP
        mock_method = Mock()
        mock_method.value = "SMS"

        with pytest.raises(HTTPException, match="not implemented"):
            await auth_service.enable_mfa(mock_db, "user123", mock_method)

    @pytest.mark.asyncio
    async def test_change_password_weak_password(self, auth_service):
        """Test password change with weak new password."""
        from ai_engine.models.database import User

        mock_db = AsyncMock()
        user = User(
            id="user123",
            username="testuser",
            password_hash=auth_service.hash_password("OldPassword123!"),
        )

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = user
        mock_db.execute.return_value = mock_result

        with pytest.raises(HTTPException):
            await auth_service.change_password(
                mock_db, "user123", "OldPassword123!", "weak"
            )

    @pytest.mark.asyncio
    async def test_verify_token_invalid(self, auth_service):
        """Test token verification with invalid token."""
        with pytest.raises(HTTPException, match="Invalid token"):
            await auth_service.verify_token("invalid_token")

    @pytest.mark.asyncio
    async def test_create_api_key_with_permissions(self, auth_service):
        """Test API key creation with specific permissions."""
        mock_db = AsyncMock()

        result = await auth_service.create_api_key(
            mock_db,
            user_id="user123",
            name="Test API Key",
            permissions=["read:data", "write:data"],
        )

        assert "api_key" in result
        mock_db.add.assert_called_once()

    def test_requires_mfa_privileged_user(self, auth_service):
        """Test MFA requirement for privileged user."""
        from ai_engine.models.database import User, UserRole

        user = User(
            id="user123",
            username="security_admin",
            role=UserRole.SECURITY_ADMIN,
            created_at=datetime.utcnow() - timedelta(days=30),
        )

        assert auth_service._requires_mfa(user) is True

    def test_requires_mfa_compliance_officer(self, auth_service):
        """Test MFA requirement for compliance officer."""
        from ai_engine.models.database import User, UserRole

        user = User(
            id="user123",
            username="compliance",
            role=UserRole.COMPLIANCE_OFFICER,
            created_at=datetime.utcnow() - timedelta(days=30),
        )

        assert auth_service._requires_mfa(user) is True

    @pytest.mark.asyncio
    async def test_authenticate_user_mfa_not_enabled_privileged(self, auth_service):
        """Test authentication for privileged user without MFA enabled."""
        from ai_engine.models.database import User, UserRole

        password = "TestPassword123!"
        mock_db = AsyncMock()
        user = User(
            id="user123",
            username="admin",
            password_hash=auth_service.hash_password(password),
            is_active=True,
            failed_login_attempts=0,
            mfa_enabled=False,
            role=UserRole.ADMINISTRATOR,
            created_at=datetime.utcnow() - timedelta(days=30),
        )

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = user
        mock_db.execute.return_value = mock_result

        with pytest.raises(HTTPException, match="MFA is required"):
            await auth_service.authenticate_user(mock_db, "admin", password)


class TestGetAuthService:
    """Test suite for get_auth_service function."""

    @pytest.mark.asyncio
    async def test_get_auth_service(self):
        """Test getting auth service instance."""
        from ai_engine.api.auth_enterprise import get_auth_service

        with patch("ai_engine.api.auth_enterprise.EnterpriseAuthenticationService"):
            service = await get_auth_service()
            assert service is not None

    @pytest.mark.asyncio
    async def test_get_auth_service_singleton(self):
        """Test that get_auth_service returns singleton instance."""
        from ai_engine.api import auth_enterprise

        # Reset global instance
        auth_enterprise._auth_service = None

        with patch("ai_engine.api.auth_enterprise.get_config") as mock_config:
            mock_config.return_value = Mock()
            mock_config.return_value.security = Mock()
            mock_config.return_value.security.jwt_secret = (
                "test_secret_key_with_at_least_32_characters_long"
            )

            service1 = await auth_enterprise.get_auth_service()
            service2 = await auth_enterprise.get_auth_service()

            assert service1 is service2
