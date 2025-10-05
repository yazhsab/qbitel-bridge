"""
Comprehensive tests for ai_engine/api/auth_enterprise.py

Tests cover:
- Password management and validation
- Multi-factor authentication (TOTP, backup codes)
- API key management
- JWT token creation and verification
- Session management
- User authentication with MFA
- Account lockout policies
- Audit logging
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, MagicMock, patch, call
import hashlib
import secrets
import jwt
from jwt import ExpiredSignatureError, PyJWTError
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ai_engine.api.auth_enterprise import (
    EnterpriseAuthenticationService,
    get_auth_service,
)
from ai_engine.models.database import (
    User,
    APIKey,
    UserSession,
    AuditLog,
    UserRole,
    MFAMethod,
    APIKeyStatus,
    AuditAction,
)


# Fixtures

@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = Mock()
    config.security = Mock()
    config.security.jwt_secret = "test_secret_key_that_is_at_least_32_characters_long"
    return config


@pytest.fixture
def auth_service(mock_config):
    """Create EnterpriseAuthenticationService instance."""
    with patch("ai_engine.api.auth_enterprise.get_config", return_value=mock_config):
        with patch("ai_engine.api.auth_enterprise.get_audit_logger"):
            service = EnterpriseAuthenticationService(mock_config)
            return service


@pytest.fixture
async def mock_db():
    """Create mock database session."""
    db = AsyncMock(spec=AsyncSession)
    return db


@pytest.fixture
def mock_user():
    """Create mock user."""
    user = Mock(spec=User)
    user.id = "user-123"
    user.username = "testuser"
    user.email = "test@example.com"
    user.password_hash = "$2b$12$test_hash"
    user.is_active = True
    user.role = UserRole.USER
    user.mfa_enabled = False
    user.mfa_method = None
    user.mfa_secret = None
    user.mfa_backup_codes = []
    user.failed_login_attempts = 0
    user.account_locked_until = None
    user.created_at = datetime.utcnow()
    user.last_login = None
    user.last_login_ip = None
    user.must_change_password = False
    user.password_changed_at = datetime.utcnow()
    return user


# Initialization Tests

def test_service_initialization(auth_service):
    """Test service initialization."""
    assert auth_service.config is not None
    assert auth_service.secret_key is not None
    assert auth_service.algorithm == "HS256"
    assert auth_service.access_token_expire_minutes == 30
    assert auth_service.refresh_token_expire_days == 7


def test_service_initialization_missing_secret():
    """Test initialization fails with missing or short secret."""
    config = Mock()
    config.security = Mock()
    config.security.jwt_secret = "short"  # Too short

    with patch("ai_engine.api.auth_enterprise.get_config", return_value=config):
        with patch("ai_engine.api.auth_enterprise.get_audit_logger"):
            with pytest.raises(ValueError, match="JWT secret not configured"):
                EnterpriseAuthenticationService(config)


# Password Management Tests

def test_hash_password(auth_service):
    """Test password hashing."""
    password = "test_password_123"
    hashed = auth_service.hash_password(password)

    assert hashed != password
    assert hashed.startswith("$2b$")


def test_verify_password_success(auth_service):
    """Test password verification success."""
    password = "test_password_123"
    hashed = auth_service.hash_password(password)

    assert auth_service.verify_password(password, hashed) is True


def test_verify_password_failure(auth_service):
    """Test password verification failure."""
    password = "test_password_123"
    hashed = auth_service.hash_password(password)

    assert auth_service.verify_password("wrong_password", hashed) is False


def test_validate_password_strength_success(auth_service):
    """Test password strength validation success."""
    password = "StrongP@ssw0rd123!"
    is_valid, error = auth_service.validate_password_strength(password)

    assert is_valid is True
    assert error is None


def test_validate_password_too_short(auth_service):
    """Test password too short."""
    password = "Short1!"
    is_valid, error = auth_service.validate_password_strength(password)

    assert is_valid is False
    assert "12 characters" in error


def test_validate_password_no_uppercase(auth_service):
    """Test password missing uppercase."""
    password = "lowercase123!@#"
    is_valid, error = auth_service.validate_password_strength(password)

    assert is_valid is False
    assert "uppercase" in error


def test_validate_password_no_lowercase(auth_service):
    """Test password missing lowercase."""
    password = "UPPERCASE123!@#"
    is_valid, error = auth_service.validate_password_strength(password)

    assert is_valid is False
    assert "lowercase" in error


def test_validate_password_no_digit(auth_service):
    """Test password missing digit."""
    password = "NoDigitsHere!@#"
    is_valid, error = auth_service.validate_password_strength(password)

    assert is_valid is False
    assert "digit" in error


def test_validate_password_no_special_char(auth_service):
    """Test password missing special character."""
    password = "NoSpecialChar123"
    is_valid, error = auth_service.validate_password_strength(password)

    assert is_valid is False
    assert "special character" in error


def test_validate_password_common_pattern(auth_service):
    """Test password with common pattern."""
    password = "Password123!@#"
    is_valid, error = auth_service.validate_password_strength(password)

    assert is_valid is False
    assert "common patterns" in error


@pytest.mark.asyncio
async def test_change_password_success(auth_service, mock_db, mock_user):
    """Test password change success."""
    old_password = "OldPassword123!"
    new_password = "NewPassword456!"

    mock_user.password_hash = auth_service.hash_password(old_password)

    # Mock database query
    mock_result = AsyncMock()
    mock_result.scalar_one_or_none.return_value = mock_user
    mock_db.execute.return_value = mock_result

    result = await auth_service.change_password(
        mock_db, str(mock_user.id), old_password, new_password
    )

    assert result is True
    assert mock_user.must_change_password is False
    assert mock_user.password_changed_at is not None
    mock_db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_change_password_user_not_found(auth_service, mock_db):
    """Test password change with non-existent user."""
    mock_result = AsyncMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_db.execute.return_value = mock_result

    with pytest.raises(HTTPException, match="User not found"):
        await auth_service.change_password(
            mock_db, "non-existent", "old", "new"
        )


@pytest.mark.asyncio
async def test_change_password_invalid_old_password(auth_service, mock_db, mock_user):
    """Test password change with invalid old password."""
    mock_user.password_hash = auth_service.hash_password("CorrectPassword123!")

    mock_result = AsyncMock()
    mock_result.scalar_one_or_none.return_value = mock_user
    mock_db.execute.return_value = mock_result

    with pytest.raises(HTTPException, match="Invalid old password"):
        await auth_service.change_password(
            mock_db, str(mock_user.id), "WrongPassword!", "NewPassword456!"
        )


@pytest.mark.asyncio
async def test_change_password_weak_new_password(auth_service, mock_db, mock_user):
    """Test password change with weak new password."""
    old_password = "OldPassword123!"
    mock_user.password_hash = auth_service.hash_password(old_password)

    mock_result = AsyncMock()
    mock_result.scalar_one_or_none.return_value = mock_user
    mock_db.execute.return_value = mock_result

    with pytest.raises(HTTPException, match="12 characters"):
        await auth_service.change_password(
            mock_db, str(mock_user.id), old_password, "weak"
        )


# Multi-Factor Authentication Tests

def test_generate_totp_secret(auth_service):
    """Test TOTP secret generation."""
    secret = auth_service.generate_totp_secret()

    assert isinstance(secret, str)
    assert len(secret) > 0


def test_generate_totp_qr_code(auth_service):
    """Test TOTP QR code generation."""
    secret = auth_service.generate_totp_secret()
    qr_code = auth_service.generate_totp_qr_code("testuser", secret)

    assert isinstance(qr_code, str)
    assert len(qr_code) > 0  # Base64 encoded


def test_verify_totp_success(auth_service):
    """Test TOTP verification success."""
    import pyotp

    secret = auth_service.generate_totp_secret()
    totp = pyotp.TOTP(secret)
    token = totp.now()

    assert auth_service.verify_totp(secret, token) is True


def test_verify_totp_failure(auth_service):
    """Test TOTP verification failure."""
    secret = auth_service.generate_totp_secret()
    invalid_token = "000000"

    assert auth_service.verify_totp(secret, invalid_token) is False


def test_generate_backup_codes(auth_service):
    """Test backup codes generation."""
    codes = auth_service.generate_backup_codes(count=10)

    assert len(codes) == 10
    assert all(isinstance(code, str) for code in codes)
    assert len(set(codes)) == 10  # All unique


@pytest.mark.asyncio
async def test_enable_mfa_totp(auth_service, mock_db, mock_user):
    """Test enabling TOTP MFA."""
    mock_result = AsyncMock()
    mock_result.scalar_one_or_none.return_value = mock_user
    mock_db.execute.return_value = mock_result

    result = await auth_service.enable_mfa(
        mock_db, str(mock_user.id), MFAMethod.TOTP
    )

    assert "qr_code" in result
    assert "secret" in result
    assert "backup_codes" in result
    assert len(result["backup_codes"]) == 10
    assert mock_user.mfa_enabled is True
    assert mock_user.mfa_method == MFAMethod.TOTP
    mock_db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_enable_mfa_user_not_found(auth_service, mock_db):
    """Test enabling MFA for non-existent user."""
    mock_result = AsyncMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_db.execute.return_value = mock_result

    with pytest.raises(HTTPException, match="User not found"):
        await auth_service.enable_mfa(mock_db, "non-existent", MFAMethod.TOTP)


@pytest.mark.asyncio
async def test_verify_mfa_token_success(auth_service, mock_db, mock_user):
    """Test MFA token verification success."""
    import pyotp

    secret = auth_service.generate_totp_secret()
    mock_user.mfa_enabled = True
    mock_user.mfa_method = MFAMethod.TOTP
    mock_user.mfa_secret = secret

    mock_result = AsyncMock()
    mock_result.scalar_one_or_none.return_value = mock_user
    mock_db.execute.return_value = mock_result

    totp = pyotp.TOTP(secret)
    token = totp.now()

    result = await auth_service.verify_mfa_token(mock_db, str(mock_user.id), token)

    assert result is True


@pytest.mark.asyncio
async def test_verify_mfa_token_failure(auth_service, mock_db, mock_user):
    """Test MFA token verification failure."""
    mock_user.mfa_enabled = True
    mock_user.mfa_method = MFAMethod.TOTP
    mock_user.mfa_secret = auth_service.generate_totp_secret()

    mock_result = AsyncMock()
    mock_result.scalar_one_or_none.return_value = mock_user
    mock_db.execute.return_value = mock_result

    result = await auth_service.verify_mfa_token(
        mock_db, str(mock_user.id), "000000"
    )

    assert result is False


# API Key Management Tests

def test_generate_api_key(auth_service):
    """Test API key generation."""
    api_key, key_hash = auth_service.generate_api_key()

    assert api_key.startswith("cronos_")
    assert len(api_key) > 40
    assert len(key_hash) == 64  # SHA256 hex digest


@pytest.mark.asyncio
async def test_create_api_key(auth_service, mock_db):
    """Test API key creation."""
    mock_db.add = Mock()
    mock_db.refresh = AsyncMock()

    result = await auth_service.create_api_key(
        mock_db,
        user_id="user-123",
        name="Test API Key",
        description="For testing",
        expires_days=30,
        permissions=["read", "write"],
    )

    assert "api_key" in result
    assert "key_id" in result
    assert "key_prefix" in result
    assert "expires_at" in result
    assert result["api_key"].startswith("cronos_")
    mock_db.add.assert_called_once()
    mock_db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_revoke_api_key(auth_service, mock_db):
    """Test API key revocation."""
    mock_api_key = Mock(spec=APIKey)
    mock_api_key.id = "key-123"

    mock_result = AsyncMock()
    mock_result.scalar_one_or_none.return_value = mock_api_key
    mock_db.execute.return_value = mock_result

    result = await auth_service.revoke_api_key(
        mock_db, "key-123", "admin-user", reason="Security concern"
    )

    assert result is True
    assert mock_api_key.status == APIKeyStatus.REVOKED
    assert mock_api_key.revoked_at is not None
    assert mock_api_key.revoked_by == "admin-user"
    assert mock_api_key.revoked_reason == "Security concern"
    mock_db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_revoke_api_key_not_found(auth_service, mock_db):
    """Test revoking non-existent API key."""
    mock_result = AsyncMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_db.execute.return_value = mock_result

    with pytest.raises(HTTPException, match="API key not found"):
        await auth_service.revoke_api_key(mock_db, "non-existent", "admin")


@pytest.mark.asyncio
async def test_verify_api_key_success(auth_service, mock_db, mock_user):
    """Test API key verification success."""
    api_key, key_hash = auth_service.generate_api_key()

    mock_api_key = Mock(spec=APIKey)
    mock_api_key.key_hash = key_hash
    mock_api_key.status = APIKeyStatus.ACTIVE
    mock_api_key.expires_at = None
    mock_api_key.last_used_at = None
    mock_api_key.usage_count = 0

    mock_result = AsyncMock()
    mock_result.first.return_value = (mock_api_key, mock_user)
    mock_db.execute.return_value = mock_result

    user = await auth_service.verify_api_key(mock_db, api_key)

    assert user == mock_user
    assert mock_api_key.last_used_at is not None
    assert mock_api_key.usage_count == 1
    mock_db.commit.assert_called()


@pytest.mark.asyncio
async def test_verify_api_key_expired(auth_service, mock_db, mock_user):
    """Test API key verification with expired key."""
    api_key, key_hash = auth_service.generate_api_key()

    mock_api_key = Mock(spec=APIKey)
    mock_api_key.key_hash = key_hash
    mock_api_key.status = APIKeyStatus.ACTIVE
    mock_api_key.expires_at = datetime.utcnow() - timedelta(days=1)
    mock_api_key.last_used_at = None
    mock_api_key.usage_count = 0

    mock_result = AsyncMock()
    mock_result.first.return_value = (mock_api_key, mock_user)
    mock_db.execute.return_value = mock_result

    user = await auth_service.verify_api_key(mock_db, api_key)

    assert user is None
    assert mock_api_key.status == APIKeyStatus.EXPIRED


@pytest.mark.asyncio
async def test_verify_api_key_invalid(auth_service, mock_db):
    """Test API key verification with invalid key."""
    mock_result = AsyncMock()
    mock_result.first.return_value = None
    mock_db.execute.return_value = mock_result

    user = await auth_service.verify_api_key(mock_db, "invalid_key")

    assert user is None


# JWT Token Management Tests

def test_create_access_token(auth_service):
    """Test access token creation."""
    data = {"user_id": "user-123", "username": "testuser"}
    token = auth_service.create_access_token(data)

    assert isinstance(token, str)
    assert len(token) > 0

    # Decode and verify
    payload = jwt.decode(token, auth_service.secret_key, algorithms=[auth_service.algorithm])
    assert payload["user_id"] == "user-123"
    assert payload["type"] == "access"
    assert "exp" in payload


def test_create_refresh_token(auth_service):
    """Test refresh token creation."""
    data = {"user_id": "user-123"}
    token = auth_service.create_refresh_token(data)

    assert isinstance(token, str)

    payload = jwt.decode(token, auth_service.secret_key, algorithms=[auth_service.algorithm])
    assert payload["user_id"] == "user-123"
    assert payload["type"] == "refresh"


@pytest.mark.asyncio
async def test_verify_token_success(auth_service):
    """Test token verification success."""
    data = {"user_id": "user-123"}
    token = auth_service.create_access_token(data)

    payload = await auth_service.verify_token(token)

    assert payload["user_id"] == "user-123"
    assert payload["type"] == "access"


@pytest.mark.asyncio
async def test_verify_token_expired(auth_service):
    """Test token verification with expired token."""
    # Create expired token
    data = {"user_id": "user-123"}
    expire = datetime.utcnow() - timedelta(minutes=10)
    data.update({"exp": expire})

    token = jwt.encode(data, auth_service.secret_key, algorithm=auth_service.algorithm)

    with pytest.raises(HTTPException, match="Token has expired"):
        await auth_service.verify_token(token)


@pytest.mark.asyncio
async def test_verify_token_invalid(auth_service):
    """Test token verification with invalid token."""
    with pytest.raises(HTTPException, match="Invalid token"):
        await auth_service.verify_token("invalid.token.here")


# Session Management Tests

@pytest.mark.asyncio
async def test_create_session(auth_service, mock_db):
    """Test session creation."""
    mock_db.refresh = AsyncMock()

    session = await auth_service.create_session(
        mock_db,
        user_id="user-123",
        ip_address="192.168.1.1",
        user_agent="Mozilla/5.0",
    )

    assert isinstance(session, UserSession)
    assert session.user_id == "user-123"
    assert session.ip_address == "192.168.1.1"
    assert session.session_token is not None
    mock_db.add.assert_called_once()
    mock_db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_revoke_session_success(auth_service, mock_db):
    """Test session revocation success."""
    mock_session = Mock(spec=UserSession)
    mock_session.session_token = "test-token"

    mock_result = AsyncMock()
    mock_result.scalar_one_or_none.return_value = mock_session
    mock_db.execute.return_value = mock_result

    result = await auth_service.revoke_session(mock_db, "test-token")

    assert result is True
    assert mock_session.revoked_at is not None
    mock_db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_revoke_session_not_found(auth_service, mock_db):
    """Test session revocation with non-existent session."""
    mock_result = AsyncMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_db.execute.return_value = mock_result

    result = await auth_service.revoke_session(mock_db, "non-existent")

    assert result is False


# User Authentication Tests

@pytest.mark.asyncio
async def test_authenticate_user_success(auth_service, mock_db, mock_user):
    """Test user authentication success."""
    password = "TestPassword123!"
    mock_user.password_hash = auth_service.hash_password(password)

    mock_result = AsyncMock()
    mock_result.scalar_one_or_none.return_value = mock_user
    mock_db.execute.return_value = mock_result

    user = await auth_service.authenticate_user(
        mock_db, "testuser", password, ip_address="192.168.1.1"
    )

    assert user == mock_user
    assert mock_user.failed_login_attempts == 0
    assert mock_user.last_login is not None
    assert mock_user.last_login_ip == "192.168.1.1"
    mock_db.commit.assert_called()


@pytest.mark.asyncio
async def test_authenticate_user_not_found(auth_service, mock_db):
    """Test authentication with non-existent user."""
    mock_result = AsyncMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_db.execute.return_value = mock_result

    user = await auth_service.authenticate_user(
        mock_db, "nonexistent", "password"
    )

    assert user is None


@pytest.mark.asyncio
async def test_authenticate_user_account_locked(auth_service, mock_db, mock_user):
    """Test authentication with locked account."""
    mock_user.account_locked_until = datetime.utcnow() + timedelta(minutes=30)

    mock_result = AsyncMock()
    mock_result.scalar_one_or_none.return_value = mock_user
    mock_db.execute.return_value = mock_result

    with pytest.raises(HTTPException, match="Account locked"):
        await auth_service.authenticate_user(mock_db, "testuser", "password")


@pytest.mark.asyncio
async def test_authenticate_user_wrong_password(auth_service, mock_db, mock_user):
    """Test authentication with wrong password."""
    mock_user.password_hash = auth_service.hash_password("CorrectPassword123!")

    mock_result = AsyncMock()
    mock_result.scalar_one_or_none.return_value = mock_user
    mock_db.execute.return_value = mock_result

    user = await auth_service.authenticate_user(
        mock_db, "testuser", "WrongPassword123!"
    )

    assert user is None
    assert mock_user.failed_login_attempts == 1
    mock_db.commit.assert_called()


@pytest.mark.asyncio
async def test_authenticate_user_lockout_after_failed_attempts(auth_service, mock_db, mock_user):
    """Test account lockout after multiple failed attempts."""
    mock_user.password_hash = auth_service.hash_password("CorrectPassword123!")
    mock_user.failed_login_attempts = 4  # One more will trigger lockout

    mock_result = AsyncMock()
    mock_result.scalar_one_or_none.return_value = mock_user
    mock_db.execute.return_value = mock_result

    user = await auth_service.authenticate_user(
        mock_db, "testuser", "WrongPassword123!"
    )

    assert user is None
    assert mock_user.failed_login_attempts == 5
    assert mock_user.account_locked_until is not None


@pytest.mark.asyncio
async def test_authenticate_user_with_mfa_required(auth_service, mock_db, mock_user):
    """Test authentication requiring MFA token."""
    import pyotp

    password = "TestPassword123!"
    secret = auth_service.generate_totp_secret()

    mock_user.password_hash = auth_service.hash_password(password)
    mock_user.mfa_enabled = True
    mock_user.mfa_method = MFAMethod.TOTP
    mock_user.mfa_secret = secret

    mock_result = AsyncMock()
    mock_result.scalar_one_or_none.return_value = mock_user
    mock_db.execute.return_value = mock_result

    # Without MFA token
    with pytest.raises(HTTPException, match="MFA token required"):
        await auth_service.authenticate_user(mock_db, "testuser", password)

    # With valid MFA token
    totp = pyotp.TOTP(secret)
    token = totp.now()

    user = await auth_service.authenticate_user(
        mock_db, "testuser", password, mfa_token=token
    )

    assert user == mock_user


@pytest.mark.asyncio
async def test_authenticate_user_with_invalid_mfa(auth_service, mock_db, mock_user):
    """Test authentication with invalid MFA token."""
    password = "TestPassword123!"
    mock_user.password_hash = auth_service.hash_password(password)
    mock_user.mfa_enabled = True
    mock_user.mfa_method = MFAMethod.TOTP
    mock_user.mfa_secret = auth_service.generate_totp_secret()

    mock_result = AsyncMock()
    mock_result.scalar_one_or_none.return_value = mock_user
    mock_db.execute.return_value = mock_result

    user = await auth_service.authenticate_user(
        mock_db, "testuser", password, mfa_token="000000"
    )

    assert user is None


@pytest.mark.asyncio
async def test_authenticate_admin_without_mfa_enforced(auth_service, mock_db, mock_user):
    """Test admin user required to have MFA enabled."""
    password = "TestPassword123!"
    mock_user.password_hash = auth_service.hash_password(password)
    mock_user.role = UserRole.ADMINISTRATOR
    mock_user.mfa_enabled = False
    mock_user.created_at = datetime.utcnow() - timedelta(days=30)  # Past grace period

    mock_result = AsyncMock()
    mock_result.scalar_one_or_none.return_value = mock_user
    mock_db.execute.return_value = mock_result

    with pytest.raises(HTTPException, match="MFA is required"):
        await auth_service.authenticate_user(mock_db, "testuser", password)


# MFA Policy Tests

def test_requires_mfa_for_admin(auth_service, mock_user):
    """Test MFA requirement for admin users."""
    mock_user.role = UserRole.ADMINISTRATOR
    mock_user.created_at = datetime.utcnow() - timedelta(days=30)

    assert auth_service._requires_mfa(mock_user) is True


def test_requires_mfa_grace_period(auth_service, mock_user):
    """Test MFA grace period for new users."""
    mock_user.role = UserRole.ADMINISTRATOR
    mock_user.created_at = datetime.utcnow() - timedelta(days=3)  # Within grace period

    assert auth_service._requires_mfa(mock_user) is False


def test_is_privileged_user(auth_service, mock_user):
    """Test privileged user detection."""
    mock_user.role = UserRole.ADMINISTRATOR
    assert auth_service._is_privileged_user(mock_user) is True

    mock_user.role = UserRole.SECURITY_ADMIN
    assert auth_service._is_privileged_user(mock_user) is True

    mock_user.role = UserRole.COMPLIANCE_OFFICER
    assert auth_service._is_privileged_user(mock_user) is True

    mock_user.role = UserRole.USER
    assert auth_service._is_privileged_user(mock_user) is False


# Audit Logging Tests

@pytest.mark.asyncio
async def test_log_audit(auth_service, mock_db):
    """Test audit logging."""
    await auth_service._log_audit(
        mock_db,
        user_id="user-123",
        action=AuditAction.LOGIN_SUCCESS,
        success=True,
        details={"ip": "192.168.1.1"},
        ip_address="192.168.1.1",
        user_agent="Mozilla/5.0",
    )

    mock_db.add.assert_called_once()
    mock_db.commit.assert_called_once()


# Global Service Instance Tests

@pytest.mark.asyncio
async def test_get_auth_service():
    """Test getting global auth service instance."""
    with patch("ai_engine.api.auth_enterprise.get_config"):
        with patch("ai_engine.api.auth_enterprise.get_audit_logger"):
            service1 = await get_auth_service()
            service2 = await get_auth_service()

            assert service1 is service2  # Same instance


# Integration Tests

@pytest.mark.asyncio
async def test_full_authentication_flow(auth_service, mock_db, mock_user):
    """Test complete authentication flow."""
    # 1. User registration (password setup)
    password = "SecurePassword123!"
    mock_user.password_hash = auth_service.hash_password(password)

    # 2. Enable MFA
    mock_result = AsyncMock()
    mock_result.scalar_one_or_none.return_value = mock_user
    mock_db.execute.return_value = mock_result

    mfa_result = await auth_service.enable_mfa(
        mock_db, str(mock_user.id), MFAMethod.TOTP
    )

    secret = mfa_result["secret"]

    # 3. Authenticate with password and MFA
    import pyotp

    totp = pyotp.TOTP(secret)
    token = totp.now()

    user = await auth_service.authenticate_user(
        mock_db, mock_user.username, password, mfa_token=token
    )

    assert user == mock_user

    # 4. Create session
    session = await auth_service.create_session(mock_db, str(user.id))
    assert session is not None

    # 5. Create API key
    api_key_result = await auth_service.create_api_key(
        mock_db, str(user.id), "Test Key"
    )
    assert "api_key" in api_key_result


@pytest.mark.asyncio
async def test_api_key_lifecycle(auth_service, mock_db, mock_user):
    """Test complete API key lifecycle."""
    # 1. Create API key
    mock_db.refresh = AsyncMock()
    api_key_result = await auth_service.create_api_key(
        mock_db,
        user_id=str(mock_user.id),
        name="Test Key",
        expires_days=30,
    )

    api_key = api_key_result["api_key"]
    key_id = api_key_result["key_id"]

    # 2. Verify API key
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    mock_api_key = Mock(spec=APIKey)
    mock_api_key.key_hash = key_hash
    mock_api_key.status = APIKeyStatus.ACTIVE
    mock_api_key.expires_at = None
    mock_api_key.last_used_at = None
    mock_api_key.usage_count = 0

    mock_result = AsyncMock()
    mock_result.first.return_value = (mock_api_key, mock_user)
    mock_result.scalar_one_or_none.return_value = mock_api_key
    mock_db.execute.return_value = mock_result

    verified_user = await auth_service.verify_api_key(mock_db, api_key)
    assert verified_user == mock_user

    # 3. Revoke API key
    revoke_result = await auth_service.revoke_api_key(
        mock_db, key_id, "admin-user", reason="Test complete"
    )
    assert revoke_result is True
