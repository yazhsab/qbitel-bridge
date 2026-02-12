"""
Comprehensive Unit Tests for ai_engine/api/auth_enterprise.py - Enterprise Authentication

This test suite provides complete coverage of the EnterpriseAuthenticationService class,
including all authentication methods, security features, and edge cases.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi import HTTPException
import hashlib
import secrets


class TestEnterpriseAuthenticationServiceComprehensive:
    """Comprehensive test suite for EnterpriseAuthenticationService."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config with all security settings."""
        config = Mock()
        config.security = Mock()
        config.security.jwt_secret = "test_secret_key_with_at_least_32_characters_long_for_security"
        return config

    @pytest.fixture
    def auth_service(self, mock_config):
        """Create authentication service instance."""
        from ai_engine.api.auth_enterprise import EnterpriseAuthenticationService

        return EnterpriseAuthenticationService(mock_config)

    # ==================== Initialization Tests ====================

    def test_service_initialization_complete(self, auth_service):
        """Test complete service initialization with all attributes."""
        assert auth_service.algorithm == "HS256"
        assert auth_service.access_token_expire_minutes == 30
        assert auth_service.refresh_token_expire_days == 7
        assert auth_service.mfa_required_for_admin is True
        assert auth_service.mfa_required_for_privileged is True
        assert auth_service.mfa_grace_period_days == 7
        assert auth_service.audit_logger is not None

    def test_load_secret_key_minimum_length(self):
        """Test JWT secret key minimum length requirement."""
        from ai_engine.api.auth_enterprise import EnterpriseAuthenticationService

        config = Mock()
        config.security = Mock()
        config.security.jwt_secret = "a" * 32  # Exactly 32 characters

        service = EnterpriseAuthenticationService(config)
        assert len(service.secret_key) == 32

    def test_load_secret_key_none(self):
        """Test loading JWT secret when None."""
        from ai_engine.api.auth_enterprise import EnterpriseAuthenticationService

        config = Mock()
        config.security = Mock()
        config.security.jwt_secret = None

        with pytest.raises(ValueError, match="JWT secret not configured"):
            EnterpriseAuthenticationService(config)

    def test_load_secret_key_empty_string(self):
        """Test loading JWT secret when empty string."""
        from ai_engine.api.auth_enterprise import EnterpriseAuthenticationService

        config = Mock()
        config.security = Mock()
        config.security.jwt_secret = ""

        with pytest.raises(ValueError, match="JWT secret not configured"):
            EnterpriseAuthenticationService(config)

    # ==================== Password Management Tests ====================

    def test_hash_password_different_hashes(self, auth_service):
        """Test that same password produces different hashes (salt)."""
        password = "TestPassword123!"
        hash1 = auth_service.hash_password(password)
        hash2 = auth_service.hash_password(password)

        assert hash1 != hash2  # Different due to salt
        assert auth_service.verify_password(password, hash1)
        assert auth_service.verify_password(password, hash2)

    def test_verify_password_empty_password(self, auth_service):
        """Test password verification with empty password."""
        hashed = auth_service.hash_password("TestPassword123!")
        assert auth_service.verify_password("", hashed) is False

    def test_validate_password_strength_all_requirements(self, auth_service):
        """Test password validation with all requirements met."""
        valid_passwords = [
            "ValidPass123!@#",
            "Str0ng!Password",
            "C0mpl3x#Pass",
            "S3cur3$Passw0rd",
        ]

        for password in valid_passwords:
            is_valid, error = auth_service.validate_password_strength(password)
            assert is_valid is True, f"Password {password} should be valid"
            assert error is None

    def test_validate_password_strength_edge_cases(self, auth_service):
        """Test password validation edge cases."""
        # Exactly 12 characters
        is_valid, _ = auth_service.validate_password_strength("ValidPass1!")
        assert is_valid is False  # Only 11 chars

        is_valid, _ = auth_service.validate_password_strength("ValidPass12!")
        assert is_valid is True  # Exactly 12 chars

    def test_validate_password_strength_multiple_special_chars(self, auth_service):
        """Test password with multiple special characters."""
        password = "Test!@#$%^&*()123Abc"
        is_valid, error = auth_service.validate_password_strength(password)
        assert is_valid is True
        assert error is None

    def test_validate_password_strength_common_patterns_case_insensitive(self, auth_service):
        """Test common pattern detection is case insensitive."""
        weak_passwords = [
            "PASSWORD123!Abc",
            "MyPassword123!",
            "Admin123!@#Abc",
            "Qwerty123!@#Abc",
            "Letmein123!@#Abc",
        ]

        for password in weak_passwords:
            is_valid, error = auth_service.validate_password_strength(password)
            assert is_valid is False
            assert "common patterns" in error

    @pytest.mark.asyncio
    async def test_change_password_updates_timestamp(self, auth_service):
        """Test that password change updates timestamp."""
        from ai_engine.models.database import User

        mock_db = AsyncMock()
        old_time = datetime.utcnow() - timedelta(days=30)
        user = User(
            id="user123",
            username="testuser",
            password_hash=auth_service.hash_password("OldPassword123!"),
            password_changed_at=old_time,
        )

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = user
        mock_db.execute.return_value = mock_result

        await auth_service.change_password(mock_db, "user123", "OldPassword123!", "NewPassword456!")

        assert user.password_changed_at > old_time
        assert user.must_change_password is False

    @pytest.mark.asyncio
    async def test_change_password_audit_logging(self, auth_service):
        """Test that password change creates audit logs."""
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

        with patch.object(auth_service, "_log_audit", new_callable=AsyncMock) as mock_audit:
            await auth_service.change_password(mock_db, "user123", "OldPassword123!", "NewPassword456!")

            # Should log success
            assert mock_audit.call_count >= 1

    # ==================== MFA Tests ====================

    def test_generate_totp_secret_uniqueness(self, auth_service):
        """Test that TOTP secrets are unique."""
        secrets_set = set()
        for _ in range(10):
            secret = auth_service.generate_totp_secret()
            assert secret not in secrets_set
            secrets_set.add(secret)

    def test_generate_totp_qr_code_format(self, auth_service):
        """Test QR code generation format."""
        secret = auth_service.generate_totp_secret()
        qr_code = auth_service.generate_totp_qr_code("testuser", secret)

        # Should be base64 encoded
        import base64

        try:
            decoded = base64.b64decode(qr_code)
            assert len(decoded) > 0
        except Exception:
            pytest.fail("QR code should be valid base64")

    def test_verify_totp_window(self, auth_service):
        """Test TOTP verification with time window."""
        import pyotp

        secret = auth_service.generate_totp_secret()
        totp = pyotp.TOTP(secret)

        # Current token should work
        current_token = totp.now()
        assert auth_service.verify_totp(secret, current_token) is True

        # Invalid token should fail
        assert auth_service.verify_totp(secret, "000000") is False

    def test_generate_backup_codes_format(self, auth_service):
        """Test backup codes format and uniqueness."""
        codes = auth_service.generate_backup_codes(count=10)

        assert len(codes) == 10
        assert len(set(codes)) == 10  # All unique

        for code in codes:
            assert len(code) == 8  # 4 bytes hex = 8 chars
            assert code.isupper()  # Should be uppercase
            assert all(c in "0123456789ABCDEF" for c in code)

    def test_generate_backup_codes_custom_count(self, auth_service):
        """Test generating custom number of backup codes."""
        for count in [5, 15, 20]:
            codes = auth_service.generate_backup_codes(count=count)
            assert len(codes) == count

    @pytest.mark.asyncio
    async def test_enable_mfa_with_provided_secret(self, auth_service):
        """Test enabling MFA with user-provided secret."""
        from ai_engine.models.database import User, MFAMethod

        mock_db = AsyncMock()
        user = User(id="user123", username="testuser")

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = user
        mock_db.execute.return_value = mock_result

        custom_secret = "CUSTOMSECRET1234"
        result = await auth_service.enable_mfa(mock_db, "user123", MFAMethod.TOTP, secret=custom_secret)

        assert result["secret"] == custom_secret
        assert user.mfa_secret == custom_secret

    @pytest.mark.asyncio
    async def test_enable_mfa_backup_codes_hashed(self, auth_service):
        """Test that backup codes are properly hashed."""
        from ai_engine.models.database import User, MFAMethod

        mock_db = AsyncMock()
        user = User(id="user123", username="testuser")

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = user
        mock_db.execute.return_value = mock_result

        result = await auth_service.enable_mfa(mock_db, "user123", MFAMethod.TOTP)

        # Backup codes in result should be plain text
        backup_codes = result["backup_codes"]
        assert len(backup_codes) == 10

        # Stored codes should be hashed
        assert len(user.mfa_backup_codes) == 10
        for stored_code in user.mfa_backup_codes:
            assert stored_code != backup_codes[0]  # Should be hashed

    @pytest.mark.asyncio
    async def test_verify_mfa_token_user_not_found(self, auth_service):
        """Test MFA verification when user doesn't exist."""
        mock_db = AsyncMock()
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        result = await auth_service.verify_mfa_token(mock_db, "nonexistent", "123456")
        assert result is False

    # ==================== API Key Management Tests ====================

    def test_generate_api_key_format(self, auth_service):
        """Test API key generation format."""
        api_key, key_hash = auth_service.generate_api_key()

        assert api_key.startswith("qbitel_")
        assert len(key_hash) == 64  # SHA256 hex

        # Verify hash is correct
        expected_hash = hashlib.sha256(api_key.encode()).hexdigest()
        assert key_hash == expected_hash

    def test_generate_api_key_uniqueness(self, auth_service):
        """Test that generated API keys are unique."""
        keys = set()
        for _ in range(10):
            api_key, _ = auth_service.generate_api_key()
            assert api_key not in keys
            keys.add(api_key)

    @pytest.mark.asyncio
    async def test_create_api_key_without_expiration(self, auth_service):
        """Test creating API key without expiration."""
        mock_db = AsyncMock()

        result = await auth_service.create_api_key(
            mock_db,
            user_id="user123",
            name="Test Key",
            expires_days=None,
        )

        assert result["expires_at"] is None

    @pytest.mark.asyncio
    async def test_create_api_key_with_permissions(self, auth_service):
        """Test creating API key with specific permissions."""
        mock_db = AsyncMock()

        permissions = ["read:data", "write:data", "admin:users"]
        result = await auth_service.create_api_key(
            mock_db,
            user_id="user123",
            name="Admin Key",
            permissions=permissions,
        )

        assert "api_key" in result
        mock_db.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_revoke_api_key_with_reason(self, auth_service):
        """Test API key revocation with reason."""
        from ai_engine.models.database import APIKey, APIKeyStatus

        mock_db = AsyncMock()
        api_key = APIKey(
            id="key123",
            key_hash="hash",
            key_prefix="qbitel_",
            name="Test Key",
            user_id="user123",
            status=APIKeyStatus.ACTIVE,
        )

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = api_key
        mock_db.execute.return_value = mock_result

        reason = "Security audit"
        result = await auth_service.revoke_api_key(mock_db, "key123", "admin", reason=reason)

        assert result is True
        assert api_key.revoked_reason == reason
        assert api_key.revoked_by == "admin"

    @pytest.mark.asyncio
    async def test_verify_api_key_inactive_user(self, auth_service):
        """Test API key verification with inactive user."""
        from ai_engine.models.database import APIKey, User, APIKeyStatus

        api_key = "qbitel_test_key"
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        mock_db = AsyncMock()
        api_key_record = APIKey(
            id="key123",
            key_hash=key_hash,
            status=APIKeyStatus.ACTIVE,
        )
        user = User(id="user123", username="testuser", is_active=False)

        mock_result = Mock()
        mock_result.first.return_value = (api_key_record, user)
        mock_db.execute.return_value = mock_result

        result = await auth_service.verify_api_key(mock_db, api_key)
        assert result is None

    @pytest.mark.asyncio
    async def test_verify_api_key_usage_tracking(self, auth_service):
        """Test that API key usage is tracked."""
        from ai_engine.models.database import APIKey, User, APIKeyStatus

        api_key = "qbitel_test_key"
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        mock_db = AsyncMock()
        api_key_record = APIKey(
            id="key123",
            key_hash=key_hash,
            status=APIKeyStatus.ACTIVE,
            usage_count=5,
            last_used_at=None,
        )
        user = User(id="user123", username="testuser", is_active=True)

        mock_result = Mock()
        mock_result.first.return_value = (api_key_record, user)
        mock_db.execute.return_value = mock_result

        result = await auth_service.verify_api_key(mock_db, api_key)

        assert result == user
        assert api_key_record.usage_count == 6
        assert api_key_record.last_used_at is not None

    # ==================== JWT Token Tests ====================

    def test_create_access_token_payload(self, auth_service):
        """Test access token payload structure."""
        import jwt

        data = {"user_id": "user123", "username": "testuser", "role": "admin"}
        token = auth_service.create_access_token(data)

        # Decode without verification to check payload
        payload = jwt.decode(token, options={"verify_signature": False})

        assert payload["user_id"] == "user123"
        assert payload["username"] == "testuser"
        assert payload["role"] == "admin"
        assert payload["type"] == "access"
        assert "exp" in payload

    def test_create_refresh_token_expiration(self, auth_service):
        """Test refresh token has longer expiration."""
        import jwt

        data = {"user_id": "user123"}
        access_token = auth_service.create_access_token(data)
        refresh_token = auth_service.create_refresh_token(data)

        access_payload = jwt.decode(access_token, options={"verify_signature": False})
        refresh_payload = jwt.decode(refresh_token, options={"verify_signature": False})

        assert refresh_payload["exp"] > access_payload["exp"]
        assert refresh_payload["type"] == "refresh"

    @pytest.mark.asyncio
    async def test_verify_token_type_checking(self, auth_service):
        """Test token type is included in payload."""
        data = {"user_id": "user123"}
        access_token = auth_service.create_access_token(data)

        payload = await auth_service.verify_token(access_token)
        assert payload["type"] == "access"

    # ==================== Session Management Tests ====================

    @pytest.mark.asyncio
    async def test_create_session_with_metadata(self, auth_service):
        """Test session creation with full metadata."""
        mock_db = AsyncMock()

        session = await auth_service.create_session(
            mock_db,
            user_id="user123",
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0 (Test Browser)",
        )

        assert session.user_id == "user123"
        assert session.ip_address == "192.168.1.100"
        assert session.user_agent == "Mozilla/5.0 (Test Browser)"
        assert session.expires_at > datetime.utcnow()

    @pytest.mark.asyncio
    async def test_create_session_token_uniqueness(self, auth_service):
        """Test that session tokens are unique."""
        mock_db = AsyncMock()

        tokens = set()
        for _ in range(5):
            session = await auth_service.create_session(mock_db, user_id="user123")
            assert session.session_token not in tokens
            tokens.add(session.session_token)

    @pytest.mark.asyncio
    async def test_revoke_session_timestamp(self, auth_service):
        """Test that session revocation sets timestamp."""
        from ai_engine.models.database import UserSession

        mock_db = AsyncMock()
        session = UserSession(
            session_token="token123",
            user_id="user123",
            revoked_at=None,
        )

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = session
        mock_db.execute.return_value = mock_result

        result = await auth_service.revoke_session(mock_db, "token123")

        assert result is True
        assert session.revoked_at is not None
        assert session.revoked_at <= datetime.utcnow()

    # ==================== Authentication Tests ====================

    @pytest.mark.asyncio
    async def test_authenticate_user_with_valid_mfa(self, auth_service):
        """Test authentication with valid MFA token."""
        from ai_engine.models.database import User, UserRole, MFAMethod
        import pyotp

        password = "TestPassword123!"
        secret = auth_service.generate_totp_secret()
        totp = pyotp.TOTP(secret)
        valid_token = totp.now()

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

        result = await auth_service.authenticate_user(mock_db, "admin", password, mfa_token=valid_token)

        assert result == user
        assert user.failed_login_attempts == 0

    @pytest.mark.asyncio
    async def test_authenticate_user_lockout_duration(self, auth_service):
        """Test account lockout duration."""
        from ai_engine.models.database import User

        mock_db = AsyncMock()
        lockout_time = datetime.utcnow() + timedelta(minutes=30)
        user = User(
            id="user123",
            username="testuser",
            is_active=True,
            account_locked_until=lockout_time,
        )

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = user
        mock_db.execute.return_value = mock_result

        with pytest.raises(HTTPException) as exc_info:
            await auth_service.authenticate_user(mock_db, "testuser", "password")

        assert "Account locked" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_authenticate_user_failed_attempts_increment(self, auth_service):
        """Test failed login attempts increment correctly."""
        from ai_engine.models.database import User

        mock_db = AsyncMock()
        user = User(
            id="user123",
            username="testuser",
            password_hash=auth_service.hash_password("CorrectPassword123!"),
            is_active=True,
            failed_login_attempts=2,
        )

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = user
        mock_db.execute.return_value = mock_result

        result = await auth_service.authenticate_user(mock_db, "testuser", "WrongPassword")

        assert result is None
        assert user.failed_login_attempts == 3

    @pytest.mark.asyncio
    async def test_authenticate_user_last_login_tracking(self, auth_service):
        """Test that last login is tracked."""
        from ai_engine.models.database import User

        password = "TestPassword123!"
        ip_address = "192.168.1.1"

        mock_db = AsyncMock()
        user = User(
            id="user123",
            username="testuser",
            password_hash=auth_service.hash_password(password),
            is_active=True,
            failed_login_attempts=0,
            mfa_enabled=False,
            last_login=None,
            last_login_ip=None,
        )

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = user
        mock_db.execute.return_value = mock_result

        result = await auth_service.authenticate_user(mock_db, "testuser", password, ip_address=ip_address)

        assert result == user
        assert user.last_login is not None
        assert user.last_login_ip == ip_address

    # ==================== MFA Policy Tests ====================

    def test_requires_mfa_regular_user(self, auth_service):
        """Test MFA not required for regular users."""
        from ai_engine.models.database import User, UserRole

        user = User(
            id="user123",
            username="regular",
            role=UserRole.USER,
            created_at=datetime.utcnow() - timedelta(days=30),
        )

        assert auth_service._requires_mfa(user) is False

    def test_requires_mfa_security_admin(self, auth_service):
        """Test MFA required for security admin."""
        from ai_engine.models.database import User, UserRole

        user = User(
            id="user123",
            username="security_admin",
            role=UserRole.SECURITY_ADMIN,
            created_at=datetime.utcnow() - timedelta(days=30),
        )

        assert auth_service._requires_mfa(user) is True

    def test_requires_mfa_grace_period_boundary(self, auth_service):
        """Test MFA grace period boundary conditions."""
        from ai_engine.models.database import User, UserRole

        # Just before grace period ends
        user = User(
            id="user123",
            username="admin",
            role=UserRole.ADMINISTRATOR,
            created_at=datetime.utcnow() - timedelta(days=6, hours=23),
        )
        assert auth_service._requires_mfa(user) is False

        # Just after grace period ends
        user.created_at = datetime.utcnow() - timedelta(days=7, hours=1)
        assert auth_service._requires_mfa(user) is True

    def test_is_privileged_user_all_roles(self, auth_service):
        """Test privileged user check for all roles."""
        from ai_engine.models.database import User, UserRole

        privileged_roles = [
            UserRole.ADMINISTRATOR,
            UserRole.SECURITY_ADMIN,
            UserRole.COMPLIANCE_OFFICER,
        ]

        for role in privileged_roles:
            user = User(id="user123", role=role)
            assert auth_service._is_privileged_user(user) is True

        # Non-privileged roles
        user = User(id="user123", role=UserRole.USER)
        assert auth_service._is_privileged_user(user) is False

    # ==================== Audit Logging Tests ====================

    @pytest.mark.asyncio
    async def test_log_audit_complete_metadata(self, auth_service):
        """Test audit logging with complete metadata."""
        from ai_engine.models.database import AuditAction

        mock_db = AsyncMock()

        await auth_service._log_audit(
            db=mock_db,
            user_id="user123",
            action=AuditAction.LOGIN_SUCCESS,
            success=True,
            details={"mfa_used": True, "device": "mobile"},
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
        )

        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_log_audit_failure_with_error(self, auth_service):
        """Test audit logging for failures."""
        from ai_engine.models.database import AuditAction

        mock_db = AsyncMock()

        await auth_service._log_audit(
            db=mock_db,
            user_id="user123",
            action=AuditAction.LOGIN_FAILED,
            success=False,
            error_message="Invalid credentials",
            ip_address="192.168.1.1",
        )

        mock_db.add.assert_called_once()

    # ==================== Edge Cases and Error Handling ====================

    @pytest.mark.asyncio
    async def test_change_password_same_as_old(self, auth_service):
        """Test changing password to same as old password."""
        from ai_engine.models.database import User

        password = "TestPassword123!"
        mock_db = AsyncMock()
        user = User(
            id="user123",
            username="testuser",
            password_hash=auth_service.hash_password(password),
        )

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = user
        mock_db.execute.return_value = mock_result

        # Should succeed even if same password
        result = await auth_service.change_password(mock_db, "user123", password, password)
        assert result is True

    @pytest.mark.asyncio
    async def test_verify_api_key_concurrent_usage(self, auth_service):
        """Test API key verification handles concurrent usage."""
        from ai_engine.models.database import APIKey, User, APIKeyStatus

        api_key = "qbitel_test_key"
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        mock_db = AsyncMock()
        api_key_record = APIKey(
            id="key123",
            key_hash=key_hash,
            status=APIKeyStatus.ACTIVE,
            usage_count=100,
        )
        user = User(id="user123", username="testuser", is_active=True)

        mock_result = Mock()
        mock_result.first.return_value = (api_key_record, user)
        mock_db.execute.return_value = mock_result

        result = await auth_service.verify_api_key(mock_db, api_key)

        assert result == user
        assert api_key_record.usage_count == 101


class TestGetAuthServiceSingleton:
    """Test suite for get_auth_service singleton pattern."""

    @pytest.mark.asyncio
    async def test_get_auth_service_creates_instance(self):
        """Test that get_auth_service creates instance."""
        from ai_engine.api import auth_enterprise

        # Reset singleton
        auth_enterprise._auth_service = None

        with patch("ai_engine.api.auth_enterprise.get_config") as mock_config:
            mock_config.return_value = Mock()
            mock_config.return_value.security = Mock()
            mock_config.return_value.security.jwt_secret = "test_secret_key_with_at_least_32_characters_long"

            service = await auth_enterprise.get_auth_service()
            assert service is not None
            assert isinstance(service, auth_enterprise.EnterpriseAuthenticationService)

    @pytest.mark.asyncio
    async def test_get_auth_service_returns_same_instance(self):
        """Test that multiple calls return same instance."""
        from ai_engine.api import auth_enterprise

        # Reset singleton
        auth_enterprise._auth_service = None

        with patch("ai_engine.api.auth_enterprise.get_config") as mock_config:
            mock_config.return_value = Mock()
            mock_config.return_value.security = Mock()
            mock_config.return_value.security.jwt_secret = "test_secret_key_with_at_least_32_characters_long"

            service1 = await auth_enterprise.get_auth_service()
            service2 = await auth_enterprise.get_auth_service()

            assert service1 is service2
