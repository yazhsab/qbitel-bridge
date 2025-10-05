"""
Tests for database models.
Covers User, APIKey, UserSession, and AuditLog models.
"""

import pytest
import uuid
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ai_engine.models.database import (
    Base,
    User,
    UserRole,
    MFAMethod,
    APIKey,
    APIKeyStatus,
    AuditAction,
)


class TestUserRole:
    """Test UserRole enumeration."""

    def test_user_role_values(self):
        """Test user role enum values."""
        assert UserRole.ADMINISTRATOR.value == "administrator"
        assert UserRole.SECURITY_ANALYST.value == "security_analyst"
        assert UserRole.OPERATOR.value == "operator"
        assert UserRole.VIEWER.value == "viewer"
        assert UserRole.API_USER.value == "api_user"


class TestMFAMethod:
    """Test MFAMethod enumeration."""

    def test_mfa_method_values(self):
        """Test MFA method enum values."""
        assert MFAMethod.TOTP.value == "totp"
        assert MFAMethod.SMS.value == "sms"
        assert MFAMethod.EMAIL.value == "email"
        assert MFAMethod.HARDWARE_TOKEN.value == "hardware_token"


class TestAPIKeyStatus:
    """Test APIKeyStatus enumeration."""

    def test_api_key_status_values(self):
        """Test API key status enum values."""
        assert APIKeyStatus.ACTIVE.value == "active"
        assert APIKeyStatus.REVOKED.value == "revoked"
        assert APIKeyStatus.EXPIRED.value == "expired"


class TestAuditAction:
    """Test AuditAction enumeration."""

    def test_audit_action_values(self):
        """Test audit action enum values."""
        assert AuditAction.LOGIN.value == "login"
        assert AuditAction.LOGOUT.value == "logout"
        assert AuditAction.LOGIN_FAILED.value == "login_failed"
        assert AuditAction.PASSWORD_CHANGE.value == "password_change"
        assert AuditAction.MFA_ENABLED.value == "mfa_enabled"
        assert AuditAction.API_KEY_CREATED.value == "api_key_created"


class TestUserModel:
    """Test User model."""

    @pytest.fixture
    def db_session(self):
        """Create in-memory database session."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        yield session
        session.close()

    def test_user_creation(self, db_session):
        """Test creating a user."""
        user = User(
            username="testuser",
            email="test@example.com",
            full_name="Test User",
            password_hash="hashed_password",
            role=UserRole.VIEWER
        )

        db_session.add(user)
        db_session.commit()

        assert user.id is not None
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.role == UserRole.VIEWER
        assert user.is_active is True
        assert user.is_verified is False

    def test_user_defaults(self, db_session):
        """Test user default values."""
        user = User(
            username="defaultuser",
            email="default@example.com",
            full_name="Default User"
        )

        db_session.add(user)
        db_session.commit()

        assert user.is_active is True
        assert user.is_verified is False
        assert user.role == UserRole.VIEWER
        assert user.mfa_enabled is False
        assert user.failed_login_attempts == 0
        assert user.must_change_password is False

    def test_user_with_mfa(self, db_session):
        """Test user with MFA enabled."""
        user = User(
            username="mfauser",
            email="mfa@example.com",
            full_name="MFA User",
            mfa_enabled=True,
            mfa_method=MFAMethod.TOTP,
            mfa_secret="encrypted_secret"
        )

        db_session.add(user)
        db_session.commit()

        assert user.mfa_enabled is True
        assert user.mfa_method == MFAMethod.TOTP
        assert user.mfa_secret == "encrypted_secret"

    def test_user_oauth(self, db_session):
        """Test user with OAuth authentication."""
        user = User(
            username="oauthuser",
            email="oauth@example.com",
            full_name="OAuth User",
            oauth_provider="google",
            oauth_id="google_12345",
            password_hash=None  # OAuth users don't have passwords
        )

        db_session.add(user)
        db_session.commit()

        assert user.oauth_provider == "google"
        assert user.oauth_id == "google_12345"
        assert user.password_hash is None

    def test_user_security_tracking(self, db_session):
        """Test user security tracking fields."""
        now = datetime.utcnow()
        user = User(
            username="secureuser",
            email="secure@example.com",
            full_name="Secure User",
            last_login=now,
            last_login_ip="192.168.1.1",
            failed_login_attempts=2,
            account_locked_until=now + timedelta(hours=1)
        )

        db_session.add(user)
        db_session.commit()

        assert user.last_login is not None
        assert user.last_login_ip == "192.168.1.1"
        assert user.failed_login_attempts == 2
        assert user.account_locked_until is not None

    def test_user_soft_delete(self, db_session):
        """Test user soft delete."""
        user = User(
            username="deleteduser",
            email="deleted@example.com",
            full_name="Deleted User",
            deleted_at=datetime.utcnow()
        )

        db_session.add(user)
        db_session.commit()

        assert user.deleted_at is not None

    def test_user_repr(self, db_session):
        """Test user string representation."""
        user = User(
            username="repruser",
            email="repr@example.com",
            full_name="Repr User",
            role=UserRole.ADMINISTRATOR
        )

        db_session.add(user)
        db_session.commit()

        repr_str = repr(user)
        assert "User" in repr_str
        assert "repruser" in repr_str
        assert "administrator" in repr_str


class TestAPIKeyModel:
    """Test APIKey model."""

    @pytest.fixture
    def db_session(self):
        """Create in-memory database session."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        yield session
        session.close()

    @pytest.fixture
    def test_user(self, db_session):
        """Create a test user."""
        user = User(
            username="keyuser",
            email="key@example.com",
            full_name="Key User"
        )
        db_session.add(user)
        db_session.commit()
        return user

    def test_api_key_creation(self, db_session, test_user):
        """Test creating an API key."""
        api_key = APIKey(
            key_hash="sha256_hash_value",
            key_prefix="cronos_",
            name="Production Key",
            description="Key for production use",
            user_id=test_user.id,
            status=APIKeyStatus.ACTIVE
        )

        db_session.add(api_key)
        db_session.commit()

        assert api_key.id is not None
        assert api_key.key_hash == "sha256_hash_value"
        assert api_key.key_prefix == "cronos_"
        assert api_key.name == "Production Key"
        assert api_key.status == APIKeyStatus.ACTIVE
        assert api_key.user_id == test_user.id

    def test_api_key_defaults(self, db_session, test_user):
        """Test API key default values."""
        api_key = APIKey(
            key_hash="hash",
            key_prefix="key_",
            name="Default Key",
            user_id=test_user.id
        )

        db_session.add(api_key)
        db_session.commit()

        assert api_key.status == APIKeyStatus.ACTIVE
        assert api_key.usage_count == 0
        assert api_key.rate_limit_per_minute == 100
        assert api_key.permissions == []

    def test_api_key_with_expiration(self, db_session, test_user):
        """Test API key with expiration."""
        expires_at = datetime.utcnow() + timedelta(days=30)
        api_key = APIKey(
            key_hash="expiring_hash",
            key_prefix="exp_",
            name="Expiring Key",
            user_id=test_user.id,
            expires_at=expires_at
        )

        db_session.add(api_key)
        db_session.commit()

        assert api_key.expires_at == expires_at

    def test_api_key_usage_tracking(self, db_session, test_user):
        """Test API key usage tracking."""
        now = datetime.utcnow()
        api_key = APIKey(
            key_hash="used_hash",
            key_prefix="used_",
            name="Used Key",
            user_id=test_user.id,
            last_used_at=now,
            last_used_ip="10.0.0.1",
            usage_count=42
        )

        db_session.add(api_key)
        db_session.commit()

        assert api_key.last_used_at is not None
        assert api_key.last_used_ip == "10.0.0.1"
        assert api_key.usage_count == 42

    def test_api_key_revoked(self, db_session, test_user):
        """Test revoked API key."""
        api_key = APIKey(
            key_hash="revoked_hash",
            key_prefix="rev_",
            name="Revoked Key",
            user_id=test_user.id,
            status=APIKeyStatus.REVOKED
        )

        db_session.add(api_key)
        db_session.commit()

        assert api_key.status == APIKeyStatus.REVOKED

    def test_api_key_with_permissions(self, db_session, test_user):
        """Test API key with custom permissions."""
        permissions = ["read:data", "write:data", "delete:data"]
        api_key = APIKey(
            key_hash="perm_hash",
            key_prefix="perm_",
            name="Permission Key",
            user_id=test_user.id,
            permissions=permissions
        )

        db_session.add(api_key)
        db_session.commit()

        assert api_key.permissions == permissions

    def test_api_key_rate_limiting(self, db_session, test_user):
        """Test API key with custom rate limit."""
        api_key = APIKey(
            key_hash="rate_hash",
            key_prefix="rate_",
            name="Rate Limited Key",
            user_id=test_user.id,
            rate_limit_per_minute=500
        )

        db_session.add(api_key)
        db_session.commit()

        assert api_key.rate_limit_per_minute == 500


class TestUserAPIKeyRelationship:
    """Test User-APIKey relationship."""

    @pytest.fixture
    def db_session(self):
        """Create in-memory database session."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        yield session
        session.close()

    def test_user_api_keys_relationship(self, db_session):
        """Test user can have multiple API keys."""
        user = User(
            username="multikey",
            email="multikey@example.com",
            full_name="Multi Key User"
        )
        db_session.add(user)
        db_session.commit()

        # Add multiple API keys
        for i in range(3):
            api_key = APIKey(
                key_hash=f"hash_{i}",
                key_prefix=f"key_{i}_",
                name=f"Key {i}",
                user_id=user.id
            )
            db_session.add(api_key)

        db_session.commit()

        # Query user with keys
        user = db_session.query(User).filter_by(username="multikey").first()
        assert len(user.api_keys) == 3

    def test_cascade_delete_api_keys(self, db_session):
        """Test API keys are deleted when user is deleted."""
        user = User(
            username="cascadeuser",
            email="cascade@example.com",
            full_name="Cascade User"
        )
        db_session.add(user)
        db_session.commit()

        # Add API key
        api_key = APIKey(
            key_hash="cascade_hash",
            key_prefix="cascade_",
            name="Cascade Key",
            user_id=user.id
        )
        db_session.add(api_key)
        db_session.commit()

        # Delete user
        user_id = user.id
        db_session.delete(user)
        db_session.commit()

        # API key should be deleted
        remaining_keys = db_session.query(APIKey).filter_by(user_id=user_id).all()
        assert len(remaining_keys) == 0


class TestModelConstraints:
    """Test database model constraints."""

    @pytest.fixture
    def db_session(self):
        """Create in-memory database session."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        yield session
        session.close()

    def test_user_unique_username(self, db_session):
        """Test username must be unique."""
        user1 = User(
            username="uniqueuser",
            email="user1@example.com",
            full_name="User 1"
        )
        db_session.add(user1)
        db_session.commit()

        user2 = User(
            username="uniqueuser",  # Duplicate
            email="user2@example.com",
            full_name="User 2"
        )
        db_session.add(user2)

        with pytest.raises(Exception):  # Integrity error
            db_session.commit()

    def test_user_unique_email(self, db_session):
        """Test email must be unique."""
        user1 = User(
            username="user1",
            email="unique@example.com",
            full_name="User 1"
        )
        db_session.add(user1)
        db_session.commit()

        user2 = User(
            username="user2",
            email="unique@example.com",  # Duplicate
            full_name="User 2"
        )
        db_session.add(user2)

        with pytest.raises(Exception):  # Integrity error
            db_session.commit()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
