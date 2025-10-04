"""
CRONOS AI Engine - Database Models

Production-ready database models for user management, authentication, and audit logging.
"""

import uuid
from datetime import datetime, timedelta
from typing import Optional, List
from enum import Enum as PyEnum

from sqlalchemy import (
    Column,
    String,
    Integer,
    Boolean,
    DateTime,
    Text,
    JSON,
    ForeignKey,
    Index,
    UniqueConstraint,
    Enum,
    LargeBinary,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, INET
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class UserRole(str, PyEnum):
    """User role enumeration."""

    ADMINISTRATOR = "administrator"
    SECURITY_ANALYST = "security_analyst"
    OPERATOR = "operator"
    VIEWER = "viewer"
    API_USER = "api_user"


class MFAMethod(str, PyEnum):
    """Multi-factor authentication method."""

    TOTP = "totp"  # Time-based One-Time Password
    SMS = "sms"
    EMAIL = "email"
    HARDWARE_TOKEN = "hardware_token"


class APIKeyStatus(str, PyEnum):
    """API key status."""

    ACTIVE = "active"
    REVOKED = "revoked"
    EXPIRED = "expired"


class AuditAction(str, PyEnum):
    """Audit action types."""

    LOGIN = "login"
    LOGOUT = "logout"
    LOGIN_FAILED = "login_failed"
    PASSWORD_CHANGE = "password_change"
    MFA_ENABLED = "mfa_enabled"
    MFA_DISABLED = "mfa_disabled"
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_REVOKED = "permission_revoked"
    USER_CREATED = "user_created"
    USER_UPDATED = "user_updated"
    USER_DELETED = "user_deleted"
    OAUTH_LOGIN = "oauth_login"
    SAML_LOGIN = "saml_login"


class User(Base):
    """User model with enterprise authentication features."""

    __tablename__ = "users"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Basic information
    username = Column(String(255), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    full_name = Column(String(255), nullable=False)

    # Authentication
    password_hash = Column(String(255), nullable=True)  # Nullable for OAuth/SAML users
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)

    # Role and permissions
    role = Column(Enum(UserRole), nullable=False, default=UserRole.VIEWER)
    permissions = Column(JSONB, default=list, nullable=False)

    # Multi-factor authentication
    mfa_enabled = Column(Boolean, default=False, nullable=False)
    mfa_method = Column(Enum(MFAMethod), nullable=True)
    mfa_secret = Column(String(255), nullable=True)  # Encrypted TOTP secret
    mfa_backup_codes = Column(JSONB, nullable=True)  # Encrypted backup codes

    # OAuth/SAML integration
    oauth_provider = Column(String(50), nullable=True)
    oauth_id = Column(String(255), nullable=True)
    saml_name_id = Column(String(255), nullable=True)

    # Security tracking
    last_login = Column(DateTime(timezone=True), nullable=True)
    last_login_ip = Column(INET, nullable=True)
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    account_locked_until = Column(DateTime(timezone=True), nullable=True)
    password_changed_at = Column(DateTime(timezone=True), nullable=True)
    must_change_password = Column(Boolean, default=False, nullable=False)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    deleted_at = Column(DateTime(timezone=True), nullable=True)  # Soft delete

    # Relationships
    api_keys = relationship(
        "APIKey", back_populates="user", cascade="all, delete-orphan"
    )
    sessions = relationship(
        "UserSession", back_populates="user", cascade="all, delete-orphan"
    )
    audit_logs = relationship(
        "AuditLog", back_populates="user", cascade="all, delete-orphan"
    )

    # Indexes
    __table_args__ = (
        Index("idx_users_username", "username"),
        Index("idx_users_email", "email"),
        Index("idx_users_role", "role"),
        Index("idx_users_is_active", "is_active"),
        Index("idx_users_oauth_provider_id", "oauth_provider", "oauth_id"),
        Index("idx_users_created_at", "created_at"),
    )

    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, role={self.role})>"


class APIKey(Base):
    """API key model with rotation and expiration support."""

    __tablename__ = "api_keys"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Key information
    key_hash = Column(
        String(255), unique=True, nullable=False, index=True
    )  # SHA-256 hash
    key_prefix = Column(
        String(20), nullable=False
    )  # First few chars for identification
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    # User association
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )

    # Status and expiration
    status = Column(Enum(APIKeyStatus), default=APIKeyStatus.ACTIVE, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)

    # Usage tracking
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    last_used_ip = Column(INET, nullable=True)
    usage_count = Column(Integer, default=0, nullable=False)

    # Rate limiting
    rate_limit_per_minute = Column(Integer, default=100, nullable=False)

    # Permissions (can be more restrictive than user permissions)
    permissions = Column(JSONB, default=list, nullable=False)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    revoked_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    revoked_reason = Column(Text, nullable=True)

    # Relationships
    user = relationship("User", back_populates="api_keys", foreign_keys=[user_id])

    # Indexes
    __table_args__ = (
        Index("idx_api_keys_key_hash", "key_hash"),
        Index("idx_api_keys_user_id", "user_id"),
        Index("idx_api_keys_status", "status"),
        Index("idx_api_keys_expires_at", "expires_at"),
    )

    def __repr__(self):
        return f"<APIKey(id={self.id}, name={self.name}, status={self.status})>"


class UserSession(Base):
    """User session model for session management."""

    __tablename__ = "user_sessions"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Session information
    session_token = Column(String(255), unique=True, nullable=False, index=True)
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )

    # Session metadata
    ip_address = Column(INET, nullable=True)
    user_agent = Column(Text, nullable=True)
    device_info = Column(JSONB, nullable=True)

    # Session lifecycle
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    expires_at = Column(DateTime(timezone=True), nullable=False)
    last_activity = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    revoked_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    user = relationship("User", back_populates="sessions")

    # Indexes
    __table_args__ = (
        Index("idx_user_sessions_token", "session_token"),
        Index("idx_user_sessions_user_id", "user_id"),
        Index("idx_user_sessions_expires_at", "expires_at"),
    )

    def __repr__(self):
        return f"<UserSession(id={self.id}, user_id={self.user_id})>"


class AuditLog(Base):
    """Audit log model for security and compliance tracking."""

    __tablename__ = "audit_logs"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Action information
    action = Column(Enum(AuditAction), nullable=False, index=True)
    resource_type = Column(String(100), nullable=True)
    resource_id = Column(String(255), nullable=True)

    # User information
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    username = Column(String(255), nullable=True)  # Denormalized for deleted users

    # Request information
    ip_address = Column(INET, nullable=True)
    user_agent = Column(Text, nullable=True)
    request_id = Column(String(255), nullable=True)

    # Action details
    details = Column(JSONB, nullable=True)
    success = Column(Boolean, nullable=False)
    error_message = Column(Text, nullable=True)

    # Timestamp
    timestamp = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    # Relationships
    user = relationship("User", back_populates="audit_logs")

    # Indexes
    __table_args__ = (
        Index("idx_audit_logs_action", "action"),
        Index("idx_audit_logs_user_id", "user_id"),
        Index("idx_audit_logs_timestamp", "timestamp"),
        Index("idx_audit_logs_resource", "resource_type", "resource_id"),
        Index("idx_audit_logs_success", "success"),
    )

    def __repr__(self):
        return f"<AuditLog(id={self.id}, action={self.action}, timestamp={self.timestamp})>"


class OAuthProvider(Base):
    """OAuth provider configuration."""

    __tablename__ = "oauth_providers"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Provider information
    name = Column(String(100), unique=True, nullable=False)
    provider_type = Column(String(50), nullable=False)  # google, github, azure, etc.

    # OAuth configuration
    client_id = Column(String(255), nullable=False)
    client_secret_encrypted = Column(LargeBinary, nullable=False)  # Encrypted
    authorization_endpoint = Column(String(500), nullable=False)
    token_endpoint = Column(String(500), nullable=False)
    userinfo_endpoint = Column(String(500), nullable=True)
    scopes = Column(JSONB, default=list, nullable=False)

    # Status
    is_active = Column(Boolean, default=True, nullable=False)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def __repr__(self):
        return f"<OAuthProvider(name={self.name}, type={self.provider_type})>"


class SAMLProvider(Base):
    """SAML provider configuration."""

    __tablename__ = "saml_providers"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Provider information
    name = Column(String(100), unique=True, nullable=False)
    entity_id = Column(String(500), nullable=False)

    # SAML configuration
    sso_url = Column(String(500), nullable=False)
    slo_url = Column(String(500), nullable=True)
    x509_cert = Column(Text, nullable=False)

    # Attribute mapping
    attribute_mapping = Column(
        JSONB, nullable=False
    )  # Map SAML attributes to user fields

    # Status
    is_active = Column(Boolean, default=True, nullable=False)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def __repr__(self):
        return f"<SAMLProvider(name={self.name}, entity_id={self.entity_id})>"


class PasswordResetToken(Base):
    """Password reset token model."""

    __tablename__ = "password_reset_tokens"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Token information
    token_hash = Column(String(255), unique=True, nullable=False, index=True)
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )

    # Expiration
    expires_at = Column(DateTime(timezone=True), nullable=False)
    used_at = Column(DateTime(timezone=True), nullable=True)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Indexes
    __table_args__ = (
        Index("idx_password_reset_tokens_token_hash", "token_hash"),
        Index("idx_password_reset_tokens_user_id", "user_id"),
        Index("idx_password_reset_tokens_expires_at", "expires_at"),
    )

    def __repr__(self):
        return f"<PasswordResetToken(id={self.id}, user_id={self.user_id})>"
