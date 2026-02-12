"""
QBITEL Engine - Security Audit Logging

Comprehensive audit logging for security events with compliance support.
"""

import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Audit event types."""

    # Authentication events
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILED = "login_failed"
    LOGOUT = "logout"
    TOKEN_CREATED = "token_created"
    TOKEN_REFRESHED = "token_refreshed"
    TOKEN_REVOKED = "token_revoked"

    # Authorization events
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    PERMISSION_CHANGED = "permission_changed"
    ROLE_CHANGED = "role_changed"

    # MFA events
    MFA_ENABLED = "mfa_enabled"
    MFA_DISABLED = "mfa_disabled"
    MFA_VERIFIED = "mfa_verified"
    MFA_FAILED = "mfa_failed"

    # Password events
    PASSWORD_CHANGED = "password_changed"
    PASSWORD_RESET_REQUESTED = "password_reset_requested"
    PASSWORD_RESET_COMPLETED = "password_reset_completed"

    # API Key events
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"
    API_KEY_USED = "api_key_used"

    # Account events
    ACCOUNT_CREATED = "account_created"
    ACCOUNT_UPDATED = "account_updated"
    ACCOUNT_DELETED = "account_deleted"
    ACCOUNT_LOCKED = "account_locked"
    ACCOUNT_UNLOCKED = "account_unlocked"

    # Secret events
    SECRET_ACCESSED = "secret_accessed"
    SECRET_CREATED = "secret_created"
    SECRET_UPDATED = "secret_updated"
    SECRET_DELETED = "secret_deleted"
    SECRET_ROTATED = "secret_rotated"

    # Configuration events
    CONFIG_CHANGED = "config_changed"
    SECURITY_POLICY_CHANGED = "security_policy_changed"

    # Data access events
    SENSITIVE_DATA_ACCESSED = "sensitive_data_accessed"
    DATA_EXPORTED = "data_exported"

    # System events
    SYSTEM_ERROR = "system_error"
    SECURITY_ALERT = "security_alert"


class AuditSeverity(str, Enum):
    """Audit event severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Audit event data structure."""

    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str] = None
    username: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    result: str = "success"  # success, failure, error
    severity: AuditSeverity = AuditSeverity.LOW
    details: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["event_type"] = self.event_type.value
        data["severity"] = self.severity.value
        return data

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class AuditLogger:
    """
    Security audit logger with compliance support.

    Provides comprehensive audit logging for security events with
    support for SOC2, HIPAA, PCI-DSS, and other compliance frameworks.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize audit logger."""
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.log_to_file = self.config.get("log_to_file", True)
        self.log_to_syslog = self.config.get("log_to_syslog", False)
        self.log_to_siem = self.config.get("log_to_siem", False)
        self.audit_file = self.config.get("audit_file", "security_audit.log")

        # Setup file handler if enabled
        if self.log_to_file:
            self._setup_file_handler()

    def _setup_file_handler(self):
        """Setup file handler for audit logs."""
        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(self.audit_file, maxBytes=100 * 1024 * 1024, backupCount=10)  # 100MB
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - AUDIT - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)

        audit_logger = logging.getLogger("qbitel.security.audit")
        audit_logger.addHandler(file_handler)
        audit_logger.setLevel(logging.INFO)

    def log_event(self, event: AuditEvent):
        """
        Log an audit event.

        Args:
            event: AuditEvent to log
        """
        if not self.enabled:
            return

        audit_logger = logging.getLogger("qbitel.security.audit")

        # Log to file
        if self.log_to_file:
            audit_logger.info(event.to_json())

        # Log to syslog if configured
        if self.log_to_syslog:
            self._log_to_syslog(event)

        # Log to SIEM if configured
        if self.log_to_siem:
            self._log_to_siem(event)

        # Log critical events to main logger
        if event.severity == AuditSeverity.CRITICAL:
            logger.critical(
                f"SECURITY AUDIT: {event.event_type.value} - {event.result}",
                extra=event.to_dict(),
            )

    def _log_to_syslog(self, event: AuditEvent):
        """Log to syslog."""
        # TODO: Implement syslog integration
        pass

    def _log_to_siem(self, event: AuditEvent):
        """Log to SIEM system."""
        # TODO: Implement SIEM integration (Splunk, ELK, etc.)
        pass

    # Convenience methods for common events

    def log_login_success(
        self,
        user_id: str,
        username: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        mfa_used: bool = False,
        **kwargs,
    ):
        """Log successful login."""
        event = AuditEvent(
            event_type=AuditEventType.LOGIN_SUCCESS,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            username=username,
            ip_address=ip_address,
            user_agent=user_agent,
            result="success",
            severity=AuditSeverity.LOW,
            details={"mfa_used": mfa_used, **kwargs},
        )
        self.log_event(event)

    def log_login_failed(
        self,
        username: str,
        reason: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        **kwargs,
    ):
        """Log failed login attempt."""
        event = AuditEvent(
            event_type=AuditEventType.LOGIN_FAILED,
            timestamp=datetime.utcnow(),
            username=username,
            ip_address=ip_address,
            user_agent=user_agent,
            result="failure",
            severity=AuditSeverity.MEDIUM,
            error_message=reason,
            details=kwargs,
        )
        self.log_event(event)

    def log_mfa_enabled(self, user_id: str, username: str, method: str, **kwargs):
        """Log MFA enabled."""
        event = AuditEvent(
            event_type=AuditEventType.MFA_ENABLED,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            username=username,
            result="success",
            severity=AuditSeverity.LOW,
            details={"method": method, **kwargs},
        )
        self.log_event(event)

    def log_password_changed(self, user_id: str, username: str, forced: bool = False, **kwargs):
        """Log password change."""
        event = AuditEvent(
            event_type=AuditEventType.PASSWORD_CHANGED,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            username=username,
            result="success",
            severity=AuditSeverity.MEDIUM,
            details={"forced": forced, **kwargs},
        )
        self.log_event(event)

    def log_api_key_created(self, user_id: str, key_name: str, key_id: str, **kwargs):
        """Log API key creation."""
        event = AuditEvent(
            event_type=AuditEventType.API_KEY_CREATED,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            result="success",
            severity=AuditSeverity.MEDIUM,
            details={"key_name": key_name, "key_id": key_id, **kwargs},
        )
        self.log_event(event)

    def log_secret_accessed(self, user_id: str, secret_key: str, **kwargs):
        """Log secret access."""
        event = AuditEvent(
            event_type=AuditEventType.SECRET_ACCESSED,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            resource=secret_key,
            result="success",
            severity=AuditSeverity.HIGH,
            details=kwargs,
        )
        self.log_event(event)

    def log_secret_rotated(self, user_id: str, secret_key: str, **kwargs):
        """Log secret rotation."""
        event = AuditEvent(
            event_type=AuditEventType.SECRET_ROTATED,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            resource=secret_key,
            result="success",
            severity=AuditSeverity.HIGH,
            details=kwargs,
        )
        self.log_event(event)

    def log_access_denied(self, user_id: Optional[str], resource: str, action: str, reason: str, **kwargs):
        """Log access denied."""
        event = AuditEvent(
            event_type=AuditEventType.ACCESS_DENIED,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            resource=resource,
            action=action,
            result="failure",
            severity=AuditSeverity.MEDIUM,
            error_message=reason,
            details=kwargs,
        )
        self.log_event(event)

    def log_security_alert(
        self,
        alert_type: str,
        description: str,
        severity: AuditSeverity = AuditSeverity.HIGH,
        **kwargs,
    ):
        """Log security alert."""
        event = AuditEvent(
            event_type=AuditEventType.SECURITY_ALERT,
            timestamp=datetime.utcnow(),
            result="alert",
            severity=severity,
            details={"alert_type": alert_type, "description": description, **kwargs},
        )
        self.log_event(event)


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger(config: Optional[Dict[str, Any]] = None) -> AuditLogger:
    """Get global audit logger instance."""
    global _audit_logger

    if _audit_logger is None:
        _audit_logger = AuditLogger(config)

    return _audit_logger


def log_audit_event(event: AuditEvent):
    """Convenience function to log audit event."""
    logger = get_audit_logger()
    logger.log_event(event)
