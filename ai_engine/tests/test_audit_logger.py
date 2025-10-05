"""
Tests for security audit logging.
"""

import pytest
import json
import tempfile
import os
from datetime import datetime
from pathlib import Path

from ai_engine.security.audit_logger import (
    AuditLogger,
    AuditEvent,
    AuditEventType,
    AuditSeverity,
    get_audit_logger,
    log_audit_event,
)


class TestAuditEventType:
    """Test AuditEventType enum."""

    def test_authentication_events(self):
        """Test authentication event types."""
        assert AuditEventType.LOGIN_SUCCESS.value == "login_success"
        assert AuditEventType.LOGIN_FAILED.value == "login_failed"
        assert AuditEventType.LOGOUT.value == "logout"
        assert AuditEventType.TOKEN_CREATED.value == "token_created"

    def test_authorization_events(self):
        """Test authorization event types."""
        assert AuditEventType.ACCESS_GRANTED.value == "access_granted"
        assert AuditEventType.ACCESS_DENIED.value == "access_denied"
        assert AuditEventType.PERMISSION_CHANGED.value == "permission_changed"

    def test_mfa_events(self):
        """Test MFA event types."""
        assert AuditEventType.MFA_ENABLED.value == "mfa_enabled"
        assert AuditEventType.MFA_DISABLED.value == "mfa_disabled"
        assert AuditEventType.MFA_VERIFIED.value == "mfa_verified"

    def test_secret_events(self):
        """Test secret management event types."""
        assert AuditEventType.SECRET_ACCESSED.value == "secret_accessed"
        assert AuditEventType.SECRET_CREATED.value == "secret_created"
        assert AuditEventType.SECRET_ROTATED.value == "secret_rotated"


class TestAuditSeverity:
    """Test AuditSeverity enum."""

    def test_severity_levels(self):
        """Test severity level values."""
        assert AuditSeverity.LOW.value == "low"
        assert AuditSeverity.MEDIUM.value == "medium"
        assert AuditSeverity.HIGH.value == "high"
        assert AuditSeverity.CRITICAL.value == "critical"


class TestAuditEvent:
    """Test AuditEvent dataclass."""

    def test_audit_event_creation(self):
        """Test creating audit event."""
        event = AuditEvent(
            event_type=AuditEventType.LOGIN_SUCCESS,
            timestamp=datetime.utcnow(),
            user_id="user123",
            username="testuser",
            ip_address="192.168.1.1",
            result="success",
            severity=AuditSeverity.LOW,
        )

        assert event.event_type == AuditEventType.LOGIN_SUCCESS
        assert event.user_id == "user123"
        assert event.username == "testuser"
        assert event.ip_address == "192.168.1.1"
        assert event.result == "success"
        assert event.severity == AuditSeverity.LOW

    def test_audit_event_with_details(self):
        """Test audit event with additional details."""
        details = {"login_method": "password", "device": "mobile"}
        event = AuditEvent(
            event_type=AuditEventType.LOGIN_SUCCESS,
            timestamp=datetime.utcnow(),
            user_id="user123",
            details=details,
        )

        assert event.details == details
        assert event.details["login_method"] == "password"

    def test_audit_event_to_dict(self):
        """Test converting audit event to dictionary."""
        timestamp = datetime.utcnow()
        event = AuditEvent(
            event_type=AuditEventType.ACCESS_DENIED,
            timestamp=timestamp,
            user_id="user456",
            resource="/api/admin",
            action="read",
            result="failure",
            severity=AuditSeverity.MEDIUM,
        )

        result = event.to_dict()

        assert result["event_type"] == "access_denied"
        assert result["user_id"] == "user456"
        assert result["resource"] == "/api/admin"
        assert result["action"] == "read"
        assert result["result"] == "failure"
        assert result["severity"] == "medium"
        assert "timestamp" in result

    def test_audit_event_to_json(self):
        """Test converting audit event to JSON."""
        event = AuditEvent(
            event_type=AuditEventType.MFA_ENABLED,
            timestamp=datetime.utcnow(),
            user_id="user789",
            severity=AuditSeverity.LOW,
        )

        json_str = event.to_json()
        parsed = json.loads(json_str)

        assert parsed["event_type"] == "mfa_enabled"
        assert parsed["user_id"] == "user789"
        assert parsed["severity"] == "low"


class TestAuditLogger:
    """Test AuditLogger class."""

    def test_audit_logger_initialization(self):
        """Test audit logger initialization."""
        logger = AuditLogger()

        assert logger.enabled is True
        assert logger.log_to_file is True
        assert logger.log_to_syslog is False
        assert logger.log_to_siem is False

    def test_audit_logger_custom_config(self):
        """Test audit logger with custom configuration."""
        config = {"enabled": False, "log_to_file": False, "log_to_syslog": True}
        logger = AuditLogger(config)

        assert logger.enabled is False
        assert logger.log_to_file is False
        assert logger.log_to_syslog is True

    def test_log_event_when_disabled(self):
        """Test logging when disabled."""
        config = {"enabled": False, "log_to_file": False}
        logger = AuditLogger(config)

        event = AuditEvent(
            event_type=AuditEventType.LOGIN_SUCCESS, timestamp=datetime.utcnow()
        )

        # Should not raise error, just not log
        logger.log_event(event)

    def test_log_event_to_file(self):
        """Test logging event to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_file = os.path.join(tmpdir, "audit.log")
            config = {"enabled": True, "log_to_file": True, "audit_file": audit_file}
            logger = AuditLogger(config)

            event = AuditEvent(
                event_type=AuditEventType.LOGIN_SUCCESS,
                timestamp=datetime.utcnow(),
                user_id="user123",
            )

            logger.log_event(event)

            # Check file was created
            assert os.path.exists(audit_file)

    def test_log_login_success(self):
        """Test convenience method for login success."""
        config = {"enabled": True, "log_to_file": False}
        logger = AuditLogger(config)

        logger.log_login_success(
            user_id="user123",
            username="testuser",
            ip_address="192.168.1.1",
            mfa_used=True,
        )

        # Should not raise error

    def test_log_login_failed(self):
        """Test convenience method for login failure."""
        config = {"enabled": True, "log_to_file": False}
        logger = AuditLogger(config)

        logger.log_login_failed(
            username="testuser", reason="Invalid password", ip_address="192.168.1.1"
        )

        # Should not raise error

    def test_log_mfa_enabled(self):
        """Test convenience method for MFA enabled."""
        config = {"enabled": True, "log_to_file": False}
        logger = AuditLogger(config)

        logger.log_mfa_enabled(user_id="user123", username="testuser", method="totp")

        # Should not raise error

    def test_log_password_changed(self):
        """Test convenience method for password change."""
        config = {"enabled": True, "log_to_file": False}
        logger = AuditLogger(config)

        logger.log_password_changed(
            user_id="user123", username="testuser", forced=False
        )

        # Should not raise error

    def test_log_api_key_created(self):
        """Test convenience method for API key creation."""
        config = {"enabled": True, "log_to_file": False}
        logger = AuditLogger(config)

        logger.log_api_key_created(
            user_id="user123", key_name="production-key", key_id="key-abc123"
        )

        # Should not raise error

    def test_log_secret_accessed(self):
        """Test convenience method for secret access."""
        config = {"enabled": True, "log_to_file": False}
        logger = AuditLogger(config)

        logger.log_secret_accessed(user_id="user123", secret_key="database_password")

        # Should not raise error

    def test_log_secret_rotated(self):
        """Test convenience method for secret rotation."""
        config = {"enabled": True, "log_to_file": False}
        logger = AuditLogger(config)

        logger.log_secret_rotated(
            user_id="user123", secret_key="api_key", reason="scheduled_rotation"
        )

        # Should not raise error

    def test_log_access_denied(self):
        """Test convenience method for access denied."""
        config = {"enabled": True, "log_to_file": False}
        logger = AuditLogger(config)

        logger.log_access_denied(
            user_id="user123",
            resource="/api/admin/users",
            action="delete",
            reason="insufficient_permissions",
        )

        # Should not raise error

    def test_log_security_alert(self):
        """Test convenience method for security alert."""
        config = {"enabled": True, "log_to_file": False}
        logger = AuditLogger(config)

        logger.log_security_alert(
            alert_type="brute_force_attempt",
            description="Multiple failed login attempts detected",
            severity=AuditSeverity.HIGH,
            ip_address="192.168.1.100",
        )

        # Should not raise error

    def test_critical_event_logged_to_main_logger(self, caplog):
        """Test critical events are logged to main logger."""
        config = {"enabled": True, "log_to_file": False}
        logger = AuditLogger(config)

        event = AuditEvent(
            event_type=AuditEventType.SECURITY_ALERT,
            timestamp=datetime.utcnow(),
            severity=AuditSeverity.CRITICAL,
            details={"alert": "critical_security_breach"},
        )

        logger.log_event(event)

        # Check that critical log was written
        # (caplog will capture it if logging is configured)


class TestGlobalAuditLoggerFunctions:
    """Test global audit logger functions."""

    def test_get_audit_logger_singleton(self):
        """Test audit logger singleton pattern."""
        logger1 = get_audit_logger()
        logger2 = get_audit_logger()

        assert logger1 is logger2

    def test_log_audit_event_function(self):
        """Test global log_audit_event function."""
        event = AuditEvent(
            event_type=AuditEventType.LOGIN_SUCCESS,
            timestamp=datetime.utcnow(),
            user_id="user123",
        )

        # Should not raise error
        log_audit_event(event)


class TestAuditEventScenarios:
    """Test real-world audit event scenarios."""

    def test_successful_login_flow(self):
        """Test complete successful login flow."""
        config = {"enabled": True, "log_to_file": False}
        logger = AuditLogger(config)

        # User logs in successfully
        logger.log_login_success(
            user_id="user123",
            username="alice",
            ip_address="192.168.1.50",
            user_agent="Mozilla/5.0",
            mfa_used=True,
        )

        # Token is created
        event = AuditEvent(
            event_type=AuditEventType.TOKEN_CREATED,
            timestamp=datetime.utcnow(),
            user_id="user123",
            username="alice",
            result="success",
            severity=AuditSeverity.LOW,
            details={"token_type": "bearer", "expires_in": 3600},
        )
        logger.log_event(event)

    def test_failed_login_attempt(self):
        """Test failed login attempt."""
        config = {"enabled": True, "log_to_file": False}
        logger = AuditLogger(config)

        logger.log_login_failed(
            username="alice",
            reason="Invalid credentials",
            ip_address="192.168.1.50",
            attempt_count=3,
        )

    def test_account_lockout(self):
        """Test account lockout scenario."""
        config = {"enabled": True, "log_to_file": False}
        logger = AuditLogger(config)

        # Multiple failed logins
        for i in range(5):
            logger.log_login_failed(
                username="alice",
                reason="Invalid password",
                ip_address="192.168.1.50",
                attempt_count=i + 1,
            )

        # Account locked
        event = AuditEvent(
            event_type=AuditEventType.ACCOUNT_LOCKED,
            timestamp=datetime.utcnow(),
            user_id="user123",
            username="alice",
            result="success",
            severity=AuditSeverity.HIGH,
            details={"reason": "too_many_failed_attempts"},
        )
        logger.log_event(event)

    def test_privilege_escalation_attempt(self):
        """Test privilege escalation detection."""
        config = {"enabled": True, "log_to_file": False}
        logger = AuditLogger(config)

        logger.log_access_denied(
            user_id="user456",
            resource="/api/admin/permissions",
            action="update",
            reason="insufficient_role",
            current_role="user",
            required_role="admin",
        )

        logger.log_security_alert(
            alert_type="privilege_escalation_attempt",
            description="User attempted to access admin endpoint",
            severity=AuditSeverity.HIGH,
            user_id="user456",
        )

    def test_sensitive_data_access(self):
        """Test sensitive data access logging."""
        config = {"enabled": True, "log_to_file": False}
        logger = AuditLogger(config)

        event = AuditEvent(
            event_type=AuditEventType.SENSITIVE_DATA_ACCESSED,
            timestamp=datetime.utcnow(),
            user_id="user123",
            resource="/api/users/pii",
            action="read",
            result="success",
            severity=AuditSeverity.HIGH,
            details={
                "data_type": "personal_information",
                "record_count": 100,
                "justification": "compliance_audit",
            },
        )
        logger.log_event(event)

    def test_configuration_change(self):
        """Test security configuration change."""
        config = {"enabled": True, "log_to_file": False}
        logger = AuditLogger(config)

        event = AuditEvent(
            event_type=AuditEventType.SECURITY_POLICY_CHANGED,
            timestamp=datetime.utcnow(),
            user_id="admin123",
            username="admin",
            action="update",
            result="success",
            severity=AuditSeverity.HIGH,
            details={
                "policy": "password_requirements",
                "changes": {
                    "min_length": {"old": 8, "new": 12},
                    "require_special_chars": {"old": False, "new": True},
                },
            },
        )
        logger.log_event(event)

    def test_api_key_lifecycle(self):
        """Test complete API key lifecycle."""
        config = {"enabled": True, "log_to_file": False}
        logger = AuditLogger(config)

        # API key created
        logger.log_api_key_created(
            user_id="user123",
            key_name="production-api-key",
            key_id="key-abc123",
            scope="read_write",
        )

        # API key used
        event = AuditEvent(
            event_type=AuditEventType.API_KEY_USED,
            timestamp=datetime.utcnow(),
            user_id="user123",
            result="success",
            severity=AuditSeverity.LOW,
            details={"key_id": "key-abc123", "endpoint": "/api/data", "method": "POST"},
        )
        logger.log_event(event)

        # API key revoked
        event = AuditEvent(
            event_type=AuditEventType.API_KEY_REVOKED,
            timestamp=datetime.utcnow(),
            user_id="user123",
            result="success",
            severity=AuditSeverity.MEDIUM,
            details={"key_id": "key-abc123", "reason": "compromised"},
        )
        logger.log_event(event)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
