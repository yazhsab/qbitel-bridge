"""
QBITEL Engine - Security Orchestrator Enterprise Logging

Enterprise-grade structured logging with security event tracking, audit trails,
and compliance features.
"""

import logging
import logging.handlers
import json
import time
import uuid
import traceback
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
import threading
from contextlib import contextmanager

from .models import (
    SecurityEvent,
    ThreatAnalysis,
    AutomatedResponse,
    SecurityEventType,
    ThreatLevel,
)
from .config import get_security_config


class LogLevel(str, Enum):
    """Enhanced log levels for security operations."""

    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"
    AUDIT = "AUDIT"  # Special audit log level
    SECURITY = "SECURITY"  # Security event logs
    COMPLIANCE = "COMPLIANCE"  # Compliance-related logs


class SecurityLogType(str, Enum):
    """Types of security logs."""

    EVENT_DETECTION = "event_detection"
    THREAT_ANALYSIS = "threat_analysis"
    DECISION_MAKING = "decision_making"
    RESPONSE_EXECUTION = "response_execution"
    SYSTEM_QUARANTINE = "system_quarantine"
    ACCESS_CONTROL = "access_control"
    CONFIGURATION_CHANGE = "configuration_change"
    AUDIT_TRAIL = "audit_trail"
    COMPLIANCE_CHECK = "compliance_check"
    PERFORMANCE_METRIC = "performance_metric"


@dataclass
class SecurityLogEntry:
    """Structured security log entry."""

    timestamp: datetime
    log_id: str
    log_type: SecurityLogType
    log_level: LogLevel
    message: str

    # Core identification
    event_id: Optional[str] = None
    incident_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    # Security context
    threat_level: Optional[str] = None
    confidence_score: Optional[float] = None
    affected_systems: List[str] = None
    source_ip: Optional[str] = None

    # Business context
    business_impact: Optional[str] = None
    compliance_framework: Optional[str] = None
    data_classification: Optional[str] = None

    # Technical details
    component: Optional[str] = None
    function_name: Optional[str] = None
    execution_time_ms: Optional[float] = None
    error_code: Optional[str] = None
    stack_trace: Optional[str] = None

    # Additional context
    metadata: Dict[str, Any] = None
    tags: List[str] = None

    def __post_init__(self):
        if self.affected_systems is None:
            self.affected_systems = []
        if self.metadata is None:
            self.metadata = {}
        if self.tags is None:
            self.tags = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class SensitiveDataMasker:
    """Masks sensitive data in logs for compliance."""

    def __init__(self):
        self.pii_patterns = {
            "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "ip_private": r"\b(?:10\.|172\.(?:1[6-9]|2[0-9]|3[01])\.|192\.168\.)\d{1,3}\.\d{1,3}\b",
            "password": r"(?i)(password|pwd|pass|secret|token|key)\s*[:=]\s*[^\s]+",
            "api_key": r"(?i)(api[_-]?key|apikey|access[_-]?token)\s*[:=]\s*[^\s]+",
        }

        self.sensitive_fields = {
            "password",
            "secret",
            "token",
            "key",
            "credential",
            "ssn",
            "social_security",
            "credit_card",
            "card_number",
        }

    def mask_message(self, message: str) -> str:
        """Mask sensitive data in message."""
        import re

        for pattern_name, pattern in self.pii_patterns.items():
            if pattern_name == "ip_private":
                # Keep last octet for debugging, mask others
                message = re.sub(pattern, lambda m: self._mask_ip(m.group()), message)
            else:
                message = re.sub(pattern, "[REDACTED]", message)

        return message

    def mask_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive data in metadata dictionary."""
        if not metadata:
            return metadata

        masked = {}
        for key, value in metadata.items():
            key_lower = key.lower()

            if any(sensitive in key_lower for sensitive in self.sensitive_fields):
                masked[key] = "[REDACTED]"
            elif isinstance(value, str):
                masked[key] = self.mask_message(value)
            elif isinstance(value, dict):
                masked[key] = self.mask_metadata(value)
            else:
                masked[key] = value

        return masked

    def _mask_ip(self, ip: str) -> str:
        """Mask private IP address partially."""
        parts = ip.split(".")
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.xxx.{parts[3]}"
        return "[REDACTED_IP]"


class SecurityLogger:
    """
    Enterprise security logger with structured logging, audit trails, and compliance features.
    """

    def __init__(self, name: str = "qbitel.security"):
        self.logger_name = name
        self.logger = logging.getLogger(name)
        self.config = get_security_config()

        # Thread-local storage for context
        self.context = threading.local()

        # Sensitive data masker
        self.masker = SensitiveDataMasker()

        # Setup logger
        self._setup_logger()

        # Performance tracking
        self.log_counts = {level.value: 0 for level in LogLevel}
        self.start_time = time.time()

    def _setup_logger(self):
        """Setup enterprise logging configuration."""
        # Clear existing handlers
        self.logger.handlers.clear()

        # Set log level
        log_level = self.config.monitoring.logging.get("level", "INFO")
        self.logger.setLevel(getattr(logging, log_level))

        # Console handler for development
        if self.config.monitoring.logging.get("console_enabled", True):
            console_handler = logging.StreamHandler()
            console_formatter = self._create_formatter(structured=False)
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

        # File handler for production
        if self.config.monitoring.logging.get("file_enabled", True):
            log_dir = Path(self.config.monitoring.logging.get("log_directory", "logs"))
            log_dir.mkdir(exist_ok=True)

            # Main application log
            app_log_file = log_dir / f"{self.logger_name}.log"
            app_handler = logging.handlers.RotatingFileHandler(
                app_log_file, maxBytes=100 * 1024 * 1024, backupCount=10  # 100MB
            )
            app_formatter = self._create_formatter(structured=True)
            app_handler.setFormatter(app_formatter)
            self.logger.addHandler(app_handler)

            # Security events log
            security_log_file = log_dir / f"{self.logger_name}_security.log"
            security_handler = logging.handlers.RotatingFileHandler(
                security_log_file, maxBytes=500 * 1024 * 1024, backupCount=20  # 500MB
            )
            security_formatter = self._create_formatter(structured=True)
            security_handler.setFormatter(security_formatter)
            security_handler.addFilter(self._security_log_filter)
            self.logger.addHandler(security_handler)

            # Audit log (if compliance enabled)
            if self.config.security.audit.get("enabled", False):
                audit_log_file = log_dir / f"{self.logger_name}_audit.log"
                audit_handler = logging.handlers.RotatingFileHandler(
                    audit_log_file,
                    maxBytes=1 * 1024 * 1024 * 1024,  # 1GB
                    backupCount=50,  # Long retention for compliance
                )
                audit_formatter = self._create_formatter(structured=True)
                audit_handler.setFormatter(audit_formatter)
                audit_handler.addFilter(self._audit_log_filter)
                self.logger.addHandler(audit_handler)

        # Syslog handler for enterprise integration
        if self.config.monitoring.logging.get("syslog_enabled", False):
            syslog_handler = logging.handlers.SysLogHandler(
                address=self.config.monitoring.logging.get("syslog_address", "/dev/log")
            )
            syslog_formatter = self._create_formatter(structured=True)
            syslog_handler.setFormatter(syslog_formatter)
            self.logger.addHandler(syslog_handler)

    def _create_formatter(self, structured: bool = True):
        """Create log formatter."""
        if structured:
            return StructuredSecurityFormatter()
        else:
            return logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

    def _security_log_filter(self, record):
        """Filter for security logs only."""
        return hasattr(record, "log_type") and record.log_type in [
            SecurityLogType.EVENT_DETECTION.value,
            SecurityLogType.THREAT_ANALYSIS.value,
            SecurityLogType.RESPONSE_EXECUTION.value,
            SecurityLogType.SYSTEM_QUARANTINE.value,
        ]

    def _audit_log_filter(self, record):
        """Filter for audit logs only."""
        return hasattr(record, "log_level") and record.log_level in [
            LogLevel.AUDIT.value,
            LogLevel.COMPLIANCE.value,
        ]

    def log_security_event(
        self,
        log_type: SecurityLogType,
        message: str,
        level: LogLevel = LogLevel.INFO,
        **kwargs,
    ) -> str:
        """
        Log a security event with structured data.

        Args:
            log_type: Type of security log
            message: Log message
            level: Log level
            **kwargs: Additional context data

        Returns:
            Log entry ID
        """

        log_entry = self._create_log_entry(
            log_type=log_type, message=message, log_level=level, **kwargs
        )

        # Apply sensitive data masking if enabled
        if self.config.security.privacy.get("data_masking", True):
            log_entry.message = self.masker.mask_message(log_entry.message)
            log_entry.metadata = self.masker.mask_metadata(log_entry.metadata)

        # Log the entry
        self._log_entry(log_entry)

        # Update metrics
        self.log_counts[level.value] += 1

        return log_entry.log_id

    def log_security_event_obj(
        self,
        security_event: SecurityEvent,
        log_type: SecurityLogType = SecurityLogType.EVENT_DETECTION,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log a security event object."""

        context = {
            "event_id": security_event.event_id,
            "threat_level": security_event.threat_level.value,
            "confidence_score": security_event.confidence_score,
            "affected_systems": security_event.affected_systems,
            "source_ip": security_event.source_ip,
            "event_type": security_event.event_type.value,
            "detection_method": security_event.detection_method,
            "indicators_count": len(security_event.indicators_of_compromise),
            "metadata": {
                "event_timestamp": security_event.event_timestamp.isoformat(),
                "detection_timestamp": security_event.detection_timestamp.isoformat(),
                "false_positive_likelihood": security_event.false_positive_likelihood,
            },
        }

        if additional_context:
            context["metadata"].update(additional_context)

        return self.log_security_event(
            log_type=log_type,
            message=f"Security event detected: {security_event.event_type.value} - {security_event.description}",
            level=self._threat_level_to_log_level(security_event.threat_level),
            **context,
        )

    def log_threat_analysis(
        self,
        threat_analysis: ThreatAnalysis,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log threat analysis results."""

        context = {
            "event_id": threat_analysis.event_id,
            "analysis_id": threat_analysis.analysis_id,
            "threat_level": threat_analysis.threat_level.value,
            "confidence_score": threat_analysis.confidence_score,
            "business_impact_score": threat_analysis.business_impact_score,
            "affected_systems": threat_analysis.affected_assets,
            "metadata": {
                "threat_classification": threat_analysis.threat_classification.value,
                "confidence_level": threat_analysis.confidence.value,
                "processing_time_ms": threat_analysis.processing_time_ms,
                "ttps": threat_analysis.ttps,
                "mitre_techniques": threat_analysis.mitre_attack_techniques,
                "threat_actor": threat_analysis.threat_actor,
                "financial_impact": threat_analysis.financial_impact_estimate,
            },
        }

        if additional_context:
            context["metadata"].update(additional_context)

        return self.log_security_event(
            log_type=SecurityLogType.THREAT_ANALYSIS,
            message=f"Threat analysis completed: {threat_analysis.threat_classification.value} "
            f"(confidence: {threat_analysis.confidence_score:.2f})",
            level=self._threat_level_to_log_level(threat_analysis.threat_level),
            **context,
        )

    def log_automated_response(
        self,
        automated_response: AutomatedResponse,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log automated response execution."""

        context = {
            "event_id": automated_response.event_id,
            "response_id": automated_response.response_id,
            "analysis_id": automated_response.analysis_id,
            "confidence_score": automated_response.confidence_score,
            "auto_execute": automated_response.auto_execute,
            "requires_approval": automated_response.requires_human_approval,
            "metadata": {
                "response_strategy": automated_response.response_strategy,
                "confidence_level": automated_response.confidence.value,
                "actions_count": len(automated_response.actions),
                "overall_risk_score": automated_response.overall_risk_score,
                "approved_by": automated_response.approved_by,
                "approved_at": (
                    automated_response.approved_at.isoformat()
                    if automated_response.approved_at
                    else None
                ),
                "actions": [
                    {
                        "action_id": action.action_id,
                        "action_type": action.action_type.value,
                        "priority": action.priority,
                        "risk_level": action.risk_level.value,
                        "requires_approval": action.requires_approval,
                    }
                    for action in automated_response.actions
                ],
            },
        }

        if additional_context:
            context["metadata"].update(additional_context)

        log_level = (
            LogLevel.WARNING if automated_response.auto_execute else LogLevel.INFO
        )

        return self.log_security_event(
            log_type=SecurityLogType.DECISION_MAKING,
            message=f"Automated response created: {automated_response.response_strategy} "
            f"({'auto-execute' if automated_response.auto_execute else 'requires approval'})",
            level=log_level,
            **context,
        )

    def log_audit_event(
        self,
        action: str,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        result: str = "SUCCESS",
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log audit event for compliance."""

        context = {
            "user_id": user_id,
            "metadata": {
                "action": action,
                "resource": resource,
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "source_component": self.logger_name,
            },
        }

        if additional_context:
            context["metadata"].update(additional_context)

        return self.log_security_event(
            log_type=SecurityLogType.AUDIT_TRAIL,
            message=f"Audit: {action} on {resource} by {user_id} - {result}",
            level=LogLevel.AUDIT,
            **context,
        )

    def _create_log_entry(
        self, log_type: SecurityLogType, message: str, log_level: LogLevel, **kwargs
    ) -> SecurityLogEntry:
        """Create structured log entry."""

        return SecurityLogEntry(
            timestamp=datetime.now(),
            log_id=str(uuid.uuid4()),
            log_type=log_type,
            log_level=log_level,
            message=message,
            event_id=kwargs.get("event_id"),
            incident_id=kwargs.get("incident_id"),
            user_id=kwargs.get("user_id"),
            session_id=kwargs.get("session_id"),
            threat_level=kwargs.get("threat_level"),
            confidence_score=kwargs.get("confidence_score"),
            affected_systems=kwargs.get("affected_systems", []),
            source_ip=kwargs.get("source_ip"),
            business_impact=kwargs.get("business_impact"),
            compliance_framework=kwargs.get("compliance_framework"),
            data_classification=kwargs.get("data_classification"),
            component=kwargs.get("component", self.logger_name),
            function_name=kwargs.get("function_name"),
            execution_time_ms=kwargs.get("execution_time_ms"),
            error_code=kwargs.get("error_code"),
            stack_trace=kwargs.get("stack_trace"),
            metadata=kwargs.get("metadata", {}),
            tags=kwargs.get("tags", []),
        )

    def _log_entry(self, log_entry: SecurityLogEntry):
        """Log the structured entry."""

        # Create log record
        record = logging.LogRecord(
            name=self.logger_name,
            level=getattr(logging, log_entry.log_level.value),
            pathname="",
            lineno=0,
            msg=log_entry.message,
            args=(),
            exc_info=None,
        )

        # Add structured data to record
        for field, value in asdict(log_entry).items():
            setattr(record, field, value)

        # Log through standard Python logging
        self.logger.handle(record)

    def _threat_level_to_log_level(self, threat_level: ThreatLevel) -> LogLevel:
        """Convert threat level to log level."""
        mapping = {
            ThreatLevel.CRITICAL: LogLevel.CRITICAL,
            ThreatLevel.HIGH: LogLevel.ERROR,
            ThreatLevel.MEDIUM: LogLevel.WARNING,
            ThreatLevel.LOW: LogLevel.INFO,
            ThreatLevel.INFO: LogLevel.DEBUG,
        }
        return mapping.get(threat_level, LogLevel.INFO)

    @contextmanager
    def security_context(
        self,
        event_id: Optional[str] = None,
        incident_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        """Context manager for adding security context to all logs."""

        # Store previous context
        previous_context = getattr(self.context, "security_data", {})

        # Set new context
        self.context.security_data = {
            "event_id": event_id,
            "incident_id": incident_id,
            "user_id": user_id,
            "session_id": session_id,
        }

        try:
            yield
        finally:
            # Restore previous context
            self.context.security_data = previous_context

    def get_log_statistics(self) -> Dict[str, Any]:
        """Get logging statistics."""

        uptime_seconds = time.time() - self.start_time

        return {
            "uptime_seconds": uptime_seconds,
            "log_counts": self.log_counts.copy(),
            "logs_per_second": sum(self.log_counts.values()) / max(uptime_seconds, 1),
            "configuration": {
                "log_level": self.logger.level,
                "handlers_count": len(self.logger.handlers),
                "sensitive_data_masking": self.config.security.privacy.get(
                    "data_masking", True
                ),
                "audit_enabled": self.config.security.audit.get("enabled", False),
            },
        }


class StructuredSecurityFormatter(logging.Formatter):
    """Custom formatter for structured security logs."""

    def format(self, record):
        """Format log record as structured JSON."""

        # Basic log data
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add structured security data if available
        if hasattr(record, "log_id"):
            log_data["log_id"] = record.log_id

        if hasattr(record, "log_type"):
            log_data["log_type"] = record.log_type

        if hasattr(record, "event_id"):
            log_data["event_id"] = record.event_id

        if hasattr(record, "incident_id"):
            log_data["incident_id"] = record.incident_id

        if hasattr(record, "threat_level"):
            log_data["threat_level"] = record.threat_level

        if hasattr(record, "confidence_score"):
            log_data["confidence_score"] = record.confidence_score

        if hasattr(record, "affected_systems"):
            log_data["affected_systems"] = record.affected_systems

        if hasattr(record, "source_ip"):
            log_data["source_ip"] = record.source_ip

        if hasattr(record, "component"):
            log_data["component"] = record.component

        if hasattr(record, "metadata") and record.metadata:
            log_data["metadata"] = record.metadata

        if hasattr(record, "tags") and record.tags:
            log_data["tags"] = record.tags

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add stack trace if available
        if hasattr(record, "stack_trace") and record.stack_trace:
            log_data["stack_trace"] = record.stack_trace

        return json.dumps(log_data, default=str)


# Global logger instance
_security_logger: Optional[SecurityLogger] = None


def get_security_logger(name: str = "qbitel.security") -> SecurityLogger:
    """Get global security logger instance."""
    global _security_logger
    if _security_logger is None:
        _security_logger = SecurityLogger(name)
    return _security_logger


def log_security_event(log_type: SecurityLogType, message: str, **kwargs) -> str:
    """Convenience function for logging security events."""
    logger = get_security_logger()
    return logger.log_security_event(log_type, message, **kwargs)


def log_audit_event(action: str, user_id: Optional[str] = None, **kwargs) -> str:
    """Convenience function for logging audit events."""
    logger = get_security_logger()
    return logger.log_audit_event(action, user_id, **kwargs)
