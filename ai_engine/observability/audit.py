"""
Audit Logging

Compliance-ready audit logging for:
- Security events
- Access control
- Data modifications
- Cryptographic operations
- Compliance reporting

Supports:
- Tamper-evident logging
- SIEM integration
- Regulatory compliance (PCI-DSS, SOX, GDPR)
- Long-term retention
"""

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import uuid


class AuditCategory(Enum):
    """Categories of audit events."""

    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    CONFIGURATION = "configuration"
    CRYPTO_OPERATION = "crypto_operation"
    KEY_MANAGEMENT = "key_management"
    CERTIFICATE = "certificate"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    SYSTEM = "system"


class AuditSeverity(Enum):
    """Severity levels for audit events."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuditOutcome(Enum):
    """Outcome of audited action."""

    SUCCESS = "success"
    FAILURE = "failure"
    DENIED = "denied"
    ERROR = "error"


@dataclass
class AuditActor:
    """Actor who performed the action."""

    actor_id: str
    actor_type: str = "user"  # user, service, system
    username: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    roles: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "username": self.username,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "session_id": self.session_id,
            "roles": self.roles,
        }


@dataclass
class AuditResource:
    """Resource that was acted upon."""

    resource_id: str
    resource_type: str
    resource_name: Optional[str] = None
    path: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "resource_id": self.resource_id,
            "resource_type": self.resource_type,
            "resource_name": self.resource_name,
            "path": self.path,
            "attributes": self.attributes,
        }


@dataclass
class AuditEvent:
    """Audit event record."""

    event_id: str
    timestamp: datetime
    category: AuditCategory
    action: str
    outcome: AuditOutcome
    severity: AuditSeverity
    actor: AuditActor
    description: str
    resource: Optional[AuditResource] = None
    correlation_id: Optional[str] = None
    trace_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Tamper-evident fields
    sequence_number: int = 0
    previous_hash: str = ""
    event_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "category": self.category.value,
            "action": self.action,
            "outcome": self.outcome.value,
            "severity": self.severity.value,
            "actor": self.actor.to_dict(),
            "description": self.description,
            "resource": self.resource.to_dict() if self.resource else None,
            "correlation_id": self.correlation_id,
            "trace_id": self.trace_id,
            "details": self.details,
            "metadata": self.metadata,
            "sequence_number": self.sequence_number,
            "previous_hash": self.previous_hash,
            "event_hash": self.event_hash,
        }

    def calculate_hash(self) -> str:
        """Calculate hash for tamper detection."""
        data = {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "category": self.category.value,
            "action": self.action,
            "outcome": self.outcome.value,
            "actor": self.actor.to_dict(),
            "description": self.description,
            "sequence_number": self.sequence_number,
            "previous_hash": self.previous_hash,
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()


class AuditLogger:
    """
    Audit logger for compliance-ready logging.

    Provides:
    - Structured audit events
    - Tamper-evident logging
    - Multiple export targets
    - Compliance reporting
    """

    def __init__(
        self,
        service_name: str = "qbitel",
        enable_tamper_detection: bool = True,
        retention_days: int = 2555,  # ~7 years for compliance
    ):
        self.service_name = service_name
        self.enable_tamper_detection = enable_tamper_detection
        self.retention_days = retention_days

        self._events: List[AuditEvent] = []
        self._sequence = 0
        self._last_hash = "genesis"
        self._lock = threading.Lock()
        self._exporters: List[Callable[[AuditEvent], None]] = []

        # Add default logging exporter
        self._logger = logging.getLogger(f"audit.{service_name}")

    def log(
        self,
        category: AuditCategory,
        action: str,
        outcome: AuditOutcome,
        actor: AuditActor,
        description: str,
        severity: AuditSeverity = AuditSeverity.INFO,
        resource: Optional[AuditResource] = None,
        correlation_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """
        Log an audit event.

        Args:
            category: Event category
            action: Action performed
            outcome: Action outcome
            actor: Who performed the action
            description: Human-readable description
            severity: Event severity
            resource: Resource acted upon
            correlation_id: Correlation ID for tracing
            trace_id: Distributed trace ID
            details: Additional details
            metadata: Metadata for indexing

        Returns:
            Created AuditEvent
        """
        with self._lock:
            self._sequence += 1

            event = AuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                category=category,
                action=action,
                outcome=outcome,
                severity=severity,
                actor=actor,
                description=description,
                resource=resource,
                correlation_id=correlation_id,
                trace_id=trace_id,
                details=details or {},
                metadata={
                    "service": self.service_name,
                    **(metadata or {}),
                },
                sequence_number=self._sequence,
                previous_hash=self._last_hash,
            )

            # Calculate tamper-evident hash
            if self.enable_tamper_detection:
                event.event_hash = event.calculate_hash()
                self._last_hash = event.event_hash

            # Store event
            self._events.append(event)

        # Export to all exporters
        self._export(event)

        # Log to standard logger
        log_level = {
            AuditSeverity.INFO: logging.INFO,
            AuditSeverity.LOW: logging.INFO,
            AuditSeverity.MEDIUM: logging.WARNING,
            AuditSeverity.HIGH: logging.WARNING,
            AuditSeverity.CRITICAL: logging.ERROR,
        }[severity]

        self._logger.log(
            log_level,
            f"[{category.value}] {action}: {outcome.value} - {description}",
            extra={"audit_event": event.to_dict()},
        )

        return event

    # Convenience methods for common audit scenarios

    def log_authentication(
        self,
        actor: AuditActor,
        outcome: AuditOutcome,
        method: str = "password",
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Log authentication event."""
        return self.log(
            category=AuditCategory.AUTHENTICATION,
            action=f"authenticate_{method}",
            outcome=outcome,
            actor=actor,
            description=f"Authentication attempt via {method}",
            severity=AuditSeverity.MEDIUM if outcome != AuditOutcome.SUCCESS else AuditSeverity.INFO,
            details=details,
        )

    def log_authorization(
        self,
        actor: AuditActor,
        resource: AuditResource,
        action: str,
        outcome: AuditOutcome,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Log authorization event."""
        return self.log(
            category=AuditCategory.AUTHORIZATION,
            action=f"authorize_{action}",
            outcome=outcome,
            actor=actor,
            description=f"Authorization check for {action} on {resource.resource_type}",
            severity=AuditSeverity.MEDIUM if outcome == AuditOutcome.DENIED else AuditSeverity.INFO,
            resource=resource,
            details=details,
        )

    def log_data_access(
        self,
        actor: AuditActor,
        resource: AuditResource,
        access_type: str = "read",
        outcome: AuditOutcome = AuditOutcome.SUCCESS,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Log data access event."""
        return self.log(
            category=AuditCategory.DATA_ACCESS,
            action=f"data_{access_type}",
            outcome=outcome,
            actor=actor,
            description=f"Data {access_type} on {resource.resource_type}:{resource.resource_id}",
            resource=resource,
            details=details,
        )

    def log_data_modification(
        self,
        actor: AuditActor,
        resource: AuditResource,
        modification_type: str,
        outcome: AuditOutcome = AuditOutcome.SUCCESS,
        before: Optional[Dict[str, Any]] = None,
        after: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Log data modification event."""
        event_details = details or {}
        if before:
            event_details["before"] = before
        if after:
            event_details["after"] = after

        return self.log(
            category=AuditCategory.DATA_MODIFICATION,
            action=f"data_{modification_type}",
            outcome=outcome,
            actor=actor,
            description=f"Data {modification_type} on {resource.resource_type}:{resource.resource_id}",
            severity=AuditSeverity.MEDIUM,
            resource=resource,
            details=event_details,
        )

    def log_crypto_operation(
        self,
        actor: AuditActor,
        operation: str,
        algorithm: str,
        outcome: AuditOutcome = AuditOutcome.SUCCESS,
        key_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Log cryptographic operation."""
        event_details = {
            "algorithm": algorithm,
            **(details or {}),
        }
        if key_id:
            event_details["key_id"] = key_id

        resource = None
        if key_id:
            resource = AuditResource(
                resource_id=key_id,
                resource_type="cryptographic_key",
            )

        return self.log(
            category=AuditCategory.CRYPTO_OPERATION,
            action=f"crypto_{operation}",
            outcome=outcome,
            actor=actor,
            description=f"Cryptographic {operation} using {algorithm}",
            severity=AuditSeverity.INFO,
            resource=resource,
            details=event_details,
        )

    def log_key_management(
        self,
        actor: AuditActor,
        action: str,
        key_id: str,
        key_type: str,
        outcome: AuditOutcome = AuditOutcome.SUCCESS,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Log key management event."""
        resource = AuditResource(
            resource_id=key_id,
            resource_type="cryptographic_key",
            attributes={"key_type": key_type},
        )

        return self.log(
            category=AuditCategory.KEY_MANAGEMENT,
            action=f"key_{action}",
            outcome=outcome,
            actor=actor,
            description=f"Key {action} for {key_type} key {key_id[:8]}...",
            severity=AuditSeverity.HIGH if action in ("destroy", "compromise") else AuditSeverity.MEDIUM,
            resource=resource,
            details=details,
        )

    def log_certificate(
        self,
        actor: AuditActor,
        action: str,
        cert_id: str,
        common_name: str,
        outcome: AuditOutcome = AuditOutcome.SUCCESS,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Log certificate event."""
        resource = AuditResource(
            resource_id=cert_id,
            resource_type="certificate",
            resource_name=common_name,
        )

        return self.log(
            category=AuditCategory.CERTIFICATE,
            action=f"cert_{action}",
            outcome=outcome,
            actor=actor,
            description=f"Certificate {action} for {common_name}",
            severity=AuditSeverity.HIGH if action == "revoke" else AuditSeverity.MEDIUM,
            resource=resource,
            details=details,
        )

    def log_security_event(
        self,
        actor: AuditActor,
        event_type: str,
        description: str,
        severity: AuditSeverity = AuditSeverity.HIGH,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Log security event."""
        return self.log(
            category=AuditCategory.SECURITY,
            action=f"security_{event_type}",
            outcome=AuditOutcome.SUCCESS,  # Security events are informational
            actor=actor,
            description=description,
            severity=severity,
            details=details,
        )

    def add_exporter(self, exporter: Callable[[AuditEvent], None]) -> None:
        """Add an audit event exporter."""
        self._exporters.append(exporter)

    def _export(self, event: AuditEvent) -> None:
        """Export event to all exporters."""
        for exporter in self._exporters:
            try:
                exporter(event)
            except Exception as e:
                self._logger.error(f"Audit export failed: {e}")

    def verify_integrity(self) -> Dict[str, Any]:
        """
        Verify integrity of audit log chain.

        Returns:
            Verification result
        """
        with self._lock:
            if not self._events:
                return {"valid": True, "events_checked": 0}

            errors = []
            expected_hash = "genesis"

            for i, event in enumerate(self._events):
                # Check sequence
                if event.sequence_number != i + 1:
                    errors.append(f"Sequence break at event {event.event_id}")

                # Check previous hash
                if event.previous_hash != expected_hash:
                    errors.append(f"Hash chain break at event {event.event_id}")

                # Verify event hash
                calculated = event.calculate_hash()
                if calculated != event.event_hash:
                    errors.append(f"Tampered event detected: {event.event_id}")

                expected_hash = event.event_hash

            return {
                "valid": len(errors) == 0,
                "events_checked": len(self._events),
                "errors": errors,
            }

    def get_events(
        self,
        category: Optional[AuditCategory] = None,
        severity: Optional[AuditSeverity] = None,
        actor_id: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """
        Query audit events.

        Args:
            category: Filter by category
            severity: Filter by minimum severity
            actor_id: Filter by actor
            since: Filter by timestamp
            limit: Maximum events to return

        Returns:
            List of matching events
        """
        with self._lock:
            events = self._events.copy()

        # Apply filters
        if category:
            events = [e for e in events if e.category == category]
        if severity:
            severity_order = list(AuditSeverity)
            min_index = severity_order.index(severity)
            events = [e for e in events if severity_order.index(e.severity) >= min_index]
        if actor_id:
            events = [e for e in events if e.actor.actor_id == actor_id]
        if since:
            events = [e for e in events if e.timestamp >= since]

        # Sort by timestamp descending and limit
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]

    def generate_compliance_report(
        self,
        framework: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, Any]:
        """
        Generate compliance report.

        Args:
            framework: Compliance framework (PCI-DSS, SOX, GDPR)
            start_date: Report start date
            end_date: Report end date

        Returns:
            Compliance report
        """
        events = [
            e for e in self._events
            if start_date <= e.timestamp <= end_date
        ]

        # Category breakdown
        by_category = {}
        for event in events:
            cat = event.category.value
            by_category[cat] = by_category.get(cat, 0) + 1

        # Outcome breakdown
        by_outcome = {}
        for event in events:
            outcome = event.outcome.value
            by_outcome[outcome] = by_outcome.get(outcome, 0) + 1

        # Severity breakdown
        by_severity = {}
        for event in events:
            sev = event.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1

        return {
            "framework": framework,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "summary": {
                "total_events": len(events),
                "by_category": by_category,
                "by_outcome": by_outcome,
                "by_severity": by_severity,
            },
            "integrity": self.verify_integrity(),
            "generated_at": datetime.utcnow().isoformat(),
        }


# Global audit logger
_audit_logger: Optional[AuditLogger] = None
_audit_lock = threading.Lock()


def get_audit_logger(service_name: str = "qbitel") -> AuditLogger:
    """Get or create the global audit logger."""
    global _audit_logger
    with _audit_lock:
        if _audit_logger is None:
            _audit_logger = AuditLogger(service_name)
        return _audit_logger


# SIEM exporter
def create_siem_exporter(
    endpoint: str,
    api_key: Optional[str] = None,
) -> Callable[[AuditEvent], None]:
    """Create a SIEM exporter function."""
    def exporter(event: AuditEvent) -> None:
        try:
            import urllib.request
            import json

            data = json.dumps(event.to_dict()).encode()
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            request = urllib.request.Request(
                endpoint,
                data=data,
                headers=headers,
                method="POST",
            )
            urllib.request.urlopen(request, timeout=5)
        except Exception:
            pass  # Don't fail on SIEM export

    return exporter
