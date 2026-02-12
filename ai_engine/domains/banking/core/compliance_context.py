"""
Banking Compliance Context Module

Manages compliance context for banking operations including
regulatory framework tracking, audit trail, and compliance status.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4
import logging
import hashlib
import json

logger = logging.getLogger(__name__)


class ComplianceFramework(Enum):
    """Compliance frameworks supported."""

    PCI_DSS_4_0 = ("PCI-DSS", "4.0", "Payment Card Industry Data Security Standard")
    DORA = ("DORA", "1.0", "Digital Operational Resilience Act")
    BASEL_III = ("Basel", "III", "Basel III Capital Framework")
    BASEL_IV = ("Basel", "IV", "Basel IV Final Reforms")
    BCBS_239 = ("BCBS", "239", "Risk Data Aggregation and Reporting")
    GDPR = ("GDPR", "2016/679", "General Data Protection Regulation")
    SOX = ("SOX", "2002", "Sarbanes-Oxley Act")
    GLBA = ("GLBA", "1999", "Gramm-Leach-Bliley Act")
    NIST_PQC = ("NIST", "PQC", "Post-Quantum Cryptography Standards")

    def __init__(self, framework: str, version: str, description: str):
        self.framework = framework
        self.version = version
        self.description = description

    @property
    def full_name(self) -> str:
        return f"{self.framework} {self.version}"


class ComplianceStatus(Enum):
    """Compliance status for requirements."""

    COMPLIANT = auto()  # Fully compliant
    PARTIALLY_COMPLIANT = auto()  # Some gaps exist
    NON_COMPLIANT = auto()  # Not compliant
    NOT_APPLICABLE = auto()  # Requirement doesn't apply
    UNDER_REVIEW = auto()  # Being assessed
    REMEDIATION = auto()  # Remediation in progress


class ControlCategory(Enum):
    """Compliance control categories."""

    ACCESS_CONTROL = ("AC", "Access Control")
    AUDIT_LOGGING = ("AU", "Audit and Accountability")
    CONFIGURATION = ("CM", "Configuration Management")
    CRYPTOGRAPHY = ("SC", "System and Communications Protection")
    DATA_PROTECTION = ("DP", "Data Protection")
    INCIDENT_RESPONSE = ("IR", "Incident Response")
    RISK_ASSESSMENT = ("RA", "Risk Assessment")
    SECURITY_AWARENESS = ("AT", "Awareness and Training")
    VENDOR_MANAGEMENT = ("VM", "Vendor Management")

    def __init__(self, code: str, description: str):
        self.code = code
        self.description = description


@dataclass
class ComplianceControl:
    """Individual compliance control."""

    control_id: str
    framework: ComplianceFramework
    category: ControlCategory
    title: str
    description: str
    status: ComplianceStatus = ComplianceStatus.UNDER_REVIEW

    # Assessment details
    last_assessed: Optional[datetime] = None
    assessed_by: Optional[str] = None
    evidence_references: List[str] = field(default_factory=list)

    # Remediation
    remediation_plan: Optional[str] = None
    remediation_due_date: Optional[date] = None
    remediation_owner: Optional[str] = None

    # Risk
    risk_rating: str = "MEDIUM"  # LOW, MEDIUM, HIGH, CRITICAL
    compensating_controls: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "control_id": self.control_id,
            "framework": self.framework.full_name,
            "category": self.category.description,
            "title": self.title,
            "status": self.status.name,
            "risk_rating": self.risk_rating,
            "last_assessed": self.last_assessed.isoformat() if self.last_assessed else None,
        }


@dataclass
class AuditEvent:
    """Audit trail event."""

    event_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Event details
    event_type: str = ""
    event_category: str = ""
    description: str = ""

    # Actor
    user_id: Optional[str] = None
    user_role: Optional[str] = None
    source_ip: Optional[str] = None
    source_system: Optional[str] = None

    # Target
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    action: str = ""

    # Outcome
    outcome: str = "SUCCESS"  # SUCCESS, FAILURE, PARTIAL
    error_message: Optional[str] = None

    # Compliance
    compliance_frameworks: Set[ComplianceFramework] = field(default_factory=set)
    control_ids: List[str] = field(default_factory=list)

    # Data
    request_data: Optional[Dict] = None
    response_data: Optional[Dict] = None

    # Integrity
    previous_hash: Optional[str] = None
    event_hash: Optional[str] = None

    def __post_init__(self):
        """Calculate event hash for integrity."""
        self.event_hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """Calculate cryptographic hash of event."""
        data = {
            "event_id": str(self.event_id),
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "user_id": self.user_id,
            "resource_id": self.resource_id,
            "action": self.action,
            "outcome": self.outcome,
            "previous_hash": self.previous_hash,
        }
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha3_256(content.encode()).hexdigest()

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage/transmission."""
        return {
            "event_id": str(self.event_id),
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "event_category": self.event_category,
            "description": self.description,
            "user_id": self.user_id,
            "source_ip": self.source_ip,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "action": self.action,
            "outcome": self.outcome,
            "compliance_frameworks": [f.full_name for f in self.compliance_frameworks],
            "event_hash": self.event_hash,
        }


class ComplianceContext:
    """
    Manages compliance context for banking operations.

    Provides:
    - Compliance status tracking
    - Audit trail management
    - Control assessment
    - Evidence collection
    """

    def __init__(
        self,
        frameworks: Optional[Set[ComplianceFramework]] = None,
        organization_id: Optional[str] = None,
    ):
        self.frameworks = frameworks or {
            ComplianceFramework.PCI_DSS_4_0,
            ComplianceFramework.GDPR,
            ComplianceFramework.NIST_PQC,
        }
        self.organization_id = organization_id or "DEFAULT"

        self._controls: Dict[str, ComplianceControl] = {}
        self._audit_trail: List[AuditEvent] = []
        self._last_hash: Optional[str] = None

        # Initialize standard controls
        self._initialize_controls()

        logger.info(f"Compliance context initialized for frameworks: " f"{[f.full_name for f in self.frameworks]}")

    def _initialize_controls(self):
        """Initialize standard compliance controls."""
        # PCI-DSS 4.0 Controls
        if ComplianceFramework.PCI_DSS_4_0 in self.frameworks:
            self._add_pci_dss_controls()

        # DORA Controls
        if ComplianceFramework.DORA in self.frameworks:
            self._add_dora_controls()

        # NIST PQC Controls
        if ComplianceFramework.NIST_PQC in self.frameworks:
            self._add_nist_pqc_controls()

    def _add_pci_dss_controls(self):
        """Add PCI-DSS 4.0 controls."""
        controls = [
            ComplianceControl(
                control_id="PCI-3.5.1",
                framework=ComplianceFramework.PCI_DSS_4_0,
                category=ControlCategory.CRYPTOGRAPHY,
                title="Document cryptographic architecture",
                description="Document and maintain cryptographic architecture including algorithms, key strengths, and key management procedures",
            ),
            ComplianceControl(
                control_id="PCI-3.6.1",
                framework=ComplianceFramework.PCI_DSS_4_0,
                category=ControlCategory.CRYPTOGRAPHY,
                title="Strong cryptographic key generation",
                description="Procedures for generating strong cryptographic keys",
            ),
            ComplianceControl(
                control_id="PCI-3.7.1",
                framework=ComplianceFramework.PCI_DSS_4_0,
                category=ControlCategory.CRYPTOGRAPHY,
                title="Key management procedures",
                description="Key management procedures include generation, distribution, storage, and destruction",
            ),
            ComplianceControl(
                control_id="PCI-4.2.1",
                framework=ComplianceFramework.PCI_DSS_4_0,
                category=ControlCategory.CRYPTOGRAPHY,
                title="Strong cryptography for transmission",
                description="Strong cryptography is used to safeguard cardholder data during transmission",
            ),
            ComplianceControl(
                control_id="PCI-10.2.1",
                framework=ComplianceFramework.PCI_DSS_4_0,
                category=ControlCategory.AUDIT_LOGGING,
                title="Audit logs enabled",
                description="Audit logs are enabled and active for all system components",
            ),
            ComplianceControl(
                control_id="PCI-12.3.3",
                framework=ComplianceFramework.PCI_DSS_4_0,
                category=ControlCategory.CRYPTOGRAPHY,
                title="Cryptographic cipher inventories",
                description="Maintain cryptographic cipher and protocol inventories",
            ),
        ]

        for control in controls:
            self._controls[control.control_id] = control

    def _add_dora_controls(self):
        """Add DORA controls."""
        controls = [
            ComplianceControl(
                control_id="DORA-5.1",
                framework=ComplianceFramework.DORA,
                category=ControlCategory.RISK_ASSESSMENT,
                title="ICT risk management framework",
                description="Establish ICT risk management framework",
            ),
            ComplianceControl(
                control_id="DORA-17.1",
                framework=ComplianceFramework.DORA,
                category=ControlCategory.INCIDENT_RESPONSE,
                title="ICT incident management",
                description="Establish ICT-related incident management process",
            ),
            ComplianceControl(
                control_id="DORA-24.1",
                framework=ComplianceFramework.DORA,
                category=ControlCategory.RISK_ASSESSMENT,
                title="Digital operational resilience testing",
                description="Establish digital operational resilience testing program",
            ),
            ComplianceControl(
                control_id="DORA-28.1",
                framework=ComplianceFramework.DORA,
                category=ControlCategory.VENDOR_MANAGEMENT,
                title="ICT third-party risk management",
                description="Manage ICT third-party risk effectively",
            ),
        ]

        for control in controls:
            self._controls[control.control_id] = control

    def _add_nist_pqc_controls(self):
        """Add NIST PQC controls."""
        controls = [
            ComplianceControl(
                control_id="PQC-1.1",
                framework=ComplianceFramework.NIST_PQC,
                category=ControlCategory.CRYPTOGRAPHY,
                title="Cryptographic inventory",
                description="Maintain inventory of all cryptographic assets",
            ),
            ComplianceControl(
                control_id="PQC-1.2",
                framework=ComplianceFramework.NIST_PQC,
                category=ControlCategory.CRYPTOGRAPHY,
                title="Quantum risk assessment",
                description="Assess quantum computing risk to cryptographic assets",
            ),
            ComplianceControl(
                control_id="PQC-2.1",
                framework=ComplianceFramework.NIST_PQC,
                category=ControlCategory.CRYPTOGRAPHY,
                title="PQC algorithm adoption",
                description="Adopt NIST-approved PQC algorithms (ML-KEM, ML-DSA)",
            ),
            ComplianceControl(
                control_id="PQC-2.2",
                framework=ComplianceFramework.NIST_PQC,
                category=ControlCategory.CRYPTOGRAPHY,
                title="Hybrid cryptography",
                description="Implement hybrid classical/PQC cryptography during transition",
            ),
            ComplianceControl(
                control_id="PQC-3.1",
                framework=ComplianceFramework.NIST_PQC,
                category=ControlCategory.CRYPTOGRAPHY,
                title="Crypto agility",
                description="Implement crypto-agile architecture for algorithm updates",
            ),
        ]

        for control in controls:
            self._controls[control.control_id] = control

    def get_control(self, control_id: str) -> Optional[ComplianceControl]:
        """Get a specific control."""
        return self._controls.get(control_id)

    def update_control_status(
        self,
        control_id: str,
        status: ComplianceStatus,
        assessed_by: str,
        evidence: Optional[List[str]] = None,
    ) -> bool:
        """Update control compliance status."""
        control = self._controls.get(control_id)
        if not control:
            logger.warning(f"Control not found: {control_id}")
            return False

        control.status = status
        control.last_assessed = datetime.utcnow()
        control.assessed_by = assessed_by
        if evidence:
            control.evidence_references.extend(evidence)

        # Log audit event
        self.log_event(
            event_type="CONTROL_ASSESSMENT",
            event_category="COMPLIANCE",
            description=f"Control {control_id} assessed as {status.name}",
            action="UPDATE_CONTROL_STATUS",
            resource_type="ComplianceControl",
            resource_id=control_id,
        )

        logger.info(f"Control {control_id} updated to {status.name}")
        return True

    def log_event(
        self,
        event_type: str,
        event_category: str,
        description: str,
        action: str,
        user_id: Optional[str] = None,
        source_ip: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        outcome: str = "SUCCESS",
        request_data: Optional[Dict] = None,
        response_data: Optional[Dict] = None,
    ) -> AuditEvent:
        """Log an audit event."""
        event = AuditEvent(
            event_type=event_type,
            event_category=event_category,
            description=description,
            action=action,
            user_id=user_id,
            source_ip=source_ip,
            resource_type=resource_type,
            resource_id=resource_id,
            outcome=outcome,
            request_data=request_data,
            response_data=response_data,
            compliance_frameworks=self.frameworks,
            previous_hash=self._last_hash,
        )

        self._audit_trail.append(event)
        self._last_hash = event.event_hash

        logger.debug(f"Audit event logged: {event.event_id}")
        return event

    def get_audit_trail(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Query audit trail."""
        events = self._audit_trail

        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if user_id:
            events = [e for e in events if e.user_id == user_id]

        return events[-limit:]

    def verify_audit_chain(self) -> bool:
        """Verify integrity of audit trail."""
        if not self._audit_trail:
            return True

        previous_hash = None
        for event in self._audit_trail:
            if event.previous_hash != previous_hash:
                logger.error(f"Audit chain broken at event {event.event_id}")
                return False
            previous_hash = event.event_hash

        return True

    def get_compliance_summary(self) -> Dict[str, Any]:
        """Get compliance summary across all frameworks."""
        summary = {
            "organization_id": self.organization_id,
            "frameworks": [f.full_name for f in self.frameworks],
            "assessment_date": datetime.utcnow().isoformat(),
            "overall_status": "COMPLIANT",
            "controls": {
                "total": len(self._controls),
                "compliant": 0,
                "partially_compliant": 0,
                "non_compliant": 0,
                "under_review": 0,
            },
            "by_framework": {},
            "high_risk_controls": [],
        }

        # Count by status
        for control in self._controls.values():
            if control.status == ComplianceStatus.COMPLIANT:
                summary["controls"]["compliant"] += 1
            elif control.status == ComplianceStatus.PARTIALLY_COMPLIANT:
                summary["controls"]["partially_compliant"] += 1
            elif control.status == ComplianceStatus.NON_COMPLIANT:
                summary["controls"]["non_compliant"] += 1
            else:
                summary["controls"]["under_review"] += 1

            # Track high risk
            if control.risk_rating in ("HIGH", "CRITICAL"):
                summary["high_risk_controls"].append(control.to_dict())

            # Group by framework
            framework_name = control.framework.full_name
            if framework_name not in summary["by_framework"]:
                summary["by_framework"][framework_name] = {
                    "compliant": 0,
                    "non_compliant": 0,
                    "total": 0,
                }
            summary["by_framework"][framework_name]["total"] += 1
            if control.status == ComplianceStatus.COMPLIANT:
                summary["by_framework"][framework_name]["compliant"] += 1
            elif control.status == ComplianceStatus.NON_COMPLIANT:
                summary["by_framework"][framework_name]["non_compliant"] += 1

        # Determine overall status
        if summary["controls"]["non_compliant"] > 0:
            summary["overall_status"] = "NON_COMPLIANT"
        elif summary["controls"]["partially_compliant"] > 0:
            summary["overall_status"] = "PARTIALLY_COMPLIANT"
        elif summary["controls"]["under_review"] > 0:
            summary["overall_status"] = "UNDER_REVIEW"

        return summary

    def export_controls(self) -> List[Dict]:
        """Export all controls as list of dictionaries."""
        return [control.to_dict() for control in self._controls.values()]

    def export_audit_trail(self) -> List[Dict]:
        """Export audit trail as list of dictionaries."""
        return [event.to_dict() for event in self._audit_trail]
