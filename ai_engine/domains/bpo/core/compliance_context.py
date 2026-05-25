"""
BPO Compliance Context Module

Manages compliance context for BPO/call center operations including
regulatory framework tracking, audit trail, and compliance status.

Covers BPO-specific compliance requirements:
- PCI-DSS 4.0 for contact centers (Requirement 3, 4, 8, 10, 12)
- TCPA (Telephone Consumer Protection Act) - consent management
- HIPAA (Healthcare BPO) - PHI protection
- SOC 2 Type II - service organization controls
- GDPR - data subject rights in call recordings
- Call recording retention and disposal policies
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


class BPOComplianceFramework(Enum):
    """Compliance frameworks supported for BPO operations."""

    PCI_DSS_4_0 = ("PCI-DSS", "4.0", "Payment Card Industry Data Security Standard")
    TCPA = ("TCPA", "1991", "Telephone Consumer Protection Act")
    TSR = ("TSR", "2003", "Telemarketing Sales Rule")
    HIPAA = ("HIPAA", "1996", "Health Insurance Portability and Accountability Act")
    SOC_2 = ("SOC 2", "Type II", "Service Organization Control 2")
    GDPR = ("GDPR", "2016/679", "General Data Protection Regulation")
    CCPA = ("CCPA", "2018", "California Consumer Privacy Act")
    SOX = ("SOX", "2002", "Sarbanes-Oxley Act")
    ISO_27001 = ("ISO", "27001:2022", "Information Security Management System")
    NIST_PQC = ("NIST", "PQC", "Post-Quantum Cryptography Standards")
    FCA = ("FCA", "2022", "Financial Conduct Authority Recording Rules")

    def __init__(self, framework: str, version: str, description: str):
        self.framework = framework
        self.version = version
        self.description = description

    @property
    def full_name(self) -> str:
        return f"{self.framework} {self.version}"


class ComplianceStatus(Enum):
    """Compliance status for requirements."""

    COMPLIANT = auto()
    PARTIALLY_COMPLIANT = auto()
    NON_COMPLIANT = auto()
    NOT_APPLICABLE = auto()
    UNDER_REVIEW = auto()
    REMEDIATION = auto()


class BPOControlCategory(Enum):
    """Compliance control categories specific to BPO."""

    ACCESS_CONTROL = ("AC", "Access Control")
    AUDIT_LOGGING = ("AU", "Audit and Accountability")
    CALL_RECORDING = ("CR", "Call Recording and Retention")
    VOICE_SECURITY = ("VS", "Voice Channel Security")
    AGENT_SECURITY = ("AS", "Agent Desktop Security")
    DATA_PROTECTION = ("DP", "Data Protection and PII Handling")
    CONSENT_MANAGEMENT = ("CM", "Consent Management")
    CRYPTOGRAPHY = ("SC", "System and Communications Protection")
    INCIDENT_RESPONSE = ("IR", "Incident Response")
    REMOTE_ACCESS = ("RA", "Remote Agent Security")
    FRAUD_PREVENTION = ("FP", "Toll Fraud Prevention")
    TENANT_ISOLATION = ("TI", "Multi-Tenant Isolation")

    def __init__(self, code: str, description: str):
        self.code = code
        self.description = description


@dataclass
class ComplianceControl:
    """Individual compliance control for BPO operations."""

    control_id: str
    framework: BPOComplianceFramework
    category: BPOControlCategory
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
    risk_rating: str = "MEDIUM"
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
class BPOAuditEvent:
    """Audit trail event for BPO operations."""

    event_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Event details
    event_type: str = ""
    event_category: str = ""
    description: str = ""

    # Actor
    agent_id: Optional[str] = None
    agent_role: Optional[str] = None
    source_ip: Optional[str] = None
    source_system: Optional[str] = None
    tenant_id: Optional[str] = None

    # Call context
    call_id: Optional[str] = None
    session_id: Optional[str] = None
    campaign_id: Optional[str] = None
    queue_name: Optional[str] = None

    # Target
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    action: str = ""

    # Outcome
    outcome: str = "SUCCESS"
    error_message: Optional[str] = None

    # Compliance
    compliance_frameworks: Set[BPOComplianceFramework] = field(default_factory=set)
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
            "agent_id": self.agent_id,
            "call_id": self.call_id,
            "tenant_id": self.tenant_id,
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
            "agent_id": self.agent_id,
            "tenant_id": self.tenant_id,
            "call_id": self.call_id,
            "session_id": self.session_id,
            "source_ip": self.source_ip,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "action": self.action,
            "outcome": self.outcome,
            "compliance_frameworks": [f.full_name for f in self.compliance_frameworks],
            "event_hash": self.event_hash,
        }


class BPOComplianceContext:
    """
    Manages compliance context for BPO/call center operations.

    Provides:
    - BPO-specific compliance status tracking
    - Call-aware audit trail management
    - Control assessment for voice and agent channels
    - Evidence collection for contact center compliance
    - Multi-tenant compliance isolation
    """

    def __init__(
        self,
        frameworks: Optional[Set[BPOComplianceFramework]] = None,
        organization_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ):
        self.frameworks = frameworks or {
            BPOComplianceFramework.PCI_DSS_4_0,
            BPOComplianceFramework.GDPR,
            BPOComplianceFramework.SOC_2,
            BPOComplianceFramework.NIST_PQC,
        }
        self.organization_id = organization_id or "DEFAULT"
        self.tenant_id = tenant_id

        self._controls: Dict[str, ComplianceControl] = {}
        self._audit_trail: List[BPOAuditEvent] = []
        self._last_hash: Optional[str] = None

        # Initialize standard controls
        self._initialize_controls()

        logger.info(
            f"BPO compliance context initialized for frameworks: "
            f"{[f.full_name for f in self.frameworks]}"
        )

    def _initialize_controls(self):
        """Initialize BPO-specific compliance controls."""
        if BPOComplianceFramework.PCI_DSS_4_0 in self.frameworks:
            self._add_pci_dss_voice_controls()

        if BPOComplianceFramework.TCPA in self.frameworks:
            self._add_tcpa_controls()

        if BPOComplianceFramework.HIPAA in self.frameworks:
            self._add_hipaa_controls()

        if BPOComplianceFramework.SOC_2 in self.frameworks:
            self._add_soc2_controls()

        if BPOComplianceFramework.NIST_PQC in self.frameworks:
            self._add_nist_pqc_controls()

        if BPOComplianceFramework.GDPR in self.frameworks:
            self._add_gdpr_controls()

    def _add_pci_dss_voice_controls(self):
        """Add PCI-DSS 4.0 controls specific to voice/contact center channels."""
        controls = [
            ComplianceControl(
                control_id="PCI-CC-3.4.1",
                framework=BPOComplianceFramework.PCI_DSS_4_0,
                category=BPOControlCategory.VOICE_SECURITY,
                title="DTMF masking for cardholder data",
                description="DTMF tones containing cardholder data are masked/clamped to prevent capture in recordings and agent headsets",
            ),
            ComplianceControl(
                control_id="PCI-CC-3.5.1",
                framework=BPOComplianceFramework.PCI_DSS_4_0,
                category=BPOControlCategory.CALL_RECORDING,
                title="Call recording encryption",
                description="Call recordings containing cardholder data are encrypted with strong cryptography",
            ),
            ComplianceControl(
                control_id="PCI-CC-4.2.1",
                framework=BPOComplianceFramework.PCI_DSS_4_0,
                category=BPOControlCategory.CRYPTOGRAPHY,
                title="Voice channel encryption",
                description="Strong cryptography (SRTP-PQC) protects cardholder data during voice transmission",
            ),
            ComplianceControl(
                control_id="PCI-CC-8.3.1",
                framework=BPOComplianceFramework.PCI_DSS_4_0,
                category=BPOControlCategory.ACCESS_CONTROL,
                title="Agent authentication for payment systems",
                description="Multi-factor authentication for agents accessing cardholder data",
            ),
            ComplianceControl(
                control_id="PCI-CC-8.6.1",
                framework=BPOComplianceFramework.PCI_DSS_4_0,
                category=BPOControlCategory.AGENT_SECURITY,
                title="Agent desktop PCI compliance",
                description="Agent desktops prevent storage/display of full cardholder data; clipboard and screen capture blocked",
            ),
            ComplianceControl(
                control_id="PCI-CC-10.2.1",
                framework=BPOComplianceFramework.PCI_DSS_4_0,
                category=BPOControlCategory.AUDIT_LOGGING,
                title="Payment call audit logging",
                description="All payment-related call events are logged with cryptographic integrity",
            ),
            ComplianceControl(
                control_id="PCI-CC-12.3.3",
                framework=BPOComplianceFramework.PCI_DSS_4_0,
                category=BPOControlCategory.CRYPTOGRAPHY,
                title="Voice cryptographic inventory",
                description="Maintain inventory of voice channel cryptographic algorithms including PQC status",
            ),
        ]
        for control in controls:
            self._controls[control.control_id] = control

    def _add_tcpa_controls(self):
        """Add TCPA controls for telemarketing compliance."""
        controls = [
            ComplianceControl(
                control_id="TCPA-1.1",
                framework=BPOComplianceFramework.TCPA,
                category=BPOControlCategory.CONSENT_MANAGEMENT,
                title="Prior express consent tracking",
                description="Track and verify prior express consent for automated/prerecorded calls",
            ),
            ComplianceControl(
                control_id="TCPA-1.2",
                framework=BPOComplianceFramework.TCPA,
                category=BPOControlCategory.CONSENT_MANAGEMENT,
                title="Do-Not-Call (DNC) list compliance",
                description="Maintain internal DNC list and honor National DNC Registry",
            ),
            ComplianceControl(
                control_id="TCPA-2.1",
                framework=BPOComplianceFramework.TCPA,
                category=BPOControlCategory.CALL_RECORDING,
                title="Call time restrictions",
                description="Enforce calling time restrictions (8 AM to 9 PM local time)",
            ),
            ComplianceControl(
                control_id="TCPA-2.2",
                framework=BPOComplianceFramework.TCPA,
                category=BPOControlCategory.AUDIT_LOGGING,
                title="Consent revocation tracking",
                description="Process and log opt-out/consent revocation requests within required timeframes",
            ),
        ]
        for control in controls:
            self._controls[control.control_id] = control

    def _add_hipaa_controls(self):
        """Add HIPAA controls for healthcare BPO."""
        controls = [
            ComplianceControl(
                control_id="HIPAA-CC-1.1",
                framework=BPOComplianceFramework.HIPAA,
                category=BPOControlCategory.DATA_PROTECTION,
                title="PHI protection in voice channels",
                description="Protected Health Information is encrypted during voice transmission and in call recordings",
            ),
            ComplianceControl(
                control_id="HIPAA-CC-1.2",
                framework=BPOComplianceFramework.HIPAA,
                category=BPOControlCategory.AGENT_SECURITY,
                title="Minimum necessary PHI access",
                description="Agents access only the minimum necessary PHI for their role and current interaction",
            ),
            ComplianceControl(
                control_id="HIPAA-CC-2.1",
                framework=BPOComplianceFramework.HIPAA,
                category=BPOControlCategory.AUDIT_LOGGING,
                title="PHI access audit trail",
                description="All access to PHI is logged with agent identity, timestamp, and justification",
            ),
            ComplianceControl(
                control_id="HIPAA-CC-3.1",
                framework=BPOComplianceFramework.HIPAA,
                category=BPOControlCategory.REMOTE_ACCESS,
                title="Remote agent PHI safeguards",
                description="Remote agents handling PHI have additional safeguards (encryption, endpoint compliance)",
            ),
        ]
        for control in controls:
            self._controls[control.control_id] = control

    def _add_soc2_controls(self):
        """Add SOC 2 Type II controls for BPO operations."""
        controls = [
            ComplianceControl(
                control_id="SOC2-CC-6.1",
                framework=BPOComplianceFramework.SOC_2,
                category=BPOControlCategory.ACCESS_CONTROL,
                title="Logical access security",
                description="Logical access to systems and data is restricted based on agent roles and client requirements",
            ),
            ComplianceControl(
                control_id="SOC2-CC-6.6",
                framework=BPOComplianceFramework.SOC_2,
                category=BPOControlCategory.TENANT_ISOLATION,
                title="Multi-tenant data isolation",
                description="Client data is logically isolated across BPO tenants with separate encryption keys",
            ),
            ComplianceControl(
                control_id="SOC2-CC-7.2",
                framework=BPOComplianceFramework.SOC_2,
                category=BPOControlCategory.INCIDENT_RESPONSE,
                title="Security incident monitoring",
                description="Security events are monitored with automated detection and alerting",
            ),
        ]
        for control in controls:
            self._controls[control.control_id] = control

    def _add_nist_pqc_controls(self):
        """Add NIST PQC controls for quantum-safe BPO operations."""
        controls = [
            ComplianceControl(
                control_id="PQC-BPO-1.1",
                framework=BPOComplianceFramework.NIST_PQC,
                category=BPOControlCategory.CRYPTOGRAPHY,
                title="Voice channel PQC adoption",
                description="Adopt NIST-approved PQC algorithms for voice signaling and media encryption",
            ),
            ComplianceControl(
                control_id="PQC-BPO-1.2",
                framework=BPOComplianceFramework.NIST_PQC,
                category=BPOControlCategory.CRYPTOGRAPHY,
                title="Recording PQC encryption",
                description="Call recordings encrypted with quantum-safe algorithms for long-term protection",
            ),
            ComplianceControl(
                control_id="PQC-BPO-2.1",
                framework=BPOComplianceFramework.NIST_PQC,
                category=BPOControlCategory.CRYPTOGRAPHY,
                title="Hybrid cryptography for voice",
                description="Implement hybrid classical/PQC cryptography for voice channel transition period",
            ),
        ]
        for control in controls:
            self._controls[control.control_id] = control

    def _add_gdpr_controls(self):
        """Add GDPR controls for BPO data handling."""
        controls = [
            ComplianceControl(
                control_id="GDPR-CC-1.1",
                framework=BPOComplianceFramework.GDPR,
                category=BPOControlCategory.CONSENT_MANAGEMENT,
                title="Call recording consent",
                description="Obtain and track explicit consent for call recording under GDPR",
            ),
            ComplianceControl(
                control_id="GDPR-CC-1.2",
                framework=BPOComplianceFramework.GDPR,
                category=BPOControlCategory.DATA_PROTECTION,
                title="Data subject access requests",
                description="Process DSAR for call recordings and associated personal data within 30 days",
            ),
            ComplianceControl(
                control_id="GDPR-CC-2.1",
                framework=BPOComplianceFramework.GDPR,
                category=BPOControlCategory.CALL_RECORDING,
                title="Recording retention and deletion",
                description="Automated retention policies with secure deletion of recordings past retention period",
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

        self.log_event(
            event_type="CONTROL_ASSESSMENT",
            event_category="COMPLIANCE",
            description=f"Control {control_id} assessed as {status.name}",
            action="UPDATE_CONTROL_STATUS",
            resource_type="ComplianceControl",
            resource_id=control_id,
        )

        logger.info(f"BPO control {control_id} updated to {status.name}")
        return True

    def log_event(
        self,
        event_type: str,
        event_category: str,
        description: str,
        action: str,
        agent_id: Optional[str] = None,
        source_ip: Optional[str] = None,
        call_id: Optional[str] = None,
        session_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        outcome: str = "SUCCESS",
        request_data: Optional[Dict] = None,
        response_data: Optional[Dict] = None,
    ) -> BPOAuditEvent:
        """Log a BPO audit event."""
        event = BPOAuditEvent(
            event_type=event_type,
            event_category=event_category,
            description=description,
            action=action,
            agent_id=agent_id,
            source_ip=source_ip,
            call_id=call_id,
            session_id=session_id,
            tenant_id=self.tenant_id,
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

        logger.debug(f"BPO audit event logged: {event.event_id}")
        return event

    def log_call_event(
        self,
        call_id: str,
        event_type: str,
        agent_id: Optional[str] = None,
        description: str = "",
        **kwargs,
    ) -> BPOAuditEvent:
        """Convenience method to log a call-related event."""
        return self.log_event(
            event_type=event_type,
            event_category="CALL",
            description=description,
            action=event_type,
            agent_id=agent_id,
            call_id=call_id,
            **kwargs,
        )

    def log_dtmf_masking_event(
        self,
        call_id: str,
        agent_id: str,
        masking_mode: str = "CLAMP",
    ) -> BPOAuditEvent:
        """Log a DTMF masking event for PCI-DSS compliance."""
        return self.log_event(
            event_type="DTMF_MASKING",
            event_category="PCI",
            description=f"DTMF masking activated (mode: {masking_mode})",
            action="DTMF_MASK_ACTIVATED",
            agent_id=agent_id,
            call_id=call_id,
            resource_type="VoiceChannel",
            request_data={"masking_mode": masking_mode},
        )

    def log_recording_pause_resume(
        self,
        call_id: str,
        agent_id: str,
        action: str,  # "PAUSE" or "RESUME"
        reason: str = "PCI_PAYMENT",
    ) -> BPOAuditEvent:
        """Log recording pause/resume for PCI-DSS compliance."""
        return self.log_event(
            event_type=f"RECORDING_{action}",
            event_category="PCI",
            description=f"Call recording {action.lower()}d for {reason}",
            action=f"RECORDING_{action}",
            agent_id=agent_id,
            call_id=call_id,
            resource_type="CallRecording",
            request_data={"reason": reason},
        )

    def get_audit_trail(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_type: Optional[str] = None,
        agent_id: Optional[str] = None,
        call_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[BPOAuditEvent]:
        """Query audit trail with BPO-specific filters."""
        events = self._audit_trail

        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if agent_id:
            events = [e for e in events if e.agent_id == agent_id]
        if call_id:
            events = [e for e in events if e.call_id == call_id]

        return events[-limit:]

    def verify_audit_chain(self) -> bool:
        """Verify integrity of audit trail."""
        if not self._audit_trail:
            return True

        previous_hash = None
        for event in self._audit_trail:
            if event.previous_hash != previous_hash:
                logger.error(f"BPO audit chain broken at event {event.event_id}")
                return False
            previous_hash = event.event_hash

        return True

    def get_compliance_summary(self) -> Dict[str, Any]:
        """Get compliance summary across all BPO frameworks."""
        summary = {
            "organization_id": self.organization_id,
            "tenant_id": self.tenant_id,
            "domain": "BPO/Call Center",
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
            "by_category": {},
            "high_risk_controls": [],
        }

        for control in self._controls.values():
            if control.status == ComplianceStatus.COMPLIANT:
                summary["controls"]["compliant"] += 1
            elif control.status == ComplianceStatus.PARTIALLY_COMPLIANT:
                summary["controls"]["partially_compliant"] += 1
            elif control.status == ComplianceStatus.NON_COMPLIANT:
                summary["controls"]["non_compliant"] += 1
            else:
                summary["controls"]["under_review"] += 1

            if control.risk_rating in ("HIGH", "CRITICAL"):
                summary["high_risk_controls"].append(control.to_dict())

            # Group by framework
            framework_name = control.framework.full_name
            if framework_name not in summary["by_framework"]:
                summary["by_framework"][framework_name] = {
                    "compliant": 0, "non_compliant": 0, "total": 0,
                }
            summary["by_framework"][framework_name]["total"] += 1
            if control.status == ComplianceStatus.COMPLIANT:
                summary["by_framework"][framework_name]["compliant"] += 1
            elif control.status == ComplianceStatus.NON_COMPLIANT:
                summary["by_framework"][framework_name]["non_compliant"] += 1

            # Group by category
            category_name = control.category.description
            if category_name not in summary["by_category"]:
                summary["by_category"][category_name] = {
                    "compliant": 0, "non_compliant": 0, "total": 0,
                }
            summary["by_category"][category_name]["total"] += 1
            if control.status == ComplianceStatus.COMPLIANT:
                summary["by_category"][category_name]["compliant"] += 1

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
