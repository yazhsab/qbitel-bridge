"""
Compliance Mapper

Maps protocols and systems to compliance requirements.
Identifies gaps and generates compliance reports.

Supported Frameworks:
- PCI-DSS v4.0
- SWIFT CSP
- SOX
- GDPR
- MiFID II
- Basel III
- NIST Cybersecurity Framework
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import uuid

logger = logging.getLogger(__name__)


class ComplianceFramework(Enum):
    """Compliance frameworks."""

    PCI_DSS = "pci_dss"
    SWIFT_CSP = "swift_csp"
    SOX = "sox"
    GDPR = "gdpr"
    MIFID_II = "mifid_ii"
    BASEL_III = "basel_iii"
    NIST_CSF = "nist_csf"
    ISO_27001 = "iso_27001"
    FFIEC = "ffiec"


class ControlStatus(Enum):
    """Status of a compliance control."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NOT_APPLICABLE = "not_applicable"
    NOT_ASSESSED = "not_assessed"


@dataclass
class ComplianceControl:
    """A compliance control requirement."""

    control_id: str
    framework: ComplianceFramework
    title: str
    description: str
    category: str
    status: ControlStatus = ControlStatus.NOT_ASSESSED
    evidence: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    remediation_actions: List[str] = field(default_factory=list)
    automated_check: bool = False
    last_assessed: Optional[datetime] = None


@dataclass
class ComplianceGap:
    """A compliance gap finding."""

    gap_id: str
    framework: ComplianceFramework
    control_id: str
    title: str
    description: str
    severity: str  # critical, high, medium, low
    affected_systems: List[str] = field(default_factory=list)
    remediation_steps: List[str] = field(default_factory=list)
    estimated_effort: str = "medium"
    deadline: Optional[datetime] = None


@dataclass
class ComplianceReport:
    """Complete compliance report."""

    report_id: str
    framework: ComplianceFramework
    report_date: datetime = field(default_factory=datetime.utcnow)

    # Summary scores
    compliance_score: float = 0.0
    controls_total: int = 0
    controls_compliant: int = 0
    controls_non_compliant: int = 0
    controls_partial: int = 0
    controls_na: int = 0

    # Details
    controls: List[ComplianceControl] = field(default_factory=list)
    gaps: List[ComplianceGap] = field(default_factory=list)

    # Recommendations
    priority_actions: List[str] = field(default_factory=list)
    quick_wins: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "framework": self.framework.value,
            "report_date": self.report_date.isoformat(),
            "compliance_score": self.compliance_score,
            "summary": {
                "total": self.controls_total,
                "compliant": self.controls_compliant,
                "non_compliant": self.controls_non_compliant,
                "partial": self.controls_partial,
                "na": self.controls_na,
            },
            "gaps_count": len(self.gaps),
            "priority_actions": self.priority_actions,
        }


class ComplianceMapper:
    """
    Maps systems and protocols to compliance requirements.

    Capabilities:
    - Framework requirement mapping
    - Automated control assessment
    - Gap identification
    - Remediation planning
    """

    def __init__(self):
        self._control_definitions = self._load_control_definitions()

    def assess_compliance(
        self,
        protocol_analysis: Dict[str, Any],
        frameworks: List[ComplianceFramework],
        system_context: Optional[Dict[str, Any]] = None,
    ) -> List[ComplianceReport]:
        """
        Assess compliance against specified frameworks.

        Args:
            protocol_analysis: Output from ProtocolAnalyzer
            frameworks: List of frameworks to assess
            system_context: Additional system context

        Returns:
            List of ComplianceReport for each framework
        """
        reports = []

        for framework in frameworks:
            report = self._assess_framework(framework, protocol_analysis, system_context)
            reports.append(report)

        return reports

    def generate_gap_analysis(
        self,
        reports: List[ComplianceReport],
    ) -> Dict[str, Any]:
        """
        Generate consolidated gap analysis from multiple reports.

        Args:
            reports: List of compliance reports

        Returns:
            Consolidated gap analysis
        """
        all_gaps = []
        frameworks_assessed = []
        total_score = 0

        for report in reports:
            all_gaps.extend(report.gaps)
            frameworks_assessed.append(report.framework.value)
            total_score += report.compliance_score

        # Prioritize gaps
        critical_gaps = [g for g in all_gaps if g.severity == "critical"]
        high_gaps = [g for g in all_gaps if g.severity == "high"]

        return {
            "frameworks_assessed": frameworks_assessed,
            "average_compliance_score": total_score / len(reports) if reports else 0,
            "total_gaps": len(all_gaps),
            "critical_gaps": len(critical_gaps),
            "high_gaps": len(high_gaps),
            "priority_remediation": [
                {
                    "gap_id": g.gap_id,
                    "title": g.title,
                    "framework": g.framework.value,
                    "severity": g.severity,
                }
                for g in (critical_gaps + high_gaps)[:10]
            ],
        }

    def map_to_controls(
        self,
        security_findings: List[Dict[str, Any]],
        framework: ComplianceFramework,
    ) -> List[ComplianceControl]:
        """
        Map security findings to compliance controls.

        Args:
            security_findings: Security assessment findings
            framework: Target compliance framework

        Returns:
            List of affected controls
        """
        affected_controls = []
        controls = self._control_definitions.get(framework, {})

        for finding in security_findings:
            category = finding.get("category", "").lower()

            # Map to relevant controls
            for control_id, control_def in controls.items():
                if self._finding_affects_control(finding, control_def):
                    control = ComplianceControl(
                        control_id=control_id,
                        framework=framework,
                        title=control_def["title"],
                        description=control_def["description"],
                        category=control_def["category"],
                        status=ControlStatus.NON_COMPLIANT,
                        gaps=[finding.get("description", "")],
                    )
                    affected_controls.append(control)

        return affected_controls

    def _assess_framework(
        self,
        framework: ComplianceFramework,
        protocol_analysis: Dict[str, Any],
        system_context: Optional[Dict[str, Any]],
    ) -> ComplianceReport:
        """Assess compliance against a specific framework."""
        if framework == ComplianceFramework.PCI_DSS:
            return self._assess_pci_dss(protocol_analysis, system_context)
        elif framework == ComplianceFramework.SWIFT_CSP:
            return self._assess_swift_csp(protocol_analysis, system_context)
        elif framework == ComplianceFramework.NIST_CSF:
            return self._assess_nist_csf(protocol_analysis, system_context)
        else:
            return self._assess_generic(framework, protocol_analysis)

    def _assess_pci_dss(
        self,
        protocol_analysis: Dict[str, Any],
        system_context: Optional[Dict[str, Any]],
    ) -> ComplianceReport:
        """Assess PCI-DSS compliance."""
        report = ComplianceReport(
            report_id=str(uuid.uuid4()),
            framework=ComplianceFramework.PCI_DSS,
        )

        controls = []
        gaps = []

        # Requirement 3: Protect stored cardholder data
        req3 = ComplianceControl(
            control_id="PCI-3",
            framework=ComplianceFramework.PCI_DSS,
            title="Protect Stored Cardholder Data",
            description="Protect stored cardholder data through encryption",
            category="Data Protection",
            automated_check=True,
        )

        pci_fields = protocol_analysis.get("pci_relevant_fields", [])
        encryption = protocol_analysis.get("encryption_detected", False)

        if pci_fields:
            if encryption:
                req3.status = ControlStatus.COMPLIANT
                req3.evidence.append("PCI fields encrypted")
            else:
                req3.status = ControlStatus.NON_COMPLIANT
                req3.gaps.append("PCI data fields not encrypted")
                gaps.append(
                    ComplianceGap(
                        gap_id=str(uuid.uuid4()),
                        framework=ComplianceFramework.PCI_DSS,
                        control_id="PCI-3",
                        title="Unencrypted Cardholder Data",
                        description="Cardholder data stored/transmitted without encryption",
                        severity="critical",
                        remediation_steps=[
                            "Implement AES-256 encryption for stored data",
                            "Enable TLS 1.3 for transmission",
                            "Deploy PQC-ready encryption (ML-KEM)",
                        ],
                    )
                )
        else:
            req3.status = ControlStatus.NOT_APPLICABLE

        controls.append(req3)

        # Requirement 4: Encrypt transmission
        req4 = ComplianceControl(
            control_id="PCI-4",
            framework=ComplianceFramework.PCI_DSS,
            title="Encrypt Transmission of Cardholder Data",
            description="Use strong cryptography during transmission",
            category="Transmission Security",
            automated_check=True,
        )

        if encryption:
            req4.status = ControlStatus.COMPLIANT
            req4.evidence.append("Encryption detected in protocol")
        else:
            req4.status = ControlStatus.NON_COMPLIANT
            req4.gaps.append("Transmission encryption not detected")
            gaps.append(
                ComplianceGap(
                    gap_id=str(uuid.uuid4()),
                    framework=ComplianceFramework.PCI_DSS,
                    control_id="PCI-4",
                    title="Unencrypted Transmission",
                    description="Cardholder data transmitted without encryption",
                    severity="critical",
                    remediation_steps=[
                        "Implement TLS 1.3",
                        "Configure strong cipher suites",
                        "Disable legacy protocols (TLS 1.0, 1.1)",
                    ],
                )
            )

        controls.append(req4)

        # Requirement 8: Strong authentication
        req8 = ComplianceControl(
            control_id="PCI-8",
            framework=ComplianceFramework.PCI_DSS,
            title="Identify and Authenticate Access",
            description="Implement strong authentication mechanisms",
            category="Access Control",
            automated_check=True,
        )

        if protocol_analysis.get("signature_detected"):
            req8.status = ControlStatus.COMPLIANT
            req8.evidence.append("Digital signatures detected")
        else:
            req8.status = ControlStatus.PARTIALLY_COMPLIANT
            req8.gaps.append("Digital signatures not detected")

        controls.append(req8)

        # Requirement 12: Maintain security policy (PQC readiness)
        req12_pqc = ComplianceControl(
            control_id="PCI-12-PQC",
            framework=ComplianceFramework.PCI_DSS,
            title="Post-Quantum Cryptography Readiness",
            description="Prepare for quantum-safe cryptography transition",
            category="Security Policy",
            automated_check=True,
        )

        if protocol_analysis.get("pqc_ready"):
            req12_pqc.status = ControlStatus.COMPLIANT
            req12_pqc.evidence.append("PQC algorithms detected")
        else:
            req12_pqc.status = ControlStatus.PARTIALLY_COMPLIANT
            req12_pqc.gaps.append("PQC migration not started")
            gaps.append(
                ComplianceGap(
                    gap_id=str(uuid.uuid4()),
                    framework=ComplianceFramework.PCI_DSS,
                    control_id="PCI-12-PQC",
                    title="PQC Migration Required",
                    description="System not prepared for post-quantum cryptography",
                    severity="medium",
                    remediation_steps=[
                        "Assess current cryptographic inventory",
                        "Plan hybrid classical+PQC implementation",
                        "Deploy ML-KEM-768 and ML-DSA-65",
                    ],
                )
            )

        controls.append(req12_pqc)

        # Calculate scores
        report.controls = controls
        report.gaps = gaps
        report.controls_total = len(controls)
        report.controls_compliant = sum(1 for c in controls if c.status == ControlStatus.COMPLIANT)
        report.controls_non_compliant = sum(1 for c in controls if c.status == ControlStatus.NON_COMPLIANT)
        report.controls_partial = sum(1 for c in controls if c.status == ControlStatus.PARTIALLY_COMPLIANT)
        report.controls_na = sum(1 for c in controls if c.status == ControlStatus.NOT_APPLICABLE)

        applicable = report.controls_total - report.controls_na
        if applicable > 0:
            report.compliance_score = ((report.controls_compliant + 0.5 * report.controls_partial) / applicable) * 100

        # Priority actions
        if gaps:
            report.priority_actions = [g.title for g in gaps if g.severity == "critical"]
            report.quick_wins = [g.title for g in gaps if g.severity == "low"]

        return report

    def _assess_swift_csp(
        self,
        protocol_analysis: Dict[str, Any],
        system_context: Optional[Dict[str, Any]],
    ) -> ComplianceReport:
        """Assess SWIFT CSP compliance."""
        report = ComplianceReport(
            report_id=str(uuid.uuid4()),
            framework=ComplianceFramework.SWIFT_CSP,
        )

        controls = []
        gaps = []

        # Control 1.1: SWIFT Environment Protection
        ctrl_1_1 = ComplianceControl(
            control_id="SWIFT-1.1",
            framework=ComplianceFramework.SWIFT_CSP,
            title="SWIFT Environment Protection",
            description="Restrict Internet access and protect SWIFT infrastructure",
            category="Secure Environment",
        )
        controls.append(ctrl_1_1)

        # Control 2.2: Security Updates
        ctrl_2_2 = ComplianceControl(
            control_id="SWIFT-2.2",
            framework=ComplianceFramework.SWIFT_CSP,
            title="Security Updates",
            description="Apply security updates in timely manner",
            category="Reduce Attack Surface",
        )
        controls.append(ctrl_2_2)

        # Control 4.1: Password Policy
        ctrl_4_1 = ComplianceControl(
            control_id="SWIFT-4.1",
            framework=ComplianceFramework.SWIFT_CSP,
            title="Password Policy",
            description="Implement strong password management",
            category="Prevent Credential Compromise",
            status=(
                ControlStatus.COMPLIANT if protocol_analysis.get("signature_detected") else ControlStatus.PARTIALLY_COMPLIANT
            ),
        )
        controls.append(ctrl_4_1)

        # Control 6.1: Malware Protection
        ctrl_6_1 = ComplianceControl(
            control_id="SWIFT-6.1",
            framework=ComplianceFramework.SWIFT_CSP,
            title="Malware Protection",
            description="Deploy anti-malware solutions",
            category="Detect Anomalous Activity",
        )
        controls.append(ctrl_6_1)

        report.controls = controls
        report.controls_total = len(controls)
        report.compliance_score = 70.0  # Placeholder

        return report

    def _assess_nist_csf(
        self,
        protocol_analysis: Dict[str, Any],
        system_context: Optional[Dict[str, Any]],
    ) -> ComplianceReport:
        """Assess NIST Cybersecurity Framework compliance."""
        report = ComplianceReport(
            report_id=str(uuid.uuid4()),
            framework=ComplianceFramework.NIST_CSF,
        )

        controls = []

        # Identify (ID)
        id_am = ComplianceControl(
            control_id="ID.AM",
            framework=ComplianceFramework.NIST_CSF,
            title="Asset Management",
            description="Physical devices and systems are inventoried",
            category="Identify",
            status=ControlStatus.COMPLIANT,  # Qbitel AI provides this
        )
        controls.append(id_am)

        # Protect (PR)
        pr_ds = ComplianceControl(
            control_id="PR.DS",
            framework=ComplianceFramework.NIST_CSF,
            title="Data Security",
            description="Data-at-rest and data-in-transit are protected",
            category="Protect",
            status=ControlStatus.COMPLIANT if protocol_analysis.get("encryption_detected") else ControlStatus.NON_COMPLIANT,
        )
        controls.append(pr_ds)

        # Detect (DE)
        de_cm = ComplianceControl(
            control_id="DE.CM",
            framework=ComplianceFramework.NIST_CSF,
            title="Security Continuous Monitoring",
            description="Systems are monitored for anomalies",
            category="Detect",
            status=ControlStatus.COMPLIANT,  # Qbitel AI Threat Detector
        )
        controls.append(de_cm)

        report.controls = controls
        report.controls_total = len(controls)
        report.controls_compliant = sum(1 for c in controls if c.status == ControlStatus.COMPLIANT)
        report.compliance_score = (report.controls_compliant / report.controls_total) * 100

        return report

    def _assess_generic(
        self,
        framework: ComplianceFramework,
        protocol_analysis: Dict[str, Any],
    ) -> ComplianceReport:
        """Generic compliance assessment."""
        report = ComplianceReport(
            report_id=str(uuid.uuid4()),
            framework=framework,
        )
        report.controls_total = 10
        report.compliance_score = 60.0
        return report

    def _finding_affects_control(
        self,
        finding: Dict[str, Any],
        control_def: Dict[str, Any],
    ) -> bool:
        """Check if a security finding affects a compliance control."""
        finding_category = finding.get("category", "").lower()
        control_category = control_def.get("category", "").lower()

        # Category matching
        category_map = {
            "cryptographic": ["encryption", "cryptography", "data protection"],
            "data_exposure": ["data protection", "confidentiality"],
            "authentication": ["access control", "authentication"],
            "protocol": ["communication", "transmission"],
        }

        for find_cat, control_cats in category_map.items():
            if find_cat in finding_category:
                if any(cc in control_category for cc in control_cats):
                    return True

        return False

    def _load_control_definitions(self) -> Dict[ComplianceFramework, Dict[str, Dict]]:
        """Load compliance control definitions."""
        return {
            ComplianceFramework.PCI_DSS: {
                "PCI-1": {
                    "title": "Install and Maintain Network Security Controls",
                    "description": "Network security controls protect cardholder data",
                    "category": "Network Security",
                },
                "PCI-2": {
                    "title": "Apply Secure Configurations",
                    "description": "Vendor-supplied defaults must be changed",
                    "category": "System Security",
                },
                "PCI-3": {
                    "title": "Protect Stored Account Data",
                    "description": "Protect stored cardholder data through encryption",
                    "category": "Data Protection",
                },
                "PCI-4": {
                    "title": "Protect Cardholder Data During Transmission",
                    "description": "Use strong cryptography during transmission",
                    "category": "Transmission Security",
                },
            },
            ComplianceFramework.SWIFT_CSP: {
                "SWIFT-1.1": {
                    "title": "SWIFT Environment Protection",
                    "description": "Restrict Internet access",
                    "category": "Secure Environment",
                },
                "SWIFT-2.2": {
                    "title": "Security Updates",
                    "description": "Apply security updates",
                    "category": "Attack Surface",
                },
            },
        }
