"""
Security Assessor

AI-powered security posture assessment for banking systems.
Evaluates cryptographic strength, identifies vulnerabilities,
and recommends remediation actions.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import uuid


logger = logging.getLogger(__name__)


class SecurityRisk(Enum):
    """Security risk severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class VulnerabilityCategory(Enum):
    """Categories of security vulnerabilities."""

    CRYPTOGRAPHIC = "cryptographic"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_EXPOSURE = "data_exposure"
    INJECTION = "injection"
    PROTOCOL = "protocol"
    CONFIGURATION = "configuration"
    QUANTUM_VULNERABLE = "quantum_vulnerable"


@dataclass
class VulnerabilityFinding:
    """A detected security vulnerability."""

    finding_id: str
    title: str
    category: VulnerabilityCategory
    severity: SecurityRisk
    description: str
    evidence: List[str] = field(default_factory=list)
    affected_components: List[str] = field(default_factory=list)
    cve_ids: List[str] = field(default_factory=list)
    cwe_id: Optional[str] = None
    cvss_score: Optional[float] = None
    exploitability: str = "unknown"
    business_impact: str = ""


@dataclass
class RemediationAction:
    """Recommended remediation action."""

    action_id: str
    finding_id: str
    title: str
    description: str
    priority: SecurityRisk
    effort: str = "medium"  # low, medium, high
    automated: bool = False
    automation_tool: Optional[str] = None
    verification_steps: List[str] = field(default_factory=list)


@dataclass
class CryptoAssessment:
    """Cryptographic assessment details."""

    algorithms_detected: List[str] = field(default_factory=list)
    quantum_safe: bool = False
    weak_algorithms: List[str] = field(default_factory=list)
    key_lengths: Dict[str, int] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    pqc_migration_urgency: str = "medium"


@dataclass
class ComplianceStatus:
    """Compliance status for a framework."""

    framework: str
    controls_assessed: int = 0
    controls_compliant: int = 0
    controls_non_compliant: int = 0
    compliance_score: float = 0.0
    gaps: List[str] = field(default_factory=list)


@dataclass
class SecurityAssessment:
    """Complete security assessment."""

    assessment_id: str
    target_name: str
    assessment_type: str

    # Overall scores
    risk_score: float = 0.0  # 0-100, higher is worse
    security_score: float = 0.0  # 0-100, higher is better
    pqc_readiness_score: float = 0.0  # 0-100

    # Findings
    findings: List[VulnerabilityFinding] = field(default_factory=list)
    remediations: List[RemediationAction] = field(default_factory=list)

    # Crypto assessment
    crypto_assessment: Optional[CryptoAssessment] = None

    # Compliance
    compliance_statuses: List[ComplianceStatus] = field(default_factory=list)

    # Summary
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0

    # Metadata
    assessed_at: datetime = field(default_factory=datetime.utcnow)
    assessment_duration_ms: float = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "assessment_id": self.assessment_id,
            "target_name": self.target_name,
            "assessment_type": self.assessment_type,
            "scores": {
                "risk_score": self.risk_score,
                "security_score": self.security_score,
                "pqc_readiness_score": self.pqc_readiness_score,
            },
            "findings_summary": {
                "critical": self.critical_count,
                "high": self.high_count,
                "medium": self.medium_count,
                "low": self.low_count,
                "total": len(self.findings),
            },
            "crypto_assessment": {
                "quantum_safe": self.crypto_assessment.quantum_safe if self.crypto_assessment else False,
                "weak_algorithms": self.crypto_assessment.weak_algorithms if self.crypto_assessment else [],
            },
            "compliance": [
                {
                    "framework": cs.framework,
                    "score": cs.compliance_score,
                    "gaps": len(cs.gaps),
                }
                for cs in self.compliance_statuses
            ],
            "assessed_at": self.assessed_at.isoformat(),
        }


class SecurityAssessor:
    """
    AI-powered security assessor.

    Evaluates:
    - Cryptographic strength and PQC readiness
    - Protocol security
    - Data protection
    - Authentication/Authorization
    - Compliance posture
    """

    def __init__(self):
        self._crypto_rules = self._build_crypto_rules()
        self._vulnerability_patterns = self._build_vulnerability_patterns()

    def assess(
        self,
        protocol_analysis: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> SecurityAssessment:
        """
        Perform security assessment.

        Args:
            protocol_analysis: Output from ProtocolAnalyzer
            context: Additional context (system type, compliance requirements)

        Returns:
            SecurityAssessment with findings and recommendations
        """
        start_time = datetime.utcnow()

        assessment = SecurityAssessment(
            assessment_id=str(uuid.uuid4()),
            target_name=protocol_analysis.get("protocol_type", "unknown"),
            assessment_type="protocol_security",
        )

        # Assess cryptography
        assessment.crypto_assessment = self._assess_cryptography(protocol_analysis)

        # Find vulnerabilities
        assessment.findings = self._find_vulnerabilities(protocol_analysis, context)

        # Generate remediations
        assessment.remediations = self._generate_remediations(assessment.findings)

        # Assess compliance
        if context and context.get("compliance_frameworks"):
            assessment.compliance_statuses = self._assess_compliance(
                protocol_analysis, context["compliance_frameworks"]
            )

        # Calculate scores
        self._calculate_scores(assessment)

        # Calculate duration
        assessment.assessment_duration_ms = (
            datetime.utcnow() - start_time
        ).total_seconds() * 1000

        return assessment

    def assess_pqc_readiness(
        self,
        current_crypto: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Assess PQC migration readiness.

        Args:
            current_crypto: Current cryptographic inventory

        Returns:
            PQC readiness report
        """
        report = {
            "readiness_score": 0,
            "quantum_vulnerable_algorithms": [],
            "migration_required": [],
            "timeline_recommendation": "",
            "recommendations": [],
        }

        # Identify quantum-vulnerable algorithms
        quantum_vulnerable = [
            "RSA", "DSA", "ECDSA", "ECDH", "DH",
            "RSA-2048", "RSA-3072", "RSA-4096",
            "P-256", "P-384", "P-521",
        ]

        for algo in current_crypto.get("algorithms", []):
            for qv in quantum_vulnerable:
                if qv.lower() in algo.lower():
                    report["quantum_vulnerable_algorithms"].append(algo)
                    report["migration_required"].append({
                        "current": algo,
                        "recommended": self._get_pqc_replacement(algo),
                    })

        # Calculate readiness score
        if current_crypto.get("pqc_algorithms"):
            report["readiness_score"] = 80
        elif not report["quantum_vulnerable_algorithms"]:
            report["readiness_score"] = 50
        else:
            vuln_ratio = len(report["quantum_vulnerable_algorithms"]) / max(
                len(current_crypto.get("algorithms", [])), 1
            )
            report["readiness_score"] = int((1 - vuln_ratio) * 40)

        # Timeline recommendation
        if report["readiness_score"] < 30:
            report["timeline_recommendation"] = "Immediate: Begin PQC migration planning"
        elif report["readiness_score"] < 60:
            report["timeline_recommendation"] = "Near-term: Implement hybrid cryptography within 12 months"
        else:
            report["timeline_recommendation"] = "On-track: Continue monitoring NIST standards"

        # Recommendations
        report["recommendations"] = [
            "Implement hybrid classical+PQC key encapsulation (ML-KEM)",
            "Deploy ML-DSA for quantum-resistant signatures",
            "Update HSM firmware for PQC support",
            "Test PQC interoperability with partners",
        ]

        return report

    def _assess_cryptography(
        self,
        protocol_analysis: Dict[str, Any],
    ) -> CryptoAssessment:
        """Assess cryptographic strength."""
        assessment = CryptoAssessment()

        # Analyze detected security characteristics
        security_chars = protocol_analysis.get("security_characteristics", [])

        for char in security_chars:
            if char.get("present"):
                algo_name = char.get("name", "")
                assessment.algorithms_detected.append(algo_name)

                # Check for weak algorithms
                weak_patterns = ["DES", "MD5", "SHA-1", "RC4", "3DES"]
                if any(weak in algo_name for weak in weak_patterns):
                    assessment.weak_algorithms.append(algo_name)

        # Check PQC readiness
        pqc_patterns = ["ML-KEM", "ML-DSA", "Kyber", "Dilithium"]
        assessment.quantum_safe = any(
            any(pqc in algo for pqc in pqc_patterns)
            for algo in assessment.algorithms_detected
        )

        if not assessment.quantum_safe:
            assessment.pqc_migration_urgency = "high"
            assessment.recommendations.extend([
                "Implement ML-KEM-768 for key encapsulation",
                "Deploy ML-DSA-65 for digital signatures",
                "Consider hybrid mode during transition",
            ])

        if assessment.weak_algorithms:
            assessment.recommendations.extend([
                f"Replace weak algorithm: {algo}"
                for algo in assessment.weak_algorithms
            ])

        return assessment

    def _find_vulnerabilities(
        self,
        protocol_analysis: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> List[VulnerabilityFinding]:
        """Find security vulnerabilities."""
        findings = []

        # Check for PCI-relevant fields without encryption
        pci_fields = protocol_analysis.get("pci_relevant_fields", [])
        if pci_fields and not protocol_analysis.get("encryption_detected"):
            findings.append(VulnerabilityFinding(
                finding_id=str(uuid.uuid4()),
                title="Unencrypted PCI Data Detected",
                category=VulnerabilityCategory.DATA_EXPOSURE,
                severity=SecurityRisk.CRITICAL,
                description="Payment card data fields detected without encryption",
                evidence=pci_fields,
                cwe_id="CWE-311",
                business_impact="PCI-DSS violation, potential card data breach",
            ))

        # Check for weak cryptography
        security_chars = protocol_analysis.get("security_characteristics", [])
        for char in security_chars:
            if char.get("strength") == "weak":
                findings.append(VulnerabilityFinding(
                    finding_id=str(uuid.uuid4()),
                    title=f"Weak Cryptographic Algorithm: {char.get('name')}",
                    category=VulnerabilityCategory.CRYPTOGRAPHIC,
                    severity=SecurityRisk.HIGH,
                    description=f"Weak algorithm {char.get('name')} provides insufficient security",
                    cwe_id="CWE-327",
                    business_impact="Data may be decrypted by attackers",
                ))

        # Check for quantum vulnerability
        if not protocol_analysis.get("pqc_ready"):
            findings.append(VulnerabilityFinding(
                finding_id=str(uuid.uuid4()),
                title="Quantum-Vulnerable Cryptography",
                category=VulnerabilityCategory.QUANTUM_VULNERABLE,
                severity=SecurityRisk.MEDIUM,
                description="Current cryptographic algorithms are vulnerable to quantum attacks",
                cwe_id="CWE-327",
                business_impact="Future quantum computers may decrypt historical data",
            ))

        # Protocol-specific vulnerabilities
        protocol_type = protocol_analysis.get("protocol_type", "")

        if protocol_type == "swift_mt":
            findings.append(VulnerabilityFinding(
                finding_id=str(uuid.uuid4()),
                title="Legacy SWIFT MT Format",
                category=VulnerabilityCategory.PROTOCOL,
                severity=SecurityRisk.LOW,
                description="SWIFT MT format being deprecated in favor of ISO 20022",
                business_impact="Migration required by industry deadlines",
            ))

        if protocol_type == "fix":
            # Check for FIX without encryption
            if not protocol_analysis.get("encryption_detected"):
                findings.append(VulnerabilityFinding(
                    finding_id=str(uuid.uuid4()),
                    title="Unencrypted FIX Protocol",
                    category=VulnerabilityCategory.DATA_EXPOSURE,
                    severity=SecurityRisk.HIGH,
                    description="FIX trading messages transmitted without encryption",
                    business_impact="Trade data exposure, market manipulation risk",
                ))

        return findings

    def _generate_remediations(
        self,
        findings: List[VulnerabilityFinding],
    ) -> List[RemediationAction]:
        """Generate remediation actions for findings."""
        remediations = []

        for finding in findings:
            remediation = self._get_remediation_for_finding(finding)
            if remediation:
                remediations.append(remediation)

        return remediations

    def _get_remediation_for_finding(
        self,
        finding: VulnerabilityFinding,
    ) -> Optional[RemediationAction]:
        """Get remediation action for a specific finding."""
        remediation_map = {
            VulnerabilityCategory.CRYPTOGRAPHIC: RemediationAction(
                action_id=str(uuid.uuid4()),
                finding_id=finding.finding_id,
                title=f"Upgrade Cryptography: {finding.title}",
                description="Replace weak cryptographic algorithm with strong alternative",
                priority=finding.severity,
                effort="medium",
                automated=True,
                automation_tool="Qbitel AI PQC Provider",
                verification_steps=[
                    "Verify new algorithm implementation",
                    "Test key operations",
                    "Validate performance",
                ],
            ),
            VulnerabilityCategory.QUANTUM_VULNERABLE: RemediationAction(
                action_id=str(uuid.uuid4()),
                finding_id=finding.finding_id,
                title="Implement Post-Quantum Cryptography",
                description="Deploy ML-KEM and ML-DSA for quantum resistance",
                priority=finding.severity,
                effort="high",
                automated=True,
                automation_tool="Qbitel AI PQC Migration",
                verification_steps=[
                    "Deploy PQC algorithms",
                    "Configure hybrid mode",
                    "Test interoperability",
                    "Update key management",
                ],
            ),
            VulnerabilityCategory.DATA_EXPOSURE: RemediationAction(
                action_id=str(uuid.uuid4()),
                finding_id=finding.finding_id,
                title="Implement Data Encryption",
                description="Encrypt sensitive data at rest and in transit",
                priority=finding.severity,
                effort="medium",
                verification_steps=[
                    "Enable TLS 1.3",
                    "Implement field-level encryption",
                    "Configure key rotation",
                ],
            ),
            VulnerabilityCategory.PROTOCOL: RemediationAction(
                action_id=str(uuid.uuid4()),
                finding_id=finding.finding_id,
                title="Protocol Migration",
                description="Migrate to modern protocol version",
                priority=finding.severity,
                effort="high",
                automated=True,
                automation_tool="Qbitel AI Protocol Migrator",
                verification_steps=[
                    "Map protocol fields",
                    "Implement transformations",
                    "Test compatibility",
                ],
            ),
        }

        return remediation_map.get(finding.category)

    def _assess_compliance(
        self,
        protocol_analysis: Dict[str, Any],
        frameworks: List[str],
    ) -> List[ComplianceStatus]:
        """Assess compliance against frameworks."""
        statuses = []

        for framework in frameworks:
            status = ComplianceStatus(framework=framework)

            if framework == "PCI-DSS":
                status = self._assess_pci_dss(protocol_analysis)
            elif framework == "SWIFT-CSP":
                status = self._assess_swift_csp(protocol_analysis)
            elif framework == "SOX":
                status = self._assess_sox(protocol_analysis)

            statuses.append(status)

        return statuses

    def _assess_pci_dss(
        self,
        protocol_analysis: Dict[str, Any],
    ) -> ComplianceStatus:
        """Assess PCI-DSS compliance."""
        status = ComplianceStatus(framework="PCI-DSS")
        status.controls_assessed = 12  # Main requirements

        gaps = []

        # Req 3: Protect stored cardholder data
        if protocol_analysis.get("pci_relevant_fields") and not protocol_analysis.get("encryption_detected"):
            gaps.append("Req 3: Cardholder data not encrypted")
        else:
            status.controls_compliant += 1

        # Req 4: Encrypt transmission
        if not protocol_analysis.get("encryption_detected"):
            gaps.append("Req 4: Transmission not encrypted")
        else:
            status.controls_compliant += 1

        # Req 8: Authentication
        if protocol_analysis.get("signature_detected"):
            status.controls_compliant += 1
        else:
            gaps.append("Req 8: Strong authentication not detected")

        status.controls_non_compliant = len(gaps)
        status.gaps = gaps
        status.compliance_score = (status.controls_compliant / status.controls_assessed) * 100

        return status

    def _assess_swift_csp(
        self,
        protocol_analysis: Dict[str, Any],
    ) -> ComplianceStatus:
        """Assess SWIFT CSP compliance."""
        status = ComplianceStatus(framework="SWIFT-CSP")
        status.controls_assessed = 7  # Mandatory controls

        gaps = []

        # Control 1.2: Operating System Privileged Accounts
        # Control 2.2: Token Management
        if not protocol_analysis.get("encryption_detected"):
            gaps.append("Control 2.2: Encryption not implemented")

        # Control 4.1: Password Policy
        # Control 5.1: Logical Access Control

        status.controls_compliant = status.controls_assessed - len(gaps)
        status.controls_non_compliant = len(gaps)
        status.gaps = gaps
        status.compliance_score = (status.controls_compliant / status.controls_assessed) * 100

        return status

    def _assess_sox(
        self,
        protocol_analysis: Dict[str, Any],
    ) -> ComplianceStatus:
        """Assess SOX compliance."""
        status = ComplianceStatus(framework="SOX")
        status.controls_assessed = 5
        status.controls_compliant = 3
        status.controls_non_compliant = 2
        status.gaps = [
            "Section 404: Audit trail requirements",
        ]
        status.compliance_score = 60.0
        return status

    def _calculate_scores(self, assessment: SecurityAssessment) -> None:
        """Calculate security scores."""
        # Count findings by severity
        for finding in assessment.findings:
            if finding.severity == SecurityRisk.CRITICAL:
                assessment.critical_count += 1
            elif finding.severity == SecurityRisk.HIGH:
                assessment.high_count += 1
            elif finding.severity == SecurityRisk.MEDIUM:
                assessment.medium_count += 1
            elif finding.severity == SecurityRisk.LOW:
                assessment.low_count += 1

        # Calculate risk score (higher is worse)
        assessment.risk_score = min(
            assessment.critical_count * 30 +
            assessment.high_count * 15 +
            assessment.medium_count * 5 +
            assessment.low_count * 1,
            100
        )

        # Calculate security score (higher is better)
        assessment.security_score = 100 - assessment.risk_score

        # Calculate PQC readiness score
        if assessment.crypto_assessment:
            if assessment.crypto_assessment.quantum_safe:
                assessment.pqc_readiness_score = 90
            elif not assessment.crypto_assessment.weak_algorithms:
                assessment.pqc_readiness_score = 50
            else:
                assessment.pqc_readiness_score = 20

    def _get_pqc_replacement(self, algorithm: str) -> str:
        """Get PQC replacement for classical algorithm."""
        replacements = {
            "RSA": "ML-KEM + ML-DSA hybrid",
            "ECDSA": "ML-DSA-65",
            "ECDH": "ML-KEM-768",
            "DH": "ML-KEM-768",
            "DSA": "ML-DSA-65",
        }

        for classic, pqc in replacements.items():
            if classic.lower() in algorithm.lower():
                return pqc

        return "ML-KEM-768 + ML-DSA-65 hybrid"

    def _build_crypto_rules(self) -> Dict[str, Dict[str, Any]]:
        """Build cryptographic assessment rules."""
        return {
            "AES-256": {"strength": "strong", "quantum_safe": False},
            "AES-128": {"strength": "moderate", "quantum_safe": False},
            "RSA-2048": {"strength": "moderate", "quantum_safe": False},
            "RSA-4096": {"strength": "strong", "quantum_safe": False},
            "DES": {"strength": "weak", "quantum_safe": False},
            "3DES": {"strength": "weak", "quantum_safe": False},
            "MD5": {"strength": "weak", "quantum_safe": False},
            "SHA-1": {"strength": "weak", "quantum_safe": False},
            "SHA-256": {"strength": "strong", "quantum_safe": False},
            "ML-KEM-768": {"strength": "strong", "quantum_safe": True},
            "ML-DSA-65": {"strength": "strong", "quantum_safe": True},
        }

    def _build_vulnerability_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Build vulnerability detection patterns."""
        return {
            "sql_injection": {
                "pattern": r"SELECT.*FROM.*WHERE.*=.*\$",
                "severity": SecurityRisk.CRITICAL,
                "cwe": "CWE-89",
            },
            "hardcoded_credentials": {
                "pattern": r"password\s*=\s*['\"][^'\"]+['\"]",
                "severity": SecurityRisk.HIGH,
                "cwe": "CWE-798",
            },
        }
