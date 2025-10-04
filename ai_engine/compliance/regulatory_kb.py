"""
CRONOS AI - Regulatory Knowledge Base

Comprehensive regulatory framework knowledge for enterprise compliance reporting.
Supports PCI-DSS 4.0, Basel III, HIPAA, NERC CIP, and FDA medical device regulations.
"""

import asyncio
import logging
import json
import hashlib
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

from ..llm.unified_llm_service import UnifiedLLMService, LLMRequest, get_llm_service
from ..core.config import Config
from ..core.exceptions import CronosAIException

logger = logging.getLogger(__name__)


class ComplianceException(CronosAIException):
    """Compliance-specific exception."""

    pass


class RequirementSeverity(Enum):
    """Severity levels for compliance requirements."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ControlType(Enum):
    """Types of compliance controls."""

    ADMINISTRATIVE = "administrative"
    TECHNICAL = "technical"
    PHYSICAL = "physical"
    OPERATIONAL = "operational"


@dataclass
class ComplianceRequirement:
    """Individual compliance requirement."""

    id: str
    title: str
    description: str
    severity: RequirementSeverity
    control_type: ControlType
    section: str
    subsection: Optional[str] = None
    validation_criteria: List[str] = field(default_factory=list)
    evidence_required: List[str] = field(default_factory=list)
    automated_checks: List[str] = field(default_factory=list)
    remediation_guidance: str = ""
    references: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ComplianceAssessment:
    """Results of compliance assessment."""

    framework: str
    version: str
    assessment_date: datetime
    overall_compliance_score: float
    compliant_requirements: int
    non_compliant_requirements: int
    partially_compliant_requirements: int
    not_assessed_requirements: int
    gaps: List["ComplianceGap"]
    recommendations: List["ComplianceRecommendation"]
    risk_score: float
    next_assessment_due: datetime
    assessment_id: str = field(
        default_factory=lambda: hashlib.md5(
            str(datetime.utcnow()).encode()
        ).hexdigest()[:12]
    )


@dataclass
class ComplianceGap:
    """Identified compliance gap."""

    requirement_id: str
    requirement_title: str
    severity: RequirementSeverity
    current_state: str
    required_state: str
    gap_description: str
    impact_assessment: str
    remediation_effort: str  # low, medium, high
    estimated_cost: Optional[float] = None
    target_completion_date: Optional[datetime] = None


@dataclass
class ComplianceRecommendation:
    """Compliance improvement recommendation."""

    id: str
    title: str
    description: str
    priority: RequirementSeverity
    implementation_steps: List[str]
    estimated_effort_days: int
    cost_estimate: Optional[float] = None
    business_impact: str
    technical_requirements: List[str] = field(default_factory=list)
    success_metrics: List[str] = field(default_factory=list)


class ComplianceFramework(ABC):
    """Abstract base class for compliance frameworks."""

    def __init__(self, version: str):
        self.version = version
        self.name = self.__class__.__name__.replace("Framework", "")
        self.requirements: Dict[str, ComplianceRequirement] = {}
        self.sections: Dict[str, List[str]] = {}
        self.load_requirements()

    @abstractmethod
    def load_requirements(self) -> None:
        """Load framework-specific requirements."""
        pass

    def get_requirement(self, req_id: str) -> Optional[ComplianceRequirement]:
        """Get specific requirement by ID."""
        return self.requirements.get(req_id)

    def get_requirements_by_section(self, section: str) -> List[ComplianceRequirement]:
        """Get all requirements for a specific section."""
        return [req for req in self.requirements.values() if req.section == section]

    def get_requirements_by_severity(
        self, severity: RequirementSeverity
    ) -> List[ComplianceRequirement]:
        """Get requirements by severity level."""
        return [req for req in self.requirements.values() if req.severity == severity]

    def get_all_requirements(self) -> List[ComplianceRequirement]:
        """Get all requirements in this framework."""
        return list(self.requirements.values())


class PCIDSSFramework(ComplianceFramework):
    """PCI-DSS 4.0 compliance framework."""

    def __init__(self):
        super().__init__("4.0")

    def load_requirements(self) -> None:
        """Load PCI-DSS 4.0 requirements."""
        # Build and maintain a secure network and systems
        self.requirements["1.1"] = ComplianceRequirement(
            id="1.1",
            title="Install and maintain network security controls",
            description="Establish, implement, and maintain network security controls (NSCs) to protect the CDE.",
            severity=RequirementSeverity.CRITICAL,
            control_type=ControlType.TECHNICAL,
            section="1",
            subsection="1.1",
            validation_criteria=[
                "Network security controls are installed and configured",
                "Network segmentation isolates CDE from other networks",
                "Firewall rules restrict traffic to necessary communications",
            ],
            evidence_required=[
                "Network architecture diagram",
                "Firewall configuration files",
                "Network security policy documentation",
            ],
            automated_checks=[
                "firewall_config_scan",
                "network_segmentation_test",
                "port_vulnerability_scan",
            ],
            remediation_guidance="Implement proper network segmentation using firewalls, routers, or other security devices.",
        )

        self.requirements["2.1"] = ComplianceRequirement(
            id="2.1",
            title="Change all vendor-supplied default passwords",
            description="Always change vendor-supplied defaults before installing a system component on the network.",
            severity=RequirementSeverity.HIGH,
            control_type=ControlType.ADMINISTRATIVE,
            section="2",
            subsection="2.1",
            validation_criteria=[
                "Default passwords changed on all systems",
                "Default accounts removed or disabled",
                "System hardening procedures followed",
            ],
            evidence_required=[
                "System configuration documentation",
                "Change logs for default credentials",
                "Hardening checklists",
            ],
            automated_checks=[
                "default_password_scan",
                "default_account_check",
                "system_hardening_validation",
            ],
            remediation_guidance="Change all default passwords and remove or disable unnecessary default accounts.",
        )

        # Protect stored cardholder data
        self.requirements["3.1"] = ComplianceRequirement(
            id="3.1",
            title="Keep cardholder data storage to a minimum",
            description="Limit cardholder data storage and retention time to that which is required for legal, regulatory, and/or business requirements.",
            severity=RequirementSeverity.CRITICAL,
            control_type=ControlType.ADMINISTRATIVE,
            section="3",
            subsection="3.1",
            validation_criteria=[
                "Data retention policy implemented",
                "Unnecessary data purged regularly",
                "Business justification for stored data",
            ],
            evidence_required=[
                "Data retention policy",
                "Data inventory documentation",
                "Purging procedures and logs",
            ],
            automated_checks=[
                "data_inventory_scan",
                "retention_compliance_check",
                "unnecessary_data_detection",
            ],
            remediation_guidance="Implement data retention policy and regularly purge unnecessary cardholder data.",
        )

        # Encrypt transmission of cardholder data
        self.requirements["4.1"] = ComplianceRequirement(
            id="4.1",
            title="Use strong cryptography and security protocols",
            description="Use strong cryptography and security protocols to safeguard sensitive cardholder data during transmission over open, public networks.",
            severity=RequirementSeverity.CRITICAL,
            control_type=ControlType.TECHNICAL,
            section="4",
            subsection="4.1",
            validation_criteria=[
                "Strong encryption for data in transit",
                "TLS 1.2 or higher for web applications",
                "VPN or equivalent for remote access",
            ],
            evidence_required=[
                "Encryption implementation documentation",
                "SSL/TLS configuration files",
                "Network traffic analysis",
            ],
            automated_checks=[
                "ssl_configuration_scan",
                "encryption_strength_test",
                "protocol_version_check",
            ],
            remediation_guidance="Implement strong encryption protocols (TLS 1.2+) for all cardholder data transmissions.",
        )

        self.sections = {
            "1": ["Network Security"],
            "2": ["System Configuration"],
            "3": ["Data Protection"],
            "4": ["Encryption"],
            "5": ["Anti-virus"],
            "6": ["Secure Development"],
            "7": ["Access Control"],
            "8": ["Identity Management"],
            "9": ["Physical Security"],
            "10": ["Logging and Monitoring"],
            "11": ["Security Testing"],
            "12": ["Information Security Policy"],
        }


class BaselIIIFramework(ComplianceFramework):
    """Basel III compliance framework for financial institutions."""

    def __init__(self):
        super().__init__("2022")

    def load_requirements(self) -> None:
        """Load Basel III requirements."""
        # Capital Requirements
        self.requirements["CAP.1"] = ComplianceRequirement(
            id="CAP.1",
            title="Minimum Capital Requirements",
            description="Banks must maintain minimum capital ratios as specified by Basel III standards.",
            severity=RequirementSeverity.CRITICAL,
            control_type=ControlType.OPERATIONAL,
            section="Capital",
            subsection="Minimum Requirements",
            validation_criteria=[
                "Common Equity Tier 1 ratio ≥ 4.5%",
                "Tier 1 capital ratio ≥ 6%",
                "Total capital ratio ≥ 8%",
            ],
            evidence_required=[
                "Capital adequacy reports",
                "Regulatory capital calculations",
                "Stress test results",
            ],
            automated_checks=[
                "capital_ratio_calculation",
                "regulatory_reporting_validation",
                "stress_test_compliance",
            ],
            remediation_guidance="Maintain adequate capital buffers and implement capital planning processes.",
        )

        # Liquidity Requirements
        self.requirements["LIQ.1"] = ComplianceRequirement(
            id="LIQ.1",
            title="Liquidity Coverage Ratio",
            description="Banks must maintain high-quality liquid assets to cover net cash outflows under stress scenarios.",
            severity=RequirementSeverity.CRITICAL,
            control_type=ControlType.OPERATIONAL,
            section="Liquidity",
            subsection="LCR",
            validation_criteria=[
                "LCR ≥ 100%",
                "High-quality liquid assets adequately maintained",
                "Net cash outflow calculations accurate",
            ],
            evidence_required=[
                "LCR calculations and reports",
                "Liquid asset portfolio documentation",
                "Cash flow projections",
            ],
            automated_checks=[
                "lcr_calculation_validation",
                "liquid_asset_quality_check",
                "cash_flow_stress_test",
            ],
            remediation_guidance="Maintain sufficient high-quality liquid assets and monitor LCR daily.",
        )

        # Risk Management
        self.requirements["RISK.1"] = ComplianceRequirement(
            id="RISK.1",
            title="Credit Risk Management",
            description="Implement comprehensive credit risk management framework.",
            severity=RequirementSeverity.HIGH,
            control_type=ControlType.OPERATIONAL,
            section="Risk Management",
            subsection="Credit Risk",
            validation_criteria=[
                "Credit risk policies established",
                "Risk assessment procedures implemented",
                "Regular portfolio stress testing",
            ],
            evidence_required=[
                "Credit risk management policy",
                "Risk assessment documentation",
                "Stress testing results",
            ],
            automated_checks=[
                "credit_policy_compliance",
                "risk_assessment_validation",
                "portfolio_concentration_check",
            ],
            remediation_guidance="Establish robust credit risk management processes and regular monitoring.",
        )

        self.sections = {
            "Capital": ["Minimum Capital Requirements", "Capital Buffers"],
            "Liquidity": ["LCR", "NSFR", "Monitoring Tools"],
            "Risk Management": ["Credit Risk", "Market Risk", "Operational Risk"],
            "Leverage": ["Leverage Ratio"],
            "Disclosure": ["Pillar 3 Requirements"],
        }


class HIPAAFramework(ComplianceFramework):
    """HIPAA compliance framework for healthcare organizations."""

    def __init__(self):
        super().__init__("2013")

    def load_requirements(self) -> None:
        """Load HIPAA requirements."""
        # Administrative Safeguards
        self.requirements["164.308(a)(1)"] = ComplianceRequirement(
            id="164.308(a)(1)",
            title="Security Officer",
            description="Assign security responsibilities to an individual with appropriate authority.",
            severity=RequirementSeverity.HIGH,
            control_type=ControlType.ADMINISTRATIVE,
            section="Administrative Safeguards",
            subsection="Security Management Process",
            validation_criteria=[
                "Security officer designated",
                "Security responsibilities documented",
                "Authority and accountability established",
            ],
            evidence_required=[
                "Security officer designation document",
                "Job description and responsibilities",
                "Organizational chart",
            ],
            automated_checks=["security_officer_validation", "role_assignment_check"],
            remediation_guidance="Designate a qualified security officer and document their responsibilities.",
        )

        # Physical Safeguards
        self.requirements["164.310(a)(1)"] = ComplianceRequirement(
            id="164.310(a)(1)",
            title="Facility Access Controls",
            description="Implement policies and procedures to limit physical access to electronic information systems.",
            severity=RequirementSeverity.HIGH,
            control_type=ControlType.PHYSICAL,
            section="Physical Safeguards",
            subsection="Facility Access Controls",
            validation_criteria=[
                "Physical access controls implemented",
                "Access logging and monitoring",
                "Visitor access procedures",
            ],
            evidence_required=[
                "Physical security policy",
                "Access control system documentation",
                "Visitor logs and procedures",
            ],
            automated_checks=["physical_access_audit", "facility_security_scan"],
            remediation_guidance="Implement comprehensive physical access controls and monitoring.",
        )

        # Technical Safeguards
        self.requirements["164.312(a)(1)"] = ComplianceRequirement(
            id="164.312(a)(1)",
            title="Access Control",
            description="Implement technical policies and procedures for electronic information systems.",
            severity=RequirementSeverity.CRITICAL,
            control_type=ControlType.TECHNICAL,
            section="Technical Safeguards",
            subsection="Access Control",
            validation_criteria=[
                "Unique user identification assigned",
                "Role-based access controls implemented",
                "Access authorization procedures",
            ],
            evidence_required=[
                "Access control policy",
                "User access documentation",
                "Role-based access matrix",
            ],
            automated_checks=[
                "user_access_audit",
                "role_validation_check",
                "unauthorized_access_detection",
            ],
            remediation_guidance="Implement comprehensive technical access controls and user authentication.",
        )

        self.sections = {
            "Administrative Safeguards": [
                "Security Management",
                "Access Management",
                "Workforce Training",
            ],
            "Physical Safeguards": [
                "Facility Access",
                "Workstation Use",
                "Device Controls",
            ],
            "Technical Safeguards": [
                "Access Control",
                "Audit Controls",
                "Integrity",
                "Transmission Security",
            ],
        }


class NERCCIPFramework(ComplianceFramework):
    """NERC CIP compliance framework for critical infrastructure protection."""

    def __init__(self):
        super().__init__("014")

    def load_requirements(self) -> None:
        """Load NERC CIP requirements."""
        # Cyber Security Organization
        self.requirements["CIP-003-8"] = ComplianceRequirement(
            id="CIP-003-8",
            title="Cyber Security - Security Management Controls",
            description="Specify consistent and sustainable security management controls.",
            severity=RequirementSeverity.CRITICAL,
            control_type=ControlType.ADMINISTRATIVE,
            section="CIP-003",
            subsection="Security Management",
            validation_criteria=[
                "Cyber security policy documented",
                "Leadership commitment established",
                "Delegated authority defined",
            ],
            evidence_required=[
                "Cyber security policy document",
                "Senior manager designation",
                "Authority delegation documentation",
            ],
            automated_checks=["policy_compliance_check", "authority_validation"],
            remediation_guidance="Develop comprehensive cyber security management program with clear leadership.",
        )

        # Electronic Security Perimeters
        self.requirements["CIP-005-6"] = ComplianceRequirement(
            id="CIP-005-6",
            title="Cyber Security - Electronic Security Perimeter(s)",
            description="Manage electronic access to BES Cyber Systems by specifying a controlled Electronic Security Perimeter.",
            severity=RequirementSeverity.CRITICAL,
            control_type=ControlType.TECHNICAL,
            section="CIP-005",
            subsection="Electronic Security Perimeters",
            validation_criteria=[
                "Electronic Security Perimeters defined",
                "Electronic Access Control implemented",
                "Remote access controls established",
            ],
            evidence_required=[
                "ESP documentation and diagrams",
                "Access control documentation",
                "Remote access procedures",
            ],
            automated_checks=[
                "esp_boundary_validation",
                "access_point_audit",
                "remote_access_compliance",
            ],
            remediation_guidance="Implement secure electronic perimeters with appropriate access controls.",
        )

        # Physical Security
        self.requirements["CIP-006-6"] = ComplianceRequirement(
            id="CIP-006-6",
            title="Cyber Security - Physical Security of BES Cyber Systems",
            description="Manage physical access to BES Cyber Systems by specifying a controlled Physical Security Perimeter.",
            severity=RequirementSeverity.HIGH,
            control_type=ControlType.PHYSICAL,
            section="CIP-006",
            subsection="Physical Security",
            validation_criteria=[
                "Physical Security Perimeters defined",
                "Physical access controls implemented",
                "Monitoring and logging systems active",
            ],
            evidence_required=[
                "PSP documentation",
                "Physical access control procedures",
                "Monitoring system documentation",
            ],
            automated_checks=[
                "physical_perimeter_audit",
                "access_control_validation",
                "monitoring_system_check",
            ],
            remediation_guidance="Establish robust physical security controls for critical cyber systems.",
        )

        self.sections = {
            "CIP-002": ["Cyber Asset Categorization"],
            "CIP-003": ["Security Management Controls"],
            "CIP-004": ["Personnel & Training"],
            "CIP-005": ["Electronic Security Perimeters"],
            "CIP-006": ["Physical Security"],
            "CIP-007": ["System Security Management"],
            "CIP-008": ["Incident Reporting"],
            "CIP-009": ["Recovery Plans"],
            "CIP-010": ["Configuration Change Management"],
            "CIP-011": ["Information Protection"],
            "CIP-013": ["Supply Chain Risk Management"],
            "CIP-014": ["Physical Security"],
        }


class FDAMedicalFramework(ComplianceFramework):
    """FDA medical device regulations compliance framework."""

    def __init__(self):
        super().__init__("2022")

    def load_requirements(self) -> None:
        """Load FDA medical device requirements."""
        # Quality Management System
        self.requirements["820.20"] = ComplianceRequirement(
            id="820.20",
            title="Management Responsibility",
            description="Management with executive responsibility shall establish quality policy and objectives.",
            severity=RequirementSeverity.HIGH,
            control_type=ControlType.ADMINISTRATIVE,
            section="820.20",
            subsection="Management Responsibility",
            validation_criteria=[
                "Quality policy established",
                "Quality objectives defined",
                "Management review process implemented",
            ],
            evidence_required=[
                "Quality policy documentation",
                "Quality objectives documentation",
                "Management review records",
            ],
            automated_checks=[
                "quality_policy_validation",
                "management_review_compliance",
            ],
            remediation_guidance="Establish comprehensive quality management system with clear policies.",
        )

        # Design Controls
        self.requirements["820.30"] = ComplianceRequirement(
            id="820.30",
            title="Design Controls",
            description="Each manufacturer shall establish and maintain procedures to control the design of the device.",
            severity=RequirementSeverity.CRITICAL,
            control_type=ControlType.TECHNICAL,
            section="820.30",
            subsection="Design Controls",
            validation_criteria=[
                "Design control procedures established",
                "Design inputs and outputs documented",
                "Design verification and validation performed",
            ],
            evidence_required=[
                "Design control procedures",
                "Design history file",
                "Verification and validation records",
            ],
            automated_checks=["design_control_audit", "design_documentation_check"],
            remediation_guidance="Implement comprehensive design control processes for medical devices.",
        )

        # Risk Management
        self.requirements["ISO14971"] = ComplianceRequirement(
            id="ISO14971",
            title="Risk Management for Medical Devices",
            description="Application of risk management to medical devices per ISO 14971.",
            severity=RequirementSeverity.CRITICAL,
            control_type=ControlType.OPERATIONAL,
            section="Risk Management",
            subsection="ISO 14971",
            validation_criteria=[
                "Risk management process established",
                "Risk analysis conducted",
                "Risk control measures implemented",
            ],
            evidence_required=[
                "Risk management plan",
                "Risk analysis documentation",
                "Risk control verification",
            ],
            automated_checks=["risk_management_compliance", "risk_analysis_validation"],
            remediation_guidance="Implement ISO 14971 compliant risk management for medical devices.",
        )

        self.sections = {
            "820.20": ["Management Responsibility"],
            "820.25": ["Personnel"],
            "820.30": ["Design Controls"],
            "820.40": ["Document Controls"],
            "820.50": ["Purchasing Controls"],
            "820.70": ["Production and Process Controls"],
            "820.80": ["Identification and Traceability"],
            "820.90": ["Nonconforming Product"],
            "820.100": ["Corrective and Preventive Action"],
            "820.180": ["General Requirements Records"],
            "820.198": ["Complaint Files"],
        }


class RegulatoryKnowledgeBase:
    """Comprehensive regulatory framework knowledge management system."""

    def __init__(self, config: Config, llm_service: Optional[UnifiedLLMService] = None):
        self.config = config
        self.llm_service = llm_service or get_llm_service()
        self.logger = logging.getLogger(__name__)

        # Initialize frameworks
        self.frameworks = {
            "PCI_DSS_4_0": PCIDSSFramework(),
            "BASEL_III": BaselIIIFramework(),
            "HIPAA": HIPAAFramework(),
            "NERC_CIP": NERCCIPFramework(),
            "FDA_MEDICAL": FDAMedicalFramework(),
        }

        # Framework metadata
        self.framework_metadata = {
            "PCI_DSS_4_0": {
                "full_name": "Payment Card Industry Data Security Standard v4.0",
                "applicable_to": [
                    "Payment processors",
                    "E-commerce",
                    "Financial services",
                ],
                "last_updated": "2022-03-31",
                "next_review": "2024-03-31",
            },
            "BASEL_III": {
                "full_name": "Basel III International Regulatory Framework",
                "applicable_to": ["Banks", "Financial institutions", "Credit unions"],
                "last_updated": "2022-01-01",
                "next_review": "2025-01-01",
            },
            "HIPAA": {
                "full_name": "Health Insurance Portability and Accountability Act",
                "applicable_to": [
                    "Healthcare providers",
                    "Health plans",
                    "Healthcare clearinghouses",
                ],
                "last_updated": "2013-01-25",
                "next_review": "2024-01-25",
            },
            "NERC_CIP": {
                "full_name": "NERC Critical Infrastructure Protection",
                "applicable_to": [
                    "Electric utilities",
                    "Power generation",
                    "Grid operators",
                ],
                "last_updated": "2021-10-01",
                "next_review": "2024-10-01",
            },
            "FDA_MEDICAL": {
                "full_name": "FDA Medical Device Regulations",
                "applicable_to": [
                    "Medical device manufacturers",
                    "Healthcare technology",
                ],
                "last_updated": "2022-09-01",
                "next_review": "2025-09-01",
            },
        }

        self.logger.info(
            "Regulatory Knowledge Base initialized with %d frameworks",
            len(self.frameworks),
        )

    async def assess_compliance(
        self, system_data: Dict[str, Any], framework: str
    ) -> ComplianceAssessment:
        """
        Assess system compliance against specified regulatory framework.

        Args:
            system_data: System configuration and state data
            framework: Framework identifier (e.g., 'PCI_DSS_4_0')

        Returns:
            Comprehensive compliance assessment
        """
        try:
            if framework not in self.frameworks:
                raise ComplianceException(f"Unknown framework: {framework}")

            framework_instance = self.frameworks[framework]
            self.logger.info(f"Starting compliance assessment for {framework}")

            # Collect relevant compliance data
            compliance_data = await self._collect_compliance_data(
                system_data, framework_instance.requirements
            )

            # LLM-powered gap analysis
            gap_analysis = await self._perform_llm_gap_analysis(
                compliance_data, framework_instance, framework
            )

            # Calculate compliance metrics
            assessment = await self._calculate_compliance_metrics(
                gap_analysis, framework_instance, framework
            )

            self.logger.info(
                f"Compliance assessment completed for {framework}: {assessment.overall_compliance_score:.2f}%"
            )
            return assessment

        except Exception as e:
            self.logger.error(f"Compliance assessment failed for {framework}: {e}")
            raise ComplianceException(f"Assessment failed: {e}")

    async def _collect_compliance_data(
        self,
        system_data: Dict[str, Any],
        requirements: Dict[str, ComplianceRequirement],
    ) -> Dict[str, Any]:
        """Collect and organize compliance-relevant data."""
        compliance_data = {
            "system_configuration": system_data.get("configuration", {}),
            "security_controls": system_data.get("security", {}),
            "access_controls": system_data.get("access", {}),
            "network_config": system_data.get("network", {}),
            "data_handling": system_data.get("data", {}),
            "monitoring": system_data.get("monitoring", {}),
            "policies": system_data.get("policies", {}),
            "training": system_data.get("training", {}),
            "physical_security": system_data.get("physical", {}),
            "incident_response": system_data.get("incidents", {}),
            "requirements_count": len(requirements),
            "framework_sections": list(
                set(req.section for req in requirements.values())
            ),
        }

        return compliance_data

    async def _perform_llm_gap_analysis(
        self,
        compliance_data: Dict[str, Any],
        framework_instance: ComplianceFramework,
        framework: str,
    ) -> Dict[str, Any]:
        """Use LLM to perform intelligent gap analysis."""
        try:
            # Create comprehensive prompt for gap analysis
            prompt = self._create_gap_analysis_prompt(
                compliance_data, framework_instance, framework
            )

            # Request LLM analysis
            llm_request = LLMRequest(
                prompt=prompt,
                feature_domain="compliance_reporter",
                context={
                    "framework": framework,
                    "requirements_count": len(framework_instance.requirements),
                    "system_data_keys": list(compliance_data.keys()),
                },
                max_tokens=3000,
                temperature=0.2,  # Low temperature for consistent compliance analysis
            )

            response = await self.llm_service.process_request(llm_request)

            # Parse LLM response
            try:
                gap_analysis = json.loads(response.content)
            except json.JSONDecodeError:
                # Fallback parsing if JSON is malformed
                gap_analysis = self._parse_llm_text_response(response.content)

            return gap_analysis

        except Exception as e:
            self.logger.error(f"LLM gap analysis failed: {e}")
            # Return basic analysis if LLM fails
            return self._fallback_gap_analysis(compliance_data, framework_instance)

    def _create_gap_analysis_prompt(
        self,
        compliance_data: Dict[str, Any],
        framework_instance: ComplianceFramework,
        framework: str,
    ) -> str:
        """Create detailed prompt for LLM gap analysis."""
        metadata = self.framework_metadata.get(framework, {})

        prompt = f"""
You are a compliance expert analyzing {metadata.get('full_name', framework)} compliance.

FRAMEWORK: {framework}
REQUIREMENTS: {len(framework_instance.requirements)} total requirements
SECTIONS: {list(framework_instance.sections.keys())}

SYSTEM DATA ANALYSIS:
{json.dumps(compliance_data, indent=2)}

TASK: Perform a comprehensive compliance gap analysis and provide:

1. COMPLIANCE SCORE (0-100): Overall compliance percentage
2. GAPS: List of identified compliance gaps with:
   - Requirement ID and title
   - Severity (critical/high/medium/low)
   - Current state vs required state
   - Gap description and impact
   - Remediation effort estimate
3. RECOMMENDATIONS: Priority-ordered recommendations with:
   - Implementation steps
   - Estimated effort (days)
   - Business impact assessment
   - Technical requirements

Return response as JSON with this structure:
{{
  "compliance_score": <0-100>,
  "compliant_count": <number>,
  "non_compliant_count": <number>,
  "partially_compliant_count": <number>,
  "risk_score": <0-100>,
  "gaps": [
    {{
      "requirement_id": "string",
      "requirement_title": "string", 
      "severity": "critical|high|medium|low",
      "current_state": "string",
      "required_state": "string",
      "gap_description": "string",
      "impact_assessment": "string",
      "remediation_effort": "low|medium|high"
    }}
  ],
  "recommendations": [
    {{
      "id": "string",
      "title": "string",
      "description": "string",
      "priority": "critical|high|medium|low",
      "implementation_steps": ["step1", "step2"],
      "estimated_effort_days": <number>,
      "business_impact": "string",
      "technical_requirements": ["req1", "req2"]
    }}
  ]
}}

Focus on practical, actionable insights for enterprise-grade compliance improvement.
"""
        return prompt

    def _parse_llm_text_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM text response when JSON parsing fails."""
        # Basic fallback parsing - in production would be more sophisticated
        return {
            "compliance_score": 50.0,
            "compliant_count": 0,
            "non_compliant_count": 0,
            "partially_compliant_count": 0,
            "risk_score": 50.0,
            "gaps": [],
            "recommendations": [],
            "analysis_text": response_text,
        }

    def _fallback_gap_analysis(
        self, compliance_data: Dict[str, Any], framework_instance: ComplianceFramework
    ) -> Dict[str, Any]:
        """Provide basic gap analysis when LLM is unavailable."""
        total_requirements = len(framework_instance.requirements)

        return {
            "compliance_score": 25.0,  # Conservative estimate
            "compliant_count": 0,
            "non_compliant_count": total_requirements,
            "partially_compliant_count": 0,
            "risk_score": 75.0,
            "gaps": [
                {
                    "requirement_id": req.id,
                    "requirement_title": req.title,
                    "severity": req.severity.value,
                    "current_state": "Not assessed",
                    "required_state": "Full compliance required",
                    "gap_description": f"Unable to assess compliance for {req.title}",
                    "impact_assessment": "Unknown impact - manual review required",
                    "remediation_effort": "medium",
                }
                for req in list(framework_instance.requirements.values())[
                    :5
                ]  # Limit to first 5
            ],
            "recommendations": [
                {
                    "id": "manual_review",
                    "title": "Manual Compliance Review Required",
                    "description": "Automated analysis unavailable - conduct manual review",
                    "priority": "high",
                    "implementation_steps": [
                        "Engage compliance expert",
                        "Review requirements manually",
                    ],
                    "estimated_effort_days": 30,
                    "business_impact": "Risk of non-compliance",
                    "technical_requirements": ["Compliance expertise"],
                }
            ],
        }

    async def _calculate_compliance_metrics(
        self,
        gap_analysis: Dict[str, Any],
        framework_instance: ComplianceFramework,
        framework: str,
    ) -> ComplianceAssessment:
        """Calculate final compliance assessment metrics."""
        now = datetime.utcnow()

        # Parse gaps and recommendations
        gaps = []
        for gap_data in gap_analysis.get("gaps", []):
            gaps.append(
                ComplianceGap(
                    requirement_id=gap_data.get("requirement_id", ""),
                    requirement_title=gap_data.get("requirement_title", ""),
                    severity=RequirementSeverity(gap_data.get("severity", "medium")),
                    current_state=gap_data.get("current_state", ""),
                    required_state=gap_data.get("required_state", ""),
                    gap_description=gap_data.get("gap_description", ""),
                    impact_assessment=gap_data.get("impact_assessment", ""),
                    remediation_effort=gap_data.get("remediation_effort", "medium"),
                )
            )

        recommendations = []
        for rec_data in gap_analysis.get("recommendations", []):
            recommendations.append(
                ComplianceRecommendation(
                    id=rec_data.get("id", ""),
                    title=rec_data.get("title", ""),
                    description=rec_data.get("description", ""),
                    priority=RequirementSeverity(rec_data.get("priority", "medium")),
                    implementation_steps=rec_data.get("implementation_steps", []),
                    estimated_effort_days=rec_data.get("estimated_effort_days", 5),
                    business_impact=rec_data.get("business_impact", ""),
                    technical_requirements=rec_data.get("technical_requirements", []),
                )
            )

        return ComplianceAssessment(
            framework=framework,
            version=framework_instance.version,
            assessment_date=now,
            overall_compliance_score=gap_analysis.get("compliance_score", 0.0),
            compliant_requirements=gap_analysis.get("compliant_count", 0),
            non_compliant_requirements=gap_analysis.get("non_compliant_count", 0),
            partially_compliant_requirements=gap_analysis.get(
                "partially_compliant_count", 0
            ),
            not_assessed_requirements=0,
            gaps=gaps,
            recommendations=recommendations,
            risk_score=gap_analysis.get("risk_score", 100.0),
            next_assessment_due=now + timedelta(days=90),  # Quarterly assessments
        )

    def get_framework(self, framework_name: str) -> Optional[ComplianceFramework]:
        """Get specific compliance framework."""
        return self.frameworks.get(framework_name)

    def get_available_frameworks(self) -> List[str]:
        """Get list of available compliance frameworks."""
        return list(self.frameworks.keys())

    def get_framework_metadata(self, framework_name: str) -> Dict[str, Any]:
        """Get metadata for specific framework."""
        return self.framework_metadata.get(framework_name, {})

    async def get_compliance_summary(self, framework: str) -> Dict[str, Any]:
        """Get high-level compliance framework summary."""
        if framework not in self.frameworks:
            raise ComplianceException(f"Unknown framework: {framework}")

        framework_instance = self.frameworks[framework]
        metadata = self.framework_metadata[framework]

        return {
            "framework": framework,
            "name": metadata["full_name"],
            "version": framework_instance.version,
            "total_requirements": len(framework_instance.requirements),
            "sections": list(framework_instance.sections.keys()),
            "critical_requirements": len(
                framework_instance.get_requirements_by_severity(
                    RequirementSeverity.CRITICAL
                )
            ),
            "high_requirements": len(
                framework_instance.get_requirements_by_severity(
                    RequirementSeverity.HIGH
                )
            ),
            "applicable_to": metadata["applicable_to"],
            "last_updated": metadata["last_updated"],
            "next_review": metadata["next_review"],
        }
