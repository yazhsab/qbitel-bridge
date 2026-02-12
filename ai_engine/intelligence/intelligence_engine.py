"""
Intelligence Engine

The main orchestration engine for AI Protocol Intelligence.
Combines all intelligence components into a unified interface.

Capabilities:
- Automated protocol discovery and analysis
- Legacy system fingerprinting and migration planning
- Security assessment and threat detection
- Compliance mapping and reporting
- PQC migration planning
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set
import uuid

from ai_engine.intelligence.protocol_analyzer import (
    ProtocolAnalyzer,
    ProtocolAnalysisResult,
    ProtocolType,
)
from ai_engine.intelligence.legacy_fingerprinter import (
    LegacyFingerprinter,
    SystemFingerprint,
    LegacySystemType,
)
from ai_engine.intelligence.migration_planner import (
    MigrationPlanner,
    MigrationPlan,
    MigrationStrategy,
)
from ai_engine.intelligence.security_assessor import (
    SecurityAssessor,
    SecurityAssessment,
)
from ai_engine.intelligence.threat_detector import (
    ThreatDetector,
    ThreatAlert,
    ThreatLevel,
)
from ai_engine.intelligence.compliance_mapper import (
    ComplianceMapper,
    ComplianceReport,
    ComplianceFramework,
)

logger = logging.getLogger(__name__)


@dataclass
class IntelligenceConfig:
    """Configuration for Intelligence Engine."""

    # Component enablement
    enable_protocol_analysis: bool = True
    enable_fingerprinting: bool = True
    enable_security_assessment: bool = True
    enable_threat_detection: bool = True
    enable_compliance_mapping: bool = True

    # Threat detection settings
    baseline_window_size: int = 1000
    alert_callback: Optional[Callable[[ThreatAlert], None]] = None

    # Compliance frameworks to assess
    compliance_frameworks: List[ComplianceFramework] = field(
        default_factory=lambda: [
            ComplianceFramework.PCI_DSS,
            ComplianceFramework.SWIFT_CSP,
        ]
    )

    # PQC settings
    pqc_target_level: str = "level3"  # level1, level3, level5

    # Caching
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600


@dataclass
class IntelligenceResult:
    """Complete intelligence analysis result."""

    result_id: str
    analyzed_at: datetime = field(default_factory=datetime.utcnow)

    # Analysis results
    protocol_analysis: Optional[ProtocolAnalysisResult] = None
    system_fingerprint: Optional[SystemFingerprint] = None
    security_assessment: Optional[SecurityAssessment] = None
    migration_plan: Optional[MigrationPlan] = None
    compliance_reports: List[ComplianceReport] = field(default_factory=list)
    threat_alerts: List[ThreatAlert] = field(default_factory=list)

    # Summary
    protocol_type: str = "unknown"
    system_type: str = "unknown"
    pqc_ready: bool = False
    security_score: float = 0.0
    compliance_score: float = 0.0
    modernization_complexity: str = "unknown"

    # Recommendations
    recommendations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "result_id": self.result_id,
            "analyzed_at": self.analyzed_at.isoformat(),
            "summary": {
                "protocol_type": self.protocol_type,
                "system_type": self.system_type,
                "pqc_ready": self.pqc_ready,
                "security_score": self.security_score,
                "compliance_score": self.compliance_score,
                "modernization_complexity": self.modernization_complexity,
            },
            "protocol_analysis": self.protocol_analysis.to_dict() if self.protocol_analysis else None,
            "fingerprint": self.system_fingerprint.to_dict() if self.system_fingerprint else None,
            "security": self.security_assessment.to_dict() if self.security_assessment else None,
            "migration": self.migration_plan.to_dict() if self.migration_plan else None,
            "compliance": [r.to_dict() for r in self.compliance_reports],
            "alerts": [a.to_dict() for a in self.threat_alerts],
            "recommendations": self.recommendations,
        }


class IntelligenceEngine:
    """
    Main AI Protocol Intelligence Engine.

    Orchestrates all intelligence components to provide:
    - Comprehensive protocol analysis
    - Automated legacy system discovery
    - Security and compliance assessment
    - Migration planning with PQC support
    - Real-time threat detection
    """

    def __init__(self, config: Optional[IntelligenceConfig] = None):
        self._config = config or IntelligenceConfig()

        # Initialize components
        self._protocol_analyzer = ProtocolAnalyzer()
        self._fingerprinter = LegacyFingerprinter()
        self._migration_planner = MigrationPlanner()
        self._security_assessor = SecurityAssessor()
        self._threat_detector = ThreatDetector(
            baseline_window_size=self._config.baseline_window_size,
            alert_callback=self._config.alert_callback,
        )
        self._compliance_mapper = ComplianceMapper()

        # Cache
        self._cache: Dict[str, Any] = {}

        logger.info("Intelligence Engine initialized")

    def analyze(
        self,
        data: bytes,
        context: Optional[Dict[str, Any]] = None,
    ) -> IntelligenceResult:
        """
        Perform comprehensive analysis on data.

        Args:
            data: Raw data to analyze (message, file, etc.)
            context: Optional context (filename, source, etc.)

        Returns:
            IntelligenceResult with complete analysis
        """
        result = IntelligenceResult(result_id=str(uuid.uuid4()))
        context = context or {}

        try:
            # Protocol Analysis
            if self._config.enable_protocol_analysis:
                result.protocol_analysis = self._protocol_analyzer.analyze(data)
                result.protocol_type = result.protocol_analysis.protocol_type.value
                result.pqc_ready = result.protocol_analysis.pqc_ready

            # System Fingerprinting
            if self._config.enable_fingerprinting:
                result.system_fingerprint = self._fingerprinter.fingerprint(data, context)
                result.system_type = result.system_fingerprint.system_type.value
                result.modernization_complexity = result.system_fingerprint.modernization_complexity

            # Security Assessment
            if self._config.enable_security_assessment and result.protocol_analysis:
                analysis_dict = result.protocol_analysis.to_dict()
                result.security_assessment = self._security_assessor.assess(analysis_dict, context)
                result.security_score = result.security_assessment.security_score

            # Compliance Mapping
            if self._config.enable_compliance_mapping and result.protocol_analysis:
                result.compliance_reports = self._compliance_mapper.assess_compliance(
                    result.protocol_analysis.to_dict(),
                    self._config.compliance_frameworks,
                )
                if result.compliance_reports:
                    result.compliance_score = sum(r.compliance_score for r in result.compliance_reports) / len(
                        result.compliance_reports
                    )

            # Generate recommendations
            result.recommendations = self._generate_recommendations(result)

        except Exception as e:
            logger.error(f"Intelligence analysis failed: {e}")
            result.recommendations.append(
                {
                    "type": "error",
                    "message": f"Analysis error: {e}",
                }
            )

        return result

    def analyze_for_migration(
        self,
        source_data: bytes,
        target_requirements: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> IntelligenceResult:
        """
        Analyze for migration planning.

        Args:
            source_data: Source system data/schema
            target_requirements: Target system requirements
            context: Optional context

        Returns:
            IntelligenceResult with migration plan
        """
        # First, perform standard analysis
        result = self.analyze(source_data, context)

        # Generate migration plan
        if result.protocol_analysis and result.system_fingerprint:
            source_analysis = {
                **result.protocol_analysis.to_dict(),
                **result.system_fingerprint.to_dict(),
            }

            result.migration_plan = self._migration_planner.create_plan(
                source_analysis,
                target_requirements,
            )

        return result

    def analyze_message(
        self,
        message_data: bytes,
        protocol_hint: Optional[ProtocolType] = None,
        monitor_threats: bool = True,
    ) -> IntelligenceResult:
        """
        Analyze a single message with optional threat monitoring.

        Args:
            message_data: Message data
            protocol_hint: Optional protocol type hint
            monitor_threats: Whether to monitor for threats

        Returns:
            IntelligenceResult
        """
        result = IntelligenceResult(result_id=str(uuid.uuid4()))

        # Protocol analysis
        result.protocol_analysis = self._protocol_analyzer.analyze(message_data, hint=protocol_hint)
        result.protocol_type = result.protocol_analysis.protocol_type.value

        # Threat detection
        if monitor_threats and self._config.enable_threat_detection:
            # Convert to dict for threat analysis
            message_dict = {
                "protocol_type": result.protocol_type,
                "fields": {f.name: f.example for f in result.protocol_analysis.fields},
            }
            result.threat_alerts = self._threat_detector.analyze(message_dict, result.protocol_type)

        return result

    def assess_pqc_readiness(
        self,
        current_crypto: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Assess PQC migration readiness.

        Args:
            current_crypto: Current cryptographic inventory

        Returns:
            PQC readiness assessment
        """
        return self._security_assessor.assess_pqc_readiness(current_crypto)

    def create_pqc_migration_plan(
        self,
        current_crypto: Dict[str, Any],
    ) -> MigrationPlan:
        """
        Create PQC migration plan.

        Args:
            current_crypto: Current cryptographic inventory

        Returns:
            MigrationPlan for PQC migration
        """
        return self._migration_planner.create_pqc_migration_plan(
            current_crypto,
            target_security_level=self._config.pqc_target_level,
        )

    def compare_protocols(
        self,
        source_data: bytes,
        target_data: bytes,
    ) -> Dict[str, Any]:
        """
        Compare two protocols for mapping/migration.

        Args:
            source_data: Source protocol data
            target_data: Target protocol data

        Returns:
            Protocol comparison report
        """
        source_analysis = self._protocol_analyzer.analyze(source_data)
        target_analysis = self._protocol_analyzer.analyze(target_data)

        return self._protocol_analyzer.compare_protocols(source_analysis, target_analysis)

    def get_compliance_report(
        self,
        protocol_analysis: Dict[str, Any],
        frameworks: Optional[List[ComplianceFramework]] = None,
    ) -> List[ComplianceReport]:
        """
        Generate compliance reports.

        Args:
            protocol_analysis: Protocol analysis result
            frameworks: Frameworks to assess (defaults to config)

        Returns:
            List of compliance reports
        """
        frameworks = frameworks or self._config.compliance_frameworks
        return self._compliance_mapper.assess_compliance(protocol_analysis, frameworks)

    def train_threat_baseline(
        self,
        historical_data: List[Dict[str, Any]],
    ) -> None:
        """
        Train threat detection baselines from historical data.

        Args:
            historical_data: List of historical messages/transactions
        """
        # Extract amounts for baseline
        amounts = [float(d.get("amount", 0) or 0) for d in historical_data if d.get("amount") is not None]

        if amounts:
            self._threat_detector.train_baseline("amount", amounts)

        logger.info(f"Trained threat baselines from {len(historical_data)} records")

    def get_threat_summary(
        self,
        since: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get summary of threat detection activity."""
        return self._threat_detector.get_alerts_summary(since)

    def add_threat_rule(
        self,
        rule_id: str,
        name: str,
        condition: Callable[[Dict[str, Any]], bool],
        threat_level: ThreatLevel = ThreatLevel.MEDIUM,
    ) -> None:
        """
        Add a custom threat detection rule.

        Args:
            rule_id: Unique rule ID
            name: Rule name
            condition: Function that returns True if threat detected
            threat_level: Severity level
        """
        from ai_engine.intelligence.threat_detector import ThreatRule, AnomalyType

        rule = ThreatRule(
            rule_id=rule_id,
            name=name,
            description=f"Custom rule: {name}",
            anomaly_type=AnomalyType.BEHAVIORAL_ANOMALY,
            threat_level=threat_level,
            condition=condition,
        )
        self._threat_detector.add_rule(rule)

    def _generate_recommendations(
        self,
        result: IntelligenceResult,
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations from analysis results."""
        recommendations = []

        # PQC recommendation
        if not result.pqc_ready:
            recommendations.append(
                {
                    "type": "security",
                    "priority": "high",
                    "title": "Implement Post-Quantum Cryptography",
                    "description": "Current cryptographic algorithms are vulnerable to quantum attacks",
                    "actions": [
                        "Deploy ML-KEM-768 for key encapsulation",
                        "Deploy ML-DSA-65 for digital signatures",
                        "Implement hybrid classical+PQC mode during transition",
                    ],
                    "automated": True,
                    "tool": "Qbitel AI PQC Provider",
                }
            )

        # Security recommendations
        if result.security_assessment:
            if result.security_score < 70:
                recommendations.append(
                    {
                        "type": "security",
                        "priority": "critical" if result.security_score < 50 else "high",
                        "title": "Address Security Vulnerabilities",
                        "description": f"Security score ({result.security_score:.0f}) below acceptable threshold",
                        "actions": [f.title for f in result.security_assessment.findings[:5]],
                    }
                )

        # Compliance recommendations
        for report in result.compliance_reports:
            if report.compliance_score < 80:
                recommendations.append(
                    {
                        "type": "compliance",
                        "priority": "high",
                        "title": f"Address {report.framework.value} Compliance Gaps",
                        "description": f"Compliance score ({report.compliance_score:.0f}%) needs improvement",
                        "actions": report.priority_actions[:3],
                    }
                )

        # Modernization recommendation
        if result.modernization_complexity in ("high", "very_high"):
            recommendations.append(
                {
                    "type": "modernization",
                    "priority": "medium",
                    "title": "Plan System Modernization",
                    "description": f"Legacy system with {result.modernization_complexity} modernization complexity",
                    "actions": [
                        "Conduct detailed migration assessment",
                        "Identify quick-win modernization opportunities",
                        "Plan phased migration approach",
                    ],
                    "automated": True,
                    "tool": "Qbitel AI Migration Planner",
                }
            )

        # Protocol upgrade
        if result.protocol_type in ("swift_mt", "cobol_copybook", "ebcdic_fixed"):
            recommendations.append(
                {
                    "type": "protocol",
                    "priority": "medium",
                    "title": "Protocol Migration Recommended",
                    "description": f"Legacy protocol ({result.protocol_type}) should be migrated",
                    "actions": [
                        "Map to ISO 20022 format",
                        "Implement protocol adapters",
                        "Plan parallel run period",
                    ],
                    "automated": True,
                    "tool": "Qbitel AI Protocol Migrator",
                }
            )

        return recommendations

    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            "status": "operational",
            "components": {
                "protocol_analyzer": self._config.enable_protocol_analysis,
                "fingerprinter": self._config.enable_fingerprinting,
                "security_assessor": self._config.enable_security_assessment,
                "threat_detector": self._config.enable_threat_detection,
                "compliance_mapper": self._config.enable_compliance_mapping,
            },
            "config": {
                "compliance_frameworks": [f.value for f in self._config.compliance_frameworks],
                "pqc_target_level": self._config.pqc_target_level,
            },
        }


# Convenience function for quick analysis
def quick_analyze(data: bytes, context: Optional[Dict[str, Any]] = None) -> IntelligenceResult:
    """
    Quick analysis using default configuration.

    Args:
        data: Data to analyze
        context: Optional context

    Returns:
        IntelligenceResult
    """
    engine = IntelligenceEngine()
    return engine.analyze(data, context)
