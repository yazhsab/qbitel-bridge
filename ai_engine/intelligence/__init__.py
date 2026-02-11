"""
AI Protocol Intelligence Engine

The core intelligence layer for Qbitel AI that provides:
- Protocol discovery and analysis
- Legacy system fingerprinting
- Migration path recommendations
- Security posture assessment
- Automated threat detection
- Compliance mapping
- PQC migration planning

Components:
- ProtocolAnalyzer: Deep analysis of protocol structures and behaviors
- LegacyFingerprinter: Identifies legacy system characteristics
- MigrationPlanner: Plans migration paths from legacy to modern
- SecurityAssessor: Evaluates security posture and risks
- ComplianceMapper: Maps protocols to compliance requirements
- ThreatDetector: Real-time threat detection and anomaly analysis
"""

from ai_engine.intelligence.protocol_analyzer import (
    ProtocolAnalyzer,
    ProtocolAnalysisResult,
    ProtocolType,
    ProtocolCapability,
)
from ai_engine.intelligence.legacy_fingerprinter import (
    LegacyFingerprinter,
    SystemFingerprint,
    LegacySystemType,
    FingerprintConfidence,
)
from ai_engine.intelligence.migration_planner import (
    MigrationPlanner,
    MigrationPlan,
    MigrationPhase,
    MigrationRisk,
    MigrationStrategy,
)
from ai_engine.intelligence.security_assessor import (
    SecurityAssessor,
    SecurityAssessment,
    VulnerabilityFinding,
    SecurityRisk,
    RemediationAction,
)
from ai_engine.intelligence.threat_detector import (
    ThreatDetector,
    ThreatAlert,
    ThreatLevel,
    AnomalyType,
)
from ai_engine.intelligence.compliance_mapper import (
    ComplianceMapper,
    ComplianceReport,
    ComplianceFramework,
    ComplianceControl,
    ComplianceGap,
)
from ai_engine.intelligence.intelligence_engine import (
    IntelligenceEngine,
    IntelligenceConfig,
    IntelligenceResult,
)

__all__ = [
    # Protocol Analysis
    "ProtocolAnalyzer",
    "ProtocolAnalysisResult",
    "ProtocolType",
    "ProtocolCapability",
    # Legacy Fingerprinting
    "LegacyFingerprinter",
    "SystemFingerprint",
    "LegacySystemType",
    "FingerprintConfidence",
    # Migration Planning
    "MigrationPlanner",
    "MigrationPlan",
    "MigrationPhase",
    "MigrationRisk",
    "MigrationStrategy",
    # Security Assessment
    "SecurityAssessor",
    "SecurityAssessment",
    "VulnerabilityFinding",
    "SecurityRisk",
    "RemediationAction",
    # Threat Detection
    "ThreatDetector",
    "ThreatAlert",
    "ThreatLevel",
    "AnomalyType",
    # Compliance
    "ComplianceMapper",
    "ComplianceReport",
    "ComplianceFramework",
    "ComplianceControl",
    "ComplianceGap",
    # Main Engine
    "IntelligenceEngine",
    "IntelligenceConfig",
    "IntelligenceResult",
]
