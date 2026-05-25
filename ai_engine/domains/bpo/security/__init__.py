"""
BPO Security Module

Comprehensive security infrastructure for BPO and contact center operations including:
- Toll fraud detection and prevention (IRSF, PBX hacking, Wangiri)
- PCI-DSS voice channel compliance (DTMF masking, pause/resume recording)
- Agent session security monitoring (anomaly detection, data access tracking)
- Data loss prevention (PII detection, exfiltration prevention)
- Remote agent security (VPN-less PQC tunnels, endpoint compliance)

This module provides the security foundation for BPO operations,
supporting both traditional and post-quantum cryptography (PQC) approaches.
"""

from ai_engine.domains.bpo.security.toll_fraud import (
    TollFraudDetector,
    FraudPattern,
    TollFraudRule,
    FraudAction,
    FraudPatternType,
    FraudSeverity,
)
from ai_engine.domains.bpo.security.pci_voice import (
    PCIVoiceProtector,
    DTMFMaskingMode,
    PANDetector,
    RecordingController,
    AgentScreenMasker,
    PCIScopeManager,
    ComplianceReporter,
)
from ai_engine.domains.bpo.security.session_monitor import (
    AgentSessionMonitor,
    SessionEvent,
    AnomalyDetector,
    DataAccessPattern,
    AlertManager,
    SessionPolicy,
)
from ai_engine.domains.bpo.security.data_loss_prevention import (
    BPODataLossPreventionEngine,
    DLPRule,
    PIIDetector,
    DataType,
    DLPAction,
    ChannelMonitor,
    IncidentReporter,
)
from ai_engine.domains.bpo.security.remote_access import (
    RemoteAgentSecurityManager,
    EndpointRequirements,
    DevicePosture,
    TunnelConfig,
    ComplianceChecker,
    GeoFencePolicy,
    WatermarkGenerator,
    NetworkAssessor,
)

__all__ = [
    # Toll Fraud Detection
    "TollFraudDetector",
    "FraudPattern",
    "TollFraudRule",
    "FraudAction",
    "FraudPatternType",
    "FraudSeverity",
    # PCI Voice Security
    "PCIVoiceProtector",
    "DTMFMaskingMode",
    "PANDetector",
    "RecordingController",
    "AgentScreenMasker",
    "PCIScopeManager",
    "ComplianceReporter",
    # Session Monitoring
    "AgentSessionMonitor",
    "SessionEvent",
    "AnomalyDetector",
    "DataAccessPattern",
    "AlertManager",
    "SessionPolicy",
    # Data Loss Prevention
    "BPODataLossPreventionEngine",
    "DLPRule",
    "PIIDetector",
    "DataType",
    "DLPAction",
    "ChannelMonitor",
    "IncidentReporter",
    # Remote Access Security
    "RemoteAgentSecurityManager",
    "EndpointRequirements",
    "DevicePosture",
    "TunnelConfig",
    "ComplianceChecker",
    "GeoFencePolicy",
    "WatermarkGenerator",
    "NetworkAssessor",
]
