"""
QBITEL Bridge - Legacy System Whisperer (Backward Compatibility Shim)

This module provides backward compatibility by re-exporting from the
new modular services structure at ai_engine/services/legacy_whisperer/.

The Legacy Whisperer has been refactored into 4 specialized services:
- COBOLParserService: Traffic pattern analysis and protocol structure inference
- BusinessRulesExtractor: Documentation and behavior analysis
- DataFlowAnalyzer: Protocol differences and risk assessment
- ModernizationRecommender: Adapter code generation

For new code, prefer importing directly from:
    from ai_engine.services.legacy_whisperer import (
        LegacySystemWhisperer,
        COBOLParserService,
        BusinessRulesExtractor,
        DataFlowAnalyzer,
        ModernizationRecommender,
        ...
    )
"""

# Re-export everything from the new modular structure
from ..services.legacy_whisperer import (
    # Exception
    LegacyWhispererException,
    # Enums
    ProtocolComplexity,
    ModernizationRisk,
    AdapterLanguage,
    # Data classes
    ProtocolPattern,
    ProtocolField,
    ProtocolSpecification,
    AdapterCode,
    Explanation,
    # Orchestrator (main class)
    LegacySystemWhisperer,
    create_legacy_whisperer,
    # Individual services (for advanced usage)
    COBOLParserService,
    BusinessRulesExtractor,
    DataFlowAnalyzer,
    ModernizationRecommender,
)

__all__ = [
    # Exception
    "LegacyWhispererException",
    # Enums
    "ProtocolComplexity",
    "ModernizationRisk",
    "AdapterLanguage",
    # Data classes
    "ProtocolPattern",
    "ProtocolField",
    "ProtocolSpecification",
    "AdapterCode",
    "Explanation",
    # Orchestrator
    "LegacySystemWhisperer",
    "create_legacy_whisperer",
    # Individual services
    "COBOLParserService",
    "BusinessRulesExtractor",
    "DataFlowAnalyzer",
    "ModernizationRecommender",
]
