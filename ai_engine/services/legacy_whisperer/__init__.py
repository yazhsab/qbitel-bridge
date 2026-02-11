"""
QBITEL Bridge - Legacy Whisperer Services

Modular architecture for legacy system understanding and modernization.

Services:
- models: Shared data models and enums
- cobol_parser: Traffic pattern analysis and protocol structure inference
- business_rules: Documentation generation and behavior analysis
- data_flow: Protocol differences and risk assessment
- modernization: Adapter code generation and deployment guides
"""

from .models import (
    ProtocolComplexity,
    ModernizationRisk,
    AdapterLanguage,
    ProtocolPattern,
    ProtocolField,
    ProtocolSpecification,
    AdapterCode,
    Explanation,
    LegacyWhispererException,
)

from .cobol_parser import COBOLParserService
from .business_rules import BusinessRulesExtractor
from .data_flow import DataFlowAnalyzer
from .modernization import ModernizationRecommender
from .orchestrator import LegacySystemWhisperer, create_legacy_whisperer

__all__ = [
    # Models
    "ProtocolComplexity",
    "ModernizationRisk",
    "AdapterLanguage",
    "ProtocolPattern",
    "ProtocolField",
    "ProtocolSpecification",
    "AdapterCode",
    "Explanation",
    "LegacyWhispererException",
    # Services
    "COBOLParserService",
    "BusinessRulesExtractor",
    "DataFlowAnalyzer",
    "ModernizationRecommender",
    # Orchestrator
    "LegacySystemWhisperer",
    "create_legacy_whisperer",
]
