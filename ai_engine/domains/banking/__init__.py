"""
QBITEL - Banking Domain Module

Quantum-secure cloud migration support for banking institutions.

This module provides:
- Post-quantum cryptography profiles optimized for banking workloads
- Protocol handlers for payment systems (ISO 8583, ISO 20022, ACH, SWIFT, etc.)
- Capital markets support (FIX protocol, market data)
- HSM integration (Thales, Utimaco, nCipher, Cloud HSMs)
- Compliance frameworks (PCI-DSS 4.0, DORA, Basel III/IV)
- Legacy system modernization (Mainframe, AS/400)

Feature flag: QBITEL_FEATURE_BANKING_DOMAIN=true
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__all__: List[str] = []


def _load_banking_submodules():
    """Load banking submodules based on availability."""
    global __all__

    submodules = [
        ("core", "Banking core utilities and profiles"),
        ("protocols", "Payment and trading protocol handlers"),
        ("security", "HSM, PKI, and key management"),
        ("compliance", "Regulatory compliance frameworks"),
        ("legacy", "Legacy system modernization"),
        ("cloud", "Multi-cloud security integration"),
        ("analytics", "Protocol discovery and analysis"),
    ]

    for module_name, description in submodules:
        try:
            module = __import__(f"ai_engine.domains.banking.{module_name}", fromlist=[module_name])
            globals()[module_name] = module
            __all__.append(module_name)
            logger.debug(f"Loaded banking submodule: {module_name} - {description}")
        except ImportError as e:
            logger.warning(f"Failed to load banking submodule {module_name}: {e}")

    if __all__:
        logger.info(f"Banking domain loaded with submodules: {', '.join(__all__)}")
    else:
        logger.warning("No banking submodules loaded")


# Lazy loading - modules loaded on first access
_initialized = False


def _ensure_initialized():
    """Ensure submodules are loaded."""
    global _initialized
    if not _initialized:
        _load_banking_submodules()
        _initialized = True


def __getattr__(name: str):
    """Lazy load submodules on attribute access."""
    _ensure_initialized()
    if name in __all__:
        return globals()[name]
    raise AttributeError(f"module 'ai_engine.domains.banking' has no attribute '{name}'")


# Version info
def get_version() -> str:
    """Get banking domain version."""
    return __version__


def get_supported_protocols() -> List[str]:
    """Get list of supported banking protocols."""
    return [
        # Payment protocols
        "ISO 8583",
        "ISO 20022 (pain.*, pacs.*, camt.*)",
        "SWIFT MT",
        "SWIFT MX",
        "ACH/NACHA",
        "FedWire",
        "FedNow",
        "SEPA (SCT, SCT Inst, SDD)",
        "CHIPS",
        # Card protocols
        "EMV",
        "3D Secure 2.0",
        # Trading protocols
        "FIX 4.2/4.4/5.0",
        "FpML",
        "FAST",
    ]


def get_supported_compliance_frameworks() -> List[str]:
    """Get list of supported compliance frameworks."""
    return [
        "PCI-DSS 4.0",
        "DORA (Digital Operational Resilience Act)",
        "Basel III/IV",
        "BCBS 239",
        "GDPR",
        "SOX",
    ]
