"""
QBITEL - BPO AI-Powered Protocol Discovery

Protocol fingerprinting and discovery for BPO/call center protocols.
Integrates with the platform's discovery engine to provide specialized
detection for telephony, terminal, and contact center protocols.

This module provides:
- BPO Discovery Profile: Feature extraction tuned for voice/signaling protocols
- BPO Protocol Signatures: Fingerprint database for SIP, CTI, IVR, TN3270e, etc.
- Registration hooks for the platform's ProtocolSignatureDatabase

Feature flag: QBITEL_FEATURE_BPO_DISCOVERY=true
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__all__: List[str] = []

# Lazy-loaded module references
_profile_module = None
_signatures_module = None


def _load_profile():
    """Lazy-load the BPO Discovery Profile module."""
    global _profile_module
    if _profile_module is None:
        try:
            from . import bpo_discovery_profile as mod
            _profile_module = mod
            logger.debug("Loaded BPO Discovery Profile module")
        except ImportError as e:
            logger.warning(f"Failed to load BPO Discovery Profile: {e}")
    return _profile_module


def _load_signatures():
    """Lazy-load the BPO Protocol Signatures module."""
    global _signatures_module
    if _signatures_module is None:
        try:
            from . import bpo_protocol_signatures as mod
            _signatures_module = mod
            logger.debug("Loaded BPO Protocol Signatures module")
        except ImportError as e:
            logger.warning(f"Failed to load BPO Protocol Signatures: {e}")
    return _signatures_module


def __getattr__(name: str):
    """Lazy load submodules and key classes on attribute access."""

    # Discovery profile classes
    profile_exports = {
        "BPODiscoveryProfile",
        "BPOFeatureExtractor",
        "BPOProtocolClassifier",
        "BPODiscoveryConfig",
    }

    if name in profile_exports:
        mod = _load_profile()
        if mod is not None:
            return getattr(mod, name)
        raise AttributeError(f"module 'discovery' has no attribute '{name}'")

    # Protocol signature classes
    signature_exports = {
        "BPOProtocolSignatureProvider",
        "get_bpo_signatures",
        "register_bpo_signatures",
    }

    if name in signature_exports:
        mod = _load_signatures()
        if mod is not None:
            return getattr(mod, name)
        raise AttributeError(f"module 'discovery' has no attribute '{name}'")

    # Module-level access
    if name == "bpo_discovery_profile":
        mod = _load_profile()
        if mod is not None:
            return mod
        raise AttributeError(f"module 'discovery' has no attribute '{name}'")

    if name == "bpo_protocol_signatures":
        mod = _load_signatures()
        if mod is not None:
            return mod
        raise AttributeError(f"module 'discovery' has no attribute '{name}'")

    raise AttributeError(f"module 'ai_engine.domains.bpo.discovery' has no attribute '{name}'")


def get_version() -> str:
    """Get discovery module version."""
    return __version__


def get_capabilities() -> List[str]:
    """Get list of discovery capabilities."""
    return [
        "SIP Protocol Fingerprinting (SIP/2.0, SDP, SIP-PQC-TLS)",
        "CTI Protocol Detection (CSTA, TSAPI, Finesse XML)",
        "IVR Protocol Analysis (VoiceXML, MRCP, CCXML)",
        "TN3270e Terminal Emulation Detection",
        "RTP/SRTP Media Stream Classification",
        "DTMF Tone Pattern Analysis",
        "SS7/ISUP Legacy Signaling Detection",
        "Hybrid ML + Signature Classification",
        "Real-time Voice Protocol Monitoring",
        "PBX Vendor Fingerprinting (Avaya, Cisco, Genesys)",
    ]
