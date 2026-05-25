"""
QBITEL - BPO & Call Center Domain Module

Quantum-safe security for Business Process Outsourcing and Contact Center operations.

This module provides:
- Post-quantum cryptography profiles optimized for call center workloads
- Protocol handlers for telephony systems (SIP, TN3270e, IVR, CTI)
- Voice & signaling security (toll fraud prevention, DTMF masking)
- PBX integration (Avaya, Cisco, Genesys)
- Compliance frameworks (PCI-DSS voice, TCPA, HIPAA for healthcare BPOs)
- Agent desktop security and data loss prevention
- Remote/work-from-home agent security (VPN-less quantum-safe tunnels)
- Call recording encryption and compliance
- AI-powered protocol discovery for telephony and contact center protocols
- Agentic AI with 8 specialized BPO agents (call processing, PCI, fraud, etc.)
- Zero-touch orchestration for autonomous BPO security deployment

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                  BPO Domain Module                          │
    ├──────────┬──────────┬──────────┬──────────┬────────────────┤
    │  core/   │protocols/│security/ │integr./  │  AI Layer      │
    │          │          │          │          │                │
    │ Profiles │ SIP      │ Toll     │ PBX     │ discovery/     │
    │ Policies │ TN3270e  │ Fraud    │ CRM     │  └ Fingerprint │
    │ Compli-  │ IVR      │ PCI     │ WFM     │ agents/        │
    │  ance    │ CTI      │ DLP     │          │  └ 8 Agents    │
    │          │          │ Remote  │          │ zero_touch/    │
    │          │          │ Session │          │  └ Orchestrator │
    └──────────┴──────────┴──────────┴──────────┴────────────────┘

Feature flag: QBITEL_FEATURE_BPO_DOMAIN=true
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__all__: List[str] = []


def _load_bpo_submodules():
    """Load BPO submodules based on availability."""
    global __all__

    submodules = [
        ("core", "BPO core utilities, domain profiles, and security policies"),
        ("protocols", "Telephony and terminal protocol handlers (SIP, TN3270e, IVR, CTI)"),
        ("security", "Toll fraud prevention, PCI voice masking, DLP, session monitoring"),
        ("integrations", "PBX, CRM, and WFM integration bridges"),
        ("discovery", "AI-powered protocol discovery and fingerprinting for BPO protocols"),
        ("agents", "Specialized BPO agents (call processing, PCI, fraud, CRM, recording, audit)"),
        ("zero_touch", "Zero-touch orchestration, auto-provisioning, and automation recipes"),
    ]

    for module_name, description in submodules:
        try:
            module = __import__(f"ai_engine.domains.bpo.{module_name}", fromlist=[module_name])
            globals()[module_name] = module
            __all__.append(module_name)
            logger.debug(f"Loaded BPO submodule: {module_name} - {description}")
        except ImportError as e:
            logger.warning(f"Failed to load BPO submodule {module_name}: {e}")

    if __all__:
        logger.info(f"BPO domain loaded with submodules: {', '.join(__all__)}")
    else:
        logger.warning("No BPO submodules loaded")


# Lazy loading - modules loaded on first access
_initialized = False


def _ensure_initialized():
    """Ensure submodules are loaded."""
    global _initialized
    if not _initialized:
        _load_bpo_submodules()
        _initialized = True


def __getattr__(name: str):
    """Lazy load submodules on attribute access."""
    _ensure_initialized()
    if name in __all__:
        return globals()[name]
    raise AttributeError(f"module 'ai_engine.domains.bpo' has no attribute '{name}'")


# Version info
def get_version() -> str:
    """Get BPO domain version."""
    return __version__


def get_supported_protocols() -> List[str]:
    """Get list of supported BPO/call center protocols."""
    return [
        # Voice/Telephony protocols
        "SIP (Session Initiation Protocol)",
        "SDP (Session Description Protocol)",
        "RTP/SRTP (Real-time Transport Protocol)",
        "SRTP-PQC (Quantum-safe SRTP)",
        # Terminal protocols
        "TN3270e (IBM Terminal Emulation)",
        "TN5250 (AS/400 Terminal Emulation)",
        "SSH (Secure Shell for terminal access)",
        # Signaling protocols
        "DTMF (Dual-Tone Multi-Frequency)",
        "SS7/ISUP (Legacy signaling)",
        # Contact center protocols
        "CTI (Computer Telephony Integration)",
        "TAPI (Telephony API)",
        "JTAPI (Java Telephony API)",
        "CSTA (Computer Supported Telecommunications Applications)",
        # IVR protocols
        "VXML (VoiceXML)",
        "MRCP (Media Resource Control Protocol)",
        "CCXML (Call Control eXtensible Markup Language)",
        # Integration protocols
        "SMPP (Short Message Peer-to-Peer)",
        "XMPP (Extensible Messaging and Presence Protocol)",
    ]


def get_supported_compliance_frameworks() -> List[str]:
    """Get list of supported BPO compliance frameworks."""
    return [
        "PCI-DSS 4.0 (Contact Center)",
        "TCPA (Telephone Consumer Protection Act)",
        "HIPAA (Healthcare BPO operations)",
        "SOC 2 Type II",
        "GDPR (Data Protection)",
        "ISO 27001 (Information Security)",
        "NIST 800-53 (Cybersecurity Framework)",
        "SOX (Financial Services BPO)",
        "FCA/MiFID II (UK/EU Financial Services BPO)",
        "NIST PQC (Post-Quantum Cryptography)",
    ]


def get_supported_integrations() -> List[str]:
    """Get list of supported BPO platform integrations."""
    return [
        # PBX Systems
        "Avaya Aura / Communication Manager",
        "Cisco Unified Communications Manager (CUCM)",
        "Genesys Cloud / PureConnect",
        "Mitel MiVoice / MiContact Center",
        "Asterisk / FreePBX",
        # CRM Systems
        "Salesforce Service Cloud",
        "Zendesk",
        "ServiceNow",
        "Microsoft Dynamics 365",
        "Freshdesk",
        # WFM Systems
        "NICE WFM",
        "Verint Workforce Management",
        "Aspect Workforce Management",
        "Calabrio WFM",
        # Quality Monitoring
        "NICE Quality Central",
        "Verint Quality Management",
        "Calabrio Quality Management",
    ]
