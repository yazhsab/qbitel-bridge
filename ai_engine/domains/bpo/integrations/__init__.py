"""
BPO Integration Bridges

Connectors for PBX systems, CRM platforms, and Workforce Management systems.
Provides quantum-safe communication bridges between QBITEL and external
contact center infrastructure.

Supported integrations:
- PBX/ACD: Avaya Aura, Cisco CUCM, Genesys Cloud, Mitel, Asterisk, FreeSWITCH
- CRM: Salesforce, Zendesk, ServiceNow, Dynamics 365, Freshdesk, HubSpot, Zoho
- WFM: NICE WFM, Verint, Aspect, Calabrio, Genesys WFM

All integration bridges enforce:
- Post-quantum cryptographic tunneling for legacy API wrapping
- PII/PHI automatic masking during data transit
- Audit logging of all cross-system data access
- Rate limiting to prevent bulk data extraction
- Data residency enforcement per tenant policy
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__all__: List[str] = []


def _load_integration_submodules():
    """Load integration submodules based on availability."""
    global __all__

    submodules = [
        ("pbx", "PBX/ACD system connectors (Avaya, Cisco, Genesys, Asterisk)"),
        ("crm", "CRM platform bridges (Salesforce, Zendesk, ServiceNow)"),
        ("wfm", "Workforce Management bridges (NICE, Verint, Calabrio)"),
    ]

    for module_name, description in submodules:
        try:
            module = __import__(
                f"ai_engine.domains.bpo.integrations.{module_name}",
                fromlist=[module_name],
            )
            globals()[module_name] = module
            __all__.append(module_name)
            logger.debug(f"Loaded BPO integration submodule: {module_name} - {description}")
        except ImportError as e:
            logger.warning(f"Failed to load BPO integration submodule {module_name}: {e}")

    if __all__:
        logger.info(f"BPO integrations loaded with submodules: {', '.join(__all__)}")
    else:
        logger.warning("No BPO integration submodules loaded")


# Lazy loading - modules loaded on first access
_initialized = False


def _ensure_initialized():
    """Ensure submodules are loaded."""
    global _initialized
    if not _initialized:
        _load_integration_submodules()
        _initialized = True


def __getattr__(name: str):
    """Lazy load submodules on attribute access."""
    _ensure_initialized()
    if name in __all__:
        return globals()[name]
    raise AttributeError(f"module 'ai_engine.domains.bpo.integrations' has no attribute '{name}'")


# Version info
def get_version() -> str:
    """Get BPO integrations version."""
    return __version__


def get_supported_pbx_systems() -> List[str]:
    """Get list of supported PBX/ACD systems."""
    return [
        "Avaya Aura / Communication Manager (TSAPI, DMCC)",
        "Cisco Unified Communications Manager (CTI-OS, JTAPI, Finesse)",
        "Genesys Cloud / PureConnect",
        "Mitel MiVoice / MiContact Center",
        "Asterisk (AMI, ARI)",
        "FreeSWITCH (ESL)",
        "BroadSoft / Cisco BroadWorks",
        "RingCentral (REST API)",
        "Five9 (VCC API)",
    ]


def get_supported_crm_systems() -> List[str]:
    """Get list of supported CRM platforms."""
    return [
        "Salesforce Service Cloud (REST, Streaming API)",
        "Zendesk (REST API, Webhooks)",
        "ServiceNow (REST, GlideRecord)",
        "Microsoft Dynamics 365 (OData, Dataverse)",
        "Freshdesk (REST API)",
        "HubSpot (REST API)",
        "Zoho CRM (REST API)",
    ]


def get_supported_wfm_systems() -> List[str]:
    """Get list of supported WFM platforms."""
    return [
        "NICE WFM (REST API)",
        "Verint Workforce Management (REST API)",
        "Aspect Workforce Management (REST API)",
        "Calabrio WFM (REST API)",
        "Genesys WFM (Platform API)",
    ]
