"""
QBITEL - BPO Specialized Agents

Agentic AI agents for autonomous BPO/call center security operations.
Each agent inherits from the platform's BaseAgent and provides specialized
capabilities for different aspects of contact center security.

This module provides:
- 8 Specialized BPO Agents:
  1. CallProcessingAgent - Real-time call security monitoring
  2. PCIComplianceAgent - PCI-DSS voice compliance enforcement
  3. FraudDetectionAgent - Toll fraud and social engineering detection
  4. CRMScreenPopAgent - CRM data protection and masking
  5. RecordingEncryptionAgent - Call recording encryption management
  6. SessionSecurityAgent - Agent desktop session monitoring
  7. RemoteAccessAgent - Work-from-home agent security
  8. ComplianceAuditAgent - Continuous compliance auditing
- BPOAgentCoordinator: Multi-agent orchestration for BPO operations

Feature flag: QBITEL_FEATURE_BPO_AGENTS=true
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__all__: List[str] = []

# Lazy-loaded module references
_agents_module = None
_coordinator_module = None


def _load_agents():
    """Lazy-load the BPO Agents module."""
    global _agents_module
    if _agents_module is None:
        try:
            from . import bpo_agents as mod
            _agents_module = mod
            logger.debug("Loaded BPO Agents module")
        except ImportError as e:
            logger.warning(f"Failed to load BPO Agents: {e}")
    return _agents_module


def _load_coordinator():
    """Lazy-load the BPO Agent Coordinator module."""
    global _coordinator_module
    if _coordinator_module is None:
        try:
            from . import agent_coordinator as mod
            _coordinator_module = mod
            logger.debug("Loaded BPO Agent Coordinator module")
        except ImportError as e:
            logger.warning(f"Failed to load BPO Agent Coordinator: {e}")
    return _coordinator_module


def __getattr__(name: str):
    """Lazy load submodules and key classes on attribute access."""

    # Agent classes
    agent_exports = {
        "CallProcessingAgent",
        "PCIComplianceAgent",
        "FraudDetectionAgent",
        "CRMScreenPopAgent",
        "RecordingEncryptionAgent",
        "SessionSecurityAgent",
        "RemoteAccessAgent",
        "ComplianceAuditAgent",
        "BPO_AGENT_REGISTRY",
    }

    if name in agent_exports:
        mod = _load_agents()
        if mod is not None:
            return getattr(mod, name)
        raise AttributeError(f"module 'agents' has no attribute '{name}'")

    # Coordinator classes
    coordinator_exports = {
        "BPOAgentCoordinator",
        "BPOAgentPool",
        "CoordinationStrategy",
    }

    if name in coordinator_exports:
        mod = _load_coordinator()
        if mod is not None:
            return getattr(mod, name)
        raise AttributeError(f"module 'agents' has no attribute '{name}'")

    # Module-level access
    if name == "bpo_agents":
        mod = _load_agents()
        if mod is not None:
            return mod
        raise AttributeError(f"module 'agents' has no attribute '{name}'")

    if name == "agent_coordinator":
        mod = _load_coordinator()
        if mod is not None:
            return mod
        raise AttributeError(f"module 'agents' has no attribute '{name}'")

    raise AttributeError(f"module 'ai_engine.domains.bpo.agents' has no attribute '{name}'")


def get_version() -> str:
    """Get agents module version."""
    return __version__


def get_agent_types() -> List[str]:
    """Get list of available BPO agent types."""
    return [
        "CallProcessingAgent - Real-time call security monitoring",
        "PCIComplianceAgent - PCI-DSS voice compliance enforcement",
        "FraudDetectionAgent - Toll fraud and social engineering detection",
        "CRMScreenPopAgent - CRM data protection and screen pop security",
        "RecordingEncryptionAgent - Call recording encryption management",
        "SessionSecurityAgent - Agent desktop session monitoring",
        "RemoteAccessAgent - Work-from-home agent security",
        "ComplianceAuditAgent - Continuous compliance auditing",
    ]
