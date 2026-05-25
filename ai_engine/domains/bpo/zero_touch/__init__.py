"""
QBITEL - BPO Zero-Touch Orchestration & Automation

Autonomous discovery, provisioning, and configuration of BPO/call center
security environments with zero human intervention.

This module provides:
- Zero-Touch Orchestrator: Fully autonomous security deployment pipeline
  (discover -> assess -> generate policies -> provision -> monitor)
- Automation Recipes: Pre-built deployment templates for common BPO security
  patterns (PCI-DSS voice, toll fraud prevention, remote workforce, etc.)
- Recipe Registry: Central catalog and execution engine for automation recipes

Feature flag: QBITEL_FEATURE_BPO_ZERO_TOUCH=true
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__all__: List[str] = []

# Lazy-loaded module references
_orchestrator_module = None
_recipes_module = None


def _load_orchestrator():
    """Lazy-load the BPO Zero-Touch Orchestrator module."""
    global _orchestrator_module
    if _orchestrator_module is None:
        try:
            from . import bpo_zero_touch_orchestrator as mod
            _orchestrator_module = mod
            logger.debug("Loaded BPO Zero-Touch Orchestrator module")
        except ImportError as e:
            logger.warning(f"Failed to load BPO Zero-Touch Orchestrator: {e}")
    return _orchestrator_module


def _load_recipes():
    """Lazy-load the BPO Automation Recipes module."""
    global _recipes_module
    if _recipes_module is None:
        try:
            from . import automation_recipes as mod
            _recipes_module = mod
            logger.debug("Loaded BPO Automation Recipes module")
        except ImportError as e:
            logger.warning(f"Failed to load BPO Automation Recipes: {e}")
    return _recipes_module


def __getattr__(name: str):
    """Lazy load submodules and key classes on attribute access."""

    # Orchestrator classes
    orchestrator_exports = {
        "BPOZeroTouchOrchestrator",
        "EnvironmentDiscoveryResult",
        "SecurityAssessmentResult",
        "SecurityGap",
        "PolicyGenerationResult",
        "GeneratedPolicy",
        "ProvisioningResult",
        "ZeroTouchDeploymentResult",
    }

    if name in orchestrator_exports:
        mod = _load_orchestrator()
        if mod is not None:
            return getattr(mod, name)
        raise AttributeError(f"module 'zero_touch' has no attribute '{name}'")

    # Automation recipe classes
    recipe_exports = {
        "BPOAutomationRecipe",
        "AutomationStep",
        "RecipeRegistry",
        "PCIDSSVoiceComplianceRecipe",
        "TollFraudPreventionRecipe",
        "RemoteWorkforceSecurityRecipe",
        "QuantumSafeVoiceRecipe",
        "ComplianceSuiteRecipe",
        "FullBPOSecurityRecipe",
    }

    if name in recipe_exports:
        mod = _load_recipes()
        if mod is not None:
            return getattr(mod, name)
        raise AttributeError(f"module 'zero_touch' has no attribute '{name}'")

    # Module-level access
    if name == "bpo_zero_touch_orchestrator":
        mod = _load_orchestrator()
        if mod is not None:
            return mod
        raise AttributeError(f"module 'zero_touch' has no attribute '{name}'")

    if name == "automation_recipes":
        mod = _load_recipes()
        if mod is not None:
            return mod
        raise AttributeError(f"module 'zero_touch' has no attribute '{name}'")

    raise AttributeError(f"module 'ai_engine.domains.bpo.zero_touch' has no attribute '{name}'")


def get_version() -> str:
    """Get zero-touch module version."""
    return __version__


def get_capabilities() -> List[str]:
    """Get list of zero-touch capabilities."""
    return [
        "Environment Discovery (PBX, CRM, WFM, SIP, terminal)",
        "Security Posture Assessment (voice encryption, DTMF, PCI-DSS)",
        "LLM-Powered Policy Generation",
        "Auto-Provisioning of Security Controls",
        "Continuous Monitoring and Self-Healing",
        "PCI-DSS Voice Compliance Recipe",
        "Toll Fraud Prevention Recipe",
        "Remote Workforce Security Recipe",
        "Quantum-Safe Voice Upgrade Recipe",
        "Full Compliance Suite Recipe",
        "Unified Full BPO Security Recipe",
    ]
