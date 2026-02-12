"""
QBITEL Engine - Feature Flags System

Centralized feature flag management for enabling/disabling modules and features.
Supports environment variable configuration and runtime toggling.

Usage:
    from ai_engine.core.feature_flags import feature_flags

    if feature_flags.is_enabled("healthcare_domain"):
        from ai_engine.domains.healthcare import ...

Environment Variables:
    QBITEL_FEATURE_HEALTHCARE_DOMAIN=true
    QBITEL_FEATURE_AUTOMOTIVE_DOMAIN=false
    QBITEL_FEATURE_AVIATION_DOMAIN=false
    QBITEL_FEATURE_INDUSTRIAL_DOMAIN=false
    QBITEL_FEATURE_MARKETPLACE=false
    QBITEL_FEATURE_AGENTIC_SECURITY=true
    QBITEL_FEATURE_QUANTUM_CRYPTO=true
"""

import os
import logging
from typing import Dict, Any, Optional, Set
from enum import Enum
from dataclasses import dataclass, field
from functools import lru_cache

logger = logging.getLogger(__name__)


class FeatureCategory(str, Enum):
    """Categories of features for grouping and reporting."""

    DOMAIN = "domain"
    SECURITY = "security"
    INTEGRATION = "integration"
    EXPERIMENTAL = "experimental"
    CORE = "core"


@dataclass
class FeatureDefinition:
    """Definition of a feature flag."""

    name: str
    description: str
    category: FeatureCategory
    default_enabled: bool = False
    requires: Set[str] = field(default_factory=set)  # Dependencies on other features
    env_var: Optional[str] = None  # Override env var name


# Feature flag definitions
FEATURE_DEFINITIONS: Dict[str, FeatureDefinition] = {
    # Domain modules
    "healthcare_domain": FeatureDefinition(
        name="healthcare_domain",
        description="Healthcare-specific PQC modules (FDA compliance, FHIR, medical devices)",
        category=FeatureCategory.DOMAIN,
        default_enabled=False,
        env_var="QBITEL_FEATURE_HEALTHCARE_DOMAIN",
    ),
    "automotive_domain": FeatureDefinition(
        name="automotive_domain",
        description="Automotive V2X protocols and SCMS integration",
        category=FeatureCategory.DOMAIN,
        default_enabled=False,
        env_var="QBITEL_FEATURE_AUTOMOTIVE_DOMAIN",
    ),
    "aviation_domain": FeatureDefinition(
        name="aviation_domain",
        description="Aviation protocols (ADS-B, LDACS, ARINC 653)",
        category=FeatureCategory.DOMAIN,
        default_enabled=False,
        env_var="QBITEL_FEATURE_AVIATION_DOMAIN",
    ),
    "industrial_domain": FeatureDefinition(
        name="industrial_domain",
        description="Industrial protocols (Modbus, IEC 62351, SCADA)",
        category=FeatureCategory.DOMAIN,
        default_enabled=False,
        env_var="QBITEL_FEATURE_INDUSTRIAL_DOMAIN",
    ),
    # Core features
    "protocol_discovery": FeatureDefinition(
        name="protocol_discovery",
        description="AI-powered protocol discovery engine",
        category=FeatureCategory.CORE,
        default_enabled=True,
        env_var="QBITEL_FEATURE_PROTOCOL_DISCOVERY",
    ),
    "field_detection": FeatureDefinition(
        name="field_detection",
        description="BiLSTM-CRF field boundary detection",
        category=FeatureCategory.CORE,
        default_enabled=True,
        env_var="QBITEL_FEATURE_FIELD_DETECTION",
    ),
    "legacy_whisperer": FeatureDefinition(
        name="legacy_whisperer",
        description="COBOL/mainframe modernization service",
        category=FeatureCategory.CORE,
        default_enabled=True,
        env_var="QBITEL_FEATURE_LEGACY_WHISPERER",
    ),
    # Security features
    "agentic_security": FeatureDefinition(
        name="agentic_security",
        description="Zero-touch autonomous security decision engine",
        category=FeatureCategory.SECURITY,
        default_enabled=True,
        env_var="QBITEL_FEATURE_AGENTIC_SECURITY",
    ),
    "quantum_crypto": FeatureDefinition(
        name="quantum_crypto",
        description="Post-quantum cryptography (Kyber, Dilithium)",
        category=FeatureCategory.SECURITY,
        default_enabled=True,
        env_var="QBITEL_FEATURE_QUANTUM_CRYPTO",
    ),
    "threat_intelligence": FeatureDefinition(
        name="threat_intelligence",
        description="MITRE ATT&CK and STIX/TAXII integration",
        category=FeatureCategory.SECURITY,
        default_enabled=True,
        env_var="QBITEL_FEATURE_THREAT_INTELLIGENCE",
    ),
    # Integration features
    "marketplace": FeatureDefinition(
        name="marketplace",
        description="Protocol marketplace for sharing and monetization",
        category=FeatureCategory.INTEGRATION,
        default_enabled=False,
        env_var="QBITEL_FEATURE_MARKETPLACE",
    ),
    "cloud_integrations": FeatureDefinition(
        name="cloud_integrations",
        description="AWS Security Hub, Azure Sentinel, GCP SCC integrations",
        category=FeatureCategory.INTEGRATION,
        default_enabled=True,
        env_var="QBITEL_FEATURE_CLOUD_INTEGRATIONS",
    ),
    "compliance_automation": FeatureDefinition(
        name="compliance_automation",
        description="Automated compliance reporting (SOC2, GDPR, HIPAA, PCI-DSS)",
        category=FeatureCategory.INTEGRATION,
        default_enabled=True,
        env_var="QBITEL_FEATURE_COMPLIANCE_AUTOMATION",
    ),
    # Experimental features
    "llm_copilot": FeatureDefinition(
        name="llm_copilot",
        description="LLM-powered protocol analysis copilot",
        category=FeatureCategory.EXPERIMENTAL,
        default_enabled=True,
        env_var="QBITEL_FEATURE_LLM_COPILOT",
    ),
    "translation_studio": FeatureDefinition(
        name="translation_studio",
        description="Protocol to REST API/SDK generation",
        category=FeatureCategory.EXPERIMENTAL,
        default_enabled=True,
        env_var="QBITEL_FEATURE_TRANSLATION_STUDIO",
    ),
}


class FeatureFlags:
    """
    Feature flag manager for QBITEL.

    Provides centralized control over feature availability with:
    - Environment variable configuration
    - Runtime toggling (for testing)
    - Dependency checking
    - Audit logging
    """

    def __init__(self):
        self._overrides: Dict[str, bool] = {}
        self._loaded = False
        self._cache: Dict[str, bool] = {}

    def _load_from_env(self, feature: FeatureDefinition) -> bool:
        """Load feature flag value from environment variable."""
        env_var = feature.env_var or f"QBITEL_FEATURE_{feature.name.upper()}"
        env_value = os.getenv(env_var, "").lower()

        if env_value in ("true", "1", "yes", "on", "enabled"):
            return True
        elif env_value in ("false", "0", "no", "off", "disabled"):
            return False
        else:
            return feature.default_enabled

    def is_enabled(self, feature_name: str) -> bool:
        """
        Check if a feature is enabled.

        Args:
            feature_name: Name of the feature to check

        Returns:
            True if feature is enabled, False otherwise
        """
        # Check runtime overrides first
        if feature_name in self._overrides:
            return self._overrides[feature_name]

        # Check cache
        if feature_name in self._cache:
            return self._cache[feature_name]

        # Get feature definition
        feature = FEATURE_DEFINITIONS.get(feature_name)
        if not feature:
            logger.warning(f"Unknown feature flag: {feature_name}, defaulting to disabled")
            return False

        # Check dependencies
        for dependency in feature.requires:
            if not self.is_enabled(dependency):
                logger.debug(f"Feature {feature_name} disabled: missing dependency {dependency}")
                self._cache[feature_name] = False
                return False

        # Load from environment
        enabled = self._load_from_env(feature)
        self._cache[feature_name] = enabled

        return enabled

    def enable(self, feature_name: str) -> None:
        """
        Enable a feature at runtime (for testing).

        Args:
            feature_name: Name of the feature to enable
        """
        if feature_name not in FEATURE_DEFINITIONS:
            logger.warning(f"Enabling unknown feature: {feature_name}")

        self._overrides[feature_name] = True
        self._cache.pop(feature_name, None)
        logger.info(f"Feature enabled: {feature_name}")

    def disable(self, feature_name: str) -> None:
        """
        Disable a feature at runtime (for testing).

        Args:
            feature_name: Name of the feature to disable
        """
        self._overrides[feature_name] = False
        self._cache.pop(feature_name, None)
        logger.info(f"Feature disabled: {feature_name}")

    def reset(self, feature_name: Optional[str] = None) -> None:
        """
        Reset feature flag(s) to environment/default values.

        Args:
            feature_name: Specific feature to reset, or None to reset all
        """
        if feature_name:
            self._overrides.pop(feature_name, None)
            self._cache.pop(feature_name, None)
        else:
            self._overrides.clear()
            self._cache.clear()

    def get_all_flags(self) -> Dict[str, bool]:
        """
        Get current state of all feature flags.

        Returns:
            Dictionary of feature name to enabled status
        """
        return {name: self.is_enabled(name) for name in FEATURE_DEFINITIONS.keys()}

    def get_enabled_features(self) -> Set[str]:
        """
        Get set of all enabled feature names.

        Returns:
            Set of enabled feature names
        """
        return {name for name in FEATURE_DEFINITIONS.keys() if self.is_enabled(name)}

    def get_disabled_features(self) -> Set[str]:
        """
        Get set of all disabled feature names.

        Returns:
            Set of disabled feature names
        """
        return {name for name in FEATURE_DEFINITIONS.keys() if not self.is_enabled(name)}

    def get_features_by_category(self, category: FeatureCategory) -> Dict[str, bool]:
        """
        Get features filtered by category.

        Args:
            category: Category to filter by

        Returns:
            Dictionary of feature name to enabled status
        """
        return {
            name: self.is_enabled(name) for name, definition in FEATURE_DEFINITIONS.items() if definition.category == category
        }

    def get_feature_info(self, feature_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a feature.

        Args:
            feature_name: Name of the feature

        Returns:
            Feature information dictionary or None if not found
        """
        feature = FEATURE_DEFINITIONS.get(feature_name)
        if not feature:
            return None

        return {
            "name": feature.name,
            "description": feature.description,
            "category": feature.category.value,
            "default_enabled": feature.default_enabled,
            "env_var": feature.env_var or f"QBITEL_FEATURE_{feature.name.upper()}",
            "requires": list(feature.requires),
            "enabled": self.is_enabled(feature_name),
        }

    def get_status_report(self) -> Dict[str, Any]:
        """
        Get a comprehensive status report of all feature flags.

        Returns:
            Status report dictionary
        """
        enabled = self.get_enabled_features()
        disabled = self.get_disabled_features()

        return {
            "total_features": len(FEATURE_DEFINITIONS),
            "enabled_count": len(enabled),
            "disabled_count": len(disabled),
            "enabled_features": sorted(enabled),
            "disabled_features": sorted(disabled),
            "by_category": {category.value: self.get_features_by_category(category) for category in FeatureCategory},
            "overrides_active": len(self._overrides),
        }


# Global singleton instance
feature_flags = FeatureFlags()


# Convenience functions
def is_feature_enabled(feature_name: str) -> bool:
    """Check if a feature is enabled."""
    return feature_flags.is_enabled(feature_name)


def require_feature(feature_name: str) -> None:
    """
    Raise an error if feature is not enabled.

    Args:
        feature_name: Name of the required feature

    Raises:
        RuntimeError: If feature is not enabled
    """
    if not feature_flags.is_enabled(feature_name):
        raise RuntimeError(
            f"Feature '{feature_name}' is not enabled. " f"Set {FEATURE_DEFINITIONS[feature_name].env_var}=true to enable it."
        )


def feature_guard(feature_name: str):
    """
    Decorator to guard functions with feature flags.

    Args:
        feature_name: Name of the required feature

    Example:
        @feature_guard("marketplace")
        def create_listing():
            ...
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            if not feature_flags.is_enabled(feature_name):
                raise RuntimeError(f"Feature '{feature_name}' is disabled. " f"Cannot call {func.__name__}.")
            return func(*args, **kwargs)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator


# Log feature status on import
def _log_feature_status():
    """Log feature flag status on startup."""
    enabled = feature_flags.get_enabled_features()
    disabled = feature_flags.get_disabled_features()

    logger.info(f"Feature flags loaded: {len(enabled)} enabled, {len(disabled)} disabled")
    if enabled:
        logger.debug(f"Enabled features: {', '.join(sorted(enabled))}")
    if disabled:
        logger.debug(f"Disabled features: {', '.join(sorted(disabled))}")


# Initialize on import
_log_feature_status()
