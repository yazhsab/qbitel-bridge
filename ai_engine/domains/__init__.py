"""
QBITEL - Domain-Specific PQC Modules

Optimized post-quantum cryptography implementations for constrained environments:
- Healthcare: Medical devices with 64KB RAM, 10+ year battery life
- Automotive: V2X with <10ms latency, 1000+ msg/sec verification
- Aviation: Bandwidth-constrained channels (600bps - 2.4kbps)
- Industrial: Safety-critical systems with deterministic timing (IEC 61508)
- BPO/Call Centers: Voice channel security, DTMF masking, remote agent protection

Feature flags control which domain modules are loaded:
- QBITEL_FEATURE_HEALTHCARE_DOMAIN=true
- QBITEL_FEATURE_AUTOMOTIVE_DOMAIN=true
- QBITEL_FEATURE_AVIATION_DOMAIN=true
- QBITEL_FEATURE_INDUSTRIAL_DOMAIN=true
- QBITEL_FEATURE_BPO_DOMAIN=true
"""

import logging
import warnings
from enum import Enum
from typing import Dict, List

logger = logging.getLogger(__name__)


class DomainMaturity(Enum):
    """Maturity level for domain modules."""

    GA = "ga"                      # General Availability — production ready, audited
    PREVIEW = "preview"            # Feature-complete, not fully hardened
    EXPERIMENTAL = "experimental"  # Early development, APIs may change


# Maps domain names to their maturity level.
# GA: Banking, Healthcare, BPO (user-selected for initial release)
# PREVIEW: Automotive, Aviation (strong PQC use cases, needs more hardening)
# EXPERIMENTAL: Industrial (safety-critical, requires formal certification)
DOMAIN_MATURITY: Dict[str, DomainMaturity] = {
    "banking": DomainMaturity.GA,
    "healthcare": DomainMaturity.GA,
    "bpo": DomainMaturity.GA,
    "automotive": DomainMaturity.PREVIEW,
    "aviation": DomainMaturity.PREVIEW,
    "industrial": DomainMaturity.EXPERIMENTAL,
}

# Dynamic exports based on feature flags
__all__: List[str] = ["DomainMaturity", "DOMAIN_MATURITY"]

# Import feature flags
try:
    from ..core.feature_flags import feature_flags

    _feature_flags_available = True
except ImportError:
    _feature_flags_available = False
    logger.warning("Feature flags not available, loading all domain modules")


def _load_domain_modules():
    """Load domain modules based on feature flags, with maturity warnings."""
    global __all__

    _domains = [
        ("healthcare", "healthcare_domain"),
        ("automotive", "automotive_domain"),
        ("aviation", "aviation_domain"),
        ("industrial", "industrial_domain"),
        ("bpo", "bpo_domain"),
    ]

    for domain_name, flag_name in _domains:
        if _feature_flags_available and not feature_flags.is_enabled(flag_name):
            continue
        try:
            __import__(f"{__name__}.{domain_name}", fromlist=[domain_name])
            __all__.append(domain_name)

            maturity = DOMAIN_MATURITY.get(domain_name, DomainMaturity.EXPERIMENTAL)
            if maturity == DomainMaturity.PREVIEW:
                warnings.warn(
                    f"Domain module '{domain_name}' is PREVIEW. "
                    f"APIs are stable but security has not been fully audited.",
                    UserWarning,
                    stacklevel=2,
                )
            elif maturity == DomainMaturity.EXPERIMENTAL:
                warnings.warn(
                    f"Domain module '{domain_name}' is EXPERIMENTAL. "
                    f"APIs may change and formal safety certification is pending.",
                    FutureWarning,
                    stacklevel=2,
                )
            logger.info(f"{domain_name.title()} domain module loaded (maturity={maturity.value})")
        except ImportError as e:
            logger.warning(f"Failed to load {domain_name} domain: {e}")

    loaded = [d for d in __all__ if d not in ("DomainMaturity", "DOMAIN_MATURITY")]
    if not loaded:
        logger.info("No domain modules enabled via feature flags")
    else:
        logger.info(f"Loaded domain modules: {', '.join(loaded)}")


# Load modules on import
_load_domain_modules()
