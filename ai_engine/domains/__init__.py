"""
QBITEL - Domain-Specific PQC Modules

Optimized post-quantum cryptography implementations for constrained environments:
- Healthcare: Medical devices with 64KB RAM, 10+ year battery life
- Automotive: V2X with <10ms latency, 1000+ msg/sec verification
- Aviation: Bandwidth-constrained channels (600bps - 2.4kbps)
- Industrial: Safety-critical systems with deterministic timing (IEC 61508)

Feature flags control which domain modules are loaded:
- QBITEL_FEATURE_HEALTHCARE_DOMAIN=true
- QBITEL_FEATURE_AUTOMOTIVE_DOMAIN=true
- QBITEL_FEATURE_AVIATION_DOMAIN=true
- QBITEL_FEATURE_INDUSTRIAL_DOMAIN=true
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

# Dynamic exports based on feature flags
__all__: List[str] = []

# Import feature flags
try:
    from ..core.feature_flags import feature_flags
    _feature_flags_available = True
except ImportError:
    _feature_flags_available = False
    logger.warning("Feature flags not available, loading all domain modules")


def _load_domain_modules():
    """Load domain modules based on feature flags."""
    global __all__

    # Healthcare domain
    if not _feature_flags_available or feature_flags.is_enabled("healthcare_domain"):
        try:
            from . import healthcare
            __all__.append("healthcare")
            logger.info("Healthcare domain module loaded")
        except ImportError as e:
            logger.warning(f"Failed to load healthcare domain: {e}")

    # Automotive domain
    if not _feature_flags_available or feature_flags.is_enabled("automotive_domain"):
        try:
            from . import automotive
            __all__.append("automotive")
            logger.info("Automotive domain module loaded")
        except ImportError as e:
            logger.warning(f"Failed to load automotive domain: {e}")

    # Aviation domain
    if not _feature_flags_available or feature_flags.is_enabled("aviation_domain"):
        try:
            from . import aviation
            __all__.append("aviation")
            logger.info("Aviation domain module loaded")
        except ImportError as e:
            logger.warning(f"Failed to load aviation domain: {e}")

    # Industrial domain
    if not _feature_flags_available or feature_flags.is_enabled("industrial_domain"):
        try:
            from . import industrial
            __all__.append("industrial")
            logger.info("Industrial domain module loaded")
        except ImportError as e:
            logger.warning(f"Failed to load industrial domain: {e}")

    if not __all__:
        logger.info("No domain modules enabled via feature flags")
    else:
        logger.info(f"Loaded domain modules: {', '.join(__all__)}")


# Load modules on import
_load_domain_modules()
