"""
CRONOS AI - Vendor Shims Package
Provides fallback implementations for optional dependencies in air-gapped deployments.
"""

import logging

logger = logging.getLogger(__name__)

__all__ = ['llm_shims', 'ml_shims', 'monitoring_shims']

logger.info("CRONOS AI vendor shims package loaded")