"""
QBITEL - Protocol Marketplace Module

The Protocol Marketplace transforms QBITEL from a product into an ecosystem by
enabling community contributions, vendor partnerships, and protocol monetization.

Key Features:
- Protocol discovery and search
- Automated validation pipeline
- Revenue sharing for protocol creators
- Integration with Protocol Knowledge Base
- Translation Studio deployment
"""

from .protocol_validator import ProtocolValidator, ValidationResult, run_protocol_validation
from .knowledge_base_integration import (
    MarketplaceKnowledgeBaseIntegration,
    MarketplaceProtocolDeployer,
)
from .stripe_integration import StripeConnectManager, get_stripe_manager
from .s3_file_manager import S3FileManager, get_s3_manager

__all__ = [
    "ProtocolValidator",
    "ValidationResult",
    "run_protocol_validation",
    "MarketplaceKnowledgeBaseIntegration",
    "MarketplaceProtocolDeployer",
    "StripeConnectManager",
    "get_stripe_manager",
    "S3FileManager",
    "get_s3_manager",
]
