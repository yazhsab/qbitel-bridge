"""
Envoy Proxy Integration

Provides quantum-safe encryption for east-west traffic using Envoy xDS APIs
and custom filters for service-to-service communication.
"""

from .xds_server import EnvoyXDSServer
from .traffic_encryption import EastWestEncryption
from .policy_engine import TrafficPolicyEngine
from .observability import EnvoyObservability

__all__ = [
    "EnvoyXDSServer",
    "EastWestEncryption",
    "TrafficPolicyEngine",
    "EnvoyObservability",
]
