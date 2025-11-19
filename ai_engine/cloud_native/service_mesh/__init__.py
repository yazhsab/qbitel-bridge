"""
Service Mesh Security Module

Provides quantum-safe security for Kubernetes service meshes including:
- Istio sidecar injection and configuration
- Envoy proxy filters for east-west traffic encryption
- Quantum key distribution for certificate management
- Service mesh policy enforcement
"""

from typing import Dict, List, Optional

__all__ = [
    "istio",
    "envoy",
    "helm",
]
