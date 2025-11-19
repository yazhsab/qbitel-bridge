"""
Istio Service Mesh Integration

Provides automatic sidecar injection with quantum-safe encryption for Istio service meshes.
"""

from .sidecar_injector import IstioSidecarInjector
from .qkd_certificate_manager import QuantumCertificateManager
from .mtls_config import MutualTLSConfigurator
from .mesh_policy import MeshPolicyManager

__all__ = [
    "IstioSidecarInjector",
    "QuantumCertificateManager",
    "MutualTLSConfigurator",
    "MeshPolicyManager",
]
