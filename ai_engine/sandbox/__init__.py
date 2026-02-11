"""
Firecracker Sandbox Module

Secure sandbox environment for marketplace validation:
- MicroVM isolation using Firecracker
- Resource quotas and limits
- Network isolation
- Ephemeral execution environments
- Audit logging

Features:
- Fast (<125ms) microVM boot times
- Memory and CPU isolation
- Network namespacing
- Filesystem snapshots
- Execution timeouts

Usage:
    from ai_engine.sandbox import FirecrackerSandbox, SandboxConfig

    sandbox = FirecrackerSandbox(config=SandboxConfig(
        memory_mb=512,
        vcpu_count=2,
        timeout_seconds=60
    ))

    async with sandbox:
        result = await sandbox.execute(code="print('Hello from sandbox')")
"""

from ai_engine.sandbox.firecracker import (
    FirecrackerSandbox,
    SandboxConfig,
    SandboxResult,
    SandboxStatus,
    ResourceLimits,
    NetworkConfig,
)
from ai_engine.sandbox.manager import (
    SandboxManager,
    SandboxPool,
    PoolConfig,
)
from ai_engine.sandbox.validation import (
    MarketplaceValidator,
    ValidationResult,
    ValidationConfig,
    SecurityCheck,
    ComplianceCheck,
)

__all__ = [
    # Core sandbox
    "FirecrackerSandbox",
    "SandboxConfig",
    "SandboxResult",
    "SandboxStatus",
    "ResourceLimits",
    "NetworkConfig",
    # Management
    "SandboxManager",
    "SandboxPool",
    "PoolConfig",
    # Validation
    "MarketplaceValidator",
    "ValidationResult",
    "ValidationConfig",
    "SecurityCheck",
    "ComplianceCheck",
]
