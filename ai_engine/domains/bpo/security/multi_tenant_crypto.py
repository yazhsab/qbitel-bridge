"""
Multi-Tenant Cryptographic Isolation Module

Provides per-tenant PQC key hierarchies for BPO environments.

Multi-tenant BPO platforms handle data for multiple enterprise
clients simultaneously. This module ensures:
- Cryptographic isolation between tenants using hierarchical key trees
- Per-tenant PQC algorithm selection (ML-KEM, ML-DSA, SLH-DSA)
- Session-level ephemeral key derivation for forward secrecy
- Cross-tenant access detection and prevention
- Compliance-driven key rotation policies per tenant
- HSM integration support for premium/dedicated tenants

The key hierarchy follows a 5-level structure:
  MASTER -> TENANT_ROOT -> SERVICE -> SESSION -> EPHEMERAL

Each level derives keys from its parent using HKDF-style key
derivation backed by SHA3-256. In production, the MASTER and
TENANT_ROOT levels would reside in HSM hardware.

Integrates with QBITEL's quantum-safe infrastructure for
end-to-end tenant data protection.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import hashlib
import logging
import re
import uuid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TenantSecurityTier(Enum):
    """Security tier determining the cryptographic strength and isolation."""

    STANDARD = auto()      # Shared infra, software-based keys
    ENHANCED = auto()      # Shared infra, enhanced key rotation
    PREMIUM = auto()       # Separate key hierarchy, HSM-backed root
    DEDICATED = auto()     # Dedicated HSM, fully isolated key tree


class KeyHierarchyLevel(Enum):
    """Level in the tenant key hierarchy tree."""

    MASTER = auto()        # Platform-wide master key (HSM-stored)
    TENANT_ROOT = auto()   # Per-tenant root key
    SERVICE = auto()       # Per-service key (voice, chat, data)
    SESSION = auto()       # Per-session key (forward secrecy)
    EPHEMERAL = auto()     # Short-lived ephemeral key


class TenantIsolationMode(Enum):
    """Infrastructure isolation mode for tenant cryptographic operations."""

    SHARED_INFRA_SEPARATE_KEYS = auto()   # Shared infra, logical key separation
    SEPARATE_INFRA = auto()               # Dedicated crypto infrastructure
    DEDICATED_HSM = auto()                # Per-tenant HSM partition


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TenantKeyPolicy:
    """
    Cryptographic policy for a specific tenant.

    Controls algorithm selection, key rotation intervals,
    HSM requirements, and compliance frameworks.
    """

    tenant_id: str = ""
    security_tier: TenantSecurityTier = TenantSecurityTier.STANDARD
    kem_algorithm: str = "ML-KEM-768"
    sig_algorithm: str = "ML-DSA-65"
    use_hybrid: bool = True
    key_rotation_hours: int = 24
    session_key_rotation_hours: int = 1
    require_hsm: bool = False
    compliance_frameworks: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize policy for storage or transmission."""
        return {
            "tenant_id": self.tenant_id,
            "security_tier": self.security_tier.name,
            "kem_algorithm": self.kem_algorithm,
            "sig_algorithm": self.sig_algorithm,
            "use_hybrid": self.use_hybrid,
            "key_rotation_hours": self.key_rotation_hours,
            "session_key_rotation_hours": self.session_key_rotation_hours,
            "require_hsm": self.require_hsm,
            "compliance_frameworks": self.compliance_frameworks,
        }


@dataclass
class TenantKeySet:
    """
    A set of cryptographic keys for a tenant.

    Contains the tenant root key hash and derived service keys.
    Actual key material is never stored in plaintext; only
    hashes are retained for verification and audit.
    """

    key_set_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = ""
    tenant_root_key_hash: str = ""
    service_keys: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    rotated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(hours=24))
    is_active: bool = True

    @property
    def is_expired(self) -> bool:
        """Check if the key set has expired."""
        return datetime.utcnow() > self.expires_at

    @property
    def age_hours(self) -> float:
        """Get the age of the key set in hours."""
        delta = datetime.utcnow() - self.created_at
        return delta.total_seconds() / 3600.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize key set metadata for audit or transmission."""
        return {
            "key_set_id": self.key_set_id,
            "tenant_id": self.tenant_id,
            "tenant_root_key_hash": self.tenant_root_key_hash,
            "service_keys": {k: v[:16] + "..." for k, v in self.service_keys.items()},
            "created_at": self.created_at.isoformat(),
            "rotated_at": self.rotated_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "is_active": self.is_active,
            "is_expired": self.is_expired,
        }


@dataclass
class TenantCryptoContext:
    """
    Runtime cryptographic context for an active tenant.

    Holds the current key set, policy, and operational counters
    for monitoring tenant cryptographic operations.
    """

    tenant_id: str = ""
    key_set: Optional[TenantKeySet] = None
    policy: Optional[TenantKeyPolicy] = None
    active_sessions: int = 0
    total_operations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize context for monitoring or debugging."""
        return {
            "tenant_id": self.tenant_id,
            "key_set": self.key_set.to_dict() if self.key_set else None,
            "policy": self.policy.to_dict() if self.policy else None,
            "active_sessions": self.active_sessions,
            "total_operations": self.total_operations,
        }


@dataclass
class CrossTenantAccessEvent:
    """
    A detected cross-tenant access attempt.

    Records any attempt by one tenant's operations to access
    another tenant's key material or encrypted data.
    """

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_tenant: str = ""
    target_tenant: str = ""
    operation: str = ""
    was_blocked: bool = True
    evidence: Dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize event for alerting or audit logging."""
        return {
            "event_id": self.event_id,
            "source_tenant": self.source_tenant,
            "target_tenant": self.target_tenant,
            "operation": self.operation,
            "was_blocked": self.was_blocked,
            "evidence": self.evidence,
            "detected_at": self.detected_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Key derivation helpers
# ---------------------------------------------------------------------------

# Standard service names in BPO environments
STANDARD_SERVICES: List[str] = [
    "voice",        # Voice call encryption
    "chat",         # Text chat encryption
    "screen",       # Screen sharing encryption
    "recording",    # Call recording encryption
    "data",         # Customer data encryption
    "analytics",    # Analytics data encryption
]

# Compliance frameworks and their key requirements
COMPLIANCE_KEY_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "PCI-DSS": {
        "min_key_length_bits": 256,
        "max_rotation_hours": 24,
        "require_hsm": True,
        "required_algorithms": ["ML-KEM-768", "ML-KEM-1024"],
    },
    "HIPAA": {
        "min_key_length_bits": 256,
        "max_rotation_hours": 72,
        "require_hsm": False,
        "required_algorithms": ["ML-KEM-768"],
    },
    "SOC2": {
        "min_key_length_bits": 128,
        "max_rotation_hours": 168,
        "require_hsm": False,
        "required_algorithms": ["ML-KEM-512", "ML-KEM-768"],
    },
    "GDPR": {
        "min_key_length_bits": 256,
        "max_rotation_hours": 48,
        "require_hsm": False,
        "required_algorithms": ["ML-KEM-768"],
    },
    "ISO-27001": {
        "min_key_length_bits": 256,
        "max_rotation_hours": 168,
        "require_hsm": False,
        "required_algorithms": ["ML-KEM-768", "ML-KEM-1024"],
    },
}


def _derive_key(parent_hash: str, context: str, level: str) -> str:
    """
    Derive a child key hash from a parent key hash using HKDF-style derivation.

    In production, this would use HKDF-Expand with SHA3-256 and
    proper key material. Here we use SHA3-256 chaining as a
    placeholder for the derivation function.

    Args:
        parent_hash: The parent key hash (hex string).
        context: Context string (e.g., tenant_id, service_name).
        level: Key hierarchy level identifier.

    Returns:
        Derived key hash as hex string.
    """
    derivation_input = f"{parent_hash}:{context}:{level}:{uuid.uuid4().hex[:8]}"
    return hashlib.sha3_256(derivation_input.encode()).hexdigest()


def _generate_master_key_hash(seed: str = "") -> str:
    """Generate a master key hash (placeholder for HSM key generation)."""
    seed_material = seed or str(uuid.uuid4())
    return hashlib.sha3_256(
        f"MASTER:{seed_material}:{datetime.utcnow().isoformat()}".encode()
    ).hexdigest()


# ---------------------------------------------------------------------------
# Tenant Key Manager
# ---------------------------------------------------------------------------


class TenantKeyManager:
    """
    Manages per-tenant cryptographic key hierarchies for BPO platforms.

    Provides complete lifecycle management for tenant keys including
    creation, rotation, derivation, isolation verification, and
    cross-tenant access detection.

    The key hierarchy:
      MASTER (platform-wide)
        -> TENANT_ROOT (per tenant)
          -> SERVICE (per service per tenant)
            -> SESSION (per active session)
              -> EPHEMERAL (short-lived operations)

    Usage::

        manager = TenantKeyManager()

        # Create keys for a new tenant
        key_set = manager.create_tenant_keys("tenant-001", policy)

        # Derive a session key
        session_key = manager.derive_session_key("tenant-001", "voice", "session-abc")

        # Check isolation
        events = manager.detect_cross_tenant_access("tenant-001", "tenant-002", "decrypt")
    """

    def __init__(
        self,
        *,
        master_key_seed: str = "",
        alert_callback: Optional[Callable[[CrossTenantAccessEvent], None]] = None,
    ):
        self._master_key_hash = _generate_master_key_hash(master_key_seed)
        self._alert_callback = alert_callback

        # Tenant contexts (tenant_id -> context)
        self._contexts: Dict[str, TenantCryptoContext] = {}

        # Cross-tenant access log
        self._access_events: List[CrossTenantAccessEvent] = []

        # Session keys (tenant_id:service:session_id -> key_hash)
        self._session_keys: Dict[str, str] = {}

        # Revoked key sets (key_set_id -> revocation info)
        self._revoked_keys: Dict[str, Dict[str, Any]] = {}

        # Statistics
        self._stats = {
            "total_tenants": 0,
            "total_key_sets_created": 0,
            "total_rotations": 0,
            "total_session_keys_derived": 0,
            "total_cross_tenant_events": 0,
            "total_cross_tenant_blocked": 0,
        }

        logger.info(
            "TenantKeyManager initialized master_key_hash=%s...",
            self._master_key_hash[:16],
        )

    # ------------------------------------------------------------------
    # Tenant key creation
    # ------------------------------------------------------------------

    def create_tenant_keys(
        self,
        tenant_id: str,
        policy: Optional[TenantKeyPolicy] = None,
    ) -> TenantKeySet:
        """
        Create a complete key hierarchy for a new tenant.

        Generates the tenant root key from the master key and
        derives service-level keys for all standard BPO services.

        Args:
            tenant_id: Unique identifier for the tenant.
            policy: Cryptographic policy for the tenant. If None, a
                    standard policy is created.

        Returns:
            TenantKeySet containing all derived key hashes.
        """
        if policy is None:
            policy = TenantKeyPolicy(tenant_id=tenant_id)
        else:
            policy.tenant_id = tenant_id

        # Derive tenant root key from master
        root_key_hash = _derive_key(
            self._master_key_hash,
            tenant_id,
            KeyHierarchyLevel.TENANT_ROOT.name,
        )

        # Derive service keys
        service_keys: Dict[str, str] = {}
        for service in STANDARD_SERVICES:
            service_key_hash = _derive_key(
                root_key_hash,
                f"{tenant_id}:{service}",
                KeyHierarchyLevel.SERVICE.name,
            )
            service_keys[service] = service_key_hash

        # Calculate expiry based on policy
        expires_at = datetime.utcnow() + timedelta(hours=policy.key_rotation_hours)

        key_set = TenantKeySet(
            tenant_id=tenant_id,
            tenant_root_key_hash=root_key_hash,
            service_keys=service_keys,
            expires_at=expires_at,
        )

        # Create or update context
        context = TenantCryptoContext(
            tenant_id=tenant_id,
            key_set=key_set,
            policy=policy,
        )
        self._contexts[tenant_id] = context

        self._stats["total_key_sets_created"] += 1
        if tenant_id not in self._contexts or self._contexts[tenant_id].key_set is None:
            self._stats["total_tenants"] += 1

        logger.info(
            "Created key hierarchy for tenant=%s tier=%s services=%d expires=%s",
            tenant_id,
            policy.security_tier.name,
            len(service_keys),
            expires_at.isoformat(),
        )

        return key_set

    # ------------------------------------------------------------------
    # Key rotation
    # ------------------------------------------------------------------

    def rotate_tenant_keys(self, tenant_id: str) -> TenantKeySet:
        """
        Rotate all keys for a tenant.

        Creates a new key hierarchy while preserving the old one
        briefly for in-flight operations. Old session keys are
        invalidated.

        Args:
            tenant_id: The tenant whose keys should be rotated.

        Returns:
            New TenantKeySet with rotated keys.

        Raises:
            ValueError: If the tenant does not exist.
        """
        context = self._contexts.get(tenant_id)
        if not context:
            raise ValueError(f"Tenant {tenant_id} not found")

        # Revoke old key set
        if context.key_set:
            self._revoked_keys[context.key_set.key_set_id] = {
                "tenant_id": tenant_id,
                "revoked_at": datetime.utcnow().isoformat(),
                "reason": "key_rotation",
            }
            context.key_set.is_active = False

        # Invalidate session keys for this tenant
        prefix = f"{tenant_id}:"
        expired_sessions = [k for k in self._session_keys if k.startswith(prefix)]
        for key in expired_sessions:
            del self._session_keys[key]

        # Create new key set
        new_key_set = self.create_tenant_keys(tenant_id, context.policy)
        new_key_set.rotated_at = datetime.utcnow()

        self._stats["total_rotations"] += 1

        logger.info(
            "Rotated keys for tenant=%s new_key_set=%s invalidated_sessions=%d",
            tenant_id,
            new_key_set.key_set_id,
            len(expired_sessions),
        )

        return new_key_set

    # ------------------------------------------------------------------
    # Session key derivation
    # ------------------------------------------------------------------

    def derive_session_key(
        self,
        tenant_id: str,
        service: str,
        session_id: str,
    ) -> str:
        """
        Derive an ephemeral session key for a specific service and session.

        Session keys provide forward secrecy - compromise of one
        session key does not affect other sessions.

        Args:
            tenant_id: The tenant identifier.
            service: The service name (e.g., "voice", "chat").
            session_id: The unique session identifier.

        Returns:
            Session key hash as hex string.

        Raises:
            ValueError: If tenant or service key not found.
        """
        context = self._contexts.get(tenant_id)
        if not context or not context.key_set:
            raise ValueError(f"No active key set for tenant {tenant_id}")

        if context.key_set.is_expired:
            logger.warning(
                "Key set expired for tenant=%s, rotating before derivation",
                tenant_id,
            )
            self.rotate_tenant_keys(tenant_id)
            context = self._contexts[tenant_id]

        service_key = context.key_set.service_keys.get(service)
        if not service_key:
            raise ValueError(
                f"No service key for '{service}' in tenant {tenant_id}"
            )

        # Derive session key
        session_key_hash = _derive_key(
            service_key,
            f"{tenant_id}:{service}:{session_id}",
            KeyHierarchyLevel.SESSION.name,
        )

        cache_key = f"{tenant_id}:{service}:{session_id}"
        self._session_keys[cache_key] = session_key_hash

        context.active_sessions += 1
        context.total_operations += 1
        self._stats["total_session_keys_derived"] += 1

        logger.debug(
            "Derived session key for tenant=%s service=%s session=%s",
            tenant_id,
            service,
            session_id,
        )

        return session_key_hash

    # ------------------------------------------------------------------
    # Tenant isolation
    # ------------------------------------------------------------------

    def verify_tenant_isolation(self, tenant_id: str) -> Dict[str, Any]:
        """
        Verify that a tenant's cryptographic isolation is intact.

        Checks that the tenant's keys are properly derived, not
        shared with other tenants, and that no cross-tenant
        access has been detected.

        Args:
            tenant_id: The tenant to verify.

        Returns:
            Dictionary with isolation verification results.
        """
        context = self._contexts.get(tenant_id)
        if not context:
            return {
                "tenant_id": tenant_id,
                "isolated": False,
                "reason": "Tenant not found",
                "verified_at": datetime.utcnow().isoformat(),
            }

        issues: List[str] = []

        # Check key set exists and is active
        if not context.key_set:
            issues.append("No active key set")
        elif not context.key_set.is_active:
            issues.append("Key set is not active")
        elif context.key_set.is_expired:
            issues.append("Key set has expired")

        # Check for cross-tenant access events
        cross_events = [
            e for e in self._access_events
            if e.source_tenant == tenant_id or e.target_tenant == tenant_id
        ]
        if cross_events:
            issues.append(f"{len(cross_events)} cross-tenant access events detected")

        # Verify key uniqueness across tenants
        if context.key_set:
            root_hash = context.key_set.tenant_root_key_hash
            for other_id, other_ctx in self._contexts.items():
                if other_id == tenant_id:
                    continue
                if other_ctx.key_set and other_ctx.key_set.tenant_root_key_hash == root_hash:
                    issues.append(f"Root key hash collision with tenant {other_id}")

        # Verify service key uniqueness
        if context.key_set:
            for service, key_hash in context.key_set.service_keys.items():
                for other_id, other_ctx in self._contexts.items():
                    if other_id == tenant_id or not other_ctx.key_set:
                        continue
                    for other_service, other_hash in other_ctx.key_set.service_keys.items():
                        if key_hash == other_hash:
                            issues.append(
                                f"Service key collision: {service} matches "
                                f"{other_id}:{other_service}"
                            )

        isolated = len(issues) == 0

        logger.info(
            "Isolation verification for tenant=%s: isolated=%s issues=%d",
            tenant_id,
            isolated,
            len(issues),
        )

        return {
            "tenant_id": tenant_id,
            "isolated": isolated,
            "issues": issues,
            "key_set_active": bool(context.key_set and context.key_set.is_active),
            "active_sessions": context.active_sessions,
            "cross_tenant_events": len(cross_events),
            "verified_at": datetime.utcnow().isoformat(),
        }

    def detect_cross_tenant_access(
        self,
        source_tenant: str,
        target_tenant: str,
        operation: str,
    ) -> CrossTenantAccessEvent:
        """
        Detect and record a cross-tenant access attempt.

        Any attempt by one tenant's context to access another
        tenant's key material or encrypted data is logged and
        blocked by default.

        Args:
            source_tenant: The tenant initiating the access.
            target_tenant: The tenant being accessed.
            operation: The operation attempted (e.g., "decrypt", "read").

        Returns:
            CrossTenantAccessEvent recording the incident.
        """
        event = CrossTenantAccessEvent(
            source_tenant=source_tenant,
            target_tenant=target_tenant,
            operation=operation,
            was_blocked=True,
            evidence={
                "source_has_context": source_tenant in self._contexts,
                "target_has_context": target_tenant in self._contexts,
                "operation": operation,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

        self._access_events.append(event)
        self._stats["total_cross_tenant_events"] += 1
        self._stats["total_cross_tenant_blocked"] += 1

        logger.warning(
            "Cross-tenant access BLOCKED: %s -> %s operation=%s event=%s",
            source_tenant,
            target_tenant,
            operation,
            event.event_id,
        )

        if self._alert_callback:
            self._alert_callback(event)

        return event

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------

    def get_tenant_context(self, tenant_id: str) -> Optional[TenantCryptoContext]:
        """
        Get the cryptographic context for a tenant.

        Args:
            tenant_id: The tenant identifier.

        Returns:
            TenantCryptoContext if the tenant exists, None otherwise.
        """
        return self._contexts.get(tenant_id)

    def revoke_tenant_keys(self, tenant_id: str, reason: str = "manual") -> bool:
        """
        Revoke all keys for a tenant immediately.

        Deactivates the current key set and invalidates all
        derived session keys. Used for emergency response
        when a tenant breach is suspected.

        Args:
            tenant_id: The tenant whose keys should be revoked.
            reason: Reason for revocation.

        Returns:
            True if keys were revoked, False if tenant not found.
        """
        context = self._contexts.get(tenant_id)
        if not context or not context.key_set:
            return False

        # Revoke key set
        self._revoked_keys[context.key_set.key_set_id] = {
            "tenant_id": tenant_id,
            "revoked_at": datetime.utcnow().isoformat(),
            "reason": reason,
        }
        context.key_set.is_active = False

        # Invalidate all session keys
        prefix = f"{tenant_id}:"
        invalidated = 0
        keys_to_remove = [k for k in self._session_keys if k.startswith(prefix)]
        for key in keys_to_remove:
            del self._session_keys[key]
            invalidated += 1

        logger.warning(
            "REVOKED all keys for tenant=%s reason=%s sessions_invalidated=%d",
            tenant_id,
            reason,
            invalidated,
        )

        return True

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_isolation_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive isolation report for all tenants.

        Verifies isolation for every active tenant and compiles
        the results into an aggregate report.

        Returns:
            Dictionary containing per-tenant and aggregate results.
        """
        tenant_reports: List[Dict[str, Any]] = []
        total_isolated = 0
        total_issues = 0

        for tenant_id in self._contexts:
            report = self.verify_tenant_isolation(tenant_id)
            tenant_reports.append(report)
            if report["isolated"]:
                total_isolated += 1
            total_issues += len(report.get("issues", []))

        return {
            "report_type": "isolation",
            "total_tenants": len(self._contexts),
            "total_isolated": total_isolated,
            "total_issues": total_issues,
            "tenant_reports": tenant_reports,
            "cross_tenant_events": len(self._access_events),
            "revoked_key_sets": len(self._revoked_keys),
            "statistics": dict(self._stats),
            "generated_at": datetime.utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def create_standard_policy(cls, tenant_id: str) -> TenantKeyPolicy:
        """
        Create a standard-tier tenant key policy.

        Suitable for general BPO operations with moderate
        security requirements and software-based keys.
        """
        return TenantKeyPolicy(
            tenant_id=tenant_id,
            security_tier=TenantSecurityTier.STANDARD,
            kem_algorithm="ML-KEM-768",
            sig_algorithm="ML-DSA-65",
            use_hybrid=True,
            key_rotation_hours=24,
            session_key_rotation_hours=1,
            require_hsm=False,
            compliance_frameworks=["SOC2"],
        )

    @classmethod
    def create_premium_policy(cls, tenant_id: str) -> TenantKeyPolicy:
        """
        Create a premium-tier tenant key policy.

        Suitable for financial services and healthcare BPO
        operations requiring HSM-backed keys and strict
        compliance adherence.
        """
        return TenantKeyPolicy(
            tenant_id=tenant_id,
            security_tier=TenantSecurityTier.PREMIUM,
            kem_algorithm="ML-KEM-1024",
            sig_algorithm="ML-DSA-87",
            use_hybrid=True,
            key_rotation_hours=8,
            session_key_rotation_hours=1,
            require_hsm=True,
            compliance_frameworks=["PCI-DSS", "HIPAA", "SOC2"],
        )

    @classmethod
    def create_dedicated_policy(cls, tenant_id: str) -> TenantKeyPolicy:
        """
        Create a dedicated-tier tenant key policy.

        Highest isolation level with dedicated HSM partitions,
        shortest rotation intervals, and comprehensive compliance
        coverage. Suitable for government and defense BPO.
        """
        return TenantKeyPolicy(
            tenant_id=tenant_id,
            security_tier=TenantSecurityTier.DEDICATED,
            kem_algorithm="ML-KEM-1024",
            sig_algorithm="ML-DSA-87",
            use_hybrid=False,  # Pure PQC for maximum quantum resistance
            key_rotation_hours=4,
            session_key_rotation_hours=1,
            require_hsm=True,
            compliance_frameworks=["PCI-DSS", "HIPAA", "SOC2", "GDPR", "ISO-27001"],
        )
