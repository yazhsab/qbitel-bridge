"""
Key Management Type Definitions

Defines key states, purposes, and metadata structures.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional


class KeyState(Enum):
    """
    Key lifecycle states as per NIST SP 800-57.

    The key states follow the key lifecycle:
    PRE_ACTIVATION -> ACTIVE -> DEACTIVATED -> COMPROMISED/DESTROYED
    """

    PRE_ACTIVATION = "pre_activation"  # Generated but not yet active
    ACTIVE = "active"  # Can be used for all permitted operations
    SUSPENDED = "suspended"  # Temporarily disabled
    DEACTIVATED = "deactivated"  # Can only decrypt/verify, not encrypt/sign
    COMPROMISED = "compromised"  # Should not be used, retained for verification
    DESTROYED = "destroyed"  # Key material securely deleted

    @property
    def can_encrypt(self) -> bool:
        """Check if key can be used for encryption."""
        return self == KeyState.ACTIVE

    @property
    def can_decrypt(self) -> bool:
        """Check if key can be used for decryption."""
        return self in (KeyState.ACTIVE, KeyState.DEACTIVATED)

    @property
    def can_sign(self) -> bool:
        """Check if key can be used for signing."""
        return self == KeyState.ACTIVE

    @property
    def can_verify(self) -> bool:
        """Check if key can be used for verification."""
        return self in (KeyState.ACTIVE, KeyState.DEACTIVATED, KeyState.COMPROMISED)


class KeyPurpose(Enum):
    """Key usage purposes."""

    # Encryption purposes
    DATA_ENCRYPTION = "data_encryption"  # Encrypting data at rest
    KEY_ENCRYPTION = "key_encryption"  # Encrypting other keys (KEK)
    TRANSPORT_ENCRYPTION = "transport_encryption"  # TLS, etc.

    # Signing purposes
    DIGITAL_SIGNATURE = "digital_signature"  # General signing
    CODE_SIGNING = "code_signing"  # Software signing
    CERTIFICATE_SIGNING = "certificate_signing"  # CA operations

    # Authentication
    AUTHENTICATION = "authentication"  # User/system authentication
    MAC = "mac"  # Message authentication codes

    # Key agreement
    KEY_AGREEMENT = "key_agreement"  # Diffie-Hellman, etc.
    KEY_DERIVATION = "key_derivation"  # Deriving other keys

    # Payment specific
    PIN_ENCRYPTION = "pin_encryption"  # PIN block encryption
    PIN_VERIFICATION = "pin_verification"  # PVV generation
    CVV_GENERATION = "cvv_generation"  # Card verification
    EMV_PROCESSING = "emv_processing"  # EMV cryptogram operations

    # PQC specific
    PQC_KEY_ENCAPSULATION = "pqc_kem"  # ML-KEM operations
    PQC_DIGITAL_SIGNATURE = "pqc_signature"  # ML-DSA operations


@dataclass
class KeyRotationPolicy:
    """Policy for key rotation."""

    # Rotation triggers
    max_age_days: int = 365  # Maximum key age
    max_usage_count: Optional[int] = None  # Maximum uses before rotation
    max_data_encrypted_bytes: Optional[int] = None  # Maximum data encrypted

    # Pre-rotation settings
    pre_rotation_days: int = 30  # Days before expiry to generate new key
    overlap_days: int = 7  # Days both old and new keys are active

    # Rotation behavior
    auto_rotate: bool = True
    rotate_on_compromise: bool = True
    notify_before_days: int = 60  # Days before expiry to send notification

    # Post-rotation
    deactivate_old_key: bool = True
    retain_old_key_days: int = 90  # Days to retain old key for decryption

    def should_rotate(
        self,
        created_at: datetime,
        usage_count: int = 0,
        bytes_encrypted: int = 0,
    ) -> bool:
        """Check if key should be rotated based on policy."""
        now = datetime.utcnow()

        # Check age
        age_days = (now - created_at).days
        if age_days >= self.max_age_days:
            return True

        # Check usage count
        if self.max_usage_count and usage_count >= self.max_usage_count:
            return True

        # Check data encrypted
        if self.max_data_encrypted_bytes and bytes_encrypted >= self.max_data_encrypted_bytes:
            return True

        return False

    def should_pre_rotate(self, created_at: datetime) -> bool:
        """Check if we should start pre-rotation."""
        now = datetime.utcnow()
        age_days = (now - created_at).days
        rotation_threshold = self.max_age_days - self.pre_rotation_days
        return age_days >= rotation_threshold


@dataclass
class KeyInfo:
    """Complete key information and metadata."""

    # Identity
    key_id: str
    alias: str
    key_type: str  # e.g., "AES-256", "RSA-2048", "ML-KEM-768"

    # State
    state: KeyState = KeyState.PRE_ACTIVATION

    # Purpose
    purpose: KeyPurpose = KeyPurpose.DATA_ENCRYPTION
    allowed_purposes: List[KeyPurpose] = field(default_factory=list)

    # Lifecycle dates
    created_at: Optional[datetime] = None
    activated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    deactivated_at: Optional[datetime] = None
    destroyed_at: Optional[datetime] = None

    # Rotation
    rotation_policy: Optional[KeyRotationPolicy] = None
    previous_key_id: Optional[str] = None
    next_key_id: Optional[str] = None
    version: int = 1

    # Usage tracking
    usage_count: int = 0
    bytes_encrypted: int = 0
    last_used_at: Optional[datetime] = None

    # Storage
    hsm_key_handle: Optional[str] = None
    key_store: str = "hsm"  # "hsm", "software", "cloud_kms"

    # Attributes
    extractable: bool = False
    exportable: bool = False
    sensitive: bool = True

    # Access control
    owner: Optional[str] = None
    authorized_users: List[str] = field(default_factory=list)
    authorized_applications: List[str] = field(default_factory=list)

    # Metadata
    description: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    custom_attributes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow()
        if not self.allowed_purposes:
            self.allowed_purposes = [self.purpose]

    @property
    def is_active(self) -> bool:
        """Check if key is currently active."""
        if self.state != KeyState.ACTIVE:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True

    @property
    def is_expired(self) -> bool:
        """Check if key has expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at

    @property
    def days_until_expiry(self) -> Optional[int]:
        """Get days until key expires."""
        if not self.expires_at:
            return None
        delta = self.expires_at - datetime.utcnow()
        return max(0, delta.days)

    @property
    def needs_rotation(self) -> bool:
        """Check if key needs rotation based on policy."""
        if not self.rotation_policy:
            return False
        if not self.created_at:
            return False
        return self.rotation_policy.should_rotate(
            self.created_at,
            self.usage_count,
            self.bytes_encrypted,
        )

    def can_perform(self, operation: str) -> bool:
        """Check if key can perform operation in current state."""
        if operation in ("encrypt", "wrap"):
            return self.state.can_encrypt
        elif operation in ("decrypt", "unwrap"):
            return self.state.can_decrypt
        elif operation == "sign":
            return self.state.can_sign
        elif operation == "verify":
            return self.state.can_verify
        return False

    def record_usage(self, bytes_processed: int = 0) -> None:
        """Record key usage."""
        self.usage_count += 1
        self.bytes_encrypted += bytes_processed
        self.last_used_at = datetime.utcnow()

    def activate(self) -> None:
        """Activate the key."""
        if self.state not in (KeyState.PRE_ACTIVATION, KeyState.SUSPENDED):
            raise ValueError(f"Cannot activate key in state: {self.state}")
        self.state = KeyState.ACTIVE
        self.activated_at = datetime.utcnow()

    def deactivate(self) -> None:
        """Deactivate the key."""
        if self.state != KeyState.ACTIVE:
            raise ValueError(f"Cannot deactivate key in state: {self.state}")
        self.state = KeyState.DEACTIVATED
        self.deactivated_at = datetime.utcnow()

    def compromise(self) -> None:
        """Mark key as compromised."""
        self.state = KeyState.COMPROMISED

    def destroy(self) -> None:
        """Mark key as destroyed."""
        self.state = KeyState.DESTROYED
        self.destroyed_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key_id": self.key_id,
            "alias": self.alias,
            "key_type": self.key_type,
            "state": self.state.value,
            "purpose": self.purpose.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "activated_at": self.activated_at.isoformat() if self.activated_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "version": self.version,
            "usage_count": self.usage_count,
            "is_active": self.is_active,
            "needs_rotation": self.needs_rotation,
            "owner": self.owner,
            "tags": self.tags,
        }
