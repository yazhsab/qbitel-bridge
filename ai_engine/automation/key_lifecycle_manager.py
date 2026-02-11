"""
Key Lifecycle Manager

Automated cryptographic key lifecycle management including:
- Key generation with PQC support
- Automatic rotation
- Secure destruction
- Audit logging
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import uuid
import threading


logger = logging.getLogger(__name__)


class KeyState(Enum):
    """Key lifecycle states."""

    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DEACTIVATED = "deactivated"
    COMPROMISED = "compromised"
    DESTROYED = "destroyed"


class KeyType(Enum):
    """Types of cryptographic keys."""

    KEK = "key_encryption_key"
    DEK = "data_encryption_key"
    SIGNING = "signing_key"
    VERIFICATION = "verification_key"
    TRANSPORT = "transport_key"
    MASTER = "master_key"


@dataclass
class KeyPolicy:
    """Policy for key management."""

    key_type: KeyType
    algorithm: str = "ML-KEM-768"  # Default to PQC
    key_size: int = 256
    rotation_period_days: int = 90
    activation_delay_hours: int = 0
    expiry_warning_days: int = 14
    auto_rotate: bool = True
    requires_hsm: bool = True
    backup_required: bool = True
    dual_control: bool = False
    pqc_hybrid: bool = True  # Use hybrid classical+PQC


@dataclass
class KeyMetadata:
    """Metadata for a managed key."""

    key_id: str
    key_type: KeyType
    state: KeyState
    algorithm: str
    created_at: datetime
    activated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    last_rotated_at: Optional[datetime] = None
    rotation_count: int = 0
    hsm_id: Optional[str] = None
    backup_id: Optional[str] = None
    policy_id: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class KeyRotationEvent:
    """Event for key rotation."""

    event_id: str
    old_key_id: str
    new_key_id: str
    rotated_at: datetime
    reason: str
    automatic: bool
    completed: bool = False
    error: Optional[str] = None


class KeyLifecycleManager:
    """
    Automated key lifecycle manager.

    Capabilities:
    - PQC-ready key generation
    - Automatic rotation scheduling
    - HSM integration
    - Audit logging
    - Compliance tracking
    """

    def __init__(
        self,
        hsm_provider: Optional[Any] = None,
        rotation_callback: Optional[Callable[[KeyRotationEvent], None]] = None,
    ):
        self._hsm = hsm_provider
        self._rotation_callback = rotation_callback
        self._keys: Dict[str, KeyMetadata] = {}
        self._policies: Dict[str, KeyPolicy] = {}
        self._rotation_events: List[KeyRotationEvent] = []
        self._lock = threading.RLock()
        self._rotation_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()

        # Initialize default policies
        self._init_default_policies()

    def generate_key(
        self,
        key_type: KeyType,
        policy_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> KeyMetadata:
        """
        Generate a new cryptographic key.

        Args:
            key_type: Type of key to generate
            policy_id: Policy to apply (uses default if None)
            tags: Optional tags for the key

        Returns:
            KeyMetadata for the generated key
        """
        with self._lock:
            # Get policy
            policy = self._get_policy(key_type, policy_id)

            # Generate key ID
            key_id = str(uuid.uuid4())

            # Calculate expiry
            expires_at = datetime.utcnow() + timedelta(days=policy.rotation_period_days)

            # Generate in HSM if available and required
            hsm_id = None
            if self._hsm and policy.requires_hsm:
                try:
                    # Generate key in HSM
                    hsm_key = self._hsm.generate_key(
                        key_type=self._map_key_type(policy.algorithm),
                        label=key_id,
                        extractable=False,
                    )
                    hsm_id = hsm_key.key_id
                except Exception as e:
                    logger.warning(f"HSM key generation failed, using software: {e}")

            # Create metadata
            metadata = KeyMetadata(
                key_id=key_id,
                key_type=key_type,
                state=KeyState.PENDING if policy.activation_delay_hours > 0 else KeyState.ACTIVE,
                algorithm=policy.algorithm,
                created_at=datetime.utcnow(),
                activated_at=datetime.utcnow() if policy.activation_delay_hours == 0 else None,
                expires_at=expires_at,
                hsm_id=hsm_id,
                policy_id=policy_id,
                tags=tags or {},
            )

            self._keys[key_id] = metadata

            logger.info(f"Generated key {key_id} ({key_type.value}, {policy.algorithm})")

            return metadata

    def rotate_key(
        self,
        key_id: str,
        reason: str = "Scheduled rotation",
        automatic: bool = False,
    ) -> KeyRotationEvent:
        """
        Rotate a key.

        Args:
            key_id: ID of key to rotate
            reason: Reason for rotation
            automatic: Whether this is automatic rotation

        Returns:
            KeyRotationEvent
        """
        with self._lock:
            if key_id not in self._keys:
                raise ValueError(f"Key not found: {key_id}")

            old_metadata = self._keys[key_id]

            # Generate new key
            new_metadata = self.generate_key(
                key_type=old_metadata.key_type,
                policy_id=old_metadata.policy_id,
                tags=old_metadata.tags,
            )

            # Update old key state
            old_metadata.state = KeyState.DEACTIVATED

            # Update new key rotation count
            new_metadata.rotation_count = old_metadata.rotation_count + 1
            new_metadata.last_rotated_at = datetime.utcnow()

            # Create rotation event
            event = KeyRotationEvent(
                event_id=str(uuid.uuid4()),
                old_key_id=key_id,
                new_key_id=new_metadata.key_id,
                rotated_at=datetime.utcnow(),
                reason=reason,
                automatic=automatic,
                completed=True,
            )

            self._rotation_events.append(event)

            # Notify callback
            if self._rotation_callback:
                try:
                    self._rotation_callback(event)
                except Exception as e:
                    logger.error(f"Rotation callback failed: {e}")

            logger.info(f"Rotated key {key_id} -> {new_metadata.key_id}")

            return event

    def destroy_key(
        self,
        key_id: str,
        reason: str = "End of lifecycle",
    ) -> bool:
        """
        Securely destroy a key.

        Args:
            key_id: ID of key to destroy
            reason: Reason for destruction

        Returns:
            True if successful
        """
        with self._lock:
            if key_id not in self._keys:
                return False

            metadata = self._keys[key_id]

            # Destroy in HSM if applicable
            if metadata.hsm_id and self._hsm:
                try:
                    hsm_key = self._hsm.get_key(metadata.hsm_id)
                    if hsm_key:
                        self._hsm.delete_key(hsm_key)
                except Exception as e:
                    logger.error(f"HSM key destruction failed: {e}")

            # Update state
            metadata.state = KeyState.DESTROYED

            logger.info(f"Destroyed key {key_id}: {reason}")

            return True

    def suspend_key(self, key_id: str, reason: str = "") -> bool:
        """Suspend a key temporarily."""
        with self._lock:
            if key_id not in self._keys:
                return False

            self._keys[key_id].state = KeyState.SUSPENDED
            logger.info(f"Suspended key {key_id}: {reason}")
            return True

    def activate_key(self, key_id: str) -> bool:
        """Activate a pending or suspended key."""
        with self._lock:
            if key_id not in self._keys:
                return False

            metadata = self._keys[key_id]
            if metadata.state in (KeyState.PENDING, KeyState.SUSPENDED):
                metadata.state = KeyState.ACTIVE
                metadata.activated_at = datetime.utcnow()
                logger.info(f"Activated key {key_id}")
                return True

            return False

    def mark_compromised(self, key_id: str, reason: str = "") -> bool:
        """Mark a key as compromised."""
        with self._lock:
            if key_id not in self._keys:
                return False

            self._keys[key_id].state = KeyState.COMPROMISED

            logger.warning(f"Key {key_id} marked COMPROMISED: {reason}")

            # Auto-rotate
            self.rotate_key(key_id, f"Compromised: {reason}", automatic=True)

            return True

    def get_key(self, key_id: str) -> Optional[KeyMetadata]:
        """Get key metadata."""
        return self._keys.get(key_id)

    def list_keys(
        self,
        key_type: Optional[KeyType] = None,
        state: Optional[KeyState] = None,
    ) -> List[KeyMetadata]:
        """List keys with optional filtering."""
        keys = list(self._keys.values())

        if key_type:
            keys = [k for k in keys if k.key_type == key_type]

        if state:
            keys = [k for k in keys if k.state == state]

        return keys

    def get_expiring_keys(
        self,
        days: int = 14,
    ) -> List[KeyMetadata]:
        """Get keys expiring within specified days."""
        cutoff = datetime.utcnow() + timedelta(days=days)
        return [
            k for k in self._keys.values()
            if k.expires_at and k.expires_at < cutoff and k.state == KeyState.ACTIVE
        ]

    def start_rotation_scheduler(self, check_interval_seconds: int = 3600) -> None:
        """Start background rotation scheduler."""
        if self._rotation_thread and self._rotation_thread.is_alive():
            return

        self._shutdown_event.clear()

        def rotation_loop():
            while not self._shutdown_event.is_set():
                try:
                    self._check_and_rotate()
                except Exception as e:
                    logger.error(f"Rotation check failed: {e}")

                self._shutdown_event.wait(check_interval_seconds)

        self._rotation_thread = threading.Thread(
            target=rotation_loop,
            daemon=True,
            name="key-rotation-scheduler",
        )
        self._rotation_thread.start()
        logger.info("Key rotation scheduler started")

    def stop_rotation_scheduler(self) -> None:
        """Stop the rotation scheduler."""
        self._shutdown_event.set()
        if self._rotation_thread:
            self._rotation_thread.join(timeout=5)
        logger.info("Key rotation scheduler stopped")

    def _check_and_rotate(self) -> None:
        """Check for keys needing rotation and rotate them."""
        now = datetime.utcnow()

        for key_id, metadata in list(self._keys.items()):
            if metadata.state != KeyState.ACTIVE:
                continue

            # Get policy
            policy = self._get_policy(metadata.key_type, metadata.policy_id)

            if not policy.auto_rotate:
                continue

            # Check if rotation is due
            if metadata.expires_at and now >= metadata.expires_at:
                try:
                    self.rotate_key(key_id, "Automatic scheduled rotation", automatic=True)
                except Exception as e:
                    logger.error(f"Auto-rotation failed for {key_id}: {e}")

    def _init_default_policies(self) -> None:
        """Initialize default key policies."""
        self._policies["default_kek"] = KeyPolicy(
            key_type=KeyType.KEK,
            algorithm="ML-KEM-768",
            rotation_period_days=365,
            requires_hsm=True,
        )

        self._policies["default_dek"] = KeyPolicy(
            key_type=KeyType.DEK,
            algorithm="AES-256",
            rotation_period_days=90,
            requires_hsm=False,
        )

        self._policies["default_signing"] = KeyPolicy(
            key_type=KeyType.SIGNING,
            algorithm="ML-DSA-65",
            rotation_period_days=365,
            requires_hsm=True,
        )

    def _get_policy(
        self,
        key_type: KeyType,
        policy_id: Optional[str],
    ) -> KeyPolicy:
        """Get policy for key type."""
        if policy_id and policy_id in self._policies:
            return self._policies[policy_id]

        # Return type-specific default
        default_map = {
            KeyType.KEK: "default_kek",
            KeyType.DEK: "default_dek",
            KeyType.SIGNING: "default_signing",
        }

        policy_key = default_map.get(key_type, "default_dek")
        return self._policies.get(policy_key, KeyPolicy(key_type=key_type))

    def _map_key_type(self, algorithm: str) -> Any:
        """Map algorithm string to HSM key type."""
        # This would map to actual HSMKeyType enum
        return algorithm

    def get_rotation_history(
        self,
        key_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[KeyRotationEvent]:
        """Get rotation history."""
        events = self._rotation_events

        if key_id:
            events = [
                e for e in events
                if e.old_key_id == key_id or e.new_key_id == key_id
            ]

        return events[-limit:]
