"""
Key Manager

Central component for managing cryptographic keys throughout their lifecycle.
"""

import uuid
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from ai_engine.domains.banking.security.hsm import (
    HSMProvider,
    HSMKeyType,
    HSMAlgorithm,
    HSMKeyHandle,
)
from ai_engine.domains.banking.security.key_management.key_types import (
    KeyState,
    KeyPurpose,
    KeyInfo,
    KeyRotationPolicy,
)


class KeyManagerError(Exception):
    """Exception for key management operations."""

    pass


class KeyManager:
    """
    Key Manager for comprehensive key lifecycle management.

    Features:
    - Key generation with HSM
    - Key state management
    - Key rotation
    - Key versioning
    - Access control
    - Audit logging
    """

    def __init__(
        self,
        hsm_provider: HSMProvider,
        audit_callback: Optional[Callable[[str, Dict], None]] = None,
    ):
        """
        Initialize key manager.

        Args:
            hsm_provider: HSM provider for key operations
            audit_callback: Optional callback for audit logging
        """
        self._hsm = hsm_provider
        self._audit_callback = audit_callback
        self._keys: Dict[str, KeyInfo] = {}
        self._aliases: Dict[str, str] = {}  # alias -> key_id

        # Default policies
        self._default_rotation_policy = KeyRotationPolicy()

    def generate_key(
        self,
        alias: str,
        key_type: HSMKeyType,
        purpose: KeyPurpose,
        owner: Optional[str] = None,
        rotation_policy: Optional[KeyRotationPolicy] = None,
        auto_activate: bool = True,
        expires_in_days: Optional[int] = None,
        tags: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> KeyInfo:
        """
        Generate a new key.

        Args:
            alias: Human-readable alias for the key
            key_type: Type of key to generate
            purpose: Primary purpose of the key
            owner: Key owner identifier
            rotation_policy: Rotation policy for the key
            auto_activate: If True, activate key immediately
            expires_in_days: Days until key expires
            tags: Additional tags for the key

        Returns:
            KeyInfo for the generated key
        """
        # Check alias uniqueness
        if alias in self._aliases:
            raise KeyManagerError(f"Alias already exists: {alias}")

        key_id = str(uuid.uuid4())

        # Generate key in HSM
        if key_type.is_symmetric:
            hsm_handle = self._hsm.generate_key(
                key_type=key_type,
                label=f"{alias}_{key_id[:8]}",
                extractable=kwargs.get("extractable", False),
            )
        else:
            pub_handle, priv_handle = self._hsm.generate_key_pair(
                key_type=key_type,
                label=f"{alias}_{key_id[:8]}",
                extractable=kwargs.get("extractable", False),
            )
            # For asymmetric keys, track the private key handle
            hsm_handle = priv_handle
            kwargs["public_key_handle"] = pub_handle.key_id

        # Calculate expiry
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        # Create key info
        key_info = KeyInfo(
            key_id=key_id,
            alias=alias,
            key_type=key_type.algorithm_name,
            state=KeyState.PRE_ACTIVATION,
            purpose=purpose,
            rotation_policy=rotation_policy or self._default_rotation_policy,
            expires_at=expires_at,
            hsm_key_handle=hsm_handle.key_id,
            owner=owner,
            tags=tags or {},
            extractable=kwargs.get("extractable", False),
        )

        # Store key info
        self._keys[key_id] = key_info
        self._aliases[alias] = key_id

        # Auto-activate if requested
        if auto_activate:
            key_info.activate()

        # Audit log
        self._audit(
            "key_generated",
            {
                "key_id": key_id,
                "alias": alias,
                "key_type": key_type.algorithm_name,
                "purpose": purpose.value,
                "owner": owner,
            },
        )

        return key_info

    def get_key(self, key_id_or_alias: str) -> Optional[KeyInfo]:
        """Get key by ID or alias."""
        # Check if it's an alias
        if key_id_or_alias in self._aliases:
            key_id = self._aliases[key_id_or_alias]
        else:
            key_id = key_id_or_alias

        return self._keys.get(key_id)

    def get_active_key(self, alias: str) -> Optional[KeyInfo]:
        """Get the current active key for an alias."""
        key_info = self.get_key(alias)
        if key_info and key_info.is_active:
            return key_info

        # If key has been rotated, find the newest active version
        if key_info and key_info.next_key_id:
            return self.get_active_key(key_info.next_key_id)

        return None

    def list_keys(
        self,
        state: Optional[KeyState] = None,
        purpose: Optional[KeyPurpose] = None,
        owner: Optional[str] = None,
        include_destroyed: bool = False,
    ) -> List[KeyInfo]:
        """List keys matching criteria."""
        keys = []

        for key_info in self._keys.values():
            # Filter by state
            if state and key_info.state != state:
                continue

            # Filter by purpose
            if purpose and key_info.purpose != purpose:
                continue

            # Filter by owner
            if owner and key_info.owner != owner:
                continue

            # Filter destroyed
            if not include_destroyed and key_info.state == KeyState.DESTROYED:
                continue

            keys.append(key_info)

        return keys

    def activate_key(self, key_id_or_alias: str) -> KeyInfo:
        """Activate a key."""
        key_info = self.get_key(key_id_or_alias)
        if not key_info:
            raise KeyManagerError(f"Key not found: {key_id_or_alias}")

        key_info.activate()

        self._audit(
            "key_activated",
            {
                "key_id": key_info.key_id,
                "alias": key_info.alias,
            },
        )

        return key_info

    def deactivate_key(self, key_id_or_alias: str) -> KeyInfo:
        """Deactivate a key."""
        key_info = self.get_key(key_id_or_alias)
        if not key_info:
            raise KeyManagerError(f"Key not found: {key_id_or_alias}")

        key_info.deactivate()

        self._audit(
            "key_deactivated",
            {
                "key_id": key_info.key_id,
                "alias": key_info.alias,
            },
        )

        return key_info

    def rotate_key(
        self,
        key_id_or_alias: str,
        reason: str = "scheduled",
    ) -> KeyInfo:
        """
        Rotate a key by generating a new version.

        Args:
            key_id_or_alias: Key to rotate
            reason: Reason for rotation

        Returns:
            New key info
        """
        old_key = self.get_key(key_id_or_alias)
        if not old_key:
            raise KeyManagerError(f"Key not found: {key_id_or_alias}")

        # Get HSM key type from string
        key_type = self._get_hsm_key_type(old_key.key_type)

        # Generate new key with same parameters
        new_alias = f"{old_key.alias}_v{old_key.version + 1}"

        new_key = self.generate_key(
            alias=new_alias,
            key_type=key_type,
            purpose=old_key.purpose,
            owner=old_key.owner,
            rotation_policy=old_key.rotation_policy,
            tags=old_key.tags,
        )

        # Link keys
        new_key.previous_key_id = old_key.key_id
        new_key.version = old_key.version + 1
        old_key.next_key_id = new_key.key_id

        # Update alias to point to new key
        self._aliases[old_key.alias] = new_key.key_id

        # Deactivate old key if policy says so
        if old_key.rotation_policy and old_key.rotation_policy.deactivate_old_key:
            old_key.deactivate()

        self._audit(
            "key_rotated",
            {
                "old_key_id": old_key.key_id,
                "new_key_id": new_key.key_id,
                "reason": reason,
            },
        )

        return new_key

    def destroy_key(
        self,
        key_id_or_alias: str,
        reason: str = "end_of_life",
    ) -> None:
        """
        Destroy a key (mark as destroyed and delete from HSM).

        Args:
            key_id_or_alias: Key to destroy
            reason: Reason for destruction
        """
        key_info = self.get_key(key_id_or_alias)
        if not key_info:
            raise KeyManagerError(f"Key not found: {key_id_or_alias}")

        if key_info.state == KeyState.DESTROYED:
            return  # Already destroyed

        # Delete from HSM
        if key_info.hsm_key_handle:
            hsm_handle = self._hsm.get_key(key_info.hsm_key_handle)
            if hsm_handle:
                self._hsm.delete_key(hsm_handle)

        # Mark as destroyed
        key_info.destroy()

        self._audit(
            "key_destroyed",
            {
                "key_id": key_info.key_id,
                "alias": key_info.alias,
                "reason": reason,
            },
        )

    def mark_compromised(
        self,
        key_id_or_alias: str,
        reason: str = "unknown",
    ) -> None:
        """Mark a key as compromised."""
        key_info = self.get_key(key_id_or_alias)
        if not key_info:
            raise KeyManagerError(f"Key not found: {key_id_or_alias}")

        key_info.compromise()

        # Rotate immediately if policy says so
        if key_info.rotation_policy and key_info.rotation_policy.rotate_on_compromise:
            self.rotate_key(key_id_or_alias, reason="compromise")

        self._audit(
            "key_compromised",
            {
                "key_id": key_info.key_id,
                "alias": key_info.alias,
                "reason": reason,
            },
        )

    def get_keys_needing_rotation(self) -> List[KeyInfo]:
        """Get list of keys that need rotation."""
        return [key for key in self._keys.values() if key.state == KeyState.ACTIVE and key.needs_rotation]

    def get_expiring_keys(self, days: int = 30) -> List[KeyInfo]:
        """Get list of keys expiring within specified days."""
        keys = []
        cutoff = datetime.utcnow() + timedelta(days=days)

        for key_info in self._keys.values():
            if key_info.state != KeyState.ACTIVE:
                continue
            if key_info.expires_at and key_info.expires_at <= cutoff:
                keys.append(key_info)

        return sorted(keys, key=lambda k: k.expires_at or datetime.max)

    def record_usage(
        self,
        key_id_or_alias: str,
        operation: str,
        bytes_processed: int = 0,
    ) -> None:
        """Record key usage for tracking."""
        key_info = self.get_key(key_id_or_alias)
        if not key_info:
            return

        key_info.record_usage(bytes_processed)

        self._audit(
            "key_used",
            {
                "key_id": key_info.key_id,
                "operation": operation,
                "bytes_processed": bytes_processed,
            },
        )

    def export_key_inventory(self) -> List[Dict[str, Any]]:
        """Export key inventory for reporting."""
        return [key.to_dict() for key in self._keys.values()]

    def _get_hsm_key_type(self, key_type_str: str) -> HSMKeyType:
        """Convert key type string to HSMKeyType."""
        mapping = {
            "AES-128": HSMKeyType.AES_128,
            "AES-192": HSMKeyType.AES_192,
            "AES-256": HSMKeyType.AES_256,
            "3DES": HSMKeyType.DES3,
            "RSA-2048": HSMKeyType.RSA_2048,
            "RSA-3072": HSMKeyType.RSA_3072,
            "RSA-4096": HSMKeyType.RSA_4096,
            "EC-P256": HSMKeyType.EC_P256,
            "EC-P384": HSMKeyType.EC_P384,
            "EC-P521": HSMKeyType.EC_P521,
            "ML-KEM-512": HSMKeyType.ML_KEM_512,
            "ML-KEM-768": HSMKeyType.ML_KEM_768,
            "ML-KEM-1024": HSMKeyType.ML_KEM_1024,
            "ML-DSA-44": HSMKeyType.ML_DSA_44,
            "ML-DSA-65": HSMKeyType.ML_DSA_65,
            "ML-DSA-87": HSMKeyType.ML_DSA_87,
        }

        if key_type_str not in mapping:
            raise KeyManagerError(f"Unknown key type: {key_type_str}")

        return mapping[key_type_str]

    def _audit(self, event: str, data: Dict[str, Any]) -> None:
        """Log audit event."""
        if self._audit_callback:
            audit_record = {
                "event": event,
                "timestamp": datetime.utcnow().isoformat(),
                "data": data,
            }
            self._audit_callback(event, audit_record)
