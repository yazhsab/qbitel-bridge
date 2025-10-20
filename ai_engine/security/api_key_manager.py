"""
CRONOS AI Engine - API Key Management and Rotation

Production-ready API key lifecycle management with automatic expiration,
rotation, and compliance enforcement.
"""

import secrets
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
from sqlalchemy.exc import IntegrityError

from ai_engine.models.database import APIKey, User
from ai_engine.core.transaction_manager import transactional

logger = logging.getLogger(__name__)


class APIKeyStatus(str, Enum):
    """API key status."""

    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    PENDING_ROTATION = "pending_rotation"


class APIKeyManager:
    """
    Manages API key lifecycle including creation, rotation, and expiration.

    Features:
    - Automatic key generation with cryptographic randomness
    - Configurable expiration policies
    - Warning period before expiration
    - Automatic rotation support
    - Revocation and audit logging
    """

    def __init__(
        self,
        default_expiration_days: int = 90,
        warning_days_before_expiration: int = 7,
        max_keys_per_user: int = 5,
    ):
        """
        Initialize API key manager.

        Args:
            default_expiration_days: Default key expiration in days
            warning_days_before_expiration: Days before expiration to warn
            max_keys_per_user: Maximum active keys per user
        """
        self.default_expiration_days = default_expiration_days
        self.warning_days_before_expiration = warning_days_before_expiration
        self.max_keys_per_user = max_keys_per_user

        logger.info(
            f"API Key Manager initialized "
            f"(expiration={default_expiration_days}d, warning={warning_days_before_expiration}d)"
        )

    def generate_api_key(self, prefix: str = "cai") -> str:
        """
        Generate a cryptographically secure API key.

        Format: <prefix>_<random_token>
        Example: cai_1a2b3c4d5e6f7g8h9i0j

        Args:
            prefix: Key prefix for identification

        Returns:
            str: Generated API key
        """
        # Generate 32 bytes of cryptographically secure random data
        random_bytes = secrets.token_bytes(32)

        # Convert to URL-safe base64 (44 characters)
        token = secrets.token_urlsafe(32)

        # Combine prefix with token
        api_key = f"{prefix}_{token}"

        logger.debug(f"Generated new API key with prefix: {prefix}")
        return api_key

    def hash_api_key(self, api_key: str) -> str:
        """
        Hash an API key for secure storage.

        Uses SHA-256 for fast lookups while maintaining security.

        Args:
            api_key: Plain API key

        Returns:
            str: Hashed API key (hex)
        """
        return hashlib.sha256(api_key.encode()).hexdigest()

    @transactional()
    async def create_api_key(
        self,
        db: AsyncSession,
        user_id: int,
        name: str,
        scopes: Optional[List[str]] = None,
        expires_in_days: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new API key for a user.

        Args:
            db: Database session
            user_id: User ID
            name: Human-readable key name
            scopes: API scopes/permissions
            expires_in_days: Custom expiration (overrides default)
            metadata: Additional metadata

        Returns:
            Dict with api_key (plaintext, shown once) and key info

        Raises:
            ValueError: If max keys reached
            IntegrityError: If key name already exists for user
        """
        # Check if user exists
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            raise ValueError(f"User {user_id} not found")

        # Check max keys limit
        result = await db.execute(
            select(APIKey).where(
                and_(
                    APIKey.user_id == user_id,
                    APIKey.revoked_at.is_(None),
                    or_(
                        APIKey.expires_at.is_(None),
                        APIKey.expires_at > datetime.utcnow(),
                    ),
                )
            )
        )
        active_keys = result.scalars().all()
        if len(active_keys) >= self.max_keys_per_user:
            raise ValueError(
                f"Maximum {self.max_keys_per_user} active API keys per user. "
                f"Revoke an existing key first."
            )

        # Generate API key
        api_key_plain = self.generate_api_key()
        api_key_hash = self.hash_api_key(api_key_plain)

        # Calculate expiration
        expiration_days = expires_in_days or self.default_expiration_days
        expires_at = datetime.utcnow() + timedelta(days=expiration_days)

        # Create API key record
        api_key_obj = APIKey(
            user_id=user_id,
            key_hash=api_key_hash,
            key_prefix=api_key_plain[:8],  # Store prefix for identification
            name=name,
            scopes=scopes or [],
            expires_at=expires_at,
            metadata=metadata or {},
        )

        db.add(api_key_obj)
        await db.flush()

        logger.info(
            f"✅ Created API key '{name}' for user {user_id} "
            f"(expires: {expires_at.date()})"
        )

        # Return key info (plaintext key shown only once!)
        return {
            "api_key": api_key_plain,  # ONLY TIME THIS IS SHOWN
            "key_id": api_key_obj.id,
            "name": name,
            "scopes": scopes or [],
            "expires_at": expires_at.isoformat(),
            "warning": (
                "⚠️  This is the only time the API key will be shown. "
                "Store it securely."
            ),
        }

    @transactional()
    async def verify_api_key(
        self,
        db: AsyncSession,
        api_key: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Verify an API key and return associated user/permissions.

        Args:
            db: Database session
            api_key: API key to verify

        Returns:
            Dict with user_id, scopes, etc. if valid, None if invalid
        """
        # Hash the provided key
        api_key_hash = self.hash_api_key(api_key)

        # Look up key
        result = await db.execute(select(APIKey).where(APIKey.key_hash == api_key_hash))
        key_obj = result.scalar_one_or_none()

        if not key_obj:
            logger.warning(f"API key verification failed: key not found")
            return None

        # Check if revoked
        if key_obj.revoked_at:
            logger.warning(
                f"API key verification failed: key revoked "
                f"(key_id={key_obj.id}, revoked={key_obj.revoked_at})"
            )
            return None

        # Check if expired
        if key_obj.expires_at and key_obj.expires_at < datetime.utcnow():
            logger.warning(
                f"API key verification failed: key expired "
                f"(key_id={key_obj.id}, expired={key_obj.expires_at})"
            )
            return None

        # Update last used timestamp
        key_obj.last_used_at = datetime.utcnow()
        await db.flush()

        # Check if expiring soon (warning)
        days_until_expiration = None
        if key_obj.expires_at:
            days_until_expiration = (key_obj.expires_at - datetime.utcnow()).days
            if days_until_expiration <= self.warning_days_before_expiration:
                logger.warning(
                    f"⚠️  API key expiring soon: {days_until_expiration} days "
                    f"(key_id={key_obj.id}, name='{key_obj.name}')"
                )

        logger.debug(
            f"✅ API key verified (key_id={key_obj.id}, user_id={key_obj.user_id})"
        )

        return {
            "key_id": key_obj.id,
            "user_id": key_obj.user_id,
            "name": key_obj.name,
            "scopes": key_obj.scopes,
            "expires_at": (
                key_obj.expires_at.isoformat() if key_obj.expires_at else None
            ),
            "days_until_expiration": days_until_expiration,
            "expiring_soon": (
                days_until_expiration is not None
                and days_until_expiration <= self.warning_days_before_expiration
            ),
        }

    @transactional()
    async def revoke_api_key(
        self,
        db: AsyncSession,
        key_id: int,
        revoked_by_user_id: Optional[int] = None,
    ) -> bool:
        """
        Revoke an API key.

        Args:
            db: Database session
            key_id: API key ID to revoke
            revoked_by_user_id: User who performed revocation

        Returns:
            bool: True if revoked, False if not found
        """
        result = await db.execute(select(APIKey).where(APIKey.id == key_id))
        key_obj = result.scalar_one_or_none()

        if not key_obj:
            logger.warning(f"Cannot revoke: API key {key_id} not found")
            return False

        if key_obj.revoked_at:
            logger.warning(f"API key {key_id} already revoked")
            return True

        key_obj.revoked_at = datetime.utcnow()
        await db.flush()

        logger.info(
            f"✅ Revoked API key {key_id} (name='{key_obj.name}', "
            f"user_id={key_obj.user_id}, revoked_by={revoked_by_user_id})"
        )

        return True

    @transactional()
    async def rotate_api_key(
        self,
        db: AsyncSession,
        key_id: int,
        grace_period_days: int = 7,
    ) -> Dict[str, Any]:
        """
        Rotate an API key (create new, mark old for deprecation).

        The old key remains valid for a grace period to allow clients to update.

        Args:
            db: Database session
            key_id: Existing key ID to rotate
            grace_period_days: Days to keep old key valid

        Returns:
            Dict with new API key and rotation info
        """
        # Get existing key
        result = await db.execute(select(APIKey).where(APIKey.id == key_id))
        old_key = result.scalar_one_or_none()

        if not old_key:
            raise ValueError(f"API key {key_id} not found")

        if old_key.revoked_at:
            raise ValueError(f"Cannot rotate revoked key {key_id}")

        # Create new key with same properties
        new_key_info = await self.create_api_key(
            db=db,
            user_id=old_key.user_id,
            name=f"{old_key.name} (rotated)",
            scopes=old_key.scopes,
            expires_in_days=self.default_expiration_days,
            metadata={
                **old_key.metadata,
                "rotated_from": key_id,
                "rotation_date": datetime.utcnow().isoformat(),
            },
        )

        # Mark old key for future revocation (after grace period)
        old_key.expires_at = datetime.utcnow() + timedelta(days=grace_period_days)
        await db.flush()

        logger.info(
            f"✅ Rotated API key {key_id} → {new_key_info['key_id']} "
            f"(grace period: {grace_period_days} days)"
        )

        return {
            **new_key_info,
            "old_key_id": key_id,
            "old_key_expires_at": old_key.expires_at.isoformat(),
            "grace_period_days": grace_period_days,
            "migration_warning": (
                f"⚠️  Update your application to use the new API key. "
                f"Old key will expire on {old_key.expires_at.date()}"
            ),
        }

    async def get_expiring_keys(
        self,
        db: AsyncSession,
        days_threshold: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get API keys expiring soon.

        Args:
            db: Database session
            days_threshold: Days threshold (default: warning period)

        Returns:
            List of expiring keys with details
        """
        threshold = days_threshold or self.warning_days_before_expiration
        threshold_date = datetime.utcnow() + timedelta(days=threshold)

        result = await db.execute(
            select(APIKey).where(
                and_(
                    APIKey.expires_at.isnot(None),
                    APIKey.expires_at <= threshold_date,
                    APIKey.expires_at > datetime.utcnow(),
                    APIKey.revoked_at.is_(None),
                )
            )
        )
        expiring_keys = result.scalars().all()

        return [
            {
                "key_id": key.id,
                "user_id": key.user_id,
                "name": key.name,
                "key_prefix": key.key_prefix,
                "expires_at": key.expires_at.isoformat(),
                "days_until_expiration": (key.expires_at - datetime.utcnow()).days,
                "last_used_at": (
                    key.last_used_at.isoformat() if key.last_used_at else None
                ),
            }
            for key in expiring_keys
        ]

    async def get_user_keys(
        self,
        db: AsyncSession,
        user_id: int,
        include_revoked: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Get all API keys for a user.

        Args:
            db: Database session
            user_id: User ID
            include_revoked: Include revoked keys

        Returns:
            List of API keys
        """
        query = select(APIKey).where(APIKey.user_id == user_id)

        if not include_revoked:
            query = query.where(APIKey.revoked_at.is_(None))

        result = await db.execute(query)
        keys = result.scalars().all()

        return [
            {
                "key_id": key.id,
                "name": key.name,
                "key_prefix": key.key_prefix,
                "scopes": key.scopes,
                "created_at": key.created_at.isoformat(),
                "expires_at": key.expires_at.isoformat() if key.expires_at else None,
                "last_used_at": (
                    key.last_used_at.isoformat() if key.last_used_at else None
                ),
                "revoked_at": key.revoked_at.isoformat() if key.revoked_at else None,
                "is_active": (
                    key.revoked_at is None
                    and (key.expires_at is None or key.expires_at > datetime.utcnow())
                ),
                "is_expiring_soon": (
                    key.expires_at is not None
                    and (key.expires_at - datetime.utcnow()).days
                    <= self.warning_days_before_expiration
                    and key.expires_at > datetime.utcnow()
                ),
            }
            for key in keys
        ]


# Global API key manager
_api_key_manager: Optional[APIKeyManager] = None


def get_api_key_manager() -> APIKeyManager:
    """
    Get global API key manager.

    Returns:
        APIKeyManager: Global instance
    """
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager
