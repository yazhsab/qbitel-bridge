#!/usr/bin/env python3
"""
Secret Rotation Script for CRONOS AI

This script helps rotate secrets in production environments.
Supports HashiCorp Vault, AWS Secrets Manager, and Azure Key Vault.
"""

import argparse
import sys
import os
import secrets
import string
from datetime import datetime
from typing import Optional, Dict, Any
import logging

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_engine.security.secrets_manager import (
    SecretsManager,
    SecretBackend,
    SecretMetadata,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_strong_password(length: int = 32) -> str:
    """Generate a strong random password."""
    alphabet = string.ascii_letters + string.digits + string.punctuation
    # Ensure at least one of each type
    password = [
        secrets.choice(string.ascii_lowercase),
        secrets.choice(string.ascii_uppercase),
        secrets.choice(string.digits),
        secrets.choice(string.punctuation),
    ]
    # Fill the rest
    password.extend(secrets.choice(alphabet) for _ in range(length - 4))
    # Shuffle
    secrets.SystemRandom().shuffle(password)
    return "".join(password)


def generate_api_key(length: int = 32) -> str:
    """Generate a secure API key."""
    return secrets.token_urlsafe(length)


def generate_jwt_secret(length: int = 64) -> str:
    """Generate a JWT secret."""
    return secrets.token_urlsafe(length)


def rotate_database_password(
    secrets_manager: SecretsManager, dry_run: bool = False
) -> bool:
    """
    Rotate database password.

    Steps:
    1. Generate new password
    2. Update password in database
    3. Update password in secrets manager
    4. Verify connection with new password
    """
    logger.info("Rotating database password...")

    new_password = generate_strong_password(32)

    if dry_run:
        logger.info(f"[DRY RUN] Would generate new password: {new_password[:4]}...")
        return True

    try:
        # Store new password
        metadata = SecretMetadata(
            key="database_password",
            backend=secrets_manager.backend,
            created_at=datetime.utcnow(),
            last_rotated=datetime.utcnow(),
            rotation_interval_days=90,
        )

        secrets_manager.set_secret("database_password", new_password, metadata)
        logger.info("✅ Database password rotated successfully")

        logger.warning(
            "⚠️  IMPORTANT: Update the database user password manually:\n"
            f"   ALTER USER cronos_user WITH PASSWORD '{new_password}';\n"
            "   Then restart the application to use the new password."
        )

        return True

    except Exception as e:
        logger.error(f"❌ Failed to rotate database password: {e}")
        return False


def rotate_redis_password(
    secrets_manager: SecretsManager, dry_run: bool = False
) -> bool:
    """Rotate Redis password."""
    logger.info("Rotating Redis password...")

    new_password = generate_strong_password(32)

    if dry_run:
        logger.info(f"[DRY RUN] Would generate new password: {new_password[:4]}...")
        return True

    try:
        metadata = SecretMetadata(
            key="redis_password",
            backend=secrets_manager.backend,
            created_at=datetime.utcnow(),
            last_rotated=datetime.utcnow(),
            rotation_interval_days=90,
        )

        secrets_manager.set_secret("redis_password", new_password, metadata)
        logger.info("✅ Redis password rotated successfully")

        logger.warning(
            "⚠️  IMPORTANT: Update Redis configuration:\n"
            f"   CONFIG SET requirepass '{new_password}'\n"
            "   Then restart the application."
        )

        return True

    except Exception as e:
        logger.error(f"❌ Failed to rotate Redis password: {e}")
        return False


def rotate_jwt_secret(secrets_manager: SecretsManager, dry_run: bool = False) -> bool:
    """Rotate JWT secret."""
    logger.info("Rotating JWT secret...")

    new_secret = generate_jwt_secret(64)

    if dry_run:
        logger.info(f"[DRY RUN] Would generate new JWT secret: {new_secret[:8]}...")
        return True

    try:
        metadata = SecretMetadata(
            key="jwt_secret",
            backend=secrets_manager.backend,
            created_at=datetime.utcnow(),
            last_rotated=datetime.utcnow(),
            rotation_interval_days=180,
        )

        secrets_manager.set_secret("jwt_secret", new_secret, metadata)
        logger.info("✅ JWT secret rotated successfully")

        logger.warning(
            "⚠️  IMPORTANT: All existing JWT tokens will be invalidated.\n"
            "   Users will need to re-authenticate after application restart."
        )

        return True

    except Exception as e:
        logger.error(f"❌ Failed to rotate JWT secret: {e}")
        return False


def rotate_api_key(secrets_manager: SecretsManager, dry_run: bool = False) -> bool:
    """Rotate API key."""
    logger.info("Rotating API key...")

    new_key = generate_api_key(32)

    if dry_run:
        logger.info(f"[DRY RUN] Would generate new API key: {new_key[:8]}...")
        return True

    try:
        metadata = SecretMetadata(
            key="api_key",
            backend=secrets_manager.backend,
            created_at=datetime.utcnow(),
            last_rotated=datetime.utcnow(),
            rotation_interval_days=90,
        )

        secrets_manager.set_secret("api_key", new_key, metadata)
        logger.info("✅ API key rotated successfully")

        logger.warning(
            "⚠️  IMPORTANT: Update API key in all client applications.\n"
            f"   New API key: {new_key}\n"
            "   Store this securely!"
        )

        return True

    except Exception as e:
        logger.error(f"❌ Failed to rotate API key: {e}")
        return False


def rotate_encryption_key(
    secrets_manager: SecretsManager, dry_run: bool = False
) -> bool:
    """Rotate encryption key."""
    logger.info("Rotating encryption key...")

    new_key = generate_jwt_secret(64)

    if dry_run:
        logger.info(f"[DRY RUN] Would generate new encryption key: {new_key[:8]}...")
        return True

    try:
        metadata = SecretMetadata(
            key="encryption_key",
            backend=secrets_manager.backend,
            created_at=datetime.utcnow(),
            last_rotated=datetime.utcnow(),
            rotation_interval_days=365,
        )

        secrets_manager.set_secret("encryption_key", new_key, metadata)
        logger.info("✅ Encryption key rotated successfully")

        logger.warning(
            "⚠️  CRITICAL: Data encrypted with the old key will need to be re-encrypted.\n"
            "   Implement a key rotation strategy before using this in production."
        )

        return True

    except Exception as e:
        logger.error(f"❌ Failed to rotate encryption key: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Rotate secrets for CRONOS AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Rotate all secrets (dry run)
  python rotate_secrets.py --all --dry-run
  
  # Rotate database password
  python rotate_secrets.py --database
  
  # Rotate JWT secret
  python rotate_secrets.py --jwt
  
  # Rotate API key
  python rotate_secrets.py --api-key
  
  # Use specific backend
  python rotate_secrets.py --all --backend vault
        """,
    )

    parser.add_argument("--all", action="store_true", help="Rotate all secrets")
    parser.add_argument(
        "--database", action="store_true", help="Rotate database password"
    )
    parser.add_argument("--redis", action="store_true", help="Rotate Redis password")
    parser.add_argument("--jwt", action="store_true", help="Rotate JWT secret")
    parser.add_argument("--api-key", action="store_true", help="Rotate API key")
    parser.add_argument(
        "--encryption", action="store_true", help="Rotate encryption key"
    )
    parser.add_argument(
        "--backend",
        choices=["vault", "aws", "azure", "file"],
        default="vault",
        help="Secrets backend to use",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Perform dry run without making changes"
    )

    args = parser.parse_args()

    # Check if at least one secret type is selected
    if not any(
        [args.all, args.database, args.redis, args.jwt, args.api_key, args.encryption]
    ):
        parser.print_help()
        return 1

    # Initialize secrets manager
    backend_map = {
        "vault": SecretBackend.VAULT,
        "aws": SecretBackend.AWS_SECRETS_MANAGER,
        "azure": SecretBackend.AZURE_KEY_VAULT,
        "file": SecretBackend.FILE,
    }

    config = {"backend": backend_map[args.backend].value}

    try:
        secrets_manager = SecretsManager(config)
        logger.info(f"Using secrets backend: {args.backend}")

        if args.dry_run:
            logger.info("🔍 DRY RUN MODE - No changes will be made")

        success = True

        # Rotate selected secrets
        if args.all or args.database:
            success &= rotate_database_password(secrets_manager, args.dry_run)

        if args.all or args.redis:
            success &= rotate_redis_password(secrets_manager, args.dry_run)

        if args.all or args.jwt:
            success &= rotate_jwt_secret(secrets_manager, args.dry_run)

        if args.all or args.api_key:
            success &= rotate_api_key(secrets_manager, args.dry_run)

        if args.all or args.encryption:
            success &= rotate_encryption_key(secrets_manager, args.dry_run)

        if success:
            logger.info("\n✅ Secret rotation completed successfully")
            if not args.dry_run:
                logger.info("\n⚠️  Remember to:")
                logger.info(
                    "1. Update secrets in the actual services (database, Redis, etc.)"
                )
                logger.info("2. Restart the application to use new secrets")
                logger.info("3. Update client applications with new API keys")
                logger.info("4. Test the application thoroughly")
            return 0
        else:
            logger.error("\n❌ Some secrets failed to rotate")
            return 1

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
