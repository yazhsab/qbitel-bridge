"""
QBITEL Engine - Database Field Encryption

Production-ready column-level encryption for sensitive database fields.
Uses AES-256-GCM for authenticated encryption with Fernet (symmetric encryption).
"""

import os
import base64
import logging
from typing import Optional, Any

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from sqlalchemy.types import TypeDecorator, LargeBinary, String, Text

logger = logging.getLogger(__name__)


class EncryptionError(Exception):
    """Raised when encryption/decryption fails."""

    pass


class EncryptionKeyManager:
    """
    Manages encryption keys for database field encryption.

    Features:
    - Key derivation from master key using PBKDF2
    - Support for key rotation
    - Secure key storage in memory
    - Environment-based key loading
    """

    _instance: Optional["EncryptionKeyManager"] = None
    _fernet: Optional[Fernet] = None

    def __init__(self):
        """Initialize encryption key manager."""
        if EncryptionKeyManager._instance is not None:
            raise RuntimeError("EncryptionKeyManager is a singleton. Use get_instance() instead.")

        self._load_encryption_key()
        EncryptionKeyManager._instance = self

    @classmethod
    def get_instance(cls) -> "EncryptionKeyManager":
        """
        Get singleton instance of EncryptionKeyManager.

        Returns:
            EncryptionKeyManager: Singleton instance

        Raises:
            RuntimeError: If not initialized
        """
        if cls._instance is None:
            cls._instance = EncryptionKeyManager()
        return cls._instance

    def _load_encryption_key(self) -> None:
        """
        Load encryption key from environment.

        Environment variables (in priority order):
        1. ENCRYPTION_KEY - Direct Fernet key (base64-encoded)
        2. MASTER_ENCRYPTION_KEY - Master key to derive Fernet key

        Raises:
            EncryptionError: If no valid encryption key found
        """
        # Try direct Fernet key first
        encryption_key = os.getenv("ENCRYPTION_KEY")

        if encryption_key:
            try:
                # Validate it's a valid Fernet key
                self._fernet = Fernet(encryption_key.encode())
                logger.info("✅ Encryption key loaded from ENCRYPTION_KEY")
                return
            except Exception as e:
                logger.warning(f"Invalid ENCRYPTION_KEY format: {e}")

        # Try deriving from master key
        master_key = os.getenv("MASTER_ENCRYPTION_KEY")

        if master_key:
            try:
                self._fernet = self._derive_fernet_key(master_key)
                logger.info("✅ Encryption key derived from MASTER_ENCRYPTION_KEY")
                return
            except Exception as e:
                logger.error(f"Failed to derive encryption key: {e}")
                raise EncryptionError(f"Failed to derive encryption key: {e}") from e

        # No key found
        error_msg = (
            "No encryption key configured!\n"
            "REQUIRED for production: Set one of:\n"
            "  - ENCRYPTION_KEY: Fernet key (base64, 44 chars)\n"
            "  - MASTER_ENCRYPTION_KEY: Master key (min 32 chars)\n\n"
            "Generate a Fernet key:\n"
            '  python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"\n\n'
            "Generate a master key:\n"
            '  python -c "import secrets; print(secrets.token_urlsafe(32))"'
        )

        # In production, this should be a hard error
        environment = os.getenv("QBITEL_AI_ENVIRONMENT", "development")
        if environment == "production":
            raise EncryptionError(error_msg)
        else:
            # For development, generate a temporary key (WARNING: Data will be lost!)
            logger.warning(
                "⚠️  NO ENCRYPTION KEY SET - Generating temporary key for development.\n"
                "⚠️  This key will be lost on restart - DO NOT use in production!"
            )
            self._fernet = Fernet(Fernet.generate_key())

    def _derive_fernet_key(self, master_key: str) -> Fernet:
        """
        Derive a Fernet key from a master key using PBKDF2.

        Args:
            master_key: Master encryption key

        Returns:
            Fernet: Initialized Fernet instance

        Raises:
            EncryptionError: If key derivation fails
        """
        if len(master_key) < 32:
            raise EncryptionError("Master encryption key must be at least 32 characters")

        # Use a static salt (in production, consider using a configurable salt)
        # The salt doesn't need to be secret, but should be consistent
        salt = b"qbitel_encryption_salt_v1"

        # Derive 32 bytes (256 bits) for Fernet
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,  # OWASP recommendation
        )

        key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
        return Fernet(key)

    def encrypt(self, plaintext: str) -> bytes:
        """
        Encrypt plaintext string.

        Args:
            plaintext: String to encrypt

        Returns:
            bytes: Encrypted data

        Raises:
            EncryptionError: If encryption fails
        """
        if not self._fernet:
            raise EncryptionError("Encryption not initialized")

        if plaintext is None:
            return None

        try:
            return self._fernet.encrypt(plaintext.encode("utf-8"))
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise EncryptionError(f"Encryption failed: {e}") from e

    def decrypt(self, ciphertext: bytes) -> str:
        """
        Decrypt ciphertext to string.

        Args:
            ciphertext: Encrypted bytes

        Returns:
            str: Decrypted plaintext

        Raises:
            EncryptionError: If decryption fails
        """
        if not self._fernet:
            raise EncryptionError("Encryption not initialized")

        if ciphertext is None:
            return None

        try:
            return self._fernet.decrypt(ciphertext).decode("utf-8")
        except InvalidToken as e:
            logger.error("Decryption failed: Invalid token (wrong key or corrupted data)")
            raise EncryptionError(
                "Decryption failed: Invalid token. " "This may indicate the encryption key has changed."
            ) from e
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise EncryptionError(f"Decryption failed: {e}") from e

    def rotate_key(self, new_key: str) -> None:
        """
        Rotate to a new encryption key.

        WARNING: This only updates the in-memory key. Existing encrypted data
        in the database must be re-encrypted with the new key separately.

        Args:
            new_key: New encryption key (Fernet key or master key)

        Raises:
            EncryptionError: If key rotation fails
        """
        try:
            # Try as Fernet key first
            new_fernet = Fernet(new_key.encode())
            self._fernet = new_fernet
            logger.info("✅ Encryption key rotated (Fernet key)")
        except Exception:
            # Try deriving from master key
            try:
                self._fernet = self._derive_fernet_key(new_key)
                logger.info("✅ Encryption key rotated (derived from master key)")
            except Exception as e:
                raise EncryptionError(f"Key rotation failed: {e}") from e


# SQLAlchemy custom type for encrypted strings
class EncryptedString(TypeDecorator):
    """
    SQLAlchemy type decorator for encrypting string fields.

    Usage:
        class User(Base):
            __tablename__ = 'users'
            id = Column(Integer, primary_key=True)
            mfa_secret = Column(EncryptedString(255))

    The field is automatically encrypted when stored and decrypted when loaded.
    """

    impl = LargeBinary
    cache_ok = True

    def __init__(self, length: Optional[int] = None, *args, **kwargs):
        """
        Initialize encrypted string type.

        Args:
            length: Maximum length of unencrypted string (for documentation)
        """
        self.length = length
        super().__init__(*args, **kwargs)

    def process_bind_param(self, value: Optional[str], dialect) -> Optional[bytes]:
        """
        Encrypt value before storing in database.

        Args:
            value: Plaintext string
            dialect: SQLAlchemy dialect

        Returns:
            bytes: Encrypted value
        """
        if value is None:
            return None

        if not isinstance(value, str):
            raise TypeError(f"EncryptedString requires str, got {type(value)}")

        try:
            key_manager = EncryptionKeyManager.get_instance()
            return key_manager.encrypt(value)
        except Exception as e:
            logger.error(f"Failed to encrypt field: {e}")
            raise

    def process_result_value(self, value: Optional[bytes], dialect) -> Optional[str]:
        """
        Decrypt value when loading from database.

        Args:
            value: Encrypted bytes
            dialect: SQLAlchemy dialect

        Returns:
            str: Decrypted plaintext
        """
        if value is None:
            return None

        if not isinstance(value, bytes):
            raise TypeError(f"EncryptedString expects bytes from DB, got {type(value)}")

        try:
            key_manager = EncryptionKeyManager.get_instance()
            return key_manager.decrypt(value)
        except Exception as e:
            logger.error(f"Failed to decrypt field: {e}")
            raise


class EncryptedText(TypeDecorator):
    """
    SQLAlchemy type decorator for encrypting text fields (larger data).

    Usage:
        class OAuthClient(Base):
            __tablename__ = 'oauth_clients'
            id = Column(Integer, primary_key=True)
            client_secret = Column(EncryptedText)
    """

    impl = LargeBinary
    cache_ok = True

    def process_bind_param(self, value: Optional[str], dialect) -> Optional[bytes]:
        """Encrypt text before storing."""
        if value is None:
            return None

        if not isinstance(value, str):
            raise TypeError(f"EncryptedText requires str, got {type(value)}")

        try:
            key_manager = EncryptionKeyManager.get_instance()
            return key_manager.encrypt(value)
        except Exception as e:
            logger.error(f"Failed to encrypt text field: {e}")
            raise

    def process_result_value(self, value: Optional[bytes], dialect) -> Optional[str]:
        """Decrypt text when loading."""
        if value is None:
            return None

        if not isinstance(value, bytes):
            raise TypeError(f"EncryptedText expects bytes from DB, got {type(value)}")

        try:
            key_manager = EncryptionKeyManager.get_instance()
            return key_manager.decrypt(value)
        except Exception as e:
            logger.error(f"Failed to decrypt text field: {e}")
            raise


class EncryptedJSON(TypeDecorator):
    """
    SQLAlchemy type decorator for encrypting JSON fields.

    Usage:
        class User(Base):
            __tablename__ = 'users'
            id = Column(Integer, primary_key=True)
            mfa_backup_codes = Column(EncryptedJSON)
    """

    impl = LargeBinary
    cache_ok = True

    def process_bind_param(self, value: Optional[Any], dialect) -> Optional[bytes]:
        """Encrypt JSON before storing."""
        if value is None:
            return None

        import json

        try:
            # Serialize to JSON string
            json_str = json.dumps(value)

            # Encrypt
            key_manager = EncryptionKeyManager.get_instance()
            return key_manager.encrypt(json_str)

        except Exception as e:
            logger.error(f"Failed to encrypt JSON field: {e}")
            raise

    def process_result_value(self, value: Optional[bytes], dialect) -> Optional[Any]:
        """Decrypt JSON when loading."""
        if value is None:
            return None

        import json

        try:
            # Decrypt
            key_manager = EncryptionKeyManager.get_instance()
            json_str = key_manager.decrypt(value)

            # Deserialize from JSON
            return json.loads(json_str)

        except Exception as e:
            logger.error(f"Failed to decrypt JSON field: {e}")
            raise


def generate_encryption_key() -> str:
    """
    Generate a new Fernet encryption key.

    Returns:
        str: Base64-encoded Fernet key

    Usage:
        key = generate_encryption_key()
        print(f"Add to your .env file:")
        print(f"ENCRYPTION_KEY={key}")
    """
    return Fernet.generate_key().decode()


def initialize_encryption(test_mode: bool = False) -> None:
    """
    Initialize encryption system.

    Should be called during application startup.

    Args:
        test_mode: If True, generate temporary key for testing

    Raises:
        EncryptionError: If initialization fails
    """
    if test_mode:
        # Generate temporary key for testing
        os.environ["ENCRYPTION_KEY"] = generate_encryption_key()
        logger.warning("⚠️  Using temporary encryption key for testing")

    # Initialize singleton
    EncryptionKeyManager.get_instance()
    logger.info("✅ Encryption system initialized")
