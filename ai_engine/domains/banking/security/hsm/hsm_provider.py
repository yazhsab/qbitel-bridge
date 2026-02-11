"""
HSM Provider Abstract Interface

Defines the abstract interface for HSM operations that all
vendor implementations must follow.
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple
from datetime import datetime

from ai_engine.domains.banking.security.hsm.hsm_types import (
    HSMConfig,
    HSMKeyType,
    HSMAlgorithm,
    HSMKeyHandle,
    HSMError,
)


@dataclass
class EncryptionResult:
    """Result of an encryption operation."""

    ciphertext: bytes
    iv: Optional[bytes] = None
    tag: Optional[bytes] = None
    algorithm: Optional[str] = None


@dataclass
class DecryptionResult:
    """Result of a decryption operation."""

    plaintext: bytes
    verified: bool = True


@dataclass
class SignatureResult:
    """Result of a signing operation."""

    signature: bytes
    algorithm: str
    key_id: str


@dataclass
class VerificationResult:
    """Result of a signature verification."""

    valid: bool
    key_id: str


@dataclass
class KEMEncapsulationResult:
    """Result of KEM encapsulation (for PQC)."""

    ciphertext: bytes
    shared_secret: bytes


class HSMSession:
    """
    Session with an HSM.

    Manages the connection state and provides context for operations.
    """

    def __init__(self, provider: "HSMProvider", session_handle: Any):
        self._provider = provider
        self._session_handle = session_handle
        self._is_open = True
        self._transaction_active = False

    @property
    def is_open(self) -> bool:
        return self._is_open

    def close(self) -> None:
        """Close the session."""
        if self._is_open:
            self._provider._close_session(self._session_handle)
            self._is_open = False

    def begin_transaction(self) -> None:
        """Begin a transaction for atomic operations."""
        if self._transaction_active:
            raise HSMError("Transaction already active")
        self._transaction_active = True

    def commit_transaction(self) -> None:
        """Commit the current transaction."""
        if not self._transaction_active:
            raise HSMError("No active transaction")
        self._transaction_active = False

    def rollback_transaction(self) -> None:
        """Rollback the current transaction."""
        if not self._transaction_active:
            raise HSMError("No active transaction")
        self._transaction_active = False

    def __enter__(self) -> "HSMSession":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class HSMProvider(ABC):
    """
    Abstract base class for HSM providers.

    All vendor-specific implementations must inherit from this class
    and implement the abstract methods.
    """

    def __init__(self, config: HSMConfig):
        self._config = config
        self._connected = False
        self._capabilities: Dict[str, bool] = {}

    @property
    def config(self) -> HSMConfig:
        return self._config

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass

    @property
    @abstractmethod
    def supports_pqc(self) -> bool:
        """Check if provider supports post-quantum cryptography."""
        pass

    # Connection management

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to HSM."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from HSM."""
        pass

    @abstractmethod
    def open_session(self, read_write: bool = True) -> HSMSession:
        """Open a session with the HSM."""
        pass

    @abstractmethod
    def _close_session(self, session_handle: Any) -> None:
        """Close a session (internal use)."""
        pass

    @contextmanager
    def session(self, read_write: bool = True) -> Generator[HSMSession, None, None]:
        """Context manager for HSM session."""
        session = self.open_session(read_write)
        try:
            yield session
        finally:
            session.close()

    # Key management

    @abstractmethod
    def generate_key(
        self,
        key_type: HSMKeyType,
        label: str,
        extractable: bool = False,
        **kwargs,
    ) -> HSMKeyHandle:
        """Generate a new key in the HSM."""
        pass

    @abstractmethod
    def generate_key_pair(
        self,
        key_type: HSMKeyType,
        label: str,
        extractable: bool = False,
        **kwargs,
    ) -> Tuple[HSMKeyHandle, HSMKeyHandle]:
        """Generate a key pair (public, private)."""
        pass

    @abstractmethod
    def import_key(
        self,
        key_type: HSMKeyType,
        key_data: bytes,
        label: str,
        **kwargs,
    ) -> HSMKeyHandle:
        """Import a key into the HSM."""
        pass

    @abstractmethod
    def export_public_key(self, key_handle: HSMKeyHandle) -> bytes:
        """Export a public key from the HSM."""
        pass

    @abstractmethod
    def delete_key(self, key_handle: HSMKeyHandle) -> None:
        """Delete a key from the HSM."""
        pass

    @abstractmethod
    def get_key(self, key_id: str) -> Optional[HSMKeyHandle]:
        """Get a key handle by ID."""
        pass

    @abstractmethod
    def list_keys(
        self,
        key_type: Optional[HSMKeyType] = None,
        label_pattern: Optional[str] = None,
    ) -> List[HSMKeyHandle]:
        """List keys in the HSM."""
        pass

    @abstractmethod
    def get_key_info(self, key_handle: HSMKeyHandle) -> Dict[str, Any]:
        """Get detailed information about a key."""
        pass

    # Cryptographic operations

    @abstractmethod
    def encrypt(
        self,
        key_handle: HSMKeyHandle,
        plaintext: bytes,
        algorithm: HSMAlgorithm,
        iv: Optional[bytes] = None,
        aad: Optional[bytes] = None,
    ) -> EncryptionResult:
        """Encrypt data using a key."""
        pass

    @abstractmethod
    def decrypt(
        self,
        key_handle: HSMKeyHandle,
        ciphertext: bytes,
        algorithm: HSMAlgorithm,
        iv: Optional[bytes] = None,
        tag: Optional[bytes] = None,
        aad: Optional[bytes] = None,
    ) -> DecryptionResult:
        """Decrypt data using a key."""
        pass

    @abstractmethod
    def sign(
        self,
        key_handle: HSMKeyHandle,
        data: bytes,
        algorithm: HSMAlgorithm,
    ) -> SignatureResult:
        """Sign data using a private key."""
        pass

    @abstractmethod
    def verify(
        self,
        key_handle: HSMKeyHandle,
        data: bytes,
        signature: bytes,
        algorithm: HSMAlgorithm,
    ) -> VerificationResult:
        """Verify a signature using a public key."""
        pass

    @abstractmethod
    def wrap_key(
        self,
        wrapping_key: HSMKeyHandle,
        key_to_wrap: HSMKeyHandle,
        algorithm: HSMAlgorithm,
    ) -> bytes:
        """Wrap (encrypt) a key for export."""
        pass

    @abstractmethod
    def unwrap_key(
        self,
        wrapping_key: HSMKeyHandle,
        wrapped_key: bytes,
        key_type: HSMKeyType,
        algorithm: HSMAlgorithm,
        label: str,
    ) -> HSMKeyHandle:
        """Unwrap (decrypt) a key and import it."""
        pass

    @abstractmethod
    def derive_key(
        self,
        base_key: HSMKeyHandle,
        derivation_data: bytes,
        key_type: HSMKeyType,
        label: str,
    ) -> HSMKeyHandle:
        """Derive a new key from an existing key."""
        pass

    # Hash and MAC operations

    @abstractmethod
    def hash(self, data: bytes, algorithm: HSMAlgorithm) -> bytes:
        """Compute hash of data."""
        pass

    @abstractmethod
    def mac(
        self,
        key_handle: HSMKeyHandle,
        data: bytes,
        algorithm: HSMAlgorithm,
    ) -> bytes:
        """Compute MAC of data."""
        pass

    @abstractmethod
    def verify_mac(
        self,
        key_handle: HSMKeyHandle,
        data: bytes,
        mac_value: bytes,
        algorithm: HSMAlgorithm,
    ) -> bool:
        """Verify MAC of data."""
        pass

    # PQC operations

    def kem_encapsulate(
        self,
        public_key: HSMKeyHandle,
    ) -> KEMEncapsulationResult:
        """
        KEM encapsulation for post-quantum key exchange.

        Args:
            public_key: Public key for encapsulation

        Returns:
            KEMEncapsulationResult with ciphertext and shared secret
        """
        raise NotImplementedError("PQC not supported by this provider")

    def kem_decapsulate(
        self,
        private_key: HSMKeyHandle,
        ciphertext: bytes,
    ) -> bytes:
        """
        KEM decapsulation for post-quantum key exchange.

        Args:
            private_key: Private key for decapsulation
            ciphertext: Encapsulated ciphertext

        Returns:
            Shared secret
        """
        raise NotImplementedError("PQC not supported by this provider")

    # Random number generation

    @abstractmethod
    def generate_random(self, length: int) -> bytes:
        """Generate cryptographically secure random bytes."""
        pass

    # Utility methods

    def get_capabilities(self) -> Dict[str, bool]:
        """Get HSM capabilities."""
        return self._capabilities.copy()

    def check_health(self) -> Dict[str, Any]:
        """Check HSM health status."""
        return {
            "connected": self._connected,
            "provider": self.provider_name,
            "supports_pqc": self.supports_pqc,
            "timestamp": datetime.utcnow().isoformat(),
        }
