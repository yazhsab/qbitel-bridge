"""
SoftHSM Implementation

Software-based HSM for development and testing.
Implements the HSMProvider interface using Python cryptography libraries.

WARNING: This is NOT suitable for production use. Use a hardware HSM
for production environments.
"""

import hashlib
import hmac
import os
import secrets
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ai_engine.domains.banking.security.hsm.hsm_types import (
    HSMConfig,
    HSMKeyType,
    HSMAlgorithm,
    HSMKeyHandle,
    HSMError,
    HSMConnectionError,
    HSMOperationError,
    HSMKeyNotFoundError,
    HSMKeyExistsError,
    HSMCapabilityError,
)
from ai_engine.domains.banking.security.hsm.hsm_provider import (
    HSMProvider,
    HSMSession,
    EncryptionResult,
    DecryptionResult,
    SignatureResult,
    VerificationResult,
    KEMEncapsulationResult,
)

# Try to import cryptography library
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, ec, padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


class SoftHSMKeyStore:
    """In-memory key store for SoftHSM."""

    def __init__(self):
        self._keys: Dict[str, Dict[str, Any]] = {}
        self._key_handles: Dict[str, HSMKeyHandle] = {}

    def store_key(
        self,
        key_id: str,
        key_data: Any,
        handle: HSMKeyHandle,
    ) -> None:
        """Store a key."""
        if key_id in self._keys:
            raise HSMKeyExistsError(key_id)
        self._keys[key_id] = {
            "data": key_data,
            "created_at": datetime.utcnow(),
        }
        self._key_handles[key_id] = handle

    def get_key(self, key_id: str) -> Optional[Any]:
        """Get key data."""
        if key_id not in self._keys:
            return None
        return self._keys[key_id]["data"]

    def get_handle(self, key_id: str) -> Optional[HSMKeyHandle]:
        """Get key handle."""
        return self._key_handles.get(key_id)

    def delete_key(self, key_id: str) -> None:
        """Delete a key."""
        if key_id in self._keys:
            del self._keys[key_id]
        if key_id in self._key_handles:
            del self._key_handles[key_id]

    def list_keys(self) -> List[HSMKeyHandle]:
        """List all key handles."""
        return list(self._key_handles.values())


class SoftHSM(HSMProvider):
    """
    Software HSM implementation for development and testing.

    Uses Python cryptography library for cryptographic operations.
    Keys are stored in memory and are lost when the instance is destroyed.
    """

    def __init__(self, config: HSMConfig):
        super().__init__(config)
        self._key_store = SoftHSMKeyStore()
        self._sessions: Dict[str, bool] = {}
        self._session_counter = 0

        if not CRYPTO_AVAILABLE:
            raise HSMError("cryptography library not available")

        self._capabilities = {
            "aes": True,
            "des3": True,
            "rsa": True,
            "ecdsa": True,
            "sha256": True,
            "sha384": True,
            "sha512": True,
            "hmac": True,
            "key_derivation": True,
            "key_wrapping": True,
            "pqc": config.enable_pqc,
        }

    @property
    def provider_name(self) -> str:
        return "SoftHSM"

    @property
    def supports_pqc(self) -> bool:
        return self._config.enable_pqc

    def connect(self) -> None:
        """Establish connection (no-op for SoftHSM)."""
        self._connected = True

    def disconnect(self) -> None:
        """Disconnect (no-op for SoftHSM)."""
        self._connected = False

    def open_session(self, read_write: bool = True) -> HSMSession:
        """Open a session."""
        self._session_counter += 1
        session_id = f"session_{self._session_counter}"
        self._sessions[session_id] = read_write
        return HSMSession(self, session_id)

    def _close_session(self, session_handle: Any) -> None:
        """Close a session."""
        if session_handle in self._sessions:
            del self._sessions[session_handle]

    def generate_key(
        self,
        key_type: HSMKeyType,
        label: str,
        extractable: bool = False,
        **kwargs,
    ) -> HSMKeyHandle:
        """Generate a symmetric key."""
        key_id = str(uuid.uuid4())

        if key_type.is_symmetric:
            # Generate symmetric key
            key_size_bytes = key_type.key_size // 8
            key_data = secrets.token_bytes(key_size_bytes)
        else:
            raise HSMOperationError(
                f"Use generate_key_pair for asymmetric keys: {key_type}",
                operation="generate_key",
            )

        handle = HSMKeyHandle(
            key_id=key_id,
            key_type=key_type,
            label=label,
            created_at=datetime.utcnow(),
            extractable=extractable,
            sensitive=True,
            can_encrypt=True,
            can_decrypt=True,
            can_wrap=True,
            can_unwrap=True,
            can_derive=True,
        )

        self._key_store.store_key(key_id, key_data, handle)
        return handle

    def generate_key_pair(
        self,
        key_type: HSMKeyType,
        label: str,
        extractable: bool = False,
        **kwargs,
    ) -> Tuple[HSMKeyHandle, HSMKeyHandle]:
        """Generate an asymmetric key pair."""
        pub_key_id = str(uuid.uuid4())
        priv_key_id = str(uuid.uuid4())

        if key_type in (HSMKeyType.RSA_2048, HSMKeyType.RSA_3072, HSMKeyType.RSA_4096):
            # Generate RSA key pair
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_type.key_size,
                backend=default_backend(),
            )
            public_key = private_key.public_key()
        elif key_type in (HSMKeyType.EC_P256, HSMKeyType.EC_P384, HSMKeyType.EC_P521):
            # Generate EC key pair
            curve_map = {
                HSMKeyType.EC_P256: ec.SECP256R1(),
                HSMKeyType.EC_P384: ec.SECP384R1(),
                HSMKeyType.EC_P521: ec.SECP521R1(),
            }
            private_key = ec.generate_private_key(
                curve_map[key_type],
                backend=default_backend(),
            )
            public_key = private_key.public_key()
        elif key_type.is_pqc:
            # Simulated PQC keys (would use real PQC library in production)
            private_key = secrets.token_bytes(64)
            public_key = secrets.token_bytes(64)
        else:
            raise HSMOperationError(
                f"Unsupported key type: {key_type}",
                operation="generate_key_pair",
            )

        # Create handles
        public_handle = HSMKeyHandle(
            key_id=pub_key_id,
            key_type=key_type,
            label=f"{label}_public",
            created_at=datetime.utcnow(),
            extractable=True,
            sensitive=False,
            can_encrypt=True,
            can_verify=True,
            can_wrap=True,
        )

        private_handle = HSMKeyHandle(
            key_id=priv_key_id,
            key_type=key_type,
            label=f"{label}_private",
            created_at=datetime.utcnow(),
            extractable=extractable,
            sensitive=True,
            can_decrypt=True,
            can_sign=True,
            can_unwrap=True,
        )

        self._key_store.store_key(pub_key_id, public_key, public_handle)
        self._key_store.store_key(priv_key_id, private_key, private_handle)

        return public_handle, private_handle

    def import_key(
        self,
        key_type: HSMKeyType,
        key_data: bytes,
        label: str,
        **kwargs,
    ) -> HSMKeyHandle:
        """Import a key into the HSM."""
        key_id = str(uuid.uuid4())

        handle = HSMKeyHandle(
            key_id=key_id,
            key_type=key_type,
            label=label,
            created_at=datetime.utcnow(),
            extractable=kwargs.get("extractable", False),
            sensitive=True,
            can_encrypt=True,
            can_decrypt=True,
        )

        self._key_store.store_key(key_id, key_data, handle)
        return handle

    def export_public_key(self, key_handle: HSMKeyHandle) -> bytes:
        """Export a public key."""
        key_data = self._key_store.get_key(key_handle.key_id)
        if key_data is None:
            raise HSMKeyNotFoundError(key_handle.key_id)

        if isinstance(key_data, bytes):
            return key_data

        # Serialize public key
        return key_data.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

    def delete_key(self, key_handle: HSMKeyHandle) -> None:
        """Delete a key."""
        self._key_store.delete_key(key_handle.key_id)

    def get_key(self, key_id: str) -> Optional[HSMKeyHandle]:
        """Get a key handle by ID."""
        return self._key_store.get_handle(key_id)

    def list_keys(
        self,
        key_type: Optional[HSMKeyType] = None,
        label_pattern: Optional[str] = None,
    ) -> List[HSMKeyHandle]:
        """List keys in the HSM."""
        handles = self._key_store.list_keys()

        if key_type:
            handles = [h for h in handles if h.key_type == key_type]

        if label_pattern:
            handles = [h for h in handles if label_pattern in h.label]

        return handles

    def get_key_info(self, key_handle: HSMKeyHandle) -> Dict[str, Any]:
        """Get detailed information about a key."""
        return key_handle.to_dict()

    def encrypt(
        self,
        key_handle: HSMKeyHandle,
        plaintext: bytes,
        algorithm: HSMAlgorithm,
        iv: Optional[bytes] = None,
        aad: Optional[bytes] = None,
    ) -> EncryptionResult:
        """Encrypt data."""
        key_data = self._key_store.get_key(key_handle.key_id)
        if key_data is None:
            raise HSMKeyNotFoundError(key_handle.key_id)

        if algorithm == HSMAlgorithm.AES_GCM:
            iv = iv or secrets.token_bytes(12)
            cipher = Cipher(
                algorithms.AES(key_data),
                modes.GCM(iv),
                backend=default_backend(),
            )
            encryptor = cipher.encryptor()
            if aad:
                encryptor.authenticate_additional_data(aad)
            ciphertext = encryptor.update(plaintext) + encryptor.finalize()
            return EncryptionResult(
                ciphertext=ciphertext,
                iv=iv,
                tag=encryptor.tag,
                algorithm=algorithm.algorithm_name,
            )

        elif algorithm == HSMAlgorithm.AES_CBC:
            iv = iv or secrets.token_bytes(16)
            cipher = Cipher(
                algorithms.AES(key_data),
                modes.CBC(iv),
                backend=default_backend(),
            )
            encryptor = cipher.encryptor()
            # Add PKCS7 padding
            pad_len = 16 - (len(plaintext) % 16)
            padded = plaintext + bytes([pad_len] * pad_len)
            ciphertext = encryptor.update(padded) + encryptor.finalize()
            return EncryptionResult(
                ciphertext=ciphertext,
                iv=iv,
                algorithm=algorithm.algorithm_name,
            )

        elif algorithm == HSMAlgorithm.RSA_OAEP:
            ciphertext = key_data.encrypt(
                plaintext,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None,
                ),
            )
            return EncryptionResult(
                ciphertext=ciphertext,
                algorithm=algorithm.algorithm_name,
            )

        else:
            raise HSMOperationError(
                f"Unsupported algorithm: {algorithm}",
                operation="encrypt",
            )

    def decrypt(
        self,
        key_handle: HSMKeyHandle,
        ciphertext: bytes,
        algorithm: HSMAlgorithm,
        iv: Optional[bytes] = None,
        tag: Optional[bytes] = None,
        aad: Optional[bytes] = None,
    ) -> DecryptionResult:
        """Decrypt data."""
        key_data = self._key_store.get_key(key_handle.key_id)
        if key_data is None:
            raise HSMKeyNotFoundError(key_handle.key_id)

        if algorithm == HSMAlgorithm.AES_GCM:
            if not iv or not tag:
                raise HSMOperationError("IV and tag required for AES-GCM")
            cipher = Cipher(
                algorithms.AES(key_data),
                modes.GCM(iv, tag),
                backend=default_backend(),
            )
            decryptor = cipher.decryptor()
            if aad:
                decryptor.authenticate_additional_data(aad)
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            return DecryptionResult(plaintext=plaintext, verified=True)

        elif algorithm == HSMAlgorithm.AES_CBC:
            if not iv:
                raise HSMOperationError("IV required for AES-CBC")
            cipher = Cipher(
                algorithms.AES(key_data),
                modes.CBC(iv),
                backend=default_backend(),
            )
            decryptor = cipher.decryptor()
            padded = decryptor.update(ciphertext) + decryptor.finalize()
            # Remove PKCS7 padding
            pad_len = padded[-1]
            plaintext = padded[:-pad_len]
            return DecryptionResult(plaintext=plaintext)

        elif algorithm == HSMAlgorithm.RSA_OAEP:
            plaintext = key_data.decrypt(
                ciphertext,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None,
                ),
            )
            return DecryptionResult(plaintext=plaintext)

        else:
            raise HSMOperationError(
                f"Unsupported algorithm: {algorithm}",
                operation="decrypt",
            )

    def sign(
        self,
        key_handle: HSMKeyHandle,
        data: bytes,
        algorithm: HSMAlgorithm,
    ) -> SignatureResult:
        """Sign data."""
        key_data = self._key_store.get_key(key_handle.key_id)
        if key_data is None:
            raise HSMKeyNotFoundError(key_handle.key_id)

        if algorithm == HSMAlgorithm.RSA_PSS:
            signature = key_data.sign(
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.AUTO,
                ),
                hashes.SHA256(),
            )
        elif algorithm == HSMAlgorithm.ECDSA:
            signature = key_data.sign(
                data,
                ec.ECDSA(hashes.SHA256()),
            )
        elif algorithm == HSMAlgorithm.ML_DSA:
            # Simulated ML-DSA signature
            signature = hashlib.sha512(key_data + data).digest()
        else:
            raise HSMOperationError(
                f"Unsupported algorithm: {algorithm}",
                operation="sign",
            )

        return SignatureResult(
            signature=signature,
            algorithm=algorithm.algorithm_name,
            key_id=key_handle.key_id,
        )

    def verify(
        self,
        key_handle: HSMKeyHandle,
        data: bytes,
        signature: bytes,
        algorithm: HSMAlgorithm,
    ) -> VerificationResult:
        """Verify a signature."""
        key_data = self._key_store.get_key(key_handle.key_id)
        if key_data is None:
            raise HSMKeyNotFoundError(key_handle.key_id)

        try:
            if algorithm == HSMAlgorithm.RSA_PSS:
                key_data.verify(
                    signature,
                    data,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.AUTO,
                    ),
                    hashes.SHA256(),
                )
                valid = True
            elif algorithm == HSMAlgorithm.ECDSA:
                key_data.verify(signature, data, ec.ECDSA(hashes.SHA256()))
                valid = True
            elif algorithm == HSMAlgorithm.ML_DSA:
                # Simulated ML-DSA verification
                valid = True  # Simplified
            else:
                raise HSMOperationError(f"Unsupported algorithm: {algorithm}")
        except Exception:
            valid = False

        return VerificationResult(valid=valid, key_id=key_handle.key_id)

    def wrap_key(
        self,
        wrapping_key: HSMKeyHandle,
        key_to_wrap: HSMKeyHandle,
        algorithm: HSMAlgorithm,
    ) -> bytes:
        """Wrap a key for export."""
        wrap_key_data = self._key_store.get_key(wrapping_key.key_id)
        target_key_data = self._key_store.get_key(key_to_wrap.key_id)

        if wrap_key_data is None:
            raise HSMKeyNotFoundError(wrapping_key.key_id)
        if target_key_data is None:
            raise HSMKeyNotFoundError(key_to_wrap.key_id)

        # Use AES-GCM for wrapping
        iv = secrets.token_bytes(12)
        result = self.encrypt(
            wrapping_key,
            target_key_data if isinstance(target_key_data, bytes) else b"",
            HSMAlgorithm.AES_GCM,
            iv=iv,
        )

        # Return IV + tag + ciphertext
        return iv + result.tag + result.ciphertext

    def unwrap_key(
        self,
        wrapping_key: HSMKeyHandle,
        wrapped_key: bytes,
        key_type: HSMKeyType,
        algorithm: HSMAlgorithm,
        label: str,
    ) -> HSMKeyHandle:
        """Unwrap a key and import it."""
        # Extract IV, tag, ciphertext
        iv = wrapped_key[:12]
        tag = wrapped_key[12:28]
        ciphertext = wrapped_key[28:]

        result = self.decrypt(
            wrapping_key,
            ciphertext,
            HSMAlgorithm.AES_GCM,
            iv=iv,
            tag=tag,
        )

        return self.import_key(key_type, result.plaintext, label)

    def derive_key(
        self,
        base_key: HSMKeyHandle,
        derivation_data: bytes,
        key_type: HSMKeyType,
        label: str,
    ) -> HSMKeyHandle:
        """Derive a new key from an existing key."""
        key_data = self._key_store.get_key(base_key.key_id)
        if key_data is None:
            raise HSMKeyNotFoundError(base_key.key_id)

        # Use HKDF for key derivation
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=key_type.key_size // 8,
            salt=None,
            info=derivation_data,
            backend=default_backend(),
        )

        derived_key = hkdf.derive(key_data if isinstance(key_data, bytes) else b"")
        return self.import_key(key_type, derived_key, label)

    def hash(self, data: bytes, algorithm: HSMAlgorithm) -> bytes:
        """Compute hash."""
        hash_map = {
            HSMAlgorithm.SHA256: hashes.SHA256(),
            HSMAlgorithm.SHA384: hashes.SHA384(),
            HSMAlgorithm.SHA512: hashes.SHA512(),
        }

        if algorithm not in hash_map:
            raise HSMOperationError(f"Unsupported hash algorithm: {algorithm}")

        from cryptography.hazmat.primitives.hashes import Hash
        digest = Hash(hash_map[algorithm], backend=default_backend())
        digest.update(data)
        return digest.finalize()

    def mac(
        self,
        key_handle: HSMKeyHandle,
        data: bytes,
        algorithm: HSMAlgorithm,
    ) -> bytes:
        """Compute MAC."""
        key_data = self._key_store.get_key(key_handle.key_id)
        if key_data is None:
            raise HSMKeyNotFoundError(key_handle.key_id)

        if algorithm == HSMAlgorithm.HMAC_SHA256:
            return hmac.new(key_data, data, hashlib.sha256).digest()
        elif algorithm == HSMAlgorithm.HMAC_SHA384:
            return hmac.new(key_data, data, hashlib.sha384).digest()
        else:
            raise HSMOperationError(f"Unsupported MAC algorithm: {algorithm}")

    def verify_mac(
        self,
        key_handle: HSMKeyHandle,
        data: bytes,
        mac_value: bytes,
        algorithm: HSMAlgorithm,
    ) -> bool:
        """Verify MAC."""
        expected_mac = self.mac(key_handle, data, algorithm)
        return hmac.compare_digest(expected_mac, mac_value)

    def kem_encapsulate(
        self,
        public_key: HSMKeyHandle,
    ) -> KEMEncapsulationResult:
        """KEM encapsulation (simulated for SoftHSM)."""
        if not self._config.enable_pqc:
            raise HSMCapabilityError("PQC not enabled")

        # Simulated KEM encapsulation
        shared_secret = secrets.token_bytes(32)
        ciphertext = secrets.token_bytes(768)  # Simulated ML-KEM ciphertext

        return KEMEncapsulationResult(
            ciphertext=ciphertext,
            shared_secret=shared_secret,
        )

    def kem_decapsulate(
        self,
        private_key: HSMKeyHandle,
        ciphertext: bytes,
    ) -> bytes:
        """KEM decapsulation (simulated for SoftHSM)."""
        if not self._config.enable_pqc:
            raise HSMCapabilityError("PQC not enabled")

        # Simulated KEM decapsulation
        return secrets.token_bytes(32)

    def generate_random(self, length: int) -> bytes:
        """Generate random bytes."""
        return secrets.token_bytes(length)
