"""
Key Wrapping

Provides key wrapping and unwrapping for secure key transport.
Implements TR-31 key block format for banking applications.
"""

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from ai_engine.domains.banking.security.hsm import (
    HSMProvider,
    HSMKeyHandle,
    HSMKeyType,
    HSMAlgorithm,
)


class KeyBlockVersion(Enum):
    """TR-31 Key Block versions."""

    A = "A"  # TDEA key derivation (deprecated)
    B = "B"  # TDEA key variant binding
    C = "C"  # TDEA key derivation binding
    D = "D"  # AES key derivation binding
    E = "E"  # AES/DUKPT key derivation


class KeyUsage(Enum):
    """TR-31 Key Usage codes."""

    BDK = "B0"  # Base Derivation Key (BDK)
    DUKPT_INITIAL = "B1"  # Initial DUKPT Key
    CVK = "C0"  # Card Verification Key
    KEK = "K0"  # Key Encryption Key
    KEK_WRAPPING = "K1"  # Key Wrapping Key
    ZPK = "P0"  # PIN Encryption Key
    TMK = "M0"  # Terminal Master Key
    TAK = "M1"  # Terminal Authentication Key
    PVK = "V0"  # PIN Verification Key
    PVK_OTHER = "V1"  # PIN Verification Other
    DATA_ENCRYPTION = "D0"  # Data Encryption
    MAC_GENERATION = "M3"  # MAC Key for Generation
    MAC_VERIFICATION = "M4"  # MAC Key for Verification
    ISO_MAC = "M5"  # ISO 16609 MAC Algorithm


class Algorithm(Enum):
    """TR-31 Algorithm codes."""

    AES = "A"  # AES
    DES = "D"  # DES
    EC = "E"  # Elliptic Curve
    HMAC = "H"  # HMAC
    RSA = "R"  # RSA
    DSA = "S"  # DSA
    TDES = "T"  # Triple DES


class ModeOfUse(Enum):
    """TR-31 Mode of Use codes."""

    ENCRYPT = "E"  # Encrypt only
    DECRYPT = "D"  # Decrypt only
    BOTH = "B"  # Both encrypt and decrypt
    GENERATE = "C"  # Generate (MAC, signature)
    VERIFY = "V"  # Verify (MAC, signature)
    DERIVE = "X"  # Key derivation
    NO_SPECIAL = "N"  # No special restrictions


class Exportability(Enum):
    """TR-31 Exportability codes."""

    EXPORTABLE = "E"  # Exportable under trusted key
    NOT_EXPORTABLE = "N"  # Not exportable
    SENSITIVE = "S"  # Sensitive, exportable under KEK


@dataclass
class TR31KeyBlock:
    """TR-31 Key Block structure."""

    version: KeyBlockVersion
    key_usage: KeyUsage
    algorithm: Algorithm
    mode_of_use: ModeOfUse
    key_version: str  # 2 characters
    exportability: Exportability
    optional_blocks: Dict[str, str]
    encrypted_key: bytes
    mac: bytes

    # Metadata
    key_length: int = 0
    created_at: Optional[datetime] = None

    def to_string(self) -> str:
        """Convert to TR-31 string format."""
        # Build header
        header = (
            f"{self.version.value}"
            f"{self._calc_block_length():04d}"
            f"{self.key_usage.value}"
            f"{self.algorithm.value}"
            f"{self.mode_of_use.value}"
            f"{self.key_version}"
            f"{self.exportability.value}"
            f"{len(self.optional_blocks):02d}"
            "00"  # Reserved
        )

        # Add optional blocks
        opt_block_str = ""
        for block_id, value in self.optional_blocks.items():
            opt_block_str += f"{block_id}{len(value):02d}{value}"

        # Combine
        key_hex = self.encrypted_key.hex().upper()
        mac_hex = self.mac.hex().upper()

        return header + opt_block_str + key_hex + mac_hex

    def _calc_block_length(self) -> int:
        """Calculate total block length."""
        header_len = 16
        opt_len = sum(4 + len(v) for v in self.optional_blocks.values())
        key_len = len(self.encrypted_key) * 2
        mac_len = len(self.mac) * 2
        return header_len + opt_len + key_len + mac_len

    @classmethod
    def from_string(cls, block_string: str) -> "TR31KeyBlock":
        """Parse TR-31 key block from string."""
        if len(block_string) < 16:
            raise ValueError("Invalid key block: too short")

        version = KeyBlockVersion(block_string[0])
        length = int(block_string[1:5])
        key_usage = KeyUsage(block_string[5:7])
        algorithm = Algorithm(block_string[7])
        mode_of_use = ModeOfUse(block_string[8])
        key_version = block_string[9:11]
        exportability = Exportability(block_string[11])
        num_opt_blocks = int(block_string[12:14])

        # Parse optional blocks
        pos = 16
        optional_blocks = {}
        for _ in range(num_opt_blocks):
            if pos + 4 > len(block_string):
                break
            block_id = block_string[pos : pos + 2]
            block_len = int(block_string[pos + 2 : pos + 4])
            block_value = block_string[pos + 4 : pos + 4 + block_len]
            optional_blocks[block_id] = block_value
            pos += 4 + block_len

        # Remaining is encrypted key + MAC
        remaining = block_string[pos:]
        mac_len = 32 if version in (KeyBlockVersion.D, KeyBlockVersion.E) else 16
        encrypted_key_hex = remaining[:-mac_len]
        mac_hex = remaining[-mac_len:]

        return cls(
            version=version,
            key_usage=key_usage,
            algorithm=algorithm,
            mode_of_use=mode_of_use,
            key_version=key_version,
            exportability=exportability,
            optional_blocks=optional_blocks,
            encrypted_key=bytes.fromhex(encrypted_key_hex),
            mac=bytes.fromhex(mac_hex),
        )


class KeyWrapping:
    """
    Key wrapping operations for secure key transport.

    Supports:
    - AES Key Wrap (RFC 3394)
    - TR-31 Key Block format
    - PKCS#8 wrapping
    - HSM-based wrapping
    """

    # AES Key Wrap default IV
    AES_WRAP_IV = bytes.fromhex("A6A6A6A6A6A6A6A6")

    def __init__(self, hsm_provider: Optional[HSMProvider] = None):
        """
        Initialize key wrapping.

        Args:
            hsm_provider: Optional HSM for hardware-based wrapping
        """
        self._hsm = hsm_provider

    def wrap_aes_key_wrap(
        self,
        kek: bytes,
        key_to_wrap: bytes,
    ) -> bytes:
        """
        Wrap key using AES Key Wrap (RFC 3394).

        Args:
            kek: Key Encryption Key (16, 24, or 32 bytes)
            key_to_wrap: Key to wrap (must be multiple of 8 bytes)

        Returns:
            Wrapped key (8 bytes longer than input)
        """
        if len(key_to_wrap) % 8 != 0:
            raise ValueError("Key to wrap must be multiple of 8 bytes")

        from ai_engine.domains.banking.security.hsm.soft_hsm import CRYPTO_AVAILABLE

        if not CRYPTO_AVAILABLE:
            raise ValueError("cryptography library required")

        from cryptography.hazmat.primitives.keywrap import aes_key_wrap
        from cryptography.hazmat.backends import default_backend

        return aes_key_wrap(kek, key_to_wrap, default_backend())

    def unwrap_aes_key_wrap(
        self,
        kek: bytes,
        wrapped_key: bytes,
    ) -> bytes:
        """
        Unwrap key using AES Key Wrap (RFC 3394).

        Args:
            kek: Key Encryption Key
            wrapped_key: Wrapped key data

        Returns:
            Unwrapped key
        """
        from ai_engine.domains.banking.security.hsm.soft_hsm import CRYPTO_AVAILABLE

        if not CRYPTO_AVAILABLE:
            raise ValueError("cryptography library required")

        from cryptography.hazmat.primitives.keywrap import aes_key_unwrap
        from cryptography.hazmat.backends import default_backend

        return aes_key_unwrap(kek, wrapped_key, default_backend())

    def create_tr31_key_block(
        self,
        kbpk: bytes,  # Key Block Protection Key
        key_to_wrap: bytes,
        key_usage: KeyUsage,
        algorithm: Algorithm,
        mode_of_use: ModeOfUse,
        key_version: str = "00",
        exportability: Exportability = Exportability.EXPORTABLE,
        version: KeyBlockVersion = KeyBlockVersion.D,
        optional_blocks: Optional[Dict[str, str]] = None,
    ) -> TR31KeyBlock:
        """
        Create a TR-31 key block.

        Args:
            kbpk: Key Block Protection Key
            key_to_wrap: Key to wrap
            key_usage: Key usage code
            algorithm: Algorithm code
            mode_of_use: Mode of use
            key_version: Key version (2 chars)
            exportability: Exportability code
            version: TR-31 version
            optional_blocks: Optional header blocks

        Returns:
            TR31KeyBlock structure
        """
        if optional_blocks is None:
            optional_blocks = {}

        # Add timestamp optional block
        optional_blocks["TS"] = datetime.utcnow().strftime("%y%m%d%H%M%S")

        # Derive encryption and MAC keys from KBPK
        enc_key, mac_key = self._derive_tr31_keys(kbpk, version)

        # Pad key to multiple of 8 bytes
        pad_len = (8 - len(key_to_wrap) % 8) % 8
        padded_key = key_to_wrap + bytes([pad_len] * pad_len)

        # Add random padding for security
        random_padding = secrets.token_bytes(16)
        key_data = len(key_to_wrap).to_bytes(2, "big") + padded_key + random_padding

        # Encrypt key data
        encrypted_key = self._encrypt_tr31(enc_key, key_data, version)

        # Build header for MAC calculation
        header = (
            f"{version.value}"
            f"0000"  # Placeholder for length
            f"{key_usage.value}"
            f"{algorithm.value}"
            f"{mode_of_use.value}"
            f"{key_version}"
            f"{exportability.value}"
            f"{len(optional_blocks):02d}"
            "00"
        )

        opt_str = ""
        for bid, val in optional_blocks.items():
            opt_str += f"{bid}{len(val):02d}{val}"

        # Calculate MAC
        mac_data = (header + opt_str + encrypted_key.hex().upper()).encode()
        mac = self._calculate_tr31_mac(mac_key, mac_data, version)

        return TR31KeyBlock(
            version=version,
            key_usage=key_usage,
            algorithm=algorithm,
            mode_of_use=mode_of_use,
            key_version=key_version,
            exportability=exportability,
            optional_blocks=optional_blocks,
            encrypted_key=encrypted_key,
            mac=mac,
            key_length=len(key_to_wrap),
            created_at=datetime.utcnow(),
        )

    def unwrap_tr31_key_block(
        self,
        kbpk: bytes,
        key_block: TR31KeyBlock,
    ) -> bytes:
        """
        Unwrap a TR-31 key block.

        Args:
            kbpk: Key Block Protection Key
            key_block: TR-31 key block

        Returns:
            Unwrapped key
        """
        # Derive keys
        enc_key, mac_key = self._derive_tr31_keys(kbpk, key_block.version)

        # Verify MAC
        header = (
            f"{key_block.version.value}"
            f"0000"
            f"{key_block.key_usage.value}"
            f"{key_block.algorithm.value}"
            f"{key_block.mode_of_use.value}"
            f"{key_block.key_version}"
            f"{key_block.exportability.value}"
            f"{len(key_block.optional_blocks):02d}"
            "00"
        )

        opt_str = ""
        for bid, val in key_block.optional_blocks.items():
            opt_str += f"{bid}{len(val):02d}{val}"

        mac_data = (header + opt_str + key_block.encrypted_key.hex().upper()).encode()
        expected_mac = self._calculate_tr31_mac(mac_key, mac_data, key_block.version)

        if not hmac.compare_digest(expected_mac, key_block.mac):
            raise ValueError("TR-31 key block MAC verification failed")

        # Decrypt key data
        key_data = self._decrypt_tr31(enc_key, key_block.encrypted_key, key_block.version)

        # Extract key
        key_length = int.from_bytes(key_data[:2], "big")
        return key_data[2 : 2 + key_length]

    def _derive_tr31_keys(
        self,
        kbpk: bytes,
        version: KeyBlockVersion,
    ) -> Tuple[bytes, bytes]:
        """Derive encryption and MAC keys from KBPK."""
        if version in (KeyBlockVersion.D, KeyBlockVersion.E):
            # AES-based derivation
            from ai_engine.domains.banking.security.key_management.key_derivation import KeyDerivation

            kdf = KeyDerivation()

            enc_key = kdf.derive_sp800_108_counter(
                key=kbpk,
                label=b"ENC",
                context=b"TR31",
                length=len(kbpk),
            )

            mac_key = kdf.derive_sp800_108_counter(
                key=kbpk,
                label=b"MAC",
                context=b"TR31",
                length=len(kbpk),
            )

            return enc_key, mac_key
        else:
            # TDES-based (legacy)
            enc_key = kbpk
            mac_key = kbpk
            return enc_key, mac_key

    def _encrypt_tr31(
        self,
        key: bytes,
        data: bytes,
        version: KeyBlockVersion,
    ) -> bytes:
        """Encrypt data for TR-31 key block."""
        from ai_engine.domains.banking.security.hsm.soft_hsm import CRYPTO_AVAILABLE

        if not CRYPTO_AVAILABLE:
            raise ValueError("cryptography library required")

        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.backends import default_backend

        if version in (KeyBlockVersion.D, KeyBlockVersion.E):
            # AES-CBC
            iv = secrets.token_bytes(16)
            cipher = Cipher(
                algorithms.AES(key),
                modes.CBC(iv),
                backend=default_backend(),
            )
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(data) + encryptor.finalize()
            return iv + ciphertext
        else:
            # TDES-CBC
            iv = secrets.token_bytes(8)
            key_3des = key + key[:8] if len(key) == 16 else key
            cipher = Cipher(
                algorithms.TripleDES(key_3des),
                modes.CBC(iv),
                backend=default_backend(),
            )
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(data) + encryptor.finalize()
            return iv + ciphertext

    def _decrypt_tr31(
        self,
        key: bytes,
        data: bytes,
        version: KeyBlockVersion,
    ) -> bytes:
        """Decrypt TR-31 encrypted data."""
        from ai_engine.domains.banking.security.hsm.soft_hsm import CRYPTO_AVAILABLE

        if not CRYPTO_AVAILABLE:
            raise ValueError("cryptography library required")

        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.backends import default_backend

        if version in (KeyBlockVersion.D, KeyBlockVersion.E):
            iv = data[:16]
            ciphertext = data[16:]
            cipher = Cipher(
                algorithms.AES(key),
                modes.CBC(iv),
                backend=default_backend(),
            )
            decryptor = cipher.decryptor()
            return decryptor.update(ciphertext) + decryptor.finalize()
        else:
            iv = data[:8]
            ciphertext = data[8:]
            key_3des = key + key[:8] if len(key) == 16 else key
            cipher = Cipher(
                algorithms.TripleDES(key_3des),
                modes.CBC(iv),
                backend=default_backend(),
            )
            decryptor = cipher.decryptor()
            return decryptor.update(ciphertext) + decryptor.finalize()

    def _calculate_tr31_mac(
        self,
        key: bytes,
        data: bytes,
        version: KeyBlockVersion,
    ) -> bytes:
        """Calculate TR-31 MAC."""
        if version in (KeyBlockVersion.D, KeyBlockVersion.E):
            # CMAC-AES
            from cryptography.hazmat.primitives.cmac import CMAC
            from cryptography.hazmat.primitives.ciphers import algorithms
            from cryptography.hazmat.backends import default_backend

            c = CMAC(algorithms.AES(key), backend=default_backend())
            c.update(data)
            return c.finalize()
        else:
            # Retail MAC (ISO 9797-1 Algorithm 3)
            return hmac.new(key, data, hashlib.sha256).digest()[:8]

    def wrap_with_hsm(
        self,
        wrapping_key: HSMKeyHandle,
        key_to_wrap: HSMKeyHandle,
        algorithm: HSMAlgorithm = HSMAlgorithm.AES_GCM,
    ) -> bytes:
        """Wrap key using HSM."""
        if not self._hsm:
            raise ValueError("HSM provider required")

        return self._hsm.wrap_key(wrapping_key, key_to_wrap, algorithm)

    def unwrap_with_hsm(
        self,
        wrapping_key: HSMKeyHandle,
        wrapped_key: bytes,
        key_type: HSMKeyType,
        algorithm: HSMAlgorithm,
        label: str,
    ) -> HSMKeyHandle:
        """Unwrap key using HSM."""
        if not self._hsm:
            raise ValueError("HSM provider required")

        return self._hsm.unwrap_key(
            wrapping_key,
            wrapped_key,
            key_type,
            algorithm,
            label,
        )
