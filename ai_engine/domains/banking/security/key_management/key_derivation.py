"""
Key Derivation

Provides key derivation functions for banking applications.
"""

import hashlib
import hmac
from enum import Enum
from typing import Optional, Tuple

from ai_engine.domains.banking.security.hsm import (
    HSMProvider,
    HSMKeyHandle,
    HSMKeyType,
    HSMAlgorithm,
)


class DerivationMethod(Enum):
    """Key derivation methods."""

    HKDF = "hkdf"  # HMAC-based Key Derivation Function
    PBKDF2 = "pbkdf2"  # Password-Based Key Derivation Function 2
    SP800_108_CTR = "sp800_108_ctr"  # NIST SP 800-108 Counter Mode
    SP800_108_FEEDBACK = "sp800_108_feedback"  # NIST SP 800-108 Feedback Mode
    EMV_OPTION_A = "emv_option_a"  # EMV Option A (for ICC Master Key)
    EMV_OPTION_B = "emv_option_b"  # EMV Option B
    TR31 = "tr31"  # TR-31 Key Block derivation


class KeyDerivation:
    """
    Key derivation operations for banking applications.

    Supports:
    - HKDF for general key derivation
    - NIST SP 800-108 for banking applications
    - EMV key derivation for card processing
    - TR-31 key block formats
    """

    def __init__(self, hsm_provider: Optional[HSMProvider] = None):
        """
        Initialize key derivation.

        Args:
            hsm_provider: Optional HSM for hardware-based derivation
        """
        self._hsm = hsm_provider

    def derive_hkdf(
        self,
        input_key_material: bytes,
        salt: Optional[bytes],
        info: bytes,
        length: int,
        hash_algorithm: str = "SHA256",
    ) -> bytes:
        """
        Derive key using HKDF (RFC 5869).

        Args:
            input_key_material: Input keying material
            salt: Optional salt (random value)
            info: Context and application specific information
            length: Length of output keying material in bytes

        Returns:
            Derived key material
        """
        hash_map = {
            "SHA256": hashlib.sha256,
            "SHA384": hashlib.sha384,
            "SHA512": hashlib.sha512,
        }

        if hash_algorithm not in hash_map:
            raise ValueError(f"Unsupported hash algorithm: {hash_algorithm}")

        hash_func = hash_map[hash_algorithm]
        hash_len = hash_func().digest_size

        # HKDF-Extract
        if salt is None:
            salt = bytes(hash_len)
        prk = hmac.new(salt, input_key_material, hash_func).digest()

        # HKDF-Expand
        okm = b""
        t = b""
        n = (length + hash_len - 1) // hash_len

        for i in range(1, n + 1):
            t = hmac.new(prk, t + info + bytes([i]), hash_func).digest()
            okm += t

        return okm[:length]

    def derive_sp800_108_counter(
        self,
        key: bytes,
        label: bytes,
        context: bytes,
        length: int,
        r_len: int = 4,
        hash_algorithm: str = "SHA256",
    ) -> bytes:
        """
        Derive key using NIST SP 800-108 Counter Mode.

        Args:
            key: Key derivation key
            label: Label identifying the purpose
            context: Context information
            length: Length of output in bytes
            r_len: Length of counter in bytes (1-4)
            hash_algorithm: Hash algorithm to use

        Returns:
            Derived key material
        """
        hash_map = {
            "SHA256": hashlib.sha256,
            "SHA384": hashlib.sha384,
            "SHA512": hashlib.sha512,
        }

        if hash_algorithm not in hash_map:
            raise ValueError(f"Unsupported hash algorithm: {hash_algorithm}")

        hash_func = hash_map[hash_algorithm]
        hash_len = hash_func().digest_size

        # Calculate number of iterations
        n = (length + hash_len - 1) // hash_len

        # Fixed input data: Label || 0x00 || Context || L
        fixed_input = label + b"\x00" + context + length.to_bytes(4, "big")

        result = b""
        for i in range(1, n + 1):
            # Counter || Fixed Input
            counter = i.to_bytes(r_len, "big")
            data = counter + fixed_input

            # K(i) = PRF(K_in, counter || fixed_input)
            k_i = hmac.new(key, data, hash_func).digest()
            result += k_i

        return result[:length]

    def derive_emv_icc_master_key(
        self,
        issuer_master_key: bytes,
        pan: str,
        pan_sequence: str = "00",
        method: DerivationMethod = DerivationMethod.EMV_OPTION_A,
    ) -> bytes:
        """
        Derive ICC Master Key from Issuer Master Key.

        Args:
            issuer_master_key: 16-byte Issuer Master Key
            pan: Primary Account Number
            pan_sequence: PAN Sequence Number (2 hex digits)
            method: EMV derivation method (A or B)

        Returns:
            16-byte ICC Master Key
        """
        if method == DerivationMethod.EMV_OPTION_A:
            return self._derive_emv_option_a(issuer_master_key, pan, pan_sequence)
        elif method == DerivationMethod.EMV_OPTION_B:
            return self._derive_emv_option_b(issuer_master_key, pan, pan_sequence)
        else:
            raise ValueError(f"Invalid EMV derivation method: {method}")

    def _derive_emv_option_a(
        self,
        imk: bytes,
        pan: str,
        psn: str,
    ) -> bytes:
        """EMV Option A derivation."""
        # Concatenate PAN and PSN, pad/truncate to 16 hex digits
        pan_psn = pan + psn
        pan_psn = pan_psn.ljust(16, "0")[:16]

        # Convert to bytes
        derivation_data = bytes.fromhex(pan_psn)

        # Triple DES encrypt for left half
        from ai_engine.domains.banking.security.hsm.soft_hsm import CRYPTO_AVAILABLE

        if CRYPTO_AVAILABLE:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            from cryptography.hazmat.backends import default_backend

            # Expand 16-byte key to 24-byte 3DES key
            key_3des = imk + imk[:8]

            cipher = Cipher(
                algorithms.TripleDES(key_3des),
                modes.ECB(),
                backend=default_backend(),
            )

            # Left half
            encryptor = cipher.encryptor()
            left = encryptor.update(derivation_data) + encryptor.finalize()

            # Right half (XOR input with 0xFFFFFFFF...)
            xored = bytes(b ^ 0xFF for b in derivation_data)
            encryptor = cipher.encryptor()
            right = encryptor.update(xored) + encryptor.finalize()

            return left + right

        raise ValueError("cryptography library required for EMV derivation")

    def _derive_emv_option_b(
        self,
        imk: bytes,
        pan: str,
        psn: str,
    ) -> bytes:
        """EMV Option B derivation (hash-based)."""
        # SHA-1 of PAN || PSN
        data = (pan + psn).encode()
        hash_result = hashlib.sha1(data).digest()

        # Use first 16 bytes
        derivation_data = hash_result[:16]

        # Same encryption as Option A
        return self._derive_emv_option_a.__func__(self, imk, derivation_data.hex(), "")

    def derive_session_key(
        self,
        master_key: bytes,
        atc: str,
        diversification_data: Optional[bytes] = None,
    ) -> Tuple[bytes, bytes]:
        """
        Derive EMV session keys (encryption and MAC).

        Args:
            master_key: ICC Master Key
            atc: Application Transaction Counter (4 hex digits)
            diversification_data: Optional additional diversification

        Returns:
            Tuple of (encryption_key, mac_key)
        """
        # Pad ATC to 4 bytes
        atc_bytes = bytes.fromhex(atc.zfill(4))

        # Derive encryption key (using "0000" prefix)
        enc_derivation = b"\x00\x00" + atc_bytes + b"\x00" * 10
        if diversification_data:
            enc_derivation = enc_derivation[:8] + diversification_data[:8]

        enc_key = self._derive_with_3des(master_key, enc_derivation)

        # Derive MAC key (using "0001" prefix)
        mac_derivation = b"\x00\x01" + atc_bytes + b"\x00" * 10
        if diversification_data:
            mac_derivation = mac_derivation[:8] + diversification_data[:8]

        mac_key = self._derive_with_3des(master_key, mac_derivation)

        return enc_key, mac_key

    def _derive_with_3des(self, key: bytes, data: bytes) -> bytes:
        """Derive key using 3DES encryption."""
        from ai_engine.domains.banking.security.hsm.soft_hsm import CRYPTO_AVAILABLE

        if CRYPTO_AVAILABLE:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            from cryptography.hazmat.backends import default_backend

            # Expand to 24-byte key
            key_3des = key + key[:8] if len(key) == 16 else key

            cipher = Cipher(
                algorithms.TripleDES(key_3des),
                modes.ECB(),
                backend=default_backend(),
            )

            encryptor = cipher.encryptor()
            result = encryptor.update(data[:8]) + encryptor.finalize()

            return result

        raise ValueError("cryptography library required")

    def derive_pin_key(
        self,
        zpk: bytes,
        derivation_data: bytes,
    ) -> bytes:
        """
        Derive PIN encryption key.

        Args:
            zpk: Zone PIN Key
            derivation_data: Derivation data (e.g., from ATM)

        Returns:
            Derived PIN key
        """
        return self.derive_sp800_108_counter(
            key=zpk,
            label=b"PIN_KEY",
            context=derivation_data,
            length=16,
        )

    def derive_with_hsm(
        self,
        base_key_handle: HSMKeyHandle,
        derivation_data: bytes,
        output_key_type: HSMKeyType,
        label: str,
    ) -> HSMKeyHandle:
        """
        Derive key using HSM.

        Args:
            base_key_handle: HSM handle of base key
            derivation_data: Derivation parameters
            output_key_type: Type of derived key
            label: Label for derived key

        Returns:
            HSM handle of derived key
        """
        if not self._hsm:
            raise ValueError("HSM provider required for HSM-based derivation")

        return self._hsm.derive_key(
            base_key=base_key_handle,
            derivation_data=derivation_data,
            key_type=output_key_type,
            label=label,
        )
