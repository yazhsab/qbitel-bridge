"""
Hybrid Post-Quantum Cryptography

Combines classical algorithms (RSA, ECDH, ECDSA) with post-quantum algorithms
(ML-KEM, ML-DSA) to provide security against both classical and quantum attacks.

Hybrid modes ensure security even if one algorithm is compromised:
- Hybrid KEM: ECDH + ML-KEM combined key establishment
- Hybrid Signature: ECDSA + ML-DSA dual signatures

These hybrid schemes are recommended during the PQC transition period
and follow NIST recommendations for algorithm agility.
"""

import hashlib
import secrets
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Union

from ai_engine.domains.banking.security.pqc.ml_kem import (
    MLKEM768,
    MLKEMKeyPair,
    MLKEMPublicKey,
    MLKEMPrivateKey,
    MLKEMEncapsulation,
    create_ml_kem,
)
from ai_engine.domains.banking.security.pqc.ml_dsa import (
    MLDSA65,
    MLDSAKeyPair,
    MLDSAPublicKey,
    MLDSAPrivateKey,
    MLDSASignature,
    create_ml_dsa,
)

# Try to import cryptography for classical algorithms
try:
    from cryptography.hazmat.primitives.asymmetric import ec, rsa, padding
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


class HybridMode(Enum):
    """Hybrid cryptography modes."""

    # KEM modes
    ECDH_MLKEM_768 = ("ECDH-ML-KEM-768", "kem", "P-384", "ML-KEM-768")
    ECDH_MLKEM_1024 = ("ECDH-ML-KEM-1024", "kem", "P-521", "ML-KEM-1024")
    RSA_MLKEM_768 = ("RSA-ML-KEM-768", "kem", "RSA-3072", "ML-KEM-768")

    # Signature modes
    ECDSA_MLDSA_65 = ("ECDSA-ML-DSA-65", "sig", "P-384", "ML-DSA-65")
    ECDSA_MLDSA_87 = ("ECDSA-ML-DSA-87", "sig", "P-521", "ML-DSA-87")
    RSA_MLDSA_65 = ("RSA-ML-DSA-65", "sig", "RSA-3072", "ML-DSA-65")

    def __init__(self, name: str, mode_type: str, classical: str, pqc: str):
        self.algorithm_name = name
        self.mode_type = mode_type
        self.classical_algorithm = classical
        self.pqc_algorithm = pqc


@dataclass
class HybridPublicKey:
    """Hybrid public key containing both classical and PQC components."""

    classical_key: bytes
    pqc_key: bytes
    mode: HybridMode

    def to_bytes(self) -> bytes:
        """Serialize hybrid public key."""
        # Format: [4-byte classical len][classical key][pqc key]
        classical_len = len(self.classical_key).to_bytes(4, "big")
        return classical_len + self.classical_key + self.pqc_key

    @classmethod
    def from_bytes(cls, data: bytes, mode: HybridMode) -> "HybridPublicKey":
        """Deserialize hybrid public key."""
        classical_len = int.from_bytes(data[:4], "big")
        classical_key = data[4:4 + classical_len]
        pqc_key = data[4 + classical_len:]
        return cls(classical_key=classical_key, pqc_key=pqc_key, mode=mode)


@dataclass
class HybridPrivateKey:
    """Hybrid private key containing both classical and PQC components."""

    classical_key: bytes
    pqc_key: bytes
    mode: HybridMode

    def to_bytes(self) -> bytes:
        """Serialize hybrid private key."""
        classical_len = len(self.classical_key).to_bytes(4, "big")
        return classical_len + self.classical_key + self.pqc_key

    @classmethod
    def from_bytes(cls, data: bytes, mode: HybridMode) -> "HybridPrivateKey":
        """Deserialize hybrid private key."""
        classical_len = int.from_bytes(data[:4], "big")
        classical_key = data[4:4 + classical_len]
        pqc_key = data[4 + classical_len:]
        return cls(classical_key=classical_key, pqc_key=pqc_key, mode=mode)


@dataclass
class HybridKeyPair:
    """Hybrid key pair."""

    public_key: HybridPublicKey
    private_key: HybridPrivateKey
    mode: HybridMode

    # Store typed components for operations
    _classical_public: Optional[object] = None
    _classical_private: Optional[object] = None
    _pqc_keypair: Optional[object] = None


@dataclass
class HybridEncapsulation:
    """Result of hybrid KEM encapsulation."""

    classical_ciphertext: bytes
    pqc_ciphertext: bytes
    combined_shared_secret: bytes

    def to_bytes(self) -> bytes:
        """Serialize encapsulation."""
        ct1_len = len(self.classical_ciphertext).to_bytes(4, "big")
        ct2_len = len(self.pqc_ciphertext).to_bytes(4, "big")
        return ct1_len + self.classical_ciphertext + ct2_len + self.pqc_ciphertext


@dataclass
class HybridSignature:
    """Hybrid signature containing both classical and PQC signatures."""

    classical_signature: bytes
    pqc_signature: bytes
    mode: HybridMode

    def to_bytes(self) -> bytes:
        """Serialize hybrid signature."""
        sig1_len = len(self.classical_signature).to_bytes(4, "big")
        return sig1_len + self.classical_signature + self.pqc_signature

    @classmethod
    def from_bytes(cls, data: bytes, mode: HybridMode) -> "HybridSignature":
        """Deserialize hybrid signature."""
        sig1_len = int.from_bytes(data[:4], "big")
        classical_sig = data[4:4 + sig1_len]
        pqc_sig = data[4 + sig1_len:]
        return cls(classical_signature=classical_sig, pqc_signature=pqc_sig, mode=mode)


class HybridKEM:
    """
    Hybrid Key Encapsulation Mechanism.

    Combines ECDH or RSA with ML-KEM for quantum-resistant key establishment.
    The shared secret is derived by combining both classical and PQC secrets.
    """

    def __init__(self, mode: HybridMode = HybridMode.ECDH_MLKEM_768):
        if mode.mode_type != "kem":
            raise ValueError(f"Invalid mode for KEM: {mode}")

        self.mode = mode

        # Initialize PQC KEM
        pqc_level = mode.pqc_algorithm.split("-")[-1]
        self._pqc_kem = create_ml_kem(pqc_level)

        # Classical curve mapping
        self._curve_map = {
            "P-256": ec.SECP256R1(),
            "P-384": ec.SECP384R1(),
            "P-521": ec.SECP521R1(),
        }

    def generate_keypair(self) -> HybridKeyPair:
        """Generate hybrid key pair."""
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography library required for hybrid KEM")

        # Generate classical key
        if self.mode.classical_algorithm.startswith("P-"):
            curve = self._curve_map[self.mode.classical_algorithm]
            classical_private = ec.generate_private_key(curve, default_backend())
            classical_public = classical_private.public_key()

            classical_pub_bytes = classical_public.public_bytes(
                serialization.Encoding.X962,
                serialization.PublicFormat.UncompressedPoint
            )
            classical_priv_bytes = classical_private.private_bytes(
                serialization.Encoding.DER,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption()
            )
        else:
            # RSA
            classical_private = rsa.generate_private_key(
                public_exponent=65537,
                key_size=3072,
                backend=default_backend()
            )
            classical_public = classical_private.public_key()

            classical_pub_bytes = classical_public.public_bytes(
                serialization.Encoding.DER,
                serialization.PublicFormat.SubjectPublicKeyInfo
            )
            classical_priv_bytes = classical_private.private_bytes(
                serialization.Encoding.DER,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption()
            )

        # Generate PQC key
        pqc_keypair = self._pqc_kem.generate_keypair()

        return HybridKeyPair(
            public_key=HybridPublicKey(
                classical_key=classical_pub_bytes,
                pqc_key=pqc_keypair.public_key.to_bytes(),
                mode=self.mode
            ),
            private_key=HybridPrivateKey(
                classical_key=classical_priv_bytes,
                pqc_key=pqc_keypair.private_key.to_bytes(),
                mode=self.mode
            ),
            mode=self.mode,
            _classical_public=classical_public,
            _classical_private=classical_private,
            _pqc_keypair=pqc_keypair,
        )

    def encapsulate(self, public_key: HybridPublicKey) -> HybridEncapsulation:
        """
        Encapsulate shared secret using hybrid scheme.

        Both classical and PQC shared secrets are combined using HKDF.
        """
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography library required for hybrid KEM")

        # Classical encapsulation (ECDH or RSA)
        if self.mode.classical_algorithm.startswith("P-"):
            # ECDH: generate ephemeral key pair
            curve = self._curve_map[self.mode.classical_algorithm]
            ephemeral_private = ec.generate_private_key(curve, default_backend())
            ephemeral_public = ephemeral_private.public_key()

            # Load peer's public key
            peer_public = ec.EllipticCurvePublicKey.from_encoded_point(
                curve, public_key.classical_key
            )

            # Derive shared secret
            classical_shared = ephemeral_private.exchange(ec.ECDH(), peer_public)

            # Ciphertext is ephemeral public key
            classical_ciphertext = ephemeral_public.public_bytes(
                serialization.Encoding.X962,
                serialization.PublicFormat.UncompressedPoint
            )
        else:
            # RSA-KEM: encrypt random value
            peer_public = serialization.load_der_public_key(
                public_key.classical_key, default_backend()
            )
            classical_shared = secrets.token_bytes(32)
            classical_ciphertext = peer_public.encrypt(
                classical_shared,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )

        # PQC encapsulation
        from ai_engine.domains.banking.security.pqc.ml_kem import MLKEMPublicKey
        pqc_public = MLKEMPublicKey.from_bytes(
            public_key.pqc_key,
            self._pqc_kem.level
        )
        pqc_encap = self._pqc_kem.encapsulate(pqc_public)

        # Combine shared secrets using HKDF
        combined_ikm = classical_shared + pqc_encap.shared_secret
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"hybrid-kem-v1",
            info=self.mode.algorithm_name.encode(),
            backend=default_backend()
        )
        combined_shared = hkdf.derive(combined_ikm)

        return HybridEncapsulation(
            classical_ciphertext=classical_ciphertext,
            pqc_ciphertext=pqc_encap.ciphertext,
            combined_shared_secret=combined_shared,
        )

    def decapsulate(
        self,
        encapsulation: HybridEncapsulation,
        private_key: HybridPrivateKey,
    ) -> bytes:
        """
        Decapsulate to recover combined shared secret.
        """
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography library required for hybrid KEM")

        # Classical decapsulation
        classical_private = serialization.load_der_private_key(
            private_key.classical_key, None, default_backend()
        )

        if self.mode.classical_algorithm.startswith("P-"):
            # ECDH
            curve = self._curve_map[self.mode.classical_algorithm]
            ephemeral_public = ec.EllipticCurvePublicKey.from_encoded_point(
                curve, encapsulation.classical_ciphertext
            )
            classical_shared = classical_private.exchange(ec.ECDH(), ephemeral_public)
        else:
            # RSA
            classical_shared = classical_private.decrypt(
                encapsulation.classical_ciphertext,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )

        # PQC decapsulation
        from ai_engine.domains.banking.security.pqc.ml_kem import MLKEMPrivateKey
        pqc_private = MLKEMPrivateKey.from_bytes(
            private_key.pqc_key,
            self._pqc_kem.level
        )
        pqc_shared = self._pqc_kem.decapsulate(
            encapsulation.pqc_ciphertext,
            pqc_private
        )

        # Combine shared secrets
        combined_ikm = classical_shared + pqc_shared
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"hybrid-kem-v1",
            info=self.mode.algorithm_name.encode(),
            backend=default_backend()
        )

        return hkdf.derive(combined_ikm)


class HybridSigner:
    """
    Hybrid Digital Signature Scheme.

    Combines ECDSA or RSA with ML-DSA for quantum-resistant signatures.
    Both signatures must verify for the hybrid signature to be valid.
    """

    def __init__(self, mode: HybridMode = HybridMode.ECDSA_MLDSA_65):
        if mode.mode_type != "sig":
            raise ValueError(f"Invalid mode for signatures: {mode}")

        self.mode = mode

        # Initialize PQC DSA
        pqc_level = mode.pqc_algorithm.split("-")[-1]
        self._pqc_dsa = create_ml_dsa(pqc_level)

        # Classical curve mapping
        self._curve_map = {
            "P-256": ec.SECP256R1(),
            "P-384": ec.SECP384R1(),
            "P-521": ec.SECP521R1(),
        }

    def generate_keypair(self) -> HybridKeyPair:
        """Generate hybrid signing key pair."""
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography library required for hybrid signatures")

        # Generate classical key
        if self.mode.classical_algorithm.startswith("P-"):
            curve = self._curve_map[self.mode.classical_algorithm]
            classical_private = ec.generate_private_key(curve, default_backend())
            classical_public = classical_private.public_key()

            classical_pub_bytes = classical_public.public_bytes(
                serialization.Encoding.X962,
                serialization.PublicFormat.UncompressedPoint
            )
            classical_priv_bytes = classical_private.private_bytes(
                serialization.Encoding.DER,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption()
            )
        else:
            # RSA
            classical_private = rsa.generate_private_key(
                public_exponent=65537,
                key_size=3072,
                backend=default_backend()
            )
            classical_public = classical_private.public_key()

            classical_pub_bytes = classical_public.public_bytes(
                serialization.Encoding.DER,
                serialization.PublicFormat.SubjectPublicKeyInfo
            )
            classical_priv_bytes = classical_private.private_bytes(
                serialization.Encoding.DER,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption()
            )

        # Generate PQC key
        pqc_keypair = self._pqc_dsa.generate_keypair()

        return HybridKeyPair(
            public_key=HybridPublicKey(
                classical_key=classical_pub_bytes,
                pqc_key=pqc_keypair.public_key.to_bytes(),
                mode=self.mode
            ),
            private_key=HybridPrivateKey(
                classical_key=classical_priv_bytes,
                pqc_key=pqc_keypair.private_key.to_bytes(),
                mode=self.mode
            ),
            mode=self.mode,
        )

    def sign(
        self,
        message: bytes,
        private_key: HybridPrivateKey,
        context: bytes = b"",
    ) -> HybridSignature:
        """
        Sign message using both classical and PQC algorithms.

        Both signatures are required for verification.
        """
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography library required for hybrid signatures")

        # Load classical private key
        classical_private = serialization.load_der_private_key(
            private_key.classical_key, None, default_backend()
        )

        # Classical signature
        if self.mode.classical_algorithm.startswith("P-"):
            classical_sig = classical_private.sign(
                message,
                ec.ECDSA(hashes.SHA384())
            )
        else:
            classical_sig = classical_private.sign(
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.AUTO
                ),
                hashes.SHA256()
            )

        # PQC signature
        from ai_engine.domains.banking.security.pqc.ml_dsa import MLDSAPrivateKey
        pqc_private = MLDSAPrivateKey.from_bytes(
            private_key.pqc_key,
            self._pqc_dsa.level
        )
        pqc_sig = self._pqc_dsa.sign(message, pqc_private, context)

        return HybridSignature(
            classical_signature=classical_sig,
            pqc_signature=pqc_sig.to_bytes(),
            mode=self.mode,
        )

    def verify(
        self,
        message: bytes,
        signature: HybridSignature,
        public_key: HybridPublicKey,
        context: bytes = b"",
    ) -> bool:
        """
        Verify hybrid signature.

        BOTH classical and PQC signatures must be valid.
        """
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography library required for hybrid signatures")

        # Verify classical signature
        try:
            if self.mode.classical_algorithm.startswith("P-"):
                curve = self._curve_map[self.mode.classical_algorithm]
                classical_public = ec.EllipticCurvePublicKey.from_encoded_point(
                    curve, public_key.classical_key
                )
                classical_public.verify(
                    signature.classical_signature,
                    message,
                    ec.ECDSA(hashes.SHA384())
                )
            else:
                classical_public = serialization.load_der_public_key(
                    public_key.classical_key, default_backend()
                )
                classical_public.verify(
                    signature.classical_signature,
                    message,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.AUTO
                    ),
                    hashes.SHA256()
                )
            classical_valid = True
        except Exception:
            classical_valid = False

        # Verify PQC signature
        from ai_engine.domains.banking.security.pqc.ml_dsa import (
            MLDSAPublicKey,
            MLDSASignature,
        )
        pqc_public = MLDSAPublicKey.from_bytes(
            public_key.pqc_key,
            self._pqc_dsa.level
        )
        pqc_sig = MLDSASignature.from_bytes(
            signature.pqc_signature,
            self._pqc_dsa.level
        )
        pqc_valid = self._pqc_dsa.verify(message, pqc_sig, pqc_public, context)

        # Both must be valid
        return classical_valid and pqc_valid
