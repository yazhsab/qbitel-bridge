"""
HSM Type Definitions

Core types, enums, and exceptions for HSM operations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime


class HSMKeyType(Enum):
    """Types of keys supported by HSM."""

    # Symmetric keys
    AES_128 = ("AES-128", 128, "symmetric")
    AES_192 = ("AES-192", 192, "symmetric")
    AES_256 = ("AES-256", 256, "symmetric")
    DES = ("DES", 56, "symmetric")
    DES3 = ("3DES", 168, "symmetric")

    # Traditional asymmetric keys
    RSA_2048 = ("RSA-2048", 2048, "asymmetric")
    RSA_3072 = ("RSA-3072", 3072, "asymmetric")
    RSA_4096 = ("RSA-4096", 4096, "asymmetric")
    EC_P256 = ("EC-P256", 256, "asymmetric")
    EC_P384 = ("EC-P384", 384, "asymmetric")
    EC_P521 = ("EC-P521", 521, "asymmetric")

    # Post-quantum keys (ML-KEM / Kyber)
    ML_KEM_512 = ("ML-KEM-512", 512, "pqc_kem")
    ML_KEM_768 = ("ML-KEM-768", 768, "pqc_kem")
    ML_KEM_1024 = ("ML-KEM-1024", 1024, "pqc_kem")

    # Post-quantum signatures (ML-DSA / Dilithium)
    ML_DSA_44 = ("ML-DSA-44", 44, "pqc_signature")
    ML_DSA_65 = ("ML-DSA-65", 65, "pqc_signature")
    ML_DSA_87 = ("ML-DSA-87", 87, "pqc_signature")

    # Hybrid keys (classical + PQC)
    HYBRID_RSA_ML_KEM = ("Hybrid-RSA-ML-KEM", 0, "hybrid")
    HYBRID_EC_ML_KEM = ("Hybrid-EC-ML-KEM", 0, "hybrid")

    def __init__(self, algorithm_name: str, key_size: int, category: str):
        self.algorithm_name = algorithm_name
        self.key_size = key_size
        self.category = category

    @property
    def is_symmetric(self) -> bool:
        return self.category == "symmetric"

    @property
    def is_asymmetric(self) -> bool:
        return self.category == "asymmetric"

    @property
    def is_pqc(self) -> bool:
        return self.category in ("pqc_kem", "pqc_signature")

    @property
    def is_hybrid(self) -> bool:
        return self.category == "hybrid"


class HSMAlgorithm(Enum):
    """Cryptographic algorithms supported by HSM."""

    # Encryption
    AES_ECB = ("AES-ECB", "encryption")
    AES_CBC = ("AES-CBC", "encryption")
    AES_GCM = ("AES-GCM", "encryption")
    AES_CTR = ("AES-CTR", "encryption")
    DES3_CBC = ("3DES-CBC", "encryption")
    RSA_OAEP = ("RSA-OAEP", "encryption")
    RSA_PKCS1 = ("RSA-PKCS1", "encryption")

    # Signatures
    RSA_PSS = ("RSA-PSS", "signature")
    RSA_PKCS1_SIGN = ("RSA-PKCS1-SIGN", "signature")
    ECDSA = ("ECDSA", "signature")
    ED25519 = ("Ed25519", "signature")
    ML_DSA = ("ML-DSA", "signature")

    # Key encapsulation
    ML_KEM_ENCAP = ("ML-KEM-ENCAP", "kem")
    ML_KEM_DECAP = ("ML-KEM-DECAP", "kem")

    # Hash
    SHA256 = ("SHA-256", "hash")
    SHA384 = ("SHA-384", "hash")
    SHA512 = ("SHA-512", "hash")
    SHA3_256 = ("SHA3-256", "hash")
    SHA3_512 = ("SHA3-512", "hash")

    # MAC
    HMAC_SHA256 = ("HMAC-SHA256", "mac")
    HMAC_SHA384 = ("HMAC-SHA384", "mac")
    CMAC = ("CMAC", "mac")

    def __init__(self, algorithm_name: str, operation_type: str):
        self.algorithm_name = algorithm_name
        self.operation_type = operation_type


@dataclass
class HSMConfig:
    """Configuration for HSM connection."""

    provider_type: str  # "softhsm", "thales", "futurex", "utimaco"
    slot_id: int = 0
    pin: str = ""

    # Connection settings
    host: Optional[str] = None
    port: Optional[int] = None

    # Library paths
    library_path: Optional[str] = None
    config_path: Optional[str] = None

    # TLS settings
    use_tls: bool = True
    ca_cert_path: Optional[str] = None
    client_cert_path: Optional[str] = None
    client_key_path: Optional[str] = None

    # Timeout settings
    connection_timeout: int = 30
    operation_timeout: int = 60

    # HA settings
    failover_hosts: List[str] = field(default_factory=list)
    load_balance: bool = False

    # Logging
    audit_logging: bool = True

    # PQC support
    enable_pqc: bool = True
    pqc_algorithms: List[str] = field(default_factory=lambda: ["ML-KEM-768", "ML-DSA-65"])

    def validate(self) -> List[str]:
        """Validate configuration."""
        errors = []

        if not self.provider_type:
            errors.append("provider_type is required")

        if self.provider_type not in ("softhsm", "thales", "futurex", "utimaco"):
            errors.append(f"Unknown provider_type: {self.provider_type}")

        if self.provider_type != "softhsm" and not self.host:
            errors.append("host is required for network HSM")

        return errors


@dataclass
class HSMKeyHandle:
    """Handle to a key stored in HSM."""

    key_id: str
    key_type: HSMKeyType
    label: str

    # Key metadata
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    # Attributes
    extractable: bool = False
    sensitive: bool = True
    modifiable: bool = False

    # Usage flags
    can_encrypt: bool = False
    can_decrypt: bool = False
    can_sign: bool = False
    can_verify: bool = False
    can_wrap: bool = False
    can_unwrap: bool = False
    can_derive: bool = False

    # Additional metadata
    owner: Optional[str] = None
    purpose: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """Check if key is currently active."""
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key_id": self.key_id,
            "key_type": self.key_type.algorithm_name,
            "label": self.label,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "extractable": self.extractable,
            "sensitive": self.sensitive,
            "can_encrypt": self.can_encrypt,
            "can_decrypt": self.can_decrypt,
            "can_sign": self.can_sign,
            "can_verify": self.can_verify,
            "can_wrap": self.can_wrap,
            "can_unwrap": self.can_unwrap,
            "owner": self.owner,
            "purpose": self.purpose,
        }


class HSMError(Exception):
    """Base exception for HSM operations."""

    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message)
        self.error_code = error_code


class HSMConnectionError(HSMError):
    """Exception for HSM connection failures."""
    pass


class HSMAuthenticationError(HSMError):
    """Exception for HSM authentication failures."""
    pass


class HSMOperationError(HSMError):
    """Exception for HSM operation failures."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        operation: Optional[str] = None,
    ):
        super().__init__(message, error_code)
        self.operation = operation


class HSMKeyNotFoundError(HSMError):
    """Exception when key is not found in HSM."""

    def __init__(self, key_id: str):
        super().__init__(f"Key not found: {key_id}")
        self.key_id = key_id


class HSMKeyExistsError(HSMError):
    """Exception when key already exists."""

    def __init__(self, key_id: str):
        super().__init__(f"Key already exists: {key_id}")
        self.key_id = key_id


class HSMCapabilityError(HSMError):
    """Exception when HSM doesn't support requested capability."""

    def __init__(self, capability: str):
        super().__init__(f"HSM does not support: {capability}")
        self.capability = capability
