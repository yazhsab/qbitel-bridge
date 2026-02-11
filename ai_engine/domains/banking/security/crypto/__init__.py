"""
Cryptographic Verification Module

Production-ready cryptographic verification for banking protocols:
- EMV cryptogram verification (ARQC, AAC, TC, ARPC)
- 3D Secure CAVV/AAV verification
- Digital signature verification (Classical + PQC)
- MAC/HMAC verification

All verification operations use:
- Constant-time comparisons to prevent timing attacks
- Proper key derivation per EMV/3DS specifications
- Support for post-quantum cryptography
"""

from ai_engine.domains.banking.security.crypto.verification import (
    # Result types
    VerificationStatus,
    VerificationResult,
    # EMV
    EMVCryptogramVerifier,
    EMVCryptogramData,
    CryptogramType,
    KeyDerivationType,
    # 3DS
    ThreeDSVerifier,
    ThreeDSVerificationData,
    # Signatures
    SignatureVerifier,
    SignatureVerificationData,
    # MAC
    MACVerifier,
    # Utilities
    secure_compare,
    # Convenience functions
    verify_emv_arqc,
    verify_cavv,
    verify_signature,
)

__all__ = [
    # Result types
    "VerificationStatus",
    "VerificationResult",
    # EMV
    "EMVCryptogramVerifier",
    "EMVCryptogramData",
    "CryptogramType",
    "KeyDerivationType",
    # 3DS
    "ThreeDSVerifier",
    "ThreeDSVerificationData",
    # Signatures
    "SignatureVerifier",
    "SignatureVerificationData",
    # MAC
    "MACVerifier",
    # Utilities
    "secure_compare",
    # Convenience functions
    "verify_emv_arqc",
    "verify_cavv",
    "verify_signature",
]
