"""
Unit tests for ai_engine.crypto.dilithium — ML-DSA / Dilithium (FIPS 204) module.

All tests run against the FALLBACK provider (no real PQC crypto libs required).
The pqc_fallback_mode fixture (from conftest.py) sets QBITEL_PQC_ALLOW_FALLBACK=1.
"""

import pytest

from ai_engine.crypto.dilithium import (
    DilithiumEngine,
    DilithiumKeyPair,
    DilithiumPrivateKey,
    DilithiumPublicKey,
    DilithiumSecurityLevel,
    DilithiumSignature,
)
from ai_engine.crypto.mlkem import PQCProviderUnavailableError


# ---------------------------------------------------------------------------
# Security level property tests
# ---------------------------------------------------------------------------

class TestSecurityLevelProperties:
    """Verify size, NIST level, and NIST name for all six enum members."""

    EXPECTED = {
        DilithiumSecurityLevel.DILITHIUM_2: {
            "public_key_size": 1312,
            "private_key_size": 2528,
            "signature_size": 2420,
            "nist_level": 2,
            "nist_name": "ML-DSA-44",
        },
        DilithiumSecurityLevel.MLDSA_44: {
            "public_key_size": 1312,
            "private_key_size": 2528,
            "signature_size": 2420,
            "nist_level": 2,
            "nist_name": "ML-DSA-44",
        },
        DilithiumSecurityLevel.DILITHIUM_3: {
            "public_key_size": 1952,
            "private_key_size": 4000,
            "signature_size": 3293,
            "nist_level": 3,
            "nist_name": "ML-DSA-65",
        },
        DilithiumSecurityLevel.MLDSA_65: {
            "public_key_size": 1952,
            "private_key_size": 4000,
            "signature_size": 3293,
            "nist_level": 3,
            "nist_name": "ML-DSA-65",
        },
        DilithiumSecurityLevel.DILITHIUM_5: {
            "public_key_size": 2592,
            "private_key_size": 4864,
            "signature_size": 4595,
            "nist_level": 5,
            "nist_name": "ML-DSA-87",
        },
        DilithiumSecurityLevel.MLDSA_87: {
            "public_key_size": 2592,
            "private_key_size": 4864,
            "signature_size": 4595,
            "nist_level": 5,
            "nist_name": "ML-DSA-87",
        },
    }

    @pytest.mark.parametrize("level", list(EXPECTED.keys()), ids=lambda l: l.name)
    def test_security_level_properties(self, level):
        expected = self.EXPECTED[level]
        assert level.public_key_size == expected["public_key_size"]
        assert level.private_key_size == expected["private_key_size"]
        assert level.signature_size == expected["signature_size"]
        assert level.nist_level == expected["nist_level"]
        assert level.nist_name == expected["nist_name"]


class TestNistNameMapping:
    """Verify the legacy → NIST name mapping is correct."""

    @pytest.mark.parametrize(
        "level, expected_name",
        [
            (DilithiumSecurityLevel.DILITHIUM_2, "ML-DSA-44"),
            (DilithiumSecurityLevel.DILITHIUM_3, "ML-DSA-65"),
            (DilithiumSecurityLevel.DILITHIUM_5, "ML-DSA-87"),
            (DilithiumSecurityLevel.MLDSA_44, "ML-DSA-44"),
            (DilithiumSecurityLevel.MLDSA_65, "ML-DSA-65"),
            (DilithiumSecurityLevel.MLDSA_87, "ML-DSA-87"),
        ],
        ids=lambda v: v if isinstance(v, str) else v.name,
    )
    def test_nist_name_mapping(self, level, expected_name):
        assert level.nist_name == expected_name


# ---------------------------------------------------------------------------
# Engine initialisation
# ---------------------------------------------------------------------------

class TestEngineInit:

    def test_engine_init_fallback(self, pqc_fallback_mode):
        """Engine should initialise with fallback when no real libs exist."""
        engine = DilithiumEngine(DilithiumSecurityLevel.DILITHIUM_3, strict_mode=False)
        assert engine.provider is not None
        assert engine.level == DilithiumSecurityLevel.DILITHIUM_3

    def test_engine_strict_raises(self, pqc_fallback_mode):
        """With strict_mode=True and no real providers, PQCProviderUnavailableError is raised."""
        with pytest.raises(PQCProviderUnavailableError):
            DilithiumEngine(DilithiumSecurityLevel.DILITHIUM_3, strict_mode=True)


# ---------------------------------------------------------------------------
# Key generation
# ---------------------------------------------------------------------------

class TestKeygen:

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "level",
        [
            DilithiumSecurityLevel.DILITHIUM_2,
            DilithiumSecurityLevel.DILITHIUM_3,
            DilithiumSecurityLevel.DILITHIUM_5,
        ],
        ids=lambda l: l.name,
    )
    async def test_keygen_all_levels(self, pqc_fallback_mode, level):
        """Keygen must produce keys whose sizes match the level specification."""
        engine = DilithiumEngine(level, strict_mode=False)
        kp = await engine.generate_keypair()

        assert isinstance(kp, DilithiumKeyPair)
        assert kp.level == level
        assert len(kp.public_key.data) == level.public_key_size
        assert len(kp.private_key.data) == level.private_key_size


# ---------------------------------------------------------------------------
# Sign / Verify
# ---------------------------------------------------------------------------

class TestSignVerify:

    @pytest.mark.asyncio
    async def test_sign_verify_roundtrip(self, pqc_fallback_mode):
        """
        Sign a message and then verify it.

        NOTE: The fallback verify() always returns True regardless of inputs,
        so this test only confirms that the full sign-then-verify path
        executes without error and returns True.
        """
        level = DilithiumSecurityLevel.DILITHIUM_3
        engine = DilithiumEngine(level, strict_mode=False)
        kp = await engine.generate_keypair()

        message = b"QBITEL Bridge quantum-safe test message"
        signature = await engine.sign(message, kp.private_key)

        assert isinstance(signature, DilithiumSignature)
        assert signature.level == level
        assert len(signature.data) == level.signature_size

        valid = await engine.verify(message, signature, kp.public_key)
        assert valid is True

    @pytest.mark.asyncio
    async def test_verify_wrong_message(self, pqc_fallback_mode):
        """
        Sign message_a, verify with message_b.

        In fallback mode the verify() always returns True because it has no
        real cryptographic verification logic. This test documents that
        behaviour explicitly so it is not mistaken for a real security
        guarantee.
        """
        level = DilithiumSecurityLevel.DILITHIUM_2
        engine = DilithiumEngine(level, strict_mode=False)
        kp = await engine.generate_keypair()

        message_a = b"original message"
        message_b = b"tampered message"

        signature = await engine.sign(message_a, kp.private_key)

        # Fallback: verify always returns True, even for the wrong message.
        valid = await engine.verify(message_b, signature, kp.public_key)
        assert valid is True, (
            "Fallback verify() is expected to return True unconditionally. "
            "This does NOT indicate real signature validity."
        )


# ---------------------------------------------------------------------------
# Signature dataclass
# ---------------------------------------------------------------------------

class TestSignatureDataclass:

    @pytest.mark.asyncio
    async def test_signature_size_property(self, pqc_fallback_mode):
        """DilithiumSignature.size must equal len(data)."""
        engine = DilithiumEngine(DilithiumSecurityLevel.DILITHIUM_3, strict_mode=False)
        kp = await engine.generate_keypair()
        sig = await engine.sign(b"test", kp.private_key)

        assert sig.size == len(sig.data)
        assert sig.size == DilithiumSecurityLevel.DILITHIUM_3.signature_size


# ---------------------------------------------------------------------------
# Factory methods
# ---------------------------------------------------------------------------

class TestFactoryMethods:

    def test_factory_for_enterprise(self, pqc_fallback_mode):
        """for_enterprise() should create an engine at DILITHIUM_3."""
        engine = DilithiumEngine.for_enterprise()
        assert engine.level == DilithiumSecurityLevel.DILITHIUM_3

    def test_factory_for_maximum_security(self, pqc_fallback_mode):
        """for_maximum_security() should create an engine at DILITHIUM_5."""
        engine = DilithiumEngine.for_maximum_security()
        assert engine.level == DilithiumSecurityLevel.DILITHIUM_5

    def test_factory_for_government(self, pqc_fallback_mode):
        """for_government() should create an engine at MLDSA_87."""
        engine = DilithiumEngine.for_government()
        assert engine.level == DilithiumSecurityLevel.MLDSA_87
