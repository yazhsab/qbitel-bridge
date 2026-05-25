"""
Unit tests for ai_engine.crypto.falcon — Falcon digital signature module.

All tests run against the FALLBACK provider (no real PQC crypto libs required).
The pqc_fallback_mode fixture (from conftest.py) sets QBITEL_PQC_ALLOW_FALLBACK=1.
"""

import pytest

from ai_engine.crypto.falcon import (
    FalconBatchVerifier,
    FalconEngine,
    FalconKeyPair,
    FalconPrivateKey,
    FalconPublicKey,
    FalconSecurityLevel,
    FalconSignature,
)
from ai_engine.crypto.mlkem import PQCProviderUnavailableError


# ---------------------------------------------------------------------------
# Security level property tests
# ---------------------------------------------------------------------------

class TestSecurityLevelProperties:
    """Verify size/level constants for both Falcon parameter sets."""

    EXPECTED = {
        FalconSecurityLevel.FALCON_512: {
            "public_key_size": 897,
            "private_key_size": 1281,
            "signature_size_max": 690,
            "signature_size_typical": 666,
            "nist_level": 1,
            "size_advantage_vs_dilithium": 2420 / 666,
        },
        FalconSecurityLevel.FALCON_1024: {
            "public_key_size": 1793,
            "private_key_size": 2305,
            "signature_size_max": 1330,
            "signature_size_typical": 1280,
            "nist_level": 5,
            "size_advantage_vs_dilithium": 4595 / 1280,
        },
    }

    @pytest.mark.parametrize("level", list(EXPECTED.keys()), ids=lambda l: l.name)
    def test_security_level_properties(self, level):
        expected = self.EXPECTED[level]
        assert level.public_key_size == expected["public_key_size"]
        assert level.private_key_size == expected["private_key_size"]
        assert level.signature_size_max == expected["signature_size_max"]
        assert level.signature_size_typical == expected["signature_size_typical"]
        assert level.nist_level == expected["nist_level"]
        assert level.size_advantage_vs_dilithium == pytest.approx(
            expected["size_advantage_vs_dilithium"]
        )


# ---------------------------------------------------------------------------
# Engine initialisation
# ---------------------------------------------------------------------------

class TestEngineInit:

    def test_engine_init_fallback(self, pqc_fallback_mode):
        """Engine should initialise with the fallback provider when no real libs exist."""
        engine = FalconEngine(FalconSecurityLevel.FALCON_512, strict_mode=False)
        assert engine.provider is not None
        assert engine.level == FalconSecurityLevel.FALCON_512

    def test_engine_strict_raises(self, pqc_fallback_mode):
        """With strict_mode=True and no real providers, PQCProviderUnavailableError is raised."""
        with pytest.raises(PQCProviderUnavailableError):
            FalconEngine(FalconSecurityLevel.FALCON_512, strict_mode=True)


# ---------------------------------------------------------------------------
# Key generation
# ---------------------------------------------------------------------------

class TestKeygen:

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "level",
        [FalconSecurityLevel.FALCON_512, FalconSecurityLevel.FALCON_1024],
        ids=lambda l: l.name,
    )
    async def test_keygen_both_levels(self, pqc_fallback_mode, level):
        """Keygen must produce keys whose sizes match the level specification."""
        engine = FalconEngine(level, strict_mode=False)
        kp = await engine.generate_keypair()

        assert isinstance(kp, FalconKeyPair)
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
        Sign then verify should complete without error.

        NOTE: The fallback provider generates random bytes for signatures, so
        verify always returns True. We only verify that the operations succeed
        and produce correctly-typed outputs.
        """
        engine = FalconEngine(FalconSecurityLevel.FALCON_512, strict_mode=False)
        kp = await engine.generate_keypair()

        message = b"test message for Falcon signing"
        signature = await engine.sign(message, kp.private_key)

        assert isinstance(signature, FalconSignature)
        assert signature.level == FalconSecurityLevel.FALCON_512
        assert signature.size == len(signature.data)

        valid = await engine.verify(message, signature, kp.public_key)
        # Fallback: verify always returns True
        assert valid is True

    @pytest.mark.asyncio
    async def test_sign_level_mismatch(self, pqc_fallback_mode):
        """Sign with a private key from a different level must raise ValueError."""
        engine_512 = FalconEngine(FalconSecurityLevel.FALCON_512, strict_mode=False)
        engine_1024 = FalconEngine(FalconSecurityLevel.FALCON_1024, strict_mode=False)

        kp_1024 = await engine_1024.generate_keypair()

        with pytest.raises(ValueError, match="[Ss]ecurity level mismatch"):
            await engine_512.sign(b"test", kp_1024.private_key)

    @pytest.mark.asyncio
    async def test_verify_level_mismatch(self, pqc_fallback_mode):
        """Verify with mismatched levels must raise ValueError."""
        engine_512 = FalconEngine(FalconSecurityLevel.FALCON_512, strict_mode=False)
        engine_1024 = FalconEngine(FalconSecurityLevel.FALCON_1024, strict_mode=False)

        kp_512 = await engine_512.generate_keypair()
        sig_512 = await engine_512.sign(b"test", kp_512.private_key)

        kp_1024 = await engine_1024.generate_keypair()

        # Signature level does not match engine level
        with pytest.raises(ValueError, match="[Ss]ecurity level mismatch"):
            await engine_1024.verify(b"test", sig_512, kp_1024.public_key)


# ---------------------------------------------------------------------------
# Batch verifier
# ---------------------------------------------------------------------------

class TestBatchVerifier:

    @pytest.mark.asyncio
    async def test_batch_verifier_basic(self, pqc_fallback_mode):
        """Add items and verify_batch returns a list of bools."""
        engine = FalconEngine(FalconSecurityLevel.FALCON_512, strict_mode=False)
        kp = await engine.generate_keypair()
        sig = await engine.sign(b"msg1", kp.private_key)

        verifier = FalconBatchVerifier(
            level=FalconSecurityLevel.FALCON_512, batch_size=8,
        )
        verifier.add(b"msg1", sig, kp.public_key)

        results = await verifier.verify_batch()
        assert isinstance(results, list)
        assert len(results) == 1
        assert all(isinstance(r, bool) for r in results)

    def test_batch_verifier_is_ready(self, pqc_fallback_mode):
        """Batch with batch_size=2: not ready after 1 add, ready after 2."""
        verifier = FalconBatchVerifier(
            level=FalconSecurityLevel.FALCON_512, batch_size=2,
        )

        dummy_sig = FalconSignature(FalconSecurityLevel.FALCON_512, b"\x00" * 666)
        dummy_pk = FalconPublicKey(FalconSecurityLevel.FALCON_512, b"\x00" * 897)

        verifier.add(b"msg1", dummy_sig, dummy_pk)
        assert not verifier.is_ready()

        verifier.add(b"msg2", dummy_sig, dummy_pk)
        assert verifier.is_ready()

    @pytest.mark.asyncio
    async def test_batch_verifier_empty(self, pqc_fallback_mode):
        """verify_batch on an empty verifier should return []."""
        verifier = FalconBatchVerifier(
            level=FalconSecurityLevel.FALCON_512, batch_size=8,
        )
        results = await verifier.verify_batch()
        assert results == []

    def test_batch_verifier_level_mismatch(self, pqc_fallback_mode):
        """Adding an item with the wrong level must raise ValueError."""
        verifier = FalconBatchVerifier(
            level=FalconSecurityLevel.FALCON_512, batch_size=8,
        )

        wrong_sig = FalconSignature(FalconSecurityLevel.FALCON_1024, b"\x00" * 1280)
        wrong_pk = FalconPublicKey(FalconSecurityLevel.FALCON_1024, b"\x00" * 1793)

        with pytest.raises(ValueError, match="[Ss]ecurity level mismatch"):
            verifier.add(b"msg", wrong_sig, wrong_pk)


# ---------------------------------------------------------------------------
# Factory methods
# ---------------------------------------------------------------------------

class TestFactoryMethods:

    def test_factory_bandwidth_constrained(self, pqc_fallback_mode):
        """for_bandwidth_constrained() should create an engine at FALCON_512."""
        engine = FalconEngine.for_bandwidth_constrained()
        assert engine.level == FalconSecurityLevel.FALCON_512

    def test_factory_maximum_security(self, pqc_fallback_mode):
        """for_maximum_security() should create an engine at FALCON_1024."""
        engine = FalconEngine.for_maximum_security()
        assert engine.level == FalconSecurityLevel.FALCON_1024

    def test_factory_automotive_v2x(self, pqc_fallback_mode):
        """for_automotive_v2x() should return a verifier with batch_size=64."""
        verifier = FalconBatchVerifier.for_automotive_v2x()
        assert isinstance(verifier, FalconBatchVerifier)
        assert verifier.batch_size == 64
        assert verifier.level == FalconSecurityLevel.FALCON_512
