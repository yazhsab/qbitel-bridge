"""
Unit tests for ai_engine.crypto.mlkem — ML-KEM (FIPS 203) module.

All tests run against the FALLBACK provider (no real PQC crypto libs required).
The pqc_fallback_mode fixture (from conftest.py) sets QBITEL_PQC_ALLOW_FALLBACK=1.
"""

import pytest

from ai_engine.crypto.mlkem import (
    MlKemCiphertext,
    MlKemEngine,
    MlKemKeyPair,
    MlKemPrivateKey,
    MlKemPublicKey,
    MlKemSecurityLevel,
    MlKemSharedSecret,
    PQCProviderUnavailableError,
)


# ---------------------------------------------------------------------------
# Security level property tests
# ---------------------------------------------------------------------------

class TestSecurityLevelProperties:
    """Verify size/level constants for all three ML-KEM parameter sets."""

    EXPECTED = {
        MlKemSecurityLevel.MLKEM_512: {
            "public_key_size": 800,
            "private_key_size": 1632,
            "ciphertext_size": 768,
            "shared_secret_size": 32,
            "nist_level": 1,
        },
        MlKemSecurityLevel.MLKEM_768: {
            "public_key_size": 1184,
            "private_key_size": 2400,
            "ciphertext_size": 1088,
            "shared_secret_size": 32,
            "nist_level": 3,
        },
        MlKemSecurityLevel.MLKEM_1024: {
            "public_key_size": 1568,
            "private_key_size": 3168,
            "ciphertext_size": 1568,
            "shared_secret_size": 32,
            "nist_level": 5,
        },
    }

    @pytest.mark.parametrize("level", list(EXPECTED.keys()), ids=lambda l: l.name)
    def test_security_level_properties(self, level):
        expected = self.EXPECTED[level]
        assert level.public_key_size == expected["public_key_size"]
        assert level.private_key_size == expected["private_key_size"]
        assert level.ciphertext_size == expected["ciphertext_size"]
        assert level.shared_secret_size == expected["shared_secret_size"]
        assert level.nist_level == expected["nist_level"]


# ---------------------------------------------------------------------------
# Engine initialisation
# ---------------------------------------------------------------------------

class TestEngineInit:

    def test_engine_init_fallback_mode(self, pqc_fallback_mode):
        """Engine should initialise with the fallback provider when no real libs exist."""
        engine = MlKemEngine(MlKemSecurityLevel.MLKEM_768, strict_mode=False)
        assert engine.provider is not None
        assert engine.level == MlKemSecurityLevel.MLKEM_768

    def test_engine_init_strict_mode_raises(self, pqc_fallback_mode):
        """With strict_mode=True and no real providers, PQCProviderUnavailableError is raised."""
        with pytest.raises(PQCProviderUnavailableError):
            MlKemEngine(MlKemSecurityLevel.MLKEM_768, strict_mode=True)


# ---------------------------------------------------------------------------
# Key generation
# ---------------------------------------------------------------------------

class TestKeygen:

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "level",
        [MlKemSecurityLevel.MLKEM_512, MlKemSecurityLevel.MLKEM_768, MlKemSecurityLevel.MLKEM_1024],
        ids=lambda l: l.name,
    )
    async def test_keygen_all_levels(self, pqc_fallback_mode, level):
        """Keygen must produce keys whose sizes match the level specification."""
        engine = MlKemEngine(level, strict_mode=False)
        kp = await engine.generate_keypair()

        assert isinstance(kp, MlKemKeyPair)
        assert kp.level == level
        assert len(kp.public_key.data) == level.public_key_size
        assert len(kp.private_key.data) == level.private_key_size


# ---------------------------------------------------------------------------
# Encapsulate / Decapsulate
# ---------------------------------------------------------------------------

class TestEncapsDecaps:

    @pytest.mark.asyncio
    async def test_encapsulate_decapsulate_roundtrip(self, pqc_fallback_mode):
        """
        Encapsulate then decapsulate should complete without error.

        NOTE: The fallback provider generates independent random bytes for each
        operation, so the shared secrets from encapsulate and decapsulate will
        NOT match.  We only verify that the operations succeed and produce
        correctly-sized outputs.
        """
        level = MlKemSecurityLevel.MLKEM_768
        engine = MlKemEngine(level, strict_mode=False)
        kp = await engine.generate_keypair()

        ciphertext, shared_secret_enc = await engine.encapsulate(kp.public_key)

        assert isinstance(ciphertext, MlKemCiphertext)
        assert len(ciphertext.data) == level.ciphertext_size
        assert isinstance(shared_secret_enc, MlKemSharedSecret)
        assert len(shared_secret_enc.data) == 32

        shared_secret_dec = await engine.decapsulate(ciphertext, kp.private_key)

        assert isinstance(shared_secret_dec, MlKemSharedSecret)
        assert len(shared_secret_dec.data) == 32
        # Fallback: secrets are random, so they will differ.

    @pytest.mark.asyncio
    async def test_encapsulate_level_mismatch(self, pqc_fallback_mode):
        """Encapsulate with a key from a different level must raise ValueError."""
        engine_768 = MlKemEngine(MlKemSecurityLevel.MLKEM_768, strict_mode=False)
        engine_512 = MlKemEngine(MlKemSecurityLevel.MLKEM_512, strict_mode=False)

        kp_512 = await engine_512.generate_keypair()

        with pytest.raises(ValueError, match="[Ss]ecurity level mismatch"):
            await engine_768.encapsulate(kp_512.public_key)

    @pytest.mark.asyncio
    async def test_decapsulate_level_mismatch(self, pqc_fallback_mode):
        """Decapsulate with mismatched level must raise ValueError."""
        engine_768 = MlKemEngine(MlKemSecurityLevel.MLKEM_768, strict_mode=False)
        engine_512 = MlKemEngine(MlKemSecurityLevel.MLKEM_512, strict_mode=False)

        kp_768 = await engine_768.generate_keypair()
        ct_768, _ = await engine_768.encapsulate(kp_768.public_key)

        kp_512 = await engine_512.generate_keypair()

        with pytest.raises(ValueError, match="[Ss]ecurity level mismatch"):
            await engine_512.decapsulate(ct_768, kp_512.private_key)


# ---------------------------------------------------------------------------
# Dataclass validation
# ---------------------------------------------------------------------------

class TestDataclassValidation:

    def test_public_key_invalid_size(self):
        """MlKemPublicKey must reject data whose length doesn't match the level."""
        with pytest.raises(ValueError, match="Invalid public key size"):
            MlKemPublicKey(MlKemSecurityLevel.MLKEM_768, b"\x00" * 100)

    def test_shared_secret_invalid_size(self):
        """MlKemSharedSecret must reject data that isn't exactly 32 bytes."""
        with pytest.raises(ValueError, match="Invalid shared secret size"):
            MlKemSharedSecret(b"\x00" * 16)

        with pytest.raises(ValueError, match="Invalid shared secret size"):
            MlKemSharedSecret(b"\x00" * 64)


# ---------------------------------------------------------------------------
# Factory methods
# ---------------------------------------------------------------------------

class TestFactoryMethods:

    def test_factory_for_tls_hybrid(self, pqc_fallback_mode):
        """for_tls_hybrid() should create an engine at MLKEM_768."""
        engine = MlKemEngine.for_tls_hybrid()
        assert engine.level == MlKemSecurityLevel.MLKEM_768

    def test_factory_for_constrained(self, pqc_fallback_mode):
        """for_constrained_devices() should create an engine at MLKEM_512."""
        engine = MlKemEngine.for_constrained_devices()
        assert engine.level == MlKemSecurityLevel.MLKEM_512

    def test_factory_for_maximum_security(self, pqc_fallback_mode):
        """for_maximum_security() should create an engine at MLKEM_1024."""
        engine = MlKemEngine.for_maximum_security()
        assert engine.level == MlKemSecurityLevel.MLKEM_1024
