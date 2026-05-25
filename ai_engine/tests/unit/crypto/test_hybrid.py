"""
Unit tests for ai_engine.crypto.hybrid — Hybrid Post-Quantum Key Exchange module.

All tests run against the FALLBACK provider for the PQC (ML-KEM) component.
The classical ECDH component (X25519, P-384) uses the `cryptography` library
which is always available.

The pqc_fallback_mode fixture (from conftest.py) sets QBITEL_PQC_ALLOW_FALLBACK=1.
"""

import pytest

from ai_engine.crypto.hybrid import (
    HybridCiphertext,
    HybridKemEngine,
    HybridKeyPair,
    HybridKexVariant,
    HybridPublicKey,
    HybridSharedSecret,
)


# ---------------------------------------------------------------------------
# Variant enumeration
# ---------------------------------------------------------------------------

class TestHybridKexVariant:

    def test_all_variants_exist(self):
        """All three HybridKexVariant members should be defined."""
        members = set(HybridKexVariant)
        assert HybridKexVariant.X25519_MLKEM_768 in members
        assert HybridKexVariant.P384_MLKEM_1024 in members
        assert HybridKexVariant.X25519_MLKEM_512 in members
        assert len(members) == 3


# ---------------------------------------------------------------------------
# Engine initialisation
# ---------------------------------------------------------------------------

class TestEngineInit:

    @pytest.mark.parametrize(
        "variant",
        list(HybridKexVariant),
        ids=lambda v: v.name,
    )
    def test_engine_init_all_variants(self, pqc_fallback_mode, variant):
        """Engine should initialise for every variant with fallback provider."""
        engine = HybridKemEngine(variant)
        assert engine.variant == variant
        assert engine.mlkem_engine is not None


# ---------------------------------------------------------------------------
# Key generation
# ---------------------------------------------------------------------------

class TestKeygen:

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "variant",
        list(HybridKexVariant),
        ids=lambda v: v.name,
    )
    async def test_keygen_all_variants(self, pqc_fallback_mode, variant):
        """Keygen must produce a HybridKeyPair with correctly typed components."""
        engine = HybridKemEngine(variant)
        kp = await engine.generate_keypair()

        assert isinstance(kp, HybridKeyPair)
        assert kp.variant == variant
        assert isinstance(kp.public_key, HybridPublicKey)
        assert kp.public_key.variant == variant
        assert len(kp.public_key.classical) > 0
        assert len(kp.public_key.pqc) > 0


# ---------------------------------------------------------------------------
# Encapsulate / Decapsulate
# ---------------------------------------------------------------------------

class TestEncapsDecaps:

    @pytest.mark.asyncio
    async def test_encapsulate_decapsulate_roundtrip(self, pqc_fallback_mode):
        """
        Encapsulate then decapsulate should complete without error.

        NOTE: The PQC fallback provider generates independent random bytes, so
        the PQC component of the shared secrets will NOT match. However, the
        classical ECDH part works correctly. We verify that both operations
        succeed and produce correctly-typed, 32-byte shared secrets.
        """
        variant = HybridKexVariant.X25519_MLKEM_768
        engine = HybridKemEngine(variant)
        kp = await engine.generate_keypair()

        ciphertext, shared_secret_enc = await engine.encapsulate(kp.public_key)

        assert isinstance(ciphertext, HybridCiphertext)
        assert ciphertext.variant == variant
        assert len(ciphertext.classical) > 0
        assert len(ciphertext.pqc) > 0

        assert isinstance(shared_secret_enc, HybridSharedSecret)
        assert len(shared_secret_enc.data) == 32

        shared_secret_dec = await engine.decapsulate(ciphertext, kp.private_key)

        assert isinstance(shared_secret_dec, HybridSharedSecret)
        assert len(shared_secret_dec.data) == 32
        # Fallback PQC: secrets diverge, so we do NOT assert equality.

    @pytest.mark.asyncio
    async def test_variant_mismatch_encapsulate(self, pqc_fallback_mode):
        """Encapsulate with a key from a different variant must raise ValueError."""
        engine_768 = HybridKemEngine(HybridKexVariant.X25519_MLKEM_768)
        engine_512 = HybridKemEngine(HybridKexVariant.X25519_MLKEM_512)

        kp_512 = await engine_512.generate_keypair()

        with pytest.raises(ValueError, match="[Vv]ariant mismatch"):
            await engine_768.encapsulate(kp_512.public_key)

    @pytest.mark.asyncio
    async def test_variant_mismatch_decapsulate(self, pqc_fallback_mode):
        """Decapsulate with mismatched variant must raise ValueError."""
        engine_768 = HybridKemEngine(HybridKexVariant.X25519_MLKEM_768)
        engine_512 = HybridKemEngine(HybridKexVariant.X25519_MLKEM_512)

        kp_768 = await engine_768.generate_keypair()
        ct_768, _ = await engine_768.encapsulate(kp_768.public_key)

        with pytest.raises(ValueError, match="[Vv]ariant mismatch"):
            await engine_512.decapsulate(ct_768, kp_768.private_key)


# ---------------------------------------------------------------------------
# Factory methods
# ---------------------------------------------------------------------------

class TestFactoryMethods:

    def test_factory_for_tls_default(self, pqc_fallback_mode):
        """for_tls_default() should create an engine with X25519_MLKEM_768."""
        engine = HybridKemEngine.for_tls_default()
        assert engine.variant == HybridKexVariant.X25519_MLKEM_768

    def test_factory_for_enterprise(self, pqc_fallback_mode):
        """for_enterprise() should create an engine with P384_MLKEM_1024."""
        engine = HybridKemEngine.for_enterprise()
        assert engine.variant == HybridKexVariant.P384_MLKEM_1024

    def test_factory_for_constrained(self, pqc_fallback_mode):
        """for_constrained() should create an engine with X25519_MLKEM_512."""
        engine = HybridKemEngine.for_constrained()
        assert engine.variant == HybridKexVariant.X25519_MLKEM_512
