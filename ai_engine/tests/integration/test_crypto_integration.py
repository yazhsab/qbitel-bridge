"""Integration tests for the PQC crypto pipeline.

Tests the full flow through the unified PQC engine, provider registry,
and domain-specific algorithm selection.
"""

import pytest

from ai_engine.crypto.providers import ProviderRegistry, ProviderTier


@pytest.fixture(autouse=True)
def _reset_registry():
    ProviderRegistry.reset()
    yield
    ProviderRegistry.reset()


@pytest.fixture
def _fallback_mode(monkeypatch):
    monkeypatch.setenv("QBITEL_PQC_ALLOW_FALLBACK", "1")
    import ai_engine.crypto.providers as _providers
    import ai_engine.crypto.mlkem as _mlkem
    import ai_engine.crypto.falcon as _falcon
    monkeypatch.setattr(_providers, "_GLOBAL_STRICT_MODE", False)
    monkeypatch.setattr(_mlkem, "_GLOBAL_STRICT_MODE", False)
    monkeypatch.setattr(_falcon, "_GLOBAL_STRICT_MODE", False)
    ProviderRegistry.reset()


class TestProviderRegistryConsistency:
    """Verify provider registry returns consistent results across modules."""

    def test_registry_is_singleton(self):
        r1 = ProviderRegistry()
        r2 = ProviderRegistry()
        assert r1 is r2

    def test_status_contains_all_providers(self):
        status = ProviderRegistry().status()
        expected_keys = {"liboqs", "pqcrypto", "kyber-py", "dilithium-py"}
        assert set(status.keys()) == expected_keys

    def test_all_providers_have_tier(self):
        status = ProviderRegistry().status()
        valid_tiers = {t.value for t in ProviderTier}
        for provider, info in status.items():
            assert info["tier"] in valid_tiers


class TestUnifiedEngineFlow:
    """Test PQCEngine through domain profiles."""

    @pytest.mark.asyncio
    async def test_enterprise_engine_keygen(self, _fallback_mode):
        from ai_engine.crypto.pqc_unified import PQCEngine, DomainProfile
        engine = PQCEngine(DomainProfile.ENTERPRISE)
        assert engine.domain == DomainProfile.ENTERPRISE

    @pytest.mark.asyncio
    async def test_healthcare_engine_keygen(self, _fallback_mode):
        from ai_engine.crypto.pqc_unified import PQCEngine, DomainProfile
        engine = PQCEngine(DomainProfile.HEALTHCARE)
        assert engine.domain == DomainProfile.HEALTHCARE

    @pytest.mark.asyncio
    async def test_cnsa2_engine_keygen(self, _fallback_mode):
        from ai_engine.crypto.pqc_unified import PQCEngine, DomainProfile
        engine = PQCEngine(DomainProfile.CNSA2_DEFENSE)
        assert engine.domain == DomainProfile.CNSA2_DEFENSE


class TestDomainSpecificAlgorithmSelection:
    """Verify each GA domain selects appropriate algorithms."""

    def test_banking_enterprise_algorithms(self):
        from ai_engine.crypto.pqc_unified import DomainProfile, PQCAlgorithm
        assert DomainProfile.ENTERPRISE.default_kem == PQCAlgorithm.MLKEM_768
        assert DomainProfile.ENTERPRISE.default_signature == PQCAlgorithm.DILITHIUM_3

    def test_healthcare_constrained_algorithms(self):
        from ai_engine.crypto.pqc_unified import DomainProfile, PQCAlgorithm
        assert DomainProfile.HEALTHCARE.default_kem == PQCAlgorithm.MLKEM_512
        assert DomainProfile.HEALTHCARE.default_signature == PQCAlgorithm.FALCON_512

    def test_automotive_preview_algorithms(self):
        from ai_engine.crypto.pqc_unified import DomainProfile, PQCAlgorithm
        assert DomainProfile.AUTOMOTIVE.default_signature == PQCAlgorithm.FALCON_512

    def test_cnsa2_mandated_algorithms(self):
        from ai_engine.crypto.pqc_unified import DomainProfile, PQCAlgorithm
        assert DomainProfile.CNSA2_DEFENSE.default_kem == PQCAlgorithm.MLKEM_1024
        assert DomainProfile.CNSA2_DEFENSE.default_signature == PQCAlgorithm.MLDSA_87


class TestHybridKemFullFlow:
    """Test Hybrid KEM end-to-end with classical + PQC."""

    @pytest.mark.asyncio
    async def test_hybrid_keygen_encapsulate_decapsulate(self, _fallback_mode):
        from ai_engine.crypto.hybrid import HybridKemEngine, HybridKexVariant
        engine = HybridKemEngine(HybridKexVariant.X25519_MLKEM_768)
        keypair = await engine.generate_keypair()
        assert keypair.public_key is not None
        assert keypair.private_key is not None

        ciphertext, shared_secret = await engine.encapsulate(keypair.public_key)
        assert ciphertext is not None
        assert shared_secret is not None
        assert len(shared_secret.data) == 32


class TestDomainMaturityIntegration:
    """Test domain maturity labels are correctly set."""

    def test_ga_domains(self):
        from ai_engine.domains import DomainMaturity, DOMAIN_MATURITY
        assert DOMAIN_MATURITY["banking"] == DomainMaturity.GA
        assert DOMAIN_MATURITY["healthcare"] == DomainMaturity.GA
        assert DOMAIN_MATURITY["bpo"] == DomainMaturity.GA

    def test_preview_domains(self):
        from ai_engine.domains import DomainMaturity, DOMAIN_MATURITY
        assert DOMAIN_MATURITY["automotive"] == DomainMaturity.PREVIEW
        assert DOMAIN_MATURITY["aviation"] == DomainMaturity.PREVIEW

    def test_experimental_domains(self):
        from ai_engine.domains import DomainMaturity, DOMAIN_MATURITY
        assert DOMAIN_MATURITY["industrial"] == DomainMaturity.EXPERIMENTAL
