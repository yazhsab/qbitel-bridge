"""Unit tests for ai_engine.crypto.providers module."""

import pytest

from ai_engine.crypto.providers import (
    CryptoProvider,
    PROVIDER_TIERS,
    ProviderRegistry,
    ProviderTier,
)


class TestCryptoProviderEnum:
    """Tests for the CryptoProvider enum."""

    def test_crypto_provider_enum(self):
        """Verify all 5 members exist."""
        expected = {"LIBOQS", "PQCRYPTO", "KYBER_PY", "DILITHIUM_PY", "FALLBACK"}
        actual = {member.name for member in CryptoProvider}
        assert actual == expected

    def test_crypto_provider_values(self):
        assert CryptoProvider.LIBOQS.value == "liboqs"
        assert CryptoProvider.PQCRYPTO.value == "pqcrypto"
        assert CryptoProvider.KYBER_PY.value == "kyber-py"
        assert CryptoProvider.DILITHIUM_PY.value == "dilithium-py"
        assert CryptoProvider.FALLBACK.value == "fallback"


class TestProviderTierEnum:
    """Tests for the ProviderTier enum."""

    def test_provider_tier_enum(self):
        """Verify PRODUCTION, DEV_ONLY, UNSAFE members."""
        expected = {"PRODUCTION", "DEV_ONLY", "UNSAFE"}
        actual = {member.name for member in ProviderTier}
        assert actual == expected


class TestProviderTiersMapping:
    """Tests for the PROVIDER_TIERS mapping."""

    def test_liboqs_is_production(self):
        assert PROVIDER_TIERS[CryptoProvider.LIBOQS] == ProviderTier.PRODUCTION

    def test_pqcrypto_is_production(self):
        assert PROVIDER_TIERS[CryptoProvider.PQCRYPTO] == ProviderTier.PRODUCTION

    def test_kyber_py_is_dev_only(self):
        assert PROVIDER_TIERS[CryptoProvider.KYBER_PY] == ProviderTier.DEV_ONLY

    def test_dilithium_py_is_dev_only(self):
        assert PROVIDER_TIERS[CryptoProvider.DILITHIUM_PY] == ProviderTier.DEV_ONLY

    def test_fallback_is_unsafe(self):
        assert PROVIDER_TIERS[CryptoProvider.FALLBACK] == ProviderTier.UNSAFE

    def test_provider_tiers_mapping(self):
        """Verify key mappings in PROVIDER_TIERS."""
        assert PROVIDER_TIERS[CryptoProvider.LIBOQS] == ProviderTier.PRODUCTION
        assert PROVIDER_TIERS[CryptoProvider.KYBER_PY] == ProviderTier.DEV_ONLY
        assert PROVIDER_TIERS[CryptoProvider.FALLBACK] == ProviderTier.UNSAFE


class TestProviderRegistry:
    """Tests for the ProviderRegistry singleton."""

    @pytest.fixture(autouse=True)
    def _reset_registry(self):
        """Reset the singleton before and after each test."""
        ProviderRegistry.reset()
        yield
        ProviderRegistry.reset()

    def test_registry_singleton(self):
        """Two calls to ProviderRegistry() return the same instance."""
        reg_a = ProviderRegistry()
        reg_b = ProviderRegistry()
        assert reg_a is reg_b

    def test_registry_reset(self):
        """After reset(), a new instance is different from the old one."""
        old = ProviderRegistry()
        old_id = id(old)
        ProviderRegistry.reset()
        new = ProviderRegistry()
        assert id(new) != old_id

    def test_status_returns_dict(self, pqc_fallback_mode):
        """status() returns a dict with provider name keys."""
        reg = ProviderRegistry()
        result = reg.status()
        assert isinstance(result, dict)
        # Each value should have 'available' and 'tier' keys
        for provider_name, info in result.items():
            assert isinstance(provider_name, str)
            assert "available" in info
            assert "tier" in info

    def test_available_providers_returns_list(self, pqc_fallback_mode):
        """available_providers() returns a list of strings."""
        reg = ProviderRegistry()
        result = reg.available_providers()
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, str)

    def test_get_kem_provider_returns_string(self, pqc_fallback_mode):
        """get_kem_provider(strict_mode=False) returns a string."""
        reg = ProviderRegistry()
        result = reg.get_kem_provider(strict_mode=False)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_sig_provider_returns_string(self, pqc_fallback_mode):
        """get_sig_provider(strict_mode=False) returns a string."""
        reg = ProviderRegistry()
        result = reg.get_sig_provider(strict_mode=False)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_provider_tier(self, pqc_fallback_mode):
        """For known provider names, returns correct tier."""
        reg = ProviderRegistry()
        assert reg.get_provider_tier("liboqs") == ProviderTier.PRODUCTION
        assert reg.get_provider_tier("pqcrypto") == ProviderTier.PRODUCTION
        assert reg.get_provider_tier("kyber-py") == ProviderTier.DEV_ONLY
        assert reg.get_provider_tier("dilithium-py") == ProviderTier.DEV_ONLY
        assert reg.get_provider_tier("fallback") == ProviderTier.UNSAFE

    def test_get_provider_tier_unknown_defaults_to_unsafe(self, pqc_fallback_mode):
        """Unknown provider names should map to UNSAFE tier (via FALLBACK)."""
        reg = ProviderRegistry()
        assert reg.get_provider_tier("unknown-provider") == ProviderTier.UNSAFE

    def test_is_production_ready(self, pqc_fallback_mode):
        """is_production_ready() returns a bool."""
        reg = ProviderRegistry()
        result = reg.is_production_ready()
        assert isinstance(result, bool)
