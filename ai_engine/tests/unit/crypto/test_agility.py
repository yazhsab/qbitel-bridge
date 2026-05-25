"""Unit tests for ai_engine.crypto.agility module."""

import pytest

from ai_engine.crypto.agility import (
    ALGORITHM_REGISTRY,
    CNSA2_POLICY,
    LEGACY_COMPATIBLE_POLICY,
    POST_QUANTUM_POLICY,
    TRANSITIONAL_POLICY,
    AlgorithmDescriptor,
    AlgorithmFamily,
    AlgorithmStatus,
    CryptoAgilityNegotiator,
    CryptoCapability,
    CryptoPolicy,
    NegotiatedSuite,
    NegotiationFailedError,
    SecurityEra,
)


class TestAlgorithmRegistry:
    """Tests for the ALGORITHM_REGISTRY constant."""

    def test_algorithm_registry_not_empty(self):
        """ALGORITHM_REGISTRY has entries."""
        assert len(ALGORITHM_REGISTRY) > 0

    def test_algorithm_registry_values_are_descriptors(self):
        """All values are AlgorithmDescriptor instances."""
        for name, algo in ALGORITHM_REGISTRY.items():
            assert isinstance(algo, AlgorithmDescriptor), f"{name} is not AlgorithmDescriptor"
            assert algo.name == name

    def test_registry_contains_expected_kems(self):
        """Registry includes key ML-KEM variants."""
        assert "ML-KEM-768" in ALGORITHM_REGISTRY
        assert "ML-KEM-1024" in ALGORITHM_REGISTRY

    def test_registry_contains_expected_sigs(self):
        """Registry includes key signature algorithms."""
        assert "ML-DSA-65" in ALGORITHM_REGISTRY
        assert "Falcon-512" in ALGORITHM_REGISTRY


class TestSecurityEraOrdering:
    """Tests for SecurityEra ordering."""

    def test_security_era_ordering(self):
        """CLASSICAL < HYBRID < POST_QUANTUM < CNSA2 by enum definition order."""
        eras = list(SecurityEra)
        assert eras.index(SecurityEra.CLASSICAL) < eras.index(SecurityEra.HYBRID)
        assert eras.index(SecurityEra.HYBRID) < eras.index(SecurityEra.POST_QUANTUM)
        assert eras.index(SecurityEra.POST_QUANTUM) < eras.index(SecurityEra.CNSA2)

    def test_security_era_values(self):
        assert SecurityEra.CLASSICAL.value == "classical"
        assert SecurityEra.HYBRID.value == "hybrid"
        assert SecurityEra.POST_QUANTUM.value == "post-quantum"
        assert SecurityEra.CNSA2.value == "cnsa-2.0"


class TestPredefinedPolicies:
    """Tests for pre-defined CryptoPolicy instances."""

    def test_predefined_policies_exist(self):
        """All 4 policies are importable and are CryptoPolicy instances."""
        assert isinstance(TRANSITIONAL_POLICY, CryptoPolicy)
        assert isinstance(POST_QUANTUM_POLICY, CryptoPolicy)
        assert isinstance(CNSA2_POLICY, CryptoPolicy)
        assert isinstance(LEGACY_COMPATIBLE_POLICY, CryptoPolicy)

    def test_policies_have_minimum_security_bits(self):
        """Each policy has a minimum_security_bits attribute."""
        assert TRANSITIONAL_POLICY.minimum_security_bits >= 128
        assert POST_QUANTUM_POLICY.minimum_security_bits >= 128
        assert CNSA2_POLICY.minimum_security_bits >= 256
        assert LEGACY_COMPATIBLE_POLICY.minimum_security_bits >= 128

    def test_cnsa2_policy_properties(self):
        """Verify CNSA 2.0 policy has strict settings."""
        assert CNSA2_POLICY.minimum_security_bits == 256
        assert CNSA2_POLICY.required_era == SecurityEra.CNSA2
        assert CNSA2_POLICY.cnsa2_required is True
        assert CNSA2_POLICY.minimum_nist_level == 5
        assert CNSA2_POLICY.allow_classical_only is False

    def test_transitional_policy_requires_hybrid(self):
        assert TRANSITIONAL_POLICY.require_hybrid is True
        assert TRANSITIONAL_POLICY.allow_classical_only is False

    def test_post_quantum_policy_bans_classical(self):
        assert "ECDSA-P256" in POST_QUANTUM_POLICY.banned_algorithms
        assert POST_QUANTUM_POLICY.allow_classical_only is False
        assert POST_QUANTUM_POLICY.required_era == SecurityEra.POST_QUANTUM

    def test_legacy_compatible_policy_allows_classical(self):
        assert LEGACY_COMPATIBLE_POLICY.allow_classical_only is True
        assert LEGACY_COMPATIBLE_POLICY.require_hybrid is False


class TestNegotiation:
    """Tests for CryptoAgilityNegotiator."""

    def test_negotiate_success(self):
        """Two peers with overlapping capabilities negotiate a suite."""
        negotiator = CryptoAgilityNegotiator(
            "peer-a", policy=LEGACY_COMPATIBLE_POLICY
        )

        local_caps = CryptoCapability(
            peer_id="peer-a",
            supported_kems=["ML-KEM-768", "X25519"],
            supported_sigs=["ML-DSA-65", "ECDSA-P256"],
            supported_symmetric=["AES-256-GCM"],
            supported_hashes=["SHA3-256", "SHA-256"],
            supported_kdfs=["HKDF-SHA3-256"],
        )
        remote_caps = CryptoCapability(
            peer_id="peer-b",
            supported_kems=["ML-KEM-768", "ML-KEM-1024"],
            supported_sigs=["ML-DSA-65", "ML-DSA-87"],
            supported_symmetric=["AES-256-GCM", "AES-128-GCM"],
            supported_hashes=["SHA3-256"],
            supported_kdfs=["HKDF-SHA3-256"],
        )

        suite = negotiator.negotiate(local_caps, remote_caps)
        assert isinstance(suite, NegotiatedSuite)
        assert suite.kem.name == "ML-KEM-768"
        assert suite.signature.name == "ML-DSA-65"
        assert suite.symmetric.name == "AES-256-GCM"
        assert suite.hash_algo.name == "SHA3-256"
        assert suite.initiator_id == "peer-a"
        assert suite.responder_id == "peer-b"
        assert isinstance(suite.suite_id, bytes)
        assert len(suite.suite_id) > 0

    def test_negotiate_empty_fails(self):
        """Empty capabilities should raise NegotiationFailedError."""
        negotiator = CryptoAgilityNegotiator(
            "peer-a", policy=LEGACY_COMPATIBLE_POLICY
        )

        local_caps = CryptoCapability(
            peer_id="peer-a",
            supported_kems=[],
            supported_sigs=[],
            supported_symmetric=[],
            supported_hashes=[],
            supported_kdfs=[],
        )
        remote_caps = CryptoCapability(
            peer_id="peer-b",
            supported_kems=[],
            supported_sigs=[],
            supported_symmetric=[],
            supported_hashes=[],
            supported_kdfs=[],
        )

        with pytest.raises(NegotiationFailedError):
            negotiator.negotiate(local_caps, remote_caps)

    def test_negotiate_no_overlap_fails(self):
        """Non-overlapping capabilities should raise NegotiationFailedError."""
        negotiator = CryptoAgilityNegotiator(
            "peer-a", policy=LEGACY_COMPATIBLE_POLICY
        )

        local_caps = CryptoCapability(
            peer_id="peer-a",
            supported_kems=["ML-KEM-768"],
            supported_sigs=["ML-DSA-65"],
            supported_symmetric=["AES-256-GCM"],
            supported_hashes=["SHA3-256"],
            supported_kdfs=["HKDF-SHA3-256"],
        )
        remote_caps = CryptoCapability(
            peer_id="peer-b",
            supported_kems=["X25519"],
            supported_sigs=["ECDSA-P256"],
            supported_symmetric=["AES-128-GCM"],
            supported_hashes=["SHA-256"],
            supported_kdfs=["HKDF-SHA256"],
        )

        with pytest.raises(NegotiationFailedError):
            negotiator.negotiate(local_caps, remote_caps)

    def test_advertise_capabilities(self):
        """advertise_capabilities() returns a CryptoCapability."""
        negotiator = CryptoAgilityNegotiator(
            "peer-a", policy=LEGACY_COMPATIBLE_POLICY
        )
        caps = negotiator.advertise_capabilities()
        assert isinstance(caps, CryptoCapability)
        assert caps.peer_id == "peer-a"
        assert isinstance(caps.supported_kems, list)
        assert isinstance(caps.supported_sigs, list)

    def test_negotiated_suite_security_level(self):
        """NegotiatedSuite reports correct security_level property."""
        negotiator = CryptoAgilityNegotiator(
            "peer-a", policy=LEGACY_COMPATIBLE_POLICY
        )
        local_caps = CryptoCapability(
            peer_id="peer-a",
            supported_kems=["ML-KEM-768"],
            supported_sigs=["ML-DSA-65"],
            supported_symmetric=["AES-256-GCM"],
            supported_hashes=["SHA3-256"],
            supported_kdfs=["HKDF-SHA3-256"],
        )
        remote_caps = CryptoCapability(
            peer_id="peer-b",
            supported_kems=["ML-KEM-768"],
            supported_sigs=["ML-DSA-65"],
            supported_symmetric=["AES-256-GCM"],
            supported_hashes=["SHA3-256"],
            supported_kdfs=["HKDF-SHA3-256"],
        )
        suite = negotiator.negotiate(local_caps, remote_caps)
        assert suite.security_level >= 192
        assert suite.is_fully_post_quantum is True
