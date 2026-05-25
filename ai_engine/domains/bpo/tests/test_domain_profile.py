"""
Tests for BPO Domain PQC Profiles

Tests cover:
- BPOSubdomain enumeration values
- BPOSecurityConstraints validation (valid and invalid)
- BPOPQCProfile.for_subdomain() for each subdomain
- Voice latency constraints
- DTMF masking constraints
- Pre-configured profiles (voice_signaling, agent_desktop, payment_processing, etc.)
- Profile serialization (to_dict)
- get_profile() and list_profiles()
"""

import pytest

from ai_engine.domains.bpo.core.domain_profile import (
    BPOSubdomain,
    PQCAlgorithm,
    BPOSecurityConstraints,
    BPOPQCProfile,
    BPO_PROFILES,
    get_profile,
    list_profiles,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_constraints():
    """Return default BPOSecurityConstraints."""
    return BPOSecurityConstraints()


@pytest.fixture
def voice_profile():
    """Return voice signaling profile."""
    return BPOPQCProfile.for_subdomain(BPOSubdomain.VOICE_SIGNALING)


@pytest.fixture
def payment_profile():
    """Return payment processing profile."""
    return BPOPQCProfile.for_subdomain(BPOSubdomain.PAYMENT_PROCESSING)


@pytest.fixture
def agent_desktop_profile():
    """Return agent desktop profile."""
    return BPOPQCProfile.for_subdomain(BPOSubdomain.AGENT_DESKTOP)


# ---------------------------------------------------------------------------
# BPOSubdomain Enumeration
# ---------------------------------------------------------------------------

class TestBPOSubdomain:
    """Tests for BPOSubdomain enumeration."""

    def test_voice_subdomains_exist(self):
        """Test that voice-related subdomains are defined."""
        assert BPOSubdomain.VOICE_SIGNALING is not None
        assert BPOSubdomain.VOICE_MEDIA is not None
        assert BPOSubdomain.IVR_SYSTEMS is not None
        assert BPOSubdomain.CALL_RECORDING is not None

    def test_agent_subdomains_exist(self):
        """Test that agent-related subdomains are defined."""
        assert BPOSubdomain.AGENT_DESKTOP is not None
        assert BPOSubdomain.TERMINAL_EMULATION is not None
        assert BPOSubdomain.SCREEN_RECORDING is not None

    def test_infrastructure_subdomains_exist(self):
        """Test that infrastructure subdomains are defined."""
        assert BPOSubdomain.PBX_INTEGRATION is not None
        assert BPOSubdomain.CTI_MIDDLEWARE is not None
        assert BPOSubdomain.WFM_SYSTEMS is not None

    def test_customer_data_subdomains_exist(self):
        """Test that customer data subdomains are defined."""
        assert BPOSubdomain.CRM_INTEGRATION is not None
        assert BPOSubdomain.PAYMENT_PROCESSING is not None
        assert BPOSubdomain.CUSTOMER_DATA is not None

    def test_remote_subdomains_exist(self):
        """Test that remote operations subdomains are defined."""
        assert BPOSubdomain.REMOTE_AGENT is not None
        assert BPOSubdomain.VPN_LESS_ACCESS is not None

    def test_analytics_subdomains_exist(self):
        """Test that analytics subdomains are defined."""
        assert BPOSubdomain.QUALITY_MONITORING is not None
        assert BPOSubdomain.SPEECH_ANALYTICS is not None
        assert BPOSubdomain.COMPLIANCE_RECORDING is not None

    def test_all_subdomains_are_unique(self):
        """Test that all subdomain values are unique."""
        values = [member.value for member in BPOSubdomain]
        assert len(values) == len(set(values)), "BPOSubdomain values must be unique"

    @pytest.mark.parametrize("subdomain", list(BPOSubdomain))
    def test_subdomain_has_name(self, subdomain):
        """Test that each subdomain has a valid name."""
        assert subdomain.name is not None
        assert len(subdomain.name) > 0


# ---------------------------------------------------------------------------
# PQCAlgorithm Enumeration
# ---------------------------------------------------------------------------

class TestPQCAlgorithm:
    """Tests for PQCAlgorithm enumeration."""

    def test_kem_algorithms(self):
        """Test KEM algorithm properties."""
        assert PQCAlgorithm.ML_KEM_512.is_kem is True
        assert PQCAlgorithm.ML_KEM_768.is_kem is True
        assert PQCAlgorithm.ML_KEM_1024.is_kem is True

    def test_signature_algorithms(self):
        """Test signature algorithm properties."""
        assert PQCAlgorithm.ML_DSA_44.is_signature is True
        assert PQCAlgorithm.ML_DSA_65.is_signature is True
        assert PQCAlgorithm.ML_DSA_87.is_signature is True

    def test_falcon_algorithms(self):
        """Test Falcon signature algorithms."""
        assert PQCAlgorithm.FALCON_512.is_signature is True
        assert PQCAlgorithm.FALCON_1024.is_signature is True
        assert PQCAlgorithm.FALCON_512.security_bits == 128
        assert PQCAlgorithm.FALCON_1024.security_bits == 256

    def test_hybrid_algorithms(self):
        """Test hybrid algorithm properties."""
        assert PQCAlgorithm.X25519_ML_KEM_768.is_hybrid is True
        assert PQCAlgorithm.P384_ML_KEM_1024.is_hybrid is True
        assert PQCAlgorithm.ED25519_ML_DSA_65.is_hybrid is True

    def test_kem_is_not_signature(self):
        """Test that KEM algorithms are not signatures."""
        assert PQCAlgorithm.ML_KEM_512.is_signature is False
        assert PQCAlgorithm.ML_KEM_768.is_signature is False

    def test_signature_is_not_kem(self):
        """Test that signature algorithms are not KEMs."""
        assert PQCAlgorithm.ML_DSA_44.is_kem is False
        assert PQCAlgorithm.ML_DSA_65.is_kem is False

    @pytest.mark.parametrize("algo,expected_bits", [
        (PQCAlgorithm.ML_KEM_512, 128),
        (PQCAlgorithm.ML_KEM_768, 192),
        (PQCAlgorithm.ML_KEM_1024, 256),
        (PQCAlgorithm.ML_DSA_44, 128),
        (PQCAlgorithm.ML_DSA_65, 192),
        (PQCAlgorithm.ML_DSA_87, 256),
    ])
    def test_security_bits(self, algo, expected_bits):
        """Test security bits for each algorithm."""
        assert algo.security_bits == expected_bits, (
            f"{algo.algo_name} should have {expected_bits} security bits"
        )


# ---------------------------------------------------------------------------
# BPOSecurityConstraints Validation
# ---------------------------------------------------------------------------

class TestBPOSecurityConstraints:
    """Tests for BPOSecurityConstraints validation."""

    def test_default_constraints_are_valid(self, default_constraints):
        """Test that default constraints pass validation."""
        errors = default_constraints.validate()
        assert errors == [], f"Default constraints should be valid, got: {errors}"

    def test_min_security_bits_below_128(self):
        """Test validation rejects security bits below 128."""
        constraints = BPOSecurityConstraints(min_security_bits=64)
        errors = constraints.validate()
        assert any("128" in e for e in errors), (
            "Should require at least 128 security bits"
        )

    def test_pqc_overhead_exceeds_voice_latency(self):
        """Test validation catches PQC overhead exceeding max voice latency."""
        constraints = BPOSecurityConstraints(
            voice_codec_overhead_ms=200.0,
            max_voice_latency_ms=150.0,
        )
        errors = constraints.validate()
        assert any("overhead" in e.lower() or "latency" in e.lower() for e in errors), (
            "Should detect PQC overhead exceeding voice latency"
        )

    def test_target_latency_exceeds_max(self):
        """Test validation catches target latency exceeding maximum."""
        constraints = BPOSecurityConstraints(
            max_latency_ms=50.0,
            target_latency_ms=100.0,
        )
        errors = constraints.validate()
        assert any("target" in e.lower() for e in errors), (
            "Should detect target latency exceeding maximum"
        )

    def test_pci_level1_requires_dtmf_masking(self):
        """Test validation enforces DTMF masking for PCI-DSS Level 1."""
        constraints = BPOSecurityConstraints(
            pci_dss_level=1,
            require_dtmf_masking=False,
        )
        errors = constraints.validate()
        assert any("dtmf" in e.lower() or "pci" in e.lower() for e in errors), (
            "PCI-DSS Level 1 must require DTMF masking"
        )

    def test_recording_retention_minimum(self):
        """Test validation requires at least 1 year recording retention."""
        constraints = BPOSecurityConstraints(
            require_call_recording=True,
            retention_years=0,
        )
        errors = constraints.validate()
        assert any("retention" in e.lower() for e in errors), (
            "Call recording retention must be at least 1 year"
        )

    def test_valid_custom_constraints(self):
        """Test a fully valid custom constraints object."""
        constraints = BPOSecurityConstraints(
            max_latency_ms=200.0,
            target_latency_ms=50.0,
            throughput_sessions=10000,
            voice_codec_overhead_ms=10.0,
            max_voice_latency_ms=150.0,
            min_security_bits=192,
            pci_dss_level=1,
            require_dtmf_masking=True,
            require_call_recording=True,
            retention_years=7,
        )
        errors = constraints.validate()
        assert errors == [], f"Valid custom constraints should pass, got: {errors}"

    def test_multiple_errors_returned(self):
        """Test that multiple validation errors are all reported."""
        constraints = BPOSecurityConstraints(
            min_security_bits=64,
            max_latency_ms=50.0,
            target_latency_ms=100.0,
            pci_dss_level=1,
            require_dtmf_masking=False,
        )
        errors = constraints.validate()
        assert len(errors) >= 3, (
            f"Should report multiple errors, got {len(errors)}: {errors}"
        )


# ---------------------------------------------------------------------------
# Voice Latency Constraints
# ---------------------------------------------------------------------------

class TestVoiceLatencyConstraints:
    """Tests for voice-specific latency constraints."""

    def test_voice_signaling_low_latency(self, voice_profile):
        """Test voice signaling has low latency requirements."""
        assert voice_profile.constraints.max_latency_ms <= 100.0, (
            "Voice signaling max latency should be <= 100ms"
        )

    def test_voice_media_ultra_low_latency(self):
        """Test voice media has ultra-low latency requirements."""
        profile = BPOPQCProfile.for_subdomain(BPOSubdomain.VOICE_MEDIA)
        assert profile.constraints.max_latency_ms <= 30.0, (
            "Voice media max latency should be <= 30ms"
        )

    def test_voice_codec_overhead_within_budget(self, voice_profile):
        """Test voice codec overhead is within latency budget."""
        c = voice_profile.constraints
        assert c.voice_codec_overhead_ms <= c.max_voice_latency_ms, (
            "Voice codec overhead must not exceed max voice latency"
        )

    def test_itu_g114_voice_latency(self, voice_profile):
        """Test ITU-T G.114 recommendation compliance (150ms max)."""
        assert voice_profile.constraints.max_voice_latency_ms <= 150.0, (
            "Max voice latency should comply with ITU-T G.114 (150ms)"
        )


# ---------------------------------------------------------------------------
# DTMF Masking Constraints
# ---------------------------------------------------------------------------

class TestDTMFMaskingConstraints:
    """Tests for DTMF masking constraints."""

    def test_ivr_requires_dtmf_masking(self):
        """Test IVR systems require DTMF masking."""
        profile = BPOPQCProfile.for_subdomain(BPOSubdomain.IVR_SYSTEMS)
        assert profile.constraints.require_dtmf_masking is True, (
            "IVR systems should require DTMF masking"
        )

    def test_payment_requires_dtmf_masking(self, payment_profile):
        """Test payment processing requires DTMF masking."""
        assert payment_profile.constraints.require_dtmf_masking is True, (
            "Payment processing must require DTMF masking"
        )

    def test_payment_is_pci_level1(self, payment_profile):
        """Test payment processing is PCI-DSS Level 1."""
        assert payment_profile.constraints.pci_dss_level == 1, (
            "Payment processing should be PCI-DSS Level 1"
        )


# ---------------------------------------------------------------------------
# BPOPQCProfile.for_subdomain()
# ---------------------------------------------------------------------------

class TestBPOPQCProfileForSubdomain:
    """Tests for BPOPQCProfile.for_subdomain() factory method."""

    @pytest.mark.parametrize("subdomain", list(BPOSubdomain))
    def test_for_subdomain_returns_profile(self, subdomain):
        """Test that for_subdomain returns a profile for every subdomain."""
        profile = BPOPQCProfile.for_subdomain(subdomain)
        assert profile is not None
        assert isinstance(profile, BPOPQCProfile)
        assert profile.subdomain == subdomain

    def test_voice_signaling_uses_fast_kem(self, voice_profile):
        """Test voice signaling uses fast KEM algorithm."""
        assert voice_profile.kem_algorithm == PQCAlgorithm.ML_KEM_512, (
            "Voice signaling should use ML-KEM-512 for speed"
        )

    def test_voice_signaling_uses_compact_signatures(self, voice_profile):
        """Test voice signaling uses Falcon for compact signatures."""
        assert voice_profile.sig_algorithm == PQCAlgorithm.FALCON_512, (
            "Voice signaling should use Falcon-512 for compact signatures"
        )

    def test_voice_media_pure_pqc(self):
        """Test voice media uses pure PQC (no hybrid) for speed."""
        profile = BPOPQCProfile.for_subdomain(BPOSubdomain.VOICE_MEDIA)
        assert profile.use_hybrid_mode is False, (
            "Voice media should use pure PQC for speed"
        )

    def test_call_recording_max_security(self):
        """Test call recording uses maximum security algorithms."""
        profile = BPOPQCProfile.for_subdomain(BPOSubdomain.CALL_RECORDING)
        assert profile.kem_algorithm == PQCAlgorithm.ML_KEM_1024, (
            "Call recording should use ML-KEM-1024 for max security"
        )
        assert profile.sig_algorithm == PQCAlgorithm.ML_DSA_87, (
            "Call recording should use ML-DSA-87 for max security"
        )

    def test_payment_processing_max_security(self, payment_profile):
        """Test payment processing uses max security with hybrid mode."""
        assert payment_profile.kem_algorithm == PQCAlgorithm.ML_KEM_1024
        assert payment_profile.sig_algorithm == PQCAlgorithm.ML_DSA_87
        assert payment_profile.use_hybrid_mode is True
        assert payment_profile.strict_mode is True

    def test_agent_desktop_balanced(self, agent_desktop_profile):
        """Test agent desktop has balanced latency/security."""
        assert agent_desktop_profile.kem_algorithm == PQCAlgorithm.ML_KEM_768
        assert agent_desktop_profile.use_hybrid_mode is True
        assert agent_desktop_profile.key_rotation_hours == 8  # Per shift

    def test_remote_agent_frequent_key_rotation(self):
        """Test remote agent has more frequent key rotation."""
        profile = BPOPQCProfile.for_subdomain(BPOSubdomain.REMOTE_AGENT)
        assert profile.key_rotation_hours <= 4, (
            "Remote agent should have frequent key rotation (<= 4 hours)"
        )

    def test_terminal_emulation_fips_required(self):
        """Test terminal emulation requires FIPS compliance."""
        profile = BPOPQCProfile.for_subdomain(BPOSubdomain.TERMINAL_EMULATION)
        assert profile.constraints.require_fips is True, (
            "Terminal emulation should require FIPS compliance"
        )


# ---------------------------------------------------------------------------
# Profile helper methods
# ---------------------------------------------------------------------------

class TestBPOPQCProfileHelpers:
    """Tests for BPOPQCProfile helper methods."""

    def test_get_kem_for_128_bits(self, voice_profile):
        """Test KEM selection for 128-bit security."""
        kem = voice_profile.get_kem_for_security_level(128)
        assert kem == PQCAlgorithm.ML_KEM_512

    def test_get_kem_for_192_bits(self, voice_profile):
        """Test KEM selection for 192-bit security."""
        kem = voice_profile.get_kem_for_security_level(192)
        assert kem == PQCAlgorithm.ML_KEM_768

    def test_get_kem_for_256_bits(self, voice_profile):
        """Test KEM selection for 256-bit security."""
        kem = voice_profile.get_kem_for_security_level(256)
        assert kem == PQCAlgorithm.ML_KEM_1024

    def test_get_sig_for_128_bits(self, voice_profile):
        """Test signature selection for 128-bit security."""
        sig = voice_profile.get_sig_for_security_level(128)
        assert sig == PQCAlgorithm.ML_DSA_44

    def test_get_sig_for_192_bits(self, voice_profile):
        """Test signature selection for 192-bit security."""
        sig = voice_profile.get_sig_for_security_level(192)
        assert sig == PQCAlgorithm.ML_DSA_65

    def test_get_sig_for_256_bits(self, voice_profile):
        """Test signature selection for 256-bit security."""
        sig = voice_profile.get_sig_for_security_level(256)
        assert sig == PQCAlgorithm.ML_DSA_87


# ---------------------------------------------------------------------------
# Profile Serialization
# ---------------------------------------------------------------------------

class TestProfileSerialization:
    """Tests for BPOPQCProfile serialization."""

    def test_to_dict_has_subdomain(self, voice_profile):
        """Test serialization includes subdomain."""
        d = voice_profile.to_dict()
        assert "subdomain" in d
        assert d["subdomain"] == "VOICE_SIGNALING"

    def test_to_dict_has_constraints(self, voice_profile):
        """Test serialization includes constraints."""
        d = voice_profile.to_dict()
        assert "constraints" in d
        assert "max_latency_ms" in d["constraints"]
        assert "target_latency_ms" in d["constraints"]
        assert "require_fips" in d["constraints"]
        assert "require_dtmf_masking" in d["constraints"]

    def test_to_dict_has_algorithms(self, voice_profile):
        """Test serialization includes algorithm selections."""
        d = voice_profile.to_dict()
        assert "algorithms" in d
        assert "kem" in d["algorithms"]
        assert "signature" in d["algorithms"]
        assert "hash" in d["algorithms"]

    def test_to_dict_has_settings(self, voice_profile):
        """Test serialization includes settings."""
        d = voice_profile.to_dict()
        assert "settings" in d
        assert "use_hybrid_mode" in d["settings"]
        assert "strict_mode" in d["settings"]
        assert "key_rotation_hours" in d["settings"]

    def test_to_dict_hybrid_kem_present(self, voice_profile):
        """Test serialization includes hybrid KEM when set."""
        d = voice_profile.to_dict()
        algorithms = d["algorithms"]
        # Voice signaling has hybrid mode enabled, so hybrid_kem may be set
        assert "hybrid_kem" in algorithms

    def test_to_dict_round_trip_consistency(self, payment_profile):
        """Test serialization produces consistent output."""
        d1 = payment_profile.to_dict()
        d2 = payment_profile.to_dict()
        assert d1 == d2, "Repeated serialization should produce identical output"


# ---------------------------------------------------------------------------
# Pre-configured Profiles
# ---------------------------------------------------------------------------

class TestPreConfiguredProfiles:
    """Tests for pre-configured BPO profile collection."""

    def test_voice_signaling_profile_exists(self):
        """Test voice_signaling profile is pre-configured."""
        assert "voice_signaling" in BPO_PROFILES

    def test_agent_desktop_profile_exists(self):
        """Test agent_desktop profile is pre-configured."""
        assert "agent_desktop" in BPO_PROFILES

    def test_payment_processing_profile_exists(self):
        """Test payment_processing profile is pre-configured."""
        assert "payment_processing" in BPO_PROFILES

    def test_call_recording_profile_exists(self):
        """Test call_recording profile is pre-configured."""
        assert "call_recording" in BPO_PROFILES

    def test_remote_agent_profile_exists(self):
        """Test remote_agent profile is pre-configured."""
        assert "remote_agent" in BPO_PROFILES

    def test_cti_middleware_profile_exists(self):
        """Test cti_middleware profile is pre-configured."""
        assert "cti_middleware" in BPO_PROFILES

    @pytest.mark.parametrize("profile_name", [
        "voice_signaling",
        "voice_media",
        "ivr_systems",
        "call_recording",
        "agent_desktop",
        "terminal_emulation",
        "payment_processing",
        "remote_agent",
        "quality_monitoring",
        "crm_integration",
        "cti_middleware",
        "speech_analytics",
    ])
    def test_all_expected_profiles_present(self, profile_name):
        """Test that all expected pre-configured profiles exist."""
        assert profile_name in BPO_PROFILES, (
            f"Pre-configured profile '{profile_name}' should exist"
        )

    @pytest.mark.parametrize("profile_name", list(BPO_PROFILES.keys()))
    def test_all_profiles_are_valid(self, profile_name):
        """Test that all pre-configured profiles are valid BPOPQCProfile instances."""
        profile = BPO_PROFILES[profile_name]
        assert isinstance(profile, BPOPQCProfile)


# ---------------------------------------------------------------------------
# get_profile() and list_profiles()
# ---------------------------------------------------------------------------

class TestProfileAccessFunctions:
    """Tests for get_profile() and list_profiles() functions."""

    def test_get_profile_existing(self):
        """Test get_profile returns existing profile."""
        profile = get_profile("voice_signaling")
        assert profile is not None
        assert isinstance(profile, BPOPQCProfile)

    def test_get_profile_nonexistent(self):
        """Test get_profile returns None for unknown profile."""
        profile = get_profile("nonexistent_profile")
        assert profile is None

    def test_list_profiles_returns_list(self):
        """Test list_profiles returns a list of strings."""
        profiles = list_profiles()
        assert isinstance(profiles, list)
        assert len(profiles) > 0
        assert all(isinstance(p, str) for p in profiles)

    def test_list_profiles_contains_expected(self):
        """Test list_profiles includes expected profile names."""
        profiles = list_profiles()
        assert "voice_signaling" in profiles
        assert "payment_processing" in profiles
        assert "agent_desktop" in profiles

    def test_get_profile_matches_list(self):
        """Test every listed profile is retrievable."""
        for name in list_profiles():
            profile = get_profile(name)
            assert profile is not None, f"Profile '{name}' listed but not retrievable"
