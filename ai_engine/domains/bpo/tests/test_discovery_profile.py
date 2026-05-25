"""
Tests for BPO AI-powered protocol discovery profile.

Tests cover:
- BPOProtocolType enumeration (SIP, CTI, IVR, TN3270e, RTP, SRTP, etc.)
- BPOProtocolFingerprint pattern matching (SIP, TN3270e, RTP, CTI-CSTA, IVR-VXML)
- BPODiscoveryProfile creation and classification methods
- Discovery configuration (timeouts, confidence thresholds, sample sizes)
- Environment-specific profiles (on-premise, cloud, remote, hybrid)

Note: These tests define the expected behavior for the BPO discovery profile
module (ai_engine.domains.bpo.discovery.bpo_discovery_profile). The tests
serve as a specification and will pass once the module is implemented.
"""

import pytest

from ai_engine.domains.bpo.discovery.bpo_discovery_profile import (
    BPODiscoveryProfile,
    BPOProtocolType,
    BPODiscoveryHint,
    BPOProtocolFingerprint,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_profile():
    """Return a default BPODiscoveryProfile."""
    return BPODiscoveryProfile()


@pytest.fixture
def on_premise_profile():
    """Return an on-premise environment discovery profile."""
    return BPODiscoveryProfile.for_environment("ON_PREMISE")


@pytest.fixture
def cloud_profile():
    """Return a cloud CCaaS environment discovery profile."""
    return BPODiscoveryProfile.for_environment("CLOUD_CCaaS")


@pytest.fixture
def remote_profile():
    """Return a remote workforce environment discovery profile."""
    return BPODiscoveryProfile.for_environment("REMOTE_WORKFORCE")


@pytest.fixture
def hybrid_profile():
    """Return a hybrid environment discovery profile."""
    return BPODiscoveryProfile.for_environment("HYBRID")


@pytest.fixture
def sip_fingerprint():
    """Return the SIP protocol fingerprint."""
    return BPOProtocolFingerprint.for_protocol(BPOProtocolType.SIP)


@pytest.fixture
def tn3270e_fingerprint():
    """Return the TN3270e protocol fingerprint."""
    return BPOProtocolFingerprint.for_protocol(BPOProtocolType.TN3270E)


@pytest.fixture
def rtp_fingerprint():
    """Return the RTP protocol fingerprint."""
    return BPOProtocolFingerprint.for_protocol(BPOProtocolType.RTP)


# ---------------------------------------------------------------------------
# BPOProtocolType Enumeration
# ---------------------------------------------------------------------------

class TestBPOProtocolType:
    """Tests for BPOProtocolType enumeration."""

    def test_voice_protocol_types_exist(self):
        """Test that voice-related protocol types are defined."""
        assert BPOProtocolType.SIP is not None
        assert BPOProtocolType.RTP is not None
        assert BPOProtocolType.SRTP is not None
        assert BPOProtocolType.SDP is not None

    def test_signaling_protocol_types_exist(self):
        """Test that signaling protocol types are defined."""
        assert BPOProtocolType.DTMF is not None
        assert BPOProtocolType.SS7 is not None

    def test_terminal_protocol_types_exist(self):
        """Test that terminal protocol types are defined."""
        assert BPOProtocolType.TN3270E is not None
        assert BPOProtocolType.TN5250 is not None
        assert BPOProtocolType.SSH is not None

    def test_cti_protocol_types_exist(self):
        """Test that CTI protocol types are defined."""
        assert BPOProtocolType.CTI_CSTA is not None
        assert BPOProtocolType.CTI_TAPI is not None
        assert BPOProtocolType.CTI_JTAPI is not None

    def test_ivr_protocol_types_exist(self):
        """Test that IVR protocol types are defined."""
        assert BPOProtocolType.IVR_VXML is not None
        assert BPOProtocolType.IVR_MRCP is not None
        assert BPOProtocolType.IVR_CCXML is not None

    def test_pqc_protocol_types_exist(self):
        """Test that PQC-enhanced protocol types are defined."""
        assert BPOProtocolType.SIP_PQC is not None
        assert BPOProtocolType.SRTP_PQC is not None

    def test_integration_protocol_types_exist(self):
        """Test that integration protocol types are defined."""
        assert BPOProtocolType.MGCP is not None

    def test_at_least_15_protocol_types(self):
        """Test that at least 15 BPO protocol types are defined."""
        protocol_count = len(list(BPOProtocolType))
        assert protocol_count >= 15, (
            f"Expected at least 15 BPO protocol types, got {protocol_count}"
        )

    def test_all_protocol_types_unique(self):
        """Test that all protocol type values are unique."""
        values = [member.value for member in BPOProtocolType]
        assert len(values) == len(set(values)), (
            "BPOProtocolType values must be unique"
        )

    @pytest.mark.parametrize("protocol_type", list(BPOProtocolType))
    def test_protocol_type_has_name(self, protocol_type):
        """Test that each protocol type has a valid name."""
        assert protocol_type.name is not None
        assert len(protocol_type.name) > 0

    @pytest.mark.parametrize("protocol_type", list(BPOProtocolType))
    def test_protocol_type_has_display_name(self, protocol_type):
        """Test that each protocol type has a human-readable display name."""
        assert hasattr(protocol_type, "display_name")
        assert protocol_type.display_name is not None
        assert len(protocol_type.display_name) > 0, (
            f"Protocol type {protocol_type.name} should have a display name"
        )


# ---------------------------------------------------------------------------
# BPOProtocolFingerprint Pattern Matching
# ---------------------------------------------------------------------------

class TestBPOProtocolFingerprint:
    """Tests for BPOProtocolFingerprint pattern matching."""

    def test_sip_fingerprint_matches_invite(self, sip_fingerprint):
        """Test SIP fingerprint matches INVITE request."""
        raw_data = b"INVITE sip:user@example.com SIP/2.0\r\nVia: SIP/2.0/TLS\r\n"
        assert sip_fingerprint.matches(raw_data) is True, (
            "SIP fingerprint should match INVITE request"
        )

    def test_sip_fingerprint_matches_response(self, sip_fingerprint):
        """Test SIP fingerprint matches SIP/2.0 response."""
        raw_data = b"SIP/2.0 200 OK\r\nVia: SIP/2.0/TLS\r\n"
        assert sip_fingerprint.matches(raw_data) is True, (
            "SIP fingerprint should match SIP/2.0 response"
        )

    def test_sip_fingerprint_rejects_non_sip(self, sip_fingerprint):
        """Test SIP fingerprint rejects non-SIP data."""
        raw_data = b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n"
        assert sip_fingerprint.matches(raw_data) is False, (
            "SIP fingerprint should not match HTTP data"
        )

    def test_tn3270e_fingerprint_matches_iac_eor(self, tn3270e_fingerprint):
        """Test TN3270e fingerprint matches IAC/EOR byte sequences."""
        # IAC (0xFF) followed by EOR (0xEF)
        raw_data = bytes([0xFF, 0xEF, 0x00, 0x01, 0x02])
        assert tn3270e_fingerprint.matches(raw_data) is True, (
            "TN3270e fingerprint should match IAC/EOR byte sequence"
        )

    def test_tn3270e_fingerprint_matches_telnet_negotiation(self, tn3270e_fingerprint):
        """Test TN3270e fingerprint matches telnet negotiation sequences."""
        # IAC (0xFF) DO (0xFD) TN3270E (0x28)
        raw_data = bytes([0xFF, 0xFD, 0x28, 0x00])
        assert tn3270e_fingerprint.matches(raw_data) is True, (
            "TN3270e fingerprint should match telnet negotiation"
        )

    def test_tn3270e_fingerprint_rejects_plain_text(self, tn3270e_fingerprint):
        """Test TN3270e fingerprint rejects plain text data."""
        raw_data = b"Hello World\r\n"
        assert tn3270e_fingerprint.matches(raw_data) is False, (
            "TN3270e fingerprint should not match plain text"
        )

    def test_rtp_fingerprint_matches_version2(self, rtp_fingerprint):
        """Test RTP fingerprint matches version=2 header byte."""
        # RTP version 2: first byte has version=2 in high 2 bits (0x80)
        raw_data = bytes([0x80, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
                          0x00, 0x00, 0x00, 0x01])
        assert rtp_fingerprint.matches(raw_data) is True, (
            "RTP fingerprint should match version=2 header byte"
        )

    def test_rtp_fingerprint_rejects_wrong_version(self, rtp_fingerprint):
        """Test RTP fingerprint rejects non-version-2 packets."""
        # Version 0 in high 2 bits
        raw_data = bytes([0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
                          0x00, 0x00, 0x00, 0x01])
        assert rtp_fingerprint.matches(raw_data) is False, (
            "RTP fingerprint should reject packets with wrong version"
        )

    def test_cti_csta_fingerprint_matches_xml_pattern(self):
        """Test CTI-CSTA fingerprint matches XML CSTA patterns."""
        fingerprint = BPOProtocolFingerprint.for_protocol(BPOProtocolType.CTI_CSTA)
        raw_data = b'<?xml version="1.0"?><CSTARequest xmlns="http://www.ecma.ch/standards/ecma-323/csta/ed3">'
        assert fingerprint.matches(raw_data) is True, (
            "CTI-CSTA fingerprint should match CSTA XML patterns"
        )

    def test_cti_csta_fingerprint_rejects_non_csta(self):
        """Test CTI-CSTA fingerprint rejects non-CSTA XML."""
        fingerprint = BPOProtocolFingerprint.for_protocol(BPOProtocolType.CTI_CSTA)
        raw_data = b'<?xml version="1.0"?><html><body>Hello</body></html>'
        assert fingerprint.matches(raw_data) is False, (
            "CTI-CSTA fingerprint should not match generic XML"
        )

    def test_ivr_vxml_fingerprint_matches_voicexml(self):
        """Test IVR-VXML fingerprint matches VoiceXML namespace."""
        fingerprint = BPOProtocolFingerprint.for_protocol(BPOProtocolType.IVR_VXML)
        raw_data = b'<?xml version="1.0"?><vxml xmlns="http://www.w3.org/2001/vxml" version="2.1">'
        assert fingerprint.matches(raw_data) is True, (
            "IVR-VXML fingerprint should match VoiceXML namespace"
        )

    def test_ivr_vxml_fingerprint_rejects_non_vxml(self):
        """Test IVR-VXML fingerprint rejects non-VoiceXML data."""
        fingerprint = BPOProtocolFingerprint.for_protocol(BPOProtocolType.IVR_VXML)
        raw_data = b'<?xml version="1.0"?><rss version="2.0"><channel></channel></rss>'
        assert fingerprint.matches(raw_data) is False, (
            "IVR-VXML fingerprint should not match non-VoiceXML XML"
        )

    @pytest.mark.parametrize("protocol_type", list(BPOProtocolType))
    def test_all_protocols_have_fingerprint(self, protocol_type):
        """Test that every protocol type has a corresponding fingerprint."""
        fingerprint = BPOProtocolFingerprint.for_protocol(protocol_type)
        assert fingerprint is not None, (
            f"Protocol type {protocol_type.name} should have a fingerprint"
        )

    def test_fingerprint_has_confidence_range(self, sip_fingerprint):
        """Test fingerprint confidence is within valid range."""
        raw_data = b"INVITE sip:user@host SIP/2.0\r\n"
        result = sip_fingerprint.match_with_confidence(raw_data)
        assert 0.0 <= result.confidence <= 1.0, (
            "Fingerprint confidence must be between 0.0 and 1.0"
        )


# ---------------------------------------------------------------------------
# BPODiscoveryProfile Creation and Methods
# ---------------------------------------------------------------------------

class TestBPODiscoveryProfile:
    """Tests for BPODiscoveryProfile creation and classification methods."""

    def test_profile_creation(self, default_profile):
        """Test BPODiscoveryProfile can be instantiated."""
        assert default_profile is not None
        assert isinstance(default_profile, BPODiscoveryProfile)

    def test_get_protocol_hints_sip_port(self, default_profile):
        """Test get_protocol_hints returns SIP hint for port 5060."""
        hints = default_profile.get_protocol_hints(port=5060)
        protocol_types = [h.protocol_type for h in hints]
        assert BPOProtocolType.SIP in protocol_types, (
            "Port 5060 should hint at SIP protocol"
        )

    def test_get_protocol_hints_tn3270e_port(self, default_profile):
        """Test get_protocol_hints returns TN3270e hint for port 23."""
        hints = default_profile.get_protocol_hints(port=23)
        protocol_types = [h.protocol_type for h in hints]
        assert BPOProtocolType.TN3270E in protocol_types, (
            "Port 23 should hint at TN3270e protocol"
        )

    def test_get_protocol_hints_mgcp_port(self, default_profile):
        """Test get_protocol_hints returns MGCP hint for port 2427."""
        hints = default_profile.get_protocol_hints(port=2427)
        protocol_types = [h.protocol_type for h in hints]
        assert BPOProtocolType.MGCP in protocol_types, (
            "Port 2427 should hint at MGCP protocol"
        )

    def test_get_protocol_hints_sip_tls_port(self, default_profile):
        """Test get_protocol_hints returns SIP hint for port 5061."""
        hints = default_profile.get_protocol_hints(port=5061)
        protocol_types = [h.protocol_type for h in hints]
        assert any(
            pt in (BPOProtocolType.SIP, BPOProtocolType.SIP_PQC)
            for pt in protocol_types
        ), "Port 5061 should hint at SIP or SIP-PQC protocol"

    def test_classify_sip_from_raw_bytes(self, default_profile):
        """Test classify_bpo_protocol correctly identifies SIP."""
        raw_data = (
            b"INVITE sip:user@host SIP/2.0\r\n"
            b"Via: SIP/2.0/TLS 10.0.0.1:5061\r\n"
            b"From: <sip:agent@pbx.example.com>;tag=abc123\r\n"
            b"To: <sip:customer@external.com>\r\n"
            b"Call-ID: 12345@10.0.0.1\r\n"
            b"CSeq: 1 INVITE\r\n\r\n"
        )
        result = default_profile.classify_bpo_protocol(raw_data)
        assert result.protocol_type == BPOProtocolType.SIP, (
            "Should classify SIP INVITE as SIP protocol"
        )

    def test_classify_tn3270e_from_binary_data(self, default_profile):
        """Test classify_bpo_protocol correctly identifies TN3270e."""
        # TN3270e negotiation: IAC DO TN3270E
        raw_data = bytes([0xFF, 0xFD, 0x28]) + b"\x00\x01\x02\x03"
        result = default_profile.classify_bpo_protocol(raw_data)
        assert result.protocol_type == BPOProtocolType.TN3270E, (
            "Should classify IAC DO TN3270E as TN3270e protocol"
        )

    def test_classify_returns_confidence(self, default_profile):
        """Test classify_bpo_protocol returns a confidence score."""
        raw_data = b"INVITE sip:user@host SIP/2.0\r\n\r\n"
        result = default_profile.classify_bpo_protocol(raw_data)
        assert hasattr(result, "confidence")
        assert 0.0 <= result.confidence <= 1.0, (
            "Classification confidence must be between 0.0 and 1.0"
        )

    def test_get_security_requirements_sip(self, default_profile):
        """Test get_security_requirements returns PCI-DSS for SIP voice channels."""
        requirements = default_profile.get_security_requirements(BPOProtocolType.SIP)
        requirement_names = [r.name if hasattr(r, "name") else str(r) for r in requirements]
        assert any("PCI" in name.upper() for name in requirement_names), (
            "SIP voice channels should have PCI-DSS security requirements"
        )

    def test_get_compliance_mapping_ivr(self, default_profile):
        """Test get_compliance_mapping maps IVR to TCPA requirements."""
        mapping = default_profile.get_compliance_mapping(BPOProtocolType.IVR_VXML)
        framework_names = [
            m.framework if hasattr(m, "framework") else str(m) for m in mapping
        ]
        assert any("TCPA" in name.upper() for name in framework_names), (
            "IVR protocols should map to TCPA compliance requirements"
        )

    def test_get_pqc_recommendation_voice_low_latency(self, default_profile):
        """Test get_pqc_recommendation recommends low-latency PQC for voice."""
        recommendation = default_profile.get_pqc_recommendation(BPOProtocolType.SIP)
        kem_name = recommendation.kem_algorithm
        assert any(algo in kem_name for algo in ("ML-KEM-512", "Falcon-512")), (
            "Voice protocols should recommend low-latency PQC (ML-KEM-512 or Falcon-512)"
        )

    def test_get_pqc_recommendation_rtp(self, default_profile):
        """Test get_pqc_recommendation recommends low-latency PQC for RTP."""
        recommendation = default_profile.get_pqc_recommendation(BPOProtocolType.RTP)
        kem_name = recommendation.kem_algorithm
        assert "ML-KEM-512" in kem_name or "Falcon" in kem_name, (
            "RTP should recommend low-latency PQC for real-time media"
        )

    def test_get_pqc_recommendation_terminal(self, default_profile):
        """Test get_pqc_recommendation recommends ML-KEM-768 for terminal sessions."""
        recommendation = default_profile.get_pqc_recommendation(BPOProtocolType.TN3270E)
        assert "ML-KEM-768" in recommendation.kem_algorithm, (
            "Terminal sessions should recommend ML-KEM-768 for balanced security"
        )

    def test_classify_unknown_data(self, default_profile):
        """Test classify_bpo_protocol handles unknown data gracefully."""
        raw_data = b"\x00\x01\x02\x03\x04\x05\x06\x07"
        result = default_profile.classify_bpo_protocol(raw_data)
        assert result.confidence < 0.5, (
            "Unknown data should yield low classification confidence"
        )


# ---------------------------------------------------------------------------
# Discovery Configuration
# ---------------------------------------------------------------------------

class TestBPODiscoveryConfig:
    """Tests for discovery configuration parameters."""

    def test_voice_protocols_tight_timeouts(self, default_profile):
        """Test voice protocols have tighter timeouts (< 200ms)."""
        sip_config = default_profile.get_discovery_config(BPOProtocolType.SIP)
        assert sip_config.timeout_ms < 200, (
            "Voice protocol (SIP) discovery timeout should be < 200ms"
        )

    def test_rtp_tight_timeout(self, default_profile):
        """Test RTP has tight discovery timeout."""
        rtp_config = default_profile.get_discovery_config(BPOProtocolType.RTP)
        assert rtp_config.timeout_ms < 200, (
            "RTP discovery timeout should be < 200ms for real-time constraints"
        )

    def test_signaling_protocols_high_confidence(self, default_profile):
        """Test signaling protocols have higher confidence thresholds."""
        cti_config = default_profile.get_discovery_config(BPOProtocolType.CTI_CSTA)
        assert cti_config.min_confidence >= 0.7, (
            "Signaling protocol confidence threshold should be >= 0.7"
        )

    def test_terminal_protocols_larger_sample_size(self, default_profile):
        """Test terminal protocols allow larger sample sizes."""
        tn3270e_config = default_profile.get_discovery_config(BPOProtocolType.TN3270E)
        sip_config = default_profile.get_discovery_config(BPOProtocolType.SIP)
        assert tn3270e_config.max_sample_bytes >= sip_config.max_sample_bytes, (
            "Terminal protocols should allow larger sample sizes than voice protocols"
        )

    def test_srtp_timeout_similar_to_rtp(self, default_profile):
        """Test SRTP discovery timeout is similar to RTP."""
        srtp_config = default_profile.get_discovery_config(BPOProtocolType.SRTP)
        rtp_config = default_profile.get_discovery_config(BPOProtocolType.RTP)
        assert abs(srtp_config.timeout_ms - rtp_config.timeout_ms) < 50, (
            "SRTP timeout should be close to RTP timeout"
        )

    def test_ivr_moderate_timeout(self, default_profile):
        """Test IVR protocols have moderate discovery timeouts."""
        vxml_config = default_profile.get_discovery_config(BPOProtocolType.IVR_VXML)
        assert 100 <= vxml_config.timeout_ms <= 500, (
            "IVR discovery timeout should be moderate (100-500ms)"
        )


# ---------------------------------------------------------------------------
# Environment-Specific Profiles
# ---------------------------------------------------------------------------

class TestBPOEnvironmentProfiles:
    """Tests for environment-specific discovery profile factory methods."""

    def test_on_premise_includes_tn3270e(self, on_premise_profile):
        """Test ON_PREMISE environment includes TN3270e."""
        supported = on_premise_profile.supported_protocols
        assert BPOProtocolType.TN3270E in supported, (
            "On-premise environment should include TN3270e for legacy mainframe access"
        )

    def test_on_premise_includes_sip(self, on_premise_profile):
        """Test ON_PREMISE environment includes SIP."""
        supported = on_premise_profile.supported_protocols
        assert BPOProtocolType.SIP in supported, (
            "On-premise environment should include SIP"
        )

    def test_on_premise_includes_cti(self, on_premise_profile):
        """Test ON_PREMISE environment includes CTI protocols."""
        supported = on_premise_profile.supported_protocols
        assert BPOProtocolType.CTI_CSTA in supported, (
            "On-premise environment should include CTI-CSTA"
        )

    def test_cloud_ccaas_prioritizes_sip_pqc(self, cloud_profile):
        """Test CLOUD_CCaaS environment prioritizes SIP-PQC."""
        supported = cloud_profile.supported_protocols
        assert BPOProtocolType.SIP_PQC in supported, (
            "Cloud CCaaS should support SIP-PQC"
        )
        # SIP-PQC should be prioritized (appear before plain SIP in priority list)
        priority = cloud_profile.get_protocol_priority()
        pqc_idx = priority.index(BPOProtocolType.SIP_PQC)
        if BPOProtocolType.SIP in priority:
            sip_idx = priority.index(BPOProtocolType.SIP)
            assert pqc_idx < sip_idx, (
                "Cloud CCaaS should prioritize SIP-PQC over plain SIP"
            )

    def test_cloud_ccaas_includes_srtp_pqc(self, cloud_profile):
        """Test CLOUD_CCaaS environment includes SRTP-PQC."""
        supported = cloud_profile.supported_protocols
        assert BPOProtocolType.SRTP_PQC in supported, (
            "Cloud CCaaS should support SRTP-PQC for quantum-safe voice"
        )

    def test_remote_workforce_enables_vpnless_tunnel(self, remote_profile):
        """Test REMOTE_WORKFORCE enables VPN-less tunnel discovery."""
        features = remote_profile.get_enabled_features()
        feature_names = [f if isinstance(f, str) else f.name for f in features]
        assert any("vpn" in name.lower() or "tunnel" in name.lower()
                    for name in feature_names), (
            "Remote workforce should enable VPN-less tunnel discovery"
        )

    def test_remote_workforce_includes_sip(self, remote_profile):
        """Test REMOTE_WORKFORCE includes SIP for remote agents."""
        supported = remote_profile.supported_protocols
        assert BPOProtocolType.SIP in supported or BPOProtocolType.SIP_PQC in supported, (
            "Remote workforce should include SIP or SIP-PQC"
        )

    def test_hybrid_includes_all_protocol_types(self, hybrid_profile):
        """Test HYBRID environment includes all protocol types."""
        supported = hybrid_profile.supported_protocols
        # Hybrid should include both legacy (TN3270e) and modern (SIP-PQC)
        assert BPOProtocolType.TN3270E in supported, (
            "Hybrid should include TN3270e for legacy systems"
        )
        assert BPOProtocolType.SIP in supported or BPOProtocolType.SIP_PQC in supported, (
            "Hybrid should include SIP or SIP-PQC"
        )
        assert BPOProtocolType.CTI_CSTA in supported, (
            "Hybrid should include CTI-CSTA"
        )
        assert BPOProtocolType.IVR_VXML in supported, (
            "Hybrid should include IVR-VXML"
        )

    def test_hybrid_has_most_protocols(self, hybrid_profile, on_premise_profile,
                                       cloud_profile, remote_profile):
        """Test HYBRID environment supports the most protocols."""
        hybrid_count = len(hybrid_profile.supported_protocols)
        on_prem_count = len(on_premise_profile.supported_protocols)
        cloud_count = len(cloud_profile.supported_protocols)
        remote_count = len(remote_profile.supported_protocols)
        assert hybrid_count >= on_prem_count, (
            "Hybrid should support at least as many protocols as on-premise"
        )
        assert hybrid_count >= cloud_count, (
            "Hybrid should support at least as many protocols as cloud"
        )
        assert hybrid_count >= remote_count, (
            "Hybrid should support at least as many protocols as remote workforce"
        )

    @pytest.mark.parametrize("environment", [
        "ON_PREMISE",
        "CLOUD_CCaaS",
        "REMOTE_WORKFORCE",
        "HYBRID",
    ])
    def test_environment_profile_has_supported_protocols(self, environment):
        """Test that all environment profiles have supported protocols."""
        profile = BPODiscoveryProfile.for_environment(environment)
        assert profile is not None
        assert len(profile.supported_protocols) > 0, (
            f"{environment} profile should have at least one supported protocol"
        )

    @pytest.mark.parametrize("environment", [
        "ON_PREMISE",
        "CLOUD_CCaaS",
        "REMOTE_WORKFORCE",
        "HYBRID",
    ])
    def test_environment_profile_is_bpo_discovery_profile(self, environment):
        """Test that all environment profiles are BPODiscoveryProfile instances."""
        profile = BPODiscoveryProfile.for_environment(environment)
        assert isinstance(profile, BPODiscoveryProfile), (
            f"{environment} profile should be a BPODiscoveryProfile instance"
        )


# ---------------------------------------------------------------------------
# BPODiscoveryHint Structure
# ---------------------------------------------------------------------------

class TestBPODiscoveryHint:
    """Tests for BPODiscoveryHint structure."""

    def test_hint_has_protocol_type(self, default_profile):
        """Test discovery hint includes protocol type."""
        hints = default_profile.get_protocol_hints(port=5060)
        assert len(hints) > 0, "Should return at least one hint for port 5060"
        assert hints[0].protocol_type is not None

    def test_hint_has_confidence(self, default_profile):
        """Test discovery hint includes confidence score."""
        hints = default_profile.get_protocol_hints(port=5060)
        assert len(hints) > 0
        assert hasattr(hints[0], "confidence")
        assert 0.0 <= hints[0].confidence <= 1.0

    def test_hint_has_description(self, default_profile):
        """Test discovery hint includes a description."""
        hints = default_profile.get_protocol_hints(port=5060)
        assert len(hints) > 0
        assert hasattr(hints[0], "description")
        assert len(hints[0].description) > 0

    def test_unknown_port_returns_empty_or_low_confidence(self, default_profile):
        """Test unknown port returns empty hints or low-confidence results."""
        hints = default_profile.get_protocol_hints(port=99999)
        if len(hints) > 0:
            assert all(h.confidence < 0.3 for h in hints), (
                "Unknown port should yield only low-confidence hints"
            )
