"""
Tests for SIP Protocol Handler

Tests cover:
- SIPMethod enum
- SIPResponseCode properties (is_provisional, is_success, is_error)
- SIPSecurityProfile defaults and to_dict()
- Premium rate number lists
- SIP transport enum values
"""

import pytest

from ai_engine.domains.bpo.protocols.sip.sip_codes import (
    SIPMethod,
    SIPResponseCode,
    SIPHeaderName,
    SDPMediaType,
    SIPTransport,
    SIPSecurityProfile,
    CALL_CENTER_SIP_HEADERS,
    PREMIUM_RATE_PREFIXES,
    AGENT_ALLOWED_METHODS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_security_profile():
    """Return a default SIPSecurityProfile."""
    return SIPSecurityProfile()


# ---------------------------------------------------------------------------
# SIPMethod Enum
# ---------------------------------------------------------------------------

class TestSIPMethod:
    """Tests for SIPMethod enumeration."""

    def test_core_methods_exist(self):
        """Test core SIP methods (RFC 3261) are defined."""
        assert SIPMethod.INVITE.method == "INVITE"
        assert SIPMethod.ACK.method == "ACK"
        assert SIPMethod.BYE.method == "BYE"
        assert SIPMethod.CANCEL.method == "CANCEL"
        assert SIPMethod.REGISTER.method == "REGISTER"
        assert SIPMethod.OPTIONS.method == "OPTIONS"

    def test_extension_methods_exist(self):
        """Test SIP extension methods are defined."""
        assert SIPMethod.REFER.method == "REFER"
        assert SIPMethod.SUBSCRIBE.method == "SUBSCRIBE"
        assert SIPMethod.NOTIFY.method == "NOTIFY"
        assert SIPMethod.MESSAGE.method == "MESSAGE"
        assert SIPMethod.INFO.method == "INFO"
        assert SIPMethod.UPDATE.method == "UPDATE"
        assert SIPMethod.PRACK.method == "PRACK"
        assert SIPMethod.PUBLISH.method == "PUBLISH"

    @pytest.mark.parametrize("method", list(SIPMethod))
    def test_all_methods_have_description(self, method):
        """Test that all SIP methods have descriptions."""
        assert method.description is not None
        assert len(method.description) > 0

    @pytest.mark.parametrize("method", list(SIPMethod))
    def test_all_methods_have_method_string(self, method):
        """Test that all SIP methods have a method string."""
        assert method.method is not None
        assert method.method == method.method.upper(), (
            f"SIP method {method.method} should be uppercase"
        )


# ---------------------------------------------------------------------------
# SIPResponseCode Properties
# ---------------------------------------------------------------------------

class TestSIPResponseCode:
    """Tests for SIPResponseCode enumeration."""

    def test_provisional_responses(self):
        """Test 1xx responses are provisional."""
        assert SIPResponseCode.TRYING.is_provisional is True
        assert SIPResponseCode.RINGING.is_provisional is True
        assert SIPResponseCode.SESSION_PROGRESS.is_provisional is True

    def test_provisional_not_success(self):
        """Test 1xx responses are not success."""
        assert SIPResponseCode.TRYING.is_success is False
        assert SIPResponseCode.RINGING.is_success is False

    def test_provisional_not_error(self):
        """Test 1xx responses are not errors."""
        assert SIPResponseCode.TRYING.is_error is False
        assert SIPResponseCode.RINGING.is_error is False

    def test_success_responses(self):
        """Test 2xx responses are success."""
        assert SIPResponseCode.OK.is_success is True
        assert SIPResponseCode.ACCEPTED.is_success is True

    def test_success_not_provisional(self):
        """Test 2xx responses are not provisional."""
        assert SIPResponseCode.OK.is_provisional is False

    def test_success_not_error(self):
        """Test 2xx responses are not errors."""
        assert SIPResponseCode.OK.is_error is False

    def test_client_error_responses(self):
        """Test 4xx responses are errors."""
        assert SIPResponseCode.BAD_REQUEST.is_error is True
        assert SIPResponseCode.UNAUTHORIZED.is_error is True
        assert SIPResponseCode.FORBIDDEN.is_error is True
        assert SIPResponseCode.NOT_FOUND.is_error is True
        assert SIPResponseCode.BUSY_HERE.is_error is True

    def test_server_error_responses(self):
        """Test 5xx responses are errors."""
        assert SIPResponseCode.SERVER_INTERNAL_ERROR.is_error is True
        assert SIPResponseCode.SERVICE_UNAVAILABLE.is_error is True

    def test_global_failure_responses(self):
        """Test 6xx responses are errors."""
        assert SIPResponseCode.BUSY_EVERYWHERE.is_error is True
        assert SIPResponseCode.DECLINE.is_error is True

    def test_error_not_provisional(self):
        """Test error responses are not provisional."""
        assert SIPResponseCode.BAD_REQUEST.is_provisional is False
        assert SIPResponseCode.SERVER_INTERNAL_ERROR.is_provisional is False

    def test_error_not_success(self):
        """Test error responses are not success."""
        assert SIPResponseCode.BAD_REQUEST.is_success is False
        assert SIPResponseCode.SERVER_INTERNAL_ERROR.is_success is False

    @pytest.mark.parametrize("response,expected_code", [
        (SIPResponseCode.TRYING, 100),
        (SIPResponseCode.RINGING, 180),
        (SIPResponseCode.OK, 200),
        (SIPResponseCode.BAD_REQUEST, 400),
        (SIPResponseCode.NOT_FOUND, 404),
        (SIPResponseCode.SERVER_INTERNAL_ERROR, 500),
        (SIPResponseCode.BUSY_EVERYWHERE, 600),
    ])
    def test_response_codes(self, response, expected_code):
        """Test response code values."""
        assert response.code == expected_code

    @pytest.mark.parametrize("response", list(SIPResponseCode))
    def test_all_responses_have_reason(self, response):
        """Test all responses have a reason phrase."""
        assert response.reason is not None
        assert len(response.reason) > 0

    @pytest.mark.parametrize("response", list(SIPResponseCode))
    def test_all_responses_have_category(self, response):
        """Test all responses have a category."""
        assert response.category in (
            "provisional", "success", "redirection",
            "client_error", "server_error", "global_failure"
        )

    def test_call_center_specific_codes(self):
        """Test call center relevant response codes."""
        # Queued is important for ACD systems
        assert SIPResponseCode.QUEUED.code == 182
        assert SIPResponseCode.QUEUED.is_provisional is True

        # Call Being Forwarded for transfers
        assert SIPResponseCode.CALL_BEING_FORWARDED.code == 181
        assert SIPResponseCode.CALL_BEING_FORWARDED.is_provisional is True

        # Busy is important for agent availability
        assert SIPResponseCode.BUSY_HERE.code == 486
        assert SIPResponseCode.BUSY_HERE.is_error is True

        # Request Terminated for cancelled calls
        assert SIPResponseCode.REQUEST_TERMINATED.code == 487


# ---------------------------------------------------------------------------
# SIPSecurityProfile
# ---------------------------------------------------------------------------

class TestSIPSecurityProfile:
    """Tests for SIPSecurityProfile defaults and serialization."""

    def test_default_tls_required(self, default_security_profile):
        """Test TLS is required by default."""
        assert default_security_profile.require_tls is True

    def test_default_min_tls_version(self, default_security_profile):
        """Test minimum TLS version is 1.3."""
        assert default_security_profile.min_tls_version == "TLS 1.3"

    def test_default_pqc_tls_required(self, default_security_profile):
        """Test PQC TLS is required by default."""
        assert default_security_profile.require_pqc_tls is True

    def test_default_digest_auth(self, default_security_profile):
        """Test digest authentication is required."""
        assert default_security_profile.require_digest_auth is True

    def test_default_srtp_required(self, default_security_profile):
        """Test SRTP is required by default."""
        assert default_security_profile.require_srtp is True
        assert default_security_profile.require_srtp_pqc is True

    def test_default_call_center_settings(self, default_security_profile):
        """Test call center specific defaults."""
        assert default_security_profile.allow_transfer is True
        assert default_security_profile.allow_forward is True
        assert default_security_profile.max_concurrent_calls == 10000
        assert default_security_profile.max_call_duration_seconds == 14400  # 4 hours

    def test_default_fraud_prevention(self, default_security_profile):
        """Test toll fraud prevention defaults."""
        assert default_security_profile.block_premium_rate is True
        assert default_security_profile.rate_limit_per_second == 100

    def test_to_dict_structure(self, default_security_profile):
        """Test to_dict returns expected structure."""
        d = default_security_profile.to_dict()
        assert "transport" in d
        assert "authentication" in d
        assert "media" in d
        assert "fraud_prevention" in d

    def test_to_dict_transport(self, default_security_profile):
        """Test to_dict transport section."""
        d = default_security_profile.to_dict()
        assert d["transport"]["require_tls"] is True
        assert d["transport"]["min_tls_version"] == "TLS 1.3"
        assert d["transport"]["require_pqc_tls"] is True

    def test_to_dict_authentication(self, default_security_profile):
        """Test to_dict authentication section."""
        d = default_security_profile.to_dict()
        assert d["authentication"]["require_digest_auth"] is True
        assert d["authentication"]["auth_algorithm"] == "SHA-256"

    def test_to_dict_media(self, default_security_profile):
        """Test to_dict media section."""
        d = default_security_profile.to_dict()
        assert d["media"]["require_srtp"] is True
        assert d["media"]["require_srtp_pqc"] is True

    def test_to_dict_fraud_prevention(self, default_security_profile):
        """Test to_dict fraud prevention section."""
        d = default_security_profile.to_dict()
        assert d["fraud_prevention"]["block_premium_rate"] is True
        assert d["fraud_prevention"]["rate_limit_per_second"] == 100

    def test_custom_security_profile(self):
        """Test creating a custom security profile."""
        profile = SIPSecurityProfile(
            require_tls=True,
            min_tls_version="TLS 1.3",
            require_pqc_tls=True,
            require_mutual_tls=True,
            block_premium_rate=True,
            block_international=True,
            rate_limit_per_second=50,
        )
        assert profile.require_mutual_tls is True
        assert profile.block_international is True
        assert profile.rate_limit_per_second == 50


# ---------------------------------------------------------------------------
# Premium Rate Number Lists
# ---------------------------------------------------------------------------

class TestPremiumRatePrefixes:
    """Tests for premium rate number prefix lists."""

    def test_us_premium_rate_prefixes(self):
        """Test US premium rate prefixes are defined."""
        assert "US" in PREMIUM_RATE_PREFIXES
        assert "900" in PREMIUM_RATE_PREFIXES["US"]
        assert "976" in PREMIUM_RATE_PREFIXES["US"]

    def test_uk_premium_rate_prefixes(self):
        """Test UK premium rate prefixes are defined."""
        assert "UK" in PREMIUM_RATE_PREFIXES
        assert "09" in PREMIUM_RATE_PREFIXES["UK"]

    def test_eu_premium_rate_prefixes(self):
        """Test EU premium rate prefixes are defined."""
        assert "EU" in PREMIUM_RATE_PREFIXES
        assert len(PREMIUM_RATE_PREFIXES["EU"]) > 0

    def test_irsf_prefixes(self):
        """Test International Revenue Share Fraud prefixes are defined."""
        assert "IRSF" in PREMIUM_RATE_PREFIXES
        assert "882" in PREMIUM_RATE_PREFIXES["IRSF"]
        assert "883" in PREMIUM_RATE_PREFIXES["IRSF"]

    def test_all_prefixes_are_strings(self):
        """Test all prefix values are strings."""
        for region, prefixes in PREMIUM_RATE_PREFIXES.items():
            assert isinstance(prefixes, list), f"{region} prefixes should be a list"
            for prefix in prefixes:
                assert isinstance(prefix, str), (
                    f"Prefix {prefix} in {region} should be a string"
                )


# ---------------------------------------------------------------------------
# SIP Transport
# ---------------------------------------------------------------------------

class TestSIPTransport:
    """Tests for SIPTransport enumeration."""

    def test_udp_transport(self):
        """Test UDP transport properties."""
        assert SIPTransport.UDP.transport == "UDP"
        assert SIPTransport.UDP.default_port == 5060
        assert SIPTransport.UDP.encrypted is False

    def test_tcp_transport(self):
        """Test TCP transport properties."""
        assert SIPTransport.TCP.transport == "TCP"
        assert SIPTransport.TCP.default_port == 5060
        assert SIPTransport.TCP.encrypted is False

    def test_tls_transport(self):
        """Test TLS transport properties."""
        assert SIPTransport.TLS.transport == "TLS"
        assert SIPTransport.TLS.default_port == 5061
        assert SIPTransport.TLS.encrypted is True

    def test_wss_transport(self):
        """Test WebSocket Secure transport properties."""
        assert SIPTransport.WSS.transport == "WSS"
        assert SIPTransport.WSS.default_port == 443
        assert SIPTransport.WSS.encrypted is True

    def test_tls_pqc_transport(self):
        """Test quantum-safe TLS transport properties."""
        assert SIPTransport.TLS_PQC.transport == "TLS-PQC"
        assert SIPTransport.TLS_PQC.encrypted is True

    @pytest.mark.parametrize("transport", list(SIPTransport))
    def test_all_transports_have_port(self, transport):
        """Test all transports have a default port."""
        assert transport.default_port > 0

    def test_encrypted_transports(self):
        """Test only secure transports are marked encrypted."""
        unencrypted = [t for t in SIPTransport if not t.encrypted]
        encrypted = [t for t in SIPTransport if t.encrypted]
        assert len(unencrypted) >= 2  # UDP, TCP
        assert len(encrypted) >= 3   # TLS, WSS, TLS-PQC


# ---------------------------------------------------------------------------
# SIP Headers
# ---------------------------------------------------------------------------

class TestSIPHeaders:
    """Tests for SIP header definitions."""

    def test_core_headers_exist(self):
        """Test core SIP headers are defined."""
        assert SIPHeaderName.VIA is not None
        assert SIPHeaderName.FROM is not None
        assert SIPHeaderName.TO is not None
        assert SIPHeaderName.CALL_ID is not None
        assert SIPHeaderName.CSEQ is not None

    def test_required_headers(self):
        """Test mandatory headers are marked as required."""
        assert SIPHeaderName.VIA.required is True
        assert SIPHeaderName.FROM.required is True
        assert SIPHeaderName.TO.required is True
        assert SIPHeaderName.CALL_ID.required is True

    def test_compact_forms(self):
        """Test compact forms for key headers."""
        assert SIPHeaderName.VIA.compact_form == "v"
        assert SIPHeaderName.FROM.compact_form == "f"
        assert SIPHeaderName.TO.compact_form == "t"
        assert SIPHeaderName.CALL_ID.compact_form == "i"

    def test_call_center_headers(self):
        """Test call center specific SIP headers are defined."""
        assert "X-CC-Queue" in CALL_CENTER_SIP_HEADERS
        assert "X-CC-Agent" in CALL_CENTER_SIP_HEADERS
        assert "X-CC-Tenant" in CALL_CENTER_SIP_HEADERS
        assert "X-CC-PCI-Mode" in CALL_CENTER_SIP_HEADERS

    def test_sdp_media_types(self):
        """Test SDP media types are defined."""
        assert SDPMediaType.AUDIO.media_type == "audio"
        assert SDPMediaType.VIDEO.media_type == "video"
        assert SDPMediaType.APPLICATION.media_type == "application"


# ---------------------------------------------------------------------------
# Agent Allowed Methods
# ---------------------------------------------------------------------------

class TestAgentAllowedMethods:
    """Tests for agent-allowed SIP methods list."""

    def test_invite_allowed(self):
        """Test agents can initiate calls (INVITE)."""
        assert SIPMethod.INVITE in AGENT_ALLOWED_METHODS

    def test_bye_allowed(self):
        """Test agents can end calls (BYE)."""
        assert SIPMethod.BYE in AGENT_ALLOWED_METHODS

    def test_refer_allowed(self):
        """Test agents can transfer calls (REFER)."""
        assert SIPMethod.REFER in AGENT_ALLOWED_METHODS

    def test_info_allowed(self):
        """Test agents can send DTMF relay (INFO)."""
        assert SIPMethod.INFO in AGENT_ALLOWED_METHODS

    def test_register_not_allowed(self):
        """Test agents cannot register endpoints."""
        assert SIPMethod.REGISTER not in AGENT_ALLOWED_METHODS

    def test_subscribe_not_allowed(self):
        """Test agents cannot subscribe to events."""
        assert SIPMethod.SUBSCRIBE not in AGENT_ALLOWED_METHODS
