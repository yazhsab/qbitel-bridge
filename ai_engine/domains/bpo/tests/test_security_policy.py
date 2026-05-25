"""
Tests for BPO Security Policy Module

Tests cover:
- SecurityLevel enum
- DataClassification enum
- VoiceSecurityPolicy defaults
- AgentSecurityPolicy defaults
- BPOSecurityPolicy validation (PCI-DSS, TCPA, HIPAA rules)
- for_financial_services_bpo() factory
- for_healthcare_bpo() factory
- for_general_customer_service() factory
- for_remote_workforce() factory
- Policy serialization (to_dict)
"""

import pytest
from datetime import datetime, timedelta

from ai_engine.domains.bpo.core.security_policy import (
    SecurityLevel,
    DataClassification,
    BPOComplianceRequirement,
    VoiceSecurityPolicy,
    AgentSecurityPolicy,
    EncryptionPolicy,
    AuditPolicy,
    MultiTenantPolicy,
    BPOSecurityPolicy,
    BPO_POLICIES,
    get_policy,
    list_policies,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_policy():
    """Return a default BPOSecurityPolicy."""
    return BPOSecurityPolicy()


@pytest.fixture
def financial_policy():
    """Return financial services BPO policy."""
    return BPOSecurityPolicy.for_financial_services_bpo()


@pytest.fixture
def healthcare_policy():
    """Return healthcare BPO policy."""
    return BPOSecurityPolicy.for_healthcare_bpo()


@pytest.fixture
def customer_service_policy():
    """Return general customer service BPO policy."""
    return BPOSecurityPolicy.for_general_customer_service()


@pytest.fixture
def remote_workforce_policy():
    """Return remote workforce BPO policy."""
    return BPOSecurityPolicy.for_remote_workforce()


# ---------------------------------------------------------------------------
# SecurityLevel Enum
# ---------------------------------------------------------------------------

class TestSecurityLevel:
    """Tests for SecurityLevel enumeration."""

    def test_standard_level(self):
        """Test STANDARD security level properties."""
        level = SecurityLevel.STANDARD
        assert level.level == 1
        assert level.display_name == "Standard"
        assert level.min_key_bits == 128

    def test_enhanced_level(self):
        """Test ENHANCED security level properties."""
        level = SecurityLevel.ENHANCED
        assert level.level == 2
        assert level.display_name == "Enhanced"
        assert level.min_key_bits == 192

    def test_critical_level(self):
        """Test CRITICAL security level properties."""
        level = SecurityLevel.CRITICAL
        assert level.level == 3
        assert level.display_name == "Critical"
        assert level.min_key_bits == 256

    def test_restricted_level(self):
        """Test RESTRICTED security level properties."""
        level = SecurityLevel.RESTRICTED
        assert level.level == 4
        assert level.display_name == "Restricted"
        assert level.min_key_bits == 256

    def test_level_ordering(self):
        """Test that security levels are ordered correctly."""
        assert SecurityLevel.STANDARD.level < SecurityLevel.ENHANCED.level
        assert SecurityLevel.ENHANCED.level < SecurityLevel.CRITICAL.level
        assert SecurityLevel.CRITICAL.level < SecurityLevel.RESTRICTED.level


# ---------------------------------------------------------------------------
# DataClassification Enum
# ---------------------------------------------------------------------------

class TestDataClassification:
    """Tests for DataClassification enumeration."""

    def test_public_classification(self):
        """Test PUBLIC data classification."""
        dc = DataClassification.PUBLIC
        assert dc.level == 0
        assert dc.requires_encryption is False
        assert dc.requires_hsm is False

    def test_confidential_classification(self):
        """Test CONFIDENTIAL data classification."""
        dc = DataClassification.CONFIDENTIAL
        assert dc.level == 2
        assert dc.requires_encryption is True
        assert dc.requires_hsm is False

    def test_restricted_classification(self):
        """Test RESTRICTED data classification."""
        dc = DataClassification.RESTRICTED
        assert dc.level == 3
        assert dc.requires_encryption is True
        assert dc.requires_hsm is True

    def test_top_secret_classification(self):
        """Test TOP_SECRET data classification."""
        dc = DataClassification.TOP_SECRET
        assert dc.level == 4
        assert dc.requires_encryption is True
        assert dc.requires_hsm is True
        assert dc.min_retention_years == 10

    def test_classification_ordering(self):
        """Test data classification levels are ordered."""
        assert DataClassification.PUBLIC.level < DataClassification.INTERNAL.level
        assert DataClassification.INTERNAL.level < DataClassification.CONFIDENTIAL.level
        assert DataClassification.CONFIDENTIAL.level < DataClassification.RESTRICTED.level
        assert DataClassification.RESTRICTED.level < DataClassification.TOP_SECRET.level

    @pytest.mark.parametrize("classification,expected_encryption", [
        (DataClassification.PUBLIC, False),
        (DataClassification.INTERNAL, False),
        (DataClassification.CONFIDENTIAL, True),
        (DataClassification.RESTRICTED, True),
        (DataClassification.TOP_SECRET, True),
    ])
    def test_encryption_requirements(self, classification, expected_encryption):
        """Test encryption requirements per classification."""
        assert classification.requires_encryption == expected_encryption


# ---------------------------------------------------------------------------
# VoiceSecurityPolicy Defaults
# ---------------------------------------------------------------------------

class TestVoiceSecurityPolicy:
    """Tests for VoiceSecurityPolicy defaults."""

    def test_default_recording_enabled(self):
        """Test that call recording is enabled by default."""
        policy = VoiceSecurityPolicy()
        assert policy.record_all_calls is True
        assert policy.encrypt_recordings is True

    def test_default_dtmf_masking(self):
        """Test DTMF masking defaults."""
        policy = VoiceSecurityPolicy()
        assert policy.enable_dtmf_masking is True
        assert policy.dtmf_masking_mode == "CLAMP"
        assert policy.mask_dtmf_in_recording is True
        assert policy.mask_dtmf_in_screen is True

    def test_default_pause_resume(self):
        """Test pause/resume defaults for PCI."""
        policy = VoiceSecurityPolicy()
        assert policy.enable_pause_resume is True
        assert policy.auto_pause_on_payment is True
        assert policy.auto_resume_timeout_seconds == 120

    def test_default_encryption(self):
        """Test voice encryption defaults."""
        policy = VoiceSecurityPolicy()
        assert policy.encrypt_voice_media is True
        assert policy.require_srtp is True
        assert "PQC" in policy.srtp_profile

    def test_default_toll_fraud(self):
        """Test toll fraud prevention defaults."""
        policy = VoiceSecurityPolicy()
        assert policy.enable_toll_fraud_detection is True
        assert policy.premium_rate_blocking is True
        assert policy.max_call_duration_hours == 4

    def test_default_biometrics_disabled(self):
        """Test voice biometrics disabled by default."""
        policy = VoiceSecurityPolicy()
        assert policy.enable_voice_biometrics is False


# ---------------------------------------------------------------------------
# AgentSecurityPolicy Defaults
# ---------------------------------------------------------------------------

class TestAgentSecurityPolicy:
    """Tests for AgentSecurityPolicy defaults."""

    def test_default_mfa_required(self):
        """Test MFA is required by default."""
        policy = AgentSecurityPolicy()
        assert policy.require_mfa is True
        assert len(policy.mfa_methods) >= 2

    def test_default_session_management(self):
        """Test session management defaults."""
        policy = AgentSecurityPolicy()
        assert policy.max_session_duration_hours == 10
        assert policy.idle_timeout_minutes == 5
        assert policy.concurrent_sessions_allowed == 1
        assert policy.session_lock_on_idle is True

    def test_default_desktop_security(self):
        """Test desktop security defaults."""
        policy = AgentSecurityPolicy()
        assert policy.block_clipboard is True
        assert policy.block_screen_capture is True
        assert policy.block_usb_storage is True
        assert policy.watermark_screen is True

    def test_default_data_masking(self):
        """Test data masking defaults."""
        policy = AgentSecurityPolicy()
        assert policy.mask_pii_on_screen is True
        assert policy.mask_credit_card is True
        assert policy.mask_ssn is True
        assert policy.auto_clear_after_call is True

    def test_default_remote_agent(self):
        """Test remote agent defaults."""
        policy = AgentSecurityPolicy()
        assert policy.require_endpoint_compliance is True
        assert policy.require_antivirus is True
        assert policy.require_disk_encryption is True
        assert policy.home_network_requirements == "WPA3"


# ---------------------------------------------------------------------------
# BPOSecurityPolicy Validation
# ---------------------------------------------------------------------------

class TestBPOSecurityPolicyValidation:
    """Tests for BPOSecurityPolicy validation rules."""

    def test_default_policy_is_valid(self, default_policy):
        """Test that default policy passes validation."""
        errors = default_policy.validate()
        assert errors == [], f"Default policy should be valid, got: {errors}"

    def test_pci_dss_requires_dtmf_masking(self):
        """Test PCI-DSS validation requires DTMF masking."""
        policy = BPOSecurityPolicy(
            compliance_frameworks={BPOComplianceRequirement.PCI_DSS_4_0},
            voice_security=VoiceSecurityPolicy(enable_dtmf_masking=False),
        )
        errors = policy.validate()
        assert any("dtmf" in e.lower() for e in errors), (
            "PCI-DSS should require DTMF masking"
        )

    def test_pci_dss_requires_encrypted_recordings(self):
        """Test PCI-DSS validation requires encrypted recordings."""
        policy = BPOSecurityPolicy(
            compliance_frameworks={BPOComplianceRequirement.PCI_DSS_4_0},
            voice_security=VoiceSecurityPolicy(encrypt_recordings=False),
        )
        errors = policy.validate()
        assert any("recording" in e.lower() or "encrypt" in e.lower() for e in errors), (
            "PCI-DSS should require encrypted recordings"
        )

    def test_pci_dss_requires_encryption_in_transit(self):
        """Test PCI-DSS validation requires encryption in transit."""
        policy = BPOSecurityPolicy(
            compliance_frameworks={BPOComplianceRequirement.PCI_DSS_4_0},
            encryption=EncryptionPolicy(encrypt_in_transit=False),
        )
        errors = policy.validate()
        assert any("transit" in e.lower() for e in errors), (
            "PCI-DSS should require encryption in transit"
        )

    def test_pci_dss_requires_card_masking(self):
        """Test PCI-DSS validation requires credit card masking."""
        policy = BPOSecurityPolicy(
            compliance_frameworks={BPOComplianceRequirement.PCI_DSS_4_0},
            agent_security=AgentSecurityPolicy(mask_credit_card=False),
        )
        errors = policy.validate()
        assert any("credit card" in e.lower() or "card" in e.lower() for e in errors), (
            "PCI-DSS should require credit card masking"
        )

    def test_pci_dss_requires_data_access_logging(self):
        """Test PCI-DSS validation requires data access logging."""
        policy = BPOSecurityPolicy(
            compliance_frameworks={BPOComplianceRequirement.PCI_DSS_4_0},
            audit=AuditPolicy(log_data_access=False),
        )
        errors = policy.validate()
        assert any("data access" in e.lower() or "logging" in e.lower() for e in errors), (
            "PCI-DSS should require data access logging"
        )

    def test_tcpa_requires_call_event_logging(self):
        """Test TCPA validation requires call event logging."""
        policy = BPOSecurityPolicy(
            compliance_frameworks={BPOComplianceRequirement.TCPA},
            audit=AuditPolicy(log_call_events=False),
        )
        errors = policy.validate()
        assert any("tcpa" in e.lower() or "call event" in e.lower() for e in errors), (
            "TCPA should require call event logging"
        )

    def test_hipaa_requires_encryption_at_rest(self):
        """Test HIPAA validation requires encryption at rest."""
        policy = BPOSecurityPolicy(
            compliance_frameworks={BPOComplianceRequirement.HIPAA},
            encryption=EncryptionPolicy(encrypt_at_rest=False),
        )
        errors = policy.validate()
        assert any("hipaa" in e.lower() or "at rest" in e.lower() for e in errors), (
            "HIPAA should require encryption at rest"
        )

    def test_hipaa_requires_clipboard_blocking(self):
        """Test HIPAA validation recommends clipboard blocking."""
        policy = BPOSecurityPolicy(
            compliance_frameworks={BPOComplianceRequirement.HIPAA},
            agent_security=AgentSecurityPolicy(block_clipboard=False),
        )
        errors = policy.validate()
        assert any("clipboard" in e.lower() for e in errors), (
            "HIPAA should recommend clipboard blocking"
        )

    def test_hipaa_requires_audit_retention(self):
        """Test HIPAA validation requires minimum 6 year audit retention."""
        policy = BPOSecurityPolicy(
            compliance_frameworks={BPOComplianceRequirement.HIPAA},
            audit=AuditPolicy(retention_days=365),  # Only 1 year
        )
        errors = policy.validate()
        assert any("retention" in e.lower() or "6 year" in e.lower() for e in errors), (
            "HIPAA should require minimum 6 year audit retention"
        )

    def test_soc2_requires_real_time_alerting(self):
        """Test SOC 2 validation requires real-time alerting."""
        policy = BPOSecurityPolicy(
            compliance_frameworks={BPOComplianceRequirement.SOC_2},
            audit=AuditPolicy(real_time_alerting=False),
        )
        errors = policy.validate()
        assert any("alerting" in e.lower() or "soc 2" in e.lower() for e in errors), (
            "SOC 2 should require real-time alerting"
        )

    def test_top_secret_requires_critical_level(self):
        """Test Top Secret data requires Critical security level."""
        policy = BPOSecurityPolicy(
            security_level=SecurityLevel.STANDARD,
            data_classification=DataClassification.TOP_SECRET,
            compliance_frameworks=set(),  # No frameworks to avoid other errors
        )
        errors = policy.validate()
        assert any("top secret" in e.lower() or "critical" in e.lower() for e in errors), (
            "Top Secret data should require Critical security level"
        )

    def test_confidential_data_requires_encryption(self):
        """Test Confidential data requires encryption at rest."""
        policy = BPOSecurityPolicy(
            data_classification=DataClassification.CONFIDENTIAL,
            encryption=EncryptionPolicy(encrypt_at_rest=False),
            compliance_frameworks=set(),
        )
        errors = policy.validate()
        assert any("encrypt" in e.lower() for e in errors), (
            "Confidential data should require encryption at rest"
        )

    def test_review_date_auto_set(self):
        """Test review date is auto-set to 1 year from effective date."""
        policy = BPOSecurityPolicy()
        assert policy.review_date is not None
        expected = policy.effective_date + timedelta(days=365)
        assert policy.review_date.date() == expected.date()


# ---------------------------------------------------------------------------
# Factory Methods
# ---------------------------------------------------------------------------

class TestFinancialServicesBPO:
    """Tests for for_financial_services_bpo() factory."""

    def test_policy_id(self, financial_policy):
        """Test financial policy has correct ID."""
        assert financial_policy.policy_id == "BPO-FIN-001"

    def test_security_level(self, financial_policy):
        """Test financial policy is Critical level."""
        assert financial_policy.security_level == SecurityLevel.CRITICAL

    def test_data_classification(self, financial_policy):
        """Test financial policy has Restricted classification."""
        assert financial_policy.data_classification == DataClassification.RESTRICTED

    def test_compliance_frameworks(self, financial_policy):
        """Test financial policy includes required frameworks."""
        frameworks = financial_policy.compliance_frameworks
        assert BPOComplianceRequirement.PCI_DSS_4_0 in frameworks
        assert BPOComplianceRequirement.SOX in frameworks
        assert BPOComplianceRequirement.GLBA in frameworks

    def test_dtmf_masking_enabled(self, financial_policy):
        """Test financial policy enables DTMF masking."""
        assert financial_policy.voice_security.enable_dtmf_masking is True

    def test_toll_fraud_detection(self, financial_policy):
        """Test financial policy enables toll fraud detection."""
        assert financial_policy.voice_security.enable_toll_fraud_detection is True

    def test_pqc_algorithms(self, financial_policy):
        """Test financial policy uses strongest PQC algorithms."""
        assert financial_policy.encryption.pqc_kem_algorithm == "ML-KEM-1024"
        assert financial_policy.encryption.pqc_sig_algorithm == "ML-DSA-87"

    def test_low_risk_tolerance(self, financial_policy):
        """Test financial policy has low risk tolerance."""
        assert financial_policy.risk_tolerance == "LOW"

    def test_validates_without_errors(self, financial_policy):
        """Test financial policy passes validation."""
        errors = financial_policy.validate()
        assert errors == [], f"Financial policy should be valid, got: {errors}"


class TestHealthcareBPO:
    """Tests for for_healthcare_bpo() factory."""

    def test_policy_id(self, healthcare_policy):
        """Test healthcare policy has correct ID."""
        assert healthcare_policy.policy_id == "BPO-HC-001"

    def test_security_level(self, healthcare_policy):
        """Test healthcare policy is Critical level."""
        assert healthcare_policy.security_level == SecurityLevel.CRITICAL

    def test_hipaa_compliance(self, healthcare_policy):
        """Test healthcare policy includes HIPAA."""
        assert BPOComplianceRequirement.HIPAA in healthcare_policy.compliance_frameworks
        assert BPOComplianceRequirement.HITECH in healthcare_policy.compliance_frameworks

    def test_recording_encryption(self, healthcare_policy):
        """Test healthcare policy encrypts recordings."""
        assert healthcare_policy.voice_security.encrypt_recordings is True

    def test_clipboard_blocking(self, healthcare_policy):
        """Test healthcare policy blocks clipboard."""
        assert healthcare_policy.agent_security.block_clipboard is True

    def test_ssn_masking(self, healthcare_policy):
        """Test healthcare policy masks SSN."""
        assert healthcare_policy.agent_security.mask_ssn is True

    def test_audit_retention(self, healthcare_policy):
        """Test healthcare policy has 6 year audit retention (HIPAA)."""
        assert healthcare_policy.audit.retention_days >= 2190  # 6 years

    def test_voice_biometrics(self, healthcare_policy):
        """Test healthcare policy enables voice biometrics."""
        assert healthcare_policy.voice_security.enable_voice_biometrics is True

    def test_validates_without_errors(self, healthcare_policy):
        """Test healthcare policy passes validation."""
        errors = healthcare_policy.validate()
        assert errors == [], f"Healthcare policy should be valid, got: {errors}"


class TestGeneralCustomerService:
    """Tests for for_general_customer_service() factory."""

    def test_policy_id(self, customer_service_policy):
        """Test customer service policy has correct ID."""
        assert customer_service_policy.policy_id == "BPO-CS-001"

    def test_security_level(self, customer_service_policy):
        """Test customer service policy is Enhanced level."""
        assert customer_service_policy.security_level == SecurityLevel.ENHANCED

    def test_data_classification(self, customer_service_policy):
        """Test customer service policy is Confidential classification."""
        assert customer_service_policy.data_classification == DataClassification.CONFIDENTIAL

    def test_medium_risk_tolerance(self, customer_service_policy):
        """Test customer service policy has medium risk tolerance."""
        assert customer_service_policy.risk_tolerance == "MEDIUM"

    def test_pqc_enabled(self, customer_service_policy):
        """Test customer service policy enables PQC."""
        assert customer_service_policy.encryption.require_pqc is True

    def test_validates_without_errors(self, customer_service_policy):
        """Test customer service policy passes validation."""
        errors = customer_service_policy.validate()
        assert errors == [], f"Customer service policy should be valid, got: {errors}"


class TestRemoteWorkforce:
    """Tests for for_remote_workforce() factory."""

    def test_policy_id(self, remote_workforce_policy):
        """Test remote workforce policy has correct ID."""
        assert remote_workforce_policy.policy_id == "BPO-REMOTE-001"

    def test_biometric_mfa(self, remote_workforce_policy):
        """Test remote workforce policy includes biometric MFA."""
        assert "biometric" in remote_workforce_policy.agent_security.mfa_methods

    def test_aggressive_idle_timeout(self, remote_workforce_policy):
        """Test remote workforce has aggressive idle timeout."""
        assert remote_workforce_policy.agent_security.idle_timeout_minutes <= 3

    def test_all_blocking_enabled(self, remote_workforce_policy):
        """Test remote workforce blocks all exfiltration channels."""
        agent = remote_workforce_policy.agent_security
        assert agent.block_clipboard is True
        assert agent.block_usb_storage is True
        assert agent.block_personal_email is True
        assert agent.block_cloud_storage is True

    def test_watermark_screen(self, remote_workforce_policy):
        """Test remote workforce enables screen watermarking."""
        assert remote_workforce_policy.agent_security.watermark_screen is True

    def test_endpoint_compliance(self, remote_workforce_policy):
        """Test remote workforce requires endpoint compliance."""
        agent = remote_workforce_policy.agent_security
        assert agent.require_endpoint_compliance is True
        assert agent.require_antivirus is True
        assert agent.require_disk_encryption is True

    def test_wpa3_required(self, remote_workforce_policy):
        """Test remote workforce requires WPA3 for home network."""
        assert remote_workforce_policy.agent_security.home_network_requirements == "WPA3"

    def test_pqc_voice_encryption(self, remote_workforce_policy):
        """Test remote workforce uses PQC voice encryption."""
        assert "PQC" in remote_workforce_policy.encryption.voice_encryption

    def test_validates_without_errors(self, remote_workforce_policy):
        """Test remote workforce policy passes validation."""
        errors = remote_workforce_policy.validate()
        assert errors == [], f"Remote workforce policy should be valid, got: {errors}"


# ---------------------------------------------------------------------------
# Policy Serialization
# ---------------------------------------------------------------------------

class TestPolicySerialization:
    """Tests for BPOSecurityPolicy serialization."""

    def test_to_dict_has_policy_id(self, financial_policy):
        """Test serialization includes policy_id."""
        d = financial_policy.to_dict()
        assert d["policy_id"] == "BPO-FIN-001"

    def test_to_dict_has_security_level(self, financial_policy):
        """Test serialization includes security level."""
        d = financial_policy.to_dict()
        assert d["security_level"] == "Critical"

    def test_to_dict_has_data_classification(self, financial_policy):
        """Test serialization includes data classification."""
        d = financial_policy.to_dict()
        assert d["data_classification"] == "Restricted"

    def test_to_dict_has_compliance_frameworks(self, financial_policy):
        """Test serialization includes compliance frameworks."""
        d = financial_policy.to_dict()
        assert "compliance_frameworks" in d
        assert isinstance(d["compliance_frameworks"], list)
        assert len(d["compliance_frameworks"]) > 0

    def test_to_dict_has_voice_security(self, financial_policy):
        """Test serialization includes voice security settings."""
        d = financial_policy.to_dict()
        assert "voice_security" in d
        assert "dtmf_masking" in d["voice_security"]

    def test_to_dict_has_agent_security(self, financial_policy):
        """Test serialization includes agent security settings."""
        d = financial_policy.to_dict()
        assert "agent_security" in d
        assert "require_mfa" in d["agent_security"]

    def test_to_dict_has_encryption(self, financial_policy):
        """Test serialization includes encryption settings."""
        d = financial_policy.to_dict()
        assert "encryption" in d
        assert "require_pqc" in d["encryption"]
        assert "kem_algorithm" in d["encryption"]

    def test_to_dict_has_risk_tolerance(self, financial_policy):
        """Test serialization includes risk tolerance."""
        d = financial_policy.to_dict()
        assert "risk_tolerance" in d

    def test_to_dict_effective_date_is_iso(self, financial_policy):
        """Test effective date is ISO format string."""
        d = financial_policy.to_dict()
        assert "effective_date" in d
        # Verify it can be parsed back
        datetime.fromisoformat(d["effective_date"])

    def test_to_dict_round_trip_consistency(self, financial_policy):
        """Test serialization produces consistent output."""
        d1 = financial_policy.to_dict()
        d2 = financial_policy.to_dict()
        assert d1 == d2


# ---------------------------------------------------------------------------
# Pre-defined Policies & Access Functions
# ---------------------------------------------------------------------------

class TestPolicyAccessFunctions:
    """Tests for get_policy() and list_policies() functions."""

    def test_get_policy_existing(self):
        """Test get_policy returns existing policy."""
        policy = get_policy("financial_services")
        assert policy is not None
        assert isinstance(policy, BPOSecurityPolicy)

    def test_get_policy_nonexistent(self):
        """Test get_policy returns None for unknown policy."""
        assert get_policy("nonexistent") is None

    def test_list_policies_returns_list(self):
        """Test list_policies returns a list of strings."""
        policies = list_policies()
        assert isinstance(policies, list)
        assert len(policies) > 0

    @pytest.mark.parametrize("policy_name", [
        "financial_services",
        "healthcare",
        "general_customer_service",
        "remote_workforce",
    ])
    def test_predefined_policy_exists(self, policy_name):
        """Test each pre-defined policy exists."""
        assert policy_name in BPO_POLICIES
        policy = get_policy(policy_name)
        assert policy is not None

    def test_all_listed_policies_retrievable(self):
        """Test every listed policy is retrievable."""
        for name in list_policies():
            policy = get_policy(name)
            assert policy is not None, f"Policy '{name}' listed but not retrievable"
