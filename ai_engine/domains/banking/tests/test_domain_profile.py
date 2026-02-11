"""
Tests for Banking Domain Profile and Security Policies
"""

import pytest
from datetime import datetime, timedelta

from ai_engine.domains.banking.core.domain_profile import (
    BankingSubdomain,
    BankingSecurityConstraints,
    BankingPQCProfile,
    PQCAlgorithm,
    BANKING_PROFILES,
)
from ai_engine.domains.banking.core.security_policy import (
    SecurityLevel,
    DataClassification,
    ComplianceRequirement,
    AccessControl,
    EncryptionPolicy,
    AuditPolicy,
    BankingSecurityPolicy,
    BANKING_POLICIES,
    get_policy,
    list_policies,
)


class TestBankingSubdomain:
    """Tests for BankingSubdomain enum."""

    def test_all_subdomains_defined(self):
        """Verify all expected subdomains are defined."""
        expected = [
            "CARD_PAYMENTS", "WIRE_TRANSFERS", "DOMESTIC_CLEARING",
            "REAL_TIME_PAYMENTS", "TRADING", "CORRESPONDENT_BANKING",
            "TREASURY", "CORE_BANKING",
        ]
        for subdomain in expected:
            assert hasattr(BankingSubdomain, subdomain)

    def test_subdomain_values_unique(self):
        """Verify all subdomain values are unique."""
        values = [s.value for s in BankingSubdomain]
        assert len(values) == len(set(values))


class TestBankingSecurityConstraints:
    """Tests for BankingSecurityConstraints dataclass."""

    def test_default_constraints(self):
        """Test default constraint values."""
        constraints = BankingSecurityConstraints()
        assert constraints.max_latency_ms > 0
        assert constraints.min_key_bits >= 128
        assert constraints.hsm_required is True
        assert constraints.pci_dss_required is True

    def test_custom_constraints(self):
        """Test custom constraint values."""
        constraints = BankingSecurityConstraints(
            max_latency_ms=50,
            min_key_bits=256,
            hsm_required=True,
            require_dual_control=True,
        )
        assert constraints.max_latency_ms == 50
        assert constraints.min_key_bits == 256
        assert constraints.require_dual_control is True

    def test_compliance_frameworks_list(self):
        """Test compliance frameworks as a list."""
        constraints = BankingSecurityConstraints(
            compliance_frameworks=["PCI-DSS 4.0", "DORA", "NIST PQC"]
        )
        assert "PCI-DSS 4.0" in constraints.compliance_frameworks
        assert "DORA" in constraints.compliance_frameworks


class TestBankingPQCProfile:
    """Tests for BankingPQCProfile dataclass."""

    def test_default_profile(self):
        """Test default profile creation."""
        profile = BankingPQCProfile(
            subdomain=BankingSubdomain.CARD_PAYMENTS,
            constraints=BankingSecurityConstraints(),
        )
        assert profile.subdomain == BankingSubdomain.CARD_PAYMENTS
        assert profile.kem_algorithm == PQCAlgorithm.ML_KEM_768
        assert profile.hybrid_mode is True

    def test_profile_for_wire_transfers(self):
        """Test profile for wire transfers."""
        profile = BankingPQCProfile(
            subdomain=BankingSubdomain.WIRE_TRANSFERS,
            constraints=BankingSecurityConstraints(
                max_latency_ms=500,
                min_key_bits=256,
                require_dual_control=True,
            ),
            kem_algorithm=PQCAlgorithm.ML_KEM_1024,
            sig_algorithm=PQCAlgorithm.ML_DSA_87,
        )
        assert profile.kem_algorithm == PQCAlgorithm.ML_KEM_1024
        assert profile.sig_algorithm == PQCAlgorithm.ML_DSA_87
        assert profile.constraints.require_dual_control is True

    def test_trading_profile_low_latency(self):
        """Test trading profile with low latency requirements."""
        profile = BankingPQCProfile(
            subdomain=BankingSubdomain.TRADING,
            constraints=BankingSecurityConstraints(max_latency_ms=10),
            kem_algorithm=PQCAlgorithm.ML_KEM_512,  # Faster
            hybrid_mode=False,  # Lower latency
        )
        assert profile.constraints.max_latency_ms == 10
        assert profile.hybrid_mode is False

    def test_predefined_profiles_exist(self):
        """Test that predefined profiles are available."""
        assert "swift_messaging" in BANKING_PROFILES
        assert "real_time_payments" in BANKING_PROFILES
        assert "card_processing" in BANKING_PROFILES
        assert "trading_systems" in BANKING_PROFILES
        assert "hsm_operations" in BANKING_PROFILES


class TestPQCAlgorithm:
    """Tests for PQCAlgorithm enum."""

    def test_ml_kem_variants(self):
        """Test ML-KEM algorithm variants."""
        assert PQCAlgorithm.ML_KEM_512.key_size == 512
        assert PQCAlgorithm.ML_KEM_768.key_size == 768
        assert PQCAlgorithm.ML_KEM_1024.key_size == 1024

    def test_ml_dsa_variants(self):
        """Test ML-DSA algorithm variants."""
        assert PQCAlgorithm.ML_DSA_44.key_size == 44
        assert PQCAlgorithm.ML_DSA_65.key_size == 65
        assert PQCAlgorithm.ML_DSA_87.key_size == 87

    def test_algorithm_names(self):
        """Test algorithm name property."""
        assert "ML-KEM" in PQCAlgorithm.ML_KEM_768.name
        assert "ML-DSA" in PQCAlgorithm.ML_DSA_65.name


class TestSecurityLevel:
    """Tests for SecurityLevel enum."""

    def test_security_levels(self):
        """Test security level properties."""
        assert SecurityLevel.STANDARD.level == 1
        assert SecurityLevel.ENHANCED.level == 2
        assert SecurityLevel.CRITICAL.level == 3
        assert SecurityLevel.RESTRICTED.level == 4

    def test_min_key_bits(self):
        """Test minimum key bits per level."""
        assert SecurityLevel.STANDARD.min_key_bits == 128
        assert SecurityLevel.ENHANCED.min_key_bits == 192
        assert SecurityLevel.CRITICAL.min_key_bits == 256
        assert SecurityLevel.RESTRICTED.min_key_bits == 256


class TestDataClassification:
    """Tests for DataClassification enum."""

    def test_classification_levels(self):
        """Test data classification levels."""
        assert DataClassification.PUBLIC.level == 0
        assert DataClassification.INTERNAL.level == 1
        assert DataClassification.CONFIDENTIAL.level == 2
        assert DataClassification.RESTRICTED.level == 3
        assert DataClassification.TOP_SECRET.level == 4

    def test_encryption_requirements(self):
        """Test encryption requirements per classification."""
        assert DataClassification.PUBLIC.requires_encryption is False
        assert DataClassification.CONFIDENTIAL.requires_encryption is True
        assert DataClassification.RESTRICTED.requires_encryption is True
        assert DataClassification.TOP_SECRET.requires_encryption is True

    def test_hsm_requirements(self):
        """Test HSM requirements per classification."""
        assert DataClassification.CONFIDENTIAL.requires_hsm is False
        assert DataClassification.RESTRICTED.requires_hsm is True
        assert DataClassification.TOP_SECRET.requires_hsm is True


class TestComplianceRequirement:
    """Tests for ComplianceRequirement enum."""

    def test_pci_dss(self):
        """Test PCI-DSS compliance requirement."""
        pci = ComplianceRequirement.PCI_DSS_4_0
        assert pci.code == "PCI-DSS 4.0"
        assert "Payment Card" in pci.description

    def test_dora(self):
        """Test DORA compliance requirement."""
        dora = ComplianceRequirement.DORA
        assert dora.code == "DORA"
        assert "Resilience" in dora.description

    def test_nist_pqc(self):
        """Test NIST PQC compliance requirement."""
        nist = ComplianceRequirement.NIST_PQC
        assert "PQC" in nist.code
        assert "Quantum" in nist.description


class TestAccessControl:
    """Tests for AccessControl dataclass."""

    def test_default_access_control(self):
        """Test default access control settings."""
        ac = AccessControl()
        assert ac.require_mfa is True
        assert ac.max_session_duration_hours == 8
        assert ac.idle_timeout_minutes == 15

    def test_mfa_methods(self):
        """Test MFA method configuration."""
        ac = AccessControl(mfa_methods={"totp", "hardware_token", "biometric"})
        assert "totp" in ac.mfa_methods
        assert "hardware_token" in ac.mfa_methods

    def test_dual_control(self):
        """Test dual control settings."""
        ac = AccessControl(
            dual_control_required=True,
            four_eyes_principle=True,
        )
        assert ac.dual_control_required is True
        assert ac.four_eyes_principle is True


class TestEncryptionPolicy:
    """Tests for EncryptionPolicy dataclass."""

    def test_default_encryption_policy(self):
        """Test default encryption policy."""
        ep = EncryptionPolicy()
        assert ep.encrypt_at_rest is True
        assert ep.encrypt_in_transit is True
        assert ep.min_tls_version == "TLS 1.3"
        assert ep.require_pqc is True

    def test_pqc_algorithms(self):
        """Test PQC algorithm settings."""
        ep = EncryptionPolicy(
            pqc_kem_algorithm="ML-KEM-1024",
            pqc_sig_algorithm="ML-DSA-87",
        )
        assert ep.pqc_kem_algorithm == "ML-KEM-1024"
        assert ep.pqc_sig_algorithm == "ML-DSA-87"

    def test_hybrid_mode(self):
        """Test hybrid cryptography mode."""
        ep = EncryptionPolicy(hybrid_mode=True)
        assert ep.hybrid_mode is True


class TestBankingSecurityPolicy:
    """Tests for BankingSecurityPolicy dataclass."""

    def test_default_policy(self):
        """Test default security policy."""
        policy = BankingSecurityPolicy()
        assert policy.security_level == SecurityLevel.ENHANCED
        assert policy.data_classification == DataClassification.CONFIDENTIAL

    def test_policy_validation_valid(self):
        """Test validation of a valid policy."""
        policy = BankingSecurityPolicy.for_payment_processing()
        errors = policy.validate()
        assert len(errors) == 0

    def test_policy_validation_invalid(self):
        """Test validation catches mismatches."""
        policy = BankingSecurityPolicy(
            data_classification=DataClassification.TOP_SECRET,
            security_level=SecurityLevel.STANDARD,  # Too low for TOP_SECRET
        )
        errors = policy.validate()
        assert len(errors) > 0
        assert any("Top Secret" in e for e in errors)

    def test_payment_processing_policy(self):
        """Test payment processing policy factory."""
        policy = BankingSecurityPolicy.for_payment_processing()
        assert policy.security_level == SecurityLevel.CRITICAL
        assert policy.data_classification == DataClassification.RESTRICTED
        assert ComplianceRequirement.PCI_DSS_4_0 in policy.compliance_frameworks
        assert policy.access_control.dual_control_required is True

    def test_wire_transfer_policy(self):
        """Test wire transfer policy factory."""
        policy = BankingSecurityPolicy.for_wire_transfer()
        assert policy.security_level == SecurityLevel.CRITICAL
        assert policy.access_control.four_eyes_principle is True
        assert ComplianceRequirement.DORA in policy.compliance_frameworks
        assert policy.max_transaction_amount == 1000000.0

    def test_trading_policy(self):
        """Test trading systems policy factory."""
        policy = BankingSecurityPolicy.for_trading()
        assert ComplianceRequirement.MIFID_II in policy.compliance_frameworks
        assert policy.encryption.hybrid_mode is False  # Speed priority

    def test_hsm_operations_policy(self):
        """Test HSM operations policy factory."""
        policy = BankingSecurityPolicy.for_hsm_operations()
        assert policy.security_level == SecurityLevel.RESTRICTED
        assert policy.data_classification == DataClassification.TOP_SECRET
        assert policy.incident_response_sla_minutes == 5
        assert policy.access_control.max_session_duration_hours == 1

    def test_policy_to_dict(self):
        """Test policy serialization."""
        policy = BankingSecurityPolicy.for_payment_processing()
        data = policy.to_dict()
        assert "policy_id" in data
        assert "security_level" in data
        assert "encryption" in data
        assert data["encryption"]["require_pqc"] is True

    def test_review_date_auto_set(self):
        """Test review date is automatically set."""
        policy = BankingSecurityPolicy()
        assert policy.review_date is not None
        assert policy.review_date > policy.effective_date


class TestPredefinedPolicies:
    """Tests for predefined policy registry."""

    def test_banking_policies_exist(self):
        """Test that all expected policies exist."""
        assert "payment_processing" in BANKING_POLICIES
        assert "wire_transfer" in BANKING_POLICIES
        assert "trading" in BANKING_POLICIES
        assert "hsm_operations" in BANKING_POLICIES

    def test_get_policy(self):
        """Test policy retrieval function."""
        policy = get_policy("payment_processing")
        assert policy is not None
        assert policy.policy_id == "PAYMENT-001"

    def test_get_policy_not_found(self):
        """Test retrieval of non-existent policy."""
        policy = get_policy("non_existent")
        assert policy is None

    def test_list_policies(self):
        """Test listing available policies."""
        policies = list_policies()
        assert len(policies) >= 4
        assert "payment_processing" in policies
