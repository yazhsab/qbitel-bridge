"""
Comprehensive Unit Tests for GDPR Compliance Manager
Tests all functionality in ai_engine/compliance/gdpr_compliance.py
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from ai_engine.compliance.gdpr_compliance import (
    GDPRComplianceManager,
    GDPRException,
    DataSubjectRight,
    LegalBasis,
    DataSubjectRequest,
    ConsentRecord,
    DataProcessingRecord,
)
from ai_engine.compliance.audit_trail import AuditTrailManager, EventType
from ai_engine.core.config import Config


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = Config()
    config.data_controller_name = "QBITEL Test"
    config.dpo_contact = "dpo@test.com"
    return config


@pytest.fixture
def mock_audit_manager():
    """Create mock audit manager."""
    manager = Mock(spec=AuditTrailManager)
    manager.record_compliance_event = AsyncMock()
    return manager


@pytest.fixture
def gdpr_manager(mock_config, mock_audit_manager):
    """Create GDPRComplianceManager instance."""
    return GDPRComplianceManager(mock_config, mock_audit_manager)


class TestGDPRComplianceManagerInitialization:
    """Test GDPRComplianceManager initialization."""

    def test_initialization(self, gdpr_manager, mock_config, mock_audit_manager):
        """Test successful initialization."""
        assert gdpr_manager.config == mock_config
        assert gdpr_manager.audit_manager == mock_audit_manager
        assert gdpr_manager.data_controller == "QBITEL Test"
        assert gdpr_manager.dpo_contact == "dpo@test.com"
        assert gdpr_manager.breach_notification_hours == 72
        assert len(gdpr_manager.consent_records) == 0
        assert len(gdpr_manager.processing_records) == 0
        assert len(gdpr_manager.subject_requests) == 0

    def test_initialization_default_values(self):
        """Test initialization with default config values."""
        config = Config()
        manager = GDPRComplianceManager(config, None)

        assert manager.data_controller == "QBITEL"
        assert manager.dpo_contact == "dpo@qbitel.com"

    @pytest.mark.asyncio
    async def test_initialize(self, gdpr_manager):
        """Test manager initialization."""
        with patch.object(gdpr_manager, "_load_records", new_callable=AsyncMock):
            await gdpr_manager.initialize()

            # Should complete without error


class TestConsentManagement:
    """Test consent recording and management."""

    @pytest.mark.asyncio
    async def test_record_consent_success(self, gdpr_manager, mock_audit_manager):
        """Test successful consent recording."""
        consent_id = await gdpr_manager.record_consent(
            subject_id="user_123",
            purpose="Marketing communications",
            legal_basis=LegalBasis.CONSENT,
            expires_in_days=365,
            metadata={"channel": "email"},
        )

        assert consent_id is not None
        assert consent_id in gdpr_manager.consent_records

        consent = gdpr_manager.consent_records[consent_id]
        assert consent.subject_id == "user_123"
        assert consent.purpose == "Marketing communications"
        assert consent.legal_basis == LegalBasis.CONSENT
        assert consent.expires_at is not None
        assert consent.metadata["channel"] == "email"

        mock_audit_manager.record_compliance_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_consent_without_expiry(self, gdpr_manager):
        """Test recording consent without expiry date."""
        consent_id = await gdpr_manager.record_consent(
            subject_id="user_456",
            purpose="Service provision",
            legal_basis=LegalBasis.CONTRACT,
        )

        consent = gdpr_manager.consent_records[consent_id]
        assert consent.expires_at is None

    @pytest.mark.asyncio
    async def test_record_consent_different_legal_bases(self, gdpr_manager):
        """Test recording consent with different legal bases."""
        for basis in LegalBasis:
            consent_id = await gdpr_manager.record_consent(
                subject_id=f"user_{basis.value}",
                purpose=f"Purpose for {basis.value}",
                legal_basis=basis,
            )

            consent = gdpr_manager.consent_records[consent_id]
            assert consent.legal_basis == basis

    @pytest.mark.asyncio
    async def test_record_consent_without_audit_manager(self, mock_config):
        """Test recording consent without audit manager."""
        manager = GDPRComplianceManager(mock_config, None)

        consent_id = await manager.record_consent(subject_id="user_123", purpose="Test purpose")

        assert consent_id is not None

    @pytest.mark.asyncio
    async def test_withdraw_consent_success(self, gdpr_manager, mock_audit_manager):
        """Test successful consent withdrawal."""
        consent_id = await gdpr_manager.record_consent(subject_id="user_123", purpose="Marketing")

        result = await gdpr_manager.withdraw_consent(consent_id, "user_123")

        assert result is True
        consent = gdpr_manager.consent_records[consent_id]
        assert consent.withdrawn_at is not None

    @pytest.mark.asyncio
    async def test_withdraw_consent_nonexistent(self, gdpr_manager):
        """Test withdrawing non-existent consent."""
        with pytest.raises(GDPRException, match="Consent .* not found"):
            await gdpr_manager.withdraw_consent("invalid_id", "user_123")

    @pytest.mark.asyncio
    async def test_withdraw_consent_wrong_subject(self, gdpr_manager):
        """Test withdrawing consent with wrong subject ID."""
        consent_id = await gdpr_manager.record_consent(subject_id="user_123", purpose="Marketing")

        with pytest.raises(GDPRException, match="Subject ID mismatch"):
            await gdpr_manager.withdraw_consent(consent_id, "user_456")


class TestDataSubjectRights:
    """Test data subject rights handling."""

    @pytest.mark.asyncio
    async def test_handle_access_request(self, gdpr_manager, mock_audit_manager):
        """Test handling right to access request."""
        request_id = await gdpr_manager.handle_subject_request(
            subject_id="user_123",
            right=DataSubjectRight.ACCESS,
            details={"scope": "all_data"},
        )

        assert request_id is not None
        assert request_id in gdpr_manager.subject_requests

        request = gdpr_manager.subject_requests[request_id]
        assert request.subject_id == "user_123"
        assert request.right == DataSubjectRight.ACCESS
        assert request.status == "completed"
        assert request.response_data is not None

    @pytest.mark.asyncio
    async def test_handle_erasure_request(self, gdpr_manager):
        """Test handling right to erasure request."""
        request_id = await gdpr_manager.handle_subject_request(subject_id="user_123", right=DataSubjectRight.ERASURE)

        request = gdpr_manager.subject_requests[request_id]
        assert request.right == DataSubjectRight.ERASURE
        assert request.status == "completed"

    @pytest.mark.asyncio
    async def test_handle_portability_request(self, gdpr_manager):
        """Test handling right to data portability request."""
        request_id = await gdpr_manager.handle_subject_request(subject_id="user_123", right=DataSubjectRight.PORTABILITY)

        request = gdpr_manager.subject_requests[request_id]
        assert request.right == DataSubjectRight.PORTABILITY
        assert request.status == "completed"

    @pytest.mark.asyncio
    async def test_handle_rectification_request(self, gdpr_manager):
        """Test handling right to rectification request."""
        request_id = await gdpr_manager.handle_subject_request(
            subject_id="user_123",
            right=DataSubjectRight.RECTIFICATION,
            details={"field": "email", "new_value": "new@example.com"},
        )

        request = gdpr_manager.subject_requests[request_id]
        assert request.right == DataSubjectRight.RECTIFICATION
        assert request.details["field"] == "email"

    @pytest.mark.asyncio
    async def test_handle_restriction_request(self, gdpr_manager):
        """Test handling right to restriction request."""
        request_id = await gdpr_manager.handle_subject_request(subject_id="user_123", right=DataSubjectRight.RESTRICTION)

        request = gdpr_manager.subject_requests[request_id]
        assert request.right == DataSubjectRight.RESTRICTION

    @pytest.mark.asyncio
    async def test_handle_object_request(self, gdpr_manager):
        """Test handling right to object request."""
        request_id = await gdpr_manager.handle_subject_request(subject_id="user_123", right=DataSubjectRight.OBJECT)

        request = gdpr_manager.subject_requests[request_id]
        assert request.right == DataSubjectRight.OBJECT

    @pytest.mark.asyncio
    async def test_access_request_includes_rights_info(self, gdpr_manager):
        """Test access request includes rights information."""
        request_id = await gdpr_manager.handle_subject_request(subject_id="user_123", right=DataSubjectRight.ACCESS)

        request = gdpr_manager.subject_requests[request_id]
        assert "rights_information" in request.response_data
        assert "dpo_contact" in request.response_data["rights_information"]


class TestProcessingActivities:
    """Test processing activities recording (Article 30)."""

    @pytest.mark.asyncio
    async def test_record_processing_activity(self, gdpr_manager):
        """Test recording processing activity."""
        record_id = await gdpr_manager.record_processing_activity(
            purpose="User authentication",
            legal_basis=LegalBasis.CONTRACT,
            data_categories=["email", "password_hash"],
            data_subjects=["customers"],
            recipients=["internal_systems"],
            retention_period="Account lifetime + 90 days",
            security_measures=["encryption", "access_control"],
        )

        assert record_id is not None
        assert record_id in gdpr_manager.processing_records

        record = gdpr_manager.processing_records[record_id]
        assert record.purpose == "User authentication"
        assert record.legal_basis == LegalBasis.CONTRACT
        assert "email" in record.data_categories
        assert "encryption" in record.security_measures

    @pytest.mark.asyncio
    async def test_record_processing_activity_sets_controller(self, gdpr_manager):
        """Test processing activity records controller name."""
        record_id = await gdpr_manager.record_processing_activity(
            purpose="Analytics",
            legal_basis=LegalBasis.LEGITIMATE_INTERESTS,
            data_categories=["usage_data"],
            data_subjects=["users"],
            recipients=["analytics_team"],
            retention_period="2 years",
            security_measures=["pseudonymization"],
        )

        record = gdpr_manager.processing_records[record_id]
        assert record.controller_name == "QBITEL Test"

    @pytest.mark.asyncio
    async def test_record_processing_activity_timestamps(self, gdpr_manager):
        """Test processing activity has timestamps."""
        record_id = await gdpr_manager.record_processing_activity(
            purpose="Test",
            legal_basis=LegalBasis.CONSENT,
            data_categories=["test"],
            data_subjects=["test"],
            recipients=["test"],
            retention_period="test",
            security_measures=["test"],
        )

        record = gdpr_manager.processing_records[record_id]
        assert record.created_at is not None
        assert record.updated_at is not None


class TestComplianceVerification:
    """Test compliance verification."""

    @pytest.mark.asyncio
    async def test_verify_compliance_no_issues(self, gdpr_manager):
        """Test compliance verification with no issues."""
        status = await gdpr_manager.verify_compliance()

        assert status["compliant"] is True
        assert "issues" in status
        assert "recommendations" in status
        assert "verified_at" in status

    @pytest.mark.asyncio
    async def test_verify_compliance_expired_consents(self, gdpr_manager):
        """Test verification detects expired consents."""
        # Create expired consent
        consent_id = await gdpr_manager.record_consent(subject_id="user_123", purpose="Marketing", expires_in_days=1)

        # Set expiry to past
        consent = gdpr_manager.consent_records[consent_id]
        consent.expires_at = datetime.utcnow() - timedelta(days=1)

        status = await gdpr_manager.verify_compliance()

        assert len(status["issues"]) > 0
        assert any("expired consents" in issue.lower() for issue in status["issues"])

    @pytest.mark.asyncio
    async def test_verify_compliance_overdue_requests(self, gdpr_manager):
        """Test verification detects overdue subject requests."""
        # Create old pending request
        request_id = await gdpr_manager.handle_subject_request(
            subject_id="user_123",
            right=DataSubjectRight.RESTRICTION,  # Won't auto-complete
        )

        # Set request date to past
        request = gdpr_manager.subject_requests[request_id]
        request.requested_at = datetime.utcnow() - timedelta(days=35)
        request.status = "pending"

        status = await gdpr_manager.verify_compliance()

        assert status["compliant"] is False
        assert any("30-day response time" in issue for issue in status["issues"])

    @pytest.mark.asyncio
    async def test_verify_compliance_no_processing_records(self, gdpr_manager):
        """Test verification recommends processing records."""
        status = await gdpr_manager.verify_compliance()

        assert any("processing activities" in rec.lower() for rec in status["recommendations"])


class TestStatistics:
    """Test statistics functionality."""

    def test_get_statistics_empty(self, gdpr_manager):
        """Test statistics with no data."""
        stats = gdpr_manager.get_statistics()

        assert stats["total_consents"] == 0
        assert stats["active_consents"] == 0
        assert stats["withdrawn_consents"] == 0
        assert stats["subject_requests"]["total"] == 0
        assert stats["processing_records"] == 0

    @pytest.mark.asyncio
    async def test_get_statistics_with_data(self, gdpr_manager):
        """Test statistics with data."""
        # Add consents
        await gdpr_manager.record_consent("user_1", "Purpose 1")
        await gdpr_manager.record_consent("user_2", "Purpose 2")
        consent_id = await gdpr_manager.record_consent("user_3", "Purpose 3")
        await gdpr_manager.withdraw_consent(consent_id, "user_3")

        # Add subject request
        await gdpr_manager.handle_subject_request("user_1", DataSubjectRight.ACCESS)

        # Add processing record
        await gdpr_manager.record_processing_activity(
            "Test",
            LegalBasis.CONSENT,
            ["data"],
            ["subjects"],
            ["recipients"],
            "period",
            ["measures"],
        )

        stats = gdpr_manager.get_statistics()

        assert stats["total_consents"] == 3
        assert stats["active_consents"] == 2
        assert stats["withdrawn_consents"] == 1
        assert stats["subject_requests"]["total"] == 1
        assert stats["subject_requests"]["completed"] == 1
        assert stats["processing_records"] == 1


class TestDataClasses:
    """Test data classes."""

    def test_data_subject_request_creation(self):
        """Test creating DataSubjectRequest."""
        request = DataSubjectRequest(
            request_id="req_123",
            subject_id="user_123",
            right=DataSubjectRight.ACCESS,
            requested_at=datetime.utcnow(),
        )

        assert request.request_id == "req_123"
        assert request.subject_id == "user_123"
        assert request.right == DataSubjectRight.ACCESS
        assert request.status == "pending"

    def test_consent_record_creation(self):
        """Test creating ConsentRecord."""
        now = datetime.utcnow()
        consent = ConsentRecord(
            consent_id="con_123",
            subject_id="user_123",
            purpose="Marketing",
            legal_basis=LegalBasis.CONSENT,
            granted_at=now,
        )

        assert consent.consent_id == "con_123"
        assert consent.subject_id == "user_123"
        assert consent.purpose == "Marketing"
        assert consent.withdrawn_at is None

    def test_data_processing_record_creation(self):
        """Test creating DataProcessingRecord."""
        now = datetime.utcnow()
        record = DataProcessingRecord(
            record_id="proc_123",
            controller_name="Test Controller",
            purpose="Test Purpose",
            legal_basis=LegalBasis.CONTRACT,
            data_categories=["email"],
            data_subjects=["customers"],
            recipients=["internal"],
            retention_period="1 year",
            security_measures=["encryption"],
            created_at=now,
            updated_at=now,
        )

        assert record.record_id == "proc_123"
        assert record.controller_name == "Test Controller"
        assert "email" in record.data_categories


class TestEnums:
    """Test enum classes."""

    def test_data_subject_right_values(self):
        """Test DataSubjectRight enum values."""
        assert DataSubjectRight.ACCESS.value == "right_to_access"
        assert DataSubjectRight.RECTIFICATION.value == "right_to_rectification"
        assert DataSubjectRight.ERASURE.value == "right_to_erasure"
        assert DataSubjectRight.RESTRICTION.value == "right_to_restriction"
        assert DataSubjectRight.PORTABILITY.value == "right_to_portability"
        assert DataSubjectRight.OBJECT.value == "right_to_object"

    def test_legal_basis_values(self):
        """Test LegalBasis enum values."""
        assert LegalBasis.CONSENT.value == "consent"
        assert LegalBasis.CONTRACT.value == "contract"
        assert LegalBasis.LEGAL_OBLIGATION.value == "legal_obligation"
        assert LegalBasis.VITAL_INTERESTS.value == "vital_interests"
        assert LegalBasis.PUBLIC_TASK.value == "public_task"
        assert LegalBasis.LEGITIMATE_INTERESTS.value == "legitimate_interests"


class TestRightsInformation:
    """Test rights information."""

    def test_get_rights_information(self, gdpr_manager):
        """Test getting rights information."""
        info = gdpr_manager._get_rights_information()

        assert "right_to_access" in info
        assert "right_to_rectification" in info
        assert "right_to_erasure" in info
        assert "right_to_restriction" in info
        assert "right_to_portability" in info
        assert "right_to_object" in info
        assert "dpo_contact" in info
        assert info["dpo_contact"] == "dpo@test.com"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_multiple_consents_same_subject(self, gdpr_manager):
        """Test multiple consents for same subject."""
        consent_id1 = await gdpr_manager.record_consent("user_123", "Purpose 1")
        consent_id2 = await gdpr_manager.record_consent("user_123", "Purpose 2")

        assert consent_id1 != consent_id2
        assert len(gdpr_manager.consent_records) == 2

    @pytest.mark.asyncio
    async def test_consent_with_metadata(self, gdpr_manager):
        """Test consent with custom metadata."""
        consent_id = await gdpr_manager.record_consent(
            "user_123", "Marketing", metadata={"source": "web", "ip": "192.168.1.1"}
        )

        consent = gdpr_manager.consent_records[consent_id]
        assert consent.metadata["source"] == "web"
        assert consent.metadata["ip"] == "192.168.1.1"

    @pytest.mark.asyncio
    async def test_subject_request_without_details(self, gdpr_manager):
        """Test subject request without details."""
        request_id = await gdpr_manager.handle_subject_request("user_123", DataSubjectRight.ACCESS)

        request = gdpr_manager.subject_requests[request_id]
        assert request.details == {}

    @pytest.mark.asyncio
    async def test_verify_compliance_with_valid_consents(self, gdpr_manager):
        """Test verification with valid non-expired consents."""
        await gdpr_manager.record_consent("user_123", "Marketing", expires_in_days=365)

        status = await gdpr_manager.verify_compliance()

        # Should not have expired consent issues
        assert not any("expired consents" in issue.lower() for issue in status["issues"])

    @pytest.mark.asyncio
    async def test_consent_without_expiry_not_expired(self, gdpr_manager):
        """Test consent without expiry is not considered expired."""
        await gdpr_manager.record_consent("user_123", "Service provision", legal_basis=LegalBasis.CONTRACT)

        status = await gdpr_manager.verify_compliance()

        # Should not have expired consent issues
        assert not any("expired consents" in issue.lower() for issue in status["issues"])

    @pytest.mark.asyncio
    async def test_load_records(self, gdpr_manager):
        """Test loading records from database."""
        # Should not raise exception
        await gdpr_manager._load_records()

    @pytest.mark.asyncio
    async def test_processing_request_methods(self, gdpr_manager):
        """Test all processing request methods."""
        # Create requests for all rights
        request_id = await gdpr_manager.handle_subject_request("user_123", DataSubjectRight.ACCESS)
        request = gdpr_manager.subject_requests[request_id]
        assert request.completed_at is not None

        # Test that processing methods complete requests
        for right in [
            DataSubjectRight.ERASURE,
            DataSubjectRight.PORTABILITY,
            DataSubjectRight.RECTIFICATION,
        ]:
            request_id = await gdpr_manager.handle_subject_request("user_123", right)
            request = gdpr_manager.subject_requests[request_id]
            assert request.status == "completed"
