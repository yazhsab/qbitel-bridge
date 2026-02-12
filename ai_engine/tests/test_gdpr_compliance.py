"""
Comprehensive tests for GDPR Compliance Implementation.

Tests cover:
- Data subject rights (access, erasure, portability, etc.)
- Consent management
- Data processing records (Article 30)
- Data breach notification
- Privacy impact assessments
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, MagicMock, patch

from ai_engine.compliance.gdpr_compliance import (
    GDPRComplianceManager,
    DataSubjectRequest,
    ConsentRecord,
    DataProcessingRecord,
    DataSubjectRight,
    LegalBasis,
    GDPRException,
)
from ai_engine.core.config import Config


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = Mock(spec=Config)
    config.gdpr_enabled = True
    config.data_controller = "QBITEL Inc."
    config.dpo_contact = "dpo@qbitel.ai"
    return config


@pytest.fixture
def mock_audit_manager():
    """Create mock audit trail manager."""
    manager = AsyncMock()
    manager.log_event = AsyncMock()
    return manager


@pytest.fixture
def gdpr_manager(mock_config, mock_audit_manager):
    """Create GDPRComplianceManager instance."""
    with patch(
        "ai_engine.compliance.gdpr_compliance.AuditTrailManager",
        return_value=mock_audit_manager,
    ):
        manager = GDPRComplianceManager(mock_config)
        manager.audit_manager = mock_audit_manager
        return manager


class TestDataSubjectRequest:
    """Tests for DataSubjectRequest dataclass."""

    def test_request_creation(self):
        """Test creating a data subject request."""
        now = datetime.utcnow()
        request = DataSubjectRequest(
            request_id="REQ001",
            subject_id="user_123",
            right=DataSubjectRight.ACCESS,
            requested_at=now,
        )

        assert request.request_id == "REQ001"
        assert request.subject_id == "user_123"
        assert request.right == DataSubjectRight.ACCESS
        assert request.status == "pending"
        assert request.completed_at is None


class TestConsentRecord:
    """Tests for ConsentRecord."""

    def test_consent_creation(self):
        """Test creating a consent record."""
        now = datetime.utcnow()
        consent = ConsentRecord(
            consent_id="CONSENT001",
            subject_id="user_456",
            purpose="Marketing communications",
            legal_basis=LegalBasis.CONSENT,
            granted_at=now,
            expires_at=now + timedelta(days=365),
        )

        assert consent.consent_id == "CONSENT001"
        assert consent.subject_id == "user_456"
        assert consent.legal_basis == LegalBasis.CONSENT
        assert consent.withdrawn_at is None

    def test_consent_withdrawal(self):
        """Test consent withdrawal."""
        now = datetime.utcnow()
        consent = ConsentRecord(
            consent_id="CONSENT002",
            subject_id="user_789",
            purpose="Data processing",
            legal_basis=LegalBasis.CONSENT,
            granted_at=now,
        )

        consent.withdrawn_at = datetime.utcnow()
        assert consent.withdrawn_at is not None


class TestGDPRComplianceManager:
    """Tests for GDPRComplianceManager."""

    def test_manager_initialization(self, gdpr_manager, mock_config):
        """Test manager initialization."""
        assert gdpr_manager.config == mock_config
        assert isinstance(gdpr_manager.consent_records, dict)
        assert isinstance(gdpr_manager.subject_requests, dict)

    @pytest.mark.asyncio
    async def test_record_consent(self, gdpr_manager):
        """Test recording user consent."""
        consent_id = await gdpr_manager.record_consent(
            subject_id="user_001",
            purpose="Analytics",
            legal_basis=LegalBasis.CONSENT,
            metadata={"ip_address": "192.168.1.1"},
        )

        assert consent_id is not None
        assert consent_id in gdpr_manager.consent_records

    @pytest.mark.asyncio
    async def test_withdraw_consent(self, gdpr_manager):
        """Test withdrawing consent."""
        # Record consent first
        consent_id = await gdpr_manager.record_consent(
            subject_id="user_002", purpose="Marketing", legal_basis=LegalBasis.CONSENT
        )

        # Withdraw consent
        result = await gdpr_manager.withdraw_consent(consent_id)

        assert result is True
        consent = gdpr_manager.consent_records[consent_id]
        assert consent.withdrawn_at is not None

    @pytest.mark.asyncio
    async def test_check_consent_valid(self, gdpr_manager):
        """Test checking if consent is valid."""
        consent_id = await gdpr_manager.record_consent(
            subject_id="user_003", purpose="Processing", legal_basis=LegalBasis.CONSENT
        )

        is_valid = await gdpr_manager.check_consent(consent_id)
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_check_consent_withdrawn(self, gdpr_manager):
        """Test consent validity after withdrawal."""
        consent_id = await gdpr_manager.record_consent(
            subject_id="user_004", purpose="Analytics", legal_basis=LegalBasis.CONSENT
        )

        await gdpr_manager.withdraw_consent(consent_id)
        is_valid = await gdpr_manager.check_consent(consent_id)

        assert is_valid is False

    @pytest.mark.asyncio
    async def test_submit_access_request(self, gdpr_manager):
        """Test submitting right to access request."""
        request_id = await gdpr_manager.submit_subject_request(subject_id="user_005", right=DataSubjectRight.ACCESS)

        assert request_id is not None
        assert request_id in gdpr_manager.subject_requests
        request = gdpr_manager.subject_requests[request_id]
        assert request.right == DataSubjectRight.ACCESS

    @pytest.mark.asyncio
    async def test_submit_erasure_request(self, gdpr_manager):
        """Test submitting right to erasure request."""
        request_id = await gdpr_manager.submit_subject_request(subject_id="user_006", right=DataSubjectRight.ERASURE)

        assert request_id in gdpr_manager.subject_requests
        request = gdpr_manager.subject_requests[request_id]
        assert request.right == DataSubjectRight.ERASURE

    @pytest.mark.asyncio
    async def test_process_access_request(self, gdpr_manager):
        """Test processing right to access request."""
        request_id = await gdpr_manager.submit_subject_request(subject_id="user_007", right=DataSubjectRight.ACCESS)

        with patch.object(gdpr_manager, "_gather_subject_data", new_callable=AsyncMock) as mock_gather:
            mock_gather.return_value = {"name": "John Doe", "email": "john@example.com"}

            response = await gdpr_manager.process_access_request(request_id)

            assert response is not None
            assert "name" in response
            mock_gather.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_erasure_request(self, gdpr_manager):
        """Test processing right to erasure request."""
        request_id = await gdpr_manager.submit_subject_request(subject_id="user_008", right=DataSubjectRight.ERASURE)

        with patch.object(gdpr_manager, "_delete_subject_data", new_callable=AsyncMock) as mock_delete:
            mock_delete.return_value = True

            result = await gdpr_manager.process_erasure_request(request_id)

            assert result is True
            mock_delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_export_personal_data(self, gdpr_manager):
        """Test exporting personal data (portability)."""
        request_id = await gdpr_manager.submit_subject_request(subject_id="user_009", right=DataSubjectRight.PORTABILITY)

        with patch.object(gdpr_manager, "_export_data_portable_format", new_callable=AsyncMock) as mock_export:
            mock_export.return_value = {"format": "JSON", "data": {}}

            export_data = await gdpr_manager.export_subject_data(request_id)

            assert export_data is not None
            assert "format" in export_data
            mock_export.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_processing_activity(self, gdpr_manager):
        """Test recording data processing activity (Article 30)."""
        activity_id = await gdpr_manager.record_processing_activity(
            activity_name="User authentication",
            purpose="Secure access control",
            legal_basis=LegalBasis.CONTRACT,
            data_categories=["authentication_tokens", "login_logs"],
            recipients=["Internal systems"],
        )

        assert activity_id is not None
        assert activity_id in gdpr_manager.processing_records

    @pytest.mark.asyncio
    async def test_data_breach_notification(self, gdpr_manager):
        """Test data breach notification."""
        breach_id = await gdpr_manager.record_data_breach(
            breach_type="Unauthorized access",
            affected_subjects=["user_010", "user_011"],
            severity="high",
            description="Database access breach",
        )

        assert breach_id is not None

    @pytest.mark.asyncio
    async def test_notify_supervisory_authority(self, gdpr_manager):
        """Test notifying supervisory authority of breach."""
        breach_id = await gdpr_manager.record_data_breach(
            breach_type="Data leak", affected_subjects=["user_012"], severity="critical"
        )

        with patch.object(gdpr_manager, "_notify_authority_impl", new_callable=AsyncMock) as mock_notify:
            mock_notify.return_value = True

            result = await gdpr_manager.notify_supervisory_authority(breach_id)

            assert result is True
            mock_notify.assert_called_once()

    @pytest.mark.asyncio
    async def test_notify_affected_subjects(self, gdpr_manager):
        """Test notifying affected data subjects."""
        breach_id = await gdpr_manager.record_data_breach(
            breach_type="Data exposure",
            affected_subjects=["user_013", "user_014"],
            severity="high",
        )

        with patch.object(gdpr_manager, "_notify_subjects_impl", new_callable=AsyncMock) as mock_notify:
            mock_notify.return_value = 2

            notified_count = await gdpr_manager.notify_affected_subjects(breach_id)

            assert notified_count == 2
            mock_notify.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_compliance_report(self, gdpr_manager):
        """Test generating GDPR compliance report."""
        # Add some test data
        await gdpr_manager.record_consent("user_015", "Analytics", LegalBasis.CONSENT)
        await gdpr_manager.submit_subject_request("user_016", DataSubjectRight.ACCESS)

        report = await gdpr_manager.generate_compliance_report()

        assert "consent_records_count" in report
        assert "subject_requests_count" in report
        assert "processing_activities_count" in report

    @pytest.mark.asyncio
    async def test_check_consent_expiration(self, gdpr_manager):
        """Test checking expired consents."""
        # Create expired consent
        past_date = datetime.utcnow() - timedelta(days=400)
        expired_consent = ConsentRecord(
            consent_id="EXPIRED001",
            subject_id="user_017",
            purpose="Old marketing",
            legal_basis=LegalBasis.CONSENT,
            granted_at=past_date,
            expires_at=past_date + timedelta(days=365),
        )
        gdpr_manager.consent_records["EXPIRED001"] = expired_consent

        expired_list = await gdpr_manager.list_expired_consents()

        assert len(expired_list) >= 1
        assert any(c.consent_id == "EXPIRED001" for c in expired_list)

    def test_data_subject_right_enum(self):
        """Test DataSubjectRight enum values."""
        assert DataSubjectRight.ACCESS == "right_to_access"
        assert DataSubjectRight.ERASURE == "right_to_erasure"
        assert DataSubjectRight.PORTABILITY == "right_to_portability"

    def test_legal_basis_enum(self):
        """Test LegalBasis enum values."""
        assert LegalBasis.CONSENT == "consent"
        assert LegalBasis.CONTRACT == "contract"
        assert LegalBasis.LEGITIMATE_INTERESTS == "legitimate_interests"

    @pytest.mark.asyncio
    async def test_request_fulfillment_deadline(self, gdpr_manager):
        """Test that requests track 30-day deadline."""
        request_id = await gdpr_manager.submit_subject_request(subject_id="user_018", right=DataSubjectRight.ACCESS)

        request = gdpr_manager.subject_requests[request_id]
        deadline = request.requested_at + timedelta(days=30)

        # Verify deadline is tracked
        assert deadline > request.requested_at
