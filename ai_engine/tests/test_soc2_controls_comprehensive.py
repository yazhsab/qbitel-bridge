"""
Comprehensive Unit Tests for SOC2 Controls Manager
Tests all functionality in ai_engine/compliance/soc2_controls.py
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from ai_engine.compliance.soc2_controls import (
    SOC2ControlsManager,
    SOC2Exception,
    TrustServiceCriteria,
    ControlStatus,
    SOC2Control,
)
from ai_engine.compliance.audit_trail import AuditTrailManager, EventType
from ai_engine.core.config import Config


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    return Config()


@pytest.fixture
def mock_audit_manager():
    """Create mock audit manager."""
    manager = Mock(spec=AuditTrailManager)
    manager.record_compliance_event = AsyncMock()
    return manager


@pytest.fixture
def soc2_manager(mock_config, mock_audit_manager):
    """Create SOC2ControlsManager instance."""
    return SOC2ControlsManager(mock_config, mock_audit_manager)


class TestSOC2ControlsManagerInitialization:
    """Test SOC2ControlsManager initialization."""

    def test_initialization(self, soc2_manager, mock_config, mock_audit_manager):
        """Test successful initialization."""
        assert soc2_manager.config == mock_config
        assert soc2_manager.audit_manager == mock_audit_manager
        assert len(soc2_manager.controls) > 0

    def test_security_controls_initialized(self, soc2_manager):
        """Test security controls are initialized."""
        assert "CC6.1" in soc2_manager.controls
        assert "CC6.2" in soc2_manager.controls
        assert "CC6.3" in soc2_manager.controls
        assert "CC6.6" in soc2_manager.controls
        assert "CC6.7" in soc2_manager.controls
        assert "CC6.8" in soc2_manager.controls

    def test_availability_controls_initialized(self, soc2_manager):
        """Test availability controls are initialized."""
        assert "A1.1" in soc2_manager.controls
        assert "A1.2" in soc2_manager.controls
        assert "A1.3" in soc2_manager.controls

    def test_processing_integrity_controls_initialized(self, soc2_manager):
        """Test processing integrity controls are initialized."""
        assert "PI1.1" in soc2_manager.controls
        assert "PI1.4" in soc2_manager.controls

    def test_confidentiality_controls_initialized(self, soc2_manager):
        """Test confidentiality controls are initialized."""
        assert "C1.1" in soc2_manager.controls
        assert "C1.2" in soc2_manager.controls

    def test_privacy_controls_initialized(self, soc2_manager):
        """Test privacy controls are initialized."""
        assert "P3.1" in soc2_manager.controls
        assert "P4.1" in soc2_manager.controls


class TestControlDetails:
    """Test control details and properties."""

    def test_cc61_control_details(self, soc2_manager):
        """Test CC6.1 control details."""
        control = soc2_manager.controls["CC6.1"]
        
        assert control.control_id == "CC6.1"
        assert control.criteria == TrustServiceCriteria.SECURITY
        assert control.title == "Logical and Physical Access Controls"
        assert control.status == ControlStatus.IMPLEMENTED
        assert control.responsible_party == "Security Team"
        assert len(control.evidence) > 0

    def test_a11_control_details(self, soc2_manager):
        """Test A1.1 control details."""
        control = soc2_manager.controls["A1.1"]
        
        assert control.control_id == "A1.1"
        assert control.criteria == TrustServiceCriteria.AVAILABILITY
        assert "capacity" in control.title.lower()
        assert control.status == ControlStatus.IMPLEMENTED

    def test_pi11_control_details(self, soc2_manager):
        """Test PI1.1 control details."""
        control = soc2_manager.controls["PI1.1"]
        
        assert control.control_id == "PI1.1"
        assert control.criteria == TrustServiceCriteria.PROCESSING_INTEGRITY
        assert control.status == ControlStatus.IMPLEMENTED

    def test_c11_control_details(self, soc2_manager):
        """Test C1.1 control details."""
        control = soc2_manager.controls["C1.1"]
        
        assert control.control_id == "C1.1"
        assert control.criteria == TrustServiceCriteria.CONFIDENTIALITY
        assert control.status == ControlStatus.IMPLEMENTED

    def test_p31_control_details(self, soc2_manager):
        """Test P3.1 control details."""
        control = soc2_manager.controls["P3.1"]
        
        assert control.control_id == "P3.1"
        assert control.criteria == TrustServiceCriteria.PRIVACY
        assert "notice" in control.title.lower()


class TestControlTesting:
    """Test control testing functionality."""

    @pytest.mark.asyncio
    async def test_test_control_success(self, soc2_manager, mock_audit_manager):
        """Test successful control testing."""
        result = await soc2_manager.test_control("CC6.1")
        
        assert result["control_id"] == "CC6.1"
        assert "tested_at" in result
        assert result["status"] == "passed"
        assert "findings" in result
        assert "recommendations" in result
        
        mock_audit_manager.record_compliance_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_test_control_updates_control(self, soc2_manager):
        """Test control testing updates control record."""
        control = soc2_manager.controls["CC6.2"]
        assert control.last_tested is None
        
        await soc2_manager.test_control("CC6.2")
        
        assert control.last_tested is not None
        assert control.test_results is not None

    @pytest.mark.asyncio
    async def test_test_control_nonexistent(self, soc2_manager):
        """Test testing non-existent control."""
        with pytest.raises(SOC2Exception, match="Control .* not found"):
            await soc2_manager.test_control("INVALID")

    @pytest.mark.asyncio
    async def test_test_control_without_audit_manager(self, mock_config):
        """Test control testing without audit manager."""
        manager = SOC2ControlsManager(mock_config, None)
        
        result = await manager.test_control("CC6.1")
        
        assert result is not None
        assert result["status"] == "passed"

    @pytest.mark.asyncio
    async def test_test_multiple_controls(self, soc2_manager):
        """Test testing multiple controls."""
        control_ids = ["CC6.1", "CC6.2", "A1.1"]
        
        for control_id in control_ids:
            result = await soc2_manager.test_control(control_id)
            assert result["control_id"] == control_id


class TestComplianceVerification:
    """Test compliance verification."""

    @pytest.mark.asyncio
    async def test_verify_compliance_all_implemented(self, soc2_manager):
        """Test compliance verification when all controls implemented."""
        status = await soc2_manager.verify_compliance()
        
        assert status["compliant"] is True
        assert "criteria_status" in status
        assert "issues" in status
        assert "recommendations" in status
        assert "verified_at" in status

    @pytest.mark.asyncio
    async def test_verify_compliance_criteria_status(self, soc2_manager):
        """Test compliance verification includes criteria status."""
        status = await soc2_manager.verify_compliance()
        
        for criteria in TrustServiceCriteria:
            assert criteria.value in status["criteria_status"]
            criteria_status = status["criteria_status"][criteria.value]
            assert "implemented" in criteria_status
            assert "total" in criteria_status
            assert "percentage" in criteria_status

    @pytest.mark.asyncio
    async def test_verify_compliance_with_unimplemented(self, soc2_manager):
        """Test compliance verification with unimplemented controls."""
        # Mark a control as not implemented
        soc2_manager.controls["CC6.1"].status = ControlStatus.NOT_IMPLEMENTED
        
        status = await soc2_manager.verify_compliance()
        
        assert status["compliant"] is False
        assert len(status["issues"]) > 0

    @pytest.mark.asyncio
    async def test_verify_compliance_untested_controls(self, soc2_manager):
        """Test verification detects untested controls."""
        # Set a control as tested long ago
        control = soc2_manager.controls["CC6.1"]
        control.last_tested = datetime.utcnow() - timedelta(days=100)
        
        status = await soc2_manager.verify_compliance()
        
        assert any("testing" in rec.lower() for rec in status["recommendations"])

    @pytest.mark.asyncio
    async def test_verify_compliance_partially_implemented(self, soc2_manager):
        """Test compliance with partially implemented controls."""
        soc2_manager.controls["CC6.2"].status = ControlStatus.PARTIALLY_IMPLEMENTED
        
        status = await soc2_manager.verify_compliance()
        
        assert status["compliant"] is False


class TestControlRetrieval:
    """Test control retrieval methods."""

    def test_get_control_exists(self, soc2_manager):
        """Test getting existing control."""
        control = soc2_manager.get_control("CC6.1")
        
        assert control is not None
        assert control.control_id == "CC6.1"

    def test_get_control_nonexistent(self, soc2_manager):
        """Test getting non-existent control."""
        control = soc2_manager.get_control("INVALID")
        
        assert control is None

    def test_get_controls_by_criteria_security(self, soc2_manager):
        """Test getting controls by security criteria."""
        controls = soc2_manager.get_controls_by_criteria(TrustServiceCriteria.SECURITY)
        
        assert len(controls) > 0
        assert all(c.criteria == TrustServiceCriteria.SECURITY for c in controls)

    def test_get_controls_by_criteria_availability(self, soc2_manager):
        """Test getting controls by availability criteria."""
        controls = soc2_manager.get_controls_by_criteria(TrustServiceCriteria.AVAILABILITY)
        
        assert len(controls) == 3  # A1.1, A1.2, A1.3
        assert all(c.criteria == TrustServiceCriteria.AVAILABILITY for c in controls)

    def test_get_controls_by_criteria_all(self, soc2_manager):
        """Test getting controls for all criteria."""
        for criteria in TrustServiceCriteria:
            controls = soc2_manager.get_controls_by_criteria(criteria)
            assert len(controls) > 0


class TestStatistics:
    """Test statistics functionality."""

    def test_get_statistics_structure(self, soc2_manager):
        """Test statistics structure."""
        stats = soc2_manager.get_statistics()
        
        assert "total_controls" in stats
        assert "by_status" in stats
        assert "by_criteria" in stats
        assert "testing_status" in stats

    def test_get_statistics_total_controls(self, soc2_manager):
        """Test total controls count."""
        stats = soc2_manager.get_statistics()
        
        assert stats["total_controls"] == len(soc2_manager.controls)
        assert stats["total_controls"] > 0

    def test_get_statistics_by_status(self, soc2_manager):
        """Test statistics by status."""
        stats = soc2_manager.get_statistics()
        
        for status in ControlStatus:
            assert status.value in stats["by_status"]

    def test_get_statistics_by_criteria(self, soc2_manager):
        """Test statistics by criteria."""
        stats = soc2_manager.get_statistics()
        
        for criteria in TrustServiceCriteria:
            assert criteria.value in stats["by_criteria"]

    def test_get_statistics_testing_status(self, soc2_manager):
        """Test testing status statistics."""
        stats = soc2_manager.get_statistics()
        
        assert "tested_last_30_days" in stats["testing_status"]
        assert "tested_last_90_days" in stats["testing_status"]
        assert "never_tested" in stats["testing_status"]

    @pytest.mark.asyncio
    async def test_get_statistics_after_testing(self, soc2_manager):
        """Test statistics after testing controls."""
        # Test a control
        await soc2_manager.test_control("CC6.1")
        
        stats = soc2_manager.get_statistics()
        
        assert stats["testing_status"]["tested_last_30_days"] > 0


class TestComplianceReport:
    """Test compliance report generation."""

    @pytest.mark.asyncio
    async def test_generate_compliance_report(self, soc2_manager):
        """Test generating compliance report."""
        report = await soc2_manager.generate_compliance_report()
        
        assert "report_metadata" in report
        assert "executive_summary" in report
        assert "controls_assessment" in report
        assert "compliance_status" in report
        assert "statistics" in report
        assert "recommendations" in report

    @pytest.mark.asyncio
    async def test_report_metadata(self, soc2_manager):
        """Test report metadata."""
        report = await soc2_manager.generate_compliance_report()
        
        metadata = report["report_metadata"]
        assert "generated_at" in metadata
        assert metadata["report_type"] == "SOC2 Type II Compliance"
        assert "period_covered" in metadata

    @pytest.mark.asyncio
    async def test_report_executive_summary(self, soc2_manager):
        """Test report executive summary."""
        report = await soc2_manager.generate_compliance_report()
        
        summary = report["executive_summary"]
        assert "total_controls" in summary
        assert "implemented_controls" in summary
        assert "compliance_percentage" in summary
        assert "overall_status" in summary

    @pytest.mark.asyncio
    async def test_report_controls_assessment(self, soc2_manager):
        """Test report controls assessment."""
        report = await soc2_manager.generate_compliance_report()
        
        assessment = report["controls_assessment"]
        assert len(assessment) == len(soc2_manager.controls)
        
        for control_data in assessment:
            assert "control_id" in control_data
            assert "criteria" in control_data
            assert "title" in control_data
            assert "status" in control_data
            assert "evidence_count" in control_data

    @pytest.mark.asyncio
    async def test_report_includes_compliance_status(self, soc2_manager):
        """Test report includes compliance status."""
        report = await soc2_manager.generate_compliance_report()
        
        assert "compliance_status" in report
        assert "compliant" in report["compliance_status"]

    @pytest.mark.asyncio
    async def test_report_compliance_percentage(self, soc2_manager):
        """Test report calculates compliance percentage."""
        report = await soc2_manager.generate_compliance_report()
        
        summary = report["executive_summary"]
        percentage = summary["compliance_percentage"]
        
        assert 0 <= percentage <= 100


class TestDataClasses:
    """Test data classes."""

    def test_soc2_control_creation(self):
        """Test creating SOC2Control."""
        control = SOC2Control(
            control_id="TEST.1",
            criteria=TrustServiceCriteria.SECURITY,
            title="Test Control",
            description="Test Description",
            status=ControlStatus.IMPLEMENTED,
            evidence=["evidence1", "evidence2"],
            responsible_party="Test Team"
        )
        
        assert control.control_id == "TEST.1"
        assert control.criteria == TrustServiceCriteria.SECURITY
        assert control.title == "Test Control"
        assert control.status == ControlStatus.IMPLEMENTED
        assert len(control.evidence) == 2

    def test_soc2_control_defaults(self):
        """Test SOC2Control default values."""
        control = SOC2Control(
            control_id="TEST.1",
            criteria=TrustServiceCriteria.SECURITY,
            title="Test",
            description="Test",
            status=ControlStatus.IMPLEMENTED
        )
        
        assert control.evidence == []
        assert control.last_tested is None
        assert control.test_results is None
        assert control.responsible_party == ""


class TestEnums:
    """Test enum classes."""

    def test_trust_service_criteria_values(self):
        """Test TrustServiceCriteria enum values."""
        assert TrustServiceCriteria.SECURITY.value == "security"
        assert TrustServiceCriteria.AVAILABILITY.value == "availability"
        assert TrustServiceCriteria.PROCESSING_INTEGRITY.value == "processing_integrity"
        assert TrustServiceCriteria.CONFIDENTIALITY.value == "confidentiality"
        assert TrustServiceCriteria.PRIVACY.value == "privacy"

    def test_control_status_values(self):
        """Test ControlStatus enum values."""
        assert ControlStatus.IMPLEMENTED.value == "implemented"
        assert ControlStatus.PARTIALLY_IMPLEMENTED.value == "partially_implemented"
        assert ControlStatus.NOT_IMPLEMENTED.value == "not_implemented"
        assert ControlStatus.NOT_APPLICABLE.value == "not_applicable"


class TestControlEvidence:
    """Test control evidence handling."""

    def test_control_has_evidence(self, soc2_manager):
        """Test controls have evidence."""
        for control in soc2_manager.controls.values():
            if control.status == ControlStatus.IMPLEMENTED:
                assert len(control.evidence) > 0

    def test_security_control_evidence(self, soc2_manager):
        """Test security controls have appropriate evidence."""
        control = soc2_manager.controls["CC6.1"]
        
        assert any("IAM" in evidence or "MFA" in evidence or "access" in evidence.lower() 
                  for evidence in control.evidence)

    def test_availability_control_evidence(self, soc2_manager):
        """Test availability controls have appropriate evidence."""
        control = soc2_manager.controls["A1.2"]
        
        assert any("backup" in evidence.lower() or "recovery" in evidence.lower() 
                  for evidence in control.evidence)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_test_control_multiple_times(self, soc2_manager):
        """Test testing same control multiple times."""
        control_id = "CC6.1"
        
        result1 = await soc2_manager.test_control(control_id)
        first_test_time = soc2_manager.controls[control_id].last_tested
        
        await asyncio.sleep(0.01)  # Small delay
        
        result2 = await soc2_manager.test_control(control_id)
        second_test_time = soc2_manager.controls[control_id].last_tested
        
        assert result1 is not None
        assert result2 is not None
        assert second_test_time > first_test_time

    @pytest.mark.asyncio
    async def test_verify_compliance_empty_controls(self, mock_config):
        """Test verification with no controls."""
        manager = SOC2ControlsManager(mock_config, None)
        manager.controls.clear()
        
        status = await manager.verify_compliance()
        
        # Should handle gracefully
        assert "criteria_status" in status

    def test_get_controls_by_criteria_empty(self, mock_config):
        """Test getting controls when none exist for criteria."""
        manager = SOC2ControlsManager(mock_config, None)
        manager.controls.clear()
        
        controls = manager.get_controls_by_criteria(TrustServiceCriteria.SECURITY)
        
        assert len(controls) == 0

    @pytest.mark.asyncio
    async def test_report_with_mixed_statuses(self, soc2_manager):
        """Test report generation with mixed control statuses."""
        soc2_manager.controls["CC6.1"].status = ControlStatus.IMPLEMENTED
        soc2_manager.controls["CC6.2"].status = ControlStatus.PARTIALLY_IMPLEMENTED
        soc2_manager.controls["CC6.3"].status = ControlStatus.NOT_IMPLEMENTED
        
        report = await soc2_manager.generate_compliance_report()
        
        assert report is not None
        assert "executive_summary" in report

    def test_statistics_with_no_tests(self, soc2_manager):
        """Test statistics when no controls have been tested."""
        stats = soc2_manager.get_statistics()
        
        assert stats["testing_status"]["never_tested"] == len(soc2_manager.controls)
        assert stats["testing_status"]["tested_last_30_days"] == 0

    @pytest.mark.asyncio
    async def test_control_testing_records_timestamp(self, soc2_manager):
        """Test control testing records accurate timestamp."""
        before_test = datetime.utcnow()
        
        await soc2_manager.test_control("CC6.1")
        
        after_test = datetime.utcnow()
        control = soc2_manager.controls["CC6.1"]
        
        assert before_test <= control.last_tested <= after_test

    def test_all_controls_have_required_fields(self, soc2_manager):
        """Test all controls have required fields."""
        for control in soc2_manager.controls.values():
            assert control.control_id
            assert control.criteria
            assert control.title
            assert control.description
            assert control.status
            assert control.responsible_party


class TestControlCoverage:
    """Test control coverage across criteria."""

    def test_security_criteria_coverage(self, soc2_manager):
        """Test security criteria has adequate control coverage."""
        controls = soc2_manager.get_controls_by_criteria(TrustServiceCriteria.SECURITY)
        
        assert len(controls) >= 6  # At least CC6.1-CC6.8

    def test_all_criteria_have_controls(self, soc2_manager):
        """Test all trust service criteria have controls."""
        for criteria in TrustServiceCriteria:
            controls = soc2_manager.get_controls_by_criteria(criteria)
            assert len(controls) > 0, f"No controls for {criteria.value}"

    def test_control_distribution(self, soc2_manager):
        """Test controls are distributed across criteria."""
        stats = soc2_manager.get_statistics()
        
        # Each criteria should have at least one control
        for criteria in TrustServiceCriteria:
            assert stats["by_criteria"][criteria.value] > 0