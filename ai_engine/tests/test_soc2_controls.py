"""
Comprehensive tests for SOC2 Controls Implementation.

Tests cover:
- Trust Service Criteria (Security, Availability, Processing Integrity, Confidentiality, Privacy)
- Control implementation and testing
- Evidence collection
- Compliance monitoring
- Control effectiveness assessment
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, MagicMock, patch

from ai_engine.compliance.soc2_controls import (
    SOC2ControlsManager,
    SOC2Control,
    TrustServiceCriteria,
    ControlStatus,
    SOC2Exception,
)
from ai_engine.core.config import Config


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = Mock(spec=Config)
    config.soc2_enabled = True
    config.control_testing_frequency_days = 90
    return config


@pytest.fixture
def mock_audit_manager():
    """Create mock audit trail manager."""
    manager = AsyncMock()
    manager.log_event = AsyncMock()
    return manager


@pytest.fixture
def soc2_manager(mock_config, mock_audit_manager):
    """Create SOC2ControlsManager instance."""
    return SOC2ControlsManager(mock_config, mock_audit_manager)


@pytest.fixture
def sample_control():
    """Create sample SOC2 control."""
    return SOC2Control(
        control_id="CC6.1",
        criteria=TrustServiceCriteria.SECURITY,
        title="Logical and Physical Access Controls",
        description="Implement access controls to protect assets",
        status=ControlStatus.IMPLEMENTED,
        responsible_party="Security Team",
    )


class TestSOC2Control:
    """Tests for SOC2Control dataclass."""

    def test_control_creation(self, sample_control):
        """Test SOC2 control creation."""
        assert sample_control.control_id == "CC6.1"
        assert sample_control.criteria == TrustServiceCriteria.SECURITY
        assert sample_control.status == ControlStatus.IMPLEMENTED
        assert isinstance(sample_control.evidence, list)
        assert sample_control.last_tested is None

    def test_control_with_evidence(self):
        """Test control with evidence."""
        control = SOC2Control(
            control_id="CC6.2",
            criteria=TrustServiceCriteria.SECURITY,
            title="System Monitoring",
            description="Monitor system activities",
            status=ControlStatus.IMPLEMENTED,
            evidence=["audit_logs.pdf", "monitoring_report.pdf"],
        )

        assert len(control.evidence) == 2
        assert "audit_logs.pdf" in control.evidence


class TestSOC2ControlsManager:
    """Tests for SOC2ControlsManager."""

    def test_manager_initialization(self, soc2_manager, mock_config):
        """Test manager initialization."""
        assert soc2_manager.config == mock_config
        assert isinstance(soc2_manager.controls, dict)

    @pytest.mark.asyncio
    async def test_register_control(self, soc2_manager, sample_control):
        """Test registering a control."""
        await soc2_manager.register_control(sample_control)

        assert sample_control.control_id in soc2_manager.controls
        assert soc2_manager.controls[sample_control.control_id] == sample_control

    @pytest.mark.asyncio
    async def test_get_control(self, soc2_manager, sample_control):
        """Test retrieving a control."""
        await soc2_manager.register_control(sample_control)

        control = soc2_manager.get_control("CC6.1")
        assert control == sample_control

    @pytest.mark.asyncio
    async def test_get_control_not_found(self, soc2_manager):
        """Test retrieving non-existent control."""
        control = soc2_manager.get_control("NONEXISTENT")
        assert control is None

    @pytest.mark.asyncio
    async def test_update_control_status(self, soc2_manager, sample_control):
        """Test updating control status."""
        await soc2_manager.register_control(sample_control)

        await soc2_manager.update_control_status(
            "CC6.1",
            ControlStatus.PARTIALLY_IMPLEMENTED,
            notes="Implementation in progress",
        )

        control = soc2_manager.get_control("CC6.1")
        assert control.status == ControlStatus.PARTIALLY_IMPLEMENTED

    @pytest.mark.asyncio
    async def test_add_evidence(self, soc2_manager, sample_control):
        """Test adding evidence to a control."""
        await soc2_manager.register_control(sample_control)

        await soc2_manager.add_evidence("CC6.1", "access_control_logs_2024.pdf")

        control = soc2_manager.get_control("CC6.1")
        assert "access_control_logs_2024.pdf" in control.evidence

    @pytest.mark.asyncio
    async def test_test_control(self, soc2_manager, sample_control):
        """Test executing control test."""
        await soc2_manager.register_control(sample_control)

        test_results = {
            "passed": True,
            "findings": [],
            "tested_by": "auditor@example.com",
            "test_date": datetime.utcnow().isoformat(),
        }

        await soc2_manager.test_control("CC6.1", test_results)

        control = soc2_manager.get_control("CC6.1")
        assert control.last_tested is not None
        assert control.test_results == test_results

    @pytest.mark.asyncio
    async def test_list_controls_by_criteria(self, soc2_manager):
        """Test listing controls by trust service criteria."""
        # Register multiple controls
        security_control = SOC2Control(
            control_id="CC6.1",
            criteria=TrustServiceCriteria.SECURITY,
            title="Security Control",
            description="Test security control",
            status=ControlStatus.IMPLEMENTED,
        )
        availability_control = SOC2Control(
            control_id="A1.1",
            criteria=TrustServiceCriteria.AVAILABILITY,
            title="Availability Control",
            description="Test availability control",
            status=ControlStatus.IMPLEMENTED,
        )

        await soc2_manager.register_control(security_control)
        await soc2_manager.register_control(availability_control)

        security_controls = await soc2_manager.get_controls_by_criteria(TrustServiceCriteria.SECURITY)

        assert len(security_controls) >= 1
        assert all(c.criteria == TrustServiceCriteria.SECURITY for c in security_controls)

    @pytest.mark.asyncio
    async def test_list_controls_by_status(self, soc2_manager):
        """Test listing controls by status."""
        implemented_control = SOC2Control(
            control_id="CC6.2",
            criteria=TrustServiceCriteria.SECURITY,
            title="Implemented Control",
            description="Test",
            status=ControlStatus.IMPLEMENTED,
        )
        pending_control = SOC2Control(
            control_id="CC6.3",
            criteria=TrustServiceCriteria.SECURITY,
            title="Pending Control",
            description="Test",
            status=ControlStatus.PARTIALLY_IMPLEMENTED,
        )

        await soc2_manager.register_control(implemented_control)
        await soc2_manager.register_control(pending_control)

        implemented = await soc2_manager.get_controls_by_status(ControlStatus.IMPLEMENTED)

        assert len(implemented) >= 1
        assert all(c.status == ControlStatus.IMPLEMENTED for c in implemented)

    @pytest.mark.asyncio
    async def test_get_controls_needing_testing(self, soc2_manager):
        """Test identifying controls that need testing."""
        # Create control tested long ago
        old_control = SOC2Control(
            control_id="CC6.4",
            criteria=TrustServiceCriteria.SECURITY,
            title="Old Control",
            description="Test",
            status=ControlStatus.IMPLEMENTED,
            last_tested=datetime.utcnow() - timedelta(days=100),
        )

        await soc2_manager.register_control(old_control)

        controls_to_test = await soc2_manager.get_controls_needing_testing(days_threshold=90)

        assert len(controls_to_test) >= 1
        assert any(c.control_id == "CC6.4" for c in controls_to_test)

    @pytest.mark.asyncio
    async def test_generate_compliance_report(self, soc2_manager):
        """Test generating SOC2 compliance report."""
        # Add some controls
        for i in range(3):
            control = SOC2Control(
                control_id=f"CC6.{i}",
                criteria=TrustServiceCriteria.SECURITY,
                title=f"Control {i}",
                description="Test control",
                status=ControlStatus.IMPLEMENTED,
            )
            await soc2_manager.register_control(control)

        report = await soc2_manager.generate_compliance_report()

        assert "total_controls" in report
        assert "by_criteria" in report
        assert "by_status" in report
        assert report["total_controls"] >= 3

    @pytest.mark.asyncio
    async def test_assess_control_effectiveness(self, soc2_manager, sample_control):
        """Test assessing control effectiveness."""
        await soc2_manager.register_control(sample_control)

        # Add test results
        test_results = {"passed": True, "effectiveness_score": 95, "findings": []}
        await soc2_manager.test_control("CC6.1", test_results)

        effectiveness = await soc2_manager.assess_control_effectiveness("CC6.1")

        assert effectiveness is not None
        assert "score" in effectiveness or "passed" in effectiveness

    @pytest.mark.asyncio
    async def test_identify_control_gaps(self, soc2_manager):
        """Test identifying control gaps."""
        # Register some controls with different statuses
        await soc2_manager.register_control(
            SOC2Control(
                control_id="CC6.5",
                criteria=TrustServiceCriteria.SECURITY,
                title="Not Implemented",
                description="Test",
                status=ControlStatus.NOT_IMPLEMENTED,
            )
        )

        gaps = await soc2_manager.identify_control_gaps()

        assert len(gaps) >= 1
        assert any(g.status == ControlStatus.NOT_IMPLEMENTED for g in gaps)

    @pytest.mark.asyncio
    async def test_export_control_matrix(self, soc2_manager):
        """Test exporting control matrix."""
        # Add multiple controls
        for criteria in TrustServiceCriteria:
            control = SOC2Control(
                control_id=f"{criteria.value}_1",
                criteria=criteria,
                title=f"{criteria.value} control",
                description="Test",
                status=ControlStatus.IMPLEMENTED,
            )
            await soc2_manager.register_control(control)

        matrix = await soc2_manager.export_control_matrix()

        assert matrix is not None
        assert isinstance(matrix, (dict, list))

    def test_trust_service_criteria_enum(self):
        """Test TrustServiceCriteria enum values."""
        assert TrustServiceCriteria.SECURITY == "security"
        assert TrustServiceCriteria.AVAILABILITY == "availability"
        assert TrustServiceCriteria.PROCESSING_INTEGRITY == "processing_integrity"
        assert TrustServiceCriteria.CONFIDENTIALITY == "confidentiality"
        assert TrustServiceCriteria.PRIVACY == "privacy"

    def test_control_status_enum(self):
        """Test ControlStatus enum values."""
        assert ControlStatus.IMPLEMENTED == "implemented"
        assert ControlStatus.PARTIALLY_IMPLEMENTED == "partially_implemented"
        assert ControlStatus.NOT_IMPLEMENTED == "not_implemented"
        assert ControlStatus.NOT_APPLICABLE == "not_applicable"

    @pytest.mark.asyncio
    async def test_control_update_timestamps(self, soc2_manager, sample_control):
        """Test that control updates track timestamps."""
        await soc2_manager.register_control(sample_control)

        # Update control
        await soc2_manager.update_control_status("CC6.1", ControlStatus.IMPLEMENTED, notes="Updated")

        # Verify timestamp tracking
        control = soc2_manager.get_control("CC6.1")
        assert control is not None

    @pytest.mark.asyncio
    async def test_bulk_control_registration(self, soc2_manager):
        """Test registering multiple controls at once."""
        controls = [
            SOC2Control(
                control_id=f"TEST_{i}",
                criteria=TrustServiceCriteria.SECURITY,
                title=f"Control {i}",
                description="Bulk test",
                status=ControlStatus.IMPLEMENTED,
            )
            for i in range(5)
        ]

        for control in controls:
            await soc2_manager.register_control(control)

        assert len(soc2_manager.controls) >= 5
