"""
Tests for ai_engine/api/compliance_reporter_endpoints.py
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi import HTTPException
from datetime import datetime, timedelta


class TestComplianceReportEndpoints:
    """Test suite for compliance reporter API endpoints."""

    @pytest.fixture
    def mock_reporter(self):
        """Create mock compliance reporter."""
        reporter = AsyncMock()
        reporter.generate_compliance_report = AsyncMock()
        reporter.continuous_compliance_monitoring = AsyncMock()
        reporter.generate_audit_evidence = AsyncMock()
        reporter.get_service_status = Mock(
            return_value={
                "service_running": True,
                "monitoring_active": True,
                "active_monitors": [],
                "alert_queue_size": 0,
                "performance_metrics": {},
                "llm_service_status": "healthy",
            }
        )
        reporter.get_performance_metrics = Mock(
            return_value={
                "report_generation": {"target_met": True},
                "compliance_accuracy": {"target_met": True},
                "audit_pass_rate": {"target_met": True},
                "monitoring": {},
                "service_status": {},
            }
        )
        reporter.regulatory_kb = Mock()
        reporter.regulatory_kb.get_available_frameworks = Mock(return_value=["GDPR", "SOC2"])
        reporter.regulatory_kb.get_framework_metadata = Mock(
            return_value={
                "full_name": "General Data Protection Regulation",
                "applicable_to": ["EU"],
                "last_updated": "2024-01-01",
                "next_review": "2025-01-01",
            }
        )
        reporter.alert_queue = Mock()
        reporter.alert_queue.qsize = Mock(return_value=0)
        reporter.alert_queue.get_nowait = Mock(side_effect=Exception("Empty"))
        return reporter

    @pytest.mark.asyncio
    async def test_generate_compliance_report_success(self, mock_reporter):
        """Test successful compliance report generation."""
        from ai_engine.api.compliance_reporter_endpoints import (
            generate_compliance_report,
        )
        from ai_engine.api.compliance_reporter_endpoints import ComplianceReportRequest
        from ai_engine.compliance.report_generator import ReportType, ReportFormat

        # Mock report result
        mock_report = Mock()
        mock_report.report_id = "report123"
        mock_report.framework = "GDPR"
        mock_report.report_type = ReportType.DETAILED_TECHNICAL
        mock_report.format = ReportFormat.PDF
        mock_report.generated_date = datetime.utcnow()
        mock_report.file_name = "gdpr_report.pdf"
        mock_report.file_size = 1024
        mock_report.metadata = {"compliance_score": 0.95, "risk_score": 0.1}

        mock_reporter.generate_compliance_report.return_value = mock_report

        request = ComplianceReportRequest(
            protocol="HTTP",
            standard="GDPR",
            evidence={"test": "data"},
            report_type="detailed_technical",
            format="pdf",
        )

        with patch(
            "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
            return_value=mock_reporter,
        ):
            with patch("ai_engine.api.compliance_reporter_endpoints.metrics"):
                result = await generate_compliance_report(request, Mock())

        assert result.report_id == "report123"
        assert result.framework == "GDPR"
        assert result.compliance_score == 0.95

    @pytest.mark.asyncio
    async def test_generate_compliance_report_failure(self, mock_reporter):
        """Test compliance report generation failure."""
        from ai_engine.api.compliance_reporter_endpoints import (
            generate_compliance_report,
        )
        from ai_engine.api.compliance_reporter_endpoints import ComplianceReportRequest

        mock_reporter.generate_compliance_report.side_effect = Exception("Generation failed")

        request = ComplianceReportRequest(protocol="HTTP", standard="GDPR")

        with patch(
            "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
            return_value=mock_reporter,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await generate_compliance_report(request, Mock())

            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_download_compliance_report(self):
        """Test compliance report download endpoint."""
        from ai_engine.api.compliance_reporter_endpoints import (
            download_compliance_report,
        )

        with pytest.raises(HTTPException) as exc_info:
            await download_compliance_report("report123")

        assert exc_info.value.status_code == 501

    @pytest.mark.asyncio
    async def test_start_continuous_monitoring_success(self, mock_reporter):
        """Test starting continuous monitoring."""
        from ai_engine.api.compliance_reporter_endpoints import (
            start_continuous_monitoring,
        )
        from ai_engine.api.compliance_reporter_endpoints import MonitoringConfigRequest

        async def mock_monitoring_generator(*args, **kwargs):
            yield Mock(alert_id="alert1")

        mock_reporter.continuous_compliance_monitoring = mock_monitoring_generator
        mock_reporter.monitoring_config = None

        request = MonitoringConfigRequest(
            enabled=True,
            frequency="hourly",
            frameworks=["GDPR", "SOC2"],
            protocols=["HTTP", "HTTPS"],
            auto_remediation=False,
        )

        with patch(
            "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
            return_value=mock_reporter,
        ):
            result = await start_continuous_monitoring(request, Mock())

        assert result["status"] == "monitoring_started"
        assert "GDPR" in result["frameworks"]

    @pytest.mark.asyncio
    async def test_start_continuous_monitoring_failure(self, mock_reporter):
        """Test continuous monitoring startup failure."""
        from ai_engine.api.compliance_reporter_endpoints import (
            start_continuous_monitoring,
        )
        from ai_engine.api.compliance_reporter_endpoints import MonitoringConfigRequest

        mock_reporter.continuous_compliance_monitoring.side_effect = Exception("Monitoring failed")

        request = MonitoringConfigRequest(enabled=True, frequency="hourly", frameworks=["GDPR"])

        with patch(
            "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
            return_value=mock_reporter,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await start_continuous_monitoring(request, Mock())

            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_get_monitoring_status(self, mock_reporter):
        """Test getting monitoring status."""
        from ai_engine.api.compliance_reporter_endpoints import get_monitoring_status

        with patch(
            "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
            return_value=mock_reporter,
        ):
            result = await get_monitoring_status()

        assert result["monitoring_active"] is True
        assert "performance_metrics" in result

    @pytest.mark.asyncio
    async def test_get_monitoring_status_failure(self, mock_reporter):
        """Test monitoring status retrieval failure."""
        from ai_engine.api.compliance_reporter_endpoints import get_monitoring_status

        mock_reporter.get_service_status.side_effect = Exception("Status failed")

        with patch(
            "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
            return_value=mock_reporter,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_monitoring_status()

            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_get_compliance_alerts(self, mock_reporter):
        """Test getting compliance alerts."""
        from ai_engine.api.compliance_reporter_endpoints import get_compliance_alerts
        from ai_engine.compliance.compliance_reporter import (
            ComplianceAlert,
            AlertSeverity,
        )

        # Mock alerts
        alert = ComplianceAlert(
            alert_id="alert1",
            timestamp=datetime.utcnow(),
            severity=AlertSeverity.HIGH,
            standard="GDPR",
            requirement_id="REQ-001",
            requirement_title="Data Protection",
            violation_description="Violation detected",
            recommended_actions=["Action 1"],
            auto_remediation_available=True,
        )

        mock_reporter.alert_queue.qsize.return_value = 1
        mock_reporter.alert_queue.get_nowait.return_value = alert

        with patch(
            "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
            return_value=mock_reporter,
        ):
            result = await get_compliance_alerts(limit=100)

        assert len(result) == 1
        assert result[0].alert_id == "alert1"

    @pytest.mark.asyncio
    async def test_get_compliance_alerts_with_filters(self, mock_reporter):
        """Test getting compliance alerts with filters."""
        from ai_engine.api.compliance_reporter_endpoints import get_compliance_alerts
        from ai_engine.compliance.compliance_reporter import (
            ComplianceAlert,
            AlertSeverity,
        )

        alert = ComplianceAlert(
            alert_id="alert1",
            timestamp=datetime.utcnow(),
            severity=AlertSeverity.HIGH,
            standard="GDPR",
            requirement_id="REQ-001",
            requirement_title="Data Protection",
            violation_description="Violation",
            recommended_actions=[],
            auto_remediation_available=False,
        )

        mock_reporter.alert_queue.qsize.return_value = 1
        mock_reporter.alert_queue.get_nowait.return_value = alert

        with patch(
            "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
            return_value=mock_reporter,
        ):
            result = await get_compliance_alerts(limit=100, severity="high", framework="GDPR")

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_compliance_alerts_failure(self, mock_reporter):
        """Test compliance alerts retrieval failure."""
        from ai_engine.api.compliance_reporter_endpoints import get_compliance_alerts

        mock_reporter.alert_queue.qsize.side_effect = Exception("Queue error")

        with patch(
            "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
            return_value=mock_reporter,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_compliance_alerts()

            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_generate_audit_evidence_success(self, mock_reporter):
        """Test successful audit evidence generation."""
        from ai_engine.api.compliance_reporter_endpoints import generate_audit_evidence
        from ai_engine.api.compliance_reporter_endpoints import AuditRequestModel
        from ai_engine.compliance.compliance_reporter import AuditEvidence

        mock_evidence = AuditEvidence(
            evidence_id="evidence123",
            request_id="request123",
            framework="GDPR",
            generated_date=datetime.utcnow(),
            evidence_items=[],
            verification_status="verified",
            digital_signature="signature123",
        )

        mock_reporter.generate_audit_evidence.return_value = mock_evidence

        request = AuditRequestModel(
            auditor="John Doe",
            framework="GDPR",
            requirements=["REQ-001"],
            format="comprehensive",
        )

        with patch(
            "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
            return_value=mock_reporter,
        ):
            with patch("ai_engine.api.compliance_reporter_endpoints.metrics"):
                result = await generate_audit_evidence(request)

        assert result.evidence_id == "evidence123"
        assert result.framework == "GDPR"

    @pytest.mark.asyncio
    async def test_generate_audit_evidence_with_dates(self, mock_reporter):
        """Test audit evidence generation with date range."""
        from ai_engine.api.compliance_reporter_endpoints import generate_audit_evidence
        from ai_engine.api.compliance_reporter_endpoints import AuditRequestModel
        from ai_engine.compliance.compliance_reporter import AuditEvidence

        mock_evidence = AuditEvidence(
            evidence_id="evidence123",
            request_id="request123",
            framework="SOC2",
            generated_date=datetime.utcnow(),
            evidence_items=[],
            verification_status="verified",
            digital_signature="signature123",
        )

        mock_reporter.generate_audit_evidence.return_value = mock_evidence

        start_date = (datetime.utcnow() - timedelta(days=30)).isoformat()
        end_date = datetime.utcnow().isoformat()

        request = AuditRequestModel(
            auditor="Jane Smith",
            framework="SOC2",
            start_date=start_date,
            end_date=end_date,
        )

        with patch(
            "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
            return_value=mock_reporter,
        ):
            with patch("ai_engine.api.compliance_reporter_endpoints.metrics"):
                result = await generate_audit_evidence(request)

        assert result.evidence_id == "evidence123"

    @pytest.mark.asyncio
    async def test_generate_audit_evidence_failure(self, mock_reporter):
        """Test audit evidence generation failure."""
        from ai_engine.api.compliance_reporter_endpoints import generate_audit_evidence
        from ai_engine.api.compliance_reporter_endpoints import AuditRequestModel

        mock_reporter.generate_audit_evidence.side_effect = Exception("Evidence generation failed")

        request = AuditRequestModel(auditor="John Doe", framework="GDPR")

        with patch(
            "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
            return_value=mock_reporter,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await generate_audit_evidence(request)

            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_download_audit_evidence(self):
        """Test audit evidence download endpoint."""
        from ai_engine.api.compliance_reporter_endpoints import download_audit_evidence

        with pytest.raises(HTTPException) as exc_info:
            await download_audit_evidence("evidence123")

        assert exc_info.value.status_code == 501

    @pytest.mark.asyncio
    async def test_list_compliance_frameworks(self, mock_reporter):
        """Test listing compliance frameworks."""
        from ai_engine.api.compliance_reporter_endpoints import (
            list_compliance_frameworks,
        )

        with patch(
            "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
            return_value=mock_reporter,
        ):
            result = await list_compliance_frameworks()

        assert "frameworks" in result
        assert result["total_count"] == 2
        assert len(result["frameworks"]) == 2

    @pytest.mark.asyncio
    async def test_list_compliance_frameworks_failure(self, mock_reporter):
        """Test listing frameworks failure."""
        from ai_engine.api.compliance_reporter_endpoints import (
            list_compliance_frameworks,
        )

        mock_reporter.regulatory_kb.get_available_frameworks.side_effect = Exception("KB error")

        with patch(
            "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
            return_value=mock_reporter,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await list_compliance_frameworks()

            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_get_performance_metrics(self, mock_reporter):
        """Test getting performance metrics."""
        from ai_engine.api.compliance_reporter_endpoints import get_performance_metrics

        with patch(
            "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
            return_value=mock_reporter,
        ):
            result = await get_performance_metrics()

        assert "report_generation" in result
        assert "compliance_accuracy" in result
        assert "audit_pass_rate" in result
        assert "success_criteria" in result

    @pytest.mark.asyncio
    async def test_get_performance_metrics_failure(self, mock_reporter):
        """Test performance metrics retrieval failure."""
        from ai_engine.api.compliance_reporter_endpoints import get_performance_metrics

        mock_reporter.get_performance_metrics.side_effect = Exception("Metrics error")

        with patch(
            "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
            return_value=mock_reporter,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_performance_metrics()

            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, mock_reporter):
        """Test health check endpoint when healthy."""
        from ai_engine.api.compliance_reporter_endpoints import health_check

        with patch(
            "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
            return_value=mock_reporter,
        ):
            result = await health_check()

        assert result["status"] == "healthy"
        assert result["service_running"] is True

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, mock_reporter):
        """Test health check endpoint when unhealthy."""
        from ai_engine.api.compliance_reporter_endpoints import health_check

        mock_reporter.get_service_status.return_value = {
            "service_running": False,
            "monitoring_active": False,
            "llm_service_status": "unhealthy",
        }

        with patch(
            "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
            return_value=mock_reporter,
        ):
            result = await health_check()

        assert result["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_health_check_exception(self, mock_reporter):
        """Test health check with exception."""
        from ai_engine.api.compliance_reporter_endpoints import health_check

        mock_reporter.get_service_status.side_effect = Exception("Service error")

        with patch(
            "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
            return_value=mock_reporter,
        ):
            result = await health_check()

        assert result["status"] == "unhealthy"
        assert "error" in result

    def test_include_compliance_reporter_routes(self):
        """Test including compliance reporter routes."""
        from ai_engine.api.compliance_reporter_endpoints import (
            include_compliance_reporter_routes,
        )

        mock_app = Mock()
        include_compliance_reporter_routes(mock_app)

        mock_app.include_router.assert_called_once()
