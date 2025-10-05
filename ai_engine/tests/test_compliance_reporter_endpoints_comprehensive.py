"""
Comprehensive tests for ai_engine/api/compliance_reporter_endpoints.py

Tests cover:
- Compliance report generation
- Report download endpoints
- Continuous monitoring start/stop
- Monitoring status retrieval
- Compliance alerts management
- Audit evidence generation
- Framework listing
- Performance metrics
- Health checks
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, MagicMock, patch, call
from fastapi import HTTPException, BackgroundTasks
from fastapi.testclient import TestClient

from ai_engine.api.compliance_reporter_endpoints import (
    router,
    ComplianceReportRequest,
    ComplianceReportResponse,
    MonitoringConfigRequest,
    AuditRequestModel,
    AuditEvidenceResponse,
    ComplianceAlertResponse,
    generate_compliance_report,
    download_compliance_report,
    start_continuous_monitoring,
    get_monitoring_status,
    get_compliance_alerts,
    generate_audit_evidence,
    download_audit_evidence,
    list_compliance_frameworks,
    get_performance_metrics,
    health_check,
)
from ai_engine.compliance.compliance_reporter import (
    ComplianceReport,
    ComplianceAlert,
    AuditEvidence,
    AlertSeverity,
    MonitoringFrequency,
)
from ai_engine.compliance.report_generator import ReportFormat, ReportType


# Fixtures

@pytest.fixture
def mock_compliance_reporter():
    """Create mock compliance reporter."""
    reporter = AsyncMock()
    reporter.monitoring_config = Mock()
    reporter.alert_queue = asyncio.Queue()
    reporter.regulatory_kb = Mock()
    return reporter


@pytest.fixture
def mock_background_tasks():
    """Create mock background tasks."""
    return Mock(spec=BackgroundTasks)


@pytest.fixture
def sample_compliance_report():
    """Create sample compliance report."""
    return Mock(
        report_id="report-123",
        framework="SOC2",
        report_type=ReportType.DETAILED_TECHNICAL,
        format=ReportFormat.PDF,
        generated_date=datetime.utcnow(),
        file_name="compliance_report.pdf",
        file_size=1024000,
        metadata={
            "compliance_score": 0.95,
            "risk_score": 0.15,
            "total_controls": 100,
            "compliant_controls": 95,
        },
    )


@pytest.fixture
def sample_audit_evidence():
    """Create sample audit evidence."""
    return Mock(
        evidence_id="evidence-456",
        request_id="req-789",
        framework="GDPR",
        generated_date=datetime.utcnow(),
        evidence_items=[{"type": "log", "data": "sample"}] * 10,
        verification_status="verified",
        digital_signature="signature-abc123",
    )


@pytest.fixture
def sample_compliance_alert():
    """Create sample compliance alert."""
    return ComplianceAlert(
        alert_id="alert-001",
        timestamp=datetime.utcnow(),
        severity=AlertSeverity.CRITICAL,
        standard="PCI-DSS",
        requirement_id="REQ-3.4",
        requirement_title="Encryption of cardholder data",
        violation_description="Unencrypted data transmission detected",
        recommended_actions=[
            "Enable TLS 1.3",
            "Review encryption policies",
            "Update security configurations",
        ],
        auto_remediation_available=True,
        protocol="payment_gateway",
    )


# Report Generation Tests

@pytest.mark.asyncio
async def test_generate_compliance_report_success(
    mock_compliance_reporter, mock_background_tasks, sample_compliance_report
):
    """Test successful compliance report generation."""
    request = ComplianceReportRequest(
        protocol="test_protocol",
        standard="SOC2",
        evidence={"key": "value"},
        report_type="detailed_technical",
        format="pdf",
    )

    mock_compliance_reporter.generate_compliance_report.return_value = (
        sample_compliance_report
    )

    with patch(
        "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
        return_value=mock_compliance_reporter,
    ):
        with patch(
            "ai_engine.api.compliance_reporter_endpoints.get_enterprise_metrics"
        ):
            response = await generate_compliance_report(request, mock_background_tasks)

            assert isinstance(response, ComplianceReportResponse)
            assert response.report_id == "report-123"
            assert response.framework == "SOC2"
            assert response.compliance_score == 0.95
            assert response.risk_score == 0.15
            assert "report-123" in response.download_url


@pytest.mark.asyncio
async def test_generate_compliance_report_failure(
    mock_compliance_reporter, mock_background_tasks
):
    """Test compliance report generation failure."""
    request = ComplianceReportRequest(
        protocol="test_protocol",
        standard="SOC2",
    )

    mock_compliance_reporter.generate_compliance_report.side_effect = Exception(
        "Generation failed"
    )

    with patch(
        "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
        return_value=mock_compliance_reporter,
    ):
        with patch(
            "ai_engine.api.compliance_reporter_endpoints.get_enterprise_metrics"
        ):
            with pytest.raises(HTTPException) as exc_info:
                await generate_compliance_report(request, mock_background_tasks)

            assert exc_info.value.status_code == 500
            assert "Report generation failed" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_generate_compliance_report_all_formats():
    """Test report generation for all supported formats."""
    formats = ["pdf", "json", "html", "excel"]

    for fmt in formats:
        request = ComplianceReportRequest(
            protocol="test", standard="SOC2", format=fmt
        )

        mock_reporter = AsyncMock()
        mock_report = Mock(
            report_id=f"report-{fmt}",
            framework="SOC2",
            report_type=ReportType.DETAILED_TECHNICAL,
            format=ReportFormat[fmt.upper()],
            generated_date=datetime.utcnow(),
            file_name=f"report.{fmt}",
            file_size=1024,
            metadata={"compliance_score": 0.9, "risk_score": 0.1},
        )
        mock_reporter.generate_compliance_report.return_value = mock_report

        with patch(
            "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
            return_value=mock_reporter,
        ):
            with patch(
                "ai_engine.api.compliance_reporter_endpoints.get_enterprise_metrics"
            ):
                response = await generate_compliance_report(
                    request, Mock(spec=BackgroundTasks)
                )

                assert response.format == fmt


# Report Download Tests

@pytest.mark.asyncio
async def test_download_compliance_report_not_implemented():
    """Test report download endpoint (not yet implemented)."""
    with pytest.raises(HTTPException) as exc_info:
        await download_compliance_report("report-123")

    # The endpoint catches HTTPException and re-raises as 500
    assert exc_info.value.status_code == 500
    assert "storage backend" in str(exc_info.value.detail)


# Continuous Monitoring Tests

@pytest.mark.asyncio
async def test_start_continuous_monitoring_success(
    mock_compliance_reporter, mock_background_tasks
):
    """Test starting continuous monitoring."""
    config = MonitoringConfigRequest(
        enabled=True,
        frequency="hourly",
        frameworks=["SOC2", "GDPR"],
        protocols=["web_app", "api"],
        auto_remediation=True,
        alert_thresholds={"critical": 0.9, "high": 0.7},
    )

    async def mock_monitoring_generator(*args, **kwargs):
        """Mock async generator for monitoring."""
        yield ComplianceAlert(
            alert_id="test-alert",
            timestamp=datetime.utcnow(),
            severity=AlertSeverity.MEDIUM,
            standard="SOC2",
            requirement_id="REQ-1",
            requirement_title="Test",
            violation_description="Test violation",
            recommended_actions=["Fix it"],
            auto_remediation_available=False,
            protocol="test",
        )

    mock_compliance_reporter.continuous_compliance_monitoring = (
        mock_monitoring_generator
    )

    with patch(
        "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
        return_value=mock_compliance_reporter,
    ):
        response = await start_continuous_monitoring(config, mock_background_tasks)

        assert response["status"] == "monitoring_started"
        assert response["frameworks"] == ["SOC2", "GDPR"]
        assert response["frequency"] == "hourly"
        assert response["auto_remediation"] is True
        mock_background_tasks.add_task.assert_called_once()


@pytest.mark.asyncio
async def test_start_continuous_monitoring_failure(
    mock_compliance_reporter, mock_background_tasks
):
    """Test monitoring start failure."""
    config = MonitoringConfigRequest(
        enabled=True, frequency="hourly", frameworks=["SOC2"]
    )

    with patch(
        "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
        side_effect=Exception("Monitoring failed"),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await start_continuous_monitoring(config, mock_background_tasks)

        assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_start_monitoring_all_frequencies():
    """Test monitoring with different frequencies."""
    frequencies = ["hourly", "daily", "real_time"]  # Valid frequencies

    for freq in frequencies:
        config = MonitoringConfigRequest(frequency=freq, frameworks=["SOC2"])

        mock_reporter = AsyncMock()

        async def mock_gen(*args, **kwargs):
            if False:
                yield None

        mock_reporter.continuous_compliance_monitoring = mock_gen

        with patch(
            "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
            return_value=mock_reporter,
        ):
            response = await start_continuous_monitoring(
                config, Mock(spec=BackgroundTasks)
            )

            assert response["frequency"] == freq


# Monitoring Status Tests

@pytest.mark.asyncio
async def test_get_monitoring_status_success():
    """Test getting monitoring status."""
    mock_reporter = AsyncMock()
    mock_reporter.get_service_status.return_value = {
        "monitoring_active": True,
        "active_monitors": ["SOC2", "GDPR"],
        "alert_queue_size": 5,
        "performance_metrics": {"uptime": 99.9},
    }

    async def mock_get_reporter():
        return mock_reporter

    with patch(
        "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
        side_effect=mock_get_reporter,
    ):
        response = await get_monitoring_status()

        assert response["monitoring_active"] is True
        assert response["frameworks_monitored"] == 2
        assert response["alert_queue_size"] == 5


@pytest.mark.asyncio
async def test_get_monitoring_status_failure(mock_compliance_reporter):
    """Test monitoring status retrieval failure."""
    mock_compliance_reporter.get_service_status.side_effect = Exception("Status error")

    with patch(
        "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
        return_value=mock_compliance_reporter,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await get_monitoring_status()

        assert exc_info.value.status_code == 500


# Compliance Alerts Tests

@pytest.mark.asyncio
async def test_get_compliance_alerts_success(
    mock_compliance_reporter, sample_compliance_alert
):
    """Test retrieving compliance alerts."""
    # Add alerts to queue
    await mock_compliance_reporter.alert_queue.put(sample_compliance_alert)
    await mock_compliance_reporter.alert_queue.put(sample_compliance_alert)

    with patch(
        "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
        return_value=mock_compliance_reporter,
    ):
        alerts = await get_compliance_alerts(limit=10)

        assert isinstance(alerts, list)
        assert len(alerts) == 2
        assert all(isinstance(a, ComplianceAlertResponse) for a in alerts)


@pytest.mark.asyncio
async def test_get_compliance_alerts_with_filters(mock_compliance_reporter):
    """Test retrieving alerts with severity and framework filters."""
    alert1 = ComplianceAlert(
        alert_id="alert-1",
        timestamp=datetime.utcnow(),
        severity=AlertSeverity.CRITICAL,
        standard="SOC2",
        requirement_id="REQ-1",
        requirement_title="Test 1",
        violation_description="Violation 1",
        recommended_actions=["Action 1"],
        auto_remediation_available=False,
        protocol="test",
    )

    alert2 = ComplianceAlert(
        alert_id="alert-2",
        timestamp=datetime.utcnow(),
        severity=AlertSeverity.WARNING,
        standard="GDPR",
        requirement_id="REQ-2",
        requirement_title="Test 2",
        violation_description="Violation 2",
        recommended_actions=["Action 2"],
        auto_remediation_available=False,
        protocol="test",
    )

    await mock_compliance_reporter.alert_queue.put(alert1)
    await mock_compliance_reporter.alert_queue.put(alert2)

    with patch(
        "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
        return_value=mock_compliance_reporter,
    ):
        # Filter by severity
        alerts = await get_compliance_alerts(limit=10, severity="critical")
        assert len(alerts) == 1
        assert alerts[0].severity == "critical"

        # Reset queue
        mock_compliance_reporter.alert_queue = asyncio.Queue()
        await mock_compliance_reporter.alert_queue.put(alert1)
        await mock_compliance_reporter.alert_queue.put(alert2)

        # Filter by framework
        alerts = await get_compliance_alerts(limit=10, framework="GDPR")
        assert len(alerts) == 1
        assert alerts[0].standard == "GDPR"


@pytest.mark.asyncio
async def test_get_compliance_alerts_empty_queue(mock_compliance_reporter):
    """Test retrieving alerts from empty queue."""
    with patch(
        "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
        return_value=mock_compliance_reporter,
    ):
        alerts = await get_compliance_alerts(limit=10)

        assert isinstance(alerts, list)
        assert len(alerts) == 0


@pytest.mark.asyncio
async def test_get_compliance_alerts_limit(mock_compliance_reporter):
    """Test alert limit parameter."""
    # Add more alerts than limit
    for i in range(50):
        alert = ComplianceAlert(
            alert_id=f"alert-{i}",
            timestamp=datetime.utcnow(),
            severity=AlertSeverity.INFO,
            standard="SOC2",
            requirement_id=f"REQ-{i}",
            requirement_title=f"Test {i}",
            violation_description=f"Violation {i}",
            recommended_actions=[f"Action {i}"],
            auto_remediation_available=False,
            protocol="test",
        )
        await mock_compliance_reporter.alert_queue.put(alert)

    with patch(
        "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
        return_value=mock_compliance_reporter,
    ):
        alerts = await get_compliance_alerts(limit=20)

        assert len(alerts) <= 20


# Audit Evidence Tests

@pytest.mark.asyncio
async def test_generate_audit_evidence_success(
    mock_compliance_reporter, sample_audit_evidence
):
    """Test audit evidence generation."""
    request = AuditRequestModel(
        auditor="external_auditor",
        framework="GDPR",
        requirements=["Art. 25", "Art. 32"],
        start_date="2024-01-01T00:00:00",
        end_date="2024-12-31T23:59:59",
        evidence_types=["logs", "configurations"],
        format="comprehensive",
    )

    mock_compliance_reporter.generate_audit_evidence.return_value = (
        sample_audit_evidence
    )

    with patch(
        "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
        return_value=mock_compliance_reporter,
    ):
        with patch(
            "ai_engine.api.compliance_reporter_endpoints.get_enterprise_metrics"
        ):
            response = await generate_audit_evidence(request)

            assert isinstance(response, AuditEvidenceResponse)
            assert response.evidence_id == "evidence-456"
            assert response.framework == "GDPR"
            assert response.evidence_count == 10
            assert response.verification_status == "verified"
            assert "evidence-456" in response.download_url


@pytest.mark.asyncio
async def test_generate_audit_evidence_default_dates(mock_compliance_reporter):
    """Test audit evidence with default date range."""
    request = AuditRequestModel(
        auditor="auditor", framework="SOC2", start_date=None, end_date=None
    )

    mock_evidence = Mock(
        evidence_id="evidence-123",
        request_id="req-456",
        framework="SOC2",
        generated_date=datetime.utcnow(),
        evidence_items=[],
        verification_status="pending",
        digital_signature="sig-123",
    )
    mock_compliance_reporter.generate_audit_evidence.return_value = mock_evidence

    with patch(
        "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
        return_value=mock_compliance_reporter,
    ):
        with patch(
            "ai_engine.api.compliance_reporter_endpoints.get_enterprise_metrics"
        ):
            response = await generate_audit_evidence(request)

            # Verify call was made with default date range (30 days)
            call_args = mock_compliance_reporter.generate_audit_evidence.call_args
            audit_request = call_args[0][0]
            assert (audit_request.end_date - audit_request.start_date).days >= 29


@pytest.mark.asyncio
async def test_generate_audit_evidence_failure(mock_compliance_reporter):
    """Test audit evidence generation failure."""
    request = AuditRequestModel(auditor="auditor", framework="SOC2")

    mock_compliance_reporter.generate_audit_evidence.side_effect = Exception(
        "Evidence generation failed"
    )

    with patch(
        "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
        return_value=mock_compliance_reporter,
    ):
        with patch(
            "ai_engine.api.compliance_reporter_endpoints.get_enterprise_metrics"
        ):
            with pytest.raises(HTTPException) as exc_info:
                await generate_audit_evidence(request)

            assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_download_audit_evidence_not_implemented():
    """Test audit evidence download endpoint (not yet implemented)."""
    with pytest.raises(HTTPException) as exc_info:
        await download_audit_evidence("evidence-123")

    # The endpoint catches HTTPException and re-raises as 500
    assert exc_info.value.status_code == 500
    assert "storage backend" in str(exc_info.value.detail)


# Framework Listing Tests

@pytest.mark.asyncio
async def test_list_compliance_frameworks_success(mock_compliance_reporter):
    """Test listing available compliance frameworks."""
    mock_compliance_reporter.regulatory_kb.get_available_frameworks.return_value = [
        "SOC2",
        "GDPR",
        "HIPAA",
        "PCI-DSS",
    ]

    def get_metadata(framework):
        return {
            "full_name": f"{framework} Full Name",
            "applicable_to": ["organizations"],
            "last_updated": "2024-01-01",
            "next_review": "2025-01-01",
        }

    mock_compliance_reporter.regulatory_kb.get_framework_metadata = get_metadata

    with patch(
        "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
        return_value=mock_compliance_reporter,
    ):
        response = await list_compliance_frameworks()

        assert "frameworks" in response
        assert "total_count" in response
        assert response["total_count"] == 4
        assert len(response["frameworks"]) == 4

        for framework in response["frameworks"]:
            assert "id" in framework
            assert "name" in framework
            assert "applicable_to" in framework


@pytest.mark.asyncio
async def test_list_compliance_frameworks_failure(mock_compliance_reporter):
    """Test framework listing failure."""
    mock_compliance_reporter.regulatory_kb.get_available_frameworks.side_effect = (
        Exception("KB error")
    )

    with patch(
        "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
        return_value=mock_compliance_reporter,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await list_compliance_frameworks()

        assert exc_info.value.status_code == 500


# Performance Metrics Tests

@pytest.mark.asyncio
async def test_get_performance_metrics_success():
    """Test getting performance metrics."""
    mock_reporter = AsyncMock()
    mock_reporter.get_performance_metrics.return_value = {
        "report_generation": {
            "avg_time": 300,
            "target": 600,
            "target_met": True,
        },
        "compliance_accuracy": {
            "score": 0.97,
            "target": 0.95,
            "target_met": True,
        },
        "audit_pass_rate": {
            "rate": 0.99,
            "target": 0.98,
            "target_met": True,
        },
        "monitoring": {"active": True},
        "service_status": {"running": True},
    }

    async def mock_get_reporter():
        return mock_reporter

    with patch(
        "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
        side_effect=mock_get_reporter,
    ):
        response = await get_performance_metrics()

        assert "report_generation" in response
        assert "compliance_accuracy" in response
        assert "audit_pass_rate" in response
        assert "success_criteria" in response

        # Verify success criteria
        criteria = response["success_criteria"]
        assert criteria["report_generation_met"] is True
        assert criteria["compliance_accuracy_met"] is True
        assert criteria["audit_pass_rate_met"] is True


@pytest.mark.asyncio
async def test_get_performance_metrics_failure(mock_compliance_reporter):
    """Test performance metrics retrieval failure."""
    mock_compliance_reporter.get_performance_metrics.side_effect = Exception(
        "Metrics error"
    )

    with patch(
        "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
        return_value=mock_compliance_reporter,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await get_performance_metrics()

        assert exc_info.value.status_code == 500


# Health Check Tests

@pytest.mark.asyncio
async def test_health_check_healthy():
    """Test health check when service is healthy."""
    mock_reporter = AsyncMock()
    mock_reporter.get_service_status.return_value = {
        "service_running": True,
        "monitoring_active": True,
        "llm_service_status": "connected",
    }

    async def mock_get_reporter():
        return mock_reporter

    with patch(
        "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
        side_effect=mock_get_reporter,
    ):
        response = await health_check()

        assert response["status"] == "healthy"
        assert response["service_running"] is True
        assert response["monitoring_active"] is True
        assert response["llm_service"] == "connected"
        assert "timestamp" in response


@pytest.mark.asyncio
async def test_health_check_unhealthy():
    """Test health check when service is unhealthy."""
    mock_reporter = AsyncMock()
    mock_reporter.get_service_status.return_value = {
        "service_running": False,
        "monitoring_active": False,
        "llm_service_status": "disconnected",
    }

    async def mock_get_reporter():
        return mock_reporter

    with patch(
        "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
        side_effect=mock_get_reporter,
    ):
        response = await health_check()

        assert response["status"] == "unhealthy"
        assert response["service_running"] is False


@pytest.mark.asyncio
async def test_health_check_exception():
    """Test health check with exception."""
    with patch(
        "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
        side_effect=Exception("Service unavailable"),
    ):
        response = await health_check()

        assert response["status"] == "unhealthy"
        assert "error" in response
        assert "Service unavailable" in response["error"]


# Integration Tests

@pytest.mark.asyncio
async def test_full_compliance_workflow():
    """Test complete compliance workflow."""
    mock_reporter = AsyncMock()

    # 1. Check health
    mock_reporter.get_service_status.return_value = {
        "service_running": True,
        "monitoring_active": False,
        "llm_service_status": "connected",
    }

    async def mock_get_reporter():
        return mock_reporter

    with patch(
        "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
        side_effect=mock_get_reporter,
    ):
        health = await health_check()
        assert health["status"] == "healthy"

        # 2. List frameworks
        mock_reporter.regulatory_kb.get_available_frameworks.return_value = [
            "SOC2"
        ]
        mock_reporter.regulatory_kb.get_framework_metadata.return_value = {
            "full_name": "SOC 2",
            "applicable_to": ["service_orgs"],
            "last_updated": "2024-01-01",
            "next_review": "2025-01-01",
        }

        frameworks = await list_compliance_frameworks()
        assert frameworks["total_count"] > 0

        # 3. Generate report
        mock_report = Mock(
            report_id="report-workflow",
            framework="SOC2",
            report_type=ReportType.DETAILED_TECHNICAL,
            format=ReportFormat.PDF,
            generated_date=datetime.utcnow(),
            file_name="report.pdf",
            file_size=1024,
            metadata={"compliance_score": 0.95, "risk_score": 0.1},
        )
        mock_reporter.generate_compliance_report.return_value = mock_report

        with patch(
            "ai_engine.api.compliance_reporter_endpoints.get_enterprise_metrics"
        ):
            request = ComplianceReportRequest(
                protocol="workflow_test", standard="SOC2"
            )
            report = await generate_compliance_report(
                request, Mock(spec=BackgroundTasks)
            )
            assert report.report_id == "report-workflow"

        # 4. Start monitoring
        async def mock_gen(*args, **kwargs):
            if False:
                yield None

        mock_reporter.continuous_compliance_monitoring = mock_gen

        config = MonitoringConfigRequest(
            enabled=True, frequency="hourly", frameworks=["SOC2"]
        )
        monitoring_result = await start_continuous_monitoring(
            config, Mock(spec=BackgroundTasks)
        )
        assert monitoring_result["status"] == "monitoring_started"

        # 5. Generate audit evidence
        mock_evidence = Mock(
            evidence_id="evidence-workflow",
            request_id="req-workflow",
            framework="SOC2",
            generated_date=datetime.utcnow(),
            evidence_items=[{"data": "test"}],
            verification_status="verified",
            digital_signature="sig-workflow",
        )
        mock_reporter.generate_audit_evidence.return_value = mock_evidence

        audit_request = AuditRequestModel(auditor="auditor", framework="SOC2")
        evidence = await generate_audit_evidence(audit_request)
        assert evidence.evidence_id == "evidence-workflow"


@pytest.mark.asyncio
async def test_concurrent_report_generation(mock_compliance_reporter):
    """Test concurrent report generation."""
    mock_reports = []
    for i in range(5):
        mock_report = Mock(
            report_id=f"report-{i}",
            framework="SOC2",
            report_type=ReportType.DETAILED_TECHNICAL,
            format=ReportFormat.PDF,
            generated_date=datetime.utcnow(),
            file_name=f"report-{i}.pdf",
            file_size=1024,
            metadata={"compliance_score": 0.9, "risk_score": 0.1},
        )
        mock_reports.append(mock_report)

    mock_compliance_reporter.generate_compliance_report.side_effect = mock_reports

    with patch(
        "ai_engine.api.compliance_reporter_endpoints.get_compliance_reporter",
        return_value=mock_compliance_reporter,
    ):
        with patch(
            "ai_engine.api.compliance_reporter_endpoints.get_enterprise_metrics"
        ):
            tasks = []
            for i in range(5):
                request = ComplianceReportRequest(
                    protocol=f"protocol-{i}", standard="SOC2"
                )
                task = generate_compliance_report(request, Mock(spec=BackgroundTasks))
                tasks.append(task)

            results = await asyncio.gather(*tasks)

            assert len(results) == 5
            assert all(isinstance(r, ComplianceReportResponse) for r in results)


# Request/Response Model Tests

def test_compliance_report_request_validation():
    """Test ComplianceReportRequest validation."""
    request = ComplianceReportRequest(
        protocol="test_protocol",
        standard="SOC2",
        evidence={"key": "value"},
        report_type="detailed_technical",
        format="pdf",
    )

    assert request.protocol == "test_protocol"
    assert request.standard == "SOC2"
    assert request.format == "pdf"


def test_monitoring_config_request_defaults():
    """Test MonitoringConfigRequest default values."""
    config = MonitoringConfigRequest(
        enabled=True, frequency="hourly", frameworks=["SOC2"]
    )

    assert config.enabled is True
    assert config.auto_remediation is False
    assert isinstance(config.alert_thresholds, dict)


def test_audit_request_model_validation():
    """Test AuditRequestModel validation."""
    request = AuditRequestModel(
        auditor="test_auditor",
        framework="GDPR",
        requirements=["Art. 25"],
        start_date="2024-01-01T00:00:00",
        end_date="2024-12-31T23:59:59",
    )

    assert request.auditor == "test_auditor"
    assert request.framework == "GDPR"
    assert len(request.requirements) == 1


# Router Integration Tests

def test_router_configuration():
    """Test router is properly configured."""
    assert router.prefix == "/api/v1/compliance"
    assert "compliance-reporter" in router.tags
    assert len(router.routes) > 0  # Has routes defined
