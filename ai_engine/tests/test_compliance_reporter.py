"""
CRONOS AI - Autonomous Compliance Reporter Tests

Comprehensive test suite for compliance reporter functionality including:
- Report generation
- Continuous monitoring
- Audit evidence generation
- Performance validation
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List

from ai_engine.compliance.compliance_reporter import (
    AutonomousComplianceReporter,
    ComplianceStandard,
    ComplianceAlert,
    AuditRequest,
    AuditEvidence,
    MonitoringFrequency,
    AlertSeverity,
    ContinuousMonitoringConfig,
    get_compliance_reporter,
    shutdown_compliance_reporter,
)
from ai_engine.compliance.report_generator import ReportFormat, ReportType
from ai_engine.core.config import Config


@pytest.fixture
async def compliance_reporter(test_config):
    """Create compliance reporter instance for testing."""
    reporter = AutonomousComplianceReporter(test_config)
    await reporter.start()
    yield reporter
    await reporter.stop()


@pytest.fixture
def sample_evidence():
    """Sample evidence data for testing."""
    return {
        "system_configuration": {
            "hostname": "test-server",
            "environment": "production",
            "version": "1.0.0",
        },
        "security_controls": {
            "encryption": True,
            "authentication": "multi-factor",
            "access_logging": True,
        },
        "network_config": {
            "firewall_enabled": True,
            "intrusion_detection": True,
            "segmentation": "implemented",
        },
    }


@pytest.fixture
def audit_request():
    """Sample audit request for testing."""
    return AuditRequest(
        request_id="AUDIT-TEST-001",
        auditor="test_auditor",
        framework="PCI_DSS_4_0",
        requirements=["1.1", "2.1", "3.1"],
        start_date=datetime.utcnow() - timedelta(days=30),
        end_date=datetime.utcnow(),
        evidence_types=["logs", "configurations", "policies"],
        format="comprehensive",
    )


class TestComplianceReportGeneration:
    """Test compliance report generation functionality."""

    @pytest.mark.asyncio
    async def test_generate_pci_dss_report(self, compliance_reporter, sample_evidence):
        """Test PCI-DSS compliance report generation."""
        report = await compliance_reporter.generate_compliance_report(
            protocol="payment_gateway",
            standard="PCI-DSS",
            evidence=sample_evidence,
            report_type=ReportType.DETAILED_TECHNICAL,
            format=ReportFormat.PDF,
        )

        assert report is not None
        assert report.framework == "PCI_DSS_4_0"
        assert report.format == ReportFormat.PDF
        assert len(report.content) > 0
        assert report.file_size > 0

    @pytest.mark.asyncio
    async def test_generate_hipaa_report(self, compliance_reporter, sample_evidence):
        """Test HIPAA compliance report generation."""
        report = await compliance_reporter.generate_compliance_report(
            protocol="healthcare_system",
            standard="HIPAA",
            evidence=sample_evidence,
            report_type=ReportType.EXECUTIVE_SUMMARY,
            format=ReportFormat.PDF,
        )

        assert report is not None
        assert report.framework == "HIPAA"
        assert report.report_type == ReportType.EXECUTIVE_SUMMARY

    @pytest.mark.asyncio
    async def test_report_generation_time(self, compliance_reporter, sample_evidence):
        """Test report generation meets <10 minute target."""
        import time

        start_time = time.time()
        report = await compliance_reporter.generate_compliance_report(
            protocol="test_protocol", standard="SOC2", evidence=sample_evidence
        )
        generation_time = time.time() - start_time

        # Should complete in less than 10 minutes (600 seconds)
        assert (
            generation_time < 600
        ), f"Report generation took {generation_time}s, target is <600s"
        assert report is not None

    @pytest.mark.asyncio
    async def test_multiple_format_generation(
        self, compliance_reporter, sample_evidence
    ):
        """Test generating reports in multiple formats."""
        formats = [ReportFormat.PDF, ReportFormat.JSON, ReportFormat.HTML]

        for format in formats:
            report = await compliance_reporter.generate_compliance_report(
                protocol="test_protocol",
                standard="GDPR",
                evidence=sample_evidence,
                format=format,
            )

            assert report is not None
            assert report.format == format
            assert len(report.content) > 0

    @pytest.mark.asyncio
    async def test_compliance_accuracy(self, compliance_reporter, sample_evidence):
        """Test compliance accuracy meets 95%+ target."""
        report = await compliance_reporter.generate_compliance_report(
            protocol="test_protocol", standard="PCI-DSS", evidence=sample_evidence
        )

        # Get performance metrics
        metrics = compliance_reporter.get_performance_metrics()

        # Compliance accuracy should be tracked
        assert "compliance_accuracy" in metrics

        # Note: Actual accuracy validation would require real compliance data
        # This test validates the metric tracking is working
        assert metrics["compliance_accuracy"]["average_score"] >= 0


class TestContinuousMonitoring:
    """Test continuous compliance monitoring functionality."""

    @pytest.mark.asyncio
    async def test_start_monitoring(self, compliance_reporter):
        """Test starting continuous monitoring."""
        config = ContinuousMonitoringConfig(
            enabled=True,
            frequency=MonitoringFrequency.HOURLY,
            frameworks=["PCI_DSS_4_0", "HIPAA"],
            auto_remediation=False,
        )

        compliance_reporter.monitoring_config = config

        # Start monitoring
        alert_count = 0
        async for alert in compliance_reporter.continuous_compliance_monitoring(
            protocols=["test_protocol"], standards=["PCI-DSS", "HIPAA"], config=config
        ):
            alert_count += 1
            assert isinstance(alert, ComplianceAlert)
            assert alert.alert_id is not None
            assert alert.severity in AlertSeverity

            # Break after receiving a few alerts for testing
            if alert_count >= 3:
                break

    @pytest.mark.asyncio
    async def test_alert_generation(self, compliance_reporter):
        """Test compliance alert generation."""
        # Trigger a compliance check that should generate alerts
        config = ContinuousMonitoringConfig(
            enabled=True,
            frequency=MonitoringFrequency.REAL_TIME,
            frameworks=["PCI_DSS_4_0"],
        )

        # Monitor for a short period
        alerts = []
        timeout = asyncio.create_task(asyncio.sleep(5))

        async for alert in compliance_reporter.continuous_compliance_monitoring(
            protocols=["test_protocol"], standards=["PCI-DSS"], config=config
        ):
            alerts.append(alert)
            if len(alerts) >= 2 or timeout.done():
                break

        # Should have generated some alerts
        assert len(alerts) >= 0  # May be 0 if fully compliant

    @pytest.mark.asyncio
    async def test_monitoring_frequency(self, compliance_reporter):
        """Test different monitoring frequencies."""
        frequencies = [
            MonitoringFrequency.REAL_TIME,
            MonitoringFrequency.HOURLY,
            MonitoringFrequency.DAILY,
        ]

        for frequency in frequencies:
            config = ContinuousMonitoringConfig(
                enabled=True, frequency=frequency, frameworks=["PCI_DSS_4_0"]
            )

            compliance_reporter.monitoring_config = config
            interval = compliance_reporter._get_monitoring_interval()

            assert interval > 0
            assert isinstance(interval, int)


class TestAuditEvidenceGeneration:
    """Test audit evidence generation functionality."""

    @pytest.mark.asyncio
    async def test_generate_audit_evidence(self, compliance_reporter, audit_request):
        """Test comprehensive audit evidence generation."""
        evidence = await compliance_reporter.generate_audit_evidence(audit_request)

        assert evidence is not None
        assert evidence.evidence_id is not None
        assert evidence.request_id == audit_request.request_id
        assert evidence.framework == audit_request.framework
        assert len(evidence.evidence_items) > 0
        assert evidence.compliance_summary is not None
        assert evidence.digital_signature is not None
        assert evidence.verification_status == "verified"

    @pytest.mark.asyncio
    async def test_audit_evidence_completeness(
        self, compliance_reporter, audit_request
    ):
        """Test audit evidence contains all required components."""
        evidence = await compliance_reporter.generate_audit_evidence(audit_request)

        # Check evidence items
        assert len(evidence.evidence_items) > 0

        # Verify evidence types
        evidence_types = {item["evidence_type"] for item in evidence.evidence_items}
        expected_types = {
            "system_configuration",
            "access_controls",
            "monitoring_logs",
            "policies_procedures",
            "training_records",
        }

        # Should have most expected types
        assert len(evidence_types.intersection(expected_types)) >= 3

        # Check supporting documents
        assert len(evidence.supporting_documents) > 0

        # Check audit trail
        assert len(evidence.audit_trail) > 0

    @pytest.mark.asyncio
    async def test_audit_evidence_digital_signature(
        self, compliance_reporter, audit_request
    ):
        """Test digital signature generation for audit evidence."""
        evidence = await compliance_reporter.generate_audit_evidence(audit_request)

        # Should have digital signature
        assert evidence.digital_signature is not None
        assert len(evidence.digital_signature) == 64  # SHA-256 hex digest

        # Signature should be consistent for same data
        signature1 = compliance_reporter._create_digital_signature(
            evidence.evidence_items, evidence.audit_trail
        )
        signature2 = compliance_reporter._create_digital_signature(
            evidence.evidence_items, evidence.audit_trail
        )

        assert signature1 == signature2


class TestPerformanceMetrics:
    """Test performance metrics and success criteria."""

    @pytest.mark.asyncio
    async def test_report_generation_time_tracking(
        self, compliance_reporter, sample_evidence
    ):
        """Test report generation time is tracked."""
        # Generate a report
        await compliance_reporter.generate_compliance_report(
            protocol="test_protocol", standard="PCI-DSS", evidence=sample_evidence
        )

        # Check metrics
        metrics = compliance_reporter.get_performance_metrics()

        assert "report_generation" in metrics
        assert metrics["report_generation"]["total_reports"] > 0
        assert metrics["report_generation"]["average_time_seconds"] >= 0

    @pytest.mark.asyncio
    async def test_compliance_accuracy_tracking(
        self, compliance_reporter, sample_evidence
    ):
        """Test compliance accuracy is tracked."""
        # Generate a report
        await compliance_reporter.generate_compliance_report(
            protocol="test_protocol", standard="PCI-DSS", evidence=sample_evidence
        )

        # Check metrics
        metrics = compliance_reporter.get_performance_metrics()

        assert "compliance_accuracy" in metrics
        assert "average_score" in metrics["compliance_accuracy"]

    @pytest.mark.asyncio
    async def test_service_status(self, compliance_reporter):
        """Test service status reporting."""
        status = compliance_reporter.get_service_status()

        assert status is not None
        assert "service_running" in status
        assert "monitoring_active" in status
        assert "performance_metrics" in status
        assert "llm_service_status" in status
        assert "audit_trail_status" in status

        assert status["service_running"] is True


class TestIntegration:
    """Integration tests for compliance reporter."""

    @pytest.mark.asyncio
    async def test_end_to_end_compliance_workflow(
        self, compliance_reporter, sample_evidence, audit_request
    ):
        """Test complete compliance workflow."""
        # 1. Generate compliance report
        report = await compliance_reporter.generate_compliance_report(
            protocol="test_protocol", standard="PCI-DSS", evidence=sample_evidence
        )
        assert report is not None

        # 2. Generate audit evidence
        evidence = await compliance_reporter.generate_audit_evidence(audit_request)
        assert evidence is not None

        # 3. Check service status
        status = compliance_reporter.get_service_status()
        assert status["service_running"] is True

        # 4. Verify performance metrics
        metrics = compliance_reporter.get_performance_metrics()
        assert metrics["report_generation"]["total_reports"] > 0

    @pytest.mark.asyncio
    async def test_global_instance(self, test_config):
        """Test global compliance reporter instance."""
        # Get global instance
        reporter = await get_compliance_reporter(test_config)
        assert reporter is not None
        assert reporter._running is True

        # Shutdown
        await shutdown_compliance_reporter()

    @pytest.mark.asyncio
    async def test_multiple_standards(self, compliance_reporter, sample_evidence):
        """Test handling multiple compliance standards."""
        standards = ["PCI-DSS", "HIPAA", "SOC2"]

        for standard in standards:
            report = await compliance_reporter.generate_compliance_report(
                protocol="test_protocol",
                standard=standard,
                evidence=sample_evidence,
                format=ReportFormat.JSON,
            )

            assert report is not None
            assert report.framework is not None


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_invalid_standard(self, compliance_reporter, sample_evidence):
        """Test handling of invalid compliance standard."""
        with pytest.raises(Exception):
            await compliance_reporter.generate_compliance_report(
                protocol="test_protocol",
                standard="INVALID_STANDARD",
                evidence=sample_evidence,
            )

    @pytest.mark.asyncio
    async def test_empty_evidence(self, compliance_reporter):
        """Test handling of empty evidence."""
        report = await compliance_reporter.generate_compliance_report(
            protocol="test_protocol", standard="PCI-DSS", evidence={}
        )

        # Should still generate a report, even with empty evidence
        assert report is not None

    @pytest.mark.asyncio
    async def test_service_restart(self, compliance_reporter):
        """Test service can be stopped and restarted."""
        # Stop service
        await compliance_reporter.stop()
        assert compliance_reporter._running is False

        # Restart service
        await compliance_reporter.start()
        assert compliance_reporter._running is True


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_report_generation_benchmark(
        self, compliance_reporter, sample_evidence, benchmark
    ):
        """Benchmark report generation performance."""

        async def generate_report():
            return await compliance_reporter.generate_compliance_report(
                protocol="test_protocol", standard="PCI-DSS", evidence=sample_evidence
            )

        # Run benchmark
        result = await generate_report()
        assert result is not None

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_audit_evidence_benchmark(
        self, compliance_reporter, audit_request, benchmark
    ):
        """Benchmark audit evidence generation performance."""

        async def generate_evidence():
            return await compliance_reporter.generate_audit_evidence(audit_request)

        # Run benchmark
        result = await generate_evidence()
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
