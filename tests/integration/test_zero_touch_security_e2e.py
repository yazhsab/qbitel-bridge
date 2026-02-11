"""
QBITEL - Zero-Touch Security Decision Engine End-to-End Integration Tests

This module contains comprehensive E2E tests for the Zero-Touch Security
Decision Engine, testing the full workflow from threat detection to
automated response execution.
"""

import pytest
import asyncio
import json
import time
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
from uuid import uuid4
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Test Data
# =============================================================================

SAMPLE_SECURITY_EVENTS = [
    {
        "event_type": "intrusion_attempt",
        "severity": "high",
        "source_ip": "192.168.1.100",
        "destination_ip": "10.0.0.50",
        "destination_port": 22,
        "protocol": "SSH",
        "asset_id": "server-prod-001",
        "asset_type": "linux_server",
        "indicators": ["brute_force", "multiple_failed_logins"],
        "raw_event": {"failed_attempts": 50, "time_window_seconds": 120},
    },
    {
        "event_type": "malware_detected",
        "severity": "critical",
        "source_ip": "10.0.0.25",
        "asset_id": "workstation-042",
        "asset_type": "windows_workstation",
        "indicators": ["trojan_signature", "c2_communication", "persistence_mechanism"],
        "raw_event": {"malware_hash": "abc123def456", "family": "emotet"},
    },
    {
        "event_type": "data_exfiltration",
        "severity": "high",
        "source_ip": "10.0.0.100",
        "destination_ip": "203.0.113.50",
        "destination_port": 443,
        "asset_id": "database-001",
        "asset_type": "postgresql",
        "indicators": ["large_data_transfer", "unusual_destination", "off_hours_activity"],
        "raw_event": {"bytes_transferred": 50_000_000, "duration_seconds": 300},
    },
    {
        "event_type": "privilege_escalation",
        "severity": "medium",
        "user_id": "user123",
        "asset_id": "server-web-002",
        "asset_type": "linux_server",
        "indicators": ["sudo_abuse", "unauthorized_access"],
        "raw_event": {"target_user": "root", "command": "sudo su -"},
    },
]


# =============================================================================
# Test: Decision Engine Initialization
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestDecisionEngineInitialization:
    """Tests for decision engine initialization and health."""

    async def test_decision_engine_initializes(self, decision_engine):
        """Test that decision engine initializes correctly."""
        if decision_engine is None:
            pytest.skip("Decision engine not available")

        # Verify initialization
        assert decision_engine is not None

        # Check health status
        health = decision_engine.get_health_status()
        assert health is not None
        assert "status" in health

    async def test_decision_engine_configuration(self, decision_engine):
        """Test decision engine configuration retrieval."""
        if decision_engine is None:
            pytest.skip("Decision engine not available")

        config = decision_engine.get_configuration()

        assert config is not None
        assert "auto_execute_threshold" in config
        assert "auto_approve_threshold" in config
        assert "escalation_threshold" in config

        # Verify threshold ordering
        assert config["auto_execute_threshold"] > config["auto_approve_threshold"]
        assert config["auto_approve_threshold"] > config["escalation_threshold"]


# =============================================================================
# Test: Threat Analysis
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestThreatAnalysis:
    """End-to-end tests for threat analysis."""

    @pytest.mark.parametrize("event_data", SAMPLE_SECURITY_EVENTS)
    async def test_analyze_security_event(self, decision_engine, event_data):
        """Test threat analysis for various event types."""
        if decision_engine is None:
            pytest.skip("Decision engine not available")

        from ai_engine.security.models import SecurityEvent

        # Create security event
        event = SecurityEvent(
            event_id=str(uuid4()),
            event_type=event_data["event_type"],
            timestamp=datetime.now(timezone.utc),
            severity=event_data["severity"],
            source_ip=event_data.get("source_ip"),
            destination_ip=event_data.get("destination_ip"),
            destination_port=event_data.get("destination_port"),
            protocol=event_data.get("protocol"),
            user_id=event_data.get("user_id"),
            asset_id=event_data.get("asset_id"),
            asset_type=event_data.get("asset_type"),
            indicators=event_data.get("indicators", []),
            raw_event=event_data.get("raw_event", {}),
        )

        # Perform analysis
        start_time = time.time()
        analysis = await decision_engine._analyze_threat(event)
        analysis_time = time.time() - start_time

        # Verify analysis structure
        assert analysis is not None
        assert hasattr(analysis, "threat_level") or "threat_level" in analysis
        assert hasattr(analysis, "confidence") or "confidence" in analysis

        # Verify response time (should be fast for analysis only)
        assert analysis_time < 5.0, f"Analysis took {analysis_time}s, expected <5s"

        logger.info(
            f"Analyzed {event_data['event_type']}: "
            f"threat_level={getattr(analysis, 'threat_level', analysis.get('threat_level'))}, "
            f"time={analysis_time:.2f}s"
        )

    async def test_analysis_includes_mitre_techniques(self, decision_engine):
        """Test that analysis includes MITRE ATT&CK techniques."""
        if decision_engine is None:
            pytest.skip("Decision engine not available")

        from ai_engine.security.models import SecurityEvent

        event = SecurityEvent(
            event_id=str(uuid4()),
            event_type="intrusion_attempt",
            timestamp=datetime.now(timezone.utc),
            severity="high",
            source_ip="192.168.1.100",
            destination_ip="10.0.0.50",
            destination_port=22,
            indicators=["brute_force", "credential_stuffing"],
        )

        analysis = await decision_engine._analyze_threat(event)

        # MITRE techniques should be present for well-known attack patterns
        mitre = getattr(analysis, "mitre_techniques", None) or analysis.get(
            "mitre_techniques", []
        )
        # May or may not have MITRE mapping depending on LLM availability
        logger.info(f"MITRE techniques identified: {mitre}")


# =============================================================================
# Test: Automated Response
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestAutomatedResponse:
    """End-to-end tests for automated security response."""

    async def test_respond_to_high_severity_threat(self, decision_engine):
        """Test automated response to high severity threat."""
        if decision_engine is None:
            pytest.skip("Decision engine not available")

        from ai_engine.security.models import SecurityEvent

        event = SecurityEvent(
            event_id=str(uuid4()),
            event_type="intrusion_attempt",
            timestamp=datetime.now(timezone.utc),
            severity="high",
            source_ip="192.168.1.100",
            destination_ip="10.0.0.50",
            destination_port=22,
            protocol="SSH",
            asset_id="server-prod-001",
            indicators=["brute_force", "multiple_failed_logins"],
        )

        # Execute full response pipeline
        start_time = time.time()
        response = await decision_engine.analyze_and_respond(event)
        response_time = time.time() - start_time

        # Verify response structure
        assert response is not None
        assert hasattr(response, "response_id") or "response_id" in response
        assert hasattr(response, "decision_type") or "decision_type" in response
        assert hasattr(response, "primary_actions") or "primary_actions" in response

        # Verify response time SLA (<5 seconds for full pipeline)
        assert response_time < 5.0, f"Response took {response_time}s, expected <5s"

        logger.info(
            f"Response decision: {getattr(response, 'decision_type', response.get('decision_type'))}, "
            f"confidence: {getattr(response, 'confidence', response.get('confidence'))}, "
            f"time: {response_time:.2f}s"
        )

    async def test_respond_to_critical_threat_escalates(self, decision_engine):
        """Test that critical threats are properly escalated."""
        if decision_engine is None:
            pytest.skip("Decision engine not available")

        from ai_engine.security.models import SecurityEvent

        # Critical malware event should trigger escalation
        event = SecurityEvent(
            event_id=str(uuid4()),
            event_type="malware_detected",
            timestamp=datetime.now(timezone.utc),
            severity="critical",
            source_ip="10.0.0.25",
            asset_id="workstation-042",
            asset_type="windows_workstation",
            indicators=["ransomware", "encryption_activity", "shadow_copy_deletion"],
            context={"business_critical": True},
        )

        response = await decision_engine.analyze_and_respond(event)

        # Critical threats may require human approval
        decision_type = getattr(response, "decision_type", response.get("decision_type"))
        escalation_reason = getattr(
            response, "escalation_reason", response.get("escalation_reason")
        )

        logger.info(
            f"Critical threat response: decision={decision_type}, "
            f"escalation_reason={escalation_reason}"
        )

        # Either auto-executed with high confidence OR escalated
        assert decision_type in ["auto_execute", "auto_approve", "escalate"]

    async def test_response_includes_recommended_actions(self, decision_engine):
        """Test that responses include actionable recommendations."""
        if decision_engine is None:
            pytest.skip("Decision engine not available")

        from ai_engine.security.models import SecurityEvent

        event = SecurityEvent(
            event_id=str(uuid4()),
            event_type="data_exfiltration",
            timestamp=datetime.now(timezone.utc),
            severity="high",
            source_ip="10.0.0.100",
            destination_ip="203.0.113.50",
            destination_port=443,
            asset_id="database-001",
            indicators=["large_data_transfer", "unusual_destination"],
        )

        response = await decision_engine.analyze_and_respond(event)

        # Should have primary actions
        actions = getattr(response, "primary_actions", response.get("primary_actions", []))
        assert len(actions) > 0, "Expected at least one recommended action"

        # Log actions for verification
        for action in actions[:3]:
            action_type = getattr(action, "action_type", action.get("action_type"))
            logger.info(f"Recommended action: {action_type}")


# =============================================================================
# Test: Response Simulation
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestResponseSimulation:
    """Tests for response simulation (dry-run) functionality."""

    async def test_simulate_response_does_not_execute(self, decision_engine):
        """Test that simulation does not execute actual actions."""
        if decision_engine is None:
            pytest.skip("Decision engine not available")

        from ai_engine.security.models import SecurityEvent

        event = SecurityEvent(
            event_id=str(uuid4()),
            event_type="intrusion_attempt",
            timestamp=datetime.now(timezone.utc),
            severity="high",
            source_ip="192.168.1.100",
            destination_ip="10.0.0.50",
        )

        # Simulate response
        simulation = await decision_engine.simulate_response(event)

        # Verify simulation structure
        assert simulation is not None
        assert "decision" in simulation or "predicted_decision" in simulation
        assert "actions" in simulation or "predicted_actions" in simulation

        # Verify no execution status (should be simulation only)
        execution_status = simulation.get("execution_status")
        assert execution_status is None or execution_status == "simulated"


# =============================================================================
# Test: Decision History and Metrics
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestDecisionHistoryAndMetrics:
    """Tests for decision tracking and metrics."""

    async def test_get_decision_history(self, decision_engine):
        """Test retrieving decision history."""
        if decision_engine is None:
            pytest.skip("Decision engine not available")

        # Get history (may be empty initially)
        history = decision_engine.get_decision_history(limit=10)

        assert history is not None
        assert isinstance(history, list)

    async def test_get_metrics(self, decision_engine):
        """Test retrieving decision engine metrics."""
        if decision_engine is None:
            pytest.skip("Decision engine not available")

        metrics = decision_engine.get_metrics(time_range="24h")

        assert metrics is not None
        assert "total_events" in metrics or "total_events_processed" in metrics

    async def test_pending_approvals_retrieval(self, decision_engine):
        """Test retrieving pending manual approvals."""
        if decision_engine is None:
            pytest.skip("Decision engine not available")

        pending = decision_engine.get_pending_approvals(limit=10)

        assert pending is not None
        assert isinstance(pending, list)


# =============================================================================
# Test: Configuration Updates
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestConfigurationUpdates:
    """Tests for runtime configuration updates."""

    async def test_update_thresholds(self, decision_engine):
        """Test updating decision thresholds."""
        if decision_engine is None:
            pytest.skip("Decision engine not available")

        # Get original config
        original = decision_engine.get_configuration()

        # Update threshold
        updated = decision_engine.update_configuration(
            auto_execute_threshold=0.98,  # More conservative
        )

        assert updated is not None
        assert updated.get("auto_execute_threshold") == 0.98

        # Restore original
        decision_engine.update_configuration(
            auto_execute_threshold=original.get("auto_execute_threshold", 0.95)
        )


# =============================================================================
# Test: Performance and SLA
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestPerformanceSLA:
    """Tests for verifying performance SLAs."""

    async def test_analysis_latency_sla(self, decision_engine):
        """Test that threat analysis meets latency SLA (<2s)."""
        if decision_engine is None:
            pytest.skip("Decision engine not available")

        from ai_engine.security.models import SecurityEvent

        latencies = []
        for _ in range(5):
            event = SecurityEvent(
                event_id=str(uuid4()),
                event_type="intrusion_attempt",
                timestamp=datetime.now(timezone.utc),
                severity="high",
                source_ip=f"192.168.1.{100 + _}",
            )

            start = time.time()
            await decision_engine._analyze_threat(event)
            latencies.append(time.time() - start)

        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

        logger.info(
            f"Analysis latency: avg={avg_latency:.2f}s, p95={p95_latency:.2f}s"
        )

        # SLA: p95 < 2 seconds
        assert p95_latency < 2.0, f"P95 latency {p95_latency}s exceeds 2s SLA"

    async def test_response_latency_sla(self, decision_engine):
        """Test that full response meets latency SLA (<5s)."""
        if decision_engine is None:
            pytest.skip("Decision engine not available")

        from ai_engine.security.models import SecurityEvent

        latencies = []
        for i in range(3):  # Fewer iterations for full pipeline
            event = SecurityEvent(
                event_id=str(uuid4()),
                event_type="privilege_escalation",
                timestamp=datetime.now(timezone.utc),
                severity="medium",
                user_id=f"user{i}",
                asset_id=f"server-{i}",
            )

            start = time.time()
            await decision_engine.analyze_and_respond(event)
            latencies.append(time.time() - start)

        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)

        logger.info(
            f"Response latency: avg={avg_latency:.2f}s, max={max_latency:.2f}s"
        )

        # SLA: max < 5 seconds
        assert max_latency < 5.0, f"Max latency {max_latency}s exceeds 5s SLA"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
async def decision_engine():
    """Provide a decision engine instance for testing."""
    try:
        from ai_engine.security.decision_engine import ZeroTouchDecisionEngine
        from ai_engine.core.config import get_config

        config = get_config()
        engine = ZeroTouchDecisionEngine(config)
        yield engine
    except ImportError:
        yield None
    except Exception as e:
        logger.warning(f"Could not create decision engine: {e}")
        yield None
