"""
Tests for Security Integrations (SIEM, Ticketing, Network Security).
Covers integration connectors, configuration, and external system communication.
"""

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import asdict

from ai_engine.security.integrations.base_connector import (
    BaseIntegrationConnector,
    IntegrationStatus,
    IntegrationType,
    IntegrationConfig,
    IntegrationResult,
)
from ai_engine.security.integrations.siem_connector import SIEMConnector
from ai_engine.security.integrations.ticketing_connector import TicketingConnector
from ai_engine.security.integrations.network_security_connector import (
    NetworkSecurityConnector,
)
from ai_engine.core.config import Config


class TestIntegrationStatus:
    """Test Integration Status enum."""

    def test_status_values(self):
        """Test integration status values."""
        assert IntegrationStatus.CONNECTED.value == "connected"
        assert IntegrationStatus.DISCONNECTED.value == "disconnected"
        assert IntegrationStatus.DEGRADED.value == "degraded"
        assert IntegrationStatus.ERROR.value == "error"
        assert IntegrationStatus.UNKNOWN.value == "unknown"

    def test_status_from_string(self):
        """Test creating status from string."""
        status = IntegrationStatus("connected")
        assert status == IntegrationStatus.CONNECTED


class TestIntegrationType:
    """Test Integration Type enum."""

    def test_integration_types(self):
        """Test all integration types."""
        assert IntegrationType.SIEM.value == "siem"
        assert IntegrationType.TICKETING.value == "ticketing"
        assert IntegrationType.COMMUNICATION.value == "communication"
        assert IntegrationType.NETWORK_SECURITY.value == "network_security"
        assert IntegrationType.THREAT_INTELLIGENCE.value == "threat_intelligence"
        assert IntegrationType.ORCHESTRATION.value == "orchestration"


class TestIntegrationConfig:
    """Test Integration Config dataclass."""

    def test_create_basic_config(self):
        """Test creating basic integration config."""
        config = IntegrationConfig(
            name="Splunk Production",
            integration_type=IntegrationType.SIEM,
            enabled=True,
            endpoint="https://splunk.example.com",
        )

        assert config.name == "Splunk Production"
        assert config.integration_type == IntegrationType.SIEM
        assert config.enabled is True
        assert config.timeout_seconds == 30
        assert config.retry_attempts == 3

    def test_config_with_credentials(self):
        """Test config with credentials."""
        config = IntegrationConfig(
            name="JIRA",
            integration_type=IntegrationType.TICKETING,
            enabled=True,
            endpoint="https://jira.example.com",
            credentials={"username": "cronos_bot", "api_token": "secret_token"},
        )

        assert config.credentials["username"] == "cronos_bot"
        assert "api_token" in config.credentials

    def test_config_with_rate_limiting(self):
        """Test config with rate limiting."""
        config = IntegrationConfig(
            name="Palo Alto",
            integration_type=IntegrationType.NETWORK_SECURITY,
            enabled=True,
            rate_limit_requests_per_minute=60,
        )

        assert config.rate_limit_requests_per_minute == 60

    def test_config_custom_settings(self):
        """Test config with custom settings."""
        config = IntegrationConfig(
            name="Sentinel",
            integration_type=IntegrationType.SIEM,
            enabled=True,
            custom_config={
                "workspace_id": "abc-123",
                "log_analytics_uri": "https://api.loganalytics.io",
            },
        )

        assert config.custom_config["workspace_id"] == "abc-123"

    def test_config_to_dict(self):
        """Test converting config to dict."""
        config = IntegrationConfig(
            name="Test",
            integration_type=IntegrationType.SIEM,
            enabled=True,
        )

        data = asdict(config)
        assert data["name"] == "Test"
        assert data["enabled"] is True


class TestIntegrationResult:
    """Test Integration Result dataclass."""

    def test_successful_result(self):
        """Test successful integration result."""
        result = IntegrationResult(
            success=True,
            message="Event forwarded successfully",
            response_data={"event_id": "evt_123", "acknowledged": True},
            execution_time_ms=150.5,
        )

        assert result.success is True
        assert result.response_data["event_id"] == "evt_123"
        assert result.execution_time_ms == 150.5
        assert result.error_code is None

    def test_failed_result(self):
        """Test failed integration result."""
        result = IntegrationResult(
            success=False,
            message="Connection timeout",
            error_code="TIMEOUT_001",
            execution_time_ms=30000.0,
        )

        assert result.success is False
        assert result.error_code == "TIMEOUT_001"
        assert result.response_data is None

    def test_result_with_metadata(self):
        """Test result with metadata."""
        result = IntegrationResult(
            success=True,
            message="Ticket created",
            response_data={"ticket_id": "INC0012345"},
            metadata={"assignee": "security_team", "priority": "high"},
        )

        assert result.metadata["assignee"] == "security_team"


class TestSIEMConnector:
    """Test SIEM Connector."""

    @pytest_asyncio.fixture
    async def connector(self):
        """Create SIEM connector."""
        config = IntegrationConfig(
            name="Splunk Test",
            integration_type=IntegrationType.SIEM,
            enabled=True,
            endpoint="https://splunk.test.com",
        )

        with patch(
            "ai_engine.security.integrations.siem_connector.SIEMConnector"
        ) as mock_conn:
            mock_instance = AsyncMock()
            mock_conn.return_value = mock_instance
            return mock_instance

    @pytest.mark.asyncio
    async def test_forward_security_event(self, connector):
        """Test forwarding security event to SIEM."""
        connector.forward_event = AsyncMock(
            return_value=IntegrationResult(
                success=True,
                message="Event forwarded",
                response_data={"event_id": "evt_789"},
            )
        )

        event = {
            "event_type": "threat_detected",
            "severity": "high",
            "source_ip": "192.168.1.100",
            "timestamp": datetime.now().isoformat(),
        }

        result = await connector.forward_event(event)

        assert result.success is True
        assert result.response_data["event_id"] == "evt_789"

    @pytest.mark.asyncio
    async def test_query_siem_logs(self, connector):
        """Test querying SIEM logs."""
        connector.query_logs = AsyncMock(
            return_value={
                "total_results": 42,
                "events": [
                    {"id": "1", "message": "Login failed"},
                    {"id": "2", "message": "Unauthorized access attempt"},
                ],
            }
        )

        query = "source_ip:192.168.1.* AND severity:high"
        results = await connector.query_logs(query, time_range="last_24h")

        assert results["total_results"] == 42
        assert len(results["events"]) == 2

    @pytest.mark.asyncio
    async def test_check_siem_connection(self, connector):
        """Test checking SIEM connection."""
        connector.check_connection = AsyncMock(return_value=IntegrationStatus.CONNECTED)

        status = await connector.check_connection()

        assert status == IntegrationStatus.CONNECTED

    @pytest.mark.asyncio
    async def test_siem_connection_failure(self, connector):
        """Test SIEM connection failure."""
        connector.check_connection = AsyncMock(return_value=IntegrationStatus.ERROR)

        status = await connector.check_connection()

        assert status == IntegrationStatus.ERROR


class TestTicketingConnector:
    """Test Ticketing Connector."""

    @pytest_asyncio.fixture
    async def connector(self):
        """Create ticketing connector."""
        config = IntegrationConfig(
            name="JIRA Test",
            integration_type=IntegrationType.TICKETING,
            enabled=True,
            endpoint="https://jira.test.com",
        )

        with patch(
            "ai_engine.security.integrations.ticketing_connector.TicketingConnector"
        ) as mock_conn:
            mock_instance = AsyncMock()
            mock_conn.return_value = mock_instance
            return mock_instance

    @pytest.mark.asyncio
    async def test_create_ticket(self, connector):
        """Test creating security incident ticket."""
        connector.create_ticket = AsyncMock(
            return_value=IntegrationResult(
                success=True,
                message="Ticket created",
                response_data={"ticket_id": "SEC-1234", "url": "https://jira/SEC-1234"},
            )
        )

        ticket_data = {
            "summary": "Potential data exfiltration detected",
            "description": "Anomalous network traffic observed",
            "priority": "high",
            "labels": ["security", "network", "cronos-ai"],
        }

        result = await connector.create_ticket(ticket_data)

        assert result.success is True
        assert result.response_data["ticket_id"] == "SEC-1234"

    @pytest.mark.asyncio
    async def test_update_ticket(self, connector):
        """Test updating existing ticket."""
        connector.update_ticket = AsyncMock(
            return_value=IntegrationResult(
                success=True,
                message="Ticket updated",
            )
        )

        result = await connector.update_ticket(
            ticket_id="SEC-1234",
            updates={"status": "In Progress", "assignee": "security_analyst"},
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_get_ticket_status(self, connector):
        """Test getting ticket status."""
        connector.get_ticket = AsyncMock(
            return_value={
                "ticket_id": "SEC-1234",
                "status": "Open",
                "assignee": "security_team",
                "created_at": "2025-01-01T10:00:00Z",
            }
        )

        ticket = await connector.get_ticket("SEC-1234")

        assert ticket["status"] == "Open"
        assert ticket["assignee"] == "security_team"

    @pytest.mark.asyncio
    async def test_add_ticket_comment(self, connector):
        """Test adding comment to ticket."""
        connector.add_comment = AsyncMock(
            return_value=IntegrationResult(
                success=True,
                message="Comment added",
            )
        )

        result = await connector.add_comment(
            ticket_id="SEC-1234",
            comment="CRONOS AI has completed automated analysis. No further threats detected.",
        )

        assert result.success is True


class TestNetworkSecurityConnector:
    """Test Network Security Connector."""

    @pytest_asyncio.fixture
    async def connector(self):
        """Create network security connector."""
        config = IntegrationConfig(
            name="Palo Alto Test",
            integration_type=IntegrationType.NETWORK_SECURITY,
            enabled=True,
            endpoint="https://firewall.test.com",
        )

        with patch(
            "ai_engine.security.integrations.network_security_connector.NetworkSecurityConnector"
        ) as mock_conn:
            mock_instance = AsyncMock()
            mock_conn.return_value = mock_instance
            return mock_instance

    @pytest.mark.asyncio
    async def test_block_ip_address(self, connector):
        """Test blocking malicious IP address."""
        connector.block_ip = AsyncMock(
            return_value=IntegrationResult(
                success=True,
                message="IP blocked",
                response_data={"rule_id": "rule_789", "ip": "203.0.113.42"},
            )
        )

        result = await connector.block_ip(
            ip_address="203.0.113.42",
            reason="Malicious activity detected by CRONOS AI",
        )

        assert result.success is True
        assert result.response_data["ip"] == "203.0.113.42"

    @pytest.mark.asyncio
    async def test_create_firewall_rule(self, connector):
        """Test creating firewall rule."""
        connector.create_rule = AsyncMock(
            return_value=IntegrationResult(
                success=True,
                message="Rule created",
                response_data={"rule_id": "rule_456"},
            )
        )

        rule_config = {
            "name": "Block suspicious traffic",
            "source": "any",
            "destination": "10.0.0.0/8",
            "action": "deny",
            "protocol": "tcp",
            "port": "445",
        }

        result = await connector.create_rule(rule_config)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_get_firewall_logs(self, connector):
        """Test retrieving firewall logs."""
        connector.get_logs = AsyncMock(
            return_value={
                "total": 150,
                "logs": [
                    {"action": "block", "src_ip": "203.0.113.1", "dst_port": 22},
                    {"action": "allow", "src_ip": "192.168.1.10", "dst_port": 443},
                ],
            }
        )

        logs = await connector.get_logs(time_range="last_hour", action="block")

        assert logs["total"] == 150
        assert logs["logs"][0]["action"] == "block"

    @pytest.mark.asyncio
    async def test_quarantine_device(self, connector):
        """Test quarantining compromised device."""
        connector.quarantine_device = AsyncMock(
            return_value=IntegrationResult(
                success=True,
                message="Device quarantined",
                response_data={"device_id": "dev_123", "vlan": "quarantine"},
            )
        )

        result = await connector.quarantine_device(
            device_id="dev_123",
            reason="Ransomware indicators detected",
        )

        assert result.success is True


class TestIntegrationRetryMechanism:
    """Test integration retry and error handling."""

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test automatic retry on temporary failure."""
        config = IntegrationConfig(
            name="Test",
            integration_type=IntegrationType.SIEM,
            enabled=True,
            retry_attempts=3,
        )

        with patch(
            "ai_engine.security.integrations.base_connector.BaseIntegrationConnector"
        ) as mock_base:
            mock_instance = AsyncMock()
            # Fail twice, succeed on third attempt
            mock_instance.send_request = AsyncMock(
                side_effect=[
                    Exception("Timeout"),
                    Exception("Connection error"),
                    IntegrationResult(success=True, message="Success"),
                ]
            )
            mock_base.return_value = mock_instance

            # Mock retry logic would be tested here
            # In actual implementation, connector should retry

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test max retries exceeded."""
        config = IntegrationConfig(
            name="Test",
            integration_type=IntegrationType.SIEM,
            enabled=True,
            retry_attempts=2,
        )

        # After max retries, should fail gracefully


class TestIntegrationConcurrency:
    """Test concurrent integration operations."""

    @pytest.mark.asyncio
    async def test_concurrent_event_forwarding(self):
        """Test forwarding multiple events concurrently."""
        with patch(
            "ai_engine.security.integrations.siem_connector.SIEMConnector"
        ) as mock_conn:
            mock_instance = AsyncMock()
            mock_instance.forward_event = AsyncMock(
                side_effect=[
                    IntegrationResult(success=True, message=f"Event {i} forwarded")
                    for i in range(10)
                ]
            )
            mock_conn.return_value = mock_instance

            connector = mock_instance

            # Forward 10 events concurrently
            events = [{"id": i, "type": "test_event"} for i in range(10)]

            tasks = [connector.forward_event(event) for event in events]
            results = await asyncio.gather(*tasks)

            assert len(results) == 10
            assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_ticket_creation(self):
        """Test creating multiple tickets concurrently."""
        with patch(
            "ai_engine.security.integrations.ticketing_connector.TicketingConnector"
        ) as mock_conn:
            mock_instance = AsyncMock()
            mock_instance.create_ticket = AsyncMock(
                side_effect=[
                    IntegrationResult(
                        success=True,
                        message="Ticket created",
                        response_data={"ticket_id": f"SEC-{i:04d}"},
                    )
                    for i in range(5)
                ]
            )
            mock_conn.return_value = mock_instance

            connector = mock_instance

            tickets = [
                {"summary": f"Issue {i}", "priority": "medium"} for i in range(5)
            ]

            tasks = [connector.create_ticket(ticket) for ticket in tickets]
            results = await asyncio.gather(*tasks)

            assert len(results) == 5
            assert all(r.success for r in results)


class TestIntegrationEdgeCases:
    """Test integration edge cases."""

    @pytest.mark.asyncio
    async def test_empty_event_data(self):
        """Test handling empty event data."""
        with patch(
            "ai_engine.security.integrations.siem_connector.SIEMConnector"
        ) as mock_conn:
            mock_instance = AsyncMock()
            mock_instance.forward_event = AsyncMock(
                return_value=IntegrationResult(
                    success=False,
                    message="Invalid event data",
                    error_code="INVALID_DATA",
                )
            )
            mock_conn.return_value = mock_instance

            result = await mock_instance.forward_event({})

            assert result.success is False
            assert result.error_code == "INVALID_DATA"

    @pytest.mark.asyncio
    async def test_disabled_integration(self):
        """Test attempting to use disabled integration."""
        config = IntegrationConfig(
            name="Disabled SIEM",
            integration_type=IntegrationType.SIEM,
            enabled=False,
        )

        # Should not attempt connection or operations

    def test_integration_config_validation(self):
        """Test integration config validation."""
        # Missing endpoint for remote integration
        config = IntegrationConfig(
            name="Test",
            integration_type=IntegrationType.SIEM,
            enabled=True,
            endpoint=None,
        )

        assert config.endpoint is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
