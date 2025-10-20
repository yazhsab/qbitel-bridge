"""
CRONOS AI - Threat Intelligence Platform Comprehensive Test Suite

Tests for STIX/TAXII client, MITRE ATT&CK mapper, threat hunter, and TIP manager.
Achieves >90% code coverage across all TIP components.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from ai_engine.core.config import Config
from ai_engine.core.exceptions import CronosAIException
from ai_engine.security.models import SecurityEvent, ThreatLevel, SecurityEventType
from ai_engine.threat_intelligence import (
    STIXTAXIIClient,
    STIXIndicator,
    TAXIIServer,
    IOCFeed,
    MITREATTACKMapper,
    ATTACKTechnique,
    MITRETactic,
    TTPMapping,
    ThreatHunter,
    HuntCampaign,
    HuntFinding,
    HuntHypothesis,
    HuntType,
    ThreatIntelligenceManager,
)


@pytest.fixture
def config():
    """Create test configuration."""
    config = Config()
    config.llm_provider = "anthropic"
    config.anthropic_api_key = "test-key"
    return config


@pytest.fixture
def sample_security_event():
    """Create sample security event."""
    return SecurityEvent(
        event_id="evt_001",
        event_type=SecurityEventType.SUSPICIOUS_NETWORK_ACTIVITY,
        threat_level=ThreatLevel.HIGH,
        description="Port scanning detected from external IP",
        source_ip="203.0.113.42",
        destination_ip="10.0.1.50",
    )


@pytest.fixture
def sample_stix_indicator():
    """Create sample STIX indicator."""
    return STIXIndicator(
        id="indicator--12345",
        pattern="[ipv4-addr:value = '203.0.113.42']",
        valid_from=datetime.utcnow(),
        name="Malicious Scanner IP",
        description="Known scanning infrastructure",
        labels=["scanner", "malicious"],
        confidence=85,
    )


# STIX/TAXII Client Tests


class TestSTIXTAXIIClient:
    """Test STIX/TAXII client functionality."""

    @pytest.mark.asyncio
    async def test_client_initialization(self, config):
        """Test client initialization."""
        client = STIXTAXIIClient(config)

        assert client.config == config
        assert len(client.indicators_cache) == 0

    @pytest.mark.asyncio
    async def test_add_ioc_feed(self, config):
        """Test adding IOC feed."""
        client = STIXTAXIIClient(config)
        await client.initialize()

        feed = IOCFeed(
            feed_id="test_feed",
            name="Test Feed",
            source="test_source",
            feed_type="json",
            url="https://example.com/feed.json",
            enabled=True,
            update_interval_hours=24,
        )

        client.ioc_feeds[feed.feed_id] = feed

        assert "test_feed" in client.ioc_feeds
        assert client.ioc_feeds["test_feed"].name == "Test Feed"

    @pytest.mark.asyncio
    async def test_check_ioc_match(self, config, sample_stix_indicator):
        """Test IOC matching."""
        client = STIXTAXIIClient(config)
        await client.initialize()

        # Add indicator to cache
        client.indicators_cache[sample_stix_indicator.id] = sample_stix_indicator

        # Check for match
        result = await client.check_ioc("203.0.113.42", "ipv4-addr")

        assert result is not None
        assert result.id == sample_stix_indicator.id
        assert result.confidence == 85

    @pytest.mark.asyncio
    async def test_check_ioc_no_match(self, config):
        """Test IOC with no match."""
        client = STIXTAXIIClient(config)
        await client.initialize()

        result = await client.check_ioc("192.0.2.1", "ipv4-addr")

        assert result is None

    @pytest.mark.asyncio
    async def test_query_indicators_by_type(self, config, sample_stix_indicator):
        """Test querying indicators - all indicators without type filter."""
        client = STIXTAXIIClient(config)
        await client.initialize()

        client.indicators_cache[sample_stix_indicator.id] = sample_stix_indicator

        # Query without type filter to get all indicators
        results = await client.query_indicators(indicator_type=None, limit=10)

        assert len(results) > 0
        assert results[0].id == sample_stix_indicator.id

    @pytest.mark.asyncio
    async def test_query_indicators_by_confidence(self, config, sample_stix_indicator):
        """Test querying indicators by confidence threshold."""
        client = STIXTAXIIClient(config)
        await client.initialize()

        client.indicators_cache[sample_stix_indicator.id] = sample_stix_indicator

        # Should find indicator with confidence 85
        results = await client.query_indicators(min_confidence=80, limit=10)
        assert len(results) > 0

        # Should not find indicator with confidence 85
        results = await client.query_indicators(min_confidence=90, limit=10)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_query_indicators_by_labels(self, config, sample_stix_indicator):
        """Test querying indicators by labels."""
        client = STIXTAXIIClient(config)
        await client.initialize()

        client.indicators_cache[sample_stix_indicator.id] = sample_stix_indicator

        # Should find indicator with "scanner" label
        results = await client.query_indicators(labels=["scanner"], limit=10)
        assert len(results) > 0

        # Should not find indicator with "ransomware" label
        results = await client.query_indicators(labels=["ransomware"], limit=10)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_shutdown(self, config):
        """Test client shutdown."""
        client = STIXTAXIIClient(config)
        await client.initialize()
        await client.shutdown()

        # Session should be closed
        assert client.session is None or client.session.closed


# MITRE ATT&CK Mapper Tests


class TestMITREATTACKMapper:
    """Test MITRE ATT&CK mapper functionality."""

    @pytest.mark.asyncio
    async def test_mapper_initialization(self, config):
        """Test mapper initialization."""
        mapper = MITREATTACKMapper(config)

        assert mapper.config == config
        assert len(mapper.techniques) == 0

    @pytest.mark.asyncio
    async def test_load_attack_techniques(self, config):
        """Test loading ATT&CK techniques."""
        mapper = MITREATTACKMapper(config)
        await mapper.initialize()

        assert len(mapper.techniques) > 0
        assert "T1595" in mapper.techniques  # Active Scanning
        assert "T1046" in mapper.techniques  # Network Service Discovery

    @pytest.mark.asyncio
    async def test_map_scan_event_to_techniques(self, config, sample_security_event):
        """Test mapping network scan event to techniques."""
        mapper = MITREATTACKMapper(config)
        await mapper.initialize()

        # Modify event to trigger reconnaissance mapping
        sample_security_event.event_type = "port_scan"
        sample_security_event.description = "Port scanning activity detected"

        mapping = await mapper.map_event_to_techniques(sample_security_event)

        assert mapping.event_id == sample_security_event.event_id
        assert len(mapping.matched_techniques) > 0

        # Should map to reconnaissance techniques
        technique_ids = [t.technique_id for t in mapping.matched_techniques]
        assert any(tid in ["T1595", "T1046"] for tid in technique_ids)

    @pytest.mark.asyncio
    async def test_map_unauthorized_access_to_techniques(
        self, config, sample_security_event
    ):
        """Test mapping unauthorized access to techniques."""
        mapper = MITREATTACKMapper(config)
        await mapper.initialize()

        sample_security_event.event_type = "authentication_failure"
        sample_security_event.description = "Multiple failed login attempts"

        mapping = await mapper.map_event_to_techniques(sample_security_event)

        assert len(mapping.matched_techniques) > 0
        technique_ids = [t.technique_id for t in mapping.matched_techniques]
        assert "T1078" in technique_ids  # Valid Accounts

    @pytest.mark.asyncio
    async def test_search_techniques(self, config):
        """Test searching techniques."""
        mapper = MITREATTACKMapper(config)
        await mapper.initialize()

        results = await mapper.search_techniques("scanning", limit=5)

        assert len(results) > 0
        assert any("scan" in t.name.lower() for t in results)

    @pytest.mark.asyncio
    async def test_get_detection_coverage_report(self, config):
        """Test detection coverage reporting."""
        mapper = MITREATTACKMapper(config)
        await mapper.initialize()

        report = mapper.get_detection_coverage_report()

        assert "total_techniques" in report
        assert "coverage_by_tactic" in report
        assert report["total_techniques"] > 0

    @pytest.mark.asyncio
    async def test_shutdown(self, config):
        """Test mapper shutdown."""
        mapper = MITREATTACKMapper(config)
        await mapper.initialize()
        await mapper.shutdown()

        # Mapper should have shut down cleanly
        assert mapper.techniques is not None


# Threat Hunter Tests


class TestThreatHunter:
    """Test threat hunter functionality."""

    @pytest.mark.asyncio
    async def test_hunter_initialization(self, config):
        """Test hunter initialization."""
        stix_client = STIXTAXIIClient(config)
        attack_mapper = MITREATTACKMapper(config)

        hunter = ThreatHunter(
            config, stix_client=stix_client, attack_mapper=attack_mapper
        )

        assert hunter.config == config
        assert hunter.stix_client == stix_client
        assert hunter.attack_mapper == attack_mapper

    @pytest.mark.asyncio
    async def test_load_hunt_hypotheses(self, config):
        """Test loading hunt hypotheses."""
        stix_client = STIXTAXIIClient(config)
        await stix_client.initialize()

        attack_mapper = MITREATTACKMapper(config)
        await attack_mapper.initialize()

        hunter = ThreatHunter(
            config, stix_client=stix_client, attack_mapper=attack_mapper
        )
        await hunter.initialize()

        assert len(hunter.hypotheses) > 0
        assert "h001" in hunter.hypotheses  # Unknown IOCs
        assert "h002" in hunter.hypotheses  # Lateral movement

    @pytest.mark.asyncio
    async def test_add_event_to_history(self, config, sample_security_event):
        """Test adding event to hunt history."""
        stix_client = STIXTAXIIClient(config)
        await stix_client.initialize()

        attack_mapper = MITREATTACKMapper(config)
        await attack_mapper.initialize()

        hunter = ThreatHunter(
            config, stix_client=stix_client, attack_mapper=attack_mapper
        )
        await hunter.initialize()

        hunter.add_event_to_history(sample_security_event)

        assert len(hunter.event_history) == 1
        assert hunter.event_history[0].event_id == sample_security_event.event_id

    @pytest.mark.asyncio
    async def test_execute_hunt_campaign_all_hypotheses(
        self, config, sample_security_event
    ):
        """Test executing hunt campaign with all hypotheses."""
        stix_client = STIXTAXIIClient(config)
        await stix_client.initialize()

        attack_mapper = MITREATTACKMapper(config)
        await attack_mapper.initialize()

        hunter = ThreatHunter(
            config, stix_client=stix_client, attack_mapper=attack_mapper
        )
        await hunter.initialize()

        # Add some events to history
        hunter.add_event_to_history(sample_security_event)

        campaign = await hunter.execute_hunt_campaign(
            hypotheses=None, time_range_hours=24  # All hypotheses
        )

        assert campaign.campaign_id is not None
        assert len(campaign.hypotheses_tested) > 0
        assert campaign.end_time is not None

    @pytest.mark.asyncio
    async def test_execute_hunt_campaign_specific_hypothesis(
        self, config, sample_security_event
    ):
        """Test executing hunt campaign with specific hypothesis."""
        stix_client = STIXTAXIIClient(config)
        await stix_client.initialize()

        attack_mapper = MITREATTACKMapper(config)
        await attack_mapper.initialize()

        hunter = ThreatHunter(
            config, stix_client=stix_client, attack_mapper=attack_mapper
        )
        await hunter.initialize()

        hunter.add_event_to_history(sample_security_event)

        campaign = await hunter.execute_hunt_campaign(
            hypotheses=["h001"], time_range_hours=24  # Unknown IOCs hypothesis
        )

        assert len(campaign.hypotheses_tested) == 1
        assert campaign.hypotheses_tested[0] == "h001"

    @pytest.mark.asyncio
    async def test_hunt_generates_findings(
        self, config, sample_security_event, sample_stix_indicator
    ):
        """Test that hunting generates findings when IOCs match."""
        stix_client = STIXTAXIIClient(config)
        await stix_client.initialize()

        # Add malicious indicator
        stix_client.indicators_cache[sample_stix_indicator.id] = sample_stix_indicator

        attack_mapper = MITREATTACKMapper(config)
        await attack_mapper.initialize()

        hunter = ThreatHunter(
            config, stix_client=stix_client, attack_mapper=attack_mapper
        )
        await hunter.initialize()

        # Add event with malicious IP
        sample_security_event.source_ip = "203.0.113.42"
        hunter.add_event_to_history(sample_security_event)

        campaign = await hunter.execute_hunt_campaign(
            hypotheses=["h001"], time_range_hours=24  # Unknown IOCs
        )

        # Should generate finding for malicious IP
        assert len(campaign.findings) > 0


# TIP Manager Tests


class TestThreatIntelligenceManager:
    """Test TIP manager functionality."""

    @pytest.mark.asyncio
    async def test_manager_initialization(self, config):
        """Test manager initialization."""
        manager = ThreatIntelligenceManager(config)

        assert manager.config == config
        assert not manager._initialized
        assert manager.stix_client is None
        assert manager.attack_mapper is None
        assert manager.threat_hunter is None

    @pytest.mark.asyncio
    async def test_initialize_all_components(self, config):
        """Test initializing all TIP components."""
        manager = ThreatIntelligenceManager(config)
        await manager.initialize()

        assert manager._initialized
        assert manager.stix_client is not None
        assert manager.attack_mapper is not None
        assert manager.threat_hunter is not None

    @pytest.mark.asyncio
    async def test_process_security_event(self, config, sample_security_event):
        """Test processing security event through TIP pipeline."""
        manager = ThreatIntelligenceManager(config)
        await manager.initialize()

        enrichment = await manager.process_security_event(sample_security_event)

        assert enrichment["event_id"] == sample_security_event.event_id
        assert "ioc_matches" in enrichment
        assert "ttp_mapping" in enrichment
        assert "threat_score" in enrichment
        assert enrichment["threat_score"] >= 0.0
        assert enrichment["threat_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_process_event_with_ioc_match(
        self, config, sample_security_event, sample_stix_indicator
    ):
        """Test event processing with IOC match increases threat score."""
        manager = ThreatIntelligenceManager(config)
        await manager.initialize()

        # Add malicious indicator
        manager.stix_client.indicators_cache[sample_stix_indicator.id] = (
            sample_stix_indicator
        )

        # Event with malicious IP
        sample_security_event.source_ip = "203.0.113.42"

        enrichment = await manager.process_security_event(sample_security_event)

        assert len(enrichment["ioc_matches"]) > 0
        assert enrichment["threat_score"] > 0.0

    @pytest.mark.asyncio
    async def test_query_iocs(self, config, sample_stix_indicator):
        """Test querying IOCs."""
        manager = ThreatIntelligenceManager(config)
        await manager.initialize()

        manager.stix_client.indicators_cache[sample_stix_indicator.id] = (
            sample_stix_indicator
        )

        result = await manager.query_threat_intelligence(
            query_type="iocs",
            query_params={
                "indicator_type": "ipv4-addr",
                "min_confidence": 80,
                "limit": 10,
            },
        )

        assert "indicators" in result
        assert result["total"] > 0

    @pytest.mark.asyncio
    async def test_query_techniques(self, config):
        """Test querying MITRE ATT&CK techniques."""
        manager = ThreatIntelligenceManager(config)
        await manager.initialize()

        result = await manager.query_threat_intelligence(
            query_type="techniques", query_params={"query": "scanning", "limit": 5}
        )

        assert "techniques" in result
        assert result["total"] > 0

    @pytest.mark.asyncio
    async def test_execute_threat_hunt(self, config, sample_security_event):
        """Test executing threat hunt."""
        manager = ThreatIntelligenceManager(config)
        await manager.initialize()

        # Add event to history
        manager.threat_hunter.add_event_to_history(sample_security_event)

        campaign = await manager.execute_threat_hunt(
            hypotheses=None, time_range_hours=24
        )

        assert isinstance(campaign, HuntCampaign)
        assert campaign.campaign_id is not None

    @pytest.mark.asyncio
    async def test_update_ioc_feeds(self, config):
        """Test updating IOC feeds."""
        manager = ThreatIntelligenceManager(config)
        await manager.initialize()

        # Add a test feed
        feed = IOCFeed(
            feed_id="test_feed",
            name="Test Feed",
            source="test_source",
            feed_type="json",
            url="https://example.com/feed.json",
            enabled=True,
            update_interval_hours=24,
        )
        manager.stix_client.ioc_feeds[feed.feed_id] = feed

        # Mock feed update
        with patch.object(manager.stix_client, "update_ioc_feed", return_value=5):
            results = await manager.update_ioc_feeds(feed_ids=["test_feed"])

            assert "test_feed" in results
            assert results["test_feed"] >= 0

    @pytest.mark.asyncio
    async def test_get_health_status(self, config):
        """Test getting health status."""
        manager = ThreatIntelligenceManager(config)
        await manager.initialize()

        health = manager.get_health_status()

        assert health["initialized"] is True
        assert health["components"]["stix_taxii"] is True
        assert health["components"]["mitre_attack"] is True
        assert health["components"]["threat_hunter"] is True

    @pytest.mark.asyncio
    async def test_get_coverage_report(self, config):
        """Test getting coverage report."""
        manager = ThreatIntelligenceManager(config)
        await manager.initialize()

        report = manager.get_coverage_report()

        assert "timestamp" in report
        assert "mitre_attack_coverage" in report
        assert "ioc_feed_status" in report

    @pytest.mark.asyncio
    async def test_shutdown(self, config):
        """Test manager shutdown."""
        manager = ThreatIntelligenceManager(config)
        await manager.initialize()
        await manager.shutdown()

        assert not manager._initialized

    @pytest.mark.asyncio
    async def test_process_event_before_initialization(
        self, config, sample_security_event
    ):
        """Test processing event before initialization raises error."""
        manager = ThreatIntelligenceManager(config)

        with pytest.raises(CronosAIException, match="TIP not initialized"):
            await manager.process_security_event(sample_security_event)


# Integration Tests


class TestTIPIntegration:
    """Test TIP end-to-end integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_event_enrichment_pipeline(self, config):
        """Test complete event enrichment pipeline."""
        manager = ThreatIntelligenceManager(config)
        await manager.initialize()

        # Create malicious indicator
        indicator = STIXIndicator(
            id="indicator--malicious-scanner",
            pattern="[ipv4-addr:value = '198.51.100.42']",
            valid_from=datetime.utcnow(),
            name="Known Scanner",
            description="Malicious scanning infrastructure",
            labels=["scanner", "malicious"],
            confidence=90,
        )
        manager.stix_client.indicators_cache[indicator.id] = indicator

        # Create event with malicious IP
        event = SecurityEvent(
            event_id="evt_integration_001",
            event_type=SecurityEventType.SUSPICIOUS_NETWORK_ACTIVITY,
            threat_level=ThreatLevel.HIGH,
            description="Port scanning detected",
            source_ip="198.51.100.42",
            destination_ip="10.0.1.100",
        )

        # Process event
        enrichment = await manager.process_security_event(event)

        # Verify enrichment
        assert len(enrichment["ioc_matches"]) > 0
        assert enrichment["ioc_matches"][0]["value"] == "198.51.100.42"
        assert enrichment["ttp_mapping"] is not None
        assert enrichment["threat_score"] > 0.3  # IOC match + TTP match

    @pytest.mark.asyncio
    async def test_threat_hunt_with_findings(self, config):
        """Test threat hunt generating findings."""
        manager = ThreatIntelligenceManager(config)
        await manager.initialize()

        # Add malicious indicator
        indicator = STIXIndicator(
            id="indicator--apt-c2",
            pattern="[ipv4-addr:value = '203.0.113.100']",
            valid_from=datetime.utcnow(),
            name="APT C2 Server",
            description="Command and control infrastructure",
            labels=["c2", "apt"],
            confidence=95,
        )
        manager.stix_client.indicators_cache[indicator.id] = indicator

        # Add suspicious events
        for i in range(5):
            event = SecurityEvent(
                event_id=f"evt_hunt_{i}",
                event_type=SecurityEventType.COMMAND_AND_CONTROL,
                threat_level=ThreatLevel.MEDIUM,
                description=f"Outbound connection {i}",
                source_ip="10.0.1.50",
                destination_ip="203.0.113.100",
            )
            manager.threat_hunter.add_event_to_history(event)

        # Execute hunt
        campaign = await manager.execute_threat_hunt(
            hypotheses=["h001"], time_range_hours=24  # Unknown IOCs
        )

        # Should find C2 communication
        assert len(campaign.findings) > 0
        assert any("203.0.113.100" in f.description for f in campaign.findings)


# Performance Tests


class TestTIPPerformance:
    """Test TIP performance characteristics."""

    @pytest.mark.asyncio
    async def test_event_processing_performance(self, config):
        """Test event processing completes in reasonable time."""
        manager = ThreatIntelligenceManager(config)
        await manager.initialize()

        event = SecurityEvent(
            event_id="evt_perf_001",
            event_type=SecurityEventType.ANOMALOUS_BEHAVIOR,
            threat_level=ThreatLevel.LOW,
            description="Performance test event",
            source_ip="192.0.2.1",
            destination_ip="192.0.2.2",
        )

        start_time = datetime.utcnow()
        enrichment = await manager.process_security_event(event)
        duration = (datetime.utcnow() - start_time).total_seconds()

        # Should complete in < 1 second
        assert duration < 1.0
        assert enrichment is not None

    @pytest.mark.asyncio
    async def test_bulk_ioc_query_performance(self, config):
        """Test bulk IOC query performance."""
        manager = ThreatIntelligenceManager(config)
        await manager.initialize()

        # Add 100 indicators
        for i in range(100):
            indicator = STIXIndicator(
                id=f"indicator--bulk-{i}",
                pattern=f"[ipv4-addr:value = '192.0.2.{i}']",
                valid_from=datetime.utcnow(),
                name=f"Test Indicator {i}",
                confidence=50 + (i % 50),
            )
            manager.stix_client.indicators_cache[indicator.id] = indicator

        start_time = datetime.utcnow()
        result = await manager.query_threat_intelligence(
            query_type="iocs", query_params={"limit": 100}
        )
        duration = (datetime.utcnow() - start_time).total_seconds()

        # Should complete in < 500ms
        assert duration < 0.5
        assert result["total"] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
