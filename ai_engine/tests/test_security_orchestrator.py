"""
CRONOS AI - Zero-Touch Security Orchestrator Tests
Comprehensive test suite for security automation functionality.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from ai_engine.llm.security_orchestrator import (
    ZeroTouchSecurityOrchestrator,
    SecurityEvent,
    SecurityRequirements,
    ThreatData,
    ThreatType,
    ThreatSeverity,
    ResponseAction,
    IncidentStatus,
    SecurityException
)
from ai_engine.llm.unified_llm_service import LLMResponse
from ai_engine.core.config import Config


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = Mock(spec=Config)
    config.openai_api_key = "test-key"
    config.anthropic_api_key = "test-key"
    return config


@pytest.fixture
def mock_llm_service():
    """Create mock LLM service."""
    service = AsyncMock()
    service.process_request = AsyncMock(return_value=LLMResponse(
        content='{"threat_type": "malware", "confidence": 0.95, "risk_score": 85, "attack_vector": "phishing", "affected_assets": ["server1"], "indicators_of_compromise": ["malicious.exe"], "analysis_summary": "Malware detected", "recommended_actions": ["Block", "Quarantine"], "mitigation_strategies": ["Update antivirus"]}',
        provider="openai_gpt4",
        tokens_used=500,
        processing_time=1.5,
        confidence=0.9
    ))
    return service


@pytest.fixture
def mock_alert_manager():
    """Create mock alert manager."""
    manager = Mock()
    manager.active_alerts = {}
    return manager


@pytest.fixture
def mock_policy_engine():
    """Create mock policy engine."""
    return Mock()


@pytest.fixture
async def security_orchestrator(mock_config, mock_llm_service, mock_alert_manager, mock_policy_engine):
    """Create security orchestrator instance."""
    orchestrator = ZeroTouchSecurityOrchestrator(
        config=mock_config,
        llm_service=mock_llm_service,
        alert_manager=mock_alert_manager,
        policy_engine=mock_policy_engine
    )
    return orchestrator


@pytest.fixture
def sample_security_event():
    """Create sample security event."""
    return SecurityEvent(
        event_id="EVT-001",
        event_type=ThreatType.MALWARE,
        severity=ThreatSeverity.HIGH,
        timestamp=datetime.utcnow(),
        source_ip="192.168.1.100",
        destination_ip="10.0.0.50",
        user_id="user123",
        resource="/api/sensitive-data",
        description="Suspicious malware activity detected",
        indicators=["malicious.exe", "suspicious-hash-123"],
        raw_data={"process": "malicious.exe", "pid": 1234}
    )


@pytest.fixture
def sample_security_requirements():
    """Create sample security requirements."""
    return SecurityRequirements(
        requirement_id="REQ-001",
        framework="NIST",
        controls=["AC-1", "AC-2", "SC-7"],
        risk_level="high",
        compliance_requirements=["PCI-DSS", "HIPAA"],
        business_context={"industry": "healthcare", "data_sensitivity": "high"}
    )


@pytest.fixture
def sample_threat_data():
    """Create sample threat intelligence data."""
    return ThreatData(
        data_id="TI-001",
        source="threat-feed-alpha",
        threat_indicators=["192.168.1.100", "malicious-domain.com"],
        threat_actors=["APT28", "Lazarus Group"],
        attack_patterns=["spear-phishing", "credential-theft"],
        vulnerabilities=["CVE-2023-1234", "CVE-2023-5678"],
        timestamp=datetime.utcnow(),
        confidence=0.85
    )


class TestSecurityOrchestrator:
    """Test suite for ZeroTouchSecurityOrchestrator."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, security_orchestrator):
        """Test orchestrator initialization."""
        assert security_orchestrator is not None
        assert len(security_orchestrator.active_incidents) == 0
        assert len(security_orchestrator.response_playbooks) > 0
        assert security_orchestrator.stats['total_events'] == 0
    
    @pytest.mark.asyncio
    async def test_detect_and_respond_success(self, security_orchestrator, sample_security_event):
        """Test successful threat detection and response."""
        response = await security_orchestrator.detect_and_respond(sample_security_event)
        
        assert response is not None
        assert response.event_id == sample_security_event.event_id
        assert response.success is True
        assert len(response.actions_taken) > 0
        assert response.execution_time > 0
        
        # Verify statistics updated
        assert security_orchestrator.stats['total_events'] == 1
        assert security_orchestrator.stats['threats_detected'] == 1
        assert security_orchestrator.stats['automated_responses'] == 1
    
    @pytest.mark.asyncio
    async def test_detect_and_respond_critical_threat(self, security_orchestrator):
        """Test response to critical threat."""
        critical_event = SecurityEvent(
            event_id="EVT-CRITICAL",
            event_type=ThreatType.DATA_EXFILTRATION,
            severity=ThreatSeverity.CRITICAL,
            timestamp=datetime.utcnow(),
            source_ip="192.168.1.100",
            description="Critical data exfiltration detected"
        )
        
        response = await security_orchestrator.detect_and_respond(critical_event)
        
        assert response.success is True
        # Critical threats should trigger multiple actions
        assert ResponseAction.BLOCK in response.actions_taken or 'block' in [a.value for a in response.actions_taken]
        assert ResponseAction.ALERT in response.actions_taken or 'alert' in [a.value for a in response.actions_taken]
    
    @pytest.mark.asyncio
    async def test_threat_analysis_caching(self, security_orchestrator, sample_security_event):
        """Test threat analysis caching mechanism."""
        # First analysis
        response1 = await security_orchestrator.detect_and_respond(sample_security_event)
        
        # Second analysis with same event (should use cache)
        response2 = await security_orchestrator.detect_and_respond(sample_security_event)
        
        assert response1.event_id == response2.event_id
        # Cache should have entry
        assert len(security_orchestrator.threat_intelligence_cache) > 0
    
    @pytest.mark.asyncio
    async def test_generate_security_policies(self, security_orchestrator, sample_security_requirements, mock_llm_service):
        """Test security policy generation."""
        # Mock LLM response for policy generation
        mock_llm_service.process_request.return_value = LLMResponse(
            content='[{"name": "Access Control Policy", "description": "Control access to resources", "policy_type": "access_control", "rules": [{"condition": "user.role != admin", "action": "deny"}], "enforcement_level": "enforce", "scope": ["all"]}]',
            provider="openai_gpt4",
            tokens_used=800,
            processing_time=2.0,
            confidence=0.9
        )
        
        policies = await security_orchestrator.generate_security_policies(sample_security_requirements)
        
        assert len(policies) > 0
        assert policies[0].name is not None
        assert policies[0].policy_type is not None
        assert len(policies[0].rules) > 0
        assert policies[0].enforcement_level in ['monitor', 'warn', 'enforce']
        
        # Verify statistics updated
        assert security_orchestrator.stats['policies_generated'] > 0
    
    @pytest.mark.asyncio
    async def test_threat_intelligence_analysis(self, security_orchestrator, sample_threat_data, mock_llm_service):
        """Test threat intelligence analysis."""
        # Mock LLM response for threat analysis
        mock_llm_service.process_request.return_value = LLMResponse(
            content='{"threat_type": "intrusion", "severity": "high", "confidence": 0.9, "risk_score": 80, "attack_vector": "network", "affected_assets": ["server1", "server2"], "indicators_of_compromise": ["192.168.1.100"], "analysis_summary": "Advanced persistent threat detected", "recommended_actions": ["Block IPs", "Isolate systems"], "mitigation_strategies": ["Update firewall rules", "Patch vulnerabilities"]}',
            provider="anthropic_claude",
            tokens_used=1000,
            processing_time=2.5,
            confidence=0.95
        )
        
        analysis = await security_orchestrator.threat_intelligence_analysis(sample_threat_data)
        
        assert analysis is not None
        assert analysis.threat_id is not None
        assert analysis.threat_type in ThreatType
        assert analysis.severity in ThreatSeverity
        assert 0.0 <= analysis.confidence <= 1.0
        assert 0 <= analysis.risk_score <= 100
        assert len(analysis.recommended_actions) > 0
        assert len(analysis.mitigation_strategies) > 0
    
    @pytest.mark.asyncio
    async def test_assess_security_posture(self, security_orchestrator, sample_security_event):
        """Test security posture assessment."""
        # Create some incidents first
        await security_orchestrator.detect_and_respond(sample_security_event)
        
        assessment = await security_orchestrator.assess_security_posture()
        
        assert assessment is not None
        assert 'timestamp' in assessment
        assert 'overall_status' in assessment
        assert 'active_threats' in assessment
        assert 'severity_distribution' in assessment
        assert 'statistics' in assessment
        assert 'recommendations' in assessment
        
        # Verify statistics
        stats = assessment['statistics']
        assert 'total_events_processed' in stats
        assert 'threats_detected' in stats
        assert 'detection_accuracy' in stats
    
    @pytest.mark.asyncio
    async def test_incident_management(self, security_orchestrator, sample_security_event):
        """Test incident creation and management."""
        response = await security_orchestrator.detect_and_respond(sample_security_event)
        
        # Verify incident was created
        assert len(security_orchestrator.active_incidents) > 0
        
        # Get incident
        incident_id = list(security_orchestrator.active_incidents.keys())[0]
        incident = security_orchestrator.active_incidents[incident_id]
        
        assert incident.incident_id is not None
        assert incident.title is not None
        assert incident.severity in ThreatSeverity
        assert incident.status in IncidentStatus
        assert len(incident.events) > 0
        assert incident.analysis is not None
        assert incident.response is not None
    
    @pytest.mark.asyncio
    async def test_response_playbooks(self, security_orchestrator):
        """Test response playbook execution."""
        # Test different threat types
        threat_types = [
            ThreatType.MALWARE,
            ThreatType.INTRUSION,
            ThreatType.DATA_EXFILTRATION,
            ThreatType.DDoS
        ]
        
        for threat_type in threat_types:
            actions = security_orchestrator._get_playbook_actions(threat_type)
            assert len(actions) > 0
            assert all(isinstance(action, ResponseAction) for action in actions)
    
    @pytest.mark.asyncio
    async def test_alert_generation(self, security_orchestrator, sample_security_event, mock_alert_manager):
        """Test security alert generation."""
        # Set event to high severity to trigger alert
        sample_security_event.severity = ThreatSeverity.HIGH
        
        response = await security_orchestrator.detect_and_respond(sample_security_event)
        
        # Verify alert was generated for high severity
        assert len(response.alerts_generated) > 0 or len(mock_alert_manager.active_alerts) > 0
    
    @pytest.mark.asyncio
    async def test_statistics_tracking(self, security_orchestrator, sample_security_event):
        """Test statistics tracking."""
        initial_stats = security_orchestrator.get_statistics()
        
        # Process event
        await security_orchestrator.detect_and_respond(sample_security_event)
        
        updated_stats = security_orchestrator.get_statistics()
        
        assert updated_stats['total_events'] > initial_stats['total_events']
        assert updated_stats['threats_detected'] > initial_stats['threats_detected']
        assert updated_stats['automated_responses'] > initial_stats['automated_responses']
    
    @pytest.mark.asyncio
    async def test_error_handling(self, security_orchestrator, mock_llm_service):
        """Test error handling in threat detection."""
        # Mock LLM service to raise exception
        mock_llm_service.process_request.side_effect = Exception("LLM service error")
        
        event = SecurityEvent(
            event_id="EVT-ERROR",
            event_type=ThreatType.UNKNOWN,
            severity=ThreatSeverity.MEDIUM,
            timestamp=datetime.utcnow(),
            description="Test error handling"
        )
        
        response = await security_orchestrator.detect_and_respond(event)
        
        # Should return error response
        assert response.success is False
        assert 'error' in response.metadata or 'failed' in response.details.lower()
    
    @pytest.mark.asyncio
    async def test_policy_validation(self, security_orchestrator, sample_security_requirements):
        """Test policy validation."""
        from ai_engine.llm.security_orchestrator import SecurityPolicy
        
        # Create test policies
        policies = [
            SecurityPolicy(
                policy_id="POL-001",
                name="Test Policy",
                description="Test policy description",
                policy_type="security",
                rules=[{"condition": "test", "action": "allow"}],
                enforcement_level="enforce",
                scope=["all"]
            ),
            SecurityPolicy(
                policy_id="POL-002",
                name="",  # Invalid: empty name
                description="Invalid policy",
                policy_type="security",
                rules=[],  # Invalid: no rules
                enforcement_level="enforce",
                scope=["all"]
            )
        ]
        
        validated = await security_orchestrator._validate_policies(policies, sample_security_requirements)
        
        # Only valid policy should pass
        assert len(validated) == 1
        assert validated[0].policy_id == "POL-001"
    
    @pytest.mark.asyncio
    async def test_concurrent_event_processing(self, security_orchestrator):
        """Test concurrent processing of multiple events."""
        events = [
            SecurityEvent(
                event_id=f"EVT-{i}",
                event_type=ThreatType.MALWARE,
                severity=ThreatSeverity.MEDIUM,
                timestamp=datetime.utcnow(),
                description=f"Test event {i}"
            )
            for i in range(5)
        ]
        
        # Process events concurrently
        responses = await asyncio.gather(*[
            security_orchestrator.detect_and_respond(event)
            for event in events
        ])
        
        assert len(responses) == 5
        assert all(r.success for r in responses)
        assert security_orchestrator.stats['total_events'] == 5


class TestSecurityModels:
    """Test security data models."""
    
    def test_security_event_creation(self):
        """Test SecurityEvent creation."""
        event = SecurityEvent(
            event_id="EVT-001",
            event_type=ThreatType.INTRUSION,
            severity=ThreatSeverity.HIGH,
            timestamp=datetime.utcnow(),
            description="Test event"
        )
        
        assert event.event_id == "EVT-001"
        assert event.event_type == ThreatType.INTRUSION
        assert event.severity == ThreatSeverity.HIGH
    
    def test_security_requirements_creation(self):
        """Test SecurityRequirements creation."""
        requirements = SecurityRequirements(
            requirement_id="REQ-001",
            framework="NIST",
            controls=["AC-1"],
            risk_level="high",
            compliance_requirements=["PCI-DSS"]
        )
        
        assert requirements.requirement_id == "REQ-001"
        assert requirements.framework == "NIST"
        assert len(requirements.controls) == 1
    
    def test_threat_data_creation(self):
        """Test ThreatData creation."""
        threat_data = ThreatData(
            data_id="TI-001",
            source="test-source",
            threat_indicators=["indicator1"],
            threat_actors=["actor1"],
            attack_patterns=["pattern1"],
            vulnerabilities=["CVE-2023-1234"],
            timestamp=datetime.utcnow(),
            confidence=0.9
        )
        
        assert threat_data.data_id == "TI-001"
        assert threat_data.confidence == 0.9
        assert len(threat_data.threat_indicators) == 1


@pytest.mark.integration
class TestSecurityOrchestratorIntegration:
    """Integration tests for security orchestrator."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_threat_response(self, security_orchestrator, sample_security_event):
        """Test complete end-to-end threat response workflow."""
        # 1. Detect and respond
        response = await security_orchestrator.detect_and_respond(sample_security_event)
        assert response.success is True
        
        # 2. Verify incident created
        assert len(security_orchestrator.active_incidents) > 0
        
        # 3. Assess security posture
        assessment = await security_orchestrator.assess_security_posture()
        assert assessment['active_threats'] > 0
        
        # 4. Get statistics
        stats = security_orchestrator.get_statistics()
        assert stats['total_events'] > 0
        assert stats['threats_detected'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])