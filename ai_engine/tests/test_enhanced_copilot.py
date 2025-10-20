"""
CRONOS AI - Enhanced LLM Copilot Test Suite

Comprehensive tests for enhanced copilot capabilities including predictive
threat modeling, playbook generation, fuzzing, and code generation.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from ai_engine.copilot.predictive_threat_modeler import (
    PredictiveThreatModeler,
    ThreatScenario,
    ScenarioType,
    RiskImpact,
)
from ai_engine.copilot.playbook_generator import (
    PlaybookGenerator,
    PlaybookPhase,
    ActionPriority,
)
from ai_engine.copilot.protocol_fuzzer import (
    ProtocolFuzzer,
    MutationStrategy,
    VulnerabilityType,
)
from ai_engine.copilot.protocol_handler_generator import (
    ProtocolHandlerGenerator,
    ProtocolSpec,
    ProtocolMessage,
    ProtocolField,
    ProgrammingLanguage,
    ComponentType,
)
from ai_engine.security.models import SecurityEvent, SecurityEventType, ThreatLevel
from ai_engine.core.config import Config


# Fixtures


@pytest.fixture
def config():
    """Test configuration."""
    return Config()


@pytest.fixture
def mock_llm_service():
    """Mock LLM service."""
    llm_service = Mock()
    llm_service.query = AsyncMock()

    # Default response
    mock_response = Mock()
    mock_response.content = '{"threat_vectors": [], "confidence": 0.8}'
    mock_response.tokens_used = 100
    mock_response.processing_time = 0.5
    llm_service.query.return_value = mock_response

    return llm_service


# Predictive Threat Modeler Tests


class TestPredictiveThreatModeler:
    """Test suite for PredictiveThreatModeler."""

    @pytest.mark.asyncio
    async def test_model_port_change_scenario(self, config, mock_llm_service):
        """Test modeling a port change scenario."""
        modeler = PredictiveThreatModeler(config, llm_service=mock_llm_service)

        scenario = ThreatScenario(
            scenario_id="test_scenario_1",
            scenario_type=ScenarioType.PORT_CHANGE,
            description="Open port 443 for HTTPS traffic",
            proposed_change={"ports": [443]},
        )

        # Mock LLM response with proper JSON
        mock_llm_service.query.return_value.content = """
        {
          "threat_vectors": [
            {
              "name": "Exposed HTTPS Service",
              "description": "Opening port 443 exposes HTTPS service to internet",
              "likelihood": 0.6,
              "impact": "medium",
              "attack_techniques": ["T1046", "T1190"],
              "mitigation_recommendations": ["Use WAF", "Enable TLS 1.3"],
              "cve_references": []
            }
          ]
        }
        """

        model = await modeler.model_threat_scenario(scenario)

        assert model.scenario.scenario_id == "test_scenario_1"
        assert model.scenario.scenario_type == ScenarioType.PORT_CHANGE
        assert len(model.threat_vectors) > 0
        assert 0.0 <= model.overall_risk_score <= 10.0
        assert model.risk_impact in RiskImpact
        assert len(model.recommendations) > 0
        assert 0.0 <= model.confidence_score <= 1.0

    @pytest.mark.asyncio
    async def test_model_encryption_change_scenario(self, config, mock_llm_service):
        """Test modeling an encryption change scenario."""
        modeler = PredictiveThreatModeler(config, llm_service=mock_llm_service)

        scenario = ThreatScenario(
            scenario_id="test_scenario_2",
            scenario_type=ScenarioType.ENCRYPTION_CHANGE,
            description="Downgrade encryption to TLS 1.0",
            proposed_change={
                "encryption": {"algorithm": "TLS 1.0", "strength": "weaken"}
            },
        )

        mock_llm_service.query.return_value.content = """
        {
          "threat_vectors": [
            {
              "name": "Weak Encryption",
              "description": "TLS 1.0 is deprecated and vulnerable",
              "likelihood": 0.9,
              "impact": "critical",
              "attack_techniques": ["T1557", "T1040"],
              "mitigation_recommendations": ["Use TLS 1.3", "Disable TLS 1.0"],
              "cve_references": ["CVE-2014-3566"]
            }
          ]
        }
        """

        model = await modeler.model_threat_scenario(scenario)

        # Should identify high risk due to encryption weakening
        assert model.overall_risk_score > 5.0  # Should be high risk
        assert model.simulation_results.recommendation in ["REJECT", "MODIFY"]

    @pytest.mark.asyncio
    async def test_attack_surface_analysis(self, config, mock_llm_service):
        """Test attack surface change analysis."""
        modeler = PredictiveThreatModeler(config, llm_service=mock_llm_service)

        scenario = ThreatScenario(
            scenario_id="test_scenario_3",
            scenario_type=ScenarioType.PROTOCOL_CHANGE,
            description="Add new protocol handler",
            proposed_change={"protocols": ["SNMP", "Telnet"]},
        )

        attack_surface = await modeler._analyze_attack_surface_changes(scenario, None)

        assert "new_protocols" in attack_surface
        assert attack_surface["new_protocols"] == ["SNMP", "Telnet"]
        assert attack_surface["network_exposure_change"] > 0  # Increased exposure


# Playbook Generator Tests


class TestPlaybookGenerator:
    """Test suite for PlaybookGenerator."""

    @pytest.mark.asyncio
    async def test_generate_playbook_for_unauthorized_access(
        self, config, mock_llm_service
    ):
        """Test playbook generation for unauthorized access incident."""
        generator = PlaybookGenerator(config, llm_service=mock_llm_service)

        incident = SecurityEvent(
            event_id="incident_001",
            event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
            timestamp=datetime.utcnow(),
            source_ip="192.168.1.100",
            destination_ip="10.0.0.50",
            threat_level=ThreatLevel.HIGH,
            description="Unauthorized access attempt detected",
            metadata={"failed_login_attempts": 10},
        )

        # Mock LLM response
        mock_llm_service.query.return_value.content = """
        {
          "actions": [
            {
              "phase": "containment",
              "priority": "critical",
              "title": "Block Source IP",
              "description": "Immediately block the source IP address",
              "commands": ["iptables -A INPUT -s 192.168.1.100 -j DROP"],
              "validation_criteria": ["Verify IP is blocked"],
              "estimated_duration_minutes": 5
            },
            {
              "phase": "investigation",
              "priority": "high",
              "title": "Review Access Logs",
              "description": "Analyze access logs for attack pattern",
              "commands": ["grep 192.168.1.100 /var/log/auth.log"],
              "validation_criteria": ["Identify attack pattern"],
              "estimated_duration_minutes": 15
            }
          ],
          "success_criteria": ["Threat contained", "Access restored"],
          "escalation_triggers": ["Unable to contain within 2 hours"],
          "references": [],
          "confidence": 0.85
        }
        """

        playbook = await generator.generate_playbook(incident)

        assert playbook.playbook_id.startswith("playbook_")
        assert playbook.title
        assert len(playbook.actions) > 0
        assert len(playbook.success_criteria) > 0
        assert len(playbook.escalation_triggers) > 0

        # Verify actions are sorted by phase and priority
        containment_actions = playbook.get_actions_by_phase(PlaybookPhase.CONTAINMENT)
        assert len(containment_actions) > 0

        critical_actions = playbook.get_critical_actions()
        assert len(critical_actions) > 0

    @pytest.mark.asyncio
    async def test_playbook_classification(self, config, mock_llm_service):
        """Test incident classification logic."""
        generator = PlaybookGenerator(config, llm_service=mock_llm_service)

        # Test various event types
        access_event = SecurityEvent(
            event_id="test1",
            event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
            timestamp=datetime.utcnow(),
            source_ip="1.2.3.4",
            destination_ip="5.6.7.8",
            threat_level=ThreatLevel.MEDIUM,
            description="unauthorized access attempt",
        )

        incident_type = generator._classify_incident(access_event)
        assert incident_type == "unauthorized_access"

    @pytest.mark.asyncio
    async def test_playbook_to_dict(self, config, mock_llm_service):
        """Test playbook serialization."""
        generator = PlaybookGenerator(config, llm_service=mock_llm_service)

        incident = SecurityEvent(
            event_id="incident_002",
            event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
            timestamp=datetime.utcnow(),
            source_ip="1.1.1.1",
            destination_ip="2.2.2.2",
            threat_level=ThreatLevel.LOW,
            description="Test incident",
        )

        playbook = await generator.generate_playbook(incident)
        playbook_dict = playbook.to_dict()

        assert "playbook_id" in playbook_dict
        assert "actions" in playbook_dict
        assert isinstance(playbook_dict["actions"], list)
        assert "success_criteria" in playbook_dict


# Protocol Fuzzer Tests


class TestProtocolFuzzer:
    """Test suite for ProtocolFuzzer."""

    @pytest.mark.asyncio
    async def test_fuzzing_session(self, config, mock_llm_service):
        """Test complete fuzzing session."""
        fuzzer = ProtocolFuzzer(config, llm_service=mock_llm_service)

        protocol_spec = {"message_format": {"fields": ["length", "type", "data"]}}

        session = await fuzzer.start_fuzzing_session(
            protocol_name="TestProtocol",
            protocol_spec=protocol_spec,
            max_test_cases=100,
            duration_minutes=1,
        )

        assert session.session_id.startswith("fuzz_")
        assert session.protocol_name == "TestProtocol"
        assert session.test_cases_generated > 0
        assert session.status == "completed"
        assert "mutation_strategy_coverage" in session.coverage_metrics

    @pytest.mark.asyncio
    async def test_mutation_strategies(self, config, mock_llm_service):
        """Test different mutation strategies."""
        fuzzer = ProtocolFuzzer(config, llm_service=mock_llm_service)

        base_data = b"\x00\x01\x02\x03\x04\x05"

        # Test bit flip
        mutated, desc = fuzzer._mutate_bit_flip(base_data)
        assert len(mutated) == len(base_data)
        assert "Flipped" in desc

        # Test byte flip
        mutated, desc = fuzzer._mutate_byte_flip(base_data)
        assert len(mutated) == len(base_data)

        # Test boundary values
        mutated, desc = fuzzer._mutate_boundary_values(base_data)
        assert "boundary value" in desc.lower()

        # Test magic numbers
        mutated, desc = fuzzer._mutate_magic_numbers(base_data)
        assert "magic number" in desc.lower()

    @pytest.mark.asyncio
    async def test_vulnerability_detection(self, config, mock_llm_service):
        """Test vulnerability detection from crashes."""
        fuzzer = ProtocolFuzzer(config, llm_service=mock_llm_service)

        # Simulate finding vulnerabilities
        protocol_spec = {"message_format": {}}

        session = await fuzzer.start_fuzzing_session(
            protocol_name="VulnProtocol",
            protocol_spec=protocol_spec,
            max_test_cases=500,  # More tests = higher chance of simulated vuln
            duration_minutes=1,
        )

        # Should potentially find some vulnerabilities (simulated)
        assert isinstance(session.vulnerabilities_found, list)


# Protocol Handler Generator Tests


class TestProtocolHandlerGenerator:
    """Test suite for ProtocolHandlerGenerator."""

    @pytest.mark.asyncio
    async def test_generate_python_parser(self, config, mock_llm_service):
        """Test Python parser generation."""
        generator = ProtocolHandlerGenerator(config, llm_service=mock_llm_service)

        # Mock LLM response with Python code
        mock_llm_service.query.return_value.content = """
        {
          "source_code": "import struct\\n\\nclass MessageParser:\\n    def parse(self, data: bytes):\\n        length, msg_type = struct.unpack('!HB', data[:3])\\n        return {'length': length, 'type': msg_type}",
          "dependencies": ["struct"],
          "security_notes": ["Input validation implemented", "Bounds checking added"]
        }
        """

        protocol_spec = ProtocolSpec(
            protocol_name="SimpleProtocol",
            protocol_version="1.0",
            description="A simple test protocol",
            messages=[
                ProtocolMessage(
                    message_name="Hello",
                    message_type=1,
                    fields=[
                        ProtocolField(
                            name="length",
                            field_type="uint16",
                            length=2,
                            required=True,
                            description="Message length",
                        ),
                        ProtocolField(
                            name="type",
                            field_type="uint8",
                            length=1,
                            required=True,
                            description="Message type",
                        ),
                    ],
                    description="Hello message",
                )
            ],
        )

        artifact = await generator.generate_handler(
            protocol_spec=protocol_spec,
            language=ProgrammingLanguage.PYTHON,
            component_type=ComponentType.PARSER,
        )

        assert artifact.artifact_id.startswith("gen_")
        assert artifact.language == ProgrammingLanguage.PYTHON
        assert artifact.component_type == ComponentType.PARSER
        assert len(artifact.source_code) > 0
        assert artifact.file_name.endswith(".py")
        assert len(artifact.dependencies) > 0

    @pytest.mark.asyncio
    async def test_generate_rust_handler(self, config, mock_llm_service):
        """Test Rust handler generation."""
        generator = ProtocolHandlerGenerator(config, llm_service=mock_llm_service)

        mock_llm_service.query.return_value.content = """
        {
          "source_code": "pub struct MessageHandler;\\n\\nimpl MessageHandler {\\n    pub fn handle(&self, data: &[u8]) -> Result<(), Error> {\\n        Ok(())\\n    }\\n}",
          "dependencies": ["tokio", "bytes"],
          "security_notes": ["Safe Rust practices used"]
        }
        """

        protocol_spec = ProtocolSpec(
            protocol_name="TestProtocol",
            protocol_version="1.0",
            description="Test protocol",
            messages=[],
        )

        artifact = await generator.generate_handler(
            protocol_spec=protocol_spec,
            language=ProgrammingLanguage.RUST,
            component_type=ComponentType.HANDLER,
        )

        assert artifact.language == ProgrammingLanguage.RUST
        assert artifact.file_name.endswith(".rs")

    def test_file_name_generation(self, config, mock_llm_service):
        """Test file name generation for different languages."""
        generator = ProtocolHandlerGenerator(config, llm_service=mock_llm_service)

        # Test different languages
        py_file = generator._determine_file_name(
            "MyProtocol", ComponentType.PARSER, ProgrammingLanguage.PYTHON
        )
        assert py_file == "myprotocol_parser.py"

        rs_file = generator._determine_file_name(
            "MyProtocol", ComponentType.HANDLER, ProgrammingLanguage.RUST
        )
        assert rs_file == "myprotocol_handler.rs"

        go_file = generator._determine_file_name(
            "MyProtocol", ComponentType.VALIDATOR, ProgrammingLanguage.GO
        )
        assert go_file == "myprotocol_validator.go"


# Integration Tests


class TestEnhancedCopilotIntegration:
    """Integration tests for enhanced copilot features."""

    @pytest.mark.asyncio
    async def test_threat_model_to_playbook_workflow(self, config, mock_llm_service):
        """Test workflow from threat modeling to playbook generation."""
        # Step 1: Model a threat scenario
        modeler = PredictiveThreatModeler(config, llm_service=mock_llm_service)

        scenario = ThreatScenario(
            scenario_id="workflow_test",
            scenario_type=ScenarioType.ACCESS_CHANGE,
            description="Expand access to production database",
            proposed_change={"access_control": {"direction": "expand"}},
        )

        mock_llm_service.query.return_value.content = """
        {
          "threat_vectors": [
            {
              "name": "Unauthorized Access Risk",
              "description": "Expanded access increases risk",
              "likelihood": 0.7,
              "impact": "high",
              "attack_techniques": ["T1078"],
              "mitigation_recommendations": ["Implement MFA"],
              "cve_references": []
            }
          ]
        }
        """

        threat_model = await modeler.model_threat_scenario(scenario)

        # Verify high risk identified
        assert threat_model.overall_risk_score > 4.0

        # Step 2: If risk is high, generate incident response playbook
        if threat_model.risk_impact in [RiskImpact.HIGH, RiskImpact.CRITICAL]:
            generator = PlaybookGenerator(config, llm_service=mock_llm_service)

            incident = SecurityEvent(
                event_id="workflow_incident",
                event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
                timestamp=datetime.utcnow(),
                source_ip="internal",
                destination_ip="production_db",
                threat_level=ThreatLevel.HIGH,
                description="Unauthorized access to production database",
            )

            mock_llm_service.query.return_value.content = """
            {
              "actions": [
                {
                  "phase": "containment",
                  "priority": "critical",
                  "title": "Revoke Access",
                  "description": "Immediately revoke expanded access",
                  "commands": ["revoke_access production_db"],
                  "validation_criteria": ["Access revoked"],
                  "estimated_duration_minutes": 5
                }
              ],
              "success_criteria": ["Access secured"],
              "escalation_triggers": ["Access still active after 30 minutes"],
              "references": [],
              "confidence": 0.9
            }
            """

            playbook = await generator.generate_playbook(incident)

            assert len(playbook.actions) > 0
            assert any(
                action.priority == ActionPriority.CRITICAL
                for action in playbook.actions
            )


# Performance Tests


class TestEnhancedCopilotPerformance:
    """Performance tests for enhanced copilot."""

    @pytest.mark.asyncio
    async def test_threat_modeling_performance(self, config, mock_llm_service):
        """Test threat modeling completes in reasonable time."""
        modeler = PredictiveThreatModeler(config, llm_service=mock_llm_service)

        scenario = ThreatScenario(
            scenario_id="perf_test",
            scenario_type=ScenarioType.PORT_CHANGE,
            description="Performance test",
            proposed_change={"ports": [8080]},
        )

        import time

        start = time.time()
        await modeler.model_threat_scenario(scenario)
        duration = time.time() - start

        # Should complete within reasonable time
        assert duration < 5.0  # 5 seconds max (with mock LLM)

    @pytest.mark.asyncio
    async def test_playbook_generation_performance(self, config, mock_llm_service):
        """Test playbook generation performance."""
        generator = PlaybookGenerator(config, llm_service=mock_llm_service)

        incident = SecurityEvent(
            event_id="perf_test",
            event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
            timestamp=datetime.utcnow(),
            source_ip="1.1.1.1",
            destination_ip="2.2.2.2",
            threat_level=ThreatLevel.MEDIUM,
            description="Performance test",
        )

        import time

        start = time.time()
        await generator.generate_playbook(incident)
        duration = time.time() - start

        assert duration < 5.0  # 5 seconds max


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
