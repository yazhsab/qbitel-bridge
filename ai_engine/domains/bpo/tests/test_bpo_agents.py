"""
Tests for BPO specialized AI agents.

Tests cover:
- CallProcessingAgent (SIP/call handling, PROTOCOL_ANALYSIS + PROTOCOL_DISCOVERY)
- PCIComplianceAgent (DTMF masking, PCI checks)
- FraudDetectionAgent (toll fraud, premium rate detection)
- CRMScreenPopAgent (legacy CRM integration)
- RecordingEncryptionAgent (quantum-safe recording encryption)
- SessionSecurityAgent (behavioral modeling, anomaly detection)
- RemoteAccessAgent (VPN-less quantum-safe tunnels)
- ComplianceAuditAgent (audit trail, compliance reporting)
- BPOAgentCoordinator (orchestration, dependency graph, lifecycle)

Note: These tests define the expected behavior for the BPO agent modules
(ai_engine.domains.bpo.agents). The tests serve as a specification and
will pass once the modules are implemented.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from ai_engine.agents.base_agent import (
    AgentCapability,
    AgentConfig,
    AgentPriority,
    AgentTask,
    TaskResult,
)
from ai_engine.domains.bpo.agents.bpo_agents import (
    CallProcessingAgent,
    PCIComplianceAgent,
    FraudDetectionAgent,
    CRMScreenPopAgent,
    RecordingEncryptionAgent,
    SessionSecurityAgent,
    RemoteAccessAgent,
    ComplianceAuditAgent,
)
from ai_engine.domains.bpo.agents.agent_coordinator import BPOAgentCoordinator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def call_processing_agent():
    """Return a CallProcessingAgent instance."""
    return CallProcessingAgent()


@pytest.fixture
def pci_compliance_agent():
    """Return a PCIComplianceAgent instance."""
    return PCIComplianceAgent()


@pytest.fixture
def fraud_detection_agent():
    """Return a FraudDetectionAgent instance."""
    return FraudDetectionAgent()


@pytest.fixture
def crm_screen_pop_agent():
    """Return a CRMScreenPopAgent instance."""
    return CRMScreenPopAgent()


@pytest.fixture
def recording_encryption_agent():
    """Return a RecordingEncryptionAgent instance."""
    return RecordingEncryptionAgent()


@pytest.fixture
def session_security_agent():
    """Return a SessionSecurityAgent instance."""
    return SessionSecurityAgent()


@pytest.fixture
def remote_access_agent():
    """Return a RemoteAccessAgent instance."""
    return RemoteAccessAgent()


@pytest.fixture
def compliance_audit_agent():
    """Return a ComplianceAuditAgent instance."""
    return ComplianceAuditAgent()


@pytest.fixture
def coordinator():
    """Return a BPOAgentCoordinator instance."""
    return BPOAgentCoordinator()


@pytest.fixture
def sample_sip_task():
    """Return a sample SIP processing task."""
    return AgentTask(
        task_id="task-sip-001",
        task_type="process_sip_invite",
        payload={
            "raw_data": "INVITE sip:user@host SIP/2.0\r\n",
            "source_ip": "10.0.0.1",
            "source_port": 5060,
        },
    )


@pytest.fixture
def sample_pci_task():
    """Return a sample PCI compliance check task."""
    return AgentTask(
        task_id="task-pci-001",
        task_type="check_pci_compliance",
        payload={
            "call_id": "CALL-001",
            "agent_id": "AGENT-001",
            "channel": "voice",
        },
    )


@pytest.fixture
def sample_fraud_task():
    """Return a sample fraud detection task."""
    return AgentTask(
        task_id="task-fraud-001",
        task_type="detect_toll_fraud",
        payload={
            "destination_number": "+19001234567",
            "source_agent": "AGENT-001",
            "call_id": "CALL-002",
        },
    )


# ---------------------------------------------------------------------------
# CallProcessingAgent Tests
# ---------------------------------------------------------------------------

class TestCallProcessingAgent:
    """Tests for CallProcessingAgent."""

    def test_agent_type(self, call_processing_agent):
        """Test agent type string is correct."""
        assert call_processing_agent.agent_type == "call_processing", (
            "CallProcessingAgent should have agent_type='call_processing'"
        )

    def test_has_protocol_analysis_capability(self, call_processing_agent):
        """Test agent has PROTOCOL_ANALYSIS capability."""
        assert call_processing_agent.has_capability(AgentCapability.PROTOCOL_ANALYSIS), (
            "CallProcessingAgent should have PROTOCOL_ANALYSIS capability"
        )

    def test_has_protocol_discovery_capability(self, call_processing_agent):
        """Test agent has PROTOCOL_DISCOVERY capability."""
        assert call_processing_agent.has_capability(AgentCapability.PROTOCOL_DISCOVERY), (
            "CallProcessingAgent should have PROTOCOL_DISCOVERY capability"
        )

    def test_priority_is_critical(self, call_processing_agent):
        """Test agent priority is CRITICAL."""
        assert call_processing_agent.priority == AgentPriority.CRITICAL, (
            "CallProcessingAgent should have CRITICAL priority"
        )

    @pytest.mark.asyncio
    async def test_execute_process_sip_invite(self, call_processing_agent, sample_sip_task):
        """Test agent can process a SIP INVITE task."""
        result = await call_processing_agent.execute(sample_sip_task)
        assert result is not None, (
            "CallProcessingAgent should return a result for process_sip_invite"
        )

    @pytest.mark.asyncio
    async def test_execute_returns_call_metadata(self, call_processing_agent, sample_sip_task):
        """Test agent returns call metadata dict."""
        result = await call_processing_agent.execute(sample_sip_task)
        assert isinstance(result, dict), (
            "CallProcessingAgent should return a dict with call metadata"
        )

    @pytest.mark.asyncio
    async def test_execute_unknown_task_type(self, call_processing_agent):
        """Test agent returns error for unknown task type."""
        task = AgentTask(
            task_id="task-unknown-001",
            task_type="unknown_task_type",
            payload={},
        )
        result = await call_processing_agent.execute(task)
        assert result is not None
        # Should indicate error or unsupported task type
        if isinstance(result, dict):
            assert "error" in result or "unsupported" in str(result).lower()

    def test_config_capabilities(self, call_processing_agent):
        """Test default config has correct capabilities list."""
        capabilities = call_processing_agent.capabilities
        assert AgentCapability.PROTOCOL_ANALYSIS in capabilities
        assert AgentCapability.PROTOCOL_DISCOVERY in capabilities


# ---------------------------------------------------------------------------
# PCIComplianceAgent Tests
# ---------------------------------------------------------------------------

class TestPCIComplianceAgent:
    """Tests for PCIComplianceAgent."""

    def test_agent_type(self, pci_compliance_agent):
        """Test agent type string is correct."""
        assert pci_compliance_agent.agent_type == "pci_compliance", (
            "PCIComplianceAgent should have agent_type='pci_compliance'"
        )

    def test_has_compliance_check_capability(self, pci_compliance_agent):
        """Test agent has COMPLIANCE_CHECK capability."""
        assert pci_compliance_agent.has_capability(AgentCapability.COMPLIANCE_CHECK), (
            "PCIComplianceAgent should have COMPLIANCE_CHECK capability"
        )

    def test_has_policy_enforcement_capability(self, pci_compliance_agent):
        """Test agent has POLICY_ENFORCEMENT capability."""
        assert pci_compliance_agent.has_capability(AgentCapability.POLICY_ENFORCEMENT), (
            "PCIComplianceAgent should have POLICY_ENFORCEMENT capability"
        )

    def test_priority_is_critical(self, pci_compliance_agent):
        """Test agent priority is CRITICAL."""
        assert pci_compliance_agent.priority == AgentPriority.CRITICAL, (
            "PCIComplianceAgent should have CRITICAL priority"
        )

    @pytest.mark.asyncio
    async def test_execute_check_pci_compliance(self, pci_compliance_agent, sample_pci_task):
        """Test agent can process a PCI compliance check task."""
        result = await pci_compliance_agent.execute(sample_pci_task)
        assert result is not None

    @pytest.mark.asyncio
    async def test_execute_enforce_dtmf_masking(self, pci_compliance_agent):
        """Test agent can enforce DTMF masking."""
        task = AgentTask(
            task_id="task-dtmf-001",
            task_type="enforce_dtmf_masking",
            payload={
                "call_id": "CALL-001",
                "agent_id": "AGENT-001",
                "masking_mode": "CLAMP",
            },
        )
        result = await pci_compliance_agent.execute(task)
        assert result is not None

    @pytest.mark.asyncio
    async def test_execute_unknown_task_type(self, pci_compliance_agent):
        """Test agent returns error for unknown task type."""
        task = AgentTask(
            task_id="task-unknown-002",
            task_type="unknown_pci_task",
            payload={},
        )
        result = await pci_compliance_agent.execute(task)
        assert result is not None
        if isinstance(result, dict):
            assert "error" in result or "unsupported" in str(result).lower()


# ---------------------------------------------------------------------------
# FraudDetectionAgent Tests
# ---------------------------------------------------------------------------

class TestFraudDetectionAgent:
    """Tests for FraudDetectionAgent."""

    def test_agent_type(self, fraud_detection_agent):
        """Test agent type string is correct."""
        assert fraud_detection_agent.agent_type == "fraud_detection", (
            "FraudDetectionAgent should have agent_type='fraud_detection'"
        )

    def test_has_anomaly_detection_capability(self, fraud_detection_agent):
        """Test agent has ANOMALY_DETECTION capability."""
        assert fraud_detection_agent.has_capability(AgentCapability.ANOMALY_DETECTION), (
            "FraudDetectionAgent should have ANOMALY_DETECTION capability"
        )

    def test_has_threat_analysis_capability(self, fraud_detection_agent):
        """Test agent has THREAT_ANALYSIS capability."""
        assert fraud_detection_agent.has_capability(AgentCapability.THREAT_ANALYSIS), (
            "FraudDetectionAgent should have THREAT_ANALYSIS capability"
        )

    def test_has_threat_mitigation_capability(self, fraud_detection_agent):
        """Test agent has THREAT_MITIGATION capability."""
        assert fraud_detection_agent.has_capability(AgentCapability.THREAT_MITIGATION), (
            "FraudDetectionAgent should have THREAT_MITIGATION capability"
        )

    def test_priority_is_critical(self, fraud_detection_agent):
        """Test agent priority is CRITICAL."""
        assert fraud_detection_agent.priority == AgentPriority.CRITICAL, (
            "FraudDetectionAgent should have CRITICAL priority"
        )

    @pytest.mark.asyncio
    async def test_execute_detect_toll_fraud(self, fraud_detection_agent, sample_fraud_task):
        """Test agent can detect toll fraud."""
        result = await fraud_detection_agent.execute(sample_fraud_task)
        assert result is not None

    @pytest.mark.asyncio
    async def test_execute_check_premium_rate(self, fraud_detection_agent):
        """Test agent can check premium rate numbers."""
        task = AgentTask(
            task_id="task-premium-001",
            task_type="check_premium_rate",
            payload={
                "destination_number": "+19001234567",
                "call_id": "CALL-003",
            },
        )
        result = await fraud_detection_agent.execute(task)
        assert result is not None

    @pytest.mark.asyncio
    async def test_execute_unknown_task_type(self, fraud_detection_agent):
        """Test agent returns error for unknown task type."""
        task = AgentTask(
            task_id="task-unknown-003",
            task_type="unknown_fraud_task",
            payload={},
        )
        result = await fraud_detection_agent.execute(task)
        assert result is not None
        if isinstance(result, dict):
            assert "error" in result or "unsupported" in str(result).lower()


# ---------------------------------------------------------------------------
# CRMScreenPopAgent Tests
# ---------------------------------------------------------------------------

class TestCRMScreenPopAgent:
    """Tests for CRMScreenPopAgent."""

    def test_agent_type(self, crm_screen_pop_agent):
        """Test agent type string is correct."""
        assert crm_screen_pop_agent.agent_type == "crm_screen_pop", (
            "CRMScreenPopAgent should have agent_type='crm_screen_pop'"
        )

    def test_has_legacy_system_analysis_capability(self, crm_screen_pop_agent):
        """Test agent has LEGACY_SYSTEM_ANALYSIS capability."""
        assert crm_screen_pop_agent.has_capability(AgentCapability.LEGACY_SYSTEM_ANALYSIS), (
            "CRMScreenPopAgent should have LEGACY_SYSTEM_ANALYSIS capability"
        )

    def test_priority_is_high(self, crm_screen_pop_agent):
        """Test agent priority is HIGH."""
        assert crm_screen_pop_agent.priority == AgentPriority.HIGH, (
            "CRMScreenPopAgent should have HIGH priority"
        )

    @pytest.mark.asyncio
    async def test_execute_screen_pop_lookup(self, crm_screen_pop_agent):
        """Test agent can perform a screen pop lookup."""
        task = AgentTask(
            task_id="task-crm-001",
            task_type="screen_pop_lookup",
            payload={
                "caller_number": "+12125551234",
                "call_id": "CALL-004",
                "agent_id": "AGENT-001",
            },
        )
        result = await crm_screen_pop_agent.execute(task)
        assert result is not None

    @pytest.mark.asyncio
    async def test_execute_unknown_task_type(self, crm_screen_pop_agent):
        """Test agent returns error for unknown task type."""
        task = AgentTask(
            task_id="task-unknown-004",
            task_type="unknown_crm_task",
            payload={},
        )
        result = await crm_screen_pop_agent.execute(task)
        assert result is not None
        if isinstance(result, dict):
            assert "error" in result or "unsupported" in str(result).lower()


# ---------------------------------------------------------------------------
# RecordingEncryptionAgent Tests
# ---------------------------------------------------------------------------

class TestRecordingEncryptionAgent:
    """Tests for RecordingEncryptionAgent."""

    def test_agent_type(self, recording_encryption_agent):
        """Test agent type string is correct."""
        assert recording_encryption_agent.agent_type == "recording_encryption", (
            "RecordingEncryptionAgent should have agent_type='recording_encryption'"
        )

    def test_has_quantum_cryptography_capability(self, recording_encryption_agent):
        """Test agent has QUANTUM_CRYPTOGRAPHY capability."""
        assert recording_encryption_agent.has_capability(AgentCapability.QUANTUM_CRYPTOGRAPHY), (
            "RecordingEncryptionAgent should have QUANTUM_CRYPTOGRAPHY capability"
        )

    def test_has_compliance_check_capability(self, recording_encryption_agent):
        """Test agent has COMPLIANCE_CHECK capability."""
        assert recording_encryption_agent.has_capability(AgentCapability.COMPLIANCE_CHECK), (
            "RecordingEncryptionAgent should have COMPLIANCE_CHECK capability"
        )

    def test_priority_is_high(self, recording_encryption_agent):
        """Test agent priority is HIGH."""
        assert recording_encryption_agent.priority == AgentPriority.HIGH, (
            "RecordingEncryptionAgent should have HIGH priority"
        )

    @pytest.mark.asyncio
    async def test_execute_encrypt_recording(self, recording_encryption_agent):
        """Test agent can encrypt a call recording."""
        task = AgentTask(
            task_id="task-encrypt-001",
            task_type="encrypt_recording",
            payload={
                "call_id": "CALL-005",
                "recording_path": "/recordings/CALL-005.wav",
                "encryption_algorithm": "ML-KEM-1024",
            },
        )
        result = await recording_encryption_agent.execute(task)
        assert result is not None

    @pytest.mark.asyncio
    async def test_execute_unknown_task_type(self, recording_encryption_agent):
        """Test agent returns error for unknown task type."""
        task = AgentTask(
            task_id="task-unknown-005",
            task_type="unknown_recording_task",
            payload={},
        )
        result = await recording_encryption_agent.execute(task)
        assert result is not None
        if isinstance(result, dict):
            assert "error" in result or "unsupported" in str(result).lower()


# ---------------------------------------------------------------------------
# SessionSecurityAgent Tests
# ---------------------------------------------------------------------------

class TestSessionSecurityAgent:
    """Tests for SessionSecurityAgent."""

    def test_agent_type(self, session_security_agent):
        """Test agent type string is correct."""
        assert session_security_agent.agent_type == "session_security", (
            "SessionSecurityAgent should have agent_type='session_security'"
        )

    def test_has_behavioral_modeling_capability(self, session_security_agent):
        """Test agent has BEHAVIORAL_MODELING capability."""
        assert session_security_agent.has_capability(AgentCapability.BEHAVIORAL_MODELING), (
            "SessionSecurityAgent should have BEHAVIORAL_MODELING capability"
        )

    def test_has_anomaly_detection_capability(self, session_security_agent):
        """Test agent has ANOMALY_DETECTION capability."""
        assert session_security_agent.has_capability(AgentCapability.ANOMALY_DETECTION), (
            "SessionSecurityAgent should have ANOMALY_DETECTION capability"
        )

    def test_priority_is_high(self, session_security_agent):
        """Test agent priority is HIGH."""
        assert session_security_agent.priority == AgentPriority.HIGH, (
            "SessionSecurityAgent should have HIGH priority"
        )

    @pytest.mark.asyncio
    async def test_execute_unknown_task_type(self, session_security_agent):
        """Test agent returns error for unknown task type."""
        task = AgentTask(
            task_id="task-unknown-006",
            task_type="unknown_session_task",
            payload={},
        )
        result = await session_security_agent.execute(task)
        assert result is not None
        if isinstance(result, dict):
            assert "error" in result or "unsupported" in str(result).lower()


# ---------------------------------------------------------------------------
# RemoteAccessAgent Tests
# ---------------------------------------------------------------------------

class TestRemoteAccessAgent:
    """Tests for RemoteAccessAgent."""

    def test_agent_type(self, remote_access_agent):
        """Test agent type string is correct."""
        assert remote_access_agent.agent_type == "remote_access", (
            "RemoteAccessAgent should have agent_type='remote_access'"
        )

    def test_has_quantum_cryptography_capability(self, remote_access_agent):
        """Test agent has QUANTUM_CRYPTOGRAPHY capability."""
        assert remote_access_agent.has_capability(AgentCapability.QUANTUM_CRYPTOGRAPHY), (
            "RemoteAccessAgent should have QUANTUM_CRYPTOGRAPHY capability"
        )

    def test_has_threat_analysis_capability(self, remote_access_agent):
        """Test agent has THREAT_ANALYSIS capability."""
        assert remote_access_agent.has_capability(AgentCapability.THREAT_ANALYSIS), (
            "RemoteAccessAgent should have THREAT_ANALYSIS capability"
        )

    def test_has_policy_enforcement_capability(self, remote_access_agent):
        """Test agent has POLICY_ENFORCEMENT capability."""
        assert remote_access_agent.has_capability(AgentCapability.POLICY_ENFORCEMENT), (
            "RemoteAccessAgent should have POLICY_ENFORCEMENT capability"
        )

    def test_priority_is_high(self, remote_access_agent):
        """Test agent priority is HIGH."""
        assert remote_access_agent.priority == AgentPriority.HIGH, (
            "RemoteAccessAgent should have HIGH priority"
        )

    @pytest.mark.asyncio
    async def test_execute_unknown_task_type(self, remote_access_agent):
        """Test agent returns error for unknown task type."""
        task = AgentTask(
            task_id="task-unknown-007",
            task_type="unknown_remote_task",
            payload={},
        )
        result = await remote_access_agent.execute(task)
        assert result is not None
        if isinstance(result, dict):
            assert "error" in result or "unsupported" in str(result).lower()


# ---------------------------------------------------------------------------
# ComplianceAuditAgent Tests
# ---------------------------------------------------------------------------

class TestComplianceAuditAgent:
    """Tests for ComplianceAuditAgent."""

    def test_agent_type(self, compliance_audit_agent):
        """Test agent type string is correct."""
        assert compliance_audit_agent.agent_type == "compliance_audit", (
            "ComplianceAuditAgent should have agent_type='compliance_audit'"
        )

    def test_has_compliance_check_capability(self, compliance_audit_agent):
        """Test agent has COMPLIANCE_CHECK capability."""
        assert compliance_audit_agent.has_capability(AgentCapability.COMPLIANCE_CHECK), (
            "ComplianceAuditAgent should have COMPLIANCE_CHECK capability"
        )

    def test_has_report_generation_capability(self, compliance_audit_agent):
        """Test agent has REPORT_GENERATION capability."""
        assert compliance_audit_agent.has_capability(AgentCapability.REPORT_GENERATION), (
            "ComplianceAuditAgent should have REPORT_GENERATION capability"
        )

    def test_priority_is_normal(self, compliance_audit_agent):
        """Test agent priority is NORMAL."""
        assert compliance_audit_agent.priority == AgentPriority.NORMAL, (
            "ComplianceAuditAgent should have NORMAL priority"
        )

    @pytest.mark.asyncio
    async def test_execute_unknown_task_type(self, compliance_audit_agent):
        """Test agent returns error for unknown task type."""
        task = AgentTask(
            task_id="task-unknown-008",
            task_type="unknown_audit_task",
            payload={},
        )
        result = await compliance_audit_agent.execute(task)
        assert result is not None
        if isinstance(result, dict):
            assert "error" in result or "unsupported" in str(result).lower()


# ---------------------------------------------------------------------------
# BPOAgentCoordinator Tests
# ---------------------------------------------------------------------------

class TestBPOAgentCoordinator:
    """Tests for BPOAgentCoordinator."""

    def test_coordinator_creation(self, coordinator):
        """Test coordinator can be instantiated."""
        assert coordinator is not None
        assert isinstance(coordinator, BPOAgentCoordinator)

    def test_coordinator_creates_all_agents(self, coordinator):
        """Test coordinator creates all 8 BPO agents."""
        agents = coordinator.get_agents()
        assert len(agents) == 8, (
            f"Coordinator should create exactly 8 agents, got {len(agents)}"
        )

    def test_coordinator_has_call_processing_agent(self, coordinator):
        """Test coordinator includes CallProcessingAgent."""
        agents = coordinator.get_agents()
        agent_types = [a.agent_type for a in agents]
        assert "call_processing" in agent_types, (
            "Coordinator should include a CallProcessingAgent"
        )

    def test_coordinator_has_pci_compliance_agent(self, coordinator):
        """Test coordinator includes PCIComplianceAgent."""
        agents = coordinator.get_agents()
        agent_types = [a.agent_type for a in agents]
        assert "pci_compliance" in agent_types, (
            "Coordinator should include a PCIComplianceAgent"
        )

    def test_coordinator_has_fraud_detection_agent(self, coordinator):
        """Test coordinator includes FraudDetectionAgent."""
        agents = coordinator.get_agents()
        agent_types = [a.agent_type for a in agents]
        assert "fraud_detection" in agent_types, (
            "Coordinator should include a FraudDetectionAgent"
        )

    def test_coordinator_has_crm_screen_pop_agent(self, coordinator):
        """Test coordinator includes CRMScreenPopAgent."""
        agents = coordinator.get_agents()
        agent_types = [a.agent_type for a in agents]
        assert "crm_screen_pop" in agent_types, (
            "Coordinator should include a CRMScreenPopAgent"
        )

    def test_coordinator_has_recording_encryption_agent(self, coordinator):
        """Test coordinator includes RecordingEncryptionAgent."""
        agents = coordinator.get_agents()
        agent_types = [a.agent_type for a in agents]
        assert "recording_encryption" in agent_types, (
            "Coordinator should include a RecordingEncryptionAgent"
        )

    def test_coordinator_has_session_security_agent(self, coordinator):
        """Test coordinator includes SessionSecurityAgent."""
        agents = coordinator.get_agents()
        agent_types = [a.agent_type for a in agents]
        assert "session_security" in agent_types, (
            "Coordinator should include a SessionSecurityAgent"
        )

    def test_coordinator_has_remote_access_agent(self, coordinator):
        """Test coordinator includes RemoteAccessAgent."""
        agents = coordinator.get_agents()
        agent_types = [a.agent_type for a in agents]
        assert "remote_access" in agent_types, (
            "Coordinator should include a RemoteAccessAgent"
        )

    def test_coordinator_has_compliance_audit_agent(self, coordinator):
        """Test coordinator includes ComplianceAuditAgent."""
        agents = coordinator.get_agents()
        agent_types = [a.agent_type for a in agents]
        assert "compliance_audit" in agent_types, (
            "Coordinator should include a ComplianceAuditAgent"
        )

    @pytest.mark.asyncio
    async def test_process_incoming_call(self, coordinator):
        """Test coordinator orchestrates agents for incoming call."""
        call_data = {
            "call_id": "CALL-COORD-001",
            "caller_number": "+12125551234",
            "destination_number": "+14155559876",
            "agent_id": "AGENT-001",
            "raw_sip": "INVITE sip:user@host SIP/2.0\r\n",
        }
        result = await coordinator.process_incoming_call(call_data)
        assert result is not None, (
            "Coordinator should return a result for incoming call processing"
        )

    @pytest.mark.asyncio
    async def test_process_incoming_call_returns_agent_results(self, coordinator):
        """Test coordinator returns results from multiple agents."""
        call_data = {
            "call_id": "CALL-COORD-002",
            "caller_number": "+12125551234",
            "destination_number": "+14155559876",
            "agent_id": "AGENT-001",
        }
        result = await coordinator.process_incoming_call(call_data)
        assert isinstance(result, dict), (
            "Coordinator result should be a dict with agent results"
        )

    def test_dependency_graph_call_processing_first(self, coordinator):
        """Test dependency graph: CallProcessing runs first."""
        dep_graph = coordinator.get_dependency_graph()
        assert dep_graph is not None
        # CallProcessing should have no dependencies (runs first)
        call_processing_deps = dep_graph.get("call_processing", [])
        assert len(call_processing_deps) == 0, (
            "CallProcessingAgent should have no dependencies (runs first)"
        )

    def test_dependency_graph_parallel_after_call_processing(self, coordinator):
        """Test dependency graph: PCI, Fraud, CRM run after CallProcessing."""
        dep_graph = coordinator.get_dependency_graph()
        # PCI, Fraud, and CRM should depend on CallProcessing
        pci_deps = dep_graph.get("pci_compliance", [])
        fraud_deps = dep_graph.get("fraud_detection", [])
        crm_deps = dep_graph.get("crm_screen_pop", [])
        assert "call_processing" in pci_deps, (
            "PCIComplianceAgent should depend on CallProcessingAgent"
        )
        assert "call_processing" in fraud_deps, (
            "FraudDetectionAgent should depend on CallProcessingAgent"
        )
        assert "call_processing" in crm_deps, (
            "CRMScreenPopAgent should depend on CallProcessingAgent"
        )

    def test_agent_status_reporting(self, coordinator):
        """Test coordinator can report status of all agents."""
        status = coordinator.get_status()
        assert status is not None
        assert isinstance(status, dict)
        assert "agents" in status
        assert len(status["agents"]) == 8, (
            "Status should include all 8 agents"
        )

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, coordinator):
        """Test graceful shutdown stops all agents."""
        await coordinator.shutdown()
        agents = coordinator.get_agents()
        for agent in agents:
            assert agent.status.value in ("terminated", "shutting_down", "initializing"), (
                f"Agent {agent.agent_type} should be terminated or shutting down after shutdown"
            )

    @pytest.mark.parametrize("agent_type,expected_capabilities", [
        ("call_processing", {AgentCapability.PROTOCOL_ANALYSIS, AgentCapability.PROTOCOL_DISCOVERY}),
        ("pci_compliance", {AgentCapability.COMPLIANCE_CHECK, AgentCapability.POLICY_ENFORCEMENT}),
        ("fraud_detection", {AgentCapability.ANOMALY_DETECTION, AgentCapability.THREAT_ANALYSIS, AgentCapability.THREAT_MITIGATION}),
        ("crm_screen_pop", {AgentCapability.LEGACY_SYSTEM_ANALYSIS}),
        ("recording_encryption", {AgentCapability.QUANTUM_CRYPTOGRAPHY, AgentCapability.COMPLIANCE_CHECK}),
        ("session_security", {AgentCapability.BEHAVIORAL_MODELING, AgentCapability.ANOMALY_DETECTION}),
        ("remote_access", {AgentCapability.QUANTUM_CRYPTOGRAPHY, AgentCapability.THREAT_ANALYSIS, AgentCapability.POLICY_ENFORCEMENT}),
        ("compliance_audit", {AgentCapability.COMPLIANCE_CHECK, AgentCapability.REPORT_GENERATION}),
    ])
    def test_agent_capabilities_from_coordinator(self, coordinator, agent_type, expected_capabilities):
        """Test each agent in the coordinator has expected capabilities."""
        agents = coordinator.get_agents()
        agent = next((a for a in agents if a.agent_type == agent_type), None)
        assert agent is not None, f"Agent {agent_type} not found in coordinator"
        for cap in expected_capabilities:
            assert agent.has_capability(cap), (
                f"Agent {agent_type} should have capability {cap.value}"
            )

    @pytest.mark.parametrize("agent_type,expected_priority", [
        ("call_processing", AgentPriority.CRITICAL),
        ("pci_compliance", AgentPriority.CRITICAL),
        ("fraud_detection", AgentPriority.CRITICAL),
        ("crm_screen_pop", AgentPriority.HIGH),
        ("recording_encryption", AgentPriority.HIGH),
        ("session_security", AgentPriority.HIGH),
        ("remote_access", AgentPriority.HIGH),
        ("compliance_audit", AgentPriority.NORMAL),
    ])
    def test_agent_priorities_from_coordinator(self, coordinator, agent_type, expected_priority):
        """Test each agent in the coordinator has expected priority."""
        agents = coordinator.get_agents()
        agent = next((a for a in agents if a.agent_type == agent_type), None)
        assert agent is not None, f"Agent {agent_type} not found in coordinator"
        assert agent.priority == expected_priority, (
            f"Agent {agent_type} should have priority {expected_priority.name}, "
            f"got {agent.priority.name}"
        )
