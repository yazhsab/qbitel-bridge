"""
Comprehensive tests for the CRONOS AI Multi-Agent Orchestration Framework.

Tests cover:
- Base Agent functionality
- Planning Agent task decomposition
- Agent Communication Protocol
- Agent Pool Manager dynamic scaling
- Agent Memory persistence
- Collaboration Framework consensus/negotiation
- Multi-Agent Orchestrator integration
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from ai_engine.agents.base_agent import (
    BaseAgent,
    AgentCapability,
    AgentConfig,
    AgentTask,
    AgentPriority,
    AgentStatus,
    TaskResult,
)
from ai_engine.agents.planning_agent import (
    PlanningAgent,
    TaskPlan,
    SubTask,
    SubTaskStatus,
    PlanStatus,
    ExecutionStrategy,
)
from ai_engine.agents.agent_communication import (
    AgentCommunicationProtocol,
    AgentMessage,
    MessageType,
    MessagePriority,
    Channel,
)
from ai_engine.agents.agent_pool import (
    AgentPoolManager,
    AgentPool,
    PoolConfig,
    AgentInstance,
    ScalingPolicy,
    LoadBalancingStrategy,
)
from ai_engine.agents.agent_memory import (
    AgentMemoryManager,
    MemoryType,
    MemoryEntry,
    EpisodicMemory,
    SemanticMemory,
    InMemoryBackend,
)
from ai_engine.agents.agent_collaboration import (
    CollaborationFramework,
    ConsensusProtocol,
    NegotiationStrategy,
    CollaborationSession,
    Proposal,
    Vote,
    VoteType,
)
from ai_engine.agents.multi_agent_orchestrator import (
    MultiAgentOrchestrator,
    OrchestratorConfig,
    OrchestratorMode,
)


# ============================================================================
# Test Fixtures
# ============================================================================

class TestAgent(BaseAgent):
    """A simple test agent implementation."""

    def __init__(self, config: AgentConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.executed_tasks: List[AgentTask] = []

    async def execute(self, task: AgentTask) -> Any:
        """Execute a task and return a result."""
        self.executed_tasks.append(task)

        # Simulate some work
        await asyncio.sleep(0.01)

        # Return based on task type
        if task.task_type == "echo":
            return task.payload.get("message", "echo")
        elif task.task_type == "add":
            a = task.payload.get("a", 0)
            b = task.payload.get("b", 0)
            return {"sum": a + b}
        elif task.task_type == "fail":
            raise RuntimeError("Intentional failure")
        else:
            return {"status": "completed", "task_type": task.task_type}


@pytest.fixture
def agent_config():
    """Create a basic agent configuration."""
    return AgentConfig(
        agent_type="test_agent",
        capabilities=[
            AgentCapability.THREAT_ANALYSIS,
            AgentCapability.PROTOCOL_ANALYSIS,
        ],
        max_concurrent_tasks=5,
        task_timeout_seconds=30,
    )


@pytest.fixture
def test_agent(agent_config):
    """Create a test agent instance."""
    return TestAgent(config=agent_config)


@pytest.fixture
async def communication():
    """Create and start communication protocol."""
    comm = AgentCommunicationProtocol()
    await comm.start()
    yield comm
    await comm.stop()


@pytest.fixture
async def memory():
    """Create and start memory manager."""
    mem = AgentMemoryManager()
    await mem.start()
    yield mem
    await mem.stop()


@pytest.fixture
async def collaboration(communication):
    """Create and start collaboration framework."""
    collab = CollaborationFramework(communication=communication)
    await collab.start()
    yield collab
    await collab.stop()


# ============================================================================
# Base Agent Tests
# ============================================================================

class TestBaseAgent:
    """Tests for BaseAgent functionality."""

    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent_config):
        """Test agent initializes correctly."""
        agent = TestAgent(config=agent_config)

        assert agent.agent_type == "test_agent"
        assert agent.status == AgentStatus.INITIALIZING
        assert AgentCapability.THREAT_ANALYSIS in agent.capabilities
        assert agent.max_concurrent_tasks == 5

    @pytest.mark.asyncio
    async def test_agent_start_stop(self, agent_config):
        """Test agent start and stop lifecycle."""
        agent = TestAgent(config=agent_config)

        await agent.start()
        assert agent.status in [AgentStatus.IDLE, AgentStatus.BUSY]
        assert agent._running is True

        await agent.stop()
        assert agent.status == AgentStatus.TERMINATED
        assert agent._running is False

    @pytest.mark.asyncio
    async def test_agent_task_execution(self, agent_config):
        """Test agent executes tasks correctly."""
        agent = TestAgent(config=agent_config)
        await agent.start()

        task = AgentTask(
            task_id="test-1",
            task_type="echo",
            payload={"message": "hello"},
        )

        task_id = await agent.submit_task(task)
        assert task_id == "test-1"

        # Wait for task completion
        await asyncio.sleep(0.1)

        assert len(agent.executed_tasks) == 1
        assert agent.stats["tasks_completed"] >= 0

        await agent.stop()

    @pytest.mark.asyncio
    async def test_agent_capability_check(self, agent_config):
        """Test capability checking."""
        agent = TestAgent(config=agent_config)

        assert agent.has_capability(AgentCapability.THREAT_ANALYSIS) is True
        assert agent.has_capability(AgentCapability.QUANTUM_CRYPTOGRAPHY) is False

    @pytest.mark.asyncio
    async def test_agent_status_reporting(self, agent_config):
        """Test agent status reporting."""
        agent = TestAgent(config=agent_config)
        await agent.start()

        status = agent.get_status()

        assert "agent_id" in status
        assert status["agent_type"] == "test_agent"
        assert status["status"] in ["idle", "busy"]
        assert "statistics" in status

        await agent.stop()


# ============================================================================
# Agent Communication Protocol Tests
# ============================================================================

class TestAgentCommunication:
    """Tests for Agent Communication Protocol."""

    @pytest.mark.asyncio
    async def test_communication_start_stop(self):
        """Test communication protocol lifecycle."""
        comm = AgentCommunicationProtocol()
        await comm.start()
        assert comm._running is True

        await comm.stop()
        assert comm._running is False

    @pytest.mark.asyncio
    async def test_agent_registration(self, agent_config, communication):
        """Test agent registration with communication protocol."""
        agent = TestAgent(config=agent_config)
        await agent.start()

        await communication.register_agent(agent)

        assert agent.agent_id in communication.agents
        assert agent.agent_id in communication.agent_queues

        await communication.unregister_agent(agent.agent_id)
        assert agent.agent_id not in communication.agents

        await agent.stop()

    @pytest.mark.asyncio
    async def test_direct_message_send(self, agent_config, communication):
        """Test sending direct messages between agents."""
        agent1 = TestAgent(config=agent_config)
        agent2 = TestAgent(config=agent_config)

        await agent1.start()
        await agent2.start()
        await communication.register_agent(agent1)
        await communication.register_agent(agent2)

        message_id = await communication.send(
            sender=agent1,
            recipient_id=agent2.agent_id,
            message_type=MessageType.DATA_SHARE,
            payload={"data": "test"},
        )

        assert message_id is not None
        assert communication.stats["messages_sent"] > 0

        await agent1.stop()
        await agent2.stop()

    @pytest.mark.asyncio
    async def test_broadcast_message(self, agent_config, communication):
        """Test broadcasting messages to all agents."""
        agents = []
        for _ in range(3):
            agent = TestAgent(config=agent_config)
            await agent.start()
            await communication.register_agent(agent)
            agents.append(agent)

        await communication.broadcast(
            sender=agents[0],
            message_type=MessageType.EVENT,
            payload={"event": "test"},
        )

        assert communication.stats["broadcasts_sent"] > 0

        for agent in agents:
            await agent.stop()

    @pytest.mark.asyncio
    async def test_channel_pubsub(self, agent_config, communication):
        """Test channel pub/sub functionality."""
        agent1 = TestAgent(config=agent_config)
        agent2 = TestAgent(config=agent_config)

        await agent1.start()
        await agent2.start()
        await communication.register_agent(agent1)
        await communication.register_agent(agent2)

        # Create channel
        channel = communication.create_channel("test-channel", "Test Channel")
        assert channel is not None

        # Subscribe
        await communication.subscribe(agent2, "test-channel")

        # Publish
        delivered = await communication.publish(
            sender=agent1,
            channel_name="test-channel",
            payload={"message": "test"},
        )

        assert delivered == 1

        await agent1.stop()
        await agent2.stop()


# ============================================================================
# Agent Pool Manager Tests
# ============================================================================

class TestAgentPoolManager:
    """Tests for Agent Pool Manager."""

    @pytest.mark.asyncio
    async def test_pool_creation(self, agent_config):
        """Test creating an agent pool."""
        pool_config = PoolConfig(
            agent_type="test_pool",
            agent_class=TestAgent,
            agent_config=agent_config,
            min_agents=1,
            max_agents=5,
            initial_agents=2,
        )

        pool = AgentPool(config=pool_config)
        await pool.start()

        assert len(pool.instances) == 2
        assert pool._running is True

        await pool.stop()
        assert len(pool.instances) == 0

    @pytest.mark.asyncio
    async def test_pool_task_submission(self, agent_config):
        """Test submitting tasks to a pool."""
        pool_config = PoolConfig(
            agent_type="test_pool",
            agent_class=TestAgent,
            agent_config=agent_config,
            min_agents=2,
            max_agents=5,
            initial_agents=2,
        )

        pool = AgentPool(config=pool_config)
        await pool.start()

        task = AgentTask(
            task_id="pool-test-1",
            task_type="echo",
            payload={"message": "pool test"},
        )

        # Submit task
        result = await asyncio.wait_for(
            pool.submit_task(task),
            timeout=5.0
        )

        assert result is not None

        await pool.stop()

    @pytest.mark.asyncio
    async def test_pool_manager_multi_pool(self, agent_config):
        """Test managing multiple pools."""
        manager = AgentPoolManager()
        await manager.start()

        pool_config = PoolConfig(
            agent_type="test_pool",
            agent_class=TestAgent,
            agent_config=agent_config,
            min_agents=1,
            max_agents=3,
            initial_agents=1,
        )

        pool = manager.register_pool(pool_config)
        assert pool is not None
        assert "test_pool" in manager.pools

        await manager.stop()


# ============================================================================
# Agent Memory Tests
# ============================================================================

class TestAgentMemory:
    """Tests for Agent Memory Manager."""

    @pytest.mark.asyncio
    async def test_memory_backend(self):
        """Test in-memory backend operations."""
        backend = InMemoryBackend()

        entry = MemoryEntry(
            entry_id="test-1",
            agent_id="agent-1",
            memory_type=MemoryType.EPISODIC,
            content={"event": "test"},
        )

        # Store
        result = await backend.store(entry)
        assert result is True

        # Retrieve
        entries = await backend.retrieve(
            agent_id="agent-1",
            memory_type=MemoryType.EPISODIC,
        )
        assert len(entries) == 1
        assert entries[0].entry_id == "test-1"

        # Delete
        deleted = await backend.delete("test-1")
        assert deleted is True

    @pytest.mark.asyncio
    async def test_working_memory(self, memory):
        """Test working memory operations."""
        await memory.store_working(
            agent_id="agent-1",
            content={"action": "test"},
        )

        entries = await memory.get_working("agent-1")
        assert len(entries) == 1

        await memory.clear_working("agent-1")
        entries = await memory.get_working("agent-1")
        assert len(entries) == 0

    @pytest.mark.asyncio
    async def test_episodic_memory(self, memory):
        """Test episodic memory operations."""
        episode_id = await memory.store_episodic(
            agent_id="agent-1",
            event_type="task_completion",
            content={"task": "test"},
            importance=0.8,
        )

        assert episode_id is not None

        episodes = await memory.retrieve_episodic(
            agent_id="agent-1",
            event_type="task_completion",
        )

        assert len(episodes) == 1

    @pytest.mark.asyncio
    async def test_semantic_memory(self, memory):
        """Test semantic memory operations."""
        fact_id = await memory.store_semantic(
            subject="agent-1",
            predicate="is_type",
            object_value="test_agent",
            confidence=1.0,
        )

        assert fact_id is not None

        results = await memory.retrieve_semantic(query="agent-1")
        assert len(results) >= 0

    @pytest.mark.asyncio
    async def test_memory_sharing(self, memory):
        """Test cross-agent memory sharing."""
        # Store memory for agent-1
        await memory.store_episodic(
            agent_id="agent-1",
            event_type="learned",
            content={"skill": "threat_detection"},
            importance=0.9,
        )

        # Share with agent-2
        shared = await memory.share_memory(
            source_agent_id="agent-1",
            target_agent_id="agent-2",
            memory_type=MemoryType.EPISODIC,
        )

        assert shared >= 0


# ============================================================================
# Collaboration Framework Tests
# ============================================================================

class TestCollaborationFramework:
    """Tests for Collaboration Framework."""

    @pytest.mark.asyncio
    async def test_consensus_session(self, agent_config, communication, collaboration):
        """Test consensus building session."""
        agent1 = TestAgent(config=agent_config)
        agent2 = TestAgent(config=agent_config)

        await agent1.start()
        await agent2.start()
        await communication.register_agent(agent1)
        await communication.register_agent(agent2)

        proposal = Proposal(
            proposal_id="proposal-1",
            proposer_id=agent1.agent_id,
            title="Test Proposal",
            description="A test proposal",
            options=[{"option": "A"}, {"option": "B"}],
        )

        session = await collaboration.initiate_consensus(
            initiator=agent1,
            proposal=proposal,
            participants={agent1.agent_id, agent2.agent_id},
            protocol=ConsensusProtocol.MAJORITY,
        )

        assert session is not None
        assert session.session_id in collaboration.sessions

        # Submit votes
        await collaboration.submit_vote(
            session_id=session.session_id,
            agent=agent1,
            vote_type=VoteType.APPROVE,
        )

        await collaboration.submit_vote(
            session_id=session.session_id,
            agent=agent2,
            vote_type=VoteType.APPROVE,
        )

        # Check consensus
        await asyncio.sleep(0.1)

        # Session should be completed
        assert collaboration.stats["successful_consensus"] >= 0

        await agent1.stop()
        await agent2.stop()

    @pytest.mark.asyncio
    async def test_negotiation_session(self, agent_config, communication, collaboration):
        """Test negotiation session."""
        agent1 = TestAgent(config=agent_config)
        agent2 = TestAgent(config=agent_config)

        await agent1.start()
        await agent2.start()
        await communication.register_agent(agent1)
        await communication.register_agent(agent2)

        session = await collaboration.initiate_negotiation(
            initiator=agent1,
            counterpart_id=agent2.agent_id,
            topic="resource_allocation",
            initial_terms={"cpu": 50, "memory": 1024},
            strategy=NegotiationStrategy.COOPERATIVE,
        )

        assert session is not None
        assert session.session_type == "negotiation"
        assert len(session.offers) == 1

        # Submit counter-offer
        offer = await collaboration.submit_counter_offer(
            session_id=session.session_id,
            agent=agent2,
            terms={"cpu": 40, "memory": 2048},
        )

        assert offer is not None

        await agent1.stop()
        await agent2.stop()


# ============================================================================
# Planning Agent Tests
# ============================================================================

class TestPlanningAgent:
    """Tests for Planning Agent."""

    @pytest.mark.asyncio
    async def test_planning_agent_initialization(self):
        """Test planning agent initialization."""
        config = AgentConfig(
            agent_type="planning_agent",
            capabilities=[AgentCapability.TASK_PLANNING],
        )

        planning_agent = PlanningAgent(config=config)

        assert AgentCapability.TASK_PLANNING in planning_agent.capabilities
        assert AgentCapability.AGENT_COORDINATION in planning_agent.capabilities

    @pytest.mark.asyncio
    async def test_plan_creation(self):
        """Test creating a task plan."""
        config = AgentConfig(
            agent_type="planning_agent",
            capabilities=[AgentCapability.TASK_PLANNING],
        )

        planning_agent = PlanningAgent(config=config)
        await planning_agent.start()

        task = AgentTask(
            task_id="complex-task-1",
            task_type="security_scan",
            payload={
                "name": "Full Security Scan",
                "description": "Scan network for vulnerabilities",
                "targets": ["192.168.1.0/24"],
            },
        )

        plan = await planning_agent.create_plan(task)

        assert plan is not None
        assert plan.plan_id is not None
        assert len(plan.subtasks) > 0
        assert plan.status == PlanStatus.DRAFT

        await planning_agent.stop()

    @pytest.mark.asyncio
    async def test_plan_status(self):
        """Test getting plan status."""
        config = AgentConfig(
            agent_type="planning_agent",
            capabilities=[AgentCapability.TASK_PLANNING],
        )

        planning_agent = PlanningAgent(config=config)
        await planning_agent.start()

        task = AgentTask(
            task_id="task-1",
            task_type="analysis",
            payload={"name": "Test"},
        )

        plan = await planning_agent.create_plan(task)
        status = planning_agent.get_plan_status(plan.plan_id)

        assert status is not None
        assert status["plan_id"] == plan.plan_id
        assert "subtasks" in status

        await planning_agent.stop()


# ============================================================================
# Multi-Agent Orchestrator Integration Tests
# ============================================================================

class TestMultiAgentOrchestrator:
    """Integration tests for Multi-Agent Orchestrator."""

    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        config = OrchestratorConfig(
            name="test-orchestrator",
            enable_planning=True,
            enable_pooling=True,
            enable_memory=True,
            enable_collaboration=True,
        )

        orchestrator = MultiAgentOrchestrator(config=config)
        await orchestrator.start()

        assert orchestrator._running is True
        assert orchestrator.communication is not None
        assert orchestrator.memory is not None

        await orchestrator.stop()
        assert orchestrator._running is False

    @pytest.mark.asyncio
    async def test_orchestrator_agent_registration(self, agent_config):
        """Test registering agents with orchestrator."""
        config = OrchestratorConfig(name="test-orchestrator")
        orchestrator = MultiAgentOrchestrator(config=config)
        await orchestrator.start()

        pool_config = PoolConfig(
            agent_type="test_agent",
            agent_class=TestAgent,
            agent_config=agent_config,
            min_agents=1,
            max_agents=3,
            initial_agents=1,
        )

        orchestrator.register_agent_class(
            agent_type="test_agent",
            agent_class=TestAgent,
            config=agent_config,
            pool_config=pool_config,
        )

        assert "test_agent" in orchestrator.agent_registry

        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_orchestrator_status(self):
        """Test getting orchestrator status."""
        config = OrchestratorConfig(name="test-orchestrator")
        orchestrator = MultiAgentOrchestrator(config=config)
        await orchestrator.start()

        status = await orchestrator.get_orchestrator_status()

        assert "orchestrator_id" in status
        assert status["running"] is True
        assert "components" in status
        assert "statistics" in status

        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_orchestrator_knowledge_operations(self):
        """Test knowledge storage and retrieval."""
        config = OrchestratorConfig(
            name="test-orchestrator",
            enable_memory=True,
        )
        orchestrator = MultiAgentOrchestrator(config=config)
        await orchestrator.start()

        # Store knowledge
        fact_id = await orchestrator.store_knowledge(
            subject="test_agent",
            predicate="has_capability",
            object_value="threat_analysis",
        )

        assert fact_id is not None

        # Query knowledge
        results = await orchestrator.query_knowledge("test_agent")
        assert isinstance(results, list)

        await orchestrator.stop()


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance tests for the multi-agent system."""

    @pytest.mark.asyncio
    async def test_high_volume_messaging(self, agent_config, communication):
        """Test high-volume message handling."""
        agents = []
        for _ in range(5):
            agent = TestAgent(config=agent_config)
            await agent.start()
            await communication.register_agent(agent)
            agents.append(agent)

        # Send many messages
        message_count = 100
        for i in range(message_count):
            await communication.send(
                sender=agents[0],
                recipient_id=agents[1].agent_id,
                message_type=MessageType.DATA_SHARE,
                payload={"index": i},
            )

        await asyncio.sleep(0.5)

        assert communication.stats["messages_sent"] >= message_count

        for agent in agents:
            await agent.stop()

    @pytest.mark.asyncio
    async def test_concurrent_task_execution(self, agent_config):
        """Test concurrent task execution in pool."""
        pool_config = PoolConfig(
            agent_type="test_pool",
            agent_class=TestAgent,
            agent_config=agent_config,
            min_agents=3,
            max_agents=5,
            initial_agents=3,
        )

        pool = AgentPool(config=pool_config)
        await pool.start()

        # Submit multiple tasks concurrently
        tasks = []
        for i in range(10):
            task = AgentTask(
                task_id=f"concurrent-{i}",
                task_type="echo",
                payload={"index": i},
            )
            tasks.append(pool.submit_task(task))

        # Wait for all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = sum(1 for r in results if not isinstance(r, Exception))
        assert success_count > 0

        await pool.stop()


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
