"""
CRONOS AI - Multi-Agent Orchestrator

The central orchestrator that integrates all multi-agent components:
- Planning Agent for task decomposition
- Agent Communication Protocol for messaging
- Agent Pool Manager for dynamic scaling
- Agent Memory Manager for persistence
- Collaboration Framework for consensus/negotiation

This is the main entry point for the multi-agent system.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type, TYPE_CHECKING

from prometheus_client import Counter, Gauge, Histogram

from .base_agent import (
    BaseAgent,
    AgentCapability,
    AgentConfig,
    AgentTask,
    AgentPriority,
    AgentStatus,
)
from .planning_agent import (
    PlanningAgent,
    TaskPlan,
    ExecutionStrategy,
)
from .agent_communication import (
    AgentCommunicationProtocol,
    AgentMessage,
    MessageType,
)
from .agent_pool import (
    AgentPoolManager,
    AgentPool,
    PoolConfig,
    ScalingPolicy,
    LoadBalancingStrategy,
)
from .agent_memory import (
    AgentMemoryManager,
    MemoryType,
)
from .agent_collaboration import (
    CollaborationFramework,
    ConsensusProtocol,
    Proposal,
)

if TYPE_CHECKING:
    from ..llm.unified_llm_service import UnifiedLLMService

# Prometheus metrics
ORCHESTRATOR_TASKS = Counter(
    "cronos_orchestrator_tasks_total",
    "Total tasks processed by orchestrator",
    ["task_type", "status"],
)
ORCHESTRATOR_AGENTS = Gauge(
    "cronos_orchestrator_agents_total",
    "Total agents managed by orchestrator",
    ["agent_type"],
)
ORCHESTRATOR_LATENCY = Histogram(
    "cronos_orchestrator_latency_seconds",
    "Task processing latency",
    ["task_type"],
)

logger = logging.getLogger(__name__)


class OrchestratorMode(str, Enum):
    """Operating modes for the orchestrator."""

    STANDALONE = "standalone"  # Single orchestrator
    DISTRIBUTED = "distributed"  # Multiple orchestrators
    HIERARCHICAL = "hierarchical"  # Parent-child orchestrators


@dataclass
class OrchestratorConfig:
    """Configuration for the Multi-Agent Orchestrator."""

    name: str = "cronos-orchestrator"
    mode: OrchestratorMode = OrchestratorMode.STANDALONE

    # Planning Agent config
    enable_planning: bool = True
    max_concurrent_plans: int = 10
    default_execution_strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE

    # Pool config
    enable_pooling: bool = True
    default_min_agents: int = 1
    default_max_agents: int = 10
    auto_scaling: bool = True

    # Memory config
    enable_memory: bool = True
    memory_consolidation_interval: int = 300  # seconds

    # Collaboration config
    enable_collaboration: bool = True
    default_consensus_protocol: ConsensusProtocol = ConsensusProtocol.MAJORITY

    # Communication config
    max_message_queue_size: int = 10000

    # Performance config
    task_timeout_seconds: int = 300
    health_check_interval: int = 30

    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiAgentOrchestrator:
    """
    Central orchestrator for the CRONOS AI multi-agent system.

    Provides unified access to:
    - Task submission and planning
    - Agent pool management
    - Inter-agent communication
    - Memory and learning
    - Collaboration and consensus

    Usage:
        orchestrator = MultiAgentOrchestrator(config, llm_service)
        await orchestrator.start()

        # Submit a complex task
        result = await orchestrator.submit_task(
            task_type="security_analysis",
            payload={"target": "network"},
            require_planning=True,
        )

        # Or use specific capabilities
        agents = await orchestrator.get_agents_with_capability(
            AgentCapability.THREAT_ANALYSIS
        )
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        llm_service: Optional["UnifiedLLMService"] = None,
    ):
        """Initialize the Multi-Agent Orchestrator."""
        self.config = config
        self.orchestrator_id = str(uuid.uuid4())
        self.llm_service = llm_service

        # Core components
        self.communication: Optional[AgentCommunicationProtocol] = None
        self.memory: Optional[AgentMemoryManager] = None
        self.pool_manager: Optional[AgentPoolManager] = None
        self.planning_agent: Optional[PlanningAgent] = None
        self.collaboration: Optional[CollaborationFramework] = None

        # Registered agent classes
        self.agent_registry: Dict[str, Type[BaseAgent]] = {}

        # Active plans
        self.active_plans: Dict[str, TaskPlan] = {}

        # Task tracking
        self.pending_tasks: Dict[str, asyncio.Future] = {}

        # Statistics
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "plans_created": 0,
            "consensus_sessions": 0,
        }

        # Lifecycle
        self._running = False
        self._background_tasks: List[asyncio.Task] = []

        self.logger = logging.getLogger(
            f"{__name__}.Orchestrator.{self.orchestrator_id[:8]}"
        )

    async def start(self) -> None:
        """Start the orchestrator and all components."""
        if self._running:
            self.logger.warning("Orchestrator already running")
            return

        self.logger.info(f"Starting Multi-Agent Orchestrator: {self.config.name}")

        # Initialize communication protocol
        self.communication = AgentCommunicationProtocol(
            max_queue_size=self.config.max_message_queue_size
        )
        await self.communication.start()

        # Initialize memory manager
        if self.config.enable_memory:
            self.memory = AgentMemoryManager()
            await self.memory.start()

        # Initialize pool manager
        if self.config.enable_pooling:
            self.pool_manager = AgentPoolManager(
                communication=self.communication,
                memory=self.memory,
            )
            await self.pool_manager.start()

        # Initialize collaboration framework
        if self.config.enable_collaboration:
            self.collaboration = CollaborationFramework(
                communication=self.communication
            )
            await self.collaboration.start()

        # Initialize planning agent
        if self.config.enable_planning:
            planning_config = AgentConfig(
                agent_type="planning_agent",
                capabilities=[
                    AgentCapability.TASK_PLANNING,
                    AgentCapability.AGENT_COORDINATION,
                    AgentCapability.WORKFLOW_ORCHESTRATION,
                ],
                max_concurrent_tasks=self.config.max_concurrent_plans,
            )
            self.planning_agent = PlanningAgent(
                config=planning_config,
                llm_service=self.llm_service,
                agent_pool=self.pool_manager,
                communication=self.communication,
                memory=self.memory,
            )
            await self.planning_agent.start()

        # Start background tasks
        self._background_tasks.append(
            asyncio.create_task(self._health_monitor())
        )
        self._background_tasks.append(
            asyncio.create_task(self._metrics_collector())
        )

        self._running = True
        self.logger.info("Multi-Agent Orchestrator started")

    async def stop(self) -> None:
        """Stop the orchestrator and all components."""
        if not self._running:
            return

        self.logger.info("Stopping Multi-Agent Orchestrator")
        self._running = False

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Stop components in reverse order
        if self.planning_agent:
            await self.planning_agent.stop()

        if self.collaboration:
            await self.collaboration.stop()

        if self.pool_manager:
            await self.pool_manager.stop()

        if self.memory:
            await self.memory.stop()

        if self.communication:
            await self.communication.stop()

        self.logger.info("Multi-Agent Orchestrator stopped")

    # Agent Registration

    def register_agent_class(
        self,
        agent_type: str,
        agent_class: Type[BaseAgent],
        config: AgentConfig,
        pool_config: Optional[PoolConfig] = None,
    ) -> None:
        """
        Register an agent class with the orchestrator.

        Args:
            agent_type: Unique identifier for this agent type
            agent_class: The agent class to register
            config: Default configuration for agents of this type
            pool_config: Optional pool configuration for auto-scaling
        """
        self.agent_registry[agent_type] = agent_class

        # Create pool if configured
        if pool_config and self.pool_manager:
            pool_config.agent_class = agent_class
            pool_config.agent_config = config
            self.pool_manager.register_pool(pool_config)

        self.logger.info(f"Registered agent class: {agent_type}")

    # Task Submission

    async def submit_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: AgentPriority = AgentPriority.NORMAL,
        require_planning: bool = False,
        required_capabilities: Optional[List[AgentCapability]] = None,
        execution_strategy: Optional[ExecutionStrategy] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """
        Submit a task for execution.

        This is the main entry point for task submission. The orchestrator will:
        1. Determine if planning is needed
        2. Decompose complex tasks into sub-tasks
        3. Route tasks to appropriate agents
        4. Monitor execution and return results

        Args:
            task_type: Type of task to execute
            payload: Task data
            priority: Task priority
            require_planning: Force task planning
            required_capabilities: Capabilities needed
            execution_strategy: How to execute sub-tasks
            timeout: Task timeout

        Returns:
            Task execution result
        """
        task_id = str(uuid.uuid4())
        start_time = time.time()

        self.stats["tasks_submitted"] += 1
        self.logger.info(f"Task submitted: {task_id[:8]} ({task_type})")

        task = AgentTask(
            task_id=task_id,
            task_type=task_type,
            payload=payload,
            priority=priority,
            timeout=timeout or self.config.task_timeout_seconds,
        )

        try:
            # Determine execution path
            if require_planning or self._should_use_planning(task):
                # Use planning agent for complex tasks
                result = await self._execute_with_planning(
                    task,
                    required_capabilities,
                    execution_strategy,
                )
            else:
                # Direct execution
                result = await self._execute_direct(
                    task,
                    required_capabilities,
                )

            self.stats["tasks_completed"] += 1
            ORCHESTRATOR_TASKS.labels(
                task_type=task_type,
                status="success"
            ).inc()

            execution_time = time.time() - start_time
            ORCHESTRATOR_LATENCY.labels(task_type=task_type).observe(execution_time)

            self.logger.info(
                f"Task completed: {task_id[:8]} in {execution_time:.2f}s"
            )

            return result

        except Exception as e:
            self.stats["tasks_failed"] += 1
            ORCHESTRATOR_TASKS.labels(
                task_type=task_type,
                status="failed"
            ).inc()
            self.logger.error(f"Task failed: {task_id[:8]} - {e}")
            raise

    async def _execute_with_planning(
        self,
        task: AgentTask,
        required_capabilities: Optional[List[AgentCapability]],
        execution_strategy: Optional[ExecutionStrategy],
    ) -> Any:
        """Execute task using the planning agent."""
        if not self.planning_agent:
            raise RuntimeError("Planning agent not available")

        # Add planning metadata
        task.payload["execution_strategy"] = (
            execution_strategy.value if execution_strategy
            else self.config.default_execution_strategy.value
        )
        task.payload["required_capabilities"] = (
            [c.value for c in required_capabilities] if required_capabilities else []
        )

        # Submit to planning agent
        planning_task = AgentTask(
            task_id=str(uuid.uuid4()),
            task_type="decompose_and_execute",
            payload=task.payload,
            priority=task.priority,
            timeout=task.timeout,
            correlation_id=task.task_id,
        )

        await self.planning_agent.submit_task(planning_task)
        self.stats["plans_created"] += 1

        # Wait for completion
        return await self._wait_for_task(planning_task.task_id)

    async def _execute_direct(
        self,
        task: AgentTask,
        required_capabilities: Optional[List[AgentCapability]],
    ) -> Any:
        """Execute task directly via agent pool."""
        if not self.pool_manager:
            raise RuntimeError("Pool manager not available")

        return await self.pool_manager.submit_task(
            task=task,
            required_capabilities=required_capabilities,
        )

    async def _wait_for_task(
        self,
        task_id: str,
        timeout: Optional[float] = None
    ) -> Any:
        """Wait for a task to complete."""
        timeout = timeout or self.config.task_timeout_seconds
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check planning agent's completed tasks
            if self.planning_agent:
                for result in self.planning_agent.completed_tasks:
                    if result.task_id == task_id:
                        if result.success:
                            return result.result
                        else:
                            raise RuntimeError(result.error or "Task failed")

            await asyncio.sleep(0.1)

        raise asyncio.TimeoutError(f"Task {task_id} timed out")

    def _should_use_planning(self, task: AgentTask) -> bool:
        """Determine if a task should use planning."""
        # Complex tasks that benefit from planning
        complex_task_types = {
            "security_incident_response",
            "protocol_discovery",
            "compliance_audit",
            "threat_hunting",
            "multi_step_analysis",
        }

        if task.task_type in complex_task_types:
            return True

        # Check payload for complexity indicators
        payload = task.payload
        if payload.get("multi_step"):
            return True
        if payload.get("subtasks"):
            return True
        if len(payload.get("targets", [])) > 5:
            return True

        return False

    # Consensus and Collaboration

    async def request_consensus(
        self,
        proposal: Proposal,
        participants: Optional[Set[str]] = None,
        protocol: Optional[ConsensusProtocol] = None,
        timeout: int = 300,
    ) -> Dict[str, Any]:
        """
        Request consensus from a group of agents.

        Args:
            proposal: The proposal to vote on
            participants: Agent IDs to participate (or all if None)
            protocol: Consensus protocol to use
            timeout: Voting timeout

        Returns:
            Consensus result
        """
        if not self.collaboration:
            raise RuntimeError("Collaboration framework not available")

        # Get participants
        if participants is None:
            # Use all available agents
            participants = set()
            if self.pool_manager:
                status = await self.pool_manager.get_status()
                for agent in status.get("agents", []):
                    participants.add(agent["agent_id"])

        if not participants:
            raise ValueError("No participants available for consensus")

        # Use the planning agent as initiator
        if self.planning_agent:
            session = await self.collaboration.initiate_consensus(
                initiator=self.planning_agent,
                proposal=proposal,
                participants=participants,
                protocol=protocol or self.config.default_consensus_protocol,
                timeout_seconds=timeout,
            )

            self.stats["consensus_sessions"] += 1

            # Wait for consensus
            while session.session_id in self.collaboration.sessions:
                await asyncio.sleep(0.5)

            # Get result
            for completed in self.collaboration.completed_sessions:
                if completed.session_id == session.session_id:
                    return completed.result or {}

        return {"outcome": "failed", "reason": "session_not_found"}

    # Query Methods

    async def get_agents_with_capability(
        self,
        capability: AgentCapability
    ) -> List[Dict[str, Any]]:
        """Get all agents with a specific capability."""
        if not self.pool_manager:
            return []

        return await self.pool_manager.get_agents_with_capabilities([capability])

    async def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific agent."""
        if self.communication:
            for agent in self.communication.agents.values():
                if agent.agent_id == agent_id:
                    return agent.get_status()
        return None

    async def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status."""
        status = {
            "orchestrator_id": self.orchestrator_id,
            "name": self.config.name,
            "mode": self.config.mode.value,
            "running": self._running,
            "statistics": self.stats,
            "components": {
                "communication": bool(self.communication),
                "memory": bool(self.memory),
                "pool_manager": bool(self.pool_manager),
                "planning_agent": bool(self.planning_agent),
                "collaboration": bool(self.collaboration),
            },
        }

        # Add component-specific stats
        if self.communication:
            status["communication_stats"] = self.communication.get_stats()

        if self.memory:
            status["memory_stats"] = self.memory.get_stats()

        if self.pool_manager:
            status["pool_stats"] = await self.pool_manager.get_status()

        if self.collaboration:
            status["collaboration_stats"] = self.collaboration.get_stats()

        return status

    # Memory Operations

    async def store_knowledge(
        self,
        subject: str,
        predicate: str,
        object_value: str,
        confidence: float = 1.0,
    ) -> Optional[str]:
        """Store knowledge in the semantic memory."""
        if not self.memory:
            return None

        return await self.memory.store_semantic(
            subject=subject,
            predicate=predicate,
            object_value=object_value,
            confidence=confidence,
            source="orchestrator",
        )

    async def query_knowledge(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Query the knowledge base."""
        if not self.memory:
            return []

        return await self.memory.retrieve_semantic(query=query, limit=limit)

    # Background Tasks

    async def _health_monitor(self) -> None:
        """Monitor health of all components."""
        while self._running:
            try:
                # Check communication
                if self.communication:
                    agents = self.communication.get_agents()
                    for agent_info in agents:
                        ORCHESTRATOR_AGENTS.labels(
                            agent_type=agent_info["agent_type"]
                        ).set(1)

                # Check pools
                if self.pool_manager:
                    status = await self.pool_manager.get_status()
                    for agent_type, pool_stat in status.get("pools", {}).items():
                        ORCHESTRATOR_AGENTS.labels(
                            agent_type=agent_type
                        ).set(pool_stat["total_agents"])

                await asyncio.sleep(self.config.health_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(self.config.health_check_interval)

    async def _metrics_collector(self) -> None:
        """Collect and publish metrics."""
        while self._running:
            try:
                # Memory consolidation trigger
                if self.memory and self.communication:
                    for agent in self.communication.agents.values():
                        await self.memory.schedule_consolidation(agent.agent_id)

                await asyncio.sleep(self.config.memory_consolidation_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Metrics collector error: {e}")
                await asyncio.sleep(60)


# Convenience function to create a fully configured orchestrator
async def create_orchestrator(
    llm_service: Optional["UnifiedLLMService"] = None,
    **config_kwargs
) -> MultiAgentOrchestrator:
    """
    Create and start a fully configured Multi-Agent Orchestrator.

    Args:
        llm_service: Optional LLM service for planning
        **config_kwargs: Configuration overrides

    Returns:
        Started orchestrator instance
    """
    config = OrchestratorConfig(**config_kwargs)
    orchestrator = MultiAgentOrchestrator(config, llm_service)
    await orchestrator.start()
    return orchestrator
