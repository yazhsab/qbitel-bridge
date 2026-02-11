"""
QBITEL - Orchestrator Agent & Multi-Agent System

Coordinates the multi-agent system for legacy protocol analysis
and modernization.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Set
import uuid

from prometheus_client import Counter, Histogram, Gauge

from .base import (
    BaseAgent,
    AgentConfig,
    AgentRole,
    AgentMessage,
    MessageType,
    AgentState,
)
from .memory import (
    SharedMemory,
    ConversationMemory,
    EpisodicMemory,
    SemanticMemory,
    MemoryType,
)
from .protocol_analyst import ProtocolAnalystAgent
from .documentation_agent import DocumentationAgent
from .risk_assessor import RiskAssessorAgent
from .code_generator import CodeGeneratorAgent


logger = logging.getLogger(__name__)


# =============================================================================
# Metrics
# =============================================================================

ORCHESTRATOR_TASKS = Counter(
    "qbitel_orchestrator_tasks_total",
    "Total tasks processed by orchestrator",
    ["task_type", "status"]
)
ORCHESTRATOR_DELEGATIONS = Counter(
    "qbitel_orchestrator_delegations_total",
    "Total task delegations",
    ["target_agent"]
)
ORCHESTRATOR_DURATION = Histogram(
    "qbitel_orchestrator_task_duration_seconds",
    "Task processing duration",
    ["task_type"]
)
ACTIVE_AGENTS = Gauge(
    "qbitel_multi_agent_active_agents",
    "Number of active agents"
)


# =============================================================================
# Task Types
# =============================================================================

@dataclass
class OrchestratorTask:
    """A task managed by the orchestrator."""

    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_type: str = "general"
    description: str = ""
    input_data: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending, in_progress, completed, failed
    assigned_agents: Set[AgentRole] = field(default_factory=set)
    sub_tasks: List["OrchestratorTask"] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


# =============================================================================
# Orchestrator Agent
# =============================================================================

class OrchestratorAgent(BaseAgent):
    """
    Orchestrator agent that coordinates multi-agent collaboration.

    Responsibilities:
    - Task decomposition
    - Agent assignment
    - Progress monitoring
    - Result aggregation
    - Error recovery
    """

    def __init__(
        self,
        llm_service: Any,
        agents: Dict[AgentRole, BaseAgent],
        config: Optional[AgentConfig] = None
    ):
        config = config or AgentConfig(
            role=AgentRole.ORCHESTRATOR,
            name="Task Orchestrator",
            description=(
                "Coordinates the multi-agent system for legacy protocol analysis. "
                "Decomposes complex tasks, assigns work to specialized agents, "
                "monitors progress, and aggregates results."
            ),
            model="gpt-4o",
            temperature=0.5,
            max_tokens=4096,
            max_iterations=20,
            enable_reflection=True,
        )

        super().__init__(config, llm_service, tools=[])

        self.agents = agents
        self.tasks: Dict[str, OrchestratorTask] = {}
        self.task_history: List[OrchestratorTask] = []

    @property
    def system_prompt(self) -> str:
        """Orchestrator specific system prompt."""
        agent_list = "\n".join([
            f"- {role.value}: {agent.config.description[:100]}..."
            for role, agent in self.agents.items()
        ])

        return f"""You are the Task Orchestrator for the QBITEL Legacy Whisperer system.

Your role is to coordinate a team of specialized agents:

{agent_list}

## Coordination Responsibilities

1. **Task Analysis**: Understand the user's request and break it into subtasks
2. **Agent Assignment**: Assign subtasks to the most appropriate agent(s)
3. **Progress Monitoring**: Track progress and handle blockers
4. **Result Aggregation**: Combine results from multiple agents
5. **Quality Assurance**: Ensure completeness and accuracy

## Delegation Guidelines

- **Protocol Analyst**: Traffic analysis, pattern recognition, structure inference
- **Documentation**: Documentation generation, behavior explanations
- **Risk Assessor**: Risk evaluation, mitigation strategies
- **Code Generator**: Adapter code, tests, deployment configs

## Task Decomposition

For complex tasks, break them down:

1. Analyze requirements
2. Identify dependencies between subtasks
3. Determine parallelizable work
4. Assign agents appropriately
5. Define success criteria

## Output Format

When planning task execution:

```
TASK PLAN
=========

Objective: [Main goal]

Subtasks:
1. [Subtask 1]
   - Agent: [assigned agent]
   - Dependencies: [none or list]
   - Expected output: [description]

2. [Subtask 2]
   ...

Execution Order:
- Parallel: [1, 2]
- Then: [3]
- Finally: [4]

Success Criteria:
- [Criterion 1]
- [Criterion 2]
```
"""

    async def execute_task(
        self,
        task_type: str,
        description: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a high-level task by coordinating agents.

        Args:
            task_type: Type of task (reverse_engineer, generate_adapter, etc.)
            description: Human-readable description
            input_data: Input data for the task

        Returns:
            Aggregated results from all agents
        """
        start_time = time.time()
        ORCHESTRATOR_TASKS.labels(task_type=task_type, status="started").inc()

        task = OrchestratorTask(
            task_type=task_type,
            description=description,
            input_data=input_data,
            status="in_progress"
        )
        self.tasks[task.task_id] = task

        try:
            logger.info(f"Starting task {task.task_id}: {task_type}")

            # Decompose task into subtasks
            subtasks = await self._decompose_task(task)

            # Execute subtasks (with parallelization where possible)
            results = await self._execute_subtasks(subtasks, task)

            # Aggregate results
            aggregated = await self._aggregate_results(task_type, results)

            task.results = aggregated
            task.status = "completed"
            task.completed_at = datetime.now(timezone.utc)

            ORCHESTRATOR_TASKS.labels(task_type=task_type, status="completed").inc()
            ORCHESTRATOR_DURATION.labels(task_type=task_type).observe(
                time.time() - start_time
            )

            logger.info(f"Task {task.task_id} completed successfully")

            return aggregated

        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            task.status = "failed"
            task.error = str(e)
            ORCHESTRATOR_TASKS.labels(task_type=task_type, status="failed").inc()
            raise

        finally:
            self.task_history.append(task)

    async def _decompose_task(
        self,
        task: OrchestratorTask
    ) -> List[OrchestratorTask]:
        """Decompose a task into subtasks."""
        task_type = task.task_type

        if task_type == "reverse_engineer":
            return self._decompose_reverse_engineer(task)
        elif task_type == "generate_adapter":
            return self._decompose_generate_adapter(task)
        elif task_type == "explain_behavior":
            return self._decompose_explain_behavior(task)
        else:
            # Use LLM for generic decomposition
            return await self._llm_decompose(task)

    def _decompose_reverse_engineer(
        self,
        task: OrchestratorTask
    ) -> List[OrchestratorTask]:
        """Decompose reverse engineering task."""
        return [
            OrchestratorTask(
                task_type="analyze_traffic",
                description="Analyze traffic patterns and structure",
                input_data=task.input_data,
                assigned_agents={AgentRole.PROTOCOL_ANALYST}
            ),
            OrchestratorTask(
                task_type="generate_documentation",
                description="Generate protocol documentation",
                input_data=task.input_data,
                assigned_agents={AgentRole.DOCUMENTATION}
            ),
            OrchestratorTask(
                task_type="assess_risks",
                description="Assess modernization risks",
                input_data=task.input_data,
                assigned_agents={AgentRole.RISK_ASSESSOR}
            ),
        ]

    def _decompose_generate_adapter(
        self,
        task: OrchestratorTask
    ) -> List[OrchestratorTask]:
        """Decompose adapter generation task."""
        return [
            OrchestratorTask(
                task_type="assess_risks",
                description="Assess adapter generation risks",
                input_data=task.input_data,
                assigned_agents={AgentRole.RISK_ASSESSOR}
            ),
            OrchestratorTask(
                task_type="generate_code",
                description="Generate adapter code",
                input_data=task.input_data,
                assigned_agents={AgentRole.CODE_GENERATOR}
            ),
            OrchestratorTask(
                task_type="generate_tests",
                description="Generate test suite",
                input_data=task.input_data,
                assigned_agents={AgentRole.CODE_GENERATOR}
            ),
            OrchestratorTask(
                task_type="generate_documentation",
                description="Generate integration guide",
                input_data=task.input_data,
                assigned_agents={AgentRole.DOCUMENTATION}
            ),
        ]

    def _decompose_explain_behavior(
        self,
        task: OrchestratorTask
    ) -> List[OrchestratorTask]:
        """Decompose behavior explanation task."""
        return [
            OrchestratorTask(
                task_type="explain_behavior",
                description="Explain legacy behavior",
                input_data=task.input_data,
                assigned_agents={AgentRole.DOCUMENTATION}
            ),
            OrchestratorTask(
                task_type="assess_risks",
                description="Assess risks of addressing behavior",
                input_data=task.input_data,
                assigned_agents={AgentRole.RISK_ASSESSOR}
            ),
        ]

    async def _llm_decompose(
        self,
        task: OrchestratorTask
    ) -> List[OrchestratorTask]:
        """Use LLM to decompose unknown task types."""
        prompt = f"""Decompose this task into subtasks for the agent team.

Task Type: {task.task_type}
Description: {task.description}
Input Data Keys: {list(task.input_data.keys())}

Available agents:
- protocol_analyst: Traffic analysis, pattern recognition
- documentation: Documentation, explanations
- risk_assessor: Risk evaluation
- code_generator: Code generation

Provide subtasks in format:
SUBTASK: [name]
AGENT: [agent_role]
DESCRIPTION: [what to do]
---
"""
        response = await self._call_llm(prompt)

        # Parse response into subtasks (simplified parsing)
        subtasks = []
        # In production, use proper parsing
        subtasks.append(OrchestratorTask(
            task_type="generic",
            description=task.description,
            input_data=task.input_data,
            assigned_agents={AgentRole.PROTOCOL_ANALYST}
        ))

        return subtasks

    async def _execute_subtasks(
        self,
        subtasks: List[OrchestratorTask],
        parent_task: OrchestratorTask
    ) -> Dict[str, Any]:
        """Execute subtasks, parallelizing where possible."""
        results = {}

        # Group subtasks by dependency (simplified: assume all parallel)
        # In production, implement proper dependency resolution

        for subtask in subtasks:
            parent_task.sub_tasks.append(subtask)

        # Execute subtasks
        execution_tasks = []
        for subtask in subtasks:
            for agent_role in subtask.assigned_agents:
                if agent_role in self.agents:
                    ORCHESTRATOR_DELEGATIONS.labels(
                        target_agent=agent_role.value
                    ).inc()
                    execution_tasks.append(
                        self._delegate_to_agent(agent_role, subtask)
                    )

        # Wait for all subtasks
        if execution_tasks:
            subtask_results = await asyncio.gather(
                *execution_tasks,
                return_exceptions=True
            )

            for i, result in enumerate(subtask_results):
                if isinstance(result, Exception):
                    logger.error(f"Subtask failed: {result}")
                    results[f"subtask_{i}"] = {"error": str(result)}
                else:
                    results[f"subtask_{i}"] = result

        return results

    async def _delegate_to_agent(
        self,
        agent_role: AgentRole,
        subtask: OrchestratorTask
    ) -> Dict[str, Any]:
        """Delegate a subtask to an agent."""
        agent = self.agents.get(agent_role)
        if not agent:
            raise ValueError(f"Agent {agent_role.value} not available")

        message = AgentMessage(
            message_type=MessageType.TASK,
            sender=AgentRole.ORCHESTRATOR,
            recipient=agent_role,
            content=subtask.description,
            data=subtask.input_data
        )

        response = await agent.process_message(message)

        subtask.status = "completed"
        subtask.results = response.data

        return response.data

    async def _aggregate_results(
        self,
        task_type: str,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Aggregate results from multiple agents."""
        aggregated = {
            "task_type": task_type,
            "subtask_results": results,
            "summary": "",
            "recommendations": []
        }

        # Generate summary using LLM
        summary_prompt = f"""Summarize these results from a multi-agent analysis:

Task Type: {task_type}
Results: {results}

Provide:
1. A brief summary (2-3 sentences)
2. Key findings
3. Recommendations
"""
        summary_response = await self._call_llm(summary_prompt)
        aggregated["summary"] = summary_response

        return aggregated


# =============================================================================
# Multi-Agent Orchestrator (High-Level API)
# =============================================================================

class MultiAgentOrchestrator:
    """
    High-level API for the multi-agent system.

    Provides simple methods for common tasks while managing
    the underlying agent coordination.
    """

    def __init__(self, llm_service: Any, rag_engine: Any = None):
        self.llm_service = llm_service
        self.rag_engine = rag_engine
        self.logger = logging.getLogger(__name__)

        # Initialize memory systems
        self.shared_memory = SharedMemory()
        self.conversation_memory = ConversationMemory()
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()

        # Initialize agents
        self.agents: Dict[AgentRole, BaseAgent] = {}
        self.orchestrator: Optional[OrchestratorAgent] = None

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all agents and memory systems."""
        self.logger.info("Initializing Multi-Agent Orchestrator")

        # Load semantic knowledge
        self.semantic_memory.preload_protocol_knowledge()

        # Create specialized agents
        self.agents[AgentRole.PROTOCOL_ANALYST] = ProtocolAnalystAgent(
            self.llm_service
        )
        self.agents[AgentRole.DOCUMENTATION] = DocumentationAgent(
            self.llm_service, self.rag_engine
        )
        self.agents[AgentRole.RISK_ASSESSOR] = RiskAssessorAgent(
            self.llm_service
        )
        self.agents[AgentRole.CODE_GENERATOR] = CodeGeneratorAgent(
            self.llm_service
        )

        # Create orchestrator
        self.orchestrator = OrchestratorAgent(
            self.llm_service, self.agents
        )

        ACTIVE_AGENTS.set(len(self.agents) + 1)  # +1 for orchestrator

        self._initialized = True
        self.logger.info(
            f"Multi-Agent Orchestrator initialized with {len(self.agents)} agents"
        )

    async def reverse_engineer_protocol(
        self,
        traffic_samples: List[bytes],
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Reverse engineer a protocol from traffic samples.

        Coordinates Protocol Analyst, Documentation, and Risk Assessor agents.
        """
        if not self._initialized:
            await self.initialize()

        # Start episodic memory
        episode_id = await self.episodic_memory.start_episode(
            f"Reverse engineering protocol from {len(traffic_samples)} samples",
            {AgentRole.PROTOCOL_ANALYST, AgentRole.DOCUMENTATION, AgentRole.RISK_ASSESSOR}
        )

        try:
            result = await self.orchestrator.execute_task(
                task_type="reverse_engineer",
                description="Reverse engineer legacy protocol from traffic samples",
                input_data={
                    "traffic_samples": traffic_samples,
                    "context": context
                }
            )

            # Store in shared memory
            await self.shared_memory.store(
                f"Protocol analysis completed: {result.get('summary', '')[:200]}",
                MemoryType.EPISODIC,
                AgentRole.ORCHESTRATOR,
                importance=0.8
            )

            # End episode
            await self.episodic_memory.end_episode(
                episode_id,
                "Protocol reverse engineering completed successfully",
                success=True,
                lessons_learned=["Successfully analyzed traffic patterns"]
            )

            return result

        except Exception as e:
            await self.episodic_memory.end_episode(
                episode_id,
                f"Failed: {str(e)}",
                success=False,
                lessons_learned=[f"Error during reverse engineering: {str(e)}"]
            )
            raise

    async def generate_adapter(
        self,
        protocol_spec: Dict[str, Any],
        target_protocol: str,
        language: str = "python"
    ) -> Dict[str, Any]:
        """
        Generate a protocol adapter.

        Coordinates Risk Assessor and Code Generator agents.
        """
        if not self._initialized:
            await self.initialize()

        episode_id = await self.episodic_memory.start_episode(
            f"Generating {language} adapter for {target_protocol}",
            {AgentRole.RISK_ASSESSOR, AgentRole.CODE_GENERATOR, AgentRole.DOCUMENTATION}
        )

        try:
            result = await self.orchestrator.execute_task(
                task_type="generate_adapter",
                description=f"Generate {language} adapter to {target_protocol}",
                input_data={
                    "protocol_spec": protocol_spec,
                    "target_protocol": target_protocol,
                    "language": language
                }
            )

            await self.episodic_memory.end_episode(
                episode_id,
                "Adapter generation completed successfully",
                success=True
            )

            return result

        except Exception as e:
            await self.episodic_memory.end_episode(
                episode_id,
                f"Failed: {str(e)}",
                success=False
            )
            raise

    async def explain_behavior(
        self,
        behavior: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Explain legacy system behavior.

        Coordinates Documentation and Risk Assessor agents.
        """
        if not self._initialized:
            await self.initialize()

        result = await self.orchestrator.execute_task(
            task_type="explain_behavior",
            description="Explain legacy system behavior",
            input_data={
                "behavior": behavior,
                "context": context
            }
        )

        return result

    async def get_agent_states(self) -> Dict[str, AgentState]:
        """Get current state of all agents."""
        states = {}
        for role, agent in self.agents.items():
            states[role.value] = agent.get_state()
        if self.orchestrator:
            states["orchestrator"] = self.orchestrator.get_state()
        return states

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        return {
            "shared_memory": self.shared_memory.get_stats(),
            "conversation_threads": len(self.conversation_memory._threads),
            "episodes": len(self.episodic_memory._episodes),
            "semantic_facts": len(self.semantic_memory._facts),
        }

    async def shutdown(self) -> None:
        """Shutdown all agents and cleanup resources."""
        self.logger.info("Shutting down Multi-Agent Orchestrator")

        for agent in self.agents.values():
            await agent.shutdown()

        if self.orchestrator:
            await self.orchestrator.shutdown()

        ACTIVE_AGENTS.set(0)

        self.logger.info("Multi-Agent Orchestrator shutdown complete")


# =============================================================================
# Factory Function
# =============================================================================

async def create_multi_agent_orchestrator(
    llm_service: Any,
    rag_engine: Any = None
) -> MultiAgentOrchestrator:
    """Create and initialize a Multi-Agent Orchestrator."""
    orchestrator = MultiAgentOrchestrator(llm_service, rag_engine)
    await orchestrator.initialize()
    return orchestrator


__all__ = [
    "OrchestratorTask",
    "OrchestratorAgent",
    "MultiAgentOrchestrator",
    "create_multi_agent_orchestrator",
]
