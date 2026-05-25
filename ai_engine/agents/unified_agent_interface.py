"""
QBITEL - Unified Agent Interface

Bridges the three disconnected agent systems in QBITEL:
  1. Core Framework   (ai_engine.agents.base_agent.BaseAgent)
  2. Legacy Whisperer  (ai_engine.services.legacy_whisperer.agents.base.BaseAgent)
  3. BPO Agents        (ai_engine.domains.bpo.agents.bpo_agents.*)

This module provides:
- UnifiedAgentProtocol: a Protocol class that all agent types can satisfy
- AgentAdapter:         wraps any agent system's agent to the unified interface
- UnifiedAgentRegistry: central registry that discovers and manages agents from
                        all three subsystems through a single API
- Cross-system message routing and task delegation

This allows the orchestrator to treat agents from ANY subsystem interchangeably
for task routing, health monitoring, and inter-agent communication.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    Union,
    runtime_checkable,
)

from prometheus_client import Counter, Gauge

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

UNIFIED_AGENT_COUNT = Gauge(
    "qbitel_unified_agents_total",
    "Total agents registered in unified registry",
    ["subsystem", "agent_type"],
)
UNIFIED_AGENT_TASKS = Counter(
    "qbitel_unified_agent_tasks_total",
    "Tasks routed through unified agent interface",
    ["subsystem", "agent_type", "status"],
)
UNIFIED_CROSS_SYSTEM_CALLS = Counter(
    "qbitel_unified_cross_system_calls_total",
    "Cross-subsystem agent interactions",
    ["source_subsystem", "target_subsystem"],
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AgentSubsystem(str, Enum):
    """Identifies which subsystem an agent belongs to."""

    CORE = "core"  # ai_engine.agents.*
    LEGACY_WHISPERER = "legacy_whisperer"  # ai_engine.services.legacy_whisperer.*
    BPO = "bpo"  # ai_engine.domains.bpo.*
    EXTERNAL = "external"  # Third-party / MCP agents


class UnifiedAgentStatus(str, Enum):
    """Normalized agent status across all subsystems."""

    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    PAUSED = "paused"
    OFFLINE = "offline"


class UnifiedCapability(str, Enum):
    """Superset of capabilities across all subsystems."""

    # Core Framework capabilities
    THREAT_ANALYSIS = "threat_analysis"
    PROTOCOL_ANALYSIS = "protocol_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    COMPLIANCE_CHECK = "compliance_check"
    INCIDENT_RESPONSE = "incident_response"
    TASK_PLANNING = "task_planning"

    # Legacy Whisperer capabilities
    LEGACY_CODE_ANALYSIS = "legacy_code_analysis"
    PROTOCOL_DISCOVERY = "protocol_discovery"
    DOCUMENTATION_GENERATION = "documentation_generation"
    CODE_GENERATION = "code_generation"
    RISK_ASSESSMENT = "risk_assessment"

    # BPO capabilities
    DOCUMENT_PROCESSING = "document_processing"
    DATA_EXTRACTION = "data_extraction"
    QUALITY_ASSURANCE = "quality_assurance"
    WORKFLOW_AUTOMATION = "workflow_automation"
    REPORTING = "reporting"

    # Cross-cutting
    NATURAL_LANGUAGE = "natural_language"
    REASONING = "reasoning"
    SEARCH = "search"


# ---------------------------------------------------------------------------
# Unified Task / Result
# ---------------------------------------------------------------------------


@dataclass
class UnifiedTask:
    """
    A task that can be handled by any agent from any subsystem.

    Acts as a lingua franca between the three different task formats.
    """

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str = "general"
    prompt: str = ""  # Natural language task description
    payload: Dict[str, Any] = field(default_factory=dict)
    required_capabilities: List[UnifiedCapability] = field(default_factory=list)
    priority: int = 3  # 1=critical, 5=background
    timeout_seconds: float = 300.0
    source_subsystem: Optional[AgentSubsystem] = None
    source_agent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class UnifiedResult:
    """Normalized result from any agent subsystem."""

    task_id: str
    agent_id: str
    subsystem: AgentSubsystem
    success: bool
    content: Any = None
    error: Optional[str] = None
    execution_time_seconds: float = 0.0
    tokens_used: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Unified Agent Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class UnifiedAgentProtocol(Protocol):
    """
    Protocol (structural typing) that any agent from any subsystem can satisfy.

    Agents don't need to inherit from this — they just need to have these
    attributes/methods. The AgentAdapter fills the gap for agents that don't
    fully conform.
    """

    agent_id: str
    agent_type: str

    def get_status(self) -> Dict[str, Any]:
        ...

    async def execute_unified_task(self, task: UnifiedTask) -> UnifiedResult:
        ...


# ---------------------------------------------------------------------------
# Agent Adapters
# ---------------------------------------------------------------------------


class BaseAgentAdapter:
    """
    Adapter that wraps any subsystem's agent to the UnifiedAgentProtocol.

    Subclassed per subsystem to handle the different APIs.
    """

    def __init__(
        self,
        wrapped_agent: Any,
        subsystem: AgentSubsystem,
        agent_type: str,
        capabilities: Optional[List[UnifiedCapability]] = None,
    ):
        self._wrapped = wrapped_agent
        self.subsystem = subsystem
        self.agent_type = agent_type
        self.capabilities: Set[UnifiedCapability] = set(capabilities or [])
        self.registered_at = datetime.utcnow()

        # Try to extract agent_id from wrapped agent
        self.agent_id: str = getattr(
            wrapped_agent, "agent_id",
            getattr(wrapped_agent, "id", str(uuid.uuid4())),
        )

        self.logger = logging.getLogger(
            f"{__name__}.{subsystem.value}.{agent_type}"
        )

    @property
    def wrapped(self) -> Any:
        """Access the underlying agent."""
        return self._wrapped

    def get_status(self) -> Dict[str, Any]:
        """Get normalized agent status."""
        raise NotImplementedError

    async def execute_unified_task(self, task: UnifiedTask) -> UnifiedResult:
        """Execute a unified task by adapting to the subsystem's API."""
        raise NotImplementedError

    def get_unified_status(self) -> UnifiedAgentStatus:
        """Map subsystem status to unified status."""
        raise NotImplementedError

    def has_capability(self, cap: UnifiedCapability) -> bool:
        return cap in self.capabilities

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id={self.agent_id[:8]}, "
            f"type={self.agent_type}, "
            f"subsystem={self.subsystem.value})"
        )


class CoreAgentAdapter(BaseAgentAdapter):
    """Adapter for Core Framework agents (ai_engine.agents.base_agent.BaseAgent)."""

    def __init__(self, agent: Any, capabilities: Optional[List[UnifiedCapability]] = None):
        # Map core AgentCapability to UnifiedCapability
        mapped_caps = capabilities or []
        if not mapped_caps and hasattr(agent, "capabilities"):
            mapped_caps = self._map_core_capabilities(agent.capabilities)

        super().__init__(
            wrapped_agent=agent,
            subsystem=AgentSubsystem.CORE,
            agent_type=getattr(agent, "agent_type", "core_agent"),
            capabilities=mapped_caps,
        )

    def get_status(self) -> Dict[str, Any]:
        if hasattr(self._wrapped, "get_status"):
            return self._wrapped.get_status()
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "subsystem": self.subsystem.value,
            "status": self.get_unified_status().value,
        }

    def get_unified_status(self) -> UnifiedAgentStatus:
        status = getattr(self._wrapped, "status", None)
        if status is None:
            return UnifiedAgentStatus.OFFLINE

        status_val = status.value if hasattr(status, "value") else str(status)
        mapping = {
            "idle": UnifiedAgentStatus.READY,
            "busy": UnifiedAgentStatus.BUSY,
            "error": UnifiedAgentStatus.ERROR,
            "paused": UnifiedAgentStatus.PAUSED,
            "initializing": UnifiedAgentStatus.BUSY,
            "shutting_down": UnifiedAgentStatus.OFFLINE,
            "terminated": UnifiedAgentStatus.OFFLINE,
        }
        return mapping.get(status_val, UnifiedAgentStatus.OFFLINE)

    async def execute_unified_task(self, task: UnifiedTask) -> UnifiedResult:
        """Adapt unified task to Core Framework's AgentTask and execute."""
        start_time = time.time()
        try:
            # Dynamically import to avoid circular deps
            from .base_agent import AgentTask, AgentPriority

            core_task = AgentTask(
                task_id=task.task_id,
                task_type=task.task_type,
                payload={
                    "prompt": task.prompt,
                    **task.payload,
                },
                priority=AgentPriority(min(task.priority, 5)),
                timeout=task.timeout_seconds,
                metadata=task.metadata,
            )

            await self._wrapped.submit_task(core_task)

            # Wait for completion by polling completed_tasks
            result = await self._wait_for_completion(
                task.task_id, task.timeout_seconds
            )

            elapsed = time.time() - start_time

            return UnifiedResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                subsystem=self.subsystem,
                success=result.get("success", True),
                content=result.get("result"),
                execution_time_seconds=elapsed,
                metadata=result.get("metadata", {}),
            )

        except Exception as e:
            elapsed = time.time() - start_time
            return UnifiedResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                subsystem=self.subsystem,
                success=False,
                error=str(e),
                execution_time_seconds=elapsed,
            )

    async def _wait_for_completion(
        self, task_id: str, timeout: float
    ) -> Dict[str, Any]:
        """Wait for a core task to complete."""
        start = time.time()
        while time.time() - start < timeout:
            for result in getattr(self._wrapped, "completed_tasks", []):
                if result.task_id == task_id:
                    return {
                        "success": result.success,
                        "result": result.result,
                        "error": result.error,
                        "metadata": getattr(result, "metadata", {}),
                    }
            await asyncio.sleep(0.1)
        raise asyncio.TimeoutError(f"Core task {task_id} timed out")

    @staticmethod
    def _map_core_capabilities(
        core_caps: set,
    ) -> List[UnifiedCapability]:
        """Map Core AgentCapability enum values to UnifiedCapability."""
        mapping = {
            "threat_analysis": UnifiedCapability.THREAT_ANALYSIS,
            "protocol_analysis": UnifiedCapability.PROTOCOL_ANALYSIS,
            "anomaly_detection": UnifiedCapability.ANOMALY_DETECTION,
            "compliance_check": UnifiedCapability.COMPLIANCE_CHECK,
            "incident_response": UnifiedCapability.INCIDENT_RESPONSE,
            "task_planning": UnifiedCapability.TASK_PLANNING,
            "natural_language": UnifiedCapability.NATURAL_LANGUAGE,
            "legacy_system_analysis": UnifiedCapability.LEGACY_CODE_ANALYSIS,
        }
        result = []
        for cap in core_caps:
            val = cap.value if hasattr(cap, "value") else str(cap)
            if val in mapping:
                result.append(mapping[val])
        return result


class LegacyWhispererAdapter(BaseAgentAdapter):
    """Adapter for Legacy Whisperer agents."""

    def __init__(self, agent: Any, capabilities: Optional[List[UnifiedCapability]] = None):
        # Map role to capabilities
        mapped_caps = capabilities or []
        if not mapped_caps:
            mapped_caps = self._map_lw_role(agent)

        agent_type = "lw_agent"
        if hasattr(agent, "config") and hasattr(agent.config, "role"):
            agent_type = f"lw_{agent.config.role.value}"

        super().__init__(
            wrapped_agent=agent,
            subsystem=AgentSubsystem.LEGACY_WHISPERER,
            agent_type=agent_type,
            capabilities=mapped_caps,
        )

        # Use the agent's state.agent_id if available
        if hasattr(agent, "state") and hasattr(agent.state, "agent_id"):
            self.agent_id = agent.state.agent_id

    def get_status(self) -> Dict[str, Any]:
        state = getattr(self._wrapped, "state", None)
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "subsystem": self.subsystem.value,
            "status": self.get_unified_status().value,
            "is_busy": getattr(state, "is_busy", False) if state else False,
            "iteration_count": getattr(state, "iteration_count", 0) if state else 0,
        }

    def get_unified_status(self) -> UnifiedAgentStatus:
        state = getattr(self._wrapped, "state", None)
        if state is None:
            return UnifiedAgentStatus.OFFLINE
        if getattr(state, "is_busy", False):
            return UnifiedAgentStatus.BUSY
        if getattr(state, "errors_count", 0) > 5:
            return UnifiedAgentStatus.ERROR
        return UnifiedAgentStatus.READY

    async def execute_unified_task(self, task: UnifiedTask) -> UnifiedResult:
        """Adapt unified task to Legacy Whisperer's message-based API."""
        start_time = time.time()

        try:
            # Legacy Whisperer uses AgentMessage with process_message()
            # Import dynamically to avoid circular deps
            from ai_engine.services.legacy_whisperer.agents.base import (
                AgentMessage as LWMessage,
                MessageType as LWMessageType,
                AgentRole as LWRole,
            )

            message = LWMessage(
                message_type=LWMessageType.TASK,
                sender=LWRole.ORCHESTRATOR,
                content=task.prompt,
                data=task.payload,
                metadata={
                    "unified_task_id": task.task_id,
                    **task.metadata,
                },
            )

            response = await self._wrapped.process_message(message)
            elapsed = time.time() - start_time

            return UnifiedResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                subsystem=self.subsystem,
                success=(response.message_type.value != "error"),
                content=response.content if hasattr(response, "content") else str(response),
                execution_time_seconds=elapsed,
                metadata=getattr(response, "data", {}),
            )

        except Exception as e:
            elapsed = time.time() - start_time
            return UnifiedResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                subsystem=self.subsystem,
                success=False,
                error=str(e),
                execution_time_seconds=elapsed,
            )

    @staticmethod
    def _map_lw_role(agent: Any) -> List[UnifiedCapability]:
        """Map Legacy Whisperer roles to unified capabilities."""
        mapping = {
            "orchestrator": [
                UnifiedCapability.TASK_PLANNING,
                UnifiedCapability.REASONING,
            ],
            "protocol_analyst": [
                UnifiedCapability.PROTOCOL_ANALYSIS,
                UnifiedCapability.PROTOCOL_DISCOVERY,
            ],
            "documentation": [
                UnifiedCapability.DOCUMENTATION_GENERATION,
                UnifiedCapability.NATURAL_LANGUAGE,
            ],
            "risk_assessor": [
                UnifiedCapability.RISK_ASSESSMENT,
                UnifiedCapability.THREAT_ANALYSIS,
            ],
            "code_generator": [
                UnifiedCapability.CODE_GENERATION,
                UnifiedCapability.LEGACY_CODE_ANALYSIS,
            ],
        }
        role = "orchestrator"
        if hasattr(agent, "config") and hasattr(agent.config, "role"):
            role = agent.config.role.value
        return mapping.get(role, [UnifiedCapability.REASONING])


class BPOAgentAdapter(BaseAgentAdapter):
    """Adapter for BPO domain agents."""

    def __init__(self, agent: Any, capabilities: Optional[List[UnifiedCapability]] = None):
        mapped_caps = capabilities or self._map_bpo_capabilities(agent)

        agent_type = "bpo_agent"
        if hasattr(agent, "name"):
            agent_type = f"bpo_{agent.name}"
        elif hasattr(agent, "agent_name"):
            agent_type = f"bpo_{agent.agent_name}"

        super().__init__(
            wrapped_agent=agent,
            subsystem=AgentSubsystem.BPO,
            agent_type=agent_type,
            capabilities=mapped_caps,
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "subsystem": self.subsystem.value,
            "status": self.get_unified_status().value,
        }

    def get_unified_status(self) -> UnifiedAgentStatus:
        # BPO agents don't have standardized status; assume READY
        return UnifiedAgentStatus.READY

    async def execute_unified_task(self, task: UnifiedTask) -> UnifiedResult:
        """Adapt unified task to BPO agent's process() or execute() method."""
        start_time = time.time()

        try:
            # BPO agents typically have process() or execute() methods
            if hasattr(self._wrapped, "process"):
                result = await self._wrapped.process(
                    task.prompt, **task.payload
                )
            elif hasattr(self._wrapped, "execute"):
                result = await self._wrapped.execute(task.payload)
            elif hasattr(self._wrapped, "run"):
                result = await self._wrapped.run(task.prompt, **task.payload)
            else:
                raise AttributeError(
                    f"BPO agent {self.agent_type} has no process/execute/run method"
                )

            elapsed = time.time() - start_time

            return UnifiedResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                subsystem=self.subsystem,
                success=True,
                content=result,
                execution_time_seconds=elapsed,
            )

        except Exception as e:
            elapsed = time.time() - start_time
            return UnifiedResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                subsystem=self.subsystem,
                success=False,
                error=str(e),
                execution_time_seconds=elapsed,
            )

    @staticmethod
    def _map_bpo_capabilities(agent: Any) -> List[UnifiedCapability]:
        """Infer BPO capabilities from agent attributes."""
        caps = [UnifiedCapability.DOCUMENT_PROCESSING]
        name = getattr(agent, "name", getattr(agent, "agent_name", "")).lower()
        if "extract" in name:
            caps.append(UnifiedCapability.DATA_EXTRACTION)
        if "quality" in name or "qa" in name:
            caps.append(UnifiedCapability.QUALITY_ASSURANCE)
        if "workflow" in name or "automat" in name:
            caps.append(UnifiedCapability.WORKFLOW_AUTOMATION)
        if "report" in name:
            caps.append(UnifiedCapability.REPORTING)
        return caps


# ---------------------------------------------------------------------------
# Unified Agent Registry
# ---------------------------------------------------------------------------


class UnifiedAgentRegistry:
    """
    Central registry that unifies agents from all three subsystems.

    Provides:
    - Agent discovery by capability, subsystem, or type
    - Cross-system task routing
    - Health aggregation across all subsystems
    - Automatic adapter wrapping
    """

    def __init__(self):
        self._adapters: Dict[str, BaseAgentAdapter] = {}  # agent_id -> adapter
        self._by_subsystem: Dict[AgentSubsystem, Set[str]] = {
            s: set() for s in AgentSubsystem
        }
        self._by_capability: Dict[UnifiedCapability, Set[str]] = {
            c: set() for c in UnifiedCapability
        }
        self._lock = asyncio.Lock()
        self.logger = logging.getLogger(f"{__name__}.UnifiedRegistry")

    async def register(self, adapter: BaseAgentAdapter) -> str:
        """Register an agent adapter."""
        async with self._lock:
            self._adapters[adapter.agent_id] = adapter
            self._by_subsystem[adapter.subsystem].add(adapter.agent_id)
            for cap in adapter.capabilities:
                self._by_capability[cap].add(adapter.agent_id)

            UNIFIED_AGENT_COUNT.labels(
                subsystem=adapter.subsystem.value,
                agent_type=adapter.agent_type,
            ).inc()

            self.logger.info(
                f"Registered {adapter.subsystem.value} agent: "
                f"{adapter.agent_type} ({adapter.agent_id[:8]})"
            )
            return adapter.agent_id

    async def unregister(self, agent_id: str) -> bool:
        """Unregister an agent."""
        async with self._lock:
            adapter = self._adapters.pop(agent_id, None)
            if adapter is None:
                return False

            self._by_subsystem[adapter.subsystem].discard(agent_id)
            for cap in adapter.capabilities:
                self._by_capability[cap].discard(agent_id)

            UNIFIED_AGENT_COUNT.labels(
                subsystem=adapter.subsystem.value,
                agent_type=adapter.agent_type,
            ).dec()

            self.logger.info(f"Unregistered agent: {agent_id[:8]}")
            return True

    def register_core_agent(
        self, agent: Any, capabilities: Optional[List[UnifiedCapability]] = None
    ) -> str:
        """Convenience: wrap and register a Core Framework agent."""
        adapter = CoreAgentAdapter(agent, capabilities)
        # Use sync path for convenience (caller should await register)
        self._adapters[adapter.agent_id] = adapter
        self._by_subsystem[adapter.subsystem].add(adapter.agent_id)
        for cap in adapter.capabilities:
            self._by_capability[cap].add(adapter.agent_id)
        return adapter.agent_id

    def register_lw_agent(
        self, agent: Any, capabilities: Optional[List[UnifiedCapability]] = None
    ) -> str:
        """Convenience: wrap and register a Legacy Whisperer agent."""
        adapter = LegacyWhispererAdapter(agent, capabilities)
        self._adapters[adapter.agent_id] = adapter
        self._by_subsystem[adapter.subsystem].add(adapter.agent_id)
        for cap in adapter.capabilities:
            self._by_capability[cap].add(adapter.agent_id)
        return adapter.agent_id

    def register_bpo_agent(
        self, agent: Any, capabilities: Optional[List[UnifiedCapability]] = None
    ) -> str:
        """Convenience: wrap and register a BPO agent."""
        adapter = BPOAgentAdapter(agent, capabilities)
        self._adapters[adapter.agent_id] = adapter
        self._by_subsystem[adapter.subsystem].add(adapter.agent_id)
        for cap in adapter.capabilities:
            self._by_capability[cap].add(adapter.agent_id)
        return adapter.agent_id

    # ------ Discovery ------

    def get_agent(self, agent_id: str) -> Optional[BaseAgentAdapter]:
        """Get an agent adapter by ID."""
        return self._adapters.get(agent_id)

    def get_agents_by_subsystem(
        self, subsystem: AgentSubsystem
    ) -> List[BaseAgentAdapter]:
        """Get all agents from a specific subsystem."""
        return [
            self._adapters[aid]
            for aid in self._by_subsystem.get(subsystem, set())
            if aid in self._adapters
        ]

    def get_agents_by_capability(
        self,
        capability: UnifiedCapability,
        status_filter: Optional[UnifiedAgentStatus] = None,
    ) -> List[BaseAgentAdapter]:
        """Get all agents with a specific capability, optionally filtered by status."""
        agents = [
            self._adapters[aid]
            for aid in self._by_capability.get(capability, set())
            if aid in self._adapters
        ]
        if status_filter:
            agents = [a for a in agents if a.get_unified_status() == status_filter]
        return agents

    def find_best_agent(
        self,
        required_capabilities: List[UnifiedCapability],
        prefer_subsystem: Optional[AgentSubsystem] = None,
        prefer_local: bool = True,
    ) -> Optional[BaseAgentAdapter]:
        """
        Find the best available agent matching the required capabilities.

        Priority:
          1. Prefer agents matching ALL capabilities
          2. Prefer agents from the specified subsystem
          3. Prefer local (Core) agents if prefer_local=True
          4. Pick the one with READY status
        """
        # Get candidates matching all capabilities
        candidate_ids: Optional[Set[str]] = None
        for cap in required_capabilities:
            cap_set = self._by_capability.get(cap, set())
            if candidate_ids is None:
                candidate_ids = set(cap_set)
            else:
                candidate_ids &= cap_set

        if not candidate_ids:
            return None

        candidates = [
            self._adapters[aid]
            for aid in candidate_ids
            if aid in self._adapters
        ]

        if not candidates:
            return None

        # Filter to READY agents
        ready = [c for c in candidates if c.get_unified_status() == UnifiedAgentStatus.READY]
        if ready:
            candidates = ready

        # Prefer specified subsystem
        if prefer_subsystem:
            subsys_match = [c for c in candidates if c.subsystem == prefer_subsystem]
            if subsys_match:
                candidates = subsys_match

        # Prefer local agents (Core > Legacy Whisperer > BPO > External)
        if prefer_local:
            local_priority = {
                AgentSubsystem.CORE: 0,
                AgentSubsystem.LEGACY_WHISPERER: 1,
                AgentSubsystem.BPO: 2,
                AgentSubsystem.EXTERNAL: 3,
            }
            candidates.sort(key=lambda c: local_priority.get(c.subsystem, 99))

        return candidates[0] if candidates else None

    # ------ Task Routing ------

    async def route_task(
        self,
        task: UnifiedTask,
        target_agent_id: Optional[str] = None,
        prefer_subsystem: Optional[AgentSubsystem] = None,
    ) -> UnifiedResult:
        """
        Route a task to the best available agent.

        Args:
            task: The unified task to execute.
            target_agent_id: Specific agent to target (optional).
            prefer_subsystem: Preferred subsystem (optional).

        Returns:
            UnifiedResult from the executing agent.
        """
        # Find target agent
        if target_agent_id:
            adapter = self.get_agent(target_agent_id)
            if adapter is None:
                return UnifiedResult(
                    task_id=task.task_id,
                    agent_id=target_agent_id,
                    subsystem=AgentSubsystem.CORE,
                    success=False,
                    error=f"Agent {target_agent_id} not found",
                )
        else:
            adapter = self.find_best_agent(
                required_capabilities=task.required_capabilities,
                prefer_subsystem=prefer_subsystem,
            )
            if adapter is None:
                return UnifiedResult(
                    task_id=task.task_id,
                    agent_id="none",
                    subsystem=AgentSubsystem.CORE,
                    success=False,
                    error=(
                        f"No agent found with capabilities: "
                        f"{[c.value for c in task.required_capabilities]}"
                    ),
                )

        # Track cross-system calls
        if task.source_subsystem and task.source_subsystem != adapter.subsystem:
            UNIFIED_CROSS_SYSTEM_CALLS.labels(
                source_subsystem=task.source_subsystem.value,
                target_subsystem=adapter.subsystem.value,
            ).inc()

        # Execute
        self.logger.info(
            f"Routing task {task.task_id[:8]} to {adapter.agent_type} "
            f"({adapter.subsystem.value})"
        )
        result = await adapter.execute_unified_task(task)

        # Track metrics
        status = "success" if result.success else "failure"
        UNIFIED_AGENT_TASKS.labels(
            subsystem=adapter.subsystem.value,
            agent_type=adapter.agent_type,
            status=status,
        ).inc()

        return result

    # ------ Health Dashboard ------

    def get_dashboard(self) -> Dict[str, Any]:
        """Get unified health dashboard across all subsystems."""
        total = len(self._adapters)
        by_subsystem = {}

        for subsystem in AgentSubsystem:
            agents = self.get_agents_by_subsystem(subsystem)
            if agents:
                ready = sum(
                    1 for a in agents
                    if a.get_unified_status() == UnifiedAgentStatus.READY
                )
                busy = sum(
                    1 for a in agents
                    if a.get_unified_status() == UnifiedAgentStatus.BUSY
                )
                error = sum(
                    1 for a in agents
                    if a.get_unified_status() == UnifiedAgentStatus.ERROR
                )
                by_subsystem[subsystem.value] = {
                    "total": len(agents),
                    "ready": ready,
                    "busy": busy,
                    "error": error,
                    "agents": [
                        {
                            "id": a.agent_id[:8],
                            "type": a.agent_type,
                            "status": a.get_unified_status().value,
                            "capabilities": [c.value for c in a.capabilities],
                        }
                        for a in agents
                    ],
                }

        # Capability coverage
        capability_coverage = {}
        for cap in UnifiedCapability:
            agents = self.get_agents_by_capability(cap)
            if agents:
                capability_coverage[cap.value] = len(agents)

        return {
            "total_agents": total,
            "subsystems": by_subsystem,
            "capability_coverage": capability_coverage,
            "health": (
                "healthy"
                if all(
                    a.get_unified_status() != UnifiedAgentStatus.ERROR
                    for a in self._adapters.values()
                )
                else "degraded"
            ),
        }

    def __len__(self) -> int:
        return len(self._adapters)
