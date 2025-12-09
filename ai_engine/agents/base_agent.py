"""
CRONOS AI - Base Agent Implementation

Provides the foundational agent class that all specialized agents inherit from.
Implements core agent lifecycle, capability management, and communication interfaces.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

from prometheus_client import Counter, Histogram, Gauge

if TYPE_CHECKING:
    from .agent_communication import AgentCommunicationProtocol, AgentMessage
    from .agent_memory import AgentMemoryManager

# Prometheus metrics
AGENT_TASKS_COUNTER = Counter(
    "cronos_agent_tasks_total",
    "Total tasks processed by agents",
    ["agent_type", "agent_id", "status"],
)
AGENT_TASK_DURATION = Histogram(
    "cronos_agent_task_duration_seconds",
    "Agent task processing duration",
    ["agent_type", "task_type"],
)
AGENT_ACTIVE_COUNT = Gauge(
    "cronos_agent_active_count",
    "Number of active agents",
    ["agent_type"],
)

logger = logging.getLogger(__name__)


class AgentCapability(str, Enum):
    """Agent capabilities defining what an agent can do."""

    # Analysis capabilities
    THREAT_ANALYSIS = "threat_analysis"
    PROTOCOL_ANALYSIS = "protocol_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    COMPLIANCE_CHECK = "compliance_check"

    # Response capabilities
    INCIDENT_RESPONSE = "incident_response"
    POLICY_ENFORCEMENT = "policy_enforcement"
    THREAT_MITIGATION = "threat_mitigation"

    # Learning capabilities
    PATTERN_LEARNING = "pattern_learning"
    PROTOCOL_DISCOVERY = "protocol_discovery"
    BEHAVIORAL_MODELING = "behavioral_modeling"

    # Coordination capabilities
    TASK_PLANNING = "task_planning"
    AGENT_COORDINATION = "agent_coordination"
    WORKFLOW_ORCHESTRATION = "workflow_orchestration"

    # Communication capabilities
    NATURAL_LANGUAGE = "natural_language"
    REPORT_GENERATION = "report_generation"
    ALERT_GENERATION = "alert_generation"

    # Specialized capabilities
    LEGACY_SYSTEM_ANALYSIS = "legacy_system_analysis"
    QUANTUM_CRYPTOGRAPHY = "quantum_cryptography"
    SBOM_ANALYSIS = "sbom_analysis"


class AgentStatus(str, Enum):
    """Agent operational status."""

    INITIALIZING = "initializing"
    IDLE = "idle"
    BUSY = "busy"
    PAUSED = "paused"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    TERMINATED = "terminated"


class AgentPriority(int, Enum):
    """Agent priority levels for task assignment."""

    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class AgentConfig:
    """Configuration for agent initialization."""

    agent_type: str
    capabilities: List[AgentCapability]
    max_concurrent_tasks: int = 5
    task_timeout_seconds: int = 300
    heartbeat_interval_seconds: int = 10
    memory_enabled: bool = True
    learning_enabled: bool = True
    priority: AgentPriority = AgentPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Result of a task execution."""

    task_id: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTask:
    """Task to be executed by an agent."""

    task_id: str
    task_type: str
    payload: Dict[str, Any]
    priority: AgentPriority = AgentPriority.NORMAL
    timeout: Optional[float] = None
    callback: Optional[Callable] = None
    correlation_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Base class for all CRONOS AI agents.

    Provides core functionality including:
    - Lifecycle management (start, stop, pause, resume)
    - Task queue management
    - Communication interface
    - Memory interface
    - Health monitoring
    - Metrics collection
    """

    def __init__(
        self,
        config: AgentConfig,
        communication: Optional["AgentCommunicationProtocol"] = None,
        memory: Optional["AgentMemoryManager"] = None,
    ):
        """Initialize the base agent."""
        self.agent_id = str(uuid.uuid4())
        self.config = config
        self.agent_type = config.agent_type
        self.capabilities: Set[AgentCapability] = set(config.capabilities)
        self.status = AgentStatus.INITIALIZING
        self.priority = config.priority

        # Communication and memory
        self.communication = communication
        self.memory = memory

        # Task management
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.active_tasks: Dict[str, AgentTask] = {}
        self.completed_tasks: List[TaskResult] = []
        self.max_concurrent_tasks = config.max_concurrent_tasks

        # Lifecycle management
        self._running = False
        self._paused = False
        self._task_semaphore = asyncio.Semaphore(config.max_concurrent_tasks)
        self._shutdown_event = asyncio.Event()

        # Statistics
        self.stats = {
            "tasks_received": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "last_activity": None,
        }

        # Timestamps
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.last_heartbeat: Optional[datetime] = None

        self.logger = logging.getLogger(f"{__name__}.{self.agent_type}.{self.agent_id[:8]}")
        self.logger.info(f"Agent initialized: {self.agent_type} ({self.agent_id[:8]})")

    async def start(self) -> None:
        """Start the agent and begin processing tasks."""
        if self._running:
            self.logger.warning("Agent already running")
            return

        self.logger.info(f"Starting agent: {self.agent_type}")
        self._running = True
        self.started_at = datetime.utcnow()
        self.status = AgentStatus.IDLE

        # Update metrics
        AGENT_ACTIVE_COUNT.labels(agent_type=self.agent_type).inc()

        # Register with communication protocol
        if self.communication:
            await self.communication.register_agent(self)

        # Start background tasks
        asyncio.create_task(self._task_processor())
        asyncio.create_task(self._heartbeat_loop())

        # Call agent-specific initialization
        await self._on_start()

        self.logger.info(f"Agent started: {self.agent_type}")

    async def stop(self) -> None:
        """Stop the agent gracefully."""
        if not self._running:
            return

        self.logger.info(f"Stopping agent: {self.agent_type}")
        self.status = AgentStatus.SHUTTING_DOWN
        self._running = False

        # Signal shutdown
        self._shutdown_event.set()

        # Wait for active tasks to complete (with timeout)
        if self.active_tasks:
            self.logger.info(f"Waiting for {len(self.active_tasks)} active tasks to complete")
            try:
                await asyncio.wait_for(self._wait_for_tasks(), timeout=30.0)
            except asyncio.TimeoutError:
                self.logger.warning("Timeout waiting for tasks, forcing shutdown")

        # Unregister from communication protocol
        if self.communication:
            await self.communication.unregister_agent(self.agent_id)

        # Call agent-specific cleanup
        await self._on_stop()

        self.status = AgentStatus.TERMINATED
        AGENT_ACTIVE_COUNT.labels(agent_type=self.agent_type).dec()
        self.logger.info(f"Agent stopped: {self.agent_type}")

    async def pause(self) -> None:
        """Pause task processing."""
        self._paused = True
        self.status = AgentStatus.PAUSED
        self.logger.info(f"Agent paused: {self.agent_type}")

    async def resume(self) -> None:
        """Resume task processing."""
        self._paused = False
        self.status = AgentStatus.IDLE if not self.active_tasks else AgentStatus.BUSY
        self.logger.info(f"Agent resumed: {self.agent_type}")

    async def submit_task(self, task: AgentTask) -> str:
        """Submit a task to the agent's queue."""
        self.stats["tasks_received"] += 1
        await self.task_queue.put(task)
        self.logger.debug(f"Task submitted: {task.task_id}")
        return task.task_id

    async def _task_processor(self) -> None:
        """Main task processing loop."""
        while self._running:
            try:
                # Wait for shutdown or task
                if self._shutdown_event.is_set():
                    break

                # Check if paused
                if self._paused:
                    await asyncio.sleep(0.1)
                    continue

                # Get task with timeout
                try:
                    task = await asyncio.wait_for(
                        self.task_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Acquire semaphore for concurrency control
                async with self._task_semaphore:
                    self.status = AgentStatus.BUSY
                    await self._execute_task(task)

                # Update status
                if not self.active_tasks and not self._paused:
                    self.status = AgentStatus.IDLE

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in task processor: {e}")
                self.status = AgentStatus.ERROR

    async def _execute_task(self, task: AgentTask) -> TaskResult:
        """Execute a single task."""
        self.active_tasks[task.task_id] = task
        start_time = time.time()

        try:
            self.logger.debug(f"Executing task: {task.task_id} ({task.task_type})")

            # Execute with timeout
            timeout = task.timeout or self.config.task_timeout_seconds
            result = await asyncio.wait_for(
                self.execute(task),
                timeout=timeout
            )

            execution_time = time.time() - start_time

            task_result = TaskResult(
                task_id=task.task_id,
                success=True,
                result=result,
                execution_time=execution_time,
                metadata={"correlation_id": task.correlation_id}
            )

            # Update statistics
            self.stats["tasks_completed"] += 1
            self.stats["total_processing_time"] += execution_time
            self.stats["average_processing_time"] = (
                self.stats["total_processing_time"] / self.stats["tasks_completed"]
            )
            self.stats["last_activity"] = datetime.utcnow().isoformat()

            # Update metrics
            AGENT_TASKS_COUNTER.labels(
                agent_type=self.agent_type,
                agent_id=self.agent_id[:8],
                status="success"
            ).inc()
            AGENT_TASK_DURATION.labels(
                agent_type=self.agent_type,
                task_type=task.task_type
            ).observe(execution_time)

            self.logger.debug(f"Task completed: {task.task_id} in {execution_time:.2f}s")

            # Store in memory if enabled
            if self.memory and self.config.memory_enabled:
                await self._store_task_memory(task, task_result)

            # Execute callback if provided
            if task.callback:
                try:
                    await task.callback(task_result)
                except Exception as e:
                    self.logger.error(f"Task callback error: {e}")

            return task_result

        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            task_result = TaskResult(
                task_id=task.task_id,
                success=False,
                result=None,
                error=f"Task timeout after {execution_time:.2f}s",
                execution_time=execution_time
            )
            self.stats["tasks_failed"] += 1
            AGENT_TASKS_COUNTER.labels(
                agent_type=self.agent_type,
                agent_id=self.agent_id[:8],
                status="timeout"
            ).inc()
            self.logger.warning(f"Task timeout: {task.task_id}")
            return task_result

        except Exception as e:
            execution_time = time.time() - start_time
            task_result = TaskResult(
                task_id=task.task_id,
                success=False,
                result=None,
                error=str(e),
                execution_time=execution_time
            )
            self.stats["tasks_failed"] += 1
            AGENT_TASKS_COUNTER.labels(
                agent_type=self.agent_type,
                agent_id=self.agent_id[:8],
                status="error"
            ).inc()
            self.logger.error(f"Task failed: {task.task_id} - {e}")
            return task_result

        finally:
            del self.active_tasks[task.task_id]
            self.completed_tasks.append(task_result)
            # Keep only last 1000 completed tasks
            if len(self.completed_tasks) > 1000:
                self.completed_tasks = self.completed_tasks[-1000:]

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats."""
        while self._running:
            try:
                self.last_heartbeat = datetime.utcnow()

                # Send heartbeat via communication protocol
                if self.communication:
                    await self.communication.send_heartbeat(self)

                await asyncio.sleep(self.config.heartbeat_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(self.config.heartbeat_interval_seconds)

    async def _wait_for_tasks(self) -> None:
        """Wait for all active tasks to complete."""
        while self.active_tasks:
            await asyncio.sleep(0.1)

    async def _store_task_memory(self, task: AgentTask, result: TaskResult) -> None:
        """Store task execution in memory for learning."""
        if not self.memory:
            return

        memory_entry = {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "payload_summary": self._summarize_payload(task.payload),
            "success": result.success,
            "execution_time": result.execution_time,
            "result_summary": self._summarize_result(result.result) if result.success else result.error,
            "timestamp": datetime.utcnow().isoformat(),
        }

        await self.memory.store_episodic(
            agent_id=self.agent_id,
            event_type="task_execution",
            content=memory_entry
        )

    def _summarize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of task payload for memory storage."""
        # Truncate large values
        summary = {}
        for key, value in payload.items():
            if isinstance(value, str) and len(value) > 500:
                summary[key] = value[:500] + "..."
            elif isinstance(value, (list, dict)) and len(str(value)) > 500:
                summary[key] = f"<{type(value).__name__} with {len(value)} items>"
            else:
                summary[key] = value
        return summary

    def _summarize_result(self, result: Any) -> Any:
        """Create a summary of task result for memory storage."""
        if result is None:
            return None
        if isinstance(result, str) and len(result) > 500:
            return result[:500] + "..."
        if isinstance(result, dict):
            return self._summarize_payload(result)
        return result

    def has_capability(self, capability: AgentCapability) -> bool:
        """Check if agent has a specific capability."""
        return capability in self.capabilities

    def get_status(self) -> Dict[str, Any]:
        """Get agent status information."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "status": self.status.value,
            "capabilities": [c.value for c in self.capabilities],
            "priority": self.priority.value,
            "queue_size": self.task_queue.qsize(),
            "active_tasks": len(self.active_tasks),
            "statistics": self.stats,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
        }

    # Abstract methods to be implemented by specialized agents

    @abstractmethod
    async def execute(self, task: AgentTask) -> Any:
        """
        Execute a task. Must be implemented by specialized agents.

        Args:
            task: The task to execute

        Returns:
            The result of task execution
        """
        pass

    async def _on_start(self) -> None:
        """Called when agent starts. Override for custom initialization."""
        pass

    async def _on_stop(self) -> None:
        """Called when agent stops. Override for custom cleanup."""
        pass

    async def handle_message(self, message: "AgentMessage") -> Optional[Any]:
        """
        Handle an incoming message from another agent.
        Override for custom message handling.

        Args:
            message: The incoming message

        Returns:
            Optional response to the message
        """
        self.logger.debug(f"Received message: {message.message_type} from {message.sender_id}")
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id[:8]}, type={self.agent_type}, status={self.status.value})"
