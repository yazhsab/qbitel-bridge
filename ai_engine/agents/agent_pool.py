"""
QBITEL - Dynamic Agent Pool Manager

Provides dynamic agent lifecycle management with:
- On-demand agent spawning
- Agent pool scaling (up/down)
- Load balancing across agents
- Agent health monitoring
- Resource-aware scheduling
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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
    TaskResult,
)

if TYPE_CHECKING:
    from .agent_communication import AgentCommunicationProtocol
    from .agent_memory import AgentMemoryManager

# Prometheus metrics
POOL_AGENTS_TOTAL = Gauge(
    "qbitel_pool_agents_total",
    "Total agents in pool",
    ["agent_type", "status"],
)
POOL_SCALING_EVENTS = Counter(
    "qbitel_pool_scaling_events_total",
    "Total scaling events",
    ["agent_type", "direction"],
)
POOL_TASK_QUEUE = Gauge(
    "qbitel_pool_task_queue_size",
    "Size of pool task queue",
    ["agent_type"],
)

logger = logging.getLogger(__name__)


class ScalingPolicy(str, Enum):
    """Agent pool scaling policies."""

    FIXED = "fixed"  # Fixed number of agents
    AUTO = "auto"  # Auto-scale based on load
    THRESHOLD = "threshold"  # Scale at specific thresholds
    SCHEDULED = "scheduled"  # Scale based on time/schedule
    PREDICTIVE = "predictive"  # ML-based predictive scaling


class LoadBalancingStrategy(str, Enum):
    """Strategy for distributing tasks across agents."""

    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    RANDOM = "random"
    CAPABILITY_MATCH = "capability_match"
    WEIGHTED = "weighted"


@dataclass
class PoolConfig:
    """Configuration for an agent pool."""

    agent_type: str
    agent_class: Type[BaseAgent]
    agent_config: AgentConfig
    min_agents: int = 1
    max_agents: int = 10
    initial_agents: int = 2
    scaling_policy: ScalingPolicy = ScalingPolicy.AUTO
    load_balancing: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_LOADED
    scale_up_threshold: float = 0.8  # Scale up when load > 80%
    scale_down_threshold: float = 0.3  # Scale down when load < 30%
    scale_cooldown_seconds: int = 60  # Minimum time between scaling events
    idle_timeout_seconds: int = 300  # Time before idle agent is removed
    health_check_interval: int = 30  # Health check interval
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentInstance:
    """Represents a running agent instance in the pool."""

    instance_id: str
    agent: BaseAgent
    agent_type: str
    pool_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_task_at: Optional[datetime] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_processing_time: float = 0.0
    is_warm: bool = False  # Whether agent is warmed up and ready

    @property
    def success_rate(self) -> float:
        """Calculate task success rate."""
        total = self.tasks_completed + self.tasks_failed
        return self.tasks_completed / total if total > 0 else 1.0

    @property
    def average_task_time(self) -> float:
        """Calculate average task processing time."""
        return (
            self.total_processing_time / self.tasks_completed
            if self.tasks_completed > 0
            else 0.0
        )

    @property
    def idle_time(self) -> float:
        """Time since last task in seconds."""
        if self.last_task_at:
            return (datetime.utcnow() - self.last_task_at).total_seconds()
        return (datetime.utcnow() - self.created_at).total_seconds()


@dataclass
class PoolStats:
    """Statistics for an agent pool."""

    pool_id: str
    agent_type: str
    total_agents: int
    active_agents: int
    idle_agents: int
    total_tasks_queued: int
    total_tasks_completed: int
    total_tasks_failed: int
    average_response_time: float
    current_load: float  # 0.0 to 1.0
    last_scale_event: Optional[datetime] = None
    scale_up_count: int = 0
    scale_down_count: int = 0


class AgentPool:
    """
    Manages a pool of agents of a specific type.

    Provides:
    - Dynamic scaling based on load
    - Load balancing across agents
    - Health monitoring
    - Task queuing and distribution
    """

    def __init__(
        self,
        config: PoolConfig,
        communication: Optional["AgentCommunicationProtocol"] = None,
        memory: Optional["AgentMemoryManager"] = None,
    ):
        """Initialize the agent pool."""
        self.pool_id = str(uuid.uuid4())
        self.config = config
        self.communication = communication
        self.memory = memory

        # Agent instances
        self.instances: Dict[str, AgentInstance] = {}

        # Task queue for the pool
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.pending_tasks: Dict[str, AgentTask] = {}

        # Scaling tracking
        self.last_scale_event: Optional[datetime] = None
        self.scale_up_count = 0
        self.scale_down_count = 0

        # Round-robin tracking
        self._round_robin_index = 0

        # Background tasks
        self._running = False
        self._task_dispatcher: Optional[asyncio.Task] = None
        self._health_checker: Optional[asyncio.Task] = None
        self._scaler: Optional[asyncio.Task] = None

        # Statistics
        self.total_tasks_completed = 0
        self.total_tasks_failed = 0
        self.total_response_time = 0.0

        self.logger = logging.getLogger(
            f"{__name__}.Pool.{config.agent_type}.{self.pool_id[:8]}"
        )

    async def start(self) -> None:
        """Start the agent pool."""
        if self._running:
            return

        self._running = True
        self.logger.info(f"Starting agent pool: {self.config.agent_type}")

        # Spawn initial agents
        for _ in range(self.config.initial_agents):
            await self._spawn_agent()

        # Start background tasks
        self._task_dispatcher = asyncio.create_task(self._dispatch_tasks())
        self._health_checker = asyncio.create_task(self._health_check_loop())
        self._scaler = asyncio.create_task(self._scaling_loop())

        self.logger.info(
            f"Agent pool started with {len(self.instances)} agents"
        )

    async def stop(self) -> None:
        """Stop the agent pool."""
        if not self._running:
            return

        self._running = False
        self.logger.info(f"Stopping agent pool: {self.config.agent_type}")

        # Cancel background tasks
        for task in [self._task_dispatcher, self._health_checker, self._scaler]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Stop all agents
        for instance in list(self.instances.values()):
            await self._terminate_agent(instance.instance_id)

        self.logger.info("Agent pool stopped")

    async def submit_task(
        self,
        task: AgentTask,
        required_capabilities: Optional[List[AgentCapability]] = None,
    ) -> Any:
        """
        Submit a task to the pool for execution.

        Args:
            task: The task to execute
            required_capabilities: Optional list of required capabilities

        Returns:
            Task result
        """
        # Create a future for the result
        future = asyncio.get_event_loop().create_future()
        task.metadata["_future"] = future
        task.metadata["_required_capabilities"] = required_capabilities or []

        # Add to queue
        await self.task_queue.put(task)
        self.pending_tasks[task.task_id] = task

        POOL_TASK_QUEUE.labels(agent_type=self.config.agent_type).set(
            self.task_queue.qsize()
        )

        # Wait for result
        try:
            result = await future
            return result
        except Exception as e:
            self.logger.error(f"Task failed: {task.task_id} - {e}")
            raise

    async def _dispatch_tasks(self) -> None:
        """Dispatch tasks from queue to available agents."""
        while self._running:
            try:
                # Get task from queue with timeout
                try:
                    task = await asyncio.wait_for(
                        self.task_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Find suitable agent
                required_caps = task.metadata.get("_required_capabilities", [])
                agent_instance = await self._select_agent(required_caps)

                if not agent_instance:
                    # No agent available, put back in queue
                    await self.task_queue.put(task)
                    await asyncio.sleep(0.1)
                    continue

                # Submit task to agent
                asyncio.create_task(
                    self._execute_task_on_agent(agent_instance, task)
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Task dispatch error: {e}")
                await asyncio.sleep(0.1)

    async def _execute_task_on_agent(
        self,
        instance: AgentInstance,
        task: AgentTask
    ) -> None:
        """Execute a task on a specific agent instance."""
        start_time = time.time()
        future = task.metadata.get("_future")

        try:
            # Submit to agent
            await instance.agent.submit_task(task)

            # Wait for completion (poll agent's completed tasks)
            result = await self._wait_for_task_completion(instance, task)

            # Update instance stats
            execution_time = time.time() - start_time
            instance.tasks_completed += 1
            instance.total_processing_time += execution_time
            instance.last_task_at = datetime.utcnow()

            # Update pool stats
            self.total_tasks_completed += 1
            self.total_response_time += execution_time

            # Set result on future
            if future and not future.done():
                future.set_result(result)

        except Exception as e:
            instance.tasks_failed += 1
            self.total_tasks_failed += 1
            self.logger.error(f"Task execution failed: {task.task_id} - {e}")

            if future and not future.done():
                future.set_exception(e)

        finally:
            # Remove from pending
            if task.task_id in self.pending_tasks:
                del self.pending_tasks[task.task_id]

    async def _wait_for_task_completion(
        self,
        instance: AgentInstance,
        task: AgentTask,
        timeout: float = 300.0
    ) -> Any:
        """Wait for a task to complete on an agent."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check if task is in completed list
            for result in instance.agent.completed_tasks:
                if result.task_id == task.task_id:
                    if result.success:
                        return result.result
                    else:
                        raise RuntimeError(result.error or "Task failed")

            # Check if task is still active
            if task.task_id in instance.agent.active_tasks:
                await asyncio.sleep(0.1)
                continue

            # Task not found anywhere - might have completed
            await asyncio.sleep(0.1)

        raise asyncio.TimeoutError(f"Task {task.task_id} timed out")

    async def _select_agent(
        self,
        required_capabilities: List[AgentCapability]
    ) -> Optional[AgentInstance]:
        """Select the best agent for a task based on load balancing strategy."""
        # Get eligible agents
        eligible = []
        for instance in self.instances.values():
            # Check status
            if instance.agent.status not in [AgentStatus.IDLE, AgentStatus.BUSY]:
                continue

            # Check capabilities
            if required_capabilities:
                if not all(
                    instance.agent.has_capability(cap)
                    for cap in required_capabilities
                ):
                    continue

            # Check capacity
            if instance.agent.task_queue.qsize() >= instance.agent.max_concurrent_tasks * 2:
                continue

            eligible.append(instance)

        if not eligible:
            return None

        # Apply load balancing strategy
        strategy = self.config.load_balancing

        if strategy == LoadBalancingStrategy.ROUND_ROBIN:
            self._round_robin_index = (self._round_robin_index + 1) % len(eligible)
            return eligible[self._round_robin_index]

        elif strategy == LoadBalancingStrategy.LEAST_LOADED:
            return min(eligible, key=lambda i: i.agent.task_queue.qsize())

        elif strategy == LoadBalancingStrategy.RANDOM:
            import random
            return random.choice(eligible)

        elif strategy == LoadBalancingStrategy.CAPABILITY_MATCH:
            # Prefer agents with exact capability match
            if required_capabilities:
                exact_match = [
                    i for i in eligible
                    if set(required_capabilities).issubset(i.agent.capabilities)
                ]
                if exact_match:
                    return min(exact_match, key=lambda i: i.agent.task_queue.qsize())
            return min(eligible, key=lambda i: i.agent.task_queue.qsize())

        elif strategy == LoadBalancingStrategy.WEIGHTED:
            # Weight by success rate and response time
            def score(instance: AgentInstance) -> float:
                load = instance.agent.task_queue.qsize() / max(
                    instance.agent.max_concurrent_tasks, 1
                )
                return (1 - instance.success_rate) + load + (instance.average_task_time / 100)

            return min(eligible, key=score)

        return eligible[0] if eligible else None

    async def _spawn_agent(self) -> Optional[AgentInstance]:
        """Spawn a new agent instance."""
        if len(self.instances) >= self.config.max_agents:
            self.logger.warning(
                f"Cannot spawn agent: max agents ({self.config.max_agents}) reached"
            )
            return None

        try:
            # Create agent
            agent = self.config.agent_class(
                config=self.config.agent_config,
                communication=self.communication,
                memory=self.memory,
            )

            # Start agent
            await agent.start()

            # Create instance tracking
            instance = AgentInstance(
                instance_id=str(uuid.uuid4()),
                agent=agent,
                agent_type=self.config.agent_type,
                pool_id=self.pool_id,
            )

            self.instances[instance.instance_id] = instance

            # Update metrics
            POOL_AGENTS_TOTAL.labels(
                agent_type=self.config.agent_type,
                status="active"
            ).inc()

            self.logger.info(
                f"Spawned agent: {instance.instance_id[:8]} "
                f"(total: {len(self.instances)})"
            )

            return instance

        except Exception as e:
            self.logger.error(f"Failed to spawn agent: {e}")
            return None

    async def _terminate_agent(self, instance_id: str) -> bool:
        """Terminate an agent instance."""
        instance = self.instances.get(instance_id)
        if not instance:
            return False

        if len(self.instances) <= self.config.min_agents:
            self.logger.warning(
                f"Cannot terminate agent: min agents ({self.config.min_agents}) reached"
            )
            return False

        try:
            # Stop the agent
            await instance.agent.stop()

            # Remove from pool
            del self.instances[instance_id]

            # Update metrics
            POOL_AGENTS_TOTAL.labels(
                agent_type=self.config.agent_type,
                status="active"
            ).dec()

            self.logger.info(
                f"Terminated agent: {instance_id[:8]} "
                f"(total: {len(self.instances)})"
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to terminate agent {instance_id}: {e}")
            return False

    async def _health_check_loop(self) -> None:
        """Periodically check agent health."""
        while self._running:
            try:
                unhealthy = []

                for instance_id, instance in self.instances.items():
                    # Check if agent is responsive
                    if instance.agent.status == AgentStatus.ERROR:
                        unhealthy.append(instance_id)
                        continue

                    # Check heartbeat
                    if instance.agent.last_heartbeat:
                        heartbeat_age = (
                            datetime.utcnow() - instance.agent.last_heartbeat
                        ).total_seconds()
                        if heartbeat_age > self.config.health_check_interval * 3:
                            unhealthy.append(instance_id)

                # Replace unhealthy agents
                for instance_id in unhealthy:
                    self.logger.warning(f"Replacing unhealthy agent: {instance_id[:8]}")
                    await self._terminate_agent(instance_id)
                    await self._spawn_agent()

                await asyncio.sleep(self.config.health_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                await asyncio.sleep(self.config.health_check_interval)

    async def _scaling_loop(self) -> None:
        """Auto-scale the pool based on load."""
        if self.config.scaling_policy == ScalingPolicy.FIXED:
            return

        while self._running:
            try:
                current_load = self._calculate_load()

                # Check cooldown
                if self.last_scale_event:
                    cooldown_remaining = (
                        datetime.utcnow() - self.last_scale_event
                    ).total_seconds()
                    if cooldown_remaining < self.config.scale_cooldown_seconds:
                        await asyncio.sleep(10)
                        continue

                # Scale up
                if current_load > self.config.scale_up_threshold:
                    if len(self.instances) < self.config.max_agents:
                        agents_to_add = min(
                            2,  # Add up to 2 at a time
                            self.config.max_agents - len(self.instances)
                        )
                        for _ in range(agents_to_add):
                            await self._spawn_agent()

                        self.last_scale_event = datetime.utcnow()
                        self.scale_up_count += 1
                        POOL_SCALING_EVENTS.labels(
                            agent_type=self.config.agent_type,
                            direction="up"
                        ).inc()
                        self.logger.info(
                            f"Scaled up pool: load={current_load:.2f}, "
                            f"agents={len(self.instances)}"
                        )

                # Scale down
                elif current_load < self.config.scale_down_threshold:
                    if len(self.instances) > self.config.min_agents:
                        # Find idle agents to terminate
                        idle_agents = [
                            instance for instance in self.instances.values()
                            if (
                                instance.agent.status == AgentStatus.IDLE and
                                instance.idle_time > self.config.idle_timeout_seconds
                            )
                        ]

                        if idle_agents:
                            # Terminate oldest idle agent
                            oldest = min(idle_agents, key=lambda i: i.created_at)
                            await self._terminate_agent(oldest.instance_id)

                            self.last_scale_event = datetime.utcnow()
                            self.scale_down_count += 1
                            POOL_SCALING_EVENTS.labels(
                                agent_type=self.config.agent_type,
                                direction="down"
                            ).inc()
                            self.logger.info(
                                f"Scaled down pool: load={current_load:.2f}, "
                                f"agents={len(self.instances)}"
                            )

                await asyncio.sleep(10)  # Check every 10 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Scaling error: {e}")
                await asyncio.sleep(10)

    def _calculate_load(self) -> float:
        """Calculate current pool load (0.0 to 1.0)."""
        if not self.instances:
            return 0.0

        total_capacity = sum(
            i.agent.max_concurrent_tasks for i in self.instances.values()
        )
        total_load = sum(
            i.agent.task_queue.qsize() + len(i.agent.active_tasks)
            for i in self.instances.values()
        )

        return total_load / total_capacity if total_capacity > 0 else 0.0

    def get_stats(self) -> PoolStats:
        """Get pool statistics."""
        active = sum(
            1 for i in self.instances.values()
            if i.agent.status == AgentStatus.BUSY
        )
        idle = sum(
            1 for i in self.instances.values()
            if i.agent.status == AgentStatus.IDLE
        )

        avg_response = (
            self.total_response_time / self.total_tasks_completed
            if self.total_tasks_completed > 0
            else 0.0
        )

        return PoolStats(
            pool_id=self.pool_id,
            agent_type=self.config.agent_type,
            total_agents=len(self.instances),
            active_agents=active,
            idle_agents=idle,
            total_tasks_queued=self.task_queue.qsize(),
            total_tasks_completed=self.total_tasks_completed,
            total_tasks_failed=self.total_tasks_failed,
            average_response_time=avg_response,
            current_load=self._calculate_load(),
            last_scale_event=self.last_scale_event,
            scale_up_count=self.scale_up_count,
            scale_down_count=self.scale_down_count,
        )


class AgentPoolManager:
    """
    Manages multiple agent pools.

    Provides:
    - Pool lifecycle management
    - Cross-pool task routing
    - Global resource management
    - Unified monitoring
    """

    def __init__(
        self,
        communication: Optional["AgentCommunicationProtocol"] = None,
        memory: Optional["AgentMemoryManager"] = None,
    ):
        """Initialize the pool manager."""
        self.pools: Dict[str, AgentPool] = {}
        self.communication = communication
        self.memory = memory
        self._running = False

        self.logger = logging.getLogger(f"{__name__}.PoolManager")

    async def start(self) -> None:
        """Start the pool manager."""
        self._running = True
        for pool in self.pools.values():
            await pool.start()
        self.logger.info(f"Pool manager started with {len(self.pools)} pools")

    async def stop(self) -> None:
        """Stop the pool manager."""
        self._running = False
        for pool in self.pools.values():
            await pool.stop()
        self.logger.info("Pool manager stopped")

    def register_pool(self, config: PoolConfig) -> AgentPool:
        """Register and create a new agent pool."""
        pool = AgentPool(
            config=config,
            communication=self.communication,
            memory=self.memory,
        )
        self.pools[config.agent_type] = pool
        self.logger.info(f"Registered pool: {config.agent_type}")
        return pool

    def get_pool(self, agent_type: str) -> Optional[AgentPool]:
        """Get a pool by agent type."""
        return self.pools.get(agent_type)

    async def submit_task(
        self,
        task: AgentTask,
        agent_type: Optional[str] = None,
        required_capabilities: Optional[List[AgentCapability]] = None,
    ) -> Any:
        """
        Submit a task to an appropriate pool.

        Args:
            task: The task to execute
            agent_type: Optional specific agent type
            required_capabilities: Required capabilities

        Returns:
            Task result
        """
        # Find appropriate pool
        if agent_type and agent_type in self.pools:
            pool = self.pools[agent_type]
        elif required_capabilities:
            pool = self._find_pool_with_capabilities(required_capabilities)
        else:
            # Use first available pool
            pool = next(iter(self.pools.values())) if self.pools else None

        if not pool:
            raise ValueError("No suitable pool found for task")

        return await pool.submit_task(task, required_capabilities)

    def _find_pool_with_capabilities(
        self,
        capabilities: List[AgentCapability]
    ) -> Optional[AgentPool]:
        """Find a pool that has agents with the required capabilities."""
        for pool in self.pools.values():
            pool_caps = set(pool.config.agent_config.capabilities)
            if all(cap in pool_caps for cap in capabilities):
                return pool
        return None

    async def get_agents_with_capabilities(
        self,
        capabilities: List[AgentCapability]
    ) -> List[Dict[str, Any]]:
        """Get all agents with the specified capabilities."""
        agents = []
        for pool in self.pools.values():
            for instance in pool.instances.values():
                if all(instance.agent.has_capability(cap) for cap in capabilities):
                    agents.append({
                        "agent_id": instance.agent.agent_id,
                        "agent_type": instance.agent_type,
                        "capabilities": [c.value for c in instance.agent.capabilities],
                        "status": instance.agent.status.value,
                        "queue_size": instance.agent.task_queue.qsize(),
                        "success_rate": instance.success_rate,
                    })
        return agents

    async def get_status(self) -> Dict[str, Any]:
        """Get status of all pools."""
        pool_stats = {}
        for agent_type, pool in self.pools.items():
            stats = pool.get_stats()
            pool_stats[agent_type] = {
                "pool_id": stats.pool_id,
                "total_agents": stats.total_agents,
                "active_agents": stats.active_agents,
                "idle_agents": stats.idle_agents,
                "current_load": stats.current_load,
                "tasks_queued": stats.total_tasks_queued,
                "tasks_completed": stats.total_tasks_completed,
                "tasks_failed": stats.total_tasks_failed,
            }

        total_agents = sum(s["total_agents"] for s in pool_stats.values())
        total_active = sum(s["active_agents"] for s in pool_stats.values())

        return {
            "total_pools": len(self.pools),
            "total_agents": total_agents,
            "total_active": total_active,
            "pools": pool_stats,
            "agents": await self._get_all_agents(),
        }

    async def _get_all_agents(self) -> List[Dict[str, Any]]:
        """Get list of all agents across all pools."""
        agents = []
        for pool in self.pools.values():
            for instance in pool.instances.values():
                agents.append({
                    "agent_id": instance.agent.agent_id,
                    "agent_type": instance.agent_type,
                    "pool_id": pool.pool_id,
                    "status": instance.agent.status.value,
                    "capabilities": [c.value for c in instance.agent.capabilities],
                    "tasks_completed": instance.tasks_completed,
                    "success_rate": instance.success_rate,
                })
        return agents
