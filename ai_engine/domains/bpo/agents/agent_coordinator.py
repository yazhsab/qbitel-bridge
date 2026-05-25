"""
QBITEL - BPO Agent Coordinator

Multi-agent orchestration for BPO/call center security operations.
Coordinates the 8 specialized BPO agents and manages task routing,
parallel execution, and result aggregation.

Integration points:
- ai_engine.agents.multi_agent_orchestrator: MultiAgentOrchestrator
- ai_engine.agents.base_agent: AgentPool, PlanningAgent
- ai_engine.domains.bpo.agents.bpo_agents: All 8 BPO agents
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from prometheus_client import Counter, Histogram, Gauge

from .bpo_agents import (
    BaseBPOAgent,
    BPOAgentConfig,
    BPOAgentType,
    BPOAlert,
    AlertSeverity,
    BPO_AGENT_REGISTRY,
    create_agent,
)

logger = logging.getLogger(__name__)

# Prometheus metrics
COORDINATOR_TASKS = Counter(
    "qbitel_bpo_coordinator_tasks_total",
    "Total tasks coordinated",
    ["strategy", "status"],
)
COORDINATOR_DURATION = Histogram(
    "qbitel_bpo_coordinator_duration_seconds",
    "Coordination task duration",
    ["strategy"],
)
COORDINATOR_AGENTS = Gauge(
    "qbitel_bpo_coordinator_active_agents",
    "Number of active BPO agents",
    ["agent_type"],
)


class CoordinationStrategy(str, Enum):
    """Strategy for multi-agent task coordination."""

    SEQUENTIAL = "sequential"       # Execute tasks one by one
    PARALLEL = "parallel"           # Execute all tasks simultaneously
    PIPELINE = "pipeline"           # Output of one agent feeds into next
    ADAPTIVE = "adaptive"           # Choose strategy based on task analysis
    CONSENSUS = "consensus"         # Multiple agents vote on decision
    BROADCAST = "broadcast"         # Send same task to all agents


@dataclass
class CoordinationTask:
    """Task to be coordinated across multiple agents."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    target_agents: List[str] = field(default_factory=list)
    strategy: CoordinationStrategy = CoordinationStrategy.ADAPTIVE
    timeout_seconds: float = 300.0
    priority: int = 3  # 1=critical, 5=background
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoordinationResult:
    """Result from coordinated multi-agent execution."""
    task_id: str
    strategy: str
    success: bool
    agent_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    aggregated_result: Optional[Dict[str, Any]] = None
    alerts_generated: List[BPOAlert] = field(default_factory=list)
    duration: float = 0.0
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class BPOAgentPool:
    """
    Pool of BPO agents for the coordinator.

    Manages agent lifecycle, health monitoring, and load balancing
    across multiple agent instances.
    """

    def __init__(self):
        """Initialize the agent pool."""
        self.agents: Dict[str, BaseBPOAgent] = {}
        self.agent_type_map: Dict[str, List[str]] = {}
        self.logger = logging.getLogger(f"{__name__}.BPOAgentPool")

    async def initialize_agents(
        self, agent_types: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Initialize BPO agents.

        Args:
            agent_types: Specific agent types to initialize. None = all.

        Returns:
            Dict mapping agent_type to agent_id
        """
        types = agent_types or list(BPO_AGENT_REGISTRY.keys())
        result = {}

        for agent_type in types:
            try:
                agent = create_agent(agent_type)
                await agent.start()
                self.agents[agent.agent_id] = agent

                if agent_type not in self.agent_type_map:
                    self.agent_type_map[agent_type] = []
                self.agent_type_map[agent_type].append(agent.agent_id)

                COORDINATOR_AGENTS.labels(agent_type=agent_type).inc()
                result[agent_type] = agent.agent_id
                self.logger.info(f"Initialized agent: {agent_type} ({agent.agent_id[:8]})")

            except Exception as e:
                self.logger.error(f"Failed to initialize agent {agent_type}: {e}")

        return result

    async def shutdown_all(self) -> None:
        """Shutdown all agents in the pool."""
        for agent_id, agent in list(self.agents.items()):
            try:
                await agent.stop()
                COORDINATOR_AGENTS.labels(agent_type=agent.agent_type).dec()
            except Exception as e:
                self.logger.error(f"Error shutting down agent {agent_id[:8]}: {e}")
        self.agents.clear()
        self.agent_type_map.clear()

    def get_agent(self, agent_type: str) -> Optional[BaseBPOAgent]:
        """Get an agent by type (returns first available)."""
        agent_ids = self.agent_type_map.get(agent_type, [])
        for aid in agent_ids:
            agent = self.agents.get(aid)
            if agent and agent.status in ("idle", "busy"):
                return agent
        return None

    def get_agent_by_id(self, agent_id: str) -> Optional[BaseBPOAgent]:
        """Get an agent by ID."""
        return self.agents.get(agent_id)

    def get_all_agents(self) -> List[BaseBPOAgent]:
        """Get all agents."""
        return list(self.agents.values())

    def get_pool_status(self) -> Dict[str, Any]:
        """Get pool status summary."""
        return {
            "total_agents": len(self.agents),
            "agent_types": {
                agent_type: len(agent_ids)
                for agent_type, agent_ids in self.agent_type_map.items()
            },
            "agents": [
                agent.get_status() for agent in self.agents.values()
            ],
        }


class BPOAgentCoordinator:
    """
    Multi-agent coordinator for BPO security operations.

    Manages task routing, parallel execution, and result aggregation
    across the 8 specialized BPO agents.

    Usage:
        coordinator = BPOAgentCoordinator()
        await coordinator.initialize()
        result = await coordinator.execute(task)
    """

    # Task type to agent type mapping
    TASK_ROUTING: Dict[str, str] = {
        # Call Processing
        "analyze_call": BPOAgentType.CALL_PROCESSING.value,
        "monitor_signaling": BPOAgentType.CALL_PROCESSING.value,
        "detect_anomaly": BPOAgentType.CALL_PROCESSING.value,
        # PCI Compliance
        "mask_dtmf": BPOAgentType.PCI_COMPLIANCE.value,
        "validate_pci": BPOAgentType.PCI_COMPLIANCE.value,
        "enforce_pause_resume": BPOAgentType.PCI_COMPLIANCE.value,
        # Fraud Detection
        "detect_toll_fraud": BPOAgentType.FRAUD_DETECTION.value,
        "analyze_call_pattern": BPOAgentType.FRAUD_DETECTION.value,
        "detect_social_engineering": BPOAgentType.FRAUD_DETECTION.value,
        # CRM
        "mask_screen_data": BPOAgentType.CRM_SCREEN_POP.value,
        "validate_screen_pop": BPOAgentType.CRM_SCREEN_POP.value,
        "audit_crm_access": BPOAgentType.CRM_SCREEN_POP.value,
        # Recording
        "encrypt_recording": BPOAgentType.RECORDING_ENCRYPTION.value,
        "rotate_keys": BPOAgentType.RECORDING_ENCRYPTION.value,
        "verify_encryption": BPOAgentType.RECORDING_ENCRYPTION.value,
        # Session Security
        "monitor_session": BPOAgentType.SESSION_SECURITY.value,
        "detect_dlp_violation": BPOAgentType.SESSION_SECURITY.value,
        "validate_desktop": BPOAgentType.SESSION_SECURITY.value,
        # Remote Access
        "provision_tunnel": BPOAgentType.REMOTE_ACCESS.value,
        "validate_endpoint": BPOAgentType.REMOTE_ACCESS.value,
        "monitor_remote": BPOAgentType.REMOTE_ACCESS.value,
        # Compliance Audit
        "run_audit": BPOAgentType.COMPLIANCE_AUDIT.value,
        "generate_report": BPOAgentType.COMPLIANCE_AUDIT.value,
        "check_policy": BPOAgentType.COMPLIANCE_AUDIT.value,
    }

    def __init__(self):
        """Initialize the BPO agent coordinator."""
        self.pool = BPOAgentPool()
        self._initialized = False
        self._task_history: List[CoordinationResult] = []
        self.logger = logging.getLogger(f"{__name__}.BPOAgentCoordinator")

    async def initialize(self, agent_types: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Initialize the coordinator and all agents.

        Args:
            agent_types: Specific agent types to initialize. None = all.

        Returns:
            Dict mapping agent_type to agent_id
        """
        self.logger.info("Initializing BPO Agent Coordinator")
        result = await self.pool.initialize_agents(agent_types)
        self._initialized = True
        self.logger.info(f"BPO Agent Coordinator initialized with {len(result)} agents")
        return result

    async def shutdown(self) -> None:
        """Shutdown the coordinator and all agents."""
        self.logger.info("Shutting down BPO Agent Coordinator")
        await self.pool.shutdown_all()
        self._initialized = False

    async def execute(self, task: CoordinationTask) -> CoordinationResult:
        """
        Execute a coordination task.

        Routes the task to appropriate agent(s) based on strategy.
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        strategy = task.strategy

        try:
            if strategy == CoordinationStrategy.SEQUENTIAL:
                result = await self._execute_sequential(task)
            elif strategy == CoordinationStrategy.PARALLEL:
                result = await self._execute_parallel(task)
            elif strategy == CoordinationStrategy.PIPELINE:
                result = await self._execute_pipeline(task)
            elif strategy == CoordinationStrategy.BROADCAST:
                result = await self._execute_broadcast(task)
            elif strategy == CoordinationStrategy.CONSENSUS:
                result = await self._execute_consensus(task)
            else:
                # Adaptive - determine best strategy
                result = await self._execute_adaptive(task)

            result.duration = time.time() - start_time
            COORDINATOR_TASKS.labels(strategy=strategy.value, status="success").inc()
            COORDINATOR_DURATION.labels(strategy=strategy.value).observe(result.duration)

            # Store in history
            self._task_history.append(result)
            if len(self._task_history) > 10000:
                self._task_history = self._task_history[-10000:]

            return result

        except Exception as e:
            duration = time.time() - start_time
            COORDINATOR_TASKS.labels(strategy=strategy.value, status="error").inc()
            self.logger.error(f"Coordination task failed: {e}")
            return CoordinationResult(
                task_id=task.task_id,
                strategy=strategy.value,
                success=False,
                error=str(e),
                duration=duration,
            )

    async def route_task(self, task_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a single task to the appropriate agent.

        Convenience method for simple task routing.
        """
        agent_type = self.TASK_ROUTING.get(task_type)
        if not agent_type:
            raise ValueError(f"Unknown task type: {task_type}")

        agent = self.pool.get_agent(agent_type)
        if not agent:
            raise RuntimeError(f"No available agent for type: {agent_type}")

        return await agent.execute(task_type, payload)

    async def _execute_sequential(self, task: CoordinationTask) -> CoordinationResult:
        """Execute tasks sequentially across agents."""
        agent_results = {}
        target_agents = task.target_agents or self._resolve_agents(task)

        for agent_type in target_agents:
            agent = self.pool.get_agent(agent_type)
            if agent:
                result = await agent.execute(task.task_type, task.payload)
                agent_results[agent_type] = result

        return CoordinationResult(
            task_id=task.task_id,
            strategy=CoordinationStrategy.SEQUENTIAL.value,
            success=all(r.get("success", False) for r in agent_results.values()),
            agent_results=agent_results,
        )

    async def _execute_parallel(self, task: CoordinationTask) -> CoordinationResult:
        """Execute tasks in parallel across agents."""
        target_agents = task.target_agents or self._resolve_agents(task)

        async def _run_agent(agent_type: str) -> Tuple[str, Dict[str, Any]]:
            agent = self.pool.get_agent(agent_type)
            if agent:
                result = await agent.execute(task.task_type, task.payload)
                return (agent_type, result)
            return (agent_type, {"success": False, "error": "Agent not available"})

        tasks = [_run_agent(at) for at in target_agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        agent_results = {}
        for r in results:
            if isinstance(r, Exception):
                self.logger.error(f"Parallel task error: {r}")
            elif isinstance(r, tuple):
                agent_results[r[0]] = r[1]

        return CoordinationResult(
            task_id=task.task_id,
            strategy=CoordinationStrategy.PARALLEL.value,
            success=any(r.get("success", False) for r in agent_results.values()),
            agent_results=agent_results,
        )

    async def _execute_pipeline(self, task: CoordinationTask) -> CoordinationResult:
        """Execute tasks as a pipeline where output feeds into next agent."""
        target_agents = task.target_agents or self._resolve_agents(task)
        agent_results = {}
        current_payload = dict(task.payload)

        for agent_type in target_agents:
            agent = self.pool.get_agent(agent_type)
            if agent:
                result = await agent.execute(task.task_type, current_payload)
                agent_results[agent_type] = result

                # Feed result into next stage
                if result.get("success") and result.get("result"):
                    current_payload.update(result["result"])
                else:
                    break  # Pipeline stops on failure

        return CoordinationResult(
            task_id=task.task_id,
            strategy=CoordinationStrategy.PIPELINE.value,
            success=all(r.get("success", False) for r in agent_results.values()),
            agent_results=agent_results,
            aggregated_result=current_payload,
        )

    async def _execute_broadcast(self, task: CoordinationTask) -> CoordinationResult:
        """Broadcast task to all agents."""
        all_agents = list(BPO_AGENT_REGISTRY.keys())

        async def _run_agent(agent_type: str) -> Tuple[str, Dict[str, Any]]:
            agent = self.pool.get_agent(agent_type)
            if agent:
                try:
                    result = await agent.execute(task.task_type, task.payload)
                    return (agent_type, result)
                except Exception as e:
                    return (agent_type, {"success": False, "error": str(e)})
            return (agent_type, {"success": False, "error": "Agent not available"})

        tasks = [_run_agent(at) for at in all_agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        agent_results = {}
        for r in results:
            if isinstance(r, tuple):
                agent_results[r[0]] = r[1]

        return CoordinationResult(
            task_id=task.task_id,
            strategy=CoordinationStrategy.BROADCAST.value,
            success=True,
            agent_results=agent_results,
        )

    async def _execute_consensus(self, task: CoordinationTask) -> CoordinationResult:
        """Execute consensus-based decision across agents."""
        # Run parallel first
        parallel_result = await self._execute_parallel(task)

        # Aggregate decisions
        decisions = {}
        for agent_type, result in parallel_result.agent_results.items():
            if result.get("success") and result.get("result"):
                action = result["result"].get("action", "unknown")
                if action not in decisions:
                    decisions[action] = []
                decisions[action].append(agent_type)

        # Majority vote
        consensus_action = max(decisions, key=lambda k: len(decisions[k])) if decisions else "unknown"

        return CoordinationResult(
            task_id=task.task_id,
            strategy=CoordinationStrategy.CONSENSUS.value,
            success=True,
            agent_results=parallel_result.agent_results,
            aggregated_result={
                "consensus_action": consensus_action,
                "votes": {k: len(v) for k, v in decisions.items()},
            },
        )

    async def _execute_adaptive(self, task: CoordinationTask) -> CoordinationResult:
        """Adaptively choose the best strategy based on task characteristics."""
        # Determine strategy based on task type
        agent_type = self.TASK_ROUTING.get(task.task_type)

        if agent_type:
            # Single agent task
            agent = self.pool.get_agent(agent_type)
            if agent:
                result = await agent.execute(task.task_type, task.payload)
                return CoordinationResult(
                    task_id=task.task_id,
                    strategy=CoordinationStrategy.ADAPTIVE.value,
                    success=result.get("success", False),
                    agent_results={agent_type: result},
                )

        # Multi-agent task - use parallel
        return await self._execute_parallel(task)

    def _resolve_agents(self, task: CoordinationTask) -> List[str]:
        """Resolve which agents should handle a task."""
        agent_type = self.TASK_ROUTING.get(task.task_type)
        if agent_type:
            return [agent_type]
        # Default to all agents
        return list(BPO_AGENT_REGISTRY.keys())

    # =========================================================================
    # Convenience methods for common BPO operations
    # =========================================================================

    async def analyze_incoming_call(self, call_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an incoming call through the full security pipeline.

        Runs call processing, fraud detection, and PCI compliance checks.
        """
        task = CoordinationTask(
            task_type="analyze_call",
            payload=call_data,
            target_agents=[
                BPOAgentType.CALL_PROCESSING.value,
                BPOAgentType.FRAUD_DETECTION.value,
                BPOAgentType.PCI_COMPLIANCE.value,
            ],
            strategy=CoordinationStrategy.PARALLEL,
        )
        return (await self.execute(task)).agent_results

    async def run_full_compliance_audit(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """Run full compliance audit across all frameworks."""
        return await self.route_task("run_audit", {
            "framework": "PCI-DSS 4.0",
            "scope": "full",
            "environment": environment,
        })

    async def provision_remote_agent(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Provision a remote/WFH agent with quantum-safe tunnel."""
        # Validate endpoint first
        validation = await self.route_task("validate_endpoint", {
            "endpoint": agent_data.get("endpoint", {}),
        })

        # Provision tunnel if endpoint is compliant
        result = validation.get("result", {})
        if result.get("compliant", False) or result.get("access_decision") == "allow":
            tunnel = await self.route_task("provision_tunnel", agent_data)
            return {"validation": validation, "tunnel": tunnel}

        return {"validation": validation, "tunnel": None, "reason": "Endpoint non-compliant"}

    def get_all_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        limit: int = 100,
    ) -> List[BPOAlert]:
        """Get all alerts from all agents."""
        all_alerts = []
        for agent in self.pool.get_all_agents():
            alerts = agent.get_alerts(severity=severity, limit=limit)
            all_alerts.extend(alerts)

        # Sort by timestamp descending
        all_alerts.sort(key=lambda a: a.timestamp, reverse=True)
        return all_alerts[:limit]

    def get_status(self) -> Dict[str, Any]:
        """Get coordinator status."""
        return {
            "initialized": self._initialized,
            "pool": self.pool.get_pool_status(),
            "task_history_count": len(self._task_history),
            "supported_task_types": list(self.TASK_ROUTING.keys()),
        }
