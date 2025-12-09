"""
CRONOS AI - Planning/Supervisor Agent (Meta-Agent)

The Planning Agent is a meta-agent responsible for:
- Decomposing complex tasks into sub-tasks
- Assigning sub-tasks to appropriate specialist agents
- Monitoring execution and adapting plans
- Coordinating multi-agent workflows
- Providing strategic decision making
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, TYPE_CHECKING

from .base_agent import (
    BaseAgent,
    AgentCapability,
    AgentConfig,
    AgentTask,
    AgentPriority,
    TaskResult,
)

if TYPE_CHECKING:
    from .agent_communication import AgentCommunicationProtocol
    from .agent_memory import AgentMemoryManager
    from .agent_pool import AgentPoolManager
    from ..llm.unified_llm_service import UnifiedLLMService, LLMRequest

logger = logging.getLogger(__name__)


class PlanStatus(str, Enum):
    """Status of a task plan."""

    DRAFT = "draft"
    APPROVED = "approved"
    EXECUTING = "executing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionStrategy(str, Enum):
    """Strategy for executing sub-tasks."""

    SEQUENTIAL = "sequential"  # Execute tasks one after another
    PARALLEL = "parallel"  # Execute all tasks concurrently
    PIPELINE = "pipeline"  # Output of one feeds into next
    ADAPTIVE = "adaptive"  # Dynamically choose based on conditions
    CONSENSUS = "consensus"  # Multiple agents vote on result


class SubTaskStatus(str, Enum):
    """Status of a sub-task."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


@dataclass
class SubTask:
    """A sub-task within a plan."""

    subtask_id: str
    name: str
    description: str
    required_capabilities: List[AgentCapability]
    input_data: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)  # IDs of dependent subtasks
    assigned_agent_id: Optional[str] = None
    status: SubTaskStatus = SubTaskStatus.PENDING
    priority: AgentPriority = AgentPriority.NORMAL
    timeout_seconds: int = 300
    retry_count: int = 0
    max_retries: int = 3
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskPlan:
    """A complete plan for executing a complex task."""

    plan_id: str
    name: str
    description: str
    original_task: AgentTask
    subtasks: List[SubTask]
    execution_strategy: ExecutionStrategy
    status: PlanStatus = PlanStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0  # 0.0 to 1.0
    final_result: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_ready_subtasks(self) -> List[SubTask]:
        """Get subtasks that are ready to execute (dependencies satisfied)."""
        completed_ids = {
            st.subtask_id for st in self.subtasks
            if st.status == SubTaskStatus.COMPLETED
        }
        ready = []
        for subtask in self.subtasks:
            if subtask.status == SubTaskStatus.PENDING:
                if all(dep_id in completed_ids for dep_id in subtask.dependencies):
                    ready.append(subtask)
        return ready

    def update_progress(self) -> None:
        """Update plan progress based on subtask completion."""
        if not self.subtasks:
            self.progress = 0.0
            return
        completed = sum(
            1 for st in self.subtasks
            if st.status in [SubTaskStatus.COMPLETED, SubTaskStatus.SKIPPED]
        )
        self.progress = completed / len(self.subtasks)


@dataclass
class PlanningContext:
    """Context for task planning."""

    available_agents: List[Dict[str, Any]]
    available_capabilities: Set[AgentCapability]
    system_load: float  # 0.0 to 1.0
    historical_performance: Dict[str, float]  # agent_type -> avg_success_rate
    constraints: Dict[str, Any] = field(default_factory=dict)


class PlanningAgent(BaseAgent):
    """
    Meta-agent responsible for planning and coordinating multi-agent workflows.

    The Planning Agent:
    1. Receives complex tasks from the orchestrator
    2. Decomposes tasks into sub-tasks using LLM reasoning
    3. Assigns sub-tasks to appropriate specialist agents
    4. Monitors execution and handles failures
    5. Adapts plans based on runtime conditions
    6. Aggregates results and provides final output
    """

    def __init__(
        self,
        config: AgentConfig,
        llm_service: Optional["UnifiedLLMService"] = None,
        agent_pool: Optional["AgentPoolManager"] = None,
        communication: Optional["AgentCommunicationProtocol"] = None,
        memory: Optional["AgentMemoryManager"] = None,
    ):
        """Initialize the Planning Agent."""
        # Ensure planning capabilities are included
        planning_capabilities = [
            AgentCapability.TASK_PLANNING,
            AgentCapability.AGENT_COORDINATION,
            AgentCapability.WORKFLOW_ORCHESTRATION,
        ]
        for cap in planning_capabilities:
            if cap not in config.capabilities:
                config.capabilities.append(cap)

        super().__init__(config, communication, memory)

        self.llm_service = llm_service
        self.agent_pool = agent_pool

        # Active plans
        self.active_plans: Dict[str, TaskPlan] = {}
        self.completed_plans: List[TaskPlan] = []

        # Plan execution tracking
        self._plan_executors: Dict[str, asyncio.Task] = {}

        # Callbacks for plan events
        self._plan_callbacks: Dict[str, List[Callable]] = {
            "on_plan_created": [],
            "on_plan_started": [],
            "on_subtask_completed": [],
            "on_plan_completed": [],
            "on_plan_failed": [],
        }

        self.logger = logging.getLogger(f"{__name__}.PlanningAgent.{self.agent_id[:8]}")

    async def execute(self, task: AgentTask) -> Any:
        """
        Execute a planning task.

        The main entry point for complex tasks that need decomposition.
        """
        task_type = task.task_type

        if task_type == "create_plan":
            return await self.create_plan(task)
        elif task_type == "execute_plan":
            plan_id = task.payload.get("plan_id")
            return await self.execute_plan(plan_id)
        elif task_type == "monitor_plan":
            plan_id = task.payload.get("plan_id")
            return self.get_plan_status(plan_id)
        elif task_type == "cancel_plan":
            plan_id = task.payload.get("plan_id")
            return await self.cancel_plan(plan_id)
        elif task_type == "decompose_and_execute":
            # Convenience method: create plan and execute immediately
            plan = await self.create_plan(task)
            return await self.execute_plan(plan.plan_id)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    async def create_plan(self, task: AgentTask) -> TaskPlan:
        """
        Create a plan for executing a complex task.

        Uses LLM to decompose the task into sub-tasks.
        """
        self.logger.info(f"Creating plan for task: {task.task_id}")

        # Get planning context
        context = await self._get_planning_context()

        # Use LLM to decompose task
        subtasks = await self._decompose_task(task, context)

        # Determine execution strategy
        strategy = await self._determine_strategy(task, subtasks, context)

        # Create the plan
        plan = TaskPlan(
            plan_id=str(uuid.uuid4()),
            name=task.payload.get("name", f"Plan for {task.task_type}"),
            description=task.payload.get("description", ""),
            original_task=task,
            subtasks=subtasks,
            execution_strategy=strategy,
            status=PlanStatus.DRAFT,
            metadata={
                "context": {
                    "system_load": context.system_load,
                    "available_agents": len(context.available_agents),
                }
            }
        )

        self.active_plans[plan.plan_id] = plan
        self.logger.info(f"Plan created: {plan.plan_id} with {len(subtasks)} subtasks")

        # Trigger callbacks
        await self._trigger_callbacks("on_plan_created", plan)

        return plan

    async def execute_plan(self, plan_id: str) -> Any:
        """Execute a plan and return the final result."""
        plan = self.active_plans.get(plan_id)
        if not plan:
            raise ValueError(f"Plan not found: {plan_id}")

        if plan.status not in [PlanStatus.DRAFT, PlanStatus.APPROVED]:
            raise ValueError(f"Plan cannot be executed in status: {plan.status}")

        self.logger.info(f"Executing plan: {plan_id}")
        plan.status = PlanStatus.EXECUTING
        plan.started_at = datetime.utcnow()

        # Trigger callbacks
        await self._trigger_callbacks("on_plan_started", plan)

        try:
            # Execute based on strategy
            if plan.execution_strategy == ExecutionStrategy.SEQUENTIAL:
                result = await self._execute_sequential(plan)
            elif plan.execution_strategy == ExecutionStrategy.PARALLEL:
                result = await self._execute_parallel(plan)
            elif plan.execution_strategy == ExecutionStrategy.PIPELINE:
                result = await self._execute_pipeline(plan)
            elif plan.execution_strategy == ExecutionStrategy.ADAPTIVE:
                result = await self._execute_adaptive(plan)
            elif plan.execution_strategy == ExecutionStrategy.CONSENSUS:
                result = await self._execute_consensus(plan)
            else:
                result = await self._execute_sequential(plan)

            plan.status = PlanStatus.COMPLETED
            plan.completed_at = datetime.utcnow()
            plan.final_result = result
            plan.progress = 1.0

            # Move to completed
            self.completed_plans.append(plan)
            del self.active_plans[plan_id]

            # Trigger callbacks
            await self._trigger_callbacks("on_plan_completed", plan)

            self.logger.info(f"Plan completed: {plan_id}")
            return result

        except Exception as e:
            plan.status = PlanStatus.FAILED
            plan.metadata["error"] = str(e)
            self.logger.error(f"Plan failed: {plan_id} - {e}")

            # Trigger callbacks
            await self._trigger_callbacks("on_plan_failed", plan)

            raise

    async def cancel_plan(self, plan_id: str) -> bool:
        """Cancel an executing plan."""
        plan = self.active_plans.get(plan_id)
        if not plan:
            return False

        plan.status = PlanStatus.CANCELLED

        # Cancel any running executors
        if plan_id in self._plan_executors:
            self._plan_executors[plan_id].cancel()
            del self._plan_executors[plan_id]

        self.logger.info(f"Plan cancelled: {plan_id}")
        return True

    def get_plan_status(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a plan."""
        plan = self.active_plans.get(plan_id)
        if not plan:
            # Check completed plans
            for completed in self.completed_plans:
                if completed.plan_id == plan_id:
                    plan = completed
                    break

        if not plan:
            return None

        return {
            "plan_id": plan.plan_id,
            "name": plan.name,
            "status": plan.status.value,
            "progress": plan.progress,
            "subtasks": [
                {
                    "subtask_id": st.subtask_id,
                    "name": st.name,
                    "status": st.status.value,
                    "assigned_agent": st.assigned_agent_id,
                }
                for st in plan.subtasks
            ],
            "created_at": plan.created_at.isoformat(),
            "started_at": plan.started_at.isoformat() if plan.started_at else None,
            "completed_at": plan.completed_at.isoformat() if plan.completed_at else None,
        }

    async def _get_planning_context(self) -> PlanningContext:
        """Gather context for planning decisions."""
        available_agents = []
        available_capabilities = set()

        if self.agent_pool:
            # Get available agents from pool
            pool_status = await self.agent_pool.get_status()
            for agent_info in pool_status.get("agents", []):
                available_agents.append(agent_info)
                for cap in agent_info.get("capabilities", []):
                    try:
                        available_capabilities.add(AgentCapability(cap))
                    except ValueError:
                        pass

        # Get historical performance from memory
        historical_performance = {}
        if self.memory:
            perf_data = await self.memory.retrieve_semantic(
                query="agent performance statistics",
                agent_id=self.agent_id,
                limit=10
            )
            for entry in perf_data:
                agent_type = entry.get("agent_type")
                success_rate = entry.get("success_rate", 0.8)
                if agent_type:
                    historical_performance[agent_type] = success_rate

        return PlanningContext(
            available_agents=available_agents,
            available_capabilities=available_capabilities,
            system_load=0.5,  # Would come from monitoring
            historical_performance=historical_performance,
        )

    async def _decompose_task(
        self,
        task: AgentTask,
        context: PlanningContext
    ) -> List[SubTask]:
        """Use LLM to decompose a task into sub-tasks."""
        if not self.llm_service:
            # Fallback: create a single subtask
            return [
                SubTask(
                    subtask_id=str(uuid.uuid4()),
                    name=task.task_type,
                    description=f"Execute {task.task_type}",
                    required_capabilities=[],
                    input_data=task.payload,
                )
            ]

        # Prepare prompt for LLM
        capabilities_list = [cap.value for cap in context.available_capabilities]

        prompt = f"""You are a task planning agent. Decompose the following complex task into smaller sub-tasks.

Task Type: {task.task_type}
Task Description: {task.payload.get('description', 'No description provided')}
Task Data: {task.payload}

Available Capabilities:
{', '.join(capabilities_list)}

Decompose this task into sub-tasks. For each sub-task, provide:
1. A unique name
2. A description of what needs to be done
3. Required capabilities (from the available list)
4. Dependencies on other sub-tasks (by name)
5. Input data needed

Format your response as a JSON array of sub-tasks:
[
    {{
        "name": "subtask_name",
        "description": "what this subtask does",
        "required_capabilities": ["capability1", "capability2"],
        "dependencies": ["name_of_dependent_subtask"],
        "input_keys": ["key1", "key2"]
    }}
]

Important:
- Break down into 2-5 sub-tasks unless the task is very complex
- Ensure dependencies form a valid DAG (no circular dependencies)
- Each sub-task should be atomic and have clear boundaries
- Consider parallel execution where possible
"""

        from ..llm.unified_llm_service import LLMRequest

        llm_request = LLMRequest(
            prompt=prompt,
            feature_domain="planning_agent",
            context={"task": task.payload},
            max_tokens=2000,
            temperature=0.3,
        )

        response = await self.llm_service.process_request(llm_request)

        # Parse LLM response
        import json
        try:
            subtask_data = json.loads(response.content)
        except json.JSONDecodeError:
            # Fallback to single subtask
            self.logger.warning("Failed to parse LLM response, using fallback")
            return [
                SubTask(
                    subtask_id=str(uuid.uuid4()),
                    name=task.task_type,
                    description=f"Execute {task.task_type}",
                    required_capabilities=[],
                    input_data=task.payload,
                )
            ]

        # Convert to SubTask objects
        subtasks = []
        name_to_id = {}

        for st_data in subtask_data:
            subtask_id = str(uuid.uuid4())
            name = st_data.get("name", f"subtask_{len(subtasks)}")
            name_to_id[name] = subtask_id

            # Parse capabilities
            capabilities = []
            for cap_str in st_data.get("required_capabilities", []):
                try:
                    capabilities.append(AgentCapability(cap_str))
                except ValueError:
                    pass

            # Extract input data
            input_keys = st_data.get("input_keys", [])
            input_data = {k: task.payload.get(k) for k in input_keys if k in task.payload}
            if not input_data:
                input_data = task.payload

            subtask = SubTask(
                subtask_id=subtask_id,
                name=name,
                description=st_data.get("description", ""),
                required_capabilities=capabilities,
                input_data=input_data,
                priority=task.priority,
            )
            subtasks.append(subtask)

        # Resolve dependencies
        for i, st_data in enumerate(subtask_data):
            dep_names = st_data.get("dependencies", [])
            subtasks[i].dependencies = [
                name_to_id[name] for name in dep_names
                if name in name_to_id
            ]

        return subtasks

    async def _determine_strategy(
        self,
        task: AgentTask,
        subtasks: List[SubTask],
        context: PlanningContext
    ) -> ExecutionStrategy:
        """Determine the best execution strategy for the plan."""
        # Check for dependencies
        has_dependencies = any(st.dependencies for st in subtasks)

        # Check task hints
        strategy_hint = task.payload.get("execution_strategy")
        if strategy_hint:
            try:
                return ExecutionStrategy(strategy_hint)
            except ValueError:
                pass

        # Auto-determine strategy
        if len(subtasks) == 1:
            return ExecutionStrategy.SEQUENTIAL

        if not has_dependencies:
            # All independent - can run in parallel
            if context.system_load < 0.7:
                return ExecutionStrategy.PARALLEL
            else:
                return ExecutionStrategy.SEQUENTIAL

        # Check if it's a pipeline (linear dependencies)
        is_pipeline = True
        for i, st in enumerate(subtasks):
            if i > 0 and subtasks[i-1].subtask_id not in st.dependencies:
                is_pipeline = False
                break
            if len(st.dependencies) > 1:
                is_pipeline = False
                break

        if is_pipeline:
            return ExecutionStrategy.PIPELINE

        return ExecutionStrategy.ADAPTIVE

    async def _execute_sequential(self, plan: TaskPlan) -> Any:
        """Execute subtasks sequentially."""
        results = {}

        for subtask in plan.subtasks:
            result = await self._execute_subtask(plan, subtask, results)
            results[subtask.subtask_id] = result
            plan.update_progress()

            if subtask.status == SubTaskStatus.FAILED:
                raise RuntimeError(f"Subtask failed: {subtask.name} - {subtask.error}")

        return self._aggregate_results(plan, results)

    async def _execute_parallel(self, plan: TaskPlan) -> Any:
        """Execute all subtasks in parallel."""
        tasks = []
        results = {}

        for subtask in plan.subtasks:
            task = asyncio.create_task(
                self._execute_subtask(plan, subtask, results)
            )
            tasks.append((subtask.subtask_id, task))

        # Wait for all tasks
        for subtask_id, task in tasks:
            try:
                result = await task
                results[subtask_id] = result
            except Exception as e:
                self.logger.error(f"Parallel subtask failed: {subtask_id} - {e}")
                results[subtask_id] = None

        plan.update_progress()
        return self._aggregate_results(plan, results)

    async def _execute_pipeline(self, plan: TaskPlan) -> Any:
        """Execute subtasks in a pipeline, passing output to next input."""
        pipeline_data = {}
        results = {}

        for subtask in plan.subtasks:
            # Merge pipeline data into subtask input
            subtask.input_data = {**subtask.input_data, **pipeline_data}

            result = await self._execute_subtask(plan, subtask, results)
            results[subtask.subtask_id] = result

            # Update pipeline data for next subtask
            if isinstance(result, dict):
                pipeline_data.update(result)
            else:
                pipeline_data["previous_result"] = result

            plan.update_progress()

            if subtask.status == SubTaskStatus.FAILED:
                raise RuntimeError(f"Pipeline failed at: {subtask.name}")

        return self._aggregate_results(plan, results)

    async def _execute_adaptive(self, plan: TaskPlan) -> Any:
        """Execute subtasks adaptively based on dependencies and availability."""
        results = {}
        pending = {st.subtask_id: st for st in plan.subtasks}
        executing: Dict[str, asyncio.Task] = {}

        while pending or executing:
            # Find ready subtasks
            ready = []
            for subtask_id, subtask in list(pending.items()):
                deps_satisfied = all(
                    dep_id in results for dep_id in subtask.dependencies
                )
                if deps_satisfied:
                    ready.append(subtask)
                    del pending[subtask_id]

            # Start ready subtasks (up to concurrency limit)
            max_concurrent = 5
            for subtask in ready:
                if len(executing) < max_concurrent:
                    # Pass dependent results as input
                    for dep_id in subtask.dependencies:
                        if dep_id in results and isinstance(results[dep_id], dict):
                            subtask.input_data.update(results[dep_id])

                    task = asyncio.create_task(
                        self._execute_subtask(plan, subtask, results)
                    )
                    executing[subtask.subtask_id] = task
                else:
                    # Put back in pending
                    pending[subtask.subtask_id] = subtask

            if not executing:
                if pending:
                    # Deadlock - circular dependencies
                    raise RuntimeError("Deadlock detected in plan execution")
                break

            # Wait for at least one task to complete
            done, _ = await asyncio.wait(
                executing.values(),
                return_when=asyncio.FIRST_COMPLETED
            )

            # Process completed tasks
            for completed_task in done:
                for subtask_id, task in list(executing.items()):
                    if task == completed_task:
                        try:
                            result = await task
                            results[subtask_id] = result
                        except Exception as e:
                            self.logger.error(f"Subtask failed: {subtask_id} - {e}")
                            results[subtask_id] = None
                        del executing[subtask_id]
                        break

            plan.update_progress()

        return self._aggregate_results(plan, results)

    async def _execute_consensus(self, plan: TaskPlan) -> Any:
        """Execute subtasks and use consensus for final result."""
        # Execute all in parallel
        results = await self._execute_parallel(plan)

        # For consensus, we need multiple agents to agree
        # This is useful for high-stakes decisions
        if isinstance(results, dict):
            # Count votes for each result value
            votes: Dict[str, int] = {}
            for subtask_id, result in results.items():
                result_key = str(result)
                votes[result_key] = votes.get(result_key, 0) + 1

            # Return most common result
            if votes:
                consensus_result = max(votes.items(), key=lambda x: x[1])[0]
                return {
                    "consensus_result": consensus_result,
                    "all_results": results,
                    "votes": votes,
                }

        return results

    async def _execute_subtask(
        self,
        plan: TaskPlan,
        subtask: SubTask,
        results: Dict[str, Any]
    ) -> Any:
        """Execute a single subtask by assigning it to an appropriate agent."""
        subtask.status = SubTaskStatus.EXECUTING
        subtask.started_at = datetime.utcnow()

        try:
            # Find appropriate agent
            agent = await self._find_agent_for_subtask(subtask)

            if not agent:
                raise RuntimeError(f"No suitable agent found for subtask: {subtask.name}")

            subtask.assigned_agent_id = agent.get("agent_id")

            # Create task for agent
            agent_task = AgentTask(
                task_id=str(uuid.uuid4()),
                task_type=subtask.name,
                payload=subtask.input_data,
                priority=subtask.priority,
                timeout=subtask.timeout_seconds,
                correlation_id=plan.plan_id,
            )

            # Submit to agent via pool or communication
            if self.agent_pool:
                result = await self.agent_pool.submit_task(
                    agent_task,
                    required_capabilities=subtask.required_capabilities
                )
            elif self.communication:
                result = await self.communication.request(
                    target_agent_id=subtask.assigned_agent_id,
                    message_type="execute_task",
                    payload={"task": agent_task},
                    timeout=subtask.timeout_seconds
                )
            else:
                raise RuntimeError("No agent pool or communication available")

            subtask.status = SubTaskStatus.COMPLETED
            subtask.completed_at = datetime.utcnow()
            subtask.result = result

            # Trigger callback
            await self._trigger_callbacks("on_subtask_completed", plan, subtask)

            return result

        except Exception as e:
            subtask.status = SubTaskStatus.FAILED
            subtask.error = str(e)
            subtask.retry_count += 1

            # Retry if possible
            if subtask.retry_count < subtask.max_retries:
                self.logger.warning(
                    f"Subtask {subtask.name} failed, retrying ({subtask.retry_count}/{subtask.max_retries})"
                )
                subtask.status = SubTaskStatus.PENDING
                return await self._execute_subtask(plan, subtask, results)

            self.logger.error(f"Subtask {subtask.name} failed after {subtask.retry_count} retries: {e}")
            raise

    async def _find_agent_for_subtask(self, subtask: SubTask) -> Optional[Dict[str, Any]]:
        """Find the best agent to execute a subtask."""
        if not self.agent_pool:
            return {"agent_id": "default"}

        # Get available agents with required capabilities
        agents = await self.agent_pool.get_agents_with_capabilities(
            subtask.required_capabilities
        )

        if not agents:
            return None

        # Sort by availability and performance
        best_agent = min(
            agents,
            key=lambda a: (a.get("queue_size", 0), -a.get("success_rate", 0.8))
        )

        return best_agent

    def _aggregate_results(
        self,
        plan: TaskPlan,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Aggregate results from all subtasks."""
        aggregated = {
            "plan_id": plan.plan_id,
            "status": "completed",
            "subtask_results": {},
            "summary": {},
        }

        for subtask in plan.subtasks:
            aggregated["subtask_results"][subtask.name] = {
                "status": subtask.status.value,
                "result": results.get(subtask.subtask_id),
                "error": subtask.error,
            }

        # Create summary
        successful = sum(1 for st in plan.subtasks if st.status == SubTaskStatus.COMPLETED)
        aggregated["summary"] = {
            "total_subtasks": len(plan.subtasks),
            "successful": successful,
            "failed": len(plan.subtasks) - successful,
            "execution_time": (
                (plan.completed_at - plan.started_at).total_seconds()
                if plan.completed_at and plan.started_at
                else None
            ),
        }

        return aggregated

    async def _trigger_callbacks(self, event: str, *args) -> None:
        """Trigger registered callbacks for an event."""
        for callback in self._plan_callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args)
                else:
                    callback(*args)
            except Exception as e:
                self.logger.error(f"Callback error for {event}: {e}")

    def register_callback(self, event: str, callback: Callable) -> None:
        """Register a callback for plan events."""
        if event in self._plan_callbacks:
            self._plan_callbacks[event].append(callback)

    async def _on_start(self) -> None:
        """Initialize planning agent."""
        self.logger.info("Planning Agent started")

    async def _on_stop(self) -> None:
        """Cleanup planning agent."""
        # Cancel all active plan executors
        for plan_id, executor in self._plan_executors.items():
            executor.cancel()
        self._plan_executors.clear()
        self.logger.info("Planning Agent stopped")
