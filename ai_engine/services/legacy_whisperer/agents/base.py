"""
QBITEL - Multi-Agent Base Classes

Core abstractions for the multi-agent system including:
- Agent base class with tool use and reflection
- Message passing between agents
- Memory management
- Tool interface and results

Inspired by modern agent frameworks like AutoGen, CrewAI, and LangGraph.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union, TypeVar, Generic, Set, Tuple, AsyncIterator

from prometheus_client import Counter, Histogram, Gauge

# =============================================================================
# Metrics
# =============================================================================

AGENT_ACTIONS = Counter("qbitel_agent_actions_total", "Total agent actions", ["agent_role", "action_type", "status"])
AGENT_TOOL_CALLS = Counter(
    "qbitel_agent_tool_calls_total", "Total tool calls by agents", ["agent_role", "tool_name", "status"]
)
AGENT_REFLECTION_COUNT = Counter(
    "qbitel_agent_reflections_total", "Total agent reflections", ["agent_role", "reflection_type"]
)
AGENT_RESPONSE_TIME = Histogram("qbitel_agent_response_seconds", "Agent response time", ["agent_role"])
AGENT_TOKEN_USAGE = Counter(
    "qbitel_agent_tokens_total", "Total tokens used by agents", ["agent_role", "token_type"]  # input, output
)


# =============================================================================
# Enums and Types
# =============================================================================


class AgentRole(str, Enum):
    """Agent role types in the multi-agent system."""

    ORCHESTRATOR = "orchestrator"
    PROTOCOL_ANALYST = "protocol_analyst"
    DOCUMENTATION = "documentation"
    RISK_ASSESSOR = "risk_assessor"
    CODE_GENERATOR = "code_generator"
    HUMAN = "human"  # For human-in-the-loop


class MessageType(str, Enum):
    """Types of messages between agents."""

    TASK = "task"  # New task assignment
    RESULT = "result"  # Task result
    QUESTION = "question"  # Clarification request
    ANSWER = "answer"  # Answer to question
    HANDOFF = "handoff"  # Task handoff to another agent
    FEEDBACK = "feedback"  # Feedback on work
    ERROR = "error"  # Error notification
    STATUS = "status"  # Status update


class ToolStatus(str, Enum):
    """Tool execution status."""

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    TIMEOUT = "timeout"


class ReflectionType(str, Enum):
    """Types of agent self-reflection."""

    QUALITY_CHECK = "quality_check"
    ERROR_ANALYSIS = "error_analysis"
    IMPROVEMENT = "improvement"
    VALIDATION = "validation"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class AgentConfig:
    """Configuration for an agent."""

    role: AgentRole
    name: str
    description: str

    # LLM settings
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096

    # Behavior settings
    max_iterations: int = 10
    max_tool_calls_per_iteration: int = 5
    enable_reflection: bool = True
    reflection_frequency: int = 3  # Reflect every N iterations
    enable_self_correction: bool = True

    # Memory settings
    memory_capacity: int = 100
    context_window_messages: int = 20

    # Timeouts
    tool_timeout: float = 60.0  # seconds
    iteration_timeout: float = 300.0  # seconds

    # Collaboration
    can_delegate: bool = True
    can_receive_delegation: bool = True
    allowed_handoff_targets: List[AgentRole] = field(default_factory=list)


@dataclass
class AgentMessage:
    """Message passed between agents."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType = MessageType.TASK
    sender: AgentRole = AgentRole.ORCHESTRATOR
    recipient: AgentRole = AgentRole.PROTOCOL_ANALYST
    content: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None  # For threading
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    priority: int = 0  # Higher = more urgent

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "message_type": self.message_type.value,
            "sender": self.sender.value,
            "recipient": self.recipient.value,
            "content": self.content,
            "data": self.data,
            "metadata": self.metadata,
            "parent_id": self.parent_id,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority,
        }


@dataclass
class ToolResult:
    """Result from a tool execution."""

    tool_name: str
    status: ToolStatus
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.status == ToolStatus.SUCCESS


@dataclass
class ReflectionResult:
    """Result from agent self-reflection."""

    reflection_type: ReflectionType
    assessment: str
    confidence: float
    issues_found: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    should_retry: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    """Current state of an agent."""

    agent_id: str
    role: AgentRole
    is_busy: bool = False
    current_task_id: Optional[str] = None
    iteration_count: int = 0
    total_tokens_used: int = 0
    errors_count: int = 0
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# Memory Classes
# =============================================================================


@dataclass
class MemoryEntry:
    """A single memory entry."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    entry_type: str = "general"
    importance: float = 0.5  # 0-1
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    accessed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0


class AgentMemory:
    """
    Memory system for an agent.

    Provides:
    - Short-term (conversation) memory
    - Long-term (episodic) memory
    - Semantic memory for facts and knowledge
    """

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self._short_term: List[MemoryEntry] = []
        self._long_term: List[MemoryEntry] = []
        self._semantic: Dict[str, MemoryEntry] = {}

    def add_short_term(self, content: str, importance: float = 0.5, **metadata) -> str:
        """Add to short-term memory."""
        entry = MemoryEntry(content=content, entry_type="short_term", importance=importance, metadata=metadata)

        self._short_term.append(entry)

        # Trim if over capacity
        if len(self._short_term) > self.capacity:
            # Remove lowest importance entries
            self._short_term.sort(key=lambda e: e.importance, reverse=True)
            self._short_term = self._short_term[: self.capacity]

        return entry.id

    def add_long_term(self, content: str, importance: float = 0.7, **metadata) -> str:
        """Add to long-term memory."""
        entry = MemoryEntry(content=content, entry_type="long_term", importance=importance, metadata=metadata)
        self._long_term.append(entry)
        return entry.id

    def add_semantic(self, key: str, content: str, **metadata) -> None:
        """Add to semantic memory (key-value facts)."""
        self._semantic[key] = MemoryEntry(content=content, entry_type="semantic", importance=1.0, metadata=metadata)

    def get_recent(self, n: int = 10) -> List[MemoryEntry]:
        """Get most recent short-term memories."""
        return self._short_term[-n:]

    def get_relevant(self, query: str, n: int = 5) -> List[MemoryEntry]:
        """Get most relevant memories (simple keyword matching for now)."""
        query_words = set(query.lower().split())
        scored = []

        for entry in self._short_term + self._long_term:
            entry_words = set(entry.content.lower().split())
            overlap = len(query_words & entry_words)
            if overlap > 0:
                score = overlap / len(query_words)
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:n]]

    def get_semantic(self, key: str) -> Optional[str]:
        """Get semantic memory by key."""
        entry = self._semantic.get(key)
        if entry:
            entry.accessed_at = datetime.now(timezone.utc)
            entry.access_count += 1
            return entry.content
        return None

    def clear_short_term(self) -> None:
        """Clear short-term memory."""
        self._short_term.clear()

    def summarize(self) -> str:
        """Summarize current memory state."""
        return (
            f"Memory: {len(self._short_term)} short-term, "
            f"{len(self._long_term)} long-term, "
            f"{len(self._semantic)} semantic entries"
        )


# =============================================================================
# Tool Interface
# =============================================================================


class AgentTool(ABC):
    """
    Base class for tools that agents can use.

    Tools provide agents with capabilities like:
    - Analyzing data
    - Searching documents
    - Generating code
    - Making API calls
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"{__name__}.{name}")

    @property
    @abstractmethod
    def parameters_schema(self) -> Dict[str, Any]:
        """JSON Schema for tool parameters."""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass

    def to_function_spec(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling spec."""
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters_schema},
        }


# =============================================================================
# Base Agent Class
# =============================================================================


class BaseAgent(ABC):
    """
    Base class for all agents in the multi-agent system.

    Provides:
    - LLM interaction with tool use
    - Memory management
    - Self-reflection capabilities
    - Inter-agent communication
    """

    def __init__(self, config: AgentConfig, llm_service: Any, tools: Optional[List[AgentTool]] = None):
        self.config = config
        self.llm_service = llm_service
        self.tools = {tool.name: tool for tool in (tools or [])}
        self.memory = AgentMemory(config.memory_capacity)
        self.logger = logging.getLogger(f"{__name__}.{config.role.value}")

        # State
        self.state = AgentState(agent_id=str(uuid.uuid4()), role=config.role)

        # Message queue
        self._message_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self._response_handlers: Dict[str, asyncio.Future] = {}

        # Conversation history
        self._conversation_history: List[Dict[str, Any]] = []

        self.logger.info(f"Agent {config.name} ({config.role.value}) initialized")

    @property
    def system_prompt(self) -> str:
        """Generate system prompt for the agent."""
        tool_descriptions = "\n".join([f"- {name}: {tool.description}" for name, tool in self.tools.items()])

        return f"""You are {self.config.name}, a {self.config.role.value} agent in the QBITEL Legacy Whisperer system.

{self.config.description}

Your Role:
- {self.config.role.value.replace('_', ' ').title()}

Available Tools:
{tool_descriptions if tool_descriptions else "No tools available."}

Guidelines:
1. Think step-by-step before taking actions
2. Use tools when they can help accomplish your task
3. Be precise and thorough in your analysis
4. Ask for clarification if the task is ambiguous
5. Report any errors or limitations you encounter
6. Collaborate with other agents when needed

When using tools, format your request as:
TOOL: tool_name
PARAMETERS: {{"param1": "value1", "param2": "value2"}}

Remember: Quality over speed. Take time to produce accurate results."""

    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """
        Process an incoming message and generate a response.

        This is the main entry point for agent interaction.
        """
        start_time = time.time()
        self.state.is_busy = True
        self.state.current_task_id = message.id
        self.state.last_activity = datetime.now(timezone.utc)

        AGENT_ACTIONS.labels(agent_role=self.config.role.value, action_type="process_message", status="started").inc()

        try:
            # Add message to memory
            self.memory.add_short_term(
                f"Received {message.message_type.value} from {message.sender.value}: {message.content}",
                importance=0.8 if message.priority > 0 else 0.5,
            )

            # Process based on message type
            if message.message_type == MessageType.TASK:
                response = await self._handle_task(message)
            elif message.message_type == MessageType.QUESTION:
                response = await self._handle_question(message)
            elif message.message_type == MessageType.FEEDBACK:
                response = await self._handle_feedback(message)
            else:
                response = await self._handle_generic(message)

            # Self-reflection if enabled
            if self.config.enable_reflection:
                if self.state.iteration_count % self.config.reflection_frequency == 0:
                    await self._reflect_on_response(response)

            self.state.iteration_count += 1

            AGENT_ACTIONS.labels(agent_role=self.config.role.value, action_type="process_message", status="success").inc()
            AGENT_RESPONSE_TIME.labels(agent_role=self.config.role.value).observe(time.time() - start_time)

            return response

        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            self.state.errors_count += 1
            AGENT_ACTIONS.labels(agent_role=self.config.role.value, action_type="process_message", status="error").inc()

            return AgentMessage(
                message_type=MessageType.ERROR,
                sender=self.config.role,
                recipient=message.sender,
                content=f"Error processing request: {str(e)}",
                parent_id=message.id,
                data={"error": str(e)},
            )

        finally:
            self.state.is_busy = False
            self.state.current_task_id = None

    async def _handle_task(self, message: AgentMessage) -> AgentMessage:
        """Handle a task message."""
        # Build conversation context
        context = self._build_context(message)

        # Run agentic loop
        result = await self._agentic_loop(context, message.data)

        return AgentMessage(
            message_type=MessageType.RESULT,
            sender=self.config.role,
            recipient=message.sender,
            content=result.get("summary", "Task completed"),
            data=result,
            parent_id=message.id,
        )

    async def _handle_question(self, message: AgentMessage) -> AgentMessage:
        """Handle a question from another agent."""
        # Use LLM to answer based on knowledge and memory
        relevant_memories = self.memory.get_relevant(message.content, n=5)
        memory_context = "\n".join([m.content for m in relevant_memories])

        prompt = f"""Question from {message.sender.value}:
{message.content}

Relevant context from memory:
{memory_context}

Please provide a helpful and accurate answer."""

        response = await self._call_llm(prompt)

        return AgentMessage(
            message_type=MessageType.ANSWER,
            sender=self.config.role,
            recipient=message.sender,
            content=response,
            parent_id=message.id,
        )

    async def _handle_feedback(self, message: AgentMessage) -> AgentMessage:
        """Handle feedback on previous work."""
        # Store feedback in memory
        self.memory.add_long_term(f"Feedback on task {message.parent_id}: {message.content}", importance=0.9)

        # Analyze feedback and generate improvement plan
        if self.config.enable_self_correction:
            improvement = await self._generate_improvement_plan(message.content)
            return AgentMessage(
                message_type=MessageType.STATUS,
                sender=self.config.role,
                recipient=message.sender,
                content=f"Feedback acknowledged. Improvement plan: {improvement}",
                parent_id=message.id,
                data={"improvement_plan": improvement},
            )

        return AgentMessage(
            message_type=MessageType.STATUS,
            sender=self.config.role,
            recipient=message.sender,
            content="Feedback acknowledged and stored.",
            parent_id=message.id,
        )

    async def _handle_generic(self, message: AgentMessage) -> AgentMessage:
        """Handle other message types."""
        response = await self._call_llm(f"Respond to this message: {message.content}")

        return AgentMessage(
            message_type=MessageType.RESULT,
            sender=self.config.role,
            recipient=message.sender,
            content=response,
            parent_id=message.id,
        )

    async def _agentic_loop(self, context: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the agentic loop for complex tasks.

        This implements the ReAct pattern:
        1. Reason about the current state
        2. Act using tools
        3. Observe results
        4. Repeat until done
        """
        results = {"iterations": 0, "tool_calls": [], "observations": [], "final_result": None, "summary": ""}

        iteration = 0
        max_iterations = self.config.max_iterations

        while iteration < max_iterations:
            iteration += 1
            results["iterations"] = iteration

            # Reason about current state
            reasoning_prompt = self._build_reasoning_prompt(context, task_data, results)
            response = await self._call_llm(reasoning_prompt)

            # Parse response for tool calls or final answer
            if "FINAL ANSWER:" in response:
                # Extract final answer
                final_answer = response.split("FINAL ANSWER:")[-1].strip()
                results["final_result"] = final_answer
                results["summary"] = final_answer[:500]
                break

            elif "TOOL:" in response:
                # Execute tool call
                tool_result = await self._execute_tool_from_response(response)
                results["tool_calls"].append(
                    {
                        "iteration": iteration,
                        "tool": tool_result.tool_name,
                        "status": tool_result.status.value,
                        "result_preview": str(tool_result.result)[:200],
                    }
                )

                # Add observation
                observation = f"Tool {tool_result.tool_name} returned: {tool_result.result}"
                results["observations"].append(observation)

                # Update context for next iteration
                context += f"\n\nObservation {iteration}: {observation}"

            elif "HANDOFF:" in response:
                # Request handoff to another agent
                handoff_target = self._parse_handoff(response)
                results["handoff_requested"] = handoff_target
                results["summary"] = f"Requesting handoff to {handoff_target}"
                break

            else:
                # No action specified, treat as thinking
                results["observations"].append(f"Reasoning: {response[:200]}")

            # Check for timeout
            if iteration >= max_iterations:
                results["summary"] = "Maximum iterations reached"
                break

        return results

    async def _execute_tool_from_response(self, response: str) -> ToolResult:
        """Parse and execute tool call from LLM response."""
        try:
            # Extract tool name
            tool_line = [l for l in response.split("\n") if l.startswith("TOOL:")][0]
            tool_name = tool_line.replace("TOOL:", "").strip()

            # Extract parameters
            params = {}
            if "PARAMETERS:" in response:
                params_line = response.split("PARAMETERS:")[-1].split("\n")[0]
                import json

                params = json.loads(params_line.strip())

            return await self.call_tool(tool_name, **params)

        except Exception as e:
            self.logger.error(f"Failed to parse tool call: {e}")
            return ToolResult(tool_name="unknown", status=ToolStatus.FAILURE, error=f"Failed to parse tool call: {e}")

    async def call_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool by name."""
        start_time = time.time()

        if tool_name not in self.tools:
            AGENT_TOOL_CALLS.labels(agent_role=self.config.role.value, tool_name=tool_name, status="not_found").inc()
            return ToolResult(tool_name=tool_name, status=ToolStatus.FAILURE, error=f"Tool '{tool_name}' not found")

        tool = self.tools[tool_name]

        try:
            # Execute with timeout
            result = await asyncio.wait_for(tool.execute(**kwargs), timeout=self.config.tool_timeout)
            result.execution_time = time.time() - start_time

            AGENT_TOOL_CALLS.labels(agent_role=self.config.role.value, tool_name=tool_name, status=result.status.value).inc()

            return result

        except asyncio.TimeoutError:
            AGENT_TOOL_CALLS.labels(agent_role=self.config.role.value, tool_name=tool_name, status="timeout").inc()
            return ToolResult(
                tool_name=tool_name,
                status=ToolStatus.TIMEOUT,
                error=f"Tool execution timed out after {self.config.tool_timeout}s",
                execution_time=time.time() - start_time,
            )

        except Exception as e:
            AGENT_TOOL_CALLS.labels(agent_role=self.config.role.value, tool_name=tool_name, status="error").inc()
            return ToolResult(
                tool_name=tool_name, status=ToolStatus.FAILURE, error=str(e), execution_time=time.time() - start_time
            )

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self._conversation_history[-self.config.context_window_messages :],
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self.llm_service.complete(
                messages=messages,
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            # Track token usage
            if hasattr(response, "usage"):
                self.state.total_tokens_used += response.usage.total_tokens
                AGENT_TOKEN_USAGE.labels(agent_role=self.config.role.value, token_type="total").inc(
                    response.usage.total_tokens
                )

            # Update conversation history
            self._conversation_history.append({"role": "user", "content": prompt})
            self._conversation_history.append({"role": "assistant", "content": response.content})

            return response.content

        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            raise

    async def _reflect_on_response(self, response: AgentMessage) -> ReflectionResult:
        """Perform self-reflection on the generated response."""
        AGENT_REFLECTION_COUNT.labels(agent_role=self.config.role.value, reflection_type="quality_check").inc()

        reflection_prompt = f"""Reflect on your response to ensure quality.

Your response:
{response.content[:1000]}

Consider:
1. Is the response accurate and complete?
2. Are there any errors or inconsistencies?
3. Could the response be improved?
4. Does it fully address the original request?

Provide a brief assessment and any suggestions for improvement."""

        reflection = await self._call_llm(reflection_prompt)

        # Parse reflection
        issues = []
        suggestions = []
        should_retry = "retry" in reflection.lower() or "incorrect" in reflection.lower()

        result = ReflectionResult(
            reflection_type=ReflectionType.QUALITY_CHECK,
            assessment=reflection,
            confidence=0.8 if not should_retry else 0.5,
            issues_found=issues,
            suggestions=suggestions,
            should_retry=should_retry,
        )

        # Store reflection in memory
        self.memory.add_short_term(f"Reflection: {reflection[:200]}", importance=0.6)

        return result

    async def _generate_improvement_plan(self, feedback: str) -> str:
        """Generate an improvement plan based on feedback."""
        prompt = f"""Based on this feedback, create a brief improvement plan:

Feedback:
{feedback}

Provide 2-3 specific improvements you will make."""

        return await self._call_llm(prompt)

    def _build_context(self, message: AgentMessage) -> str:
        """Build context string for task processing."""
        recent_memories = self.memory.get_recent(5)
        memory_context = "\n".join([m.content for m in recent_memories])

        return f"""Task: {message.content}

Task Data:
{message.data}

Recent Context:
{memory_context}"""

    def _build_reasoning_prompt(self, context: str, task_data: Dict[str, Any], current_results: Dict[str, Any]) -> str:
        """Build prompt for the reasoning step."""
        tools_available = ", ".join(self.tools.keys()) if self.tools else "None"

        return f"""Current Context:
{context}

Current Results:
- Iterations: {current_results['iterations']}
- Tool calls: {len(current_results['tool_calls'])}
- Observations: {current_results['observations'][-3:] if current_results['observations'] else 'None'}

Available Tools: {tools_available}

Think step by step:
1. What have I learned so far?
2. What do I still need to do?
3. Which tool should I use next, or am I ready to provide a final answer?

If you have enough information to complete the task, respond with:
FINAL ANSWER: <your complete answer>

If you need to use a tool, respond with:
TOOL: <tool_name>
PARAMETERS: {{"param1": "value1"}}

If you need another agent's help, respond with:
HANDOFF: <agent_role>
REASON: <why handoff is needed>"""

    def _parse_handoff(self, response: str) -> str:
        """Parse handoff target from response."""
        if "HANDOFF:" in response:
            return response.split("HANDOFF:")[-1].split("\n")[0].strip()
        return "orchestrator"

    def add_tool(self, tool: AgentTool) -> None:
        """Add a tool to the agent."""
        self.tools[tool.name] = tool
        self.logger.info(f"Added tool: {tool.name}")

    def get_state(self) -> AgentState:
        """Get current agent state."""
        return self.state

    async def shutdown(self) -> None:
        """Shutdown the agent."""
        self.logger.info(f"Shutting down agent {self.config.name}")
        self.memory.clear_short_term()
        self._conversation_history.clear()
        self.state.is_busy = False


__all__ = [
    "AgentRole",
    "MessageType",
    "ToolStatus",
    "ReflectionType",
    "AgentConfig",
    "AgentMessage",
    "ToolResult",
    "ReflectionResult",
    "AgentState",
    "MemoryEntry",
    "AgentMemory",
    "AgentTool",
    "BaseAgent",
]
