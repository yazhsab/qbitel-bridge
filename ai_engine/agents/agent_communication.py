"""
CRONOS AI - Agent Communication Protocol (ACP)

Provides direct agent-to-agent communication with:
- Point-to-point messaging
- Pub/Sub channels
- Request/Response patterns
- Broadcast capabilities
- Message routing and delivery guarantees
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING
from collections import defaultdict

from prometheus_client import Counter, Histogram

if TYPE_CHECKING:
    from .base_agent import BaseAgent

# Prometheus metrics
ACP_MESSAGES_SENT = Counter(
    "cronos_acp_messages_sent_total",
    "Total messages sent via ACP",
    ["message_type", "sender_type"],
)
ACP_MESSAGES_RECEIVED = Counter(
    "cronos_acp_messages_received_total",
    "Total messages received via ACP",
    ["message_type", "receiver_type"],
)
ACP_MESSAGE_LATENCY = Histogram(
    "cronos_acp_message_latency_seconds",
    "Message delivery latency",
    ["message_type"],
)

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """Types of inter-agent messages."""

    # Control messages
    HEARTBEAT = "heartbeat"
    REGISTER = "register"
    UNREGISTER = "unregister"
    STATUS_REQUEST = "status_request"
    STATUS_RESPONSE = "status_response"

    # Task messages
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    TASK_PROGRESS = "task_progress"
    TASK_CANCEL = "task_cancel"

    # Collaboration messages
    PROPOSAL = "proposal"
    VOTE = "vote"
    CONSENSUS_REQUEST = "consensus_request"
    CONSENSUS_RESULT = "consensus_result"
    NEGOTIATION = "negotiation"

    # Data messages
    DATA_SHARE = "data_share"
    DATA_REQUEST = "data_request"
    DATA_RESPONSE = "data_response"

    # Event messages
    EVENT = "event"
    ALERT = "alert"
    BROADCAST = "broadcast"

    # Custom
    CUSTOM = "custom"


class MessagePriority(int, Enum):
    """Message priority levels."""

    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


class DeliveryStatus(str, Enum):
    """Message delivery status."""

    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    ACKNOWLEDGED = "acknowledged"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class AgentMessage:
    """Message exchanged between agents."""

    message_id: str
    message_type: MessageType
    sender_id: str
    sender_type: str
    payload: Dict[str, Any]
    recipient_id: Optional[str] = None  # None for broadcast
    channel: Optional[str] = None  # For pub/sub
    priority: MessagePriority = MessagePriority.NORMAL
    correlation_id: Optional[str] = None  # For request/response
    reply_to: Optional[str] = None  # Message ID to reply to
    ttl_seconds: int = 300  # Time to live
    require_ack: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if message has expired."""
        age = (datetime.utcnow() - self.created_at).total_seconds()
        return age > self.ttl_seconds


@dataclass
class Channel:
    """Pub/Sub channel for agent communication."""

    channel_id: str
    name: str
    description: str = ""
    subscribers: Set[str] = field(default_factory=set)
    message_history: List[AgentMessage] = field(default_factory=list)
    max_history: int = 100
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, message: AgentMessage) -> None:
        """Add message to history."""
        self.message_history.append(message)
        if len(self.message_history) > self.max_history:
            self.message_history = self.message_history[-self.max_history:]


@dataclass
class PendingRequest:
    """Tracking for request/response pattern."""

    request_id: str
    sender_id: str
    recipient_id: str
    message: AgentMessage
    future: asyncio.Future
    timeout: float
    created_at: datetime = field(default_factory=datetime.utcnow)


class AgentCommunicationProtocol:
    """
    Agent Communication Protocol implementation.

    Provides:
    - Direct agent-to-agent messaging
    - Pub/Sub channels for topic-based communication
    - Request/Response pattern with timeouts
    - Broadcast capabilities
    - Message queuing and delivery
    - Heartbeat monitoring
    """

    def __init__(self, max_queue_size: int = 10000):
        """Initialize the communication protocol."""
        self.agents: Dict[str, "BaseAgent"] = {}
        self.agent_queues: Dict[str, asyncio.Queue] = {}
        self.channels: Dict[str, Channel] = {}
        self.pending_requests: Dict[str, PendingRequest] = {}

        # Message routing
        self._message_handlers: Dict[str, Callable] = {}
        self._global_handlers: List[Callable] = []

        # Configuration
        self.max_queue_size = max_queue_size
        self.default_timeout = 30.0

        # Background tasks
        self._running = False
        self._message_processor_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None

        # Statistics
        self.stats = {
            "messages_sent": 0,
            "messages_delivered": 0,
            "messages_failed": 0,
            "broadcasts_sent": 0,
            "requests_completed": 0,
            "requests_timeout": 0,
        }

        self.logger = logging.getLogger(f"{__name__}.ACP")

    async def start(self) -> None:
        """Start the communication protocol."""
        if self._running:
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.logger.info("Agent Communication Protocol started")

    async def stop(self) -> None:
        """Stop the communication protocol."""
        self._running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Cancel pending requests
        for request_id, pending in self.pending_requests.items():
            if not pending.future.done():
                pending.future.cancel()

        self.logger.info("Agent Communication Protocol stopped")

    async def register_agent(self, agent: "BaseAgent") -> None:
        """Register an agent with the protocol."""
        agent_id = agent.agent_id
        self.agents[agent_id] = agent
        self.agent_queues[agent_id] = asyncio.Queue(maxsize=self.max_queue_size)

        # Start message delivery task for this agent
        asyncio.create_task(self._deliver_messages(agent_id))

        self.logger.info(f"Agent registered: {agent.agent_type} ({agent_id[:8]})")

        # Broadcast registration event
        await self.broadcast(
            sender=agent,
            message_type=MessageType.REGISTER,
            payload={
                "agent_id": agent_id,
                "agent_type": agent.agent_type,
                "capabilities": [c.value for c in agent.capabilities],
            }
        )

    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the protocol."""
        if agent_id not in self.agents:
            return

        agent = self.agents[agent_id]

        # Broadcast unregistration event
        await self.broadcast(
            sender=agent,
            message_type=MessageType.UNREGISTER,
            payload={"agent_id": agent_id}
        )

        # Remove from channels
        for channel in self.channels.values():
            channel.subscribers.discard(agent_id)

        # Clean up
        del self.agents[agent_id]
        del self.agent_queues[agent_id]

        self.logger.info(f"Agent unregistered: {agent_id[:8]}")

    async def send(
        self,
        sender: "BaseAgent",
        recipient_id: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        require_ack: bool = False,
        ttl_seconds: int = 300,
        correlation_id: Optional[str] = None,
    ) -> str:
        """
        Send a message to a specific agent.

        Args:
            sender: The sending agent
            recipient_id: ID of the recipient agent
            message_type: Type of message
            payload: Message payload
            priority: Message priority
            require_ack: Whether to require acknowledgment
            ttl_seconds: Time to live
            correlation_id: Correlation ID for tracking

        Returns:
            Message ID
        """
        message = AgentMessage(
            message_id=str(uuid.uuid4()),
            message_type=message_type,
            sender_id=sender.agent_id,
            sender_type=sender.agent_type,
            recipient_id=recipient_id,
            payload=payload,
            priority=priority,
            require_ack=require_ack,
            ttl_seconds=ttl_seconds,
            correlation_id=correlation_id,
        )

        await self._route_message(message)
        return message.message_id

    async def request(
        self,
        sender: Optional["BaseAgent"] = None,
        target_agent_id: str = "",
        message_type: str = "task_request",
        payload: Dict[str, Any] = None,
        timeout: float = None,
        **kwargs
    ) -> Any:
        """
        Send a request and wait for response (request/response pattern).

        Args:
            sender: The sending agent (optional)
            target_agent_id: ID of the target agent
            message_type: Type of request
            payload: Request payload
            timeout: Response timeout in seconds

        Returns:
            Response payload
        """
        timeout = timeout or self.default_timeout
        correlation_id = str(uuid.uuid4())

        # Create future for response
        future = asyncio.get_event_loop().create_future()

        sender_id = sender.agent_id if sender else "system"
        sender_type = sender.agent_type if sender else "system"

        message = AgentMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType(message_type) if isinstance(message_type, str) else message_type,
            sender_id=sender_id,
            sender_type=sender_type,
            recipient_id=target_agent_id,
            payload=payload or {},
            priority=MessagePriority.NORMAL,
            correlation_id=correlation_id,
            require_ack=True,
        )

        # Track pending request
        pending = PendingRequest(
            request_id=correlation_id,
            sender_id=sender_id,
            recipient_id=target_agent_id,
            message=message,
            future=future,
            timeout=timeout,
        )
        self.pending_requests[correlation_id] = pending

        # Send request
        await self._route_message(message)

        try:
            # Wait for response with timeout
            result = await asyncio.wait_for(future, timeout=timeout)
            self.stats["requests_completed"] += 1
            return result
        except asyncio.TimeoutError:
            self.stats["requests_timeout"] += 1
            self.logger.warning(f"Request timeout: {correlation_id}")
            raise
        finally:
            # Cleanup
            if correlation_id in self.pending_requests:
                del self.pending_requests[correlation_id]

    async def respond(
        self,
        sender: "BaseAgent",
        original_message: AgentMessage,
        payload: Dict[str, Any],
    ) -> None:
        """
        Send a response to a request.

        Args:
            sender: The responding agent
            original_message: The original request message
            payload: Response payload
        """
        response = AgentMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.TASK_RESPONSE,
            sender_id=sender.agent_id,
            sender_type=sender.agent_type,
            recipient_id=original_message.sender_id,
            payload=payload,
            correlation_id=original_message.correlation_id,
            reply_to=original_message.message_id,
        )

        await self._route_message(response)

    async def broadcast(
        self,
        sender: "BaseAgent",
        message_type: MessageType,
        payload: Dict[str, Any],
        exclude: Optional[Set[str]] = None,
    ) -> None:
        """
        Broadcast a message to all registered agents.

        Args:
            sender: The sending agent
            message_type: Type of message
            payload: Message payload
            exclude: Set of agent IDs to exclude
        """
        exclude = exclude or set()
        exclude.add(sender.agent_id)  # Don't send to self

        message = AgentMessage(
            message_id=str(uuid.uuid4()),
            message_type=message_type,
            sender_id=sender.agent_id,
            sender_type=sender.agent_type,
            payload=payload,
            recipient_id=None,  # Broadcast marker
        )

        for agent_id in self.agents.keys():
            if agent_id not in exclude:
                # Create copy for each recipient
                agent_message = AgentMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=message.message_type,
                    sender_id=message.sender_id,
                    sender_type=message.sender_type,
                    recipient_id=agent_id,
                    payload=message.payload,
                    metadata={"broadcast_id": message.message_id}
                )
                await self._route_message(agent_message)

        self.stats["broadcasts_sent"] += 1

    # Channel (Pub/Sub) Methods

    def create_channel(
        self,
        name: str,
        description: str = "",
    ) -> Channel:
        """Create a new pub/sub channel."""
        channel_id = str(uuid.uuid4())
        channel = Channel(
            channel_id=channel_id,
            name=name,
            description=description,
        )
        self.channels[channel_id] = channel
        self.logger.info(f"Channel created: {name} ({channel_id[:8]})")
        return channel

    def get_channel(self, name: str) -> Optional[Channel]:
        """Get a channel by name."""
        for channel in self.channels.values():
            if channel.name == name:
                return channel
        return None

    async def subscribe(
        self,
        agent: "BaseAgent",
        channel_name: str,
    ) -> bool:
        """Subscribe an agent to a channel."""
        channel = self.get_channel(channel_name)
        if not channel:
            # Auto-create channel
            channel = self.create_channel(channel_name)

        channel.subscribers.add(agent.agent_id)
        self.logger.debug(f"Agent {agent.agent_id[:8]} subscribed to {channel_name}")
        return True

    async def unsubscribe(
        self,
        agent: "BaseAgent",
        channel_name: str,
    ) -> bool:
        """Unsubscribe an agent from a channel."""
        channel = self.get_channel(channel_name)
        if channel:
            channel.subscribers.discard(agent.agent_id)
            return True
        return False

    async def publish(
        self,
        sender: "BaseAgent",
        channel_name: str,
        payload: Dict[str, Any],
        message_type: MessageType = MessageType.EVENT,
    ) -> int:
        """
        Publish a message to a channel.

        Returns:
            Number of subscribers notified
        """
        channel = self.get_channel(channel_name)
        if not channel:
            self.logger.warning(f"Channel not found: {channel_name}")
            return 0

        message = AgentMessage(
            message_id=str(uuid.uuid4()),
            message_type=message_type,
            sender_id=sender.agent_id,
            sender_type=sender.agent_type,
            channel=channel_name,
            payload=payload,
        )

        # Store in channel history
        channel.add_message(message)

        # Deliver to subscribers
        delivered = 0
        for subscriber_id in channel.subscribers:
            if subscriber_id != sender.agent_id:
                subscriber_message = AgentMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=message.message_type,
                    sender_id=message.sender_id,
                    sender_type=message.sender_type,
                    recipient_id=subscriber_id,
                    channel=channel_name,
                    payload=message.payload,
                    metadata={"channel_message_id": message.message_id}
                )
                try:
                    await self._route_message(subscriber_message)
                    delivered += 1
                except Exception as e:
                    self.logger.error(f"Failed to deliver to {subscriber_id}: {e}")

        return delivered

    # Heartbeat

    async def send_heartbeat(self, agent: "BaseAgent") -> None:
        """Send a heartbeat for an agent."""
        message = AgentMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.HEARTBEAT,
            sender_id=agent.agent_id,
            sender_type=agent.agent_type,
            payload={
                "status": agent.status.value,
                "queue_size": agent.task_queue.qsize(),
                "active_tasks": len(agent.active_tasks),
                "timestamp": datetime.utcnow().isoformat(),
            },
            ttl_seconds=60,
        )

        # Broadcast heartbeat (lightweight)
        for handler in self._global_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
            except Exception as e:
                self.logger.error(f"Heartbeat handler error: {e}")

    # Internal Methods

    async def _route_message(self, message: AgentMessage) -> None:
        """Route a message to its destination."""
        if message.is_expired():
            self.logger.warning(f"Message expired: {message.message_id}")
            self.stats["messages_failed"] += 1
            return

        recipient_id = message.recipient_id

        if recipient_id and recipient_id in self.agent_queues:
            try:
                # Priority-based insertion would require a priority queue
                await self.agent_queues[recipient_id].put(message)
                self.stats["messages_sent"] += 1
                ACP_MESSAGES_SENT.labels(
                    message_type=message.message_type.value,
                    sender_type=message.sender_type
                ).inc()
            except asyncio.QueueFull:
                self.logger.error(f"Queue full for agent: {recipient_id[:8]}")
                self.stats["messages_failed"] += 1
        else:
            self.logger.warning(f"Unknown recipient: {recipient_id}")
            self.stats["messages_failed"] += 1

    async def _deliver_messages(self, agent_id: str) -> None:
        """Deliver messages to an agent."""
        queue = self.agent_queues.get(agent_id)
        if not queue:
            return

        while self._running and agent_id in self.agents:
            try:
                message = await asyncio.wait_for(queue.get(), timeout=1.0)

                agent = self.agents.get(agent_id)
                if not agent:
                    continue

                # Calculate latency
                latency = (datetime.utcnow() - message.created_at).total_seconds()
                ACP_MESSAGE_LATENCY.labels(
                    message_type=message.message_type.value
                ).observe(latency)

                # Check if this is a response to a pending request
                if message.correlation_id and message.correlation_id in self.pending_requests:
                    pending = self.pending_requests[message.correlation_id]
                    if not pending.future.done():
                        pending.future.set_result(message.payload)
                    continue

                # Deliver to agent
                try:
                    response = await agent.handle_message(message)

                    # If message requires response and we got one, send it
                    if message.require_ack and response is not None:
                        await self.respond(agent, message, response)

                    self.stats["messages_delivered"] += 1
                    ACP_MESSAGES_RECEIVED.labels(
                        message_type=message.message_type.value,
                        receiver_type=agent.agent_type
                    ).inc()

                except Exception as e:
                    self.logger.error(f"Message handling error: {e}")

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Message delivery error: {e}")

    async def _cleanup_loop(self) -> None:
        """Periodically clean up expired messages and requests."""
        while self._running:
            try:
                # Clean up expired pending requests
                now = datetime.utcnow()
                expired = []
                for request_id, pending in self.pending_requests.items():
                    age = (now - pending.created_at).total_seconds()
                    if age > pending.timeout:
                        if not pending.future.done():
                            pending.future.set_exception(
                                asyncio.TimeoutError(f"Request {request_id} expired")
                            )
                        expired.append(request_id)

                for request_id in expired:
                    del self.pending_requests[request_id]

                await asyncio.sleep(10)  # Cleanup every 10 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(10)

    def register_handler(
        self,
        message_type: MessageType,
        handler: Callable,
    ) -> None:
        """Register a handler for a specific message type."""
        self._message_handlers[message_type.value] = handler

    def register_global_handler(self, handler: Callable) -> None:
        """Register a global handler for all messages."""
        self._global_handlers.append(handler)

    def get_stats(self) -> Dict[str, Any]:
        """Get communication statistics."""
        return {
            **self.stats,
            "registered_agents": len(self.agents),
            "active_channels": len(self.channels),
            "pending_requests": len(self.pending_requests),
        }

    def get_agents(self) -> List[Dict[str, Any]]:
        """Get list of registered agents."""
        return [
            {
                "agent_id": agent.agent_id,
                "agent_type": agent.agent_type,
                "status": agent.status.value,
                "capabilities": [c.value for c in agent.capabilities],
            }
            for agent in self.agents.values()
        ]
