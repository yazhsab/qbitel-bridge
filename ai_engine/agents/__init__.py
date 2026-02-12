"""
QBITEL - Multi-Agent Orchestration Framework

This module provides a comprehensive multi-agent orchestration system with:
- Planning/Supervisor Agent for task decomposition and coordination
- Agent Communication Protocol for direct agent-to-agent messaging
- Dynamic Agent Pool Manager for on-demand agent scaling
- Agent Memory/Learning Persistence for long-term learning
- Agent Collaboration Framework for consensus and negotiation

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │               Multi-Agent Orchestration Layer                │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │  ┌─────────────────────────────────────────────────────────┐│
    │  │            Planning/Supervisor Agent (Meta-Agent)        ││
    │  │  - Task decomposition                                    ││
    │  │  - Agent assignment                                      ││
    │  │  - Execution monitoring                                  ││
    │  │  - Plan adaptation                                       ││
    │  └─────────────────────────────────────────────────────────┘│
    │                            │                                 │
    │                            ▼                                 │
    │  ┌─────────────────────────────────────────────────────────┐│
    │  │          Agent Communication Protocol (ACP)              ││
    │  │  - Direct agent-to-agent messaging                       ││
    │  │  - Pub/Sub channels                                      ││
    │  │  - Request/Response patterns                             ││
    │  │  - Broadcast capabilities                                ││
    │  └─────────────────────────────────────────────────────────┘│
    │                            │                                 │
    │            ┌───────────────┼───────────────┐                │
    │            ▼               ▼               ▼                │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
    │  │   Agent 1   │  │   Agent 2   │  │   Agent N   │         │
    │  │  (Pooled)   │  │  (Pooled)   │  │  (Pooled)   │         │
    │  └─────────────┘  └─────────────┘  └─────────────┘         │
    │            │               │               │                │
    │            └───────────────┼───────────────┘                │
    │                            ▼                                 │
    │  ┌─────────────────────────────────────────────────────────┐│
    │  │           Agent Memory & Learning Persistence            ││
    │  │  - Short-term working memory                             ││
    │  │  - Long-term episodic memory                             ││
    │  │  - Semantic knowledge store                              ││
    │  │  - Cross-agent learning transfer                         ││
    │  └─────────────────────────────────────────────────────────┘│
    │                                                              │
    └─────────────────────────────────────────────────────────────┘
"""

from .base_agent import (
    BaseAgent,
    AgentCapability,
    AgentStatus,
    AgentPriority,
    AgentConfig,
)
from .planning_agent import (
    PlanningAgent,
    TaskPlan,
    SubTask,
    PlanStatus,
    ExecutionStrategy,
)
from .agent_communication import (
    AgentCommunicationProtocol,
    AgentMessage,
    MessageType,
    MessagePriority,
    Channel,
)
from .agent_pool import (
    AgentPoolManager,
    PoolConfig,
    AgentInstance,
    ScalingPolicy,
)
from .agent_memory import (
    AgentMemoryManager,
    MemoryType,
    MemoryPriority,
    MemoryEntry,
    EpisodicMemory,
    SemanticMemory,
    MemoryBackend,
    InMemoryBackend,
)

# New: Persistent memory with compression and relevance decay
try:
    from .persistent_memory import (
        SQLitePersistentBackend,
        RedisPersistentBackend,
        PersistentMemoryEntry,
        MemoryCompressor,
        MemoryConsolidationService,
        RelevanceDecayConfig,
        CompressionConfig,
    )

    _PERSISTENT_MEMORY_AVAILABLE = True
except ImportError:
    _PERSISTENT_MEMORY_AVAILABLE = False
    SQLitePersistentBackend = None
    RedisPersistentBackend = None

from .agent_collaboration import (
    CollaborationFramework,
    ConsensusProtocol,
    NegotiationStrategy,
    CollaborationSession,
)
from .multi_agent_orchestrator import (
    MultiAgentOrchestrator,
    OrchestratorConfig,
)

__all__ = [
    # Base Agent
    "BaseAgent",
    "AgentCapability",
    "AgentStatus",
    "AgentPriority",
    "AgentConfig",
    # Planning Agent
    "PlanningAgent",
    "TaskPlan",
    "SubTask",
    "PlanStatus",
    "ExecutionStrategy",
    # Communication
    "AgentCommunicationProtocol",
    "AgentMessage",
    "MessageType",
    "MessagePriority",
    "Channel",
    # Pool Management
    "AgentPoolManager",
    "PoolConfig",
    "AgentInstance",
    "ScalingPolicy",
    # Memory
    "AgentMemoryManager",
    "MemoryType",
    "MemoryEntry",
    "EpisodicMemory",
    "SemanticMemory",
    # Collaboration
    "CollaborationFramework",
    "ConsensusProtocol",
    "NegotiationStrategy",
    "CollaborationSession",
    # Orchestrator
    "MultiAgentOrchestrator",
    "OrchestratorConfig",
]
