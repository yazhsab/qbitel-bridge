"""
QBITEL - Agent Memory and Learning Persistence

Provides long-term memory management for agents with:
- Short-term working memory (per-session)
- Long-term episodic memory (experiences)
- Semantic memory (knowledge graphs)
- Cross-agent learning transfer
- Memory consolidation and retrieval
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict

from prometheus_client import Counter, Gauge, Histogram

# Prometheus metrics
MEMORY_OPERATIONS = Counter(
    "qbitel_memory_operations_total",
    "Total memory operations",
    ["operation", "memory_type"],
)
MEMORY_SIZE = Gauge(
    "qbitel_memory_size_entries",
    "Number of entries in memory",
    ["agent_id", "memory_type"],
)
MEMORY_RETRIEVAL_TIME = Histogram(
    "qbitel_memory_retrieval_seconds",
    "Memory retrieval time",
    ["memory_type"],
)

logger = logging.getLogger(__name__)


class MemoryType(str, Enum):
    """Types of memory storage."""

    WORKING = "working"  # Short-term, per-session
    EPISODIC = "episodic"  # Long-term experiences
    SEMANTIC = "semantic"  # Knowledge and facts
    PROCEDURAL = "procedural"  # Skills and procedures


class MemoryPriority(str, Enum):
    """Memory entry priority for retention."""

    CRITICAL = "critical"  # Never forget
    HIGH = "high"  # Long retention
    NORMAL = "normal"  # Standard retention
    LOW = "low"  # Short retention


@dataclass
class MemoryEntry:
    """A single memory entry."""

    entry_id: str
    agent_id: str
    memory_type: MemoryType
    content: Dict[str, Any]
    embedding: Optional[List[float]] = None  # For semantic search
    priority: MemoryPriority = MemoryPriority.NORMAL
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if memory has expired."""
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        return False

    def touch(self) -> None:
        """Update access timestamp."""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()


@dataclass
class EpisodicMemory:
    """An episodic memory representing a specific event/experience."""

    episode_id: str
    agent_id: str
    event_type: str
    context: Dict[str, Any]  # What was happening
    action: Dict[str, Any]  # What was done
    outcome: Dict[str, Any]  # What resulted
    emotion_valence: float = 0.0  # -1 to 1 (negative to positive)
    importance: float = 0.5  # 0 to 1
    timestamp: datetime = field(default_factory=datetime.utcnow)
    related_episodes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticMemory:
    """A semantic memory representing knowledge/facts."""

    fact_id: str
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source: str = "learned"
    evidence: List[str] = field(default_factory=list)
    last_verified: datetime = field(default_factory=datetime.utcnow)
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryBackend(ABC):
    """Abstract base class for memory storage backends."""

    @abstractmethod
    async def store(self, entry: MemoryEntry) -> bool:
        """Store a memory entry."""
        pass

    @abstractmethod
    async def retrieve(
        self,
        agent_id: str,
        memory_type: MemoryType,
        query: Optional[Dict[str, Any]] = None,
        limit: int = 10,
    ) -> List[MemoryEntry]:
        """Retrieve memory entries."""
        pass

    @abstractmethod
    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        agent_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
    ) -> List[MemoryEntry]:
        """Search memories by content."""
        pass


class InMemoryBackend(MemoryBackend):
    """In-memory storage backend for development/testing."""

    def __init__(self):
        self.entries: Dict[str, MemoryEntry] = {}
        self.agent_index: Dict[str, Set[str]] = defaultdict(set)
        self.type_index: Dict[MemoryType, Set[str]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def store(self, entry: MemoryEntry) -> bool:
        async with self._lock:
            self.entries[entry.entry_id] = entry
            self.agent_index[entry.agent_id].add(entry.entry_id)
            self.type_index[entry.memory_type].add(entry.entry_id)
            return True

    async def retrieve(
        self,
        agent_id: str,
        memory_type: MemoryType,
        query: Optional[Dict[str, Any]] = None,
        limit: int = 10,
    ) -> List[MemoryEntry]:
        async with self._lock:
            # Get entries for agent and type
            agent_entries = self.agent_index.get(agent_id, set())
            type_entries = self.type_index.get(memory_type, set())
            matching_ids = agent_entries & type_entries

            entries = [self.entries[eid] for eid in matching_ids if eid in self.entries and not self.entries[eid].is_expired()]

            # Apply query filter if provided
            if query:
                entries = [e for e in entries if self._matches_query(e, query)]

            # Sort by recency and return limit
            entries.sort(key=lambda e: e.last_accessed, reverse=True)
            return entries[:limit]

    async def delete(self, entry_id: str) -> bool:
        async with self._lock:
            if entry_id in self.entries:
                entry = self.entries[entry_id]
                self.agent_index[entry.agent_id].discard(entry_id)
                self.type_index[entry.memory_type].discard(entry_id)
                del self.entries[entry_id]
                return True
            return False

    async def search(
        self,
        query: str,
        agent_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
    ) -> List[MemoryEntry]:
        async with self._lock:
            results = []
            query_lower = query.lower()

            for entry in self.entries.values():
                if entry.is_expired():
                    continue
                if agent_id and entry.agent_id != agent_id:
                    continue
                if memory_type and entry.memory_type != memory_type:
                    continue

                # Simple text search in content
                content_str = json.dumps(entry.content).lower()
                if query_lower in content_str:
                    results.append(entry)

            # Sort by relevance (access count as proxy) and recency
            results.sort(key=lambda e: (e.access_count, e.last_accessed), reverse=True)
            return results[:limit]

    def _matches_query(self, entry: MemoryEntry, query: Dict[str, Any]) -> bool:
        """Check if entry matches query criteria."""
        for key, value in query.items():
            if key == "tags":
                if not set(value).issubset(entry.tags):
                    return False
            elif key in entry.content:
                if entry.content[key] != value:
                    return False
            elif key in entry.metadata:
                if entry.metadata[key] != value:
                    return False
        return True


class AgentMemoryManager:
    """
    Manages memory for all agents.

    Provides:
    - Episodic memory for experiences
    - Semantic memory for knowledge
    - Working memory for current context
    - Memory consolidation
    - Cross-agent memory sharing
    """

    def __init__(
        self,
        backend: Optional[MemoryBackend] = None,
        embedding_service: Optional[Any] = None,
    ):
        """Initialize the memory manager."""
        self.backend = backend or InMemoryBackend()
        self.embedding_service = embedding_service

        # Working memory (in-memory, per-agent)
        self.working_memory: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.working_memory_limit = 100  # Max entries per agent

        # Episodic memory index
        self.episodic_memories: Dict[str, EpisodicMemory] = {}

        # Semantic memory (knowledge graph)
        self.semantic_memories: Dict[str, SemanticMemory] = {}
        self.knowledge_graph: Dict[str, Set[str]] = defaultdict(set)  # subject -> fact_ids

        # Memory consolidation tracking
        self.consolidation_queue: asyncio.Queue = asyncio.Queue()
        self._consolidation_task: Optional[asyncio.Task] = None

        # Statistics
        self.stats = {
            "total_stores": 0,
            "total_retrievals": 0,
            "cache_hits": 0,
            "consolidations": 0,
        }

        self._running = False
        self.logger = logging.getLogger(f"{__name__}.MemoryManager")

    async def start(self) -> None:
        """Start the memory manager."""
        self._running = True
        self._consolidation_task = asyncio.create_task(self._consolidation_loop())
        self.logger.info("Agent Memory Manager started")

    async def stop(self) -> None:
        """Stop the memory manager."""
        self._running = False
        if self._consolidation_task:
            self._consolidation_task.cancel()
            try:
                await self._consolidation_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Agent Memory Manager stopped")

    # Working Memory (Short-term)

    async def store_working(
        self,
        agent_id: str,
        content: Dict[str, Any],
    ) -> None:
        """Store in working memory."""
        self.working_memory[agent_id].append(
            {
                "content": content,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        # Limit working memory size
        if len(self.working_memory[agent_id]) > self.working_memory_limit:
            # Move oldest to episodic if important
            oldest = self.working_memory[agent_id].pop(0)
            if oldest.get("important"):
                await self.store_episodic(agent_id=agent_id, event_type="working_memory_overflow", content=oldest["content"])

        MEMORY_OPERATIONS.labels(operation="store", memory_type="working").inc()

    async def get_working(
        self,
        agent_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get recent working memory."""
        entries = self.working_memory.get(agent_id, [])
        return entries[-limit:]

    async def clear_working(self, agent_id: str) -> None:
        """Clear working memory for an agent."""
        self.working_memory[agent_id] = []

    # Episodic Memory (Experiences)

    async def store_episodic(
        self,
        agent_id: str,
        event_type: str,
        content: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        outcome: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
    ) -> str:
        """Store an episodic memory."""
        episode_id = str(uuid.uuid4())

        episode = EpisodicMemory(
            episode_id=episode_id,
            agent_id=agent_id,
            event_type=event_type,
            context=context or {},
            action=content,
            outcome=outcome or {},
            importance=importance,
        )

        self.episodic_memories[episode_id] = episode

        # Store in backend
        entry = MemoryEntry(
            entry_id=episode_id,
            agent_id=agent_id,
            memory_type=MemoryType.EPISODIC,
            content={
                "event_type": event_type,
                "context": episode.context,
                "action": episode.action,
                "outcome": episode.outcome,
            },
            priority=self._importance_to_priority(importance),
            tags={event_type},
        )

        await self.backend.store(entry)
        self.stats["total_stores"] += 1

        MEMORY_OPERATIONS.labels(operation="store", memory_type="episodic").inc()
        MEMORY_SIZE.labels(agent_id=agent_id[:8], memory_type="episodic").inc()

        self.logger.debug(f"Stored episodic memory: {episode_id[:8]} for agent {agent_id[:8]}")

        return episode_id

    async def retrieve_episodic(
        self,
        agent_id: str,
        event_type: Optional[str] = None,
        limit: int = 10,
        time_range: Optional[Tuple[datetime, datetime]] = None,
    ) -> List[EpisodicMemory]:
        """Retrieve episodic memories."""
        start_time = time.time()

        query = {}
        if event_type:
            query["event_type"] = event_type

        entries = await self.backend.retrieve(
            agent_id=agent_id,
            memory_type=MemoryType.EPISODIC,
            query=query,
            limit=limit,
        )

        # Filter by time range if specified
        if time_range:
            start, end = time_range
            entries = [e for e in entries if start <= e.created_at <= end]

        # Convert to EpisodicMemory objects
        episodes = []
        for entry in entries:
            if entry.entry_id in self.episodic_memories:
                episode = self.episodic_memories[entry.entry_id]
                episodes.append(episode)
            else:
                # Reconstruct from entry
                episode = EpisodicMemory(
                    episode_id=entry.entry_id,
                    agent_id=entry.agent_id,
                    event_type=entry.content.get("event_type", "unknown"),
                    context=entry.content.get("context", {}),
                    action=entry.content.get("action", {}),
                    outcome=entry.content.get("outcome", {}),
                    timestamp=entry.created_at,
                )
                episodes.append(episode)

        self.stats["total_retrievals"] += 1
        MEMORY_OPERATIONS.labels(operation="retrieve", memory_type="episodic").inc()
        MEMORY_RETRIEVAL_TIME.labels(memory_type="episodic").observe(time.time() - start_time)

        return episodes

    # Semantic Memory (Knowledge)

    async def store_semantic(
        self,
        subject: str,
        predicate: str,
        object_value: str,
        confidence: float = 1.0,
        source: str = "learned",
        agent_id: Optional[str] = None,
    ) -> str:
        """Store a semantic fact."""
        # Create unique ID based on triple
        fact_hash = hashlib.md5(f"{subject}:{predicate}:{object_value}".encode()).hexdigest()[:16]

        fact = SemanticMemory(
            fact_id=fact_hash,
            subject=subject,
            predicate=predicate,
            object=object_value,
            confidence=confidence,
            source=source,
        )

        # Check if exists and update confidence
        if fact_hash in self.semantic_memories:
            existing = self.semantic_memories[fact_hash]
            # Increase confidence when fact is confirmed
            existing.confidence = min(1.0, existing.confidence + 0.1)
            existing.last_verified = datetime.utcnow()
            existing.evidence.append(source)
        else:
            self.semantic_memories[fact_hash] = fact
            self.knowledge_graph[subject].add(fact_hash)

        # Store in backend
        entry = MemoryEntry(
            entry_id=fact_hash,
            agent_id=agent_id or "global",
            memory_type=MemoryType.SEMANTIC,
            content={
                "subject": subject,
                "predicate": predicate,
                "object": object_value,
            },
            priority=MemoryPriority.HIGH,  # Knowledge is important
            tags={subject, predicate},
        )

        await self.backend.store(entry)
        self.stats["total_stores"] += 1

        MEMORY_OPERATIONS.labels(operation="store", memory_type="semantic").inc()

        return fact_hash

    async def retrieve_semantic(
        self,
        query: str,
        agent_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Retrieve semantic knowledge matching query."""
        start_time = time.time()

        # Search in backend
        entries = await self.backend.search(
            query=query,
            agent_id=agent_id,
            memory_type=MemoryType.SEMANTIC,
            limit=limit,
        )

        results = []
        for entry in entries:
            if entry.entry_id in self.semantic_memories:
                fact = self.semantic_memories[entry.entry_id]
                results.append(
                    {
                        "subject": fact.subject,
                        "predicate": fact.predicate,
                        "object": fact.object,
                        "confidence": fact.confidence,
                        "source": fact.source,
                    }
                )
            else:
                results.append(entry.content)

        self.stats["total_retrievals"] += 1
        MEMORY_OPERATIONS.labels(operation="retrieve", memory_type="semantic").inc()
        MEMORY_RETRIEVAL_TIME.labels(memory_type="semantic").observe(time.time() - start_time)

        return results

    async def query_knowledge_graph(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object_value: Optional[str] = None,
    ) -> List[SemanticMemory]:
        """Query the knowledge graph."""
        results = []

        for fact in self.semantic_memories.values():
            match = True
            if subject and fact.subject != subject:
                match = False
            if predicate and fact.predicate != predicate:
                match = False
            if object_value and fact.object != object_value:
                match = False

            if match:
                results.append(fact)

        # Sort by confidence
        results.sort(key=lambda f: f.confidence, reverse=True)
        return results

    # Cross-Agent Memory Sharing

    async def share_memory(
        self,
        source_agent_id: str,
        target_agent_id: str,
        memory_type: MemoryType,
        filter_tags: Optional[Set[str]] = None,
    ) -> int:
        """Share memories from one agent to another."""
        entries = await self.backend.retrieve(
            agent_id=source_agent_id,
            memory_type=memory_type,
            limit=100,
        )

        shared_count = 0
        for entry in entries:
            # Apply tag filter
            if filter_tags and not filter_tags.issubset(entry.tags):
                continue

            # Create copy for target agent
            shared_entry = MemoryEntry(
                entry_id=str(uuid.uuid4()),
                agent_id=target_agent_id,
                memory_type=entry.memory_type,
                content=entry.content.copy(),
                priority=entry.priority,
                tags=entry.tags.copy(),
                metadata={
                    **entry.metadata,
                    "shared_from": source_agent_id,
                    "original_id": entry.entry_id,
                },
            )

            await self.backend.store(shared_entry)
            shared_count += 1

        self.logger.info(f"Shared {shared_count} memories from {source_agent_id[:8]} " f"to {target_agent_id[:8]}")

        return shared_count

    # Memory Consolidation

    async def consolidate(self, agent_id: str) -> None:
        """Consolidate memories for an agent (move important short-term to long-term)."""
        # Get working memory
        working = await self.get_working(agent_id)

        for item in working:
            content = item.get("content", {})

            # Determine importance
            importance = self._calculate_importance(content)

            if importance > 0.5:
                # Store as episodic
                await self.store_episodic(
                    agent_id=agent_id,
                    event_type="consolidated",
                    content=content,
                    importance=importance,
                )

        # Clear consolidated items
        await self.clear_working(agent_id)
        self.stats["consolidations"] += 1

        self.logger.debug(f"Consolidated memories for agent {agent_id[:8]}")

    async def _consolidation_loop(self) -> None:
        """Background loop for memory consolidation."""
        while self._running:
            try:
                # Process consolidation requests
                try:
                    agent_id = await asyncio.wait_for(self.consolidation_queue.get(), timeout=60.0)
                    await self.consolidate(agent_id)
                except asyncio.TimeoutError:
                    pass

                # Periodic cleanup of expired memories
                await self._cleanup_expired()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Consolidation error: {e}")
                await asyncio.sleep(10)

    async def _cleanup_expired(self) -> None:
        """Clean up expired memory entries from all backends."""
        for backend in self.backends.values():
            try:
                all_entries = await backend.retrieve(limit=1000)
                for entry in all_entries:
                    if hasattr(entry, "is_expired") and entry.is_expired():
                        await backend.delete(entry.id)
            except Exception as e:
                self.logger.warning(f"Cleanup error for backend: {e}")

    def _calculate_importance(self, content: Dict[str, Any]) -> float:
        """Calculate importance score for memory content."""
        importance = 0.5

        # Increase for certain content types
        if content.get("success") is True:
            importance += 0.1
        if content.get("error"):
            importance += 0.2  # Errors are important to remember
        if content.get("task_type") in ["security", "threat", "critical"]:
            importance += 0.2

        return min(1.0, importance)

    def _importance_to_priority(self, importance: float) -> MemoryPriority:
        """Convert importance score to memory priority."""
        if importance >= 0.8:
            return MemoryPriority.CRITICAL
        elif importance >= 0.6:
            return MemoryPriority.HIGH
        elif importance >= 0.4:
            return MemoryPriority.NORMAL
        else:
            return MemoryPriority.LOW

    async def schedule_consolidation(self, agent_id: str) -> None:
        """Schedule memory consolidation for an agent."""
        await self.consolidation_queue.put(agent_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get memory manager statistics."""
        return {
            **self.stats,
            "episodic_count": len(self.episodic_memories),
            "semantic_count": len(self.semantic_memories),
            "working_memory_agents": len(self.working_memory),
            "knowledge_graph_subjects": len(self.knowledge_graph),
        }
