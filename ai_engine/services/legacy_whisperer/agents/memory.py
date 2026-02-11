"""
QBITEL - Multi-Agent Memory System

Provides shared memory and context management for multi-agent collaboration:
- SharedMemory: Cross-agent memory sharing
- ConversationMemory: Thread-based conversation tracking
- EpisodicMemory: Task and event memory
- SemanticMemory: Knowledge and facts storage
"""

import asyncio
import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Set
from enum import Enum
import numpy as np

from .base import AgentRole, AgentMessage


logger = logging.getLogger(__name__)


# =============================================================================
# Memory Types
# =============================================================================

class MemoryType(str, Enum):
    """Types of memory entries."""

    CONVERSATION = "conversation"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"


@dataclass
class SharedMemoryEntry:
    """Entry in shared memory."""

    id: str
    content: str
    memory_type: MemoryType
    created_by: AgentRole
    importance: float = 0.5
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    last_accessed_by: Optional[AgentRole] = None


@dataclass
class ConversationThread:
    """A conversation thread between agents."""

    thread_id: str
    participants: Set[AgentRole]
    messages: List[AgentMessage] = field(default_factory=list)
    topic: str = ""
    status: str = "active"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Episode:
    """An episodic memory entry representing a task or event."""

    episode_id: str
    task_description: str
    agents_involved: Set[AgentRole]
    start_time: datetime
    end_time: Optional[datetime] = None
    outcome: str = ""
    success: bool = False
    observations: List[str] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Shared Memory
# =============================================================================

class SharedMemory:
    """
    Shared memory space accessible by all agents.

    Provides:
    - Cross-agent knowledge sharing
    - Memory persistence
    - Relevance-based retrieval
    - Memory consolidation
    """

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self._entries: Dict[str, SharedMemoryEntry] = {}
        self._by_type: Dict[MemoryType, List[str]] = defaultdict(list)
        self._by_tag: Dict[str, List[str]] = defaultdict(list)
        self._by_agent: Dict[AgentRole, List[str]] = defaultdict(list)
        self._lock = asyncio.Lock()

        logger.info(f"Shared memory initialized with capacity {capacity}")

    async def store(
        self,
        content: str,
        memory_type: MemoryType,
        created_by: AgentRole,
        importance: float = 0.5,
        tags: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None
    ) -> str:
        """
        Store an entry in shared memory.

        Returns:
            Entry ID
        """
        async with self._lock:
            # Generate ID
            entry_id = hashlib.sha256(
                f"{content}{created_by.value}{time.time()}".encode()
            ).hexdigest()[:16]

            entry = SharedMemoryEntry(
                id=entry_id,
                content=content,
                memory_type=memory_type,
                created_by=created_by,
                importance=importance,
                tags=tags or set(),
                metadata=metadata or {},
                embedding=embedding
            )

            # Store entry
            self._entries[entry_id] = entry

            # Index by type
            self._by_type[memory_type].append(entry_id)

            # Index by tags
            for tag in entry.tags:
                self._by_tag[tag].append(entry_id)

            # Index by agent
            self._by_agent[created_by].append(entry_id)

            # Enforce capacity
            if len(self._entries) > self.capacity:
                await self._evict_old_entries()

            logger.debug(f"Stored memory entry {entry_id} by {created_by.value}")

            return entry_id

    async def retrieve(
        self,
        entry_id: str,
        accessor: Optional[AgentRole] = None
    ) -> Optional[SharedMemoryEntry]:
        """Retrieve a specific entry by ID."""
        async with self._lock:
            entry = self._entries.get(entry_id)
            if entry and accessor:
                entry.access_count += 1
                entry.last_accessed_by = accessor
            return entry

    async def search(
        self,
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        tags: Optional[Set[str]] = None,
        created_by: Optional[AgentRole] = None,
        limit: int = 10
    ) -> List[SharedMemoryEntry]:
        """
        Search memory entries.

        Uses simple keyword matching. For production, integrate
        with vector similarity search.
        """
        async with self._lock:
            candidates = list(self._entries.values())

            # Filter by type
            if memory_types:
                candidates = [
                    e for e in candidates if e.memory_type in memory_types
                ]

            # Filter by tags
            if tags:
                candidates = [
                    e for e in candidates if e.tags & tags
                ]

            # Filter by creator
            if created_by:
                candidates = [
                    e for e in candidates if e.created_by == created_by
                ]

            # Score by relevance (simple keyword matching)
            query_words = set(query.lower().split())
            scored = []
            for entry in candidates:
                entry_words = set(entry.content.lower().split())
                overlap = len(query_words & entry_words)
                if overlap > 0 or not query:
                    relevance = overlap / max(len(query_words), 1)
                    # Factor in importance and recency
                    recency = 1.0 / (1 + (datetime.now(timezone.utc) - entry.created_at).total_seconds() / 3600)
                    score = relevance * 0.5 + entry.importance * 0.3 + recency * 0.2
                    scored.append((score, entry))

            # Sort by score and return top results
            scored.sort(key=lambda x: x[0], reverse=True)
            return [entry for _, entry in scored[:limit]]

    async def get_by_type(
        self,
        memory_type: MemoryType,
        limit: int = 50
    ) -> List[SharedMemoryEntry]:
        """Get entries by memory type."""
        async with self._lock:
            entry_ids = self._by_type.get(memory_type, [])[-limit:]
            return [
                self._entries[eid] for eid in entry_ids
                if eid in self._entries
            ]

    async def get_by_agent(
        self,
        agent: AgentRole,
        limit: int = 50
    ) -> List[SharedMemoryEntry]:
        """Get entries created by a specific agent."""
        async with self._lock:
            entry_ids = self._by_agent.get(agent, [])[-limit:]
            return [
                self._entries[eid] for eid in entry_ids
                if eid in self._entries
            ]

    async def update_importance(
        self,
        entry_id: str,
        new_importance: float
    ) -> None:
        """Update the importance of an entry."""
        async with self._lock:
            if entry_id in self._entries:
                self._entries[entry_id].importance = new_importance
                self._entries[entry_id].updated_at = datetime.now(timezone.utc)

    async def _evict_old_entries(self) -> None:
        """Evict oldest, least important entries."""
        # Sort by importance and recency
        scored = []
        for entry_id, entry in self._entries.items():
            age = (datetime.now(timezone.utc) - entry.created_at).total_seconds()
            score = entry.importance - (age / 86400)  # Decrease score with age
            scored.append((score, entry_id))

        scored.sort(key=lambda x: x[0])

        # Remove lowest scoring entries
        to_remove = self.capacity // 10  # Remove 10%
        for _, entry_id in scored[:to_remove]:
            entry = self._entries.pop(entry_id, None)
            if entry:
                # Clean up indexes
                for tag in entry.tags:
                    if entry_id in self._by_tag[tag]:
                        self._by_tag[tag].remove(entry_id)
                if entry_id in self._by_type[entry.memory_type]:
                    self._by_type[entry.memory_type].remove(entry_id)
                if entry_id in self._by_agent[entry.created_by]:
                    self._by_agent[entry.created_by].remove(entry_id)

        logger.debug(f"Evicted {to_remove} memory entries")

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "total_entries": len(self._entries),
            "capacity": self.capacity,
            "by_type": {t.value: len(ids) for t, ids in self._by_type.items()},
            "by_agent": {a.value: len(ids) for a, ids in self._by_agent.items()},
        }


# =============================================================================
# Conversation Memory
# =============================================================================

class ConversationMemory:
    """
    Manages conversation threads between agents.

    Tracks:
    - Active conversations
    - Message history
    - Context for each thread
    """

    def __init__(self, max_threads: int = 100, max_messages_per_thread: int = 200):
        self.max_threads = max_threads
        self.max_messages_per_thread = max_messages_per_thread
        self._threads: Dict[str, ConversationThread] = {}
        self._by_participant: Dict[AgentRole, Set[str]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def create_thread(
        self,
        participants: Set[AgentRole],
        topic: str = ""
    ) -> str:
        """Create a new conversation thread."""
        async with self._lock:
            thread_id = hashlib.sha256(
                f"{participants}{topic}{time.time()}".encode()
            ).hexdigest()[:12]

            thread = ConversationThread(
                thread_id=thread_id,
                participants=participants,
                topic=topic
            )

            self._threads[thread_id] = thread

            for participant in participants:
                self._by_participant[participant].add(thread_id)

            # Enforce capacity
            if len(self._threads) > self.max_threads:
                await self._close_oldest_threads()

            return thread_id

    async def add_message(
        self,
        thread_id: str,
        message: AgentMessage
    ) -> None:
        """Add a message to a thread."""
        async with self._lock:
            if thread_id not in self._threads:
                raise ValueError(f"Thread {thread_id} not found")

            thread = self._threads[thread_id]
            thread.messages.append(message)
            thread.updated_at = datetime.now(timezone.utc)

            # Trim if too many messages
            if len(thread.messages) > self.max_messages_per_thread:
                thread.messages = thread.messages[-self.max_messages_per_thread:]

    async def get_thread(self, thread_id: str) -> Optional[ConversationThread]:
        """Get a conversation thread."""
        return self._threads.get(thread_id)

    async def get_thread_messages(
        self,
        thread_id: str,
        limit: int = 50
    ) -> List[AgentMessage]:
        """Get recent messages from a thread."""
        async with self._lock:
            thread = self._threads.get(thread_id)
            if thread:
                return thread.messages[-limit:]
            return []

    async def get_participant_threads(
        self,
        participant: AgentRole,
        active_only: bool = True
    ) -> List[ConversationThread]:
        """Get threads for a participant."""
        async with self._lock:
            thread_ids = self._by_participant.get(participant, set())
            threads = [
                self._threads[tid] for tid in thread_ids
                if tid in self._threads
            ]
            if active_only:
                threads = [t for t in threads if t.status == "active"]
            return threads

    async def close_thread(self, thread_id: str) -> None:
        """Close a conversation thread."""
        async with self._lock:
            if thread_id in self._threads:
                self._threads[thread_id].status = "closed"

    async def _close_oldest_threads(self) -> None:
        """Close oldest inactive threads."""
        threads = sorted(
            self._threads.values(),
            key=lambda t: t.updated_at
        )

        to_close = len(self._threads) - self.max_threads + 10
        for thread in threads[:to_close]:
            thread.status = "closed"


# =============================================================================
# Episodic Memory
# =============================================================================

class EpisodicMemory:
    """
    Stores episodic memories (tasks, events, experiences).

    Used for:
    - Learning from past tasks
    - Avoiding repeated mistakes
    - Building expertise over time
    """

    def __init__(self, capacity: int = 500):
        self.capacity = capacity
        self._episodes: Dict[str, Episode] = {}
        self._by_agent: Dict[AgentRole, List[str]] = defaultdict(list)
        self._successful: List[str] = []
        self._failed: List[str] = []
        self._lock = asyncio.Lock()

    async def start_episode(
        self,
        task_description: str,
        agents_involved: Set[AgentRole]
    ) -> str:
        """Start a new episode."""
        async with self._lock:
            episode_id = hashlib.sha256(
                f"{task_description}{time.time()}".encode()
            ).hexdigest()[:12]

            episode = Episode(
                episode_id=episode_id,
                task_description=task_description,
                agents_involved=agents_involved,
                start_time=datetime.now(timezone.utc)
            )

            self._episodes[episode_id] = episode

            for agent in agents_involved:
                self._by_agent[agent].append(episode_id)

            return episode_id

    async def add_observation(
        self,
        episode_id: str,
        observation: str
    ) -> None:
        """Add an observation to an episode."""
        async with self._lock:
            if episode_id in self._episodes:
                self._episodes[episode_id].observations.append(observation)

    async def end_episode(
        self,
        episode_id: str,
        outcome: str,
        success: bool,
        lessons_learned: Optional[List[str]] = None
    ) -> None:
        """End an episode and record outcome."""
        async with self._lock:
            if episode_id in self._episodes:
                episode = self._episodes[episode_id]
                episode.end_time = datetime.now(timezone.utc)
                episode.outcome = outcome
                episode.success = success
                episode.lessons_learned = lessons_learned or []

                if success:
                    self._successful.append(episode_id)
                else:
                    self._failed.append(episode_id)

    async def get_similar_episodes(
        self,
        task_description: str,
        limit: int = 5,
        successful_only: bool = False
    ) -> List[Episode]:
        """
        Find similar past episodes.

        Uses keyword matching. For production, use embeddings.
        """
        async with self._lock:
            query_words = set(task_description.lower().split())
            scored = []

            for episode in self._episodes.values():
                if successful_only and not episode.success:
                    continue

                episode_words = set(episode.task_description.lower().split())
                overlap = len(query_words & episode_words)
                if overlap > 0:
                    score = overlap / max(len(query_words), 1)
                    scored.append((score, episode))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [ep for _, ep in scored[:limit]]

    async def get_lessons_learned(
        self,
        agent: Optional[AgentRole] = None,
        successful_only: bool = True
    ) -> List[str]:
        """Get lessons learned from past episodes."""
        async with self._lock:
            lessons = []

            if agent:
                episode_ids = self._by_agent.get(agent, [])
            else:
                episode_ids = list(self._episodes.keys())

            for eid in episode_ids:
                episode = self._episodes.get(eid)
                if episode and (not successful_only or episode.success):
                    lessons.extend(episode.lessons_learned)

            return lessons


# =============================================================================
# Semantic Memory
# =============================================================================

class SemanticMemory:
    """
    Stores semantic knowledge (facts, concepts, relationships).

    Provides:
    - Protocol knowledge
    - Best practices
    - Domain expertise
    """

    def __init__(self):
        self._facts: Dict[str, str] = {}
        self._concepts: Dict[str, Dict[str, Any]] = {}
        self._relationships: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def add_fact(self, key: str, fact: str) -> None:
        """Add a fact to semantic memory."""
        async with self._lock:
            self._facts[key] = fact

    async def get_fact(self, key: str) -> Optional[str]:
        """Get a fact by key."""
        return self._facts.get(key)

    async def add_concept(
        self,
        name: str,
        definition: str,
        properties: Optional[Dict[str, Any]] = None,
        examples: Optional[List[str]] = None
    ) -> None:
        """Add a concept to semantic memory."""
        async with self._lock:
            self._concepts[name] = {
                "definition": definition,
                "properties": properties or {},
                "examples": examples or []
            }

    async def get_concept(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a concept by name."""
        return self._concepts.get(name)

    async def add_relationship(
        self,
        subject: str,
        relation: str,
        obj: str
    ) -> None:
        """Add a relationship between concepts."""
        async with self._lock:
            self._relationships[subject].append({
                "relation": relation,
                "object": obj
            })

    async def get_related(
        self,
        subject: str,
        relation: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Get related concepts."""
        async with self._lock:
            relationships = self._relationships.get(subject, [])
            if relation:
                relationships = [
                    r for r in relationships if r["relation"] == relation
                ]
            return relationships

    async def search_facts(self, query: str, limit: int = 10) -> List[str]:
        """Search facts by keyword."""
        async with self._lock:
            query_lower = query.lower()
            matches = []
            for key, fact in self._facts.items():
                if query_lower in key.lower() or query_lower in fact.lower():
                    matches.append(fact)
            return matches[:limit]

    def preload_protocol_knowledge(self) -> None:
        """Preload common protocol knowledge."""
        # This would be loaded from a knowledge base in production
        protocols = {
            "fix_protocol": (
                "FIX (Financial Information eXchange) is a messaging standard "
                "for trading. Uses tag=value format with SOH (\\x01) delimiter."
            ),
            "swift_mt": (
                "SWIFT MT messages are used for international financial messaging. "
                "Structure includes 5 blocks: Basic Header, Application Header, "
                "User Header, Text Block, and Trailer."
            ),
            "iso8583": (
                "ISO 8583 is a financial transaction message standard. "
                "Uses a bitmap to indicate present data elements."
            ),
            "ebcdic_encoding": (
                "EBCDIC (Extended Binary Coded Decimal Interchange Code) is "
                "a character encoding used on IBM mainframes. Different from ASCII."
            ),
        }

        for key, fact in protocols.items():
            self._facts[key] = fact


__all__ = [
    "MemoryType",
    "SharedMemoryEntry",
    "ConversationThread",
    "Episode",
    "SharedMemory",
    "ConversationMemory",
    "EpisodicMemory",
    "SemanticMemory",
]
