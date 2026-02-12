"""
QBITEL - Persistent Agent Memory with Compression and Relevance Decay

This module provides:
- SQLite-based persistent memory storage
- Redis-based distributed memory (optional)
- Memory compression for storage efficiency
- Relevance decay for memory prioritization
- Memory consolidation and summarization
"""

import asyncio
import hashlib
import json
import logging
import math
import os
import pickle
import sqlite3
import time
import uuid
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict
from contextlib import asynccontextmanager

from prometheus_client import Counter, Gauge, Histogram

# Import base memory types
from .agent_memory import (
    MemoryType,
    MemoryPriority,
    MemoryEntry,
    EpisodicMemory,
    SemanticMemory,
    MemoryBackend,
)

# Redis async support (aioredis is deprecated; use redis.asyncio)
try:
    import redis.asyncio as aioredis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    aioredis = None

logger = logging.getLogger(__name__)

# =============================================================================
# Metrics
# =============================================================================

MEMORY_PERSIST_COUNTER = Counter(
    "qbitel_memory_persist_total", "Total memory persistence operations", ["operation", "backend"]
)
MEMORY_COMPRESSION_RATIO = Histogram(
    "qbitel_memory_compression_ratio", "Memory compression ratio", buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)
MEMORY_DECAY_APPLIED = Counter("qbitel_memory_decay_applied_total", "Total relevance decay applications")
MEMORY_CONSOLIDATED = Counter("qbitel_memory_consolidated_total", "Total memories consolidated")


# =============================================================================
# Relevance Decay Configuration
# =============================================================================


@dataclass
class RelevanceDecayConfig:
    """Configuration for memory relevance decay."""

    # Base decay rate per day (0.95 means 5% decay per day)
    base_decay_rate: float = 0.95

    # Priority-based decay multipliers
    priority_multipliers: Dict[MemoryPriority, float] = field(
        default_factory=lambda: {
            MemoryPriority.CRITICAL: 1.0,  # No decay
            MemoryPriority.HIGH: 0.99,  # Very slow decay
            MemoryPriority.NORMAL: 0.95,  # Normal decay
            MemoryPriority.LOW: 0.85,  # Fast decay
        }
    )

    # Access boost (how much accessing a memory boosts relevance)
    access_boost: float = 0.1

    # Minimum relevance before memory is eligible for pruning
    min_relevance: float = 0.1

    # Days of inactivity before pruning
    prune_after_days: int = 90


@dataclass
class CompressionConfig:
    """Configuration for memory compression."""

    # Enable compression
    enabled: bool = True

    # Compression level (1-9, higher = more compression)
    level: int = 6

    # Minimum size to compress (bytes)
    min_size: int = 256

    # Compression algorithm
    algorithm: str = "zlib"


# =============================================================================
# Memory Entry with Relevance
# =============================================================================


@dataclass
class PersistentMemoryEntry:
    """Memory entry with persistence and relevance tracking."""

    entry_id: str
    agent_id: str
    memory_type: MemoryType
    content: Dict[str, Any]

    # Relevance tracking
    relevance_score: float = 1.0
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Priority and expiration
    priority: MemoryPriority = MemoryPriority.NORMAL
    expires_at: Optional[datetime] = None

    # Embedding for semantic search
    embedding: Optional[List[float]] = None

    # Metadata
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Compression info
    is_compressed: bool = False
    original_size: int = 0
    compressed_size: int = 0

    # Consolidation tracking
    is_consolidated: bool = False
    consolidated_from: List[str] = field(default_factory=list)
    consolidation_summary: Optional[str] = None

    def calculate_relevance(self, config: RelevanceDecayConfig) -> float:
        """Calculate current relevance score with decay."""
        days_since_access = (datetime.utcnow() - self.last_accessed).days
        days_since_creation = (datetime.utcnow() - self.created_at).days

        # Get priority multiplier
        priority_mult = config.priority_multipliers.get(self.priority, config.priority_multipliers[MemoryPriority.NORMAL])

        # Apply time decay
        decay_factor = math.pow(priority_mult, days_since_access)

        # Apply access frequency boost
        access_boost = min(config.access_boost * math.log1p(self.access_count), 0.5)

        # Calculate final relevance
        relevance = self.relevance_score * decay_factor + access_boost

        # Clamp to [0, 1]
        return max(0.0, min(1.0, relevance))

    def touch(self, config: RelevanceDecayConfig) -> None:
        """Update access timestamp and boost relevance."""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()
        self.relevance_score = min(1.0, self.relevance_score + config.access_boost)

    def is_expired(self) -> bool:
        """Check if memory has expired."""
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        return False

    def should_prune(self, config: RelevanceDecayConfig) -> bool:
        """Check if memory should be pruned."""
        if self.priority == MemoryPriority.CRITICAL:
            return False

        relevance = self.calculate_relevance(config)
        if relevance < config.min_relevance:
            return True

        days_inactive = (datetime.utcnow() - self.last_accessed).days
        if days_inactive > config.prune_after_days and self.priority != MemoryPriority.HIGH:
            return True

        return False


# =============================================================================
# Compression Utilities
# =============================================================================


class MemoryCompressor:
    """Handles memory compression and decompression."""

    def __init__(self, config: CompressionConfig = None):
        self.config = config or CompressionConfig()

    def compress(self, data: Dict[str, Any]) -> Tuple[bytes, bool, int, int]:
        """
        Compress memory content.

        Returns:
            (compressed_data, was_compressed, original_size, compressed_size)
        """
        # Serialize to JSON
        json_str = json.dumps(data, default=str)
        original_bytes = json_str.encode("utf-8")
        original_size = len(original_bytes)

        # Check if compression is enabled and worthwhile
        if not self.config.enabled or original_size < self.config.min_size:
            return original_bytes, False, original_size, original_size

        # Compress
        compressed = zlib.compress(original_bytes, self.config.level)
        compressed_size = len(compressed)

        # Only use compression if it actually reduces size
        if compressed_size < original_size:
            MEMORY_COMPRESSION_RATIO.observe(compressed_size / original_size)
            return compressed, True, original_size, compressed_size
        else:
            return original_bytes, False, original_size, original_size

    def decompress(self, data: bytes, is_compressed: bool) -> Dict[str, Any]:
        """Decompress memory content."""
        if is_compressed:
            decompressed = zlib.decompress(data)
            return json.loads(decompressed.decode("utf-8"))
        else:
            return json.loads(data.decode("utf-8"))


# =============================================================================
# SQLite Persistent Backend
# =============================================================================


class SQLitePersistentBackend(MemoryBackend):
    """SQLite-based persistent memory storage."""

    def __init__(
        self,
        db_path: str = "./data/agent_memory.db",
        compression_config: CompressionConfig = None,
        decay_config: RelevanceDecayConfig = None,
    ):
        self.db_path = db_path
        self.compression_config = compression_config or CompressionConfig()
        self.decay_config = decay_config or RelevanceDecayConfig()
        self.compressor = MemoryCompressor(self.compression_config)
        self._connection: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()

        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
            self._init_schema()
        return self._connection

    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = self._connection
        cursor = conn.cursor()

        # Main memories table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                entry_id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                content BLOB NOT NULL,
                is_compressed INTEGER DEFAULT 0,
                original_size INTEGER DEFAULT 0,
                compressed_size INTEGER DEFAULT 0,
                relevance_score REAL DEFAULT 1.0,
                access_count INTEGER DEFAULT 0,
                priority TEXT DEFAULT 'normal',
                last_accessed TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT,
                tags TEXT,
                metadata TEXT,
                embedding BLOB,
                is_consolidated INTEGER DEFAULT 0,
                consolidated_from TEXT,
                consolidation_summary TEXT
            )
        """)

        # Indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_type
            ON memories(agent_id, memory_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_relevance
            ON memories(relevance_score DESC)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_last_accessed
            ON memories(last_accessed DESC)
        """)

        # Episodic memories table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS episodic_memories (
                episode_id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                context TEXT NOT NULL,
                action TEXT NOT NULL,
                outcome TEXT NOT NULL,
                emotion_valence REAL DEFAULT 0.0,
                importance REAL DEFAULT 0.5,
                timestamp TEXT NOT NULL,
                related_episodes TEXT,
                metadata TEXT
            )
        """)

        # Semantic memories (knowledge graph) table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS semantic_memories (
                fact_id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                source TEXT DEFAULT 'learned',
                evidence TEXT,
                last_verified TEXT,
                created_at TEXT NOT NULL,
                metadata TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_semantic_subject
            ON semantic_memories(subject)
        """)

        conn.commit()

    async def store(self, entry: MemoryEntry) -> bool:
        """Store a memory entry."""
        async with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                # Compress content
                compressed_data, is_compressed, orig_size, comp_size = self.compressor.compress(entry.content)

                # Serialize tags and metadata
                tags_json = json.dumps(list(entry.tags)) if entry.tags else "[]"
                metadata_json = json.dumps(entry.metadata) if entry.metadata else "{}"

                # Serialize embedding
                embedding_blob = pickle.dumps(entry.embedding) if entry.embedding else None

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO memories (
                        entry_id, agent_id, memory_type, content, is_compressed,
                        original_size, compressed_size, relevance_score, access_count,
                        priority, last_accessed, created_at, expires_at, tags,
                        metadata, embedding
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        entry.entry_id,
                        entry.agent_id,
                        entry.memory_type.value,
                        compressed_data,
                        1 if is_compressed else 0,
                        orig_size,
                        comp_size,
                        1.0,  # Initial relevance
                        entry.access_count,
                        entry.priority.value,
                        entry.last_accessed.isoformat(),
                        entry.created_at.isoformat(),
                        entry.expires_at.isoformat() if entry.expires_at else None,
                        tags_json,
                        metadata_json,
                        embedding_blob,
                    ),
                )

                conn.commit()
                MEMORY_PERSIST_COUNTER.labels(operation="store", backend="sqlite").inc()
                return True

            except Exception as e:
                logger.error(f"Failed to store memory: {e}")
                return False

    async def retrieve(
        self,
        agent_id: str,
        memory_type: MemoryType,
        query: Optional[Dict[str, Any]] = None,
        limit: int = 10,
    ) -> List[MemoryEntry]:
        """Retrieve memory entries."""
        async with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                # Build query
                sql = """
                    SELECT * FROM memories
                    WHERE agent_id = ? AND memory_type = ?
                    AND (expires_at IS NULL OR expires_at > ?)
                    ORDER BY relevance_score DESC, last_accessed DESC
                    LIMIT ?
                """
                params = [agent_id, memory_type.value, datetime.utcnow().isoformat(), limit]

                cursor.execute(sql, params)
                rows = cursor.fetchall()

                entries = []
                for row in rows:
                    entry = self._row_to_entry(row)
                    if entry:
                        # Apply query filter if provided
                        if query and not self._matches_query(entry, query):
                            continue
                        entries.append(entry)

                MEMORY_PERSIST_COUNTER.labels(operation="retrieve", backend="sqlite").inc()
                return entries

            except Exception as e:
                logger.error(f"Failed to retrieve memories: {e}")
                return []

    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        async with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("DELETE FROM memories WHERE entry_id = ?", (entry_id,))
                conn.commit()
                MEMORY_PERSIST_COUNTER.labels(operation="delete", backend="sqlite").inc()
                return cursor.rowcount > 0
            except Exception as e:
                logger.error(f"Failed to delete memory: {e}")
                return False

    async def search(
        self,
        query: str,
        agent_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
    ) -> List[MemoryEntry]:
        """Search memories by content."""
        async with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                # Build query (simple text search)
                sql = "SELECT * FROM memories WHERE 1=1"
                params = []

                if agent_id:
                    sql += " AND agent_id = ?"
                    params.append(agent_id)
                if memory_type:
                    sql += " AND memory_type = ?"
                    params.append(memory_type.value)

                sql += " ORDER BY relevance_score DESC, last_accessed DESC LIMIT ?"
                params.append(limit * 5)  # Get more to filter

                cursor.execute(sql, params)
                rows = cursor.fetchall()

                # Filter by query content
                query_lower = query.lower()
                entries = []
                for row in rows:
                    entry = self._row_to_entry(row)
                    if entry:
                        content_str = json.dumps(entry.content).lower()
                        if query_lower in content_str:
                            entries.append(entry)
                            if len(entries) >= limit:
                                break

                return entries

            except Exception as e:
                logger.error(f"Failed to search memories: {e}")
                return []

    async def apply_relevance_decay(self) -> int:
        """Apply relevance decay to all memories."""
        async with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                # Get all non-critical memories
                cursor.execute("""
                    SELECT entry_id, relevance_score, access_count,
                           priority, last_accessed, created_at
                    FROM memories
                    WHERE priority != 'critical'
                """)
                rows = cursor.fetchall()

                updated = 0
                for row in rows:
                    entry_id = row["entry_id"]
                    current_relevance = row["relevance_score"]
                    access_count = row["access_count"]
                    priority = MemoryPriority(row["priority"])
                    last_accessed = datetime.fromisoformat(row["last_accessed"])

                    # Calculate decay
                    days_since_access = (datetime.utcnow() - last_accessed).days
                    priority_mult = self.decay_config.priority_multipliers.get(
                        priority, self.decay_config.priority_multipliers[MemoryPriority.NORMAL]
                    )
                    decay_factor = math.pow(priority_mult, days_since_access)

                    # Apply access frequency boost
                    access_boost = min(self.decay_config.access_boost * math.log1p(access_count), 0.5)

                    # Calculate new relevance
                    new_relevance = max(0.0, min(1.0, current_relevance * decay_factor + access_boost))

                    # Update if changed significantly
                    if abs(new_relevance - current_relevance) > 0.01:
                        cursor.execute("UPDATE memories SET relevance_score = ? WHERE entry_id = ?", (new_relevance, entry_id))
                        updated += 1

                conn.commit()
                MEMORY_DECAY_APPLIED.inc(updated)
                return updated

            except Exception as e:
                logger.error(f"Failed to apply relevance decay: {e}")
                return 0

    async def prune_low_relevance(self) -> int:
        """Remove memories below minimum relevance threshold."""
        async with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                # Delete low relevance memories (except critical)
                cursor.execute(
                    """
                    DELETE FROM memories
                    WHERE relevance_score < ?
                    AND priority != 'critical'
                    AND priority != 'high'
                """,
                    (self.decay_config.min_relevance,),
                )

                pruned = cursor.rowcount
                conn.commit()

                if pruned > 0:
                    logger.info(f"Pruned {pruned} low-relevance memories")

                return pruned

            except Exception as e:
                logger.error(f"Failed to prune memories: {e}")
                return 0

    async def update_access(self, entry_id: str) -> None:
        """Update access timestamp and boost relevance."""
        async with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                # Get current values
                cursor.execute("SELECT relevance_score, access_count FROM memories WHERE entry_id = ?", (entry_id,))
                row = cursor.fetchone()
                if row:
                    new_relevance = min(1.0, row["relevance_score"] + self.decay_config.access_boost)
                    new_count = row["access_count"] + 1

                    cursor.execute(
                        """
                        UPDATE memories
                        SET relevance_score = ?, access_count = ?, last_accessed = ?
                        WHERE entry_id = ?
                    """,
                        (new_relevance, new_count, datetime.utcnow().isoformat(), entry_id),
                    )

                    conn.commit()

            except Exception as e:
                logger.error(f"Failed to update access: {e}")

    def _row_to_entry(self, row: sqlite3.Row) -> Optional[MemoryEntry]:
        """Convert database row to MemoryEntry."""
        try:
            # Decompress content
            content = self.compressor.decompress(row["content"], bool(row["is_compressed"]))

            # Parse tags and metadata
            tags = set(json.loads(row["tags"])) if row["tags"] else set()
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}

            # Parse embedding
            embedding = pickle.loads(row["embedding"]) if row["embedding"] else None

            return MemoryEntry(
                entry_id=row["entry_id"],
                agent_id=row["agent_id"],
                memory_type=MemoryType(row["memory_type"]),
                content=content,
                embedding=embedding,
                priority=MemoryPriority(row["priority"]),
                access_count=row["access_count"],
                last_accessed=datetime.fromisoformat(row["last_accessed"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
                tags=tags,
                metadata=metadata,
            )
        except Exception as e:
            logger.error(f"Failed to parse memory row: {e}")
            return None

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

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        async with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                cursor.execute("SELECT COUNT(*) as count FROM memories")
                total_count = cursor.fetchone()["count"]

                cursor.execute("""
                    SELECT memory_type, COUNT(*) as count
                    FROM memories GROUP BY memory_type
                """)
                by_type = {row["memory_type"]: row["count"] for row in cursor.fetchall()}

                cursor.execute("""
                    SELECT SUM(original_size) as orig, SUM(compressed_size) as comp
                    FROM memories WHERE is_compressed = 1
                """)
                compression = cursor.fetchone()

                cursor.execute("""
                    SELECT AVG(relevance_score) as avg_relevance FROM memories
                """)
                avg_relevance = cursor.fetchone()["avg_relevance"] or 0

                return {
                    "total_memories": total_count,
                    "by_type": by_type,
                    "compression": {
                        "original_bytes": compression["orig"] or 0,
                        "compressed_bytes": compression["comp"] or 0,
                        "ratio": (compression["comp"] / compression["orig"]) if compression["orig"] else 1.0,
                    },
                    "average_relevance": avg_relevance,
                }

            except Exception as e:
                logger.error(f"Failed to get stats: {e}")
                return {}

    def close(self) -> None:
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None


# =============================================================================
# Redis Distributed Backend (Optional)
# =============================================================================


class RedisPersistentBackend(MemoryBackend):
    """Redis-based distributed memory storage."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        prefix: str = "qbitel:memory:",
        compression_config: CompressionConfig = None,
        decay_config: RelevanceDecayConfig = None,
        ttl_seconds: int = 86400 * 90,  # 90 days default
    ):
        if not REDIS_AVAILABLE:
            raise RuntimeError("Redis not available. Install redis package.")

        self.redis_url = redis_url
        self.prefix = prefix
        self.compression_config = compression_config or CompressionConfig()
        self.decay_config = decay_config or RelevanceDecayConfig()
        self.compressor = MemoryCompressor(self.compression_config)
        self.ttl_seconds = ttl_seconds
        self._redis: Optional[Any] = None
        self._lock = asyncio.Lock()

    async def _get_redis(self):
        """Get or create Redis connection."""
        if self._redis is None:
            self._redis = await aioredis.from_url(self.redis_url)
        return self._redis

    def _key(self, entry_id: str) -> str:
        """Generate Redis key for entry."""
        return f"{self.prefix}entry:{entry_id}"

    def _index_key(self, agent_id: str, memory_type: str) -> str:
        """Generate Redis key for agent/type index."""
        return f"{self.prefix}index:{agent_id}:{memory_type}"

    async def store(self, entry: MemoryEntry) -> bool:
        """Store a memory entry."""
        try:
            redis = await self._get_redis()

            # Serialize entry
            entry_dict = {
                "entry_id": entry.entry_id,
                "agent_id": entry.agent_id,
                "memory_type": entry.memory_type.value,
                "content": entry.content,
                "embedding": entry.embedding,
                "priority": entry.priority.value,
                "access_count": entry.access_count,
                "last_accessed": entry.last_accessed.isoformat(),
                "created_at": entry.created_at.isoformat(),
                "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
                "tags": list(entry.tags),
                "metadata": entry.metadata,
                "relevance_score": 1.0,
            }

            # Compress
            compressed_data, is_compressed, _, _ = self.compressor.compress(entry_dict)
            entry_dict["_compressed"] = is_compressed

            # Store entry
            key = self._key(entry.entry_id)
            await redis.set(key, compressed_data, ex=self.ttl_seconds)

            # Add to index
            index_key = self._index_key(entry.agent_id, entry.memory_type.value)
            await redis.zadd(index_key, {entry.entry_id: time.time()})

            MEMORY_PERSIST_COUNTER.labels(operation="store", backend="redis").inc()
            return True

        except Exception as e:
            logger.error(f"Failed to store memory in Redis: {e}")
            return False

    async def retrieve(
        self,
        agent_id: str,
        memory_type: MemoryType,
        query: Optional[Dict[str, Any]] = None,
        limit: int = 10,
    ) -> List[MemoryEntry]:
        """Retrieve memory entries."""
        try:
            redis = await self._get_redis()

            # Get entry IDs from index
            index_key = self._index_key(agent_id, memory_type.value)
            entry_ids = await redis.zrevrange(index_key, 0, limit * 2)

            entries = []
            for entry_id in entry_ids:
                key = self._key(entry_id.decode() if isinstance(entry_id, bytes) else entry_id)
                data = await redis.get(key)
                if data:
                    entry = self._deserialize_entry(data)
                    if entry and not entry.is_expired():
                        if query and not self._matches_query(entry, query):
                            continue
                        entries.append(entry)
                        if len(entries) >= limit:
                            break

            MEMORY_PERSIST_COUNTER.labels(operation="retrieve", backend="redis").inc()
            return entries

        except Exception as e:
            logger.error(f"Failed to retrieve memories from Redis: {e}")
            return []

    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        try:
            redis = await self._get_redis()
            key = self._key(entry_id)

            # Get entry to find agent_id and type for index cleanup
            data = await redis.get(key)
            if data:
                entry = self._deserialize_entry(data)
                if entry:
                    index_key = self._index_key(entry.agent_id, entry.memory_type.value)
                    await redis.zrem(index_key, entry_id)

            await redis.delete(key)
            MEMORY_PERSIST_COUNTER.labels(operation="delete", backend="redis").inc()
            return True

        except Exception as e:
            logger.error(f"Failed to delete memory from Redis: {e}")
            return False

    async def search(
        self,
        query: str,
        agent_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
    ) -> List[MemoryEntry]:
        """Search memories by content."""
        try:
            redis = await self._get_redis()

            # Get all keys matching pattern
            pattern = f"{self.prefix}entry:*"
            keys = []
            async for key in redis.scan_iter(pattern):
                keys.append(key)

            query_lower = query.lower()
            entries = []

            for key in keys:
                data = await redis.get(key)
                if data:
                    entry = self._deserialize_entry(data)
                    if entry:
                        if agent_id and entry.agent_id != agent_id:
                            continue
                        if memory_type and entry.memory_type != memory_type:
                            continue

                        content_str = json.dumps(entry.content).lower()
                        if query_lower in content_str:
                            entries.append(entry)
                            if len(entries) >= limit:
                                break

            return entries

        except Exception as e:
            logger.error(f"Failed to search memories in Redis: {e}")
            return []

    def _deserialize_entry(self, data: bytes) -> Optional[MemoryEntry]:
        """Deserialize entry from Redis."""
        try:
            # Try to decompress
            try:
                entry_dict = json.loads(zlib.decompress(data).decode())
            except zlib.error:
                entry_dict = json.loads(data.decode())

            return MemoryEntry(
                entry_id=entry_dict["entry_id"],
                agent_id=entry_dict["agent_id"],
                memory_type=MemoryType(entry_dict["memory_type"]),
                content=entry_dict["content"],
                embedding=entry_dict.get("embedding"),
                priority=MemoryPriority(entry_dict.get("priority", "normal")),
                access_count=entry_dict.get("access_count", 0),
                last_accessed=datetime.fromisoformat(entry_dict["last_accessed"]),
                created_at=datetime.fromisoformat(entry_dict["created_at"]),
                expires_at=datetime.fromisoformat(entry_dict["expires_at"]) if entry_dict.get("expires_at") else None,
                tags=set(entry_dict.get("tags", [])),
                metadata=entry_dict.get("metadata", {}),
            )
        except Exception as e:
            logger.error(f"Failed to deserialize entry: {e}")
            return None

    def _matches_query(self, entry: MemoryEntry, query: Dict[str, Any]) -> bool:
        """Check if entry matches query criteria."""
        for key, value in query.items():
            if key == "tags":
                if not set(value).issubset(entry.tags):
                    return False
            elif key in entry.content:
                if entry.content[key] != value:
                    return False
        return True

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None


# =============================================================================
# Memory Consolidation Service
# =============================================================================


class MemoryConsolidationService:
    """
    Service for consolidating and summarizing memories.

    Consolidates similar/related memories into summaries to reduce storage
    while preserving important information.
    """

    def __init__(self, backend: MemoryBackend, llm_service: Optional[Any] = None):
        self.backend = backend
        self.llm_service = llm_service
        self.logger = logging.getLogger(__name__)

    async def consolidate_memories(
        self,
        agent_id: str,
        memory_type: MemoryType,
        max_age_days: int = 7,
        min_count: int = 5,
        similarity_threshold: float = 0.7,
    ) -> Optional[MemoryEntry]:
        """
        Consolidate old memories into a summary.

        Args:
            agent_id: Agent to consolidate for
            memory_type: Type of memories to consolidate
            max_age_days: Memories older than this are candidates
            min_count: Minimum memories needed for consolidation
            similarity_threshold: Minimum similarity for grouping

        Returns:
            Consolidated memory entry if created
        """
        try:
            # Get old memories
            cutoff = datetime.utcnow() - timedelta(days=max_age_days)
            memories = await self.backend.retrieve(agent_id=agent_id, memory_type=memory_type, limit=100)

            # Filter by age
            old_memories = [m for m in memories if m.created_at < cutoff]

            if len(old_memories) < min_count:
                return None

            # Group similar memories (simple content-based)
            groups = self._group_similar_memories(old_memories, similarity_threshold)

            # Consolidate each group
            consolidated_entries = []
            for group in groups:
                if len(group) >= min_count:
                    summary = await self._create_summary(group)
                    if summary:
                        consolidated_entries.append(summary)

                        # Mark originals as consolidated
                        for mem in group:
                            await self.backend.delete(mem.entry_id)

            MEMORY_CONSOLIDATED.inc(len(consolidated_entries))
            return consolidated_entries[0] if consolidated_entries else None

        except Exception as e:
            self.logger.error(f"Failed to consolidate memories: {e}")
            return None

    def _group_similar_memories(self, memories: List[MemoryEntry], threshold: float) -> List[List[MemoryEntry]]:
        """Group similar memories together."""
        groups = []
        used = set()

        for i, mem1 in enumerate(memories):
            if mem1.entry_id in used:
                continue

            group = [mem1]
            used.add(mem1.entry_id)

            for j, mem2 in enumerate(memories):
                if i >= j or mem2.entry_id in used:
                    continue

                similarity = self._calculate_similarity(mem1, mem2)
                if similarity >= threshold:
                    group.append(mem2)
                    used.add(mem2.entry_id)

            if len(group) > 1:
                groups.append(group)

        return groups

    def _calculate_similarity(self, mem1: MemoryEntry, mem2: MemoryEntry) -> float:
        """Calculate similarity between two memories."""
        # Simple Jaccard similarity on content keys
        keys1 = set(str(k) for k in mem1.content.keys())
        keys2 = set(str(k) for k in mem2.content.keys())

        intersection = len(keys1 & keys2)
        union = len(keys1 | keys2)

        return intersection / union if union > 0 else 0.0

    async def _create_summary(self, memories: List[MemoryEntry]) -> Optional[MemoryEntry]:
        """Create a summary memory from a group."""
        if not memories:
            return None

        # Use LLM if available, otherwise create structured summary
        if self.llm_service:
            # TODO: Use LLM to generate natural language summary
            pass

        # Create structured summary
        summary_content = {
            "type": "consolidated_summary",
            "count": len(memories),
            "date_range": {
                "start": min(m.created_at for m in memories).isoformat(),
                "end": max(m.created_at for m in memories).isoformat(),
            },
            "common_tags": list(set.intersection(*[m.tags for m in memories]) if memories else set()),
            "sample_contents": [m.content for m in memories[:3]],
        }

        summary_entry = MemoryEntry(
            entry_id=str(uuid.uuid4()),
            agent_id=memories[0].agent_id,
            memory_type=memories[0].memory_type,
            content=summary_content,
            priority=MemoryPriority.HIGH,  # Summaries are important
            tags=set.union(*[m.tags for m in memories]),
            metadata={"consolidated": True, "source_count": len(memories), "source_ids": [m.entry_id for m in memories]},
        )

        # Store the summary
        await self.backend.store(summary_entry)

        return summary_entry
