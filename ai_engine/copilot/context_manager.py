"""
CRONOS AI - Conversation Context Manager
Manages conversation history and context for the Protocol Intelligence Copilot.
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import redis.asyncio as redis
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Single conversation turn."""

    timestamp: datetime
    user_query: str
    assistant_response: str
    query_type: str
    confidence: float
    context_used: Dict[str, Any] = None


@dataclass
class ConversationSession:
    """Complete conversation session."""

    session_id: str
    user_id: str
    started_at: datetime
    last_activity: datetime
    turns: List[ConversationTurn]
    persistent_context: Dict[str, Any]
    metadata: Dict[str, Any] = None


class ConversationContextManager:
    """
    Manages conversation context and memory for Protocol Intelligence Copilot.
    Provides short-term and long-term memory capabilities with Redis backend.
    """

    def __init__(self, redis_config: Dict[str, Any] = None):
        self.redis_config = redis_config or {"host": "localhost", "port": 6379, "db": 0}
        self.logger = logging.getLogger(__name__)

        # Redis client for persistent storage
        self.redis_client: Optional[redis.Redis] = None

        # In-memory cache for active sessions
        self.active_sessions: Dict[str, ConversationSession] = {}

        # Configuration
        self.max_turns_per_session = 50
        self.session_timeout = timedelta(hours=2)
        self.context_cleanup_interval = 300  # 5 minutes

        # Background cleanup task
        self.cleanup_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Initialize context manager and Redis connection."""
        try:
            self.logger.info("Initializing Conversation Context Manager...")

            # Initialize Redis client
            self.redis_client = redis.Redis(
                host=self.redis_config["host"],
                port=self.redis_config["port"],
                db=self.redis_config["db"],
                decode_responses=True,
            )

            # Test Redis connection
            await self.redis_client.ping()

            # Start background cleanup task
            self.cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())

            self.logger.info("Conversation Context Manager initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize context manager: {e}")
            raise

    async def get_context(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """
        Get conversation context for a user session.

        Args:
            user_id: User identifier
            session_id: Session identifier

        Returns:
            Conversation context including history and persistent data
        """
        try:
            # Check in-memory cache first
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                return self._build_context_from_session(session)

            # Load from Redis
            session = await self._load_session_from_redis(user_id, session_id)
            if session:
                # Cache in memory for quick access
                self.active_sessions[session_id] = session
                return self._build_context_from_session(session)

            # Create new session
            session = ConversationSession(
                session_id=session_id,
                user_id=user_id,
                started_at=datetime.now(),
                last_activity=datetime.now(),
                turns=[],
                persistent_context={},
                metadata={"created_by": "context_manager"},
            )

            self.active_sessions[session_id] = session
            await self._save_session_to_redis(session)

            return self._build_context_from_session(session)

        except Exception as e:
            self.logger.error(f"Error getting context for {user_id}/{session_id}: {e}")
            return {}

    async def update_context(
        self,
        user_id: str,
        session_id: str,
        user_query: str,
        assistant_response: str,
        query_type: str = "general",
        confidence: float = 0.0,
        additional_context: Dict[str, Any] = None,
    ) -> None:
        """
        Update conversation context with new turn.

        Args:
            user_id: User identifier
            session_id: Session identifier
            user_query: User's query
            assistant_response: Assistant's response
            query_type: Type of query processed
            confidence: Confidence score of the response
            additional_context: Additional context to persist
        """
        try:
            # Get or create session
            session = self.active_sessions.get(session_id)
            if not session:
                session = await self._load_session_from_redis(user_id, session_id)
                if not session:
                    # Create new session
                    session = ConversationSession(
                        session_id=session_id,
                        user_id=user_id,
                        started_at=datetime.now(),
                        last_activity=datetime.now(),
                        turns=[],
                        persistent_context={},
                        metadata={"created_by": "context_manager"},
                    )

            # Create new turn
            turn = ConversationTurn(
                timestamp=datetime.now(),
                user_query=user_query,
                assistant_response=assistant_response,
                query_type=query_type,
                confidence=confidence,
                context_used=additional_context or {},
            )

            # Add to session
            session.turns.append(turn)
            session.last_activity = datetime.now()

            # Maintain maximum turns limit
            if len(session.turns) > self.max_turns_per_session:
                session.turns = session.turns[-self.max_turns_per_session :]

            # Update persistent context with learned information
            await self._update_persistent_context(session, turn, additional_context)

            # Update cache and save to Redis
            self.active_sessions[session_id] = session
            await self._save_session_to_redis(session)

        except Exception as e:
            self.logger.error(f"Error updating context for {user_id}/{session_id}: {e}")

    async def get_user_sessions(
        self, user_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent sessions for a user."""
        try:
            pattern = f"context:session:{user_id}:*"
            session_keys = await self.redis_client.keys(pattern)

            sessions = []
            for key in session_keys[-limit:]:  # Get most recent
                session_data = await self.redis_client.hgetall(key)
                if session_data:
                    sessions.append(
                        {
                            "session_id": session_data.get("session_id"),
                            "started_at": session_data.get("started_at"),
                            "last_activity": session_data.get("last_activity"),
                            "turn_count": int(session_data.get("turn_count", 0)),
                            "metadata": json.loads(session_data.get("metadata", "{}")),
                        }
                    )

            return sorted(sessions, key=lambda x: x["last_activity"], reverse=True)

        except Exception as e:
            self.logger.error(f"Error getting user sessions for {user_id}: {e}")
            return []

    async def clear_session(self, user_id: str, session_id: str) -> bool:
        """Clear a specific session."""
        try:
            # Remove from memory cache
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]

            # Remove from Redis
            redis_key = f"context:session:{user_id}:{session_id}"
            await self.redis_client.delete(redis_key)

            return True

        except Exception as e:
            self.logger.error(f"Error clearing session {user_id}/{session_id}: {e}")
            return False

    async def get_session_summary(
        self, user_id: str, session_id: str
    ) -> Dict[str, Any]:
        """Get summary of a conversation session."""
        try:
            session = await self._load_session_from_redis(user_id, session_id)
            if not session:
                return {}

            # Calculate statistics
            query_types = defaultdict(int)
            avg_confidence = 0.0
            recent_topics = []

            for turn in session.turns:
                query_types[turn.query_type] += 1
                avg_confidence += turn.confidence

                # Extract topics from queries (simple keyword extraction)
                keywords = self._extract_keywords(turn.user_query)
                recent_topics.extend(keywords)

            avg_confidence = (
                avg_confidence / len(session.turns) if session.turns else 0.0
            )

            # Get most common topics
            from collections import Counter

            top_topics = [
                topic for topic, count in Counter(recent_topics).most_common(5)
            ]

            return {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "duration": str(session.last_activity - session.started_at),
                "total_turns": len(session.turns),
                "query_types": dict(query_types),
                "average_confidence": round(avg_confidence, 2),
                "top_topics": top_topics,
                "last_activity": session.last_activity.isoformat(),
                "persistent_context_keys": list(session.persistent_context.keys()),
            }

        except Exception as e:
            self.logger.error(f"Error getting session summary: {e}")
            return {}

    def _build_context_from_session(
        self, session: ConversationSession
    ) -> Dict[str, Any]:
        """Build context dictionary from session data."""
        context = {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "session_started": session.started_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "turn_count": len(session.turns),
            "persistent_context": session.persistent_context.copy(),
        }

        # Add recent conversation history (last 5 turns)
        recent_turns = session.turns[-5:] if session.turns else []
        context["recent_conversation"] = [
            {
                "timestamp": turn.timestamp.isoformat(),
                "user_query": turn.user_query,
                "assistant_response": turn.assistant_response,
                "query_type": turn.query_type,
                "confidence": turn.confidence,
            }
            for turn in recent_turns
        ]

        # Add conversation patterns
        if session.turns:
            query_types = [turn.query_type for turn in session.turns]
            context["conversation_patterns"] = {
                "dominant_query_type": max(set(query_types), key=query_types.count),
                "query_type_diversity": len(set(query_types)),
                "average_confidence": sum(turn.confidence for turn in session.turns)
                / len(session.turns),
            }

        return context

    async def _update_persistent_context(
        self,
        session: ConversationSession,
        turn: ConversationTurn,
        additional_context: Dict[str, Any] = None,
    ) -> None:
        """Update persistent context with learned information."""
        try:
            # Extract and store user preferences
            if "protocol" in turn.user_query.lower():
                protocols_mentioned = self._extract_protocols_from_query(
                    turn.user_query
                )
                if protocols_mentioned:
                    if "preferred_protocols" not in session.persistent_context:
                        session.persistent_context["preferred_protocols"] = []

                    for protocol in protocols_mentioned:
                        if (
                            protocol
                            not in session.persistent_context["preferred_protocols"]
                        ):
                            session.persistent_context["preferred_protocols"].append(
                                protocol
                            )

            # Store user expertise level indicators
            complexity_indicators = [
                "advanced",
                "basic",
                "beginner",
                "expert",
                "detailed",
                "simple",
            ]

            for indicator in complexity_indicators:
                if indicator in turn.user_query.lower():
                    session.persistent_context["expertise_level"] = indicator
                    break

            # Add additional context if provided
            if additional_context:
                session.persistent_context.update(additional_context)

            # Limit size of persistent context
            if len(session.persistent_context) > 20:
                # Keep most recent items
                keys_to_remove = list(session.persistent_context.keys())[:-15]
                for key in keys_to_remove:
                    del session.persistent_context[key]

        except Exception as e:
            self.logger.error(f"Error updating persistent context: {e}")

    def _extract_protocols_from_query(self, query: str) -> List[str]:
        """Extract protocol names from user query."""
        protocols = [
            "http",
            "https",
            "tcp",
            "udp",
            "dns",
            "ftp",
            "ssh",
            "tls",
            "ssl",
            "smtp",
            "pop3",
            "imap",
        ]
        found_protocols = []

        query_lower = query.lower()
        for protocol in protocols:
            if protocol in query_lower:
                found_protocols.append(protocol.upper())

        return found_protocols

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text (simple implementation)."""
        # Remove common words and extract meaningful terms
        common_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
        }

        words = text.lower().split()
        keywords = [
            word.strip(".,!?;:")
            for word in words
            if len(word) > 3 and word not in common_words
        ]

        return keywords[:10]  # Return top 10 keywords

    async def _load_session_from_redis(
        self, user_id: str, session_id: str
    ) -> Optional[ConversationSession]:
        """Load session from Redis."""
        try:
            redis_key = f"context:session:{user_id}:{session_id}"
            session_data = await self.redis_client.hgetall(redis_key)

            if not session_data:
                return None

            # Reconstruct session object
            turns_data = json.loads(session_data.get("turns", "[]"))
            turns = [
                ConversationTurn(
                    timestamp=datetime.fromisoformat(turn["timestamp"]),
                    user_query=turn["user_query"],
                    assistant_response=turn["assistant_response"],
                    query_type=turn["query_type"],
                    confidence=turn["confidence"],
                    context_used=turn.get("context_used", {}),
                )
                for turn in turns_data
            ]

            return ConversationSession(
                session_id=session_data["session_id"],
                user_id=session_data["user_id"],
                started_at=datetime.fromisoformat(session_data["started_at"]),
                last_activity=datetime.fromisoformat(session_data["last_activity"]),
                turns=turns,
                persistent_context=json.loads(
                    session_data.get("persistent_context", "{}")
                ),
                metadata=json.loads(session_data.get("metadata", "{}")),
            )

        except Exception as e:
            self.logger.error(f"Error loading session from Redis: {e}")
            return None

    async def _save_session_to_redis(self, session: ConversationSession) -> None:
        """Save session to Redis."""
        try:
            redis_key = f"context:session:{session.user_id}:{session.session_id}"

            # Serialize turns
            turns_data = [
                {
                    "timestamp": turn.timestamp.isoformat(),
                    "user_query": turn.user_query,
                    "assistant_response": turn.assistant_response,
                    "query_type": turn.query_type,
                    "confidence": turn.confidence,
                    "context_used": turn.context_used or {},
                }
                for turn in session.turns
            ]

            session_data = {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "started_at": session.started_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "turns": json.dumps(turns_data),
                "persistent_context": json.dumps(session.persistent_context),
                "metadata": json.dumps(session.metadata or {}),
                "turn_count": len(session.turns),
            }

            await self.redis_client.hset(redis_key, mapping=session_data)

            # Set expiration (7 days)
            await self.redis_client.expire(redis_key, 604800)

        except Exception as e:
            self.logger.error(f"Error saving session to Redis: {e}")

    async def _cleanup_expired_sessions(self) -> None:
        """Background task to cleanup expired sessions."""
        while True:
            try:
                await asyncio.sleep(self.context_cleanup_interval)

                current_time = datetime.now()
                expired_sessions = []

                # Check in-memory cache for expired sessions
                for session_id, session in self.active_sessions.items():
                    if current_time - session.last_activity > self.session_timeout:
                        expired_sessions.append(session_id)

                # Remove expired sessions from memory
                for session_id in expired_sessions:
                    del self.active_sessions[session_id]

                if expired_sessions:
                    self.logger.info(
                        f"Cleaned up {len(expired_sessions)} expired sessions"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in session cleanup: {e}")

    async def shutdown(self) -> None:
        """Shutdown context manager."""
        try:
            if self.cleanup_task:
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass

            if self.redis_client:
                await self.redis_client.close()

            self.logger.info("Conversation Context Manager shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during context manager shutdown: {e}")
