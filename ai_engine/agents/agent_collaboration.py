"""
QBITEL - Agent Collaboration Framework

Provides multi-agent collaboration capabilities:
- Consensus protocols for decision making
- Negotiation strategies for resource allocation
- Collaborative problem solving
- Voting mechanisms
- Conflict resolution
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

from prometheus_client import Counter, Histogram

if TYPE_CHECKING:
    from .base_agent import BaseAgent
    from .agent_communication import AgentCommunicationProtocol, AgentMessage, MessageType

# Prometheus metrics
COLLABORATION_SESSIONS = Counter(
    "qbitel_collaboration_sessions_total",
    "Total collaboration sessions",
    ["session_type", "status"],
)
CONSENSUS_ROUNDS = Counter(
    "qbitel_consensus_rounds_total",
    "Total consensus rounds",
    ["protocol", "outcome"],
)
COLLABORATION_DURATION = Histogram(
    "qbitel_collaboration_duration_seconds",
    "Collaboration session duration",
    ["session_type"],
)

logger = logging.getLogger(__name__)


class ConsensusProtocol(str, Enum):
    """Types of consensus protocols."""

    MAJORITY = "majority"  # Simple majority voting
    UNANIMOUS = "unanimous"  # All must agree
    WEIGHTED = "weighted"  # Votes weighted by agent priority/expertise
    QUORUM = "quorum"  # Minimum participation required
    RANKED_CHOICE = "ranked_choice"  # Ranked choice voting
    BYZANTINE = "byzantine"  # Byzantine fault tolerant


class NegotiationStrategy(str, Enum):
    """Negotiation strategies."""

    COOPERATIVE = "cooperative"  # Win-win seeking
    COMPETITIVE = "competitive"  # Maximize own utility
    COMPROMISE = "compromise"  # Meet in the middle
    ACCOMMODATING = "accommodating"  # Prioritize relationship
    AVOIDING = "avoiding"  # Minimize conflict


class CollaborationStatus(str, Enum):
    """Status of a collaboration session."""

    INITIALIZING = "initializing"
    GATHERING = "gathering"  # Gathering participants
    DELIBERATING = "deliberating"  # Discussion phase
    VOTING = "voting"
    NEGOTIATING = "negotiating"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class VoteType(str, Enum):
    """Types of votes."""

    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"


@dataclass
class Vote:
    """A single vote from an agent."""

    vote_id: str
    agent_id: str
    agent_type: str
    vote: VoteType
    confidence: float = 1.0  # 0-1 confidence in vote
    reasoning: str = ""
    weight: float = 1.0  # Vote weight
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Proposal:
    """A proposal for collaboration."""

    proposal_id: str
    proposer_id: str
    title: str
    description: str
    options: List[Dict[str, Any]]  # Options to vote on
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    deadline: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NegotiationOffer:
    """An offer in a negotiation."""

    offer_id: str
    agent_id: str
    round_number: int
    terms: Dict[str, Any]
    utility_self: float  # How good is this for the offerer
    utility_other: float  # Estimated utility for other party
    is_final: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CollaborationSession:
    """A collaboration session between multiple agents."""

    session_id: str
    session_type: str  # consensus, negotiation, brainstorm, etc.
    initiator_id: str
    participants: Set[str] = field(default_factory=set)
    proposal: Optional[Proposal] = None
    votes: Dict[str, Vote] = field(default_factory=dict)
    offers: List[NegotiationOffer] = field(default_factory=list)
    status: CollaborationStatus = CollaborationStatus.INITIALIZING
    protocol: ConsensusProtocol = ConsensusProtocol.MAJORITY
    quorum: float = 0.5  # Minimum participation
    timeout_seconds: int = 300
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def participation_rate(self) -> float:
        """Calculate participation rate."""
        if not self.participants:
            return 0.0
        return len(self.votes) / len(self.participants)

    @property
    def is_expired(self) -> bool:
        """Check if session has expired."""
        elapsed = (datetime.utcnow() - self.created_at).total_seconds()
        return elapsed > self.timeout_seconds


class CollaborationFramework:
    """
    Framework for multi-agent collaboration.

    Provides:
    - Consensus building
    - Voting mechanisms
    - Negotiation support
    - Conflict resolution
    - Collaborative decision making
    """

    def __init__(
        self,
        communication: Optional["AgentCommunicationProtocol"] = None,
    ):
        """Initialize the collaboration framework."""
        self.communication = communication

        # Active sessions
        self.sessions: Dict[str, CollaborationSession] = {}
        self.completed_sessions: List[CollaborationSession] = []

        # Agent expertise weights (for weighted voting)
        self.agent_weights: Dict[str, Dict[str, float]] = {}

        # Statistics
        self.stats = {
            "total_sessions": 0,
            "successful_consensus": 0,
            "failed_consensus": 0,
            "negotiations_completed": 0,
        }

        self._running = False
        self._session_monitor: Optional[asyncio.Task] = None

        self.logger = logging.getLogger(f"{__name__}.CollaborationFramework")

    async def start(self) -> None:
        """Start the collaboration framework."""
        self._running = True
        self._session_monitor = asyncio.create_task(self._monitor_sessions())
        self.logger.info("Collaboration Framework started")

    async def stop(self) -> None:
        """Stop the collaboration framework."""
        self._running = False
        if self._session_monitor:
            self._session_monitor.cancel()
            try:
                await self._session_monitor
            except asyncio.CancelledError:
                pass
        self.logger.info("Collaboration Framework stopped")

    # Consensus Building

    async def initiate_consensus(
        self,
        initiator: "BaseAgent",
        proposal: Proposal,
        participants: Set[str],
        protocol: ConsensusProtocol = ConsensusProtocol.MAJORITY,
        quorum: float = 0.5,
        timeout_seconds: int = 300,
    ) -> CollaborationSession:
        """
        Initiate a consensus-building session.

        Args:
            initiator: Agent initiating the consensus
            proposal: The proposal to vote on
            participants: Set of agent IDs to participate
            protocol: Consensus protocol to use
            quorum: Minimum participation required
            timeout_seconds: Session timeout

        Returns:
            The created collaboration session
        """
        session = CollaborationSession(
            session_id=str(uuid.uuid4()),
            session_type="consensus",
            initiator_id=initiator.agent_id,
            participants=participants,
            proposal=proposal,
            protocol=protocol,
            quorum=quorum,
            timeout_seconds=timeout_seconds,
            status=CollaborationStatus.GATHERING,
        )

        self.sessions[session.session_id] = session
        self.stats["total_sessions"] += 1

        self.logger.info(f"Initiated consensus session: {session.session_id[:8]} " f"with {len(participants)} participants")

        # Notify participants
        await self._notify_participants(session)

        COLLABORATION_SESSIONS.labels(session_type="consensus", status="initiated").inc()

        return session

    async def submit_vote(
        self,
        session_id: str,
        agent: "BaseAgent",
        vote_type: VoteType,
        confidence: float = 1.0,
        reasoning: str = "",
    ) -> bool:
        """
        Submit a vote in a consensus session.

        Args:
            session_id: Session to vote in
            agent: Voting agent
            vote_type: The vote
            confidence: Confidence level
            reasoning: Explanation for vote

        Returns:
            Whether vote was accepted
        """
        session = self.sessions.get(session_id)
        if not session:
            self.logger.warning(f"Session not found: {session_id}")
            return False

        if session.status not in [CollaborationStatus.GATHERING, CollaborationStatus.VOTING]:
            self.logger.warning(f"Session not accepting votes: {session_id}")
            return False

        if agent.agent_id not in session.participants:
            self.logger.warning(f"Agent {agent.agent_id[:8]} not a participant")
            return False

        # Calculate vote weight
        weight = self._get_agent_weight(agent.agent_id, session.proposal.title)

        vote = Vote(
            vote_id=str(uuid.uuid4()),
            agent_id=agent.agent_id,
            agent_type=agent.agent_type,
            vote=vote_type,
            confidence=confidence,
            reasoning=reasoning,
            weight=weight,
        )

        session.votes[agent.agent_id] = vote
        session.status = CollaborationStatus.VOTING

        self.logger.debug(f"Vote submitted: {agent.agent_id[:8]} voted {vote_type.value} " f"in session {session_id[:8]}")

        # Check if consensus is reached
        await self._check_consensus(session)

        return True

    async def _check_consensus(self, session: CollaborationSession) -> None:
        """Check if consensus has been reached."""
        if session.participation_rate < session.quorum:
            return  # Not enough votes yet

        # Calculate result based on protocol
        result = await self._calculate_consensus_result(session)

        if result["decided"]:
            session.status = CollaborationStatus.COMPLETED
            session.completed_at = datetime.utcnow()
            session.result = result

            self.stats["successful_consensus"] += 1
            CONSENSUS_ROUNDS.labels(protocol=session.protocol.value, outcome="success").inc()

            # Move to completed
            self.completed_sessions.append(session)
            del self.sessions[session.session_id]

            self.logger.info(f"Consensus reached in session {session.session_id[:8]}: " f"{result['outcome']}")

            # Notify participants of result
            await self._notify_result(session)

    async def _calculate_consensus_result(self, session: CollaborationSession) -> Dict[str, Any]:
        """Calculate consensus result based on protocol."""
        votes = list(session.votes.values())

        if not votes:
            return {"decided": False, "reason": "no_votes"}

        approve_count = sum(1 for v in votes if v.vote == VoteType.APPROVE)
        reject_count = sum(1 for v in votes if v.vote == VoteType.REJECT)
        abstain_count = sum(1 for v in votes if v.vote == VoteType.ABSTAIN)

        # Weighted counts
        approve_weighted = sum(v.weight * v.confidence for v in votes if v.vote == VoteType.APPROVE)
        reject_weighted = sum(v.weight * v.confidence for v in votes if v.vote == VoteType.REJECT)

        total_votes = len(votes)
        total_weighted = approve_weighted + reject_weighted

        result = {
            "approve_count": approve_count,
            "reject_count": reject_count,
            "abstain_count": abstain_count,
            "approve_weighted": approve_weighted,
            "reject_weighted": reject_weighted,
            "participation_rate": session.participation_rate,
        }

        if session.protocol == ConsensusProtocol.MAJORITY:
            if approve_count > total_votes / 2:
                result["decided"] = True
                result["outcome"] = "approved"
            elif reject_count > total_votes / 2:
                result["decided"] = True
                result["outcome"] = "rejected"
            else:
                result["decided"] = False
                result["reason"] = "no_majority"

        elif session.protocol == ConsensusProtocol.UNANIMOUS:
            if approve_count == total_votes:
                result["decided"] = True
                result["outcome"] = "approved"
            elif reject_count > 0:
                result["decided"] = True
                result["outcome"] = "rejected"
            else:
                result["decided"] = False
                result["reason"] = "not_unanimous"

        elif session.protocol == ConsensusProtocol.WEIGHTED:
            if total_weighted > 0:
                if approve_weighted / total_weighted > 0.5:
                    result["decided"] = True
                    result["outcome"] = "approved"
                elif reject_weighted / total_weighted > 0.5:
                    result["decided"] = True
                    result["outcome"] = "rejected"
                else:
                    result["decided"] = False
                    result["reason"] = "no_weighted_majority"
            else:
                result["decided"] = False
                result["reason"] = "no_weighted_votes"

        elif session.protocol == ConsensusProtocol.QUORUM:
            if session.participation_rate >= session.quorum:
                if approve_count > reject_count:
                    result["decided"] = True
                    result["outcome"] = "approved"
                elif reject_count > approve_count:
                    result["decided"] = True
                    result["outcome"] = "rejected"
                else:
                    result["decided"] = False
                    result["reason"] = "tie"
            else:
                result["decided"] = False
                result["reason"] = "quorum_not_met"

        else:
            # Default to majority
            if approve_count > reject_count:
                result["decided"] = True
                result["outcome"] = "approved"
            elif reject_count > approve_count:
                result["decided"] = True
                result["outcome"] = "rejected"
            else:
                result["decided"] = False
                result["reason"] = "tie"

        return result

    # Negotiation

    async def initiate_negotiation(
        self,
        initiator: "BaseAgent",
        counterpart_id: str,
        topic: str,
        initial_terms: Dict[str, Any],
        strategy: NegotiationStrategy = NegotiationStrategy.COOPERATIVE,
        max_rounds: int = 10,
        timeout_seconds: int = 600,
    ) -> CollaborationSession:
        """
        Initiate a negotiation session.

        Args:
            initiator: Agent initiating negotiation
            counterpart_id: ID of the other negotiating agent
            topic: Subject of negotiation
            initial_terms: Initial offer terms
            strategy: Negotiation strategy
            max_rounds: Maximum negotiation rounds
            timeout_seconds: Session timeout

        Returns:
            The created negotiation session
        """
        session = CollaborationSession(
            session_id=str(uuid.uuid4()),
            session_type="negotiation",
            initiator_id=initiator.agent_id,
            participants={initiator.agent_id, counterpart_id},
            status=CollaborationStatus.NEGOTIATING,
            timeout_seconds=timeout_seconds,
            metadata={
                "topic": topic,
                "strategy": strategy.value,
                "max_rounds": max_rounds,
                "current_round": 1,
            },
        )

        # Add initial offer
        initial_offer = NegotiationOffer(
            offer_id=str(uuid.uuid4()),
            agent_id=initiator.agent_id,
            round_number=1,
            terms=initial_terms,
            utility_self=self._calculate_utility(initial_terms, initiator.agent_id),
            utility_other=0.5,  # Estimated
        )
        session.offers.append(initial_offer)

        self.sessions[session.session_id] = session
        self.stats["total_sessions"] += 1

        self.logger.info(
            f"Initiated negotiation: {session.session_id[:8]} " f"between {initiator.agent_id[:8]} and {counterpart_id[:8]}"
        )

        # Notify counterpart
        await self._notify_negotiation(session, counterpart_id)

        COLLABORATION_SESSIONS.labels(session_type="negotiation", status="initiated").inc()

        return session

    async def submit_counter_offer(
        self,
        session_id: str,
        agent: "BaseAgent",
        terms: Dict[str, Any],
        is_final: bool = False,
    ) -> Optional[NegotiationOffer]:
        """
        Submit a counter-offer in a negotiation.

        Args:
            session_id: Negotiation session
            agent: Offering agent
            terms: Counter-offer terms
            is_final: Whether this is a final offer

        Returns:
            The created offer
        """
        session = self.sessions.get(session_id)
        if not session:
            return None

        if session.status != CollaborationStatus.NEGOTIATING:
            return None

        if agent.agent_id not in session.participants:
            return None

        current_round = session.metadata.get("current_round", 1)
        max_rounds = session.metadata.get("max_rounds", 10)

        if current_round > max_rounds:
            await self._fail_negotiation(session, "max_rounds_exceeded")
            return None

        offer = NegotiationOffer(
            offer_id=str(uuid.uuid4()),
            agent_id=agent.agent_id,
            round_number=current_round,
            terms=terms,
            utility_self=self._calculate_utility(terms, agent.agent_id),
            utility_other=self._estimate_other_utility(terms, session, agent.agent_id),
            is_final=is_final,
        )

        session.offers.append(offer)
        session.metadata["current_round"] = current_round + 1

        self.logger.debug(f"Counter-offer in session {session_id[:8]}: " f"round {current_round}, final={is_final}")

        # Check for agreement
        await self._check_negotiation_outcome(session)

        return offer

    async def accept_offer(
        self,
        session_id: str,
        agent: "BaseAgent",
    ) -> bool:
        """Accept the current offer in a negotiation."""
        session = self.sessions.get(session_id)
        if not session or not session.offers:
            return False

        last_offer = session.offers[-1]
        if last_offer.agent_id == agent.agent_id:
            # Can't accept your own offer
            return False

        session.status = CollaborationStatus.COMPLETED
        session.completed_at = datetime.utcnow()
        session.result = {
            "outcome": "agreement",
            "final_terms": last_offer.terms,
            "rounds": session.metadata.get("current_round", 1),
        }

        self.stats["negotiations_completed"] += 1

        self.completed_sessions.append(session)
        del self.sessions[session_id]

        self.logger.info(f"Negotiation completed: {session_id[:8]} - agreement reached")

        return True

    async def reject_offer(
        self,
        session_id: str,
        agent: "BaseAgent",
        reason: str = "",
    ) -> bool:
        """Reject the current offer and end negotiation."""
        session = self.sessions.get(session_id)
        if not session:
            return False

        await self._fail_negotiation(session, f"rejected: {reason}")
        return True

    async def _check_negotiation_outcome(self, session: CollaborationSession) -> None:
        """Check if negotiation should conclude."""
        if len(session.offers) < 2:
            return

        last_offer = session.offers[-1]
        prev_offer = session.offers[-2]

        # Check if offers are converging
        if last_offer.is_final and prev_offer.is_final:
            # Both sides made final offers
            if self._offers_compatible(last_offer.terms, prev_offer.terms):
                session.status = CollaborationStatus.COMPLETED
                session.completed_at = datetime.utcnow()
                session.result = {
                    "outcome": "compromise",
                    "final_terms": self._merge_offers(last_offer.terms, prev_offer.terms),
                    "rounds": session.metadata.get("current_round", 1),
                }
                self.stats["negotiations_completed"] += 1
            else:
                await self._fail_negotiation(session, "incompatible_final_offers")

    async def _fail_negotiation(self, session: CollaborationSession, reason: str) -> None:
        """Mark negotiation as failed."""
        session.status = CollaborationStatus.FAILED
        session.completed_at = datetime.utcnow()
        session.result = {
            "outcome": "failed",
            "reason": reason,
            "rounds": session.metadata.get("current_round", 1),
        }

        self.completed_sessions.append(session)
        if session.session_id in self.sessions:
            del self.sessions[session.session_id]

        self.logger.info(f"Negotiation failed: {session.session_id[:8]} - {reason}")

    def _calculate_utility(self, terms: Dict[str, Any], agent_id: str) -> float:
        """Calculate utility of terms for an agent."""
        # Simplified utility calculation
        # In practice, this would be agent-specific
        return 0.5 + sum(0.1 for v in terms.values() if isinstance(v, (int, float)) and v > 0)

    def _estimate_other_utility(self, terms: Dict[str, Any], session: CollaborationSession, my_agent_id: str) -> float:
        """Estimate utility for the other party."""
        # Look at previous offers to estimate
        other_offers = [o for o in session.offers if o.agent_id != my_agent_id]
        if not other_offers:
            return 0.5
        return other_offers[-1].utility_self

    def _offers_compatible(self, terms1: Dict[str, Any], terms2: Dict[str, Any]) -> bool:
        """Check if two offers are compatible."""
        # Simplified compatibility check
        common_keys = set(terms1.keys()) & set(terms2.keys())
        if not common_keys:
            return True

        for key in common_keys:
            v1, v2 = terms1[key], terms2[key]
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                if abs(v1 - v2) / max(abs(v1), abs(v2), 1) > 0.2:
                    return False
            elif v1 != v2:
                return False

        return True

    def _merge_offers(self, terms1: Dict[str, Any], terms2: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two offers into a compromise."""
        merged = {}
        all_keys = set(terms1.keys()) | set(terms2.keys())

        for key in all_keys:
            v1 = terms1.get(key)
            v2 = terms2.get(key)

            if v1 is None:
                merged[key] = v2
            elif v2 is None:
                merged[key] = v1
            elif isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                merged[key] = (v1 + v2) / 2
            else:
                merged[key] = v1  # Default to first

        return merged

    # Agent Weights

    def set_agent_weight(self, agent_id: str, topic: str, weight: float) -> None:
        """Set an agent's expertise weight for a topic."""
        if agent_id not in self.agent_weights:
            self.agent_weights[agent_id] = {}
        self.agent_weights[agent_id][topic] = weight

    def _get_agent_weight(self, agent_id: str, topic: str) -> float:
        """Get an agent's weight for a topic."""
        if agent_id in self.agent_weights:
            return self.agent_weights[agent_id].get(topic, 1.0)
        return 1.0

    # Session Management

    async def _notify_participants(self, session: CollaborationSession) -> None:
        """Notify participants of a new session."""
        if not self.communication:
            return

        for participant_id in session.participants:
            if participant_id != session.initiator_id:
                try:
                    await self.communication.send(
                        sender=None,  # System message
                        recipient_id=participant_id,
                        message_type="PROPOSAL",
                        payload={
                            "session_id": session.session_id,
                            "proposal": (
                                {
                                    "title": session.proposal.title,
                                    "description": session.proposal.description,
                                    "options": session.proposal.options,
                                }
                                if session.proposal
                                else {}
                            ),
                            "protocol": session.protocol.value,
                            "deadline": (
                                session.proposal.deadline.isoformat()
                                if session.proposal and session.proposal.deadline
                                else None
                            ),
                        },
                    )
                except Exception as e:
                    self.logger.error(f"Failed to notify {participant_id[:8]}: {e}")

    async def _notify_negotiation(self, session: CollaborationSession, counterpart_id: str) -> None:
        """Notify counterpart of negotiation initiation."""
        if not self.communication:
            return

        try:
            await self.communication.send(
                sender=None,
                recipient_id=counterpart_id,
                message_type="NEGOTIATION",
                payload={
                    "session_id": session.session_id,
                    "topic": session.metadata.get("topic"),
                    "initial_offer": session.offers[0].terms if session.offers else {},
                    "strategy": session.metadata.get("strategy"),
                },
            )
        except Exception as e:
            self.logger.error(f"Failed to notify counterpart: {e}")

    async def _notify_result(self, session: CollaborationSession) -> None:
        """Notify participants of session result."""
        if not self.communication:
            return

        for participant_id in session.participants:
            try:
                await self.communication.send(
                    sender=None,
                    recipient_id=participant_id,
                    message_type="CONSENSUS_RESULT",
                    payload={
                        "session_id": session.session_id,
                        "result": session.result,
                    },
                )
            except Exception as e:
                self.logger.error(f"Failed to notify {participant_id[:8]}: {e}")

    async def _monitor_sessions(self) -> None:
        """Monitor sessions for timeouts."""
        while self._running:
            try:
                expired = []
                for session_id, session in self.sessions.items():
                    if session.is_expired:
                        expired.append(session_id)

                for session_id in expired:
                    session = self.sessions[session_id]
                    session.status = CollaborationStatus.TIMEOUT
                    session.completed_at = datetime.utcnow()
                    session.result = {"outcome": "timeout"}

                    self.completed_sessions.append(session)
                    del self.sessions[session_id]

                    self.logger.warning(f"Session timed out: {session_id[:8]}")
                    COLLABORATION_SESSIONS.labels(session_type=session.session_type, status="timeout").inc()

                await asyncio.sleep(10)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Session monitor error: {e}")
                await asyncio.sleep(10)

    def get_session(self, session_id: str) -> Optional[CollaborationSession]:
        """Get a session by ID."""
        return self.sessions.get(session_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get collaboration statistics."""
        return {
            **self.stats,
            "active_sessions": len(self.sessions),
            "completed_sessions": len(self.completed_sessions),
        }
