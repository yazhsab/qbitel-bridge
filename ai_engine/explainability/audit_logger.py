"""
CRONOS AI - Decision Audit Logger

Immutable audit trail logging for AI decisions with explanations.
Meets regulatory requirements for EU AI Act, FDA 21 CFR Part 11, SOC2.
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, and_, or_

from .base import ExplanationResult
from .database import AIDecisionAudit, ModelDriftMetric
from .metrics import record_audit_trail_write
from ..core.exceptions import CronosAIException

logger = logging.getLogger(__name__)


class AuditLoggerException(CronosAIException):
    """Exception raised by audit logger."""
    pass


class DecisionAuditLogger:
    """
    Audit logger for AI decisions with explanations.

    Features:
    - Immutable audit trail (append-only)
    - Async database writes for performance
    - Automatic retry on transient failures
    - Prometheus metrics integration
    - Query interface for audit retrieval
    """

    def __init__(
        self,
        database_url: str,
        max_retries: int = 3,
        batch_size: int = 100,
    ):
        """
        Initialize audit logger.

        Args:
            database_url: PostgreSQL connection URL
            max_retries: Maximum retry attempts for failed writes
            batch_size: Maximum batch size for bulk writes
        """
        self.database_url = database_url
        self.max_retries = max_retries
        self.batch_size = batch_size

        # Create async engine with appropriate parameters based on dialect
        engine_kwargs = {"echo": False}

        # SQLite doesn't support pool_size and max_overflow
        if not database_url.startswith("sqlite"):
            engine_kwargs["pool_size"] = 10
            engine_kwargs["max_overflow"] = 20

        self.engine = create_async_engine(database_url, **engine_kwargs)

        # Create async session factory
        self.async_session = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        logger.info(f"Initialized DecisionAuditLogger with database: {database_url}")

    async def log_decision(
        self,
        explanation: ExplanationResult,
        event_type: str,
        event_data: Dict[str, Any],
        compliance_framework: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        inference_time_ms: Optional[float] = None,
        **metadata,
    ) -> str:
        """
        Log an AI decision to the audit trail.

        Args:
            explanation: ExplanationResult from explainer
            event_type: Type of event (e.g., 'protocol_classification', 'security_decision')
            event_data: Event-specific data (input context)
            compliance_framework: Applicable framework ('SOC2', 'HIPAA', 'EU_AI_ACT', etc.)
            user_id: User who triggered the decision
            session_id: Session identifier
            request_id: Request tracing identifier
            inference_time_ms: Model inference time
            **metadata: Additional metadata

        Returns:
            UUID of created audit record
        """
        start_time = datetime.now(timezone.utc)

        try:
            audit_record = AIDecisionAudit(
                id=uuid4(),
                decision_id=explanation.decision_id,
                timestamp=explanation.timestamp,

                # Event context
                event_type=event_type,
                event_data=event_data,

                # Model information
                model_name=explanation.model_name,
                model_version=explanation.model_version,
                model_architecture=metadata.get('model_architecture'),

                # Decision details
                decision_output={"result": explanation.model_output},
                confidence_score=explanation.confidence_score,

                # Explainability data
                explanation_method=explanation.explanation_method.value,
                explanation_id=explanation.explanation_id,
                feature_importance=[fi.__dict__ for fi in explanation.feature_importances],
                top_features=[fi.__dict__ for fi in explanation.top_features],
                decision_rationale=explanation.explanation_summary,
                regulatory_justification=explanation.regulatory_justification,
                counterfactual=explanation.counterfactual,

                # Audit metadata
                user_id=user_id,
                session_id=session_id,
                compliance_framework=compliance_framework,
                request_id=request_id,

                # Performance
                inference_time_ms=inference_time_ms,
                explanation_time_ms=explanation.metadata.get('explanation_time_ms'),

                # Additional metadata
                additional_metadata={
                    **explanation.metadata,
                    **metadata,
                },
            )

            # Write to database with retry logic
            record_id = await self._write_with_retry(audit_record)

            # Record metrics
            duration_seconds = (datetime.now(timezone.utc) - start_time).total_seconds()
            record_audit_trail_write(
                model_name=explanation.model_name,
                compliance_framework=compliance_framework or "none",
                duration_seconds=duration_seconds,
                success=True,
            )

            logger.info(
                f"Logged decision {explanation.decision_id} to audit trail "
                f"(record_id={record_id})"
            )

            return str(record_id)

        except Exception as e:
            logger.error(f"Failed to log decision {explanation.decision_id}: {e}")

            # Record failure metric
            record_audit_trail_write(
                model_name=explanation.model_name,
                compliance_framework=compliance_framework or "none",
                duration_seconds=0,
                success=False,
            )

            raise AuditLoggerException(f"Audit logging failed: {e}") from e

    async def log_batch(
        self,
        explanations: List[ExplanationResult],
        event_type: str,
        event_data_list: List[Dict[str, Any]],
        **kwargs,
    ) -> List[str]:
        """
        Log multiple decisions in batch for efficiency.

        Args:
            explanations: List of ExplanationResult objects
            event_type: Event type for all decisions
            event_data_list: List of event data (same length as explanations)
            **kwargs: Additional parameters passed to log_decision

        Returns:
            List of audit record IDs
        """
        if len(explanations) != len(event_data_list):
            raise ValueError("explanations and event_data_list must have same length")

        record_ids = []
        for explanation, event_data in zip(explanations, event_data_list):
            record_id = await self.log_decision(
                explanation=explanation,
                event_type=event_type,
                event_data=event_data,
                **kwargs,
            )
            record_ids.append(record_id)

        return record_ids

    async def get_decision(self, decision_id: str) -> Optional[AIDecisionAudit]:
        """
        Retrieve audit record by decision ID.

        Args:
            decision_id: Decision identifier

        Returns:
            AIDecisionAudit record or None if not found
        """
        async with self.async_session() as session:
            result = await session.execute(
                select(AIDecisionAudit).where(AIDecisionAudit.decision_id == decision_id)
            )
            return result.scalar_one_or_none()

    async def query_decisions(
        self,
        model_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        compliance_framework: Optional[str] = None,
        min_confidence: Optional[float] = None,
        max_confidence: Optional[float] = None,
        human_reviewed: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AIDecisionAudit]:
        """
        Query audit trail with filters.

        Args:
            model_name: Filter by model name
            start_date: Filter by timestamp >= start_date
            end_date: Filter by timestamp <= end_date
            compliance_framework: Filter by compliance framework
            min_confidence: Filter by confidence_score >= min_confidence
            max_confidence: Filter by confidence_score <= max_confidence
            human_reviewed: Filter by human review status
            limit: Maximum results to return
            offset: Offset for pagination

        Returns:
            List of AIDecisionAudit records
        """
        async with self.async_session() as session:
            query = select(AIDecisionAudit)

            # Apply filters
            filters = []
            if model_name:
                filters.append(AIDecisionAudit.model_name == model_name)
            if start_date:
                filters.append(AIDecisionAudit.timestamp >= start_date)
            if end_date:
                filters.append(AIDecisionAudit.timestamp <= end_date)
            if compliance_framework:
                filters.append(AIDecisionAudit.compliance_framework == compliance_framework)
            if min_confidence is not None:
                filters.append(AIDecisionAudit.confidence_score >= min_confidence)
            if max_confidence is not None:
                filters.append(AIDecisionAudit.confidence_score <= max_confidence)
            if human_reviewed is not None:
                filters.append(AIDecisionAudit.human_reviewed == human_reviewed)

            if filters:
                query = query.where(and_(*filters))

            # Order by timestamp descending (most recent first)
            query = query.order_by(AIDecisionAudit.timestamp.desc())

            # Apply pagination
            query = query.limit(limit).offset(offset)

            result = await session.execute(query)
            return result.scalars().all()

    async def mark_human_reviewed(
        self,
        decision_id: str,
        reviewer_id: str,
        override: bool = False,
        override_reason: Optional[str] = None,
    ):
        """
        Mark a decision as human-reviewed.

        Args:
            decision_id: Decision to review
            reviewer_id: User ID of reviewer
            override: Whether reviewer overrode the AI decision
            override_reason: Reason for override (if applicable)
        """
        async with self.async_session() as session:
            result = await session.execute(
                select(AIDecisionAudit).where(AIDecisionAudit.decision_id == decision_id)
            )
            record = result.scalar_one_or_none()

            if not record:
                raise AuditLoggerException(f"Decision {decision_id} not found")

            record.human_reviewed = True
            record.human_override = override
            record.override_reason = override_reason
            record.reviewer_id = reviewer_id
            record.review_timestamp = datetime.now(timezone.utc)

            await session.commit()

            logger.info(
                f"Marked decision {decision_id} as human-reviewed "
                f"(override={override}, reviewer={reviewer_id})"
            )

    async def get_compliance_report(
        self,
        compliance_framework: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, Any]:
        """
        Generate compliance report for a specific framework.

        Args:
            compliance_framework: Framework to report on
            start_date: Report start date
            end_date: Report end date

        Returns:
            Compliance report dictionary
        """
        decisions = await self.query_decisions(
            compliance_framework=compliance_framework,
            start_date=start_date,
            end_date=end_date,
            limit=10000,
        )

        # Calculate statistics
        total_decisions = len(decisions)
        human_reviewed_count = sum(1 for d in decisions if d.human_reviewed)
        human_override_count = sum(1 for d in decisions if d.human_override)
        avg_confidence = sum(d.confidence_score for d in decisions) / total_decisions if total_decisions > 0 else 0

        # Confidence distribution
        confidence_buckets = {
            "0.0-0.5": 0,
            "0.5-0.7": 0,
            "0.7-0.9": 0,
            "0.9-1.0": 0,
        }
        for d in decisions:
            if d.confidence_score < 0.5:
                confidence_buckets["0.0-0.5"] += 1
            elif d.confidence_score < 0.7:
                confidence_buckets["0.5-0.7"] += 1
            elif d.confidence_score < 0.9:
                confidence_buckets["0.7-0.9"] += 1
            else:
                confidence_buckets["0.9-1.0"] += 1

        return {
            "compliance_framework": compliance_framework,
            "report_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "total_decisions": total_decisions,
            "human_reviewed_count": human_reviewed_count,
            "human_reviewed_percentage": (human_reviewed_count / total_decisions * 100) if total_decisions > 0 else 0,
            "human_override_count": human_override_count,
            "human_override_percentage": (human_override_count / total_decisions * 100) if total_decisions > 0 else 0,
            "average_confidence": avg_confidence,
            "confidence_distribution": confidence_buckets,
            "models_used": list(set(d.model_name for d in decisions)),
        }

    async def _write_with_retry(
        self,
        record: AIDecisionAudit,
        retry_count: int = 0,
    ) -> uuid4:
        """
        Write audit record with retry logic.

        Args:
            record: AIDecisionAudit to write
            retry_count: Current retry attempt

        Returns:
            Record ID

        Raises:
            AuditLoggerException: If max retries exceeded
        """
        try:
            async with self.async_session() as session:
                session.add(record)
                await session.commit()
                return record.id

        except Exception as e:
            if retry_count < self.max_retries:
                logger.warning(
                    f"Audit write failed (attempt {retry_count + 1}/{self.max_retries}): {e}"
                )
                # Exponential backoff
                await asyncio.sleep(2 ** retry_count)
                return await self._write_with_retry(record, retry_count + 1)
            else:
                raise AuditLoggerException(
                    f"Failed to write audit record after {self.max_retries} retries"
                ) from e

    async def close(self):
        """Close database connections."""
        await self.engine.dispose()
        logger.info("Closed DecisionAuditLogger database connections")


# Singleton instance
_audit_logger: Optional[DecisionAuditLogger] = None


def get_audit_logger(database_url: Optional[str] = None) -> DecisionAuditLogger:
    """
    Get singleton audit logger instance.

    Args:
        database_url: Database connection URL (required on first call)

    Returns:
        DecisionAuditLogger instance
    """
    global _audit_logger

    if _audit_logger is None:
        if database_url is None:
            raise ValueError("database_url required for first initialization")
        _audit_logger = DecisionAuditLogger(database_url)

    return _audit_logger


async def log_decision_async(
    explanation: ExplanationResult,
    event_type: str,
    event_data: Dict[str, Any],
    **kwargs,
) -> str:
    """
    Convenience function for logging a decision.

    Args:
        explanation: ExplanationResult
        event_type: Event type
        event_data: Event data
        **kwargs: Additional parameters

    Returns:
        Audit record ID
    """
    logger_instance = get_audit_logger()
    return await logger_instance.log_decision(
        explanation=explanation,
        event_type=event_type,
        event_data=event_data,
        **kwargs,
    )
