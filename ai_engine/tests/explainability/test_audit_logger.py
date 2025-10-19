"""
Tests for Decision Audit Logger.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from ai_engine.explainability.audit_logger import (
    DecisionAuditLogger,
    AuditLoggerException,
)
from ai_engine.explainability.base import (
    ExplanationResult,
    ExplanationType,
    FeatureImportance,
)
from ai_engine.explainability.database import AIDecisionAudit


@pytest.fixture
def sample_explanation():
    """Create sample explanation result for testing."""
    return ExplanationResult(
        explanation_id="exp-001",
        decision_id="dec-001",
        timestamp=datetime.utcnow(),
        model_name="test_model",
        model_version="1.0.0",
        explanation_method=ExplanationType.SHAP,
        input_data=b"test input",
        model_output="HTTP",
        confidence_score=0.95,
        feature_importances=[
            FeatureImportance("byte_0", 71, 0.15, 1, "G character"),
            FeatureImportance("byte_1", 69, 0.12, 2, "E character"),
        ],
        top_features=[
            FeatureImportance("byte_0", 71, 0.15, 1, "G character"),
        ],
        explanation_summary="Test explanation",
        regulatory_justification="Test justification",
        metadata={"explanation_time_ms": 42.5},
    )


class TestDecisionAuditLogger:
    """Tests for DecisionAuditLogger."""

    def test_initialization(self):
        """Test logger initialization."""
        logger = DecisionAuditLogger(
            database_url="sqlite+aiosqlite:///:memory:",
            max_retries=3,
            batch_size=100,
        )

        assert logger.max_retries == 3
        assert logger.batch_size == 100
        assert logger.engine is not None

    @pytest.mark.asyncio
    async def test_log_decision(self, sample_explanation):
        """Test logging a decision to audit trail."""
        logger = DecisionAuditLogger(
            database_url="sqlite+aiosqlite:///:memory:",
        )

        # Create tables (in real app, done by alembic)
        async with logger.engine.begin() as conn:
            await conn.run_sync(AIDecisionAudit.metadata.create_all)

        record_id = await logger.log_decision(
            explanation=sample_explanation,
            event_type="test_event",
            event_data={"test": "data"},
            compliance_framework="SOC2",
            user_id="test-user",
            session_id="test-session",
        )

        assert record_id is not None
        assert isinstance(record_id, str)

        # Verify record was created
        decision = await logger.get_decision(sample_explanation.decision_id)
        assert decision is not None
        assert decision.decision_id == "dec-001"
        assert decision.model_name == "test_model"
        assert decision.confidence_score == 0.95
        assert decision.compliance_framework == "SOC2"
        assert decision.user_id == "test-user"

        await logger.close()

    @pytest.mark.asyncio
    async def test_log_batch(self, sample_explanation):
        """Test batch logging."""
        logger = DecisionAuditLogger(
            database_url="sqlite+aiosqlite:///:memory:",
        )

        async with logger.engine.begin() as conn:
            await conn.run_sync(AIDecisionAudit.metadata.create_all)

        # Create multiple explanations
        explanations = []
        event_data_list = []
        for i in range(3):
            exp = ExplanationResult(
                explanation_id=f"exp-{i}",
                decision_id=f"dec-{i}",
                timestamp=datetime.utcnow(),
                model_name="test_model",
                model_version="1.0.0",
                explanation_method=ExplanationType.SHAP,
                input_data=b"test",
                model_output="HTTP",
                confidence_score=0.95,
                feature_importances=[],
                top_features=[],
                explanation_summary="Test",
                metadata={},
            )
            explanations.append(exp)
            event_data_list.append({"batch": i})

        record_ids = await logger.log_batch(
            explanations=explanations,
            event_type="batch_test",
            event_data_list=event_data_list,
        )

        assert len(record_ids) == 3
        assert all(isinstance(rid, str) for rid in record_ids)

        await logger.close()

    @pytest.mark.asyncio
    async def test_query_decisions(self, sample_explanation):
        """Test querying decisions with filters."""
        logger = DecisionAuditLogger(
            database_url="sqlite+aiosqlite:///:memory:",
        )

        async with logger.engine.begin() as conn:
            await conn.run_sync(AIDecisionAudit.metadata.create_all)

        # Log a decision
        await logger.log_decision(
            explanation=sample_explanation,
            event_type="test_event",
            event_data={},
            compliance_framework="SOC2",
        )

        # Query by model name
        results = await logger.query_decisions(
            model_name="test_model",
            limit=10,
        )

        assert len(results) == 1
        assert results[0].model_name == "test_model"

        # Query by confidence range
        results = await logger.query_decisions(
            model_name="test_model",
            min_confidence=0.90,
            max_confidence=1.0,
        )

        assert len(results) == 1

        # Query by compliance framework
        results = await logger.query_decisions(
            compliance_framework="SOC2",
        )

        assert len(results) == 1

        await logger.close()

    @pytest.mark.asyncio
    async def test_mark_human_reviewed(self, sample_explanation):
        """Test marking decision as human-reviewed."""
        logger = DecisionAuditLogger(
            database_url="sqlite+aiosqlite:///:memory:",
        )

        async with logger.engine.begin() as conn:
            await conn.run_sync(AIDecisionAudit.metadata.create_all)

        await logger.log_decision(
            explanation=sample_explanation,
            event_type="test_event",
            event_data={},
        )

        # Mark as reviewed
        await logger.mark_human_reviewed(
            decision_id="dec-001",
            reviewer_id="reviewer-123",
            override=True,
            override_reason="Incorrect classification",
        )

        # Verify review status
        decision = await logger.get_decision("dec-001")
        assert decision.human_reviewed is True
        assert decision.human_override is True
        assert decision.reviewer_id == "reviewer-123"
        assert decision.override_reason == "Incorrect classification"
        assert decision.review_timestamp is not None

        await logger.close()

    @pytest.mark.asyncio
    async def test_get_compliance_report(self, sample_explanation):
        """Test compliance report generation."""
        logger = DecisionAuditLogger(
            database_url="sqlite+aiosqlite:///:memory:",
        )

        async with logger.engine.begin() as conn:
            await conn.run_sync(AIDecisionAudit.metadata.create_all)

        # Log multiple decisions
        for i in range(5):
            exp = ExplanationResult(
                explanation_id=f"exp-{i}",
                decision_id=f"dec-{i}",
                timestamp=datetime.utcnow(),
                model_name="test_model",
                model_version="1.0.0",
                explanation_method=ExplanationType.SHAP,
                input_data=b"test",
                model_output="HTTP",
                confidence_score=0.90 + (i * 0.01),  # Varying confidence
                feature_importances=[],
                top_features=[],
                explanation_summary="Test",
                metadata={},
            )
            await logger.log_decision(
                explanation=exp,
                event_type="test_event",
                event_data={},
                compliance_framework="EU_AI_ACT",
            )

        # Generate report
        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow() + timedelta(days=1)

        report = await logger.get_compliance_report(
            compliance_framework="EU_AI_ACT",
            start_date=start_date,
            end_date=end_date,
        )

        assert report["compliance_framework"] == "EU_AI_ACT"
        assert report["total_decisions"] == 5
        assert report["human_reviewed_count"] == 0
        assert report["average_confidence"] > 0.90
        assert "confidence_distribution" in report
        assert "test_model" in report["models_used"]

        await logger.close()

    @pytest.mark.asyncio
    async def test_get_decision_not_found(self):
        """Test getting non-existent decision."""
        logger = DecisionAuditLogger(
            database_url="sqlite+aiosqlite:///:memory:",
        )

        async with logger.engine.begin() as conn:
            await conn.run_sync(AIDecisionAudit.metadata.create_all)

        decision = await logger.get_decision("nonexistent")
        assert decision is None

        await logger.close()

    @pytest.mark.asyncio
    async def test_mark_human_reviewed_not_found(self):
        """Test marking non-existent decision as reviewed."""
        logger = DecisionAuditLogger(
            database_url="sqlite+aiosqlite:///:memory:",
        )

        async with logger.engine.begin() as conn:
            await conn.run_sync(AIDecisionAudit.metadata.create_all)

        with pytest.raises(AuditLoggerException):
            await logger.mark_human_reviewed(
                decision_id="nonexistent",
                reviewer_id="reviewer-123",
            )

        await logger.close()

    @pytest.mark.asyncio
    async def test_query_with_date_range(self, sample_explanation):
        """Test querying with date range filters."""
        logger = DecisionAuditLogger(
            database_url="sqlite+aiosqlite:///:memory:",
        )

        async with logger.engine.begin() as conn:
            await conn.run_sync(AIDecisionAudit.metadata.create_all)

        await logger.log_decision(
            explanation=sample_explanation,
            event_type="test_event",
            event_data={},
        )

        # Query with date range that includes the decision
        start_date = datetime.utcnow() - timedelta(hours=1)
        end_date = datetime.utcnow() + timedelta(hours=1)

        results = await logger.query_decisions(
            start_date=start_date,
            end_date=end_date,
        )

        assert len(results) == 1

        # Query with date range that excludes the decision
        start_date = datetime.utcnow() - timedelta(days=10)
        end_date = datetime.utcnow() - timedelta(days=9)

        results = await logger.query_decisions(
            start_date=start_date,
            end_date=end_date,
        )

        assert len(results) == 0

        await logger.close()

    def test_batch_length_mismatch(self, sample_explanation):
        """Test batch logging with mismatched lengths."""
        logger = DecisionAuditLogger(
            database_url="sqlite+aiosqlite:///:memory:",
        )

        import asyncio

        async def test():
            with pytest.raises(ValueError):
                await logger.log_batch(
                    explanations=[sample_explanation],
                    event_type="test",
                    event_data_list=[{}, {}],  # Mismatched length
                )

        asyncio.run(test())
