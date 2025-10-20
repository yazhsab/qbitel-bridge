"""
CRONOS AI - Explainability Database Schema

Database models for storing AI decision audit trails and explanations.
"""

from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    CheckConstraint,
    Index,
    JSON,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import declarative_base
from sqlalchemy.types import TypeDecorator, CHAR
from sqlalchemy.dialects.postgresql import UUID as PostgreSQL_UUID
import uuid as uuid_lib

Base = declarative_base()


# Define JSONB type that falls back to JSON for SQLite
class JSONBType(TypeDecorator):
    """JSONB type for PostgreSQL, JSON for other databases."""

    impl = JSON

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB())
        else:
            return dialect.type_descriptor(JSON())


# Define UUID type that works with both PostgreSQL and SQLite
class GUID(TypeDecorator):
    """Platform-independent GUID type.
    Uses PostgreSQL's UUID type, otherwise uses CHAR(36), storing as stringified hex values.
    """

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PostgreSQL_UUID(as_uuid=True))
        else:
            return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == "postgresql":
            return value
        else:
            if isinstance(value, uuid_lib.UUID):
                return str(value)
            return str(uuid_lib.UUID(value))

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        elif isinstance(value, uuid_lib.UUID):
            return value
        else:
            return uuid_lib.UUID(value)


class AIDecisionAudit(Base):
    """
    Immutable audit trail of all AI decisions with explanations.

    This table satisfies regulatory requirements for AI transparency
    including EU AI Act, FDA 21 CFR Part 11, and SOC2 Type II.
    """

    __tablename__ = "ai_decision_audit"

    # Primary identification
    id = Column(GUID, primary_key=True, default=uuid4)
    decision_id = Column(String(255), nullable=False, unique=True, index=True)
    timestamp = Column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True
    )

    # Event context
    event_type = Column(String(50), nullable=False, index=True)
    event_data = Column(JSONBType, nullable=False)

    # Model information
    model_name = Column(String(100), nullable=False, index=True)
    model_version = Column(String(50), nullable=False)
    model_architecture = Column(String(50))  # 'cnn', 'lstm', 'llm', 'ensemble'

    # Decision details
    decision_output = Column(JSONBType, nullable=False)
    confidence_score = Column(
        Float,
        CheckConstraint(
            "confidence_score >= 0 AND confidence_score <= 1", name="valid_confidence"
        ),
        nullable=False,
    )

    # Explainability data
    explanation_method = Column(
        String(50), nullable=False
    )  # 'SHAP', 'LIME', 'RULE_BASED'
    explanation_id = Column(String(255), index=True)
    feature_importance = Column(JSONBType)  # Full feature importance data
    top_features = Column(JSONBType)  # Top N features for quick access
    decision_rationale = Column(Text)  # Human-readable explanation
    regulatory_justification = Column(Text)  # Compliance context
    counterfactual = Column(Text)  # "What if" explanation

    # Audit metadata
    user_id = Column(String(255), index=True)
    session_id = Column(String(255))
    compliance_framework = Column(
        String(50), index=True
    )  # 'SOC2', 'HIPAA', 'PCI-DSS', 'EU_AI_ACT'
    request_id = Column(String(255))  # For tracing

    # Validation and review
    human_reviewed = Column(Boolean, default=False, index=True)
    human_override = Column(Boolean, default=False)
    override_reason = Column(Text)
    reviewer_id = Column(String(255))
    review_timestamp = Column(DateTime(timezone=True))

    # Performance tracking
    inference_time_ms = Column(Float)  # Model inference time
    explanation_time_ms = Column(Float)  # Explanation generation time

    # Additional metadata
    additional_metadata = Column(JSONBType)  # Extensible metadata field

    # Composite indexes for common queries
    __table_args__ = (
        Index("idx_model_timestamp", "model_name", "timestamp"),
        Index("idx_compliance_timestamp", "compliance_framework", "timestamp"),
        Index("idx_human_review", "human_reviewed", "timestamp"),
        Index("idx_confidence_range", "confidence_score", "timestamp"),
    )

    def __repr__(self):
        return (
            f"<AIDecisionAudit(decision_id={self.decision_id}, "
            f"model={self.model_name}, confidence={self.confidence_score:.2f})>"
        )


class ModelDriftMetric(Base):
    """
    Track model performance drift over time.

    Stores periodic snapshots of model performance to detect degradation.
    """

    __tablename__ = "model_drift_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    timestamp = Column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True
    )

    # Model identification
    model_name = Column(String(100), nullable=False, index=True)
    model_version = Column(String(50), nullable=False)

    # Performance metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    average_confidence = Column(Float)

    # Distribution metrics
    prediction_distribution = Column(JSONBType)  # Distribution of predictions
    confidence_distribution = Column(JSONBType)  # Distribution of confidence scores

    # Drift detection
    drift_score = Column(Float)  # Overall drift score (0-1, higher = more drift)
    drift_detected = Column(Boolean, default=False, index=True)
    drift_details = Column(JSONBType)  # Details about detected drift

    # Baseline comparison
    baseline_date = Column(DateTime(timezone=True))
    comparison_window_hours = Column(Integer)  # Hours of data used for comparison
    sample_size = Column(Integer)  # Number of predictions in this metric

    # Alert status
    alert_triggered = Column(Boolean, default=False)
    alert_timestamp = Column(DateTime(timezone=True))

    __table_args__ = (
        Index("idx_drift_alert", "model_name", "drift_detected", "timestamp"),
    )

    def __repr__(self):
        return (
            f"<ModelDriftMetric(model={self.model_name}, "
            f"accuracy={self.accuracy:.3f}, drift={self.drift_score:.3f})>"
        )


class ExplanationCache(Base):
    """
    Cache frequently requested explanations to improve performance.

    Explanations are expensive to compute, so we cache them for
    similar inputs (based on content hash).
    """

    __tablename__ = "explanation_cache"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Cache key (hash of input + model)
    cache_key = Column(String(64), nullable=False, unique=True, index=True)

    # Model info
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)

    # Cached explanation
    explanation_data = Column(JSONBType, nullable=False)
    explanation_method = Column(String(50), nullable=False)

    # Cache metadata
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    last_accessed = Column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True
    )
    access_count = Column(Integer, default=1)

    # TTL
    expires_at = Column(DateTime(timezone=True), index=True)

    def __repr__(self):
        return f"<ExplanationCache(key={self.cache_key[:16]}..., accessed={self.access_count})>"
