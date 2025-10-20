"""Add AI Explainability tables

Revision ID: 002_explainability
Revises: 001_initial_auth_schema
Create Date: 2025-10-19 15:00:00.000000

This migration adds tables for AI explainability and audit trail:
- ai_decision_audit: Immutable log of all AI decisions with explanations
- model_drift_metrics: Track model performance over time
- explanation_cache: Cache expensive explanation computations
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = "002_explainability"
down_revision = "001_initial_auth_schema"
branch_labels = None
depends_on = None


def upgrade():
    """Add explainability tables."""

    # Create ai_decision_audit table
    op.create_table(
        "ai_decision_audit",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("decision_id", sa.String(255), nullable=False, unique=True),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        # Event context
        sa.Column("event_type", sa.String(50), nullable=False),
        sa.Column("event_data", postgresql.JSONB, nullable=False),
        # Model information
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("model_version", sa.String(50), nullable=False),
        sa.Column("model_architecture", sa.String(50)),
        # Decision details
        sa.Column("decision_output", postgresql.JSONB, nullable=False),
        sa.Column("confidence_score", sa.Float, nullable=False),
        # Explainability data
        sa.Column("explanation_method", sa.String(50), nullable=False),
        sa.Column("explanation_id", sa.String(255)),
        sa.Column("feature_importance", postgresql.JSONB),
        sa.Column("top_features", postgresql.JSONB),
        sa.Column("decision_rationale", sa.Text),
        sa.Column("regulatory_justification", sa.Text),
        sa.Column("counterfactual", sa.Text),
        # Audit metadata
        sa.Column("user_id", sa.String(255)),
        sa.Column("session_id", sa.String(255)),
        sa.Column("compliance_framework", sa.String(50)),
        sa.Column("request_id", sa.String(255)),
        # Validation
        sa.Column("human_reviewed", sa.Boolean, default=False),
        sa.Column("human_override", sa.Boolean, default=False),
        sa.Column("override_reason", sa.Text),
        sa.Column("reviewer_id", sa.String(255)),
        sa.Column("review_timestamp", sa.DateTime(timezone=True)),
        # Performance
        sa.Column("inference_time_ms", sa.Float),
        sa.Column("explanation_time_ms", sa.Float),
        # Metadata
        sa.Column("metadata", postgresql.JSONB),
        # Constraints
        sa.CheckConstraint(
            "confidence_score >= 0 AND confidence_score <= 1", name="valid_confidence"
        ),
    )

    # Create indexes for ai_decision_audit
    op.create_index("idx_decision_id", "ai_decision_audit", ["decision_id"])
    op.create_index("idx_timestamp", "ai_decision_audit", ["timestamp"])
    op.create_index("idx_event_type", "ai_decision_audit", ["event_type"])
    op.create_index("idx_model_name", "ai_decision_audit", ["model_name"])
    op.create_index("idx_explanation_id", "ai_decision_audit", ["explanation_id"])
    op.create_index("idx_user_id", "ai_decision_audit", ["user_id"])
    op.create_index(
        "idx_compliance_framework", "ai_decision_audit", ["compliance_framework"]
    )
    op.create_index("idx_human_reviewed", "ai_decision_audit", ["human_reviewed"])
    op.create_index(
        "idx_model_timestamp", "ai_decision_audit", ["model_name", "timestamp"]
    )
    op.create_index(
        "idx_compliance_timestamp",
        "ai_decision_audit",
        ["compliance_framework", "timestamp"],
    )
    op.create_index(
        "idx_human_review_timestamp",
        "ai_decision_audit",
        ["human_reviewed", "timestamp"],
    )
    op.create_index(
        "idx_confidence_range", "ai_decision_audit", ["confidence_score", "timestamp"]
    )

    # Create model_drift_metrics table
    op.create_table(
        "model_drift_metrics",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        # Model identification
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("model_version", sa.String(50), nullable=False),
        # Performance metrics
        sa.Column("accuracy", sa.Float),
        sa.Column("precision", sa.Float),
        sa.Column("recall", sa.Float),
        sa.Column("f1_score", sa.Float),
        sa.Column("average_confidence", sa.Float),
        # Distribution metrics
        sa.Column("prediction_distribution", postgresql.JSONB),
        sa.Column("confidence_distribution", postgresql.JSONB),
        # Drift detection
        sa.Column("drift_score", sa.Float),
        sa.Column("drift_detected", sa.Boolean, default=False),
        sa.Column("drift_details", postgresql.JSONB),
        # Baseline comparison
        sa.Column("baseline_date", sa.DateTime(timezone=True)),
        sa.Column("comparison_window_hours", sa.Integer),
        sa.Column("sample_size", sa.Integer),
        # Alert status
        sa.Column("alert_triggered", sa.Boolean, default=False),
        sa.Column("alert_timestamp", sa.DateTime(timezone=True)),
    )

    # Create indexes for model_drift_metrics
    op.create_index("idx_drift_timestamp", "model_drift_metrics", ["timestamp"])
    op.create_index("idx_drift_model_name", "model_drift_metrics", ["model_name"])
    op.create_index("idx_drift_detected", "model_drift_metrics", ["drift_detected"])
    op.create_index(
        "idx_drift_alert",
        "model_drift_metrics",
        ["model_name", "drift_detected", "timestamp"],
    )

    # Create explanation_cache table
    op.create_table(
        "explanation_cache",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("cache_key", sa.String(64), nullable=False, unique=True),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("model_version", sa.String(50), nullable=False),
        sa.Column("explanation_data", postgresql.JSONB, nullable=False),
        sa.Column("explanation_method", sa.String(50), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "last_accessed",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("access_count", sa.Integer, default=1),
        sa.Column("expires_at", sa.DateTime(timezone=True)),
    )

    # Create indexes for explanation_cache
    op.create_index("idx_cache_key", "explanation_cache", ["cache_key"], unique=True)
    op.create_index("idx_cache_last_accessed", "explanation_cache", ["last_accessed"])
    op.create_index("idx_cache_expires_at", "explanation_cache", ["expires_at"])


def downgrade():
    """Remove explainability tables."""

    # Drop indexes
    op.drop_index("idx_cache_expires_at", table_name="explanation_cache")
    op.drop_index("idx_cache_last_accessed", table_name="explanation_cache")
    op.drop_index("idx_cache_key", table_name="explanation_cache")

    op.drop_index("idx_drift_alert", table_name="model_drift_metrics")
    op.drop_index("idx_drift_detected", table_name="model_drift_metrics")
    op.drop_index("idx_drift_model_name", table_name="model_drift_metrics")
    op.drop_index("idx_drift_timestamp", table_name="model_drift_metrics")

    op.drop_index("idx_confidence_range", table_name="ai_decision_audit")
    op.drop_index("idx_human_review_timestamp", table_name="ai_decision_audit")
    op.drop_index("idx_compliance_timestamp", table_name="ai_decision_audit")
    op.drop_index("idx_model_timestamp", table_name="ai_decision_audit")
    op.drop_index("idx_human_reviewed", table_name="ai_decision_audit")
    op.drop_index("idx_compliance_framework", table_name="ai_decision_audit")
    op.drop_index("idx_user_id", table_name="ai_decision_audit")
    op.drop_index("idx_explanation_id", table_name="ai_decision_audit")
    op.drop_index("idx_model_name", table_name="ai_decision_audit")
    op.drop_index("idx_event_type", table_name="ai_decision_audit")
    op.drop_index("idx_timestamp", table_name="ai_decision_audit")
    op.drop_index("idx_decision_id", table_name="ai_decision_audit")

    # Drop tables
    op.drop_table("explanation_cache")
    op.drop_table("model_drift_metrics")
    op.drop_table("ai_decision_audit")
