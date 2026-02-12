"""Add model registry and versioning tables

Revision ID: 008
Revises: 007
Create Date: 2025-02-03 11:30:00.000000

This migration adds:
- ML model registry with versioning
- Model deployment tracking
- Model performance metrics
- A/B testing experiments
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "008"
down_revision = "007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create model registry and versioning tables."""

    # ═══════════════════════════════════════════════════════════════════
    # MODEL REGISTRY
    # ═══════════════════════════════════════════════════════════════════
    op.create_table(
        "ml_models",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("display_name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("model_type", sa.String(100), nullable=False),
        sa.Column("framework", sa.String(50), nullable=False),
        sa.Column("task", sa.String(100), nullable=False),
        sa.Column("domain", sa.String(50), nullable=True),
        sa.Column("owner_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("team", sa.String(100), nullable=True),
        sa.Column("tags", postgresql.ARRAY(sa.String()), default=[]),
        sa.Column("is_active", sa.Boolean(), nullable=False, default=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), default={}),
        sa.UniqueConstraint("name", name="unique_model_name"),
    )

    op.create_index("idx_mm_name", "ml_models", ["name"])
    op.create_index("idx_mm_model_type", "ml_models", ["model_type"])
    op.create_index("idx_mm_task", "ml_models", ["task"])
    op.create_index("idx_mm_domain", "ml_models", ["domain"])
    op.create_index("idx_mm_tags", "ml_models", ["tags"], postgresql_using="gin")

    # ═══════════════════════════════════════════════════════════════════
    # MODEL VERSIONS
    # ═══════════════════════════════════════════════════════════════════
    op.create_table(
        "ml_model_versions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("model_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version", sa.String(50), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(50), nullable=False, default="draft"),
        sa.Column("stage", sa.String(50), nullable=False, default="none"),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("artifact_path", sa.Text(), nullable=False),
        sa.Column("artifact_hash", sa.String(64), nullable=False),
        sa.Column("artifact_size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("framework_version", sa.String(50), nullable=True),
        sa.Column("python_version", sa.String(20), nullable=True),
        sa.Column("dependencies", postgresql.JSONB(), default={}),
        sa.Column("input_schema", postgresql.JSONB(), default={}),
        sa.Column("output_schema", postgresql.JSONB(), default={}),
        sa.Column("hyperparameters", postgresql.JSONB(), default={}),
        sa.Column("training_config", postgresql.JSONB(), default={}),
        sa.Column("training_dataset", sa.String(500), nullable=True),
        sa.Column("training_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("training_completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("training_duration_seconds", sa.Integer(), nullable=True),
        sa.Column("training_metrics", postgresql.JSONB(), default={}),
        sa.Column("validation_metrics", postgresql.JSONB(), default={}),
        sa.Column("test_metrics", postgresql.JSONB(), default={}),
        sa.Column("inference_latency_p50_ms", sa.DECIMAL(10, 2), nullable=True),
        sa.Column("inference_latency_p95_ms", sa.DECIMAL(10, 2), nullable=True),
        sa.Column("inference_latency_p99_ms", sa.DECIMAL(10, 2), nullable=True),
        sa.Column("throughput_per_second", sa.Integer(), nullable=True),
        sa.Column("memory_footprint_mb", sa.Integer(), nullable=True),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("approved_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("tags", postgresql.ARRAY(sa.String()), default=[]),
        sa.Column("metadata", postgresql.JSONB(), default={}),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["model_id"], ["ml_models.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("model_id", "version", name="unique_model_version"),
    )

    op.create_index("idx_mmv_model_id", "ml_model_versions", ["model_id"])
    op.create_index("idx_mmv_version", "ml_model_versions", ["version"])
    op.create_index("idx_mmv_status", "ml_model_versions", ["status"])
    op.create_index("idx_mmv_stage", "ml_model_versions", ["stage"])
    op.create_index("idx_mmv_artifact_hash", "ml_model_versions", ["artifact_hash"])

    # ═══════════════════════════════════════════════════════════════════
    # MODEL DEPLOYMENTS
    # ═══════════════════════════════════════════════════════════════════
    op.create_table(
        "ml_model_deployments",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("model_version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("deployment_name", sa.String(255), nullable=False),
        sa.Column("environment", sa.String(50), nullable=False),
        sa.Column("endpoint_url", sa.Text(), nullable=True),
        sa.Column("status", sa.String(50), nullable=False, default="pending"),
        sa.Column("deployment_type", sa.String(50), nullable=False),
        sa.Column("replicas", sa.Integer(), nullable=False, default=1),
        sa.Column("min_replicas", sa.Integer(), nullable=True),
        sa.Column("max_replicas", sa.Integer(), nullable=True),
        sa.Column("cpu_request", sa.String(20), nullable=True),
        sa.Column("cpu_limit", sa.String(20), nullable=True),
        sa.Column("memory_request", sa.String(20), nullable=True),
        sa.Column("memory_limit", sa.String(20), nullable=True),
        sa.Column("gpu_request", sa.String(20), nullable=True),
        sa.Column("autoscaling_config", postgresql.JSONB(), default={}),
        sa.Column("traffic_percentage", sa.Integer(), nullable=False, default=100),
        sa.Column("canary_config", postgresql.JSONB(), nullable=True),
        sa.Column("health_check_path", sa.String(255), nullable=True),
        sa.Column("deployed_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("deployed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_health_check", sa.DateTime(timezone=True), nullable=True),
        sa.Column("health_status", sa.String(50), nullable=True),
        sa.Column("rollback_version_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("terminated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("termination_reason", sa.Text(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), default={}),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["model_version_id"], ["ml_model_versions.id"]),
        sa.ForeignKeyConstraint(["rollback_version_id"], ["ml_model_versions.id"]),
    )

    op.create_index("idx_mmd_model_version_id", "ml_model_deployments", ["model_version_id"])
    op.create_index("idx_mmd_environment", "ml_model_deployments", ["environment"])
    op.create_index("idx_mmd_status", "ml_model_deployments", ["status"])
    op.create_index("idx_mmd_deployment_name", "ml_model_deployments", ["deployment_name"])

    # ═══════════════════════════════════════════════════════════════════
    # MODEL EXPERIMENTS (A/B TESTING)
    # ═══════════════════════════════════════════════════════════════════
    op.create_table(
        "ml_experiments",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("model_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("experiment_type", sa.String(50), nullable=False),
        sa.Column("status", sa.String(50), nullable=False, default="draft"),
        sa.Column("hypothesis", sa.Text(), nullable=True),
        sa.Column("success_criteria", postgresql.JSONB(), default={}),
        sa.Column("primary_metric", sa.String(100), nullable=False),
        sa.Column("secondary_metrics", postgresql.ARRAY(sa.String()), default=[]),
        sa.Column("control_version_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("treatment_version_ids", postgresql.ARRAY(postgresql.UUID(as_uuid=True)), default=[]),
        sa.Column("traffic_allocation", postgresql.JSONB(), default={}),
        sa.Column("target_sample_size", sa.Integer(), nullable=True),
        sa.Column("current_sample_size", sa.Integer(), nullable=False, default=0),
        sa.Column("confidence_level", sa.DECIMAL(5, 4), nullable=False, default=0.95),
        sa.Column("minimum_detectable_effect", sa.DECIMAL(5, 4), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("scheduled_end_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("results", postgresql.JSONB(), default={}),
        sa.Column("winner_version_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("statistical_significance", sa.DECIMAL(5, 4), nullable=True),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), default={}),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["model_id"], ["ml_models.id"]),
        sa.ForeignKeyConstraint(["control_version_id"], ["ml_model_versions.id"]),
        sa.ForeignKeyConstraint(["winner_version_id"], ["ml_model_versions.id"]),
    )

    op.create_index("idx_me_model_id", "ml_experiments", ["model_id"])
    op.create_index("idx_me_status", "ml_experiments", ["status"])
    op.create_index("idx_me_experiment_type", "ml_experiments", ["experiment_type"])

    # ═══════════════════════════════════════════════════════════════════
    # MODEL PREDICTIONS LOG (for monitoring and debugging)
    # ═══════════════════════════════════════════════════════════════════
    op.create_table(
        "ml_predictions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("model_version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("deployment_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("experiment_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("request_id", sa.String(255), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("input_hash", sa.String(64), nullable=True),
        sa.Column("input_features", postgresql.JSONB(), nullable=True),
        sa.Column("output", postgresql.JSONB(), nullable=False),
        sa.Column("confidence_score", sa.DECIMAL(5, 4), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("is_batch", sa.Boolean(), nullable=False, default=False),
        sa.Column("batch_size", sa.Integer(), nullable=True),
        sa.Column("ground_truth", postgresql.JSONB(), nullable=True),
        sa.Column("ground_truth_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_correct", sa.Boolean(), nullable=True),
        sa.Column("feedback", postgresql.JSONB(), nullable=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("session_id", sa.String(255), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), default={}),
        sa.ForeignKeyConstraint(["model_version_id"], ["ml_model_versions.id"]),
        sa.ForeignKeyConstraint(["deployment_id"], ["ml_model_deployments.id"]),
        sa.ForeignKeyConstraint(["experiment_id"], ["ml_experiments.id"]),
    )

    # Convert to TimescaleDB hypertable for efficient time-series queries
    op.execute("""
        SELECT create_hypertable(
            'ml_predictions',
            'timestamp',
            chunk_time_interval => INTERVAL '1 day',
            if_not_exists => TRUE
        )
    """)

    op.create_index("idx_mp_model_version_id", "ml_predictions", ["model_version_id", "timestamp"])
    op.create_index("idx_mp_deployment_id", "ml_predictions", ["deployment_id", "timestamp"])
    op.create_index("idx_mp_experiment_id", "ml_predictions", ["experiment_id", "timestamp"])
    op.create_index("idx_mp_request_id", "ml_predictions", ["request_id"])

    # Set retention policy
    op.execute("""
        SELECT add_retention_policy('ml_predictions', INTERVAL '90 days', if_not_exists => TRUE)
    """)

    # ═══════════════════════════════════════════════════════════════════
    # MODEL ALERTS
    # ═══════════════════════════════════════════════════════════════════
    op.create_table(
        "ml_model_alerts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("model_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("model_version_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("deployment_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("alert_type", sa.String(100), nullable=False),
        sa.Column("severity", sa.String(20), nullable=False),
        sa.Column("status", sa.String(50), nullable=False, default="active"),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("metric_name", sa.String(100), nullable=True),
        sa.Column("metric_value", sa.DECIMAL(20, 6), nullable=True),
        sa.Column("threshold_value", sa.DECIMAL(20, 6), nullable=True),
        sa.Column("condition", sa.String(20), nullable=True),
        sa.Column("triggered_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("acknowledged_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolved_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("resolution_notes", sa.Text(), nullable=True),
        sa.Column("notification_sent", sa.Boolean(), nullable=False, default=False),
        sa.Column("notification_channels", postgresql.ARRAY(sa.String()), default=[]),
        sa.Column("metadata", postgresql.JSONB(), default={}),
        sa.ForeignKeyConstraint(["model_id"], ["ml_models.id"]),
        sa.ForeignKeyConstraint(["model_version_id"], ["ml_model_versions.id"]),
        sa.ForeignKeyConstraint(["deployment_id"], ["ml_model_deployments.id"]),
    )

    op.create_index("idx_mma_model_id", "ml_model_alerts", ["model_id"])
    op.create_index("idx_mma_status", "ml_model_alerts", ["status"])
    op.create_index("idx_mma_severity", "ml_model_alerts", ["severity"])
    op.create_index("idx_mma_triggered_at", "ml_model_alerts", ["triggered_at"])


def downgrade() -> None:
    """Drop model registry and versioning tables."""

    op.drop_table("ml_model_alerts")
    op.drop_table("ml_predictions")
    op.drop_table("ml_experiments")
    op.drop_table("ml_model_deployments")
    op.drop_table("ml_model_versions")
    op.drop_table("ml_models")
