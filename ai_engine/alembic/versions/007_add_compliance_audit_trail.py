"""Add compliance audit trail and assessment tables

Revision ID: 007
Revises: 006
Create Date: 2025-02-03 11:00:00.000000

This migration adds:
- Immutable compliance audit trail
- Compliance assessments and findings
- Remediation tracking
- Evidence collection
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create compliance audit trail and assessment tables."""

    # ═══════════════════════════════════════════════════════════════════
    # COMPLIANCE FRAMEWORKS
    # ═══════════════════════════════════════════════════════════════════
    op.create_table(
        "compliance_frameworks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False, unique=True),
        sa.Column("display_name", sa.String(255), nullable=False),
        sa.Column("version", sa.String(50), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("category", sa.String(50), nullable=False),
        sa.Column("industry", sa.String(50), nullable=True),
        sa.Column("jurisdiction", sa.String(100), nullable=True),
        sa.Column("total_controls", sa.Integer(), nullable=False, default=0),
        sa.Column("control_structure", postgresql.JSONB(), default={}),
        sa.Column("is_active", sa.Boolean(), nullable=False, default=True),
        sa.Column("effective_date", sa.Date(), nullable=True),
        sa.Column("sunset_date", sa.Date(), nullable=True),
        sa.Column("documentation_url", sa.Text(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), default={}),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )

    op.create_index("idx_cf_name", "compliance_frameworks", ["name"])
    op.create_index("idx_cf_category", "compliance_frameworks", ["category"])
    op.create_index("idx_cf_industry", "compliance_frameworks", ["industry"])

    # ═══════════════════════════════════════════════════════════════════
    # COMPLIANCE CONTROLS
    # ═══════════════════════════════════════════════════════════════════
    op.create_table(
        "compliance_controls",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("framework_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("control_id", sa.String(50), nullable=False),
        sa.Column("parent_control_id", sa.String(50), nullable=True),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("guidance", sa.Text(), nullable=True),
        sa.Column("category", sa.String(100), nullable=True),
        sa.Column("subcategory", sa.String(100), nullable=True),
        sa.Column("control_type", sa.String(50), nullable=False),
        sa.Column("implementation_level", sa.String(50), nullable=True),
        sa.Column("priority", sa.String(20), nullable=True),
        sa.Column("automation_level", sa.String(50), nullable=True),
        sa.Column("testing_frequency", sa.String(50), nullable=True),
        sa.Column("evidence_requirements", postgresql.JSONB(), default=[]),
        sa.Column("related_controls", postgresql.ARRAY(sa.String()), default=[]),
        sa.Column("tags", postgresql.ARRAY(sa.String()), default=[]),
        sa.Column("metadata", postgresql.JSONB(), default={}),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["framework_id"], ["compliance_frameworks.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("framework_id", "control_id", name="unique_framework_control"),
    )

    op.create_index("idx_cc_framework_id", "compliance_controls", ["framework_id"])
    op.create_index("idx_cc_control_id", "compliance_controls", ["control_id"])
    op.create_index("idx_cc_category", "compliance_controls", ["category"])
    op.create_index("idx_cc_tags", "compliance_controls", ["tags"], postgresql_using="gin")

    # ═══════════════════════════════════════════════════════════════════
    # COMPLIANCE ASSESSMENTS
    # ═══════════════════════════════════════════════════════════════════
    op.create_table(
        "compliance_assessments",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("framework_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("assessment_name", sa.String(255), nullable=False),
        sa.Column("assessment_type", sa.String(50), nullable=False),
        sa.Column("scope", sa.Text(), nullable=True),
        sa.Column("scope_systems", postgresql.ARRAY(sa.String()), default=[]),
        sa.Column("status", sa.String(50), nullable=False, default="draft"),
        sa.Column("overall_score", sa.DECIMAL(5, 2), nullable=True),
        sa.Column("controls_assessed", sa.Integer(), nullable=False, default=0),
        sa.Column("controls_passed", sa.Integer(), nullable=False, default=0),
        sa.Column("controls_failed", sa.Integer(), nullable=False, default=0),
        sa.Column("controls_not_applicable", sa.Integer(), nullable=False, default=0),
        sa.Column("critical_findings", sa.Integer(), nullable=False, default=0),
        sa.Column("high_findings", sa.Integer(), nullable=False, default=0),
        sa.Column("medium_findings", sa.Integer(), nullable=False, default=0),
        sa.Column("low_findings", sa.Integer(), nullable=False, default=0),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("due_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("assessor_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("reviewer_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("report_url", sa.Text(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), default={}),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["framework_id"], ["compliance_frameworks.id"]),
    )

    op.create_index("idx_ca_framework_id", "compliance_assessments", ["framework_id"])
    op.create_index("idx_ca_status", "compliance_assessments", ["status"])
    op.create_index("idx_ca_due_date", "compliance_assessments", ["due_date"])
    op.create_index("idx_ca_assessor_id", "compliance_assessments", ["assessor_id"])

    # ═══════════════════════════════════════════════════════════════════
    # COMPLIANCE FINDINGS
    # ═══════════════════════════════════════════════════════════════════
    op.create_table(
        "compliance_findings",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("assessment_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("control_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("finding_id", sa.String(50), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("severity", sa.String(20), nullable=False),
        sa.Column("status", sa.String(50), nullable=False, default="open"),
        sa.Column("compliance_status", sa.String(50), nullable=False),
        sa.Column("evidence_collected", sa.Boolean(), nullable=False, default=False),
        sa.Column("evidence_summary", sa.Text(), nullable=True),
        sa.Column("root_cause", sa.Text(), nullable=True),
        sa.Column("impact", sa.Text(), nullable=True),
        sa.Column("affected_systems", postgresql.ARRAY(sa.String()), default=[]),
        sa.Column("affected_assets", postgresql.JSONB(), default=[]),
        sa.Column("recommendation", sa.Text(), nullable=True),
        sa.Column("remediation_plan", sa.Text(), nullable=True),
        sa.Column("remediation_deadline", sa.DateTime(timezone=True), nullable=True),
        sa.Column("remediation_owner_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("remediation_status", sa.String(50), nullable=True),
        sa.Column("remediation_completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("verified_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("risk_score", sa.DECIMAL(5, 2), nullable=True),
        sa.Column("compensating_controls", sa.Text(), nullable=True),
        sa.Column("exception_approved", sa.Boolean(), nullable=False, default=False),
        sa.Column("exception_reason", sa.Text(), nullable=True),
        sa.Column("exception_approved_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("exception_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), default={}),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["assessment_id"], ["compliance_assessments.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["control_id"], ["compliance_controls.id"]),
    )

    op.create_index("idx_cfn_assessment_id", "compliance_findings", ["assessment_id"])
    op.create_index("idx_cfn_control_id", "compliance_findings", ["control_id"])
    op.create_index("idx_cfn_severity", "compliance_findings", ["severity"])
    op.create_index("idx_cfn_status", "compliance_findings", ["status"])
    op.create_index("idx_cfn_remediation_status", "compliance_findings", ["remediation_status"])
    op.create_index("idx_cfn_remediation_deadline", "compliance_findings", ["remediation_deadline"])

    # ═══════════════════════════════════════════════════════════════════
    # COMPLIANCE EVIDENCE
    # ═══════════════════════════════════════════════════════════════════
    op.create_table(
        "compliance_evidence",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("finding_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("control_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("assessment_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("evidence_type", sa.String(50), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("source_system", sa.String(255), nullable=True),
        sa.Column("source_type", sa.String(50), nullable=False),
        sa.Column("file_path", sa.Text(), nullable=True),
        sa.Column("file_hash", sa.String(64), nullable=True),
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("mime_type", sa.String(100), nullable=True),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("content_structured", postgresql.JSONB(), nullable=True),
        sa.Column("collected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("collected_by", sa.String(255), nullable=True),
        sa.Column("collection_method", sa.String(50), nullable=False),
        sa.Column("is_automated", sa.Boolean(), nullable=False, default=False),
        sa.Column("retention_period_days", sa.Integer(), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("tags", postgresql.ARRAY(sa.String()), default=[]),
        sa.Column("metadata", postgresql.JSONB(), default={}),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["finding_id"], ["compliance_findings.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["control_id"], ["compliance_controls.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["assessment_id"], ["compliance_assessments.id"], ondelete="SET NULL"),
    )

    op.create_index("idx_ce_finding_id", "compliance_evidence", ["finding_id"])
    op.create_index("idx_ce_control_id", "compliance_evidence", ["control_id"])
    op.create_index("idx_ce_assessment_id", "compliance_evidence", ["assessment_id"])
    op.create_index("idx_ce_evidence_type", "compliance_evidence", ["evidence_type"])
    op.create_index("idx_ce_collected_at", "compliance_evidence", ["collected_at"])
    op.create_index("idx_ce_tags", "compliance_evidence", ["tags"], postgresql_using="gin")

    # ═══════════════════════════════════════════════════════════════════
    # IMMUTABLE COMPLIANCE AUDIT TRAIL
    # ═══════════════════════════════════════════════════════════════════
    op.create_table(
        "compliance_audit_trail",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("event_type", sa.String(100), nullable=False),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("actor_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("actor_type", sa.String(50), nullable=False),
        sa.Column("actor_name", sa.String(255), nullable=True),
        sa.Column("actor_ip", postgresql.INET(), nullable=True),
        sa.Column("resource_type", sa.String(100), nullable=False),
        sa.Column("resource_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("resource_name", sa.String(500), nullable=True),
        sa.Column("framework_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("assessment_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("control_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("finding_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("previous_state", postgresql.JSONB(), nullable=True),
        sa.Column("new_state", postgresql.JSONB(), nullable=True),
        sa.Column("changes", postgresql.JSONB(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("request_id", sa.String(255), nullable=True),
        sa.Column("session_id", sa.String(255), nullable=True),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.Column("signature", sa.String(128), nullable=True),
        sa.Column("previous_hash", sa.String(64), nullable=True),
        sa.Column("hash", sa.String(64), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), default={}),
    )

    # Create indexes for audit trail
    op.create_index("idx_cat_timestamp", "compliance_audit_trail", ["timestamp"])
    op.create_index("idx_cat_event_type", "compliance_audit_trail", ["event_type"])
    op.create_index("idx_cat_action", "compliance_audit_trail", ["action"])
    op.create_index("idx_cat_actor_id", "compliance_audit_trail", ["actor_id"])
    op.create_index("idx_cat_resource", "compliance_audit_trail", ["resource_type", "resource_id"])
    op.create_index("idx_cat_framework_id", "compliance_audit_trail", ["framework_id"])
    op.create_index("idx_cat_assessment_id", "compliance_audit_trail", ["assessment_id"])
    op.create_index("idx_cat_hash", "compliance_audit_trail", ["hash"])

    # Create function to prevent updates/deletes on audit trail
    op.execute("""
        CREATE OR REPLACE FUNCTION prevent_audit_trail_modification()
        RETURNS TRIGGER AS $$
        BEGIN
            RAISE EXCEPTION 'Compliance audit trail is immutable. Modifications are not allowed.';
        END;
        $$ LANGUAGE plpgsql;
    """)

    # Create triggers to enforce immutability
    op.execute("""
        CREATE TRIGGER prevent_audit_update
        BEFORE UPDATE ON compliance_audit_trail
        FOR EACH ROW
        EXECUTE FUNCTION prevent_audit_trail_modification();
    """)

    op.execute("""
        CREATE TRIGGER prevent_audit_delete
        BEFORE DELETE ON compliance_audit_trail
        FOR EACH ROW
        EXECUTE FUNCTION prevent_audit_trail_modification();
    """)

    # Create function to compute hash chain
    op.execute("""
        CREATE OR REPLACE FUNCTION compute_audit_hash()
        RETURNS TRIGGER AS $$
        DECLARE
            prev_hash text;
            hash_input text;
        BEGIN
            -- Get the hash of the previous record
            SELECT hash INTO prev_hash
            FROM compliance_audit_trail
            ORDER BY timestamp DESC, id DESC
            LIMIT 1;

            NEW.previous_hash := COALESCE(prev_hash, 'genesis');

            -- Compute hash of current record
            hash_input := CONCAT(
                NEW.timestamp::text,
                NEW.event_type,
                NEW.action,
                COALESCE(NEW.actor_id::text, ''),
                NEW.resource_type,
                COALESCE(NEW.resource_id::text, ''),
                COALESCE(NEW.previous_hash, 'genesis')
            );

            NEW.hash := encode(sha256(hash_input::bytea), 'hex');

            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)

    op.execute("""
        CREATE TRIGGER compute_audit_hash_trigger
        BEFORE INSERT ON compliance_audit_trail
        FOR EACH ROW
        EXECUTE FUNCTION compute_audit_hash();
    """)

    # ═══════════════════════════════════════════════════════════════════
    # COMPLIANCE REPORTS
    # ═══════════════════════════════════════════════════════════════════
    op.create_table(
        "compliance_reports",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("assessment_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("framework_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("report_type", sa.String(50), nullable=False),
        sa.Column("report_name", sa.String(500), nullable=False),
        sa.Column("report_format", sa.String(20), nullable=False),
        sa.Column("status", sa.String(50), nullable=False, default="generating"),
        sa.Column("file_path", sa.Text(), nullable=True),
        sa.Column("file_hash", sa.String(64), nullable=True),
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("generation_time_ms", sa.Integer(), nullable=True),
        sa.Column("report_period_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("report_period_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("generated_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("parameters", postgresql.JSONB(), default={}),
        sa.Column("summary", postgresql.JSONB(), default={}),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["assessment_id"], ["compliance_assessments.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["framework_id"], ["compliance_frameworks.id"]),
    )

    op.create_index("idx_cr_assessment_id", "compliance_reports", ["assessment_id"])
    op.create_index("idx_cr_framework_id", "compliance_reports", ["framework_id"])
    op.create_index("idx_cr_report_type", "compliance_reports", ["report_type"])
    op.create_index("idx_cr_status", "compliance_reports", ["status"])
    op.create_index("idx_cr_created_at", "compliance_reports", ["created_at"])


def downgrade() -> None:
    """Drop compliance audit trail and assessment tables."""

    # Drop triggers first
    op.execute("DROP TRIGGER IF EXISTS compute_audit_hash_trigger ON compliance_audit_trail")
    op.execute("DROP TRIGGER IF EXISTS prevent_audit_delete ON compliance_audit_trail")
    op.execute("DROP TRIGGER IF EXISTS prevent_audit_update ON compliance_audit_trail")

    # Drop functions
    op.execute("DROP FUNCTION IF EXISTS compute_audit_hash()")
    op.execute("DROP FUNCTION IF EXISTS prevent_audit_trail_modification()")

    # Drop tables
    op.drop_table("compliance_reports")
    op.drop_table("compliance_audit_trail")
    op.drop_table("compliance_evidence")
    op.drop_table("compliance_findings")
    op.drop_table("compliance_assessments")
    op.drop_table("compliance_controls")
    op.drop_table("compliance_frameworks")
