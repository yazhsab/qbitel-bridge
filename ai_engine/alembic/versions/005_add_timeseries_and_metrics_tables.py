"""Add TimescaleDB time-series and metrics tables

Revision ID: 005
Revises: 004
Create Date: 2025-02-03 10:00:00.000000

This migration adds:
- Time-series tables for protocol metrics (TimescaleDB hypertables)
- Security event tracking
- System performance metrics
- Protocol discovery metrics
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '005'
down_revision = '004'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create time-series and metrics tables."""

    # Enable TimescaleDB extension (idempotent)
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")

    # ═══════════════════════════════════════════════════════════════════
    # PROTOCOL DISCOVERY METRICS
    # ═══════════════════════════════════════════════════════════════════
    op.create_table(
        'protocol_discovery_metrics',
        sa.Column('time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('protocol_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('protocol_name', sa.String(255), nullable=True),
        sa.Column('discovery_phase', sa.String(50), nullable=False),
        sa.Column('messages_processed', sa.BigInteger(), nullable=False, default=0),
        sa.Column('bytes_processed', sa.BigInteger(), nullable=False, default=0),
        sa.Column('fields_detected', sa.Integer(), nullable=False, default=0),
        sa.Column('confidence_score', sa.DECIMAL(5, 4), nullable=True),
        sa.Column('grammar_rules_generated', sa.Integer(), nullable=False, default=0),
        sa.Column('processing_time_ms', sa.Integer(), nullable=False),
        sa.Column('error_count', sa.Integer(), nullable=False, default=0),
        sa.Column('source_ip', postgresql.INET(), nullable=True),
        sa.Column('destination_ip', postgresql.INET(), nullable=True),
        sa.Column('port', sa.Integer(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(), default={}),
    )

    # Convert to TimescaleDB hypertable
    op.execute("""
        SELECT create_hypertable(
            'protocol_discovery_metrics',
            'time',
            chunk_time_interval => INTERVAL '1 day',
            if_not_exists => TRUE
        )
    """)

    # Create indexes for protocol discovery metrics
    op.create_index(
        'idx_pdm_protocol_time',
        'protocol_discovery_metrics',
        ['protocol_id', 'time'],
        postgresql_using='btree'
    )
    op.create_index(
        'idx_pdm_phase',
        'protocol_discovery_metrics',
        ['discovery_phase', 'time'],
        postgresql_using='btree'
    )

    # ═══════════════════════════════════════════════════════════════════
    # SECURITY EVENTS (ZERO-TOUCH)
    # ═══════════════════════════════════════════════════════════════════
    op.create_table(
        'security_events',
        sa.Column('time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('event_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('event_type', sa.String(100), nullable=False),
        sa.Column('severity', sa.String(20), nullable=False),
        sa.Column('confidence_score', sa.DECIMAL(5, 4), nullable=False),
        sa.Column('threat_type', sa.String(100), nullable=True),
        sa.Column('mitre_technique_id', sa.String(20), nullable=True),
        sa.Column('mitre_tactic', sa.String(100), nullable=True),
        sa.Column('source_ip', postgresql.INET(), nullable=True),
        sa.Column('destination_ip', postgresql.INET(), nullable=True),
        sa.Column('source_port', sa.Integer(), nullable=True),
        sa.Column('destination_port', sa.Integer(), nullable=True),
        sa.Column('protocol', sa.String(50), nullable=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('asset_id', sa.String(255), nullable=True),
        sa.Column('asset_type', sa.String(100), nullable=True),
        sa.Column('decision', sa.String(50), nullable=False),
        sa.Column('decision_reason', sa.Text(), nullable=True),
        sa.Column('action_taken', sa.String(100), nullable=True),
        sa.Column('action_success', sa.Boolean(), nullable=True),
        sa.Column('escalated', sa.Boolean(), nullable=False, default=False),
        sa.Column('escalated_to', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('response_time_ms', sa.Integer(), nullable=True),
        sa.Column('blast_radius', sa.Integer(), nullable=True),
        sa.Column('false_positive', sa.Boolean(), nullable=True),
        sa.Column('acknowledged_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('acknowledged_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('raw_event', postgresql.JSONB(), default={}),
        sa.Column('enrichment_data', postgresql.JSONB(), default={}),
    )

    # Convert to TimescaleDB hypertable
    op.execute("""
        SELECT create_hypertable(
            'security_events',
            'time',
            chunk_time_interval => INTERVAL '1 day',
            if_not_exists => TRUE
        )
    """)

    # Create indexes for security events
    op.create_index('idx_se_event_type_time', 'security_events', ['event_type', 'time'])
    op.create_index('idx_se_severity', 'security_events', ['severity', 'time'])
    op.create_index('idx_se_decision', 'security_events', ['decision', 'time'])
    op.create_index('idx_se_source_ip', 'security_events', ['source_ip', 'time'])
    op.create_index('idx_se_mitre', 'security_events', ['mitre_technique_id', 'time'])
    op.create_index('idx_se_asset', 'security_events', ['asset_id', 'time'])
    op.create_index('idx_se_unresolved', 'security_events', ['resolved_at'], postgresql_where=sa.text('resolved_at IS NULL'))

    # ═══════════════════════════════════════════════════════════════════
    # SYSTEM PERFORMANCE METRICS
    # ═══════════════════════════════════════════════════════════════════
    op.create_table(
        'system_metrics',
        sa.Column('time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('component', sa.String(100), nullable=False),
        sa.Column('instance_id', sa.String(255), nullable=False),
        sa.Column('metric_name', sa.String(100), nullable=False),
        sa.Column('metric_value', sa.DECIMAL(20, 6), nullable=False),
        sa.Column('metric_unit', sa.String(50), nullable=True),
        sa.Column('cpu_percent', sa.DECIMAL(5, 2), nullable=True),
        sa.Column('memory_mb', sa.Integer(), nullable=True),
        sa.Column('disk_io_read_mb', sa.DECIMAL(10, 2), nullable=True),
        sa.Column('disk_io_write_mb', sa.DECIMAL(10, 2), nullable=True),
        sa.Column('network_rx_mb', sa.DECIMAL(10, 2), nullable=True),
        sa.Column('network_tx_mb', sa.DECIMAL(10, 2), nullable=True),
        sa.Column('active_connections', sa.Integer(), nullable=True),
        sa.Column('request_count', sa.BigInteger(), nullable=True),
        sa.Column('error_count', sa.Integer(), nullable=True),
        sa.Column('latency_p50_ms', sa.DECIMAL(10, 2), nullable=True),
        sa.Column('latency_p95_ms', sa.DECIMAL(10, 2), nullable=True),
        sa.Column('latency_p99_ms', sa.DECIMAL(10, 2), nullable=True),
        sa.Column('labels', postgresql.JSONB(), default={}),
    )

    # Convert to TimescaleDB hypertable
    op.execute("""
        SELECT create_hypertable(
            'system_metrics',
            'time',
            chunk_time_interval => INTERVAL '1 hour',
            if_not_exists => TRUE
        )
    """)

    # Create indexes for system metrics
    op.create_index('idx_sm_component_time', 'system_metrics', ['component', 'time'])
    op.create_index('idx_sm_metric_name', 'system_metrics', ['metric_name', 'time'])
    op.create_index('idx_sm_instance', 'system_metrics', ['instance_id', 'time'])

    # ═══════════════════════════════════════════════════════════════════
    # PQC CRYPTO OPERATIONS METRICS
    # ═══════════════════════════════════════════════════════════════════
    op.create_table(
        'pqc_operations_metrics',
        sa.Column('time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('operation_type', sa.String(50), nullable=False),
        sa.Column('algorithm', sa.String(50), nullable=False),
        sa.Column('security_level', sa.Integer(), nullable=False),
        sa.Column('key_size_bytes', sa.Integer(), nullable=True),
        sa.Column('ciphertext_size_bytes', sa.Integer(), nullable=True),
        sa.Column('signature_size_bytes', sa.Integer(), nullable=True),
        sa.Column('operation_time_us', sa.Integer(), nullable=False),
        sa.Column('success', sa.Boolean(), nullable=False),
        sa.Column('error_code', sa.String(50), nullable=True),
        sa.Column('hsm_used', sa.Boolean(), nullable=False, default=False),
        sa.Column('hsm_id', sa.String(255), nullable=True),
        sa.Column('key_id', sa.String(255), nullable=True),
        sa.Column('domain', sa.String(50), nullable=True),
        sa.Column('protocol', sa.String(100), nullable=True),
    )

    # Convert to TimescaleDB hypertable
    op.execute("""
        SELECT create_hypertable(
            'pqc_operations_metrics',
            'time',
            chunk_time_interval => INTERVAL '1 day',
            if_not_exists => TRUE
        )
    """)

    # Create indexes for PQC metrics
    op.create_index('idx_pqc_algorithm_time', 'pqc_operations_metrics', ['algorithm', 'time'])
    op.create_index('idx_pqc_operation_type', 'pqc_operations_metrics', ['operation_type', 'time'])
    op.create_index('idx_pqc_domain', 'pqc_operations_metrics', ['domain', 'time'])

    # ═══════════════════════════════════════════════════════════════════
    # LLM INFERENCE METRICS
    # ═══════════════════════════════════════════════════════════════════
    op.create_table(
        'llm_inference_metrics',
        sa.Column('time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('request_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('provider', sa.String(50), nullable=False),
        sa.Column('model', sa.String(100), nullable=False),
        sa.Column('operation', sa.String(50), nullable=False),
        sa.Column('prompt_tokens', sa.Integer(), nullable=False),
        sa.Column('completion_tokens', sa.Integer(), nullable=False),
        sa.Column('total_tokens', sa.Integer(), nullable=False),
        sa.Column('latency_ms', sa.Integer(), nullable=False),
        sa.Column('time_to_first_token_ms', sa.Integer(), nullable=True),
        sa.Column('success', sa.Boolean(), nullable=False),
        sa.Column('error_code', sa.String(50), nullable=True),
        sa.Column('fallback_used', sa.Boolean(), nullable=False, default=False),
        sa.Column('fallback_provider', sa.String(50), nullable=True),
        sa.Column('cache_hit', sa.Boolean(), nullable=False, default=False),
        sa.Column('estimated_cost_usd', sa.DECIMAL(10, 6), nullable=True),
        sa.Column('temperature', sa.DECIMAL(3, 2), nullable=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
    )

    # Convert to TimescaleDB hypertable
    op.execute("""
        SELECT create_hypertable(
            'llm_inference_metrics',
            'time',
            chunk_time_interval => INTERVAL '1 day',
            if_not_exists => TRUE
        )
    """)

    # Create indexes for LLM metrics
    op.create_index('idx_llm_provider_time', 'llm_inference_metrics', ['provider', 'time'])
    op.create_index('idx_llm_model_time', 'llm_inference_metrics', ['model', 'time'])
    op.create_index('idx_llm_operation', 'llm_inference_metrics', ['operation', 'time'])

    # ═══════════════════════════════════════════════════════════════════
    # CREATE CONTINUOUS AGGREGATES FOR COMMON QUERIES
    # ═══════════════════════════════════════════════════════════════════

    # Hourly security event summary
    op.execute("""
        CREATE MATERIALIZED VIEW IF NOT EXISTS security_events_hourly
        WITH (timescaledb.continuous) AS
        SELECT
            time_bucket('1 hour', time) AS bucket,
            event_type,
            severity,
            decision,
            COUNT(*) as event_count,
            COUNT(*) FILTER (WHERE escalated = true) as escalated_count,
            AVG(response_time_ms) as avg_response_time_ms,
            COUNT(*) FILTER (WHERE false_positive = true) as false_positive_count
        FROM security_events
        GROUP BY bucket, event_type, severity, decision
        WITH NO DATA
    """)

    # Hourly PQC operations summary
    op.execute("""
        CREATE MATERIALIZED VIEW IF NOT EXISTS pqc_operations_hourly
        WITH (timescaledb.continuous) AS
        SELECT
            time_bucket('1 hour', time) AS bucket,
            algorithm,
            operation_type,
            COUNT(*) as operation_count,
            COUNT(*) FILTER (WHERE success = true) as success_count,
            AVG(operation_time_us) as avg_operation_time_us,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY operation_time_us) as p95_operation_time_us
        FROM pqc_operations_metrics
        GROUP BY bucket, algorithm, operation_type
        WITH NO DATA
    """)

    # Hourly LLM cost summary
    op.execute("""
        CREATE MATERIALIZED VIEW IF NOT EXISTS llm_cost_hourly
        WITH (timescaledb.continuous) AS
        SELECT
            time_bucket('1 hour', time) AS bucket,
            provider,
            model,
            COUNT(*) as request_count,
            SUM(total_tokens) as total_tokens,
            SUM(estimated_cost_usd) as total_cost_usd,
            AVG(latency_ms) as avg_latency_ms,
            COUNT(*) FILTER (WHERE cache_hit = true) as cache_hits
        FROM llm_inference_metrics
        GROUP BY bucket, provider, model
        WITH NO DATA
    """)

    # Set refresh policies for continuous aggregates
    op.execute("""
        SELECT add_continuous_aggregate_policy('security_events_hourly',
            start_offset => INTERVAL '3 hours',
            end_offset => INTERVAL '1 hour',
            schedule_interval => INTERVAL '1 hour',
            if_not_exists => TRUE
        )
    """)

    op.execute("""
        SELECT add_continuous_aggregate_policy('pqc_operations_hourly',
            start_offset => INTERVAL '3 hours',
            end_offset => INTERVAL '1 hour',
            schedule_interval => INTERVAL '1 hour',
            if_not_exists => TRUE
        )
    """)

    op.execute("""
        SELECT add_continuous_aggregate_policy('llm_cost_hourly',
            start_offset => INTERVAL '3 hours',
            end_offset => INTERVAL '1 hour',
            schedule_interval => INTERVAL '1 hour',
            if_not_exists => TRUE
        )
    """)

    # ═══════════════════════════════════════════════════════════════════
    # SET DATA RETENTION POLICIES
    # ═══════════════════════════════════════════════════════════════════

    # Keep raw data for 90 days, hourly aggregates for 2 years
    op.execute("""
        SELECT add_retention_policy('protocol_discovery_metrics', INTERVAL '90 days', if_not_exists => TRUE)
    """)
    op.execute("""
        SELECT add_retention_policy('security_events', INTERVAL '90 days', if_not_exists => TRUE)
    """)
    op.execute("""
        SELECT add_retention_policy('system_metrics', INTERVAL '30 days', if_not_exists => TRUE)
    """)
    op.execute("""
        SELECT add_retention_policy('pqc_operations_metrics', INTERVAL '90 days', if_not_exists => TRUE)
    """)
    op.execute("""
        SELECT add_retention_policy('llm_inference_metrics', INTERVAL '90 days', if_not_exists => TRUE)
    """)


def downgrade() -> None:
    """Drop time-series tables and continuous aggregates."""

    # Drop continuous aggregates first
    op.execute("DROP MATERIALIZED VIEW IF EXISTS llm_cost_hourly CASCADE")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS pqc_operations_hourly CASCADE")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS security_events_hourly CASCADE")

    # Drop tables
    op.drop_table('llm_inference_metrics')
    op.drop_table('pqc_operations_metrics')
    op.drop_table('system_metrics')
    op.drop_table('security_events')
    op.drop_table('protocol_discovery_metrics')
