"""Add marketplace tables

Revision ID: 004
Revises: 003
Create Date: 2025-10-20 08:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '004'
down_revision = '003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create marketplace tables."""

    # Create marketplace_users table
    op.create_table(
        'marketplace_users',
        sa.Column('user_id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('email', sa.String(255), nullable=False, unique=True),
        sa.Column('username', sa.String(100), nullable=False, unique=True),
        sa.Column('full_name', sa.String(255)),
        sa.Column('user_type', sa.String(50), nullable=False),
        sa.Column('organization', sa.String(255)),
        sa.Column('is_verified', sa.Boolean(), default=False),
        sa.Column('verification_date', sa.DateTime(timezone=True)),
        sa.Column('verification_method', sa.String(50)),
        sa.Column('reputation_score', sa.Integer(), default=0),
        sa.Column('total_contributions', sa.Integer(), default=0),
        sa.Column('total_downloads', sa.Integer(), default=0),
        sa.Column('stripe_account_id', sa.String(255)),
        sa.Column('stripe_customer_id', sa.String(255)),
        sa.Column('payout_enabled', sa.Boolean(), default=False),
        sa.Column('total_revenue', sa.DECIMAL(12, 2), default=0.0),
        sa.Column('bio', sa.Text()),
        sa.Column('website_url', sa.String(255)),
        sa.Column('avatar_url', sa.String(255)),
        sa.Column('github_username', sa.String(100)),
        sa.Column('linkedin_url', sa.String(255)),
        sa.Column('status', sa.String(50), default='active'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('last_login_at', sa.DateTime(timezone=True)),
        sa.Column('email_notifications_enabled', sa.Boolean(), default=True),
        sa.Column('public_profile', sa.Boolean(), default=True),
    )

    # Create indexes for marketplace_users
    op.create_index('idx_marketplace_users_email', 'marketplace_users', ['email'])
    op.create_index('idx_marketplace_users_username', 'marketplace_users', ['username'])
    op.create_index('idx_marketplace_users_verified', 'marketplace_users', ['is_verified'])
    op.create_index('idx_marketplace_users_reputation', 'marketplace_users', ['reputation_score'])

    # Create marketplace_protocols table
    op.create_table(
        'marketplace_protocols',
        sa.Column('protocol_id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('protocol_name', sa.String(255), nullable=False, unique=True),
        sa.Column('display_name', sa.String(255), nullable=False),
        sa.Column('short_description', sa.Text(), nullable=False),
        sa.Column('long_description', sa.Text()),
        sa.Column('category', sa.String(50), nullable=False),
        sa.Column('subcategory', sa.String(50)),
        sa.Column('tags', postgresql.ARRAY(sa.String()), default=[]),
        sa.Column('version', sa.String(50), nullable=False),
        sa.Column('protocol_type', sa.String(50), nullable=False),
        sa.Column('industry', sa.String(50)),
        sa.Column('spec_format', sa.String(50), nullable=False),
        sa.Column('spec_file_url', sa.Text(), nullable=False),
        sa.Column('parser_code_url', sa.Text()),
        sa.Column('test_data_url', sa.Text()),
        sa.Column('documentation_url', sa.Text()),
        sa.Column('author_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('author_type', sa.String(50), nullable=False),
        sa.Column('organization', sa.String(255)),
        sa.Column('license_type', sa.String(50), nullable=False),
        sa.Column('price_model', sa.String(50)),
        sa.Column('base_price', sa.DECIMAL(10, 2)),
        sa.Column('currency', sa.String(3), default='USD'),
        sa.Column('certification_status', sa.String(50), default='pending'),
        sa.Column('certification_date', sa.DateTime(timezone=True)),
        sa.Column('average_rating', sa.DECIMAL(3, 2), default=0.0),
        sa.Column('total_ratings', sa.Integer(), default=0),
        sa.Column('download_count', sa.Integer(), default=0),
        sa.Column('active_installations', sa.Integer(), default=0),
        sa.Column('min_qbitel_version', sa.String(50), nullable=False),
        sa.Column('supported_qbitel_versions', postgresql.ARRAY(sa.String()), default=[]),
        sa.Column('dependencies', postgresql.JSONB(), default={}),
        sa.Column('status', sa.String(50), default='draft'),
        sa.Column('is_featured', sa.Boolean(), default=False),
        sa.Column('is_official', sa.Boolean(), default=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('published_at', sa.DateTime(timezone=True)),
        sa.Column('deprecated_at', sa.DateTime(timezone=True)),
        sa.Column('metadata', postgresql.JSONB(), default={}),
        sa.Column('validation_attempts', sa.Integer(), default=0),
        sa.Column('last_validation_date', sa.DateTime(timezone=True)),
        sa.Column('validation_results', postgresql.JSONB(), default={}),
        sa.ForeignKeyConstraint(['author_id'], ['marketplace_users.user_id']),
        sa.CheckConstraint(
            "(license_type = 'free' AND base_price IS NULL) OR (license_type != 'free' AND base_price > 0)",
            name='valid_price_constraint'
        ),
    )

    # Create indexes for marketplace_protocols
    op.create_index('idx_marketplace_protocols_name', 'marketplace_protocols', ['protocol_name'])
    op.create_index('idx_marketplace_protocols_category', 'marketplace_protocols', ['category'])
    op.create_index('idx_marketplace_protocols_status', 'marketplace_protocols', ['status'])
    op.create_index('idx_marketplace_protocols_certification', 'marketplace_protocols', ['certification_status'])
    op.create_index('idx_marketplace_protocols_rating', 'marketplace_protocols', ['average_rating'])
    op.create_index('idx_marketplace_protocols_downloads', 'marketplace_protocols', ['download_count'])
    op.create_index('idx_marketplace_protocols_featured', 'marketplace_protocols', ['is_featured'])
    op.create_index('idx_marketplace_protocols_official', 'marketplace_protocols', ['is_official'])
    op.create_index('idx_marketplace_protocols_tags', 'marketplace_protocols', ['tags'], postgresql_using='gin')

    # Create marketplace_installations table
    op.create_table(
        'marketplace_installations',
        sa.Column('installation_id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('protocol_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('customer_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('installed_version', sa.String(50), nullable=False),
        sa.Column('installation_date', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('last_updated', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('license_key', sa.String(255), unique=True),
        sa.Column('license_type', sa.String(50), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True)),
        sa.Column('subscription_id', sa.String(255)),
        sa.Column('next_billing_date', sa.DateTime(timezone=True)),
        sa.Column('auto_renew', sa.Boolean(), default=True),
        sa.Column('total_packets_processed', sa.BigInteger(), default=0),
        sa.Column('last_used_at', sa.DateTime(timezone=True)),
        sa.Column('usage_limit', sa.BigInteger()),
        sa.Column('status', sa.String(50), default='active'),
        sa.Column('trial_ends_at', sa.DateTime(timezone=True)),
        sa.Column('environment', sa.String(50), default='production'),
        sa.Column('configuration', postgresql.JSONB(), default={}),
        sa.ForeignKeyConstraint(['protocol_id'], ['marketplace_protocols.protocol_id']),
        sa.UniqueConstraint('protocol_id', 'customer_id', 'environment', name='unique_protocol_customer_env'),
    )

    # Create indexes for marketplace_installations
    op.create_index('idx_installations_protocol', 'marketplace_installations', ['protocol_id'])
    op.create_index('idx_installations_customer', 'marketplace_installations', ['customer_id'])
    op.create_index('idx_installations_status', 'marketplace_installations', ['status'])
    op.create_index('idx_installations_customer_status', 'marketplace_installations', ['customer_id', 'status'])
    op.create_index('idx_installations_expiry', 'marketplace_installations', ['expires_at'])

    # Create marketplace_reviews table
    op.create_table(
        'marketplace_reviews',
        sa.Column('review_id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('protocol_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('customer_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('rating', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(255)),
        sa.Column('review_text', sa.Text()),
        sa.Column('helpful_count', sa.Integer(), default=0),
        sa.Column('unhelpful_count', sa.Integer(), default=0),
        sa.Column('is_verified_purchase', sa.Boolean(), default=False),
        sa.Column('installation_id', postgresql.UUID(as_uuid=True)),
        sa.Column('status', sa.String(50), default='published'),
        sa.Column('moderation_notes', sa.Text()),
        sa.Column('moderated_by', postgresql.UUID(as_uuid=True)),
        sa.Column('moderated_at', sa.DateTime(timezone=True)),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['protocol_id'], ['marketplace_protocols.protocol_id']),
        sa.ForeignKeyConstraint(['customer_id'], ['marketplace_users.user_id']),
        sa.ForeignKeyConstraint(['installation_id'], ['marketplace_installations.installation_id']),
        sa.CheckConstraint('rating BETWEEN 1 AND 5', name='valid_rating'),
        sa.UniqueConstraint('protocol_id', 'customer_id', name='one_review_per_customer'),
    )

    # Create indexes for marketplace_reviews
    op.create_index('idx_reviews_protocol', 'marketplace_reviews', ['protocol_id'])
    op.create_index('idx_reviews_customer', 'marketplace_reviews', ['customer_id'])
    op.create_index('idx_reviews_status', 'marketplace_reviews', ['status'])
    op.create_index('idx_reviews_protocol_rating', 'marketplace_reviews', ['protocol_id', 'rating'])
    op.create_index('idx_reviews_status_created', 'marketplace_reviews', ['status', 'created_at'])

    # Create marketplace_transactions table
    op.create_table(
        'marketplace_transactions',
        sa.Column('transaction_id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('protocol_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('customer_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('installation_id', postgresql.UUID(as_uuid=True)),
        sa.Column('transaction_type', sa.String(50), nullable=False),
        sa.Column('amount', sa.DECIMAL(10, 2), nullable=False),
        sa.Column('currency', sa.String(3), default='USD'),
        sa.Column('stripe_payment_intent_id', sa.String(255)),
        sa.Column('stripe_charge_id', sa.String(255)),
        sa.Column('payment_method', sa.String(50)),
        sa.Column('platform_fee', sa.DECIMAL(10, 2)),
        sa.Column('creator_revenue', sa.DECIMAL(10, 2)),
        sa.Column('payout_status', sa.String(50), default='pending'),
        sa.Column('payout_date', sa.DateTime(timezone=True)),
        sa.Column('status', sa.String(50), default='pending'),
        sa.Column('failure_reason', sa.Text()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True)),
        sa.Column('metadata', postgresql.JSONB(), default={}),
        sa.ForeignKeyConstraint(['protocol_id'], ['marketplace_protocols.protocol_id']),
        sa.ForeignKeyConstraint(['installation_id'], ['marketplace_installations.installation_id']),
    )

    # Create indexes for marketplace_transactions
    op.create_index('idx_transactions_protocol', 'marketplace_transactions', ['protocol_id'])
    op.create_index('idx_transactions_customer', 'marketplace_transactions', ['customer_id'])
    op.create_index('idx_transactions_status', 'marketplace_transactions', ['status'])
    op.create_index('idx_transactions_customer_created', 'marketplace_transactions', ['customer_id', 'created_at'])
    op.create_index('idx_transactions_protocol_created', 'marketplace_transactions', ['protocol_id', 'created_at'])

    # Create marketplace_validations table
    op.create_table(
        'marketplace_validations',
        sa.Column('validation_id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('protocol_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('validation_type', sa.String(50), nullable=False),
        sa.Column('status', sa.String(50), nullable=False),
        sa.Column('score', sa.DECIMAL(5, 2)),
        sa.Column('test_results', postgresql.JSONB(), default={}),
        sa.Column('errors', postgresql.JSONB(), default=[]),
        sa.Column('warnings', postgresql.JSONB(), default=[]),
        sa.Column('throughput', sa.Integer()),
        sa.Column('memory_usage', sa.Integer()),
        sa.Column('latency_p50', sa.DECIMAL(10, 2)),
        sa.Column('latency_p95', sa.DECIMAL(10, 2)),
        sa.Column('latency_p99', sa.DECIMAL(10, 2)),
        sa.Column('vulnerabilities_critical', sa.Integer(), default=0),
        sa.Column('vulnerabilities_high', sa.Integer(), default=0),
        sa.Column('vulnerabilities_medium', sa.Integer(), default=0),
        sa.Column('vulnerabilities_low', sa.Integer(), default=0),
        sa.Column('reviewer_id', postgresql.UUID(as_uuid=True)),
        sa.Column('reviewer_notes', sa.Text()),
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True)),
        sa.Column('duration_seconds', sa.Integer()),
        sa.ForeignKeyConstraint(['protocol_id'], ['marketplace_protocols.protocol_id']),
    )

    # Create indexes for marketplace_validations
    op.create_index('idx_validations_protocol', 'marketplace_validations', ['protocol_id'])
    op.create_index('idx_validations_status', 'marketplace_validations', ['status'])
    op.create_index('idx_validations_protocol_type', 'marketplace_validations', ['protocol_id', 'validation_type'])
    op.create_index('idx_validations_status_started', 'marketplace_validations', ['status', 'started_at'])


def downgrade() -> None:
    """Drop marketplace tables."""
    op.drop_table('marketplace_validations')
    op.drop_table('marketplace_transactions')
    op.drop_table('marketplace_reviews')
    op.drop_table('marketplace_installations')
    op.drop_table('marketplace_protocols')
    op.drop_table('marketplace_users')
