"""
CRONOS AI - Protocol Marketplace Database Models

Database models for the Protocol Marketplace ecosystem including protocols,
users, installations, reviews, and transactions.
"""

import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum as PyEnum
from decimal import Decimal

from sqlalchemy import (
    Column,
    String,
    Integer,
    Boolean,
    DateTime,
    Text,
    ForeignKey,
    Index,
    UniqueConstraint,
    Enum,
    CheckConstraint,
    DECIMAL,
    BigInteger,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .database import Base


# Enumerations
class ProtocolCategory(str, PyEnum):
    """Protocol category enumeration."""
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    IOT = "iot"
    LEGACY = "legacy"
    MANUFACTURING = "manufacturing"
    TELECOM = "telecom"
    GAMING = "gaming"
    OTHER = "other"


class ProtocolType(str, PyEnum):
    """Protocol type enumeration."""
    BINARY = "binary"
    TEXT = "text"
    XML = "xml"
    JSON = "json"
    PROTOBUF = "protobuf"
    CUSTOM = "custom"


class SpecFormat(str, PyEnum):
    """Specification format enumeration."""
    YAML = "yaml"
    JSON = "json"
    PROTOBUF = "protobuf"
    XML = "xml"


class AuthorType(str, PyEnum):
    """Author type enumeration."""
    COMMUNITY = "community"
    VENDOR = "vendor"
    CRONOS = "cronos"
    ENTERPRISE = "enterprise"


class LicenseType(str, PyEnum):
    """License type enumeration."""
    FREE = "free"
    PAID = "paid"
    ENTERPRISE = "enterprise"
    TRIAL = "trial"


class PriceModel(str, PyEnum):
    """Price model enumeration."""
    ONE_TIME = "one_time"
    SUBSCRIPTION = "subscription"
    USAGE_BASED = "usage_based"
    TIERED = "tiered"


class CertificationStatus(str, PyEnum):
    """Certification status enumeration."""
    PENDING = "pending"
    IN_REVIEW = "in_review"
    CERTIFIED = "certified"
    REJECTED = "rejected"
    REVOKED = "revoked"


class ProtocolStatus(str, PyEnum):
    """Protocol status enumeration."""
    DRAFT = "draft"
    PENDING_VALIDATION = "pending_validation"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class UserType(str, PyEnum):
    """Marketplace user type enumeration."""
    INDIVIDUAL = "individual"
    VENDOR = "vendor"
    ENTERPRISE = "enterprise"


class InstallationStatus(str, PyEnum):
    """Installation status enumeration."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    EXPIRED = "expired"
    TRIAL = "trial"


class ReviewStatus(str, PyEnum):
    """Review status enumeration."""
    PENDING = "pending"
    PUBLISHED = "published"
    HIDDEN = "hidden"
    FLAGGED = "flagged"


# Database Models
class MarketplaceProtocol(Base):
    """
    Protocol definition in the marketplace.

    Stores comprehensive protocol metadata including technical specifications,
    licensing, quality metrics, and compatibility information.
    """
    __tablename__ = "marketplace_protocols"

    # Primary key
    protocol_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Basic information
    protocol_name = Column(String(255), unique=True, nullable=False, index=True)
    display_name = Column(String(255), nullable=False)
    short_description = Column(Text, nullable=False)
    long_description = Column(Text)

    # Categorization
    category = Column(Enum(ProtocolCategory), nullable=False, index=True)
    subcategory = Column(String(50))
    tags = Column(ARRAY(String), default=list, index=True)

    # Protocol metadata
    version = Column(String(50), nullable=False)
    protocol_type = Column(Enum(ProtocolType), nullable=False)
    industry = Column(String(50))

    # Technical specification
    spec_format = Column(Enum(SpecFormat), nullable=False)
    spec_file_url = Column(Text, nullable=False)
    parser_code_url = Column(Text)
    test_data_url = Column(Text)
    documentation_url = Column(Text)

    # Author/ownership
    author_id = Column(UUID(as_uuid=True), ForeignKey("marketplace_users.user_id"), nullable=False)
    author_type = Column(Enum(AuthorType), nullable=False)
    organization = Column(String(255))

    # Licensing
    license_type = Column(Enum(LicenseType), nullable=False)
    price_model = Column(Enum(PriceModel))
    base_price = Column(DECIMAL(10, 2))
    currency = Column(String(3), default="USD")

    # Quality metrics
    certification_status = Column(Enum(CertificationStatus), default=CertificationStatus.PENDING, index=True)
    certification_date = Column(DateTime(timezone=True))
    average_rating = Column(DECIMAL(3, 2), default=0.0, index=True)
    total_ratings = Column(Integer, default=0)
    download_count = Column(Integer, default=0, index=True)
    active_installations = Column(Integer, default=0)

    # Compatibility
    min_cronos_version = Column(String(50), nullable=False)
    supported_cronos_versions = Column(ARRAY(String), default=list)
    dependencies = Column(JSONB, default=dict)

    # Status
    status = Column(Enum(ProtocolStatus), default=ProtocolStatus.DRAFT, index=True)
    is_featured = Column(Boolean, default=False, index=True)
    is_official = Column(Boolean, default=False, index=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    published_at = Column(DateTime(timezone=True))
    deprecated_at = Column(DateTime(timezone=True))

    # Extended metadata (using protocol_metadata to avoid SQLAlchemy reserved word)
    protocol_metadata = Column("metadata", JSONB, default=dict)

    # Validation statistics
    validation_attempts = Column(Integer, default=0)
    last_validation_date = Column(DateTime(timezone=True))
    validation_results = Column(JSONB, default=dict)

    # Relationships
    author = relationship("MarketplaceUser", back_populates="protocols")
    installations = relationship("MarketplaceInstallation", back_populates="protocol", cascade="all, delete-orphan")
    reviews = relationship("MarketplaceReview", back_populates="protocol", cascade="all, delete-orphan")

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "(license_type = 'free' AND base_price IS NULL) OR (license_type != 'free' AND base_price > 0)",
            name="valid_price_constraint"
        ),
        Index("idx_marketplace_protocols_search", "protocol_name", "display_name", postgresql_using="gin"),
    )

    def __repr__(self):
        return f"<MarketplaceProtocol(protocol_id={self.protocol_id}, name='{self.protocol_name}', status='{self.status}')>"


class MarketplaceUser(Base):
    """
    Marketplace user/contributor.

    Represents protocol creators, vendors, and contributors with reputation
    tracking and payment information.
    """
    __tablename__ = "marketplace_users"

    # Primary key
    user_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Basic information
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    full_name = Column(String(255))

    # User type
    user_type = Column(Enum(UserType), nullable=False)
    organization = Column(String(255))

    # Verification
    is_verified = Column(Boolean, default=False, index=True)
    verification_date = Column(DateTime(timezone=True))
    verification_method = Column(String(50))

    # Reputation
    reputation_score = Column(Integer, default=0, index=True)
    total_contributions = Column(Integer, default=0)
    total_downloads = Column(Integer, default=0)

    # Financials
    stripe_account_id = Column(String(255))
    stripe_customer_id = Column(String(255))
    payout_enabled = Column(Boolean, default=False)
    total_revenue = Column(DECIMAL(12, 2), default=0.0)

    # Profile
    bio = Column(Text)
    website_url = Column(String(255))
    avatar_url = Column(String(255))
    github_username = Column(String(100))
    linkedin_url = Column(String(255))

    # Status
    status = Column(String(50), default="active")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    last_login_at = Column(DateTime(timezone=True))

    # Preferences
    email_notifications_enabled = Column(Boolean, default=True)
    public_profile = Column(Boolean, default=True)

    # Relationships
    protocols = relationship("MarketplaceProtocol", back_populates="author")
    reviews = relationship("MarketplaceReview", back_populates="customer")

    def __repr__(self):
        return f"<MarketplaceUser(user_id={self.user_id}, username='{self.username}', type='{self.user_type}')>"


class MarketplaceInstallation(Base):
    """
    Protocol installation tracking.

    Tracks installed protocols for customers including license information,
    usage statistics, and status.
    """
    __tablename__ = "marketplace_installations"

    # Primary key
    installation_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # References
    protocol_id = Column(UUID(as_uuid=True), ForeignKey("marketplace_protocols.protocol_id"), nullable=False, index=True)
    customer_id = Column(UUID(as_uuid=True), nullable=False, index=True)  # References main users table

    # Installation details
    installed_version = Column(String(50), nullable=False)
    installation_date = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_updated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # License
    license_key = Column(String(255), unique=True)
    license_type = Column(Enum(LicenseType), nullable=False)
    expires_at = Column(DateTime(timezone=True))

    # Subscription details
    subscription_id = Column(String(255))  # Stripe subscription ID
    next_billing_date = Column(DateTime(timezone=True))
    auto_renew = Column(Boolean, default=True)

    # Usage tracking
    total_packets_processed = Column(BigInteger, default=0)
    last_used_at = Column(DateTime(timezone=True))
    usage_limit = Column(BigInteger)  # For usage-based pricing

    # Status
    status = Column(Enum(InstallationStatus), default=InstallationStatus.ACTIVE, index=True)
    trial_ends_at = Column(DateTime(timezone=True))

    # Configuration
    environment = Column(String(50), default="production")  # production, staging, development
    configuration = Column(JSONB, default=dict)

    # Relationships
    protocol = relationship("MarketplaceProtocol", back_populates="installations")

    # Constraints
    __table_args__ = (
        UniqueConstraint("protocol_id", "customer_id", "environment", name="unique_protocol_customer_env"),
        Index("idx_installations_customer_status", "customer_id", "status"),
        Index("idx_installations_expiry", "expires_at"),
    )

    def __repr__(self):
        return f"<MarketplaceInstallation(installation_id={self.installation_id}, protocol_id={self.protocol_id}, status='{self.status}')>"


class MarketplaceReview(Base):
    """
    Protocol reviews and ratings.

    Customer reviews and ratings for marketplace protocols with verification
    and moderation support.
    """
    __tablename__ = "marketplace_reviews"

    # Primary key
    review_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # References
    protocol_id = Column(UUID(as_uuid=True), ForeignKey("marketplace_protocols.protocol_id"), nullable=False, index=True)
    customer_id = Column(UUID(as_uuid=True), ForeignKey("marketplace_users.user_id"), nullable=False, index=True)

    # Review content
    rating = Column(Integer, nullable=False)
    title = Column(String(255))
    review_text = Column(Text)

    # Helpful votes
    helpful_count = Column(Integer, default=0)
    unhelpful_count = Column(Integer, default=0)

    # Verification
    is_verified_purchase = Column(Boolean, default=False)
    installation_id = Column(UUID(as_uuid=True), ForeignKey("marketplace_installations.installation_id"))

    # Status
    status = Column(Enum(ReviewStatus), default=ReviewStatus.PUBLISHED, index=True)
    moderation_notes = Column(Text)
    moderated_by = Column(UUID(as_uuid=True))
    moderated_at = Column(DateTime(timezone=True))

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    protocol = relationship("MarketplaceProtocol", back_populates="reviews")
    customer = relationship("MarketplaceUser", back_populates="reviews")

    # Constraints
    __table_args__ = (
        CheckConstraint("rating BETWEEN 1 AND 5", name="valid_rating"),
        UniqueConstraint("protocol_id", "customer_id", name="one_review_per_customer"),
        Index("idx_reviews_protocol_rating", "protocol_id", "rating"),
        Index("idx_reviews_status_created", "status", "created_at"),
    )

    def __repr__(self):
        return f"<MarketplaceReview(review_id={self.review_id}, protocol_id={self.protocol_id}, rating={self.rating})>"


class MarketplaceTransaction(Base):
    """
    Transaction history for marketplace purchases.

    Tracks all financial transactions including purchases, subscriptions,
    and revenue sharing.
    """
    __tablename__ = "marketplace_transactions"

    # Primary key
    transaction_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # References
    protocol_id = Column(UUID(as_uuid=True), ForeignKey("marketplace_protocols.protocol_id"), nullable=False, index=True)
    customer_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    installation_id = Column(UUID(as_uuid=True), ForeignKey("marketplace_installations.installation_id"))

    # Transaction details
    transaction_type = Column(String(50), nullable=False)  # purchase, subscription, renewal, refund
    amount = Column(DECIMAL(10, 2), nullable=False)
    currency = Column(String(3), default="USD")

    # Payment provider details
    stripe_payment_intent_id = Column(String(255))
    stripe_charge_id = Column(String(255))
    payment_method = Column(String(50))

    # Revenue sharing
    platform_fee = Column(DECIMAL(10, 2))  # CRONOS AI fee (30%)
    creator_revenue = Column(DECIMAL(10, 2))  # Creator revenue (70%)
    payout_status = Column(String(50), default="pending")  # pending, paid, failed
    payout_date = Column(DateTime(timezone=True))

    # Status
    status = Column(String(50), default="pending", index=True)  # pending, completed, failed, refunded
    failure_reason = Column(Text)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    completed_at = Column(DateTime(timezone=True))

    # Extended metadata (using transaction_metadata to avoid SQLAlchemy reserved word)
    transaction_metadata = Column("metadata", JSONB, default=dict)

    # Indexes
    __table_args__ = (
        Index("idx_transactions_customer_created", "customer_id", "created_at"),
        Index("idx_transactions_protocol_created", "protocol_id", "created_at"),
        Index("idx_transactions_status", "status"),
    )

    def __repr__(self):
        return f"<MarketplaceTransaction(transaction_id={self.transaction_id}, amount={self.amount}, status='{self.status}')>"


class MarketplaceValidation(Base):
    """
    Protocol validation results.

    Stores automated and manual validation results for submitted protocols.
    """
    __tablename__ = "marketplace_validations"

    # Primary key
    validation_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Reference
    protocol_id = Column(UUID(as_uuid=True), ForeignKey("marketplace_protocols.protocol_id"), nullable=False, index=True)

    # Validation type
    validation_type = Column(String(50), nullable=False)  # automated, manual, security_scan, performance

    # Results
    status = Column(String(50), nullable=False, index=True)  # passed, failed, in_progress, skipped
    score = Column(DECIMAL(5, 2))  # 0-100

    # Details
    test_results = Column(JSONB, default=dict)
    errors = Column(JSONB, default=list)
    warnings = Column(JSONB, default=list)

    # Performance metrics
    throughput = Column(Integer)  # packets/sec
    memory_usage = Column(Integer)  # MB
    latency_p50 = Column(DECIMAL(10, 2))  # ms
    latency_p95 = Column(DECIMAL(10, 2))  # ms
    latency_p99 = Column(DECIMAL(10, 2))  # ms

    # Security scan results
    vulnerabilities_critical = Column(Integer, default=0)
    vulnerabilities_high = Column(Integer, default=0)
    vulnerabilities_medium = Column(Integer, default=0)
    vulnerabilities_low = Column(Integer, default=0)

    # Manual review
    reviewer_id = Column(UUID(as_uuid=True))
    reviewer_notes = Column(Text)

    # Timestamps
    started_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    completed_at = Column(DateTime(timezone=True))
    duration_seconds = Column(Integer)

    # Indexes
    __table_args__ = (
        Index("idx_validations_protocol_type", "protocol_id", "validation_type"),
        Index("idx_validations_status_started", "status", "started_at"),
    )

    def __repr__(self):
        return f"<MarketplaceValidation(validation_id={self.validation_id}, protocol_id={self.protocol_id}, status='{self.status}')>"
