"""
QBITEL - Marketplace API Schemas

Pydantic models for marketplace API request/response validation.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from decimal import Decimal
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field, validator, EmailStr, HttpUrl


# Enumerations
class ProtocolCategoryEnum(str, Enum):
    """Protocol category enumeration."""

    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    IOT = "iot"
    LEGACY = "legacy"
    MANUFACTURING = "manufacturing"
    TELECOM = "telecom"
    GAMING = "gaming"
    OTHER = "other"


class ProtocolTypeEnum(str, Enum):
    """Protocol type enumeration."""

    BINARY = "binary"
    TEXT = "text"
    XML = "xml"
    JSON = "json"
    PROTOBUF = "protobuf"
    CUSTOM = "custom"


class LicenseTypeEnum(str, Enum):
    """License type enumeration."""

    FREE = "free"
    PAID = "paid"
    ENTERPRISE = "enterprise"
    TRIAL = "trial"


class PriceModelEnum(str, Enum):
    """Price model enumeration."""

    ONE_TIME = "one_time"
    SUBSCRIPTION = "subscription"
    USAGE_BASED = "usage_based"
    TIERED = "tiered"


class CertificationStatusEnum(str, Enum):
    """Certification status enumeration."""

    PENDING = "pending"
    IN_REVIEW = "in_review"
    CERTIFIED = "certified"
    REJECTED = "rejected"


class ProtocolStatusEnum(str, Enum):
    """Protocol status enumeration."""

    DRAFT = "draft"
    PENDING_VALIDATION = "pending_validation"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"


class SortOrderEnum(str, Enum):
    """Sort order enumeration."""

    RATING = "rating"
    DOWNLOADS = "downloads"
    RECENT = "recent"
    ALPHABETICAL = "alphabetical"
    PRICE_LOW_HIGH = "price_low_high"
    PRICE_HIGH_LOW = "price_high_low"


# Request Schemas
class ProtocolSearchRequest(BaseModel):
    """Protocol search request parameters."""

    q: Optional[str] = Field(None, description="Search query for protocol name, description, or tags")
    category: Optional[ProtocolCategoryEnum] = Field(None, description="Filter by category")
    license_type: Optional[LicenseTypeEnum] = Field(None, description="Filter by license type")
    min_rating: Optional[float] = Field(None, ge=0.0, le=5.0, description="Minimum average rating (0-5)")
    sort: Optional[SortOrderEnum] = Field(SortOrderEnum.RATING, description="Sort order")
    page: int = Field(1, ge=1, description="Page number")
    limit: int = Field(20, ge=1, le=100, description="Results per page")
    is_featured: Optional[bool] = Field(None, description="Filter featured protocols only")
    is_official: Optional[bool] = Field(None, description="Filter official QBITEL protocols only")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")


class ProtocolSubmitRequest(BaseModel):
    """Protocol submission request."""

    protocol_name: str = Field(..., min_length=3, max_length=255, description="Unique protocol identifier")
    display_name: str = Field(..., min_length=3, max_length=255, description="Human-readable protocol name")
    short_description: str = Field(..., min_length=10, max_length=500, description="Brief protocol description")
    long_description: Optional[str] = Field(None, description="Detailed protocol description (markdown supported)")

    category: ProtocolCategoryEnum = Field(..., description="Protocol category")
    subcategory: Optional[str] = Field(None, max_length=50, description="Protocol subcategory")
    tags: List[str] = Field(default_factory=list, description="Protocol tags for searchability")

    version: str = Field(..., description="Protocol version (semantic versioning recommended)")
    protocol_type: ProtocolTypeEnum = Field(..., description="Protocol data format type")
    industry: Optional[str] = Field(None, max_length=50, description="Target industry")

    spec_format: str = Field(..., description="Specification format (yaml, json, protobuf, xml)")
    spec_file: str = Field(..., description="Base64-encoded protocol specification file")
    parser_code: Optional[str] = Field(None, description="Base64-encoded parser code (Python)")
    test_data: Optional[str] = Field(None, description="Base64-encoded test samples")

    license_type: LicenseTypeEnum = Field(..., description="License type")
    price_model: Optional[PriceModelEnum] = Field(None, description="Pricing model (required if not free)")
    base_price: Optional[Decimal] = Field(None, ge=0, description="Base price (required if not free)")

    min_qbitel_version: str = Field(..., description="Minimum compatible QBITEL version")
    dependencies: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Dependencies on other protocols or libraries"
    )

    @validator("base_price")
    def validate_price(cls, v, values):
        """Validate price is provided for paid licenses."""
        if values.get("license_type") != LicenseTypeEnum.FREE and v is None:
            raise ValueError("base_price is required for non-free licenses")
        if values.get("license_type") == LicenseTypeEnum.FREE and v is not None:
            raise ValueError("base_price must be None for free licenses")
        return v

    @validator("protocol_name")
    def validate_protocol_name(cls, v):
        """Validate protocol name format."""
        import re

        if not re.match(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$", v):
            raise ValueError("protocol_name must be lowercase alphanumeric with hyphens (kebab-case)")
        return v


class ProtocolPurchaseRequest(BaseModel):
    """Protocol purchase request."""

    license_type: LicenseTypeEnum = Field(..., description="License type to purchase")
    payment_method_id: str = Field(..., description="Stripe payment method ID")
    billing_email: EmailStr = Field(..., description="Billing email address")
    environment: str = Field("production", description="Deployment environment")


class ReviewSubmitRequest(BaseModel):
    """Review submission request."""

    rating: int = Field(..., ge=1, le=5, description="Rating (1-5 stars)")
    title: Optional[str] = Field(None, max_length=255, description="Review title")
    review_text: Optional[str] = Field(None, description="Review text")


# Response Schemas
class AuthorInfo(BaseModel):
    """Author/creator information."""

    user_id: UUID
    username: str
    full_name: Optional[str]
    organization: Optional[str]
    is_verified: bool
    reputation_score: int
    total_contributions: int
    avatar_url: Optional[str]

    class Config:
        from_attributes = True


class LicensingInfo(BaseModel):
    """Licensing information."""

    license_type: LicenseTypeEnum
    price_model: Optional[PriceModelEnum]
    base_price: Optional[Decimal]
    currency: str = "USD"
    refund_policy: Optional[str]

    class Config:
        from_attributes = True


class QualityMetrics(BaseModel):
    """Protocol quality metrics."""

    certification_status: CertificationStatusEnum
    certification_date: Optional[datetime]
    average_rating: Decimal
    total_ratings: int
    download_count: int
    active_installations: int

    class Config:
        from_attributes = True


class CompatibilityInfo(BaseModel):
    """Compatibility information."""

    min_qbitel_version: str
    supported_qbitel_versions: List[str]
    dependencies: Dict[str, Any]

    class Config:
        from_attributes = True


class TechnicalSpecs(BaseModel):
    """Technical specifications."""

    spec_format: str
    spec_file_url: str
    parser_code_url: Optional[str]
    test_data_url: Optional[str]
    documentation_url: Optional[str]

    class Config:
        from_attributes = True


class ProtocolSummary(BaseModel):
    """Protocol summary for list views."""

    protocol_id: UUID
    protocol_name: str
    display_name: str
    short_description: str
    category: ProtocolCategoryEnum
    tags: List[str]

    author: AuthorInfo

    license_type: LicenseTypeEnum
    base_price: Optional[Decimal]
    price_model: Optional[PriceModelEnum]

    average_rating: Decimal
    total_ratings: int
    download_count: int

    certification_status: CertificationStatusEnum
    is_official: bool
    is_featured: bool

    published_at: Optional[datetime]

    class Config:
        from_attributes = True


class ReviewSummary(BaseModel):
    """Review summary statistics."""

    five_star: int = Field(alias="5_star")
    four_star: int = Field(alias="4_star")
    three_star: int = Field(alias="3_star")
    two_star: int = Field(alias="2_star")
    one_star: int = Field(alias="1_star")
    recent_reviews: List[Dict[str, Any]] = Field(default_factory=list)

    class Config:
        populate_by_name = True


class ProtocolDetail(BaseModel):
    """Detailed protocol information."""

    protocol_id: UUID
    protocol_name: str
    display_name: str
    short_description: str
    long_description: Optional[str]

    category: ProtocolCategoryEnum
    subcategory: Optional[str]
    tags: List[str]

    version: str
    protocol_type: ProtocolTypeEnum
    industry: Optional[str]

    technical_specs: TechnicalSpecs
    author: AuthorInfo
    licensing: LicensingInfo
    quality_metrics: QualityMetrics
    compatibility: CompatibilityInfo

    reviews_summary: Optional[ReviewSummary]

    created_at: datetime
    published_at: Optional[datetime]
    updated_at: datetime

    class Config:
        from_attributes = True


class PaginationInfo(BaseModel):
    """Pagination information."""

    total: int
    page: int
    limit: int
    pages: int


class SearchFacets(BaseModel):
    """Search result facets."""

    categories: Dict[str, int]
    license_types: Dict[str, int]


class ProtocolSearchResponse(BaseModel):
    """Protocol search response."""

    protocols: List[ProtocolSummary]
    pagination: PaginationInfo
    facets: SearchFacets


class ValidationStep(BaseModel):
    """Validation step status."""

    step: str
    status: str
    message: str


class ValidationStatusResponse(BaseModel):
    """Protocol validation status response."""

    protocol_id: UUID
    validation_status: str
    steps: List[ValidationStep]
    estimated_completion: Optional[datetime]


class ProtocolSubmitResponse(BaseModel):
    """Protocol submission response."""

    protocol_id: UUID
    status: ProtocolStatusEnum
    message: str
    estimated_review_time: str
    validation_url: str


class InstallationInfo(BaseModel):
    """Installation information."""

    installation_id: UUID
    license_key: str
    license_type: LicenseTypeEnum
    expires_at: Optional[datetime]
    status: str
    download_urls: Dict[str, str]
    installation_instructions: str


class ProtocolPurchaseResponse(BaseModel):
    """Protocol purchase response."""

    transaction_id: UUID
    installation: InstallationInfo


class ReviewResponse(BaseModel):
    """Review response."""

    review_id: UUID
    protocol_id: UUID
    customer_id: UUID
    rating: int
    title: Optional[str]
    review_text: Optional[str]
    helpful_count: int
    is_verified_purchase: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: Optional[str]
    error_code: Optional[str]
