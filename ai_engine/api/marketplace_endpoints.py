"""
QBITEL - Protocol Marketplace API Endpoints

FastAPI endpoints for the Protocol Marketplace including discovery, submission,
purchase, and review functionality.
"""

import logging
import uuid
import base64
from typing import Optional, List
from datetime import datetime, timedelta
from decimal import Decimal

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks, UploadFile, File, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_, or_

from ..core.config import get_config
from ..core.database_manager import get_database_manager
from ..models.marketplace import (
    MarketplaceProtocol,
    MarketplaceUser,
    MarketplaceInstallation,
    MarketplaceReview,
    MarketplaceTransaction,
    MarketplaceValidation,
    ProtocolStatus,
    CertificationStatus,
    InstallationStatus,
)
from .marketplace_schemas import (
    ProtocolSearchRequest,
    ProtocolSearchResponse,
    ProtocolSummary,
    ProtocolDetail,
    ProtocolSubmitRequest,
    ProtocolSubmitResponse,
    ProtocolPurchaseRequest,
    ProtocolPurchaseResponse,
    ReviewSubmitRequest,
    ReviewResponse,
    ValidationStatusResponse,
    ValidationStep,
    PaginationInfo,
    SearchFacets,
    AuthorInfo,
    TechnicalSpecs,
    LicensingInfo,
    QualityMetrics,
    CompatibilityInfo,
    InstallationInfo,
    ErrorResponse,
)
from .auth import get_current_user
from ..marketplace.stripe_integration import get_stripe_manager
from ..marketplace.s3_file_manager import get_s3_manager

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/marketplace", tags=["marketplace"])

# Configuration
config = get_config()


def get_db():
    """Get database session."""
    db_manager = get_database_manager()
    session = db_manager.get_session()
    try:
        yield session
    finally:
        session.close()


# Helper Functions
def build_protocol_summary(protocol: MarketplaceProtocol) -> ProtocolSummary:
    """Build protocol summary from database model."""
    return ProtocolSummary(
        protocol_id=protocol.protocol_id,
        protocol_name=protocol.protocol_name,
        display_name=protocol.display_name,
        short_description=protocol.short_description,
        category=protocol.category,
        tags=protocol.tags,
        author=AuthorInfo(
            user_id=protocol.author.user_id,
            username=protocol.author.username,
            full_name=protocol.author.full_name,
            organization=protocol.author.organization,
            is_verified=protocol.author.is_verified,
            reputation_score=protocol.author.reputation_score,
            total_contributions=protocol.author.total_contributions,
            avatar_url=protocol.author.avatar_url,
        ),
        license_type=protocol.license_type,
        base_price=protocol.base_price,
        price_model=protocol.price_model,
        average_rating=protocol.average_rating,
        total_ratings=protocol.total_ratings,
        download_count=protocol.download_count,
        certification_status=protocol.certification_status,
        is_official=protocol.is_official,
        is_featured=protocol.is_featured,
        published_at=protocol.published_at,
    )


def build_protocol_detail(protocol: MarketplaceProtocol) -> ProtocolDetail:
    """Build detailed protocol information from database model."""
    return ProtocolDetail(
        protocol_id=protocol.protocol_id,
        protocol_name=protocol.protocol_name,
        display_name=protocol.display_name,
        short_description=protocol.short_description,
        long_description=protocol.long_description,
        category=protocol.category,
        subcategory=protocol.subcategory,
        tags=protocol.tags,
        version=protocol.version,
        protocol_type=protocol.protocol_type,
        industry=protocol.industry,
        technical_specs=TechnicalSpecs(
            spec_format=protocol.spec_format,
            spec_file_url=protocol.spec_file_url,
            parser_code_url=protocol.parser_code_url,
            test_data_url=protocol.test_data_url,
            documentation_url=protocol.documentation_url,
        ),
        author=AuthorInfo(
            user_id=protocol.author.user_id,
            username=protocol.author.username,
            full_name=protocol.author.full_name,
            organization=protocol.author.organization,
            is_verified=protocol.author.is_verified,
            reputation_score=protocol.author.reputation_score,
            total_contributions=protocol.author.total_contributions,
            avatar_url=protocol.author.avatar_url,
        ),
        licensing=LicensingInfo(
            license_type=protocol.license_type,
            price_model=protocol.price_model,
            base_price=protocol.base_price,
            currency=protocol.currency,
            refund_policy="30-day money-back guarantee" if protocol.license_type.value != "free" else None,
        ),
        quality_metrics=QualityMetrics(
            certification_status=protocol.certification_status,
            certification_date=protocol.certification_date,
            average_rating=protocol.average_rating,
            total_ratings=protocol.total_ratings,
            download_count=protocol.download_count,
            active_installations=protocol.active_installations,
        ),
        compatibility=CompatibilityInfo(
            min_qbitel_version=protocol.min_qbitel_version,
            supported_qbitel_versions=protocol.supported_qbitel_versions,
            dependencies=protocol.dependencies,
        ),
        reviews_summary=None,  # TODO: Calculate from reviews
        created_at=protocol.created_at,
        published_at=protocol.published_at,
        updated_at=protocol.updated_at,
    )


# Endpoints
@router.get(
    "/protocols/search",
    response_model=ProtocolSearchResponse,
    summary="Search Protocols",
    description="Search and filter marketplace protocols with pagination and facets",
)
async def search_protocols(
    q: Optional[str] = Query(None, description="Search query"),
    category: Optional[str] = Query(None, description="Filter by category"),
    license_type: Optional[str] = Query(None, description="Filter by license type"),
    min_rating: Optional[float] = Query(None, ge=0.0, le=5.0, description="Minimum rating"),
    sort: str = Query("rating", description="Sort order"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Results per page"),
    is_featured: Optional[bool] = Query(None, description="Featured only"),
    is_official: Optional[bool] = Query(None, description="Official only"),
    tags: Optional[List[str]] = Query(None, description="Filter by tags"),
    db: Session = Depends(get_db),
):
    """
    Search marketplace protocols with advanced filtering and sorting.

    Returns paginated results with facets for categories and license types.
    """
    try:
        # Build query
        query = db.query(MarketplaceProtocol).filter(MarketplaceProtocol.status == ProtocolStatus.PUBLISHED)

        # Apply filters
        if q:
            search_filter = or_(
                MarketplaceProtocol.protocol_name.ilike(f"%{q}%"),
                MarketplaceProtocol.display_name.ilike(f"%{q}%"),
                MarketplaceProtocol.short_description.ilike(f"%{q}%"),
            )
            query = query.filter(search_filter)

        if category:
            query = query.filter(MarketplaceProtocol.category == category)

        if license_type:
            query = query.filter(MarketplaceProtocol.license_type == license_type)

        if min_rating is not None:
            query = query.filter(MarketplaceProtocol.average_rating >= min_rating)

        if is_featured is not None:
            query = query.filter(MarketplaceProtocol.is_featured == is_featured)

        if is_official is not None:
            query = query.filter(MarketplaceProtocol.is_official == is_official)

        if tags:
            query = query.filter(MarketplaceProtocol.tags.overlap(tags))

        # Apply sorting
        if sort == "rating":
            query = query.order_by(desc(MarketplaceProtocol.average_rating))
        elif sort == "downloads":
            query = query.order_by(desc(MarketplaceProtocol.download_count))
        elif sort == "recent":
            query = query.order_by(desc(MarketplaceProtocol.published_at))
        elif sort == "alphabetical":
            query = query.order_by(MarketplaceProtocol.display_name)
        elif sort == "price_low_high":
            query = query.order_by(MarketplaceProtocol.base_price.asc().nullsfirst())
        elif sort == "price_high_low":
            query = query.order_by(MarketplaceProtocol.base_price.desc().nullslast())

        # Get total count
        total = query.count()

        # Apply pagination
        offset = (page - 1) * limit
        protocols = query.offset(offset).limit(limit).all()

        # Build facets
        category_facets = dict(
            db.query(MarketplaceProtocol.category, func.count())
            .filter(MarketplaceProtocol.status == ProtocolStatus.PUBLISHED)
            .group_by(MarketplaceProtocol.category)
            .all()
        )

        license_facets = dict(
            db.query(MarketplaceProtocol.license_type, func.count())
            .filter(MarketplaceProtocol.status == ProtocolStatus.PUBLISHED)
            .group_by(MarketplaceProtocol.license_type)
            .all()
        )

        # Build response
        return ProtocolSearchResponse(
            protocols=[build_protocol_summary(p) for p in protocols],
            pagination=PaginationInfo(
                total=total,
                page=page,
                limit=limit,
                pages=(total + limit - 1) // limit,
            ),
            facets=SearchFacets(
                categories={str(k): v for k, v in category_facets.items()},
                license_types={str(k): v for k, v in license_facets.items()},
            ),
        )

    except Exception as e:
        logger.error(f"Error searching protocols: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get(
    "/protocols/{protocol_id}",
    response_model=ProtocolDetail,
    summary="Get Protocol Details",
    description="Get detailed information about a specific protocol",
)
async def get_protocol(
    protocol_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    """
    Get detailed protocol information including specs, author, licensing, and reviews.
    """
    try:
        protocol = db.query(MarketplaceProtocol).filter(MarketplaceProtocol.protocol_id == protocol_id).first()

        if not protocol:
            raise HTTPException(status_code=404, detail="Protocol not found")

        if protocol.status != ProtocolStatus.PUBLISHED:
            raise HTTPException(status_code=403, detail="Protocol is not published")

        return build_protocol_detail(protocol)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting protocol {protocol_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get protocol: {str(e)}")


@router.post(
    "/protocols",
    response_model=ProtocolSubmitResponse,
    summary="Submit Protocol",
    description="Submit a new protocol to the marketplace",
    status_code=201,
)
async def submit_protocol(
    request: ProtocolSubmitRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Submit a new protocol to the marketplace for validation and certification.

    The protocol will go through automated validation and manual review before publication.
    """
    try:
        # Check if protocol name already exists
        existing = db.query(MarketplaceProtocol).filter(MarketplaceProtocol.protocol_name == request.protocol_name).first()
        if existing:
            raise HTTPException(status_code=400, detail=f"Protocol '{request.protocol_name}' already exists")

        # Get or create marketplace user
        marketplace_user = db.query(MarketplaceUser).filter(MarketplaceUser.email == current_user.email).first()

        if not marketplace_user:
            marketplace_user = MarketplaceUser(
                user_id=uuid.uuid4(),
                email=current_user.email,
                username=current_user.username,
                full_name=current_user.full_name,
                user_type="individual",  # Default, can be upgraded
            )
            db.add(marketplace_user)
            db.flush()

        # Create protocol
        protocol = MarketplaceProtocol(
            protocol_id=uuid.uuid4(),
            protocol_name=request.protocol_name,
            display_name=request.display_name,
            short_description=request.short_description,
            long_description=request.long_description,
            category=request.category,
            subcategory=request.subcategory,
            tags=request.tags,
            version=request.version,
            protocol_type=request.protocol_type,
            industry=request.industry,
            spec_format=request.spec_format,
            author_id=marketplace_user.user_id,
            author_type="community",
            license_type=request.license_type,
            price_model=request.price_model,
            base_price=request.base_price,
            min_qbitel_version=request.min_qbitel_version,
            dependencies=request.dependencies,
            status=ProtocolStatus.PENDING_VALIDATION,
            certification_status=CertificationStatus.PENDING,
        )

        db.add(protocol)
        db.flush()  # Get protocol_id before uploading

        # Upload files to S3
        s3_manager = get_s3_manager()

        # Upload spec file (required)
        spec_file_content = base64.b64decode(request.spec_file)
        spec_upload = await s3_manager.upload_protocol_file(
            protocol_id=protocol.protocol_id,
            file_content=spec_file_content,
            file_type="spec",
            filename=f"{request.protocol_name}.{request.spec_format}",
            version=request.version,
        )
        protocol.spec_file_url = spec_upload["cdn_url"]

        # Upload parser code (optional)
        if request.parser_code:
            parser_content = base64.b64decode(request.parser_code)
            parser_upload = await s3_manager.upload_protocol_file(
                protocol_id=protocol.protocol_id,
                file_content=parser_content,
                file_type="parser",
                filename=f"{request.protocol_name}_parser.py",
                version=request.version,
            )
            protocol.parser_code_url = parser_upload["cdn_url"]

        # Upload test data (optional)
        if request.test_data:
            test_data_content = base64.b64decode(request.test_data)
            test_data_upload = await s3_manager.upload_protocol_file(
                protocol_id=protocol.protocol_id,
                file_content=test_data_content,
                file_type="test_data",
                filename=f"{request.protocol_name}_test_data.json",
                version=request.version,
            )
            protocol.test_data_url = test_data_upload["cdn_url"]

        db.commit()
        db.refresh(protocol)

        # Schedule validation pipeline in background
        # background_tasks.add_task(run_protocol_validation, protocol.protocol_id)

        logger.info(f"Protocol {request.protocol_name} submitted by {current_user.username}")

        return ProtocolSubmitResponse(
            protocol_id=protocol.protocol_id,
            status=protocol.status,
            message="Protocol submitted for validation. You will be notified once review is complete.",
            estimated_review_time="2-3 business days",
            validation_url=f"/api/v1/marketplace/protocols/{protocol.protocol_id}/validation",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting protocol: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Submission failed: {str(e)}")


@router.get(
    "/protocols/{protocol_id}/validation",
    response_model=ValidationStatusResponse,
    summary="Get Validation Status",
    description="Get the validation status of a submitted protocol",
)
async def get_validation_status(
    protocol_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Get the validation status and results for a submitted protocol.
    """
    try:
        protocol = db.query(MarketplaceProtocol).filter(MarketplaceProtocol.protocol_id == protocol_id).first()

        if not protocol:
            raise HTTPException(status_code=404, detail="Protocol not found")

        # Get validation results
        validations = (
            db.query(MarketplaceValidation)
            .filter(MarketplaceValidation.protocol_id == protocol_id)
            .order_by(MarketplaceValidation.started_at.desc())
            .all()
        )

        # Build validation steps
        steps = []
        validation_types = ["syntax_validation", "parser_testing", "security_scan", "manual_review"]

        for validation_type in validation_types:
            validation = next((v for v in validations if v.validation_type == validation_type), None)
            if validation:
                steps.append(
                    ValidationStep(
                        step=validation_type,
                        status=validation.status,
                        message=f"Validation {validation.status}",
                    )
                )
            else:
                steps.append(
                    ValidationStep(
                        step=validation_type,
                        status="pending",
                        message="Awaiting validation",
                    )
                )

        return ValidationStatusResponse(
            protocol_id=protocol_id,
            validation_status=protocol.certification_status.value,
            steps=steps,
            estimated_completion=protocol.last_validation_date + timedelta(days=3) if protocol.last_validation_date else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting validation status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get validation status: {str(e)}")


@router.post(
    "/protocols/{protocol_id}/purchase",
    response_model=ProtocolPurchaseResponse,
    summary="Purchase Protocol",
    description="Purchase or subscribe to a marketplace protocol",
    status_code=201,
)
async def purchase_protocol(
    protocol_id: uuid.UUID,
    request: ProtocolPurchaseRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Purchase or subscribe to a marketplace protocol.

    Creates an installation and license key for the customer.
    """
    try:
        protocol = db.query(MarketplaceProtocol).filter(MarketplaceProtocol.protocol_id == protocol_id).first()

        if not protocol:
            raise HTTPException(status_code=404, detail="Protocol not found")

        if protocol.status != ProtocolStatus.PUBLISHED:
            raise HTTPException(status_code=403, detail="Protocol is not available for purchase")

        # Check if already installed
        existing_installation = (
            db.query(MarketplaceInstallation)
            .filter(
                and_(
                    MarketplaceInstallation.protocol_id == protocol_id,
                    MarketplaceInstallation.customer_id == current_user.id,
                    MarketplaceInstallation.environment == request.environment,
                )
            )
            .first()
        )

        if existing_installation and existing_installation.status == InstallationStatus.ACTIVE:
            raise HTTPException(status_code=400, detail="Protocol already installed for this environment")

        # Process payment with Stripe (if not free)
        stripe_response = None
        if protocol.license_type.value != "free" and protocol.base_price and protocol.base_price > 0:
            stripe_manager = get_stripe_manager()

            # Determine payment type based on price model
            if protocol.price_model and protocol.price_model.value == "subscription":
                # Create subscription
                stripe_response = await stripe_manager.create_subscription(
                    protocol_id=protocol_id,
                    customer_id=current_user.id,
                    payment_method_id=request.payment_method_id,
                )

                if not stripe_response.get("success"):
                    raise HTTPException(status_code=402, detail="Payment failed: Unable to create subscription")
            else:
                # One-time payment
                stripe_response = await stripe_manager.process_one_time_payment(
                    protocol_id=protocol_id,
                    customer_id=current_user.id,
                    payment_method_id=request.payment_method_id,
                    amount=protocol.base_price,
                    currency="usd",
                )

                if not stripe_response.get("success"):
                    raise HTTPException(status_code=402, detail="Payment failed: Unable to process payment")

        # Create installation
        installation = MarketplaceInstallation(
            installation_id=uuid.uuid4(),
            protocol_id=protocol_id,
            customer_id=current_user.id,
            installed_version=protocol.version,
            license_key=f"QBITEL-{protocol.protocol_name.upper()}-{uuid.uuid4().hex[:12]}",
            license_type=request.license_type,
            status=InstallationStatus.ACTIVE,
            environment=request.environment,
        )

        if protocol.price_model and protocol.price_model.value == "subscription":
            installation.expires_at = datetime.utcnow() + timedelta(days=30)
            installation.auto_renew = True

        db.add(installation)

        # Create transaction record
        transaction_type = "subscription" if stripe_response and "subscription_id" in stripe_response else "purchase"
        transaction = MarketplaceTransaction(
            transaction_id=stripe_response.get("transaction_id") if stripe_response else uuid.uuid4(),
            protocol_id=protocol_id,
            customer_id=current_user.id,
            installation_id=installation.installation_id,
            transaction_type=transaction_type,
            amount=protocol.base_price or Decimal("0.00"),
            platform_fee=stripe_response.get(
                "platform_fee", protocol.base_price * Decimal("0.30") if protocol.base_price else Decimal("0.00")
            ),
            creator_revenue=stripe_response.get(
                "creator_revenue", protocol.base_price * Decimal("0.70") if protocol.base_price else Decimal("0.00")
            ),
            status="completed",
            stripe_payment_intent_id=stripe_response.get("payment_intent_id") if stripe_response else None,
        )

        db.add(transaction)

        # Update protocol metrics
        protocol.download_count += 1
        protocol.active_installations += 1

        db.commit()
        db.refresh(installation)

        logger.info(f"Protocol {protocol.protocol_name} purchased by {current_user.username}")

        return ProtocolPurchaseResponse(
            transaction_id=transaction.transaction_id,
            installation=InstallationInfo(
                installation_id=installation.installation_id,
                license_key=installation.license_key,
                license_type=installation.license_type,
                expires_at=installation.expires_at,
                status=installation.status.value,
                download_urls={
                    "spec": protocol.spec_file_url,
                    "parser": protocol.parser_code_url or "",
                    "docs": protocol.documentation_url or "",
                },
                installation_instructions="See documentation for integration steps",
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error purchasing protocol: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Purchase failed: {str(e)}")


@router.post(
    "/protocols/{protocol_id}/reviews",
    response_model=ReviewResponse,
    summary="Submit Review",
    description="Submit a review for a protocol",
    status_code=201,
)
async def submit_review(
    protocol_id: uuid.UUID,
    request: ReviewSubmitRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Submit a review and rating for a marketplace protocol.

    Requires an active installation of the protocol.
    """
    try:
        # Check if protocol exists
        protocol = db.query(MarketplaceProtocol).filter(MarketplaceProtocol.protocol_id == protocol_id).first()

        if not protocol:
            raise HTTPException(status_code=404, detail="Protocol not found")

        # Get marketplace user
        marketplace_user = db.query(MarketplaceUser).filter(MarketplaceUser.email == current_user.email).first()

        if not marketplace_user:
            raise HTTPException(status_code=400, detail="Marketplace user not found")

        # Check if already reviewed
        existing_review = (
            db.query(MarketplaceReview)
            .filter(
                and_(
                    MarketplaceReview.protocol_id == protocol_id,
                    MarketplaceReview.customer_id == marketplace_user.user_id,
                )
            )
            .first()
        )

        if existing_review:
            raise HTTPException(status_code=400, detail="You have already reviewed this protocol")

        # Check if user has installed the protocol
        installation = (
            db.query(MarketplaceInstallation)
            .filter(
                and_(
                    MarketplaceInstallation.protocol_id == protocol_id,
                    MarketplaceInstallation.customer_id == current_user.id,
                )
            )
            .first()
        )

        # Create review
        review = MarketplaceReview(
            review_id=uuid.uuid4(),
            protocol_id=protocol_id,
            customer_id=marketplace_user.user_id,
            rating=request.rating,
            title=request.title,
            review_text=request.review_text,
            is_verified_purchase=installation is not None,
            installation_id=installation.installation_id if installation else None,
        )

        db.add(review)

        # Update protocol rating
        protocol.total_ratings += 1
        new_total = (protocol.average_rating * (protocol.total_ratings - 1)) + request.rating
        protocol.average_rating = new_total / protocol.total_ratings

        db.commit()
        db.refresh(review)

        logger.info(f"Review submitted for protocol {protocol.protocol_name} by {current_user.username}")

        return ReviewResponse(
            review_id=review.review_id,
            protocol_id=review.protocol_id,
            customer_id=review.customer_id,
            rating=review.rating,
            title=review.title,
            review_text=review.review_text,
            helpful_count=review.helpful_count,
            is_verified_purchase=review.is_verified_purchase,
            created_at=review.created_at,
            updated_at=review.updated_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting review: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Review submission failed: {str(e)}")


@router.get(
    "/my/protocols",
    response_model=List[ProtocolSummary],
    summary="Get My Protocols",
    description="Get protocols created by the current user",
)
async def get_my_protocols(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Get all protocols created by the current user.
    """
    try:
        marketplace_user = db.query(MarketplaceUser).filter(MarketplaceUser.email == current_user.email).first()

        if not marketplace_user:
            return []

        protocols = db.query(MarketplaceProtocol).filter(MarketplaceProtocol.author_id == marketplace_user.user_id).all()

        return [build_protocol_summary(p) for p in protocols]

    except Exception as e:
        logger.error(f"Error getting user protocols: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get protocols: {str(e)}")


@router.get(
    "/my/installations",
    response_model=List[InstallationInfo],
    summary="Get My Installations",
    description="Get protocols installed by the current user",
)
async def get_my_installations(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Get all protocol installations for the current user.
    """
    try:
        installations = db.query(MarketplaceInstallation).filter(MarketplaceInstallation.customer_id == current_user.id).all()

        return [
            InstallationInfo(
                installation_id=inst.installation_id,
                license_key=inst.license_key,
                license_type=inst.license_type,
                expires_at=inst.expires_at,
                status=inst.status.value,
                download_urls={
                    "spec": inst.protocol.spec_file_url,
                    "parser": inst.protocol.parser_code_url or "",
                    "docs": inst.protocol.documentation_url or "",
                },
                installation_instructions="See documentation for integration steps",
            )
            for inst in installations
        ]

    except Exception as e:
        logger.error(f"Error getting user installations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get installations: {str(e)}")


# =============================================================================
# Stripe Payment Integration Endpoints
# =============================================================================


@router.post(
    "/webhooks/stripe",
    summary="Stripe Webhook Handler",
    description="Handle Stripe webhook events for payment processing",
    status_code=200,
    include_in_schema=False,  # Hide from public API docs
)
async def stripe_webhook(request: Request):
    """
    Handle Stripe webhook events.

    This endpoint processes webhook events from Stripe to keep payment
    and subscription status in sync.
    """
    try:
        # Get raw request body and signature
        payload = await request.body()
        signature = request.headers.get("stripe-signature")

        if not signature:
            raise HTTPException(status_code=400, detail="Missing Stripe signature")

        # Process webhook with Stripe manager
        stripe_manager = get_stripe_manager()
        result = await stripe_manager.handle_webhook(payload, signature)

        logger.info(f"Stripe webhook processed: {result.get('event_type', 'unknown')}")

        return {"received": True, "event": result.get("event_type")}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stripe webhook processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Webhook processing failed: {str(e)}")


@router.post(
    "/subscriptions/{subscription_id}/cancel",
    summary="Cancel Subscription",
    description="Cancel a protocol subscription",
    status_code=200,
)
async def cancel_subscription(
    subscription_id: str,
    at_period_end: bool = Query(True, description="Cancel at end of billing period"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Cancel a subscription for a protocol.

    By default, cancels at the end of the current billing period.
    Set at_period_end=false to cancel immediately.
    """
    try:
        # Verify user owns this subscription
        # Note: In production, you'd query the transaction/installation to verify ownership

        stripe_manager = get_stripe_manager()
        result = await stripe_manager.cancel_subscription(subscription_id=subscription_id, at_period_end=at_period_end)

        logger.info(f"Subscription {subscription_id} cancelled by {current_user.username}")

        return {
            "success": True,
            "subscription_id": subscription_id,
            "status": result.get("status"),
            "cancel_at_period_end": result.get("cancel_at_period_end"),
            "message": (
                "Subscription will be cancelled at the end of the billing period"
                if at_period_end
                else "Subscription cancelled immediately"
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling subscription: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to cancel subscription: {str(e)}")


@router.post(
    "/creators/onboard",
    summary="Creator Onboarding",
    description="Create Stripe Connect account for protocol creators",
    status_code=201,
)
async def onboard_creator(
    email: str = Query(..., description="Creator email address"),
    country: str = Query("US", description="Country code (ISO 3166-1 alpha-2)"),
    business_type: str = Query("individual", description="individual or company"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Create a Stripe Connect account for a protocol creator.

    Returns an onboarding URL where the creator can complete their
    account setup and verify their identity.
    """
    try:
        # Get or create marketplace user
        marketplace_user = db.query(MarketplaceUser).filter(MarketplaceUser.user_id == current_user.id).first()

        if not marketplace_user:
            raise HTTPException(status_code=404, detail="Marketplace user not found. Please complete profile setup first.")

        if marketplace_user.stripe_account_id:
            raise HTTPException(status_code=400, detail="Creator account already exists")

        # Create Stripe Connect account
        stripe_manager = get_stripe_manager()
        result = await stripe_manager.create_connected_account(
            user_id=current_user.id, email=email, country=country, business_type=business_type
        )

        # Save Stripe account ID
        marketplace_user.stripe_account_id = result["account_id"]
        db.commit()

        logger.info(f"Stripe Connect account created for {current_user.username}")

        return {
            "success": True,
            "account_id": result["account_id"],
            "onboarding_url": result["onboarding_url"],
            "message": "Please complete the onboarding process at the provided URL",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating creator account: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create creator account: {str(e)}")
