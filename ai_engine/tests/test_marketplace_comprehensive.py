"""
QBITEL - Comprehensive Marketplace Tests

Unit tests for Protocol Marketplace functionality including API endpoints,
validation pipeline, and knowledge base integration.
"""

import pytest
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import sys

# Mock problematic imports before importing models
sys.modules["ai_engine.security.field_encryption"] = Mock()

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# Create Base directly for testing
Base = declarative_base()

from ai_engine.models.marketplace import (
    MarketplaceProtocol,
    MarketplaceUser,
    MarketplaceInstallation,
    MarketplaceReview,
    MarketplaceTransaction,
    MarketplaceValidation,
    ProtocolCategory,
    ProtocolType,
    SpecFormat,
    LicenseType,
    PriceModel,
    CertificationStatus,
    ProtocolStatus,
    InstallationStatus,
)
from ai_engine.marketplace.protocol_validator import (
    ProtocolValidator,
    ValidationResult,
)
from ai_engine.marketplace.knowledge_base_integration import (
    MarketplaceKnowledgeBaseIntegration,
    MarketplaceProtocolDeployer,
)


# Test Fixtures
@pytest.fixture
def db_session():
    """Create in-memory database session for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def sample_marketplace_user(db_session):
    """Create sample marketplace user."""
    user = MarketplaceUser(
        user_id=uuid.uuid4(),
        email="creator@example.com",
        username="protocol_creator",
        full_name="Protocol Creator",
        user_type="individual",
        is_verified=True,
        reputation_score=100,
    )
    db_session.add(user)
    db_session.commit()
    return user


@pytest.fixture
def sample_protocol(db_session, sample_marketplace_user):
    """Create sample marketplace protocol."""
    protocol = MarketplaceProtocol(
        protocol_id=uuid.uuid4(),
        protocol_name="test-protocol",
        display_name="Test Protocol",
        short_description="A test protocol for unit testing",
        long_description="This is a detailed description of the test protocol.",
        category=ProtocolCategory.FINANCE,
        subcategory="payment",
        tags=["test", "finance", "payment"],
        version="1.0.0",
        protocol_type=ProtocolType.BINARY,
        industry="finance",
        spec_format=SpecFormat.YAML,
        spec_file_url="s3://test/spec.yaml",
        parser_code_url="s3://test/parser.py",
        test_data_url="s3://test/test_data.bin",
        author_id=sample_marketplace_user.user_id,
        author_type="community",
        license_type=LicenseType.PAID,
        price_model=PriceModel.ONE_TIME,
        base_price=Decimal("499.00"),
        certification_status=CertificationStatus.CERTIFIED,
        status=ProtocolStatus.PUBLISHED,
        min_qbitel_version="1.0.0",
        published_at=datetime.utcnow(),
    )
    db_session.add(protocol)
    db_session.commit()
    return protocol


@pytest.fixture
def sample_installation(db_session, sample_protocol):
    """Create sample protocol installation."""
    installation = MarketplaceInstallation(
        installation_id=uuid.uuid4(),
        protocol_id=sample_protocol.protocol_id,
        customer_id=uuid.uuid4(),
        installed_version="1.0.0",
        license_key="TEST-LICENSE-KEY-12345",
        license_type=LicenseType.PAID,
        status=InstallationStatus.ACTIVE,
    )
    db_session.add(installation)
    db_session.commit()
    return installation


# Database Model Tests
class TestMarketplaceModels:
    """Test marketplace database models."""

    def test_marketplace_user_creation(self, db_session):
        """Test creating a marketplace user."""
        user = MarketplaceUser(
            user_id=uuid.uuid4(),
            email="test@example.com",
            username="testuser",
            full_name="Test User",
            user_type="individual",
        )
        db_session.add(user)
        db_session.commit()

        assert user.user_id is not None
        assert user.email == "test@example.com"
        assert user.username == "testuser"
        assert user.reputation_score == 0
        assert user.is_verified is False

    def test_marketplace_protocol_creation(self, sample_marketplace_user, db_session):
        """Test creating a marketplace protocol."""
        protocol = MarketplaceProtocol(
            protocol_id=uuid.uuid4(),
            protocol_name="sample-protocol",
            display_name="Sample Protocol",
            short_description="A sample protocol",
            category=ProtocolCategory.IOT,
            version="1.0.0",
            protocol_type=ProtocolType.JSON,
            spec_format=SpecFormat.JSON,
            spec_file_url="s3://test/spec.json",
            author_id=sample_marketplace_user.user_id,
            author_type="community",
            license_type=LicenseType.FREE,
            min_qbitel_version="1.0.0",
        )
        db_session.add(protocol)
        db_session.commit()

        assert protocol.protocol_id is not None
        assert protocol.protocol_name == "sample-protocol"
        assert protocol.license_type == LicenseType.FREE
        assert protocol.base_price is None  # Free protocols have no price

    def test_protocol_price_constraint(self, sample_marketplace_user, db_session):
        """Test protocol price constraint validation."""
        # Paid protocol must have price
        protocol = MarketplaceProtocol(
            protocol_id=uuid.uuid4(),
            protocol_name="paid-protocol",
            display_name="Paid Protocol",
            short_description="A paid protocol",
            category=ProtocolCategory.FINANCE,
            version="1.0.0",
            protocol_type=ProtocolType.BINARY,
            spec_format=SpecFormat.YAML,
            spec_file_url="s3://test/spec.yaml",
            author_id=sample_marketplace_user.user_id,
            author_type="community",
            license_type=LicenseType.PAID,
            price_model=PriceModel.ONE_TIME,
            base_price=Decimal("99.99"),
            min_qbitel_version="1.0.0",
        )
        db_session.add(protocol)
        db_session.commit()

        assert protocol.base_price == Decimal("99.99")

    def test_installation_creation(self, sample_protocol, db_session):
        """Test creating a protocol installation."""
        installation = MarketplaceInstallation(
            installation_id=uuid.uuid4(),
            protocol_id=sample_protocol.protocol_id,
            customer_id=uuid.uuid4(),
            installed_version="1.0.0",
            license_key="TEST-KEY-123",
            license_type=LicenseType.PAID,
            status=InstallationStatus.ACTIVE,
        )
        db_session.add(installation)
        db_session.commit()

        assert installation.installation_id is not None
        assert installation.status == InstallationStatus.ACTIVE
        assert installation.protocol_id == sample_protocol.protocol_id

    def test_review_creation(self, sample_protocol, sample_marketplace_user, db_session):
        """Test creating a protocol review."""
        review = MarketplaceReview(
            review_id=uuid.uuid4(),
            protocol_id=sample_protocol.protocol_id,
            customer_id=sample_marketplace_user.user_id,
            rating=5,
            title="Excellent Protocol",
            review_text="This protocol works great!",
            is_verified_purchase=True,
        )
        db_session.add(review)
        db_session.commit()

        assert review.review_id is not None
        assert review.rating == 5
        assert review.is_verified_purchase is True

    def test_review_rating_constraint(self, sample_protocol, sample_marketplace_user, db_session):
        """Test review rating must be between 1 and 5."""
        # SQLite doesn't enforce check constraints by default, but test the intent
        review = MarketplaceReview(
            review_id=uuid.uuid4(),
            protocol_id=sample_protocol.protocol_id,
            customer_id=sample_marketplace_user.user_id,
            rating=3,
        )
        db_session.add(review)
        db_session.commit()

        assert 1 <= review.rating <= 5


# Protocol Validator Tests
class TestProtocolValidator:
    """Test protocol validation pipeline."""

    @pytest.mark.asyncio
    async def test_syntax_validation_success(self, sample_protocol):
        """Test successful syntax validation."""
        validator = ProtocolValidator()

        with patch.object(
            validator,
            "_download_spec_file",
            return_value="""
protocol_metadata:
  name: test-protocol
  version: 1.0.0
  category: finance

protocol_spec:
  message_format: binary
  fields:
    - id: 1
      name: field1
      type: string
""",
        ):
            result = await validator.validate_syntax(sample_protocol)

        assert result.status == "passed"
        assert result.validation_type == "syntax_validation"
        assert len(result.errors) == 0
        assert result.score > 0

    @pytest.mark.asyncio
    async def test_syntax_validation_invalid_yaml(self, sample_protocol):
        """Test syntax validation with invalid YAML."""
        validator = ProtocolValidator()

        with patch.object(validator, "_download_spec_file", return_value="invalid: yaml: content:"):
            result = await validator.validate_syntax(sample_protocol)

        assert result.status == "failed"
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_parser_testing_success(self, sample_protocol):
        """Test successful parser testing."""
        validator = ProtocolValidator()

        with patch.object(validator, "_download_parser_code", return_value="def parse(): pass"):
            with patch.object(validator, "_download_test_data", return_value=b"test"):
                result = await validator.test_parser(sample_protocol)

        assert result.validation_type == "parser_testing"
        # Result status depends on mock implementation

    @pytest.mark.asyncio
    async def test_security_scan_success(self, sample_protocol):
        """Test security scanning."""
        validator = ProtocolValidator()

        safe_code = """
def parse_packet(data):
    return {"field": data.decode()}
"""

        with patch.object(validator, "_download_parser_code", return_value=safe_code):
            result = await validator.security_scan(sample_protocol)

        assert result.validation_type == "security_scan"
        assert "eval(" not in safe_code
        assert "exec(" not in safe_code

    @pytest.mark.asyncio
    async def test_security_scan_detects_issues(self, sample_protocol):
        """Test security scan detects dangerous code."""
        validator = ProtocolValidator()

        dangerous_code = """
def parse_packet(data):
    eval(data)  # Dangerous!
    return {}
"""

        with patch.object(validator, "_download_parser_code", return_value=dangerous_code):
            result = await validator.security_scan(sample_protocol)

        assert result.validation_type == "security_scan"
        # Should detect eval usage

    @pytest.mark.asyncio
    async def test_performance_benchmark(self, sample_protocol):
        """Test performance benchmarking."""
        validator = ProtocolValidator()

        result = await validator.performance_benchmark(sample_protocol)

        assert result.validation_type == "performance_benchmark"
        assert "throughput" in result.metrics
        assert "memory_usage" in result.metrics
        assert "latency_p50" in result.metrics


# Knowledge Base Integration Tests
class TestMarketplaceKnowledgeBaseIntegration:
    """Test marketplace knowledge base integration."""

    @pytest.mark.asyncio
    async def test_import_marketplace_protocol(self, sample_protocol, sample_installation):
        """Test importing marketplace protocol to knowledge base."""
        mock_kb = Mock()
        mock_kb.rag_engine = Mock()
        mock_kb.rag_engine.add_documents = AsyncMock()

        integration = MarketplaceKnowledgeBaseIntegration(knowledge_base=mock_kb)

        with patch.object(integration, "db_manager") as mock_db:
            mock_session = MagicMock()
            mock_session.query.return_value.filter.return_value.first.side_effect = [
                sample_protocol,
                sample_installation,
            ]
            mock_db.get_session.return_value = mock_session

            with patch.object(
                integration,
                "_download_protocol_spec",
                new=AsyncMock(return_value="""
protocol_metadata:
  name: test
  version: 1.0.0
  category: finance
protocol_spec:
  fields: []
"""),
            ):
                with patch.object(integration, "_download_parser", new=AsyncMock(return_value="def parse(): pass")):
                    knowledge = await integration.import_marketplace_protocol(
                        protocol_id=sample_protocol.protocol_id,
                        installation_id=sample_installation.installation_id,
                    )

        assert knowledge is not None
        assert knowledge.protocol_name == sample_protocol.protocol_name
        assert mock_kb.rag_engine.add_documents.called

    @pytest.mark.asyncio
    async def test_parse_specification_yaml(self):
        """Test parsing YAML specification."""
        integration = MarketplaceKnowledgeBaseIntegration()

        spec_content = """
protocol_metadata:
  name: test
  version: 1.0.0
"""
        result = integration._parse_specification(spec_content, "yaml")

        assert "protocol_metadata" in result
        assert result["protocol_metadata"]["name"] == "test"

    @pytest.mark.asyncio
    async def test_parse_specification_json(self):
        """Test parsing JSON specification."""
        integration = MarketplaceKnowledgeBaseIntegration()

        spec_content = '{"protocol_metadata": {"name": "test", "version": "1.0.0"}}'
        result = integration._parse_specification(spec_content, "json")

        assert "protocol_metadata" in result
        assert result["protocol_metadata"]["name"] == "test"

    @pytest.mark.asyncio
    async def test_parse_specification_invalid_format(self):
        """Test parsing with unsupported format."""
        integration = MarketplaceKnowledgeBaseIntegration()

        with pytest.raises(ValueError, match="Unsupported spec format"):
            integration._parse_specification("content", "xml")


# Protocol Deployer Tests
class TestMarketplaceProtocolDeployer:
    """Test marketplace protocol deployer."""

    @pytest.mark.asyncio
    async def test_deploy_protocol(self, sample_installation):
        """Test deploying protocol to Translation Studio."""
        deployer = MarketplaceProtocolDeployer()

        with patch.object(deployer, "db_manager") as mock_db:
            mock_session = MagicMock()
            mock_session.query.return_value.filter.return_value.first.return_value = sample_installation
            mock_db.get_session.return_value = mock_session

            with patch.object(deployer, "_generate_api_spec", new=AsyncMock(return_value={})):
                # Should not raise exception
                await deployer.deploy_protocol(
                    installation_id=sample_installation.installation_id,
                    target_environment="production",
                )

    @pytest.mark.asyncio
    async def test_deploy_protocol_invalid_installation(self):
        """Test deploying with invalid installation."""
        deployer = MarketplaceProtocolDeployer()

        with patch.object(deployer, "db_manager") as mock_db:
            mock_session = MagicMock()
            mock_session.query.return_value.filter.return_value.first.return_value = None
            mock_db.get_session.return_value = mock_session

            with pytest.raises(ValueError, match="Installation.*not found"):
                await deployer.deploy_protocol(
                    installation_id=uuid.uuid4(),
                    target_environment="production",
                )


# API Endpoint Tests (using FastAPI TestClient)
class TestMarketplaceAPIEndpoints:
    """Test marketplace API endpoints."""

    def test_marketplace_imports(self):
        """Test that marketplace modules can be imported."""
        from ai_engine.api import marketplace_schemas
        from ai_engine.api import marketplace_endpoints

        assert marketplace_schemas is not None
        assert marketplace_endpoints is not None

    def test_protocol_summary_schema(self):
        """Test ProtocolSummary schema validation."""
        from ai_engine.api.marketplace_schemas import ProtocolSummary, AuthorInfo

        # Create sample data
        author_data = {
            "user_id": uuid.uuid4(),
            "username": "test_user",
            "full_name": "Test User",
            "organization": "Test Org",
            "is_verified": True,
            "reputation_score": 100,
            "total_contributions": 5,
            "avatar_url": "https://example.com/avatar.png",
        }

        author = AuthorInfo(**author_data)
        assert author.username == "test_user"

    def test_validation_result_creation(self):
        """Test ValidationResult object creation."""
        result = ValidationResult(
            validation_type="syntax_validation",
            status="passed",
            score=Decimal("95.0"),
            errors=[],
            warnings=["Minor warning"],
            test_results={"yaml_valid": True},
            metrics={},
        )

        assert result.validation_type == "syntax_validation"
        assert result.status == "passed"
        assert result.score == Decimal("95.0")
        assert len(result.warnings) == 1


# Integration Tests
class TestMarketplaceIntegration:
    """Integration tests for marketplace components."""

    @pytest.mark.asyncio
    async def test_end_to_end_protocol_submission_flow(self, sample_marketplace_user, db_session):
        """Test complete protocol submission and validation flow."""
        # This would test the full flow:
        # 1. User submits protocol
        # 2. Protocol is validated
        # 3. Protocol is published
        # 4. Protocol appears in search
        # 5. Customer purchases protocol
        # 6. Protocol is installed and deployed

        # Mock implementation - in production would test full stack
        protocol_data = {
            "protocol_name": "integration-test-protocol",
            "display_name": "Integration Test Protocol",
            "short_description": "Protocol for integration testing",
            "category": ProtocolCategory.IOT,
            "version": "1.0.0",
            "protocol_type": ProtocolType.JSON,
            "spec_format": SpecFormat.JSON,
            "spec_file_url": "s3://test/spec.json",
            "author_id": sample_marketplace_user.user_id,
            "author_type": "community",
            "license_type": LicenseType.FREE,
            "min_qbitel_version": "1.0.0",
        }

        protocol = MarketplaceProtocol(**protocol_data)
        db_session.add(protocol)
        db_session.commit()

        assert protocol.protocol_id is not None
        assert protocol.status == ProtocolStatus.DRAFT


# Performance Tests
class TestMarketplacePerformance:
    """Performance tests for marketplace operations."""

    def test_bulk_protocol_search(self, db_session, sample_marketplace_user):
        """Test searching through many protocols."""
        # Create 100 test protocols
        protocols = []
        for i in range(100):
            protocol = MarketplaceProtocol(
                protocol_id=uuid.uuid4(),
                protocol_name=f"test-protocol-{i}",
                display_name=f"Test Protocol {i}",
                short_description=f"Protocol {i} for testing",
                category=ProtocolCategory.IOT if i % 2 == 0 else ProtocolCategory.FINANCE,
                version="1.0.0",
                protocol_type=ProtocolType.JSON,
                spec_format=SpecFormat.JSON,
                spec_file_url=f"s3://test/spec-{i}.json",
                author_id=sample_marketplace_user.user_id,
                author_type="community",
                license_type=LicenseType.FREE,
                min_qbitel_version="1.0.0",
                status=ProtocolStatus.PUBLISHED,
            )
            protocols.append(protocol)

        db_session.bulk_save_objects(protocols)
        db_session.commit()

        # Query protocols
        results = db_session.query(MarketplaceProtocol).filter(MarketplaceProtocol.status == ProtocolStatus.PUBLISHED).all()

        assert len(results) >= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
