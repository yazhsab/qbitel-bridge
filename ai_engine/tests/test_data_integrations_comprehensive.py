"""
Comprehensive Unit Tests for compliance/data_integrations.py
Ensures 100% code coverage for TimescaleDB, Redis, and Security integrations.
"""

import pytest
import asyncio
import json
import pickle
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import dataclass, field as dataclass_field
from typing import List, Dict, Any

from ai_engine.compliance.data_integrations import (
    IntegrationException,
    TimescaleComplianceIntegration,
    RedisComplianceCache,
    ComplianceSecurityIntegration,
)
from ai_engine.core.config import Config


# Mock classes for testing
@dataclass
class MockComplianceAssessment:
    """Mock compliance assessment for testing."""

    assessment_id: str
    framework: str
    version: str
    assessment_date: datetime
    overall_compliance_score: float
    risk_score: float
    compliant_requirements: int
    non_compliant_requirements: int
    partially_compliant_requirements: int
    not_assessed_requirements: int
    next_assessment_due: datetime
    gaps: List[Any] = dataclass_field(default_factory=list)
    recommendations: List[Any] = dataclass_field(default_factory=list)


@dataclass
class MockGap:
    """Mock compliance gap."""

    requirement_id: str
    requirement_title: str
    severity: Mock
    current_state: str
    required_state: str
    gap_description: str
    impact_assessment: str
    remediation_effort: str


@dataclass
class MockRecommendation:
    """Mock recommendation."""

    id: str
    title: str
    description: str
    priority: Mock
    implementation_steps: List[str]
    estimated_effort_days: int
    business_impact: str
    technical_requirements: Dict[str, Any]


@dataclass
class MockComplianceReport:
    """Mock compliance report."""

    report_id: str
    framework: str
    report_type: Mock
    format: Mock
    generated_date: datetime
    file_name: str
    file_size: int
    checksum: str
    metadata: Dict[str, Any]
    content: bytes = b"test content"


class TestTimescaleComplianceIntegration:
    """Test suite for TimescaleDB integration."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock(spec=Config)
        config.database = Mock()
        config.database.host = "localhost"
        config.database.port = 5432
        config.database.database = "test_db"
        config.database.username = "test_user"
        config.database.password = "test_pass"
        return config

    @pytest.fixture
    def timescale_integration(self, mock_config):
        """Create TimescaleDB integration instance."""
        return TimescaleComplianceIntegration(mock_config)

    @pytest.mark.asyncio
    async def test_initialization_success(self, timescale_integration):
        """Test successful initialization."""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire = AsyncMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )

        with patch(
            "ai_engine.compliance.data_integrations.asyncpg.create_pool",
            return_value=mock_pool,
        ):
            await timescale_integration.initialize()
            assert timescale_integration.connection_pool is not None

    @pytest.mark.asyncio
    async def test_initialization_import_error(self, timescale_integration):
        """Test initialization with import error."""
        with patch(
            "ai_engine.compliance.data_integrations.asyncpg.create_pool",
            side_effect=ImportError("asyncpg not found"),
        ):
            with pytest.raises(IntegrationException, match="asyncpg library required"):
                await timescale_integration.initialize()

    @pytest.mark.asyncio
    async def test_initialization_connection_error(self, timescale_integration):
        """Test initialization with connection error."""
        with patch(
            "ai_engine.compliance.data_integrations.asyncpg.create_pool",
            side_effect=Exception("Connection failed"),
        ):
            with pytest.raises(
                IntegrationException, match="TimescaleDB initialization failed"
            ):
                await timescale_integration.initialize()

    @pytest.mark.asyncio
    async def test_create_tables_success(self, timescale_integration):
        """Test successful table creation."""
        mock_conn = AsyncMock()
        mock_pool = AsyncMock()
        mock_pool.acquire = AsyncMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )
        timescale_integration.connection_pool = mock_pool

        await timescale_integration._create_tables()
        assert mock_conn.execute.call_count >= 5  # Multiple table creation calls

    @pytest.mark.asyncio
    async def test_create_tables_hypertable_warning(self, timescale_integration):
        """Test table creation with hypertable warning."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(
            side_effect=[
                None,
                None,
                None,
                None,
                None,
                Exception("Hypertable error"),
                None,
                None,
                None,
                None,
            ]
        )
        mock_pool = AsyncMock()
        mock_pool.acquire = AsyncMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )
        timescale_integration.connection_pool = mock_pool

        await timescale_integration._create_tables()
        # Should continue despite hypertable error

    @pytest.mark.asyncio
    async def test_create_tables_failure(self, timescale_integration):
        """Test table creation failure."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(side_effect=Exception("Table creation failed"))
        mock_pool = AsyncMock()
        mock_pool.acquire = AsyncMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )
        timescale_integration.connection_pool = mock_pool

        with pytest.raises(IntegrationException, match="Table creation failed"):
            await timescale_integration._create_tables()

    @pytest.mark.asyncio
    async def test_store_assessment_success(self, timescale_integration):
        """Test successful assessment storage."""
        # Create mock assessment
        severity_mock = Mock()
        severity_mock.value = "critical"
        priority_mock = Mock()
        priority_mock.value = "high"

        gap = MockGap(
            requirement_id="REQ-001",
            requirement_title="Test Requirement",
            severity=severity_mock,
            current_state="non-compliant",
            required_state="compliant",
            gap_description="Test gap",
            impact_assessment="High impact",
            remediation_effort="medium",
        )

        recommendation = MockRecommendation(
            id="REC-001",
            title="Test Recommendation",
            description="Test description",
            priority=priority_mock,
            implementation_steps=["step1", "step2"],
            estimated_effort_days=5,
            business_impact="Medium",
            technical_requirements={"skill": "python"},
        )

        assessment = MockComplianceAssessment(
            assessment_id="test-001",
            framework="SOC2",
            version="1.0",
            assessment_date=datetime.utcnow(),
            overall_compliance_score=85.5,
            risk_score=15.5,
            compliant_requirements=10,
            non_compliant_requirements=2,
            partially_compliant_requirements=1,
            not_assessed_requirements=0,
            next_assessment_due=datetime.utcnow() + timedelta(days=90),
            gaps=[gap],
            recommendations=[recommendation],
        )

        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value="test-001")
        mock_transaction = AsyncMock()
        mock_transaction.__aenter__ = AsyncMock(return_value=mock_transaction)
        mock_transaction.__aexit__ = AsyncMock()
        mock_conn.transaction = Mock(return_value=mock_transaction)

        mock_pool = AsyncMock()
        mock_pool.acquire = AsyncMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )
        timescale_integration.connection_pool = mock_pool

        result = await timescale_integration.store_assessment(assessment)
        assert result == "test-001"

    @pytest.mark.asyncio
    async def test_store_assessment_failure(self, timescale_integration):
        """Test assessment storage failure."""
        assessment = MockComplianceAssessment(
            assessment_id="test-001",
            framework="SOC2",
            version="1.0",
            assessment_date=datetime.utcnow(),
            overall_compliance_score=85.5,
            risk_score=15.5,
            compliant_requirements=10,
            non_compliant_requirements=2,
            partially_compliant_requirements=1,
            not_assessed_requirements=0,
            next_assessment_due=datetime.utcnow() + timedelta(days=90),
            gaps=[],
            recommendations=[],
        )

        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(side_effect=Exception("Storage failed"))
        mock_transaction = AsyncMock()
        mock_transaction.__aenter__ = AsyncMock(return_value=mock_transaction)
        mock_transaction.__aexit__ = AsyncMock()
        mock_conn.transaction = Mock(return_value=mock_transaction)

        mock_pool = AsyncMock()
        mock_pool.acquire = AsyncMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )
        timescale_integration.connection_pool = mock_pool

        with pytest.raises(IntegrationException, match="Assessment storage failed"):
            await timescale_integration.store_assessment(assessment)

    @pytest.mark.asyncio
    async def test_get_latest_assessment_success(self, timescale_integration):
        """Test getting latest assessment."""
        assessment_data = {
            "assessment_id": "test-001",
            "framework": "SOC2",
            "version": "1.0",
            "assessment_date": datetime.utcnow(),
            "overall_compliance_score": 85.5,
            "risk_score": 15.5,
            "compliant_requirements": 10,
            "non_compliant_requirements": 2,
            "partially_compliant_requirements": 1,
            "not_assessed_requirements": 0,
            "next_assessment_due": datetime.utcnow() + timedelta(days=90),
            "gaps": [],
            "recommendations": [],
        }

        mock_row = {"assessment_data": json.dumps(assessment_data, default=str)}
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=mock_row)
        mock_pool = AsyncMock()
        mock_pool.acquire = AsyncMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )
        timescale_integration.connection_pool = mock_pool

        with patch(
            "ai_engine.compliance.data_integrations.ComplianceAssessment"
        ) as mock_assessment:
            result = await timescale_integration.get_latest_assessment("SOC2")
            mock_assessment.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_latest_assessment_not_found(self, timescale_integration):
        """Test getting latest assessment when not found."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)
        mock_pool = AsyncMock()
        mock_pool.acquire = AsyncMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )
        timescale_integration.connection_pool = mock_pool

        result = await timescale_integration.get_latest_assessment("SOC2")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_latest_assessment_error(self, timescale_integration):
        """Test getting latest assessment with error."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(side_effect=Exception("Query failed"))
        mock_pool = AsyncMock()
        mock_pool.acquire = AsyncMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )
        timescale_integration.connection_pool = mock_pool

        result = await timescale_integration.get_latest_assessment("SOC2")
        assert result is None

    @pytest.mark.asyncio
    async def test_store_report_metadata(self, timescale_integration):
        """Test storing report metadata."""
        report_type_mock = Mock()
        report_type_mock.value = "detailed"
        format_mock = Mock()
        format_mock.value = "pdf"

        report = MockComplianceReport(
            report_id="report-001",
            framework="SOC2",
            report_type=report_type_mock,
            format=format_mock,
            generated_date=datetime.utcnow(),
            file_name="report.pdf",
            file_size=1024,
            checksum="abc123",
            metadata={"key": "value"},
        )

        mock_conn = AsyncMock()
        mock_pool = AsyncMock()
        mock_pool.acquire = AsyncMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )
        timescale_integration.connection_pool = mock_pool

        await timescale_integration.store_report_metadata(report)
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_compliance_trends(self, timescale_integration):
        """Test getting compliance trends."""
        mock_rows = [
            {
                "framework": "SOC2",
                "date": datetime.utcnow(),
                "avg_compliance": 85.5,
                "avg_risk": 15.5,
                "max_gaps": 5,
                "max_critical_gaps": 2,
            }
        ]

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)
        mock_pool = AsyncMock()
        mock_pool.acquire = AsyncMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )
        timescale_integration.connection_pool = mock_pool

        result = await timescale_integration.get_compliance_trends(["SOC2"], days=30)
        assert "SOC2" in result
        assert len(result["SOC2"]) == 1

    @pytest.mark.asyncio
    async def test_get_compliance_trends_error(self, timescale_integration):
        """Test getting compliance trends with error."""
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(side_effect=Exception("Query failed"))
        mock_pool = AsyncMock()
        mock_pool.acquire = AsyncMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )
        timescale_integration.connection_pool = mock_pool

        result = await timescale_integration.get_compliance_trends(["SOC2"], days=30)
        assert result == {}

    @pytest.mark.asyncio
    async def test_cleanup_old_data(self, timescale_integration):
        """Test cleaning up old data."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="DELETE 5")
        mock_pool = AsyncMock()
        mock_pool.acquire = AsyncMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )
        timescale_integration.connection_pool = mock_pool

        await timescale_integration.cleanup_old_data(retention_days=365)
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_old_data_no_deletions(self, timescale_integration):
        """Test cleanup with no deletions."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value=None)
        mock_pool = AsyncMock()
        mock_pool.acquire = AsyncMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )
        timescale_integration.connection_pool = mock_pool

        await timescale_integration.cleanup_old_data(retention_days=365)
        # Should complete without error

    @pytest.mark.asyncio
    async def test_cleanup_old_data_error(self, timescale_integration):
        """Test cleanup with error."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(side_effect=Exception("Cleanup failed"))
        mock_pool = AsyncMock()
        mock_pool.acquire = AsyncMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )
        timescale_integration.connection_pool = mock_pool

        await timescale_integration.cleanup_old_data(retention_days=365)
        # Should handle error gracefully

    @pytest.mark.asyncio
    async def test_close(self, timescale_integration):
        """Test closing connection pool."""
        mock_pool = AsyncMock()
        timescale_integration.connection_pool = mock_pool

        await timescale_integration.close()
        mock_pool.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_no_pool(self, timescale_integration):
        """Test closing when no pool exists."""
        timescale_integration.connection_pool = None
        await timescale_integration.close()
        # Should complete without error

    def test_password_from_environment(self, mock_config):
        """Test password loading from environment."""
        mock_config.database.password = ""
        with patch.dict("os.environ", {"CRONOS_AI_DB_PASSWORD": "env_password"}):
            integration = TimescaleComplianceIntegration(mock_config)
            assert integration.password == "env_password"

    def test_password_from_database_password_env(self, mock_config):
        """Test password loading from DATABASE_PASSWORD env."""
        mock_config.database.password = ""
        with patch.dict("os.environ", {"DATABASE_PASSWORD": "db_password"}, clear=True):
            integration = TimescaleComplianceIntegration(mock_config)
            assert integration.password == "db_password"


class TestRedisComplianceCache:
    """Test suite for Redis compliance cache."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock(spec=Config)
        config.redis = Mock()
        config.redis.host = "localhost"
        config.redis.port = 6379
        config.redis.password = None
        config.redis.db = 0
        return config

    @pytest.fixture
    def redis_cache(self, mock_config):
        """Create Redis cache instance."""
        return RedisComplianceCache(mock_config)

    @pytest.mark.asyncio
    async def test_initialization_success(self, redis_cache):
        """Test successful initialization."""
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()

        with patch(
            "ai_engine.compliance.data_integrations.redis.Redis",
            return_value=mock_redis,
        ):
            await redis_cache.initialize()
            assert redis_cache.redis_client is not None
            mock_redis.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialization_import_error(self, redis_cache):
        """Test initialization with import error."""
        with patch(
            "ai_engine.compliance.data_integrations.redis.Redis",
            side_effect=ImportError("redis not found"),
        ):
            with pytest.raises(IntegrationException, match="redis library required"):
                await redis_cache.initialize()

    @pytest.mark.asyncio
    async def test_initialization_connection_error(self, redis_cache):
        """Test initialization with connection error."""
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(side_effect=Exception("Connection failed"))

        with patch(
            "ai_engine.compliance.data_integrations.redis.Redis",
            return_value=mock_redis,
        ):
            with pytest.raises(
                IntegrationException, match="Redis initialization failed"
            ):
                await redis_cache.initialize()

    @pytest.mark.asyncio
    async def test_cache_assessment_with_asdict(self, redis_cache):
        """Test caching assessment using asdict."""
        assessment = MockComplianceAssessment(
            assessment_id="test-001",
            framework="SOC2",
            version="1.0",
            assessment_date=datetime.utcnow(),
            overall_compliance_score=85.5,
            risk_score=15.5,
            compliant_requirements=10,
            non_compliant_requirements=2,
            partially_compliant_requirements=1,
            not_assessed_requirements=0,
            next_assessment_due=datetime.utcnow() + timedelta(days=90),
            gaps=[],
            recommendations=[],
        )

        mock_redis = AsyncMock()
        redis_cache.redis_client = mock_redis

        await redis_cache.cache_assessment("SOC2", assessment)
        assert mock_redis.setex.call_count == 2  # Assessment and metadata

    @pytest.mark.asyncio
    async def test_cache_assessment_with_fallback(self, redis_cache):
        """Test caching assessment with fallback serialization."""
        # Create assessment that will fail asdict
        assessment = Mock()
        assessment.framework = "SOC2"
        assessment.version = "1.0"
        assessment.assessment_date = datetime.utcnow()
        assessment.overall_compliance_score = 85.5
        assessment.risk_score = 15.5
        assessment.compliant_requirements = 10
        assessment.non_compliant_requirements = 2
        assessment.partially_compliant_requirements = 1
        assessment.not_assessed_requirements = 0
        assessment.next_assessment_due = datetime.utcnow() + timedelta(days=90)

        mock_redis = AsyncMock()
        redis_cache.redis_client = mock_redis

        with patch(
            "ai_engine.compliance.data_integrations.asdict",
            side_effect=TypeError("Cannot convert"),
        ):
            await redis_cache.cache_assessment("SOC2", assessment)
            assert mock_redis.setex.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_assessment_error(self, redis_cache):
        """Test caching assessment with error."""
        assessment = MockComplianceAssessment(
            assessment_id="test-001",
            framework="SOC2",
            version="1.0",
            assessment_date=datetime.utcnow(),
            overall_compliance_score=85.5,
            risk_score=15.5,
            compliant_requirements=10,
            non_compliant_requirements=2,
            partially_compliant_requirements=1,
            not_assessed_requirements=0,
            next_assessment_due=datetime.utcnow() + timedelta(days=90),
            gaps=[],
            recommendations=[],
        )

        mock_redis = AsyncMock()
        mock_redis.setex = AsyncMock(side_effect=Exception("Cache failed"))
        redis_cache.redis_client = mock_redis

        await redis_cache.cache_assessment("SOC2", assessment)
        # Should handle error gracefully

    @pytest.mark.asyncio
    async def test_get_assessment_success(self, redis_cache):
        """Test getting cached assessment."""
        assessment_data = {
            "assessment_id": "test-001",
            "framework": "SOC2",
            "version": "1.0",
            "assessment_date": datetime.utcnow().isoformat(),
            "overall_compliance_score": 85.5,
            "risk_score": 15.5,
            "compliant_requirements": 10,
            "non_compliant_requirements": 2,
            "partially_compliant_requirements": 1,
            "not_assessed_requirements": 0,
            "next_assessment_due": (datetime.utcnow() + timedelta(days=90)).isoformat(),
            "gaps": [],
            "recommendations": [],
        }

        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=pickle.dumps(assessment_data))
        redis_cache.redis_client = mock_redis

        with patch(
            "ai_engine.compliance.data_integrations.ComplianceAssessment"
        ) as mock_assessment:
            result = await redis_cache.get_assessment("SOC2")
            mock_assessment.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_assessment_not_found(self, redis_cache):
        """Test getting assessment when not cached."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        redis_cache.redis_client = mock_redis

        result = await redis_cache.get_assessment("SOC2")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_assessment_error(self, redis_cache):
        """Test getting assessment with error."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=Exception("Get failed"))
        redis_cache.redis_client = mock_redis

        result = await redis_cache.get_assessment("SOC2")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_dashboard_data(self, redis_cache):
        """Test caching dashboard data."""
        dashboard_data = {"key": "value", "score": 85.5}

        mock_redis = AsyncMock()
        redis_cache.redis_client = mock_redis

        await redis_cache.cache_dashboard_data(dashboard_data)
        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_dashboard_data_error(self, redis_cache):
        """Test caching dashboard data with error."""
        mock_redis = AsyncMock()
        mock_redis.setex = AsyncMock(side_effect=Exception("Cache failed"))
        redis_cache.redis_client = mock_redis

        await redis_cache.cache_dashboard_data({"key": "value"})
        # Should handle error gracefully

    @pytest.mark.asyncio
    async def test_get_dashboard_data_success(self, redis_cache):
        """Test getting cached dashboard data."""
        dashboard_data = {"key": "value", "score": 85.5}

        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=json.dumps(dashboard_data).encode())
        redis_cache.redis_client = mock_redis

        result = await redis_cache.get_dashboard_data()
        assert result == dashboard_data

    @pytest.mark.asyncio
    async def test_get_dashboard_data_not_found(self, redis_cache):
        """Test getting dashboard data when not cached."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        redis_cache.redis_client = mock_redis

        result = await redis_cache.get_dashboard_data()
        assert result is None

    @pytest.mark.asyncio
    async def test_get_dashboard_data_error(self, redis_cache):
        """Test getting dashboard data with error."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=Exception("Get failed"))
        redis_cache.redis_client = mock_redis

        result = await redis_cache.get_dashboard_data()
        assert result is None

    @pytest.mark.asyncio
    async def test_invalidate_framework_cache(self, redis_cache):
        """Test invalidating framework cache."""
        mock_redis = AsyncMock()
        redis_cache.redis_client = mock_redis

        await redis_cache.invalidate_framework_cache("SOC2")
        mock_redis.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalidate_framework_cache_error(self, redis_cache):
        """Test invalidating cache with error."""
        mock_redis = AsyncMock()
        mock_redis.delete = AsyncMock(side_effect=Exception("Delete failed"))
        redis_cache.redis_client = mock_redis

        await redis_cache.invalidate_framework_cache("SOC2")
        # Should handle error gracefully

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, redis_cache):
        """Test cleaning up expired keys."""
        mock_redis = AsyncMock()
        mock_redis.keys = AsyncMock(return_value=[b"key1", b"key2", b"key3"])
        mock_pipeline = AsyncMock()
        mock_pipeline.ttl = Mock()
        mock_pipeline.execute = AsyncMock(return_value=[-1, 100, -1])
        mock_redis.pipeline = Mock(return_value=mock_pipeline)
        redis_cache.redis_client = mock_redis

        await redis_cache.cleanup_expired()
        mock_redis.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_expired_no_keys(self, redis_cache):
        """Test cleanup with no keys."""
        mock_redis = AsyncMock()
        mock_redis.keys = AsyncMock(return_value=[])
        redis_cache.redis_client = mock_redis

        await redis_cache.cleanup_expired()
        # Should complete without error

    @pytest.mark.asyncio
    async def test_cleanup_expired_error(self, redis_cache):
        """Test cleanup with error."""
        mock_redis = AsyncMock()
        mock_redis.keys = AsyncMock(side_effect=Exception("Cleanup failed"))
        redis_cache.redis_client = mock_redis

        await redis_cache.cleanup_expired()
        # Should handle error gracefully

    @pytest.mark.asyncio
    async def test_get_cache_statistics(self, redis_cache):
        """Test getting cache statistics."""
        mock_redis = AsyncMock()
        mock_redis.keys = AsyncMock(
            return_value=[
                b"compliance:assessment:SOC2",
                b"compliance:meta:SOC2",
                b"compliance:dashboard",
            ]
        )
        redis_cache.redis_client = mock_redis

        result = await redis_cache.get_cache_statistics()
        assert result["total_keys"] == 3
        assert "key_types" in result
        assert "frameworks_cached" in result

    @pytest.mark.asyncio
    async def test_get_cache_statistics_error(self, redis_cache):
        """Test getting statistics with error."""
        mock_redis = AsyncMock()
        mock_redis.keys = AsyncMock(side_effect=Exception("Stats failed"))
        redis_cache.redis_client = mock_redis

        result = await redis_cache.get_
        mock_redis = AsyncMock()
        mock_redis.keys = AsyncMock(
            return_value=[
                b"compliance:assessment:SOC2",
                b"compliance:meta:SOC2",
                b"compliance:dashboard",
            ]
        )
        redis_cache.redis_client = mock_redis

        result = await redis_cache.get_cache_statistics()
        assert result["total_keys"] == 3
        assert "key_types" in result
        assert "frameworks_cached" in result

    @pytest.mark.asyncio
    async def test_get_cache_statistics_error(self, redis_cache):
        """Test getting statistics with error."""
        mock_redis = AsyncMock()
        mock_redis.keys = AsyncMock(side_effect=Exception("Stats failed"))
        redis_cache.redis_client = mock_redis

        result = await redis_cache.get_cache_statistics()
        assert result == {}

    @pytest.mark.asyncio
    async def test_close(self, redis_cache):
        """Test closing Redis connection."""
        mock_redis = AsyncMock()
        redis_cache.redis_client = mock_redis

        await redis_cache.close()
        mock_redis.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_no_client(self, redis_cache):
        """Test closing when no client exists."""
        redis_cache.redis_client = None
        await redis_cache.close()
        # Should complete without error


class TestComplianceSecurityIntegration:
    """Test suite for compliance security integration."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock(spec=Config)
        config.security = Mock()
        config.security.enable_encryption = True
        config.security.audit_logging = True
        return config

    @pytest.fixture
    def security_integration(self, mock_config):
        """Create security integration instance."""
        return ComplianceSecurityIntegration(mock_config)

    @pytest.mark.asyncio
    async def test_initialization_success(self, security_integration):
        """Test successful initialization."""
        await security_integration.initialize()
        # Should complete without error

    @pytest.mark.asyncio
    async def test_initialization_error(self, security_integration):
        """Test initialization with error."""
        with patch.object(
            security_integration,
            "_initialize_encryption",
            side_effect=Exception("Init failed"),
        ):
            with pytest.raises(
                IntegrationException, match="Security integration failed"
            ):
                await security_integration.initialize()

    @pytest.mark.asyncio
    async def test_initialization_encryption_disabled(self, mock_config):
        """Test initialization with encryption disabled."""
        mock_config.security.enable_encryption = False
        integration = ComplianceSecurityIntegration(mock_config)
        await integration.initialize()
        # Should complete without error

    @pytest.mark.asyncio
    async def test_initialization_audit_disabled(self, mock_config):
        """Test initialization with audit logging disabled."""
        mock_config.security.audit_logging = False
        integration = ComplianceSecurityIntegration(mock_config)
        await integration.initialize()
        # Should complete without error

    @pytest.mark.asyncio
    async def test_encrypt_sensitive_data_enabled(self, security_integration):
        """Test encrypting data when encryption is enabled."""
        result = await security_integration.encrypt_sensitive_data("test data")
        assert result == "test data"  # Placeholder implementation

    @pytest.mark.asyncio
    async def test_encrypt_sensitive_data_disabled(self, mock_config):
        """Test encrypting data when encryption is disabled."""
        mock_config.security.enable_encryption = False
        integration = ComplianceSecurityIntegration(mock_config)
        result = await integration.encrypt_sensitive_data("test data")
        assert result == "test data"

    @pytest.mark.asyncio
    async def test_decrypt_sensitive_data_enabled(self, security_integration):
        """Test decrypting data when encryption is enabled."""
        result = await security_integration.decrypt_sensitive_data("encrypted data")
        assert result == "encrypted data"  # Placeholder implementation

    @pytest.mark.asyncio
    async def test_decrypt_sensitive_data_disabled(self, mock_config):
        """Test decrypting data when encryption is disabled."""
        mock_config.security.enable_encryption = False
        integration = ComplianceSecurityIntegration(mock_config)
        result = await integration.decrypt_sensitive_data("encrypted data")
        assert result == "encrypted data"

    @pytest.mark.asyncio
    async def test_log_security_event_enabled(self, security_integration):
        """Test logging security event when enabled."""
        await security_integration.log_security_event(
            event_type="access",
            resource="assessment",
            action="read",
            outcome="success",
            details={"user": "test_user"},
        )
        # Should complete without error

    @pytest.mark.asyncio
    async def test_log_security_event_disabled(self, mock_config):
        """Test logging security event when disabled."""
        mock_config.security.audit_logging = False
        integration = ComplianceSecurityIntegration(mock_config)
        await integration.log_security_event(
            event_type="access", resource="assessment", action="read", outcome="success"
        )
        # Should complete without error

    @pytest.mark.asyncio
    async def test_log_security_event_no_details(self, security_integration):
        """Test logging security event without details."""
        await security_integration.log_security_event(
            event_type="access", resource="assessment", action="read", outcome="success"
        )
        # Should complete without error

    @pytest.mark.asyncio
    async def test_log_security_event_error(self, security_integration):
        """Test logging security event with error."""
        with patch(
            "ai_engine.compliance.data_integrations.json.dumps",
            side_effect=Exception("JSON error"),
        ):
            await security_integration.log_security_event(
                event_type="access",
                resource="assessment",
                action="read",
                outcome="success",
            )
            # Should handle error gracefully

    @pytest.mark.asyncio
    async def test_validate_data_access_success(self, security_integration):
        """Test validating data access."""
        result = await security_integration.validate_data_access(
            user="test_user", resource="assessment", action="read"
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_data_access_error(self, security_integration):
        """Test validating data access with error."""
        with patch.object(
            security_integration,
            "log_security_event",
            side_effect=Exception("Log failed"),
        ):
            result = await security_integration.validate_data_access(
                user="test_user", resource="assessment", action="read"
            )
            assert result is False

    @pytest.mark.asyncio
    async def test_close(self, security_integration):
        """Test closing security integration."""
        await security_integration.close()
        # Should complete without error


class TestIntegrationException:
    """Test suite for IntegrationException."""

    def test_exception_creation(self):
        """Test creating IntegrationException."""
        exc = IntegrationException("Test error")
        assert str(exc) == "Test error"

    def test_exception_inheritance(self):
        """Test that IntegrationException inherits from CronosAIException."""
        from ai_engine.core.exceptions import CronosAIException

        exc = IntegrationException("Test error")
        assert isinstance(exc, CronosAIException)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
