"""
Comprehensive tests for Data Retention Policy Implementation.

Tests cover:
- Retention policy creation and management
- Data lifecycle tracking
- Automated retention enforcement
- Legal holds
- Archive and deletion operations
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import Dict, List

from ai_engine.compliance.data_retention import (
    DataRetentionManager,
    RetentionPolicy,
    DataLifecycleRecord,
    DataCategory,
    RetentionAction,
    DataRetentionException,
)
from ai_engine.core.config import Config


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = Mock(spec=Config)
    config.data_retention_check_interval = 3600
    config.data_retention_enabled = True
    return config


@pytest.fixture
def mock_audit_manager():
    """Create mock audit trail manager."""
    manager = AsyncMock()
    manager.log_event = AsyncMock()
    return manager


@pytest.fixture
def retention_manager(mock_config, mock_audit_manager):
    """Create DataRetentionManager instance."""
    with patch(
        "ai_engine.compliance.data_retention.AuditTrailManager",
        return_value=mock_audit_manager,
    ):
        manager = DataRetentionManager(mock_config)
        manager.audit_manager = mock_audit_manager
        return manager


@pytest.fixture
def sample_policy():
    """Create sample retention policy."""
    return RetentionPolicy(
        policy_id="POL001",
        data_category=DataCategory.AUDIT_LOGS,
        retention_period_days=90,
        archive_period_days=30,
        action_after_retention=RetentionAction.ARCHIVE,
        description="Audit log retention policy",
    )


class TestRetentionPolicy:
    """Tests for RetentionPolicy dataclass."""

    def test_policy_creation(self, sample_policy):
        """Test retention policy creation."""
        assert sample_policy.policy_id == "POL001"
        assert sample_policy.data_category == DataCategory.AUDIT_LOGS
        assert sample_policy.retention_period_days == 90
        assert sample_policy.archive_period_days == 30
        assert sample_policy.action_after_retention == RetentionAction.ARCHIVE
        assert isinstance(sample_policy.created_at, datetime)

    def test_policy_defaults(self):
        """Test policy default values."""
        policy = RetentionPolicy(
            policy_id="POL002",
            data_category=DataCategory.USER_DATA,
            retention_period_days=365,
        )
        assert policy.archive_period_days is None
        assert policy.action_after_retention == RetentionAction.DELETE
        assert policy.legal_hold_exempt is False
        assert policy.description == ""


class TestDataLifecycleRecord:
    """Tests for DataLifecycleRecord."""

    def test_record_creation(self):
        """Test lifecycle record creation."""
        now = datetime.utcnow()
        retention_date = now + timedelta(days=90)

        record = DataLifecycleRecord(
            record_id="REC001",
            data_category=DataCategory.PROTOCOL_ANALYSIS,
            data_identifier="analysis_12345",
            created_at=now,
            retention_until=retention_date,
        )

        assert record.record_id == "REC001"
        assert record.data_category == DataCategory.PROTOCOL_ANALYSIS
        assert record.action_taken is None
        assert record.legal_hold is False
        assert isinstance(record.metadata, dict)


class TestDataRetentionManager:
    """Tests for DataRetentionManager."""

    def test_manager_initialization(self, retention_manager, mock_config):
        """Test manager initialization."""
        assert retention_manager.config == mock_config
        assert isinstance(retention_manager.policies, dict)
        assert isinstance(retention_manager.lifecycle_records, dict)

    @pytest.mark.asyncio
    async def test_add_retention_policy(self, retention_manager, sample_policy):
        """Test adding a retention policy."""
        await retention_manager.add_policy(sample_policy)

        assert sample_policy.data_category in retention_manager.policies
        assert retention_manager.policies[sample_policy.data_category] == sample_policy

    @pytest.mark.asyncio
    async def test_get_policy_for_category(self, retention_manager, sample_policy):
        """Test retrieving policy for data category."""
        await retention_manager.add_policy(sample_policy)

        policy = retention_manager.get_policy(DataCategory.AUDIT_LOGS)
        assert policy == sample_policy

    @pytest.mark.asyncio
    async def test_get_policy_not_found(self, retention_manager):
        """Test retrieving non-existent policy."""
        policy = retention_manager.get_policy(DataCategory.USER_DATA)
        assert policy is None

    @pytest.mark.asyncio
    async def test_register_data(self, retention_manager, sample_policy):
        """Test registering data for retention tracking."""
        await retention_manager.add_policy(sample_policy)

        record_id = await retention_manager.register_data(
            data_category=DataCategory.AUDIT_LOGS,
            data_identifier="audit_001",
            metadata={"source": "api"},
        )

        assert record_id is not None
        assert record_id in retention_manager.lifecycle_records

    @pytest.mark.asyncio
    async def test_apply_legal_hold(self, retention_manager):
        """Test applying legal hold to data."""
        # Register data first
        record_id = await retention_manager.register_data(
            data_category=DataCategory.USER_DATA, data_identifier="user_data_001"
        )

        # Apply legal hold
        result = await retention_manager.apply_legal_hold(record_id, reason="Litigation")

        assert result is True
        record = retention_manager.lifecycle_records[record_id]
        assert record.legal_hold is True

    @pytest.mark.asyncio
    async def test_release_legal_hold(self, retention_manager):
        """Test releasing legal hold."""
        record_id = await retention_manager.register_data(
            data_category=DataCategory.USER_DATA, data_identifier="user_data_002"
        )

        await retention_manager.apply_legal_hold(record_id, reason="Investigation")
        result = await retention_manager.release_legal_hold(record_id)

        assert result is True
        record = retention_manager.lifecycle_records[record_id]
        assert record.legal_hold is False

    @pytest.mark.asyncio
    async def test_check_retention_enforcement(self, retention_manager, sample_policy):
        """Test retention enforcement check."""
        await retention_manager.add_policy(sample_policy)

        # Register expired data
        old_date = datetime.utcnow() - timedelta(days=100)
        record = DataLifecycleRecord(
            record_id="REC_EXPIRED",
            data_category=DataCategory.AUDIT_LOGS,
            data_identifier="old_audit",
            created_at=old_date,
            retention_until=old_date + timedelta(days=90),
        )
        retention_manager.lifecycle_records["REC_EXPIRED"] = record

        # Run enforcement
        enforced = await retention_manager.enforce_retention()

        assert len(enforced) > 0

    @pytest.mark.asyncio
    async def test_archive_data(self, retention_manager):
        """Test data archival."""
        record_id = await retention_manager.register_data(data_category=DataCategory.BACKUPS, data_identifier="backup_001")

        with patch.object(retention_manager, "_archive_data_impl", new_callable=AsyncMock) as mock_archive:
            mock_archive.return_value = True

            result = await retention_manager.archive_data(record_id)

            assert result is True
            mock_archive.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_data(self, retention_manager):
        """Test data deletion."""
        record_id = await retention_manager.register_data(data_category=DataCategory.SYSTEM_LOGS, data_identifier="log_001")

        with patch.object(retention_manager, "_delete_data_impl", new_callable=AsyncMock) as mock_delete:
            mock_delete.return_value = True

            result = await retention_manager.delete_data(record_id)

            assert result is True
            mock_delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_anonymize_data(self, retention_manager):
        """Test data anonymization."""
        record_id = await retention_manager.register_data(data_category=DataCategory.USER_DATA, data_identifier="user_001")

        with patch.object(retention_manager, "_anonymize_data_impl", new_callable=AsyncMock) as mock_anon:
            mock_anon.return_value = True

            result = await retention_manager.anonymize_data(record_id)

            assert result is True
            mock_anon.assert_called_once()

    @pytest.mark.asyncio
    async def test_legal_hold_prevents_deletion(self, retention_manager):
        """Test that legal hold prevents data deletion."""
        record_id = await retention_manager.register_data(
            data_category=DataCategory.USER_DATA, data_identifier="protected_data"
        )

        await retention_manager.apply_legal_hold(record_id, reason="Legal case")

        with pytest.raises(DataRetentionException, match="legal hold"):
            await retention_manager.delete_data(record_id)

    @pytest.mark.asyncio
    async def test_get_retention_statistics(self, retention_manager, sample_policy):
        """Test getting retention statistics."""
        await retention_manager.add_policy(sample_policy)

        # Register some data
        for i in range(5):
            await retention_manager.register_data(data_category=DataCategory.AUDIT_LOGS, data_identifier=f"audit_{i}")

        stats = await retention_manager.get_statistics()

        assert "total_records" in stats
        assert "policies_count" in stats
        assert stats["total_records"] >= 5

    @pytest.mark.asyncio
    async def test_list_expired_data(self, retention_manager, sample_policy):
        """Test listing expired data."""
        await retention_manager.add_policy(sample_policy)

        # Create expired record
        old_date = datetime.utcnow() - timedelta(days=100)
        record = DataLifecycleRecord(
            record_id="EXP001",
            data_category=DataCategory.AUDIT_LOGS,
            data_identifier="expired_data",
            created_at=old_date,
            retention_until=old_date + timedelta(days=90),
        )
        retention_manager.lifecycle_records["EXP001"] = record

        expired = await retention_manager.list_expired_data()

        assert len(expired) >= 1
        assert any(r.record_id == "EXP001" for r in expired)

    def test_data_category_enum(self):
        """Test DataCategory enum values."""
        assert DataCategory.AUDIT_LOGS == "audit_logs"
        assert DataCategory.USER_DATA == "user_data"
        assert DataCategory.COMPLIANCE_REPORTS == "compliance_reports"

    def test_retention_action_enum(self):
        """Test RetentionAction enum values."""
        assert RetentionAction.ARCHIVE == "archive"
        assert RetentionAction.DELETE == "delete"
        assert RetentionAction.ANONYMIZE == "anonymize"
        assert RetentionAction.RETAIN == "retain"
