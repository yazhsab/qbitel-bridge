"""
Comprehensive Unit Tests for Data Retention Manager
Tests all functionality in ai_engine/compliance/data_retention.py
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from ai_engine.compliance.data_retention import (
    DataRetentionManager,
    DataRetentionException,
    DataCategory,
    RetentionAction,
    RetentionPolicy,
    DataLifecycleRecord,
)
from ai_engine.compliance.audit_trail import AuditTrailManager, EventType, EventSeverity
from ai_engine.core.config import Config


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    return Config()


@pytest.fixture
def mock_audit_manager():
    """Create mock audit manager."""
    manager = Mock(spec=AuditTrailManager)
    manager.record_compliance_event = AsyncMock()
    return manager


@pytest.fixture
def retention_manager(mock_config, mock_audit_manager):
    """Create DataRetentionManager instance."""
    return DataRetentionManager(mock_config, mock_audit_manager)


class TestDataRetentionManagerInitialization:
    """Test DataRetentionManager initialization."""

    def test_initialization(self, retention_manager, mock_config, mock_audit_manager):
        """Test successful initialization."""
        assert retention_manager.config == mock_config
        assert retention_manager.audit_manager == mock_audit_manager
        assert len(retention_manager.policies) == 8
        assert len(retention_manager.lifecycle_records) == 0
        assert retention_manager._running is False

    def test_default_policies_created(self, retention_manager):
        """Test default retention policies are created."""
        assert "audit_logs" in retention_manager.policies
        assert "user_data" in retention_manager.policies
        assert "protocol_analysis" in retention_manager.policies
        assert "compliance_reports" in retention_manager.policies
        assert "system_logs" in retention_manager.policies
        assert "backups" in retention_manager.policies
        assert "metrics" in retention_manager.policies
        assert "traces" in retention_manager.policies

    def test_audit_logs_policy(self, retention_manager):
        """Test audit logs retention policy."""
        policy = retention_manager.policies["audit_logs"]
        assert policy.data_category == DataCategory.AUDIT_LOGS
        assert policy.retention_period_days == 2555  # 7 years
        assert policy.action_after_retention == RetentionAction.ARCHIVE
        assert policy.legal_hold_exempt is False

    def test_user_data_policy(self, retention_manager):
        """Test user data retention policy."""
        policy = retention_manager.policies["user_data"]
        assert policy.data_category == DataCategory.USER_DATA
        assert policy.retention_period_days == 90
        assert policy.action_after_retention == RetentionAction.ANONYMIZE
        assert policy.legal_hold_exempt is True


class TestDataRetentionManagerLifecycle:
    """Test lifecycle management."""

    @pytest.mark.asyncio
    async def test_initialize(self, retention_manager):
        """Test manager initialization."""
        with patch.object(retention_manager, '_load_lifecycle_records', new_callable=AsyncMock):
            await retention_manager.initialize()
            
            assert retention_manager._running is True
            assert retention_manager._enforcement_task is not None

    @pytest.mark.asyncio
    async def test_shutdown(self, retention_manager):
        """Test manager shutdown."""
        with patch.object(retention_manager, '_load_lifecycle_records', new_callable=AsyncMock):
            await retention_manager.initialize()
            await retention_manager.shutdown()
            
            assert retention_manager._running is False


class TestRegisterData:
    """Test data registration."""

    @pytest.mark.asyncio
    async def test_register_data_success(self, retention_manager, mock_audit_manager):
        """Test successful data registration."""
        record_id = await retention_manager.register_data(
            DataCategory.USER_DATA,
            "user_123",
            {"email": "test@example.com"}
        )
        
        assert record_id is not None
        assert record_id in retention_manager.lifecycle_records
        
        record = retention_manager.lifecycle_records[record_id]
        assert record.data_category == DataCategory.USER_DATA
        assert record.data_identifier == "user_123"
        assert record.metadata["email"] == "test@example.com"
        
        mock_audit_manager.record_compliance_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_data_calculates_retention(self, retention_manager):
        """Test retention period calculation."""
        record_id = await retention_manager.register_data(
            DataCategory.SYSTEM_LOGS,
            "log_456"
        )
        
        record = retention_manager.lifecycle_records[record_id]
        policy = retention_manager.policies["system_logs"]
        
        expected_retention = record.created_at + timedelta(days=policy.retention_period_days)
        assert abs((record.retention_until - expected_retention).total_seconds()) < 1

    @pytest.mark.asyncio
    async def test_register_data_with_archive_period(self, retention_manager):
        """Test registration with archive period."""
        record_id = await retention_manager.register_data(
            DataCategory.PROTOCOL_ANALYSIS,
            "analysis_789"
        )
        
        record = retention_manager.lifecycle_records[record_id]
        assert record.archive_at is not None

    @pytest.mark.asyncio
    async def test_register_data_no_policy(self, retention_manager):
        """Test registration with non-existent policy."""
        # Remove a policy
        del retention_manager.policies["user_data"]
        
        with pytest.raises(DataRetentionException, match="No policy found"):
            await retention_manager.register_data(
                DataCategory.USER_DATA,
                "user_123"
            )

    @pytest.mark.asyncio
    async def test_register_data_without_audit_manager(self, mock_config):
        """Test registration without audit manager."""
        manager = DataRetentionManager(mock_config, None)
        
        record_id = await manager.register_data(
            DataCategory.METRICS,
            "metric_001"
        )
        
        assert record_id is not None


class TestLegalHold:
    """Test legal hold functionality."""

    @pytest.mark.asyncio
    async def test_apply_legal_hold(self, retention_manager, mock_audit_manager):
        """Test applying legal hold."""
        record_id = await retention_manager.register_data(
            DataCategory.USER_DATA,
            "user_123"
        )
        
        result = await retention_manager.apply_legal_hold(
            record_id,
            "Pending litigation"
        )
        
        assert result is True
        record = retention_manager.lifecycle_records[record_id]
        assert record.legal_hold is True
        assert record.metadata["legal_hold_reason"] == "Pending litigation"
        assert "legal_hold_applied_at" in record.metadata

    @pytest.mark.asyncio
    async def test_apply_legal_hold_nonexistent_record(self, retention_manager):
        """Test applying legal hold to non-existent record."""
        with pytest.raises(DataRetentionException, match="Record .* not found"):
            await retention_manager.apply_legal_hold("invalid_id", "Test reason")

    @pytest.mark.asyncio
    async def test_release_legal_hold(self, retention_manager, mock_audit_manager):
        """Test releasing legal hold."""
        record_id = await retention_manager.register_data(
            DataCategory.USER_DATA,
            "user_123"
        )
        await retention_manager.apply_legal_hold(record_id, "Test reason")
        
        result = await retention_manager.release_legal_hold(record_id)
        
        assert result is True
        record = retention_manager.lifecycle_records[record_id]
        assert record.legal_hold is False
        assert "legal_hold_released_at" in record.metadata

    @pytest.mark.asyncio
    async def test_release_legal_hold_nonexistent_record(self, retention_manager):
        """Test releasing legal hold on non-existent record."""
        with pytest.raises(DataRetentionException, match="Record .* not found"):
            await retention_manager.release_legal_hold("invalid_id")


class TestEnforceRetentionPolicies:
    """Test retention policy enforcement."""

    @pytest.mark.asyncio
    async def test_enforce_policies_no_records(self, retention_manager):
        """Test enforcement with no records."""
        results = await retention_manager.enforce_retention_policies()
        
        assert results["records_processed"] == 0
        assert results["actions_taken"]["archived"] == 0
        assert results["actions_taken"]["deleted"] == 0

    @pytest.mark.asyncio
    async def test_enforce_policies_not_expired(self, retention_manager):
        """Test enforcement with non-expired records."""
        await retention_manager.register_data(
            DataCategory.METRICS,
            "metric_001"
        )
        
        results = await retention_manager.enforce_retention_policies()
        
        assert results["records_processed"] == 1
        assert results["actions_taken"]["retained"] == 1

    @pytest.mark.asyncio
    async def test_enforce_policies_expired_archive(self, retention_manager):
        """Test enforcement archives expired data."""
        record_id = await retention_manager.register_data(
            DataCategory.AUDIT_LOGS,
            "audit_001"
        )
        
        # Set retention to past
        record = retention_manager.lifecycle_records[record_id]
        record.retention_until = datetime.utcnow() - timedelta(days=1)
        
        results = await retention_manager.enforce_retention_policies()
        
        assert results["actions_taken"]["archived"] == 1
        assert record.action_taken == RetentionAction.ARCHIVE

    @pytest.mark.asyncio
    async def test_enforce_policies_expired_delete(self, retention_manager):
        """Test enforcement deletes expired data."""
        record_id = await retention_manager.register_data(
            DataCategory.METRICS,
            "metric_001"
        )
        
        # Set retention to past
        record = retention_manager.lifecycle_records[record_id]
        record.retention_until = datetime.utcnow() - timedelta(days=1)
        
        results = await retention_manager.enforce_retention_policies()
        
        assert results["actions_taken"]["deleted"] == 1
        assert record.action_taken == RetentionAction.DELETE

    @pytest.mark.asyncio
    async def test_enforce_policies_expired_anonymize(self, retention_manager):
        """Test enforcement anonymizes expired data."""
        record_id = await retention_manager.register_data(
            DataCategory.USER_DATA,
            "user_123"
        )
        
        # Set retention to past
        record = retention_manager.lifecycle_records[record_id]
        record.retention_until = datetime.utcnow() - timedelta(days=1)
        
        results = await retention_manager.enforce_retention_policies()
        
        assert results["actions_taken"]["anonymized"] == 1
        assert record.action_taken == RetentionAction.ANONYMIZE

    @pytest.mark.asyncio
    async def test_enforce_policies_legal_hold_skipped(self, retention_manager):
        """Test enforcement skips records with legal hold."""
        record_id = await retention_manager.register_data(
            DataCategory.USER_DATA,
            "user_123"
        )
        
        # Apply legal hold and set retention to past
        await retention_manager.apply_legal_hold(record_id, "Test")
        record = retention_manager.lifecycle_records[record_id]
        record.retention_until = datetime.utcnow() - timedelta(days=1)
        
        results = await retention_manager.enforce_retention_policies()
        
        assert results["actions_taken"]["retained"] == 1
        assert record.action_taken is None

    @pytest.mark.asyncio
    async def test_enforce_policies_handles_errors(self, retention_manager):
        """Test enforcement handles errors gracefully."""
        record_id = await retention_manager.register_data(
            DataCategory.USER_DATA,
            "user_123"
        )
        
        # Set retention to past
        record = retention_manager.lifecycle_records[record_id]
        record.retention_until = datetime.utcnow() - timedelta(days=1)
        
        # Mock _anonymize_data to raise exception
        with patch.object(retention_manager, '_anonymize_data', side_effect=Exception("Test error")):
            results = await retention_manager.enforce_retention_policies()
            
            assert len(results["errors"]) == 1
            assert results["errors"][0]["record_id"] == record_id


class TestDataActions:
    """Test data action methods."""

    @pytest.mark.asyncio
    async def test_archive_data(self, retention_manager, mock_audit_manager):
        """Test archiving data."""
        record_id = await retention_manager.register_data(
            DataCategory.AUDIT_LOGS,
            "audit_001"
        )
        record = retention_manager.lifecycle_records[record_id]
        
        await retention_manager._archive_data(record)
        
        # Verify audit event was recorded
        calls = mock_audit_manager.record_compliance_event.call_args_list
        assert any("archive_data" in str(call) for call in calls)

    @pytest.mark.asyncio
    async def test_delete_data(self, retention_manager, mock_audit_manager):
        """Test deleting data."""
        record_id = await retention_manager.register_data(
            DataCategory.METRICS,
            "metric_001"
        )
        record = retention_manager.lifecycle_records[record_id]
        
        await retention_manager._delete_data(record)
        
        # Verify audit event was recorded
        calls = mock_audit_manager.record_compliance_event.call_args_list
        assert any("delete_data" in str(call) for call in calls)

    @pytest.mark.asyncio
    async def test_anonymize_data(self, retention_manager, mock_audit_manager):
        """Test anonymizing data."""
        record_id = await retention_manager.register_data(
            DataCategory.USER_DATA,
            "user_123"
        )
        record = retention_manager.lifecycle_records[record_id]
        
        await retention_manager._anonymize_data(record)
        
        # Verify audit event was recorded
        calls = mock_audit_manager.record_compliance_event.call_args_list
        assert any("anonymize_data" in str(call) for call in calls)


class TestReporting:
    """Test reporting functionality."""

    @pytest.mark.asyncio
    async def test_generate_retention_report(self, retention_manager):
        """Test generating retention report."""
        # Register some data
        await retention_manager.register_data(DataCategory.USER_DATA, "user_1")
        await retention_manager.register_data(DataCategory.METRICS, "metric_1")
        
        report = await retention_manager.generate_retention_report()
        
        assert "report_metadata" in report
        assert "policies" in report
        assert "lifecycle_summary" in report
        assert "upcoming_actions" in report
        assert report["report_metadata"]["report_type"] == "Data Retention Compliance"

    @pytest.mark.asyncio
    async def test_report_includes_all_policies(self, retention_manager):
        """Test report includes all policies."""
        report = await retention_manager.generate_retention_report()
        
        assert len(report["policies"]) == 8
        assert "audit_logs" in report["policies"]
        assert "user_data" in report["policies"]

    @pytest.mark.asyncio
    async def test_report_lifecycle_summary(self, retention_manager):
        """Test report lifecycle summary."""
        await retention_manager.register_data(DataCategory.USER_DATA, "user_1")
        
        report = await retention_manager.generate_retention_report()
        
        assert DataCategory.USER_DATA.value in report["lifecycle_summary"]
        summary = report["lifecycle_summary"][DataCategory.USER_DATA.value]
        assert summary["total_records"] == 1
        assert summary["active"] == 1

    @pytest.mark.asyncio
    async def test_report_upcoming_actions(self, retention_manager):
        """Test report shows upcoming actions."""
        record_id = await retention_manager.register_data(
            DataCategory.METRICS,
            "metric_1"
        )
        
        # Set retention to near future
        record = retention_manager.lifecycle_records[record_id]
        record.retention_until = datetime.utcnow() + timedelta(days=15)
        
        report = await retention_manager.generate_retention_report()
        
        assert len(report["upcoming_actions"]) == 1
        assert report["upcoming_actions"][0]["record_id"] == record_id

    def test_get_statistics(self, retention_manager):
        """Test getting statistics."""
        stats = retention_manager.get_statistics()
        
        assert "total_policies" in stats
        assert "total_records" in stats
        assert "by_category" in stats
        assert "by_action" in stats
        assert "legal_holds" in stats
        assert stats["total_policies"] == 8


class TestEnforcementLoop:
    """Test background enforcement loop."""

    @pytest.mark.asyncio
    async def test_enforcement_loop_runs(self, retention_manager):
        """Test enforcement loop executes."""
        with patch.object(retention_manager, '_load_lifecycle_records', new_callable=AsyncMock), \
             patch.object(retention_manager, 'enforce_retention_policies', new_callable=AsyncMock) as mock_enforce:
            
            await retention_manager.initialize()
            
            # Wait briefly for loop to potentially run
            await asyncio.sleep(0.1)
            
            await retention_manager.shutdown()
            
            # Loop should be set up (actual execution would take 24 hours)
            assert retention_manager._enforcement_task is not None

    @pytest.mark.asyncio
    async def test_enforcement_loop_handles_errors(self, retention_manager):
        """Test enforcement loop handles errors."""
        with patch.object(retention_manager, '_load_lifecycle_records', new_callable=AsyncMock), \
             patch.object(retention_manager, 'enforce_retention_policies', 
                         new_callable=AsyncMock, side_effect=Exception("Test error")):
            
            await retention_manager.initialize()
            await asyncio.sleep(0.1)
            await retention_manager.shutdown()
            
            # Should not crash


class TestDataClasses:
    """Test data classes."""

    def test_retention_policy_creation(self):
        """Test creating RetentionPolicy."""
        policy = RetentionPolicy(
            policy_id="test_policy",
            data_category=DataCategory.USER_DATA,
            retention_period_days=90,
            action_after_retention=RetentionAction.DELETE
        )
        
        assert policy.policy_id == "test_policy"
        assert policy.data_category == DataCategory.USER_DATA
        assert policy.retention_period_days == 90

    def test_data_lifecycle_record_creation(self):
        """Test creating DataLifecycleRecord."""
        now = datetime.utcnow()
        record = DataLifecycleRecord(
            record_id="rec_123",
            data_category=DataCategory.METRICS,
            data_identifier="metric_001",
            created_at=now,
            retention_until=now + timedelta(days=90)
        )
        
        assert record.record_id == "rec_123"
        assert record.data_category == DataCategory.METRICS
        assert record.legal_hold is False


class TestEnums:
    """Test enum classes."""

    def test_data_category_values(self):
        """Test DataCategory enum values."""
        assert DataCategory.AUDIT_LOGS.value == "audit_logs"
        assert DataCategory.USER_DATA.value == "user_data"
        assert DataCategory.PROTOCOL_ANALYSIS.value == "protocol_analysis"

    def test_retention_action_values(self):
        """Test RetentionAction enum values."""
        assert RetentionAction.ARCHIVE.value == "archive"
        assert RetentionAction.DELETE.value == "delete"
        assert RetentionAction.ANONYMIZE.value == "anonymize"
        assert RetentionAction.RETAIN.value == "retain"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_multiple_registrations_same_identifier(self, retention_manager):
        """Test registering multiple records with same identifier."""
        record_id1 = await retention_manager.register_data(
            DataCategory.USER_DATA,
            "user_123"
        )
        record_id2 = await retention_manager.register_data(
            DataCategory.USER_DATA,
            "user_123"
        )
        
        assert record_id1 != record_id2
        assert len(retention_manager.lifecycle_records) == 2

    @pytest.mark.asyncio
    async def test_enforce_with_no_policy(self, retention_manager):
        """Test enforcement when policy is missing."""
        record_id = await retention_manager.register_data(
            DataCategory.METRICS,
            "metric_001"
        )
        
        # Remove policy
        del retention_manager.policies["metrics"]
        
        # Set retention to past
        record = retention_manager.lifecycle_records[record_id]
        record.retention_until = datetime.utcnow() - timedelta(days=1)
        
        # Should handle gracefully
        results = await retention_manager.enforce_retention_policies()
        assert results["records_processed"] == 1

    @pytest.mark.asyncio
    async def test_empty_metadata(self, retention_manager):
        """Test registration with empty metadata."""
        record_id = await retention_manager.register_data(
            DataCategory.METRICS,
            "metric_001",
            {}
        )
        
        record = retention_manager.lifecycle_records[record_id]
        assert record.metadata == {}