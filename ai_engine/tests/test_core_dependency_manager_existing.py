"""
Comprehensive tests for ai_engine.core.dependency_manager module.

This module provides dependency management, injection, and lifecycle
management for the QBITEL Engine components.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, Optional, List
import inspect

from ai_engine.core.dependency_manager import (
    DependencyStatus,
    DependencyInfo,
    DependencyManager,
)


class TestDependencyStatus:
    """Test DependencyStatus enum."""

    def test_dependency_status_values(self):
        """Test DependencyStatus enum values."""
        assert DependencyStatus.HEALTHY.value == "healthy"
        assert DependencyStatus.UNHEALTHY.value == "unhealthy"
        assert DependencyStatus.UNKNOWN.value == "unknown"
        assert DependencyStatus.DEGRADED.value == "degraded"


class TestDependencyInfo:
    """Test DependencyInfo class."""

    def test_initialization(self):
        """Test DependencyInfo initialization."""
        info = DependencyInfo(
            name="test_service",
            status=DependencyStatus.HEALTHY,
            version="1.0.0",
            description="Test service description",
        )

        assert info.name == "test_service"
        assert info.status == DependencyStatus.HEALTHY
        assert info.version == "1.0.0"
        assert info.description == "Test service description"
        assert info.last_check is not None
        assert info.metadata == {}

    def test_initialization_with_metadata(self):
        """Test DependencyInfo initialization with metadata."""
        metadata = {"endpoint": "http://localhost:8080", "timeout": 30}
        info = DependencyInfo(
            name="test_service",
            status=DependencyStatus.HEALTHY,
            version="1.0.0",
            description="Test service description",
            metadata=metadata,
        )

        assert info.metadata == metadata

    def test_initialization_with_custom_timestamp(self):
        """Test DependencyInfo initialization with custom timestamp."""
        from datetime import datetime

        custom_time = datetime.now()

        info = DependencyInfo(
            name="test_service",
            status=DependencyStatus.HEALTHY,
            version="1.0.0",
            description="Test service description",
            last_check=custom_time,
        )

        assert info.last_check == custom_time

    def test_to_dict(self):
        """Test converting DependencyInfo to dictionary."""
        info = DependencyInfo(
            name="test_service",
            status=DependencyStatus.HEALTHY,
            version="1.0.0",
            description="Test service description",
            metadata={"key": "value"},
        )

        info_dict = info.to_dict()

        assert info_dict["name"] == "test_service"
        assert info_dict["status"] == "healthy"
        assert info_dict["version"] == "1.0.0"
        assert info_dict["description"] == "Test service description"
        assert info_dict["metadata"]["key"] == "value"
        assert "last_check" in info_dict

    def test_from_dict(self):
        """Test creating DependencyInfo from dictionary."""
        from datetime import datetime

        info_dict = {
            "name": "test_service",
            "status": "healthy",
            "version": "1.0.0",
            "description": "Test service description",
            "metadata": {"key": "value"},
            "last_check": datetime.now().isoformat(),
        }

        info = DependencyInfo.from_dict(info_dict)

        assert info.name == "test_service"
        assert info.status == DependencyStatus.HEALTHY
        assert info.version == "1.0.0"
        assert info.description == "Test service description"
        assert info.metadata["key"] == "value"

    def test_update_status(self):
        """Test updating dependency status."""
        info = DependencyInfo(
            name="test_service",
            status=DependencyStatus.HEALTHY,
            version="1.0.0",
            description="Test service description",
        )

        original_time = info.last_check

        info.update_status(DependencyStatus.UNHEALTHY)

        assert info.status == DependencyStatus.UNHEALTHY
        assert info.last_check > original_time

    def test_update_metadata(self):
        """Test updating dependency metadata."""
        info = DependencyInfo(
            name="test_service",
            status=DependencyStatus.HEALTHY,
            version="1.0.0",
            description="Test service description",
        )

        info.update_metadata({"new_key": "new_value"})

        assert info.metadata["new_key"] == "new_value"

    def test_is_healthy(self):
        """Test checking if dependency is healthy."""
        info = DependencyInfo(
            name="test_service",
            status=DependencyStatus.HEALTHY,
            version="1.0.0",
            description="Test service description",
        )

        assert info.is_healthy() is True

        info.status = DependencyStatus.UNHEALTHY
        assert info.is_healthy() is False

    def test_is_unhealthy(self):
        """Test checking if dependency is unhealthy."""
        info = DependencyInfo(
            name="test_service",
            status=DependencyStatus.UNHEALTHY,
            version="1.0.0",
            description="Test service description",
        )

        assert info.is_unhealthy() is True

        info.status = DependencyStatus.HEALTHY
        assert info.is_unhealthy() is False

    def test_is_unknown(self):
        """Test checking if dependency status is unknown."""
        info = DependencyInfo(
            name="test_service",
            status=DependencyStatus.UNKNOWN,
            version="1.0.0",
            description="Test service description",
        )

        assert info.is_unknown() is True

        info.status = DependencyStatus.HEALTHY
        assert info.is_unknown() is False

    def test_is_degraded(self):
        """Test checking if dependency is degraded."""
        info = DependencyInfo(
            name="test_service",
            status=DependencyStatus.DEGRADED,
            version="1.0.0",
            description="Test service description",
        )

        assert info.is_degraded() is True

        info.status = DependencyStatus.HEALTHY
        assert info.is_degraded() is False


class TestDependencyManager:
    """Test DependencyManager class."""

    @pytest.fixture
    def dependency_manager(self):
        """Create DependencyManager instance."""
        return DependencyManager()

    def test_initialization(self, dependency_manager):
        """Test DependencyManager initialization."""
        assert dependency_manager._dependencies == {}
        assert dependency_manager._health_checks == {}
        assert dependency_manager._start_time is not None

    def test_register_dependency(self, dependency_manager):
        """Test registering dependency."""
        info = DependencyInfo(
            name="test_service",
            status=DependencyStatus.HEALTHY,
            version="1.0.0",
            description="Test service description",
        )

        dependency_manager.register_dependency(info)

        assert "test_service" in dependency_manager._dependencies
        assert dependency_manager._dependencies["test_service"] == info

    def test_register_dependency_duplicate(self, dependency_manager):
        """Test registering duplicate dependency."""
        info1 = DependencyInfo(
            name="test_service",
            status=DependencyStatus.HEALTHY,
            version="1.0.0",
            description="Test service description",
        )
        info2 = DependencyInfo(
            name="test_service",
            status=DependencyStatus.UNHEALTHY,
            version="2.0.0",
            description="Updated test service description",
        )

        dependency_manager.register_dependency(info1)

        with pytest.raises(ValueError, match="Dependency already registered"):
            dependency_manager.register_dependency(info2)

    def test_unregister_dependency(self, dependency_manager):
        """Test unregistering dependency."""
        info = DependencyInfo(
            name="test_service",
            status=DependencyStatus.HEALTHY,
            version="1.0.0",
            description="Test service description",
        )

        dependency_manager.register_dependency(info)
        dependency_manager.unregister_dependency("test_service")

        assert "test_service" not in dependency_manager._dependencies

    def test_unregister_dependency_not_found(self, dependency_manager):
        """Test unregistering non-existent dependency."""
        with pytest.raises(ValueError, match="Dependency not found"):
            dependency_manager.unregister_dependency("nonexistent_service")

    def test_get_dependency(self, dependency_manager):
        """Test getting dependency."""
        info = DependencyInfo(
            name="test_service",
            status=DependencyStatus.HEALTHY,
            version="1.0.0",
            description="Test service description",
        )

        dependency_manager.register_dependency(info)

        retrieved = dependency_manager.get_dependency("test_service")

        assert retrieved == info

    def test_get_dependency_not_found(self, dependency_manager):
        """Test getting non-existent dependency."""
        with pytest.raises(ValueError, match="Dependency not found"):
            dependency_manager.get_dependency("nonexistent_service")

    def test_list_dependencies(self, dependency_manager):
        """Test listing all dependencies."""
        info1 = DependencyInfo(
            name="service1",
            status=DependencyStatus.HEALTHY,
            version="1.0.0",
            description="Service 1 description",
        )
        info2 = DependencyInfo(
            name="service2",
            status=DependencyStatus.UNHEALTHY,
            version="2.0.0",
            description="Service 2 description",
        )

        dependency_manager.register_dependency(info1)
        dependency_manager.register_dependency(info2)

        dependencies = dependency_manager.list_dependencies()

        assert len(dependencies) == 2
        assert "service1" in dependencies
        assert "service2" in dependencies

    def test_update_dependency_status(self, dependency_manager):
        """Test updating dependency status."""
        info = DependencyInfo(
            name="test_service",
            status=DependencyStatus.HEALTHY,
            version="1.0.0",
            description="Test service description",
        )

        dependency_manager.register_dependency(info)

        original_time = info.last_check

        dependency_manager.update_dependency_status(
            "test_service", DependencyStatus.UNHEALTHY
        )

        updated_info = dependency_manager.get_dependency("test_service")
        assert updated_info.status == DependencyStatus.UNHEALTHY
        assert updated_info.last_check > original_time

    def test_update_dependency_status_not_found(self, dependency_manager):
        """Test updating status of non-existent dependency."""
        with pytest.raises(ValueError, match="Dependency not found"):
            dependency_manager.update_dependency_status(
                "nonexistent_service", DependencyStatus.HEALTHY
            )

    def test_update_dependency_metadata(self, dependency_manager):
        """Test updating dependency metadata."""
        info = DependencyInfo(
            name="test_service",
            status=DependencyStatus.HEALTHY,
            version="1.0.0",
            description="Test service description",
        )

        dependency_manager.register_dependency(info)

        new_metadata = {"endpoint": "http://localhost:8080", "timeout": 30}
        dependency_manager.update_dependency_metadata("test_service", new_metadata)

        updated_info = dependency_manager.get_dependency("test_service")
        assert updated_info.metadata == new_metadata

    def test_update_dependency_metadata_not_found(self, dependency_manager):
        """Test updating metadata of non-existent dependency."""
        with pytest.raises(ValueError, match="Dependency not found"):
            dependency_manager.update_dependency_metadata(
                "nonexistent_service", {"key": "value"}
            )

    def test_register_health_check(self, dependency_manager):
        """Test registering health check."""

        def health_check():
            return DependencyStatus.HEALTHY

        dependency_manager.register_health_check("test_service", health_check)

        assert "test_service" in dependency_manager._health_checks
        assert dependency_manager._health_checks["test_service"] == health_check

    def test_register_health_check_duplicate(self, dependency_manager):
        """Test registering duplicate health check."""

        def health_check1():
            return DependencyStatus.HEALTHY

        def health_check2():
            return DependencyStatus.UNHEALTHY

        dependency_manager.register_health_check("test_service", health_check1)

        with pytest.raises(ValueError, match="Health check already registered"):
            dependency_manager.register_health_check("test_service", health_check2)

    def test_unregister_health_check(self, dependency_manager):
        """Test unregistering health check."""

        def health_check():
            return DependencyStatus.HEALTHY

        dependency_manager.register_health_check("test_service", health_check)
        dependency_manager.unregister_health_check("test_service")

        assert "test_service" not in dependency_manager._health_checks

    def test_unregister_health_check_not_found(self, dependency_manager):
        """Test unregistering non-existent health check."""
        with pytest.raises(ValueError, match="Health check not found"):
            dependency_manager.unregister_health_check("nonexistent_service")

    def test_check_dependency_health(self, dependency_manager):
        """Test checking dependency health."""
        info = DependencyInfo(
            name="test_service",
            status=DependencyStatus.HEALTHY,
            version="1.0.0",
            description="Test service description",
        )

        dependency_manager.register_dependency(info)

        def health_check():
            return DependencyStatus.HEALTHY

        dependency_manager.register_health_check("test_service", health_check)

        status = dependency_manager.check_dependency_health("test_service")

        assert status == DependencyStatus.HEALTHY

    def test_check_dependency_health_not_found(self, dependency_manager):
        """Test checking health of non-existent dependency."""
        with pytest.raises(ValueError, match="Dependency not found"):
            dependency_manager.check_dependency_health("nonexistent_service")

    def test_check_dependency_health_no_health_check(self, dependency_manager):
        """Test checking health of dependency without health check."""
        info = DependencyInfo(
            name="test_service",
            status=DependencyStatus.HEALTHY,
            version="1.0.0",
            description="Test service description",
        )

        dependency_manager.register_dependency(info)

        status = dependency_manager.check_dependency_health("test_service")

        assert status == DependencyStatus.UNKNOWN

    def test_check_dependency_health_exception(self, dependency_manager):
        """Test checking health of dependency with exception in health check."""
        info = DependencyInfo(
            name="test_service",
            status=DependencyStatus.HEALTHY,
            version="1.0.0",
            description="Test service description",
        )

        dependency_manager.register_dependency(info)

        def health_check():
            raise Exception("Health check failed")

        dependency_manager.register_health_check("test_service", health_check)

        status = dependency_manager.check_dependency_health("test_service")

        assert status == DependencyStatus.UNHEALTHY

    def test_check_all_dependencies_health(self, dependency_manager):
        """Test checking health of all dependencies."""
        info1 = DependencyInfo(
            name="service1",
            status=DependencyStatus.HEALTHY,
            version="1.0.0",
            description="Service 1 description",
        )
        info2 = DependencyInfo(
            name="service2",
            status=DependencyStatus.UNHEALTHY,
            version="2.0.0",
            description="Service 2 description",
        )

        dependency_manager.register_dependency(info1)
        dependency_manager.register_dependency(info2)

        def health_check1():
            return DependencyStatus.HEALTHY

        def health_check2():
            return DependencyStatus.UNHEALTHY

        dependency_manager.register_health_check("service1", health_check1)
        dependency_manager.register_health_check("service2", health_check2)

        health_statuses = dependency_manager.check_all_dependencies_health()

        assert len(health_statuses) == 2
        assert health_statuses["service1"] == DependencyStatus.HEALTHY
        assert health_statuses["service2"] == DependencyStatus.UNHEALTHY

    def test_get_healthy_dependencies(self, dependency_manager):
        """Test getting healthy dependencies."""
        info1 = DependencyInfo(
            name="service1",
            status=DependencyStatus.HEALTHY,
            version="1.0.0",
            description="Service 1 description",
        )
        info2 = DependencyInfo(
            name="service2",
            status=DependencyStatus.UNHEALTHY,
            version="2.0.0",
            description="Service 2 description",
        )
        info3 = DependencyInfo(
            name="service3",
            status=DependencyStatus.HEALTHY,
            version="3.0.0",
            description="Service 3 description",
        )

        dependency_manager.register_dependency(info1)
        dependency_manager.register_dependency(info2)
        dependency_manager.register_dependency(info3)

        healthy_deps = dependency_manager.get_healthy_dependencies()

        assert len(healthy_deps) == 2
        assert "service1" in healthy_deps
        assert "service3" in healthy_deps
        assert "service2" not in healthy_deps

    def test_get_unhealthy_dependencies(self, dependency_manager):
        """Test getting unhealthy dependencies."""
        info1 = DependencyInfo(
            name="service1",
            status=DependencyStatus.HEALTHY,
            version="1.0.0",
            description="Service 1 description",
        )
        info2 = DependencyInfo(
            name="service2",
            status=DependencyStatus.UNHEALTHY,
            version="2.0.0",
            description="Service 2 description",
        )
        info3 = DependencyInfo(
            name="service3",
            status=DependencyStatus.DEGRADED,
            version="3.0.0",
            description="Service 3 description",
        )

        dependency_manager.register_dependency(info1)
        dependency_manager.register_dependency(info2)
        dependency_manager.register_dependency(info3)

        unhealthy_deps = dependency_manager.get_unhealthy_dependencies()

        assert len(unhealthy_deps) == 2
        assert "service2" in unhealthy_deps
        assert "service3" in unhealthy_deps
        assert "service1" not in unhealthy_deps

    def test_get_dependencies_by_status(self, dependency_manager):
        """Test getting dependencies by status."""
        info1 = DependencyInfo(
            name="service1",
            status=DependencyStatus.HEALTHY,
            version="1.0.0",
            description="Service 1 description",
        )
        info2 = DependencyInfo(
            name="service2",
            status=DependencyStatus.UNHEALTHY,
            version="2.0.0",
            description="Service 2 description",
        )
        info3 = DependencyInfo(
            name="service3",
            status=DependencyStatus.HEALTHY,
            version="3.0.0",
            description="Service 3 description",
        )

        dependency_manager.register_dependency(info1)
        dependency_manager.register_dependency(info2)
        dependency_manager.register_dependency(info3)

        healthy_deps = dependency_manager.get_dependencies_by_status(
            DependencyStatus.HEALTHY
        )
        unhealthy_deps = dependency_manager.get_dependencies_by_status(
            DependencyStatus.UNHEALTHY
        )

        assert len(healthy_deps) == 2
        assert "service1" in healthy_deps
        assert "service3" in healthy_deps

        assert len(unhealthy_deps) == 1
        assert "service2" in unhealthy_deps

    def test_get_dependency_statistics(self, dependency_manager):
        """Test getting dependency statistics."""
        info1 = DependencyInfo(
            name="service1",
            status=DependencyStatus.HEALTHY,
            version="1.0.0",
            description="Service 1 description",
        )
        info2 = DependencyInfo(
            name="service2",
            status=DependencyStatus.UNHEALTHY,
            version="2.0.0",
            description="Service 2 description",
        )
        info3 = DependencyInfo(
            name="service3",
            status=DependencyStatus.HEALTHY,
            version="3.0.0",
            description="Service 3 description",
        )

        dependency_manager.register_dependency(info1)
        dependency_manager.register_dependency(info2)
        dependency_manager.register_dependency(info3)

        stats = dependency_manager.get_dependency_statistics()

        assert stats["total_dependencies"] == 3
        assert stats["healthy_dependencies"] == 2
        assert stats["unhealthy_dependencies"] == 1
        assert stats["health_percentage"] == 2 / 3 * 100

    def test_clear_dependencies(self, dependency_manager):
        """Test clearing all dependencies."""
        info1 = DependencyInfo(
            name="service1",
            status=DependencyStatus.HEALTHY,
            version="1.0.0",
            description="Service 1 description",
        )
        info2 = DependencyInfo(
            name="service2",
            status=DependencyStatus.UNHEALTHY,
            version="2.0.0",
            description="Service 2 description",
        )

        dependency_manager.register_dependency(info1)
        dependency_manager.register_dependency(info2)

        assert len(dependency_manager._dependencies) == 2

        dependency_manager.clear_dependencies()

        assert len(dependency_manager._dependencies) == 0

    def test_clear_health_checks(self, dependency_manager):
        """Test clearing all health checks."""

        def health_check1():
            return DependencyStatus.HEALTHY

        def health_check2():
            return DependencyStatus.UNHEALTHY

        dependency_manager.register_health_check("service1", health_check1)
        dependency_manager.register_health_check("service2", health_check2)

        assert len(dependency_manager._health_checks) == 2

        dependency_manager.clear_health_checks()

        assert len(dependency_manager._health_checks) == 0

    def test_get_uptime(self, dependency_manager):
        """Test getting dependency manager uptime."""
        uptime = dependency_manager.get_uptime()

        assert uptime > 0
        assert isinstance(uptime, float)

    def test_export_dependencies(self, dependency_manager):
        """Test exporting dependencies."""
        info1 = DependencyInfo(
            name="service1",
            status=DependencyStatus.HEALTHY,
            version="1.0.0",
            description="Service 1 description",
        )
        info2 = DependencyInfo(
            name="service2",
            status=DependencyStatus.UNHEALTHY,
            version="2.0.0",
            description="Service 2 description",
        )

        dependency_manager.register_dependency(info1)
        dependency_manager.register_dependency(info2)

        exported = dependency_manager.export_dependencies()

        assert len(exported) == 2
        assert "service1" in exported
        assert "service2" in exported
        assert exported["service1"]["status"] == "healthy"
        assert exported["service2"]["status"] == "unhealthy"

    def test_import_dependencies(self, dependency_manager):
        """Test importing dependencies."""
        dependencies_data = {
            "service1": {
                "name": "service1",
                "status": "healthy",
                "version": "1.0.0",
                "description": "Service 1 description",
                "metadata": {"key": "value"},
            },
            "service2": {
                "name": "service2",
                "status": "unhealthy",
                "version": "2.0.0",
                "description": "Service 2 description",
                "metadata": {},
            },
        }

        dependency_manager.import_dependencies(dependencies_data)

        assert len(dependency_manager._dependencies) == 2
        assert "service1" in dependency_manager._dependencies
        assert "service2" in dependency_manager._dependencies

        service1 = dependency_manager.get_dependency("service1")
        assert service1.status == DependencyStatus.HEALTHY
        assert service1.version == "1.0.0"
        assert service1.metadata["key"] == "value"

    def test_import_dependencies_invalid_data(self, dependency_manager):
        """Test importing dependencies with invalid data."""
        invalid_data = "invalid_data"

        with pytest.raises(ValueError, match="Dependencies data must be a dictionary"):
            dependency_manager.import_dependencies(invalid_data)

    def test_validate_dependencies(self, dependency_manager):
        """Test validating all dependencies."""
        info1 = DependencyInfo(
            name="service1",
            status=DependencyStatus.HEALTHY,
            version="1.0.0",
            description="Service 1 description",
        )
        info2 = DependencyInfo(
            name="service2",
            status=DependencyStatus.UNHEALTHY,
            version="2.0.0",
            description="Service 2 description",
        )

        dependency_manager.register_dependency(info1)
        dependency_manager.register_dependency(info2)

        # Should not raise exception
        dependency_manager.validate_dependencies()

    def test_validate_dependencies_invalid(self, dependency_manager):
        """Test validating dependencies with invalid data."""
        # Create invalid dependency info
        info = DependencyInfo(
            name="",  # Invalid empty name
            status=DependencyStatus.HEALTHY,
            version="1.0.0",
            description="Test service description",
        )

        dependency_manager._dependencies["invalid_service"] = info

        with pytest.raises(ValueError, match="Invalid dependency"):
            dependency_manager.validate_dependencies()


class TestDependencyManagerIntegration:
    """Integration tests for dependency management system."""

    @pytest.mark.asyncio
    async def test_full_dependency_lifecycle(self):
        """Test complete dependency lifecycle."""
        manager = DependencyManager()

        # Register dependencies
        info1 = DependencyInfo(
            name="database",
            status=DependencyStatus.HEALTHY,
            version="1.0.0",
            description="Database service",
        )
        info2 = DependencyInfo(
            name="cache",
            status=DependencyStatus.HEALTHY,
            version="2.0.0",
            description="Cache service",
        )

        manager.register_dependency(info1)
        manager.register_dependency(info2)

        # Register health checks
        def database_health_check():
            return DependencyStatus.HEALTHY

        def cache_health_check():
            return DependencyStatus.HEALTHY

        manager.register_health_check("database", database_health_check)
        manager.register_health_check("cache", cache_health_check)

        # Check health
        db_status = manager.check_dependency_health("database")
        cache_status = manager.check_dependency_health("cache")

        assert db_status == DependencyStatus.HEALTHY
        assert cache_status == DependencyStatus.HEALTHY

        # Update status
        manager.update_dependency_status("database", DependencyStatus.UNHEALTHY)

        updated_db = manager.get_dependency("database")
        assert updated_db.status == DependencyStatus.UNHEALTHY

        # Get statistics
        stats = manager.get_dependency_statistics()
        assert stats["total_dependencies"] == 2
        assert stats["healthy_dependencies"] == 1
        assert stats["unhealthy_dependencies"] == 1

    @pytest.mark.asyncio
    async def test_dependency_health_monitoring(self):
        """Test dependency health monitoring."""
        manager = DependencyManager()

        # Register dependency with health check
        info = DependencyInfo(
            name="api_service",
            status=DependencyStatus.HEALTHY,
            version="1.0.0",
            description="API service",
        )

        manager.register_dependency(info)

        # Register health check that can change status
        health_status = DependencyStatus.HEALTHY

        def dynamic_health_check():
            return health_status

        manager.register_health_check("api_service", dynamic_health_check)

        # Initial health check
        status = manager.check_dependency_health("api_service")
        assert status == DependencyStatus.HEALTHY

        # Simulate service becoming unhealthy
        health_status = DependencyStatus.UNHEALTHY

        status = manager.check_dependency_health("api_service")
        assert status == DependencyStatus.UNHEALTHY

        # Simulate service recovery
        health_status = DependencyStatus.HEALTHY

        status = manager.check_dependency_health("api_service")
        assert status == DependencyStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_dependency_export_import(self):
        """Test dependency export and import functionality."""
        manager1 = DependencyManager()
        manager2 = DependencyManager()

        # Register dependencies in manager1
        info1 = DependencyInfo(
            name="service1",
            status=DependencyStatus.HEALTHY,
            version="1.0.0",
            description="Service 1",
            metadata={"endpoint": "http://localhost:8080"},
        )
        info2 = DependencyInfo(
            name="service2",
            status=DependencyStatus.UNHEALTHY,
            version="2.0.0",
            description="Service 2",
            metadata={"timeout": 30},
        )

        manager1.register_dependency(info1)
        manager1.register_dependency(info2)

        # Export from manager1
        exported = manager1.export_dependencies()

        # Import to manager2
        manager2.import_dependencies(exported)

        # Verify import
        assert len(manager2._dependencies) == 2

        service1 = manager2.get_dependency("service1")
        assert service1.status == DependencyStatus.HEALTHY
        assert service1.version == "1.0.0"
        assert service1.metadata["endpoint"] == "http://localhost:8080"

        service2 = manager2.get_dependency("service2")
        assert service2.status == DependencyStatus.UNHEALTHY
        assert service2.version == "2.0.0"
        assert service2.metadata["timeout"] == 30

    @pytest.mark.asyncio
    async def test_dependency_statistics(self):
        """Test dependency statistics calculation."""
        manager = DependencyManager()

        # Register dependencies with different statuses
        dependencies = [
            ("service1", DependencyStatus.HEALTHY),
            ("service2", DependencyStatus.HEALTHY),
            ("service3", DependencyStatus.UNHEALTHY),
            ("service4", DependencyStatus.DEGRADED),
            ("service5", DependencyStatus.UNKNOWN),
        ]

        for name, status in dependencies:
            info = DependencyInfo(
                name=name,
                status=status,
                version="1.0.0",
                description=f"{name} description",
            )
            manager.register_dependency(info)

        # Get statistics
        stats = manager.get_dependency_statistics()

        assert stats["total_dependencies"] == 5
        assert stats["healthy_dependencies"] == 2
        assert stats["unhealthy_dependencies"] == 3  # UNHEALTHY + DEGRADED + UNKNOWN
        assert stats["health_percentage"] == 40.0  # 2/5 * 100

        # Test filtered statistics
        healthy_deps = manager.get_healthy_dependencies()
        assert len(healthy_deps) == 2
        assert "service1" in healthy_deps
        assert "service2" in healthy_deps

        unhealthy_deps = manager.get_unhealthy_dependencies()
        assert len(unhealthy_deps) == 3
        assert "service3" in unhealthy_deps
        assert "service4" in unhealthy_deps
        assert "service5" in unhealthy_deps
