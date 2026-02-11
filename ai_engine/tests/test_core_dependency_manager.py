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
    DependencyManager,
    DependencyStatus,
    DependencyInfo,
    DependencyResolutionError,
    DependencyCircularReferenceError,
    DependencyNotFoundError,
    ServiceLocator,
    DependencyRegistry,
    DependencyResolver,
    DependencyValidator,
    DependencyMonitor,
    DependencyHealthCheck,
    DependencyDefinition,
    DependencyScope,
    DependencyLifecycle,
)


class TestDependencyDefinition:
    """Test DependencyDefinition class."""

    def test_initialization(self):
        """Test DependencyDefinition initialization."""
        definition = DependencyDefinition(
            name="test_service",
            factory=lambda: "test_instance",
            scope=DependencyScope.SINGLETON,
            lifecycle=DependencyLifecycle.PERMANENT,
            dependencies=["dep1", "dep2"],
        )

        assert definition.name == "test_service"
        assert definition.factory() == "test_instance"
        assert definition.scope == DependencyScope.SINGLETON
        assert definition.lifecycle == DependencyLifecycle.PERMANENT
        assert definition.dependencies == ["dep1", "dep2"]

    def test_initialization_with_defaults(self):
        """Test DependencyDefinition initialization with defaults."""
        definition = DependencyDefinition(
            name="test_service", factory=lambda: "test_instance"
        )

        assert definition.name == "test_service"
        assert definition.factory() == "test_instance"
        assert definition.scope == DependencyScope.SINGLETON
        assert definition.lifecycle == DependencyLifecycle.PERMANENT
        assert definition.dependencies == []

    def test_validation(self):
        """Test DependencyDefinition validation."""
        # Valid definition
        definition = DependencyDefinition(
            name="test_service", factory=lambda: "test_instance"
        )
        assert definition.validate() is True

        # Invalid definition - no factory
        with pytest.raises(ValueError, match="Factory is required"):
            DependencyDefinition(name="test_service", factory=None)

    def test_to_dict(self):
        """Test converting DependencyDefinition to dictionary."""
        definition = DependencyDefinition(
            name="test_service",
            factory=lambda: "test_instance",
            scope=DependencyScope.SINGLETON,
            lifecycle=DependencyLifecycle.PERMANENT,
            dependencies=["dep1"],
        )

        definition_dict = definition.to_dict()

        assert definition_dict["name"] == "test_service"
        assert definition_dict["scope"] == "singleton"
        assert definition_dict["lifecycle"] == "permanent"
        assert definition_dict["dependencies"] == ["dep1"]
        assert "factory" in definition_dict

    def test_from_dict(self):
        """Test creating DependencyDefinition from dictionary."""
        definition_dict = {
            "name": "test_service",
            "scope": "singleton",
            "lifecycle": "permanent",
            "dependencies": ["dep1"],
            "factory": lambda: "test_instance",
        }

        definition = DependencyDefinition.from_dict(definition_dict)

        assert definition.name == "test_service"
        assert definition.scope == DependencyScope.SINGLETON
        assert definition.lifecycle == DependencyLifecycle.PERMANENT
        assert definition.dependencies == ["dep1"]


class TestDependencyContainer:
    """Test DependencyContainer class."""

    @pytest.fixture
    def container(self):
        """Create DependencyContainer instance."""
        return DependencyContainer()

    def test_initialization(self, container):
        """Test DependencyContainer initialization."""
        assert container._definitions == {}
        assert container._instances == {}
        assert container._singletons == {}
        assert container._transients == {}
        assert container._scoped == {}

    def test_register_dependency(self, container):
        """Test registering dependency."""
        definition = DependencyDefinition(
            name="test_service", factory=lambda: "test_instance"
        )

        container.register_dependency(definition)

        assert "test_service" in container._definitions
        assert container._definitions["test_service"] == definition

    def test_register_dependency_duplicate(self, container):
        """Test registering duplicate dependency."""
        definition1 = DependencyDefinition(
            name="test_service", factory=lambda: "test_instance1"
        )
        definition2 = DependencyDefinition(
            name="test_service", factory=lambda: "test_instance2"
        )

        container.register_dependency(definition1)

        with pytest.raises(
            DependencyInjectionError, match="Dependency already registered"
        ):
            container.register_dependency(definition2)

    def test_resolve_dependency_singleton(self, container):
        """Test resolving singleton dependency."""
        definition = DependencyDefinition(
            name="test_service",
            factory=lambda: "test_instance",
            scope=DependencyScope.SINGLETON,
        )

        container.register_dependency(definition)

        instance1 = container.resolve("test_service")
        instance2 = container.resolve("test_service")

        assert instance1 == "test_instance"
        assert instance2 == "test_instance"
        assert instance1 is instance2  # Same instance for singleton

    def test_resolve_dependency_transient(self, container):
        """Test resolving transient dependency."""
        definition = DependencyDefinition(
            name="test_service",
            factory=lambda: "test_instance",
            scope=DependencyScope.TRANSIENT,
        )

        container.register_dependency(definition)

        instance1 = container.resolve("test_service")
        instance2 = container.resolve("test_service")

        assert instance1 == "test_instance"
        assert instance2 == "test_instance"
        assert instance1 is not instance2  # Different instances for transient

    def test_resolve_dependency_scoped(self, container):
        """Test resolving scoped dependency."""
        definition = DependencyDefinition(
            name="test_service",
            factory=lambda: "test_instance",
            scope=DependencyScope.SCOPED,
        )

        container.register_dependency(definition)

        # Create scope
        scope_id = "test_scope"
        container.create_scope(scope_id)

        instance1 = container.resolve("test_service", scope_id=scope_id)
        instance2 = container.resolve("test_service", scope_id=scope_id)

        assert instance1 == "test_instance"
        assert instance2 == "test_instance"
        assert instance1 is instance2  # Same instance within scope

    def test_resolve_dependency_not_found(self, container):
        """Test resolving non-existent dependency."""
        with pytest.raises(DependencyNotFoundError, match="Dependency not found"):
            container.resolve("nonexistent_service")

    def test_resolve_dependency_with_dependencies(self, container):
        """Test resolving dependency with dependencies."""
        # Register dependency
        dep_definition = DependencyDefinition(
            name="dependency", factory=lambda: "dependency_instance"
        )
        container.register_dependency(dep_definition)

        # Register service with dependency
        def service_factory(dependency):
            return f"service_with_{dependency}"

        service_definition = DependencyDefinition(
            name="service", factory=service_factory, dependencies=["dependency"]
        )
        container.register_dependency(service_definition)

        instance = container.resolve("service")

        assert instance == "service_with_dependency_instance"

    def test_resolve_dependency_circular_reference(self, container):
        """Test resolving dependency with circular reference."""

        # Register service A that depends on B
        def service_a_factory(service_b):
            return f"service_a_with_{service_b}"

        service_a_definition = DependencyDefinition(
            name="service_a", factory=service_a_factory, dependencies=["service_b"]
        )
        container.register_dependency(service_a_definition)

        # Register service B that depends on A
        def service_b_factory(service_a):
            return f"service_b_with_{service_a}"

        service_b_definition = DependencyDefinition(
            name="service_b", factory=service_b_factory, dependencies=["service_a"]
        )
        container.register_dependency(service_b_definition)

        with pytest.raises(
            DependencyCircularReferenceError, match="Circular dependency detected"
        ):
            container.resolve("service_a")

    def test_create_scope(self, container):
        """Test creating dependency scope."""
        scope_id = "test_scope"
        container.create_scope(scope_id)

        assert scope_id in container._scoped

    def test_dispose_scope(self, container):
        """Test disposing dependency scope."""
        scope_id = "test_scope"
        container.create_scope(scope_id)

        # Add some scoped instances
        container._scoped[scope_id] = {"test_service": "test_instance"}

        container.dispose_scope(scope_id)

        assert scope_id not in container._scoped

    def test_dispose_all(self, container):
        """Test disposing all dependencies."""
        # Register some dependencies
        definition = DependencyDefinition(
            name="test_service",
            factory=lambda: "test_instance",
            lifecycle=DependencyLifecycle.DISPOSABLE,
        )
        container.register_dependency(definition)

        # Create instance
        container.resolve("test_service")

        container.dispose_all()

        assert len(container._instances) == 0
        assert len(container._singletons) == 0
        assert len(container._transients) == 0
        assert len(container._scoped) == 0

    def test_get_dependency_info(self, container):
        """Test getting dependency information."""
        definition = DependencyDefinition(
            name="test_service",
            factory=lambda: "test_instance",
            scope=DependencyScope.SINGLETON,
            lifecycle=DependencyLifecycle.PERMANENT,
            dependencies=["dep1"],
        )

        container.register_dependency(definition)

        info = container.get_dependency_info("test_service")

        assert info["name"] == "test_service"
        assert info["scope"] == "singleton"
        assert info["lifecycle"] == "permanent"
        assert info["dependencies"] == ["dep1"]
        assert info["registered"] is True
        assert info["instantiated"] is False

    def test_list_dependencies(self, container):
        """Test listing all dependencies."""
        # Register some dependencies
        definition1 = DependencyDefinition(name="service1", factory=lambda: "instance1")
        definition2 = DependencyDefinition(name="service2", factory=lambda: "instance2")

        container.register_dependency(definition1)
        container.register_dependency(definition2)

        dependencies = container.list_dependencies()

        assert len(dependencies) == 2
        assert "service1" in dependencies
        assert "service2" in dependencies

    def test_validate_dependencies(self, container):
        """Test validating all dependencies."""
        # Register valid dependency
        definition = DependencyDefinition(
            name="test_service", factory=lambda: "test_instance"
        )
        container.register_dependency(definition)

        # Should not raise exception
        container.validate_dependencies()

    def test_validate_dependencies_invalid(self, container):
        """Test validating dependencies with invalid definition."""
        # Register invalid dependency (missing factory)
        definition = DependencyDefinition(name="test_service", factory=None)
        container._definitions["test_service"] = definition

        with pytest.raises(
            DependencyInjectionError, match="Invalid dependency definition"
        ):
            container.validate_dependencies()


class TestDependencyManager:
    """Test DependencyManager class."""

    @pytest.fixture
    def dependency_manager(self):
        """Create DependencyManager instance."""
        return DependencyManager()

    def test_initialization(self, dependency_manager):
        """Test DependencyManager initialization."""
        assert dependency_manager.container is not None
        assert dependency_manager.registry is not None
        assert dependency_manager.resolver is not None
        assert dependency_manager.validator is not None
        assert dependency_manager.monitor is not None

    def test_register_service(self, dependency_manager):
        """Test registering service."""

        def service_factory():
            return "service_instance"

        dependency_manager.register_service(
            name="test_service",
            factory=service_factory,
            scope=DependencyScope.SINGLETON,
        )

        assert "test_service" in dependency_manager.container._definitions

    def test_register_singleton(self, dependency_manager):
        """Test registering singleton service."""

        def service_factory():
            return "singleton_instance"

        dependency_manager.register_singleton("test_service", service_factory)

        definition = dependency_manager.container._definitions["test_service"]
        assert definition.scope == DependencyScope.SINGLETON

    def test_register_transient(self, dependency_manager):
        """Test registering transient service."""

        def service_factory():
            return "transient_instance"

        dependency_manager.register_transient("test_service", service_factory)

        definition = dependency_manager.container._definitions["test_service"]
        assert definition.scope == DependencyScope.TRANSIENT

    def test_register_scoped(self, dependency_manager):
        """Test registering scoped service."""

        def service_factory():
            return "scoped_instance"

        dependency_manager.register_scoped("test_service", service_factory)

        definition = dependency_manager.container._definitions["test_service"]
        assert definition.scope == DependencyScope.SCOPED

    def test_register_instance(self, dependency_manager):
        """Test registering service instance."""
        instance = "pre_created_instance"

        dependency_manager.register_instance("test_service", instance)

        # Should be available immediately
        resolved = dependency_manager.resolve("test_service")
        assert resolved == instance

    def test_resolve_service(self, dependency_manager):
        """Test resolving service."""

        def service_factory():
            return "service_instance"

        dependency_manager.register_service("test_service", service_factory)

        instance = dependency_manager.resolve("test_service")

        assert instance == "service_instance"

    def test_resolve_service_with_dependencies(self, dependency_manager):
        """Test resolving service with dependencies."""
        # Register dependency
        dependency_manager.register_service("dependency", lambda: "dep_instance")

        # Register service with dependency
        def service_factory(dependency):
            return f"service_with_{dependency}"

        dependency_manager.register_service(
            "service", service_factory, dependencies=["dependency"]
        )

        instance = dependency_manager.resolve("service")

        assert instance == "service_with_dep_instance"

    def test_resolve_service_not_found(self, dependency_manager):
        """Test resolving non-existent service."""
        with pytest.raises(DependencyNotFoundError, match="Dependency not found"):
            dependency_manager.resolve("nonexistent_service")

    def test_is_registered(self, dependency_manager):
        """Test checking if service is registered."""
        dependency_manager.register_service("test_service", lambda: "instance")

        assert dependency_manager.is_registered("test_service") is True
        assert dependency_manager.is_registered("nonexistent_service") is False

    def test_get_service_info(self, dependency_manager):
        """Test getting service information."""
        dependency_manager.register_service("test_service", lambda: "instance")

        info = dependency_manager.get_service_info("test_service")

        assert info["name"] == "test_service"
        assert info["registered"] is True

    def test_list_services(self, dependency_manager):
        """Test listing all services."""
        dependency_manager.register_service("service1", lambda: "instance1")
        dependency_manager.register_service("service2", lambda: "instance2")

        services = dependency_manager.list_services()

        assert len(services) == 2
        assert "service1" in services
        assert "service2" in services

    def test_validate_services(self, dependency_manager):
        """Test validating all services."""
        dependency_manager.register_service("test_service", lambda: "instance")

        # Should not raise exception
        dependency_manager.validate_services()

    def test_dispose_all(self, dependency_manager):
        """Test disposing all services."""
        dependency_manager.register_service("test_service", lambda: "instance")
        dependency_manager.resolve("test_service")

        dependency_manager.dispose_all()

        # Should be able to resolve again (new instance)
        instance = dependency_manager.resolve("test_service")
        assert instance == "instance"

    def test_create_scope(self, dependency_manager):
        """Test creating service scope."""
        scope_id = "test_scope"
        dependency_manager.create_scope(scope_id)

        assert scope_id in dependency_manager.container._scoped

    def test_dispose_scope(self, dependency_manager):
        """Test disposing service scope."""
        scope_id = "test_scope"
        dependency_manager.create_scope(scope_id)

        dependency_manager.dispose_scope(scope_id)

        assert scope_id not in dependency_manager.container._scoped

    def test_health_check(self, dependency_manager):
        """Test dependency manager health check."""
        dependency_manager.register_service("test_service", lambda: "instance")

        health = dependency_manager.health_check()

        assert health["status"] == "healthy"
        assert health["total_services"] == 1
        assert health["registered_services"] == 1
        assert health["instantiated_services"] == 0


class TestServiceLocator:
    """Test ServiceLocator class."""

    @pytest.fixture
    def service_locator(self):
        """Create ServiceLocator instance."""
        return ServiceLocator()

    def test_initialization(self, service_locator):
        """Test ServiceLocator initialization."""
        assert service_locator._container is not None

    def test_set_container(self, service_locator):
        """Test setting container."""
        container = DependencyContainer()
        service_locator.set_container(container)

        assert service_locator._container is container

    def test_get_service(self, service_locator):
        """Test getting service."""
        container = DependencyContainer()
        definition = DependencyDefinition(
            name="test_service", factory=lambda: "instance"
        )
        container.register_dependency(definition)
        service_locator.set_container(container)

        service = service_locator.get_service("test_service")

        assert service == "instance"

    def test_get_service_not_found(self, service_locator):
        """Test getting non-existent service."""
        with pytest.raises(DependencyNotFoundError, match="Service not found"):
            service_locator.get_service("nonexistent_service")

    def test_get_optional_service(self, service_locator):
        """Test getting optional service."""
        container = DependencyContainer()
        definition = DependencyDefinition(
            name="test_service", factory=lambda: "instance"
        )
        container.register_dependency(definition)
        service_locator.set_container(container)

        service = service_locator.get_optional_service("test_service")
        assert service == "instance"

        service = service_locator.get_optional_service("nonexistent_service")
        assert service is None

    def test_get_services_by_type(self, service_locator):
        """Test getting services by type."""
        container = DependencyContainer()

        # Register services of different types
        definition1 = DependencyDefinition(
            name="service1", factory=lambda: "string_instance"
        )
        definition2 = DependencyDefinition(name="service2", factory=lambda: 42)
        definition3 = DependencyDefinition(
            name="service3", factory=lambda: "another_string"
        )

        container.register_dependency(definition1)
        container.register_dependency(definition2)
        container.register_dependency(definition3)
        service_locator.set_container(container)

        string_services = service_locator.get_services_by_type(str)

        assert len(string_services) == 2
        assert "string_instance" in string_services
        assert "another_string" in string_services

    def test_get_all_services(self, service_locator):
        """Test getting all services."""
        container = DependencyContainer()
        definition1 = DependencyDefinition(name="service1", factory=lambda: "instance1")
        definition2 = DependencyDefinition(name="service2", factory=lambda: "instance2")

        container.register_dependency(definition1)
        container.register_dependency(definition2)
        service_locator.set_container(container)

        all_services = service_locator.get_all_services()

        assert len(all_services) == 2
        assert "service1" in all_services
        assert "service2" in all_services


class TestDependencyRegistry:
    """Test DependencyRegistry class."""

    @pytest.fixture
    def registry(self):
        """Create DependencyRegistry instance."""
        return DependencyRegistry()

    def test_initialization(self, registry):
        """Test DependencyRegistry initialization."""
        assert registry._definitions == {}
        assert registry._metadata == {}

    def test_register(self, registry):
        """Test registering dependency."""
        definition = DependencyDefinition(
            name="test_service", factory=lambda: "instance"
        )

        registry.register(definition)

        assert "test_service" in registry._definitions
        assert registry._definitions["test_service"] == definition

    def test_register_with_metadata(self, registry):
        """Test registering dependency with metadata."""
        definition = DependencyDefinition(
            name="test_service", factory=lambda: "instance"
        )
        metadata = {"version": "1.0", "description": "Test service"}

        registry.register(definition, metadata=metadata)

        assert "test_service" in registry._definitions
        assert registry._metadata["test_service"] == metadata

    def test_unregister(self, registry):
        """Test unregistering dependency."""
        definition = DependencyDefinition(
            name="test_service", factory=lambda: "instance"
        )
        registry.register(definition)

        registry.unregister("test_service")

        assert "test_service" not in registry._definitions

    def test_get_definition(self, registry):
        """Test getting dependency definition."""
        definition = DependencyDefinition(
            name="test_service", factory=lambda: "instance"
        )
        registry.register(definition)

        retrieved = registry.get_definition("test_service")

        assert retrieved == definition

    def test_get_definition_not_found(self, registry):
        """Test getting non-existent dependency definition."""
        with pytest.raises(DependencyNotFoundError, match="Definition not found"):
            registry.get_definition("nonexistent_service")

    def test_get_metadata(self, registry):
        """Test getting dependency metadata."""
        definition = DependencyDefinition(
            name="test_service", factory=lambda: "instance"
        )
        metadata = {"version": "1.0"}
        registry.register(definition, metadata=metadata)

        retrieved = registry.get_metadata("test_service")

        assert retrieved == metadata

    def test_list_definitions(self, registry):
        """Test listing all definitions."""
        definition1 = DependencyDefinition(name="service1", factory=lambda: "instance1")
        definition2 = DependencyDefinition(name="service2", factory=lambda: "instance2")

        registry.register(definition1)
        registry.register(definition2)

        definitions = registry.list_definitions()

        assert len(definitions) == 2
        assert "service1" in definitions
        assert "service2" in definitions

    def test_clear(self, registry):
        """Test clearing registry."""
        definition = DependencyDefinition(
            name="test_service", factory=lambda: "instance"
        )
        registry.register(definition)

        registry.clear()

        assert len(registry._definitions) == 0
        assert len(registry._metadata) == 0


class TestDependencyResolver:
    """Test DependencyResolver class."""

    @pytest.fixture
    def resolver(self):
        """Create DependencyResolver instance."""
        return DependencyResolver()

    def test_initialization(self, resolver):
        """Test DependencyResolver initialization."""
        assert resolver._resolution_stack == []
        assert resolver._resolved_cache == {}

    def test_resolve_dependencies(self, resolver):
        """Test resolving dependencies."""
        # Create mock container
        container = Mock()
        container.resolve.side_effect = lambda name: f"{name}_instance"

        # Create dependency definition
        definition = DependencyDefinition(
            name="test_service",
            factory=lambda dep1, dep2: f"service_with_{dep1}_{dep2}",
            dependencies=["dep1", "dep2"],
        )

        result = resolver.resolve_dependencies(definition, container)

        assert result == "service_with_dep1_instance_dep2_instance"
        assert container.resolve.call_count == 2

    def test_resolve_dependencies_no_deps(self, resolver):
        """Test resolving dependencies with no dependencies."""
        container = Mock()

        definition = DependencyDefinition(
            name="test_service", factory=lambda: "service_instance"
        )

        result = resolver.resolve_dependencies(definition, container)

        assert result == "service_instance"
        container.resolve.assert_not_called()

    def test_resolve_dependencies_circular(self, resolver):
        """Test resolving dependencies with circular reference."""
        container = Mock()

        definition = DependencyDefinition(
            name="test_service",
            factory=lambda dep: f"service_with_{dep}",
            dependencies=["test_service"],  # Circular dependency
        )

        with pytest.raises(
            DependencyCircularReferenceError, match="Circular dependency detected"
        ):
            resolver.resolve_dependencies(definition, container)

    def test_validate_dependencies(self, resolver):
        """Test validating dependencies."""
        container = Mock()
        container.is_registered.return_value = True

        definition = DependencyDefinition(
            name="test_service",
            factory=lambda dep: f"service_with_{dep}",
            dependencies=["dep1", "dep2"],
        )

        # Should not raise exception
        resolver.validate_dependencies(definition, container)

    def test_validate_dependencies_missing(self, resolver):
        """Test validating dependencies with missing dependency."""
        container = Mock()
        container.is_registered.side_effect = lambda name: name != "missing_dep"

        definition = DependencyDefinition(
            name="test_service",
            factory=lambda dep: f"service_with_{dep}",
            dependencies=["dep1", "missing_dep"],
        )

        with pytest.raises(DependencyResolutionError, match="Missing dependency"):
            resolver.validate_dependencies(definition, container)

    def test_clear_cache(self, resolver):
        """Test clearing resolution cache."""
        resolver._resolved_cache["test_service"] = "cached_instance"

        resolver.clear_cache()

        assert len(resolver._resolved_cache) == 0


class TestDependencyValidator:
    """Test DependencyValidator class."""

    @pytest.fixture
    def validator(self):
        """Create DependencyValidator instance."""
        return DependencyValidator()

    def test_initialization(self, validator):
        """Test DependencyValidator initialization."""
        assert validator._validation_rules == {}

    def test_validate_definition(self, validator):
        """Test validating dependency definition."""
        definition = DependencyDefinition(
            name="test_service", factory=lambda: "instance"
        )

        # Should not raise exception
        validator.validate_definition(definition)

    def test_validate_definition_invalid_name(self, validator):
        """Test validating dependency definition with invalid name."""
        definition = DependencyDefinition(
            name="", factory=lambda: "instance"  # Invalid empty name
        )

        with pytest.raises(DependencyInjectionError, match="Invalid dependency name"):
            validator.validate_definition(definition)

    def test_validate_definition_no_factory(self, validator):
        """Test validating dependency definition with no factory."""
        definition = DependencyDefinition(name="test_service", factory=None)

        with pytest.raises(DependencyInjectionError, match="Factory is required"):
            validator.validate_definition(definition)

    def test_validate_definition_invalid_factory(self, validator):
        """Test validating dependency definition with invalid factory."""
        definition = DependencyDefinition(name="test_service", factory="not_a_callable")

        with pytest.raises(DependencyInjectionError, match="Factory must be callable"):
            validator.validate_definition(definition)

    def test_validate_container(self, validator):
        """Test validating dependency container."""
        container = DependencyContainer()
        definition = DependencyDefinition(
            name="test_service", factory=lambda: "instance"
        )
        container.register_dependency(definition)

        # Should not raise exception
        validator.validate_container(container)

    def test_validate_container_circular_deps(self, validator):
        """Test validating container with circular dependencies."""
        container = DependencyContainer()

        # Create circular dependency
        def service_a_factory(service_b):
            return f"service_a_with_{service_b}"

        def service_b_factory(service_a):
            return f"service_b_with_{service_a}"

        definition_a = DependencyDefinition(
            name="service_a", factory=service_a_factory, dependencies=["service_b"]
        )
        definition_b = DependencyDefinition(
            name="service_b", factory=service_b_factory, dependencies=["service_a"]
        )

        container.register_dependency(definition_a)
        container.register_dependency(definition_b)

        with pytest.raises(
            DependencyCircularReferenceError, match="Circular dependency detected"
        ):
            validator.validate_container(container)

    def test_add_validation_rule(self, validator):
        """Test adding custom validation rule."""

        def custom_rule(definition):
            if definition.name.startswith("test_"):
                raise DependencyInjectionError("Name cannot start with 'test_'")

        validator.add_validation_rule("custom_rule", custom_rule)

        definition = DependencyDefinition(
            name="test_service", factory=lambda: "instance"
        )

        with pytest.raises(
            DependencyInjectionError, match="Name cannot start with 'test_'"
        ):
            validator.validate_definition(definition)

    def test_remove_validation_rule(self, validator):
        """Test removing validation rule."""

        def custom_rule(definition):
            if definition.name.startswith("test_"):
                raise DependencyInjectionError("Name cannot start with 'test_'")

        validator.add_validation_rule("custom_rule", custom_rule)
        validator.remove_validation_rule("custom_rule")

        definition = DependencyDefinition(
            name="test_service", factory=lambda: "instance"
        )

        # Should not raise exception now
        validator.validate_definition(definition)


class TestDependencyMonitor:
    """Test DependencyMonitor class."""

    @pytest.fixture
    def monitor(self):
        """Create DependencyMonitor instance."""
        return DependencyMonitor()

    def test_initialization(self, monitor):
        """Test DependencyMonitor initialization."""
        assert monitor._metrics == {}
        assert monitor._events == []

    def test_record_resolution(self, monitor):
        """Test recording dependency resolution."""
        monitor.record_resolution("test_service", 0.1, success=True)

        assert "test_service" in monitor._metrics
        assert monitor._metrics["test_service"]["total_resolutions"] == 1
        assert monitor._metrics["test_service"]["successful_resolutions"] == 1
        assert monitor._metrics["test_service"]["failed_resolutions"] == 0

    def test_record_resolution_failure(self, monitor):
        """Test recording failed dependency resolution."""
        monitor.record_resolution("test_service", 0.1, success=False)

        assert "test_service" in monitor._metrics
        assert monitor._metrics["test_service"]["total_resolutions"] == 1
        assert monitor._metrics["test_service"]["successful_resolutions"] == 0
        assert monitor._metrics["test_service"]["failed_resolutions"] == 1

    def test_record_event(self, monitor):
        """Test recording dependency event."""
        event = {
            "type": "resolution",
            "service": "test_service",
            "timestamp": "2023-01-01T00:00:00Z",
            "duration": 0.1,
            "success": True,
        }

        monitor.record_event(event)

        assert len(monitor._events) == 1
        assert monitor._events[0] == event

    def test_get_metrics(self, monitor):
        """Test getting dependency metrics."""
        monitor.record_resolution("test_service", 0.1, success=True)
        monitor.record_resolution("test_service", 0.2, success=False)

        metrics = monitor.get_metrics("test_service")

        assert metrics["total_resolutions"] == 2
        assert metrics["successful_resolutions"] == 1
        assert metrics["failed_resolutions"] == 1
        assert metrics["success_rate"] == 0.5

    def test_get_all_metrics(self, monitor):
        """Test getting all dependency metrics."""
        monitor.record_resolution("service1", 0.1, success=True)
        monitor.record_resolution("service2", 0.2, success=True)

        all_metrics = monitor.get_all_metrics()

        assert len(all_metrics) == 2
        assert "service1" in all_metrics
        assert "service2" in all_metrics

    def test_get_events(self, monitor):
        """Test getting dependency events."""
        event1 = {"type": "resolution", "service": "service1"}
        event2 = {"type": "resolution", "service": "service2"}

        monitor.record_event(event1)
        monitor.record_event(event2)

        events = monitor.get_events()

        assert len(events) == 2
        assert event1 in events
        assert event2 in events

    def test_clear_metrics(self, monitor):
        """Test clearing dependency metrics."""
        monitor.record_resolution("test_service", 0.1, success=True)

        monitor.clear_metrics()

        assert len(monitor._metrics) == 0

    def test_clear_events(self, monitor):
        """Test clearing dependency events."""
        monitor.record_event({"type": "resolution", "service": "test_service"})

        monitor.clear_events()

        assert len(monitor._events) == 0


class TestDependencyHealthCheck:
    """Test DependencyHealthCheck class."""

    @pytest.fixture
    def health_check(self):
        """Create DependencyHealthCheck instance."""
        return DependencyHealthCheck()

    def test_initialization(self, health_check):
        """Test DependencyHealthCheck initialization."""
        assert health_check._checks == {}

    def test_register_check(self, health_check):
        """Test registering health check."""

        def check_function():
            return {"status": "healthy", "message": "OK"}

        health_check.register_check("test_service", check_function)

        assert "test_service" in health_check._checks

    def test_unregister_check(self, health_check):
        """Test unregistering health check."""

        def check_function():
            return {"status": "healthy", "message": "OK"}

        health_check.register_check("test_service", check_function)
        health_check.unregister_check("test_service")

        assert "test_service" not in health_check._checks

    def test_check_service_health(self, health_check):
        """Test checking service health."""

        def check_function():
            return {"status": "healthy", "message": "OK"}

        health_check.register_check("test_service", check_function)

        result = health_check.check_service_health("test_service")

        assert result["status"] == "healthy"
        assert result["message"] == "OK"

    def test_check_service_health_not_found(self, health_check):
        """Test checking health of non-existent service."""
        result = health_check.check_service_health("nonexistent_service")

        assert result["status"] == "unknown"
        assert "not found" in result["message"]

    def test_check_service_health_exception(self, health_check):
        """Test checking service health with exception."""

        def check_function():
            raise Exception("Health check failed")

        health_check.register_check("test_service", check_function)

        result = health_check.check_service_health("test_service")

        assert result["status"] == "unhealthy"
        assert "Health check failed" in result["message"]

    def test_check_all_services_health(self, health_check):
        """Test checking all services health."""

        def check_function1():
            return {"status": "healthy", "message": "OK"}

        def check_function2():
            return {"status": "unhealthy", "message": "Error"}

        health_check.register_check("service1", check_function1)
        health_check.register_check("service2", check_function2)

        result = health_check.check_all_services_health()

        assert len(result) == 2
        assert result["service1"]["status"] == "healthy"
        assert result["service2"]["status"] == "unhealthy"

    def test_get_overall_health(self, health_check):
        """Test getting overall health status."""

        def check_function1():
            return {"status": "healthy", "message": "OK"}

        def check_function2():
            return {"status": "unhealthy", "message": "Error"}

        health_check.register_check("service1", check_function1)
        health_check.register_check("service2", check_function2)

        result = health_check.get_overall_health()

        assert (
            result["status"] == "unhealthy"
        )  # Any unhealthy service makes overall unhealthy
        assert result["total_services"] == 2
        assert result["healthy_services"] == 1
        assert result["unhealthy_services"] == 1


class TestDependencyScope:
    """Test DependencyScope enum."""

    def test_dependency_scope_values(self):
        """Test DependencyScope enum values."""
        assert DependencyScope.SINGLETON.value == "singleton"
        assert DependencyScope.TRANSIENT.value == "transient"
        assert DependencyScope.SCOPED.value == "scoped"


class TestDependencyLifecycle:
    """Test DependencyLifecycle enum."""

    def test_dependency_lifecycle_values(self):
        """Test DependencyLifecycle enum values."""
        assert DependencyLifecycle.PERMANENT.value == "permanent"
        assert DependencyLifecycle.DISPOSABLE.value == "disposable"
        assert DependencyLifecycle.EPHEMERAL.value == "ephemeral"


class TestDependencyExceptions:
    """Test dependency-related exceptions."""

    def test_dependency_injection_error(self):
        """Test DependencyInjectionError exception."""
        error = DependencyInjectionError("Test error", error_code="TEST_ERROR")

        assert str(error) == "Test error"
        assert error.error_code == "TEST_ERROR"

    def test_dependency_resolution_error(self):
        """Test DependencyResolutionError exception."""
        error = DependencyResolutionError("Resolution failed")

        assert str(error) == "Resolution failed"

    def test_dependency_circular_reference_error(self):
        """Test DependencyCircularReferenceError exception."""
        error = DependencyCircularReferenceError("Circular dependency")

        assert str(error) == "Circular dependency"

    def test_dependency_not_found_error(self):
        """Test DependencyNotFoundError exception."""
        error = DependencyNotFoundError("Dependency not found")

        assert str(error) == "Dependency not found"


class TestDependencyManagerIntegration:
    """Integration tests for dependency management system."""

    @pytest.mark.asyncio
    async def test_full_dependency_lifecycle(self):
        """Test complete dependency lifecycle."""
        manager = DependencyManager()

        # Register services
        manager.register_service("database", lambda: "database_connection")
        manager.register_service("cache", lambda: "cache_connection")

        def api_service_factory(database, cache):
            return f"api_service_with_{database}_{cache}"

        manager.register_service(
            "api_service", api_service_factory, dependencies=["database", "cache"]
        )

        # Resolve service
        api_service = manager.resolve("api_service")

        assert api_service == "api_service_with_database_connection_cache_connection"

        # Check health
        health = manager.health_check()
        assert health["status"] == "healthy"
        assert health["total_services"] == 3

    @pytest.mark.asyncio
    async def test_dependency_scopes(self):
        """Test different dependency scopes."""
        manager = DependencyManager()

        # Register singleton
        manager.register_singleton("singleton_service", lambda: "singleton_instance")

        # Register transient
        manager.register_transient("transient_service", lambda: "transient_instance")

        # Register scoped
        manager.register_scoped("scoped_service", lambda: "scoped_instance")

        # Test singleton (same instance)
        instance1 = manager.resolve("singleton_service")
        instance2 = manager.resolve("singleton_service")
        assert instance1 is instance2

        # Test transient (different instances)
        instance1 = manager.resolve("transient_service")
        instance2 = manager.resolve("transient_service")
        assert instance1 is not instance2

        # Test scoped (same instance within scope)
        scope_id = "test_scope"
        manager.create_scope(scope_id)

        instance1 = manager.resolve("scoped_service", scope_id=scope_id)
        instance2 = manager.resolve("scoped_service", scope_id=scope_id)
        assert instance1 is instance2

    @pytest.mark.asyncio
    async def test_dependency_monitoring(self):
        """Test dependency monitoring and metrics."""
        manager = DependencyManager()

        # Register service with monitoring
        manager.register_service("monitored_service", lambda: "instance")

        # Resolve service multiple times
        for _ in range(5):
            manager.resolve("monitored_service")

        # Check metrics
        metrics = manager.monitor.get_metrics("monitored_service")
        assert metrics["total_resolutions"] == 5
        assert metrics["successful_resolutions"] == 5
        assert metrics["success_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_dependency_validation(self):
        """Test dependency validation."""
        manager = DependencyManager()

        # Register valid service
        manager.register_service("valid_service", lambda: "instance")

        # Should not raise exception
        manager.validate_services()

        # Register invalid service (circular dependency)
        def service_a_factory(service_b):
            return f"service_a_with_{service_b}"

        def service_b_factory(service_a):
            return f"service_b_with_{service_a}"

        manager.register_service(
            "service_a", service_a_factory, dependencies=["service_b"]
        )
        manager.register_service(
            "service_b", service_b_factory, dependencies=["service_a"]
        )

        # Should raise exception when resolving
        with pytest.raises(DependencyCircularReferenceError):
            manager.resolve("service_a")
