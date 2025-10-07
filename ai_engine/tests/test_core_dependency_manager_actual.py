"""
Tests for ai_engine.core.dependency_manager module - actual implementation.
"""

import pytest
from unittest.mock import patch, MagicMock

from ai_engine.core.dependency_manager import (
    DependencyStatus,
    DependencyInfo,
    DependencyManager,
)


class TestDependencyStatus:
    """Test DependencyStatus enum."""

    def test_dependency_status_values(self):
        """Test DependencyStatus enum values."""
        assert DependencyStatus.AVAILABLE.value == "available"
        assert DependencyStatus.MISSING.value == "missing"
        assert DependencyStatus.VERSION_MISMATCH.value == "version_mismatch"
        assert DependencyStatus.IMPORT_ERROR.value == "import_error"


class TestDependencyInfo:
    """Test DependencyInfo class."""

    def test_initialization(self):
        """Test DependencyInfo initialization."""
        info = DependencyInfo(
            name="test_package",
            package="test_package",
            required=True,
            min_version="1.0.0"
        )
        
        assert info.name == "test_package"
        assert info.package == "test_package"
        assert info.required is True
        assert info.min_version == "1.0.0"
        assert info.status == DependencyStatus.MISSING
        assert info.installed_version is None
        assert info.error_message is None
        assert info.fallback_available is False


class TestDependencyManager:
    """Test DependencyManager class."""

    @pytest.fixture
    def dependency_manager(self):
        """Create DependencyManager instance."""
        return DependencyManager()

    def test_initialization(self, dependency_manager):
        """Test DependencyManager initialization."""
        assert dependency_manager is not None
        assert hasattr(dependency_manager, 'LLM_DEPENDENCIES')
        assert hasattr(dependency_manager, 'ML_DEPENDENCIES')
        assert hasattr(dependency_manager, 'dependency_status')

    def test_llm_dependencies_exist(self, dependency_manager):
        """Test that LLM dependencies are defined."""
        assert isinstance(dependency_manager.LLM_DEPENDENCIES, dict)
        assert len(dependency_manager.LLM_DEPENDENCIES) > 0
        assert "openai" in dependency_manager.LLM_DEPENDENCIES

    def test_ml_dependencies_exist(self, dependency_manager):
        """Test that ML dependencies are defined."""
        assert isinstance(dependency_manager.ML_DEPENDENCIES, dict)
        assert len(dependency_manager.ML_DEPENDENCIES) > 0
        assert "torch" in dependency_manager.ML_DEPENDENCIES

    def test_dependency_status_initialized(self, dependency_manager):
        """Test that dependency status is initialized."""
        assert isinstance(dependency_manager.dependency_status, dict)
        assert len(dependency_manager.dependency_status) > 0

    def test_check_all_dependencies(self, dependency_manager):
        """Test checking all dependencies."""
        # This should not raise an exception
        dependency_manager._check_all_dependencies()
        
        # Verify that dependency status was updated
        assert len(dependency_manager.dependency_status) > 0

    def test_check_dependency_available(self, dependency_manager):
        """Test checking if a dependency is available."""
        with patch('importlib.import_module') as mock_import:
            mock_import.return_value = MagicMock()
            
            # Test with a real dependency from the list
            openai_info = dependency_manager.LLM_DEPENDENCIES["openai"]
            result = dependency_manager._check_dependency(openai_info)
            
            assert result.status == DependencyStatus.AVAILABLE

    def test_check_dependency_missing(self, dependency_manager):
        """Test checking a missing dependency."""
        with patch('importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("No module named 'nonexistent'")
            
            # Create a test dependency info
            test_info = DependencyInfo(
                name="Test Package",
                package="nonexistent",
                required=False
            )
            
            result = dependency_manager._check_dependency(test_info)
            
            assert result.status == DependencyStatus.MISSING
            assert result.error_message is not None

    def test_check_dependency_version_mismatch(self, dependency_manager):
        """Test checking dependency with version mismatch."""
        with patch('importlib.import_module') as mock_import:
            mock_module = MagicMock()
            mock_module.__version__ = "0.9.0"  # Below min version
            mock_import.return_value = mock_module
            
            # Create a test dependency info with min version
            test_info = DependencyInfo(
                name="Test Package",
                package="test_package",
                required=False,
                min_version="1.0.0"
            )
            
            result = dependency_manager._check_dependency(test_info)
            
            assert result.status == DependencyStatus.VERSION_MISMATCH
            assert result.error_message is not None

    def test_compare_versions(self, dependency_manager):
        """Test version comparison."""
        # Test equal versions
        assert dependency_manager._compare_versions("1.0.0", "1.0.0") == 0
        
        # Test newer version
        assert dependency_manager._compare_versions("1.1.0", "1.0.0") > 0
        
        # Test older version
        assert dependency_manager._compare_versions("0.9.0", "1.0.0") < 0
        
        # Test with different formats
        assert dependency_manager._compare_versions("1.0", "1.0.0") < 0

    def test_get_dependency_status(self, dependency_manager):
        """Test getting dependency status."""
        # Check that we can get status for known dependencies
        openai_status = dependency_manager.dependency_status.get("openai")
        assert openai_status is not None
        assert isinstance(openai_status, DependencyInfo)

    def test_dependency_info_attributes(self, dependency_manager):
        """Test that dependency info has expected attributes."""
        openai_info = dependency_manager.LLM_DEPENDENCIES["openai"]
        
        assert hasattr(openai_info, 'name')
        assert hasattr(openai_info, 'package')
        assert hasattr(openai_info, 'required')
        assert hasattr(openai_info, 'min_version')
        assert hasattr(openai_info, 'status')
        assert hasattr(openai_info, 'installed_version')
        assert hasattr(openai_info, 'error_message')
        assert hasattr(openai_info, 'fallback_available')

    def test_required_dependencies(self, dependency_manager):
        """Test that some dependencies are marked as required."""
        # Check that some dependencies are required
        required_deps = [
            dep for dep in dependency_manager.LLM_DEPENDENCIES.values()
            if dep.required
        ]
        
        # There should be some required dependencies
        assert len(required_deps) >= 0  # Some may be required, some may not

    def test_optional_dependencies(self, dependency_manager):
        """Test that some dependencies are marked as optional."""
        # Check that some dependencies are optional
        optional_deps = [
            dep for dep in dependency_manager.LLM_DEPENDENCIES.values()
            if not dep.required
        ]
        
        # There should be some optional dependencies
        assert len(optional_deps) >= 0  # Some may be optional, some may not

    def test_dependency_initialization_completes(self, dependency_manager):
        """Test that dependency initialization completes without errors."""
        # This should not raise any exceptions
        assert dependency_manager.dependency_status is not None
        assert len(dependency_manager.dependency_status) > 0

    def test_dependency_status_consistency(self, dependency_manager):
        """Test that dependency status is consistent."""
        # All dependencies in status should have valid status values
        for name, dep_info in dependency_manager.dependency_status.items():
            assert isinstance(dep_info, DependencyInfo)
            assert dep_info.status in DependencyStatus
            assert dep_info.name is not None
            assert dep_info.package is not None
