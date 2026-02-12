"""
Simple tests for ai_engine.core.dependency_manager module.
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
            min_version="1.0.0",
        )

        assert info.name == "test_package"
        assert info.package == "test_package"
        assert info.required is True
        assert info.min_version == "1.0.0"
        assert info.status == DependencyStatus.MISSING
        assert info.installed_version is None
        assert info.error_message is None
        assert info.fallback_available is False

    def test_initialization_with_all_fields(self):
        """Test DependencyInfo initialization with all fields."""
        info = DependencyInfo(
            name="test_package",
            package="test_package",
            required=False,
            min_version="2.0.0",
            status=DependencyStatus.AVAILABLE,
            installed_version="2.1.0",
            error_message="No error",
            fallback_available=True,
        )

        assert info.name == "test_package"
        assert info.package == "test_package"
        assert info.required is False
        assert info.min_version == "2.0.0"
        assert info.status == DependencyStatus.AVAILABLE
        assert info.installed_version == "2.1.0"
        assert info.error_message == "No error"
        assert info.fallback_available is True


class TestDependencyManager:
    """Test DependencyManager class."""

    @pytest.fixture
    def dependency_manager(self):
        """Create DependencyManager instance."""
        return DependencyManager()

    def test_initialization(self, dependency_manager):
        """Test DependencyManager initialization."""
        assert dependency_manager is not None
        assert hasattr(dependency_manager, "LLM_DEPENDENCIES")
        assert hasattr(dependency_manager, "ML_DEPENDENCIES")
        assert hasattr(dependency_manager, "INFRA_DEPENDENCIES")

    def test_get_dependency_info(self, dependency_manager):
        """Test getting dependency info."""
        info = dependency_manager.get_dependency_info("openai")

        assert info is not None
        assert info.name == "OpenAI SDK"
        assert info.package == "openai"
        assert info.required is False

    def test_get_dependency_info_not_found(self, dependency_manager):
        """Test getting dependency info for non-existent dependency."""
        info = dependency_manager.get_dependency_info("nonexistent")

        assert info is None

    def test_check_dependency_available(self, dependency_manager):
        """Test checking if dependency is available."""
        with patch("importlib.import_module") as mock_import:
            mock_import.return_value = MagicMock()

            is_available = dependency_manager.check_dependency_available("openai")

            assert is_available is True

    def test_check_dependency_available_import_error(self, dependency_manager):
        """Test checking dependency availability with import error."""
        with patch("importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("No module named 'openai'")

            is_available = dependency_manager.check_dependency_available("openai")

            assert is_available is False

    def test_get_installed_version(self, dependency_manager):
        """Test getting installed version of dependency."""
        with patch("importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.__version__ = "1.10.0"
            mock_import.return_value = mock_module

            version = dependency_manager.get_installed_version("openai")

            assert version == "1.10.0"

    def test_get_installed_version_no_version(self, dependency_manager):
        """Test getting installed version when module has no version."""
        with patch("importlib.import_module") as mock_import:
            mock_module = MagicMock()
            del mock_module.__version__
            mock_import.return_value = mock_module

            version = dependency_manager.get_installed_version("openai")

            assert version is None

    def test_get_installed_version_import_error(self, dependency_manager):
        """Test getting installed version with import error."""
        with patch("importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("No module named 'openai'")

            version = dependency_manager.get_installed_version("openai")

            assert version is None

    def test_validate_dependency_requirements(self, dependency_manager):
        """Test validating dependency requirements."""
        with patch("importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.__version__ = "1.10.0"
            mock_import.return_value = mock_module

            is_valid = dependency_manager.validate_dependency_requirements("openai")

            assert is_valid is True

    def test_validate_dependency_requirements_version_mismatch(self, dependency_manager):
        """Test validating dependency requirements with version mismatch."""
        with patch("importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.__version__ = "0.9.0"  # Below min version
            mock_import.return_value = mock_module

            is_valid = dependency_manager.validate_dependency_requirements("openai")

            assert is_valid is False

    def test_get_all_dependencies(self, dependency_manager):
        """Test getting all dependencies."""
        all_deps = dependency_manager.get_all_dependencies()

        assert isinstance(all_deps, dict)
        assert "openai" in all_deps
        assert "torch" in all_deps
        assert "redis" in all_deps

    def test_get_llm_dependencies(self, dependency_manager):
        """Test getting LLM dependencies."""
        llm_deps = dependency_manager.get_llm_dependencies()

        assert isinstance(llm_deps, dict)
        assert "openai" in llm_deps
        assert "anthropic" in llm_deps

    def test_get_ml_dependencies(self, dependency_manager):
        """Test getting ML dependencies."""
        ml_deps = dependency_manager.get_ml_dependencies()

        assert isinstance(ml_deps, dict)
        assert "torch" in ml_deps
        assert "transformers" in ml_deps

    def test_get_infra_dependencies(self, dependency_manager):
        """Test getting infrastructure dependencies."""
        infra_deps = dependency_manager.get_infra_dependencies()

        assert isinstance(infra_deps, dict)
        assert "redis" in infra_deps
        assert "etcd3" in infra_deps

    def test_get_missing_dependencies(self, dependency_manager):
        """Test getting missing dependencies."""
        with patch.object(dependency_manager, "check_dependency_available", return_value=False):
            missing = dependency_manager.get_missing_dependencies()

            assert isinstance(missing, list)
            assert len(missing) > 0

    def test_get_available_dependencies(self, dependency_manager):
        """Test getting available dependencies."""
        with patch.object(dependency_manager, "check_dependency_available", return_value=True):
            available = dependency_manager.get_available_dependencies()

            assert isinstance(available, list)
            assert len(available) > 0

    def test_get_dependency_status_summary(self, dependency_manager):
        """Test getting dependency status summary."""
        with patch.object(dependency_manager, "check_dependency_available") as mock_check:
            mock_check.return_value = True

            summary = dependency_manager.get_dependency_status_summary()

            assert isinstance(summary, dict)
            assert "total" in summary
            assert "available" in summary
            assert "missing" in summary
            assert "version_mismatch" in summary
