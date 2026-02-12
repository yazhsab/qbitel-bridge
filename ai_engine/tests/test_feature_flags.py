"""
QBITEL Engine - Feature Flags Tests

Tests for the feature flag system including environment variable loading,
runtime toggling, and dependency checking.
"""

import os
import pytest
from unittest.mock import patch

from ai_engine.core.feature_flags import (
    FeatureFlags,
    FeatureDefinition,
    FeatureCategory,
    FEATURE_DEFINITIONS,
    feature_flags,
    is_feature_enabled,
    require_feature,
    feature_guard,
)


class TestFeatureDefinition:
    """Tests for FeatureDefinition dataclass."""

    def test_basic_definition(self):
        """Test creating a basic feature definition."""
        feature = FeatureDefinition(
            name="test_feature",
            description="A test feature",
            category=FeatureCategory.CORE,
            default_enabled=True,
        )
        assert feature.name == "test_feature"
        assert feature.description == "A test feature"
        assert feature.category == FeatureCategory.CORE
        assert feature.default_enabled is True
        assert feature.requires == set()
        assert feature.env_var is None

    def test_definition_with_dependencies(self):
        """Test feature definition with dependencies."""
        feature = FeatureDefinition(
            name="dependent_feature",
            description="Feature with dependencies",
            category=FeatureCategory.EXPERIMENTAL,
            default_enabled=False,
            requires={"base_feature", "other_feature"},
            env_var="CUSTOM_ENV_VAR",
        )
        assert feature.requires == {"base_feature", "other_feature"}
        assert feature.env_var == "CUSTOM_ENV_VAR"


class TestFeatureFlags:
    """Tests for FeatureFlags class."""

    def setup_method(self):
        """Reset feature flags before each test."""
        self.ff = FeatureFlags()

    def test_unknown_feature_returns_false(self):
        """Test that unknown features default to disabled."""
        assert self.ff.is_enabled("nonexistent_feature") is False

    def test_default_enabled_feature(self):
        """Test feature with default_enabled=True."""
        # protocol_discovery has default_enabled=True
        with patch.dict(os.environ, {}, clear=True):
            ff = FeatureFlags()
            assert ff.is_enabled("protocol_discovery") is True

    def test_default_disabled_feature(self):
        """Test feature with default_enabled=False."""
        # healthcare_domain has default_enabled=False
        with patch.dict(os.environ, {}, clear=True):
            ff = FeatureFlags()
            assert ff.is_enabled("healthcare_domain") is False

    def test_env_var_enables_feature(self):
        """Test enabling feature via environment variable."""
        with patch.dict(os.environ, {"QBITEL_FEATURE_HEALTHCARE_DOMAIN": "true"}):
            ff = FeatureFlags()
            assert ff.is_enabled("healthcare_domain") is True

    def test_env_var_disables_feature(self):
        """Test disabling feature via environment variable."""
        with patch.dict(os.environ, {"QBITEL_FEATURE_PROTOCOL_DISCOVERY": "false"}):
            ff = FeatureFlags()
            assert ff.is_enabled("protocol_discovery") is False

    @pytest.mark.parametrize("value", ["true", "1", "yes", "on", "enabled", "TRUE", "True"])
    def test_truthy_env_values(self, value):
        """Test various truthy environment variable values."""
        with patch.dict(os.environ, {"QBITEL_FEATURE_MARKETPLACE": value}):
            ff = FeatureFlags()
            assert ff.is_enabled("marketplace") is True

    @pytest.mark.parametrize("value", ["false", "0", "no", "off", "disabled", "FALSE", "False"])
    def test_falsy_env_values(self, value):
        """Test various falsy environment variable values."""
        with patch.dict(os.environ, {"QBITEL_FEATURE_PROTOCOL_DISCOVERY": value}):
            ff = FeatureFlags()
            assert ff.is_enabled("protocol_discovery") is False

    def test_runtime_enable(self):
        """Test enabling feature at runtime."""
        self.ff.disable("protocol_discovery")
        assert self.ff.is_enabled("protocol_discovery") is False

        self.ff.enable("protocol_discovery")
        assert self.ff.is_enabled("protocol_discovery") is True

    def test_runtime_disable(self):
        """Test disabling feature at runtime."""
        self.ff.enable("healthcare_domain")
        assert self.ff.is_enabled("healthcare_domain") is True

        self.ff.disable("healthcare_domain")
        assert self.ff.is_enabled("healthcare_domain") is False

    def test_reset_single_feature(self):
        """Test resetting a single feature override."""
        self.ff.enable("marketplace")
        assert self.ff.is_enabled("marketplace") is True

        self.ff.reset("marketplace")
        # Should return to default (False for marketplace)
        assert self.ff.is_enabled("marketplace") is False

    def test_reset_all_features(self):
        """Test resetting all feature overrides."""
        self.ff.enable("marketplace")
        self.ff.enable("healthcare_domain")
        self.ff.disable("protocol_discovery")

        self.ff.reset()

        # All should return to defaults
        assert self.ff.is_enabled("marketplace") is False
        assert self.ff.is_enabled("healthcare_domain") is False
        assert self.ff.is_enabled("protocol_discovery") is True

    def test_get_all_flags(self):
        """Test getting all flag states."""
        flags = self.ff.get_all_flags()

        assert isinstance(flags, dict)
        assert "protocol_discovery" in flags
        assert "healthcare_domain" in flags
        assert all(isinstance(v, bool) for v in flags.values())

    def test_get_enabled_features(self):
        """Test getting set of enabled features."""
        enabled = self.ff.get_enabled_features()

        assert isinstance(enabled, set)
        # protocol_discovery is enabled by default
        assert "protocol_discovery" in enabled

    def test_get_disabled_features(self):
        """Test getting set of disabled features."""
        disabled = self.ff.get_disabled_features()

        assert isinstance(disabled, set)
        # marketplace is disabled by default
        assert "marketplace" in disabled

    def test_get_features_by_category(self):
        """Test filtering features by category."""
        domain_features = self.ff.get_features_by_category(FeatureCategory.DOMAIN)

        assert "healthcare_domain" in domain_features
        assert "automotive_domain" in domain_features
        assert "aviation_domain" in domain_features
        assert "industrial_domain" in domain_features
        # Core features should not be included
        assert "protocol_discovery" not in domain_features

    def test_get_feature_info(self):
        """Test getting detailed feature information."""
        info = self.ff.get_feature_info("healthcare_domain")

        assert info is not None
        assert info["name"] == "healthcare_domain"
        assert "description" in info
        assert info["category"] == "domain"
        assert info["default_enabled"] is False
        assert info["env_var"] == "QBITEL_FEATURE_HEALTHCARE_DOMAIN"
        assert "enabled" in info

    def test_get_feature_info_unknown_feature(self):
        """Test getting info for unknown feature returns None."""
        info = self.ff.get_feature_info("nonexistent_feature")
        assert info is None

    def test_get_status_report(self):
        """Test comprehensive status report."""
        report = self.ff.get_status_report()

        assert "total_features" in report
        assert "enabled_count" in report
        assert "disabled_count" in report
        assert "enabled_features" in report
        assert "disabled_features" in report
        assert "by_category" in report
        assert "overrides_active" in report

        # Verify counts match
        assert report["enabled_count"] + report["disabled_count"] == report["total_features"]

    def test_cache_behavior(self):
        """Test that feature flag values are cached."""
        # First call populates cache
        result1 = self.ff.is_enabled("protocol_discovery")

        # Should use cached value
        result2 = self.ff.is_enabled("protocol_discovery")

        assert result1 == result2

        # Override should clear cache
        self.ff.disable("protocol_discovery")
        assert self.ff.is_enabled("protocol_discovery") is False


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def setup_method(self):
        """Reset global feature flags before each test."""
        feature_flags.reset()

    def test_is_feature_enabled(self):
        """Test is_feature_enabled convenience function."""
        # Uses global singleton
        assert is_feature_enabled("protocol_discovery") is True
        assert is_feature_enabled("marketplace") is False

    def test_require_feature_enabled(self):
        """Test require_feature with enabled feature."""
        # Should not raise for enabled feature
        require_feature("protocol_discovery")

    def test_require_feature_disabled(self):
        """Test require_feature with disabled feature raises error."""
        with pytest.raises(RuntimeError) as exc_info:
            require_feature("marketplace")

        assert "marketplace" in str(exc_info.value)
        assert "not enabled" in str(exc_info.value)


class TestFeatureGuardDecorator:
    """Tests for feature_guard decorator."""

    def setup_method(self):
        """Reset global feature flags before each test."""
        feature_flags.reset()

    def test_feature_guard_enabled(self):
        """Test feature guard allows call when feature is enabled."""

        @feature_guard("protocol_discovery")
        def my_function():
            return "success"

        result = my_function()
        assert result == "success"

    def test_feature_guard_disabled(self):
        """Test feature guard blocks call when feature is disabled."""

        @feature_guard("marketplace")
        def my_function():
            return "success"

        with pytest.raises(RuntimeError) as exc_info:
            my_function()

        assert "marketplace" in str(exc_info.value)
        assert "disabled" in str(exc_info.value)

    def test_feature_guard_preserves_metadata(self):
        """Test that feature guard preserves function name and docstring."""

        @feature_guard("protocol_discovery")
        def documented_function():
            """This is a documented function."""
            return "success"

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is a documented function."


class TestFeatureDefinitions:
    """Tests for the predefined feature definitions."""

    def test_all_categories_have_features(self):
        """Test that all categories have at least one feature."""
        categories_with_features = set()
        for definition in FEATURE_DEFINITIONS.values():
            categories_with_features.add(definition.category)

        # All categories should be represented
        for category in FeatureCategory:
            assert category in categories_with_features, f"No features in category {category}"

    def test_all_features_have_env_vars(self):
        """Test that all features have environment variable configuration."""
        for name, definition in FEATURE_DEFINITIONS.items():
            env_var = definition.env_var or f"QBITEL_FEATURE_{name.upper()}"
            assert env_var.startswith("QBITEL_FEATURE_"), f"Invalid env var for {name}"

    def test_domain_features_disabled_by_default(self):
        """Test that domain features are disabled by default."""
        domain_features = [
            "healthcare_domain",
            "automotive_domain",
            "aviation_domain",
            "industrial_domain",
        ]
        for name in domain_features:
            assert FEATURE_DEFINITIONS[name].default_enabled is False, f"{name} should be disabled by default"

    def test_core_features_enabled_by_default(self):
        """Test that core features are enabled by default."""
        core_features = [
            "protocol_discovery",
            "field_detection",
            "legacy_whisperer",
        ]
        for name in core_features:
            assert FEATURE_DEFINITIONS[name].default_enabled is True, f"{name} should be enabled by default"

    def test_feature_names_are_valid_identifiers(self):
        """Test that all feature names are valid Python identifiers."""
        import keyword

        for name in FEATURE_DEFINITIONS.keys():
            assert name.isidentifier(), f"'{name}' is not a valid identifier"
            assert not keyword.iskeyword(name), f"'{name}' is a Python keyword"


class TestFeatureDependencies:
    """Tests for feature dependency handling."""

    def test_feature_with_dependency(self):
        """Test that features with unmet dependencies are disabled."""
        # Create a test feature flags instance with a dependency
        ff = FeatureFlags()

        # Add a temporary definition with dependency
        test_def = FeatureDefinition(
            name="test_dependent",
            description="Test dependent feature",
            category=FeatureCategory.EXPERIMENTAL,
            default_enabled=True,
            requires={"marketplace"},  # marketplace is disabled by default
        )

        # Monkey-patch for testing
        from ai_engine.core import feature_flags as ff_module

        original = ff_module.FEATURE_DEFINITIONS.copy()

        try:
            ff_module.FEATURE_DEFINITIONS["test_dependent"] = test_def
            ff = FeatureFlags()

            # Should be disabled because dependency is not met
            assert ff.is_enabled("test_dependent") is False

            # Enable dependency
            ff.enable("marketplace")
            ff.reset("test_dependent")  # Clear cache

            # Now should be enabled
            assert ff.is_enabled("test_dependent") is True
        finally:
            # Restore original definitions
            ff_module.FEATURE_DEFINITIONS.clear()
            ff_module.FEATURE_DEFINITIONS.update(original)
