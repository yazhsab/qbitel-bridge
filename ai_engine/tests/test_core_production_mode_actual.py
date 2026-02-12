"""
Tests for ai_engine.core.production_mode module - actual implementation.
"""

import pytest
import os
from unittest.mock import patch

from ai_engine.core.production_mode import (
    EnvironmentMode,
    ProductionModeDetector,
    ProductionValidator,
)


class TestEnvironmentMode:
    """Test EnvironmentMode enum."""

    def test_environment_mode_values(self):
        """Test EnvironmentMode enum values."""
        assert EnvironmentMode.PRODUCTION.value == "production"
        assert EnvironmentMode.STAGING.value == "staging"
        assert EnvironmentMode.DEVELOPMENT.value == "development"
        assert EnvironmentMode.TESTING.value == "testing"


class TestProductionModeDetector:
    """Test ProductionModeDetector class."""

    def test_is_production_mode_production(self):
        """Test production mode detection when in production."""
        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "production"}):
            result = ProductionModeDetector.is_production_mode()
            assert result is True

    def test_is_production_mode_prod(self):
        """Test production mode detection when using 'prod' alias."""
        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "prod"}):
            result = ProductionModeDetector.is_production_mode()
            assert result is True

    def test_is_production_mode_development(self):
        """Test production mode detection when in development."""
        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "development"}):
            result = ProductionModeDetector.is_production_mode()
            assert result is False

    def test_is_production_mode_staging(self):
        """Test production mode detection when in staging."""
        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "staging"}):
            result = ProductionModeDetector.is_production_mode()
            assert result is False

    def test_is_production_mode_fallback_environment(self):
        """Test production mode detection with fallback to ENVIRONMENT variable."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}, clear=True):
            result = ProductionModeDetector.is_production_mode()
            assert result is True

    def test_is_production_mode_default(self):
        """Test production mode detection with default value."""
        with patch.dict(os.environ, {}, clear=True):
            result = ProductionModeDetector.is_production_mode()
            assert result is False

    def test_get_environment_mode_production(self):
        """Test getting environment mode for production."""
        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "production"}):
            mode = ProductionModeDetector.get_environment_mode()
            assert mode == EnvironmentMode.PRODUCTION

    def test_get_environment_mode_staging(self):
        """Test getting environment mode for staging."""
        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "staging"}):
            mode = ProductionModeDetector.get_environment_mode()
            assert mode == EnvironmentMode.STAGING

    def test_get_environment_mode_development(self):
        """Test getting environment mode for development."""
        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "development"}):
            mode = ProductionModeDetector.get_environment_mode()
            assert mode == EnvironmentMode.DEVELOPMENT

    def test_get_environment_mode_testing(self):
        """Test getting environment mode for testing."""
        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "testing"}):
            mode = ProductionModeDetector.get_environment_mode()
            assert mode == EnvironmentMode.TESTING

    def test_get_environment_mode_fallback_environment(self):
        """Test getting environment mode with fallback to ENVIRONMENT variable."""
        with patch.dict(os.environ, {"ENVIRONMENT": "staging"}, clear=True):
            mode = ProductionModeDetector.get_environment_mode()
            assert mode == EnvironmentMode.STAGING

    def test_get_environment_mode_default(self):
        """Test getting environment mode with default value."""
        with patch.dict(os.environ, {}, clear=True):
            mode = ProductionModeDetector.get_environment_mode()
            assert mode == EnvironmentMode.DEVELOPMENT

    def test_get_environment_mode_invalid(self):
        """Test getting environment mode with invalid value."""
        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "invalid"}):
            mode = ProductionModeDetector.get_environment_mode()
            assert mode == EnvironmentMode.DEVELOPMENT

    def test_validate_production_environment_production(self):
        """Test validating production environment when in production."""
        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "production"}):
            is_valid, errors = ProductionModeDetector.validate_production_environment()
            # Should be invalid because required env vars are missing
            assert is_valid is False
            assert len(errors) > 0

    def test_validate_production_environment_development(self):
        """Test validating production environment when in development."""
        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "development"}):
            is_valid, errors = ProductionModeDetector.validate_production_environment()
            assert is_valid is True
            assert len(errors) == 0

    def test_validate_production_environment_with_required_vars(self):
        """Test validating production environment with required variables."""
        with patch.dict(
            os.environ,
            {
                "QBITEL_AI_ENVIRONMENT": "production",
                "QBITEL_AI_DB_PASSWORD": "test_password",
                "QBITEL_AI_REDIS_PASSWORD": "test_redis_password",
                "QBITEL_AI_JWT_SECRET": "test_jwt_secret",
                "QBITEL_AI_ENCRYPTION_KEY": "test_encryption_key",
            },
        ):
            is_valid, errors = ProductionModeDetector.validate_production_environment()
            assert is_valid is True
            assert len(errors) == 0

    def test_validate_production_environment_missing_vars(self):
        """Test validating production environment with missing variables."""
        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "production"}):
            is_valid, errors = ProductionModeDetector.validate_production_environment()
            assert is_valid is False
            assert len(errors) > 0
            # Check that error messages mention the missing variables
            error_text = " ".join(errors)
            assert "QBITEL_AI_DB_PASSWORD" in error_text
            assert "QBITEL_AI_REDIS_PASSWORD" in error_text
            assert "QBITEL_AI_JWT_SECRET" in error_text
            assert "QBITEL_AI_ENCRYPTION_KEY" in error_text

    def test_log_environment_info(self):
        """Test logging environment information."""
        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "production"}):
            # This should not raise an exception
            ProductionModeDetector.log_environment_info()

    def test_production_aliases(self):
        """Test production aliases."""
        assert "production" in ProductionModeDetector.PRODUCTION_ALIASES
        assert "prod" in ProductionModeDetector.PRODUCTION_ALIASES


class TestProductionValidator:
    """Test ProductionValidator class."""

    @pytest.fixture
    def validator(self):
        """Create ProductionValidator instance."""
        return ProductionValidator()

    def test_initialization(self, validator):
        """Test ProductionValidator initialization."""
        assert validator is not None

    def test_validate_production_environment_production(self, validator):
        """Test validating production environment when in production."""
        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "production"}):
            result = validator.validate_production_environment()
            # Should be invalid because required env vars are missing
            assert result is False

    def test_validate_production_environment_development(self, validator):
        """Test validating production environment when in development."""
        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "development"}):
            result = validator.validate_production_environment()
            assert result is True

    def test_validate_production_environment_staging(self, validator):
        """Test validating production environment when in staging."""
        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "staging"}):
            result = validator.validate_production_environment()
            assert result is True

    def test_validate_production_environment_testing(self, validator):
        """Test validating production environment when in testing."""
        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "testing"}):
            result = validator.validate_production_environment()
            assert result is True

    def test_validate_production_environment_with_required_vars(self, validator):
        """Test validating production environment with required variables."""
        with patch.dict(
            os.environ,
            {
                "QBITEL_AI_ENVIRONMENT": "production",
                "QBITEL_AI_DB_PASSWORD": "test_password",
                "QBITEL_AI_REDIS_PASSWORD": "test_redis_password",
                "QBITEL_AI_JWT_SECRET": "test_jwt_secret",
                "QBITEL_AI_ENCRYPTION_KEY": "test_encryption_key",
            },
        ):
            result = validator.validate_production_environment()
            assert result is True

    def test_validate_production_environment_missing_vars(self, validator):
        """Test validating production environment with missing variables."""
        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "production"}):
            result = validator.validate_production_environment()
            assert result is False

    def test_get_validation_results(self, validator):
        """Test getting validation results."""
        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "production"}):
            results = validator.get_validation_results()

            assert isinstance(results, dict)
            assert "environment_check" in results
            assert "required_variables_check" in results

    def test_get_validation_summary(self, validator):
        """Test getting validation summary."""
        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "production"}):
            summary = validator.get_validation_summary()

            assert isinstance(summary, dict)
            assert "total_checks" in summary
            assert "passed_checks" in summary
            assert "failed_checks" in summary
            assert "success_rate" in summary
            assert summary["total_checks"] > 0

    def test_validate_production_environment_with_custom_checks(self, validator):
        """Test validating production environment with custom checks."""

        def custom_check():
            return True

        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "production"}):
            result = validator.validate_production_environment(custom_checks=[custom_check])
            # Should still be False because required env vars are missing
            assert result is False

    def test_validate_production_environment_with_failing_custom_checks(self, validator):
        """Test validating production environment with failing custom checks."""

        def custom_check():
            return False

        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "production"}):
            result = validator.validate_production_environment(custom_checks=[custom_check])
            assert result is False

    def test_validate_production_environment_with_exception_in_custom_check(self, validator):
        """Test validating production environment with exception in custom check."""

        def custom_check():
            raise Exception("Custom check failed")

        with patch.dict(os.environ, {"QBITEL_AI_ENVIRONMENT": "production"}):
            result = validator.validate_production_environment(custom_checks=[custom_check])
            assert result is False
