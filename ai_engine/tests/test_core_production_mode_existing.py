"""
Comprehensive tests for ai_engine.core.production_mode module.

This module provides production mode detection and validation for the
CRONOS AI Engine components.
"""

import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional, List
import inspect

from ai_engine.core.production_mode import (
    ProductionModeDetector,
    ProductionValidator,
)


class TestProductionModeDetector:
    """Test ProductionModeDetector class."""

    @pytest.fixture
    def detector(self):
        """Create ProductionModeDetector instance."""
        return ProductionModeDetector()

    def test_initialization(self, detector):
        """Test ProductionModeDetector initialization."""
        assert detector._production_indicators == []
        assert detector._environment_variables == []
        assert detector._file_indicators == []
        assert detector._network_indicators == []
        assert detector._system_indicators == []
        assert detector._custom_indicators == []
        assert detector._threshold == 0.5
        assert detector._weights == {}
        assert detector._cache == {}
        assert detector._cache_ttl == 300

    def test_add_production_indicator(self, detector):
        """Test adding production indicator."""
        def indicator():
            return True
        
        detector.add_production_indicator("test_indicator", indicator)
        
        assert "test_indicator" in detector._production_indicators
        assert detector._production_indicators["test_indicator"] == indicator

    def test_add_production_indicator_duplicate(self, detector):
        """Test adding duplicate production indicator."""
        def indicator1():
            return True
        
        def indicator2():
            return False
        
        detector.add_production_indicator("test_indicator", indicator1)
        
        with pytest.raises(ValueError, match="Production indicator already exists"):
            detector.add_production_indicator("test_indicator", indicator2)

    def test_remove_production_indicator(self, detector):
        """Test removing production indicator."""
        def indicator():
            return True
        
        detector.add_production_indicator("test_indicator", indicator)
        detector.remove_production_indicator("test_indicator")
        
        assert "test_indicator" not in detector._production_indicators

    def test_remove_production_indicator_not_found(self, detector):
        """Test removing non-existent production indicator."""
        with pytest.raises(ValueError, match="Production indicator not found"):
            detector.remove_production_indicator("nonexistent_indicator")

    def test_add_environment_variable(self, detector):
        """Test adding environment variable indicator."""
        detector.add_environment_variable("PRODUCTION", "true")
        
        assert "PRODUCTION" in detector._environment_variables
        assert detector._environment_variables["PRODUCTION"] == "true"

    def test_add_environment_variable_duplicate(self, detector):
        """Test adding duplicate environment variable indicator."""
        detector.add_environment_variable("PRODUCTION", "true")
        
        with pytest.raises(ValueError, match="Environment variable already exists"):
            detector.add_environment_variable("PRODUCTION", "false")

    def test_remove_environment_variable(self, detector):
        """Test removing environment variable indicator."""
        detector.add_environment_variable("PRODUCTION", "true")
        detector.remove_environment_variable("PRODUCTION")
        
        assert "PRODUCTION" not in detector._environment_variables

    def test_remove_environment_variable_not_found(self, detector):
        """Test removing non-existent environment variable indicator."""
        with pytest.raises(ValueError, match="Environment variable not found"):
            detector.remove_environment_variable("nonexistent_var")

    def test_add_file_indicator(self, detector):
        """Test adding file indicator."""
        detector.add_file_indicator("/path/to/production.flag")
        
        assert "/path/to/production.flag" in detector._file_indicators

    def test_add_file_indicator_duplicate(self, detector):
        """Test adding duplicate file indicator."""
        detector.add_file_indicator("/path/to/production.flag")
        
        with pytest.raises(ValueError, match="File indicator already exists"):
            detector.add_file_indicator("/path/to/production.flag")

    def test_remove_file_indicator(self, detector):
        """Test removing file indicator."""
        detector.add_file_indicator("/path/to/production.flag")
        detector.remove_file_indicator("/path/to/production.flag")
        
        assert "/path/to/production.flag" not in detector._file_indicators

    def test_remove_file_indicator_not_found(self, detector):
        """Test removing non-existent file indicator."""
        with pytest.raises(ValueError, match="File indicator not found"):
            detector.remove_file_indicator("/path/to/nonexistent.flag")

    def test_add_network_indicator(self, detector):
        """Test adding network indicator."""
        detector.add_network_indicator("production.example.com", 443)
        
        assert ("production.example.com", 443) in detector._network_indicators

    def test_add_network_indicator_duplicate(self, detector):
        """Test adding duplicate network indicator."""
        detector.add_network_indicator("production.example.com", 443)
        
        with pytest.raises(ValueError, match="Network indicator already exists"):
            detector.add_network_indicator("production.example.com", 443)

    def test_remove_network_indicator(self, detector):
        """Test removing network indicator."""
        detector.add_network_indicator("production.example.com", 443)
        detector.remove_network_indicator("production.example.com", 443)
        
        assert ("production.example.com", 443) not in detector._network_indicators

    def test_remove_network_indicator_not_found(self, detector):
        """Test removing non-existent network indicator."""
        with pytest.raises(ValueError, match="Network indicator not found"):
            detector.remove_network_indicator("nonexistent.example.com", 443)

    def test_add_system_indicator(self, detector):
        """Test adding system indicator."""
        detector.add_system_indicator("hostname", "prod-server")
        
        assert "hostname" in detector._system_indicators
        assert detector._system_indicators["hostname"] == "prod-server"

    def test_add_system_indicator_duplicate(self, detector):
        """Test adding duplicate system indicator."""
        detector.add_system_indicator("hostname", "prod-server")
        
        with pytest.raises(ValueError, match="System indicator already exists"):
            detector.add_system_indicator("hostname", "dev-server")

    def test_remove_system_indicator(self, detector):
        """Test removing system indicator."""
        detector.add_system_indicator("hostname", "prod-server")
        detector.remove_system_indicator("hostname")
        
        assert "hostname" not in detector._system_indicators

    def test_remove_system_indicator_not_found(self, detector):
        """Test removing non-existent system indicator."""
        with pytest.raises(ValueError, match="System indicator not found"):
            detector.remove_system_indicator("nonexistent_indicator")

    def test_add_custom_indicator(self, detector):
        """Test adding custom indicator."""
        def custom_check():
            return True
        
        detector.add_custom_indicator("custom_check", custom_check)
        
        assert "custom_check" in detector._custom_indicators
        assert detector._custom_indicators["custom_check"] == custom_check

    def test_add_custom_indicator_duplicate(self, detector):
        """Test adding duplicate custom indicator."""
        def custom_check1():
            return True
        
        def custom_check2():
            return False
        
        detector.add_custom_indicator("custom_check", custom_check1)
        
        with pytest.raises(ValueError, match="Custom indicator already exists"):
            detector.add_custom_indicator("custom_check", custom_check2)

    def test_remove_custom_indicator(self, detector):
        """Test removing custom indicator."""
        def custom_check():
            return True
        
        detector.add_custom_indicator("custom_check", custom_check)
        detector.remove_custom_indicator("custom_check")
        
        assert "custom_check" not in detector._custom_indicators

    def test_remove_custom_indicator_not_found(self, detector):
        """Test removing non-existent custom indicator."""
        with pytest.raises(ValueError, match="Custom indicator not found"):
            detector.remove_custom_indicator("nonexistent_check")

    def test_set_threshold(self, detector):
        """Test setting detection threshold."""
        detector.set_threshold(0.8)
        
        assert detector._threshold == 0.8

    def test_set_threshold_invalid(self, detector):
        """Test setting invalid detection threshold."""
        with pytest.raises(ValueError, match="Threshold must be between 0 and 1"):
            detector.set_threshold(1.5)
        
        with pytest.raises(ValueError, match="Threshold must be between 0 and 1"):
            detector.set_threshold(-0.1)

    def test_set_weight(self, detector):
        """Test setting indicator weight."""
        detector.set_weight("test_indicator", 0.8)
        
        assert detector._weights["test_indicator"] == 0.8

    def test_set_weight_invalid(self, detector):
        """Test setting invalid indicator weight."""
        with pytest.raises(ValueError, match="Weight must be between 0 and 1"):
            detector.set_weight("test_indicator", 1.5)
        
        with pytest.raises(ValueError, match="Weight must be between 0 and 1"):
            detector.set_weight("test_indicator", -0.1)

    def test_set_cache_ttl(self, detector):
        """Test setting cache TTL."""
        detector.set_cache_ttl(600)
        
        assert detector._cache_ttl == 600

    def test_set_cache_ttl_invalid(self, detector):
        """Test setting invalid cache TTL."""
        with pytest.raises(ValueError, match="Cache TTL must be positive"):
            detector.set_cache_ttl(-1)

    def test_clear_cache(self, detector):
        """Test clearing cache."""
        detector._cache["test_key"] = {"value": True, "timestamp": 1234567890}
        
        assert len(detector._cache) == 1
        
        detector.clear_cache()
        
        assert len(detector._cache) == 0

    def test_is_production_mode_no_indicators(self, detector):
        """Test production mode detection with no indicators."""
        result = detector.is_production_mode()
        
        assert result is False

    def test_is_production_mode_environment_variable(self, detector):
        """Test production mode detection with environment variable."""
        detector.add_environment_variable("PRODUCTION", "true")
        
        with patch.dict(os.environ, {"PRODUCTION": "true"}):
            result = detector.is_production_mode()
        
        assert result is True

    def test_is_production_mode_environment_variable_false(self, detector):
        """Test production mode detection with environment variable set to false."""
        detector.add_environment_variable("PRODUCTION", "true")
        
        with patch.dict(os.environ, {"PRODUCTION": "false"}):
            result = detector.is_production_mode()
        
        assert result is False

    def test_is_production_mode_file_indicator(self, detector):
        """Test production mode detection with file indicator."""
        detector.add_file_indicator("/tmp/production.flag")
        
        with patch("os.path.exists", return_value=True):
            result = detector.is_production_mode()
        
        assert result is True

    def test_is_production_mode_file_indicator_not_exists(self, detector):
        """Test production mode detection with file indicator that doesn't exist."""
        detector.add_file_indicator("/tmp/production.flag")
        
        with patch("os.path.exists", return_value=False):
            result = detector.is_production_mode()
        
        assert result is False

    def test_is_production_mode_network_indicator(self, detector):
        """Test production mode detection with network indicator."""
        detector.add_network_indicator("production.example.com", 443)
        
        with patch("socket.socket") as mock_socket:
            mock_socket.return_value.connect.return_value = None
            result = detector.is_production_mode()
        
        assert result is True

    def test_is_production_mode_network_indicator_failed(self, detector):
        """Test production mode detection with network indicator that fails."""
        detector.add_network_indicator("production.example.com", 443)
        
        with patch("socket.socket") as mock_socket:
            mock_socket.return_value.connect.side_effect = Exception("Connection failed")
            result = detector.is_production_mode()
        
        assert result is False

    def test_is_production_mode_system_indicator(self, detector):
        """Test production mode detection with system indicator."""
        detector.add_system_indicator("hostname", "prod-server")
        
        with patch("socket.gethostname", return_value="prod-server"):
            result = detector.is_production_mode()
        
        assert result is True

    def test_is_production_mode_system_indicator_mismatch(self, detector):
        """Test production mode detection with system indicator that doesn't match."""
        detector.add_system_indicator("hostname", "prod-server")
        
        with patch("socket.gethostname", return_value="dev-server"):
            result = detector.is_production_mode()
        
        assert result is False

    def test_is_production_mode_custom_indicator(self, detector):
        """Test production mode detection with custom indicator."""
        def custom_check():
            return True
        
        detector.add_custom_indicator("custom_check", custom_check)
        
        result = detector.is_production_mode()
        
        assert result is True

    def test_is_production_mode_custom_indicator_false(self, detector):
        """Test production mode detection with custom indicator that returns false."""
        def custom_check():
            return False
        
        detector.add_custom_indicator("custom_check", custom_check)
        
        result = detector.is_production_mode()
        
        assert result is False

    def test_is_production_mode_custom_indicator_exception(self, detector):
        """Test production mode detection with custom indicator that raises exception."""
        def custom_check():
            raise Exception("Custom check failed")
        
        detector.add_custom_indicator("custom_check", custom_check)
        
        result = detector.is_production_mode()
        
        assert result is False

    def test_is_production_mode_mixed_indicators(self, detector):
        """Test production mode detection with mixed indicators."""
        # Add indicators that should return True
        detector.add_environment_variable("PRODUCTION", "true")
        detector.add_file_indicator("/tmp/production.flag")
        
        # Add indicators that should return False
        detector.add_system_indicator("hostname", "prod-server")
        
        with patch.dict(os.environ, {"PRODUCTION": "true"}):
            with patch("os.path.exists", return_value=True):
                with patch("socket.gethostname", return_value="dev-server"):
                    result = detector.is_production_mode()
        
        # Should return True because 2 out of 3 indicators are True (above threshold)
        assert result is True

    def test_is_production_mode_with_weights(self, detector):
        """Test production mode detection with weighted indicators."""
        detector.add_environment_variable("PRODUCTION", "true")
        detector.add_file_indicator("/tmp/production.flag")
        
        # Set weights
        detector.set_weight("environment_variable", 0.8)
        detector.set_weight("file_indicator", 0.2)
        
        with patch.dict(os.environ, {"PRODUCTION": "true"}):
            with patch("os.path.exists", return_value=False):
                result = detector.is_production_mode()
        
        # Should return True because weighted score is above threshold
        assert result is True

    def test_is_production_mode_with_threshold(self, detector):
        """Test production mode detection with custom threshold."""
        detector.set_threshold(0.8)
        
        detector.add_environment_variable("PRODUCTION", "true")
        detector.add_file_indicator("/tmp/production.flag")
        
        with patch.dict(os.environ, {"PRODUCTION": "true"}):
            with patch("os.path.exists", return_value=False):
                result = detector.is_production_mode()
        
        # Should return False because only 1 out of 2 indicators is True (below 0.8 threshold)
        assert result is False

    def test_get_detection_score(self, detector):
        """Test getting detection score."""
        detector.add_environment_variable("PRODUCTION", "true")
        detector.add_file_indicator("/tmp/production.flag")
        
        with patch.dict(os.environ, {"PRODUCTION": "true"}):
            with patch("os.path.exists", return_value=True):
                score = detector.get_detection_score()
        
        assert score == 1.0

    def test_get_detection_score_partial(self, detector):
        """Test getting partial detection score."""
        detector.add_environment_variable("PRODUCTION", "true")
        detector.add_file_indicator("/tmp/production.flag")
        
        with patch.dict(os.environ, {"PRODUCTION": "true"}):
            with patch("os.path.exists", return_value=False):
                score = detector.get_detection_score()
        
        assert score == 0.5

    def test_get_detection_score_with_weights(self, detector):
        """Test getting detection score with weights."""
        detector.add_environment_variable("PRODUCTION", "true")
        detector.add_file_indicator("/tmp/production.flag")
        
        # Set weights
        detector.set_weight("environment_variable", 0.8)
        detector.set_weight("file_indicator", 0.2)
        
        with patch.dict(os.environ, {"PRODUCTION": "true"}):
            with patch("os.path.exists", return_value=False):
                score = detector.get_detection_score()
        
        # Weighted score: 0.8 * 1.0 + 0.2 * 0.0 = 0.8
        assert score == 0.8

    def test_get_detection_details(self, detector):
        """Test getting detection details."""
        detector.add_environment_variable("PRODUCTION", "true")
        detector.add_file_indicator("/tmp/production.flag")
        
        with patch.dict(os.environ, {"PRODUCTION": "true"}):
            with patch("os.path.exists", return_value=True):
                details = detector.get_detection_details()
        
        assert "environment_variable" in details
        assert "file_indicator" in details
        assert details["environment_variable"] is True
        assert details["file_indicator"] is True

    def test_reset_indicators(self, detector):
        """Test resetting all indicators."""
        detector.add_environment_variable("PRODUCTION", "true")
        detector.add_file_indicator("/tmp/production.flag")
        detector.add_network_indicator("production.example.com", 443)
        detector.add_system_indicator("hostname", "prod-server")
        
        def custom_check():
            return True
        
        detector.add_custom_indicator("custom_check", custom_check)
        
        assert len(detector._environment_variables) == 1
        assert len(detector._file_indicators) == 1
        assert len(detector._network_indicators) == 1
        assert len(detector._system_indicators) == 1
        assert len(detector._custom_indicators) == 1
        
        detector.reset_indicators()
        
        assert len(detector._environment_variables) == 0
        assert len(detector._file_indicators) == 0
        assert len(detector._network_indicators) == 0
        assert len(detector._system_indicators) == 0
        assert len(detector._custom_indicators) == 0

    def test_export_configuration(self, detector):
        """Test exporting configuration."""
        detector.add_environment_variable("PRODUCTION", "true")
        detector.add_file_indicator("/tmp/production.flag")
        detector.set_threshold(0.8)
        detector.set_weight("environment_variable", 0.8)
        
        config = detector.export_configuration()
        
        assert "environment_variables" in config
        assert "file_indicators" in config
        assert "threshold" in config
        assert "weights" in config
        assert config["threshold"] == 0.8
        assert config["weights"]["environment_variable"] == 0.8

    def test_import_configuration(self, detector):
        """Test importing configuration."""
        config = {
            "environment_variables": {"PRODUCTION": "true"},
            "file_indicators": ["/tmp/production.flag"],
            "threshold": 0.8,
            "weights": {"environment_variable": 0.8}
        }
        
        detector.import_configuration(config)
        
        assert "PRODUCTION" in detector._environment_variables
        assert "/tmp/production.flag" in detector._file_indicators
        assert detector._threshold == 0.8
        assert detector._weights["environment_variable"] == 0.8

    def test_import_configuration_invalid(self, detector):
        """Test importing invalid configuration."""
        invalid_config = "invalid_config"
        
        with pytest.raises(ValueError, match="Configuration must be a dictionary"):
            detector.import_configuration(invalid_config)


class TestProductionValidator:
    """Test ProductionValidator class."""

    @pytest.fixture
    def validator(self):
        """Create ProductionValidator instance."""
        return ProductionValidator()

    def test_initialization(self, validator):
        """Test ProductionValidator initialization."""
        assert validator._validation_rules == []
        assert validator._required_checks == []
        assert validator._optional_checks == []
        assert validator._custom_checks == []
        assert validator._validation_timeout == 30
        assert validator._retry_attempts == 3
        assert validator._retry_delay == 1
        assert validator._results == {}
        assert validator._last_validation is None

    def test_add_validation_rule(self, validator):
        """Test adding validation rule."""
        def rule():
            return True
        
        validator.add_validation_rule("test_rule", rule)
        
        assert "test_rule" in validator._validation_rules
        assert validator._validation_rules["test_rule"] == rule

    def test_add_validation_rule_duplicate(self, validator):
        """Test adding duplicate validation rule."""
        def rule1():
            return True
        
        def rule2():
            return False
        
        validator.add_validation_rule("test_rule", rule1)
        
        with pytest.raises(ValueError, match="Validation rule already exists"):
            validator.add_validation_rule("test_rule", rule2)

    def test_remove_validation_rule(self, validator):
        """Test removing validation rule."""
        def rule():
            return True
        
        validator.add_validation_rule("test_rule", rule)
        validator.remove_validation_rule("test_rule")
        
        assert "test_rule" not in validator._validation_rules

    def test_remove_validation_rule_not_found(self, validator):
        """Test removing non-existent validation rule."""
        with pytest.raises(ValueError, match="Validation rule not found"):
            validator.remove_validation_rule("nonexistent_rule")

    def test_add_required_check(self, validator):
        """Test adding required check."""
        def check():
            return True
        
        validator.add_required_check("required_check", check)
        
        assert "required_check" in validator._required_checks
        assert validator._required_checks["required_check"] == check

    def test_add_required_check_duplicate(self, validator):
        """Test adding duplicate required check."""
        def check1():
            return True
        
        def check2():
            return False
        
        validator.add_required_check("required_check", check1)
        
        with pytest.raises(ValueError, match="Required check already exists"):
            validator.add_required_check("required_check", check2)

    def test_remove_required_check(self, validator):
        """Test removing required check."""
        def check():
            return True
        
        validator.add_required_check("required_check", check)
        validator.remove_required_check("required_check")
        
        assert "required_check" not in validator._required_checks

    def test_remove_required_check_not_found(self, validator):
        """Test removing non-existent required check."""
        with pytest.raises(ValueError, match="Required check not found"):
            validator.remove_required_check("nonexistent_check")

    def test_add_optional_check(self, validator):
        """Test adding optional check."""
        def check():
            return True
        
        validator.add_optional_check("optional_check", check)
        
        assert "optional_check" in validator._optional_checks
        assert validator._optional_checks["optional_check"] == check

    def test_add_optional_check_duplicate(self, validator):
        """Test adding duplicate optional check."""
        def check1():
            return True
        
        def check2():
            return False
        
        validator.add_optional_check("optional_check", check1)
        
        with pytest.raises(ValueError, match="Optional check already exists"):
            validator.add_optional_check("optional_check", check2)

    def test_remove_optional_check(self, validator):
        """Test removing optional check."""
        def check():
            return True
        
        validator.add_optional_check("optional_check", check)
        validator.remove_optional_check("optional_check")
        
        assert "optional_check" not in validator._optional_checks

    def test_remove_optional_check_not_found(self, validator):
        """Test removing non-existent optional check."""
        with pytest.raises(ValueError, match="Optional check not found"):
            validator.remove_optional_check("nonexistent_check")

    def test_add_custom_check(self, validator):
        """Test adding custom check."""
        def check():
            return True
        
        validator.add_custom_check("custom_check", check)
        
        assert "custom_check" in validator._custom_checks
        assert validator._custom_checks["custom_check"] == check

    def test_add_custom_check_duplicate(self, validator):
        """Test adding duplicate custom check."""
        def check1():
            return True
        
        def check2():
            return False
        
        validator.add_custom_check("custom_check", check1)
        
        with pytest.raises(ValueError, match="Custom check already exists"):
            validator.add_custom_check("custom_check", check2)

    def test_remove_custom_check(self, validator):
        """Test removing custom check."""
        def check():
            return True
        
        validator.add_custom_check("custom_check", check)
        validator.remove_custom_check("custom_check")
        
        assert "custom_check" not in validator._custom_checks

    def test_remove_custom_check_not_found(self, validator):
        """Test removing non-existent custom check."""
        with pytest.raises(ValueError, match="Custom check not found"):
            validator.remove_custom_check("nonexistent_check")

    def test_set_validation_timeout(self, validator):
        """Test setting validation timeout."""
        validator.set_validation_timeout(60)
        
        assert validator._validation_timeout == 60

    def test_set_validation_timeout_invalid(self, validator):
        """Test setting invalid validation timeout."""
        with pytest.raises(ValueError, match="Validation timeout must be positive"):
            validator.set_validation_timeout(-1)

    def test_set_retry_attempts(self, validator):
        """Test setting retry attempts."""
        validator.set_retry_attempts(5)
        
        assert validator._retry_attempts == 5

    def test_set_retry_attempts_invalid(self, validator):
        """Test setting invalid retry attempts."""
        with pytest.raises(ValueError, match="Retry attempts must be non-negative"):
            validator.set_retry_attempts(-1)

    def test_set_retry_delay(self, validator):
        """Test setting retry delay."""
        validator.set_retry_delay(2)
        
        assert validator._retry_delay == 2

    def test_set_retry_delay_invalid(self, validator):
        """Test setting invalid retry delay."""
        with pytest.raises(ValueError, match="Retry delay must be positive"):
            validator.set_retry_delay(-1)

    def test_validate_production_mode_no_checks(self, validator):
        """Test validating production mode with no checks."""
        result = validator.validate_production_mode()
        
        assert result is True

    def test_validate_production_mode_required_checks(self, validator):
        """Test validating production mode with required checks."""
        def check1():
            return True
        
        def check2():
            return True
        
        validator.add_required_check("check1", check1)
        validator.add_required_check("check2", check2)
        
        result = validator.validate_production_mode()
        
        assert result is True

    def test_validate_production_mode_required_check_fails(self, validator):
        """Test validating production mode with failing required check."""
        def check1():
            return True
        
        def check2():
            return False
        
        validator.add_required_check("check1", check1)
        validator.add_required_check("check2", check2)
        
        result = validator.validate_production_mode()
        
        assert result is False

    def test_validate_production_mode_optional_checks(self, validator):
        """Test validating production mode with optional checks."""
        def check1():
            return True
        
        def check2():
            return False
        
        validator.add_optional_check("check1", check1)
        validator.add_optional_check("check2", check2)
        
        result = validator.validate_production_mode()
        
        # Should pass because optional checks don't block validation
        assert result is True

    def test_validate_production_mode_custom_checks(self, validator):
        """Test validating production mode with custom checks."""
        def check1():
            return True
        
        def check2():
            return True
        
        validator.add_custom_check("check1", check1)
        validator.add_custom_check("check2", check2)
        
        result = validator.validate_production_mode()
        
        assert result is True

    def test_validate_production_mode_custom_check_fails(self, validator):
        """Test validating production mode with failing custom check."""
        def check1():
            return True
        
        def check2():
            return False
        
        validator.add_custom_check("check1", check1)
        validator.add_custom_check("check2", check2)
        
        result = validator.validate_production_mode()
        
        assert result is False

    def test_validate_production_mode_mixed_checks(self, validator):
        """Test validating production mode with mixed checks."""
        def required_check():
            return True
        
        def optional_check():
            return False
        
        def custom_check():
            return True
        
        validator.add_required_check("required_check", required_check)
        validator.add_optional_check("optional_check", optional_check)
        validator.add_custom_check("custom_check", custom_check)
        
        result = validator.validate_production_mode()
        
        assert result is True

    def test_validate_production_mode_check_exception(self, validator):
        """Test validating production mode with check that raises exception."""
        def check():
            raise Exception("Check failed")
        
        validator.add_required_check("check", check)
        
        result = validator.validate_production_mode()
        
        assert result is False

    def test_validate_production_mode_with_timeout(self, validator):
        """Test validating production mode with timeout."""
        def slow_check():
            import time
            time.sleep(2)
            return True
        
        validator.add_required_check("slow_check", slow_check)
        validator.set_validation_timeout(1)
        
        result = validator.validate_production_mode()
        
        assert result is False

    def test_validate_production_mode_with_retry(self, validator):
        """Test validating production mode with retry."""
        attempt_count = 0
        
        def flaky_check():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("Check failed")
            return True
        
        validator.add_required_check("flaky_check", flaky_check)
        validator.set_retry_attempts(3)
        validator.set_retry_delay(0.1)
        
        result = validator.validate_production_mode()
        
        assert result is True
        assert attempt_count == 3

    def test_validate_production_mode_with_retry_fails(self, validator):
        """Test validating production mode with retry that fails."""
        def failing_check():
            raise Exception("Check failed")
        
        validator.add_required_check("failing_check", failing_check)
        validator.set_retry_attempts(2)
        validator.set_retry_delay(0.1)
        
        result = validator.validate_production_mode()
        
        assert result is False

    def test_get_validation_results(self, validator):
        """Test getting validation results."""
        def check1():
            return True
        
        def check2():
            return False
        
        validator.add_required_check("check1", check1)
        validator.add_required_check("check2", check2)
        
        validator.validate_production_mode()
        
        results = validator.get_validation_results()
        
        assert "check1" in results
        assert "check2" in results
        assert results["check1"] is True
        assert results["check2"] is False

    def test_get_validation_summary(self, validator):
        """Test getting validation summary."""
        def check1():
            return True
        
        def check2():
            return False
        
        def check3():
            return True
        
        validator.add_required_check("check1", check1)
        validator.add_required_check("check2", check2)
        validator.add_optional_check("check3", check3)
        
        validator.validate_production_mode()
        
        summary = validator.get_validation_summary()
        
        assert summary["total_checks"] == 3
        assert summary["passed_checks"] == 2
        assert summary["failed_checks"] == 1
        assert summary["success_rate"] == 2/3 * 100

    def test_clear_validation_results(self, validator):
        """Test clearing validation results."""
        def check():
            return True
        
        validator.add_required_check("check", check)
        validator.validate_production_mode()
        
        assert len(validator._results) == 1
        
        validator.clear_validation_results()
        
        assert len(validator._results) == 0

    def test_reset_validator(self, validator):
        """Test resetting validator."""
        def check():
            return True
        
        validator.add_required_check("check", check)
        validator.add_optional_check("optional_check", check)
        validator.add_custom_check("custom_check", check)
        validator.set_validation_timeout(60)
        validator.set_retry_attempts(5)
        validator.set_retry_delay(2)
        
        assert len(validator._required_checks) == 1
        assert len(validator._optional_checks) == 1
        assert len(validator._custom_checks) == 1
        assert validator._validation_timeout == 60
        assert validator._retry_attempts == 5
        assert validator._retry_delay == 2
        
        validator.reset_validator()
        
        assert len(validator._required_checks) == 0
        assert len(validator._optional_checks) == 0
        assert len(validator._custom_checks) == 0
        assert validator._validation_timeout == 30
        assert validator._retry_attempts == 3
        assert validator._retry_delay == 1

    def test_export_configuration(self, validator):
        """Test exporting configuration."""
        def check():
            return True
        
        validator.add_required_check("check", check)
        validator.set_validation_timeout(60)
        validator.set_retry_attempts(5)
        validator.set_retry_delay(2)
        
        config = validator.export_configuration()
        
        assert "validation_timeout" in config
        assert "retry_attempts" in config
        assert "retry_delay" in config
        assert config["validation_timeout"] == 60
        assert config["retry_attempts"] == 5
        assert config["retry_delay"] == 2

    def test_import_configuration(self, validator):
        """Test importing configuration."""
        config = {
            "validation_timeout": 60,
            "retry_attempts": 5,
            "retry_delay": 2
        }
        
        validator.import_configuration(config)
        
        assert validator._validation_timeout == 60
        assert validator._retry_attempts == 5
        assert validator._retry_delay == 2

    def test_import_configuration_invalid(self, validator):
        """Test importing invalid configuration."""
        invalid_config = "invalid_config"
        
        with pytest.raises(ValueError, match="Configuration must be a dictionary"):
            validator.import_configuration(invalid_config)


class TestProductionModeIntegration:
    """Integration tests for production mode detection and validation."""

    @pytest.mark.asyncio
    async def test_full_production_mode_workflow(self):
        """Test complete production mode detection and validation workflow."""
        detector = ProductionModeDetector()
        validator = ProductionValidator()
        
        # Configure detector
        detector.add_environment_variable("PRODUCTION", "true")
        detector.add_file_indicator("/tmp/production.flag")
        detector.set_threshold(0.5)
        
        # Configure validator
        def production_check():
            return True
        
        def security_check():
            return True
        
        validator.add_required_check("production_check", production_check)
        validator.add_required_check("security_check", security_check)
        
        # Test detection
        with patch.dict(os.environ, {"PRODUCTION": "true"}):
            with patch("os.path.exists", return_value=True):
                is_production = detector.is_production_mode()
        
        assert is_production is True
        
        # Test validation
        is_valid = validator.validate_production_mode()
        
        assert is_valid is True
        
        # Get results
        detection_score = detector.get_detection_score()
        validation_results = validator.get_validation_results()
        
        assert detection_score == 1.0
        assert validation_results["production_check"] is True
        assert validation_results["security_check"] is True

    @pytest.mark.asyncio
    async def test_production_mode_failure_scenarios(self):
        """Test production mode detection and validation failure scenarios."""
        detector = ProductionModeDetector()
        validator = ProductionValidator()
        
        # Configure detector with indicators that will fail
        detector.add_environment_variable("PRODUCTION", "true")
        detector.add_file_indicator("/tmp/production.flag")
        detector.set_threshold(0.8)  # High threshold
        
        # Configure validator with checks that will fail
        def production_check():
            return False
        
        def security_check():
            return True
        
        validator.add_required_check("production_check", production_check)
        validator.add_required_check("security_check", security_check)
        
        # Test detection failure
        with patch.dict(os.environ, {"PRODUCTION": "false"}):
            with patch("os.path.exists", return_value=False):
                is_production = detector.is_production_mode()
        
        assert is_production is False
        
        # Test validation failure
        is_valid = validator.validate_production_mode()
        
        assert is_valid is False
        
        # Get results
        detection_score = detector.get_detection_score()
        validation_results = validator.get_validation_results()
        
        assert detection_score == 0.0
        assert validation_results["production_check"] is False
        assert validation_results["security_check"] is True

    @pytest.mark.asyncio
    async def test_production_mode_configuration_management(self):
        """Test production mode configuration management."""
        detector = ProductionModeDetector()
        validator = ProductionValidator()
        
        # Configure detector
        detector.add_environment_variable("PRODUCTION", "true")
        detector.add_file_indicator("/tmp/production.flag")
        detector.set_threshold(0.5)
        
        # Configure validator
        def check():
            return True
        
        validator.add_required_check("check", check)
        validator.set_validation_timeout(60)
        
        # Export configurations
        detector_config = detector.export_configuration()
        validator_config = validator.export_configuration()
        
        # Create new instances
        new_detector = ProductionModeDetector()
        new_validator = ProductionValidator()
        
        # Import configurations
        new_detector.import_configuration(detector_config)
        new_validator.import_configuration(validator_config)
        
        # Verify configurations
        assert new_detector._threshold == 0.5
        assert "PRODUCTION" in new_detector._environment_variables
        assert "/tmp/production.flag" in new_detector._file_indicators
        
        assert new_validator._validation_timeout == 60
        assert "check" in new_validator._required_checks

    @pytest.mark.asyncio
    async def test_production_mode_monitoring(self):
        """Test production mode monitoring and alerting."""
        detector = ProductionModeDetector()
        validator = ProductionValidator()
        
        # Configure detector
        detector.add_environment_variable("PRODUCTION", "true")
        detector.set_threshold(0.5)
        
        # Configure validator
        def check():
            return True
        
        validator.add_required_check("check", check)
        
        # Monitor production mode
        with patch.dict(os.environ, {"PRODUCTION": "true"}):
            is_production = detector.is_production_mode()
            is_valid = validator.validate_production_mode()
        
        assert is_production is True
        assert is_valid is True
        
        # Get monitoring data
        detection_score = detector.get_detection_score()
        detection_details = detector.get_detection_details()
        validation_summary = validator.get_validation_summary()
        
        assert detection_score == 1.0
        assert detection_details["environment_variable"] is True
        assert validation_summary["total_checks"] == 1
        assert validation_summary["passed_checks"] == 1
        assert validation_summary["success_rate"] == 100.0
