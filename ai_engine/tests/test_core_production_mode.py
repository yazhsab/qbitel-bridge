"""
Comprehensive tests for ai_engine.core.production_mode module.

This module provides production mode detection, configuration, and
environment-specific behavior for the CRONOS AI Engine.
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import yaml
import json

from ai_engine.core.production_mode import (
    ProductionModeDetector,
    ProductionValidator,
    ProductionModeError,
    ProductionModeWarning,
    ProductionModeValidator,
    ProductionModeMonitor,
    ProductionModeHealthCheck,
    ProductionModeMetrics,
    ProductionModeLogger,
    ProductionModeSecurity,
    ProductionModePerformance,
    ProductionModeCompliance,
)


class TestEnvironmentType:
    """Test EnvironmentType enum."""

    def test_environment_type_values(self):
        """Test EnvironmentType enum values."""
        assert EnvironmentType.DEVELOPMENT.value == "development"
        assert EnvironmentType.STAGING.value == "staging"
        assert EnvironmentType.PRODUCTION.value == "production"
        assert EnvironmentType.TESTING.value == "testing"

    def test_environment_type_from_string(self):
        """Test creating EnvironmentType from string."""
        assert EnvironmentType.from_string("development") == EnvironmentType.DEVELOPMENT
        assert EnvironmentType.from_string("staging") == EnvironmentType.STAGING
        assert EnvironmentType.from_string("production") == EnvironmentType.PRODUCTION
        assert EnvironmentType.from_string("testing") == EnvironmentType.TESTING

    def test_environment_type_from_string_invalid(self):
        """Test creating EnvironmentType from invalid string."""
        with pytest.raises(ValueError, match="Invalid environment type"):
            EnvironmentType.from_string("invalid")

    def test_environment_type_is_production(self):
        """Test checking if environment type is production."""
        assert EnvironmentType.PRODUCTION.is_production() is True
        assert EnvironmentType.DEVELOPMENT.is_production() is False
        assert EnvironmentType.STAGING.is_production() is False
        assert EnvironmentType.TESTING.is_production() is False

    def test_environment_type_is_development(self):
        """Test checking if environment type is development."""
        assert EnvironmentType.DEVELOPMENT.is_development() is True
        assert EnvironmentType.PRODUCTION.is_development() is False
        assert EnvironmentType.STAGING.is_development() is False
        assert EnvironmentType.TESTING.is_development() is False


class TestProductionModeConfig:
    """Test ProductionModeConfig class."""

    def test_initialization(self):
        """Test ProductionModeConfig initialization."""
        config = ProductionModeConfig(
            environment=EnvironmentType.PRODUCTION,
            debug=False,
            log_level="INFO",
            enable_metrics=True,
            enable_tracing=True,
            security_level="high",
            performance_mode="optimized"
        )
        
        assert config.environment == EnvironmentType.PRODUCTION
        assert config.debug is False
        assert config.log_level == "INFO"
        assert config.enable_metrics is True
        assert config.enable_tracing is True
        assert config.security_level == "high"
        assert config.performance_mode == "optimized"

    def test_initialization_with_defaults(self):
        """Test ProductionModeConfig initialization with defaults."""
        config = ProductionModeConfig()
        
        assert config.environment == EnvironmentType.DEVELOPMENT
        assert config.debug is True
        assert config.log_level == "DEBUG"
        assert config.enable_metrics is False
        assert config.enable_tracing is False
        assert config.security_level == "low"
        assert config.performance_mode == "development"

    def test_validation(self):
        """Test ProductionModeConfig validation."""
        # Valid config
        config = ProductionModeConfig(
            environment=EnvironmentType.PRODUCTION,
            debug=False,
            log_level="INFO"
        )
        assert config.validate() is True
        
        # Invalid config - debug enabled in production
        config = ProductionModeConfig(
            environment=EnvironmentType.PRODUCTION,
            debug=True
        )
        with pytest.raises(ProductionModeError, match="Debug mode cannot be enabled in production"):
            config.validate()

    def test_to_dict(self):
        """Test converting ProductionModeConfig to dictionary."""
        config = ProductionModeConfig(
            environment=EnvironmentType.PRODUCTION,
            debug=False,
            log_level="INFO",
            enable_metrics=True
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["environment"] == "production"
        assert config_dict["debug"] is False
        assert config_dict["log_level"] == "INFO"
        assert config_dict["enable_metrics"] is True

    def test_from_dict(self):
        """Test creating ProductionModeConfig from dictionary."""
        config_dict = {
            "environment": "production",
            "debug": False,
            "log_level": "INFO",
            "enable_metrics": True
        }
        
        config = ProductionModeConfig.from_dict(config_dict)
        
        assert config.environment == EnvironmentType.PRODUCTION
        assert config.debug is False
        assert config.log_level == "INFO"
        assert config.enable_metrics is True

    def test_merge(self):
        """Test merging ProductionModeConfig with another config."""
        config1 = ProductionModeConfig(
            environment=EnvironmentType.PRODUCTION,
            debug=False,
            log_level="INFO"
        )
        
        config2 = ProductionModeConfig(
            enable_metrics=True,
            enable_tracing=True
        )
        
        merged = config1.merge(config2)
        
        assert merged.environment == EnvironmentType.PRODUCTION
        assert merged.debug is False
        assert merged.log_level == "INFO"
        assert merged.enable_metrics is True
        assert merged.enable_tracing is True


class TestProductionModeDetector:
    """Test ProductionModeDetector class."""

    @pytest.fixture
    def detector(self):
        """Create ProductionModeDetector instance."""
        return ProductionModeDetector()

    def test_initialization(self, detector):
        """Test ProductionModeDetector initialization."""
        assert detector._environment is None
        assert detector._config is None
        assert detector._detected is False

    def test_detect_from_environment_variable(self, detector):
        """Test detecting production mode from environment variable."""
        with patch.dict(os.environ, {"CRONOS_ENVIRONMENT": "production"}):
            result = detector.detect()
            
            assert result.environment == EnvironmentType.PRODUCTION
            assert result.debug is False
            assert result.log_level == "INFO"

    def test_detect_from_environment_variable_development(self, detector):
        """Test detecting development mode from environment variable."""
        with patch.dict(os.environ, {"CRONOS_ENVIRONMENT": "development"}):
            result = detector.detect()
            
            assert result.environment == EnvironmentType.DEVELOPMENT
            assert result.debug is True
            assert result.log_level == "DEBUG"

    def test_detect_from_environment_variable_staging(self, detector):
        """Test detecting staging mode from environment variable."""
        with patch.dict(os.environ, {"CRONOS_ENVIRONMENT": "staging"}):
            result = detector.detect()
            
            assert result.environment == EnvironmentType.STAGING
            assert result.debug is False
            assert result.log_level == "INFO"

    def test_detect_from_environment_variable_testing(self, detector):
        """Test detecting testing mode from environment variable."""
        with patch.dict(os.environ, {"CRONOS_ENVIRONMENT": "testing"}):
            result = detector.detect()
            
            assert result.environment == EnvironmentType.TESTING
            assert result.debug is True
            assert result.log_level == "DEBUG"

    def test_detect_from_environment_variable_invalid(self, detector):
        """Test detecting invalid environment variable."""
        with patch.dict(os.environ, {"CRONOS_ENVIRONMENT": "invalid"}):
            with pytest.raises(ProductionModeError, match="Invalid environment type"):
                detector.detect()

    def test_detect_from_config_file(self, detector):
        """Test detecting production mode from config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                "environment": "production",
                "debug": False,
                "log_level": "INFO",
                "enable_metrics": True
            }
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            result = detector.detect_from_config_file(config_file)
            
            assert result.environment == EnvironmentType.PRODUCTION
            assert result.debug is False
            assert result.log_level == "INFO"
            assert result.enable_metrics is True
        finally:
            os.unlink(config_file)

    def test_detect_from_config_file_not_found(self, detector):
        """Test detecting from non-existent config file."""
        with pytest.raises(ProductionModeError, match="Config file not found"):
            detector.detect_from_config_file("nonexistent.yaml")

    def test_detect_from_config_file_invalid(self, detector):
        """Test detecting from invalid config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_file = f.name

        try:
            with pytest.raises(ProductionModeError, match="Invalid config file"):
                detector.detect_from_config_file(config_file)
        finally:
            os.unlink(config_file)

    def test_detect_from_system_properties(self, detector):
        """Test detecting production mode from system properties."""
        with patch('ai_engine.core.production_mode.sys.argv', ['script.py', '--environment=production']):
            result = detector.detect_from_system_properties()
            
            assert result.environment == EnvironmentType.PRODUCTION

    def test_detect_from_system_properties_default(self, detector):
        """Test detecting default mode from system properties."""
        with patch('ai_engine.core.production_mode.sys.argv', ['script.py']):
            result = detector.detect_from_system_properties()
            
            assert result.environment == EnvironmentType.DEVELOPMENT

    def test_detect_from_heuristics(self, detector):
        """Test detecting production mode from heuristics."""
        # Mock system properties to indicate production
        with patch('ai_engine.core.production_mode.platform.system', return_value='Linux'):
            with patch('ai_engine.core.production_mode.os.getenv', return_value=None):
                with patch('ai_engine.core.production_mode.sys.argv', ['/usr/bin/cronos-ai']):
                    result = detector.detect_from_heuristics()
                    
                    # Should detect as production based on heuristics
                    assert result.environment == EnvironmentType.PRODUCTION

    def test_detect_from_heuristics_development(self, detector):
        """Test detecting development mode from heuristics."""
        # Mock system properties to indicate development
        with patch('ai_engine.core.production_mode.platform.system', return_value='Darwin'):
            with patch('ai_engine.core.production_mode.os.getenv', return_value=None):
                with patch('ai_engine.core.production_mode.sys.argv', ['python', 'main.py']):
                    result = detector.detect_from_heuristics()
                    
                    # Should detect as development based on heuristics
                    assert result.environment == EnvironmentType.DEVELOPMENT

    def test_detect_auto(self, detector):
        """Test automatic detection with multiple sources."""
        with patch.dict(os.environ, {"CRONOS_ENVIRONMENT": "production"}):
            result = detector.detect_auto()
            
            assert result.environment == EnvironmentType.PRODUCTION

    def test_detect_auto_fallback(self, detector):
        """Test automatic detection with fallback."""
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(detector, 'detect_from_heuristics') as mock_heuristics:
                mock_heuristics.return_value = ProductionModeConfig(environment=EnvironmentType.DEVELOPMENT)
                
                result = detector.detect_auto()
                
                assert result.environment == EnvironmentType.DEVELOPMENT
                mock_heuristics.assert_called_once()

    def test_get_detection_info(self, detector):
        """Test getting detection information."""
        with patch.dict(os.environ, {"CRONOS_ENVIRONMENT": "production"}):
            detector.detect()
            
            info = detector.get_detection_info()
            
            assert info["detected"] is True
            assert info["environment"] == "production"
            assert info["source"] == "environment_variable"
            assert "timestamp" in info


class TestProductionModeManager:
    """Test ProductionModeManager class."""

    @pytest.fixture
    def manager(self):
        """Create ProductionModeManager instance."""
        return ProductionModeManager()

    def test_initialization(self, manager):
        """Test ProductionModeManager initialization."""
        assert manager._config is None
        assert manager._initialized is False
        assert manager._monitor is not None
        assert manager._validator is not None

    def test_initialize(self, manager):
        """Test ProductionModeManager initialization."""
        config = ProductionModeConfig(environment=EnvironmentType.PRODUCTION)
        
        manager.initialize(config)
        
        assert manager._config == config
        assert manager._initialized is True

    def test_initialize_with_detection(self, manager):
        """Test ProductionModeManager initialization with auto-detection."""
        with patch.dict(os.environ, {"CRONOS_ENVIRONMENT": "production"}):
            manager.initialize_with_detection()
            
            assert manager._config is not None
            assert manager._config.environment == EnvironmentType.PRODUCTION
            assert manager._initialized is True

    def test_get_config(self, manager):
        """Test getting production mode config."""
        config = ProductionModeConfig(environment=EnvironmentType.PRODUCTION)
        manager.initialize(config)
        
        retrieved_config = manager.get_config()
        
        assert retrieved_config == config

    def test_get_config_not_initialized(self, manager):
        """Test getting config when not initialized."""
        with pytest.raises(ProductionModeError, match="Production mode manager not initialized"):
            manager.get_config()

    def test_is_production(self, manager):
        """Test checking if in production mode."""
        config = ProductionModeConfig(environment=EnvironmentType.PRODUCTION)
        manager.initialize(config)
        
        assert manager.is_production() is True
        
        config = ProductionModeConfig(environment=EnvironmentType.DEVELOPMENT)
        manager.initialize(config)
        
        assert manager.is_production() is False

    def test_is_development(self, manager):
        """Test checking if in development mode."""
        config = ProductionModeConfig(environment=EnvironmentType.DEVELOPMENT)
        manager.initialize(config)
        
        assert manager.is_development() is True
        
        config = ProductionModeConfig(environment=EnvironmentType.PRODUCTION)
        manager.initialize(config)
        
        assert manager.is_development() is False

    def test_is_staging(self, manager):
        """Test checking if in staging mode."""
        config = ProductionModeConfig(environment=EnvironmentType.STAGING)
        manager.initialize(config)
        
        assert manager.is_staging() is True
        
        config = ProductionModeConfig(environment=EnvironmentType.PRODUCTION)
        manager.initialize(config)
        
        assert manager.is_staging() is False

    def test_is_testing(self, manager):
        """Test checking if in testing mode."""
        config = ProductionModeConfig(environment=EnvironmentType.TESTING)
        manager.initialize(config)
        
        assert manager.is_testing() is True
        
        config = ProductionModeConfig(environment=EnvironmentType.PRODUCTION)
        manager.initialize(config)
        
        assert manager.is_testing() is False

    def test_get_environment(self, manager):
        """Test getting current environment."""
        config = ProductionModeConfig(environment=EnvironmentType.PRODUCTION)
        manager.initialize(config)
        
        environment = manager.get_environment()
        
        assert environment == EnvironmentType.PRODUCTION

    def test_get_environment_not_initialized(self, manager):
        """Test getting environment when not initialized."""
        with pytest.raises(ProductionModeError, match="Production mode manager not initialized"):
            manager.get_environment()

    def test_validate_config(self, manager):
        """Test validating production mode config."""
        config = ProductionModeConfig(environment=EnvironmentType.PRODUCTION, debug=False)
        manager.initialize(config)
        
        # Should not raise exception
        manager.validate_config()

    def test_validate_config_invalid(self, manager):
        """Test validating invalid production mode config."""
        config = ProductionModeConfig(environment=EnvironmentType.PRODUCTION, debug=True)
        manager.initialize(config)
        
        with pytest.raises(ProductionModeError, match="Invalid production mode configuration"):
            manager.validate_config()

    def test_get_health_status(self, manager):
        """Test getting health status."""
        config = ProductionModeConfig(environment=EnvironmentType.PRODUCTION)
        manager.initialize(config)
        
        health = manager.get_health_status()
        
        assert health["status"] == "healthy"
        assert health["environment"] == "production"
        assert health["initialized"] is True

    def test_get_metrics(self, manager):
        """Test getting production mode metrics."""
        config = ProductionModeConfig(environment=EnvironmentType.PRODUCTION)
        manager.initialize(config)
        
        metrics = manager.get_metrics()
        
        assert metrics["environment"] == "production"
        assert metrics["initialized"] is True
        assert "uptime" in metrics

    def test_reset(self, manager):
        """Test resetting production mode manager."""
        config = ProductionModeConfig(environment=EnvironmentType.PRODUCTION)
        manager.initialize(config)
        
        manager.reset()
        
        assert manager._config is None
        assert manager._initialized is False


class TestProductionModeValidator:
    """Test ProductionModeValidator class."""

    @pytest.fixture
    def validator(self):
        """Create ProductionModeValidator instance."""
        return ProductionModeValidator()

    def test_initialization(self, validator):
        """Test ProductionModeValidator initialization."""
        assert validator._rules == {}

    def test_validate_config(self, validator):
        """Test validating production mode config."""
        config = ProductionModeConfig(
            environment=EnvironmentType.PRODUCTION,
            debug=False,
            log_level="INFO",
            enable_metrics=True
        )
        
        # Should not raise exception
        validator.validate_config(config)

    def test_validate_config_debug_in_production(self, validator):
        """Test validating config with debug enabled in production."""
        config = ProductionModeConfig(
            environment=EnvironmentType.PRODUCTION,
            debug=True
        )
        
        with pytest.raises(ProductionModeError, match="Debug mode cannot be enabled in production"):
            validator.validate_config(config)

    def test_validate_config_debug_log_level_in_production(self, validator):
        """Test validating config with debug log level in production."""
        config = ProductionModeConfig(
            environment=EnvironmentType.PRODUCTION,
            debug=False,
            log_level="DEBUG"
        )
        
        with pytest.raises(ProductionModeError, match="Debug log level cannot be used in production"):
            validator.validate_config(config)

    def test_validate_config_metrics_disabled_in_production(self, validator):
        """Test validating config with metrics disabled in production."""
        config = ProductionModeConfig(
            environment=EnvironmentType.PRODUCTION,
            debug=False,
            enable_metrics=False
        )
        
        with pytest.raises(ProductionModeError, match="Metrics must be enabled in production"):
            validator.validate_config(config)

    def test_validate_config_tracing_disabled_in_production(self, validator):
        """Test validating config with tracing disabled in production."""
        config = ProductionModeConfig(
            environment=EnvironmentType.PRODUCTION,
            debug=False,
            enable_metrics=True,
            enable_tracing=False
        )
        
        with pytest.raises(ProductionModeError, match="Tracing must be enabled in production"):
            validator.validate_config(config)

    def test_validate_config_low_security_in_production(self, validator):
        """Test validating config with low security in production."""
        config = ProductionModeConfig(
            environment=EnvironmentType.PRODUCTION,
            debug=False,
            enable_metrics=True,
            enable_tracing=True,
            security_level="low"
        )
        
        with pytest.raises(ProductionModeError, match="Security level must be high in production"):
            validator.validate_config(config)

    def test_validate_config_development_mode_in_production(self, validator):
        """Test validating config with development performance mode in production."""
        config = ProductionModeConfig(
            environment=EnvironmentType.PRODUCTION,
            debug=False,
            enable_metrics=True,
            enable_tracing=True,
            security_level="high",
            performance_mode="development"
        )
        
        with pytest.raises(ProductionModeError, match="Performance mode must be optimized in production"):
            validator.validate_config(config)

    def test_add_validation_rule(self, validator):
        """Test adding custom validation rule."""
        def custom_rule(config):
            if config.log_level == "TRACE":
                raise ProductionModeError("TRACE log level not allowed")
        
        validator.add_validation_rule("custom_rule", custom_rule)
        
        config = ProductionModeConfig(log_level="TRACE")
        
        with pytest.raises(ProductionModeError, match="TRACE log level not allowed"):
            validator.validate_config(config)

    def test_remove_validation_rule(self, validator):
        """Test removing validation rule."""
        def custom_rule(config):
            if config.log_level == "TRACE":
                raise ProductionModeError("TRACE log level not allowed")
        
        validator.add_validation_rule("custom_rule", custom_rule)
        validator.remove_validation_rule("custom_rule")
        
        config = ProductionModeConfig(log_level="TRACE")
        
        # Should not raise exception now
        validator.validate_config(config)


class TestProductionModeMonitor:
    """Test ProductionModeMonitor class."""

    @pytest.fixture
    def monitor(self):
        """Create ProductionModeMonitor instance."""
        return ProductionModeMonitor()

    def test_initialization(self, monitor):
        """Test ProductionModeMonitor initialization."""
        assert monitor._metrics == {}
        assert monitor._events == []
        assert monitor._start_time is not None

    def test_record_event(self, monitor):
        """Test recording production mode event."""
        event = {
            "type": "config_change",
            "environment": "production",
            "timestamp": "2023-01-01T00:00:00Z",
            "details": {"debug": False}
        }
        
        monitor.record_event(event)
        
        assert len(monitor._events) == 1
        assert monitor._events[0] == event

    def test_record_metric(self, monitor):
        """Test recording production mode metric."""
        monitor.record_metric("config_validation", 1.0)
        
        assert "config_validation" in monitor._metrics
        assert monitor._metrics["config_validation"] == 1.0

    def test_record_metric_increment(self, monitor):
        """Test incrementing production mode metric."""
        monitor.record_metric("config_validation", 1.0)
        monitor.record_metric("config_validation", 1.0)
        
        assert monitor._metrics["config_validation"] == 2.0

    def test_get_metrics(self, monitor):
        """Test getting production mode metrics."""
        monitor.record_metric("config_validation", 1.0)
        monitor.record_metric("error_count", 5.0)
        
        metrics = monitor.get_metrics()
        
        assert metrics["config_validation"] == 1.0
        assert metrics["error_count"] == 5.0
        assert "uptime" in metrics

    def test_get_events(self, monitor):
        """Test getting production mode events."""
        event1 = {"type": "config_change", "environment": "production"}
        event2 = {"type": "validation_error", "environment": "production"}
        
        monitor.record_event(event1)
        monitor.record_event(event2)
        
        events = monitor.get_events()
        
        assert len(events) == 2
        assert event1 in events
        assert event2 in events

    def test_get_events_by_type(self, monitor):
        """Test getting production mode events by type."""
        event1 = {"type": "config_change", "environment": "production"}
        event2 = {"type": "validation_error", "environment": "production"}
        event3 = {"type": "config_change", "environment": "development"}
        
        monitor.record_event(event1)
        monitor.record_event(event2)
        monitor.record_event(event3)
        
        config_events = monitor.get_events_by_type("config_change")
        
        assert len(config_events) == 2
        assert event1 in config_events
        assert event3 in config_events

    def test_clear_metrics(self, monitor):
        """Test clearing production mode metrics."""
        monitor.record_metric("config_validation", 1.0)
        
        monitor.clear_metrics()
        
        assert len(monitor._metrics) == 0

    def test_clear_events(self, monitor):
        """Test clearing production mode events."""
        monitor.record_event({"type": "config_change", "environment": "production"})
        
        monitor.clear_events()
        
        assert len(monitor._events) == 0

    def test_get_uptime(self, monitor):
        """Test getting production mode uptime."""
        uptime = monitor.get_uptime()
        
        assert uptime > 0
        assert isinstance(uptime, float)


class TestProductionModeHealthCheck:
    """Test ProductionModeHealthCheck class."""

    @pytest.fixture
    def health_check(self):
        """Create ProductionModeHealthCheck instance."""
        return ProductionModeHealthCheck()

    def test_initialization(self, health_check):
        """Test ProductionModeHealthCheck initialization."""
        assert health_check._checks == {}

    def test_register_check(self, health_check):
        """Test registering health check."""
        def check_function():
            return {"status": "healthy", "message": "OK"}
        
        health_check.register_check("config_validation", check_function)
        
        assert "config_validation" in health_check._checks

    def test_unregister_check(self, health_check):
        """Test unregistering health check."""
        def check_function():
            return {"status": "healthy", "message": "OK"}
        
        health_check.register_check("config_validation", check_function)
        health_check.unregister_check("config_validation")
        
        assert "config_validation" not in health_check._checks

    def test_check_health(self, health_check):
        """Test checking health."""
        def check_function():
            return {"status": "healthy", "message": "OK"}
        
        health_check.register_check("config_validation", check_function)
        
        result = health_check.check_health("config_validation")
        
        assert result["status"] == "healthy"
        assert result["message"] == "OK"

    def test_check_health_not_found(self, health_check):
        """Test checking health of non-existent check."""
        result = health_check.check_health("nonexistent_check")
        
        assert result["status"] == "unknown"
        assert "not found" in result["message"]

    def test_check_health_exception(self, health_check):
        """Test checking health with exception."""
        def check_function():
            raise Exception("Health check failed")
        
        health_check.register_check("config_validation", check_function)
        
        result = health_check.check_health("config_validation")
        
        assert result["status"] == "unhealthy"
        assert "Health check failed" in result["message"]

    def test_check_all_health(self, health_check):
        """Test checking all health checks."""
        def check_function1():
            return {"status": "healthy", "message": "OK"}
        
        def check_function2():
            return {"status": "unhealthy", "message": "Error"}
        
        health_check.register_check("check1", check_function1)
        health_check.register_check("check2", check_function2)
        
        result = health_check.check_all_health()
        
        assert len(result) == 2
        assert result["check1"]["status"] == "healthy"
        assert result["check2"]["status"] == "unhealthy"

    def test_get_overall_health(self, health_check):
        """Test getting overall health status."""
        def check_function1():
            return {"status": "healthy", "message": "OK"}
        
        def check_function2():
            return {"status": "unhealthy", "message": "Error"}
        
        health_check.register_check("check1", check_function1)
        health_check.register_check("check2", check_function2)
        
        result = health_check.get_overall_health()
        
        assert result["status"] == "unhealthy"  # Any unhealthy check makes overall unhealthy
        assert result["total_checks"] == 2
        assert result["healthy_checks"] == 1
        assert result["unhealthy_checks"] == 1


class TestProductionModeMetrics:
    """Test ProductionModeMetrics class."""

    @pytest.fixture
    def metrics(self):
        """Create ProductionModeMetrics instance."""
        return ProductionModeMetrics()

    def test_initialization(self, metrics):
        """Test ProductionModeMetrics initialization."""
        assert metrics._counters == {}
        assert metrics._gauges == {}
        assert metrics._histograms == {}
        assert metrics._start_time is not None

    def test_increment_counter(self, metrics):
        """Test incrementing counter metric."""
        metrics.increment_counter("config_validation_count")
        
        assert metrics._counters["config_validation_count"] == 1
        
        metrics.increment_counter("config_validation_count", 5)
        
        assert metrics._counters["config_validation_count"] == 6

    def test_set_gauge(self, metrics):
        """Test setting gauge metric."""
        metrics.set_gauge("active_connections", 10)
        
        assert metrics._gauges["active_connections"] == 10

    def test_record_histogram(self, metrics):
        """Test recording histogram metric."""
        metrics.record_histogram("response_time", 0.5)
        metrics.record_histogram("response_time", 1.0)
        metrics.record_histogram("response_time", 1.5)
        
        assert "response_time" in metrics._histograms
        assert len(metrics._histograms["response_time"]) == 3

    def test_get_counter(self, metrics):
        """Test getting counter metric."""
        metrics.increment_counter("config_validation_count", 5)
        
        value = metrics.get_counter("config_validation_count")
        
        assert value == 5

    def test_get_gauge(self, metrics):
        """Test getting gauge metric."""
        metrics.set_gauge("active_connections", 10)
        
        value = metrics.get_gauge("active_connections")
        
        assert value == 10

    def test_get_histogram_stats(self, metrics):
        """Test getting histogram statistics."""
        metrics.record_histogram("response_time", 0.5)
        metrics.record_histogram("response_time", 1.0)
        metrics.record_histogram("response_time", 1.5)
        
        stats = metrics.get_histogram_stats("response_time")
        
        assert stats["count"] == 3
        assert stats["min"] == 0.5
        assert stats["max"] == 1.5
        assert stats["mean"] == 1.0

    def test_get_all_metrics(self, metrics):
        """Test getting all metrics."""
        metrics.increment_counter("config_validation_count", 5)
        metrics.set_gauge("active_connections", 10)
        metrics.record_histogram("response_time", 0.5)
        
        all_metrics = metrics.get_all_metrics()
        
        assert "counters" in all_metrics
        assert "gauges" in all_metrics
        assert "histograms" in all_metrics
        assert all_metrics["counters"]["config_validation_count"] == 5
        assert all_metrics["gauges"]["active_connections"] == 10

    def test_reset_metrics(self, metrics):
        """Test resetting metrics."""
        metrics.increment_counter("config_validation_count", 5)
        metrics.set_gauge("active_connections", 10)
        metrics.record_histogram("response_time", 0.5)
        
        metrics.reset_metrics()
        
        assert len(metrics._counters) == 0
        assert len(metrics._gauges) == 0
        assert len(metrics._histograms) == 0

    def test_get_uptime(self, metrics):
        """Test getting metrics uptime."""
        uptime = metrics.get_uptime()
        
        assert uptime > 0
        assert isinstance(uptime, float)


class TestProductionModeLogger:
    """Test ProductionModeLogger class."""

    @pytest.fixture
    def logger(self):
        """Create ProductionModeLogger instance."""
        return ProductionModeLogger()

    def test_initialization(self, logger):
        """Test ProductionModeLogger initialization."""
        assert logger._logger is not None
        assert logger._log_formatter is not None

    def test_log_info(self, logger):
        """Test logging info message."""
        with patch.object(logger._logger, 'info') as mock_log:
            logger.log_info("Test info message")
            
            mock_log.assert_called_once()
            call_args = mock_log.call_args[0][0]
            assert "Test info message" in call_args

    def test_log_warning(self, logger):
        """Test logging warning message."""
        with patch.object(logger._logger, 'warning') as mock_log:
            logger.log_warning("Test warning message")
            
            mock_log.assert_called_once()
            call_args = mock_log.call_args[0][0]
            assert "Test warning message" in call_args

    def test_log_error(self, logger):
        """Test logging error message."""
        with patch.object(logger._logger, 'error') as mock_log:
            logger.log_error("Test error message")
            
            mock_log.assert_called_once()
            call_args = mock_log.call_args[0][0]
            assert "Test error message" in call_args

    def test_log_debug(self, logger):
        """Test logging debug message."""
        with patch.object(logger._logger, 'debug') as mock_log:
            logger.log_debug("Test debug message")
            
            mock_log.assert_called_once()
            call_args = mock_log.call_args[0][0]
            assert "Test debug message" in call_args

    def test_log_structured(self, logger):
        """Test structured logging."""
        with patch.object(logger._logger, 'info') as mock_log:
            logger.log_structured("info", "Test message", {"key": "value"})
            
            mock_log.assert_called_once()
            call_args = mock_log.call_args[0][0]
            assert isinstance(call_args, dict)
            assert call_args["message"] == "Test message"
            assert call_args["key"] == "value"

    def test_set_log_level(self, logger):
        """Test setting log level."""
        logger.set_log_level("DEBUG")
        
        assert logger._logger.level == 10  # DEBUG level

    def test_get_log_level(self, logger):
        """Test getting log level."""
        logger.set_log_level("INFO")
        
        level = logger.get_log_level()
        
        assert level == "INFO"


class TestProductionModeSecurity:
    """Test ProductionModeSecurity class."""

    @pytest.fixture
    def security(self):
        """Create ProductionModeSecurity instance."""
        return ProductionModeSecurity()

    def test_initialization(self, security):
        """Test ProductionModeSecurity initialization."""
        assert security._security_level == "low"
        assert security._enabled_features == set()

    def test_set_security_level(self, security):
        """Test setting security level."""
        security.set_security_level("high")
        
        assert security._security_level == "high"

    def test_get_security_level(self, security):
        """Test getting security level."""
        security.set_security_level("medium")
        
        level = security.get_security_level()
        
        assert level == "medium"

    def test_enable_feature(self, security):
        """Test enabling security feature."""
        security.enable_feature("encryption")
        
        assert "encryption" in security._enabled_features

    def test_disable_feature(self, security):
        """Test disabling security feature."""
        security.enable_feature("encryption")
        security.disable_feature("encryption")
        
        assert "encryption" not in security._enabled_features

    def test_is_feature_enabled(self, security):
        """Test checking if security feature is enabled."""
        security.enable_feature("encryption")
        
        assert security.is_feature_enabled("encryption") is True
        assert security.is_feature_enabled("authentication") is False

    def test_validate_security_config(self, security):
        """Test validating security configuration."""
        security.set_security_level("high")
        security.enable_feature("encryption")
        security.enable_feature("authentication")
        
        # Should not raise exception
        security.validate_security_config()

    def test_validate_security_config_low_level(self, security):
        """Test validating security configuration with low level."""
        security.set_security_level("low")
        
        with pytest.raises(ProductionModeError, match="Security level too low for production"):
            security.validate_security_config()

    def test_get_security_status(self, security):
        """Test getting security status."""
        security.set_security_level("high")
        security.enable_feature("encryption")
        
        status = security.get_security_status()
        
        assert status["level"] == "high"
        assert status["enabled_features"] == ["encryption"]
        assert status["secure"] is True


class TestProductionModePerformance:
    """Test ProductionModePerformance class."""

    @pytest.fixture
    def performance(self):
        """Create ProductionModePerformance instance."""
        return ProductionModePerformance()

    def test_initialization(self, performance):
        """Test ProductionModePerformance initialization."""
        assert performance._performance_mode == "development"
        assert performance._optimizations == set()

    def test_set_performance_mode(self, performance):
        """Test setting performance mode."""
        performance.set_performance_mode("optimized")
        
        assert performance._performance_mode == "optimized"

    def test_get_performance_mode(self, performance):
        """Test getting performance mode."""
        performance.set_performance_mode("balanced")
        
        mode = performance.get_performance_mode()
        
        assert mode == "balanced"

    def test_enable_optimization(self, performance):
        """Test enabling performance optimization."""
        performance.enable_optimization("caching")
        
        assert "caching" in performance._optimizations

    def test_disable_optimization(self, performance):
        """Test disabling performance optimization."""
        performance.enable_optimization("caching")
        performance.disable_optimization("caching")
        
        assert "caching" not in performance._optimizations

    def test_is_optimization_enabled(self, performance):
        """Test checking if performance optimization is enabled."""
        performance.enable_optimization("caching")
        
        assert performance.is_optimization_enabled("caching") is True
        assert performance.is_optimization_enabled("compression") is False

    def test_validate_performance_config(self, performance):
        """Test validating performance configuration."""
        performance.set_performance_mode("optimized")
        performance.enable_optimization("caching")
        performance.enable_optimization("compression")
        
        # Should not raise exception
        performance.validate_performance_config()

    def test_validate_performance_config_development_mode(self, performance):
        """Test validating performance configuration with development mode."""
        performance.set_performance_mode("development")
        
        with pytest.raises(ProductionModeError, match="Performance mode must be optimized for production"):
            performance.validate_performance_config()

    def test_get_performance_status(self, performance):
        """Test getting performance status."""
        performance.set_performance_mode("optimized")
        performance.enable_optimization("caching")
        
        status = performance.get_performance_status()
        
        assert status["mode"] == "optimized"
        assert status["enabled_optimizations"] == ["caching"]
        assert status["optimized"] is True


class TestProductionModeCompliance:
    """Test ProductionModeCompliance class."""

    @pytest.fixture
    def compliance(self):
        """Create ProductionModeCompliance instance."""
        return ProductionModeCompliance()

    def test_initialization(self, compliance):
        """Test ProductionModeCompliance initialization."""
        assert compliance._standards == set()
        assert compliance._requirements == {}

    def test_add_standard(self, compliance):
        """Test adding compliance standard."""
        compliance.add_standard("SOC2")
        
        assert "SOC2" in compliance._standards

    def test_remove_standard(self, compliance):
        """Test removing compliance standard."""
        compliance.add_standard("SOC2")
        compliance.remove_standard("SOC2")
        
        assert "SOC2" not in compliance._standards

    def test_add_requirement(self, compliance):
        """Test adding compliance requirement."""
        compliance.add_requirement("SOC2", "encryption", "Data must be encrypted at rest")
        
        assert "SOC2" in compliance._requirements
        assert "encryption" in compliance._requirements["SOC2"]

    def test_remove_requirement(self, compliance):
        """Test removing compliance requirement."""
        compliance.add_requirement("SOC2", "encryption", "Data must be encrypted at rest")
        compliance.remove_requirement("SOC2", "encryption")
        
        assert "encryption" not in compliance._requirements["SOC2"]

    def test_validate_compliance(self, compliance):
        """Test validating compliance."""
        compliance.add_standard("SOC2")
        compliance.add_requirement("SOC2", "encryption", "Data must be encrypted at rest")
        
        # Should not raise exception
        compliance.validate_compliance("SOC2")

    def test_validate_compliance_missing_standard(self, compliance):
        """Test validating compliance with missing standard."""
        with pytest.raises(ProductionModeError, match="Compliance standard not found"):
            compliance.validate_compliance("SOC2")

    def test_get_compliance_status(self, compliance):
        """Test getting compliance status."""
        compliance.add_standard("SOC2")
        compliance.add_requirement("SOC2", "encryption", "Data must be encrypted at rest")
        
        status = compliance.get_compliance_status("SOC2")
        
        assert status["standard"] == "SOC2"
        assert status["requirements"] == {"encryption": "Data must be encrypted at rest"}
        assert status["compliant"] is True

    def test_get_all_standards(self, compliance):
        """Test getting all compliance standards."""
        compliance.add_standard("SOC2")
        compliance.add_standard("ISO27001")
        
        standards = compliance.get_all_standards()
        
        assert len(standards) == 2
        assert "SOC2" in standards
        assert "ISO27001" in standards


class TestProductionModeExceptions:
    """Test production mode-related exceptions."""

    def test_production_mode_error(self):
        """Test ProductionModeError exception."""
        error = ProductionModeError("Test error", error_code="TEST_ERROR")
        
        assert str(error) == "Test error"
        assert error.error_code == "TEST_ERROR"

    def test_production_mode_warning(self):
        """Test ProductionModeWarning exception."""
        warning = ProductionModeWarning("Test warning")
        
        assert str(warning) == "Test warning"


class TestProductionModeIntegration:
    """Integration tests for production mode system."""

    @pytest.mark.asyncio
    async def test_full_production_mode_lifecycle(self):
        """Test complete production mode lifecycle."""
        # Initialize detector
        detector = ProductionModeDetector()
        
        # Detect production mode
        with patch.dict(os.environ, {"CRONOS_ENVIRONMENT": "production"}):
            config = detector.detect()
            
            assert config.environment == EnvironmentType.PRODUCTION
            assert config.debug is False
            assert config.log_level == "INFO"
            assert config.enable_metrics is True
            assert config.enable_tracing is True
            assert config.security_level == "high"
            assert config.performance_mode == "optimized"
        
        # Initialize manager
        manager = ProductionModeManager()
        manager.initialize(config)
        
        # Validate configuration
        manager.validate_config()
        
        # Check environment
        assert manager.is_production() is True
        assert manager.is_development() is False
        
        # Get health status
        health = manager.get_health_status()
        assert health["status"] == "healthy"
        assert health["environment"] == "production"
        
        # Get metrics
        metrics = manager.get_metrics()
        assert metrics["environment"] == "production"
        assert metrics["initialized"] is True

    @pytest.mark.asyncio
    async def test_production_mode_validation(self):
        """Test production mode validation."""
        # Create invalid production config
        config = ProductionModeConfig(
            environment=EnvironmentType.PRODUCTION,
            debug=True,  # Invalid: debug enabled in production
            log_level="DEBUG",  # Invalid: debug log level in production
            enable_metrics=False,  # Invalid: metrics disabled in production
            enable_tracing=False,  # Invalid: tracing disabled in production
            security_level="low",  # Invalid: low security in production
            performance_mode="development"  # Invalid: development mode in production
        )
        
        # Initialize manager
        manager = ProductionModeManager()
        manager.initialize(config)
        
        # Should raise validation error
        with pytest.raises(ProductionModeError, match="Invalid production mode configuration"):
            manager.validate_config()

    @pytest.mark.asyncio
    async def test_production_mode_monitoring(self):
        """Test production mode monitoring."""
        # Initialize components
        detector = ProductionModeDetector()
        manager = ProductionModeManager()
        
        # Detect and initialize
        with patch.dict(os.environ, {"CRONOS_ENVIRONMENT": "production"}):
            config = detector.detect()
            manager.initialize(config)
        
        # Record some events
        manager.monitor.record_event({
            "type": "config_change",
            "environment": "production",
            "timestamp": "2023-01-01T00:00:00Z"
        })
        
        manager.monitor.record_metric("config_validation", 1.0)
        
        # Check monitoring data
        events = manager.monitor.get_events()
        assert len(events) == 1
        assert events[0]["type"] == "config_change"
        
        metrics = manager.monitor.get_metrics()
        assert metrics["config_validation"] == 1.0

    @pytest.mark.asyncio
    async def test_production_mode_health_checks(self):
        """Test production mode health checks."""
        # Initialize manager
        manager = ProductionModeManager()
        
        with patch.dict(os.environ, {"CRONOS_ENVIRONMENT": "production"}):
            config = manager.detect_and_initialize()
        
        # Register health check
        def config_validation_check():
            return {"status": "healthy", "message": "Configuration is valid"}
        
        manager.health_check.register_check("config_validation", config_validation_check)
        
        # Check health
        health = manager.health_check.check_health("config_validation")
        assert health["status"] == "healthy"
        assert health["message"] == "Configuration is valid"
        
        # Check overall health
        overall_health = manager.health_check.get_overall_health()
        assert overall_health["status"] == "healthy"
        assert overall_health["total_checks"] == 1
        assert overall_health["healthy_checks"] == 1
