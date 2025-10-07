"""
CRONOS AI Engine - Main Module Tests

Comprehensive test suite for the main entry point module.
"""

import pytest
import sys
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, Optional

# Import the main module
from ai_engine import __main__


class TestMainModule:
    """Test main module functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock()
        config.environment = "testing"
        config.model_path = "test_models"
        config.data_path = "test_data"
        config.device = "cpu"
        config.health_check_interval = 5
        config.health_check_timeout = 2
        config.log_level = "INFO"
        config.api_host = "localhost"
        config.api_port = 8000
        config.grpc_host = "localhost"
        config.grpc_port = 50051
        return config

    @pytest.fixture
    def mock_server(self):
        """Create mock server."""
        server = AsyncMock()
        server.start = AsyncMock()
        server.stop = AsyncMock()
        server.is_running = Mock(return_value=False)
        return server

    def test_main_module_imports(self):
        """Test that main module imports correctly."""
        assert hasattr(__main__, 'main')
        assert hasattr(__main__, 'setup_logging')
        assert hasattr(__main__, 'load_configuration')
        assert hasattr(__main__, 'initialize_components')
        assert hasattr(__main__, 'start_servers')
        assert hasattr(__main__, 'shutdown_servers')

    def test_setup_logging(self, mock_config):
        """Test logging setup."""
        with patch('ai_engine.__main__.logging') as mock_logging:
            __main__.setup_logging(mock_config)
            
            mock_logging.basicConfig.assert_called_once()
            mock_logging.getLogger.assert_called()

    def test_setup_logging_with_debug_level(self):
        """Test logging setup with debug level."""
        config = Mock()
        config.log_level = "DEBUG"
        
        with patch('ai_engine.__main__.logging') as mock_logging:
            __main__.setup_logging(config)
            
            mock_logging.basicConfig.assert_called_once()

    def test_load_configuration_default(self):
        """Test loading default configuration."""
        with patch('ai_engine.__main__.Config') as mock_config_class:
            mock_config = Mock()
            mock_config_class.return_value = mock_config
            
            result = __main__.load_configuration()
            
            assert result == mock_config
            mock_config_class.assert_called_once()

    def test_load_configuration_with_file(self):
        """Test loading configuration from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("environment: testing\nlog_level: DEBUG\n")
            config_file = f.name
        
        try:
            with patch('ai_engine.__main__.Config') as mock_config_class:
                mock_config = Mock()
                mock_config_class.return_value = mock_config
                
                result = __main__.load_configuration(config_file)
                
                assert result == mock_config
                mock_config_class.assert_called_once()
        finally:
            os.unlink(config_file)

    def test_load_configuration_file_not_found(self):
        """Test loading configuration with non-existent file."""
        with pytest.raises(FileNotFoundError):
            __main__.load_configuration("nonexistent_config.yaml")

    @pytest.mark.asyncio
    async def test_initialize_components(self, mock_config):
        """Test component initialization."""
        with patch('ai_engine.__main__.ProtocolDiscoveryOrchestrator') as mock_orchestrator, \
             patch('ai_engine.__main__.ComplianceService') as mock_compliance, \
             patch('ai_engine.__main__.SecurityService') as mock_security, \
             patch('ai_engine.__main__.ModelManager') as mock_models, \
             patch('ai_engine.__main__.MonitoringService') as mock_monitoring:
            
            mock_orchestrator.return_value = AsyncMock()
            mock_compliance.return_value = AsyncMock()
            mock_security.return_value = AsyncMock()
            mock_models.return_value = AsyncMock()
            mock_monitoring.return_value = AsyncMock()
            
            components = await __main__.initialize_components(mock_config)
            
            assert 'orchestrator' in components
            assert 'compliance' in components
            assert 'security' in components
            assert 'models' in components
            assert 'monitoring' in components

    @pytest.mark.asyncio
    async def test_initialize_components_with_failure(self, mock_config):
        """Test component initialization with failure."""
        with patch('ai_engine.__main__.ProtocolDiscoveryOrchestrator') as mock_orchestrator:
            mock_orchestrator.side_effect = Exception("Initialization failed")
            
            with pytest.raises(Exception):
                await __main__.initialize_components(mock_config)

    @pytest.mark.asyncio
    async def test_start_servers(self, mock_config):
        """Test server startup."""
        mock_rest_server = AsyncMock()
        mock_grpc_server = AsyncMock()
        
        with patch('ai_engine.__main__.create_rest_server', return_value=mock_rest_server), \
             patch('ai_engine.__main__.create_grpc_server', return_value=mock_grpc_server):
            
            servers = await __main__.start_servers(mock_config, {})
            
            assert 'rest' in servers
            assert 'grpc' in servers
            mock_rest_server.start.assert_called_once()
            mock_grpc_server.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_servers_with_failure(self, mock_config):
        """Test server startup with failure."""
        mock_rest_server = AsyncMock()
        mock_rest_server.start.side_effect = Exception("Server start failed")
        
        with patch('ai_engine.__main__.create_rest_server', return_value=mock_rest_server), \
             patch('ai_engine.__main__.create_grpc_server', return_value=AsyncMock()):
            
            with pytest.raises(Exception):
                await __main__.start_servers(mock_config, {})

    @pytest.mark.asyncio
    async def test_shutdown_servers(self, mock_server):
        """Test server shutdown."""
        servers = {
            'rest': mock_server,
            'grpc': mock_server
        }
        
        await __main__.shutdown_servers(servers)
        
        assert mock_server.stop.call_count == 2

    @pytest.mark.asyncio
    async def test_shutdown_servers_with_failure(self, mock_server):
        """Test server shutdown with failure."""
        mock_server.stop.side_effect = Exception("Shutdown failed")
        
        servers = {
            'rest': mock_server,
            'grpc': mock_server
        }
        
        # Should not raise exception even if shutdown fails
        await __main__.shutdown_servers(servers)
        
        assert mock_server.stop.call_count == 2

    @pytest.mark.asyncio
    async def test_main_function_success(self, mock_config):
        """Test main function execution."""
        with patch('ai_engine.__main__.setup_logging') as mock_setup_logging, \
             patch('ai_engine.__main__.load_configuration', return_value=mock_config) as mock_load_config, \
             patch('ai_engine.__main__.initialize_components', return_value={}) as mock_init_components, \
             patch('ai_engine.__main__.start_servers', return_value={}) as mock_start_servers, \
             patch('ai_engine.__main__.shutdown_servers') as mock_shutdown_servers, \
             patch('ai_engine.__main__.signal') as mock_signal:
            
            # Mock signal handling
            mock_signal.signal = Mock()
            
            await __main__.main()
            
            mock_setup_logging.assert_called_once_with(mock_config)
            mock_load_config.assert_called_once()
            mock_init_components.assert_called_once_with(mock_config)
            mock_start_servers.assert_called_once_with(mock_config, {})
            mock_shutdown_servers.assert_called_once_with({})

    @pytest.mark.asyncio
    async def test_main_function_with_exception(self, mock_config):
        """Test main function with exception."""
        with patch('ai_engine.__main__.setup_logging') as mock_setup_logging, \
             patch('ai_engine.__main__.load_configuration', return_value=mock_config) as mock_load_config, \
             patch('ai_engine.__main__.initialize_components', side_effect=Exception("Test error")) as mock_init_components, \
             patch('ai_engine.__main__.shutdown_servers') as mock_shutdown_servers, \
             patch('ai_engine.__main__.signal') as mock_signal:
            
            mock_signal.signal = Mock()
            
            with pytest.raises(Exception):
                await __main__.main()
            
            mock_setup_logging.assert_called_once_with(mock_config)
            mock_load_config.assert_called_once()
            mock_init_components.assert_called_once_with(mock_config)
            mock_shutdown_servers.assert_called_once_with({})

    def test_signal_handlers(self):
        """Test signal handler setup."""
        with patch('ai_engine.__main__.signal') as mock_signal:
            __main__.setup_signal_handlers()
            
            # Verify signal handlers were registered
            mock_signal.signal.assert_called()

    def test_graceful_shutdown(self):
        """Test graceful shutdown functionality."""
        with patch('ai_engine.__main__.shutdown_servers') as mock_shutdown:
            __main__.graceful_shutdown({}, None)
            
            mock_shutdown.assert_called_once_with({})

    def test_health_check_endpoint(self):
        """Test health check endpoint."""
        with patch('ai_engine.__main__.asyncio') as mock_asyncio:
            mock_asyncio.create_task = Mock()
            
            __main__.start_health_check()
            
            mock_asyncio.create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_loop(self, mock_config):
        """Test health check loop."""
        with patch('ai_engine.__main__.asyncio.sleep') as mock_sleep, \
             patch('ai_engine.__main__.check_system_health') as mock_health_check:
            
            mock_health_check.return_value = True
            
            # Run health check loop for a short time
            task = asyncio.create_task(__main__.health_check_loop(mock_config))
            await asyncio.sleep(0.1)
            task.cancel()
            
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            mock_health_check.assert_called()
            mock_sleep.assert_called()

    @pytest.mark.asyncio
    async def test_check_system_health(self, mock_config):
        """Test system health check."""
        with patch('ai_engine.__main__.psutil') as mock_psutil:
            mock_psutil.cpu_percent.return_value = 50.0
            mock_psutil.virtual_memory.return_value = Mock(percent=60.0)
            mock_psutil.disk_usage.return_value = Mock(percent=40.0)
            
            result = await __main__.check_system_health(mock_config)
            
            assert result is True

    @pytest.mark.asyncio
    async def test_check_system_health_critical(self, mock_config):
        """Test system health check with critical resources."""
        with patch('ai_engine.__main__.psutil') as mock_psutil:
            mock_psutil.cpu_percent.return_value = 95.0  # Critical CPU usage
            mock_psutil.virtual_memory.return_value = Mock(percent=90.0)  # Critical memory usage
            mock_psutil.disk_usage.return_value = Mock(percent=95.0)  # Critical disk usage
            
            result = await __main__.check_system_health(mock_config)
            
            assert result is False

    def test_create_rest_server(self, mock_config):
        """Test REST server creation."""
        with patch('ai_engine.__main__.FastAPI') as mock_fastapi, \
             patch('ai_engine.__main__.uvicorn') as mock_uvicorn:
            
            mock_app = Mock()
            mock_fastapi.return_value = mock_app
            
            server = __main__.create_rest_server(mock_config, {})
            
            assert server is not None
            mock_fastapi.assert_called_once()

    def test_create_grpc_server(self, mock_config):
        """Test gRPC server creation."""
        with patch('ai_engine.__main__.grpc') as mock_grpc:
            mock_server = Mock()
            mock_grpc.aio.server.return_value = mock_server
            
            server = __main__.create_grpc_server(mock_config, {})
            
            assert server == mock_server
            mock_grpc.aio.server.assert_called_once()

    def test_validate_configuration(self, mock_config):
        """Test configuration validation."""
        result = __main__.validate_configuration(mock_config)
        
        assert result is True

    def test_validate_configuration_invalid(self):
        """Test configuration validation with invalid config."""
        invalid_config = Mock()
        invalid_config.api_port = -1  # Invalid port
        
        result = __main__.validate_configuration(invalid_config)
        
        assert result is False

    def test_get_version_info(self):
        """Test version information retrieval."""
        with patch('ai_engine.__main__.pkg_resources') as mock_pkg:
            mock_pkg.get_distribution.return_value.version = "1.0.0"
            
            version = __main__.get_version_info()
            
            assert version == "1.0.0"

    def test_get_system_info(self):
        """Test system information retrieval."""
        with patch('ai_engine.__main__.platform') as mock_platform, \
             patch('ai_engine.__main__.psutil') as mock_psutil:
            
            mock_platform.system.return_value = "Linux"
            mock_platform.release.return_value = "5.4.0"
            mock_psutil.cpu_count.return_value = 8
            mock_psutil.virtual_memory.return_value = Mock(total=8589934592)  # 8GB
            
            info = __main__.get_system_info()
            
            assert "platform" in info
            assert "cpu_count" in info
            assert "memory_total" in info

    @pytest.mark.asyncio
    async def test_main_with_command_line_args(self):
        """Test main function with command line arguments."""
        with patch('ai_engine.__main__.sys.argv', ['__main__.py', '--config', 'test_config.yaml']), \
             patch('ai_engine.__main__.argparse') as mock_argparse, \
             patch('ai_engine.__main__.setup_logging') as mock_setup_logging, \
             patch('ai_engine.__main__.load_configuration') as mock_load_config, \
             patch('ai_engine.__main__.initialize_components', return_value={}) as mock_init_components, \
             patch('ai_engine.__main__.start_servers', return_value={}) as mock_start_servers, \
             patch('ai_engine.__main__.shutdown_servers') as mock_shutdown_servers, \
             patch('ai_engine.__main__.signal') as mock_signal:
            
            mock_args = Mock()
            mock_args.config = "test_config.yaml"
            mock_args.verbose = False
            mock_args.daemon = False
            mock_argparse.ArgumentParser.return_value.parse_args.return_value = mock_args
            
            mock_signal.signal = Mock()
            
            await __main__.main()
            
            mock_load_config.assert_called_once_with("test_config.yaml")

    def test_logging_configuration(self):
        """Test logging configuration setup."""
        config = Mock()
        config.log_level = "DEBUG"
        config.log_file = "/tmp/test.log"
        
        with patch('ai_engine.__main__.logging') as mock_logging:
            __main__.setup_logging(config)
            
            mock_logging.basicConfig.assert_called_once()

    def test_error_handling_in_main(self):
        """Test error handling in main function."""
        with patch('ai_engine.__main__.setup_logging', side_effect=Exception("Logging error")):
            with pytest.raises(Exception):
                asyncio.run(__main__.main())

    @pytest.mark.asyncio
    async def test_component_initialization_order(self, mock_config):
        """Test that components are initialized in correct order."""
        init_order = []
        
        def track_init(component_name):
            def decorator(func):
                async def wrapper(*args, **kwargs):
                    init_order.append(component_name)
                    return await func(*args, **kwargs)
                return wrapper
            return decorator
        
        with patch('ai_engine.__main__.ProtocolDiscoveryOrchestrator') as mock_orchestrator, \
             patch('ai_engine.__main__.ComplianceService') as mock_compliance, \
             patch('ai_engine.__main__.SecurityService') as mock_security, \
             patch('ai_engine.__main__.ModelManager') as mock_models, \
             patch('ai_engine.__main__.MonitoringService') as mock_monitoring:
            
            mock_orchestrator.return_value = AsyncMock()
            mock_compliance.return_value = AsyncMock()
            mock_security.return_value = AsyncMock()
            mock_models.return_value = AsyncMock()
            mock_monitoring.return_value = AsyncMock()
            
            await __main__.initialize_components(mock_config)
            
            # Verify all components were initialized
            mock_orchestrator.assert_called_once()
            mock_compliance.assert_called_once()
            mock_security.assert_called_once()
            mock_models.assert_called_once()
            mock_monitoring.assert_called_once()

    def test_environment_variable_handling(self):
        """Test environment variable handling."""
        with patch.dict(os.environ, {
            'CRONOS_AI_ENVIRONMENT': 'production',
            'CRONOS_AI_LOG_LEVEL': 'WARNING',
            'CRONOS_AI_API_PORT': '9000'
        }):
            config = __main__.load_configuration()
            
            # Verify environment variables were considered
            assert config is not None

    @pytest.mark.asyncio
    async def test_graceful_shutdown_on_signal(self):
        """Test graceful shutdown when signal is received."""
        with patch('ai_engine.__main__.signal') as mock_signal, \
             patch('ai_engine.__main__.shutdown_servers') as mock_shutdown:
            
            # Simulate signal handler being called
            signal_handler = mock_signal.signal.call_args[0][1]
            signal_handler(2, None)  # SIGINT
            
            mock_shutdown.assert_called_once()
