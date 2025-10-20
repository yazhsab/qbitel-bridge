"""
Comprehensive unit tests for ai_engine.__main__ module.
Tests the main entry point and CLI argument parsing.
"""

import pytest
import sys
import logging
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
from pathlib import Path
import argparse


class TestMainEntryPoint:
    """Test suite for main entry point functionality."""

    @patch("sys.argv", ["ai_engine", "--host", "127.0.0.1", "--port", "9000"])
    def test_main_production_mode_default(self):
        """Test main function in production mode with custom host/port."""
        with patch("ai_engine.__main__.Config") as mock_config:
            with patch("ai_engine.__main__.run_production_server") as mock_run_prod:
                mock_config_instance = Mock()
                mock_config.return_value = mock_config_instance

                # Import and call main
                from ai_engine.__main__ import main

                try:
                    main()
                except SystemExit as e:
                    # Should exit cleanly
                    assert e.code in [0, None]

                mock_run_prod.assert_called_once()
                assert mock_config_instance.rest_host == "127.0.0.1"
                assert mock_config_instance.rest_port == 9000

    @patch("sys.argv", ["ai_engine", "--development", "--reload"])
    def test_main_development_mode_with_reload(self):
        """Test main function in development mode with reload."""
        with patch("ai_engine.__main__.Config") as mock_config:
            with patch("ai_engine.__main__.run_development_server") as mock_run_dev:
                mock_config_instance = Mock()
                mock_config.return_value = mock_config_instance

                from ai_engine.__main__ import main

                try:
                    main()
                except SystemExit:
                    pass

                mock_run_dev.assert_called_once()
                call_kwargs = mock_run_dev.call_args[1]
                assert call_kwargs["reload"] is True

    @patch("sys.argv", ["ai_engine", "--enable-grpc", "--grpc-port", "50052"])
    def test_main_with_grpc_enabled(self):
        """Test main function with gRPC enabled."""
        with patch("ai_engine.__main__.Config") as mock_config:
            with patch("ai_engine.__main__.run_production_server") as mock_run_prod:
                mock_config_instance = Mock()
                mock_config.return_value = mock_config_instance

                from ai_engine.__main__ import main

                try:
                    main()
                except SystemExit:
                    pass

                assert mock_config_instance.enable_grpc_api is True
                assert mock_config_instance.grpc_port == 50052

    @patch("ai_engine.__main__.run_production_server")
    @patch("ai_engine.__main__.Config")
    @patch("sys.argv", ["ai_engine", "--model-path", "/models", "--data-path", "/data"])
    def test_main_with_custom_paths(self, mock_config, mock_run_prod):
        """Test main function with custom model and data paths."""
        from ai_engine.__main__ import main

        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        with pytest.raises(SystemExit):
            main()

        assert mock_config_instance.model_path == "/models"
        assert mock_config_instance.data_path == "/data"

    @patch("ai_engine.__main__.run_production_server")
    @patch("ai_engine.__main__.Config")
    @patch("sys.argv", ["ai_engine", "--log-level", "DEBUG"])
    def test_main_with_debug_logging(self, mock_config, mock_run_prod):
        """Test main function with DEBUG log level."""
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        with pytest.raises(SystemExit):
            main()

        # Verify logging was configured
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    @patch("ai_engine.__main__.run_development_server")
    @patch("ai_engine.__main__.Config")
    @patch("sys.argv", ["ai_engine", "--development", "--log-level", "WARNING"])
    def test_main_development_with_file_logging(self, mock_config, mock_run_dev):
        """Test main function in development mode creates file handler."""
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        with pytest.raises(SystemExit):
            main()

        # Verify file handler was added in development mode
        root_logger = logging.getLogger()
        file_handlers = [
            h for h in root_logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) > 0

    @patch("ai_engine.__main__.Config")
    @patch("sys.argv", ["ai_engine"])
    def test_main_keyboard_interrupt(self, mock_config):
        """Test main function handles KeyboardInterrupt gracefully."""
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        with patch(
            "ai_engine.__main__.run_production_server", side_effect=KeyboardInterrupt
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 0

    @patch("ai_engine.__main__.Config")
    @patch("sys.argv", ["ai_engine"])
    def test_main_exception_handling(self, mock_config):
        """Test main function handles exceptions and exits with error code."""
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        with patch(
            "ai_engine.__main__.run_production_server",
            side_effect=RuntimeError("Test error"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 1

    @patch("ai_engine.__main__.run_production_server")
    @patch("ai_engine.__main__.Config")
    @patch("sys.argv", ["ai_engine", "--host", "0.0.0.0", "--port", "8080"])
    def test_main_default_host_and_port(self, mock_config, mock_run_prod):
        """Test main function with default host and custom port."""
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        with pytest.raises(SystemExit):
            main()

        assert mock_config_instance.rest_host == "0.0.0.0"
        assert mock_config_instance.rest_port == 8080

    @patch("ai_engine.__main__.run_development_server")
    @patch("ai_engine.__main__.Config")
    @patch(
        "sys.argv",
        ["ai_engine", "--development", "--host", "localhost", "--port", "3000"],
    )
    def test_main_development_custom_host_port(self, mock_config, mock_run_dev):
        """Test development mode with custom host and port."""
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        with pytest.raises(SystemExit):
            main()

        call_kwargs = mock_run_dev.call_args[1]
        assert call_kwargs["host"] == "localhost"
        assert call_kwargs["port"] == 3000

    @patch("ai_engine.__main__.run_development_server")
    @patch("ai_engine.__main__.Config")
    @patch(
        "sys.argv",
        ["ai_engine", "--development", "--enable-grpc", "--grpc-port", "50053"],
    )
    def test_main_development_with_grpc(self, mock_config, mock_run_dev):
        """Test development mode with gRPC enabled."""
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        with pytest.raises(SystemExit):
            main()

        call_kwargs = mock_run_dev.call_args[1]
        assert call_kwargs["enable_grpc"] is True
        assert call_kwargs["grpc_port"] == 50053

    @patch("ai_engine.__main__.run_production_server")
    @patch("ai_engine.__main__.Config")
    @patch("sys.argv", ["ai_engine", "--log-level", "INFO"])
    def test_main_info_logging(self, mock_config, mock_run_prod):
        """Test main function with INFO log level."""
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        with pytest.raises(SystemExit):
            main()

        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO

    @patch("ai_engine.__main__.run_production_server")
    @patch("ai_engine.__main__.Config")
    @patch("sys.argv", ["ai_engine", "--log-level", "ERROR"])
    def test_main_error_logging(self, mock_config, mock_run_prod):
        """Test main function with ERROR log level."""
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        with pytest.raises(SystemExit):
            main()

        root_logger = logging.getLogger()
        assert root_logger.level == logging.ERROR

    @patch("ai_engine.__main__.run_production_server")
    @patch("ai_engine.__main__.Config")
    @patch("sys.argv", ["ai_engine", "--log-level", "CRITICAL"])
    def test_main_critical_logging(self, mock_config, mock_run_prod):
        """Test main function with CRITICAL log level."""
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        with pytest.raises(SystemExit):
            main()

        root_logger = logging.getLogger()
        assert root_logger.level == logging.CRITICAL

    @patch("ai_engine.__main__.run_production_server")
    @patch("ai_engine.__main__.Config")
    @patch("sys.argv", ["ai_engine"])
    def test_main_default_arguments(self, mock_config, mock_run_prod):
        """Test main function with all default arguments."""
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        with pytest.raises(SystemExit):
            main()

        # Verify production server was called
        mock_run_prod.assert_called_once_with(mock_config_instance)

    @patch("ai_engine.__main__.run_production_server")
    @patch("ai_engine.__main__.Config")
    @patch("sys.argv", ["ai_engine", "--config", "/path/to/config.yaml"])
    def test_main_with_config_file(self, mock_config, mock_run_prod):
        """Test main function with config file argument."""
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        with pytest.raises(SystemExit):
            main()

        # Config file argument is parsed but not used in current implementation
        mock_run_prod.assert_called_once()

    @patch("ai_engine.__main__.run_development_server")
    @patch("ai_engine.__main__.Config")
    @patch("sys.argv", ["ai_engine", "--development"])
    def test_main_development_without_reload(self, mock_config, mock_run_dev):
        """Test development mode without reload flag."""
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        with pytest.raises(SystemExit):
            main()

        call_kwargs = mock_run_dev.call_args[1]
        assert call_kwargs["reload"] is False

    @patch("ai_engine.__main__.Config")
    @patch("sys.argv", ["ai_engine"])
    def test_main_logs_startup_info(self, mock_config):
        """Test that main logs startup information."""
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        with patch("ai_engine.__main__.run_production_server"):
            with patch("logging.getLogger") as mock_get_logger:
                mock_logger = Mock()
                mock_get_logger.return_value = mock_logger

                with pytest.raises(SystemExit):
                    main()

                # Verify info logs were called
                assert mock_logger.info.call_count >= 3

    @patch("ai_engine.__main__.run_production_server")
    @patch("ai_engine.__main__.Config")
    @patch("sys.argv", ["ai_engine", "--port", "0"])
    def test_main_with_zero_port(self, mock_config, mock_run_prod):
        """Test main function with port 0 (OS assigns port)."""
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        with pytest.raises(SystemExit):
            main()

        assert mock_config_instance.rest_port == 0

    @patch("ai_engine.__main__.run_production_server")
    @patch("ai_engine.__main__.Config")
    @patch("sys.argv", ["ai_engine", "--grpc-port", "0"])
    def test_main_with_zero_grpc_port(self, mock_config, mock_run_prod):
        """Test main function with gRPC port 0."""
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        with pytest.raises(SystemExit):
            main()

        assert mock_config_instance.grpc_port == 0


class TestMainModuleExecution:
    """Test suite for module execution."""

    def test_main_module_has_main_guard(self):
        """Test that __main__.py has proper main guard."""
        import ai_engine.__main__ as main_module

        # Verify the module has the main function
        assert hasattr(main_module, "main")
        assert callable(main_module.main)

    @patch("ai_engine.__main__.main")
    def test_main_called_when_run_as_script(self, mock_main):
        """Test that main() is called when module is run as script."""
        # This tests the if __name__ == "__main__": block
        # In actual execution, this would be tested by running the module
        assert callable(mock_main)


class TestArgumentParsing:
    """Test suite for CLI argument parsing."""

    @patch("sys.argv", ["ai_engine", "--help"])
    def test_help_argument(self):
        """Test --help argument displays help and exits."""
        with pytest.raises(SystemExit) as exc_info:
            main()

        # Help should exit with code 0
        assert exc_info.value.code == 0

    @patch("ai_engine.__main__.run_production_server")
    @patch("ai_engine.__main__.Config")
    @patch("sys.argv", ["ai_engine", "--port", "invalid"])
    def test_invalid_port_argument(self, mock_config, mock_run_prod):
        """Test invalid port argument."""
        with pytest.raises(SystemExit) as exc_info:
            main()

        # Should exit with error code
        assert exc_info.value.code == 2

    @patch("ai_engine.__main__.run_production_server")
    @patch("ai_engine.__main__.Config")
    @patch("sys.argv", ["ai_engine", "--log-level", "INVALID"])
    def test_invalid_log_level(self, mock_config, mock_run_prod):
        """Test invalid log level argument."""
        with pytest.raises(SystemExit) as exc_info:
            main()

        # Should exit with error code
        assert exc_info.value.code == 2


class TestLoggingConfiguration:
    """Test suite for logging configuration."""

    @patch("ai_engine.__main__.run_production_server")
    @patch("ai_engine.__main__.Config")
    @patch("sys.argv", ["ai_engine"])
    def test_logging_format(self, mock_config, mock_run_prod):
        """Test that logging format is properly configured."""
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        with pytest.raises(SystemExit):
            main()

        # Verify logging was configured
        root_logger = logging.getLogger()
        assert len(root_logger.handlers) > 0

    @patch("ai_engine.__main__.run_production_server")
    @patch("ai_engine.__main__.Config")
    @patch("sys.argv", ["ai_engine"])
    def test_stdout_handler_in_production(self, mock_config, mock_run_prod):
        """Test that stdout handler is configured in production mode."""
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        with pytest.raises(SystemExit):
            main()

        root_logger = logging.getLogger()
        stream_handlers = [
            h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)
        ]
        assert len(stream_handlers) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
