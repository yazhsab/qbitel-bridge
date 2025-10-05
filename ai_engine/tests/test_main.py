"""
Tests for ai_engine/__main__.py - Main Entry Point
"""

import pytest
import sys
import logging
from unittest.mock import Mock, patch, MagicMock
from io import StringIO


class TestMainEntryPoint:
    """Test suite for main entry point."""

    @patch("ai_engine.__main__.run_production_server")
    @patch("ai_engine.__main__.Config")
    def test_main_production_mode(self, mock_config, mock_run_prod):
        """Test main function in production mode."""
        from ai_engine.__main__ import main

        # Mock config
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        # Mock sys.argv
        test_args = ["ai_engine", "--host", "0.0.0.0", "--port", "8080"]
        with patch.object(sys, "argv", test_args):
            try:
                main()
            except SystemExit:
                pass

        # Verify production server was called with config
        mock_run_prod.assert_called_once_with(mock_config_instance)

        # Verify config values were set from args
        assert mock_config_instance.rest_host == "0.0.0.0"
        assert mock_config_instance.rest_port == 8080

    @patch("ai_engine.__main__.run_development_server")
    @patch("ai_engine.__main__.Config")
    def test_main_development_mode(self, mock_config, mock_run_dev):
        """Test main function in development mode."""
        from ai_engine.__main__ import main

        # Mock config
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        # Mock sys.argv
        test_args = [
            "ai_engine",
            "--development",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ]
        with patch.object(sys, "argv", test_args):
            try:
                main()
            except SystemExit:
                pass

        # Verify development server was called with correct args
        mock_run_dev.assert_called_once()
        call_kwargs = mock_run_dev.call_args[1]
        assert call_kwargs["config"] == mock_config_instance
        assert call_kwargs["host"] == "127.0.0.1"
        assert call_kwargs["port"] == 8000
        assert call_kwargs["reload"] is False

    @patch("ai_engine.__main__.Config")
    def test_main_with_grpc_enabled(self, mock_config):
        """Test main function with gRPC enabled."""
        from ai_engine.__main__ import main

        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        test_args = ["ai_engine", "--enable-grpc", "--grpc-port", "50051"]
        with patch.object(sys, "argv", test_args):
            with patch("ai_engine.__main__.run_production_server"):
                try:
                    main()
                except SystemExit:
                    pass

        # Verify gRPC was enabled in config with correct port
        assert mock_config_instance.enable_grpc_api is True
        assert mock_config_instance.grpc_port == 50051

    @patch("ai_engine.__main__.Config")
    def test_main_with_custom_log_level(self, mock_config):
        """Test main function with custom log level."""
        from ai_engine.__main__ import main

        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        test_args = ["ai_engine", "--log-level", "DEBUG"]
        with patch.object(sys, "argv", test_args):
            with patch("ai_engine.__main__.run_production_server"):
                with patch("logging.basicConfig") as mock_logging:
                    try:
                        main()
                    except SystemExit:
                        pass

                    # Verify logging was configured with DEBUG level
                    assert mock_logging.called
                    call_kwargs = mock_logging.call_args[1]
                    assert call_kwargs["level"] == logging.DEBUG
                    assert "handlers" in call_kwargs
                    assert "format" in call_kwargs

    @patch("ai_engine.__main__.Config")
    def test_main_with_model_and_data_paths(self, mock_config):
        """Test main function with custom model and data paths."""
        from ai_engine.__main__ import main

        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        test_args = [
            "ai_engine",
            "--model-path",
            "/path/to/models",
            "--data-path",
            "/path/to/data",
        ]
        with patch.object(sys, "argv", test_args):
            with patch("ai_engine.__main__.run_production_server"):
                try:
                    main()
                except SystemExit:
                    pass

        # Verify paths were set
        assert mock_config_instance.model_path == "/path/to/models"
        assert mock_config_instance.data_path == "/path/to/data"

    @patch("ai_engine.__main__.Config")
    def test_main_keyboard_interrupt(self, mock_config):
        """Test main function handles keyboard interrupt."""
        from ai_engine.__main__ import main

        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        test_args = ["ai_engine"]
        with patch.object(sys, "argv", test_args):
            with patch(
                "ai_engine.__main__.run_production_server",
                side_effect=KeyboardInterrupt,
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main()

                assert exc_info.value.code == 0

    @patch("ai_engine.__main__.Config")
    def test_main_exception_handling(self, mock_config):
        """Test main function handles exceptions."""
        from ai_engine.__main__ import main

        mock_config.side_effect = Exception("Config error")

        test_args = ["ai_engine"]
        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 1

    @patch("ai_engine.__main__.run_development_server")
    @patch("ai_engine.__main__.Config")
    def test_main_with_reload_flag(self, mock_config, mock_run_dev):
        """Test main function with reload flag in development mode."""
        from ai_engine.__main__ import main

        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        test_args = ["ai_engine", "--development", "--reload"]
        with patch.object(sys, "argv", test_args):
            try:
                main()
            except SystemExit:
                pass

        # Verify reload was passed to development server
        call_kwargs = mock_run_dev.call_args[1]
        assert call_kwargs.get("reload") is True

    @patch("ai_engine.__main__.Config")
    def test_main_logging_configuration_development(self, mock_config):
        """Test logging configuration in development mode."""
        from ai_engine.__main__ import main

        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        test_args = ["ai_engine", "--development"]
        with patch.object(sys, "argv", test_args):
            with patch("ai_engine.__main__.run_development_server"):
                with patch("logging.basicConfig") as mock_logging:
                    with patch("logging.FileHandler") as mock_file_handler:
                        try:
                            main()
                        except SystemExit:
                            pass

                        # Verify file handler was added in development mode
                        assert mock_logging.called
                        call_kwargs = mock_logging.call_args[1]
                        handlers = call_kwargs["handlers"]
                        assert len(handlers) == 2  # StreamHandler + FileHandler
                        assert mock_file_handler.called

    @patch("ai_engine.__main__.Config")
    def test_main_all_arguments(self, mock_config):
        """Test main function with all possible arguments."""
        from ai_engine.__main__ import main

        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        test_args = [
            "ai_engine",
            "--host",
            "192.168.1.1",
            "--port",
            "9000",
            "--grpc-port",
            "50052",
            "--enable-grpc",
            "--development",
            "--reload",
            "--log-level",
            "WARNING",
            "--model-path",
            "/models",
            "--data-path",
            "/data",
        ]
        with patch.object(sys, "argv", test_args):
            with patch("ai_engine.__main__.run_development_server"):
                try:
                    main()
                except SystemExit:
                    pass

        # Verify all config values were set
        assert mock_config_instance.rest_host == "192.168.1.1"
        assert mock_config_instance.rest_port == 9000
        assert mock_config_instance.grpc_port == 50052
        assert mock_config_instance.enable_grpc_api is True
        assert mock_config_instance.model_path == "/models"
        assert mock_config_instance.data_path == "/data"

    def test_main_module_execution(self):
        """Test that __main__ can be executed as a module."""
        # This tests the if __name__ == "__main__" block
        with patch("ai_engine.__main__.main") as mock_main:
            # Simulate module execution
            import importlib
            import ai_engine.__main__ as main_module

            # The main function should be defined
            assert hasattr(main_module, "main")
            assert callable(main_module.main)

    @patch("ai_engine.__main__.Config")
    def test_main_version_logging(self, mock_config):
        """Test that version information is logged."""
        from ai_engine.__main__ import main

        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        test_args = ["ai_engine"]
        with patch.object(sys, "argv", test_args):
            with patch("ai_engine.__main__.run_production_server"):
                with patch("logging.getLogger") as mock_logger:
                    mock_logger_instance = Mock()
                    mock_logger.return_value = mock_logger_instance

                    try:
                        main()
                    except SystemExit:
                        pass

                    # Verify version and startup info was logged
                    assert mock_logger_instance.info.called
                    assert mock_logger_instance.info.call_count >= 3  # Starting, Version, Mode
                    info_calls = [call[0][0] for call in mock_logger_instance.info.call_args_list]
                    assert any("Starting CRONOS AI Engine" in str(call) for call in info_calls)
                    assert any("Version" in str(call) for call in info_calls)

    @patch("ai_engine.__main__.Config")
    def test_main_default_values(self, mock_config):
        """Test main function uses default values when no args provided."""
        from ai_engine.__main__ import main

        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        test_args = ["ai_engine"]
        with patch.object(sys, "argv", test_args):
            with patch("ai_engine.__main__.run_production_server"):
                try:
                    main()
                except SystemExit:
                    pass

        # Config should be created with defaults
        mock_config.assert_called_once()
