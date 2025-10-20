"""
Tests for the actual __main__.py module implementation.
"""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock
from io import StringIO
import ai_engine.__main__ as __main__


class TestMainModuleActual:
    """Tests for the actual __main__.py module."""

    def test_main_function_exists(self):
        """Test that main function exists."""
        assert hasattr(__main__, "main")
        assert callable(__main__.main)

    def test_main_function_with_help(self):
        """Test main function with --help argument."""
        with patch("sys.argv", ["__main__.py", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                __main__.main()
            assert exc_info.value.code == 0

    def test_main_function_development_mode(self):
        """Test main function in development mode."""
        with patch(
            "sys.argv",
            ["__main__.py", "--development", "--host", "127.0.0.1", "--port", "8080"],
        ):
            with patch("ai_engine.__main__.run_development_server") as mock_dev_server:
                with patch("ai_engine.__main__.Config") as mock_config_class:
                    mock_config = Mock()
                    mock_config_class.return_value = mock_config

                    __main__.main()

                    mock_dev_server.assert_called_once()
                    mock_config_class.assert_called_once()

    def test_main_function_production_mode(self):
        """Test main function in production mode."""
        with patch("sys.argv", ["__main__.py", "--host", "0.0.0.0", "--port", "8000"]):
            with patch("ai_engine.__main__.run_production_server") as mock_prod_server:
                with patch("ai_engine.__main__.Config") as mock_config_class:
                    mock_config = Mock()
                    mock_config_class.return_value = mock_config

                    __main__.main()

                    mock_prod_server.assert_called_once_with(mock_config)
                    mock_config_class.assert_called_once()

    def test_main_function_with_grpc(self):
        """Test main function with gRPC enabled."""
        with patch(
            "sys.argv", ["__main__.py", "--enable-grpc", "--grpc-port", "50052"]
        ):
            with patch("ai_engine.__main__.run_development_server") as mock_dev_server:
                with patch("ai_engine.__main__.Config") as mock_config_class:
                    mock_config = Mock()
                    mock_config_class.return_value = mock_config

                    __main__.main()

                    # Check that gRPC is enabled in config
                    assert mock_config.enable_grpc_api is True
                    assert mock_config.grpc_port == 50052

    def test_main_function_with_custom_config(self):
        """Test main function with custom configuration."""
        with patch("sys.argv", ["__main__.py", "--config", "/path/to/config.yaml"]):
            with patch("ai_engine.__main__.run_development_server") as mock_dev_server:
                with patch("ai_engine.__main__.Config") as mock_config_class:
                    mock_config = Mock()
                    mock_config_class.return_value = mock_config

                    __main__.main()

                    mock_config_class.assert_called_once()

    def test_main_function_with_model_path(self):
        """Test main function with custom model path."""
        with patch("sys.argv", ["__main__.py", "--model-path", "/path/to/models"]):
            with patch("ai_engine.__main__.run_development_server") as mock_dev_server:
                with patch("ai_engine.__main__.Config") as mock_config_class:
                    mock_config = Mock()
                    mock_config_class.return_value = mock_config

                    __main__.main()

                    assert mock_config.model_path == "/path/to/models"

    def test_main_function_with_data_path(self):
        """Test main function with custom data path."""
        with patch("sys.argv", ["__main__.py", "--data-path", "/path/to/data"]):
            with patch("ai_engine.__main__.run_development_server") as mock_dev_server:
                with patch("ai_engine.__main__.Config") as mock_config_class:
                    mock_config = Mock()
                    mock_config_class.return_value = mock_config

                    __main__.main()

                    assert mock_config.data_path == "/path/to/data"

    def test_main_function_with_log_level(self):
        """Test main function with custom log level."""
        with patch("sys.argv", ["__main__.py", "--log-level", "DEBUG"]):
            with patch("ai_engine.__main__.run_development_server") as mock_dev_server:
                with patch("ai_engine.__main__.Config") as mock_config_class:
                    mock_config = Mock()
                    mock_config_class.return_value = mock_config

                    with patch(
                        "ai_engine.__main__.logging.basicConfig"
                    ) as mock_logging:
                        __main__.main()

                        # Check that logging was configured with DEBUG level
                        mock_logging.assert_called_once()
                        call_args = mock_logging.call_args
                        assert call_args[1]["level"] == 10  # DEBUG level

    def test_main_function_with_reload(self):
        """Test main function with reload enabled."""
        with patch("sys.argv", ["__main__.py", "--development", "--reload"]):
            with patch("ai_engine.__main__.run_development_server") as mock_dev_server:
                with patch("ai_engine.__main__.Config") as mock_config_class:
                    mock_config = Mock()
                    mock_config_class.return_value = mock_config

                    __main__.main()

                    # Check that reload was passed to development server
                    mock_dev_server.assert_called_once()
                    call_args = mock_dev_server.call_args
                    assert call_args[1]["reload"] is True

    def test_main_function_keyboard_interrupt(self):
        """Test main function handles KeyboardInterrupt gracefully."""
        with patch("sys.argv", ["__main__.py"]):
            with patch(
                "ai_engine.__main__.run_development_server",
                side_effect=KeyboardInterrupt,
            ):
                with patch("ai_engine.__main__.Config") as mock_config_class:
                    mock_config = Mock()
                    mock_config_class.return_value = mock_config

                    with pytest.raises(SystemExit) as exc_info:
                        __main__.main()
                    assert exc_info.value.code == 0

    def test_main_function_exception_handling(self):
        """Test main function handles exceptions properly."""
        with patch("sys.argv", ["__main__.py"]):
            with patch(
                "ai_engine.__main__.run_development_server",
                side_effect=Exception("Test error"),
            ):
                with patch("ai_engine.__main__.Config") as mock_config_class:
                    mock_config = Mock()
                    mock_config_class.return_value = mock_config

                    with pytest.raises(SystemExit) as exc_info:
                        __main__.main()
                    assert exc_info.value.code == 1

    def test_main_function_logging_setup(self):
        """Test that logging is properly set up."""
        with patch("sys.argv", ["__main__.py", "--development"]):
            with patch("ai_engine.__main__.run_development_server") as mock_dev_server:
                with patch("ai_engine.__main__.Config") as mock_config_class:
                    mock_config = Mock()
                    mock_config_class.return_value = mock_config

                    with patch(
                        "ai_engine.__main__.logging.basicConfig"
                    ) as mock_logging:
                        with patch(
                            "ai_engine.__main__.logging.FileHandler"
                        ) as mock_file_handler:
                            __main__.main()

                            # Check that logging was configured
                            mock_logging.assert_called_once()
                            # In development mode, should have file handler
                            mock_file_handler.assert_called_once_with(
                                "cronos_ai_engine.log"
                            )

    def test_main_function_production_logging(self):
        """Test logging setup in production mode."""
        with patch("sys.argv", ["__main__.py"]):
            with patch("ai_engine.__main__.run_production_server") as mock_prod_server:
                with patch("ai_engine.__main__.Config") as mock_config_class:
                    mock_config = Mock()
                    mock_config_class.return_value = mock_config

                    with patch(
                        "ai_engine.__main__.logging.basicConfig"
                    ) as mock_logging:
                        with patch(
                            "ai_engine.__main__.logging.FileHandler"
                        ) as mock_file_handler:
                            __main__.main()

                            # Check that logging was configured
                            mock_logging.assert_called_once()
                            # In production mode, should not have file handler
                            mock_file_handler.assert_not_called()

    def test_main_function_version_info(self):
        """Test that version information is logged."""
        with patch("sys.argv", ["__main__.py"]):
            with patch("ai_engine.__main__.run_development_server") as mock_dev_server:
                with patch("ai_engine.__main__.Config") as mock_config_class:
                    mock_config = Mock()
                    mock_config_class.return_value = mock_config

                    with patch(
                        "ai_engine.__main__.logging.getLogger"
                    ) as mock_get_logger:
                        mock_logger = Mock()
                        mock_get_logger.return_value = mock_logger

                        __main__.main()

                        # Check that version info was logged
                        mock_logger.info.assert_any_call("Version: 1.0.0")
                        mock_logger.info.assert_any_call("Starting CRONOS AI Engine...")

    def test_main_function_python_version_logging(self):
        """Test that Python version is logged."""
        with patch("sys.argv", ["__main__.py"]):
            with patch("ai_engine.__main__.run_development_server") as mock_dev_server:
                with patch("ai_engine.__main__.Config") as mock_config_class:
                    mock_config = Mock()
                    mock_config_class.return_value = mock_config

                    with patch(
                        "ai_engine.__main__.logging.getLogger"
                    ) as mock_get_logger:
                        mock_logger = Mock()
                        mock_get_logger.return_value = mock_logger

                        __main__.main()

                        # Check that Python version was logged
                        mock_logger.info.assert_any_call(f"Python: {sys.version}")

    def test_main_function_mode_logging(self):
        """Test that mode (development/production) is logged."""
        with patch("sys.argv", ["__main__.py", "--development"]):
            with patch("ai_engine.__main__.run_development_server") as mock_dev_server:
                with patch("ai_engine.__main__.Config") as mock_config_class:
                    mock_config = Mock()
                    mock_config_class.return_value = mock_config

                    with patch(
                        "ai_engine.__main__.logging.getLogger"
                    ) as mock_get_logger:
                        mock_logger = Mock()
                        mock_get_logger.return_value = mock_logger

                        __main__.main()

                        # Check that mode was logged
                        mock_logger.info.assert_any_call("Mode: Development")
