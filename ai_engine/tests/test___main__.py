"""
Tests for ai_engine.__main__ module
"""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock
from io import StringIO


class TestMain:
    """Test cases for main entry point."""

    @patch('ai_engine.__main__.run_production_server')
    @patch('ai_engine.__main__.Config')
    def test_main_production_mode(self, mock_config, mock_run_prod):
        """Test main function in production mode."""
        from ai_engine.__main__ import main
        
        # Mock config
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance
        
        # Mock sys.argv
        test_args = ['ai_engine', '--host', '127.0.0.1', '--port', '9000']
        with patch.object(sys, 'argv', test_args):
            try:
                main()
            except SystemExit:
                pass
        
        # Verify production server was called
        mock_run_prod.assert_called_once()

    @patch('ai_engine.__main__.run_development_server')
    @patch('ai_engine.__main__.Config')
    def test_main_development_mode(self, mock_config, mock_run_dev):
        """Test main function in development mode."""
        from ai_engine.__main__ import main
        
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance
        
        test_args = ['ai_engine', '--development', '--host', '0.0.0.0', '--port', '8000']
        with patch.object(sys, 'argv', test_args):
            try:
                main()
            except SystemExit:
                pass
        
        mock_run_dev.assert_called_once()

    @patch('ai_engine.__main__.run_development_server')
    @patch('ai_engine.__main__.Config')
    def test_main_with_reload(self, mock_config, mock_run_dev):
        """Test main function with reload enabled."""
        from ai_engine.__main__ import main
        
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance
        
        test_args = ['ai_engine', '--development', '--reload']
        with patch.object(sys, 'argv', test_args):
            try:
                main()
            except SystemExit:
                pass
        
        # Verify reload parameter was passed
        call_kwargs = mock_run_dev.call_args[1]
        assert call_kwargs['reload'] is True

    @patch('ai_engine.__main__.Config')
    def test_main_with_grpc(self, mock_config):
        """Test main function with gRPC enabled."""
        from ai_engine.__main__ import main
        
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance
        
        test_args = ['ai_engine', '--enable-grpc', '--grpc-port', '50052']
        with patch.object(sys, 'argv', test_args):
            with patch('ai_engine.__main__.run_production_server'):
                try:
                    main()
                except SystemExit:
                    pass
        
        # Verify config was updated
        assert mock_config_instance.enable_grpc_api is True
        assert mock_config_instance.grpc_port == 50052

    @patch('ai_engine.__main__.Config')
    def test_main_with_custom_paths(self, mock_config):
        """Test main function with custom model and data paths."""
        from ai_engine.__main__ import main
        
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance
        
        test_args = [
            'ai_engine',
            '--model-path', '/custom/models',
            '--data-path', '/custom/data'
        ]
        with patch.object(sys, 'argv', test_args):
            with patch('ai_engine.__main__.run_production_server'):
                try:
                    main()
                except SystemExit:
                    pass
        
        assert mock_config_instance.model_path == '/custom/models'
        assert mock_config_instance.data_path == '/custom/data'

    @patch('ai_engine.__main__.Config')
    def test_main_log_levels(self, mock_config):
        """Test main function with different log levels."""
        from ai_engine.__main__ import main
        
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance
        
        for log_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            test_args = ['ai_engine', '--log-level', log_level]
            with patch.object(sys, 'argv', test_args):
                with patch('ai_engine.__main__.run_production_server'):
                    with patch('ai_engine.__main__.logging.basicConfig') as mock_logging:
                        try:
                            main()
                        except SystemExit:
                            pass
                        
                        # Verify logging was configured with correct level
                        import logging
                        expected_level = getattr(logging, log_level)
                        assert mock_logging.call_args[1]['level'] == expected_level

    @patch('ai_engine.__main__.Config', side_effect=Exception("Config error"))
    def test_main_config_failure(self, mock_config):
        """Test main function handles config failures."""
        from ai_engine.__main__ import main
        
        test_args = ['ai_engine']
        with patch.object(sys, 'argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()
            
            assert exc_info.value.code == 1

    @patch('ai_engine.__main__.run_production_server', side_effect=Exception("Server error"))
    @patch('ai_engine.__main__.Config')
    def test_main_server_failure(self, mock_config, mock_run_prod):
        """Test main function handles server startup failures."""
        from ai_engine.__main__ import main
        
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance
        
        test_args = ['ai_engine']
        with patch.object(sys, 'argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()
            
            assert exc_info.value.code == 1

    @patch('ai_engine.__main__.run_production_server')
    @patch('ai_engine.__main__.Config')
    def test_main_keyboard_interrupt(self, mock_config, mock_run_prod):
        """Test main function handles keyboard interrupt gracefully."""
        from ai_engine.__main__ import main
        
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance
        mock_run_prod.side_effect = KeyboardInterrupt()
        
        test_args = ['ai_engine']
        with patch.object(sys, 'argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()
            
            assert exc_info.value.code == 0

    @patch('ai_engine.__main__.Config')
    def test_main_development_with_file_logging(self, mock_config):
        """Test main function creates log file in development mode."""
        from ai_engine.__main__ import main
        
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance
        
        test_args = ['ai_engine', '--development']
        with patch.object(sys, 'argv', test_args):
            with patch('ai_engine.__main__.run_development_server'):
                with patch('ai_engine.__main__.logging.basicConfig') as mock_logging:
                    try:
                        main()
                    except SystemExit:
                        pass
                    
                    # Verify file handler was added in development mode
                    handlers = mock_logging.call_args[1]['handlers']
                    assert len(handlers) == 2  # stdout + file

    def test_main_as_module(self):
        """Test running as module with __name__ == '__main__'."""
        # This tests the if __name__ == "__main__" block
        with patch('ai_engine.__main__.main') as mock_main:
            # Simulate module execution
            import ai_engine.__main__ as main_module
            
            # The actual execution happens when module is imported with __name__ == '__main__'
            # We just verify the structure is correct
            assert hasattr(main_module, 'main')
            assert callable(main_module.main)