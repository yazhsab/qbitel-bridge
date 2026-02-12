"""
Comprehensive Unit Tests for AI Engine Server
Tests all functionality in ai_engine/api/server.py
"""

import pytest
import asyncio
import signal
import sys
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call
from fastapi import FastAPI

from ai_engine.api.server import (
    AIEngineServer,
    ServerManager,
    run_development_server,
    run_production_server,
    main,
)
from ai_engine.core.config import Config
from ai_engine.core.exceptions import AIEngineException


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = Config()
    config.rest_host = "127.0.0.1"
    config.rest_port = 8000
    config.grpc_port = 50051
    config.enable_rest_api = True
    config.enable_grpc_api = True
    return config


@pytest.fixture
def mock_rest_app():
    """Create mock REST app."""
    return Mock(spec=FastAPI)


@pytest.fixture
def mock_grpc_server():
    """Create mock gRPC server."""
    server = Mock()
    server.start = AsyncMock()
    server.stop = AsyncMock()
    server.wait_for_termination = AsyncMock()
    return server


class TestAIEngineServerInitialization:
    """Test AIEngineServer initialization."""

    def test_initialization_default(self, mock_config):
        """Test successful initialization with defaults."""
        server = AIEngineServer(mock_config)

        assert server.config == mock_config
        assert server.rest_host == "127.0.0.1"
        assert server.rest_port == 8000
        assert server.grpc_port == 50051
        assert server.enable_rest is True
        assert server.enable_grpc is True
        assert server.is_running is False
        assert server.rest_app is None
        assert server.grpc_server is None

    def test_initialization_custom_ports(self):
        """Test initialization with custom ports."""
        config = Config()
        config.rest_host = "0.0.0.0"
        config.rest_port = 9000
        config.grpc_port = 60051

        server = AIEngineServer(config)

        assert server.rest_host == "0.0.0.0"
        assert server.rest_port == 9000
        assert server.grpc_port == 60051

    def test_initialization_rest_only(self):
        """Test initialization with REST only."""
        config = Config()
        config.enable_rest_api = True
        config.enable_grpc_api = False

        server = AIEngineServer(config)

        assert server.enable_rest is True
        assert server.enable_grpc is False

    def test_initialization_grpc_only(self):
        """Test initialization with gRPC only."""
        config = Config()
        config.enable_rest_api = False
        config.enable_grpc_api = True

        server = AIEngineServer(config)

        assert server.enable_rest is False
        assert server.enable_grpc is True


class TestAIEngineServerInitialize:
    """Test server initialization."""

    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_config):
        """Test successful server initialization."""
        with (
            patch("ai_engine.api.server.initialize_auth") as mock_init_auth,
            patch("ai_engine.api.server.create_app") as mock_create_app,
            patch("ai_engine.api.server.GRPCServer") as mock_grpc_class,
        ):

            mock_create_app.return_value = Mock(spec=FastAPI)
            mock_grpc_class.return_value = Mock()

            server = AIEngineServer(mock_config)
            await server.initialize()

            mock_init_auth.assert_called_once_with(mock_config)
            mock_create_app.assert_called_once_with(mock_config)
            mock_grpc_class.assert_called_once_with(mock_config)
            assert server.rest_app is not None
            assert server.grpc_server is not None

    @pytest.mark.asyncio
    async def test_initialize_rest_only(self):
        """Test initialization with REST only."""
        config = Config()
        config.enable_rest_api = True
        config.enable_grpc_api = False

        with (
            patch("ai_engine.api.server.initialize_auth"),
            patch("ai_engine.api.server.create_app") as mock_create_app,
        ):

            mock_create_app.return_value = Mock(spec=FastAPI)

            server = AIEngineServer(config)
            await server.initialize()

            assert server.rest_app is not None
            assert server.grpc_server is None

    @pytest.mark.asyncio
    async def test_initialize_grpc_only(self):
        """Test initialization with gRPC only."""
        config = Config()
        config.enable_rest_api = False
        config.enable_grpc_api = True

        with (
            patch("ai_engine.api.server.initialize_auth"),
            patch("ai_engine.api.server.GRPCServer") as mock_grpc_class,
        ):

            mock_grpc_class.return_value = Mock()

            server = AIEngineServer(config)
            await server.initialize()

            assert server.rest_app is None
            assert server.grpc_server is not None

    @pytest.mark.asyncio
    async def test_initialize_failure(self, mock_config):
        """Test initialization failure."""
        with patch("ai_engine.api.server.initialize_auth", side_effect=Exception("Init failed")):
            server = AIEngineServer(mock_config)

            with pytest.raises(AIEngineException, match="Server initialization failed"):
                await server.initialize()


class TestAIEngineServerStart:
    """Test server start functionality."""

    @pytest.mark.asyncio
    async def test_start_already_running(self, mock_config):
        """Test starting server when already running."""
        server = AIEngineServer(mock_config)
        server.is_running = True

        await server.start()

        # Should return early without error
        assert server.is_running is True

    @pytest.mark.asyncio
    async def test_start_no_servers_enabled(self):
        """Test starting with no servers enabled."""
        config = Config()
        config.enable_rest_api = False
        config.enable_grpc_api = False

        server = AIEngineServer(config)

        with pytest.raises(AIEngineException, match="No servers enabled"):
            await server.start()

    @pytest.mark.asyncio
    async def test_start_rest_only(self, mock_config, mock_rest_app):
        """Test starting REST server only."""
        with (
            patch("ai_engine.api.server.initialize_auth"),
            patch("ai_engine.api.server.create_app", return_value=mock_rest_app),
        ):

            mock_config.enable_grpc_api = False
            server = AIEngineServer(mock_config)
            await server.initialize()

            # Mock the _run_rest_server to avoid actual server start
            server._run_rest_server = AsyncMock()

            # Start in background
            start_task = asyncio.create_task(server.start())
            await asyncio.sleep(0.1)

            assert server.is_running is True

            # Cleanup
            start_task.cancel()
            try:
                await start_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_start_failure_cleanup(self, mock_config):
        """Test start failure triggers cleanup."""
        with (
            patch("ai_engine.api.server.initialize_auth"),
            patch("ai_engine.api.server.create_app") as mock_create_app,
        ):

            mock_create_app.return_value = Mock(spec=FastAPI)

            server = AIEngineServer(mock_config)
            server.enable_grpc = False
            await server.initialize()

            # Make _run_rest_server fail
            server._run_rest_server = AsyncMock(side_effect=Exception("Start failed"))

            with pytest.raises(Exception):
                await server.start()


class TestAIEngineServerStop:
    """Test server stop functionality."""

    @pytest.mark.asyncio
    async def test_stop_not_running(self, mock_config):
        """Test stopping server when not running."""
        server = AIEngineServer(mock_config)
        server.is_running = False

        await server.stop()

        # Should return early without error
        assert server.is_running is False

    @pytest.mark.asyncio
    async def test_stop_with_grpc(self, mock_config, mock_grpc_server):
        """Test stopping server with gRPC."""
        with (
            patch("ai_engine.api.server.initialize_auth"),
            patch("ai_engine.api.server.GRPCServer", return_value=mock_grpc_server),
        ):

            mock_config.enable_rest_api = False
            server = AIEngineServer(mock_config)
            await server.initialize()

            server.is_running = True
            server.grpc_server_task = asyncio.create_task(asyncio.sleep(10))

            await server.stop()

            mock_grpc_server.stop.assert_called_once()
            assert server.is_running is False

    @pytest.mark.asyncio
    async def test_stop_with_rest(self, mock_config, mock_rest_app):
        """Test stopping server with REST."""
        with (
            patch("ai_engine.api.server.initialize_auth"),
            patch("ai_engine.api.server.create_app", return_value=mock_rest_app),
        ):

            mock_config.enable_grpc_api = False
            server = AIEngineServer(mock_config)
            await server.initialize()

            server.is_running = True
            server.rest_server_task = asyncio.create_task(asyncio.sleep(10))

            await server.stop()

            assert server.is_running is False

    @pytest.mark.asyncio
    async def test_stop_handles_exceptions(self, mock_config, mock_grpc_server):
        """Test stop handles exceptions gracefully."""
        mock_grpc_server.stop = AsyncMock(side_effect=Exception("Stop failed"))

        with (
            patch("ai_engine.api.server.initialize_auth"),
            patch("ai_engine.api.server.GRPCServer", return_value=mock_grpc_server),
        ):

            server = AIEngineServer(mock_config)
            await server.initialize()
            server.is_running = True

            # Should not raise exception
            await server.stop()


class TestAIEngineServerRestart:
    """Test server restart functionality."""

    @pytest.mark.asyncio
    async def test_restart(self, mock_config):
        """Test server restart."""
        server = AIEngineServer(mock_config)
        server.stop = AsyncMock()
        server.start = AsyncMock()

        await server.restart()

        server.stop.assert_called_once()
        server.start.assert_called_once()


class TestAIEngineServerStatus:
    """Test server status functionality."""

    def test_get_server_status(self, mock_config):
        """Test getting server status."""
        server = AIEngineServer(mock_config)
        server.is_running = True

        status = server.get_server_status()

        assert status["is_running"] is True
        assert "rest_api" in status
        assert "grpc_api" in status
        assert status["rest_api"]["enabled"] is True
        assert status["rest_api"]["host"] == "127.0.0.1"
        assert status["rest_api"]["port"] == 8000
        assert status["grpc_api"]["enabled"] is True
        assert status["grpc_api"]["port"] == 50051


class TestServerManager:
    """Test ServerManager functionality."""

    def test_initialization(self, mock_config):
        """Test ServerManager initialization."""
        manager = ServerManager(mock_config)

        assert manager.config == mock_config
        assert manager.server is None
        assert manager.shutdown_event is not None

    def test_signal_handlers_setup(self, mock_config):
        """Test signal handlers are set up."""
        with patch("signal.signal") as mock_signal:
            manager = ServerManager(mock_config)

            # Verify signal handlers were registered
            assert mock_signal.call_count >= 2  # At least SIGINT and SIGTERM

    @pytest.mark.asyncio
    async def test_run_with_shutdown(self, mock_config):
        """Test running server manager with shutdown."""
        with patch("ai_engine.api.server.AIEngineServer") as mock_server_class:
            mock_server = Mock()
            mock_server.initialize = AsyncMock()
            mock_server.start = AsyncMock()
            mock_server.stop = AsyncMock()
            mock_server_class.return_value = mock_server

            manager = ServerManager(mock_config)

            # Trigger shutdown after short delay
            async def trigger_shutdown():
                await asyncio.sleep(0.1)
                manager.shutdown_event.set()

            shutdown_task = asyncio.create_task(trigger_shutdown())

            await manager.run()

            await shutdown_task

            mock_server.initialize.assert_called_once()
            mock_server.stop.assert_called()

    @pytest.mark.asyncio
    async def test_run_handles_keyboard_interrupt(self, mock_config):
        """Test run handles KeyboardInterrupt."""
        with patch("ai_engine.api.server.AIEngineServer") as mock_server_class:
            mock_server = Mock()
            mock_server.initialize = AsyncMock()
            mock_server.start = AsyncMock(side_effect=KeyboardInterrupt())
            mock_server.stop = AsyncMock()
            mock_server_class.return_value = mock_server

            manager = ServerManager(mock_config)

            # Should not raise exception
            await manager.run()

            mock_server.stop.assert_called()

    @pytest.mark.asyncio
    async def test_run_handles_exception(self, mock_config):
        """Test run handles general exceptions."""
        with patch("ai_engine.api.server.AIEngineServer") as mock_server_class:
            mock_server = Mock()
            mock_server.initialize = AsyncMock(side_effect=Exception("Test error"))
            mock_server.stop = AsyncMock()
            mock_server_class.return_value = mock_server

            manager = ServerManager(mock_config)

            with pytest.raises(Exception, match="Test error"):
                await manager.run()

            mock_server.stop.assert_called()

    @pytest.mark.asyncio
    async def test_restart_server(self, mock_config):
        """Test restarting server."""
        manager = ServerManager(mock_config)
        manager.server = Mock()
        manager.server.restart = AsyncMock()

        await manager._restart_server()

        manager.server.restart.assert_called_once()


class TestDevelopmentServer:
    """Test development server runner."""

    def test_run_development_server_rest_only(self):
        """Test running development server with REST only."""
        with (
            patch("ai_engine.api.server.create_app") as mock_create_app,
            patch("uvicorn.run") as mock_uvicorn_run,
        ):

            mock_create_app.return_value = Mock(spec=FastAPI)

            run_development_server(host="127.0.0.1", port=8000, enable_grpc=False, reload=True)

            mock_uvicorn_run.assert_called_once()
            call_args = mock_uvicorn_run.call_args
            assert call_args[1]["host"] == "127.0.0.1"
            assert call_args[1]["port"] == 8000
            assert call_args[1]["reload"] is True

    def test_run_development_server_with_config(self):
        """Test running development server with custom config."""
        config = Config()

        with (
            patch("ai_engine.api.server.create_app") as mock_create_app,
            patch("uvicorn.run") as mock_uvicorn_run,
        ):

            mock_create_app.return_value = Mock(spec=FastAPI)

            run_development_server(config=config, host="0.0.0.0", port=9000, enable_grpc=False)

            assert config.rest_host == "0.0.0.0"
            assert config.rest_port == 9000
            mock_uvicorn_run.assert_called_once()


class TestProductionServer:
    """Test production server runner."""

    def test_run_production_server(self, mock_config):
        """Test running production server."""
        with (
            patch("ai_engine.api.server.ServerManager") as mock_manager_class,
            patch("asyncio.run") as mock_asyncio_run,
        ):

            mock_manager = Mock()
            mock_manager.run = AsyncMock()
            mock_manager_class.return_value = mock_manager

            run_production_server(mock_config)

            mock_manager_class.assert_called_once_with(mock_config)
            mock_asyncio_run.assert_called_once()


class TestMainEntryPoint:
    """Test main CLI entry point."""

    def test_main_development_mode(self):
        """Test main in development mode."""
        test_args = [
            "server.py",
            "--development",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--reload",
        ]

        with (
            patch("sys.argv", test_args),
            patch("ai_engine.api.server.run_development_server") as mock_dev_server,
        ):

            main()

            mock_dev_server.assert_called_once()

    def test_main_production_mode(self):
        """Test main in production mode."""
        test_args = ["server.py", "--host", "0.0.0.0", "--port", "8000"]

        with (
            patch("sys.argv", test_args),
            patch("ai_engine.api.server.run_production_server") as mock_prod_server,
        ):

            main()

            mock_prod_server.assert_called_once()

    def test_main_with_grpc(self):
        """Test main with gRPC enabled."""
        test_args = ["server.py", "--enable-grpc", "--grpc-port", "50051"]

        with (
            patch("sys.argv", test_args),
            patch("ai_engine.api.server.run_production_server") as mock_prod_server,
        ):

            main()

            mock_prod_server.assert_called_once()
            call_args = mock_prod_server.call_args[0][0]
            assert call_args.enable_grpc_api is True
            assert call_args.grpc_port == 50051

    def test_main_keyboard_interrupt(self):
        """Test main handles KeyboardInterrupt."""
        test_args = ["server.py"]

        with (
            patch("sys.argv", test_args),
            patch(
                "ai_engine.api.server.run_production_server",
                side_effect=KeyboardInterrupt(),
            ),
            patch("sys.exit") as mock_exit,
        ):

            main()

            mock_exit.assert_called_once_with(0)

    def test_main_exception(self):
        """Test main handles exceptions."""
        test_args = ["server.py"]

        with (
            patch("sys.argv", test_args),
            patch(
                "ai_engine.api.server.run_production_server",
                side_effect=Exception("Server error"),
            ),
            patch("sys.exit") as mock_exit,
        ):

            main()

            mock_exit.assert_called_once_with(1)

    def test_main_custom_log_level(self):
        """Test main with custom log level."""
        test_args = ["server.py", "--log-level", "DEBUG"]

        with (
            patch("sys.argv", test_args),
            patch("ai_engine.api.server.run_production_server"),
            patch("logging.basicConfig") as mock_logging,
        ):

            main()

            # Verify logging was configured
            assert mock_logging.called


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_concurrent_start_stop(self, mock_config):
        """Test concurrent start and stop operations."""
        server = AIEngineServer(mock_config)
        server.enable_rest = False
        server.enable_grpc = False

        # Should handle gracefully
        await asyncio.gather(server.stop(), server.stop(), return_exceptions=True)

    def test_server_status_with_tasks(self, mock_config):
        """Test server status with active tasks."""
        server = AIEngineServer(mock_config)
        server.rest_server_task = asyncio.create_task(asyncio.sleep(10))
        server.grpc_server_task = asyncio.create_task(asyncio.sleep(10))

        status = server.get_server_status()

        assert status["rest_api"]["running"] is True
        assert status["grpc_api"]["running"] is True

        # Cleanup
        server.rest_server_task.cancel()
        server.grpc_server_task.cancel()

    @pytest.mark.asyncio
    async def test_initialize_multiple_times(self, mock_config):
        """Test initializing server multiple times."""
        with (
            patch("ai_engine.api.server.initialize_auth"),
            patch("ai_engine.api.server.create_app") as mock_create_app,
            patch("ai_engine.api.server.GRPCServer") as mock_grpc_class,
        ):

            mock_create_app.return_value = Mock(spec=FastAPI)
            mock_grpc_class.return_value = Mock()

            server = AIEngineServer(mock_config)

            await server.initialize()
            await server.initialize()  # Second initialization

            # Should work without errors
            assert server.rest_app is not None
