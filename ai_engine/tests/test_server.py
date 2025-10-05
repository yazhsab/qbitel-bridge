"""
Tests for ai_engine/api/server.py - Server Runner
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from ai_engine.core.config import Config
from ai_engine.core.exceptions import AIEngineException


class TestAIEngineServer:
    """Test suite for AIEngineServer class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = Mock(spec=Config)
        config.rest_host = "0.0.0.0"
        config.rest_port = 8000
        config.grpc_port = 50051
        config.enable_rest_api = True
        config.enable_grpc_api = True
        return config

    @pytest.fixture
    def server(self, mock_config):
        """Create AIEngineServer instance."""
        from ai_engine.api.server import AIEngineServer

        return AIEngineServer(mock_config)

    def test_server_initialization(self, server, mock_config):
        """Test server initialization."""
        assert server.config == mock_config
        assert server.rest_host == "0.0.0.0"
        assert server.rest_port == 8000
        assert server.grpc_port == 50051
        assert server.enable_rest is True
        assert server.enable_grpc is True
        assert server.is_running is False

    @pytest.mark.asyncio
    @patch("ai_engine.api.server.initialize_auth")
    @patch("ai_engine.api.server.create_app")
    @patch("ai_engine.api.server.GRPCServer")
    async def test_server_initialize(
        self, mock_grpc, mock_create_app, mock_init_auth, server
    ):
        """Test server initialization."""
        mock_app = Mock()
        mock_create_app.return_value = mock_app
        mock_grpc_instance = Mock()
        mock_grpc.return_value = mock_grpc_instance

        await server.initialize()

        mock_init_auth.assert_called_once_with(server.config)
        mock_create_app.assert_called_once_with(server.config)
        assert server.rest_app == mock_app
        assert server.grpc_server == mock_grpc_instance

    @pytest.mark.asyncio
    @patch("ai_engine.api.server.initialize_auth")
    async def test_server_initialize_failure(self, mock_init_auth, server):
        """Test server initialization failure."""
        mock_init_auth.side_effect = Exception("Init failed")

        with pytest.raises(AIEngineException):
            await server.initialize()

    @pytest.mark.asyncio
    @patch("ai_engine.api.server.uvicorn.Server")
    async def test_server_start_rest_only(self, mock_uvicorn_server, server):
        """Test starting REST server only."""
        server.enable_grpc = False
        server.rest_app = Mock()

        mock_server_instance = AsyncMock()
        mock_server_instance.serve = AsyncMock()
        mock_uvicorn_server.return_value = mock_server_instance

        # Start server in background
        start_task = asyncio.create_task(server.start())
        await asyncio.sleep(0.1)

        assert server.is_running is True

        # Cancel task
        start_task.cancel()
        try:
            await start_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_server_start_no_servers_enabled(self, server):
        """Test starting with no servers enabled."""
        server.enable_rest = False
        server.enable_grpc = False

        with pytest.raises(AIEngineException, match="No servers enabled"):
            await server.start()

    @pytest.mark.asyncio
    async def test_server_start_already_running(self, server):
        """Test starting server when already running."""
        server.is_running = True

        await server.start()  # Should return without error

    @pytest.mark.asyncio
    async def test_server_stop(self, server):
        """Test stopping server."""
        server.is_running = True
        server.grpc_server = AsyncMock()
        server.grpc_server.stop = AsyncMock()
        server.rest_server_task = AsyncMock()
        server.grpc_server_task = AsyncMock()

        await server.stop()

        assert server.is_running is False
        server.grpc_server.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_server_stop_not_running(self, server):
        """Test stopping server when not running."""
        server.is_running = False

        await server.stop()  # Should return without error

    @pytest.mark.asyncio
    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_server_restart(self, mock_sleep, server):
        """Test restarting server."""
        server.stop = AsyncMock()
        server.start = AsyncMock()

        await server.restart()

        server.stop.assert_called_once()
        mock_sleep.assert_called_once_with(2)
        server.start.assert_called_once()

    def test_server_get_status(self, server):
        """Test getting server status."""
        server.is_running = True
        server.rest_server_task = Mock()
        server.rest_server_task.done.return_value = False
        server.grpc_server_task = Mock()
        server.grpc_server_task.done.return_value = False

        status = server.get_server_status()

        assert status["is_running"] is True
        assert status["rest_api"]["enabled"] is True
        assert status["rest_api"]["running"] is True
        assert status["grpc_api"]["enabled"] is True
        assert status["grpc_api"]["running"] is True


class TestServerManager:
    """Test suite for ServerManager class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = Mock(spec=Config)
        return config

    @pytest.fixture
    def manager(self, mock_config):
        """Create ServerManager instance."""
        from ai_engine.api.server import ServerManager

        return ServerManager(mock_config)

    def test_manager_initialization(self, manager, mock_config):
        """Test manager initialization."""
        assert manager.config == mock_config
        assert manager.server is None
        assert manager.shutdown_event is not None

    @pytest.mark.asyncio
    @patch("ai_engine.api.server.AIEngineServer")
    async def test_manager_run(self, mock_server_class, manager):
        """Test manager run."""
        mock_server = AsyncMock()
        mock_server.initialize = AsyncMock()
        mock_server.start = AsyncMock()
        mock_server.stop = AsyncMock()
        mock_server_class.return_value = mock_server

        # Run manager in background
        run_task = asyncio.create_task(manager.run())
        await asyncio.sleep(0.1)

        # Trigger shutdown
        manager.shutdown_event.set()
        await asyncio.sleep(0.1)

        # Cancel task
        run_task.cancel()
        try:
            await run_task
        except asyncio.CancelledError:
            pass

        mock_server.initialize.assert_called_once()

    @pytest.mark.asyncio
    @patch("ai_engine.api.server.AIEngineServer")
    async def test_manager_run_keyboard_interrupt(self, mock_server_class, manager):
        """Test manager handles keyboard interrupt."""
        mock_server = AsyncMock()
        mock_server.initialize = AsyncMock()
        mock_server.start = AsyncMock(side_effect=KeyboardInterrupt)
        mock_server.stop = AsyncMock()
        mock_server_class.return_value = mock_server

        # Should handle KeyboardInterrupt gracefully
        await manager.run()

    @pytest.mark.asyncio
    async def test_manager_restart_server(self, manager):
        """Test manager restart server."""
        manager.server = AsyncMock()
        manager.server.restart = AsyncMock()

        await manager._restart_server()

        manager.server.restart.assert_called_once()


class TestDevelopmentServer:
    """Test suite for development server functions."""

    @pytest.mark.asyncio
    @patch("ai_engine.api.server.uvicorn.run")
    @patch("ai_engine.api.server.create_app")
    def test_run_development_server_rest_only(self, mock_create_app, mock_uvicorn_run):
        """Test running development server with REST only."""
        from ai_engine.api.server import run_development_server

        mock_app = Mock()
        mock_create_app.return_value = mock_app
        mock_config = Mock(spec=Config)

        run_development_server(
            config=mock_config,
            host="127.0.0.1",
            port=8000,
            enable_grpc=False,
            reload=True,
        )

        mock_create_app.assert_called_once_with(mock_config)
        mock_uvicorn_run.assert_called_once()

    @pytest.mark.asyncio
    @patch("ai_engine.api.server.AIEngineServer")
    @patch("asyncio.run")
    def test_run_development_server_with_grpc(
        self, mock_asyncio_run, mock_server_class
    ):
        """Test running development server with gRPC."""
        from ai_engine.api.server import run_development_server

        mock_config = Mock(spec=Config)

        run_development_server(
            config=mock_config,
            host="127.0.0.1",
            port=8000,
            enable_grpc=True,
            grpc_port=50051,
            reload=False,
        )

        mock_asyncio_run.assert_called_once()

    def test_run_development_server_default_config(self):
        """Test running development server with default config."""
        from ai_engine.api.server import run_development_server

        with patch("ai_engine.api.server.Config") as mock_config_class:
            with patch("ai_engine.api.server.create_app"):
                with patch("ai_engine.api.server.uvicorn.run"):
                    run_development_server()

                    mock_config_class.assert_called_once()


class TestProductionServer:
    """Test suite for production server functions."""

    @pytest.mark.asyncio
    @patch("ai_engine.api.server.ServerManager")
    @patch("asyncio.run")
    def test_run_production_server(self, mock_asyncio_run, mock_manager_class):
        """Test running production server."""
        from ai_engine.api.server import run_production_server

        mock_config = Mock(spec=Config)
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager

        run_production_server(mock_config)

        mock_manager_class.assert_called_once_with(mock_config)
        mock_asyncio_run.assert_called_once()


class TestMainCLI:
    """Test suite for main CLI function."""

    @patch("ai_engine.api.server.run_production_server")
    @patch("ai_engine.api.server.Config")
    def test_main_production_mode(self, mock_config, mock_run_prod):
        """Test main CLI in production mode."""
        from ai_engine.api.server import main

        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        test_args = ["server", "--host", "0.0.0.0", "--port", "8080"]
        with patch("sys.argv", test_args):
            try:
                main()
            except SystemExit:
                pass

        mock_run_prod.assert_called_once()

    @patch("ai_engine.api.server.run_development_server")
    @patch("ai_engine.api.server.Config")
    def test_main_development_mode(self, mock_config, mock_run_dev):
        """Test main CLI in development mode."""
        from ai_engine.api.server import main

        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        test_args = ["server", "--development", "--reload"]
        with patch("sys.argv", test_args):
            try:
                main()
            except SystemExit:
                pass

        mock_run_dev.assert_called_once()

    @patch("ai_engine.api.server.Config")
    def test_main_keyboard_interrupt(self, mock_config):
        """Test main CLI handles keyboard interrupt."""
        from ai_engine.api.server import main

        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        test_args = ["server"]
        with patch("sys.argv", test_args):
            with patch(
                "ai_engine.api.server.run_production_server",
                side_effect=KeyboardInterrupt,
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main()

                assert exc_info.value.code == 0

    @patch("ai_engine.api.server.Config")
    def test_main_exception_handling(self, mock_config):
        """Test main CLI handles exceptions."""
        from ai_engine.api.server import main

        mock_config.side_effect = Exception("Config error")

        test_args = ["server"]
        with patch("sys.argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 1

    @patch("ai_engine.api.server.run_production_server")
    @patch("ai_engine.api.server.Config")
    def test_main_with_all_options(self, mock_config, mock_run_prod):
        """Test main CLI with all options."""
        from ai_engine.api.server import main

        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        test_args = [
            "server",
            "--host",
            "192.168.1.1",
            "--port",
            "9000",
            "--grpc-port",
            "50052",
            "--enable-grpc",
            "--log-level",
            "DEBUG",
        ]
        with patch("sys.argv", test_args):
            try:
                main()
            except SystemExit:
                pass

        assert mock_config_instance.rest_host == "192.168.1.1"
        assert mock_config_instance.rest_port == 9000
        assert mock_config_instance.grpc_port == 50052
        assert mock_config_instance.enable_grpc_api is True
