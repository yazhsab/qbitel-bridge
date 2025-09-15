"""
CRONOS AI Engine - Server Runner

This module provides the main server runner for both REST and gRPC APIs.
"""

import asyncio
import logging
import signal
import sys
from typing import Optional
import uvicorn
from concurrent.futures import ThreadPoolExecutor

from ..core.config import Config
from ..core.exceptions import AIEngineException
from .rest import create_app, AIEngineAPI
from .grpc import GRPCServer
from .auth import initialize_auth


class AIEngineServer:
    """
    Main server class for CRONOS AI Engine APIs.
    
    This class manages both REST and gRPC servers and provides
    unified lifecycle management for the entire API stack.
    """
    
    def __init__(self, config: Config):
        """Initialize server."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Server instances
        self.rest_app = None
        self.grpc_server: Optional[GRPCServer] = None
        
        # Server configuration
        self.rest_host = getattr(config, 'rest_host', '0.0.0.0')
        self.rest_port = getattr(config, 'rest_port', 8000)
        self.grpc_port = getattr(config, 'grpc_port', 50051)
        
        # Enable/disable servers
        self.enable_rest = getattr(config, 'enable_rest_api', True)
        self.enable_grpc = getattr(config, 'enable_grpc_api', True)
        
        # Runtime state
        self.is_running = False
        self.rest_server_task: Optional[asyncio.Task] = None
        self.grpc_server_task: Optional[asyncio.Task] = None
        
        self.logger.info(f"AIEngineServer initialized - REST: {self.enable_rest}, gRPC: {self.enable_grpc}")
    
    async def initialize(self) -> None:
        """Initialize server components."""
        try:
            self.logger.info("Initializing AI Engine Server...")
            
            # Initialize authentication
            initialize_auth(self.config)
            
            # Initialize REST API
            if self.enable_rest:
                self.rest_app = create_app(self.config)
                self.logger.info("REST API initialized")
            
            # Initialize gRPC server
            if self.enable_grpc:
                self.grpc_server = GRPCServer(self.config)
                self.logger.info("gRPC server initialized")
            
            self.logger.info("AI Engine Server initialization completed")
            
        except Exception as e:
            self.logger.error(f"Server initialization failed: {e}")
            raise AIEngineException(f"Server initialization failed: {e}")
    
    async def start(self) -> None:
        """Start all enabled servers."""
        if self.is_running:
            self.logger.warning("Server is already running")
            return
        
        try:
            self.logger.info("Starting AI Engine Server...")
            
            # Start servers concurrently
            tasks = []
            
            if self.enable_rest and self.rest_app:
                self.rest_server_task = asyncio.create_task(self._run_rest_server())
                tasks.append(self.rest_server_task)
                self.logger.info(f"REST API server starting on {self.rest_host}:{self.rest_port}")
            
            if self.enable_grpc and self.grpc_server:
                await self.grpc_server.start()
                self.grpc_server_task = asyncio.create_task(self.grpc_server.wait_for_termination())
                tasks.append(self.grpc_server_task)
                self.logger.info(f"gRPC server starting on port {self.grpc_port}")
            
            if not tasks:
                raise AIEngineException("No servers enabled")
            
            self.is_running = True
            self.logger.info("AI Engine Server started successfully")
            
            # Wait for all servers
            await asyncio.gather(*tasks)
            
        except Exception as e:
            self.logger.error(f"Server startup failed: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop all servers."""
        if not self.is_running:
            return
        
        self.logger.info("Stopping AI Engine Server...")
        
        try:
            # Stop gRPC server
            if self.grpc_server:
                await self.grpc_server.stop()
                if self.grpc_server_task:
                    self.grpc_server_task.cancel()
                    try:
                        await self.grpc_server_task
                    except asyncio.CancelledError:
                        pass
                self.logger.info("gRPC server stopped")
            
            # Stop REST server (uvicorn doesn't need explicit stop in this context)
            if self.rest_server_task:
                self.rest_server_task.cancel()
                try:
                    await self.rest_server_task
                except asyncio.CancelledError:
                    pass
                self.logger.info("REST API server stopped")
            
            self.is_running = False
            self.logger.info("AI Engine Server stopped")
            
        except Exception as e:
            self.logger.error(f"Error during server shutdown: {e}")
    
    async def restart(self) -> None:
        """Restart all servers."""
        self.logger.info("Restarting AI Engine Server...")
        await self.stop()
        await asyncio.sleep(2)  # Brief pause
        await self.start()
    
    def get_server_status(self) -> dict:
        """Get server status information."""
        return {
            "is_running": self.is_running,
            "rest_api": {
                "enabled": self.enable_rest,
                "host": self.rest_host,
                "port": self.rest_port,
                "running": self.rest_server_task is not None and not self.rest_server_task.done()
            },
            "grpc_api": {
                "enabled": self.enable_grpc,
                "port": self.grpc_port,
                "running": self.grpc_server_task is not None and not self.grpc_server_task.done()
            }
        }
    
    async def _run_rest_server(self) -> None:
        """Run REST server using uvicorn."""
        config = uvicorn.Config(
            app=self.rest_app,
            host=self.rest_host,
            port=self.rest_port,
            log_level="info",
            access_log=True,
            loop="asyncio"
        )
        
        server = uvicorn.Server(config)
        await server.serve()


class ServerManager:
    """
    Server lifecycle manager with graceful shutdown.
    
    This class handles server startup, shutdown, and signal handling
    for production deployment scenarios.
    """
    
    def __init__(self, config: Config):
        """Initialize server manager."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.server: Optional[AIEngineServer] = None
        self.shutdown_event = asyncio.Event()
        
        # Setup signal handlers
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown...")
            self.shutdown_event.set()
        
        # Handle SIGINT (Ctrl+C) and SIGTERM
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Handle SIGHUP for restart (Unix only)
        if hasattr(signal, 'SIGHUP'):
            def restart_handler(signum, frame):
                self.logger.info("Received SIGHUP, restarting server...")
                asyncio.create_task(self._restart_server())
            
            signal.signal(signal.SIGHUP, restart_handler)
    
    async def run(self) -> None:
        """Run the server with lifecycle management."""
        try:
            self.logger.info("Starting CRONOS AI Engine Server Manager...")
            
            # Initialize and start server
            self.server = AIEngineServer(self.config)
            await self.server.initialize()
            
            # Start server in background
            server_task = asyncio.create_task(self.server.start())
            
            # Wait for shutdown signal or server completion
            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(self.shutdown_event.wait()),
                    server_task
                ],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # Graceful shutdown
            if self.server:
                await self.server.stop()
            
            self.logger.info("Server manager shutdown completed")
            
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        except Exception as e:
            self.logger.error(f"Server manager error: {e}")
            raise
        finally:
            # Cleanup
            if self.server:
                await self.server.stop()
    
    async def _restart_server(self) -> None:
        """Restart the server."""
        if self.server:
            await self.server.restart()


# Development server runner

def run_development_server(
    config: Optional[Config] = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    enable_grpc: bool = False,
    grpc_port: int = 50051,
    reload: bool = True
) -> None:
    """
    Run development server with hot reloading.
    
    Args:
        config: Configuration object
        host: Host to bind to
        port: Port to bind to
        enable_grpc: Whether to enable gRPC server
        grpc_port: gRPC port
        reload: Enable hot reloading
    """
    if config is None:
        config = Config()
    
    # Update config for development
    config.rest_host = host
    config.rest_port = port
    config.enable_grpc_api = enable_grpc
    config.grpc_port = grpc_port
    
    # Setup logging for development
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if enable_grpc:
        # Run both REST and gRPC
        async def run_both():
            server = AIEngineServer(config)
            await server.initialize()
            await server.start()
        
        asyncio.run(run_both())
    else:
        # Run only REST with uvicorn
        app = create_app(config)
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )


def run_production_server(config: Config) -> None:
    """
    Run production server with proper lifecycle management.
    
    Args:
        config: Configuration object
    """
    # Setup logging for production
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run server manager
    manager = ServerManager(config)
    asyncio.run(manager.run())


# CLI entry point

def main() -> None:
    """Main entry point for server."""
    import argparse
    
    parser = argparse.ArgumentParser(description='CRONOS AI Engine Server')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--grpc-port', type=int, default=50051, help='gRPC port')
    parser.add_argument('--enable-grpc', action='store_true', help='Enable gRPC server')
    parser.add_argument('--development', action='store_true', help='Run in development mode')
    parser.add_argument('--reload', action='store_true', help='Enable hot reloading (dev only)')
    parser.add_argument('--log-level', type=str, default='INFO', help='Log level')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Load configuration
        config = Config()
        if args.config:
            # Load from file if specified
            pass  # Would implement config file loading
        
        # Override config with CLI args
        config.rest_host = args.host
        config.rest_port = args.port
        config.grpc_port = args.grpc_port
        config.enable_grpc_api = args.enable_grpc
        
        if args.development:
            run_development_server(
                config=config,
                host=args.host,
                port=args.port,
                enable_grpc=args.enable_grpc,
                grpc_port=args.grpc_port,
                reload=args.reload
            )
        else:
            run_production_server(config)
    
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        sys.exit(0)
    except Exception as e:
        print(f"Server failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()