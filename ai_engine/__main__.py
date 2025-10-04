"""
CRONOS AI Engine - Main Entry Point

This module provides the main entry point for running the AI Engine.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the ai_engine directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from .core.config import Config
from .api.server import run_production_server, run_development_server


def main():
    """Main entry point for CRONOS AI Engine."""
    import argparse

    parser = argparse.ArgumentParser(
        description="CRONOS AI Engine - Enterprise Protocol Discovery, Field Detection, and Anomaly Detection"
    )

    # Server configuration
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="REST API port")
    parser.add_argument("--grpc-port", type=int, default=50051, help="gRPC port")
    parser.add_argument("--enable-grpc", action="store_true", help="Enable gRPC server")

    # Environment
    parser.add_argument(
        "--development", action="store_true", help="Run in development mode"
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable hot reloading (dev only)"
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level",
    )

    # Configuration
    parser.add_argument("--config", type=str, help="Configuration file path")

    # AI Engine specific
    parser.add_argument("--model-path", type=str, help="Path to AI models directory")
    parser.add_argument("--data-path", type=str, help="Path to training data directory")

    args = parser.parse_args()

    # Setup logging - prefer stdout for containerized environments
    log_handlers = [logging.StreamHandler(sys.stdout)]

    # Only add file handler in development mode
    if args.development:
        log_handlers.append(logging.FileHandler("cronos_ai_engine.log"))

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=log_handlers,
    )

    logger = logging.getLogger(__name__)

    try:
        logger.info("Starting CRONOS AI Engine...")
        logger.info(f"Version: 1.0.0")
        logger.info(f"Python: {sys.version}")
        logger.info(f"Mode: {'Development' if args.development else 'Production'}")

        # Load configuration
        config = Config()

        # Override config with CLI arguments
        if args.host:
            config.rest_host = args.host
        if args.port:
            config.rest_port = args.port
        if args.grpc_port:
            config.grpc_port = args.grpc_port
        if args.enable_grpc:
            config.enable_grpc_api = True
        if args.model_path:
            config.model_path = args.model_path
        if args.data_path:
            config.data_path = args.data_path

        # Run appropriate server
        if args.development:
            logger.info("Running in development mode with hot reloading...")
            run_development_server(
                config=config,
                host=args.host,
                port=args.port,
                enable_grpc=args.enable_grpc,
                grpc_port=args.grpc_port,
                reload=args.reload,
            )
        else:
            logger.info("Running in production mode...")
            run_production_server(config)

    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to start CRONOS AI Engine: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
