"""
QBITEL - OpenTelemetry Distributed Tracing
Complete distributed tracing implementation with Jaeger backend.
"""

import logging
import os
from typing import Optional, Dict, Any, Callable
from functools import wraps

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.b3 import B3MultiFormat

logger = logging.getLogger(__name__)


class TracingConfig:
    """OpenTelemetry tracing configuration."""

    def __init__(self):
        # Service information
        self.service_name = os.getenv("OTEL_SERVICE_NAME", "qbitel-engine")
        self.service_version = os.getenv("QBITEL_VERSION", "1.0.0")
        self.environment = os.getenv("QBITEL_ENVIRONMENT", "development")

        # Tracing configuration
        self.enabled = os.getenv("OTEL_TRACING_ENABLED", "true").lower() == "true"
        self.exporter_type = os.getenv(
            "OTEL_EXPORTER_TYPE", "jaeger"
        )  # jaeger, otlp, console

        # Jaeger configuration
        self.jaeger_host = os.getenv("OTEL_JAEGER_HOST", "localhost")
        self.jaeger_port = int(os.getenv("OTEL_JAEGER_PORT", "6831"))

        # OTLP configuration
        self.otlp_endpoint = os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
        )

        # Sampling
        self.sample_rate = float(os.getenv("OTEL_TRACE_SAMPLE_RATE", "1.0"))

        # Additional attributes
        self.resource_attributes = {
            "deployment.environment": self.environment,
            "service.namespace": "qbitel",
        }


def initialize_tracing(config: Optional[TracingConfig] = None, app=None) -> bool:
    """
    Initialize OpenTelemetry distributed tracing.

    Args:
        config: Tracing configuration (uses environment variables if None)
        app: FastAPI application instance (optional, for auto-instrumentation)

    Returns:
        True if initialized successfully, False otherwise
    """
    if config is None:
        config = TracingConfig()

    if not config.enabled:
        logger.info("OpenTelemetry tracing is disabled")
        return False

    try:
        # Create resource with service information
        resource = Resource.create(
            {
                SERVICE_NAME: config.service_name,
                SERVICE_VERSION: config.service_version,
                **config.resource_attributes,
            }
        )

        # Create tracer provider
        tracer_provider = TracerProvider(resource=resource)

        # Set up exporter based on configuration
        if config.exporter_type == "jaeger":
            exporter = JaegerExporter(
                agent_host_name=config.jaeger_host,
                agent_port=config.jaeger_port,
            )
            logger.info(
                f"Using Jaeger exporter: {config.jaeger_host}:{config.jaeger_port}"
            )

        elif config.exporter_type == "otlp":
            exporter = OTLPSpanExporter(
                endpoint=config.otlp_endpoint, insecure=True  # Use TLS in production
            )
            logger.info(f"Using OTLP exporter: {config.otlp_endpoint}")

        elif config.exporter_type == "console":
            exporter = ConsoleSpanExporter()
            logger.info("Using console exporter (development)")

        else:
            raise ValueError(f"Unknown exporter type: {config.exporter_type}")

        # Add span processor
        tracer_provider.add_span_processor(BatchSpanProcessor(exporter))

        # Set global tracer provider
        trace.set_tracer_provider(tracer_provider)

        # Set up B3 propagation for compatibility
        set_global_textmap(B3MultiFormat())

        # Auto-instrument libraries
        _setup_auto_instrumentation(app)

        logger.info(
            f"✅ OpenTelemetry tracing initialized successfully "
            f"(service: {config.service_name}, "
            f"exporter: {config.exporter_type}, "
            f"sample_rate: {config.sample_rate*100}%)"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to initialize OpenTelemetry tracing: {e}")
        return False


def _setup_auto_instrumentation(app=None):
    """Setup automatic instrumentation for common libraries."""
    try:
        # Instrument FastAPI
        if app:
            FastAPIInstrumentor.instrument_app(app)
            logger.info("✅ FastAPI instrumentation enabled")

        # Instrument SQLAlchemy
        try:
            SQLAlchemyInstrumentor().instrument()
            logger.info("✅ SQLAlchemy instrumentation enabled")
        except Exception as e:
            logger.debug(f"SQLAlchemy instrumentation skipped: {e}")

        # Instrument Redis
        try:
            RedisInstrumentor().instrument()
            logger.info("✅ Redis instrumentation enabled")
        except Exception as e:
            logger.debug(f"Redis instrumentation skipped: {e}")

        # Instrument requests library
        try:
            RequestsInstrumentor().instrument()
            logger.info("✅ Requests instrumentation enabled")
        except Exception as e:
            logger.debug(f"Requests instrumentation skipped: {e}")

        # Instrument logging
        try:
            LoggingInstrumentor().instrument()
            logger.info("✅ Logging instrumentation enabled")
        except Exception as e:
            logger.debug(f"Logging instrumentation skipped: {e}")

    except Exception as e:
        logger.warning(f"Some auto-instrumentation failed: {e}")


def shutdown_tracing():
    """Shutdown tracing and flush remaining spans."""
    try:
        tracer_provider = trace.get_tracer_provider()
        if hasattr(tracer_provider, "shutdown"):
            tracer_provider.shutdown()
            logger.info("OpenTelemetry tracing shutdown complete")
    except Exception as e:
        logger.error(f"Error shutting down tracing: {e}")


def get_tracer(name: str = __name__) -> trace.Tracer:
    """
    Get a tracer instance.

    Args:
        name: Tracer name (usually __name__)

    Returns:
        Tracer instance
    """
    return trace.get_tracer(name)


def trace_function(
    span_name: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None
):
    """
    Decorator to trace a function with a custom span.

    Args:
        span_name: Span name (defaults to function name)
        attributes: Additional span attributes

    Usage:
        @trace_function('process_protocol', {'protocol_type': 'tcp'})
        async def process_protocol(data):
            ...
    """

    def decorator(func: Callable) -> Callable:
        tracer = get_tracer(func.__module__)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            name = span_name or func.__name__
            with tracer.start_as_current_span(name) as span:
                # Add attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                # Add function details
                span.set_attribute("code.function", func.__name__)
                span.set_attribute("code.namespace", func.__module__)

                try:
                    result = await func(*args, **kwargs)
                    span.set_status(trace.Status(trace.StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            name = span_name or func.__name__
            with tracer.start_as_current_span(name) as span:
                # Add attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                span.set_attribute("code.function", func.__name__)
                span.set_attribute("code.namespace", func.__module__)

                try:
                    result = func(*args, **kwargs)
                    span.set_status(trace.Status(trace.StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def add_span_attribute(key: str, value: Any):
    """
    Add an attribute to the current span.

    Args:
        key: Attribute key
        value: Attribute value
    """
    try:
        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute(key, value)
    except Exception as e:
        logger.debug(f"Failed to add span attribute: {e}")


def add_span_event(name: str, attributes: Optional[Dict[str, Any]] = None):
    """
    Add an event to the current span.

    Args:
        name: Event name
        attributes: Event attributes
    """
    try:
        span = trace.get_current_span()
        if span and span.is_recording():
            span.add_event(name, attributes=attributes or {})
    except Exception as e:
        logger.debug(f"Failed to add span event: {e}")


def record_exception(exception: Exception):
    """
    Record an exception in the current span.

    Args:
        exception: Exception to record
    """
    try:
        span = trace.get_current_span()
        if span and span.is_recording():
            span.record_exception(exception)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(exception)))
    except Exception as e:
        logger.debug(f"Failed to record exception: {e}")


class TracedOperation:
    """
    Context manager for creating traced operations.

    Usage:
        with TracedOperation('database_query', {'table': 'protocols'}):
            # Your code here
            pass
    """

    def __init__(
        self, operation_name: str, attributes: Optional[Dict[str, Any]] = None
    ):
        self.operation_name = operation_name
        self.attributes = attributes or {}
        self.span = None
        self.tracer = get_tracer(__name__)

    def __enter__(self):
        self.span = self.tracer.start_span(self.operation_name)
        self.span.__enter__()

        # Add attributes
        for key, value in self.attributes.items():
            self.span.set_attribute(key, value)

        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc_val)))
            self.span.record_exception(exc_val)
        else:
            self.span.set_status(trace.Status(trace.StatusCode.OK))

        self.span.__exit__(exc_type, exc_val, exc_tb)
        return False


# Specialized tracers for common operations


def trace_database_query(query: str, params: Optional[Dict] = None):
    """
    Create a span for a database query.

    Args:
        query: SQL query
        params: Query parameters

    Returns:
        TracedOperation context manager
    """
    attributes = {
        "db.system": "postgresql",
        "db.statement": query[:200],  # Truncate long queries
    }
    if params:
        attributes["db.params"] = str(params)[:100]

    return TracedOperation("db.query", attributes)


def trace_llm_request(provider: str, model: str, prompt_length: int):
    """
    Create a span for an LLM request.

    Args:
        provider: LLM provider (openai, anthropic, etc.)
        model: Model name
        prompt_length: Length of prompt in characters

    Returns:
        TracedOperation context manager
    """
    attributes = {
        "llm.provider": provider,
        "llm.model": model,
        "llm.prompt_length": prompt_length,
    }

    return TracedOperation("llm.request", attributes)


def trace_ai_inference(model_type: str, input_size: int):
    """
    Create a span for AI model inference.

    Args:
        model_type: Type of model (cnn, lstm, transformer, etc.)
        input_size: Size of input data

    Returns:
        TracedOperation context manager
    """
    attributes = {
        "ai.model_type": model_type,
        "ai.input_size": input_size,
    }

    return TracedOperation("ai.inference", attributes)


def trace_cache_operation(operation: str, key: str, hit: Optional[bool] = None):
    """
    Create a span for a cache operation.

    Args:
        operation: Operation type (get, set, delete)
        key: Cache key
        hit: Whether it was a cache hit (for get operations)

    Returns:
        TracedOperation context manager
    """
    attributes = {
        "cache.operation": operation,
        "cache.key": key,
    }
    if hit is not None:
        attributes["cache.hit"] = hit

    return TracedOperation("cache.operation", attributes)


# Health check helper
def get_tracing_health() -> Dict[str, Any]:
    """
    Get tracing health status.

    Returns:
        Health status dict
    """
    try:
        config = TracingConfig()
        tracer_provider = trace.get_tracer_provider()

        if not config.enabled:
            return {"status": "disabled", "enabled": False}

        if tracer_provider:
            return {
                "status": "healthy",
                "enabled": True,
                "service_name": config.service_name,
                "exporter_type": config.exporter_type,
                "sample_rate": config.sample_rate,
            }
        else:
            return {
                "status": "unhealthy",
                "enabled": True,
                "error": "Tracer provider not initialized",
            }

    except Exception as e:
        return {"status": "error", "error": str(e), "enabled": False}


# Example usage for specific QBITEL operations


@trace_function("protocol_discovery", {"component": "ai_engine"})
async def trace_protocol_discovery(packet_data: bytes):
    """Example traced protocol discovery function."""
    add_span_attribute("packet_size", len(packet_data))
    add_span_event("discovery_started", {"packet_hash": hash(packet_data)})
    # Your protocol discovery logic here
    pass


@trace_function("field_detection", {"component": "ai_engine"})
async def trace_field_detection(protocol_data: Dict):
    """Example traced field detection function."""
    add_span_attribute("field_count", len(protocol_data))
    # Your field detection logic here
    pass


@trace_function("anomaly_detection", {"component": "ai_engine"})
async def trace_anomaly_detection(sequence_data: list):
    """Example traced anomaly detection function."""
    add_span_attribute("sequence_length", len(sequence_data))
    # Your anomaly detection logic here
    pass
