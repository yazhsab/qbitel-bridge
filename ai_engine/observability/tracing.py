"""
Distributed Tracing

OpenTelemetry-compatible distributed tracing for:
- Request flow tracking
- Cross-service correlation
- Performance analysis
- Error tracing

Supports:
- Automatic span creation
- Context propagation
- Span attributes and events
- Export to various backends
"""

import logging
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class SpanKind(Enum):
    """Types of spans."""

    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(Enum):
    """Span status codes."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class SpanContext:
    """Context for distributed tracing."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    trace_flags: int = 1  # Sampled
    trace_state: Dict[str, str] = field(default_factory=dict)

    def to_traceparent(self) -> str:
        """Convert to W3C traceparent format."""
        return f"00-{self.trace_id}-{self.span_id}-{self.trace_flags:02x}"

    @classmethod
    def from_traceparent(cls, traceparent: str) -> Optional["SpanContext"]:
        """Parse W3C traceparent format."""
        try:
            parts = traceparent.split("-")
            if len(parts) >= 4 and parts[0] == "00":
                return cls(
                    trace_id=parts[1],
                    span_id=parts[2],
                    trace_flags=int(parts[3], 16),
                )
        except Exception:
            pass
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "trace_flags": self.trace_flags,
        }


@dataclass
class SpanEvent:
    """Event within a span."""

    name: str
    timestamp: float = field(default_factory=time.time)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpanLink:
    """Link to another span."""

    context: SpanContext
    attributes: Dict[str, Any] = field(default_factory=dict)


class Span:
    """
    A trace span representing a unit of work.

    Captures:
    - Timing information
    - Attributes/tags
    - Events
    - Links to other spans
    - Status and errors
    """

    def __init__(
        self,
        name: str,
        context: SpanContext,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        parent: Optional["Span"] = None,
    ):
        self.name = name
        self.context = context
        self.kind = kind
        self.attributes = attributes or {}
        self.parent = parent

        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.status = SpanStatus.UNSET
        self.status_message = ""

        self.events: List[SpanEvent] = []
        self.links: List[SpanLink] = []

        self._ended = False

    def set_attribute(self, key: str, value: Any) -> "Span":
        """Set a span attribute."""
        self.attributes[key] = value
        return self

    def set_attributes(self, attributes: Dict[str, Any]) -> "Span":
        """Set multiple attributes."""
        self.attributes.update(attributes)
        return self

    def add_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> "Span":
        """Add an event to the span."""
        self.events.append(
            SpanEvent(
                name=name,
                attributes=attributes or {},
            )
        )
        return self

    def add_link(
        self,
        context: SpanContext,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> "Span":
        """Add a link to another span."""
        self.links.append(
            SpanLink(
                context=context,
                attributes=attributes or {},
            )
        )
        return self

    def set_status(self, status: SpanStatus, message: str = "") -> "Span":
        """Set span status."""
        self.status = status
        self.status_message = message
        return self

    def record_exception(
        self,
        exception: Exception,
        escaped: bool = False,
    ) -> "Span":
        """Record an exception event."""
        self.add_event(
            "exception",
            {
                "exception.type": type(exception).__name__,
                "exception.message": str(exception),
                "exception.escaped": escaped,
            },
        )
        if escaped:
            self.set_status(SpanStatus.ERROR, str(exception))
        return self

    def end(self, end_time: Optional[float] = None) -> None:
        """End the span."""
        if self._ended:
            return
        self.end_time = end_time or time.time()
        self._ended = True

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "context": self.context.to_dict(),
            "kind": self.kind.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "status_message": self.status_message,
            "attributes": self.attributes,
            "events": [{"name": e.name, "timestamp": e.timestamp, "attributes": e.attributes} for e in self.events],
            "links": [{"context": l.context.to_dict(), "attributes": l.attributes} for l in self.links],
        }

    def __enter__(self) -> "Span":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_val is not None:
            self.record_exception(exc_val, escaped=True)
        self.end()


class TracingProvider:
    """
    Distributed tracing provider.

    Provides:
    - Span creation and management
    - Context propagation
    - Span export
    - Sampling
    """

    def __init__(
        self,
        service_name: str,
        sample_rate: float = 1.0,
        max_spans: int = 10000,
    ):
        self.service_name = service_name
        self.sample_rate = sample_rate
        self.max_spans = max_spans

        self._spans: List[Span] = []
        self._active_spans: Dict[str, Span] = {}
        self._context_var = threading.local()
        self._lock = threading.Lock()
        self._exporters: List[Callable[[List[Span]], None]] = []

    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        parent: Optional[Union[Span, SpanContext]] = None,
    ) -> Span:
        """
        Start a new span.

        Args:
            name: Span name
            kind: Span kind
            attributes: Initial attributes
            parent: Parent span or context

        Returns:
            New Span instance
        """
        # Determine parent context
        parent_span = None
        parent_context = None

        if isinstance(parent, Span):
            parent_span = parent
            parent_context = parent.context
        elif isinstance(parent, SpanContext):
            parent_context = parent
        else:
            # Check for active span in context
            parent_span = self._get_active_span()
            if parent_span:
                parent_context = parent_span.context

        # Generate context
        if parent_context:
            context = SpanContext(
                trace_id=parent_context.trace_id,
                span_id=self._generate_span_id(),
                parent_span_id=parent_context.span_id,
            )
        else:
            context = SpanContext(
                trace_id=self._generate_trace_id(),
                span_id=self._generate_span_id(),
            )

        # Create span
        span = Span(
            name=name,
            context=context,
            kind=kind,
            attributes={
                "service.name": self.service_name,
                **(attributes or {}),
            },
            parent=parent_span,
        )

        # Track span
        with self._lock:
            self._spans.append(span)
            self._active_spans[context.span_id] = span

            # Trim old spans
            if len(self._spans) > self.max_spans:
                old_spans = self._spans[: -self.max_spans]
                self._spans = self._spans[-self.max_spans :]
                # Export old spans
                self._export_spans(old_spans)

        return span

    def end_span(self, span: Span) -> None:
        """End a span and mark for export."""
        span.end()
        with self._lock:
            self._active_spans.pop(span.context.span_id, None)

    @contextmanager
    def span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[Span, None, None]:
        """Context manager for creating and ending spans."""
        span = self.start_span(name, kind, attributes)
        self._set_active_span(span)
        try:
            yield span
            if span.status == SpanStatus.UNSET:
                span.set_status(SpanStatus.OK)
        except Exception as e:
            span.record_exception(e, escaped=True)
            raise
        finally:
            self.end_span(span)
            self._clear_active_span()

    def inject_context(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Inject trace context into headers."""
        span = self._get_active_span()
        if span:
            headers["traceparent"] = span.context.to_traceparent()
            headers["x-trace-id"] = span.context.trace_id
            headers["x-span-id"] = span.context.span_id
        return headers

    def extract_context(self, headers: Dict[str, str]) -> Optional[SpanContext]:
        """Extract trace context from headers."""
        traceparent = headers.get("traceparent")
        if traceparent:
            return SpanContext.from_traceparent(traceparent)
        return None

    def add_exporter(self, exporter: Callable[[List[Span]], None]) -> None:
        """Add a span exporter."""
        self._exporters.append(exporter)

    def flush(self) -> None:
        """Flush all pending spans to exporters."""
        with self._lock:
            ended_spans = [s for s in self._spans if s._ended]
            if ended_spans:
                self._export_spans(ended_spans)
                self._spans = [s for s in self._spans if not s._ended]

    def get_trace(self, trace_id: str) -> List[Span]:
        """Get all spans for a trace."""
        with self._lock:
            return [s for s in self._spans if s.context.trace_id == trace_id]

    def _export_spans(self, spans: List[Span]) -> None:
        """Export spans to all exporters."""
        for exporter in self._exporters:
            try:
                exporter(spans)
            except Exception as e:
                logger.error(f"Span export failed: {e}")

    def _generate_trace_id(self) -> str:
        """Generate a 32-character trace ID."""
        return uuid.uuid4().hex

    def _generate_span_id(self) -> str:
        """Generate a 16-character span ID."""
        return uuid.uuid4().hex[:16]

    def _get_active_span(self) -> Optional[Span]:
        """Get the current active span."""
        return getattr(self._context_var, "active_span", None)

    def _set_active_span(self, span: Span) -> None:
        """Set the current active span."""
        self._context_var.active_span = span

    def _clear_active_span(self) -> None:
        """Clear the current active span."""
        self._context_var.active_span = None


# Global tracer instance
_tracer: Optional[TracingProvider] = None
_tracer_lock = threading.Lock()


def get_tracer(
    service_name: str = "qbitel",
    sample_rate: float = 1.0,
) -> TracingProvider:
    """Get or create the global tracer."""
    global _tracer
    with _tracer_lock:
        if _tracer is None:
            _tracer = TracingProvider(service_name, sample_rate)
        return _tracer


def trace_method(
    name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None,
) -> Callable[[F], F]:
    """
    Decorator to trace a method.

    Args:
        name: Span name (defaults to function name)
        kind: Span kind
        attributes: Static attributes

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        span_name = name or func.__name__

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            with tracer.span(span_name, kind, attributes) as span:
                # Add function info
                span.set_attribute("code.function", func.__name__)
                span.set_attribute("code.namespace", func.__module__)
                return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


# Console exporter for debugging
def console_exporter(spans: List[Span]) -> None:
    """Export spans to console."""
    for span in spans:
        print(f"[TRACE] {span.name} ({span.duration_ms:.2f}ms) - {span.status.value}")
        if span.attributes:
            for k, v in span.attributes.items():
                print(f"  {k}: {v}")


# JSON exporter
def json_exporter(output_func: Callable[[str], None]) -> Callable[[List[Span]], None]:
    """Create a JSON exporter."""
    import json

    def exporter(spans: List[Span]) -> None:
        for span in spans:
            output_func(json.dumps(span.to_dict()))

    return exporter
