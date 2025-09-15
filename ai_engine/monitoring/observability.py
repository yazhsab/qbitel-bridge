"""
CRONOS AI Engine - Observability and Distributed Tracing

This module provides comprehensive observability capabilities including
distributed tracing, performance monitoring, and system observability.
"""

import asyncio
import time
import uuid
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager, asynccontextmanager
import logging
import json

from ..core.config import Config
from ..core.exceptions import ObservabilityException
from .logging import get_logger, TraceContext


class SpanKind(str, Enum):
    """Span kind enumeration."""
    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(str, Enum):
    """Span status enumeration."""
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class SpanEvent:
    """Span event information."""
    timestamp: float
    name: str
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpanLink:
    """Link to another span."""
    trace_id: str
    span_id: str
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """Distributed tracing span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    kind: SpanKind
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    status: SpanStatus = SpanStatus.OK
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[SpanEvent] = field(default_factory=list)
    links: List[SpanLink] = field(default_factory=list)
    
    def finish(self, status: SpanStatus = SpanStatus.OK) -> None:
        """Finish the span."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.status = status
    
    def add_tag(self, key: str, value: Any) -> None:
        """Add a tag to the span."""
        self.tags[key] = value
    
    def add_log(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add a log event to the span."""
        event = SpanEvent(
            timestamp=time.time(),
            name=name,
            attributes=attributes or {}
        )
        self.logs.append(event)
    
    def add_link(self, trace_id: str, span_id: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add a link to another span."""
        link = SpanLink(
            trace_id=trace_id,
            span_id=span_id,
            attributes=attributes or {}
        )
        self.links.append(link)
    
    def set_error(self, error: Exception) -> None:
        """Mark span as error and add error information."""
        self.status = SpanStatus.ERROR
        self.add_tag("error", True)
        self.add_tag("error.type", type(error).__name__)
        self.add_tag("error.message", str(error))
        self.add_log("error", {
            "error.type": type(error).__name__,
            "error.message": str(error),
            "error.stack": str(error.__traceback__) if error.__traceback__ else ""
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "kind": self.kind.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "tags": self.tags,
            "logs": [
                {
                    "timestamp": event.timestamp,
                    "name": event.name,
                    "attributes": event.attributes
                }
                for event in self.logs
            ],
            "links": [
                {
                    "trace_id": link.trace_id,
                    "span_id": link.span_id,
                    "attributes": link.attributes
                }
                for link in self.links
            ]
        }


@dataclass
class Trace:
    """Distributed trace containing multiple spans."""
    trace_id: str
    spans: List[Span] = field(default_factory=list)
    root_span: Optional[Span] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    
    def add_span(self, span: Span) -> None:
        """Add span to trace."""
        self.spans.append(span)
        
        if self.root_span is None or span.parent_span_id is None:
            self.root_span = span
        
        if self.start_time is None or span.start_time < self.start_time:
            self.start_time = span.start_time
        
        if span.end_time:
            if self.end_time is None or span.end_time > self.end_time:
                self.end_time = span.end_time
        
        self._calculate_duration()
    
    def _calculate_duration(self) -> None:
        """Calculate trace duration."""
        if self.start_time and self.end_time:
            self.duration_ms = (self.end_time - self.start_time) * 1000
    
    def get_span_by_id(self, span_id: str) -> Optional[Span]:
        """Get span by ID."""
        for span in self.spans:
            if span.span_id == span_id:
                return span
        return None
    
    def get_child_spans(self, parent_span_id: str) -> List[Span]:
        """Get child spans of a parent."""
        return [span for span in self.spans if span.parent_span_id == parent_span_id]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary."""
        return {
            "trace_id": self.trace_id,
            "root_span": self.root_span.to_dict() if self.root_span else None,
            "spans": [span.to_dict() for span in self.spans],
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "span_count": len(self.spans)
        }


class TracingProvider:
    """
    Base tracing provider interface.
    
    This class defines the interface for different tracing backends
    such as Jaeger, Zipkin, or custom implementations.
    """
    
    def __init__(self, config: Config):
        """Initialize tracing provider."""
        self.config = config
        self.logger = get_logger(f"{__name__}.TracingProvider")
    
    async def export_span(self, span: Span) -> None:
        """Export span to tracing backend."""
        raise NotImplementedError
    
    async def export_trace(self, trace: Trace) -> None:
        """Export complete trace to tracing backend."""
        raise NotImplementedError
    
    async def flush(self) -> None:
        """Flush any pending spans."""
        raise NotImplementedError


class InMemoryTracingProvider(TracingProvider):
    """
    In-memory tracing provider for development and testing.
    
    This provider stores traces in memory and provides methods
    for retrieving and analyzing them.
    """
    
    def __init__(self, config: Config):
        """Initialize in-memory tracing provider."""
        super().__init__(config)
        
        self.traces: Dict[str, Trace] = {}
        self.spans: Dict[str, Span] = {}
        self.max_traces = getattr(config, 'max_traces_in_memory', 1000)
        
        # Thread safety
        self._lock = threading.RLock()
        
        self.logger.info("InMemoryTracingProvider initialized")
    
    async def export_span(self, span: Span) -> None:
        """Export span to memory storage."""
        with self._lock:
            self.spans[span.span_id] = span
            
            # Add to trace or create new trace
            if span.trace_id not in self.traces:
                self.traces[span.trace_id] = Trace(trace_id=span.trace_id)
            
            self.traces[span.trace_id].add_span(span)
            
            # Limit memory usage
            if len(self.traces) > self.max_traces:
                oldest_trace_id = min(self.traces.keys(), 
                                    key=lambda tid: self.traces[tid].start_time or 0)
                self._remove_trace(oldest_trace_id)
    
    async def export_trace(self, trace: Trace) -> None:
        """Export complete trace to memory storage."""
        with self._lock:
            self.traces[trace.trace_id] = trace
            
            for span in trace.spans:
                self.spans[span.span_id] = span
    
    async def flush(self) -> None:
        """Flush operation (no-op for in-memory)."""
        pass
    
    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get trace by ID."""
        with self._lock:
            return self.traces.get(trace_id)
    
    def get_span(self, span_id: str) -> Optional[Span]:
        """Get span by ID."""
        with self._lock:
            return self.spans.get(span_id)
    
    def get_all_traces(self) -> List[Trace]:
        """Get all traces."""
        with self._lock:
            return list(self.traces.values())
    
    def search_traces(
        self,
        operation_name: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        min_duration_ms: Optional[float] = None,
        max_duration_ms: Optional[float] = None,
        limit: int = 100
    ) -> List[Trace]:
        """Search traces by criteria."""
        with self._lock:
            results = []
            
            for trace in self.traces.values():
                if len(results) >= limit:
                    break
                
                # Check operation name
                if operation_name and trace.root_span:
                    if operation_name not in trace.root_span.operation_name:
                        continue
                
                # Check duration
                if trace.duration_ms:
                    if min_duration_ms and trace.duration_ms < min_duration_ms:
                        continue
                    if max_duration_ms and trace.duration_ms > max_duration_ms:
                        continue
                
                # Check tags
                if tags and trace.root_span:
                    match = True
                    for key, value in tags.items():
                        if key not in trace.root_span.tags or trace.root_span.tags[key] != value:
                            match = False
                            break
                    if not match:
                        continue
                
                results.append(trace)
            
            return results
    
    def get_trace_statistics(self) -> Dict[str, Any]:
        """Get tracing statistics."""
        with self._lock:
            if not self.traces:
                return {}
            
            durations = [t.duration_ms for t in self.traces.values() if t.duration_ms]
            span_counts = [len(t.spans) for t in self.traces.values()]
            
            stats = {
                "total_traces": len(self.traces),
                "total_spans": len(self.spans),
                "avg_spans_per_trace": sum(span_counts) / len(span_counts) if span_counts else 0,
                "operations": {}
            }
            
            if durations:
                stats.update({
                    "avg_duration_ms": sum(durations) / len(durations),
                    "min_duration_ms": min(durations),
                    "max_duration_ms": max(durations)
                })
            
            # Operation statistics
            operation_counts = {}
            for trace in self.traces.values():
                if trace.root_span:
                    op_name = trace.root_span.operation_name
                    operation_counts[op_name] = operation_counts.get(op_name, 0) + 1
            
            stats["operations"] = operation_counts
            
            return stats
    
    def _remove_trace(self, trace_id: str) -> None:
        """Remove trace and its spans from memory."""
        if trace_id in self.traces:
            trace = self.traces[trace_id]
            
            # Remove spans
            for span in trace.spans:
                self.spans.pop(span.span_id, None)
            
            # Remove trace
            del self.traces[trace_id]


class DistributedTracing:
    """
    Main distributed tracing interface for AI Engine.
    
    This class provides the primary interface for creating and managing
    distributed traces throughout the AI Engine system.
    """
    
    def __init__(self, config: Config, provider: Optional[TracingProvider] = None):
        """Initialize distributed tracing."""
        self.config = config
        self.logger = get_logger(__name__)
        
        # Tracing provider
        self.provider = provider or InMemoryTracingProvider(config)
        
        # Active spans (thread-local)
        self._active_spans: Dict[int, List[Span]] = {}
        self._spans_lock = threading.RLock()
        
        # Sampling configuration
        self.sampling_rate = getattr(config, 'tracing_sampling_rate', 1.0)  # 100% by default
        self.enabled = getattr(config, 'enable_tracing', True)
        
        # Performance tracking
        self.trace_count = 0
        self.span_count = 0
        
        self.logger.info(f"DistributedTracing initialized (enabled: {self.enabled}, sampling: {self.sampling_rate})")
    
    def should_sample(self) -> bool:
        """Determine if trace should be sampled."""
        if not self.enabled:
            return False
        
        import random
        return random.random() <= self.sampling_rate
    
    def create_span(
        self,
        operation_name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        parent_span: Optional[Span] = None,
        trace_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None
    ) -> Span:
        """Create a new span."""
        if not self.should_sample():
            # Return a no-op span
            return self._create_noop_span(operation_name)
        
        # Generate IDs
        if trace_id is None:
            if parent_span:
                trace_id = parent_span.trace_id
            else:
                trace_id = str(uuid.uuid4())
        
        span_id = str(uuid.uuid4())
        parent_span_id = parent_span.span_id if parent_span else None
        
        # Create span
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            kind=kind,
            start_time=time.time(),
            tags=tags or {}
        )
        
        # Add to active spans
        self._add_active_span(span)
        
        # Update counters
        self.span_count += 1
        if parent_span is None:
            self.trace_count += 1
        
        self.logger.debug(f"Created span: {operation_name} (trace: {trace_id}, span: {span_id})")
        
        return span
    
    def finish_span(self, span: Span, status: SpanStatus = SpanStatus.OK) -> None:
        """Finish a span."""
        if span is None or isinstance(span, NoOpSpan):
            return
        
        span.finish(status)
        
        # Remove from active spans
        self._remove_active_span(span)
        
        # Export to provider
        asyncio.create_task(self.provider.export_span(span))
        
        self.logger.debug(f"Finished span: {span.operation_name} (duration: {span.duration_ms:.2f}ms)")
    
    def get_active_span(self) -> Optional[Span]:
        """Get current active span for this thread."""
        thread_id = threading.get_ident()
        with self._spans_lock:
            spans = self._active_spans.get(thread_id, [])
            return spans[-1] if spans else None
    
    @contextmanager
    def trace(
        self,
        operation_name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        tags: Optional[Dict[str, Any]] = None
    ):
        """Context manager for creating and managing spans."""
        parent_span = self.get_active_span()
        span = self.create_span(operation_name, kind, parent_span, tags=tags)
        
        try:
            yield span
            self.finish_span(span, SpanStatus.OK)
        except Exception as e:
            span.set_error(e)
            self.finish_span(span, SpanStatus.ERROR)
            raise
    
    @asynccontextmanager
    async def async_trace(
        self,
        operation_name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        tags: Optional[Dict[str, Any]] = None
    ):
        """Async context manager for creating and managing spans."""
        parent_span = self.get_active_span()
        span = self.create_span(operation_name, kind, parent_span, tags=tags)
        
        try:
            yield span
            self.finish_span(span, SpanStatus.OK)
        except Exception as e:
            span.set_error(e)
            self.finish_span(span, SpanStatus.ERROR)
            raise
    
    def inject_trace_context(self, span: Span) -> Dict[str, str]:
        """Inject trace context for propagation."""
        return {
            "trace-id": span.trace_id,
            "span-id": span.span_id,
            "parent-span-id": span.parent_span_id or ""
        }
    
    def extract_trace_context(self, headers: Dict[str, str]) -> Optional[TraceContext]:
        """Extract trace context from headers."""
        trace_id = headers.get("trace-id")
        span_id = headers.get("span-id")
        parent_span_id = headers.get("parent-span-id")
        
        if trace_id and span_id:
            return TraceContext(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id if parent_span_id else None
            )
        
        return None
    
    def _add_active_span(self, span: Span) -> None:
        """Add span to active spans list."""
        thread_id = threading.get_ident()
        with self._spans_lock:
            if thread_id not in self._active_spans:
                self._active_spans[thread_id] = []
            self._active_spans[thread_id].append(span)
    
    def _remove_active_span(self, span: Span) -> None:
        """Remove span from active spans list."""
        thread_id = threading.get_ident()
        with self._spans_lock:
            if thread_id in self._active_spans:
                try:
                    self._active_spans[thread_id].remove(span)
                    if not self._active_spans[thread_id]:
                        del self._active_spans[thread_id]
                except ValueError:
                    pass
    
    def _create_noop_span(self, operation_name: str) -> 'NoOpSpan':
        """Create a no-op span for non-sampled traces."""
        return NoOpSpan(operation_name)
    
    def get_tracing_statistics(self) -> Dict[str, Any]:
        """Get tracing statistics."""
        with self._spans_lock:
            active_spans_count = sum(len(spans) for spans in self._active_spans.values())
        
        stats = {
            "enabled": self.enabled,
            "sampling_rate": self.sampling_rate,
            "total_traces": self.trace_count,
            "total_spans": self.span_count,
            "active_spans": active_spans_count,
            "provider_type": type(self.provider).__name__
        }
        
        # Add provider-specific stats
        if hasattr(self.provider, 'get_trace_statistics'):
            stats["provider_stats"] = self.provider.get_trace_statistics()
        
        return stats


class NoOpSpan:
    """No-op span for non-sampled traces."""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.trace_id = "noop"
        self.span_id = "noop"
    
    def add_tag(self, key: str, value: Any) -> None:
        """No-op add tag."""
        pass
    
    def add_log(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """No-op add log."""
        pass
    
    def set_error(self, error: Exception) -> None:
        """No-op set error."""
        pass
    
    def finish(self, status: SpanStatus = SpanStatus.OK) -> None:
        """No-op finish."""
        pass


class ObservabilityManager:
    """
    Central observability manager for AI Engine.
    
    This class coordinates all observability components including
    metrics, logging, tracing, and health monitoring.
    """
    
    def __init__(self, config: Config):
        """Initialize observability manager."""
        self.config = config
        self.logger = get_logger(__name__)
        
        # Components
        self.tracing: Optional[DistributedTracing] = None
        self.health_checker = None  # Will be set by health module
        self.metrics_collector = None  # Will be set by metrics module
        
        # Observability state
        self.enabled = getattr(config, 'enable_observability', True)
        self.initialization_time = time.time()
        
        self.logger.info("ObservabilityManager initialized")
    
    async def initialize(self) -> None:
        """Initialize all observability components."""
        if not self.enabled:
            self.logger.info("Observability disabled by configuration")
            return
        
        try:
            # Initialize distributed tracing
            if getattr(self.config, 'enable_tracing', True):
                self.tracing = DistributedTracing(self.config)
                self.logger.info("Distributed tracing initialized")
            
            self.logger.info("Observability components initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize observability: {e}")
            raise ObservabilityException(f"Observability initialization failed: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown observability components."""
        try:
            if self.tracing and hasattr(self.tracing.provider, 'flush'):
                await self.tracing.provider.flush()
            
            self.logger.info("Observability components shut down")
            
        except Exception as e:
            self.logger.error(f"Error during observability shutdown: {e}")
    
    def get_observability_status(self) -> Dict[str, Any]:
        """Get overall observability status."""
        status = {
            "enabled": self.enabled,
            "uptime_seconds": time.time() - self.initialization_time,
            "components": {}
        }
        
        # Tracing status
        if self.tracing:
            status["components"]["tracing"] = {
                "enabled": True,
                "statistics": self.tracing.get_tracing_statistics()
            }
        else:
            status["components"]["tracing"] = {"enabled": False}
        
        # Health monitoring status
        if self.health_checker:
            status["components"]["health_monitoring"] = {"enabled": True}
        else:
            status["components"]["health_monitoring"] = {"enabled": False}
        
        # Metrics collection status
        if self.metrics_collector:
            status["components"]["metrics"] = {"enabled": True}
        else:
            status["components"]["metrics"] = {"enabled": False}
        
        return status
    
    def create_trace_context(self, operation_name: str, **tags) -> Optional[TraceContext]:
        """Create trace context for operation."""
        if not self.tracing:
            return None
        
        span = self.tracing.create_span(operation_name, tags=tags)
        return TraceContext(
            trace_id=span.trace_id,
            span_id=span.span_id,
            operation_name=operation_name,
            tags=tags
        )


# Global observability manager instance
_observability_manager: Optional[ObservabilityManager] = None


def initialize_observability(config: Config) -> ObservabilityManager:
    """Initialize global observability manager."""
    global _observability_manager
    
    _observability_manager = ObservabilityManager(config)
    return _observability_manager


def get_observability_manager() -> Optional[ObservabilityManager]:
    """Get global observability manager."""
    return _observability_manager


def get_tracer() -> Optional[DistributedTracing]:
    """Get global tracer instance."""
    if _observability_manager and _observability_manager.tracing:
        return _observability_manager.tracing
    return None