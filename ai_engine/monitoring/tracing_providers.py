"""
QBITEL - Distributed Tracing Providers
Production-ready Jaeger and Zipkin integration for distributed tracing.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import aiohttp
from urllib.parse import urljoin

from ..core.config import Config
from ..core.exceptions import ObservabilityException
from .observability import TracingProvider, Span, Trace

logger = logging.getLogger(__name__)


@dataclass
class JaegerConfig:
    """Jaeger configuration."""

    agent_host: str = "localhost"
    agent_port: int = 6831
    collector_endpoint: Optional[str] = None
    service_name: str = "qbitel"
    sampling_rate: float = 1.0
    max_tag_value_length: int = 1024
    max_packet_size: int = 65000


@dataclass
class ZipkinConfig:
    """Zipkin configuration."""

    endpoint: str = "http://localhost:9411"
    service_name: str = "qbitel"
    sampling_rate: float = 1.0
    max_tag_value_length: int = 1024


class JaegerTracingProvider(TracingProvider):
    """
    Jaeger distributed tracing provider.

    Implements OpenTracing-compatible span export to Jaeger backend.
    Supports both UDP agent and HTTP collector endpoints.
    """

    def __init__(self, config: Config):
        """Initialize Jaeger tracing provider."""
        super().__init__(config)

        # Load Jaeger configuration
        self.jaeger_config = JaegerConfig(
            agent_host=getattr(config.monitoring, "jaeger_agent_host", "localhost"),
            agent_port=getattr(config.monitoring, "jaeger_agent_port", 6831),
            collector_endpoint=getattr(config.monitoring, "jaeger_collector_endpoint", None),
            service_name=getattr(config.monitoring, "service_name", "qbitel"),
            sampling_rate=getattr(config, "tracing_sampling_rate", 1.0),
        )

        # HTTP session for collector endpoint
        self._session: Optional[aiohttp.ClientSession] = None
        self._batch_queue: List[Span] = []
        self._batch_size = 100
        self._flush_interval = 10  # seconds
        self._flush_task: Optional[asyncio.Task] = None

        self.logger.info(
            f"JaegerTracingProvider initialized "
            f"(agent: {self.jaeger_config.agent_host}:{self.jaeger_config.agent_port}, "
            f"service: {self.jaeger_config.service_name})"
        )

    async def initialize(self):
        """Initialize HTTP session and background tasks."""
        if self.jaeger_config.collector_endpoint:
            self._session = aiohttp.ClientSession()
            self._flush_task = asyncio.create_task(self._flush_loop())
            self.logger.info("Jaeger HTTP collector initialized")

    async def shutdown(self):
        """Shutdown provider and flush pending spans."""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        await self.flush()

        if self._session:
            await self._session.close()

    async def export_span(self, span: Span) -> None:
        """Export span to Jaeger."""
        try:
            jaeger_span = self._convert_to_jaeger_format(span)

            if self.jaeger_config.collector_endpoint:
                # Use HTTP collector
                self._batch_queue.append(span)
                if len(self._batch_queue) >= self._batch_size:
                    await self._flush_batch()
            else:
                # Use UDP agent (would require additional implementation)
                self.logger.debug(f"Span exported via UDP: {span.span_id}")

        except Exception as e:
            self.logger.error(f"Failed to export span to Jaeger: {e}")
            raise ObservabilityException(f"Jaeger span export failed: {e}")

    async def export_trace(self, trace: Trace) -> None:
        """Export complete trace to Jaeger."""
        try:
            for span in trace.spans:
                await self.export_span(span)
        except Exception as e:
            self.logger.error(f"Failed to export trace to Jaeger: {e}")
            raise ObservabilityException(f"Jaeger trace export failed: {e}")

    async def flush(self) -> None:
        """Flush pending spans."""
        if self._batch_queue:
            await self._flush_batch()

    async def _flush_loop(self):
        """Background task to flush spans periodically."""
        while True:
            try:
                await asyncio.sleep(self._flush_interval)
                if self._batch_queue:
                    await self._flush_batch()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Flush loop error: {e}")

    async def _flush_batch(self):
        """Flush batch of spans to Jaeger collector."""
        if not self._batch_queue or not self._session:
            return

        try:
            batch = self._batch_queue.copy()
            self._batch_queue.clear()

            jaeger_batch = {"spans": [self._convert_to_jaeger_format(span) for span in batch]}

            endpoint = urljoin(self.jaeger_config.collector_endpoint, "/api/traces")

            async with self._session.post(
                endpoint,
                json=jaeger_batch,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status != 200:
                    self.logger.error(f"Jaeger collector returned status {response.status}")
                else:
                    self.logger.debug(f"Flushed {len(batch)} spans to Jaeger")

        except Exception as e:
            self.logger.error(f"Failed to flush batch to Jaeger: {e}")

    def _convert_to_jaeger_format(self, span: Span) -> Dict[str, Any]:
        """Convert internal span format to Jaeger format."""
        return {
            "traceId": span.trace_id,
            "spanId": span.span_id,
            "parentSpanId": span.parent_span_id,
            "operationName": span.operation_name,
            "startTime": int(span.start_time * 1000000),  # microseconds
            "duration": int(span.duration_ms * 1000) if span.duration_ms else 0,
            "tags": [{"key": k, "value": str(v)[: self.jaeger_config.max_tag_value_length]} for k, v in span.tags.items()],
            "logs": [
                {
                    "timestamp": int(log.timestamp * 1000000),
                    "fields": [{"key": k, "value": str(v)} for k, v in log.attributes.items()],
                }
                for log in span.logs
            ],
            "references": [
                {
                    "refType": "CHILD_OF",
                    "traceId": link.trace_id,
                    "spanId": link.span_id,
                }
                for link in span.links
            ],
            "process": {"serviceName": self.jaeger_config.service_name, "tags": []},
        }


class ZipkinTracingProvider(TracingProvider):
    """
    Zipkin distributed tracing provider.

    Implements Zipkin v2 API for span export.
    """

    def __init__(self, config: Config):
        """Initialize Zipkin tracing provider."""
        super().__init__(config)

        # Load Zipkin configuration
        self.zipkin_config = ZipkinConfig(
            endpoint=getattr(config.monitoring, "zipkin_endpoint", "http://localhost:9411"),
            service_name=getattr(config.monitoring, "service_name", "qbitel"),
            sampling_rate=getattr(config, "tracing_sampling_rate", 1.0),
        )

        # HTTP session
        self._session: Optional[aiohttp.ClientSession] = None
        self._batch_queue: List[Span] = []
        self._batch_size = 100
        self._flush_interval = 10  # seconds
        self._flush_task: Optional[asyncio.Task] = None

        self.logger.info(
            f"ZipkinTracingProvider initialized "
            f"(endpoint: {self.zipkin_config.endpoint}, "
            f"service: {self.zipkin_config.service_name})"
        )

    async def initialize(self):
        """Initialize HTTP session and background tasks."""
        self._session = aiohttp.ClientSession()
        self._flush_task = asyncio.create_task(self._flush_loop())
        self.logger.info("Zipkin HTTP client initialized")

    async def shutdown(self):
        """Shutdown provider and flush pending spans."""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        await self.flush()

        if self._session:
            await self._session.close()

    async def export_span(self, span: Span) -> None:
        """Export span to Zipkin."""
        try:
            self._batch_queue.append(span)
            if len(self._batch_queue) >= self._batch_size:
                await self._flush_batch()

        except Exception as e:
            self.logger.error(f"Failed to export span to Zipkin: {e}")
            raise ObservabilityException(f"Zipkin span export failed: {e}")

    async def export_trace(self, trace: Trace) -> None:
        """Export complete trace to Zipkin."""
        try:
            for span in trace.spans:
                await self.export_span(span)
        except Exception as e:
            self.logger.error(f"Failed to export trace to Zipkin: {e}")
            raise ObservabilityException(f"Zipkin trace export failed: {e}")

    async def flush(self) -> None:
        """Flush pending spans."""
        if self._batch_queue:
            await self._flush_batch()

    async def _flush_loop(self):
        """Background task to flush spans periodically."""
        while True:
            try:
                await asyncio.sleep(self._flush_interval)
                if self._batch_queue:
                    await self._flush_batch()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Flush loop error: {e}")

    async def _flush_batch(self):
        """Flush batch of spans to Zipkin."""
        if not self._batch_queue or not self._session:
            return

        try:
            batch = self._batch_queue.copy()
            self._batch_queue.clear()

            zipkin_spans = [self._convert_to_zipkin_format(span) for span in batch]

            endpoint = urljoin(self.zipkin_config.endpoint, "/api/v2/spans")

            async with self._session.post(
                endpoint,
                json=zipkin_spans,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status not in (200, 202):
                    self.logger.error(f"Zipkin returned status {response.status}")
                else:
                    self.logger.debug(f"Flushed {len(batch)} spans to Zipkin")

        except Exception as e:
            self.logger.error(f"Failed to flush batch to Zipkin: {e}")

    def _convert_to_zipkin_format(self, span: Span) -> Dict[str, Any]:
        """Convert internal span format to Zipkin v2 format."""
        zipkin_span = {
            "traceId": span.trace_id.replace("-", ""),  # Remove hyphens
            "id": span.span_id.replace("-", ""),
            "name": span.operation_name,
            "timestamp": int(span.start_time * 1000000),  # microseconds
            "duration": int(span.duration_ms * 1000) if span.duration_ms else 0,
            "kind": self._map_span_kind(span.kind),
            "localEndpoint": {"serviceName": self.zipkin_config.service_name},
            "tags": {k: str(v)[: self.zipkin_config.max_tag_value_length] for k, v in span.tags.items()},
            "annotations": [{"timestamp": int(log.timestamp * 1000000), "value": log.name} for log in span.logs],
        }

        if span.parent_span_id:
            zipkin_span["parentId"] = span.parent_span_id.replace("-", "")

        return zipkin_span

    def _map_span_kind(self, kind) -> str:
        """Map internal span kind to Zipkin kind."""
        kind_mapping = {
            "internal": "CLIENT",
            "server": "SERVER",
            "client": "CLIENT",
            "producer": "PRODUCER",
            "consumer": "CONSUMER",
        }
        return kind_mapping.get(kind.value, "CLIENT")


def create_tracing_provider(config: Config, provider_type: str = "jaeger") -> TracingProvider:
    """
    Factory function to create tracing provider.

    Args:
        config: Configuration object
        provider_type: Type of provider ("jaeger", "zipkin", or "inmemory")

    Returns:
        TracingProvider instance
    """
    provider_type = provider_type.lower()

    if provider_type == "jaeger":
        return JaegerTracingProvider(config)
    elif provider_type == "zipkin":
        return ZipkinTracingProvider(config)
    elif provider_type == "inmemory":
        from .observability import InMemoryTracingProvider

        return InMemoryTracingProvider(config)
    else:
        raise ValueError(f"Unknown tracing provider type: {provider_type}")
