"""
Tests for Observability Module

Tests cover:
- Metrics collection
- Distributed tracing
- Structured logging
- Health checks
- Audit logging
"""

import pytest
import time
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from ai_engine.observability import (
    # Metrics
    MetricsCollector,
    Counter,
    Gauge,
    Histogram,
    Summary,
    MetricLabels,
    get_metrics_collector,
    # Tracing
    TracingProvider,
    Span,
    SpanContext,
    SpanKind,
    SpanStatus,
    trace_method,
    get_tracer,
    # Logging
    StructuredLogger,
    LogLevel,
    LogContext,
    get_logger,
    configure_logging,
    # Health
    HealthRegistry,
    HealthCheck,
    HealthStatus,
    LivenessCheck,
    ReadinessCheck,
    get_health_registry,
    # Audit
    AuditLogger,
    AuditEvent,
    AuditCategory,
    AuditSeverity,
    AuditOutcome,
    AuditActor,
    AuditResource,
    get_audit_logger,
)


class TestMetricsCollector:
    """Tests for metrics collection."""

    def test_create_collector(self):
        """Test metrics collector creation."""
        collector = MetricsCollector(namespace="test")
        assert collector is not None

    def test_counter_increment(self):
        """Test counter increment."""
        collector = MetricsCollector(namespace="test", include_system_metrics=False)
        counter = collector.counter("requests_total", "Total requests")

        counter.inc()
        assert counter.get() == 1.0

        counter.inc(value=5.0)
        assert counter.get() == 6.0

    def test_counter_with_labels(self):
        """Test counter with labels."""
        collector = MetricsCollector(namespace="test", include_system_metrics=False)
        counter = collector.counter(
            "http_requests_total",
            "Total HTTP requests",
            ["method", "status"],
        )

        counter.inc({"method": "GET", "status": "200"})
        counter.inc({"method": "POST", "status": "200"}, 2)
        counter.inc({"method": "GET", "status": "500"})

        assert counter.get({"method": "GET", "status": "200"}) == 1.0
        assert counter.get({"method": "POST", "status": "200"}) == 2.0
        assert counter.get({"method": "GET", "status": "500"}) == 1.0

    def test_gauge_operations(self):
        """Test gauge operations."""
        collector = MetricsCollector(namespace="test", include_system_metrics=False)
        gauge = collector.gauge("temperature", "Temperature")

        gauge.set(25.5)
        assert gauge.get() == 25.5

        gauge.inc(value=2.0)
        assert gauge.get() == 27.5

        gauge.dec(value=5.0)
        assert gauge.get() == 22.5

    def test_histogram_observe(self):
        """Test histogram observations."""
        collector = MetricsCollector(namespace="test", include_system_metrics=False)
        histogram = collector.histogram(
            "request_duration",
            "Request duration",
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0),
        )

        histogram.observe(0.025)
        histogram.observe(0.15)
        histogram.observe(0.75)

        values = histogram.collect()
        assert len(values) > 0

    def test_histogram_timer(self):
        """Test histogram timer context manager."""
        collector = MetricsCollector(namespace="test", include_system_metrics=False)
        histogram = collector.histogram("operation_duration", "Operation duration")

        with histogram.time():
            time.sleep(0.01)

        values = histogram.collect()
        assert any(v.value > 0 for v in values)

    def test_summary_quantiles(self):
        """Test summary quantile calculation."""
        collector = MetricsCollector(namespace="test", include_system_metrics=False)
        summary = collector.summary(
            "response_size",
            "Response size",
            quantiles=(0.5, 0.9, 0.99),
        )

        for i in range(100):
            summary.observe(i * 10)

        values = summary.collect()
        assert len(values) > 0

    def test_prometheus_export(self):
        """Test Prometheus format export."""
        collector = MetricsCollector(namespace="test", include_system_metrics=False)
        counter = collector.counter("test_counter", "Test counter")
        counter.inc()

        output = collector.export_prometheus()

        assert "# HELP test_test_counter Test counter" in output
        assert "# TYPE test_test_counter counter" in output
        assert "test_test_counter" in output


class TestTracingProvider:
    """Tests for distributed tracing."""

    def test_create_tracer(self):
        """Test tracer creation."""
        tracer = TracingProvider("test-service")
        assert tracer is not None

    def test_start_span(self):
        """Test span creation."""
        tracer = TracingProvider("test-service")
        span = tracer.start_span("test-operation")

        assert span is not None
        assert span.name == "test-operation"
        assert span.context.trace_id is not None
        assert span.context.span_id is not None

    def test_span_attributes(self):
        """Test span attributes."""
        tracer = TracingProvider("test-service")
        span = tracer.start_span("test-operation")

        span.set_attribute("key", "value")
        span.set_attributes({"key2": "value2", "key3": 123})

        assert span.attributes["key"] == "value"
        assert span.attributes["key2"] == "value2"
        assert span.attributes["key3"] == 123

    def test_span_events(self):
        """Test span events."""
        tracer = TracingProvider("test-service")
        span = tracer.start_span("test-operation")

        span.add_event("event1", {"detail": "value"})

        assert len(span.events) == 1
        assert span.events[0].name == "event1"

    def test_span_context_manager(self):
        """Test span as context manager."""
        tracer = TracingProvider("test-service")

        with tracer.span("test-operation") as span:
            span.set_attribute("inside", True)
            time.sleep(0.01)

        assert span._ended
        assert span.duration_ms >= 10

    def test_nested_spans(self):
        """Test nested span hierarchy."""
        tracer = TracingProvider("test-service")

        with tracer.span("parent") as parent:
            with tracer.span("child") as child:
                pass

            # Child should have same trace_id
            assert child.context.trace_id == parent.context.trace_id
            # Child should have parent as parent_span_id
            assert child.context.parent_span_id == parent.context.span_id

    def test_span_exception(self):
        """Test span exception recording."""
        tracer = TracingProvider("test-service")

        try:
            with tracer.span("failing-operation") as span:
                raise ValueError("Test error")
        except ValueError:
            pass

        assert span.status == SpanStatus.ERROR
        assert len(span.events) > 0
        assert span.events[0].name == "exception"

    def test_trace_method_decorator(self):
        """Test trace_method decorator."""

        @trace_method(name="decorated_function")
        def my_function():
            return "result"

        result = my_function()
        assert result == "result"

    def test_context_propagation(self):
        """Test context injection and extraction."""
        tracer = TracingProvider("test-service")

        with tracer.span("source-operation") as span:
            headers = {}
            tracer.inject_context(headers)

            assert "traceparent" in headers
            assert span.context.trace_id in headers["traceparent"]

            # Extract context
            extracted = tracer.extract_context(headers)
            assert extracted is not None
            assert extracted.trace_id == span.context.trace_id


class TestStructuredLogger:
    """Tests for structured logging."""

    def test_create_logger(self):
        """Test logger creation."""
        logger = StructuredLogger("test")
        assert logger is not None

    def test_log_levels(self):
        """Test different log levels."""
        logger = StructuredLogger("test")

        # These should not raise
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")

    def test_context_propagation(self):
        """Test context propagation."""
        logger = StructuredLogger("test")

        with logger.with_context(correlation_id="test-123") as ctx_logger:
            assert ctx_logger.context.correlation_id == "test-123"

    def test_log_transaction(self):
        """Test transaction logging."""
        logger = StructuredLogger("test")

        # Should not raise
        logger.log_transaction(
            transaction_id="txn-123",
            protocol="ISO20022",
            amount=1000.00,
            currency="USD",
            status="completed",
        )

    def test_log_security_event(self):
        """Test security event logging."""
        logger = StructuredLogger("test")

        # Should not raise
        logger.log_security_event(
            event_type="access_denied",
            severity="high",
            description="Unauthorized access attempt",
            ip_address="192.168.1.1",
        )


class TestHealthRegistry:
    """Tests for health checks."""

    def test_create_registry(self):
        """Test registry creation."""
        registry = HealthRegistry()
        assert registry is not None

    def test_register_check(self):
        """Test check registration."""
        registry = HealthRegistry()

        registry.register(
            "custom_check",
            lambda: True,
            "Custom health check",
        )

        result = registry.check("custom_check")
        assert result.status == HealthStatus.HEALTHY

    def test_failing_check(self):
        """Test failing health check."""
        registry = HealthRegistry()

        registry.register(
            "failing_check",
            lambda: False,
            "Always fails",
        )

        result = registry.check("failing_check")
        assert result.status == HealthStatus.UNHEALTHY

    def test_liveness_check(self):
        """Test liveness check."""
        registry = HealthRegistry()

        registry.register_liveness(
            "basic_alive",
            lambda: True,
        )

        results = registry.check_liveness()
        assert "basic_alive" in results
        assert results["basic_alive"].status == HealthStatus.HEALTHY

    def test_readiness_check(self):
        """Test readiness check."""
        registry = HealthRegistry()

        registry.register_readiness(
            "database_ready",
            lambda: True,
        )

        results = registry.check_readiness()
        assert "database_ready" in results

    def test_overall_status_healthy(self):
        """Test overall healthy status."""
        registry = HealthRegistry()

        registry.register("check1", lambda: True, critical=True)
        registry.register("check2", lambda: True, critical=True)

        status = registry.get_overall_status()
        assert status == HealthStatus.HEALTHY

    def test_overall_status_unhealthy(self):
        """Test overall unhealthy status."""
        registry = HealthRegistry()

        registry.register("check1", lambda: True, critical=True)
        registry.register("check2", lambda: False, critical=True)

        status = registry.get_overall_status()
        assert status == HealthStatus.UNHEALTHY

    def test_health_response(self):
        """Test health response format."""
        registry = HealthRegistry()

        response = registry.get_health_response()

        assert "status" in response
        assert "timestamp" in response
        assert "checks" in response


class TestAuditLogger:
    """Tests for audit logging."""

    def test_create_audit_logger(self):
        """Test audit logger creation."""
        logger = AuditLogger("test-service")
        assert logger is not None

    def test_log_event(self):
        """Test basic event logging."""
        logger = AuditLogger("test-service")

        actor = AuditActor(actor_id="user-123", username="testuser")

        event = logger.log(
            category=AuditCategory.DATA_ACCESS,
            action="read_data",
            outcome=AuditOutcome.SUCCESS,
            actor=actor,
            description="User accessed data",
        )

        assert event is not None
        assert event.category == AuditCategory.DATA_ACCESS
        assert event.outcome == AuditOutcome.SUCCESS

    def test_tamper_detection(self):
        """Test tamper-evident logging."""
        logger = AuditLogger("test-service", enable_tamper_detection=True)
        actor = AuditActor(actor_id="user-123")

        # Log multiple events
        for i in range(5):
            logger.log(
                category=AuditCategory.DATA_ACCESS,
                action=f"action_{i}",
                outcome=AuditOutcome.SUCCESS,
                actor=actor,
                description=f"Event {i}",
            )

        # Verify integrity
        result = logger.verify_integrity()
        assert result["valid"] is True
        assert result["events_checked"] == 5

    def test_authentication_logging(self):
        """Test authentication logging."""
        logger = AuditLogger("test-service")
        actor = AuditActor(
            actor_id="user-123",
            username="testuser",
            ip_address="192.168.1.1",
        )

        event = logger.log_authentication(
            actor=actor,
            outcome=AuditOutcome.SUCCESS,
            method="password",
        )

        assert event.category == AuditCategory.AUTHENTICATION

    def test_key_management_logging(self):
        """Test key management logging."""
        logger = AuditLogger("test-service")
        actor = AuditActor(actor_id="system", actor_type="service")

        event = logger.log_key_management(
            actor=actor,
            action="generate",
            key_id="key-123",
            key_type="AES-256",
        )

        assert event.category == AuditCategory.KEY_MANAGEMENT
        assert event.resource is not None
        assert event.resource.resource_id == "key-123"

    def test_query_events(self):
        """Test event querying."""
        logger = AuditLogger("test-service")
        actor = AuditActor(actor_id="user-123")

        # Log events with different categories
        logger.log(
            category=AuditCategory.DATA_ACCESS,
            action="read",
            outcome=AuditOutcome.SUCCESS,
            actor=actor,
            description="Read data",
        )
        logger.log(
            category=AuditCategory.SECURITY,
            action="alert",
            outcome=AuditOutcome.SUCCESS,
            actor=actor,
            description="Security alert",
        )

        # Query by category
        data_events = logger.get_events(category=AuditCategory.DATA_ACCESS)
        assert len(data_events) >= 1

    def test_compliance_report(self):
        """Test compliance report generation."""
        logger = AuditLogger("test-service")
        actor = AuditActor(actor_id="user-123")

        # Log some events
        for _ in range(10):
            logger.log(
                category=AuditCategory.DATA_ACCESS,
                action="read",
                outcome=AuditOutcome.SUCCESS,
                actor=actor,
                description="Data access",
            )

        # Generate report
        report = logger.generate_compliance_report(
            framework="PCI-DSS",
            start_date=datetime.utcnow() - timedelta(days=1),
            end_date=datetime.utcnow() + timedelta(days=1),
        )

        assert report["framework"] == "PCI-DSS"
        assert report["summary"]["total_events"] >= 10


class TestIntegration:
    """Integration tests for observability components."""

    def test_metrics_with_tracing(self):
        """Test metrics integrated with tracing."""
        collector = MetricsCollector(namespace="test", include_system_metrics=False)
        tracer = TracingProvider("test-service")

        counter = collector.counter("traced_operations", "Traced operations")
        histogram = collector.histogram("traced_duration", "Traced duration")

        with tracer.span("monitored-operation") as span:
            counter.inc()
            with histogram.time():
                time.sleep(0.01)

        assert counter.get() == 1.0

    def test_logging_with_tracing(self):
        """Test logging integrated with tracing."""
        tracer = TracingProvider("test-service")
        logger = StructuredLogger("test")

        with tracer.span("logged-operation") as span:
            with logger.with_context(
                trace_id=span.context.trace_id,
                span_id=span.context.span_id,
            ):
                logger.info("Operation completed")

    def test_audit_with_health(self):
        """Test audit logging with health checks."""
        audit = AuditLogger("test-service")
        health = HealthRegistry()

        # Register audit check
        health.register(
            "audit_integrity",
            lambda: audit.verify_integrity()["valid"],
            "Audit log integrity check",
        )

        # Log some events
        actor = AuditActor(actor_id="system")
        for _ in range(5):
            audit.log(
                category=AuditCategory.SYSTEM,
                action="heartbeat",
                outcome=AuditOutcome.SUCCESS,
                actor=actor,
                description="System heartbeat",
            )

        # Check health
        result = health.check("audit_integrity")
        assert result.status == HealthStatus.HEALTHY

    def test_full_observability_stack(self):
        """Test all observability components together."""
        # Setup
        collector = MetricsCollector(namespace="full_test", include_system_metrics=False)
        tracer = TracingProvider("full-test-service")
        logger = StructuredLogger("full-test")
        audit = AuditLogger("full-test-service")
        health = HealthRegistry()

        # Register health checks
        health.register("metrics", lambda: True, "Metrics check")
        health.register("tracing", lambda: True, "Tracing check")

        # Create metrics
        requests = collector.counter("requests", "Requests")
        duration = collector.histogram("duration", "Duration")

        # Simulate operation
        with tracer.span("full-operation") as span:
            requests.inc()
            with duration.time():
                with logger.with_context(trace_id=span.context.trace_id):
                    logger.info("Processing request")

                    # Audit the operation
                    audit.log(
                        category=AuditCategory.DATA_ACCESS,
                        action="process",
                        outcome=AuditOutcome.SUCCESS,
                        actor=AuditActor(actor_id="system"),
                        description="Processed request",
                        trace_id=span.context.trace_id,
                    )

        # Verify
        assert requests.get() == 1.0
        assert health.get_overall_status() == HealthStatus.HEALTHY
        assert audit.verify_integrity()["valid"] is True
