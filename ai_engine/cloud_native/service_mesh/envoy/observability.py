"""
Envoy Observability

Provides metrics, tracing, and logging integration for Envoy proxies
with Prometheus, Jaeger, and structured logging.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Metric types"""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Metric:
    """Represents a metric"""

    name: str
    metric_type: MetricType
    value: float
    labels: Dict[str, str]
    timestamp: float = 0.0


class EnvoyObservability:
    """
    Manages observability for Envoy proxies including metrics, tracing, and logging.
    """

    def __init__(self, enable_prometheus: bool = True, enable_jaeger: bool = True, enable_access_logs: bool = True):
        """Initialize observability manager"""
        self.enable_prometheus = enable_prometheus
        self.enable_jaeger = enable_jaeger
        self.enable_access_logs = enable_access_logs

        self._metrics: Dict[str, Metric] = {}

        logger.info("Initialized EnvoyObservability")

    def create_prometheus_config(self) -> Dict[str, Any]:
        """Create Prometheus metrics configuration"""
        return {
            "name": "envoy.stat_sinks.metrics_service",
            "typed_config": {
                "@type": "type.googleapis.com/envoy.config.metrics.v3.MetricsServiceConfig",
                "transport_api_version": "V3",
                "grpc_service": {"envoy_grpc": {"cluster_name": "prometheus_cluster"}},
            },
        }

    def create_jaeger_config(self, collector_endpoint: str = "jaeger-collector:14250") -> Dict[str, Any]:
        """Create Jaeger tracing configuration"""
        return {
            "http": {
                "name": "envoy.tracers.zipkin",
                "typed_config": {
                    "@type": "type.googleapis.com/envoy.config.trace.v3.ZipkinConfig",
                    "collector_cluster": "jaeger",
                    "collector_endpoint": "/api/v2/spans",
                    "collector_endpoint_version": "HTTP_JSON",
                    "shared_span_context": False,
                },
            }
        }

    def create_access_log_config(self) -> List[Dict[str, Any]]:
        """Create access log configuration"""
        return [
            {
                "name": "envoy.access_loggers.file",
                "typed_config": {
                    "@type": "type.googleapis.com/envoy.extensions.access_loggers.file.v3.FileAccessLog",
                    "path": "/dev/stdout",
                    "log_format": {
                        "json_format": {
                            "start_time": "%START_TIME%",
                            "method": "%REQ(:METHOD)%",
                            "path": "%REQ(X-ENVOY-ORIGINAL-PATH?:PATH)%",
                            "protocol": "%PROTOCOL%",
                            "response_code": "%RESPONSE_CODE%",
                            "response_flags": "%RESPONSE_FLAGS%",
                            "bytes_received": "%BYTES_RECEIVED%",
                            "bytes_sent": "%BYTES_SENT%",
                            "duration": "%DURATION%",
                            "upstream_service_time": "%RESP(X-ENVOY-UPSTREAM-SERVICE-TIME)%",
                            "x_forwarded_for": "%REQ(X-FORWARDED-FOR)%",
                            "user_agent": "%REQ(USER-AGENT)%",
                            "request_id": "%REQ(X-REQUEST-ID)%",
                            "authority": "%REQ(:AUTHORITY)%",
                            "upstream_host": "%UPSTREAM_HOST%",
                            "quantum_encrypted": "%REQ(X-QBITEL-QUANTUM-ENCRYPTED)%",
                        }
                    },
                },
            }
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get observability statistics"""
        return {
            "prometheus_enabled": self.enable_prometheus,
            "jaeger_enabled": self.enable_jaeger,
            "access_logs_enabled": self.enable_access_logs,
            "total_metrics": len(self._metrics),
        }
