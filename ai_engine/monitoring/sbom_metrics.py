"""
QBITEL - SBOM Health Metrics
Prometheus metrics for SBOM generation and vulnerability tracking.
"""

import logging
from typing import Dict, Optional
from prometheus_client import Gauge, Counter, Histogram

logger = logging.getLogger(__name__)

# ============================================
# SBOM Component Metrics
# ============================================

SBOM_COMPONENTS = Gauge(
    "qbitel_sbom_total_components",
    "Total number of components in SBOM",
    ["version", "component_name", "component_type"],
)

SBOM_PACKAGES = Gauge(
    "qbitel_sbom_total_packages",
    "Total number of packages in SBOM",
    ["version", "component_name", "ecosystem"],
)

SBOM_DIRECT_DEPENDENCIES = Gauge(
    "qbitel_sbom_direct_dependencies",
    "Number of direct dependencies",
    ["version", "component_name"],
)

SBOM_TRANSITIVE_DEPENDENCIES = Gauge(
    "qbitel_sbom_transitive_dependencies",
    "Number of transitive (indirect) dependencies",
    ["version", "component_name"],
)

# ============================================
# Vulnerability Metrics
# ============================================

SBOM_VULNERABILITIES = Gauge(
    "qbitel_sbom_vulnerabilities",
    "Number of known vulnerabilities in SBOM",
    ["version", "component_name", "severity"],
)

SBOM_CRITICAL_VULNERABILITIES = Gauge(
    "qbitel_sbom_critical_vulnerabilities",
    "Number of CRITICAL severity vulnerabilities",
    ["version", "component_name"],
)

SBOM_HIGH_VULNERABILITIES = Gauge(
    "qbitel_sbom_high_vulnerabilities",
    "Number of HIGH severity vulnerabilities",
    ["version", "component_name"],
)

SBOM_MEDIUM_VULNERABILITIES = Gauge(
    "qbitel_sbom_medium_vulnerabilities",
    "Number of MEDIUM severity vulnerabilities",
    ["version", "component_name"],
)

SBOM_LOW_VULNERABILITIES = Gauge(
    "qbitel_sbom_low_vulnerabilities",
    "Number of LOW severity vulnerabilities",
    ["version", "component_name"],
)

SBOM_VULNERABILITY_AGE_DAYS = Histogram(
    "qbitel_sbom_vulnerability_age_days",
    "Age of vulnerabilities in days since publication",
    ["version", "component_name", "severity"],
    buckets=[1, 7, 30, 90, 180, 365, float("inf")],
)

# ============================================
# SBOM Generation Metrics
# ============================================

SBOM_GENERATION_TIME = Histogram(
    "qbitel_sbom_generation_seconds",
    "SBOM generation time in seconds",
    ["component_name", "format"],
    buckets=[0.5, 1, 2, 5, 10, 30, 60, 120, 300, float("inf")],
)

SBOM_GENERATION_SUCCESS = Counter(
    "qbitel_sbom_generation_success_total",
    "Total successful SBOM generations",
    ["component_name", "format"],
)

SBOM_GENERATION_ERRORS = Counter(
    "qbitel_sbom_generation_errors_total",
    "Total SBOM generation errors",
    ["component_name", "format", "error_type"],
)

SBOM_SIGNING_SUCCESS = Counter(
    "qbitel_sbom_signing_success_total",
    "Total successful SBOM signatures",
    ["component_name"],
)

SBOM_SIGNING_ERRORS = Counter(
    "qbitel_sbom_signing_errors_total",
    "Total SBOM signing errors",
    ["component_name", "error_type"],
)

# ============================================
# Dependency Freshness Metrics
# ============================================

SBOM_DEPENDENCY_AGE_DAYS = Histogram(
    "qbitel_sbom_dependency_age_days",
    "Age of dependencies in days since last update",
    ["version", "component_name", "ecosystem"],
    buckets=[30, 90, 180, 365, 730, float("inf")],
)

SBOM_OUTDATED_DEPENDENCIES = Gauge(
    "qbitel_sbom_outdated_dependencies",
    "Number of dependencies with available updates",
    ["version", "component_name", "update_type"],
)

SBOM_EOL_DEPENDENCIES = Gauge(
    "qbitel_sbom_eol_dependencies",
    "Number of end-of-life dependencies",
    ["version", "component_name"],
)

# ============================================
# License Compliance Metrics
# ============================================

SBOM_LICENSE_TYPES = Gauge(
    "qbitel_sbom_license_types",
    "Count of packages by license type",
    ["version", "component_name", "license"],
)

SBOM_LICENSE_VIOLATIONS = Gauge(
    "qbitel_sbom_license_violations",
    "Number of license policy violations",
    ["version", "component_name", "violation_type"],
)

# ============================================
# API Metrics
# ============================================

SBOM_API_REQUESTS = Counter(
    "qbitel_sbom_api_requests_total",
    "Total SBOM API requests",
    ["endpoint", "method", "status_code"],
)

SBOM_API_REQUEST_DURATION = Histogram(
    "qbitel_sbom_api_request_duration_seconds",
    "SBOM API request duration",
    ["endpoint", "method"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, float("inf")],
)

SBOM_DOWNLOADS = Counter(
    "qbitel_sbom_downloads_total",
    "Total SBOM downloads",
    ["version", "component_name", "format"],
)

# ============================================
# Helper Functions
# ============================================


class SBOMMetricsCollector:
    """Collector for SBOM-related metrics."""

    def __init__(self):
        """Initialize SBOM metrics collector."""
        self.logger = logging.getLogger(__name__)

    def record_sbom_generation(
        self,
        component_name: str,
        format: str,
        duration: float,
        success: bool,
        error_type: Optional[str] = None,
    ):
        """
        Record SBOM generation metrics.

        Args:
            component_name: Name of the component
            format: SBOM format (spdx or cyclonedx)
            duration: Generation time in seconds
            success: Whether generation succeeded
            error_type: Type of error if failed
        """
        SBOM_GENERATION_TIME.labels(
            component_name=component_name, format=format
        ).observe(duration)

        if success:
            SBOM_GENERATION_SUCCESS.labels(
                component_name=component_name, format=format
            ).inc()
        else:
            SBOM_GENERATION_ERRORS.labels(
                component_name=component_name,
                format=format,
                error_type=error_type or "unknown",
            ).inc()

    def update_vulnerability_metrics(
        self, version: str, component_name: str, vulnerabilities: Dict[str, int]
    ):
        """
        Update vulnerability metrics from scan results.

        Args:
            version: Component version
            component_name: Name of the component
            vulnerabilities: Dict with severity counts
        """
        for severity, count in vulnerabilities.items():
            SBOM_VULNERABILITIES.labels(
                version=version, component_name=component_name, severity=severity
            ).set(count)

        # Update specific severity gauges
        SBOM_CRITICAL_VULNERABILITIES.labels(
            version=version, component_name=component_name
        ).set(vulnerabilities.get("critical", 0))

        SBOM_HIGH_VULNERABILITIES.labels(
            version=version, component_name=component_name
        ).set(vulnerabilities.get("high", 0))

        SBOM_MEDIUM_VULNERABILITIES.labels(
            version=version, component_name=component_name
        ).set(vulnerabilities.get("medium", 0))

        SBOM_LOW_VULNERABILITIES.labels(
            version=version, component_name=component_name
        ).set(vulnerabilities.get("low", 0))

    def record_sbom_download(self, version: str, component_name: str, format: str):
        """
        Record SBOM download.

        Args:
            version: Component version
            component_name: Name of the component
            format: SBOM format
        """
        SBOM_DOWNLOADS.labels(
            version=version, component_name=component_name, format=format
        ).inc()

    def record_api_request(
        self, endpoint: str, method: str, status_code: int, duration: float
    ):
        """
        Record SBOM API request metrics.

        Args:
            endpoint: API endpoint
            method: HTTP method
            status_code: Response status code
            duration: Request duration in seconds
        """
        SBOM_API_REQUESTS.labels(
            endpoint=endpoint, method=method, status_code=str(status_code)
        ).inc()

        SBOM_API_REQUEST_DURATION.labels(endpoint=endpoint, method=method).observe(
            duration
        )


# Global instance
_sbom_metrics_collector: Optional[SBOMMetricsCollector] = None


def get_sbom_metrics_collector() -> SBOMMetricsCollector:
    """Get or create SBOM metrics collector instance."""
    global _sbom_metrics_collector
    if _sbom_metrics_collector is None:
        _sbom_metrics_collector = SBOMMetricsCollector()
    return _sbom_metrics_collector
