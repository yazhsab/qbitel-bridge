"""
CRONOS AI - Explainability Metrics

Prometheus metrics for monitoring explainability system performance.
"""

from prometheus_client import Counter, Histogram, Gauge, Info

# Explanation generation metrics
EXPLANATION_GENERATION_COUNTER = Counter(
    "cronos_explanations_generated_total",
    "Total number of explanations generated",
    ["model_name", "explanation_method", "status"],
)

EXPLANATION_GENERATION_DURATION = Histogram(
    "cronos_explanation_generation_seconds",
    "Time taken to generate explanations",
    ["model_name", "explanation_method"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

EXPLANATION_CACHE_HITS = Counter(
    "cronos_explanation_cache_hits_total",
    "Total number of explanation cache hits",
    ["model_name"],
)

EXPLANATION_CACHE_MISSES = Counter(
    "cronos_explanation_cache_misses_total",
    "Total number of explanation cache misses",
    ["model_name"],
)

# Model drift metrics
MODEL_DRIFT_SCORE = Gauge(
    "cronos_model_drift_score",
    "Current drift score for models (0-1, higher = more drift)",
    ["model_name", "model_version"],
)

MODEL_ACCURACY = Gauge(
    "cronos_model_accuracy",
    "Current model accuracy",
    ["model_name", "model_version"],
)

MODEL_CONFIDENCE_AVG = Gauge(
    "cronos_model_confidence_average",
    "Average model confidence score",
    ["model_name", "model_version"],
)

DRIFT_ALERTS_TRIGGERED = Counter(
    "cronos_drift_alerts_triggered_total",
    "Total number of drift alerts triggered",
    ["model_name", "alert_type"],
)

# Audit trail metrics
AUDIT_TRAIL_WRITES = Counter(
    "cronos_audit_trail_writes_total",
    "Total number of audit trail records written",
    ["model_name", "compliance_framework"],
)

AUDIT_TRAIL_WRITE_ERRORS = Counter(
    "cronos_audit_trail_write_errors_total",
    "Total number of audit trail write failures",
    ["error_type"],
)

AUDIT_TRAIL_WRITE_DURATION = Histogram(
    "cronos_audit_trail_write_seconds",
    "Time taken to write audit trail records",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)

# Human review metrics
HUMAN_REVIEWS_TOTAL = Counter(
    "cronos_human_reviews_total",
    "Total number of human reviews",
    ["model_name", "review_outcome"],
)

HUMAN_OVERRIDE_RATE = Gauge(
    "cronos_human_override_rate",
    "Rate of human overrides (rolling 24h)",
    ["model_name"],
)

# Explainability system info
EXPLAINABILITY_SYSTEM_INFO = Info(
    "cronos_explainability_system",
    "Information about the explainability system",
)


def record_explanation_generated(
    model_name: str,
    explanation_method: str,
    duration_seconds: float,
    status: str = "success",
):
    """Record metrics for explanation generation."""
    EXPLANATION_GENERATION_COUNTER.labels(
        model_name=model_name,
        explanation_method=explanation_method,
        status=status,
    ).inc()

    EXPLANATION_GENERATION_DURATION.labels(
        model_name=model_name,
        explanation_method=explanation_method,
    ).observe(duration_seconds)


def record_cache_hit(model_name: str):
    """Record explanation cache hit."""
    EXPLANATION_CACHE_HITS.labels(model_name=model_name).inc()


def record_cache_miss(model_name: str):
    """Record explanation cache miss."""
    EXPLANATION_CACHE_MISSES.labels(model_name=model_name).inc()


def update_drift_metrics(
    model_name: str,
    model_version: str,
    drift_score: float,
    accuracy: float,
    avg_confidence: float,
):
    """Update model drift metrics."""
    MODEL_DRIFT_SCORE.labels(
        model_name=model_name,
        model_version=model_version,
    ).set(drift_score)

    MODEL_ACCURACY.labels(
        model_name=model_name,
        model_version=model_version,
    ).set(accuracy)

    MODEL_CONFIDENCE_AVG.labels(
        model_name=model_name,
        model_version=model_version,
    ).set(avg_confidence)


def record_drift_alert(model_name: str, alert_type: str):
    """Record drift alert."""
    DRIFT_ALERTS_TRIGGERED.labels(
        model_name=model_name,
        alert_type=alert_type,
    ).inc()


def record_audit_trail_write(
    model_name: str,
    compliance_framework: str,
    duration_seconds: float,
    success: bool = True,
):
    """Record audit trail write metrics."""
    AUDIT_TRAIL_WRITES.labels(
        model_name=model_name,
        compliance_framework=compliance_framework,
    ).inc()

    AUDIT_TRAIL_WRITE_DURATION.observe(duration_seconds)

    if not success:
        AUDIT_TRAIL_WRITE_ERRORS.labels(error_type="write_failure").inc()


def record_human_review(model_name: str, review_outcome: str):
    """Record human review."""
    HUMAN_REVIEWS_TOTAL.labels(
        model_name=model_name,
        review_outcome=review_outcome,
    ).inc()


def initialize_explainability_metrics():
    """Initialize explainability system info metrics."""
    EXPLAINABILITY_SYSTEM_INFO.info(
        {
            "version": "1.0.0",
            "shap_enabled": "true",
            "lime_enabled": "true",
            "audit_trail_enabled": "true",
            "drift_monitoring_enabled": "true",
        }
    )
