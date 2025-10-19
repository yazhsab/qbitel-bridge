"""
CRONOS AI - AI Explainability Module

This module provides explainability and interpretability for AI/ML models
used in CRONOS AI, meeting regulatory requirements for transparent AI systems.

Key Features:
- SHAP-based explanations for deep learning models (CNN, LSTM)
- LIME-based explanations for LLM decisions
- Decision audit trail with immutable logging
- Model drift detection and monitoring
- Regulatory compliance reporting (EU AI Act, FDA, SOC2)

Components:
- base.py: Base explainer interfaces
- shap_explainer.py: SHAP integration for protocol classifiers
- lime_explainer.py: LIME integration for security decisions
- audit_logger.py: Immutable decision audit trail
- drift_monitor.py: Model performance drift detection
- metrics.py: Prometheus metrics for monitoring
"""

from .base import (
    BaseExplainer,
    ExplanationResult,
    ExplanationType,
    FeatureImportance,
    ExplainerRegistry,
)
from .shap_explainer import SHAPProtocolExplainer, create_background_dataset
from .lime_explainer import LIMEDecisionExplainer, LIMESecurityDecisionExplainer
from .audit_logger import DecisionAuditLogger, get_audit_logger, log_decision_async
from .drift_monitor import ModelDriftMonitor
from .metrics import (
    record_explanation_generated,
    update_drift_metrics,
    record_drift_alert,
    initialize_explainability_metrics,
)

__all__ = [
    # Base classes
    "BaseExplainer",
    "ExplanationResult",
    "ExplanationType",
    "FeatureImportance",
    "ExplainerRegistry",
    # SHAP
    "SHAPProtocolExplainer",
    "create_background_dataset",
    # LIME
    "LIMEDecisionExplainer",
    "LIMESecurityDecisionExplainer",
    # Audit logging
    "DecisionAuditLogger",
    "get_audit_logger",
    "log_decision_async",
    # Drift monitoring
    "ModelDriftMonitor",
    # Metrics
    "record_explanation_generated",
    "update_drift_metrics",
    "record_drift_alert",
    "initialize_explainability_metrics",
]

__version__ = "1.0.0"
