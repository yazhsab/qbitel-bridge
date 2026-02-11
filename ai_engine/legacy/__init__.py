"""
QBITEL Engine - Legacy System Whisperer

Enterprise-grade legacy system monitoring and predictive maintenance platform.
Combines machine learning, LLM analysis, and tribal knowledge capture for
comprehensive legacy system management.

Features:
- Enhanced anomaly detection for legacy systems
- Tribal knowledge capture and formalization
- Predictive failure analysis
- Decision support and recommendation engine
- Maintenance scheduling optimization
"""

from .enhanced_detector import EnhancedLegacySystemDetector
from .knowledge_capture import TribalKnowledgeCapture, FormalizedKnowledge
from .predictive_analytics import (
    FailurePredictor,
    PerformanceMonitor,
    MaintenanceScheduler,
)
from .decision_support import RecommendationEngine, ImpactAssessor, ActionPlanner
from .models import (
    SystemFailurePrediction,
    MaintenanceRecommendation,
    SystemBehaviorPattern,
    LegacySystemContext,
)
from .service import LegacySystemWhispererService

__all__ = [
    "EnhancedLegacySystemDetector",
    "TribalKnowledgeCapture",
    "FormalizedKnowledge",
    "FailurePredictor",
    "PerformanceMonitor",
    "MaintenanceScheduler",
    "RecommendationEngine",
    "ImpactAssessor",
    "ActionPlanner",
    "SystemFailurePrediction",
    "MaintenanceRecommendation",
    "SystemBehaviorPattern",
    "LegacySystemContext",
    "LegacySystemWhispererService",
]

__version__ = "1.0.0"
__author__ = "QBITEL Team"
__description__ = "Legacy System Whisperer - AI-powered legacy system management"
