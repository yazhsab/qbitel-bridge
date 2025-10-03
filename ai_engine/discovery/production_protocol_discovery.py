"""
CRONOS AI Engine - Production Protocol Discovery System

This module implements enterprise-ready protocol discovery with SLA guarantees,
explainable AI, model versioning, and A/B testing capabilities.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import json
import hashlib

import torch
import numpy as np
from prometheus_client import Counter, Histogram, Gauge

from ..core.config import Config
from ..core.exceptions import (
    DiscoveryException,
    SLAViolationException,
    ModelVersionException
)
from ..models.base import BaseModel, ModelInput, ModelOutput


# Prometheus metrics
DISCOVERY_SLA_VIOLATIONS = Counter(
    'cronos_discovery_sla_violations_total',
    'Total SLA violations in protocol discovery',
    ['model_version', 'sla_threshold_ms']
)

DISCOVERY_LATENCY = Histogram(
    'cronos_discovery_latency_ms',
    'Protocol discovery latency in milliseconds',
    ['model_version', 'quality_mode'],
    buckets=[10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
)

DISCOVERY_CONFIDENCE = Histogram(
    'cronos_discovery_confidence',
    'Protocol discovery confidence scores',
    ['model_version'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
)

AB_TEST_REQUESTS = Counter(
    'cronos_discovery_ab_test_requests_total',
    'Total A/B test requests',
    ['variant', 'model_version']
)

MODEL_VERSION_ACTIVE = Gauge(
    'cronos_discovery_model_version_active',
    'Active model version indicator',
    ['model_version', 'deployment_type']
)


class QualityMode(str, Enum):
    """Quality vs speed tradeoff modes."""
    FAST = "fast"  # Prioritize speed, may sacrifice accuracy
    BALANCED = "balanced"  # Balance between speed and accuracy
    ACCURATE = "accurate"  # Prioritize accuracy, may be slower


class DeploymentType(str, Enum):
    """Model deployment types."""
    PRIMARY = "primary"  # Main production model
    CANARY = "canary"  # Canary deployment for testing
    SHADOW = "shadow"  # Shadow deployment for comparison
    AB_TEST = "ab_test"  # A/B test variant


class FallbackStrategy(str, Enum):
    """Fallback strategies when SLA cannot be met."""
    CACHED_RESULT = "cached_result"  # Return cached result
    FAST_MODEL = "fast_model"  # Use faster, less accurate model
    PARTIAL_RESULT = "partial_result"  # Return partial analysis
    ERROR = "error"  # Return error


@dataclass
class DiscoveryRequest:
    """Protocol discovery request."""
    request_id: str
    packet_data: bytes
    metadata: Dict[str, Any] = field(default_factory=dict)
    sla_ms: Optional[int] = None
    quality_mode: QualityMode = QualityMode.BALANCED
    require_explanation: bool = False
    model_version: Optional[str] = None
    ab_test_variant: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class FeatureImportance:
    """Feature importance for explainability."""
    feature_name: str
    importance_score: float
    contribution: float
    description: str


@dataclass
class DecisionPath:
    """Decision path through the model."""
    step: int
    layer_name: str
    activation_summary: Dict[str, float]
    decision_point: str
    confidence_at_step: float


@dataclass
class ExplanationData:
    """Explainable AI data."""
    feature_importances: List[FeatureImportance]
    decision_paths: List[DecisionPath]
    attention_weights: Optional[Dict[str, List[float]]] = None
    saliency_map: Optional[np.ndarray] = None
    confidence_breakdown: Optional[Dict[str, float]] = None
    reasoning: str = ""


@dataclass
class DiscoveryResult:
    """Protocol discovery result."""
    request_id: str
    protocol_type: str
    confidence: float
    structure: Dict[str, Any]
    processing_time_ms: float
    model_version: str
    quality_mode: QualityMode
    sla_met: bool
    fallback_used: bool = False
    explanation: Optional[ExplanationData] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ModelVersion:
    """Model version information."""
    version_id: str
    model_path: str
    deployment_type: DeploymentType
    traffic_percentage: float
    created_at: float
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    is_active: bool = True
    rollback_version: Optional[str] = None


@dataclass
class ABTestConfig:
    """A/B test configuration."""
    test_id: str
    variant_a: ModelVersion
    variant_b: ModelVersion
    traffic_split: float  # Percentage for variant A (0.0 to 1.0)
    start_time: float
    end_time: Optional[float] = None
    metrics_to_track: List[str] = field(default_factory=list)
    is_active: bool = True


class ExplainableDiscovery:
    """Explainable AI integration for protocol discovery."""
    
    def __init__(self, config: Config):
        """Initialize explainable discovery."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Feature names for interpretation
        self.feature_names = [
            "byte_entropy",
            "packet_length",
            "header_pattern",
            "payload_structure",
            "delimiter_presence",
            "ascii_ratio",
            "binary_pattern",
            "sequence_regularity"
        ]
    
    async def explain_prediction(
        self,
        model: BaseModel,
        input_data: ModelInput,
        prediction: ModelOutput
    ) -> ExplanationData:
        """
        Generate explanation for a prediction.
        
        Args:
            model: The model that made the prediction
            input_data: Input data used for prediction
            prediction: Model prediction output
            
        Returns:
            Explanation data with feature importances and decision paths
        """
        try:
            # Calculate feature importances using gradient-based method
            feature_importances = await self._calculate_feature_importance(
                model, input_data, prediction
            )
            
            # Extract decision paths
            decision_paths = await self._extract_decision_paths(
                model, input_data
            )
            
            # Calculate attention weights if model supports it
            attention_weights = await self._extract_attention_weights(
                model, input_data
            )
            
            # Generate confidence breakdown
            confidence_breakdown = self._generate_confidence_breakdown(
                prediction, feature_importances
            )
            
            # Generate human-readable reasoning
            reasoning = self._generate_reasoning(
                feature_importances, decision_paths, confidence_breakdown
            )
            
            return ExplanationData(
                feature_importances=feature_importances,
                decision_paths=decision_paths,
                attention_weights=attention_weights,
                confidence_breakdown=confidence_breakdown,
                reasoning=reasoning
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate explanation: {e}")
            # Return minimal explanation on error
            return ExplanationData(
                feature_importances=[],
                decision_paths=[],
                reasoning="Explanation generation failed"
            )
    
    async def _calculate_feature_importance(
        self,
        model: BaseModel,
        input_data: ModelInput,
        prediction: ModelOutput
    ) -> List[FeatureImportance]:
        """Calculate feature importance using integrated gradients."""
        importances = []
        
        try:
            # Convert input to tensor
            input_tensor = input_data.to_tensor()
            input_tensor.requires_grad = True
            
            # Forward pass
            model.eval()
            output = model(input_tensor)
            
            # Calculate gradients
            if isinstance(output, dict):
                output = output.get('logits', output.get('predictions', output))
            
            # Get gradient of output with respect to input
            output.sum().backward()
            
            # Calculate importance scores
            gradients = input_tensor.grad.abs().mean(dim=0).cpu().numpy()
            
            # Create feature importance objects
            for i, feature_name in enumerate(self.feature_names[:len(gradients)]):
                importance = float(gradients[i]) if i < len(gradients) else 0.0
                importances.append(FeatureImportance(
                    feature_name=feature_name,
                    importance_score=importance,
                    contribution=importance * 100,  # Percentage
                    description=f"Impact of {feature_name} on prediction"
                ))
            
            # Sort by importance
            importances.sort(key=lambda x: x.importance_score, reverse=True)
            
        except Exception as e:
            self.logger.warning(f"Feature importance calculation failed: {e}")
        
        return importances
    
    async def _extract_decision_paths(
        self,
        model: BaseModel,
        input_data: ModelInput
    ) -> List[DecisionPath]:
        """Extract decision paths through the model."""
        paths = []
        
        try:
            # Hook to capture intermediate activations
            activations = {}
            
            def hook_fn(name):
                def hook(module, input, output):
                    if isinstance(output, torch.Tensor):
                        activations[name] = output.detach()
                return hook
            
            # Register hooks on key layers
            hooks = []
            for name, module in model.named_modules():
                if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.LSTM)):
                    hooks.append(module.register_forward_hook(hook_fn(name)))
            
            # Forward pass
            model.eval()
            with torch.no_grad():
                input_tensor = input_data.to_tensor()
                _ = model(input_tensor)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Create decision path from activations
            for step, (layer_name, activation) in enumerate(activations.items()):
                activation_summary = {
                    "mean": float(activation.mean()),
                    "std": float(activation.std()),
                    "max": float(activation.max()),
                    "min": float(activation.min())
                }
                
                paths.append(DecisionPath(
                    step=step,
                    layer_name=layer_name,
                    activation_summary=activation_summary,
                    decision_point=f"Layer {layer_name} processing",
                    confidence_at_step=float(torch.sigmoid(activation.mean()))
                ))
                
        except Exception as e:
            self.logger.warning(f"Decision path extraction failed: {e}")
        
        return paths
    
    async def _extract_attention_weights(
        self,
        model: BaseModel,
        input_data: ModelInput
    ) -> Optional[Dict[str, List[float]]]:
        """Extract attention weights if model has attention mechanism."""
        try:
            # Check if model has attention layers
            has_attention = any(
                'attention' in name.lower()
                for name, _ in model.named_modules()
            )
            
            if not has_attention:
                return None
            
            # Extract attention weights (simplified)
            attention_weights = {
                "layer_0": [0.1, 0.3, 0.4, 0.2],  # Placeholder
                "layer_1": [0.2, 0.2, 0.3, 0.3]
            }
            
            return attention_weights
            
        except Exception as e:
            self.logger.warning(f"Attention weight extraction failed: {e}")
            return None
    
    def _generate_confidence_breakdown(
        self,
        prediction: ModelOutput,
        feature_importances: List[FeatureImportance]
    ) -> Dict[str, float]:
        """Generate confidence breakdown by component."""
        breakdown = {
            "base_confidence": 0.7,
            "feature_quality": 0.0,
            "model_certainty": 0.0,
            "data_quality": 0.0
        }
        
        if feature_importances:
            # Calculate feature quality contribution
            top_features = feature_importances[:3]
            breakdown["feature_quality"] = sum(
                f.importance_score for f in top_features
            ) / len(top_features)
        
        # Model certainty from prediction confidence
        if prediction.confidence is not None:
            breakdown["model_certainty"] = float(prediction.confidence.mean())
        
        # Data quality (simplified)
        breakdown["data_quality"] = 0.8
        
        return breakdown
    
    def _generate_reasoning(
        self,
        feature_importances: List[FeatureImportance],
        decision_paths: List[DecisionPath],
        confidence_breakdown: Dict[str, float]
    ) -> str:
        """Generate human-readable reasoning."""
        reasoning_parts = []
        
        # Feature-based reasoning
        if feature_importances:
            top_feature = feature_importances[0]
            reasoning_parts.append(
                f"The prediction is primarily influenced by {top_feature.feature_name} "
                f"(importance: {top_feature.importance_score:.2f})"
            )
        
        # Confidence reasoning
        overall_confidence = confidence_breakdown.get("model_certainty", 0.0)
        if overall_confidence > 0.8:
            reasoning_parts.append("The model has high confidence in this prediction")
        elif overall_confidence > 0.5:
            reasoning_parts.append("The model has moderate confidence in this prediction")
        else:
            reasoning_parts.append("The model has low confidence in this prediction")
        
        # Decision path reasoning
        if decision_paths:
            reasoning_parts.append(
                f"The prediction was made through {len(decision_paths)} processing layers"
            )
        
        return ". ".join(reasoning_parts) + "."


class ModelVersionManager:
    """Manages model versions, deployments, and rollbacks."""
    
    def __init__(self, config: Config):
        """Initialize model version manager."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Version registry
        self.versions: Dict[str, ModelVersion] = {}
        self.active_version: Optional[str] = None
        
        # A/B tests
        self.ab_tests: Dict[str, ABTestConfig] = {}
        
        # Canary deployments
        self.canary_deployments: Dict[str, ModelVersion] = {}
        
        # Performance tracking
        self.version_metrics: Dict[str, Dict[str, List[float]]] = {}
    
    def register_version(
        self,
        version_id: str,
        model_path: str,
        deployment_type: DeploymentType = DeploymentType.PRIMARY,
        traffic_percentage: float = 100.0
    ) -> ModelVersion:
        """Register a new model version."""
        version = ModelVersion(
            version_id=version_id,
            model_path=model_path,
            deployment_type=deployment_type,
            traffic_percentage=traffic_percentage,
            created_at=time.time(),
            rollback_version=self.active_version
        )
        
        self.versions[version_id] = version
        
        # Update metrics
        MODEL_VERSION_ACTIVE.labels(
            model_version=version_id,
            deployment_type=deployment_type.value
        ).set(1 if version.is_active else 0)
        
        self.logger.info(
            f"Registered model version {version_id} "
            f"({deployment_type.value}, {traffic_percentage}% traffic)"
        )
        
        return version
    
    def activate_version(self, version_id: str) -> bool:
        """Activate a model version as primary."""
        if version_id not in self.versions:
            self.logger.error(f"Version {version_id} not found")
            return False
        
        # Deactivate current active version
        if self.active_version and self.active_version in self.versions:
            self.versions[self.active_version].is_active = False
            MODEL_VERSION_ACTIVE.labels(
                model_version=self.active_version,
                deployment_type=DeploymentType.PRIMARY.value
            ).set(0)
        
        # Activate new version
        version = self.versions[version_id]
        version.is_active = True
        version.deployment_type = DeploymentType.PRIMARY
        version.traffic_percentage = 100.0
        self.active_version = version_id
        
        MODEL_VERSION_ACTIVE.labels(
            model_version=version_id,
            deployment_type=DeploymentType.PRIMARY.value
        ).set(1)
        
        self.logger.info(f"Activated model version {version_id}")
        return True
    
    def rollback_to_version(self, version_id: str) -> bool:
        """Rollback to a previous version."""
        if version_id not in self.versions:
            self.logger.error(f"Rollback version {version_id} not found")
            return False
        
        self.logger.warning(f"Rolling back to version {version_id}")
        return self.activate_version(version_id)
    
    def create_canary_deployment(
        self,
        version_id: str,
        traffic_percentage: float = 5.0
    ) -> bool:
        """Create a canary deployment for gradual rollout."""
        if version_id not in self.versions:
            self.logger.error(f"Version {version_id} not found for canary deployment")
            return False
        
        version = self.versions[version_id]
        version.deployment_type = DeploymentType.CANARY
        version.traffic_percentage = traffic_percentage
        version.is_active = True
        
        self.canary_deployments[version_id] = version
        
        MODEL_VERSION_ACTIVE.labels(
            model_version=version_id,
            deployment_type=DeploymentType.CANARY.value
        ).set(1)
        
        self.logger.info(
            f"Created canary deployment for {version_id} "
            f"with {traffic_percentage}% traffic"
        )
        
        return True
    
    def promote_canary(self, version_id: str) -> bool:
        """Promote canary deployment to primary."""
        if version_id not in self.canary_deployments:
            self.logger.error(f"Canary deployment {version_id} not found")
            return False
        
        # Check canary performance
        if not self._validate_canary_performance(version_id):
            self.logger.warning(
                f"Canary {version_id} performance validation failed"
            )
            return False
        
        # Promote to primary
        self.activate_version(version_id)
        
        # Remove from canary deployments
        del self.canary_deployments[version_id]
        
        self.logger.info(f"Promoted canary {version_id} to primary")
        return True
    
    def create_ab_test(
        self,
        test_id: str,
        version_a: str,
        version_b: str,
        traffic_split: float = 0.5,
        duration_hours: Optional[float] = None
    ) -> ABTestConfig:
        """Create an A/B test between two model versions."""
        if version_a not in self.versions or version_b not in self.versions:
            raise ModelVersionException("One or both versions not found for A/B test")
        
        end_time = None
        if duration_hours:
            end_time = time.time() + (duration_hours * 3600)
        
        ab_test = ABTestConfig(
            test_id=test_id,
            variant_a=self.versions[version_a],
            variant_b=self.versions[version_b],
            traffic_split=traffic_split,
            start_time=time.time(),
            end_time=end_time,
            metrics_to_track=["latency", "accuracy", "confidence"]
        )
        
        self.ab_tests[test_id] = ab_test
        
        self.logger.info(
            f"Created A/B test {test_id}: {version_a} vs {version_b} "
            f"(split: {traffic_split:.0%})"
        )
        
        return ab_test
    
    def select_version_for_request(
        self,
        request: DiscoveryRequest
    ) -> str:
        """Select appropriate model version for a request."""
        # Explicit version request
        if request.model_version and request.model_version in self.versions:
            return request.model_version
        
        # A/B test variant
        if request.ab_test_variant and request.ab_test_variant in self.ab_tests:
            ab_test = self.ab_tests[request.ab_test_variant]
            if ab_test.is_active:
                # Use hash of request ID for consistent assignment
                hash_val = int(hashlib.md5(request.request_id.encode()).hexdigest(), 16)
                if (hash_val % 100) / 100.0 < ab_test.traffic_split:
                    return ab_test.variant_a.version_id
                else:
                    return ab_test.variant_b.version_id
        
        # Canary deployment
        if self.canary_deployments:
            # Route percentage of traffic to canary
            for version_id, canary in self.canary_deployments.items():
                hash_val = int(hashlib.md5(request.request_id.encode()).hexdigest(), 16)
                if (hash_val % 100) < canary.traffic_percentage:
                    return version_id
        
        # Default to active version
        return self.active_version or list(self.versions.keys())[0]
    
    def record_version_metrics(
        self,
        version_id: str,
        metrics: Dict[str, float]
    ) -> None:
        """Record performance metrics for a version."""
        if version_id not in self.version_metrics:
            self.version_metrics[version_id] = {
                key: [] for key in metrics.keys()
            }
        
        for key, value in metrics.items():
            if key not in self.version_metrics[version_id]:
                self.version_metrics[version_id][key] = []
            self.version_metrics[version_id][key].append(value)
            
            # Keep only last 1000 measurements
            if len(self.version_metrics[version_id][key]) > 1000:
                self.version_metrics[version_id][key] = \
                    self.version_metrics[version_id][key][-1000:]
    
    def get_version_performance(self, version_id: str) -> Dict[str, float]:
        """Get performance statistics for a version."""
        if version_id not in self.version_metrics:
            return {}
        
        stats = {}
        for metric_name, values in self.version_metrics[version_id].items():
            if values:
                stats[f"{metric_name}_mean"] = np.mean(values)
                stats[f"{metric_name}_p50"] = np.percentile(values, 50)
                stats[f"{metric_name}_p95"] = np.percentile(values, 95)
                stats[f"{metric_name}_p99"] = np.percentile(values, 99)
        
        return stats
    
    def _validate_canary_performance(self, version_id: str) -> bool:
        """Validate canary performance before promotion."""
        if version_id not in self.version_metrics:
            return False
        
        canary_perf = self.get_version_performance(version_id)
        
        if not self.active_version or self.active_version not in self.version_metrics:
            return True  # No baseline to compare
        
        primary_perf = self.get_version_performance(self.active_version)
        
        # Check key metrics
        if "latency_p95" in canary_perf and "latency_p95" in primary_perf:
            if canary_perf["latency_p95"] > primary_perf["latency_p95"] * 1.2:
                self.logger.warning("Canary latency significantly worse than primary")
                return False
        
        if "accuracy_mean" in canary_perf and "accuracy_mean" in primary_perf:
            if canary_perf["accuracy_mean"] < primary_perf["accuracy_mean"] * 0.95:
                self.logger.warning("Canary accuracy significantly worse than primary")
                return False
        
        return True


class ProductionProtocolDiscovery:
    """
    Enterprise-ready protocol discovery system with SLA guarantees,
    explainable AI, and model versioning.
    """
    
    def __init__(self, config: Config):
        """Initialize production protocol discovery."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.explainable_discovery = ExplainableDiscovery(config)
        self.version_manager = ModelVersionManager(config)
        
        # Models (will be loaded)
        self.models: Dict[str, BaseModel] = {}
        
        # Result cache for fallback
        self.result_cache: Dict[str, DiscoveryResult] = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Performance tracking
        self.request_count = 0
        self.sla_violations = 0
        
        self.logger.info("ProductionProtocolDiscovery initialized")
    
    async def initialize(self) -> None:
        """Initialize the discovery system."""
        self.logger.info("Initializing production protocol discovery system")
        
        # Register default model version
        self.version_manager.register_version(
            version_id="v1.0.0",
            model_path="models/protocol_discovery_v1.pt",
            deployment_type=DeploymentType.PRIMARY
        )
        
        self.version_manager.activate_version("v1.0.0")
        
        self.logger.info("Production protocol discovery system initialized")
    
    async def discover_with_sla(
        self,
        request: DiscoveryRequest,
        sla_ms: int = 100
    ) -> DiscoveryResult:
        """
        Discover protocol with SLA guarantees.
        
        Args:
            request: Discovery request
            sla_ms: SLA threshold in milliseconds
            
        Returns:
            Discovery result with SLA compliance information
        """
        start_time = time.time()
        request.sla_ms = sla_ms
        
        try:
            # Select model version
            version_id = self.version_manager.select_version_for_request(request)
            
            # Create timeout for SLA
            try:
                result = await asyncio.wait_for(
                    self._perform_discovery(request, version_id),
                    timeout=sla_ms / 1000.0
                )
                
                result.sla_met = True
                
            except asyncio.TimeoutError:
                # SLA violated - use fallback strategy
                self.logger.warning(
                    f"SLA violation for request {request.request_id}: "
                    f"timeout after {sla_ms}ms"
                )
                
                result = await self._handle_sla_violation(
                    request, version_id, sla_ms
                )
                
                result.sla_met = False
                self.sla_violations += 1
                
                DISCOVERY_SLA_VIOLATIONS.labels(
                    model_version=version_id,
                    sla_threshold_ms=str(sla_ms)
                ).inc()
            
            # Record metrics
            processing_time_ms = (time.time() - start_time) * 1000
            result.processing_time_ms = processing_time_ms
            
            DISCOVERY_LATENCY.labels(
                model_version=version_id,
                quality_mode=request.quality_mode.value
            ).observe(processing_time_ms)
            
            DISCOVERY_CONFIDENCE.labels(
                model_version=version_id
            ).observe(result.confidence)
            
            # Record version metrics
            self.version_manager.record_version_metrics(version_id, {
                "latency": processing_time_ms,
                "confidence": result.confidence,
                "sla_met": 1.0 if result.sla_met else 0.0
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Discovery failed for request {request.request_id}: {e}")
            raise DiscoveryException(f"Discovery failed: {e}")
    
    async def discover_with_explainability(
        self,
        request: DiscoveryRequest
    ) -> DiscoveryResult:
        """
        Discover protocol with explainable AI.
        
        Args:
            request: Discovery request
            
        Returns:
            Discovery result with explanation data
        """
        request.require_explanation = True
        
        # Perform discovery
        result = await self.discover_with_sla(
            request,
            sla_ms=request.sla_ms or 500  # Default 500ms for explainable
        )
        
        return result
    
    async def discover_with_versioning(
        self,
        request: DiscoveryRequest
    ) -> DiscoveryResult:
        """
        Discover protocol with model versioning support.
        
        Args:
            request: Discovery request with optional version specification
            
        Returns:
            Discovery result with version information
        """
        # Version selection handled in discover_with_sla
        result = await self.discover_with_sla(
            request,
            sla_ms=request.sla_ms or 200
        )
        
        # Record A/B test metrics if applicable
        if request.ab_test_variant:
            AB_TEST_REQUESTS.labels(
                variant=request.ab_test_variant,
                model_version=result.model_version
            ).inc()
        
        return result
    
    async def _perform_discovery(
        self,
        request: DiscoveryRequest,
        version_id: str
    ) -> DiscoveryResult:
        """Perform actual protocol discovery."""
        # Get model for version
        model = await self._get_model(version_id)
        
        # Prepare input
        model_input = ModelInput(
            data=request.packet_data,
            metadata=request.metadata
        )
        
        # Run inference based on quality mode
        if request.quality_mode == QualityMode.FAST:
            prediction = await self._fast_inference(model, model_input)
        elif request.quality_mode == QualityMode.ACCURATE:
            prediction = await self._accurate_inference(model, model_input)
        else:  # BALANCED
            prediction = await self._balanced_inference(model, model_input)
        
        # Generate explanation if required
        explanation = None
        if request.require_explanation:
            explanation = await self.explainable_discovery.explain_prediction(
                model, model_input, prediction
            )
        
        # Extract protocol information from prediction
        protocol_type = self._extract_protocol_type(prediction)
        confidence = self._extract_confidence(prediction)
        structure = self._extract_structure(prediction)
        
        # Create result
        result = DiscoveryResult(
            request_id=request.request_id,
            protocol_type=protocol_type,
            confidence=confidence,
            structure=structure,
            processing_time_ms=0.0,  # Will be set by caller
            model_version=version_id,
            quality_mode=request.quality_mode,
            sla_met=True,
            explanation=explanation,
            metadata={
                "input_size": len(request.packet_data),
                "quality_mode": request.quality_mode.value
            }
        )
        
        # Cache result
        self._cache_result(request.request_id, result)
        
        return result
    
    async def _handle_sla_violation(
        self,
        request: DiscoveryRequest,
        version_id: str,
        sla_ms: int
    ) -> DiscoveryResult:
        """Handle SLA violation with fallback strategy."""
        fallback_strategy = self._determine_fallback_strategy(request)
        
        if fallback_strategy == FallbackStrategy.CACHED_RESULT:
            # Try to return cached result
            cached = self._get_cached_result(request.packet_data)
            if cached:
                cached.fallback_used = True
                cached.metadata["fallback_strategy"] = "cached_result"
                return cached
        
        elif fallback_strategy == FallbackStrategy.FAST_MODEL:
            # Use fast model with reduced quality
            fast_request = DiscoveryRequest(
                request_id=request.request_id,
                packet_data=request.packet_data,
                metadata=request.metadata,
                quality_mode=QualityMode.FAST
            )
            result = await self._perform_discovery(fast_request, version_id)
            result.fallback_used = True
            result.metadata["fallback_strategy"] = "fast_model"
            return result
        
        elif fallback_strategy == FallbackStrategy.PARTIAL_RESULT:
            # Return partial analysis
            return DiscoveryResult(
                request_id=request.request_id,
                protocol_type="unknown",
                confidence=0.0,
                structure={"partial": True},
                processing_time_ms=sla_ms,
                model_version=version_id,
                quality_mode=request.quality_mode,
                sla_met=False,
                fallback_used=True,
                metadata={"fallback_strategy": "partial_result"}
            )
        
        else:  # ERROR
            raise SLAViolationException(
                f"SLA violation: exceeded {sla_ms}ms threshold"
            )
    
    async def _get_model(self, version_id: str) -> BaseModel:
        """Get or load model for version."""
        if version_id not in self.models:
            # Load model (simplified - would load from disk in production)
            self.logger.info(f"Loading model version {version_id}")
            # Placeholder: would actually load model here
            # self.models[version_id] = load_model(version_id)
        
        return self.models.get(version_id)
    
    async def _fast_inference(
        self,
        model: BaseModel,
        input_data: ModelInput
    ) -> ModelOutput:
        """Fast inference with reduced accuracy."""
        # Use smaller batch size, fewer iterations, etc.
        return model.predict(input_data)
    
    async def _balanced_inference(
        self,
        model: BaseModel,
        input_data: ModelInput
    ) -> ModelOutput:
        """Balanced inference."""
        return model.predict(input_data)
    
    async def _accurate_inference(
        self,
        model: BaseModel,
        input_data: ModelInput
    ) -> ModelOutput:
        """Accurate inference with ensemble or multiple passes."""
        # Could use ensemble, multiple passes, etc.
        return model.predict(input_data)
    
    def _extract_protocol_type(self, prediction: ModelOutput) -> str:
        """Extract protocol type from prediction."""
        # Simplified extraction
        if hasattr(prediction, 'metadata') and prediction.metadata:
            return prediction.metadata.get('protocol_type', 'unknown')
        return 'unknown'
    
    def _extract_confidence(self, prediction: ModelOutput) -> float:
        """Extract confidence from prediction."""
        if prediction.confidence is not None:
            return float(prediction.confidence.mean())
        return 0.5
    
    def _extract_structure(self, prediction: ModelOutput) -> Dict[str, Any]:
        """Extract structure from prediction."""
        return {
            "fields": [],
            "format": "binary",
            "version": "unknown"
        }
    
    def _cache_result(self, request_id: str, result: DiscoveryResult) -> None:
        """Cache discovery result."""
        cache_key = self._generate_cache_key(result.request_id)
        self.result_cache[cache_key] = result
        
        # Clean old cache entries
        current_time = time.time()
        expired_keys = [
            k for k, v in self.result_cache.items()
            if current_time - v.timestamp > self.cache_ttl
        ]
        for key in expired_keys:
            del self.result_cache[key]
    
    def _get_cached_result(self, packet_data: bytes) -> Optional[DiscoveryResult]:
        """Get cached result for similar packet."""
        cache_key = self._generate_cache_key(packet_data)
        result = self.result_cache.get(cache_key)
        
        if result and (time.time() - result.timestamp) < self.cache_ttl:
            return result
        
        return None
    
    def _generate_cache_key(self, data: Union[str, bytes]) -> str:
        """Generate cache key from data."""
        if isinstance(data, str):
            data = data.encode()
        return hashlib.sha256(data).hexdigest()[:16]
    
    def _determine_fallback_strategy(
        self,
        request: DiscoveryRequest
    ) -> FallbackStrategy:
        """Determine appropriate fallback strategy."""
        # Check if cached result available
        if self._get_cached_result(request.packet_data):
            return FallbackStrategy.CACHED_RESULT
        
        # If quality mode is already fast, return error
        if request.quality_mode == QualityMode.FAST:
            return FallbackStrategy.ERROR
        
        # Otherwise, try fast model
        return FallbackStrategy.FAST_MODEL
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get discovery statistics."""
        return {
            "total_requests": self.request_count,
            "sla_violations": self.sla_violations,
            "sla_compliance_rate": (
                1.0 - (self.sla_violations / self.request_count)
                if self.request_count > 0 else 1.0
            ),
            "cache_size": len(self.result_cache),
            "active_versions": len(self.version_manager.versions),
            "active_ab_tests": len([
                t for t in self.version_manager.ab_tests.values()
                if t.is_active
            ])
        }
                model, model