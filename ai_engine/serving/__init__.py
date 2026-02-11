"""
Model Serving Module

Enterprise-grade model serving infrastructure for QBITEL:
- KServe integration for standardized model serving
- Multi-model serving with intelligent routing
- Auto-scaling based on traffic patterns
- Model versioning and A/B testing
- GPU resource management
- Inference optimization (batching, caching)

Components:
- KServeManager: KServe InferenceService management
- ModelRouter: Intelligent request routing
- ModelRegistry: Model version management
- InferenceOptimizer: Batching and caching
- GPUScheduler: GPU resource allocation

Usage:
    from ai_engine.serving import KServeManager, ModelRouter

    # Initialize KServe manager
    manager = KServeManager(namespace="qbitel-vllm")

    # Deploy a model
    await manager.deploy_model(
        name="protocol-classifier",
        model_uri="gs://qbitel-models/protocol-classifier/v1",
        runtime="triton",
        resources={"gpu": 1}
    )
"""

from ai_engine.serving.kserve_manager import (
    KServeManager,
    InferenceServiceConfig,
    ModelDeployment,
    DeploymentStatus,
    ScalingConfig,
    ResourceConfig,
    RuntimeType,
)
from ai_engine.serving.model_router import (
    ModelRouter,
    RoutingStrategy,
    RoutingConfig,
    ModelEndpoint,
    TrafficSplit,
)
from ai_engine.serving.model_registry import (
    ModelRegistry,
    ModelVersion,
    ModelMetadata,
    ModelState,
)
from ai_engine.serving.inference_optimizer import (
    InferenceOptimizer,
    BatchConfig,
    CacheConfig,
    OptimizationMetrics,
)

__all__ = [
    # KServe Manager
    "KServeManager",
    "InferenceServiceConfig",
    "ModelDeployment",
    "DeploymentStatus",
    "ScalingConfig",
    "ResourceConfig",
    "RuntimeType",
    # Model Router
    "ModelRouter",
    "RoutingStrategy",
    "RoutingConfig",
    "ModelEndpoint",
    "TrafficSplit",
    # Model Registry
    "ModelRegistry",
    "ModelVersion",
    "ModelMetadata",
    "ModelState",
    # Inference Optimizer
    "InferenceOptimizer",
    "BatchConfig",
    "CacheConfig",
    "OptimizationMetrics",
]
