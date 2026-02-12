"""
KServe Model Serving Manager

Provides enterprise-grade model serving through KServe:
- InferenceService management
- Canary deployments and A/B testing
- Auto-scaling with custom metrics
- GPU resource management
- Health monitoring and alerting
- Model versioning and rollback

Supports multiple runtimes:
- vLLM for LLM serving
- Triton for general ML models
- TorchServe for PyTorch models
- Custom runtimes
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
import json
import hashlib

logger = logging.getLogger(__name__)


class RuntimeType(Enum):
    """Supported model serving runtimes."""

    VLLM = "vllm"
    TRITON = "triton"
    TORCHSERVE = "torchserve"
    TENSORFLOW = "tensorflow"
    SKLEARN = "sklearn"
    XGBOOST = "xgboost"
    CUSTOM = "custom"


class DeploymentStatus(Enum):
    """InferenceService deployment status."""

    PENDING = "pending"
    CREATING = "creating"
    READY = "ready"
    UPDATING = "updating"
    FAILED = "failed"
    TERMINATING = "terminating"
    UNKNOWN = "unknown"


@dataclass
class ResourceConfig:
    """Resource configuration for model serving."""

    # CPU resources
    cpu_request: str = "1"
    cpu_limit: str = "4"

    # Memory resources
    memory_request: str = "2Gi"
    memory_limit: str = "8Gi"

    # GPU resources
    gpu_count: int = 0
    gpu_type: str = "nvidia.com/gpu"
    gpu_memory: Optional[str] = None

    # Storage
    storage_request: str = "10Gi"

    # Node selection
    node_selector: Dict[str, str] = field(default_factory=dict)
    tolerations: List[Dict[str, Any]] = field(default_factory=list)
    affinity: Optional[Dict[str, Any]] = None

    def to_k8s_resources(self) -> Dict[str, Any]:
        """Convert to Kubernetes resource spec."""
        resources = {
            "requests": {
                "cpu": self.cpu_request,
                "memory": self.memory_request,
            },
            "limits": {
                "cpu": self.cpu_limit,
                "memory": self.memory_limit,
            },
        }

        if self.gpu_count > 0:
            resources["requests"][self.gpu_type] = str(self.gpu_count)
            resources["limits"][self.gpu_type] = str(self.gpu_count)

        return resources


@dataclass
class ScalingConfig:
    """Auto-scaling configuration."""

    min_replicas: int = 1
    max_replicas: int = 10

    # Scale to zero configuration
    scale_to_zero: bool = False
    scale_to_zero_grace_period: int = 600  # seconds

    # Target metrics
    target_cpu_utilization: int = 70
    target_memory_utilization: int = 80
    target_concurrency: Optional[int] = None
    target_rps: Optional[int] = None

    # Scale behavior
    scale_up_stabilization: int = 60  # seconds
    scale_down_stabilization: int = 300  # seconds

    # Custom metrics
    custom_metrics: List[Dict[str, Any]] = field(default_factory=list)

    def to_kserve_spec(self) -> Dict[str, Any]:
        """Convert to KServe scaling spec."""
        spec = {
            "minReplicas": self.min_replicas,
            "maxReplicas": self.max_replicas,
        }

        if self.scale_to_zero:
            spec["minReplicas"] = 0

        if self.target_concurrency:
            spec["targetUtilizationPercentage"] = self.target_concurrency

        return spec


@dataclass
class InferenceServiceConfig:
    """Configuration for KServe InferenceService."""

    name: str
    namespace: str = "qbitel-vllm"

    # Model configuration
    model_uri: str = ""
    model_format: str = "pytorch"
    runtime: RuntimeType = RuntimeType.VLLM

    # Container configuration
    image: Optional[str] = None
    command: Optional[List[str]] = None
    args: Optional[List[str]] = None
    env: Dict[str, str] = field(default_factory=dict)

    # Resources
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)

    # Serving configuration
    protocol_version: str = "v2"
    timeout: int = 300
    batch_size: Optional[int] = None

    # Canary configuration
    canary_traffic_percent: int = 0
    canary_model_uri: Optional[str] = None

    # Transformer (pre/post processing)
    transformer_image: Optional[str] = None
    transformer_resources: Optional[ResourceConfig] = None

    # Explainer (model interpretability)
    explainer_type: Optional[str] = None  # "ART", "AIX360", "Alibi"

    # Labels and annotations
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

    def get_runtime_image(self) -> str:
        """Get the default image for the runtime."""
        runtime_images = {
            RuntimeType.VLLM: "ghcr.io/qbitel/vllm-server:latest",
            RuntimeType.TRITON: "nvcr.io/nvidia/tritonserver:23.10-py3",
            RuntimeType.TORCHSERVE: "pytorch/torchserve:0.9.0-gpu",
            RuntimeType.TENSORFLOW: "tensorflow/serving:2.14.0-gpu",
            RuntimeType.SKLEARN: "kserve/sklearnserver:v0.11.0",
            RuntimeType.XGBOOST: "kserve/xgbserver:v0.11.0",
        }
        return self.image or runtime_images.get(self.runtime, "")


@dataclass
class ModelDeployment:
    """Represents a deployed model."""

    name: str
    namespace: str
    config: InferenceServiceConfig
    status: DeploymentStatus
    url: Optional[str] = None
    internal_url: Optional[str] = None

    # Versioning
    revision: str = ""
    model_hash: str = ""

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    ready_at: Optional[datetime] = None

    # Metrics
    replicas: int = 0
    ready_replicas: int = 0

    # Health
    last_health_check: Optional[datetime] = None
    health_status: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "namespace": self.namespace,
            "status": self.status.value,
            "url": self.url,
            "internal_url": self.internal_url,
            "revision": self.revision,
            "model_hash": self.model_hash,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "ready_at": self.ready_at.isoformat() if self.ready_at else None,
            "replicas": self.replicas,
            "ready_replicas": self.ready_replicas,
            "health_status": self.health_status,
        }


class KServeManager:
    """
    Manages KServe InferenceServices for model serving.

    Features:
    - Deploy, update, and delete models
    - Canary deployments and traffic splitting
    - Auto-scaling configuration
    - Health monitoring
    - Rollback support

    Example:
        manager = KServeManager(namespace="qbitel-vllm")

        # Deploy a model
        deployment = await manager.deploy_model(
            config=InferenceServiceConfig(
                name="protocol-classifier",
                model_uri="gs://qbitel-models/protocol-classifier/v1",
                runtime=RuntimeType.TRITON,
                resources=ResourceConfig(gpu_count=1)
            )
        )

        # Check status
        status = await manager.get_deployment_status("protocol-classifier")
    """

    def __init__(
        self,
        namespace: str = "qbitel-vllm",
        kubeconfig_path: Optional[str] = None,
        in_cluster: bool = True,
    ):
        """
        Initialize KServe manager.

        Args:
            namespace: Default Kubernetes namespace
            kubeconfig_path: Path to kubeconfig file
            in_cluster: Whether running inside Kubernetes
        """
        self.namespace = namespace
        self.kubeconfig_path = kubeconfig_path
        self.in_cluster = in_cluster

        self._deployments: Dict[str, ModelDeployment] = {}
        self._k8s_client = None
        self._kserve_client = None
        self._health_check_tasks: Dict[str, asyncio.Task] = {}
        self._callbacks: List[Callable[[ModelDeployment], None]] = []

        logger.info(f"KServe Manager initialized for namespace: {namespace}")

    async def initialize(self) -> None:
        """Initialize Kubernetes and KServe clients."""
        try:
            # Initialize Kubernetes client
            from kubernetes import client, config

            if self.in_cluster:
                config.load_incluster_config()
            elif self.kubeconfig_path:
                config.load_kube_config(config_file=self.kubeconfig_path)
            else:
                config.load_kube_config()

            self._k8s_client = client.CustomObjectsApi()
            logger.info("Kubernetes client initialized")

            # Load existing deployments
            await self._sync_deployments()

        except ImportError:
            logger.warning("kubernetes package not installed, running in mock mode")
            self._k8s_client = None
        except Exception as e:
            logger.error(f"Failed to initialize Kubernetes client: {e}")
            raise

    async def deploy_model(
        self,
        config: InferenceServiceConfig,
        wait_ready: bool = True,
        timeout: int = 600,
    ) -> ModelDeployment:
        """
        Deploy a model using KServe InferenceService.

        Args:
            config: InferenceService configuration
            wait_ready: Wait for deployment to be ready
            timeout: Timeout in seconds

        Returns:
            ModelDeployment instance
        """
        logger.info(f"Deploying model: {config.name}")

        # Generate InferenceService spec
        isvc_spec = self._build_inference_service_spec(config)

        # Create deployment record
        deployment = ModelDeployment(
            name=config.name,
            namespace=config.namespace or self.namespace,
            config=config,
            status=DeploymentStatus.CREATING,
            model_hash=self._compute_model_hash(config),
        )

        try:
            if self._k8s_client:
                # Create or update InferenceService
                try:
                    await self._create_inference_service(isvc_spec)
                except Exception as e:
                    if "AlreadyExists" in str(e):
                        await self._update_inference_service(isvc_spec)
                    else:
                        raise

            # Store deployment
            self._deployments[config.name] = deployment

            if wait_ready:
                deployment = await self._wait_for_ready(config.name, timeout)

            # Start health check task
            self._start_health_check(config.name)

            # Notify callbacks
            self._notify_callbacks(deployment)

            logger.info(f"Model deployed successfully: {config.name}")
            return deployment

        except Exception as e:
            deployment.status = DeploymentStatus.FAILED
            logger.error(f"Failed to deploy model {config.name}: {e}")
            raise

    async def update_model(
        self,
        name: str,
        config: InferenceServiceConfig,
        canary_percent: int = 0,
    ) -> ModelDeployment:
        """
        Update an existing model deployment.

        Args:
            name: Model name
            config: New configuration
            canary_percent: Traffic percentage for canary

        Returns:
            Updated ModelDeployment
        """
        logger.info(f"Updating model: {name} (canary: {canary_percent}%)")

        deployment = self._deployments.get(name)
        if not deployment:
            raise ValueError(f"Model not found: {name}")

        deployment.status = DeploymentStatus.UPDATING
        deployment.updated_at = datetime.utcnow()

        # Update config with canary settings
        config.canary_traffic_percent = canary_percent

        # Build and apply new spec
        isvc_spec = self._build_inference_service_spec(config)

        if self._k8s_client:
            await self._update_inference_service(isvc_spec)

        deployment.config = config
        deployment.model_hash = self._compute_model_hash(config)

        # Wait for rollout
        deployment = await self._wait_for_ready(name, timeout=600)

        logger.info(f"Model updated successfully: {name}")
        return deployment

    async def promote_canary(
        self,
        name: str,
        full_rollout: bool = False,
    ) -> ModelDeployment:
        """
        Promote canary deployment.

        Args:
            name: Model name
            full_rollout: Whether to complete rollout (100%)

        Returns:
            Updated ModelDeployment
        """
        deployment = self._deployments.get(name)
        if not deployment:
            raise ValueError(f"Model not found: {name}")

        config = deployment.config

        if full_rollout:
            # Remove canary, make it the new default
            config.canary_traffic_percent = 0
            config.model_uri = config.canary_model_uri or config.model_uri
            config.canary_model_uri = None
        else:
            # Increase canary traffic
            new_percent = min(100, config.canary_traffic_percent + 20)
            config.canary_traffic_percent = new_percent

        return await self.update_model(name, config)

    async def rollback(
        self,
        name: str,
        revision: Optional[str] = None,
    ) -> ModelDeployment:
        """
        Rollback model to previous revision.

        Args:
            name: Model name
            revision: Specific revision to rollback to

        Returns:
            Rolled back ModelDeployment
        """
        logger.info(f"Rolling back model: {name} to revision: {revision or 'previous'}")

        deployment = self._deployments.get(name)
        if not deployment:
            raise ValueError(f"Model not found: {name}")

        # Remove canary configuration for rollback
        config = deployment.config
        config.canary_traffic_percent = 0
        config.canary_model_uri = None

        return await self.update_model(name, config)

    async def delete_model(
        self,
        name: str,
        wait_deleted: bool = True,
    ) -> bool:
        """
        Delete a model deployment.

        Args:
            name: Model name
            wait_deleted: Wait for deletion to complete

        Returns:
            True if deleted successfully
        """
        logger.info(f"Deleting model: {name}")

        deployment = self._deployments.get(name)
        if deployment:
            deployment.status = DeploymentStatus.TERMINATING

        # Stop health check
        if name in self._health_check_tasks:
            self._health_check_tasks[name].cancel()
            del self._health_check_tasks[name]

        if self._k8s_client:
            try:
                await self._delete_inference_service(name, deployment.namespace if deployment else self.namespace)
            except Exception as e:
                if "NotFound" not in str(e):
                    raise

        # Remove from deployments
        if name in self._deployments:
            del self._deployments[name]

        logger.info(f"Model deleted: {name}")
        return True

    async def get_deployment(self, name: str) -> Optional[ModelDeployment]:
        """Get deployment by name."""
        return self._deployments.get(name)

    async def list_deployments(
        self,
        namespace: Optional[str] = None,
    ) -> List[ModelDeployment]:
        """List all deployments."""
        deployments = list(self._deployments.values())
        if namespace:
            deployments = [d for d in deployments if d.namespace == namespace]
        return deployments

    async def get_deployment_status(
        self,
        name: str,
    ) -> Optional[DeploymentStatus]:
        """Get deployment status."""
        deployment = self._deployments.get(name)
        if deployment:
            await self._refresh_deployment_status(deployment)
            return deployment.status
        return None

    async def get_deployment_metrics(
        self,
        name: str,
    ) -> Dict[str, Any]:
        """
        Get deployment metrics.

        Returns metrics like:
        - Request count
        - Latency percentiles
        - Error rate
        - GPU utilization
        """
        deployment = self._deployments.get(name)
        if not deployment:
            return {}

        # In production, fetch from Prometheus
        return {
            "name": name,
            "replicas": deployment.replicas,
            "ready_replicas": deployment.ready_replicas,
            "status": deployment.status.value,
            "health": deployment.health_status,
            "requests_total": 0,  # Would come from metrics
            "latency_p50_ms": 0,
            "latency_p99_ms": 0,
            "error_rate": 0.0,
        }

    def add_callback(
        self,
        callback: Callable[[ModelDeployment], None],
    ) -> None:
        """Add deployment status callback."""
        self._callbacks.append(callback)

    def _build_inference_service_spec(
        self,
        config: InferenceServiceConfig,
    ) -> Dict[str, Any]:
        """Build KServe InferenceService specification."""

        # Base predictor spec
        predictor_spec = {
            "minReplicas": config.scaling.min_replicas,
            "maxReplicas": config.scaling.max_replicas,
        }

        # Runtime-specific configuration
        if config.runtime == RuntimeType.VLLM:
            predictor_spec["containers"] = [
                {
                    "name": "kserve-container",
                    "image": config.get_runtime_image(),
                    "args": self._build_vllm_args(config),
                    "env": [{"name": k, "value": v} for k, v in config.env.items()],
                    "resources": config.resources.to_k8s_resources(),
                    "ports": [{"containerPort": 8000, "protocol": "TCP"}],
                }
            ]
        elif config.runtime == RuntimeType.TRITON:
            predictor_spec["triton"] = {
                "storageUri": config.model_uri,
                "runtimeVersion": "23.10-py3",
                "resources": config.resources.to_k8s_resources(),
            }
        elif config.runtime == RuntimeType.TORCHSERVE:
            predictor_spec["pytorch"] = {
                "storageUri": config.model_uri,
                "protocolVersion": config.protocol_version,
                "resources": config.resources.to_k8s_resources(),
            }
        elif config.runtime == RuntimeType.TENSORFLOW:
            predictor_spec["tensorflow"] = {
                "storageUri": config.model_uri,
                "resources": config.resources.to_k8s_resources(),
            }
        elif config.runtime == RuntimeType.SKLEARN:
            predictor_spec["sklearn"] = {
                "storageUri": config.model_uri,
                "resources": config.resources.to_k8s_resources(),
            }
        elif config.runtime == RuntimeType.XGBOOST:
            predictor_spec["xgboost"] = {
                "storageUri": config.model_uri,
                "resources": config.resources.to_k8s_resources(),
            }
        else:
            # Custom runtime
            predictor_spec["containers"] = [
                {
                    "name": "kserve-container",
                    "image": config.get_runtime_image(),
                    "command": config.command,
                    "args": config.args,
                    "env": [{"name": k, "value": v} for k, v in config.env.items()],
                    "resources": config.resources.to_k8s_resources(),
                }
            ]

        # Add node selector and tolerations
        if config.resources.node_selector:
            predictor_spec["nodeSelector"] = config.resources.node_selector
        if config.resources.tolerations:
            predictor_spec["tolerations"] = config.resources.tolerations
        if config.resources.affinity:
            predictor_spec["affinity"] = config.resources.affinity

        # Build full spec
        spec = {
            "predictor": predictor_spec,
        }

        # Add transformer if configured
        if config.transformer_image:
            spec["transformer"] = {
                "containers": [
                    {
                        "name": "transformer",
                        "image": config.transformer_image,
                        "resources": (
                            config.transformer_resources.to_k8s_resources()
                            if config.transformer_resources
                            else config.resources.to_k8s_resources()
                        ),
                    }
                ]
            }

        # Add explainer if configured
        if config.explainer_type:
            spec["explainer"] = {
                config.explainer_type.lower(): {
                    "type": config.explainer_type,
                }
            }

        # Add canary if configured
        if config.canary_traffic_percent > 0 and config.canary_model_uri:
            spec["canaryTrafficPercent"] = config.canary_traffic_percent

        # Build complete InferenceService
        isvc = {
            "apiVersion": "serving.kserve.io/v1beta1",
            "kind": "InferenceService",
            "metadata": {
                "name": config.name,
                "namespace": config.namespace or self.namespace,
                "labels": {
                    "app.kubernetes.io/name": config.name,
                    "app.kubernetes.io/part-of": "qbitel",
                    "app.kubernetes.io/managed-by": "kserve-manager",
                    **config.labels,
                },
                "annotations": {
                    "serving.kserve.io/autoscalerClass": "hpa",
                    "serving.kserve.io/deploymentMode": "RawDeployment",
                    **config.annotations,
                },
            },
            "spec": spec,
        }

        return isvc

    def _build_vllm_args(self, config: InferenceServiceConfig) -> List[str]:
        """Build vLLM server arguments."""
        args = [
            "--model",
            config.model_uri,
            "--port",
            "8000",
            "--host",
            "0.0.0.0",
        ]

        if config.resources.gpu_count > 1:
            args.extend(["--tensor-parallel-size", str(config.resources.gpu_count)])

        if config.batch_size:
            args.extend(["--max-batch-size", str(config.batch_size)])

        # Add GPU memory utilization
        args.extend(["--gpu-memory-utilization", "0.9"])

        # Enable streaming
        args.append("--enable-streaming")

        return args

    async def _create_inference_service(
        self,
        spec: Dict[str, Any],
    ) -> None:
        """Create InferenceService in Kubernetes."""
        if not self._k8s_client:
            logger.info(f"Mock: Creating InferenceService {spec['metadata']['name']}")
            return

        await asyncio.to_thread(
            self._k8s_client.create_namespaced_custom_object,
            group="serving.kserve.io",
            version="v1beta1",
            namespace=spec["metadata"]["namespace"],
            plural="inferenceservices",
            body=spec,
        )

    async def _update_inference_service(
        self,
        spec: Dict[str, Any],
    ) -> None:
        """Update InferenceService in Kubernetes."""
        if not self._k8s_client:
            logger.info(f"Mock: Updating InferenceService {spec['metadata']['name']}")
            return

        await asyncio.to_thread(
            self._k8s_client.patch_namespaced_custom_object,
            group="serving.kserve.io",
            version="v1beta1",
            namespace=spec["metadata"]["namespace"],
            plural="inferenceservices",
            name=spec["metadata"]["name"],
            body=spec,
        )

    async def _delete_inference_service(
        self,
        name: str,
        namespace: str,
    ) -> None:
        """Delete InferenceService from Kubernetes."""
        if not self._k8s_client:
            logger.info(f"Mock: Deleting InferenceService {name}")
            return

        await asyncio.to_thread(
            self._k8s_client.delete_namespaced_custom_object,
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            name=name,
        )

    async def _wait_for_ready(
        self,
        name: str,
        timeout: int,
    ) -> ModelDeployment:
        """Wait for deployment to be ready."""
        deployment = self._deployments[name]
        start_time = asyncio.get_event_loop().time()

        while True:
            await self._refresh_deployment_status(deployment)

            if deployment.status == DeploymentStatus.READY:
                deployment.ready_at = datetime.utcnow()
                return deployment

            if deployment.status == DeploymentStatus.FAILED:
                raise RuntimeError(f"Deployment failed: {name}")

            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Deployment timed out: {name}")

            await asyncio.sleep(5)

        return deployment

    async def _refresh_deployment_status(
        self,
        deployment: ModelDeployment,
    ) -> None:
        """Refresh deployment status from Kubernetes."""
        if not self._k8s_client:
            # Mock mode - simulate ready
            deployment.status = DeploymentStatus.READY
            deployment.replicas = deployment.config.scaling.min_replicas
            deployment.ready_replicas = deployment.replicas
            deployment.url = f"http://{deployment.name}.{deployment.namespace}.svc.cluster.local"
            return

        try:
            isvc = await asyncio.to_thread(
                self._k8s_client.get_namespaced_custom_object,
                group="serving.kserve.io",
                version="v1beta1",
                namespace=deployment.namespace,
                plural="inferenceservices",
                name=deployment.name,
            )

            # Parse status
            status = isvc.get("status", {})
            conditions = status.get("conditions", [])

            # Check conditions
            ready = False
            for condition in conditions:
                if condition.get("type") == "Ready":
                    ready = condition.get("status") == "True"
                    break

            if ready:
                deployment.status = DeploymentStatus.READY
            else:
                deployment.status = DeploymentStatus.PENDING

            # Get URLs
            deployment.url = status.get("url")
            deployment.internal_url = status.get("address", {}).get("url")

            # Get replicas
            components = status.get("components", {})
            predictor = components.get("predictor", {})
            deployment.replicas = predictor.get("replicas", 0)
            deployment.ready_replicas = predictor.get("readyReplicas", 0)

        except Exception as e:
            logger.error(f"Failed to refresh status for {deployment.name}: {e}")
            deployment.status = DeploymentStatus.UNKNOWN

    async def _sync_deployments(self) -> None:
        """Sync deployments from Kubernetes."""
        if not self._k8s_client:
            return

        try:
            isvcs = await asyncio.to_thread(
                self._k8s_client.list_namespaced_custom_object,
                group="serving.kserve.io",
                version="v1beta1",
                namespace=self.namespace,
                plural="inferenceservices",
            )

            for isvc in isvcs.get("items", []):
                name = isvc["metadata"]["name"]
                if name not in self._deployments:
                    # Create deployment record from existing
                    deployment = ModelDeployment(
                        name=name,
                        namespace=isvc["metadata"]["namespace"],
                        config=InferenceServiceConfig(name=name),
                        status=DeploymentStatus.UNKNOWN,
                    )
                    self._deployments[name] = deployment
                    await self._refresh_deployment_status(deployment)

        except Exception as e:
            logger.error(f"Failed to sync deployments: {e}")

    def _start_health_check(self, name: str) -> None:
        """Start periodic health check for deployment."""
        if name in self._health_check_tasks:
            return

        async def health_check_loop():
            while True:
                try:
                    deployment = self._deployments.get(name)
                    if not deployment:
                        break

                    await self._check_health(deployment)
                    await asyncio.sleep(30)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Health check failed for {name}: {e}")
                    await asyncio.sleep(60)

        task = asyncio.create_task(health_check_loop())
        self._health_check_tasks[name] = task

    async def _check_health(self, deployment: ModelDeployment) -> None:
        """Check deployment health."""
        deployment.last_health_check = datetime.utcnow()

        # In production, make HTTP health check to the model endpoint
        if deployment.url:
            try:
                import httpx

                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{deployment.url}/v2/health/ready",
                        timeout=10,
                    )
                    deployment.health_status = "healthy" if response.status_code == 200 else "unhealthy"
            except ImportError:
                deployment.health_status = "unknown"
            except Exception:
                deployment.health_status = "unhealthy"
        else:
            deployment.health_status = "unknown"

    def _compute_model_hash(self, config: InferenceServiceConfig) -> str:
        """Compute hash of model configuration."""
        hash_data = json.dumps(
            {
                "model_uri": config.model_uri,
                "runtime": config.runtime.value,
                "image": config.get_runtime_image(),
            },
            sort_keys=True,
        )
        return hashlib.sha256(hash_data.encode()).hexdigest()[:12]

    def _notify_callbacks(self, deployment: ModelDeployment) -> None:
        """Notify all callbacks of deployment change."""
        for callback in self._callbacks:
            try:
                callback(deployment)
            except Exception as e:
                logger.error(f"Callback failed: {e}")
