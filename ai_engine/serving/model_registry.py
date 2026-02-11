"""
Model Registry for Version Management

Centralized model registry for:
- Model version tracking
- Metadata management
- Model lifecycle management
- Artifact storage integration
- Model lineage tracking
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import re

logger = logging.getLogger(__name__)


class ModelState(Enum):
    """Model lifecycle states."""

    DRAFT = "draft"  # Initial development
    STAGING = "staging"  # Ready for testing
    PRODUCTION = "production"  # Live in production
    ARCHIVED = "archived"  # No longer active
    DEPRECATED = "deprecated"  # Will be removed


@dataclass
class ModelMetadata:
    """Model metadata and properties."""

    # Basic info
    name: str
    version: str
    description: str = ""

    # Model details
    framework: str = ""  # pytorch, tensorflow, sklearn, etc.
    task: str = ""  # classification, generation, embedding, etc.
    architecture: str = ""  # transformer, cnn, etc.

    # Training info
    training_dataset: str = ""
    training_config: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    # Performance metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    benchmark_results: Dict[str, Any] = field(default_factory=dict)

    # Resource requirements
    min_memory: str = ""
    min_gpu_memory: str = ""
    recommended_batch_size: int = 0

    # Input/output schema
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)

    # Labels and tags
    labels: Dict[str, str] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    # Author and ownership
    author: str = ""
    team: str = ""
    contact: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "framework": self.framework,
            "task": self.task,
            "architecture": self.architecture,
            "training_dataset": self.training_dataset,
            "training_config": self.training_config,
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics,
            "benchmark_results": self.benchmark_results,
            "min_memory": self.min_memory,
            "min_gpu_memory": self.min_gpu_memory,
            "recommended_batch_size": self.recommended_batch_size,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "labels": self.labels,
            "tags": self.tags,
            "author": self.author,
            "team": self.team,
            "contact": self.contact,
        }


@dataclass
class ModelVersion:
    """Represents a specific model version."""

    # Identity
    model_name: str
    version: str
    version_id: str = ""  # Unique ID

    # State
    state: ModelState = ModelState.DRAFT
    metadata: ModelMetadata = field(default_factory=lambda: ModelMetadata("", ""))

    # Storage
    artifact_uri: str = ""
    artifact_hash: str = ""
    artifact_size_bytes: int = 0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    promoted_at: Optional[datetime] = None

    # Lineage
    parent_version: Optional[str] = None
    training_run_id: Optional[str] = None

    # Approval
    approved_by: Optional[str] = None
    approval_notes: str = ""

    # Deployment info
    deployed_endpoints: List[str] = field(default_factory=list)
    deployment_count: int = 0

    def __post_init__(self):
        if not self.version_id:
            self.version_id = self._generate_version_id()
        if not self.metadata.name:
            self.metadata = ModelMetadata(
                name=self.model_name,
                version=self.version
            )

    def _generate_version_id(self) -> str:
        """Generate unique version ID."""
        data = f"{self.model_name}:{self.version}:{self.created_at.isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "version": self.version,
            "version_id": self.version_id,
            "state": self.state.value,
            "metadata": self.metadata.to_dict(),
            "artifact_uri": self.artifact_uri,
            "artifact_hash": self.artifact_hash,
            "artifact_size_bytes": self.artifact_size_bytes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "promoted_at": self.promoted_at.isoformat() if self.promoted_at else None,
            "parent_version": self.parent_version,
            "training_run_id": self.training_run_id,
            "approved_by": self.approved_by,
            "deployed_endpoints": self.deployed_endpoints,
            "deployment_count": self.deployment_count,
        }


class ModelRegistry:
    """
    Centralized model registry for version management.

    Features:
    - Register and track model versions
    - Manage model lifecycle (draft -> staging -> production)
    - Store and retrieve model metadata
    - Track model lineage
    - Integrate with artifact storage (S3, GCS, etc.)

    Example:
        registry = ModelRegistry()

        # Register a new model version
        version = await registry.register_model(
            name="protocol-classifier",
            version="1.0.0",
            artifact_uri="gs://qbitel-models/protocol-classifier/v1.0.0",
            metadata=ModelMetadata(
                name="protocol-classifier",
                version="1.0.0",
                framework="pytorch",
                task="classification",
                metrics={"accuracy": 0.95, "f1": 0.93}
            )
        )

        # Promote to production
        await registry.promote_model(
            name="protocol-classifier",
            version="1.0.0",
            target_state=ModelState.PRODUCTION
        )
    """

    def __init__(
        self,
        storage_backend: Optional[str] = None,
        storage_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize model registry.

        Args:
            storage_backend: Storage backend type (local, s3, gcs)
            storage_config: Backend-specific configuration
        """
        self.storage_backend = storage_backend or "local"
        self.storage_config = storage_config or {}

        self._models: Dict[str, Dict[str, ModelVersion]] = {}  # name -> version -> ModelVersion
        self._callbacks: Dict[str, List[Callable]] = {
            "registered": [],
            "promoted": [],
            "deployed": [],
        }
        self._lock = asyncio.Lock()

        logger.info(f"Model registry initialized with backend: {self.storage_backend}")

    async def register_model(
        self,
        name: str,
        version: str,
        artifact_uri: str,
        metadata: Optional[ModelMetadata] = None,
        parent_version: Optional[str] = None,
        training_run_id: Optional[str] = None,
    ) -> ModelVersion:
        """
        Register a new model version.

        Args:
            name: Model name
            version: Version string (semantic versioning recommended)
            artifact_uri: URI to model artifacts
            metadata: Model metadata
            parent_version: Parent version for lineage tracking
            training_run_id: ID of training run that produced this model

        Returns:
            Registered ModelVersion
        """
        logger.info(f"Registering model: {name} v{version}")

        # Validate version format
        if not self._validate_version(version):
            raise ValueError(f"Invalid version format: {version}")

        # Check for duplicate
        if name in self._models and version in self._models[name]:
            raise ValueError(f"Model version already exists: {name} v{version}")

        # Create model version
        model_version = ModelVersion(
            model_name=name,
            version=version,
            state=ModelState.DRAFT,
            metadata=metadata or ModelMetadata(name=name, version=version),
            artifact_uri=artifact_uri,
            parent_version=parent_version,
            training_run_id=training_run_id,
        )

        # Compute artifact hash
        model_version.artifact_hash = await self._compute_artifact_hash(artifact_uri)

        # Store
        async with self._lock:
            if name not in self._models:
                self._models[name] = {}
            self._models[name][version] = model_version

        # Persist metadata
        await self._persist_metadata(model_version)

        # Notify callbacks
        await self._notify("registered", model_version)

        logger.info(f"Model registered: {name} v{version} (id: {model_version.version_id})")
        return model_version

    async def get_model(
        self,
        name: str,
        version: Optional[str] = None,
        state: Optional[ModelState] = None,
    ) -> Optional[ModelVersion]:
        """
        Get a model version.

        Args:
            name: Model name
            version: Specific version (if None, gets latest)
            state: Filter by state

        Returns:
            ModelVersion or None
        """
        if name not in self._models:
            return None

        versions = self._models[name]

        if version:
            return versions.get(version)

        # Get latest version with optional state filter
        filtered = list(versions.values())
        if state:
            filtered = [v for v in filtered if v.state == state]

        if not filtered:
            return None

        # Sort by semantic version and return latest
        filtered.sort(key=lambda v: self._parse_version(v.version), reverse=True)
        return filtered[0]

    async def list_models(
        self,
        name: Optional[str] = None,
        state: Optional[ModelState] = None,
        tags: Optional[List[str]] = None,
    ) -> List[ModelVersion]:
        """
        List model versions.

        Args:
            name: Filter by model name
            state: Filter by state
            tags: Filter by tags

        Returns:
            List of ModelVersions
        """
        results = []

        for model_name, versions in self._models.items():
            if name and model_name != name:
                continue

            for version in versions.values():
                if state and version.state != state:
                    continue

                if tags:
                    if not all(t in version.metadata.tags for t in tags):
                        continue

                results.append(version)

        # Sort by name and version
        results.sort(key=lambda v: (v.model_name, self._parse_version(v.version)))
        return results

    async def promote_model(
        self,
        name: str,
        version: str,
        target_state: ModelState,
        approved_by: Optional[str] = None,
        approval_notes: str = "",
    ) -> ModelVersion:
        """
        Promote model to a new state.

        Args:
            name: Model name
            version: Version to promote
            target_state: Target state
            approved_by: Approver identifier
            approval_notes: Notes about approval

        Returns:
            Updated ModelVersion
        """
        logger.info(f"Promoting model: {name} v{version} -> {target_state.value}")

        model = await self.get_model(name, version)
        if not model:
            raise ValueError(f"Model not found: {name} v{version}")

        # Validate state transition
        if not self._validate_state_transition(model.state, target_state):
            raise ValueError(
                f"Invalid state transition: {model.state.value} -> {target_state.value}"
            )

        # Update state
        model.state = target_state
        model.updated_at = datetime.utcnow()
        model.promoted_at = datetime.utcnow()
        model.approved_by = approved_by
        model.approval_notes = approval_notes

        # Persist
        await self._persist_metadata(model)

        # Notify callbacks
        await self._notify("promoted", model)

        logger.info(f"Model promoted: {name} v{version} -> {target_state.value}")
        return model

    async def update_metadata(
        self,
        name: str,
        version: str,
        metadata: ModelMetadata,
    ) -> ModelVersion:
        """
        Update model metadata.

        Args:
            name: Model name
            version: Model version
            metadata: New metadata

        Returns:
            Updated ModelVersion
        """
        model = await self.get_model(name, version)
        if not model:
            raise ValueError(f"Model not found: {name} v{version}")

        model.metadata = metadata
        model.updated_at = datetime.utcnow()

        await self._persist_metadata(model)

        logger.info(f"Metadata updated: {name} v{version}")
        return model

    async def record_deployment(
        self,
        name: str,
        version: str,
        endpoint: str,
    ) -> ModelVersion:
        """
        Record model deployment to an endpoint.

        Args:
            name: Model name
            version: Model version
            endpoint: Deployment endpoint

        Returns:
            Updated ModelVersion
        """
        model = await self.get_model(name, version)
        if not model:
            raise ValueError(f"Model not found: {name} v{version}")

        if endpoint not in model.deployed_endpoints:
            model.deployed_endpoints.append(endpoint)
            model.deployment_count += 1
            model.updated_at = datetime.utcnow()

            await self._persist_metadata(model)
            await self._notify("deployed", model)

        logger.info(f"Deployment recorded: {name} v{version} -> {endpoint}")
        return model

    async def delete_model(
        self,
        name: str,
        version: str,
        force: bool = False,
    ) -> bool:
        """
        Delete a model version.

        Args:
            name: Model name
            version: Model version
            force: Force delete even if deployed

        Returns:
            True if deleted
        """
        model = await self.get_model(name, version)
        if not model:
            return False

        # Check if deployed
        if model.deployed_endpoints and not force:
            raise ValueError(
                f"Cannot delete deployed model. Use force=True or undeploy first."
            )

        async with self._lock:
            if name in self._models and version in self._models[name]:
                del self._models[name][version]
                if not self._models[name]:
                    del self._models[name]

        logger.info(f"Model deleted: {name} v{version}")
        return True

    async def compare_versions(
        self,
        name: str,
        version1: str,
        version2: str,
    ) -> Dict[str, Any]:
        """
        Compare two model versions.

        Returns differences in metrics, config, etc.
        """
        v1 = await self.get_model(name, version1)
        v2 = await self.get_model(name, version2)

        if not v1 or not v2:
            raise ValueError("One or both versions not found")

        return {
            "versions": [version1, version2],
            "metric_diff": {
                metric: v2.metadata.metrics.get(metric, 0) - v1.metadata.metrics.get(metric, 0)
                for metric in set(v1.metadata.metrics) | set(v2.metadata.metrics)
            },
            "hyperparameter_diff": {
                param: {
                    "v1": v1.metadata.hyperparameters.get(param),
                    "v2": v2.metadata.hyperparameters.get(param),
                }
                for param in set(v1.metadata.hyperparameters) | set(v2.metadata.hyperparameters)
                if v1.metadata.hyperparameters.get(param) != v2.metadata.hyperparameters.get(param)
            },
            "state_diff": {
                "v1": v1.state.value,
                "v2": v2.state.value,
            },
            "created_diff_days": (v2.created_at - v1.created_at).days,
        }

    def on_event(
        self,
        event: str,
        callback: Callable[[ModelVersion], None],
    ) -> None:
        """Register callback for registry events."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    async def _notify(self, event: str, model: ModelVersion) -> None:
        """Notify callbacks of an event."""
        for callback in self._callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(model)
                else:
                    callback(model)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")

    def _validate_version(self, version: str) -> bool:
        """Validate version format (semantic versioning)."""
        pattern = r"^\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$"
        return bool(re.match(pattern, version))

    def _parse_version(self, version: str) -> tuple:
        """Parse version string for comparison."""
        match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
        if match:
            return tuple(int(x) for x in match.groups())
        return (0, 0, 0)

    def _validate_state_transition(
        self,
        current: ModelState,
        target: ModelState,
    ) -> bool:
        """Validate state transition is allowed."""
        allowed_transitions = {
            ModelState.DRAFT: [ModelState.STAGING, ModelState.ARCHIVED],
            ModelState.STAGING: [ModelState.PRODUCTION, ModelState.DRAFT, ModelState.ARCHIVED],
            ModelState.PRODUCTION: [ModelState.DEPRECATED, ModelState.ARCHIVED],
            ModelState.DEPRECATED: [ModelState.ARCHIVED],
            ModelState.ARCHIVED: [],  # Terminal state
        }
        return target in allowed_transitions.get(current, [])

    async def _compute_artifact_hash(self, artifact_uri: str) -> str:
        """Compute hash of model artifacts."""
        # In production, would download and hash actual artifacts
        return hashlib.sha256(artifact_uri.encode()).hexdigest()[:16]

    async def _persist_metadata(self, model: ModelVersion) -> None:
        """Persist model metadata to storage backend."""
        if self.storage_backend == "local":
            # Local file storage
            storage_path = Path(self.storage_config.get("path", "/tmp/model_registry"))
            storage_path.mkdir(parents=True, exist_ok=True)

            file_path = storage_path / f"{model.model_name}_{model.version}.json"
            with open(file_path, "w") as f:
                json.dump(model.to_dict(), f, indent=2)

        # Add S3, GCS backends as needed
        logger.debug(f"Persisted metadata for {model.model_name} v{model.version}")

    async def _load_metadata(self) -> None:
        """Load persisted metadata from storage backend."""
        if self.storage_backend == "local":
            storage_path = Path(self.storage_config.get("path", "/tmp/model_registry"))
            if not storage_path.exists():
                return

            for file_path in storage_path.glob("*.json"):
                try:
                    with open(file_path) as f:
                        data = json.load(f)

                    model = ModelVersion(
                        model_name=data["model_name"],
                        version=data["version"],
                        version_id=data.get("version_id", ""),
                        state=ModelState(data.get("state", "draft")),
                        artifact_uri=data.get("artifact_uri", ""),
                        artifact_hash=data.get("artifact_hash", ""),
                    )

                    if model.model_name not in self._models:
                        self._models[model.model_name] = {}
                    self._models[model.model_name][model.version] = model

                except Exception as e:
                    logger.error(f"Failed to load metadata from {file_path}: {e}")

        logger.info(f"Loaded {sum(len(v) for v in self._models.values())} model versions")
