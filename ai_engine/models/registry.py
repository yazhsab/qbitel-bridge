"""
CRONOS AI Engine - Model Registry

This module implements a comprehensive model registry system for versioning,
storage, and management of AI models with MLflow integration.
"""

import logging
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import asyncio

import torch
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion as MLModelVersion
import boto3
from botocore.exceptions import ClientError

from ..core.config import Config
from ..core.exceptions import ModelRegistryException, ModelException


class ModelStatus(str, Enum):
    """Model status enumeration."""

    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    FAILED = "failed"


class ModelType(str, Enum):
    """Model type enumeration."""

    PROTOCOL_DISCOVERY = "protocol_discovery"
    FIELD_DETECTION = "field_detection"
    ANOMALY_DETECTION = "anomaly_detection"
    ENSEMBLE = "ensemble"


@dataclass
class ModelMetadata:
    """Model metadata for registry."""

    model_id: str
    name: str
    version: str
    model_type: ModelType
    status: ModelStatus
    description: Optional[str] = None
    tags: Dict[str, str] = None

    # Technical specifications
    framework: str = "pytorch"
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    model_size_mb: Optional[float] = None

    # Performance metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    inference_latency_ms: Optional[float] = None

    # Lifecycle information
    created_at: Optional[float] = None
    updated_at: Optional[float] = None
    created_by: Optional[str] = None
    training_job_id: Optional[str] = None
    parent_model_id: Optional[str] = None

    # Deployment information
    deployment_config: Optional[Dict[str, Any]] = None
    resource_requirements: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        if self.created_at is None:
            self.created_at = time.time()
        if self.updated_at is None:
            self.updated_at = self.created_at


@dataclass
class ModelVersion:
    """Model version information."""

    version: str
    model_uri: str
    checksum: str
    metadata: ModelMetadata
    artifacts: Dict[str, str] = None  # artifact_name -> uri

    def __post_init__(self):
        if self.artifacts is None:
            self.artifacts = {}


class ModelRegistry:
    """
    Comprehensive model registry with MLflow integration.

    This class provides enterprise-grade model management including
    versioning, metadata tracking, and deployment management.
    """

    def __init__(self, config: Config):
        """Initialize model registry."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # MLflow client
        self.mlflow_client: Optional[MlflowClient] = None

        # Storage backend
        self.storage_backend = "local"  # local, s3, gcs
        self.storage_config = {}

        # Local storage paths
        self.models_path = Path("models")
        self.metadata_path = Path("model_metadata")

        # In-memory cache
        self._model_cache: Dict[str, ModelMetadata] = {}
        self._version_cache: Dict[str, List[ModelVersion]] = {}

        # S3 client (if using S3 storage)
        self.s3_client = None

        self.logger.info("ModelRegistry initialized")

    async def initialize(self) -> None:
        """Initialize the model registry."""
        try:
            # Initialize MLflow client
            if self.config.mlflow.tracking_uri:
                self.mlflow_client = MlflowClient(self.config.mlflow.tracking_uri)
                self.logger.info("MLflow client initialized")

            # Initialize storage backend
            await self._initialize_storage()

            # Create local directories
            self.models_path.mkdir(exist_ok=True)
            self.metadata_path.mkdir(exist_ok=True)

            # Load existing metadata
            await self._load_existing_metadata()

            self.logger.info("ModelRegistry initialization completed")

        except Exception as e:
            self.logger.error(f"Failed to initialize ModelRegistry: {e}")
            raise ModelRegistryException(f"Registry initialization failed: {e}")

    async def register_model(
        self,
        model: Union[torch.nn.Module, str],
        metadata: ModelMetadata,
        artifacts: Optional[Dict[str, Union[str, Path]]] = None,
    ) -> ModelVersion:
        """
        Register a new model or model version.

        Args:
            model: PyTorch model or path to model file
            metadata: Model metadata
            artifacts: Optional artifacts to store with model

        Returns:
            ModelVersion with registration information
        """
        try:
            self.logger.info(f"Registering model: {metadata.name} v{metadata.version}")

            # Generate model URI
            model_uri = await self._store_model(model, metadata)

            # Calculate checksum
            checksum = await self._calculate_checksum(model_uri)

            # Store artifacts
            artifact_uris = {}
            if artifacts:
                artifact_uris = await self._store_artifacts(artifacts, metadata)

            # Create model version
            model_version = ModelVersion(
                version=metadata.version,
                model_uri=model_uri,
                checksum=checksum,
                metadata=metadata,
                artifacts=artifact_uris,
            )

            # Register with MLflow if available
            if self.mlflow_client:
                await self._register_with_mlflow(model, metadata, model_version)

            # Store metadata locally
            await self._store_metadata(metadata)

            # Update cache
            self._model_cache[metadata.model_id] = metadata
            if metadata.model_id not in self._version_cache:
                self._version_cache[metadata.model_id] = []
            self._version_cache[metadata.model_id].append(model_version)

            self.logger.info(f"Model registered successfully: {metadata.model_id}")
            return model_version

        except Exception as e:
            self.logger.error(f"Model registration failed: {e}")
            raise ModelRegistryException(f"Registration failed: {e}")

    async def get_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> ModelVersion:
        """
        Retrieve a model version.

        Args:
            model_name: Name of the model
            version: Specific version (if None, gets latest)
            stage: Model stage (staging, production)

        Returns:
            ModelVersion information
        """
        try:
            # Try MLflow first
            if self.mlflow_client:
                if stage:
                    model_version = self.mlflow_client.get_latest_versions(
                        model_name, stages=[stage]
                    )[0]
                    version = model_version.version
                elif not version:
                    model_version = self.mlflow_client.get_latest_versions(
                        model_name, stages=["production", "staging"]
                    )[0]
                    version = model_version.version

                return await self._get_model_version_from_mlflow(model_name, version)

            # Fallback to local registry
            return await self._get_model_version_local(model_name, version)

        except Exception as e:
            self.logger.error(f"Failed to get model {model_name}:{version}: {e}")
            raise ModelRegistryException(f"Model retrieval failed: {e}")

    async def list_models(
        self,
        model_type: Optional[ModelType] = None,
        status: Optional[ModelStatus] = None,
    ) -> List[ModelMetadata]:
        """
        List registered models.

        Args:
            model_type: Filter by model type
            status: Filter by status

        Returns:
            List of model metadata
        """
        models = list(self._model_cache.values())

        # Apply filters
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        if status:
            models = [m for m in models if m.status == status]

        return models

    async def list_model_versions(self, model_name: str) -> List[ModelVersion]:
        """List all versions of a model."""
        model_id = await self._get_model_id_by_name(model_name)
        return self._version_cache.get(model_id, [])

    async def update_model_status(
        self,
        model_name: str,
        version: str,
        status: ModelStatus,
        description: Optional[str] = None,
    ) -> bool:
        """
        Update model status.

        Args:
            model_name: Model name
            version: Model version
            status: New status
            description: Optional status description

        Returns:
            Success status
        """
        try:
            # Update in MLflow
            if self.mlflow_client:
                if status == ModelStatus.PRODUCTION:
                    self.mlflow_client.transition_model_version_stage(
                        model_name,
                        version,
                        "production",
                        archive_existing_versions=True,
                    )
                elif status == ModelStatus.STAGING:
                    self.mlflow_client.transition_model_version_stage(
                        model_name, version, "staging"
                    )
                elif status == ModelStatus.ARCHIVED:
                    self.mlflow_client.transition_model_version_stage(
                        model_name, version, "archived"
                    )

                if description:
                    self.mlflow_client.update_model_version(
                        model_name, version, description=description
                    )

            # Update local metadata
            model_id = await self._get_model_id_by_name(model_name)
            if model_id in self._model_cache:
                self._model_cache[model_id].status = status
                self._model_cache[model_id].updated_at = time.time()
                await self._store_metadata(self._model_cache[model_id])

            self.logger.info(
                f"Model status updated: {model_name}:{version} -> {status}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to update model status: {e}")
            return False

    async def delete_model(
        self, model_name: str, version: Optional[str] = None
    ) -> bool:
        """
        Delete a model or specific version.

        Args:
            model_name: Model name
            version: Specific version (if None, deletes all versions)

        Returns:
            Success status
        """
        try:
            if self.mlflow_client:
                if version:
                    # Delete specific version
                    self.mlflow_client.delete_model_version(model_name, version)
                else:
                    # Delete entire model
                    self.mlflow_client.delete_registered_model(model_name)

            # Clean up local storage
            await self._cleanup_local_storage(model_name, version)

            # Update cache
            model_id = await self._get_model_id_by_name(model_name)
            if version:
                # Remove specific version
                if model_id in self._version_cache:
                    self._version_cache[model_id] = [
                        v for v in self._version_cache[model_id] if v.version != version
                    ]
            else:
                # Remove entire model
                if model_id in self._model_cache:
                    del self._model_cache[model_id]
                if model_id in self._version_cache:
                    del self._version_cache[model_id]

            self.logger.info(f"Model deleted: {model_name}:{version or 'all'}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete model: {e}")
            return False

    async def load_model(
        self, model_name: str, version: Optional[str] = None
    ) -> torch.nn.Module:
        """
        Load a model for inference.

        Args:
            model_name: Model name
            version: Model version

        Returns:
            Loaded PyTorch model
        """
        try:
            model_version = await self.get_model(model_name, version)

            # Load from URI
            if model_version.model_uri.startswith("s3://"):
                # Load from S3
                return await self._load_model_from_s3(model_version.model_uri)
            elif model_version.model_uri.startswith("file://"):
                # Load from local file
                model_path = model_version.model_uri.replace("file://", "")
                return torch.load(model_path, map_location="cpu")
            elif self.mlflow_client and model_version.model_uri.startswith("models:/"):
                # Load from MLflow
                return mlflow.pytorch.load_model(model_version.model_uri)
            else:
                # Direct file path
                return torch.load(model_version.model_uri, map_location="cpu")

        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}:{version}: {e}")
            raise ModelException(f"Model loading failed: {e}")

    async def compare_models(
        self,
        model1_name: str,
        model1_version: str,
        model2_name: str,
        model2_version: str,
    ) -> Dict[str, Any]:
        """
        Compare two model versions.

        Returns:
            Comparison results including metrics differences
        """
        try:
            model1 = await self.get_model(model1_name, model1_version)
            model2 = await self.get_model(model2_name, model2_version)

            comparison = {
                "model1": {
                    "name": model1_name,
                    "version": model1_version,
                    "metadata": asdict(model1.metadata),
                },
                "model2": {
                    "name": model2_name,
                    "version": model2_version,
                    "metadata": asdict(model2.metadata),
                },
                "metrics_comparison": {},
                "size_comparison": {},
                "performance_comparison": {},
            }

            # Compare metrics
            metrics1 = {
                "accuracy": model1.metadata.accuracy,
                "precision": model1.metadata.precision,
                "recall": model1.metadata.recall,
                "f1_score": model1.metadata.f1_score,
            }

            metrics2 = {
                "accuracy": model2.metadata.accuracy,
                "precision": model2.metadata.precision,
                "recall": model2.metadata.recall,
                "f1_score": model2.metadata.f1_score,
            }

            for metric in metrics1:
                if metrics1[metric] is not None and metrics2[metric] is not None:
                    comparison["metrics_comparison"][metric] = {
                        "model1": metrics1[metric],
                        "model2": metrics2[metric],
                        "difference": metrics1[metric] - metrics2[metric],
                        "improvement": (metrics1[metric] - metrics2[metric])
                        / metrics2[metric]
                        * 100,
                    }

            # Compare sizes
            if model1.metadata.model_size_mb and model2.metadata.model_size_mb:
                comparison["size_comparison"] = {
                    "model1_mb": model1.metadata.model_size_mb,
                    "model2_mb": model2.metadata.model_size_mb,
                    "size_difference_mb": model1.metadata.model_size_mb
                    - model2.metadata.model_size_mb,
                }

            # Compare inference latency
            if (
                model1.metadata.inference_latency_ms
                and model2.metadata.inference_latency_ms
            ):
                comparison["performance_comparison"] = {
                    "model1_latency_ms": model1.metadata.inference_latency_ms,
                    "model2_latency_ms": model2.metadata.inference_latency_ms,
                    "latency_difference_ms": model1.metadata.inference_latency_ms
                    - model2.metadata.inference_latency_ms,
                }

            return comparison

        except Exception as e:
            self.logger.error(f"Model comparison failed: {e}")
            raise ModelRegistryException(f"Comparison failed: {e}")

    async def get_model_lineage(self, model_name: str, version: str) -> Dict[str, Any]:
        """Get model lineage and relationships."""
        try:
            model_version = await self.get_model(model_name, version)

            lineage = {
                "current_model": {
                    "name": model_name,
                    "version": version,
                    "model_id": model_version.metadata.model_id,
                },
                "parent_models": [],
                "child_models": [],
                "related_training_jobs": [],
            }

            # Find parent models
            if model_version.metadata.parent_model_id:
                parent_metadata = self._model_cache.get(
                    model_version.metadata.parent_model_id
                )
                if parent_metadata:
                    lineage["parent_models"].append(
                        {
                            "name": parent_metadata.name,
                            "version": parent_metadata.version,
                            "model_id": parent_metadata.model_id,
                        }
                    )

            # Find child models
            for model_id, metadata in self._model_cache.items():
                if metadata.parent_model_id == model_version.metadata.model_id:
                    lineage["child_models"].append(
                        {
                            "name": metadata.name,
                            "version": metadata.version,
                            "model_id": metadata.model_id,
                        }
                    )

            # Add training job information
            if model_version.metadata.training_job_id:
                lineage["related_training_jobs"].append(
                    {
                        "job_id": model_version.metadata.training_job_id,
                        "created_at": model_version.metadata.created_at,
                    }
                )

            return lineage

        except Exception as e:
            self.logger.error(f"Failed to get model lineage: {e}")
            raise ModelRegistryException(f"Lineage retrieval failed: {e}")

    # Private methods

    async def _initialize_storage(self) -> None:
        """Initialize storage backend."""
        # For now, use local storage
        # In production, could be S3, GCS, etc.
        self.storage_backend = "local"

        if self.storage_backend == "s3":
            try:
                self.s3_client = boto3.client("s3")
                self.logger.info("S3 storage backend initialized")
            except Exception as e:
                self.logger.warning(
                    f"S3 initialization failed, falling back to local: {e}"
                )
                self.storage_backend = "local"

    async def _store_model(
        self, model: Union[torch.nn.Module, str], metadata: ModelMetadata
    ) -> str:
        """Store model and return URI."""
        model_filename = f"{metadata.name}_{metadata.version}_{metadata.model_id}.pt"

        if self.storage_backend == "local":
            model_path = self.models_path / model_filename

            if isinstance(model, torch.nn.Module):
                torch.save(model.state_dict(), model_path)
            else:
                # Copy existing file
                import shutil

                shutil.copy(model, model_path)

            return f"file://{model_path.absolute()}"

        elif self.storage_backend == "s3":
            # Store in S3
            s3_key = f"models/{model_filename}"

            if isinstance(model, torch.nn.Module):
                # Save to temporary file first
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
                    torch.save(model.state_dict(), tmp.name)
                    self.s3_client.upload_file(
                        tmp.name, self.storage_config["bucket"], s3_key
                    )
            else:
                self.s3_client.upload_file(model, self.storage_config["bucket"], s3_key)

            return f"s3://{self.storage_config['bucket']}/{s3_key}"

        else:
            raise ModelRegistryException(
                f"Unsupported storage backend: {self.storage_backend}"
            )

    async def _calculate_checksum(self, model_uri: str) -> str:
        """Calculate model checksum for integrity verification."""
        if model_uri.startswith("file://"):
            file_path = model_uri.replace("file://", "")
            with open(file_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        elif model_uri.startswith("s3://"):
            # For S3, use ETag or calculate on download
            return "s3_etag_placeholder"  # Would implement proper S3 checksum
        else:
            return "unknown"

    async def _store_artifacts(
        self, artifacts: Dict[str, Union[str, Path]], metadata: ModelMetadata
    ) -> Dict[str, str]:
        """Store model artifacts and return URIs."""
        artifact_uris = {}

        for name, path in artifacts.items():
            artifact_filename = f"{metadata.name}_{metadata.version}_{name}"

            if self.storage_backend == "local":
                artifact_path = self.models_path / "artifacts" / artifact_filename
                artifact_path.parent.mkdir(exist_ok=True)

                import shutil

                shutil.copy(path, artifact_path)
                artifact_uris[name] = f"file://{artifact_path.absolute()}"

            # Add S3 support here if needed

        return artifact_uris

    async def _store_metadata(self, metadata: ModelMetadata) -> None:
        """Store model metadata."""
        metadata_file = self.metadata_path / f"{metadata.model_id}.json"

        with open(metadata_file, "w") as f:
            json.dump(asdict(metadata), f, indent=2)

    async def _load_existing_metadata(self) -> None:
        """Load existing model metadata from storage."""
        if not self.metadata_path.exists():
            return

        for metadata_file in self.metadata_path.glob("*.json"):
            try:
                with open(metadata_file, "r") as f:
                    metadata_dict = json.load(f)

                metadata = ModelMetadata(**metadata_dict)
                self._model_cache[metadata.model_id] = metadata

            except Exception as e:
                self.logger.warning(
                    f"Failed to load metadata from {metadata_file}: {e}"
                )

    async def _register_with_mlflow(
        self,
        model: Union[torch.nn.Module, str],
        metadata: ModelMetadata,
        model_version: ModelVersion,
    ) -> None:
        """Register model with MLflow."""
        if not self.mlflow_client:
            return

        try:
            # Create or get registered model
            try:
                self.mlflow_client.create_registered_model(
                    metadata.name, tags=metadata.tags, description=metadata.description
                )
            except Exception:
                # Model already exists
                pass

            # Create model version in MLflow
            with mlflow.start_run():
                if isinstance(model, torch.nn.Module):
                    mlflow.pytorch.log_model(
                        model,
                        artifact_path="model",
                        registered_model_name=metadata.name,
                    )

                # Log metadata as tags
                mlflow.set_tags(
                    {
                        "model_type": metadata.model_type.value,
                        "model_id": metadata.model_id,
                        "framework": metadata.framework,
                    }
                )

                # Log metrics
                if metadata.accuracy:
                    mlflow.log_metric("accuracy", metadata.accuracy)
                if metadata.precision:
                    mlflow.log_metric("precision", metadata.precision)
                if metadata.recall:
                    mlflow.log_metric("recall", metadata.recall)
                if metadata.f1_score:
                    mlflow.log_metric("f1_score", metadata.f1_score)

        except Exception as e:
            self.logger.warning(f"MLflow registration failed: {e}")

    async def _get_model_id_by_name(self, model_name: str) -> str:
        """Get model ID by name."""
        for model_id, metadata in self._model_cache.items():
            if metadata.name == model_name:
                return model_id

        raise ModelRegistryException(f"Model not found: {model_name}")

    async def _get_model_version_from_mlflow(
        self, model_name: str, version: str
    ) -> ModelVersion:
        """Get model version from MLflow."""
        # Implementation would fetch from MLflow
        # For now, return placeholder
        raise NotImplementedError("MLflow model version retrieval not implemented")

    async def _get_model_version_local(
        self, model_name: str, version: Optional[str]
    ) -> ModelVersion:
        """Get model version from local registry."""
        model_id = await self._get_model_id_by_name(model_name)

        if model_id not in self._version_cache:
            raise ModelRegistryException(f"No versions found for model: {model_name}")

        versions = self._version_cache[model_id]

        if version:
            for v in versions:
                if v.version == version:
                    return v
            raise ModelRegistryException(
                f"Version {version} not found for model: {model_name}"
            )
        else:
            # Return latest version
            if versions:
                return sorted(versions, key=lambda v: v.metadata.created_at or 0)[-1]
            else:
                raise ModelRegistryException(
                    f"No versions available for model: {model_name}"
                )

    async def _cleanup_local_storage(
        self, model_name: str, version: Optional[str]
    ) -> None:
        """Clean up local storage for deleted models."""
        # Implementation would clean up model files and artifacts
        pass

    async def _load_model_from_s3(self, s3_uri: str) -> torch.nn.Module:
        """Load model from S3."""
        # Implementation would download from S3 and load
        raise NotImplementedError("S3 model loading not implemented")
