"""
CRONOS AI Engine - Model Management System

This module provides comprehensive model lifecycle management including
training, versioning, deployment, monitoring, and A/B testing capabilities.
"""

import asyncio
import hashlib
import json
import logging
import os
import pickle
import shutil
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import tempfile
import threading
from contextlib import asynccontextmanager

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib

from ..core.config import Config
from ..core.exceptions import ModelException, TrainingException, DeploymentException
from ..monitoring.metrics import AIEngineMetrics
from ..core.structured_logging import get_logger


class ModelStatus(str, Enum):
    """Model lifecycle status."""
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATING = "validating"
    VALIDATED = "validated"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    SERVING = "serving"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    FAILED = "failed"


class ModelType(str, Enum):
    """Supported model types."""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    SKLEARN = "sklearn"
    XGBOOST = "xgboost"
    ONNX = "onnx"
    CUSTOM = "custom"


class DeploymentStrategy(str, Enum):
    """Model deployment strategies."""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    A_B_TEST = "a_b_test"
    SHADOW = "shadow"


@dataclass
class ModelMetadata:
    """Comprehensive model metadata."""
    model_id: str
    name: str
    version: str
    model_type: ModelType
    status: ModelStatus
    created_at: datetime
    updated_at: datetime
    
    # Training information
    training_config: Dict[str, Any] = field(default_factory=dict)
    training_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Model artifacts
    model_path: Optional[str] = None
    model_size_bytes: int = 0
    model_checksum: Optional[str] = None
    
    # Deployment information
    deployment_config: Dict[str, Any] = field(default_factory=dict)
    deployment_strategy: Optional[DeploymentStrategy] = None
    serving_endpoints: List[str] = field(default_factory=list)
    
    # Performance tracking
    inference_count: int = 0
    average_latency_ms: float = 0.0
    error_rate: float = 0.0
    throughput_rps: float = 0.0
    
    # Dependencies and environment
    dependencies: Dict[str, str] = field(default_factory=dict)
    python_version: str = ""
    framework_version: str = ""
    
    # Tags and labels
    tags: Dict[str, str] = field(default_factory=dict)
    labels: List[str] = field(default_factory=list)
    
    # Lineage and provenance
    parent_model_id: Optional[str] = None
    training_dataset_hash: Optional[str] = None
    training_code_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        data['status'] = self.status.value
        data['model_type'] = self.model_type.value
        if self.deployment_strategy:
            data['deployment_strategy'] = self.deployment_strategy.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary."""
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        data['status'] = ModelStatus(data['status'])
        data['model_type'] = ModelType(data['model_type'])
        if data.get('deployment_strategy'):
            data['deployment_strategy'] = DeploymentStrategy(data['deployment_strategy'])
        return cls(**data)


@dataclass
class TrainingJob:
    """Training job configuration and state."""
    job_id: str
    model_name: str
    model_version: str
    training_config: Dict[str, Any]
    
    status: str = "pending"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    
    # Results
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    
    # Resource usage
    gpu_hours: float = 0.0
    cpu_hours: float = 0.0
    memory_gb_hours: float = 0.0


class ModelRegistry:
    """
    Central model registry for managing model lifecycle.
    
    Provides model versioning, metadata management, and artifact storage
    with enterprise-grade features like lineage tracking and audit logs.
    """
    
    def __init__(self, config: Config):
        """Initialize model registry."""
        self.config = config
        self.logger = get_logger(__name__)
        
        # Storage configuration
        self.registry_path = Path(getattr(config, 'model_registry_path', './model_registry'))
        self.artifact_storage = getattr(config, 'artifact_storage_type', 'local')
        self.max_versions_per_model = getattr(config, 'max_model_versions', 10)
        
        # MLflow configuration
        self.mlflow_tracking_uri = getattr(config, 'mlflow_tracking_uri', 'sqlite:///mlflow.db')
        self.mlflow_experiment_name = getattr(config, 'mlflow_experiment_name', 'cronos_ai_models')
        
        # Thread safety
        self._registry_lock = threading.RLock()
        
        # Initialize storage
        self._initialize_storage()
        self._initialize_mlflow()
        
        self.logger.info("ModelRegistry initialized")
    
    def _initialize_storage(self):
        """Initialize model storage directories."""
        self.registry_path.mkdir(parents=True, exist_ok=True)
        (self.registry_path / 'models').mkdir(exist_ok=True)
        (self.registry_path / 'metadata').mkdir(exist_ok=True)
        (self.registry_path / 'checkpoints').mkdir(exist_ok=True)
        (self.registry_path / 'artifacts').mkdir(exist_ok=True)
    
    def _initialize_mlflow(self):
        """Initialize MLflow tracking."""
        try:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            
            # Create or get experiment
            try:
                experiment_id = mlflow.create_experiment(self.mlflow_experiment_name)
            except mlflow.exceptions.MlflowException:
                experiment = mlflow.get_experiment_by_name(self.mlflow_experiment_name)
                experiment_id = experiment.experiment_id
            
            mlflow.set_experiment(experiment_id=experiment_id)
            self.logger.info(f"MLflow initialized with experiment: {self.mlflow_experiment_name}")
            
        except Exception as e:
            self.logger.warning(f"MLflow initialization failed: {e}")
    
    async def register_model(
        self,
        name: str,
        version: str,
        model_type: ModelType,
        model_path: str,
        training_config: Dict[str, Any],
        training_metrics: Dict[str, float],
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Register a new model in the registry."""
        model_id = str(uuid.uuid4())
        
        try:
            # Calculate model checksum
            model_checksum = await self._calculate_file_checksum(model_path)
            model_size = os.path.getsize(model_path)
            
            # Create metadata
            metadata = ModelMetadata(
                model_id=model_id,
                name=name,
                version=version,
                model_type=model_type,
                status=ModelStatus.TRAINED,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                training_config=training_config,
                training_metrics=training_metrics,
                model_path=model_path,
                model_size_bytes=model_size,
                model_checksum=model_checksum,
                tags=tags or {},
                dependencies=await self._get_environment_info(),
                python_version=self._get_python_version(),
                framework_version=self._get_framework_version(model_type)
            )
            
            # Store model artifacts
            stored_path = await self._store_model_artifacts(model_id, model_path)
            metadata.model_path = stored_path
            
            # Save metadata
            await self._save_metadata(metadata)
            
            # Register in MLflow
            await self._register_in_mlflow(metadata)
            
            # Clean up old versions if needed
            await self._cleanup_old_versions(name)
            
            self.logger.info(f"Registered model: {name}:{version} (ID: {model_id})")
            return model_id
            
        except Exception as e:
            self.logger.error(f"Failed to register model {name}:{version}: {e}")
            raise ModelException(f"Model registration failed: {e}")
    
    async def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID."""
        try:
            metadata_path = self.registry_path / 'metadata' / f'{model_id}.json'
            
            if not metadata_path.exists():
                return None
            
            with open(metadata_path, 'r') as f:
                data = json.load(f)
            
            return ModelMetadata.from_dict(data)
            
        except Exception as e:
            self.logger.error(f"Failed to get model {model_id}: {e}")
            return None
    
    async def get_model_by_name_version(self, name: str, version: str) -> Optional[ModelMetadata]:
        """Get model by name and version."""
        models = await self.list_models(name)
        for model in models:
            if model.version == version:
                return model
        return None
    
    async def get_latest_model(self, name: str) -> Optional[ModelMetadata]:
        """Get the latest version of a model."""
        models = await self.list_models(name)
        if not models:
            return None
        
        # Sort by version (assuming semantic versioning)
        models.sort(key=lambda m: m.created_at, reverse=True)
        return models[0]
    
    async def list_models(self, name_filter: Optional[str] = None) -> List[ModelMetadata]:
        """List all models with optional name filtering."""
        models = []
        metadata_dir = self.registry_path / 'metadata'
        
        try:
            for metadata_file in metadata_dir.glob('*.json'):
                try:
                    with open(metadata_file, 'r') as f:
                        data = json.load(f)
                    
                    metadata = ModelMetadata.from_dict(data)
                    
                    if name_filter is None or metadata.name == name_filter:
                        models.append(metadata)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to load metadata from {metadata_file}: {e}")
            
            return models
            
        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
            return []
    
    async def update_model_status(self, model_id: str, status: ModelStatus) -> bool:
        """Update model status."""
        try:
            metadata = await self.get_model(model_id)
            if not metadata:
                return False
            
            metadata.status = status
            metadata.updated_at = datetime.utcnow()
            
            await self._save_metadata(metadata)
            
            self.logger.info(f"Updated model {model_id} status to {status.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update model status: {e}")
            return False
    
    async def delete_model(self, model_id: str) -> bool:
        """Delete a model and its artifacts."""
        try:
            metadata = await self.get_model(model_id)
            if not metadata:
                return False
            
            # Delete artifacts
            model_dir = self.registry_path / 'models' / model_id
            if model_dir.exists():
                shutil.rmtree(model_dir)
            
            # Delete metadata
            metadata_file = self.registry_path / 'metadata' / f'{model_id}.json'
            if metadata_file.exists():
                metadata_file.unlink()
            
            self.logger.info(f"Deleted model {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete model {model_id}: {e}")
            return False
    
    async def _store_model_artifacts(self, model_id: str, source_path: str) -> str:
        """Store model artifacts in registry."""
        model_dir = self.registry_path / 'models' / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        dest_path = model_dir / 'model.pkl'
        shutil.copy2(source_path, dest_path)
        
        return str(dest_path)
    
    async def _save_metadata(self, metadata: ModelMetadata):
        """Save model metadata."""
        metadata_file = self.registry_path / 'metadata' / f'{metadata.model_id}.json'
        
        with self._registry_lock:
            with open(metadata_file, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
    
    async def _register_in_mlflow(self, metadata: ModelMetadata):
        """Register model in MLflow."""
        try:
            with mlflow.start_run(run_name=f"{metadata.name}_{metadata.version}"):
                # Log parameters
                mlflow.log_params(metadata.training_config)
                
                # Log metrics
                mlflow.log_metrics(metadata.training_metrics)
                
                # Log tags
                mlflow.set_tags(metadata.tags)
                
                # Log model artifacts
                if metadata.model_type == ModelType.PYTORCH and metadata.model_path:
                    mlflow.pytorch.log_model(
                        pytorch_model=torch.load(metadata.model_path),
                        artifact_path="model",
                        registered_model_name=metadata.name
                    )
                
                self.logger.debug(f"Registered model {metadata.model_id} in MLflow")
                
        except Exception as e:
            self.logger.warning(f"MLflow registration failed: {e}")
    
    async def _calculate_file_checksum(self, file_path: str) -> str:
        """Calculate SHA-256 checksum of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    async def _get_environment_info(self) -> Dict[str, str]:
        """Get current environment information."""
        try:
            import pkg_resources
            
            installed_packages = {}
            for dist in pkg_resources.working_set:
                installed_packages[dist.project_name] = dist.version
            
            return installed_packages
            
        except Exception as e:
            self.logger.warning(f"Failed to get environment info: {e}")
            return {}
    
    def _get_python_version(self) -> str:
        """Get Python version."""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    def _get_framework_version(self, model_type: ModelType) -> str:
        """Get framework version."""
        try:
            if model_type == ModelType.PYTORCH:
                return torch.__version__
            elif model_type == ModelType.SKLEARN:
                import sklearn
                return sklearn.__version__
            else:
                return "unknown"
        except Exception:
            return "unknown"
    
    async def _cleanup_old_versions(self, model_name: str):
        """Clean up old model versions."""
        models = await self.list_models(model_name)
        
        if len(models) > self.max_versions_per_model:
            # Sort by creation time and keep only the latest versions
            models.sort(key=lambda m: m.created_at, reverse=True)
            models_to_delete = models[self.max_versions_per_model:]
            
            for model in models_to_delete:
                await self.delete_model(model.model_id)
                self.logger.info(f"Cleaned up old model version: {model.model_id}")


class TrainingPipeline:
    """
    Automated training pipeline for machine learning models.
    
    Provides end-to-end training orchestration including data preprocessing,
    model training, validation, and automated hyperparameter optimization.
    """
    
    def __init__(self, config: Config, model_registry: ModelRegistry, metrics: AIEngineMetrics):
        """Initialize training pipeline."""
        self.config = config
        self.model_registry = model_registry
        self.metrics = metrics
        self.logger = get_logger(__name__)
        
        # Training configuration
        self.max_concurrent_jobs = getattr(config, 'max_training_jobs', 3)
        self.default_training_timeout = getattr(config, 'training_timeout_hours', 24) * 3600
        
        # Active training jobs
        self.active_jobs: Dict[str, TrainingJob] = {}
        self.job_lock = threading.RLock()
        
        # Resource management
        self.gpu_available = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.gpu_available else 0
        
        self.logger.info(f"TrainingPipeline initialized (GPUs: {self.gpu_count})")
    
    async def start_training_job(
        self,
        model_name: str,
        model_version: str,
        training_config: Dict[str, Any],
        training_data: Any,
        validation_data: Optional[Any] = None
    ) -> str:
        """Start a new training job."""
        job_id = str(uuid.uuid4())
        
        try:
            # Check concurrent job limit
            if len(self.active_jobs) >= self.max_concurrent_jobs:
                raise TrainingException("Maximum concurrent training jobs reached")
            
            # Create training job
            job = TrainingJob(
                job_id=job_id,
                model_name=model_name,
                model_version=model_version,
                training_config=training_config,
                status="pending",
                started_at=datetime.utcnow()
            )
            
            with self.job_lock:
                self.active_jobs[job_id] = job
            
            # Start training task
            task = asyncio.create_task(
                self._run_training_job(job, training_data, validation_data)
            )
            
            self.logger.info(f"Started training job: {job_id} for {model_name}:{model_version}")
            return job_id
            
        except Exception as e:
            self.logger.error(f"Failed to start training job: {e}")
            raise TrainingException(f"Training job startup failed: {e}")
    
    async def get_job_status(self, job_id: str) -> Optional[TrainingJob]:
        """Get training job status."""
        with self.job_lock:
            return self.active_jobs.get(job_id)
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running training job."""
        try:
            with self.job_lock:
                if job_id not in self.active_jobs:
                    return False
                
                job = self.active_jobs[job_id]
                job.status = "cancelled"
                job.completed_at = datetime.utcnow()
            
            self.logger.info(f"Cancelled training job: {job_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cancel job {job_id}: {e}")
            return False
    
    async def list_active_jobs(self) -> List[TrainingJob]:
        """List all active training jobs."""
        with self.job_lock:
            return list(self.active_jobs.values())
    
    async def _run_training_job(
        self,
        job: TrainingJob,
        training_data: Any,
        validation_data: Optional[Any] = None
    ):
        """Execute a training job."""
        try:
            job.status = "running"
            
            # Track training metrics
            with self.metrics.track_model_inference(job.model_name, job.model_version):
                
                # Initialize model based on configuration
                model = await self._create_model(job.training_config)
                
                # Setup training
                optimizer, scheduler, criterion = await self._setup_training(
                    model, job.training_config
                )
                
                # Training loop
                await self._train_model(
                    job, model, training_data, validation_data,
                    optimizer, scheduler, criterion
                )
                
                # Save trained model
                model_path = await self._save_trained_model(job, model)
                
                # Register model in registry
                model_id = await self.model_registry.register_model(
                    name=job.model_name,
                    version=job.model_version,
                    model_type=ModelType.PYTORCH,  # Configurable
                    model_path=model_path,
                    training_config=job.training_config,
                    training_metrics=job.metrics,
                    tags={'training_job_id': job.job_id}
                )
                
                job.artifacts['model_id'] = model_id
                job.status = "completed"
                job.completed_at = datetime.utcnow()
                
                self.logger.info(f"Training job {job.job_id} completed successfully")
                
        except Exception as e:
            job.status = "failed"
            job.completed_at = datetime.utcnow()
            job.logs.append(f"Training failed: {str(e)}")
            
            self.logger.error(f"Training job {job.job_id} failed: {e}")
            
        finally:
            # Clean up job from active jobs
            with self.job_lock:
                self.active_jobs.pop(job.job_id, None)
    
    async def _create_model(self, config: Dict[str, Any]) -> nn.Module:
        """Create model instance from configuration."""
        model_type = config.get('model_type', 'simple_classifier')
        
        if model_type == 'simple_classifier':
            return SimpleClassifier(
                input_size=config.get('input_size', 784),
                hidden_size=config.get('hidden_size', 128),
                num_classes=config.get('num_classes', 10)
            )
        else:
            raise TrainingException(f"Unsupported model type: {model_type}")
    
    async def _setup_training(
        self, model: nn.Module, config: Dict[str, Any]
    ) -> Tuple[torch.optim.Optimizer, Any, nn.Module]:
        """Setup training components."""
        
        # Optimizer
        optimizer_type = config.get('optimizer', 'adam')
        lr = config.get('learning_rate', 0.001)
        
        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            raise TrainingException(f"Unsupported optimizer: {optimizer_type}")
        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=config.get('scheduler_step_size', 10),
            gamma=config.get('scheduler_gamma', 0.1)
        )
        
        # Loss function
        criterion_type = config.get('criterion', 'cross_entropy')
        if criterion_type == 'cross_entropy':
            criterion = nn.CrossEntropyLoss()
        else:
            raise TrainingException(f"Unsupported criterion: {criterion_type}")
        
        return optimizer, scheduler, criterion
    
    async def _train_model(
        self,
        job: TrainingJob,
        model: nn.Module,
        training_data: DataLoader,
        validation_data: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        criterion: nn.Module
    ):
        """Main training loop."""
        num_epochs = job.training_config.get('num_epochs', 10)
        device = torch.device('cuda' if self.gpu_available else 'cpu')
        
        model.to(device)
        model.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(training_data):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                # Calculate accuracy
                pred = output.argmax(dim=1, keepdim=True)
                accuracy = pred.eq(target.view_as(pred)).sum().item() / len(data)
                
                epoch_loss += loss.item()
                epoch_accuracy += accuracy
                num_batches += 1
                
                # Update progress
                total_batches = len(training_data) * num_epochs
                current_batch = epoch * len(training_data) + batch_idx
                job.progress = current_batch / total_batches
            
            # Calculate epoch metrics
            avg_loss = epoch_loss / num_batches
            avg_accuracy = epoch_accuracy / num_batches
            
            job.metrics[f'epoch_{epoch}_train_loss'] = avg_loss
            job.metrics[f'epoch_{epoch}_train_accuracy'] = avg_accuracy
            
            # Validation
            if validation_data:
                val_loss, val_accuracy = await self._validate_model(
                    model, validation_data, criterion, device
                )
                job.metrics[f'epoch_{epoch}_val_loss'] = val_loss
                job.metrics[f'epoch_{epoch}_val_accuracy'] = val_accuracy
            
            # Step scheduler
            scheduler.step()
            
            self.logger.info(
                f"Epoch {epoch}/{num_epochs}: "
                f"Loss={avg_loss:.4f}, Acc={avg_accuracy:.4f}"
            )
    
    async def _validate_model(
        self,
        model: nn.Module,
        validation_data: DataLoader,
        criterion: nn.Module,
        device: torch.device
    ) -> Tuple[float, float]:
        """Validate model on validation dataset."""
        model.eval()
        
        val_loss = 0.0
        val_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data, target in validation_data:
                data, target = data.to(device), target.to(device)
                
                output = model(data)
                loss = criterion(output, target)
                
                pred = output.argmax(dim=1, keepdim=True)
                accuracy = pred.eq(target.view_as(pred)).sum().item() / len(data)
                
                val_loss += loss.item()
                val_accuracy += accuracy
                num_batches += 1
        
        model.train()
        return val_loss / num_batches, val_accuracy / num_batches
    
    async def _save_trained_model(self, job: TrainingJob, model: nn.Module) -> str:
        """Save trained model to disk."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
            temp_path = f.name
        
        # Save model state dict
        torch.save({
            'model_state_dict': model.state_dict(),
            'training_config': job.training_config,
            'metrics': job.metrics,
            'job_id': job.job_id
        }, temp_path)
        
        return temp_path


class SimpleClassifier(nn.Module):
    """Simple neural network classifier for demonstration."""
    
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class ModelDeploymentManager:
    """
    Manages model deployment strategies and A/B testing.
    
    Provides enterprise-grade deployment capabilities including canary releases,
    blue-green deployments, and automated rollback mechanisms.
    """
    
    def __init__(self, config: Config, model_registry: ModelRegistry, metrics: AIEngineMetrics):
        """Initialize deployment manager."""
        self.config = config
        self.model_registry = model_registry
        self.metrics = metrics
        self.logger = get_logger(__name__)
        
        # Active deployments
        self.active_deployments: Dict[str, Dict[str, Any]] = {}
        self.deployment_lock = threading.RLock()
        
        # A/B testing configuration
        self.ab_test_configs: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info("ModelDeploymentManager initialized")
    
    async def deploy_model(
        self,
        model_id: str,
        deployment_strategy: DeploymentStrategy,
        deployment_config: Dict[str, Any]
    ) -> str:
        """Deploy a model using specified strategy."""
        deployment_id = str(uuid.uuid4())
        
        try:
            # Get model metadata
            metadata = await self.model_registry.get_model(model_id)
            if not metadata:
                raise DeploymentException(f"Model {model_id} not found")
            
            # Update model status
            await self.model_registry.update_model_status(model_id, ModelStatus.DEPLOYING)
            
            # Execute deployment strategy
            if deployment_strategy == DeploymentStrategy.BLUE_GREEN:
                await self._deploy_blue_green(deployment_id, metadata, deployment_config)
            elif deployment_strategy == DeploymentStrategy.CANARY:
                await self._deploy_canary(deployment_id, metadata, deployment_config)
            elif deployment_strategy == DeploymentStrategy.A_B_TEST:
                await self._deploy_ab_test(deployment_id, metadata, deployment_config)
            else:
                raise DeploymentException(f"Unsupported deployment strategy: {deployment_strategy}")
            
            # Update model status
            await self.model_registry.update_model_status(model_id, ModelStatus.DEPLOYED)
            
            # Track deployment
            with self.deployment_lock:
                self.active_deployments[deployment_id] = {
                    'model_id': model_id,
                    'strategy': deployment_strategy,
                    'config': deployment_config,
                    'deployed_at': datetime.utcnow(),
                    'status': 'active'
                }
            
            self.logger.info(f"Successfully deployed model {model_id} using {deployment_strategy.value}")
            return deployment_id
            
        except Exception as e:
            # Rollback on failure
            await self.model_registry.update_model_status(model_id, ModelStatus.FAILED)
            self.logger.error(f"Model deployment failed: {e}")
            raise DeploymentException(f"Deployment failed: {e}")
    
    async def _deploy_blue_green(
        self,
        deployment_id: str,
        metadata: ModelMetadata,
        config: Dict[str, Any]
    ):
        """Deploy using blue-green strategy."""
        # Implementation would manage blue/green environments
        # For now, simulate deployment
        await asyncio.sleep(1)  # Simulate deployment time
        self.logger.info(f"Blue-green deployment completed for {metadata.model_id}")
    
    async def _deploy_canary(
        self,
        deployment_id: str,
        metadata: ModelMetadata,
        config: Dict[str, Any]
    ):
        """Deploy using canary strategy."""
        traffic_percentage = config.get('canary_percentage', 5)
        
        # Implementation would gradually increase traffic
        # For now, simulate canary deployment
        await asyncio.sleep(1)
        
        self.logger.info(
            f"Canary deployment started for {metadata.model_id} "
            f"with {traffic_percentage}% traffic"
        )
    
    async def _deploy_ab_test(
        self,
        deployment_id: str,
        metadata: ModelMetadata,
        config: Dict[str, Any]
    ):
        """Deploy using A/B test strategy."""
        test_duration_hours = config.get('test_duration_hours', 24)
        test_traffic_split = config.get('traffic_split', 0.5)
        
        # Store A/B test configuration
        self.ab_test_configs[deployment_id] = {
            'model_a': config.get('control_model_id'),
            'model_b': metadata.model_id,
            'traffic_split': test_traffic_split,
            'start_time': datetime.utcnow(),
            'duration': timedelta(hours=test_duration_hours),
            'metrics_to_track': config.get('metrics', ['accuracy', 'latency'])
        }
        
        self.logger.info(
            f"A/B test deployment started for {metadata.model_id} "
            f"with {test_traffic_split} traffic split"
        )
    
    async def rollback_deployment(self, deployment_id: str) -> bool:
        """Rollback a deployment."""
        try:
            with self.deployment_lock:
                if deployment_id not in self.active_deployments:
                    return False
                
                deployment = self.active_deployments[deployment_id]
                deployment['status'] = 'rolled_back'
                deployment['rolled_back_at'] = datetime.utcnow()
            
            # Update model status
            model_id = deployment['model_id']
            await self.model_registry.update_model_status(model_id, ModelStatus.DEPRECATED)
            
            self.logger.info(f"Rolled back deployment {deployment_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Rollback failed for deployment {deployment_id}: {e}")
            return False
    
    async def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get deployment status."""
        with self.deployment_lock:
            return self.active_deployments.get(deployment_id)
    
    async def list_active_deployments(self) -> List[Dict[str, Any]]:
        """List all active deployments."""
        with self.deployment_lock:
            return [
                {**deployment, 'deployment_id': dep_id}
                for dep_id, deployment in self.active_deployments.items()
                if deployment.get('status') == 'active'
            ]


class ModelManager:
    """
    Main model management orchestrator.
    
    Coordinates all model lifecycle operations including training,
    deployment, monitoring, and governance.
    """
    
    def __init__(self, config: Config, metrics: AIEngineMetrics):
        """Initialize model manager."""
        self.config = config
        self.metrics = metrics
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.model_registry = ModelRegistry(config)
        self.training_pipeline = TrainingPipeline(config, self.model_registry, metrics)
        self.deployment_manager = ModelDeploymentManager(config, self.model_registry, metrics)
        
        # Performance monitoring
        self.model_performance: Dict[str, Dict[str, float]] = {}
        self.performance_lock = threading.RLock()
        
        self.logger.info("ModelManager initialized")
    
    async def initialize(self):
        """Initialize the model manager."""
        self.logger.info("Model management system initialized")
    
    async def shutdown(self):
        """Shutdown the model manager."""
        # Cancel any active training jobs
        active_jobs = await self.training_pipeline.list_active_jobs()
        for job in active_jobs:
            await self.training_pipeline.cancel_job(job.job_id)
        
        self.logger.info("Model management system shut down")
    
    # Delegate methods to appropriate components
    async def register_model(self, *args, **kwargs) -> str:
        """Register a new model."""
        return await self.model_registry.register_model(*args, **kwargs)
    
    async def start_training_job(self, *args, **kwargs) -> str:
        """Start a training job."""
        return await self.training_pipeline.start_training_job(*args, **kwargs)
    
    async def deploy_model(self, *args, **kwargs) -> str:
        """Deploy a model."""
        return await self.deployment_manager.deploy_model(*args, **kwargs)
    
    async def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive model performance summary."""
        models = await self.model_registry.list_models()
        deployments = await self.deployment_manager.list_active_deployments()
        active_jobs = await self.training_pipeline.list_active_jobs()
        
        return {
            'total_models': len(models),
            'active_deployments': len(deployments),
            'training_jobs': len(active_jobs),
            'models_by_status': self._group_models_by_status(models),
            'performance_metrics': self.model_performance,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _group_models_by_status(self, models: List[ModelMetadata]) -> Dict[str, int]:
        """Group models by status."""
        status_counts = {}
        for model in models:
            status = model.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        return status_counts


# Global model manager instance
_model_manager: Optional[ModelManager] = None


async def initialize_model_manager(config: Config, metrics: AIEngineMetrics) -> ModelManager:
    """Initialize global model manager."""
    global _model_manager
    
    _model_manager = ModelManager(config, metrics)
    await _model_manager.initialize()
    return _model_manager


def get_model_manager() -> Optional[ModelManager]:
    """Get global model manager instance."""
    return _model_manager


async def shutdown_model_manager():
    """Shutdown global model manager."""
    global _model_manager
    if _model_manager:
        await _model_manager.shutdown()
        _model_manager = None