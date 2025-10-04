"""
CRONOS AI Engine - Model Training Infrastructure

This module provides comprehensive model training capabilities with MLflow integration,
distributed training support, and automated hyperparameter optimization.
"""

import asyncio
import logging
import time
import uuid
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Union, Callable
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import optuna
from optuna.integration.mlflow import MLflowCallback
import wandb

from ..core.config import Config, TrainingConfig
from ..core.exceptions import TrainingException, ModelException
from ..models.registry import ModelRegistry


@dataclass
class TrainingJob:
    """Training job configuration and metadata."""
    job_id: str
    model_name: str
    model_type: str
    config: TrainingConfig
    status: str = "pending"  # pending, running, completed, failed, cancelled
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    best_metric_value: Optional[float] = None
    experiment_id: Optional[str] = None
    run_id: Optional[str] = None
    artifacts_path: Optional[str] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


class ModelTrainer:
    """
    Comprehensive model training system with MLflow integration.
    
    This class provides enterprise-grade model training capabilities including
    distributed training, experiment tracking, and automated optimization.
    """
    
    def __init__(self, config: Config):
        """Initialize model trainer."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # MLflow setup
        self.mlflow_client = None
        self.experiment_name = config.mlflow.experiment_name
        self.experiment_id = None
        
        # Training state
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        
        # Job tracking
        self.active_jobs: Dict[str, TrainingJob] = {}
        
        # Model registry integration
        self.model_registry: Optional[ModelRegistry] = None
        
        # Callbacks and hooks
        self.callbacks: List[Callable] = []
        
        self.logger.info(f"ModelTrainer initialized with device: {self.device}")
    
    async def initialize(self) -> None:
        """Initialize training infrastructure."""
        try:
            # Initialize MLflow
            await self._initialize_mlflow()
            
            # Initialize distributed training if available
            await self._initialize_distributed()
            
            # Initialize model registry
            self.model_registry = ModelRegistry(self.config)
            await self.model_registry.initialize()
            
            self.logger.info("ModelTrainer initialization completed")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ModelTrainer: {e}")
            raise TrainingException(f"Trainer initialization failed: {e}")
    
    async def train_model(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
        model_name: str = "unnamed_model",
        tags: Optional[Dict[str, str]] = None
    ) -> TrainingJob:
        """
        Train a model with comprehensive tracking and optimization.
        
        Args:
            model: PyTorch model to train
            train_dataloader: Training data loader
            val_dataloader: Optional validation data loader
            config: Training configuration
            model_name: Name for the model
            tags: Optional tags for experiment tracking
            
        Returns:
            TrainingJob with results and metadata
        """
        # Use provided config or default
        train_config = config or self.config.training
        
        # Create training job
        job = TrainingJob(
            job_id=str(uuid.uuid4()),
            model_name=model_name,
            model_type=type(model).__name__,
            config=train_config,
            status="running",
            start_time=time.time()
        )
        
        self.active_jobs[job.job_id] = job
        
        try:
            # Start MLflow run
            with mlflow.start_run(experiment_id=self.experiment_id) as run:
                job.run_id = run.info.run_id
                
                # Log parameters
                mlflow.log_params({
                    "model_name": model_name,
                    "model_type": job.model_type,
                    "learning_rate": train_config.learning_rate,
                    "batch_size": train_config.batch_size,
                    "epochs": train_config.epochs,
                    "optimizer": train_config.optimizer,
                    "device": str(self.device),
                    "world_size": self.world_size
                })
                
                # Log tags
                if tags:
                    mlflow.set_tags(tags)
                
                # Prepare model for training
                model = await self._prepare_model_for_training(model)
                
                # Setup optimizer and scheduler
                optimizer = self._create_optimizer(model, train_config)
                scheduler = self._create_scheduler(optimizer, train_config)
                
                # Setup loss criterion
                criterion = self._create_loss_criterion(train_config)
                
                # Training loop
                training_history = await self._training_loop(
                    model=model,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    criterion=criterion,
                    config=train_config,
                    job=job
                )
                
                # Save best model
                best_model_path = await self._save_best_model(model, job)
                
                # Log model to MLflow
                mlflow.pytorch.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name=model_name
                )
                
                # Update job status
                job.status = "completed"
                job.end_time = time.time()
                job.artifacts_path = best_model_path
                job.best_metric_value = training_history.get("best_val_loss")
                
                # Log final metrics
                mlflow.log_metrics({
                    "final_train_loss": training_history["train_loss"][-1],
                    "best_val_loss": training_history.get("best_val_loss", 0.0),
                    "training_time": job.end_time - job.start_time,
                    "total_epochs": len(training_history["train_loss"])
                })
                
                self.logger.info(f"Training completed for job {job.job_id}")
                
                return job
                
        except Exception as e:
            job.status = "failed"
            job.end_time = time.time()
            job.error_message = str(e)
            
            self.logger.error(f"Training failed for job {job.job_id}: {e}")
            raise TrainingException(f"Training failed: {e}")
        
        finally:
            # Cleanup
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
    
    async def hyperparameter_optimization(
        self,
        model_factory: Callable,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        search_space: Dict[str, Any],
        n_trials: int = 50,
        model_name: str = "optimized_model"
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter optimization using Optuna.
        
        Args:
            model_factory: Function that creates model instances
            train_dataloader: Training data
            val_dataloader: Validation data
            search_space: Hyperparameter search space
            n_trials: Number of optimization trials
            model_name: Base name for models
            
        Returns:
            Best hyperparameters and optimization results
        """
        self.logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
        
        # Create study
        study = optuna.create_study(
            direction="minimize",
            study_name=f"{model_name}_optimization",
            storage=None  # In-memory study
        )
        
        # MLflow callback for Optuna
        mlflow_callback = MLflowCallback(
            tracking_uri=self.config.mlflow.tracking_uri,
            metric_name="val_loss"
        )
        
        def objective(trial):
            """Objective function for optimization."""
            # Sample hyperparameters
            params = {}
            for param_name, param_config in search_space.items():
                if param_config["type"] == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config["choices"]
                    )
                elif param_config["type"] == "uniform":
                    params[param_name] = trial.suggest_uniform(
                        param_name, param_config["low"], param_config["high"]
                    )
                elif param_config["type"] == "int":
                    params[param_name] = trial.suggest_int(
                        param_name, param_config["low"], param_config["high"]
                    )
                elif param_config["type"] == "loguniform":
                    params[param_name] = trial.suggest_loguniform(
                        param_name, param_config["low"], param_config["high"]
                    )
            
            # Create model with sampled parameters
            model = model_factory(**params)
            
            # Create training config
            train_config = TrainingConfig(
                learning_rate=params.get("learning_rate", 1e-3),
                batch_size=params.get("batch_size", 32),
                epochs=params.get("epochs", 10),  # Reduced for optimization
                early_stopping_patience=5
            )
            
            # Train model
            try:
                job = asyncio.run(self.train_model(
                    model=model,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    config=train_config,
                    model_name=f"{model_name}_trial_{trial.number}",
                    tags={"optimization_trial": str(trial.number)}
                ))
                
                return job.best_metric_value or float('inf')
                
            except Exception as e:
                self.logger.warning(f"Trial {trial.number} failed: {e}")
                return float('inf')
        
        # Run optimization
        study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=[mlflow_callback]
        )
        
        best_params = study.best_params
        best_value = study.best_value
        
        self.logger.info(f"Optimization completed. Best value: {best_value}")
        self.logger.info(f"Best parameters: {best_params}")
        
        return {
            "best_params": best_params,
            "best_value": best_value,
            "study": study,
            "n_trials": n_trials
        }
    
    async def distributed_training(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
        model_name: str = "distributed_model"
    ) -> TrainingJob:
        """
        Perform distributed training across multiple GPUs/nodes.
        
        Args:
            model: Model to train
            train_dataloader: Training data
            val_dataloader: Validation data
            config: Training configuration
            model_name: Model name
            
        Returns:
            Training job results
        """
        if not torch.distributed.is_available() or self.world_size == 1:
            self.logger.warning("Distributed training not available, falling back to single GPU")
            return await self.train_model(model, train_dataloader, val_dataloader, config, model_name)
        
        self.logger.info(f"Starting distributed training on {self.world_size} processes")
        
        # Wrap model for distributed training
        if torch.cuda.is_available():
            model = model.cuda(self.local_rank)
            model = DDP(model, device_ids=[self.local_rank])
        else:
            model = DDP(model)
        
        # Use distributed sampler
        if hasattr(train_dataloader.sampler, 'set_epoch'):
            # Already has distributed sampler
            pass
        else:
            # Create distributed sampler
            train_sampler = DistributedSampler(
                train_dataloader.dataset,
                num_replicas=self.world_size,
                rank=self.rank
            )
            
            train_dataloader = DataLoader(
                train_dataloader.dataset,
                batch_size=train_dataloader.batch_size,
                sampler=train_sampler,
                num_workers=train_dataloader.num_workers,
                pin_memory=train_dataloader.pin_memory
            )
        
        # Train model (only log on rank 0)
        if self.rank == 0:
            return await self.train_model(model, train_dataloader, val_dataloader, config, model_name)
        else:
            # Non-master ranks just participate in training
            job = TrainingJob(
                job_id=str(uuid.uuid4()),
                model_name=model_name,
                model_type=type(model).__name__,
                config=config or self.config.training
            )
            
            # Participate in training without logging
            await self._training_loop_distributed(
                model, train_dataloader, val_dataloader, config or self.config.training, job
            )
            
            return job
    
    async def resume_training(
        self,
        checkpoint_path: str,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        additional_epochs: int = 10
    ) -> TrainingJob:
        """
        Resume training from a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            train_dataloader: Training data
            val_dataloader: Validation data
            additional_epochs: Additional epochs to train
            
        Returns:
            Training job results
        """
        self.logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Reconstruct model (this would need model factory)
        # For now, assume checkpoint contains everything needed
        model = checkpoint['model']
        optimizer_state = checkpoint.get('optimizer_state_dict')
        scheduler_state = checkpoint.get('scheduler_state_dict')
        epoch = checkpoint.get('epoch', 0)
        
        # Create training config with additional epochs
        config = TrainingConfig(
            epochs=epoch + additional_epochs,
            learning_rate=checkpoint.get('learning_rate', 1e-3),
            batch_size=train_dataloader.batch_size
        )
        
        # Resume training
        job = await self.train_model(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            config=config,
            model_name=f"resumed_{checkpoint.get('model_name', 'model')}"
        )
        
        return job
    
    def get_training_status(self, job_id: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Get status of training jobs."""
        if job_id:
            if job_id in self.active_jobs:
                return self.active_jobs[job_id].to_dict()
            else:
                return {"error": f"Job {job_id} not found"}
        else:
            return [job.to_dict() for job in self.active_jobs.values()]
    
    async def cancel_training(self, job_id: str) -> bool:
        """Cancel a running training job."""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            job.status = "cancelled"
            job.end_time = time.time()
            
            self.logger.info(f"Training job {job_id} cancelled")
            return True
        
        return False
    
    # Private methods
    
    async def _initialize_mlflow(self) -> None:
        """Initialize MLflow tracking."""
        try:
            mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)
            
            # Create or get experiment
            try:
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                if experiment is None:
                    self.experiment_id = mlflow.create_experiment(self.experiment_name)
                else:
                    self.experiment_id = experiment.experiment_id
            except Exception:
                self.experiment_id = mlflow.create_experiment(self.experiment_name)
            
            self.mlflow_client = MlflowClient(self.config.mlflow.tracking_uri)
            
            self.logger.info(f"MLflow initialized with experiment: {self.experiment_name}")
            
        except Exception as e:
            self.logger.warning(f"MLflow initialization failed: {e}")
            # Continue without MLflow
            self.mlflow_client = None
    
    async def _initialize_distributed(self) -> None:
        """Initialize distributed training if available."""
        try:
            if torch.distributed.is_available() and torch.cuda.device_count() > 1:
                # Initialize process group
                if not torch.distributed.is_initialized():
                    torch.distributed.init_process_group(
                        backend='nccl' if torch.cuda.is_available() else 'gloo',
                        init_method='env://',
                        world_size=torch.cuda.device_count(),
                        rank=0  # This would be set from environment
                    )
                
                self.world_size = torch.distributed.get_world_size()
                self.rank = torch.distributed.get_rank()
                self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
                
                self.logger.info(f"Distributed training initialized: world_size={self.world_size}, rank={self.rank}")
            
        except Exception as e:
            self.logger.warning(f"Distributed training initialization failed: {e}")
    
    async def _prepare_model_for_training(self, model: nn.Module) -> nn.Module:
        """Prepare model for training."""
        model = model.to(self.device)
        
        if torch.cuda.device_count() > 1 and self.world_size == 1:
            # Multi-GPU training on single node
            model = nn.DataParallel(model)
            self.logger.info(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        
        return model
    
    def _create_optimizer(self, model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        if config.optimizer.lower() == "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer.lower() == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer.lower() == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                momentum=0.9,
                weight_decay=config.weight_decay
            )
        else:
            raise TrainingException(f"Unsupported optimizer: {config.optimizer}")
    
    def _create_scheduler(
        self, 
        optimizer: torch.optim.Optimizer, 
        config: TrainingConfig
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        if config.lr_scheduler.lower() == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=config.epochs
            )
        elif config.lr_scheduler.lower() == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.epochs
            )
        elif config.lr_scheduler.lower() == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.epochs // 3,
                gamma=0.1
            )
        else:
            return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    
    def _create_loss_criterion(self, config: TrainingConfig) -> nn.Module:
        """Create loss criterion based on model type."""
        # This would be more sophisticated in practice
        return nn.CrossEntropyLoss()
    
    async def _training_loop(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        criterion: nn.Module,
        config: TrainingConfig,
        job: TrainingJob
    ) -> Dict[str, List[float]]:
        """Main training loop."""
        history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config.epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            num_batches = 0
            
            for batch_idx, (data, targets) in enumerate(train_dataloader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                
                # Gradient clipping
                if config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                
                optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
                
                # Log batch metrics
                if batch_idx % config.logging_steps == 0:
                    self.logger.debug(
                        f"Epoch {epoch+1}/{config.epochs}, "
                        f"Batch {batch_idx}/{len(train_dataloader)}, "
                        f"Loss: {loss.item():.4f}"
                    )
                    
                    if self.mlflow_client:
                        mlflow.log_metric("batch_loss", loss.item())
            
            avg_train_loss = train_loss / num_batches
            history["train_loss"].append(avg_train_loss)
            
            # Validation phase
            if val_dataloader:
                val_loss = await self._validate(model, val_dataloader, criterion)
                history["val_loss"].append(val_loss)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model checkpoint
                    await self._save_checkpoint(model, optimizer, scheduler, epoch, job)
                    
                else:
                    patience_counter += 1
                
                if patience_counter >= config.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Update learning rate
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            history["learning_rate"].append(current_lr)
            
            # Log epoch metrics
            self.logger.info(
                f"Epoch {epoch+1}/{config.epochs}: "
                f"train_loss={avg_train_loss:.4f}, "
                f"val_loss={val_loss:.4f if val_dataloader else 'N/A'}, "
                f"lr={current_lr:.6f}"
            )
            
            if self.mlflow_client:
                mlflow.log_metrics({
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss if val_dataloader else 0.0,
                    "learning_rate": current_lr
                })
        
        # Store best validation loss in history
        if val_dataloader:
            history["best_val_loss"] = best_val_loss
        
        return history
    
    async def _validate(
        self,
        model: nn.Module,
        val_dataloader: DataLoader,
        criterion: nn.Module
    ) -> float:
        """Validation loop."""
        model.eval()
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data, targets in val_dataloader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                num_batches += 1
        
        return val_loss / num_batches if num_batches > 0 else 0.0
    
    async def _save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        epoch: int,
        job: TrainingJob
    ) -> None:
        """Save training checkpoint."""
        checkpoint_path = f"checkpoints/{job.job_id}_epoch_{epoch}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'job_id': job.job_id,
            'model_name': job.model_name
        }
        
        # Ensure checkpoint directory exists
        Path("checkpoints").mkdir(exist_ok=True)
        
        torch.save(checkpoint, checkpoint_path)
        
        if self.mlflow_client:
            mlflow.log_artifact(checkpoint_path)
    
    async def _save_best_model(self, model: nn.Module, job: TrainingJob) -> str:
        """Save the best model."""
        model_path = f"models/{job.model_name}_{job.job_id}.pt"
        
        # Ensure models directory exists
        Path("models").mkdir(exist_ok=True)
        
        # Save model
        if isinstance(model, (nn.DataParallel, DDP)):
            torch.save(model.module.state_dict(), model_path)
        else:
            torch.save(model.state_dict(), model_path)
        
        return model_path
    
    async def _training_loop_distributed(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        config: TrainingConfig,
        job: TrainingJob
    ) -> None:
        """Distributed training loop for non-master ranks."""
        # Simplified distributed training participation
        # In practice, this would coordinate with the master rank
        
        optimizer = self._create_optimizer(model, config)
        criterion = self._create_loss_criterion(config)
        
        for epoch in range(config.epochs):
            model.train()
            
            for data, targets in train_dataloader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            # Synchronization points would be here in real distributed training
            torch.distributed.barrier()