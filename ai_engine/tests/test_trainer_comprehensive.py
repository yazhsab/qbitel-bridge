"""
Comprehensive Unit Tests for ModelTrainer
Tests for ai_engine/training/trainer.py
"""

import pytest
import asyncio
import torch
import torch.nn as nn
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import tempfile
import time
from datetime import datetime

from ai_engine.training.trainer import (
    ModelTrainer,
    TrainingJob,
)
from ai_engine.core.config import Config, TrainingConfig
from ai_engine.core.exceptions import TrainingException, ModelException


class SimpleTestModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


class TestTrainingJob:
    """Test TrainingJob dataclass."""

    def test_training_job_creation(self):
        """Test creating a training job."""
        config = TrainingConfig(learning_rate=0.001, batch_size=32, epochs=10)

        job = TrainingJob(
            job_id="test_job_123",
            model_name="test_model",
            model_type="SimpleTestModel",
            config=config,
            status="pending",
        )

        assert job.job_id == "test_job_123"
        assert job.model_name == "test_model"
        assert job.model_type == "SimpleTestModel"
        assert job.status == "pending"
        assert job.start_time is None
        assert job.end_time is None

    def test_training_job_to_dict(self):
        """Test converting training job to dictionary."""
        config = TrainingConfig(learning_rate=0.001, batch_size=32, epochs=10)

        job = TrainingJob(
            job_id="test_job_123",
            model_name="test_model",
            model_type="SimpleTestModel",
            config=config,
            status="running",
            start_time=time.time(),
        )

        job_dict = job.to_dict()

        assert isinstance(job_dict, dict)
        assert job_dict["job_id"] == "test_job_123"
        assert job_dict["model_name"] == "test_model"
        assert job_dict["status"] == "running"
        assert "start_time" in job_dict


class TestModelTrainer:
    """Test ModelTrainer class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Mock(spec=Config)
        config.mlflow = Mock()
        config.mlflow.experiment_name = "test_experiment"
        config.mlflow.tracking_uri = "file:///tmp/mlruns"
        config.training = TrainingConfig(
            learning_rate=0.001,
            batch_size=32,
            epochs=5,
            optimizer="adamw",
            lr_scheduler="cosine",
        )
        return config

    @pytest.fixture
    def trainer(self, config):
        """Create ModelTrainer instance."""
        return ModelTrainer(config)

    def test_trainer_initialization(self, trainer):
        """Test trainer initialization."""
        assert trainer.config is not None
        assert trainer.device is not None
        assert trainer.world_size == 1
        assert trainer.rank == 0
        assert isinstance(trainer.active_jobs, dict)
        assert len(trainer.active_jobs) == 0

    @pytest.mark.asyncio
    async def test_initialize_trainer(self, trainer):
        """Test trainer initialization."""
        with patch.object(trainer, "_initialize_mlflow", new_callable=AsyncMock):
            with patch.object(
                trainer, "_initialize_distributed", new_callable=AsyncMock
            ):
                with patch("ai_engine.training.trainer.ModelRegistry") as mock_registry:
                    mock_registry_instance = AsyncMock()
                    mock_registry.return_value = mock_registry_instance

                    await trainer.initialize()

                    assert trainer.model_registry is not None

    @pytest.mark.asyncio
    async def test_train_model_basic(self, trainer):
        """Test basic model training."""
        model = SimpleTestModel()

        # Create dummy data
        train_data = torch.randn(100, 10)
        train_targets = torch.randint(0, 2, (100,))
        train_dataset = torch.utils.data.TensorDataset(train_data, train_targets)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)

        with patch("mlflow.start_run"):
            with patch("mlflow.log_params"):
                with patch("mlflow.log_metrics"):
                    with patch("mlflow.pytorch.log_model"):
                        with patch.object(
                            trainer, "_prepare_model_for_training", return_value=model
                        ):
                            with patch.object(
                                trainer, "_training_loop", new_callable=AsyncMock
                            ) as mock_loop:
                                mock_loop.return_value = {
                                    "train_loss": [0.5, 0.4, 0.3],
                                    "best_val_loss": 0.3,
                                }
                                with patch.object(
                                    trainer, "_save_best_model", new_callable=AsyncMock
                                ) as mock_save:
                                    mock_save.return_value = "/tmp/model.pt"

                                    job = await trainer.train_model(
                                        model=model,
                                        train_dataloader=train_loader,
                                        model_name="test_model",
                                    )

                                    assert job.status == "completed"
                                    assert job.model_name == "test_model"
                                    assert job.end_time is not None

    @pytest.mark.asyncio
    async def test_train_model_with_validation(self, trainer):
        """Test model training with validation data."""
        model = SimpleTestModel()

        # Create dummy data
        train_data = torch.randn(100, 10)
        train_targets = torch.randint(0, 2, (100,))
        train_dataset = torch.utils.data.TensorDataset(train_data, train_targets)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)

        val_data = torch.randn(20, 10)
        val_targets = torch.randint(0, 2, (20,))
        val_dataset = torch.utils.data.TensorDataset(val_data, val_targets)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

        with patch("mlflow.start_run"):
            with patch("mlflow.log_params"):
                with patch("mlflow.log_metrics"):
                    with patch("mlflow.pytorch.log_model"):
                        with patch.object(
                            trainer, "_prepare_model_for_training", return_value=model
                        ):
                            with patch.object(
                                trainer, "_training_loop", new_callable=AsyncMock
                            ) as mock_loop:
                                mock_loop.return_value = {
                                    "train_loss": [0.5, 0.4],
                                    "val_loss": [0.6, 0.5],
                                    "best_val_loss": 0.5,
                                }
                                with patch.object(
                                    trainer, "_save_best_model", new_callable=AsyncMock
                                ) as mock_save:
                                    mock_save.return_value = "/tmp/model.pt"

                                    job = await trainer.train_model(
                                        model=model,
                                        train_dataloader=train_loader,
                                        val_dataloader=val_loader,
                                        model_name="test_model",
                                    )

                                    assert job.status == "completed"
                                    assert job.best_metric_value == 0.5

    @pytest.mark.asyncio
    async def test_train_model_failure(self, trainer):
        """Test model training failure handling."""
        model = SimpleTestModel()

        train_data = torch.randn(100, 10)
        train_targets = torch.randint(0, 2, (100,))
        train_dataset = torch.utils.data.TensorDataset(train_data, train_targets)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)

        with patch("mlflow.start_run"):
            with patch.object(
                trainer,
                "_prepare_model_for_training",
                side_effect=Exception("Training failed"),
            ):
                with pytest.raises(TrainingException):
                    await trainer.train_model(
                        model=model,
                        train_dataloader=train_loader,
                        model_name="test_model",
                    )

    def test_create_optimizer_adamw(self, trainer):
        """Test creating AdamW optimizer."""
        model = SimpleTestModel()
        config = TrainingConfig(
            learning_rate=0.001, optimizer="adamw", weight_decay=0.01
        )

        optimizer = trainer._create_optimizer(model, config)

        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]["lr"] == 0.001

    def test_create_optimizer_adam(self, trainer):
        """Test creating Adam optimizer."""
        model = SimpleTestModel()
        config = TrainingConfig(learning_rate=0.001, optimizer="adam")

        optimizer = trainer._create_optimizer(model, config)

        assert isinstance(optimizer, torch.optim.Adam)

    def test_create_optimizer_sgd(self, trainer):
        """Test creating SGD optimizer."""
        model = SimpleTestModel()
        config = TrainingConfig(learning_rate=0.001, optimizer="sgd")

        optimizer = trainer._create_optimizer(model, config)

        assert isinstance(optimizer, torch.optim.SGD)

    def test_create_optimizer_unsupported(self, trainer):
        """Test creating unsupported optimizer."""
        model = SimpleTestModel()
        config = TrainingConfig(learning_rate=0.001, optimizer="unsupported")

        with pytest.raises(TrainingException):
            trainer._create_optimizer(model, config)

    def test_create_scheduler_cosine(self, trainer):
        """Test creating cosine scheduler."""
        model = SimpleTestModel()
        optimizer = torch.optim.Adam(model.parameters())
        config = TrainingConfig(epochs=10, lr_scheduler="cosine")

        scheduler = trainer._create_scheduler(optimizer, config)

        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_create_scheduler_linear(self, trainer):
        """Test creating linear scheduler."""
        model = SimpleTestModel()
        optimizer = torch.optim.Adam(model.parameters())
        config = TrainingConfig(epochs=10, lr_scheduler="linear")

        scheduler = trainer._create_scheduler(optimizer, config)

        assert isinstance(scheduler, torch.optim.lr_scheduler.LinearLR)

    def test_create_scheduler_step(self, trainer):
        """Test creating step scheduler."""
        model = SimpleTestModel()
        optimizer = torch.optim.Adam(model.parameters())
        config = TrainingConfig(epochs=10, lr_scheduler="step")

        scheduler = trainer._create_scheduler(optimizer, config)

        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)

    def test_create_loss_criterion(self, trainer):
        """Test creating loss criterion."""
        config = TrainingConfig()

        criterion = trainer._create_loss_criterion(config)

        assert isinstance(criterion, nn.Module)

    def test_get_training_status_single_job(self, trainer):
        """Test getting status of single training job."""
        job = TrainingJob(
            job_id="test_123",
            model_name="test_model",
            model_type="SimpleModel",
            config=TrainingConfig(),
            status="running",
        )
        trainer.active_jobs["test_123"] = job

        status = trainer.get_training_status("test_123")

        assert isinstance(status, dict)
        assert status["job_id"] == "test_123"
        assert status["status"] == "running"

    def test_get_training_status_all_jobs(self, trainer):
        """Test getting status of all training jobs."""
        job1 = TrainingJob(
            job_id="test_1",
            model_name="model1",
            model_type="SimpleModel",
            config=TrainingConfig(),
            status="running",
        )
        job2 = TrainingJob(
            job_id="test_2",
            model_name="model2",
            model_type="SimpleModel",
            config=TrainingConfig(),
            status="completed",
        )
        trainer.active_jobs["test_1"] = job1
        trainer.active_jobs["test_2"] = job2

        statuses = trainer.get_training_status()

        assert isinstance(statuses, list)
        assert len(statuses) == 2

    def test_get_training_status_not_found(self, trainer):
        """Test getting status of non-existent job."""
        status = trainer.get_training_status("nonexistent")

        assert "error" in status

    @pytest.mark.asyncio
    async def test_cancel_training(self, trainer):
        """Test cancelling a training job."""
        job = TrainingJob(
            job_id="test_123",
            model_name="test_model",
            model_type="SimpleModel",
            config=TrainingConfig(),
            status="running",
            start_time=time.time(),
        )
        trainer.active_jobs["test_123"] = job

        result = await trainer.cancel_training("test_123")

        assert result is True
        assert job.status == "cancelled"
        assert job.end_time is not None

    @pytest.mark.asyncio
    async def test_cancel_training_not_found(self, trainer):
        """Test cancelling non-existent training job."""
        result = await trainer.cancel_training("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_validate_epoch(self, trainer):
        """Test validation epoch."""
        model = SimpleTestModel()
        criterion = nn.CrossEntropyLoss()

        val_data = torch.randn(20, 10)
        val_targets = torch.randint(0, 2, (20,))
        val_dataset = torch.utils.data.TensorDataset(val_data, val_targets)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=10)

        val_loss = await trainer._validate(model, val_loader, criterion)

        assert isinstance(val_loss, float)
        assert val_loss >= 0

    @pytest.mark.asyncio
    async def test_save_checkpoint(self, trainer):
        """Test saving training checkpoint."""
        model = SimpleTestModel()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        job = TrainingJob(
            job_id="test_123",
            model_name="test_model",
            model_type="SimpleModel",
            config=TrainingConfig(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.mkdir"):
                with patch("torch.save") as mock_save:
                    await trainer._save_checkpoint(model, optimizer, scheduler, 5, job)

                    assert mock_save.called

    @pytest.mark.asyncio
    async def test_save_best_model(self, trainer):
        """Test saving best model."""
        model = SimpleTestModel()

        job = TrainingJob(
            job_id="test_123",
            model_name="test_model",
            model_type="SimpleModel",
            config=TrainingConfig(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.mkdir"):
                with patch("torch.save") as mock_save:
                    model_path = await trainer._save_best_model(model, job)

                    assert mock_save.called
                    assert "test_model" in model_path

    @pytest.mark.asyncio
    async def test_hyperparameter_optimization(self, trainer):
        """Test hyperparameter optimization."""

        def model_factory(**kwargs):
            return SimpleTestModel()

        train_data = torch.randn(50, 10)
        train_targets = torch.randint(0, 2, (50,))
        train_dataset = torch.utils.data.TensorDataset(train_data, train_targets)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16)

        val_data = torch.randn(20, 10)
        val_targets = torch.randint(0, 2, (20,))
        val_dataset = torch.utils.data.TensorDataset(val_data, val_targets)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16)

        search_space = {
            "learning_rate": {"type": "loguniform", "low": 1e-4, "high": 1e-2},
            "batch_size": {"type": "categorical", "choices": [16, 32]},
        }

        with patch("optuna.create_study") as mock_study:
            mock_study_instance = Mock()
            mock_study_instance.best_params = {"learning_rate": 0.001, "batch_size": 32}
            mock_study_instance.best_value = 0.5
            mock_study.return_value = mock_study_instance

            result = await trainer.hyperparameter_optimization(
                model_factory=model_factory,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                search_space=search_space,
                n_trials=2,
                model_name="optimized_model",
            )

            assert "best_params" in result
            assert "best_value" in result
            assert result["n_trials"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
