"""
Tests for ai_engine/training/trainer.py
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path
import numpy as np


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_dim=10, output_dim=2):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


class TestTrainingJob:
    """Test suite for TrainingJob dataclass."""

    def test_training_job_creation(self):
        """Test TrainingJob creation."""
        from ai_engine.training.trainer import TrainingJob
        from ai_engine.core.config import TrainingConfig

        config = TrainingConfig()
        job = TrainingJob(
            job_id="job123",
            model_name="test_model",
            model_type="SimpleModel",
            config=config,
            status="pending",
        )

        assert job.job_id == "job123"
        assert job.model_name == "test_model"
        assert job.status == "pending"

    def test_training_job_to_dict(self):
        """Test TrainingJob to_dict conversion."""
        from ai_engine.training.trainer import TrainingJob
        from ai_engine.core.config import TrainingConfig

        config = TrainingConfig()
        job = TrainingJob(
            job_id="job123",
            model_name="test_model",
            model_type="SimpleModel",
            config=config,
        )

        job_dict = job.to_dict()

        assert isinstance(job_dict, dict)
        assert job_dict["job_id"] == "job123"
        assert job_dict["model_name"] == "test_model"


class TestModelTrainer:
    """Test suite for ModelTrainer class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = Mock()
        config.mlflow = Mock()
        config.mlflow.experiment_name = "test_experiment"
        config.mlflow.tracking_uri = "file:///tmp/mlflow"
        config.training = Mock()
        config.training.learning_rate = 1e-3
        config.training.batch_size = 32
        config.training.epochs = 10
        config.training.optimizer = "adamw"
        config.training.lr_scheduler = "linear"
        config.training.weight_decay = 1e-5
        config.training.gradient_clip_norm = 1.0
        config.training.logging_steps = 10
        config.training.early_stopping_patience = 5
        return config

    @pytest.fixture
    def trainer(self, mock_config):
        """Create ModelTrainer instance."""
        from ai_engine.training.trainer import ModelTrainer

        return ModelTrainer(mock_config)

    def test_trainer_initialization(self, trainer):
        """Test trainer initialization."""
        assert trainer.device is not None
        assert trainer.world_size == 1
        assert trainer.rank == 0

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
    async def test_initialize_trainer_failure(self, trainer):
        """Test trainer initialization failure."""
        from ai_engine.core.exceptions import TrainingException

        with patch.object(
            trainer, "_initialize_mlflow", side_effect=Exception("MLflow error")
        ):
            with pytest.raises(TrainingException):
                await trainer.initialize()

    @pytest.mark.asyncio
    async def test_train_model_basic(self, trainer, mock_config):
        """Test basic model training."""
        model = SimpleModel()

        # Create dummy data
        train_data = torch.randn(100, 10)
        train_labels = torch.randint(0, 2, (100,))
        train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)

        with patch("mlflow.start_run"):
            with patch("mlflow.log_params"):
                with patch("mlflow.log_metrics"):
                    with patch("mlflow.pytorch.log_model"):
                        with patch.object(
                            trainer, "_prepare_model_for_training", return_value=model
                        ):
                            with patch.object(
                                trainer,
                                "_training_loop",
                                return_value={
                                    "train_loss": [0.5],
                                    "best_val_loss": 0.4,
                                },
                            ):
                                with patch.object(
                                    trainer,
                                    "_save_best_model",
                                    return_value="/tmp/model.pt",
                                ):
                                    job = await trainer.train_model(
                                        model=model,
                                        train_dataloader=train_loader,
                                        model_name="test_model",
                                    )

        assert job.status == "completed"
        assert job.model_name == "test_model"

    @pytest.mark.asyncio
    async def test_train_model_with_validation(self, trainer):
        """Test model training with validation data."""
        model = SimpleModel()

        train_data = torch.randn(100, 10)
        train_labels = torch.randint(0, 2, (100,))
        train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)

        val_data = torch.randn(20, 10)
        val_labels = torch.randint(0, 2, (20,))
        val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

        with patch("mlflow.start_run"):
            with patch("mlflow.log_params"):
                with patch("mlflow.log_metrics"):
                    with patch("mlflow.pytorch.log_model"):
                        with patch.object(
                            trainer, "_prepare_model_for_training", return_value=model
                        ):
                            with patch.object(
                                trainer,
                                "_training_loop",
                                return_value={
                                    "train_loss": [0.5],
                                    "val_loss": [0.4],
                                    "best_val_loss": 0.4,
                                },
                            ):
                                with patch.object(
                                    trainer,
                                    "_save_best_model",
                                    return_value="/tmp/model.pt",
                                ):
                                    job = await trainer.train_model(
                                        model=model,
                                        train_dataloader=train_loader,
                                        val_dataloader=val_loader,
                                        model_name="test_model",
                                    )

        assert job.status == "completed"

    @pytest.mark.asyncio
    async def test_train_model_failure(self, trainer):
        """Test model training failure."""
        from ai_engine.core.exceptions import TrainingException

        model = SimpleModel()
        train_data = torch.randn(100, 10)
        train_labels = torch.randint(0, 2, (100,))
        train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)

        with patch("mlflow.start_run"):
            with patch.object(
                trainer,
                "_prepare_model_for_training",
                side_effect=Exception("Training error"),
            ):
                with pytest.raises(TrainingException):
                    await trainer.train_model(
                        model=model,
                        train_dataloader=train_loader,
                        model_name="test_model",
                    )

    def test_create_optimizer_adamw(self, trainer, mock_config):
        """Test AdamW optimizer creation."""
        model = SimpleModel()
        optimizer = trainer._create_optimizer(model, mock_config.training)

        assert isinstance(optimizer, torch.optim.AdamW)

    def test_create_optimizer_adam(self, trainer, mock_config):
        """Test Adam optimizer creation."""
        model = SimpleModel()
        mock_config.training.optimizer = "adam"
        optimizer = trainer._create_optimizer(model, mock_config.training)

        assert isinstance(optimizer, torch.optim.Adam)

    def test_create_optimizer_sgd(self, trainer, mock_config):
        """Test SGD optimizer creation."""
        model = SimpleModel()
        mock_config.training.optimizer = "sgd"
        optimizer = trainer._create_optimizer(model, mock_config.training)

        assert isinstance(optimizer, torch.optim.SGD)

    def test_create_optimizer_unsupported(self, trainer, mock_config):
        """Test unsupported optimizer."""
        from ai_engine.core.exceptions import TrainingException

        model = SimpleModel()
        mock_config.training.optimizer = "unsupported"

        with pytest.raises(TrainingException):
            trainer._create_optimizer(model, mock_config.training)

    def test_create_scheduler_linear(self, trainer, mock_config):
        """Test linear scheduler creation."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = trainer._create_scheduler(optimizer, mock_config.training)

        assert isinstance(scheduler, torch.optim.lr_scheduler.LinearLR)

    def test_create_scheduler_cosine(self, trainer, mock_config):
        """Test cosine scheduler creation."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        mock_config.training.lr_scheduler = "cosine"
        scheduler = trainer._create_scheduler(optimizer, mock_config.training)

        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_create_scheduler_step(self, trainer, mock_config):
        """Test step scheduler creation."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        mock_config.training.lr_scheduler = "step"
        scheduler = trainer._create_scheduler(optimizer, mock_config.training)

        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)

    def test_create_loss_criterion(self, trainer, mock_config):
        """Test loss criterion creation."""
        criterion = trainer._create_loss_criterion(mock_config.training)

        assert isinstance(criterion, nn.Module)

    @pytest.mark.asyncio
    async def test_prepare_model_for_training(self, trainer):
        """Test model preparation for training."""
        model = SimpleModel()
        prepared_model = await trainer._prepare_model_for_training(model)

        assert prepared_model is not None

    def test_get_training_status_single_job(self, trainer):
        """Test getting status of single training job."""
        from ai_engine.training.trainer import TrainingJob
        from ai_engine.core.config import TrainingConfig

        job = TrainingJob(
            job_id="job123",
            model_name="test",
            model_type="SimpleModel",
            config=TrainingConfig(),
            status="running",
        )
        trainer.active_jobs["job123"] = job

        status = trainer.get_training_status("job123")

        assert status["job_id"] == "job123"
        assert status["status"] == "running"

    def test_get_training_status_all_jobs(self, trainer):
        """Test getting status of all training jobs."""
        from ai_engine.training.trainer import TrainingJob
        from ai_engine.core.config import TrainingConfig

        job1 = TrainingJob(
            job_id="job1",
            model_name="model1",
            model_type="SimpleModel",
            config=TrainingConfig(),
        )
        job2 = TrainingJob(
            job_id="job2",
            model_name="model2",
            model_type="SimpleModel",
            config=TrainingConfig(),
        )

        trainer.active_jobs["job1"] = job1
        trainer.active_jobs["job2"] = job2

        statuses = trainer.get_training_status()

        assert len(statuses) == 2

    def test_get_training_status_not_found(self, trainer):
        """Test getting status of non-existent job."""
        status = trainer.get_training_status("nonexistent")

        assert "error" in status

    @pytest.mark.asyncio
    async def test_cancel_training(self, trainer):
        """Test cancelling training job."""
        from ai_engine.training.trainer import TrainingJob
        from ai_engine.core.config import TrainingConfig

        job = TrainingJob(
            job_id="job123",
            model_name="test",
            model_type="SimpleModel",
            config=TrainingConfig(),
            status="running",
        )
        trainer.active_jobs["job123"] = job

        result = await trainer.cancel_training("job123")

        assert result is True
        assert job.status == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_training_not_found(self, trainer):
        """Test cancelling non-existent job."""
        result = await trainer.cancel_training("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_save_checkpoint(self, trainer, mock_config, tmp_path):
        """Test saving training checkpoint."""
        from ai_engine.training.trainer import TrainingJob

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        job = TrainingJob(
            job_id="job123",
            model_name="test",
            model_type="SimpleModel",
            config=mock_config.training,
        )

        with patch("pathlib.Path.mkdir"):
            with patch("torch.save") as mock_save:
                with patch("mlflow.log_artifact"):
                    await trainer._save_checkpoint(model, optimizer, scheduler, 5, job)

                    mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_best_model(self, trainer, tmp_path):
        """Test saving best model."""
        from ai_engine.training.trainer import TrainingJob
        from ai_engine.core.config import TrainingConfig

        model = SimpleModel()
        job = TrainingJob(
            job_id="job123",
            model_name="test",
            model_type="SimpleModel",
            config=TrainingConfig(),
        )

        with patch("pathlib.Path.mkdir"):
            with patch("torch.save") as mock_save:
                model_path = await trainer._save_best_model(model, job)

                assert "test" in model_path
                mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate(self, trainer, mock_config):
        """Test validation loop."""
        model = SimpleModel()
        criterion = nn.CrossEntropyLoss()

        val_data = torch.randn(20, 10)
        val_labels = torch.randint(0, 2, (20,))
        val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=10)

        val_loss = await trainer._validate(model, val_loader, criterion)

        assert isinstance(val_loss, float)
        assert val_loss >= 0

    @pytest.mark.asyncio
    async def test_hyperparameter_optimization(self, trainer):
        """Test hyperparameter optimization."""

        def model_factory(**kwargs):
            return SimpleModel()

        train_data = torch.randn(50, 10)
        train_labels = torch.randint(0, 2, (50,))
        train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16)

        val_data = torch.randn(20, 10)
        val_labels = torch.randint(0, 2, (20,))
        val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16)

        search_space = {
            "learning_rate": {"type": "loguniform", "low": 1e-4, "high": 1e-2},
            "batch_size": {"type": "categorical", "choices": [16, 32]},
        }

        with patch("optuna.create_study") as mock_study:
            mock_study_instance = Mock()
            mock_study_instance.best_params = {"learning_rate": 1e-3, "batch_size": 32}
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

    @pytest.mark.asyncio
    async def test_distributed_training_not_available(self, trainer):
        """Test distributed training when not available."""
        model = SimpleModel()
        train_data = torch.randn(50, 10)
        train_labels = torch.randint(0, 2, (50,))
        train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16)

        with patch("torch.distributed.is_available", return_value=False):
            with patch.object(
                trainer, "train_model", new_callable=AsyncMock
            ) as mock_train:
                await trainer.distributed_training(
                    model=model,
                    train_dataloader=train_loader,
                    model_name="distributed_model",
                )

                mock_train.assert_called_once()

    @pytest.mark.asyncio
    async def test_resume_training(self, trainer, tmp_path):
        """Test resuming training from checkpoint."""
        model = SimpleModel()
        checkpoint = {
            "model": model,
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
            "epoch": 5,
            "learning_rate": 1e-3,
            "model_name": "resumed_model",
        }

        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        train_data = torch.randn(50, 10)
        train_labels = torch.randint(0, 2, (50,))
        train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16)

        with patch.object(trainer, "train_model", new_callable=AsyncMock) as mock_train:
            mock_train.return_value = Mock(status="completed")

            job = await trainer.resume_training(
                checkpoint_path=str(checkpoint_path),
                train_dataloader=train_loader,
                additional_epochs=5,
            )

            mock_train.assert_called_once()
