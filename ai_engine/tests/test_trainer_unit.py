"""
QBITEL Engine - Model Trainer Unit Tests

Comprehensive unit tests for the model training infrastructure.
"""

import pytest
import asyncio
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call
from pathlib import Path
from typing import Dict, Any

from ai_engine.training.trainer import ModelTrainer, TrainingJob
from ai_engine.core.config import Config, TrainingConfig
from ai_engine.core.exceptions import TrainingException, ModelException


class SimpleTestModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


class TestTrainingJob:
    """Test TrainingJob dataclass."""

    def test_training_job_creation(self):
        """Test creating a training job."""
        job = TrainingJob(
            job_id="test-job-123",
            model_name="test_model",
            model_type="classification",
            config=Mock(spec=TrainingConfig),
            status="pending",
        )

        assert job.job_id == "test-job-123"
        assert job.model_name == "test_model"
        assert job.model_type == "classification"
        assert job.status == "pending"
        assert job.start_time is None
        assert job.end_time is None

    def test_training_job_to_dict(self):
        """Test converting training job to dictionary."""
        config_mock = Mock(spec=TrainingConfig)
        job = TrainingJob(
            job_id="test-job-456",
            model_name="test_model",
            model_type="regression",
            config=config_mock,
            status="running",
            start_time=1234567890.0,
        )

        job_dict = job.to_dict()

        assert isinstance(job_dict, dict)
        assert job_dict["job_id"] == "test-job-456"
        assert job_dict["model_name"] == "test_model"
        assert job_dict["status"] == "running"
        assert job_dict["start_time"] == 1234567890.0


class TestModelTrainerInitialization:
    """Test ModelTrainer initialization."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Mock(spec=Config)
        config.mlflow = Mock()
        config.mlflow.experiment_name = "test_experiment"
        config.mlflow.tracking_uri = "http://localhost:5000"
        return config

    def test_trainer_initialization(self, config):
        """Test basic trainer initialization."""
        trainer = ModelTrainer(config)

        assert trainer.config == config
        assert trainer.experiment_name == "test_experiment"
        assert trainer.mlflow_client is None
        assert isinstance(trainer.active_jobs, dict)
        assert len(trainer.active_jobs) == 0
        assert isinstance(trainer.callbacks, list)
        assert trainer.world_size == 1
        assert trainer.rank == 0

    def test_trainer_device_selection_cpu(self, config):
        """Test device selection falls back to CPU."""
        with patch("torch.cuda.is_available", return_value=False):
            trainer = ModelTrainer(config)
            assert trainer.device == torch.device("cpu")

    def test_trainer_device_selection_cuda(self, config):
        """Test device selection uses CUDA when available."""
        with patch("torch.cuda.is_available", return_value=True):
            trainer = ModelTrainer(config)
            assert trainer.device == torch.device("cuda")


class TestModelTrainerInitialize:
    """Test ModelTrainer.initialize() method."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Mock(spec=Config)
        config.mlflow = Mock()
        config.mlflow.experiment_name = "test_experiment"
        config.mlflow.tracking_uri = "http://localhost:5000"
        return config

    @pytest.fixture
    def trainer(self, config):
        """Create trainer instance."""
        return ModelTrainer(config)

    @pytest.mark.asyncio
    async def test_initialize_success(self, trainer):
        """Test successful initialization."""
        with patch.object(
            trainer, "_initialize_mlflow", new_callable=AsyncMock
        ) as mock_mlflow:
            with patch.object(
                trainer, "_initialize_distributed", new_callable=AsyncMock
            ) as mock_dist:
                with patch.object(
                    trainer, "_initialize_model_registry", new_callable=AsyncMock
                ) as mock_registry:
                    await trainer.initialize()

                    mock_mlflow.assert_called_once()
                    mock_dist.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_mlflow_failure(self, trainer):
        """Test initialization handles MLflow failure."""
        with patch.object(
            trainer,
            "_initialize_mlflow",
            new_callable=AsyncMock,
            side_effect=Exception("MLflow error"),
        ):
            with pytest.raises(Exception, match="MLflow error"):
                await trainer.initialize()


class TestModelTrainerTraining:
    """Test ModelTrainer training methods."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Mock(spec=Config)
        config.mlflow = Mock()
        config.mlflow.experiment_name = "test_experiment"
        config.mlflow.tracking_uri = "http://localhost:5000"
        return config

    @pytest.fixture
    def trainer(self, config):
        """Create trainer instance."""
        return ModelTrainer(config)

    @pytest.fixture
    def training_config(self):
        """Create training configuration."""
        config = Mock(spec=TrainingConfig)
        config.epochs = 10
        config.batch_size = 32
        config.learning_rate = 0.001
        config.optimizer = "adam"
        config.scheduler = "step"
        config.loss_function = "mse"
        config.checkpoint_frequency = 5
        config.early_stopping_patience = 3
        config.validation_frequency = 1
        return config

    @pytest.mark.asyncio
    async def test_train_model_initialization(self, trainer, training_config):
        """Test train_model creates job correctly."""
        model = SimpleTestModel()
        train_loader = Mock(spec=torch.utils.data.DataLoader)
        val_loader = Mock(spec=torch.utils.data.DataLoader)

        with patch.object(
            trainer, "_training_loop", new_callable=AsyncMock
        ) as mock_loop:
            with patch("uuid.uuid4", return_value=Mock(hex="test-job-id")):
                mock_loop.return_value = {"best_metric": 0.95, "final_loss": 0.05}

                job_id = await trainer.train_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    model_name="test_model",
                    model_type="classification",
                    config=training_config,
                )

                assert job_id in trainer.active_jobs
                job = trainer.active_jobs[job_id]
                assert job.model_name == "test_model"
                assert job.model_type == "classification"
                assert job.status == "completed"

    @pytest.mark.asyncio
    async def test_train_model_failure_handling(self, trainer, training_config):
        """Test train_model handles failures."""
        model = SimpleTestModel()
        train_loader = Mock()
        val_loader = Mock()

        with patch.object(
            trainer,
            "_training_loop",
            new_callable=AsyncMock,
            side_effect=Exception("Training failed"),
        ):
            with patch("uuid.uuid4", return_value=Mock(hex="test-job-id")):
                job_id = await trainer.train_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    model_name="test_model",
                    model_type="classification",
                    config=training_config,
                )

                job = trainer.active_jobs[job_id]
                assert job.status == "failed"
                assert "Training failed" in job.error_message


class TestModelTrainerOptimizer:
    """Test optimizer creation methods."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Mock(spec=Config)
        config.mlflow = Mock()
        config.mlflow.experiment_name = "test"
        return config

    @pytest.fixture
    def trainer(self, config):
        """Create trainer instance."""
        return ModelTrainer(config)

    def test_create_optimizer_adam(self, trainer):
        """Test creating Adam optimizer."""
        model = SimpleTestModel()
        config = Mock(spec=TrainingConfig)
        config.optimizer = "adam"
        config.learning_rate = 0.001
        config.weight_decay = 0.0001

        optimizer = trainer._create_optimizer(model, config)

        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.defaults["lr"] == 0.001

    def test_create_optimizer_sgd(self, trainer):
        """Test creating SGD optimizer."""
        model = SimpleTestModel()
        config = Mock(spec=TrainingConfig)
        config.optimizer = "sgd"
        config.learning_rate = 0.01
        config.momentum = 0.9
        config.weight_decay = 0.0001

        optimizer = trainer._create_optimizer(model, config)

        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.defaults["lr"] == 0.01

    def test_create_optimizer_adamw(self, trainer):
        """Test creating AdamW optimizer."""
        model = SimpleTestModel()
        config = Mock(spec=TrainingConfig)
        config.optimizer = "adamw"
        config.learning_rate = 0.001
        config.weight_decay = 0.01

        optimizer = trainer._create_optimizer(model, config)

        assert isinstance(optimizer, torch.optim.AdamW)


class TestModelTrainerScheduler:
    """Test scheduler creation methods."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Mock(spec=Config)
        config.mlflow = Mock()
        config.mlflow.experiment_name = "test"
        return config

    @pytest.fixture
    def trainer(self, config):
        """Create trainer instance."""
        return ModelTrainer(config)

    def test_create_scheduler_step(self, trainer):
        """Test creating StepLR scheduler."""
        model = SimpleTestModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        config = Mock(spec=TrainingConfig)
        config.scheduler = "step"
        config.scheduler_step_size = 10
        config.scheduler_gamma = 0.1

        scheduler = trainer._create_scheduler(optimizer, config)

        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)

    def test_create_scheduler_cosine(self, trainer):
        """Test creating CosineAnnealingLR scheduler."""
        model = SimpleTestModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        config = Mock(spec=TrainingConfig)
        config.scheduler = "cosine"
        config.epochs = 100

        scheduler = trainer._create_scheduler(optimizer, config)

        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_create_scheduler_none(self, trainer):
        """Test no scheduler creation."""
        model = SimpleTestModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        config = Mock(spec=TrainingConfig)
        config.scheduler = None

        scheduler = trainer._create_scheduler(optimizer, config)

        assert scheduler is None


class TestModelTrainerJobManagement:
    """Test job management methods."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Mock(spec=Config)
        config.mlflow = Mock()
        config.mlflow.experiment_name = "test"
        return config

    @pytest.fixture
    def trainer(self, config):
        """Create trainer instance."""
        return ModelTrainer(config)

    def test_get_training_status(self, trainer):
        """Test getting training status."""
        # Create mock job
        job = TrainingJob(
            job_id="job-123",
            model_name="test_model",
            model_type="classification",
            config=Mock(),
            status="running",
            start_time=1234567890.0,
            best_metric_value=0.95,
        )
        trainer.active_jobs["job-123"] = job

        status = trainer.get_training_status("job-123")

        assert status is not None
        assert status["status"] == "running"
        assert status["model_name"] == "test_model"
        assert status["best_metric_value"] == 0.95

    def test_get_training_status_not_found(self, trainer):
        """Test getting status for non-existent job."""
        status = trainer.get_training_status("non-existent-job")
        assert status is None

    @pytest.mark.asyncio
    async def test_cancel_training(self, trainer):
        """Test cancelling a training job."""
        job = TrainingJob(
            job_id="job-456",
            model_name="test_model",
            model_type="classification",
            config=Mock(),
            status="running",
        )
        trainer.active_jobs["job-456"] = job

        result = await trainer.cancel_training("job-456")

        assert result is True
        assert trainer.active_jobs["job-456"].status == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_training_not_found(self, trainer):
        """Test cancelling non-existent job."""
        result = await trainer.cancel_training("non-existent-job")
        assert result is False


class TestModelTrainerCheckpointing:
    """Test model checkpointing methods."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Mock(spec=Config)
        config.mlflow = Mock()
        config.mlflow.experiment_name = "test"
        config.checkpoint_dir = "/tmp/checkpoints"
        return config

    @pytest.fixture
    def trainer(self, config):
        """Create trainer instance."""
        return ModelTrainer(config)

    @pytest.mark.asyncio
    async def test_save_checkpoint(self, trainer, tmp_path):
        """Test saving checkpoint."""
        model = SimpleTestModel()
        optimizer = torch.optim.Adam(model.parameters())
        job = TrainingJob(
            job_id="job-789",
            model_name="test_model",
            model_type="classification",
            config=Mock(),
            artifacts_path=str(tmp_path),
        )

        with patch("torch.save") as mock_save:
            checkpoint_path = await trainer._save_checkpoint(
                model=model, optimizer=optimizer, epoch=5, loss=0.123, job=job
            )

            assert checkpoint_path is not None
            mock_save.assert_called_once()

            # Verify checkpoint contains expected keys
            saved_data = mock_save.call_args[0][0]
            assert "epoch" in saved_data
            assert "model_state_dict" in saved_data
            assert "optimizer_state_dict" in saved_data
            assert "loss" in saved_data


class TestModelTrainerDistributed:
    """Test distributed training methods."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Mock(spec=Config)
        config.mlflow = Mock()
        config.mlflow.experiment_name = "test"
        return config

    @pytest.fixture
    def trainer(self, config):
        """Create trainer instance."""
        return ModelTrainer(config)

    @pytest.mark.asyncio
    async def test_initialize_distributed_not_available(self, trainer):
        """Test distributed init when not available."""
        with patch("torch.distributed.is_available", return_value=False):
            await trainer._initialize_distributed()

            assert trainer.world_size == 1
            assert trainer.rank == 0

    @pytest.mark.asyncio
    async def test_prepare_model_for_training_non_distributed(self, trainer):
        """Test preparing model without distributed training."""
        model = SimpleTestModel()
        trainer.world_size = 1

        prepared_model = await trainer._prepare_model_for_training(model)

        assert prepared_model == model
        assert prepared_model.training


class TestModelTrainerLossCriterion:
    """Test loss criterion creation."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Mock(spec=Config)
        config.mlflow = Mock()
        config.mlflow.experiment_name = "test"
        return config

    @pytest.fixture
    def trainer(self, config):
        """Create trainer instance."""
        return ModelTrainer(config)

    def test_create_loss_mse(self, trainer):
        """Test creating MSE loss."""
        config = Mock(spec=TrainingConfig)
        config.loss_function = "mse"

        criterion = trainer._create_loss_criterion(config)

        assert isinstance(criterion, nn.MSELoss)

    def test_create_loss_cross_entropy(self, trainer):
        """Test creating cross entropy loss."""
        config = Mock(spec=TrainingConfig)
        config.loss_function = "cross_entropy"

        criterion = trainer._create_loss_criterion(config)

        assert isinstance(criterion, nn.CrossEntropyLoss)

    def test_create_loss_bce(self, trainer):
        """Test creating BCE loss."""
        config = Mock(spec=TrainingConfig)
        config.loss_function = "bce"

        criterion = trainer._create_loss_criterion(config)

        assert isinstance(criterion, nn.BCEWithLogitsLoss)
