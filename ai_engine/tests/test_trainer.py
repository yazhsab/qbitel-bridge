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
        # Use weights_only=False for test compatibility
        torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=False)

        train_data = torch.randn(50, 10)
        train_labels = torch.randint(0, 2, (50,))
        train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16)

        with patch.object(trainer, "train_model", new_callable=AsyncMock) as mock_train:
            mock_train.return_value = Mock(status="completed")

            # Mock torch.load to avoid weights_only issue
            with patch("torch.load", return_value=checkpoint):
                job = await trainer.resume_training(
                    checkpoint_path=str(checkpoint_path),
                    train_dataloader=train_loader,
                    additional_epochs=5,
                )

                mock_train.assert_called_once()


class TestModelTrainerAdvanced:
    """Advanced tests for ModelTrainer."""

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

    @pytest.mark.asyncio
    async def test_initialize_mlflow_success(self, trainer):
        """Test MLflow initialization success."""
        with patch("mlflow.set_tracking_uri"):
            with patch("mlflow.get_experiment_by_name", return_value=None):
                with patch("mlflow.create_experiment", return_value="exp123"):
                    with patch("ai_engine.training.trainer.MlflowClient"):
                        await trainer._initialize_mlflow()

                        assert trainer.experiment_id == "exp123"

    @pytest.mark.asyncio
    async def test_initialize_mlflow_existing_experiment(self, trainer):
        """Test MLflow initialization with existing experiment."""
        mock_experiment = Mock()
        mock_experiment.experiment_id = "existing_exp"

        with patch("mlflow.set_tracking_uri"):
            with patch("mlflow.get_experiment_by_name", return_value=mock_experiment):
                with patch("ai_engine.training.trainer.MlflowClient"):
                    await trainer._initialize_mlflow()

                    assert trainer.experiment_id == "existing_exp"

    @pytest.mark.asyncio
    async def test_initialize_mlflow_failure(self, trainer):
        """Test MLflow initialization failure."""
        with patch("mlflow.set_tracking_uri", side_effect=Exception("MLflow error")):
            await trainer._initialize_mlflow()

            # Should continue without MLflow
            assert trainer.mlflow_client is None

    @pytest.mark.asyncio
    async def test_initialize_distributed_single_gpu(self, trainer):
        """Test distributed initialization with single GPU."""
        with patch("torch.distributed.is_available", return_value=True):
            with patch("torch.cuda.device_count", return_value=1):
                await trainer._initialize_distributed()

                assert trainer.world_size == 1

    @pytest.mark.asyncio
    async def test_initialize_distributed_multi_gpu(self, trainer):
        """Test distributed initialization with multiple GPUs."""
        with patch("torch.distributed.is_available", return_value=True):
            with patch("torch.cuda.device_count", return_value=4):
                with patch("torch.distributed.is_initialized", return_value=False):
                    with patch("torch.distributed.init_process_group"):
                        with patch("torch.distributed.get_world_size", return_value=4):
                            with patch("torch.distributed.get_rank", return_value=0):
                                with patch.dict("os.environ", {"LOCAL_RANK": "0"}):
                                    await trainer._initialize_distributed()

                                    assert trainer.world_size == 4
                                    assert trainer.rank == 0

    @pytest.mark.asyncio
    async def test_prepare_model_multi_gpu(self, trainer):
        """Test model preparation for multi-GPU."""
        model = SimpleModel()

        with patch("torch.cuda.device_count", return_value=2):
            prepared = await trainer._prepare_model_for_training(model)

            assert isinstance(prepared, (nn.Module, nn.DataParallel))

    def test_create_scheduler_constant(self, trainer, mock_config):
        """Test constant scheduler creation."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        mock_config.training.lr_scheduler = "constant"

        scheduler = trainer._create_scheduler(optimizer, mock_config.training)

        assert isinstance(scheduler, torch.optim.lr_scheduler.ConstantLR)

    @pytest.mark.asyncio
    async def test_training_loop_with_gradient_clipping(self, trainer, mock_config):
        """Test training loop with gradient clipping."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        criterion = nn.CrossEntropyLoss()

        train_data = torch.randn(50, 10)
        train_labels = torch.randint(0, 2, (50,))
        train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16)

        from ai_engine.training.trainer import TrainingJob

        job = TrainingJob(
            job_id="job123",
            model_name="test",
            model_type="SimpleModel",
            config=mock_config.training,
        )

        mock_config.training.epochs = 1
        mock_config.training.gradient_clip_norm = 1.0

        history = await trainer._training_loop(
            model,
            train_loader,
            None,
            optimizer,
            scheduler,
            criterion,
            mock_config.training,
            job,
        )

        assert "train_loss" in history
        assert len(history["train_loss"]) == 1

    @pytest.mark.asyncio
    async def test_training_loop_with_validation_and_early_stopping(
        self, trainer, mock_config
    ):
        """Test training loop with validation and early stopping."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        criterion = nn.CrossEntropyLoss()

        train_data = torch.randn(50, 10)
        train_labels = torch.randint(0, 2, (50,))
        train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16)

        val_data = torch.randn(20, 10)
        val_labels = torch.randint(0, 2, (20,))
        val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16)

        from ai_engine.training.trainer import TrainingJob

        job = TrainingJob(
            job_id="job123",
            model_name="test",
            model_type="SimpleModel",
            config=mock_config.training,
        )

        mock_config.training.epochs = 20
        mock_config.training.early_stopping_patience = 2

        with patch.object(trainer, "_save_checkpoint", new_callable=AsyncMock):
            history = await trainer._training_loop(
                model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,
                criterion,
                mock_config.training,
                job,
            )

            # Should stop early
            assert len(history["train_loss"]) < 20
            assert "best_val_loss" in history

    @pytest.mark.asyncio
    async def test_validate_empty_dataloader(self, trainer):
        """Test validation with empty dataloader."""
        model = SimpleModel()
        criterion = nn.CrossEntropyLoss()

        # Empty dataset
        val_data = torch.randn(0, 10)
        val_labels = torch.randint(0, 2, (0,))
        val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=10)

        val_loss = await trainer._validate(model, val_loader, criterion)

        assert val_loss == 0.0

    @pytest.mark.asyncio
    async def test_save_checkpoint_with_mlflow(self, trainer, mock_config, tmp_path):
        """Test checkpoint saving with MLflow logging."""
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

        trainer.mlflow_client = Mock()

        with patch("pathlib.Path.mkdir"):
            with patch("torch.save"):
                with patch("mlflow.log_artifact") as mock_log_artifact:
                    await trainer._save_checkpoint(model, optimizer, scheduler, 5, job)

                    mock_log_artifact.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_best_model_dataparallel(self, trainer, tmp_path):
        """Test saving DataParallel model."""
        from ai_engine.training.trainer import TrainingJob
        from ai_engine.core.config import TrainingConfig

        model = SimpleModel()
        model = nn.DataParallel(model)

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
    async def test_distributed_training_rank_zero(self, trainer, mock_config):
        """Test distributed training on rank 0."""
        model = SimpleModel()
        train_data = torch.randn(50, 10)
        train_labels = torch.randint(0, 2, (50,))
        train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16)

        trainer.world_size = 2
        trainer.rank = 0

        with patch("torch.distributed.is_available", return_value=True):
            with patch("torch.cuda.is_available", return_value=False):
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
    async def test_distributed_training_non_zero_rank(self, trainer, mock_config):
        """Test distributed training on non-zero rank."""
        model = SimpleModel()
        train_data = torch.randn(50, 10)
        train_labels = torch.randint(0, 2, (50,))
        train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16)

        trainer.world_size = 2
        trainer.rank = 1

        with patch("torch.distributed.is_available", return_value=True):
            with patch("torch.cuda.is_available", return_value=False):
                with patch.object(
                    trainer, "_training_loop_distributed", new_callable=AsyncMock
                ):
                    job = await trainer.distributed_training(
                        model=model,
                        train_dataloader=train_loader,
                        config=mock_config.training,
                        model_name="distributed_model",
                    )

                    assert job.model_name == "distributed_model"

    @pytest.mark.asyncio
    async def test_hyperparameter_optimization_trial_failure(self, trainer):
        """Test hyperparameter optimization with trial failures."""

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
        }

        with patch("optuna.create_study") as mock_study:
            mock_study_instance = Mock()
            mock_study_instance.best_params = {"learning_rate": 1e-3}
            mock_study_instance.best_value = 0.5
            mock_study_instance.optimize = Mock()
            mock_study.return_value = mock_study_instance

            result = await trainer.hyperparameter_optimization(
                model_factory=model_factory,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                search_space=search_space,
                n_trials=2,
            )

            assert "best_params" in result

    def test_get_training_status_empty(self, trainer):
        """Test getting training status with no jobs."""
        statuses = trainer.get_training_status()

        assert statuses == []

    @pytest.mark.asyncio
    async def test_cancel_training_nonexistent(self, trainer):
        """Test cancelling non-existent training job."""
        result = await trainer.cancel_training("nonexistent_job")

        assert result is False

    @pytest.mark.asyncio
    async def test_training_loop_distributed_with_barrier(self, trainer, mock_config):
        """Test distributed training loop with barrier."""
        model = SimpleModel()
        train_data = torch.randn(50, 10)
        train_labels = torch.randint(0, 2, (50,))
        train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16)

        from ai_engine.training.trainer import TrainingJob

        job = TrainingJob(
            job_id="job123",
            model_name="test",
            model_type="SimpleModel",
            config=mock_config.training,
        )

        mock_config.training.epochs = 1

        with patch("torch.distributed.barrier"):
            await trainer._training_loop_distributed(
                model, train_loader, None, mock_config.training, job
            )

    @pytest.mark.asyncio
    async def test_train_model_with_tags(self, trainer):
        """Test training with custom tags."""
        model = SimpleModel()
        train_data = torch.randn(50, 10)
        train_labels = torch.randint(0, 2, (50,))
        train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16)

        tags = {"experiment": "test", "version": "1.0"}

        with patch("mlflow.start_run"):
            with patch("mlflow.log_params"):
                with patch("mlflow.set_tags") as mock_set_tags:
                    with patch("mlflow.log_metrics"):
                        with patch("mlflow.pytorch.log_model"):
                            with patch.object(
                                trainer,
                                "_prepare_model_for_training",
                                return_value=model,
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
                                            tags=tags,
                                        )

                                        mock_set_tags.assert_called_once_with(tags)

    @pytest.mark.asyncio
    async def test_train_model_cleanup_on_failure(self, trainer):
        """Test that job is cleaned up on training failure."""
        model = SimpleModel()
        train_data = torch.randn(50, 10)
        train_labels = torch.randint(0, 2, (50,))
        train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16)

        with patch("mlflow.start_run"):
            with patch.object(
                trainer,
                "_prepare_model_for_training",
                side_effect=Exception("Training error"),
            ):
                from ai_engine.core.exceptions import TrainingException

                with pytest.raises(TrainingException):
                    await trainer.train_model(
                        model=model,
                        train_dataloader=train_loader,
                        model_name="test_model",
                    )

                # Job should be removed from active jobs
                assert len(trainer.active_jobs) == 0


class TestTrainingJobDataclass:
    """Test TrainingJob dataclass functionality."""

    def test_training_job_default_values(self):
        """Test TrainingJob with default values."""
        from ai_engine.training.trainer import TrainingJob
        from ai_engine.core.config import TrainingConfig

        job = TrainingJob(
            job_id="job123",
            model_name="test",
            model_type="SimpleModel",
            config=TrainingConfig(),
        )

        assert job.status == "pending"
        assert job.start_time is None
        assert job.end_time is None
        assert job.best_metric_value is None

    def test_training_job_to_dict_complete(self):
        """Test TrainingJob to_dict with all fields."""
        from ai_engine.training.trainer import TrainingJob
        from ai_engine.core.config import TrainingConfig
        import time

        job = TrainingJob(
            job_id="job123",
            model_name="test",
            model_type="SimpleModel",
            config=TrainingConfig(),
            status="completed",
            start_time=time.time(),
            end_time=time.time(),
            best_metric_value=0.5,
            experiment_id="exp123",
            run_id="run123",
            artifacts_path="/path/to/artifacts",
        )

        job_dict = job.to_dict()

        assert job_dict["status"] == "completed"
        assert job_dict["best_metric_value"] == 0.5
        assert job_dict["experiment_id"] == "exp123"
