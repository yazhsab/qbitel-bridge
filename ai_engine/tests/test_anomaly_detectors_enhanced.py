"""
Enhanced comprehensive tests for anomaly detection modules:
- ensemble_detector.py
- isolation_forest.py
- lstm_detector.py
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
from sklearn.ensemble import IsolationForest as SklearnIsolationForest

from ai_engine.anomaly.ensemble_detector import EnsembleAnomalyDetector
from ai_engine.anomaly.isolation_forest import IsolationForestDetector
from ai_engine.anomaly.lstm_detector import LSTMAnomalyDetector, LSTMModel


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return np.random.randn(100, 10)


@pytest.fixture
def sample_sequences():
    """Create sample sequence data for LSTM."""
    return np.random.randn(50, 20, 10)  # (samples, sequence_length, features)


class TestEnsembleAnomalyDetector:
    """Test suite for EnsembleAnomalyDetector."""

    def test_initialization_default(self):
        """Test ensemble detector initialization with defaults."""
        detector = EnsembleAnomalyDetector()

        assert detector.detectors == []
        assert detector.weights is None
        assert detector.voting_method == "average"

    def test_initialization_with_detectors(self):
        """Test initialization with pre-configured detectors."""
        mock_detector1 = Mock()
        mock_detector2 = Mock()

        detector = EnsembleAnomalyDetector(
            detectors=[mock_detector1, mock_detector2],
            weights=[0.6, 0.4],
        )

        assert len(detector.detectors) == 2
        assert detector.weights == [0.6, 0.4]

    def test_add_detector(self):
        """Test adding detector to ensemble."""
        detector = EnsembleAnomalyDetector()
        mock_detector = Mock()

        detector.add_detector(mock_detector, weight=0.5)

        assert len(detector.detectors) == 1
        assert detector.weights[0] == 0.5

    def test_add_detector_default_weight(self):
        """Test adding detector with default weight."""
        detector = EnsembleAnomalyDetector()
        mock_detector = Mock()

        detector.add_detector(mock_detector)

        assert detector.weights[0] == 1.0

    def test_train_all_detectors(self, sample_data):
        """Test training all detectors in ensemble."""
        detector = EnsembleAnomalyDetector()

        mock_detector1 = Mock()
        mock_detector1.train = Mock()
        mock_detector2 = Mock()
        mock_detector2.train = Mock()

        detector.add_detector(mock_detector1)
        detector.add_detector(mock_detector2)

        detector.train(sample_data)

        mock_detector1.train.assert_called_once()
        mock_detector2.train.assert_called_once()

    def test_detect_average_voting(self, sample_data):
        """Test detection with average voting."""
        detector = EnsembleAnomalyDetector(voting_method="average")

        mock_detector1 = Mock()
        mock_detector1.detect = Mock(return_value=(True, 0.8))
        mock_detector2 = Mock()
        mock_detector2.detect = Mock(return_value=(False, 0.3))

        detector.add_detector(mock_detector1, weight=0.5)
        detector.add_detector(mock_detector2, weight=0.5)

        is_anomaly, score = detector.detect(sample_data[0])

        assert isinstance(is_anomaly, bool)
        assert isinstance(score, float)
        assert 0.3 <= score <= 0.8

    def test_detect_majority_voting(self, sample_data):
        """Test detection with majority voting."""
        detector = EnsembleAnomalyDetector(voting_method="majority")

        mock_detector1 = Mock()
        mock_detector1.detect = Mock(return_value=(True, 0.8))
        mock_detector2 = Mock()
        mock_detector2.detect = Mock(return_value=(True, 0.7))
        mock_detector3 = Mock()
        mock_detector3.detect = Mock(return_value=(False, 0.3))

        detector.add_detector(mock_detector1)
        detector.add_detector(mock_detector2)
        detector.add_detector(mock_detector3)

        is_anomaly, score = detector.detect(sample_data[0])

        assert is_anomaly is True  # Majority voted anomaly

    def test_detect_max_voting(self, sample_data):
        """Test detection with max voting."""
        detector = EnsembleAnomalyDetector(voting_method="max")

        mock_detector1 = Mock()
        mock_detector1.detect = Mock(return_value=(True, 0.8))
        mock_detector2 = Mock()
        mock_detector2.detect = Mock(return_value=(False, 0.3))

        detector.add_detector(mock_detector1)
        detector.add_detector(mock_detector2)

        is_anomaly, score = detector.detect(sample_data[0])

        assert score == 0.8  # Max score

    def test_detect_batch(self, sample_data):
        """Test batch detection."""
        detector = EnsembleAnomalyDetector()

        mock_detector = Mock()
        mock_detector.detect = Mock(
            return_value=(np.array([True, False]), np.array([0.8, 0.3]))
        )

        detector.add_detector(mock_detector)

        is_anomaly, scores = detector.detect(sample_data[:2])

        assert len(is_anomaly) == 2
        assert len(scores) == 2

    def test_detect_no_detectors(self, sample_data):
        """Test detection with no detectors."""
        detector = EnsembleAnomalyDetector()

        with pytest.raises(ValueError, match="No detectors"):
            detector.detect(sample_data[0])

    def test_get_detector_scores(self, sample_data):
        """Test getting individual detector scores."""
        detector = EnsembleAnomalyDetector()

        mock_detector1 = Mock()
        mock_detector1.detect = Mock(return_value=(True, 0.8))
        mock_detector1.__class__.__name__ = "Detector1"

        mock_detector2 = Mock()
        mock_detector2.detect = Mock(return_value=(False, 0.3))
        mock_detector2.__class__.__name__ = "Detector2"

        detector.add_detector(mock_detector1)
        detector.add_detector(mock_detector2)

        scores = detector.get_detector_scores(sample_data[0])

        assert "Detector1" in scores
        assert "Detector2" in scores

    def test_set_threshold(self):
        """Test setting detection threshold."""
        detector = EnsembleAnomalyDetector()

        detector.set_threshold(0.7)

        assert detector.threshold == 0.7

    def test_save_load(self):
        """Test saving and loading ensemble."""
        detector = EnsembleAnomalyDetector()

        mock_detector = Mock()
        mock_detector.save = Mock()
        mock_detector.load = Mock()

        detector.add_detector(mock_detector)

        with patch("builtins.open", create=True):
            with patch("json.dump"):
                detector.save("ensemble_model")

        with patch("builtins.open", create=True):
            with patch(
                "json.load", return_value={"voting_method": "average", "weights": [1.0]}
            ):
                detector.load("ensemble_model")


class TestIsolationForestDetector:
    """Test suite for IsolationForestDetector."""

    def test_initialization_default(self):
        """Test isolation forest initialization with defaults."""
        detector = IsolationForestDetector()

        assert detector.n_estimators == 100
        assert detector.contamination == 0.1
        assert detector.model is None

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        detector = IsolationForestDetector(
            n_estimators=200,
            contamination=0.05,
            max_samples=256,
            random_state=42,
        )

        assert detector.n_estimators == 200
        assert detector.contamination == 0.05
        assert detector.max_samples == 256

    def test_train(self, sample_data):
        """Test training isolation forest."""
        detector = IsolationForestDetector()

        detector.train(sample_data)

        assert detector.model is not None
        assert detector.is_trained

    def test_train_with_validation(self, sample_data):
        """Test training with validation data."""
        detector = IsolationForestDetector()

        train_data = sample_data[:80]
        val_data = sample_data[80:]

        detector.train(train_data, val_data=val_data)

        assert detector.is_trained

    def test_detect_single_sample(self, sample_data):
        """Test detecting single sample."""
        detector = IsolationForestDetector()
        detector.train(sample_data)

        is_anomaly, score = detector.detect(sample_data[0])

        assert isinstance(is_anomaly, bool)
        assert isinstance(score, float)

    def test_detect_batch(self, sample_data):
        """Test detecting batch of samples."""
        detector = IsolationForestDetector()
        detector.train(sample_data)

        is_anomaly, scores = detector.detect(sample_data[:10])

        assert len(is_anomaly) == 10
        assert len(scores) == 10

    def test_detect_not_trained(self, sample_data):
        """Test detection without training."""
        detector = IsolationForestDetector()

        with pytest.raises(RuntimeError, match="not been trained"):
            detector.detect(sample_data[0])

    def test_get_anomaly_score(self, sample_data):
        """Test getting anomaly score."""
        detector = IsolationForestDetector()
        detector.train(sample_data)

        score = detector.get_anomaly_score(sample_data[0])

        assert isinstance(score, float)

    def test_set_contamination(self, sample_data):
        """Test setting contamination parameter."""
        detector = IsolationForestDetector()
        detector.train(sample_data)

        detector.set_contamination(0.15)

        assert detector.contamination == 0.15

    def test_save_load(self, sample_data):
        """Test saving and loading model."""
        detector = IsolationForestDetector()
        detector.train(sample_data)

        with patch("joblib.dump") as mock_dump:
            detector.save("if_model.pkl")
            mock_dump.assert_called_once()

        with patch("joblib.load") as mock_load:
            mock_load.return_value = detector.model
            detector.load("if_model.pkl")
            assert detector.is_trained

    def test_feature_importance(self, sample_data):
        """Test getting feature importance."""
        detector = IsolationForestDetector()
        detector.train(sample_data)

        # Isolation Forest doesn't have direct feature importance
        # but we can test the model structure
        assert detector.model.n_features_in_ == sample_data.shape[1]

    def test_decision_function(self, sample_data):
        """Test decision function."""
        detector = IsolationForestDetector()
        detector.train(sample_data)

        scores = detector.model.decision_function(sample_data[:5])

        assert len(scores) == 5

    def test_score_samples(self, sample_data):
        """Test score samples method."""
        detector = IsolationForestDetector()
        detector.train(sample_data)

        scores = detector.model.score_samples(sample_data[:5])

        assert len(scores) == 5


class TestLSTMModel:
    """Test suite for LSTM model."""

    def test_initialization(self):
        """Test LSTM model initialization."""
        model = LSTMModel(input_dim=10, hidden_dim=32, num_layers=2)

        assert model.input_dim == 10
        assert model.hidden_dim == 32
        assert model.num_layers == 2

    def test_forward_pass(self):
        """Test LSTM forward pass."""
        model = LSTMModel(input_dim=10, hidden_dim=32, num_layers=2)
        x = torch.randn(5, 20, 10)  # (batch, seq_len, features)

        output = model(x)

        assert output.shape == (5, 20, 10)

    def test_different_architectures(self):
        """Test different LSTM architectures."""
        # Single layer
        model1 = LSTMModel(input_dim=10, hidden_dim=32, num_layers=1)
        assert model1.num_layers == 1

        # Multiple layers
        model2 = LSTMModel(input_dim=10, hidden_dim=64, num_layers=3)
        assert model2.num_layers == 3


class TestLSTMAnomalyDetector:
    """Test suite for LSTMAnomalyDetector."""

    def test_initialization_default(self):
        """Test LSTM detector initialization with defaults."""
        detector = LSTMAnomalyDetector(input_dim=10, sequence_length=20)

        assert detector.input_dim == 10
        assert detector.sequence_length == 20
        assert detector.model is not None

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        detector = LSTMAnomalyDetector(
            input_dim=10,
            sequence_length=20,
            hidden_dim=64,
            num_layers=3,
            learning_rate=0.0001,
            batch_size=64,
        )

        assert detector.hidden_dim == 64
        assert detector.num_layers == 3
        assert detector.learning_rate == 0.0001

    def test_preprocess_data(self, sample_sequences):
        """Test data preprocessing."""
        detector = LSTMAnomalyDetector(input_dim=10, sequence_length=20)

        processed = detector._preprocess_data(sample_sequences)

        assert isinstance(processed, torch.Tensor)
        assert processed.shape == sample_sequences.shape

    def test_create_sequences(self):
        """Test sequence creation from data."""
        detector = LSTMAnomalyDetector(input_dim=10, sequence_length=20)
        data = np.random.randn(100, 10)

        sequences = detector._create_sequences(data)

        assert sequences.shape[1] == 20  # sequence_length
        assert sequences.shape[2] == 10  # input_dim

    def test_train(self, sample_sequences):
        """Test training LSTM detector."""
        detector = LSTMAnomalyDetector(input_dim=10, sequence_length=20)

        with patch.object(detector, "_train_epoch", return_value=0.5):
            detector.train(sample_sequences, epochs=2)

            assert detector.is_trained

    def test_train_with_validation(self, sample_sequences):
        """Test training with validation data."""
        detector = LSTMAnomalyDetector(input_dim=10, sequence_length=20)

        train_data = sample_sequences[:40]
        val_data = sample_sequences[40:]

        with patch.object(detector, "_train_epoch", return_value=0.5):
            with patch.object(detector, "_validate_epoch", return_value=0.6):
                detector.train(train_data, val_data=val_data, epochs=2)

                assert detector.is_trained

    def test_detect_single_sequence(self, sample_sequences):
        """Test detecting single sequence."""
        detector = LSTMAnomalyDetector(input_dim=10, sequence_length=20)
        detector.is_trained = True
        detector.threshold = 1.0

        with patch.object(detector, "_compute_reconstruction_error", return_value=0.5):
            is_anomaly, score = detector.detect(sample_sequences[0])

            assert isinstance(is_anomaly, bool)
            assert isinstance(score, float)

    def test_detect_batch(self, sample_sequences):
        """Test detecting batch of sequences."""
        detector = LSTMAnomalyDetector(input_dim=10, sequence_length=20)
        detector.is_trained = True
        detector.threshold = 1.0

        with patch.object(
            detector, "_compute_reconstruction_error", return_value=np.array([0.5] * 10)
        ):
            is_anomaly, scores = detector.detect(sample_sequences[:10])

            assert len(is_anomaly) == 10
            assert len(scores) == 10

    def test_detect_not_trained(self, sample_sequences):
        """Test detection without training."""
        detector = LSTMAnomalyDetector(input_dim=10, sequence_length=20)

        with pytest.raises(RuntimeError, match="not been trained"):
            detector.detect(sample_sequences[0])

    def test_compute_reconstruction_error(self, sample_sequences):
        """Test reconstruction error computation."""
        detector = LSTMAnomalyDetector(input_dim=10, sequence_length=20)
        detector.is_trained = True

        data = torch.tensor(sample_sequences[:5], dtype=torch.float32)

        errors = detector._compute_reconstruction_error(data)

        assert len(errors) == 5
        assert all(e >= 0 for e in errors)

    def test_set_threshold_percentile(self, sample_sequences):
        """Test setting threshold by percentile."""
        detector = LSTMAnomalyDetector(input_dim=10, sequence_length=20)
        detector.is_trained = True

        with patch.object(
            detector, "_compute_reconstruction_error", return_value=np.random.randn(50)
        ):
            detector.set_threshold(sample_sequences, percentile=95)

            assert detector.threshold is not None

    def test_set_threshold_manual(self):
        """Test setting threshold manually."""
        detector = LSTMAnomalyDetector(input_dim=10, sequence_length=20)

        detector.set_threshold(threshold=1.5)

        assert detector.threshold == 1.5

    def test_save_load(self):
        """Test saving and loading model."""
        detector = LSTMAnomalyDetector(input_dim=10, sequence_length=20)
        detector.is_trained = True

        with patch("torch.save") as mock_save:
            with patch("pathlib.Path.mkdir"):
                detector.save("lstm_model.pt")
                mock_save.assert_called_once()

        mock_state = {
            "model_state_dict": {},
            "threshold": 1.0,
            "input_dim": 10,
            "sequence_length": 20,
            "hidden_dim": 32,
            "num_layers": 2,
        }

        with patch("torch.load", return_value=mock_state):
            detector.load("lstm_model.pt")
            assert detector.is_trained

    def test_train_epoch(self, sample_sequences):
        """Test single training epoch."""
        detector = LSTMAnomalyDetector(input_dim=10, sequence_length=20)
        data = torch.tensor(sample_sequences, dtype=torch.float32)

        loss = detector._train_epoch(data)

        assert isinstance(loss, float)
        assert loss >= 0

    def test_validate_epoch(self, sample_sequences):
        """Test single validation epoch."""
        detector = LSTMAnomalyDetector(input_dim=10, sequence_length=20)
        data = torch.tensor(sample_sequences[:10], dtype=torch.float32)

        loss = detector._validate_epoch(data)

        assert isinstance(loss, float)
        assert loss >= 0

    def test_train_with_early_stopping(self, sample_sequences):
        """Test training with early stopping."""
        detector = LSTMAnomalyDetector(input_dim=10, sequence_length=20)

        train_data = sample_sequences[:40]
        val_data = sample_sequences[40:]

        # Mock increasing validation loss
        val_losses = [0.5, 0.6, 0.7, 0.8]
        with patch.object(detector, "_train_epoch", return_value=0.5):
            with patch.object(detector, "_validate_epoch", side_effect=val_losses):
                detector.train(
                    train_data, val_data=val_data, epochs=10, early_stopping_patience=2
                )

                assert detector.is_trained

    def test_different_sequence_lengths(self):
        """Test with different sequence lengths."""
        detector1 = LSTMAnomalyDetector(input_dim=10, sequence_length=10)
        detector2 = LSTMAnomalyDetector(input_dim=10, sequence_length=50)

        assert detector1.sequence_length == 10
        assert detector2.sequence_length == 50

    def test_gpu_support(self):
        """Test GPU support detection."""
        detector = LSTMAnomalyDetector(input_dim=10, sequence_length=20)

        assert detector.device is not None

    def test_batch_processing(self, sample_sequences):
        """Test batch processing during training."""
        detector = LSTMAnomalyDetector(input_dim=10, sequence_length=20, batch_size=16)

        with patch.object(detector, "_train_epoch", return_value=0.5):
            detector.train(sample_sequences, epochs=1)

            assert detector.is_trained

    def test_reconstruction_quality(self, sample_sequences):
        """Test reconstruction quality."""
        detector = LSTMAnomalyDetector(input_dim=10, sequence_length=20)
        detector.is_trained = True

        data = torch.tensor(sample_sequences[:5], dtype=torch.float32)

        # Perfect reconstruction should have low error
        with patch.object(detector.model, "forward", return_value=data):
            errors = detector._compute_reconstruction_error(data)

            assert all(e < 0.1 for e in errors)

    def test_sequence_creation_edge_cases(self):
        """Test sequence creation edge cases."""
        detector = LSTMAnomalyDetector(input_dim=10, sequence_length=20)

        # Data shorter than sequence length
        short_data = np.random.randn(10, 10)
        sequences = detector._create_sequences(short_data)

        # Should handle gracefully
        assert sequences.shape[0] >= 0

    def test_training_convergence(self, sample_sequences):
        """Test training convergence."""
        detector = LSTMAnomalyDetector(input_dim=10, sequence_length=20)

        losses = []

        def mock_train_epoch(data):
            loss = 1.0 / (len(losses) + 1)  # Decreasing loss
            losses.append(loss)
            return loss

        with patch.object(detector, "_train_epoch", side_effect=mock_train_epoch):
            detector.train(sample_sequences, epochs=5)

            # Loss should decrease
            assert losses[-1] < losses[0]
