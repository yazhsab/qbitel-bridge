"""
CRONOS AI Engine - Base Model Classes

This module provides base classes and interfaces for AI models in the CRONOS AI Engine.
"""

import abc
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn as nn
import numpy as np

from ..core.config import Config
from ..core.exceptions import ModelException, ValidationException


class ModelState(str, Enum):
    """Model state enumeration."""

    INITIALIZED = "initialized"
    TRAINING = "training"
    TRAINED = "trained"
    LOADED = "loaded"
    INFERENCE_READY = "inference_ready"
    FAILED = "failed"


@dataclass
class ModelInput:
    """Base class for model inputs."""

    data: Union[torch.Tensor, np.ndarray, bytes, str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_tensor(self) -> torch.Tensor:
        """Convert input data to tensor."""
        if isinstance(self.data, torch.Tensor):
            return self.data
        elif isinstance(self.data, np.ndarray):
            return torch.from_numpy(self.data)
        elif isinstance(self.data, (bytes, str)):
            # For text/byte data, convert to tensor representation
            if isinstance(self.data, str):
                # Convert string to bytes
                data_bytes = self.data.encode("utf-8")
            else:
                data_bytes = self.data

            # Convert bytes to tensor
            byte_array = np.frombuffer(data_bytes, dtype=np.uint8)
            return torch.from_numpy(byte_array).float()
        else:
            raise ModelException(f"Unsupported input data type: {type(self.data)}")


@dataclass
class ModelOutput:
    """Base class for model outputs."""

    predictions: Union[torch.Tensor, Dict[str, torch.Tensor]]
    confidence: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None
    processing_time_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert output to dictionary."""
        result = {
            "predictions": self._tensor_to_list(self.predictions),
            "processing_time_ms": self.processing_time_ms,
        }

        if self.confidence is not None:
            result["confidence"] = self._tensor_to_list(self.confidence)

        if self.metadata:
            result["metadata"] = self.metadata

        return result

    def _tensor_to_list(self, tensor_data):
        """Convert tensor data to list for serialization."""
        if isinstance(tensor_data, torch.Tensor):
            return tensor_data.detach().cpu().numpy().tolist()
        elif isinstance(tensor_data, dict):
            return {k: self._tensor_to_list(v) for k, v in tensor_data.items()}
        else:
            return tensor_data


class BaseModel(nn.Module, abc.ABC):
    """
    Abstract base class for all AI models in CRONOS AI Engine.

    This class provides common functionality for model lifecycle management,
    validation, and standardized interfaces.
    """

    def __init__(self, config: Config, model_name: str):
        """Initialize base model."""
        super().__init__()
        self.config = config
        self.model_name = model_name
        self.logger = logging.getLogger(f"{__name__}.{model_name}")

        # Model state
        self.state = ModelState.INITIALIZED
        self.version = "1.0.0"
        self.created_at = time.time()
        self.last_updated = self.created_at

        # Model metadata
        self.input_schema = None
        self.output_schema = None
        self.model_parameters = {}

        # Performance tracking
        self.training_metrics = {}
        self.inference_metrics = {
            "total_inferences": 0,
            "total_time_ms": 0.0,
            "average_latency_ms": 0.0,
            "errors": 0,
        }

        # Device management
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger.info(f"BaseModel {model_name} initialized on {self.device}")

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        pass

    @abc.abstractmethod
    def predict(self, input_data: ModelInput) -> ModelOutput:
        """Make prediction on input data."""
        pass

    @abc.abstractmethod
    def validate_input(self, input_data: ModelInput) -> bool:
        """Validate input data format and constraints."""
        pass

    @abc.abstractmethod
    def get_input_schema(self) -> Dict[str, Any]:
        """Get input data schema."""
        pass

    @abc.abstractmethod
    def get_output_schema(self) -> Dict[str, Any]:
        """Get output data schema."""
        pass

    def set_state(self, new_state: ModelState) -> None:
        """Update model state."""
        old_state = self.state
        self.state = new_state
        self.last_updated = time.time()
        self.logger.info(f"Model state changed: {old_state} -> {new_state}")

    def to_device(self, device: Optional[torch.device] = None) -> None:
        """Move model to specified device."""
        if device is None:
            device = self.device

        self.to(device)
        self.device = device
        self.logger.info(f"Model moved to device: {device}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        param_count = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "name": self.model_name,
            "version": self.version,
            "state": self.state.value,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "device": str(self.device),
            "parameters": {
                "total": param_count,
                "trainable": trainable_params,
                "frozen": param_count - trainable_params,
            },
            "input_schema": self.get_input_schema(),
            "output_schema": self.get_output_schema(),
            "inference_metrics": self.inference_metrics.copy(),
            "training_metrics": self.training_metrics.copy(),
        }

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.inference_metrics = {
            "total_inferences": 0,
            "total_time_ms": 0.0,
            "average_latency_ms": 0.0,
            "errors": 0,
        }
        self.training_metrics = {}
        self.logger.info("Model metrics reset")

    def update_inference_metrics(
        self, processing_time_ms: float, success: bool = True
    ) -> None:
        """Update inference performance metrics."""
        self.inference_metrics["total_inferences"] += 1
        self.inference_metrics["total_time_ms"] += processing_time_ms

        if success:
            # Update average latency
            total_inferences = self.inference_metrics["total_inferences"]
            total_time = self.inference_metrics["total_time_ms"]
            self.inference_metrics["average_latency_ms"] = total_time / total_inferences
        else:
            self.inference_metrics["errors"] += 1

    def validate_tensor_shape(
        self, tensor: torch.Tensor, expected_shape: Tuple[int, ...]
    ) -> bool:
        """Validate tensor shape against expected shape."""
        if len(tensor.shape) != len(expected_shape):
            return False

        for actual, expected in zip(tensor.shape, expected_shape):
            if expected != -1 and actual != expected:  # -1 means any size
                return False

        return True

    def preprocess_input(self, input_data: ModelInput) -> torch.Tensor:
        """Preprocess input data before inference."""
        tensor = input_data.to_tensor()

        # Move to correct device
        if tensor.device != self.device:
            tensor = tensor.to(self.device)

        return tensor

    def postprocess_output(
        self, raw_output: torch.Tensor, input_metadata: Optional[Dict[str, Any]] = None
    ) -> ModelOutput:
        """Postprocess model output."""
        return ModelOutput(
            predictions=raw_output,
            metadata={"model_name": self.model_name, "model_version": self.version},
        )

    def save_checkpoint(
        self, filepath: str, include_optimizer: bool = False, **kwargs
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "model_name": self.model_name,
            "version": self.version,
            "state_dict": self.state_dict(),
            "model_config": (
                self.config.__dict__ if hasattr(self.config, "__dict__") else {}
            ),
            "model_parameters": self.model_parameters,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "inference_metrics": self.inference_metrics,
            "training_metrics": self.training_metrics,
            **kwargs,
        }

        torch.save(checkpoint, filepath)
        self.logger.info(f"Model checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath: str, strict: bool = True) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)

        # Load state dict
        self.load_state_dict(checkpoint["state_dict"], strict=strict)

        # Restore metadata
        self.version = checkpoint.get("version", "unknown")
        self.model_parameters = checkpoint.get("model_parameters", {})
        self.created_at = checkpoint.get("created_at", time.time())
        self.last_updated = checkpoint.get("last_updated", time.time())
        self.inference_metrics = checkpoint.get(
            "inference_metrics", self.inference_metrics
        )
        self.training_metrics = checkpoint.get("training_metrics", {})

        self.set_state(ModelState.LOADED)
        self.logger.info(f"Model checkpoint loaded: {filepath}")

    def freeze_layers(self, layer_names: Optional[List[str]] = None) -> None:
        """Freeze model layers for transfer learning."""
        if layer_names is None:
            # Freeze all parameters
            for param in self.parameters():
                param.requires_grad = False
            self.logger.info("All model parameters frozen")
        else:
            # Freeze specific layers
            for name, module in self.named_modules():
                if name in layer_names:
                    for param in module.parameters():
                        param.requires_grad = False
                    self.logger.info(f"Layer '{name}' parameters frozen")

    def unfreeze_layers(self, layer_names: Optional[List[str]] = None) -> None:
        """Unfreeze model layers."""
        if layer_names is None:
            # Unfreeze all parameters
            for param in self.parameters():
                param.requires_grad = True
            self.logger.info("All model parameters unfrozen")
        else:
            # Unfreeze specific layers
            for name, module in self.named_modules():
                if name in layer_names:
                    for param in module.parameters():
                        param.requires_grad = True
                    self.logger.info(f"Layer '{name}' parameters unfrozen")

    def get_memory_usage(self) -> Dict[str, float]:
        """Get model memory usage information."""
        if not torch.cuda.is_available():
            return {"memory_mb": 0.0, "device": "cpu"}

        # Calculate model parameters memory
        param_size = 0
        buffer_size = 0

        for param in self.parameters():
            param_size += param.numel() * param.element_size()

        for buffer in self.buffers():
            buffer_size += buffer.numel() * buffer.element_size()

        total_size_mb = (param_size + buffer_size) / (1024 * 1024)

        memory_info = {"model_size_mb": total_size_mb, "device": str(self.device)}

        if self.device.type == "cuda":
            memory_info.update(
                {
                    "gpu_allocated_mb": torch.cuda.memory_allocated(self.device)
                    / (1024 * 1024),
                    "gpu_cached_mb": torch.cuda.memory_reserved(self.device)
                    / (1024 * 1024),
                    "gpu_max_allocated_mb": torch.cuda.max_memory_allocated(self.device)
                    / (1024 * 1024),
                }
            )

        return memory_info

    def benchmark_inference(
        self, input_data: ModelInput, num_runs: int = 100
    ) -> Dict[str, float]:
        """Benchmark model inference performance."""
        self.eval()
        times = []

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.predict(input_data)

        # Benchmark
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = self.predict(input_data)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms

        times = np.array(times)

        return {
            "mean_latency_ms": float(np.mean(times)),
            "std_latency_ms": float(np.std(times)),
            "min_latency_ms": float(np.min(times)),
            "max_latency_ms": float(np.max(times)),
            "p50_latency_ms": float(np.percentile(times, 50)),
            "p90_latency_ms": float(np.percentile(times, 90)),
            "p99_latency_ms": float(np.percentile(times, 99)),
            "throughput_per_second": 1000.0 / float(np.mean(times)),
        }


class ModelValidator:
    """Validator for model inputs and outputs."""

    @staticmethod
    def validate_input_schema(input_data: ModelInput, schema: Dict[str, Any]) -> bool:
        """Validate input data against schema."""
        try:
            # Basic validation - can be extended with more sophisticated schema validation
            if "type" in schema:
                expected_type = schema["type"]
                if expected_type == "tensor":
                    tensor = input_data.to_tensor()
                    if "shape" in schema:
                        expected_shape = tuple(schema["shape"])
                        if tensor.shape != expected_shape and -1 not in expected_shape:
                            return False

                    if "dtype" in schema:
                        expected_dtype = getattr(torch, schema["dtype"])
                        if tensor.dtype != expected_dtype:
                            return False

            return True

        except Exception as e:
            logging.getLogger(__name__).error(f"Input validation failed: {e}")
            return False

    @staticmethod
    def validate_output_schema(output: ModelOutput, schema: Dict[str, Any]) -> bool:
        """Validate output data against schema."""
        try:
            # Basic validation - can be extended
            if "predictions" in schema:
                pred_schema = schema["predictions"]
                if "type" in pred_schema and pred_schema["type"] == "tensor":
                    if not isinstance(output.predictions, torch.Tensor):
                        return False

            return True

        except Exception as e:
            logging.getLogger(__name__).error(f"Output validation failed: {e}")
            return False


class ModelFactory:
    """Factory for creating model instances."""

    _model_registry: Dict[str, type] = {}

    @classmethod
    def register_model(cls, model_name: str, model_class: type) -> None:
        """Register a model class."""
        cls._model_registry[model_name] = model_class
        logging.getLogger(__name__).info(f"Model registered: {model_name}")

    @classmethod
    def create_model(cls, model_name: str, config: Config, **kwargs) -> BaseModel:
        """Create model instance."""
        if model_name not in cls._model_registry:
            raise ModelException(f"Unknown model type: {model_name}")

        model_class = cls._model_registry[model_name]
        return model_class(config=config, **kwargs)

    @classmethod
    def list_available_models(cls) -> List[str]:
        """List available model types."""
        return list(cls._model_registry.keys())
