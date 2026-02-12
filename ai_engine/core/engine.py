"""
QBITEL Engine - Main Engine Orchestrator

This module provides the main AI Engine class that orchestrates all AI components
including protocol discovery, field detection, and anomaly detection.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import torch
import numpy as np
from prometheus_client import Counter, Histogram, Gauge

from .config import Config, get_config
from .exceptions import (
    QbitelAIException,
    ModelException,
    InferenceException,
    ConfigurationException,
    AIEngineException,
)

# Import AI components (will be implemented)
from ..discovery.pcfg_inference import PCFGInference
from ..discovery.protocol_classifier import ProtocolClassifier
from ..detection.field_detector import FieldDetector
from ..anomaly.vae_detector import VAEDetector
from ..anomaly.ensemble_detector import EnsembleAnomalyDetector
from ..features.extractors import FeatureExtractor
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from ..models.registry import ModelRegistry
    from ..monitoring.metrics import MetricsCollector


# Prometheus metrics
INFERENCE_COUNTER = Counter("qbitel_inference_total", "Total inference requests", ["component", "status"])
INFERENCE_DURATION = Histogram("qbitel_inference_duration_seconds", "Inference duration", ["component"])
MODEL_ACCURACY = Gauge("qbitel_model_accuracy", "Model accuracy", ["model_name"])
ACTIVE_MODELS = Gauge("qbitel_active_models", "Number of active models")


class AIEngine:
    """Minimal orchestration layer used by tests to exercise core workflows."""

    def __init__(self, config: Config):
        self.config = config
        self.state = "initialized"
        self.protocol_discovery = None
        self.field_detector = None
        self.anomaly_detector = None
        self.model_registry = None
        self.metrics: Dict[str, Any] = {}
        self._start_time = time.time()

    async def _initialize_components(self) -> None:
        """Hook for asynchronous component initialization."""
        # Sub-classes or tests can override/patch this method.
        return None

    async def initialize(self) -> None:
        """Initialize engine components and transition to ready state."""
        try:
            await self._initialize_components()
            self.state = "ready"
        except Exception as exc:  # pragma: no cover - defensive
            self.state = "failed"
            raise AIEngineException("Failed to initialize AI Engine") from exc

    async def shutdown(self) -> None:
        """Shutdown the engine and release resources."""
        self.state = "stopped"

    async def discover_protocol(self, model_input: Any) -> Any:
        """Delegate protocol discovery to the configured component."""
        self._ensure_ready("protocol discovery")
        if not self.protocol_discovery or not hasattr(self.protocol_discovery, "discover_protocol"):
            raise AIEngineException("Protocol discovery component not configured")

        start_time = time.time()
        try:
            result = await self.protocol_discovery.discover_protocol(model_input)
        except Exception as exc:
            raise AIEngineException("Protocol discovery failed") from exc

        self._annotate_processing_time(result, start_time)
        return result

    async def detect_fields(self, model_input: Any) -> Any:
        """Delegate field detection to the configured component."""
        self._ensure_ready("field detection")
        if not self.field_detector or not hasattr(self.field_detector, "detect_fields"):
            raise AIEngineException("Field detector component not configured")

        start_time = time.time()
        try:
            result = await self.field_detector.detect_fields(model_input)
        except Exception as exc:
            raise AIEngineException("Field detection failed") from exc

        self._annotate_processing_time(result, start_time)
        return result

    async def detect_anomalies(self, model_input: Any) -> Any:
        """Delegate anomaly detection to the configured component."""
        self._ensure_ready("anomaly detection")
        if not self.anomaly_detector:
            raise AIEngineException("Anomaly detector component not configured")

        detect_fn = None
        if hasattr(self.anomaly_detector, "detect_anomalies"):
            detect_fn = self.anomaly_detector.detect_anomalies
        elif hasattr(self.anomaly_detector, "detect"):
            detect_fn = self.anomaly_detector.detect
        else:
            raise AIEngineException("Anomaly detector does not expose a detect method")

        start_time = time.time()
        try:
            result = await detect_fn(model_input)
        except Exception as exc:
            raise AIEngineException("Anomaly detection failed") from exc

        self._annotate_processing_time(result, start_time)
        return result

    async def get_status(self) -> Dict[str, Any]:
        """Return a lightweight status summary used by health checks."""
        components = {
            "protocol_discovery": self.protocol_discovery is not None,
            "field_detector": self.field_detector is not None,
            "anomaly_detector": self.anomaly_detector is not None,
            "model_registry": self.model_registry is not None,
        }

        return {
            "status": self.state,
            "uptime_seconds": time.time() - self._start_time,
            "components": components,
            "system_metrics": dict(self.metrics),
        }

    async def cleanup(self) -> None:
        """Cleanup resources owned by the engine."""
        if self.model_registry and hasattr(self.model_registry, "cleanup"):
            cleanup_fn = self.model_registry.cleanup
            result = cleanup_fn()
            if asyncio.iscoroutine(result):
                await result
        self.state = "stopped"

    def _ensure_ready(self, operation: str) -> None:
        if self.state != "ready":
            raise AIEngineException("AI Engine not ready", component=operation)

    def _annotate_processing_time(self, result: Any, start_time: float) -> None:
        if hasattr(result, "processing_time_ms"):
            if getattr(result, "processing_time_ms", None) is None:
                result.processing_time_ms = (time.time() - start_time) * 1000.0


class QbitelAIEngine:
    """
    Main AI Engine that orchestrates all machine learning components.

    This class provides a unified interface for protocol discovery, field detection,
    and anomaly detection capabilities.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the AI Engine.

        Args:
            config: Configuration object. If None, loads from environment.
        """
        self.config = config or get_config()
        self.logger = self._setup_logging()

        # Component initialization
        self._protocol_classifier: Optional[ProtocolClassifier] = None
        self._pcfg_inference: Optional[PCFGInference] = None
        self._field_detector: Optional[FieldDetector] = None
        self._anomaly_detector: Optional[EnsembleAnomalyDetector] = None
        self._feature_extractor: Optional[FeatureExtractor] = None

        # Infrastructure components
        self._model_registry: Optional["ModelRegistry"] = None
        self._metrics_collector: Optional["MetricsCollector"] = None
        self._executor: Optional[ThreadPoolExecutor] = None

        # State management
        self._initialized = False
        self._models_loaded = {}
        self._performance_stats = {}

        self.logger.info(f"AI Engine initialized with config: {self.config.environment.value}")

    async def initialize(self) -> None:
        """Initialize all AI components and load models."""
        if self._initialized:
            self.logger.warning("AI Engine already initialized")
            return

        try:
            self.logger.info("Initializing AI Engine components...")

            # Initialize infrastructure
            await self._initialize_infrastructure()

            # Initialize AI components
            await self._initialize_ai_components()

            # Load models
            await self._load_models()

            # Start monitoring
            await self._start_monitoring()

            self._initialized = True
            ACTIVE_MODELS.set(len(self._models_loaded))

            self.logger.info("AI Engine initialization completed successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize AI Engine: {e}")
            raise QbitelAIException(f"Engine initialization failed: {e}")

    async def shutdown(self) -> None:
        """Shutdown the AI Engine and cleanup resources."""
        self.logger.info("Shutting down AI Engine...")

        try:
            # Stop monitoring
            if self._metrics_collector:
                await self._metrics_collector.stop()

            # Shutdown executor
            if self._executor:
                self._executor.shutdown(wait=True)

            # Cleanup models
            self._models_loaded.clear()

            self._initialized = False
            ACTIVE_MODELS.set(0)

            self.logger.info("AI Engine shutdown completed")

        except Exception as e:
            self.logger.error(f"Error during AI Engine shutdown: {e}")
            raise QbitelAIException(f"Engine shutdown failed: {e}")

    async def discover_protocol(self, packet_data: bytes, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Discover protocol from packet data.

        Args:
            packet_data: Raw packet data
            metadata: Optional metadata about the packet

        Returns:
            Protocol discovery results including type, confidence, and structure
        """
        if not self._initialized:
            raise QbitelAIException("AI Engine not initialized")

        start_time = time.time()

        try:
            # Extract features from packet data
            features = await self._extract_features(packet_data, "protocol_discovery")

            # Classify protocol using ML model
            protocol_result = await self._classify_protocol(features)

            # If unknown protocol, infer grammar structure
            if protocol_result.get("confidence", 0) < 0.8:
                grammar_result = await self._infer_grammar(packet_data)
                protocol_result["grammar"] = grammar_result

            # Update metrics
            INFERENCE_COUNTER.labels(component="protocol_discovery", status="success").inc()
            INFERENCE_DURATION.labels(component="protocol_discovery").observe(time.time() - start_time)

            return {
                "protocol_type": protocol_result.get("protocol_type", "unknown"),
                "confidence": protocol_result.get("confidence", 0.0),
                "structure": protocol_result.get("structure", {}),
                "grammar": protocol_result.get("grammar"),
                "metadata": metadata or {},
                "processing_time": time.time() - start_time,
            }

        except Exception as e:
            INFERENCE_COUNTER.labels(component="protocol_discovery", status="error").inc()
            self.logger.error(f"Protocol discovery failed: {e}")
            raise InferenceException(f"Protocol discovery failed: {e}")

    async def detect_fields(self, message_data: bytes, protocol_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Detect fields in a protocol message.

        Args:
            message_data: Protocol message data
            protocol_type: Optional protocol type hint

        Returns:
            List of detected fields with boundaries and types
        """
        if not self._initialized:
            raise QbitelAIException("AI Engine not initialized")

        start_time = time.time()

        try:
            # Extract features for field detection
            features = await self._extract_features(message_data, "field_detection")

            # Detect field boundaries using BiLSTM-CRF
            fields = await self._detect_field_boundaries(features, protocol_type)

            # Infer field types
            for field in fields:
                field["type"] = await self._infer_field_type(message_data[field["start"] : field["end"]])

            # Update metrics
            INFERENCE_COUNTER.labels(component="field_detection", status="success").inc()
            INFERENCE_DURATION.labels(component="field_detection").observe(time.time() - start_time)

            return fields

        except Exception as e:
            INFERENCE_COUNTER.labels(component="field_detection", status="error").inc()
            self.logger.error(f"Field detection failed: {e}")
            raise InferenceException(f"Field detection failed: {e}")

    async def detect_anomaly(
        self,
        data: Union[bytes, np.ndarray, Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Detect anomalies in the provided data.

        Args:
            data: Input data for anomaly detection
            context: Optional context information

        Returns:
            Anomaly detection results with scores and explanations
        """
        if not self._initialized:
            raise QbitelAIException("AI Engine not initialized")

        start_time = time.time()

        try:
            # Extract features for anomaly detection
            if isinstance(data, bytes):
                features = await self._extract_features(data, "anomaly_detection")
            else:
                features = data

            # Run ensemble anomaly detection
            anomaly_result = await self._detect_anomaly_ensemble(features, context)

            # Update metrics
            INFERENCE_COUNTER.labels(component="anomaly_detection", status="success").inc()
            INFERENCE_DURATION.labels(component="anomaly_detection").observe(time.time() - start_time)

            return {
                "is_anomaly": anomaly_result["score"] > self.config.anomaly_threshold,
                "anomaly_score": anomaly_result["score"],
                "confidence": anomaly_result["confidence"],
                "explanation": anomaly_result.get("explanation", ""),
                "detector_scores": anomaly_result.get("individual_scores", {}),
                "context": context or {},
                "processing_time": time.time() - start_time,
            }

        except Exception as e:
            INFERENCE_COUNTER.labels(component="anomaly_detection", status="error").inc()
            self.logger.error(f"Anomaly detection failed: {e}")
            raise InferenceException(f"Anomaly detection failed: {e}")

    async def batch_process(self, batch_data: List[bytes], operation: str = "discover_protocol") -> List[Dict[str, Any]]:
        """
        Process a batch of data efficiently.

        Args:
            batch_data: List of data items to process
            operation: Operation to perform ("discover_protocol", "detect_fields", "detect_anomaly")

        Returns:
            List of processing results
        """
        if not self._initialized:
            raise QbitelAIException("AI Engine not initialized")

        self.logger.info(f"Processing batch of {len(batch_data)} items with operation: {operation}")

        # Process in parallel using thread pool
        loop = asyncio.get_event_loop()

        if operation == "discover_protocol":
            tasks = [loop.run_in_executor(self._executor, self._sync_discover_protocol, data) for data in batch_data]
        elif operation == "detect_fields":
            tasks = [loop.run_in_executor(self._executor, self._sync_detect_fields, data) for data in batch_data]
        elif operation == "detect_anomaly":
            tasks = [loop.run_in_executor(self._executor, self._sync_detect_anomaly, data) for data in batch_data]
        else:
            raise InferenceException(f"Unknown operation: {operation}")

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Batch item {i} failed: {result}")
                processed_results.append({"error": str(result), "index": i})
            else:
                processed_results.append(result)

        return processed_results

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            "loaded_models": list(self._models_loaded.keys()),
            "model_details": {
                name: {
                    "version": model.get("version", "unknown"),
                    "type": model.get("type", "unknown"),
                    "device": model.get("device", "unknown"),
                    "last_updated": model.get("last_updated"),
                }
                for name, model in self._models_loaded.items()
            },
            "performance_stats": self._performance_stats,
            "config": asdict(self.config),
        }

    async def update_model(self, model_name: str, model_version: str) -> None:
        """Update a specific model to a new version."""
        self.logger.info(f"Updating model {model_name} to version {model_version}")

        try:
            # Load new model version
            model_info = await self._model_registry.load_model(model_name, model_version)

            # Update model in components
            if model_name == "protocol_classifier" and self._protocol_classifier:
                await self._protocol_classifier.load_model(model_info)
            elif model_name == "field_detector" and self._field_detector:
                await self._field_detector.load_model(model_info)
            elif model_name.startswith("anomaly_") and self._anomaly_detector:
                await self._anomaly_detector.update_detector(model_name, model_info)

            # Update model registry
            self._models_loaded[model_name] = {
                "version": model_version,
                "model_info": model_info,
                "last_updated": time.time(),
            }

            self.logger.info(f"Successfully updated model {model_name} to version {model_version}")

        except Exception as e:
            self.logger.error(f"Failed to update model {model_name}: {e}")
            raise ModelException(f"Model update failed: {e}", model_name, model_version)

    # Private methods

    async def _initialize_infrastructure(self) -> None:
        """Initialize infrastructure components."""
        # Initialize thread pool executor
        self._executor = ThreadPoolExecutor(
            max_workers=self.config.inference.num_workers,
            thread_name_prefix="qbitel-",
        )

        # Initialize model registry
        from ..models.registry import ModelRegistry

        self._model_registry = ModelRegistry(self.config)
        await self._model_registry.initialize()

        # Initialize metrics collector
        from ..monitoring.metrics import MetricsCollector

        self._metrics_collector = MetricsCollector(self.config)
        await self._metrics_collector.initialize()

    async def _initialize_ai_components(self) -> None:
        """Initialize AI components."""
        # Initialize feature extractor
        self._feature_extractor = FeatureExtractor(self.config)

        # Initialize protocol discovery components
        if self.config.discovery_enabled:
            self._protocol_classifier = ProtocolClassifier(self.config)
            self._pcfg_inference = PCFGInference(self.config)

        # Initialize field detection
        if self.config.field_detection_enabled:
            self._field_detector = FieldDetector(self.config)

        # Initialize anomaly detection
        if self.config.anomaly_detection_enabled:
            self._anomaly_detector = EnsembleAnomalyDetector(self.config)

    async def _load_models(self) -> None:
        """Load all configured models."""
        for model_name, model_config in self.config.models.items():
            try:
                model_info = await self._model_registry.load_model(model_name, model_config.version)
                self._models_loaded[model_name] = {
                    "version": model_config.version,
                    "config": model_config,
                    "model_info": model_info,
                    "last_updated": time.time(),
                }
                self.logger.info(f"Loaded model: {model_name} v{model_config.version}")

            except Exception as e:
                self.logger.error(f"Failed to load model {model_name}: {e}")
                if self.config.environment.value == "production":
                    raise ModelException(f"Critical model loading failed: {e}", model_name)

    async def _start_monitoring(self) -> None:
        """Start monitoring and metrics collection."""
        if self._metrics_collector:
            await self._metrics_collector.start()

    async def _extract_features(self, data: bytes, context: str) -> np.ndarray:
        """Extract features from data."""
        return await self._feature_extractor.extract(data, context)

    async def _classify_protocol(self, features: np.ndarray) -> Dict[str, Any]:
        """Classify protocol from features."""
        return await self._protocol_classifier.predict(features)

    async def _infer_grammar(self, data: bytes) -> Dict[str, Any]:
        """Infer grammar structure from data."""
        return await self._pcfg_inference.infer(data)

    async def _detect_field_boundaries(self, features: np.ndarray, protocol_type: Optional[str]) -> List[Dict[str, Any]]:
        """Detect field boundaries."""
        return await self._field_detector.detect_boundaries(features, protocol_type)

    async def _infer_field_type(self, field_data: bytes) -> str:
        """Infer the type of a field."""
        return await self._field_detector.infer_type(field_data)

    async def _detect_anomaly_ensemble(
        self,
        features: Union[np.ndarray, Dict[str, Any]],
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run ensemble anomaly detection."""
        return await self._anomaly_detector.detect(features, context)

    def _sync_discover_protocol(self, data: bytes) -> Dict[str, Any]:
        """Synchronous wrapper for protocol discovery (thread-safe)."""
        try:
            loop = asyncio.get_running_loop()
            # Already in an event loop — use concurrent.futures to bridge
            import concurrent.futures

            future = asyncio.run_coroutine_threadsafe(self.discover_protocol(data), loop)
            return future.result(timeout=30)
        except RuntimeError:
            # No event loop running — safe to create a new one
            return asyncio.run(self.discover_protocol(data))

    def _sync_detect_fields(self, data: bytes) -> List[Dict[str, Any]]:
        """Synchronous wrapper for field detection (thread-safe)."""
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures

            future = asyncio.run_coroutine_threadsafe(self.detect_fields(data), loop)
            return future.result(timeout=30)
        except RuntimeError:
            return asyncio.run(self.detect_fields(data))

    def _sync_detect_anomaly(self, data: bytes) -> Dict[str, Any]:
        """Synchronous wrapper for anomaly detection (thread-safe)."""
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures

            future = asyncio.run_coroutine_threadsafe(self.detect_anomaly(data), loop)
            return future.result(timeout=30)
        except RuntimeError:
            return asyncio.run(self.detect_anomaly(data))

    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging."""
        logger = logging.getLogger("qbitel.ai.engine")
        logger.setLevel(getattr(logging, self.config.log_level.value))

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger
