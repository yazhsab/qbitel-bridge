#!/usr/bin/env python3
"""
QBITEL - AI/ML to Protocol Processing Bridge
Connects the AI/ML engine to the protocol processing pipeline for real-time inference.
"""

import asyncio
import logging
import json
import time
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
import grpc
import grpc.aio
from pathlib import Path
import sys
import struct

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from integration.config.unified_config import get_config, get_service_config
from integration.orchestrator.service_integration import get_orchestrator, Message
from ai_engine.detection.field_detector import FieldDetector, BiLSTMCRF
from ai_engine.models.model_manager import ModelManager
from ai_engine.features.extractors import (
    StatisticalFeatureExtractor,
    StructuralFeatureExtractor,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProtocolData:
    """Protocol data structure for processing"""

    raw_data: bytes
    timestamp: float
    source_ip: str
    dest_ip: str
    source_port: int
    dest_port: int
    protocol: str = "unknown"
    application: str = "unknown"
    metadata: Dict[str, Any] = None


@dataclass
class AIAnalysisResult:
    """AI analysis result structure"""

    protocol_classification: Dict[str, Any]
    field_boundaries: List[Dict[str, Any]]
    anomaly_score: float
    threat_level: str
    confidence: float
    processing_time_ms: float
    model_versions: Dict[str, str]


class MLProtocolBridge:
    """
    Bridge between AI/ML engine and protocol processing pipeline.
    Handles real-time AI inference for protocol discovery and threat detection.
    """

    def __init__(self):
        self.config = get_service_config("ai_engine")
        self.orchestrator = get_orchestrator()

        # AI Components
        self.field_detector: Optional[FieldDetector] = None
        self.model_manager: Optional[ModelManager] = None
        self.statistical_extractor: Optional[StatisticalFeatureExtractor] = None
        self.structural_extractor: Optional[StructuralFeatureExtractor] = None

        # Processing queues
        self.inference_queue = asyncio.Queue(maxsize=10000)
        self.result_queue = asyncio.Queue(maxsize=10000)

        # Performance tracking
        self.processed_count = 0
        self.error_count = 0
        self.total_processing_time = 0.0

        # Thread pools for different AI tasks
        self.ml_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="ml_")
        self.feature_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="feature_")

        # Cache for frequent protocols
        self.protocol_cache = {}
        self.cache_hit_count = 0

        # Model performance tracking
        self.model_metrics = {
            "field_detection_accuracy": 0.0,
            "protocol_classification_accuracy": 0.0,
            "anomaly_detection_precision": 0.0,
            "average_inference_time_ms": 0.0,
        }

    async def initialize(self):
        """Initialize AI/ML components and models"""
        logger.info("Initializing AI/ML Protocol Bridge...")

        try:
            # Initialize AI components
            await self._init_ai_components()

            # Load trained models
            await self._load_models()

            # Initialize feature extractors
            await self._init_feature_extractors()

            # Start background processing tasks
            await self._start_processing_tasks()

            logger.info("AI/ML Protocol Bridge initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize AI/ML bridge: {e}")
            raise

    async def _init_ai_components(self):
        """Initialize core AI components"""
        try:
            # Initialize field detector
            device = torch.device("cuda" if torch.cuda.is_available() and self.config.get("use_gpu") else "cpu")

            self.field_detector = FieldDetector(
                vocab_size=256,  # Byte vocabulary
                embedding_dim=128,
                hidden_dim=256,
                num_tags=5,  # BIO + field types
                device=device,
            )

            # Initialize model manager
            self.model_manager = ModelManager(
                model_cache_dir=self.config.get("model_cache_dir", "/tmp/models"),
                use_gpu=self.config.get("use_gpu", False),
            )

            logger.info(f"AI components initialized on device: {device}")

        except Exception as e:
            logger.error(f"Error initializing AI components: {e}")
            raise

    async def _load_models(self):
        """Load pre-trained models"""
        try:
            # Load field detection model
            field_model_path = Path(self.config.get("model_cache_dir", "/tmp/models")) / "field_detector.pt"
            if field_model_path.exists():
                self.field_detector.load_model(str(field_model_path))
                logger.info("Field detection model loaded")
            else:
                logger.warning("Field detection model not found, using untrained model")

            # Load protocol classification models
            await self.model_manager.load_model_ensemble(
                [
                    "protocol_classifier_cnn",
                    "protocol_classifier_lstm",
                    "anomaly_detector_vae",
                ]
            )

            logger.info("ML models loaded successfully")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            # Continue with untrained models for development

    async def _init_feature_extractors(self):
        """Initialize feature extraction components"""
        try:
            self.statistical_extractor = StatisticalFeatureExtractor(window_size=self.config.get("entropy_window_size", 1000))

            self.structural_extractor = StructuralFeatureExtractor(
                max_pattern_length=self.config.get("max_sequence_length", 512)
            )

            logger.info("Feature extractors initialized")

        except Exception as e:
            logger.error(f"Error initializing feature extractors: {e}")
            raise

    async def _start_processing_tasks(self):
        """Start background processing tasks"""
        # Start inference processing
        asyncio.create_task(self._inference_processor())

        # Start result handler
        asyncio.create_task(self._result_handler())

        # Start metrics collector
        asyncio.create_task(self._metrics_collector())

        logger.info("Background processing tasks started")

    async def analyze_protocol(self, protocol_data: ProtocolData) -> AIAnalysisResult:
        """
        Main entry point for protocol analysis.
        Performs comprehensive AI analysis of protocol data.
        """
        start_time = time.time()

        try:
            # Check cache first for known protocols
            cache_key = self._generate_cache_key(protocol_data)
            if cache_key in self.protocol_cache:
                cached_result = self.protocol_cache[cache_key]
                self.cache_hit_count += 1

                # Update timestamp and return cached result
                cached_result.processing_time_ms = (time.time() - start_time) * 1000
                return cached_result

            # Extract features
            features = await self._extract_features(protocol_data)

            # Run parallel AI analyses
            tasks = [
                self._classify_protocol(features),
                self._detect_field_boundaries(protocol_data.raw_data),
                self._detect_anomalies(features),
                self._assess_threat_level(features, protocol_data),
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Combine results
            protocol_classification = results[0] if not isinstance(results[0], Exception) else {}
            field_boundaries = results[1] if not isinstance(results[1], Exception) else []
            anomaly_score = results[2] if not isinstance(results[2], Exception) else 0.0
            threat_level = results[3] if not isinstance(results[3], Exception) else "low"

            # Calculate overall confidence
            confidence = self._calculate_confidence(protocol_classification, field_boundaries, anomaly_score)

            # Create result
            result = AIAnalysisResult(
                protocol_classification=protocol_classification,
                field_boundaries=field_boundaries,
                anomaly_score=anomaly_score,
                threat_level=threat_level,
                confidence=confidence,
                processing_time_ms=(time.time() - start_time) * 1000,
                model_versions=self.model_manager.get_model_versions(),
            )

            # Cache result if confidence is high
            if confidence > 0.8:
                self.protocol_cache[cache_key] = result

                # Limit cache size
                if len(self.protocol_cache) > 1000:
                    oldest_key = next(iter(self.protocol_cache))
                    del self.protocol_cache[oldest_key]

            # Update metrics
            self.processed_count += 1
            self.total_processing_time += result.processing_time_ms

            return result

        except Exception as e:
            logger.error(f"Error analyzing protocol: {e}")
            self.error_count += 1

            # Return default result on error
            return AIAnalysisResult(
                protocol_classification={"protocol": "unknown", "confidence": 0.0},
                field_boundaries=[],
                anomaly_score=1.0,  # High anomaly score for errors
                threat_level="unknown",
                confidence=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
                model_versions={},
            )

    async def _extract_features(self, protocol_data: ProtocolData) -> Dict[str, Any]:
        """Extract features from protocol data"""
        loop = asyncio.get_event_loop()

        # Run feature extraction in thread pool
        statistical_features = await loop.run_in_executor(
            self.feature_executor,
            self.statistical_extractor.extract,
            protocol_data.raw_data,
        )

        structural_features = await loop.run_in_executor(
            self.feature_executor,
            self.structural_extractor.extract,
            protocol_data.raw_data,
        )

        # Combine features
        features = {
            **statistical_features,
            **structural_features,
            "packet_size": len(protocol_data.raw_data),
            "source_port": protocol_data.source_port,
            "dest_port": protocol_data.dest_port,
            "time_of_day": time.localtime(protocol_data.timestamp).tm_hour,
        }

        return features

    async def _classify_protocol(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Classify protocol using ensemble of models"""
        try:
            # Prepare input tensor
            feature_vector = self._features_to_tensor(features)

            # Run ensemble classification
            loop = asyncio.get_event_loop()
            predictions = await loop.run_in_executor(
                self.ml_executor,
                self.model_manager.predict_ensemble,
                "protocol_classification",
                feature_vector,
            )

            # Process ensemble results
            protocol_probabilities = {}
            for model_name, prediction in predictions.items():
                if isinstance(prediction, torch.Tensor):
                    probs = F.softmax(prediction, dim=-1).cpu().numpy()
                    protocol_probabilities[model_name] = probs.tolist()

            # Ensemble voting
            final_prediction = self._ensemble_vote(protocol_probabilities)

            return {
                "protocol": final_prediction["protocol"],
                "confidence": final_prediction["confidence"],
                "probabilities": final_prediction["probabilities"],
                "model_predictions": protocol_probabilities,
            }

        except Exception as e:
            logger.error(f"Error in protocol classification: {e}")
            return {"protocol": "unknown", "confidence": 0.0}

    async def _detect_field_boundaries(self, raw_data: bytes) -> List[Dict[str, Any]]:
        """Detect field boundaries in protocol data"""
        try:
            if not self.field_detector:
                return []

            # Convert bytes to sequence
            data_sequence = list(raw_data[:512])  # Limit sequence length

            # Run field detection
            loop = asyncio.get_event_loop()
            field_boundaries = await loop.run_in_executor(self.ml_executor, self.field_detector.detect_fields, data_sequence)

            return field_boundaries

        except Exception as e:
            logger.error(f"Error in field boundary detection: {e}")
            return []

    async def _detect_anomalies(self, features: Dict[str, Any]) -> float:
        """Detect anomalies in protocol data"""
        try:
            # Prepare features for anomaly detection
            feature_vector = self._features_to_tensor(features)

            # Run VAE anomaly detection
            loop = asyncio.get_event_loop()
            anomaly_scores = await loop.run_in_executor(
                self.ml_executor,
                self.model_manager.predict,
                "anomaly_detector_vae",
                feature_vector,
            )

            if isinstance(anomaly_scores, torch.Tensor):
                return float(anomaly_scores.mean().item())

            return 0.0

        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return 0.5  # Neutral score on error

    async def _assess_threat_level(self, features: Dict[str, Any], protocol_data: ProtocolData) -> str:
        """Assess threat level based on analysis results"""
        try:
            # Simple rule-based threat assessment
            threat_score = 0

            # Check for suspicious ports
            suspicious_ports = [1337, 4444, 5555, 6666, 31337]
            if protocol_data.dest_port in suspicious_ports or protocol_data.source_port in suspicious_ports:
                threat_score += 0.3

            # Check packet size anomalies
            packet_size = len(protocol_data.raw_data)
            if packet_size > 8192 or packet_size < 20:  # Unusually large or small
                threat_score += 0.2

            # Check entropy (encrypted/compressed data)
            entropy = features.get("entropy", 0)
            if entropy > 7.5:  # High entropy suggests encryption
                threat_score += 0.2

            # Check for uncommon protocols
            if features.get("protocol_confidence", 1.0) < 0.5:
                threat_score += 0.3

            # Convert score to level
            if threat_score >= 0.7:
                return "critical"
            elif threat_score >= 0.5:
                return "high"
            elif threat_score >= 0.3:
                return "medium"
            else:
                return "low"

        except Exception as e:
            logger.error(f"Error assessing threat level: {e}")
            return "unknown"

    def _calculate_confidence(
        self,
        protocol_classification: Dict[str, Any],
        field_boundaries: List[Dict[str, Any]],
        anomaly_score: float,
    ) -> float:
        """Calculate overall confidence score"""
        try:
            # Protocol classification confidence
            protocol_confidence = protocol_classification.get("confidence", 0.0)

            # Field detection confidence
            field_confidence = 0.0
            if field_boundaries:
                field_confidences = [f.get("confidence", 0.0) for f in field_boundaries]
                field_confidence = sum(field_confidences) / len(field_confidences)

            # Anomaly score (lower is better for confidence)
            anomaly_confidence = 1.0 - min(anomaly_score, 1.0)

            # Weighted average
            overall_confidence = protocol_confidence * 0.5 + field_confidence * 0.3 + anomaly_confidence * 0.2

            return min(max(overall_confidence, 0.0), 1.0)

        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.0

    def _features_to_tensor(self, features: Dict[str, Any]) -> torch.Tensor:
        """Convert features dictionary to tensor"""
        try:
            # Define feature order and defaults
            feature_keys = [
                "entropy",
                "byte_frequency_variance",
                "pattern_count",
                "packet_size",
                "source_port",
                "dest_port",
                "time_of_day",
            ]

            feature_vector = []
            for key in feature_keys:
                value = features.get(key, 0.0)
                if isinstance(value, (list, np.ndarray)):
                    if len(value) > 0:
                        feature_vector.append(float(np.mean(value)))
                    else:
                        feature_vector.append(0.0)
                else:
                    feature_vector.append(float(value))

            return torch.tensor(feature_vector, dtype=torch.float32)

        except Exception as e:
            logger.error(f"Error converting features to tensor: {e}")
            return torch.zeros(len(feature_keys), dtype=torch.float32)

    def _ensemble_vote(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Perform ensemble voting on model predictions"""
        try:
            if not predictions:
                return {"protocol": "unknown", "confidence": 0.0, "probabilities": {}}

            # Aggregate predictions (simple average for now)
            protocol_votes = {}
            total_confidence = 0.0

            for model_name, probs in predictions.items():
                if isinstance(probs, list) and len(probs) > 0:
                    max_idx = np.argmax(probs)
                    max_prob = probs[max_idx]

                    # Map index to protocol name (simplified)
                    protocol_names = ["HTTP", "HTTPS", "FTP", "SSH", "DNS", "UNKNOWN"]
                    protocol = protocol_names[min(max_idx, len(protocol_names) - 1)]

                    if protocol in protocol_votes:
                        protocol_votes[protocol] += max_prob
                    else:
                        protocol_votes[protocol] = max_prob

                    total_confidence += max_prob

            if not protocol_votes:
                return {"protocol": "unknown", "confidence": 0.0, "probabilities": {}}

            # Find winning protocol
            winning_protocol = max(protocol_votes, key=protocol_votes.get)
            winning_confidence = protocol_votes[winning_protocol] / len(predictions)

            # Normalize probabilities
            total_votes = sum(protocol_votes.values())
            normalized_probs = {k: v / total_votes for k, v in protocol_votes.items()}

            return {
                "protocol": winning_protocol,
                "confidence": winning_confidence,
                "probabilities": normalized_probs,
            }

        except Exception as e:
            logger.error(f"Error in ensemble voting: {e}")
            return {"protocol": "unknown", "confidence": 0.0, "probabilities": {}}

    def _generate_cache_key(self, protocol_data: ProtocolData) -> str:
        """Generate cache key for protocol data"""
        try:
            # Use hash of first 64 bytes + ports for caching
            data_sample = protocol_data.raw_data[:64]
            key_data = f"{data_sample}_{protocol_data.source_port}_{protocol_data.dest_port}"
            return str(hash(key_data))
        except Exception as e:
            logger.error(f"Error generating cache key: {e}")
            return "default"

    async def _inference_processor(self):
        """Background task for processing inference requests"""
        while True:
            try:
                # Get inference request from queue
                request = await asyncio.wait_for(self.inference_queue.get(), timeout=1.0)

                # Process request
                result = await self.analyze_protocol(request["protocol_data"])

                # Send result
                response = {
                    "correlation_id": request.get("correlation_id"),
                    "result": asdict(result),
                    "timestamp": time.time(),
                }

                await self.result_queue.put(response)

            except asyncio.TimeoutError:
                continue  # Normal timeout
            except Exception as e:
                logger.error(f"Error in inference processor: {e}")
                await asyncio.sleep(1)

    async def _result_handler(self):
        """Handle analysis results and send to orchestrator"""
        while True:
            try:
                # Get result from queue
                result = await asyncio.wait_for(self.result_queue.get(), timeout=1.0)

                # Send result via orchestrator
                message = Message(
                    id=f"ai_result_{time.time()}",
                    timestamp=time.time(),
                    source="ai_bridge",
                    destination="orchestrator",
                    message_type="ai_analysis_result",
                    payload=result,
                    correlation_id=result.get("correlation_id"),
                )

                await self.orchestrator.send_message(message)

            except asyncio.TimeoutError:
                continue  # Normal timeout
            except Exception as e:
                logger.error(f"Error in result handler: {e}")
                await asyncio.sleep(1)

    async def _metrics_collector(self):
        """Collect and publish performance metrics"""
        while True:
            try:
                await asyncio.sleep(30)  # Collect every 30 seconds

                # Calculate metrics
                total_time = self.total_processing_time
                processed = max(self.processed_count, 1)

                metrics = {
                    "processed_count": self.processed_count,
                    "error_count": self.error_count,
                    "error_rate": self.error_count / processed,
                    "average_processing_time_ms": total_time / processed,
                    "cache_hit_rate": (self.cache_hit_count / processed if processed > 0 else 0),
                    "cache_size": len(self.protocol_cache),
                    "queue_sizes": {
                        "inference_queue": self.inference_queue.qsize(),
                        "result_queue": self.result_queue.qsize(),
                    },
                    "model_metrics": self.model_metrics,
                }

                # Send metrics to orchestrator
                message = Message(
                    id=f"ai_metrics_{time.time()}",
                    timestamp=time.time(),
                    source="ai_bridge",
                    destination="orchestrator",
                    message_type="metric_update",
                    payload={
                        "component": "ai_bridge",
                        "metrics": metrics,
                    },
                )

                await self.orchestrator.send_message(message)

            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(30)

    # Public API
    async def queue_analysis(self, protocol_data: ProtocolData, correlation_id: Optional[str] = None):
        """Queue protocol data for analysis"""
        request = {
            "protocol_data": protocol_data,
            "correlation_id": correlation_id,
            "timestamp": time.time(),
        }

        await self.inference_queue.put(request)

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            "processed_count": self.processed_count,
            "error_count": self.error_count,
            "average_processing_time_ms": self.total_processing_time / max(self.processed_count, 1),
            "cache_hit_rate": self.cache_hit_count / max(self.processed_count, 1),
            "model_status": (self.model_manager.get_model_status() if self.model_manager else {}),
        }

    async def update_models(self, model_updates: Dict[str, str]):
        """Update ML models with new versions"""
        try:
            for model_name, model_path in model_updates.items():
                await self.model_manager.load_model(model_name, model_path)

            logger.info(f"Updated models: {list(model_updates.keys())}")

        except Exception as e:
            logger.error(f"Error updating models: {e}")

    async def shutdown(self):
        """Shutdown bridge and cleanup resources"""
        logger.info("Shutting down AI/ML Protocol Bridge...")

        # Shutdown thread pools
        self.ml_executor.shutdown(wait=True)
        self.feature_executor.shutdown(wait=True)

        logger.info("AI/ML Protocol Bridge shutdown complete")


# Global bridge instance
_bridge = None


def get_ml_bridge() -> MLProtocolBridge:
    """Get global ML bridge instance"""
    global _bridge
    if _bridge is None:
        _bridge = MLProtocolBridge()
    return _bridge


async def main():
    """Main entry point for AI/ML bridge"""
    bridge = MLProtocolBridge()

    try:
        await bridge.initialize()

        # Keep running
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Bridge error: {e}")
    finally:
        await bridge.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
