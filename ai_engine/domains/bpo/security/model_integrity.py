"""
AI Model Integrity Protection Module

Protects AI model artifacts used in BPO operations with PQC-signed
integrity verification and tamper detection.

AI models deployed in BPO environments face integrity threats:
- Model file tampering - modifying weights or architecture
- Training data poisoning - corrupting the training pipeline
- Inference manipulation - adversarial inputs causing misclassification
- Model drift - gradual degradation of model performance
- Supply chain attacks - compromised model distribution
- Adversarial inputs - crafted inputs to force specific outputs

Integrates with QBITEL's PQC infrastructure for ML-DSA-87 signed
model artifacts and tamper-proof audit logging.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple
import hashlib
import logging
import re
import uuid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ModelType(Enum):
    """Types of AI models deployed in BPO operations."""

    SPEECH_RECOGNITION = auto()     # ASR models for call transcription
    SENTIMENT_ANALYSIS = auto()     # Customer sentiment scoring
    QUALITY_MONITORING = auto()     # Agent quality assessment
    FRAUD_DETECTION = auto()        # Fraud pattern detection
    INTENT_CLASSIFICATION = auto()  # Caller intent classification
    ENTITY_EXTRACTION = auto()      # Named entity recognition
    SUMMARIZATION = auto()          # Call/conversation summarization
    TRANSLATION = auto()            # Real-time language translation


class IntegrityEventType(Enum):
    """Types of model integrity events."""

    MODEL_LOADED = auto()               # Model loaded into runtime
    MODEL_VERIFIED = auto()             # Model integrity verified
    MODEL_TAMPERED = auto()             # Model tampering detected
    TRAINING_DATA_VERIFIED = auto()     # Training data integrity verified
    TRAINING_DATA_TAMPERED = auto()     # Training data tampering detected
    INFERENCE_ANOMALY = auto()          # Anomalous inference behavior
    DRIFT_DETECTED = auto()             # Model drift detected
    ADVERSARIAL_INPUT_DETECTED = auto() # Adversarial input detected


class IntegrityAction(Enum):
    """Actions taken in response to integrity events."""

    ALLOW = auto()              # Allow operation to proceed
    BLOCK_INFERENCE = auto()    # Block inference with this model
    ROLLBACK_MODEL = auto()     # Rollback to previous verified version
    ALERT = auto()              # Alert the security/ML ops team
    QUARANTINE_MODEL = auto()   # Quarantine the model artifact
    RETRAIN = auto()            # Flag model for retraining


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ModelArtifact:
    """
    A signed AI model artifact with PQC integrity metadata.

    Each model artifact is hashed with SHA3-256 and signed with
    ML-DSA-87 to detect tampering at load time and periodically
    during operation.
    """

    artifact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_type: ModelType = ModelType.SPEECH_RECOGNITION
    model_name: str = ""
    version: str = "1.0.0"
    file_hash: str = ""
    file_size_bytes: int = 0
    signed_at: datetime = field(default_factory=datetime.utcnow)
    pqc_signature: str = ""
    signer_id: str = ""
    training_data_hash: str = ""
    is_verified: bool = True

    def __post_init__(self) -> None:
        if not self.file_hash and self.model_name:
            self.file_hash = self._compute_file_hash()
        if not self.pqc_signature and self.file_hash:
            self.pqc_signature = self._compute_signature()

    def _compute_file_hash(self) -> str:
        """Compute SHA3-256 hash for the model artifact."""
        payload = (
            f"{self.artifact_id}|{self.model_name}|"
            f"{self.version}|{self.file_size_bytes}"
        )
        return hashlib.sha3_256(payload.encode()).hexdigest()

    def _compute_signature(self) -> str:
        """Compute simulated ML-DSA-87 signature."""
        payload = (
            f"{self.file_hash}|{self.signer_id}|"
            f"{self.signed_at.isoformat()}|ML-DSA-87"
        )
        return hashlib.sha3_256(payload.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize artifact for storage or transmission."""
        return {
            "artifact_id": self.artifact_id,
            "model_type": self.model_type.name,
            "model_name": self.model_name,
            "version": self.version,
            "file_hash": self.file_hash[:16] + "...",
            "file_size_bytes": self.file_size_bytes,
            "signed_at": self.signed_at.isoformat(),
            "pqc_signature": self.pqc_signature[:16] + "...",
            "signer_id": self.signer_id,
            "training_data_hash": self.training_data_hash[:16] + "..." if self.training_data_hash else "",
            "is_verified": self.is_verified,
        }


@dataclass
class TrainingDataManifest:
    """
    A signed manifest of training data used to produce a model.

    Links a model artifact to the specific datasets used for
    training, enabling supply-chain verification.
    """

    manifest_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_artifact_id: str = ""
    dataset_hashes: Dict[str, str] = field(default_factory=dict)
    total_samples: int = 0
    data_sources: List[str] = field(default_factory=list)
    signed_at: datetime = field(default_factory=datetime.utcnow)
    pqc_signature: str = ""

    def __post_init__(self) -> None:
        if not self.pqc_signature and self.dataset_hashes:
            self.pqc_signature = self._compute_signature()

    def _compute_signature(self) -> str:
        """Compute simulated ML-DSA-87 signature for the manifest."""
        sorted_hashes = sorted(self.dataset_hashes.items())
        payload = (
            f"{self.manifest_id}|{self.model_artifact_id}|"
            f"{str(sorted_hashes)}|{self.signed_at.isoformat()}"
        )
        return hashlib.sha3_256(payload.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize manifest for storage or transmission."""
        return {
            "manifest_id": self.manifest_id,
            "model_artifact_id": self.model_artifact_id,
            "dataset_count": len(self.dataset_hashes),
            "total_samples": self.total_samples,
            "data_sources": self.data_sources,
            "signed_at": self.signed_at.isoformat(),
            "pqc_signature": self.pqc_signature[:16] + "...",
        }


@dataclass
class InferenceAttestation:
    """
    A PQC-signed attestation for a model inference result.

    Provides a cryptographic proof that a specific model version
    produced a specific output for a given input. Used for
    audit and non-repudiation of AI decisions.
    """

    attestation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_artifact_id: str = ""
    input_hash: str = ""
    output_hash: str = ""
    model_version: str = ""
    inference_timestamp: datetime = field(default_factory=datetime.utcnow)
    latency_ms: float = 0.0
    pqc_signature: str = ""

    def __post_init__(self) -> None:
        if not self.pqc_signature and self.input_hash:
            self.pqc_signature = self._compute_signature()

    def _compute_signature(self) -> str:
        """Compute simulated ML-DSA-87 signature for the attestation."""
        payload = (
            f"{self.attestation_id}|{self.model_artifact_id}|"
            f"{self.input_hash}|{self.output_hash}|"
            f"{self.inference_timestamp.isoformat()}"
        )
        return hashlib.sha3_256(payload.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize attestation for storage or transmission."""
        return {
            "attestation_id": self.attestation_id,
            "model_artifact_id": self.model_artifact_id,
            "input_hash": self.input_hash[:16] + "...",
            "output_hash": self.output_hash[:16] + "...",
            "model_version": self.model_version,
            "inference_timestamp": self.inference_timestamp.isoformat(),
            "latency_ms": self.latency_ms,
            "pqc_signature": self.pqc_signature[:16] + "...",
        }


@dataclass
class ModelIntegrityEvent:
    """
    A recorded model integrity event.

    Contains the event classification, affected model artifact,
    evidence, and the action taken. Each event is PQC-hashed
    for tamper-proof audit.
    """

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: IntegrityEventType = IntegrityEventType.MODEL_LOADED
    model_artifact: Optional[ModelArtifact] = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    action_taken: IntegrityAction = IntegrityAction.ALLOW
    detected_at: datetime = field(default_factory=datetime.utcnow)
    pqc_audit_hash: str = ""

    def __post_init__(self) -> None:
        if not self.pqc_audit_hash:
            self.pqc_audit_hash = self._compute_audit_hash()

    def _compute_audit_hash(self) -> str:
        """Compute SHA3-256 audit hash for this event."""
        artifact_id = self.model_artifact.artifact_id if self.model_artifact else "none"
        payload = (
            f"{self.event_id}|{self.event_type.name}|"
            f"{artifact_id}|{self.detected_at.isoformat()}"
        )
        return hashlib.sha3_256(payload.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize event for storage or transmission."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.name,
            "model_artifact": self.model_artifact.to_dict() if self.model_artifact else None,
            "evidence": self.evidence,
            "action_taken": self.action_taken.name,
            "detected_at": self.detected_at.isoformat(),
            "pqc_audit_hash": self.pqc_audit_hash,
        }


@dataclass
class ModelIntegrityPolicy:
    """
    Configuration policy for model integrity protection.

    Defines verification behavior, drift detection thresholds,
    and signing requirements.
    """

    verify_on_load: bool = True
    verify_periodically: bool = True
    verify_interval_seconds: int = 3600
    sign_inference_results: bool = False
    detect_drift: bool = True
    drift_threshold: float = 0.1
    sig_algorithm: str = "ML-DSA-87"
    block_unverified_models: bool = True
    require_training_manifest: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize policy for storage or transmission."""
        return {
            "verify_on_load": self.verify_on_load,
            "verify_periodically": self.verify_periodically,
            "verify_interval_seconds": self.verify_interval_seconds,
            "sign_inference_results": self.sign_inference_results,
            "detect_drift": self.detect_drift,
            "drift_threshold": self.drift_threshold,
            "sig_algorithm": self.sig_algorithm,
            "block_unverified_models": self.block_unverified_models,
            "require_training_manifest": self.require_training_manifest,
        }


# ---------------------------------------------------------------------------
# Model Signer
# ---------------------------------------------------------------------------


class ModelSigner:
    """
    Signs and verifies model artifacts and training manifests.

    Uses SHA3-256 hashing and simulated ML-DSA-87 signatures
    for quantum-safe integrity protection.
    """

    def __init__(self, signer_id: str = "", sig_algorithm: str = "ML-DSA-87"):
        self._signer_id = signer_id or f"signer-{uuid.uuid4().hex[:8]}"
        self._sig_algorithm = sig_algorithm

    def sign_artifact(
        self,
        model_name: str,
        model_type: ModelType,
        version: str = "1.0.0",
        file_size_bytes: int = 0,
        training_data_hash: str = "",
    ) -> ModelArtifact:
        """
        Sign a model artifact, computing SHA3-256 hash and PQC signature.

        Args:
            model_name: Name of the model.
            model_type: Type of the model.
            version: Model version string.
            file_size_bytes: Size of the model file.
            training_data_hash: Hash of the training data used.

        Returns:
            A signed ModelArtifact.
        """
        artifact = ModelArtifact(
            model_type=model_type,
            model_name=model_name,
            version=version,
            file_size_bytes=file_size_bytes,
            signer_id=self._signer_id,
            training_data_hash=training_data_hash,
            is_verified=True,
        )

        logger.info(
            "Model artifact signed: name=%s version=%s hash=%s",
            model_name, version, artifact.file_hash[:16] + "...",
        )
        return artifact

    def verify_artifact(self, artifact: ModelArtifact) -> bool:
        """
        Verify a model artifact's integrity by checking its hash.

        Recomputes the SHA3-256 hash from artifact metadata and
        compares it to the stored hash.

        Args:
            artifact: The model artifact to verify.

        Returns:
            True if the artifact integrity is valid.
        """
        expected_hash = hashlib.sha3_256(
            f"{artifact.artifact_id}|{artifact.model_name}|"
            f"{artifact.version}|{artifact.file_size_bytes}".encode()
        ).hexdigest()

        is_valid = expected_hash == artifact.file_hash
        if not is_valid:
            logger.critical(
                "Model artifact TAMPERED: name=%s version=%s "
                "expected=%s actual=%s",
                artifact.model_name, artifact.version,
                expected_hash[:16] + "...",
                artifact.file_hash[:16] + "...",
            )

        return is_valid

    def sign_training_manifest(
        self,
        model_artifact_id: str,
        dataset_hashes: Dict[str, str],
        total_samples: int = 0,
        data_sources: Optional[List[str]] = None,
    ) -> TrainingDataManifest:
        """
        Sign a training data manifest.

        Args:
            model_artifact_id: ID of the model artifact.
            dataset_hashes: Mapping of dataset names to SHA3-256 hashes.
            total_samples: Total number of training samples.
            data_sources: List of data source identifiers.

        Returns:
            A signed TrainingDataManifest.
        """
        manifest = TrainingDataManifest(
            model_artifact_id=model_artifact_id,
            dataset_hashes=dataset_hashes,
            total_samples=total_samples,
            data_sources=data_sources or [],
        )

        logger.info(
            "Training manifest signed: model=%s datasets=%d samples=%d",
            model_artifact_id, len(dataset_hashes), total_samples,
        )
        return manifest

    def verify_training_manifest(
        self,
        manifest: TrainingDataManifest,
    ) -> bool:
        """
        Verify a training data manifest's integrity.

        Recomputes the signature and compares to the stored one.

        Args:
            manifest: The manifest to verify.

        Returns:
            True if the manifest integrity is valid.
        """
        sorted_hashes = sorted(manifest.dataset_hashes.items())
        expected_payload = (
            f"{manifest.manifest_id}|{manifest.model_artifact_id}|"
            f"{str(sorted_hashes)}|{manifest.signed_at.isoformat()}"
        )
        expected_sig = hashlib.sha3_256(expected_payload.encode()).hexdigest()

        is_valid = expected_sig == manifest.pqc_signature
        if not is_valid:
            logger.critical(
                "Training manifest TAMPERED: model=%s manifest=%s",
                manifest.model_artifact_id, manifest.manifest_id,
            )

        return is_valid


# ---------------------------------------------------------------------------
# Drift Detector
# ---------------------------------------------------------------------------


class DriftDetector:
    """
    Detects model drift and adversarial inputs.

    Monitors running statistics of model inputs and outputs
    to detect distribution shifts and anomalous patterns.
    """

    def __init__(self, drift_threshold: float = 0.1):
        self._drift_threshold = drift_threshold
        self._baseline_stats: Dict[str, Dict[str, float]] = {}
        self._running_stats: Dict[str, Dict[str, float]] = {}
        self._sample_counts: Dict[str, int] = {}

    def set_baseline(
        self,
        model_id: str,
        stats: Dict[str, float],
    ) -> None:
        """
        Set baseline statistics for a model.

        Args:
            model_id: The model to set baseline for.
            stats: Baseline statistics (e.g., mean, std of outputs).
        """
        self._baseline_stats[model_id] = dict(stats)
        self._running_stats[model_id] = dict(stats)
        self._sample_counts[model_id] = 0

        logger.info(
            "Baseline set for model=%s metrics=%d",
            model_id, len(stats),
        )

    def update_running_stats(
        self,
        model_id: str,
        new_stats: Dict[str, float],
    ) -> None:
        """
        Update running statistics with new observations.

        Uses exponential moving average for smooth tracking.

        Args:
            model_id: The model to update.
            new_stats: New observation statistics.
        """
        if model_id not in self._running_stats:
            self._running_stats[model_id] = dict(new_stats)
            self._sample_counts[model_id] = 1
            return

        alpha = 0.1  # EMA smoothing factor
        current = self._running_stats[model_id]
        for key, value in new_stats.items():
            if key in current:
                current[key] = alpha * value + (1 - alpha) * current[key]
            else:
                current[key] = value

        self._sample_counts[model_id] = self._sample_counts.get(model_id, 0) + 1

    def check_distribution_drift(
        self, model_id: str
    ) -> Dict[str, Any]:
        """
        Check for distribution drift by comparing running stats to baseline.

        Args:
            model_id: The model to check.

        Returns:
            Drift analysis results.
        """
        baseline = self._baseline_stats.get(model_id)
        running = self._running_stats.get(model_id)

        if baseline is None or running is None:
            return {
                "model_id": model_id,
                "drift_detected": False,
                "reason": "no_baseline" if baseline is None else "no_running_stats",
            }

        drifted_metrics: Dict[str, Dict[str, float]] = {}
        max_drift = 0.0

        for key in baseline:
            if key in running:
                baseline_val = baseline[key]
                running_val = running[key]

                if abs(baseline_val) < 1e-10:
                    drift = abs(running_val - baseline_val)
                else:
                    drift = abs(running_val - baseline_val) / abs(baseline_val)

                if drift > self._drift_threshold:
                    drifted_metrics[key] = {
                        "baseline": baseline_val,
                        "current": running_val,
                        "drift": drift,
                    }

                max_drift = max(max_drift, drift)

        drift_detected = len(drifted_metrics) > 0

        return {
            "model_id": model_id,
            "drift_detected": drift_detected,
            "max_drift": max_drift,
            "threshold": self._drift_threshold,
            "drifted_metrics": drifted_metrics,
            "samples_observed": self._sample_counts.get(model_id, 0),
        }

    def detect_adversarial_input(
        self,
        model_id: str,
        input_features: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Check if an input appears adversarial based on feature anomalies.

        Compares input features against baseline distributions
        to detect out-of-distribution patterns.

        Args:
            model_id: The model receiving the input.
            input_features: Features of the input to check.

        Returns:
            Adversarial detection results.
        """
        baseline = self._baseline_stats.get(model_id, {})

        anomalous_features: Dict[str, Dict[str, float]] = {}
        max_anomaly = 0.0

        for key, value in input_features.items():
            baseline_key = f"{key}_mean"
            std_key = f"{key}_std"

            if baseline_key in baseline and std_key in baseline:
                mean = baseline[baseline_key]
                std = baseline[std_key]

                if std > 1e-10:
                    z_score = abs(value - mean) / std
                else:
                    z_score = abs(value - mean)

                if z_score > 3.0:  # Beyond 3 sigma
                    anomalous_features[key] = {
                        "value": value,
                        "mean": mean,
                        "std": std,
                        "z_score": z_score,
                    }
                    max_anomaly = max(max_anomaly, z_score)

        is_adversarial = len(anomalous_features) > 0

        return {
            "model_id": model_id,
            "is_adversarial": is_adversarial,
            "max_anomaly_z_score": max_anomaly,
            "anomalous_features": anomalous_features,
            "features_checked": len(input_features),
        }


# ---------------------------------------------------------------------------
# Model Integrity Engine
# ---------------------------------------------------------------------------


class ModelIntegrityEngine:
    """
    Comprehensive AI model integrity protection for BPO operations.

    Combines artifact signing, periodic verification, drift detection,
    and adversarial input detection with PQC audit logging.

    Usage::

        engine = ModelIntegrityEngine.create_strict_policy()

        # Register and sign a model
        artifact = engine.register_model(
            model_name="sentiment-v2",
            model_type=ModelType.SENTIMENT_ANALYSIS,
            version="2.1.0",
            file_size_bytes=150_000_000,
        )

        # Verify before inference
        is_valid = engine.verify_model(artifact.artifact_id)
        if not is_valid:
            # Model tampered, do not use
            ...

        # Attest inference results
        attestation = engine.attest_inference(
            model_artifact_id=artifact.artifact_id,
            input_data="customer message text",
            output_data={"sentiment": "positive", "score": 0.92},
        )
    """

    def __init__(self, policy: Optional[ModelIntegrityPolicy] = None):
        self._policy = policy or ModelIntegrityPolicy()
        self._signer = ModelSigner(sig_algorithm=self._policy.sig_algorithm)
        self._drift_detector = DriftDetector(
            drift_threshold=self._policy.drift_threshold
        )
        self._artifacts: Dict[str, ModelArtifact] = {}
        self._manifests: Dict[str, TrainingDataManifest] = {}
        self._event_log: List[ModelIntegrityEvent] = []
        self._last_verified: Dict[str, datetime] = {}
        self._stats = {
            "total_models_registered": 0,
            "total_verifications": 0,
            "verification_failures": 0,
            "total_attestations": 0,
            "drift_detections": 0,
            "adversarial_detections": 0,
            "models_quarantined": 0,
        }

        logger.info(
            "ModelIntegrityEngine initialized "
            "verify_on_load=%s periodic=%s interval=%ds "
            "drift=%s algorithm=%s",
            self._policy.verify_on_load,
            self._policy.verify_periodically,
            self._policy.verify_interval_seconds,
            self._policy.detect_drift,
            self._policy.sig_algorithm,
        )

    # ------------------------------------------------------------------
    # Model registration
    # ------------------------------------------------------------------

    def register_model(
        self,
        model_name: str,
        model_type: ModelType,
        version: str = "1.0.0",
        file_size_bytes: int = 0,
        training_data_hash: str = "",
        dataset_hashes: Optional[Dict[str, str]] = None,
        total_samples: int = 0,
        data_sources: Optional[List[str]] = None,
    ) -> ModelArtifact:
        """
        Register and sign a new model artifact.

        Optionally creates a training data manifest if dataset
        hashes are provided.

        Args:
            model_name: Name of the model.
            model_type: Type of the model.
            version: Model version string.
            file_size_bytes: Size of the model file.
            training_data_hash: Overall hash of training data.
            dataset_hashes: Per-dataset hashes for manifest.
            total_samples: Total training samples.
            data_sources: Data source identifiers.

        Returns:
            The signed ModelArtifact.
        """
        artifact = self._signer.sign_artifact(
            model_name=model_name,
            model_type=model_type,
            version=version,
            file_size_bytes=file_size_bytes,
            training_data_hash=training_data_hash,
        )

        self._artifacts[artifact.artifact_id] = artifact
        self._last_verified[artifact.artifact_id] = datetime.utcnow()
        self._stats["total_models_registered"] += 1

        # Create training manifest if dataset hashes provided
        if dataset_hashes and self._policy.require_training_manifest:
            manifest = self._signer.sign_training_manifest(
                model_artifact_id=artifact.artifact_id,
                dataset_hashes=dataset_hashes,
                total_samples=total_samples,
                data_sources=data_sources,
            )
            self._manifests[artifact.artifact_id] = manifest

        self._log_event(
            IntegrityEventType.MODEL_LOADED,
            artifact,
            IntegrityAction.ALLOW,
            {"version": version, "size_bytes": file_size_bytes},
        )

        return artifact

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_model(self, artifact_id: str) -> bool:
        """
        Verify a model artifact's integrity.

        Checks the artifact hash and optionally the training
        data manifest.

        Args:
            artifact_id: ID of the artifact to verify.

        Returns:
            True if the model passes integrity checks.
        """
        self._stats["total_verifications"] += 1
        artifact = self._artifacts.get(artifact_id)

        if artifact is None:
            logger.warning("Verification failed: artifact not found id=%s", artifact_id)
            self._stats["verification_failures"] += 1
            return False

        # Verify artifact hash
        artifact_valid = self._signer.verify_artifact(artifact)

        # Verify training manifest if required
        manifest_valid = True
        if self._policy.require_training_manifest:
            manifest = self._manifests.get(artifact_id)
            if manifest is not None:
                manifest_valid = self._signer.verify_training_manifest(manifest)
            elif self._policy.block_unverified_models:
                manifest_valid = False
                logger.warning(
                    "No training manifest for model=%s (required by policy)",
                    artifact.model_name,
                )

        is_valid = artifact_valid and manifest_valid
        self._last_verified[artifact_id] = datetime.utcnow()

        if is_valid:
            artifact.is_verified = True
            self._log_event(
                IntegrityEventType.MODEL_VERIFIED,
                artifact,
                IntegrityAction.ALLOW,
                {"artifact_valid": artifact_valid, "manifest_valid": manifest_valid},
            )
        else:
            artifact.is_verified = False
            self._stats["verification_failures"] += 1
            action = IntegrityAction.QUARANTINE_MODEL
            if self._policy.block_unverified_models:
                action = IntegrityAction.BLOCK_INFERENCE
                self._stats["models_quarantined"] += 1
            self._log_event(
                IntegrityEventType.MODEL_TAMPERED,
                artifact,
                action,
                {"artifact_valid": artifact_valid, "manifest_valid": manifest_valid},
            )

        return is_valid

    # ------------------------------------------------------------------
    # Inference attestation
    # ------------------------------------------------------------------

    def attest_inference(
        self,
        model_artifact_id: str,
        input_data: Any,
        output_data: Any,
        latency_ms: float = 0.0,
    ) -> Optional[InferenceAttestation]:
        """
        Create a PQC-signed attestation for an inference result.

        Args:
            model_artifact_id: ID of the model used.
            input_data: The input provided to the model.
            output_data: The output produced by the model.
            latency_ms: Inference latency in milliseconds.

        Returns:
            An InferenceAttestation, or None if signing is disabled.
        """
        if not self._policy.sign_inference_results:
            return None

        artifact = self._artifacts.get(model_artifact_id)
        if artifact is None:
            return None

        input_hash = hashlib.sha3_256(str(input_data).encode()).hexdigest()
        output_hash = hashlib.sha3_256(str(output_data).encode()).hexdigest()

        attestation = InferenceAttestation(
            model_artifact_id=model_artifact_id,
            input_hash=input_hash,
            output_hash=output_hash,
            model_version=artifact.version,
            latency_ms=latency_ms,
        )

        self._stats["total_attestations"] += 1
        return attestation

    # ------------------------------------------------------------------
    # Integrity monitoring
    # ------------------------------------------------------------------

    def check_integrity(self) -> Dict[str, Any]:
        """
        Run a full integrity check on all registered models.

        Verifies each artifact and checks if periodic verification
        is overdue.

        Returns:
            Summary of integrity check results.
        """
        results: Dict[str, Any] = {
            "checked_at": datetime.utcnow().isoformat(),
            "total_models": len(self._artifacts),
            "verified": 0,
            "failed": 0,
            "overdue": 0,
            "details": {},
        }

        now = datetime.utcnow()
        interval = timedelta(seconds=self._policy.verify_interval_seconds)

        for artifact_id, artifact in self._artifacts.items():
            is_valid = self.verify_model(artifact_id)

            last = self._last_verified.get(artifact_id)
            is_overdue = last is not None and (now - last) > interval

            if is_valid:
                results["verified"] += 1
            else:
                results["failed"] += 1

            if is_overdue:
                results["overdue"] += 1

            results["details"][artifact_id] = {
                "model_name": artifact.model_name,
                "version": artifact.version,
                "is_valid": is_valid,
                "is_overdue": is_overdue,
            }

        return results

    def detect_drift(
        self,
        model_artifact_id: str,
        current_stats: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Check for model drift on a specific model.

        Args:
            model_artifact_id: The model to check.
            current_stats: If provided, update running stats first.

        Returns:
            Drift analysis results.
        """
        if current_stats is not None:
            self._drift_detector.update_running_stats(
                model_artifact_id, current_stats
            )

        result = self._drift_detector.check_distribution_drift(model_artifact_id)

        if result.get("drift_detected", False):
            self._stats["drift_detections"] += 1
            artifact = self._artifacts.get(model_artifact_id)
            if artifact is not None:
                self._log_event(
                    IntegrityEventType.DRIFT_DETECTED,
                    artifact,
                    IntegrityAction.ALERT,
                    {"drift_result": result},
                )

        return result

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_integrity_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive model integrity report.

        Returns:
            Report dictionary with statistics and model details.
        """
        report_id = str(uuid.uuid4())
        report_hash = hashlib.sha3_256(
            f"{report_id}|{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()

        models_by_type: Dict[str, int] = {}
        for artifact in self._artifacts.values():
            key = artifact.model_type.name
            models_by_type[key] = models_by_type.get(key, 0) + 1

        events_by_type: Dict[str, int] = {}
        for event in self._event_log:
            key = event.event_type.name
            events_by_type[key] = events_by_type.get(key, 0) + 1

        verified_count = sum(
            1 for a in self._artifacts.values() if a.is_verified
        )

        return {
            "report_id": report_id,
            "generated_at": datetime.utcnow().isoformat(),
            "policy": self._policy.to_dict(),
            "statistics": dict(self._stats),
            "total_models": len(self._artifacts),
            "verified_models": verified_count,
            "models_by_type": models_by_type,
            "events_by_type": events_by_type,
            "total_events": len(self._event_log),
            "pqc_report_hash": report_hash,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_event(
        self,
        event_type: IntegrityEventType,
        artifact: ModelArtifact,
        action: IntegrityAction,
        evidence: Dict[str, Any],
    ) -> None:
        """Log a model integrity event."""
        event = ModelIntegrityEvent(
            event_type=event_type,
            model_artifact=artifact,
            evidence=evidence,
            action_taken=action,
        )
        self._event_log.append(event)

    # ------------------------------------------------------------------
    # Factory class methods
    # ------------------------------------------------------------------

    @classmethod
    def create_standard_policy(cls) -> "ModelIntegrityEngine":
        """
        Create an engine with standard model integrity protection.

        Verify on load, hourly periodic checks, drift detection
        enabled with 10% threshold.
        """
        policy = ModelIntegrityPolicy(
            verify_on_load=True,
            verify_periodically=True,
            verify_interval_seconds=3600,
            sign_inference_results=False,
            detect_drift=True,
            drift_threshold=0.1,
            sig_algorithm="ML-DSA-87",
            block_unverified_models=True,
            require_training_manifest=True,
        )
        return cls(policy=policy)

    @classmethod
    def create_strict_policy(cls) -> "ModelIntegrityEngine":
        """
        Create an engine with strict model integrity protection.

        Sign all inference results, 15-minute verification interval,
        tight 5% drift threshold. For high-assurance deployments.
        """
        policy = ModelIntegrityPolicy(
            verify_on_load=True,
            verify_periodically=True,
            verify_interval_seconds=900,
            sign_inference_results=True,
            detect_drift=True,
            drift_threshold=0.05,
            sig_algorithm="ML-DSA-87",
            block_unverified_models=True,
            require_training_manifest=True,
        )
        return cls(policy=policy)
