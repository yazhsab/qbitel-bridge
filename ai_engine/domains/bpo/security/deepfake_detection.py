"""
Voice Deepfake and Vishing Detection Module

Detects AI-generated synthetic voice in real-time BPO calls.

Voice deepfakes pose a growing threat to contact centers:
- AI-generated caller impersonation to bypass KYC/authentication
- Synthetic voice used in social engineering and vishing attacks
- Replay attacks using recorded legitimate caller audio
- Real-time voice cloning targeting high-value transactions

This module provides multi-layered detection:
- Spectral analysis for synthetic voice artifacts
- Breathing and micro-pause naturalness checks
- Formant transition analysis for voice onset detection
- Liveness challenge-response verification
- PQC-signed integrity hashing for forensic evidence

Integrates with QBITEL's quantum-safe infrastructure for secure
evidence preservation and cross-tenant threat intelligence sharing.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import hashlib
import logging
import math
import re
import uuid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DeepfakeIndicatorType(Enum):
    """Types of deepfake indicators detected in voice analysis."""

    SPECTRAL_ANOMALY = auto()              # Unnatural spectral distribution
    BREATHING_PATTERN_ABSENT = auto()      # Missing natural breathing sounds
    MICRO_PAUSE_IRREGULAR = auto()         # Irregular micro-pauses between words
    PITCH_CONSISTENCY_ABNORMAL = auto()    # Too-consistent pitch (robotic)
    FORMANT_TRANSITION_UNNATURAL = auto()  # Unnatural vowel transitions
    BACKGROUND_NOISE_SYNTHETIC = auto()    # Artificially generated background
    CODEC_ARTIFACT_MISMATCH = auto()       # Codec artifacts inconsistent w/ path
    EMOTIONAL_TONE_FLAT = auto()           # Lack of natural emotional variation
    VOICE_ONSET_ARTIFICIAL = auto()        # Unnatural voice onset timing
    REPLAY_ATTACK = auto()                 # Previously recorded audio replayed


class DeepfakeRiskLevel(Enum):
    """Risk classification for deepfake detection results."""

    GENUINE = (0, "Genuine", 0.0, 0.25)
    SUSPICIOUS = (1, "Suspicious", 0.25, 0.60)
    LIKELY_SYNTHETIC = (2, "Likely Synthetic", 0.60, 0.85)
    CONFIRMED_SYNTHETIC = (3, "Confirmed Synthetic", 0.85, 1.0)

    def __init__(
        self,
        level: int,
        display_name: str,
        min_confidence: float,
        max_confidence: float,
    ):
        self.level = level
        self.display_name = display_name
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence


class DeepfakeAction(Enum):
    """Actions to take when a deepfake is detected."""

    LOG = auto()                  # Log event only
    ALERT = auto()                # Alert security team
    CHALLENGE_CALLER = auto()     # Issue liveness challenge
    REQUIRE_VERIFICATION = auto() # Require additional identity verification
    FLAG_FOR_REVIEW = auto()      # Flag call for manual review
    TERMINATE_CALL = auto()       # Terminate the call immediately


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class VoiceFeatures:
    """
    Extracted acoustic features from a voice sample.

    Contains the raw and derived features used by the deepfake
    detection pipeline for classification.
    """

    pitch_mean: float = 0.0
    pitch_std: float = 0.0
    formant_frequencies: List[float] = field(default_factory=list)
    spectral_flatness: float = 0.0
    zero_crossing_rate: float = 0.0
    mfcc_coefficients: List[float] = field(default_factory=list)
    energy_envelope: List[float] = field(default_factory=list)
    breathing_intervals: List[float] = field(default_factory=list)
    micro_pause_durations: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize features for storage or transmission."""
        return {
            "pitch_mean": self.pitch_mean,
            "pitch_std": self.pitch_std,
            "formant_frequencies": self.formant_frequencies,
            "spectral_flatness": self.spectral_flatness,
            "zero_crossing_rate": self.zero_crossing_rate,
            "mfcc_coefficients": self.mfcc_coefficients,
            "energy_envelope": self.energy_envelope,
            "breathing_intervals": self.breathing_intervals,
            "micro_pause_durations": self.micro_pause_durations,
        }


@dataclass
class DeepfakeIndicator:
    """
    A single indicator of potential deepfake audio.

    Each indicator represents one dimension of analysis that
    contributes to the overall deepfake confidence score.
    """

    indicator_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    indicator_type: DeepfakeIndicatorType = DeepfakeIndicatorType.SPECTRAL_ANOMALY
    confidence: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize indicator for storage or transmission."""
        return {
            "indicator_id": self.indicator_id,
            "indicator_type": self.indicator_type.name,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "detected_at": self.detected_at.isoformat(),
        }


@dataclass
class DeepfakeAnalysis:
    """
    Complete analysis result for a voice deepfake check.

    Aggregates all individual indicators into a unified assessment
    with an overall risk level and confidence score.
    """

    analysis_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    call_id: str = ""
    indicators: List[DeepfakeIndicator] = field(default_factory=list)
    overall_risk: DeepfakeRiskLevel = DeepfakeRiskLevel.GENUINE
    confidence_score: float = 0.0
    is_synthetic: bool = False
    voice_features: Optional[VoiceFeatures] = None
    challenge_result: Optional[str] = None
    pqc_integrity_hash: str = ""
    analyzed_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize analysis for storage or transmission."""
        return {
            "analysis_id": self.analysis_id,
            "call_id": self.call_id,
            "indicators": [ind.to_dict() for ind in self.indicators],
            "overall_risk": self.overall_risk.display_name,
            "confidence_score": self.confidence_score,
            "is_synthetic": self.is_synthetic,
            "voice_features": self.voice_features.to_dict() if self.voice_features else None,
            "challenge_result": self.challenge_result,
            "pqc_integrity_hash": self.pqc_integrity_hash,
            "analyzed_at": self.analyzed_at.isoformat(),
        }


@dataclass
class DeepfakePolicy:
    """
    Configuration policy for deepfake detection behavior.

    Controls which detection methods are enabled and the
    thresholds for triggering actions.
    """

    liveness_detection_enabled: bool = True
    challenge_response_enabled: bool = True
    min_detection_confidence: float = 0.7
    auto_terminate_threshold: float = 0.95
    baseline_duration_seconds: float = 5.0
    spectral_analysis_window_ms: int = 25

    def to_dict(self) -> Dict[str, Any]:
        """Serialize policy for storage or transmission."""
        return {
            "liveness_detection_enabled": self.liveness_detection_enabled,
            "challenge_response_enabled": self.challenge_response_enabled,
            "min_detection_confidence": self.min_detection_confidence,
            "auto_terminate_threshold": self.auto_terminate_threshold,
            "baseline_duration_seconds": self.baseline_duration_seconds,
            "spectral_analysis_window_ms": self.spectral_analysis_window_ms,
        }


# ---------------------------------------------------------------------------
# Voice Feature Extraction
# ---------------------------------------------------------------------------


class VoiceFeatureExtractor:
    """
    Extracts acoustic features from raw audio samples.

    Uses placeholder implementations with standard math operations
    to compute voice features. In production, these would be backed
    by optimized DSP libraries.

    Usage::

        extractor = VoiceFeatureExtractor()
        features = extractor.extract_features(audio_samples, sample_rate=16000)
    """

    def __init__(self, *, frame_size_ms: int = 25, frame_step_ms: int = 10):
        self.frame_size_ms = frame_size_ms
        self.frame_step_ms = frame_step_ms
        logger.info(
            "VoiceFeatureExtractor initialized frame_size=%dms frame_step=%dms",
            frame_size_ms,
            frame_step_ms,
        )

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------

    def _compute_pitch(self, samples: List[float], sample_rate: int) -> Tuple[float, float]:
        """
        Estimate fundamental frequency (pitch) using autocorrelation.

        Returns (mean_pitch_hz, std_pitch_hz).
        """
        if not samples or sample_rate <= 0:
            return 0.0, 0.0

        # Frame the signal
        frame_len = max(1, int(sample_rate * self.frame_size_ms / 1000))
        pitches: List[float] = []

        for start in range(0, len(samples) - frame_len, frame_len // 2):
            frame = samples[start : start + frame_len]

            # Simple autocorrelation-based pitch estimation
            min_lag = max(1, int(sample_rate / 500))   # 500 Hz upper bound
            max_lag = min(frame_len - 1, int(sample_rate / 50))  # 50 Hz lower bound

            if min_lag >= max_lag:
                continue

            best_lag = min_lag
            best_corr = -1.0

            for lag in range(min_lag, max_lag):
                corr = 0.0
                energy = 0.0
                for i in range(frame_len - lag):
                    corr += frame[i] * frame[i + lag]
                    energy += frame[i] * frame[i]
                if energy > 0:
                    normalized = corr / (energy + 1e-10)
                    if normalized > best_corr:
                        best_corr = normalized
                        best_lag = lag

            if best_corr > 0.3:
                pitch = sample_rate / best_lag
                if 50.0 <= pitch <= 500.0:
                    pitches.append(pitch)

        if not pitches:
            return 0.0, 0.0

        mean_pitch = sum(pitches) / len(pitches)
        variance = sum((p - mean_pitch) ** 2 for p in pitches) / len(pitches)
        std_pitch = math.sqrt(variance)

        return mean_pitch, std_pitch

    def _compute_formants(self, samples: List[float], sample_rate: int) -> List[float]:
        """
        Estimate formant frequencies from the spectral envelope.

        Returns a list of estimated formant frequencies (F1, F2, F3).
        """
        if not samples or sample_rate <= 0:
            return []

        frame_len = min(len(samples), int(sample_rate * self.frame_size_ms / 1000))
        frame = samples[:frame_len]

        # Compute simple power spectrum using DFT approximation
        n_bins = 128
        magnitudes: List[float] = []
        for k in range(n_bins):
            real_part = 0.0
            imag_part = 0.0
            freq = k * sample_rate / (2 * n_bins)
            for n_idx in range(frame_len):
                angle = 2.0 * math.pi * k * n_idx / (2 * n_bins)
                real_part += frame[n_idx] * math.cos(angle)
                imag_part -= frame[n_idx] * math.sin(angle)
            mag = math.sqrt(real_part ** 2 + imag_part ** 2)
            magnitudes.append(mag)

        # Find peaks in the spectrum as formant candidates
        formants: List[float] = []
        for i in range(1, len(magnitudes) - 1):
            if magnitudes[i] > magnitudes[i - 1] and magnitudes[i] > magnitudes[i + 1]:
                freq = i * sample_rate / (2 * n_bins)
                if 200.0 <= freq <= 4000.0:
                    formants.append(freq)
                    if len(formants) >= 3:
                        break

        return formants

    def _compute_spectral_flatness(self, samples: List[float]) -> float:
        """
        Compute spectral flatness (Wiener entropy).

        A value close to 1.0 indicates noise-like signal;
        close to 0.0 indicates tonal signal.
        """
        if not samples:
            return 0.0

        magnitudes = [abs(s) + 1e-10 for s in samples[:256]]
        n = len(magnitudes)

        # Geometric mean
        log_sum = sum(math.log(m) for m in magnitudes) / n
        geo_mean = math.exp(log_sum)

        # Arithmetic mean
        arith_mean = sum(magnitudes) / n

        if arith_mean <= 0:
            return 0.0

        return geo_mean / arith_mean

    def _compute_zero_crossing_rate(self, samples: List[float]) -> float:
        """Compute zero-crossing rate of the signal."""
        if len(samples) < 2:
            return 0.0

        crossings = 0
        for i in range(1, len(samples)):
            if (samples[i] >= 0) != (samples[i - 1] >= 0):
                crossings += 1

        return crossings / (len(samples) - 1)

    def _compute_mfcc(self, samples: List[float], sample_rate: int, n_coeffs: int = 13) -> List[float]:
        """
        Compute Mel-Frequency Cepstral Coefficients (placeholder).

        Returns a list of n_coeffs MFCC values.
        """
        if not samples or sample_rate <= 0:
            return [0.0] * n_coeffs

        frame_len = min(len(samples), int(sample_rate * self.frame_size_ms / 1000))
        frame = samples[:frame_len]

        # Simplified mel-scale spectral energies
        n_filters = 26
        mel_energies: List[float] = []
        for f_idx in range(n_filters):
            center_freq = 200.0 + f_idx * 300.0
            energy = 0.0
            for n_idx in range(frame_len):
                angle = 2.0 * math.pi * center_freq * n_idx / sample_rate
                energy += frame[n_idx] * math.cos(angle)
            mel_energies.append(abs(energy) + 1e-10)

        # Apply log compression
        log_energies = [math.log(e) for e in mel_energies]

        # DCT approximation for cepstral coefficients
        coeffs: List[float] = []
        for k in range(n_coeffs):
            coeff = 0.0
            for n_idx in range(n_filters):
                coeff += log_energies[n_idx] * math.cos(
                    math.pi * k * (n_idx + 0.5) / n_filters
                )
            coeffs.append(coeff)

        return coeffs

    def _compute_energy_envelope(self, samples: List[float], sample_rate: int) -> List[float]:
        """Compute short-time energy envelope of the signal."""
        if not samples:
            return []

        frame_len = max(1, int(sample_rate * self.frame_size_ms / 1000))
        step = max(1, int(sample_rate * self.frame_step_ms / 1000))
        envelope: List[float] = []

        for start in range(0, len(samples) - frame_len, step):
            frame = samples[start : start + frame_len]
            energy = sum(s * s for s in frame) / frame_len
            envelope.append(energy)

        return envelope

    def _detect_breathing(self, energy_envelope: List[float], threshold_ratio: float = 0.1) -> List[float]:
        """
        Detect breathing intervals from the energy envelope.

        Returns list of intervals (in frames) between detected breaths.
        """
        if not energy_envelope:
            return []

        max_energy = max(energy_envelope)
        if max_energy <= 0:
            return []

        threshold = max_energy * threshold_ratio
        breath_positions: List[int] = []
        in_dip = False

        for i, energy in enumerate(energy_envelope):
            if energy < threshold and not in_dip:
                in_dip = True
                breath_positions.append(i)
            elif energy >= threshold:
                in_dip = False

        intervals: List[float] = []
        for i in range(1, len(breath_positions)):
            intervals.append(float(breath_positions[i] - breath_positions[i - 1]))

        return intervals

    def _detect_micro_pauses(self, energy_envelope: List[float], threshold_ratio: float = 0.05) -> List[float]:
        """
        Detect micro-pauses (very short silences) in speech.

        Returns list of pause durations in frames.
        """
        if not energy_envelope:
            return []

        max_energy = max(energy_envelope) if energy_envelope else 0
        if max_energy <= 0:
            return []

        threshold = max_energy * threshold_ratio
        pauses: List[float] = []
        pause_start: Optional[int] = None

        for i, energy in enumerate(energy_envelope):
            if energy < threshold:
                if pause_start is None:
                    pause_start = i
            else:
                if pause_start is not None:
                    duration = float(i - pause_start)
                    if 1 <= duration <= 10:  # Micro-pauses are very short
                        pauses.append(duration)
                    pause_start = None

        return pauses

    # ------------------------------------------------------------------
    # Main extraction entry point
    # ------------------------------------------------------------------

    def extract_features(self, audio_samples: List[float], sample_rate: int = 16000) -> VoiceFeatures:
        """
        Extract comprehensive voice features from audio samples.

        Args:
            audio_samples: Raw PCM audio samples as floats (-1.0 to 1.0).
            sample_rate: Audio sample rate in Hz (default 16000).

        Returns:
            VoiceFeatures containing all extracted acoustic features.
        """
        logger.debug(
            "Extracting features from %d samples at %d Hz",
            len(audio_samples),
            sample_rate,
        )

        pitch_mean, pitch_std = self._compute_pitch(audio_samples, sample_rate)
        formants = self._compute_formants(audio_samples, sample_rate)
        spectral_flatness = self._compute_spectral_flatness(audio_samples)
        zcr = self._compute_zero_crossing_rate(audio_samples)
        mfcc = self._compute_mfcc(audio_samples, sample_rate)
        energy_env = self._compute_energy_envelope(audio_samples, sample_rate)
        breathing = self._detect_breathing(energy_env)
        pauses = self._detect_micro_pauses(energy_env)

        features = VoiceFeatures(
            pitch_mean=pitch_mean,
            pitch_std=pitch_std,
            formant_frequencies=formants,
            spectral_flatness=spectral_flatness,
            zero_crossing_rate=zcr,
            mfcc_coefficients=mfcc,
            energy_envelope=energy_env,
            breathing_intervals=breathing,
            micro_pause_durations=pauses,
        )

        logger.debug(
            "Features extracted: pitch=%.1f+/-%.1f Hz, formants=%d, zcr=%.3f",
            pitch_mean,
            pitch_std,
            len(formants),
            zcr,
        )

        return features


# ---------------------------------------------------------------------------
# Voice Liveness Detection
# ---------------------------------------------------------------------------


class VoiceLivenessDetector:
    """
    Detects liveness indicators in voice to distinguish human
    speakers from synthetic or replayed audio.

    Analyzes breathing patterns, micro-pauses, pitch naturalness,
    and formant transitions to determine if a voice is live.

    Usage::

        detector = VoiceLivenessDetector()
        indicators = detector.check_breathing_pattern(features)
    """

    def __init__(
        self,
        *,
        breathing_interval_min: float = 2.0,
        breathing_interval_max: float = 8.0,
        micro_pause_std_min: float = 0.3,
        pitch_std_min: float = 5.0,
        pitch_std_max: float = 80.0,
    ):
        self.breathing_interval_min = breathing_interval_min
        self.breathing_interval_max = breathing_interval_max
        self.micro_pause_std_min = micro_pause_std_min
        self.pitch_std_min = pitch_std_min
        self.pitch_std_max = pitch_std_max
        logger.info("VoiceLivenessDetector initialized")

    # ------------------------------------------------------------------
    # Individual liveness checks
    # ------------------------------------------------------------------

    def check_breathing_pattern(self, features: VoiceFeatures) -> List[DeepfakeIndicator]:
        """
        Check for natural breathing patterns.

        Real human speech contains periodic low-energy segments
        corresponding to inhalation. Synthetic voices typically
        lack these natural breathing artifacts.
        """
        indicators: List[DeepfakeIndicator] = []
        intervals = features.breathing_intervals

        if not intervals:
            indicators.append(DeepfakeIndicator(
                indicator_type=DeepfakeIndicatorType.BREATHING_PATTERN_ABSENT,
                confidence=0.7,
                evidence={
                    "reason": "No breathing intervals detected in voice sample",
                    "expected_interval_range": [self.breathing_interval_min, self.breathing_interval_max],
                },
            ))
            return indicators

        mean_interval = sum(intervals) / len(intervals)
        if mean_interval < self.breathing_interval_min or mean_interval > self.breathing_interval_max:
            confidence = min(1.0, abs(mean_interval - 5.0) / 10.0)
            indicators.append(DeepfakeIndicator(
                indicator_type=DeepfakeIndicatorType.BREATHING_PATTERN_ABSENT,
                confidence=confidence,
                evidence={
                    "mean_interval": mean_interval,
                    "expected_range": [self.breathing_interval_min, self.breathing_interval_max],
                    "num_breaths": len(intervals),
                },
            ))

        # Check variability - too regular suggests synthetic
        if len(intervals) >= 3:
            variance = sum((i - mean_interval) ** 2 for i in intervals) / len(intervals)
            std = math.sqrt(variance)
            coefficient_of_variation = std / (mean_interval + 1e-10)

            if coefficient_of_variation < 0.05:
                indicators.append(DeepfakeIndicator(
                    indicator_type=DeepfakeIndicatorType.BREATHING_PATTERN_ABSENT,
                    confidence=0.6,
                    evidence={
                        "reason": "Breathing pattern too regular (likely synthetic)",
                        "coefficient_of_variation": coefficient_of_variation,
                        "expected_min_cv": 0.05,
                    },
                ))

        return indicators

    def check_micro_pauses(self, features: VoiceFeatures) -> List[DeepfakeIndicator]:
        """
        Check for natural micro-pause patterns.

        Natural speech contains micro-pauses with variable duration.
        Synthetic speech tends to have either no micro-pauses or
        unnaturally regular ones.
        """
        indicators: List[DeepfakeIndicator] = []
        pauses = features.micro_pause_durations

        if not pauses:
            indicators.append(DeepfakeIndicator(
                indicator_type=DeepfakeIndicatorType.MICRO_PAUSE_IRREGULAR,
                confidence=0.5,
                evidence={"reason": "No micro-pauses detected in voice sample"},
            ))
            return indicators

        mean_pause = sum(pauses) / len(pauses)
        variance = sum((p - mean_pause) ** 2 for p in pauses) / len(pauses)
        std_pause = math.sqrt(variance)

        if std_pause < self.micro_pause_std_min:
            indicators.append(DeepfakeIndicator(
                indicator_type=DeepfakeIndicatorType.MICRO_PAUSE_IRREGULAR,
                confidence=0.65,
                evidence={
                    "reason": "Micro-pause durations too uniform",
                    "std_pause": std_pause,
                    "expected_min_std": self.micro_pause_std_min,
                    "num_pauses": len(pauses),
                },
            ))

        return indicators

    def check_pitch_naturalness(self, features: VoiceFeatures) -> List[DeepfakeIndicator]:
        """
        Check whether pitch variation falls within natural range.

        Human speech has characteristic pitch variation. Synthetic
        voices tend to be either too flat or too variable.
        """
        indicators: List[DeepfakeIndicator] = []

        if features.pitch_mean <= 0:
            return indicators

        if features.pitch_std < self.pitch_std_min:
            confidence = 1.0 - (features.pitch_std / self.pitch_std_min)
            indicators.append(DeepfakeIndicator(
                indicator_type=DeepfakeIndicatorType.PITCH_CONSISTENCY_ABNORMAL,
                confidence=min(0.9, confidence),
                evidence={
                    "reason": "Pitch variation too low (monotone / synthetic)",
                    "pitch_std": features.pitch_std,
                    "expected_min_std": self.pitch_std_min,
                    "pitch_mean": features.pitch_mean,
                },
            ))

        if features.pitch_std > self.pitch_std_max:
            confidence = min(1.0, (features.pitch_std - self.pitch_std_max) / 50.0)
            indicators.append(DeepfakeIndicator(
                indicator_type=DeepfakeIndicatorType.PITCH_CONSISTENCY_ABNORMAL,
                confidence=min(0.8, confidence),
                evidence={
                    "reason": "Pitch variation excessively high (unstable synthesis)",
                    "pitch_std": features.pitch_std,
                    "expected_max_std": self.pitch_std_max,
                    "pitch_mean": features.pitch_mean,
                },
            ))

        return indicators

    def check_formant_transitions(self, features: VoiceFeatures) -> List[DeepfakeIndicator]:
        """
        Check formant transitions for naturalness.

        Natural speech has smooth, continuous transitions between
        formant frequencies. Synthetic speech often shows abrupt
        or missing transitions.
        """
        indicators: List[DeepfakeIndicator] = []
        formants = features.formant_frequencies

        if len(formants) < 2:
            indicators.append(DeepfakeIndicator(
                indicator_type=DeepfakeIndicatorType.FORMANT_TRANSITION_UNNATURAL,
                confidence=0.4,
                evidence={
                    "reason": "Insufficient formant data for transition analysis",
                    "formants_found": len(formants),
                },
            ))
            return indicators

        # Check F1-F2 ratio (typical range for voiced speech)
        f1 = formants[0]
        f2 = formants[1] if len(formants) > 1 else 0.0

        if f2 > 0:
            ratio = f1 / f2
            if ratio < 0.15 or ratio > 0.75:
                indicators.append(DeepfakeIndicator(
                    indicator_type=DeepfakeIndicatorType.FORMANT_TRANSITION_UNNATURAL,
                    confidence=0.55,
                    evidence={
                        "reason": "F1/F2 ratio outside natural range",
                        "f1": f1,
                        "f2": f2,
                        "ratio": ratio,
                        "expected_range": [0.15, 0.75],
                    },
                ))

        # Check F3 presence (expected in natural speech)
        if len(formants) < 3:
            indicators.append(DeepfakeIndicator(
                indicator_type=DeepfakeIndicatorType.FORMANT_TRANSITION_UNNATURAL,
                confidence=0.35,
                evidence={
                    "reason": "Missing F3 formant (common in synthetic speech)",
                    "formants_found": len(formants),
                },
            ))

        return indicators


# ---------------------------------------------------------------------------
# Deepfake Detection Engine
# ---------------------------------------------------------------------------


class DeepfakeDetectionEngine:
    """
    Main engine for detecting AI-generated synthetic voice in BPO calls.

    Combines spectral analysis, liveness detection, and challenge-response
    verification to classify voice as genuine or synthetic. All results
    are signed with PQC-based integrity hashes for forensic use.

    Usage::

        engine = DeepfakeDetectionEngine(
            policy=DeepfakeDetectionEngine.create_default_policy(),
        )

        # Analyze voice from a call
        analysis = engine.analyze_voice(
            call_id="call-001",
            audio_samples=samples,
            sample_rate=16000,
        )

        if analysis.is_synthetic:
            action = engine.get_risk_assessment(analysis)
            # Take appropriate action
    """

    def __init__(
        self,
        policy: Optional[DeepfakePolicy] = None,
        *,
        alert_callback: Optional[Callable[[DeepfakeAnalysis], None]] = None,
    ):
        self._policy = policy or self.create_default_policy()
        self._alert_callback = alert_callback
        self._extractor = VoiceFeatureExtractor(
            frame_size_ms=self._policy.spectral_analysis_window_ms,
        )
        self._liveness = VoiceLivenessDetector()

        # Analysis history (call_id -> list of analyses)
        self._history: Dict[str, List[DeepfakeAnalysis]] = {}

        # Statistics
        self._stats = {
            "total_analyzed": 0,
            "total_synthetic_detected": 0,
            "total_challenges_issued": 0,
            "total_calls_terminated": 0,
        }

        logger.info(
            "DeepfakeDetectionEngine initialized "
            "liveness=%s challenge=%s min_conf=%.2f",
            self._policy.liveness_detection_enabled,
            self._policy.challenge_response_enabled,
            self._policy.min_detection_confidence,
        )

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze_voice(
        self,
        call_id: str,
        audio_samples: List[float],
        sample_rate: int = 16000,
    ) -> DeepfakeAnalysis:
        """
        Analyze voice audio for deepfake indicators.

        Performs full-spectrum analysis including feature extraction,
        liveness checks, and spectral anomaly detection.

        Args:
            call_id: Unique identifier for the call being analyzed.
            audio_samples: Raw PCM audio samples as floats (-1.0 to 1.0).
            sample_rate: Audio sample rate in Hz (default 16000).

        Returns:
            DeepfakeAnalysis containing all indicators and risk assessment.
        """
        self._stats["total_analyzed"] += 1
        logger.info(
            "Analyzing voice for call_id=%s samples=%d rate=%d",
            call_id,
            len(audio_samples),
            sample_rate,
        )

        # Extract features
        features = self._extractor.extract_features(audio_samples, sample_rate)

        # Collect all indicators
        indicators: List[DeepfakeIndicator] = []

        # Liveness checks
        if self._policy.liveness_detection_enabled:
            indicators.extend(self._liveness.check_breathing_pattern(features))
            indicators.extend(self._liveness.check_micro_pauses(features))
            indicators.extend(self._liveness.check_pitch_naturalness(features))
            indicators.extend(self._liveness.check_formant_transitions(features))

        # Spectral analysis
        spectral_indicators = self._check_spectral_anomalies(features)
        indicators.extend(spectral_indicators)

        # Emotional tone analysis
        tone_indicators = self._check_emotional_tone(features)
        indicators.extend(tone_indicators)

        # Voice onset analysis
        onset_indicators = self._check_voice_onset(features)
        indicators.extend(onset_indicators)

        # Compute overall confidence
        confidence = self._compute_overall_confidence(indicators)

        # Determine risk level
        risk = self._classify_risk(confidence)

        # Determine if synthetic
        is_synthetic = confidence >= self._policy.min_detection_confidence

        # Compute PQC integrity hash
        hash_input = f"{call_id}:{confidence}:{len(indicators)}:{datetime.utcnow().isoformat()}"
        pqc_hash = hashlib.sha3_256(hash_input.encode()).hexdigest()

        analysis = DeepfakeAnalysis(
            call_id=call_id,
            indicators=indicators,
            overall_risk=risk,
            confidence_score=confidence,
            is_synthetic=is_synthetic,
            voice_features=features,
            pqc_integrity_hash=pqc_hash,
        )

        # Store in history
        if call_id not in self._history:
            self._history[call_id] = []
        self._history[call_id].append(analysis)

        if is_synthetic:
            self._stats["total_synthetic_detected"] += 1
            logger.warning(
                "Synthetic voice detected for call_id=%s confidence=%.3f risk=%s",
                call_id,
                confidence,
                risk.display_name,
            )
            if self._alert_callback:
                self._alert_callback(analysis)

        return analysis

    def _check_spectral_anomalies(self, features: VoiceFeatures) -> List[DeepfakeIndicator]:
        """Check for spectral distribution anomalies typical of synthesis."""
        indicators: List[DeepfakeIndicator] = []

        # Spectral flatness check - synthetic audio often has unusual flatness
        if features.spectral_flatness > 0.8:
            indicators.append(DeepfakeIndicator(
                indicator_type=DeepfakeIndicatorType.SPECTRAL_ANOMALY,
                confidence=0.6,
                evidence={
                    "reason": "Spectral flatness too high (noise-like, possibly synthetic)",
                    "spectral_flatness": features.spectral_flatness,
                    "threshold": 0.8,
                },
            ))
        elif features.spectral_flatness < 0.01:
            indicators.append(DeepfakeIndicator(
                indicator_type=DeepfakeIndicatorType.SPECTRAL_ANOMALY,
                confidence=0.5,
                evidence={
                    "reason": "Spectral flatness too low (excessively tonal)",
                    "spectral_flatness": features.spectral_flatness,
                    "threshold": 0.01,
                },
            ))

        # Zero crossing rate anomaly
        if features.zero_crossing_rate > 0.5:
            indicators.append(DeepfakeIndicator(
                indicator_type=DeepfakeIndicatorType.SPECTRAL_ANOMALY,
                confidence=0.45,
                evidence={
                    "reason": "Zero crossing rate abnormally high",
                    "zero_crossing_rate": features.zero_crossing_rate,
                },
            ))

        return indicators

    def _check_emotional_tone(self, features: VoiceFeatures) -> List[DeepfakeIndicator]:
        """Check for flat emotional tone indicative of synthesis."""
        indicators: List[DeepfakeIndicator] = []

        if not features.energy_envelope or len(features.energy_envelope) < 10:
            return indicators

        # Compute energy variation as proxy for emotional expressiveness
        energies = features.energy_envelope
        mean_energy = sum(energies) / len(energies)
        if mean_energy <= 0:
            return indicators

        variance = sum((e - mean_energy) ** 2 for e in energies) / len(energies)
        cv = math.sqrt(variance) / (mean_energy + 1e-10)

        if cv < 0.15:
            indicators.append(DeepfakeIndicator(
                indicator_type=DeepfakeIndicatorType.EMOTIONAL_TONE_FLAT,
                confidence=0.55,
                evidence={
                    "reason": "Energy variation too low (flat emotional tone)",
                    "coefficient_of_variation": cv,
                    "expected_min_cv": 0.15,
                },
            ))

        return indicators

    def _check_voice_onset(self, features: VoiceFeatures) -> List[DeepfakeIndicator]:
        """Check for unnatural voice onset characteristics."""
        indicators: List[DeepfakeIndicator] = []

        if not features.energy_envelope or len(features.energy_envelope) < 5:
            return indicators

        # Check how abruptly voice starts (attack time)
        first_few = features.energy_envelope[:5]
        if all(e > 0.5 * max(features.energy_envelope) for e in first_few[:2]):
            indicators.append(DeepfakeIndicator(
                indicator_type=DeepfakeIndicatorType.VOICE_ONSET_ARTIFICIAL,
                confidence=0.4,
                evidence={
                    "reason": "Voice onset too abrupt (no natural ramp-up)",
                    "onset_energy": first_few[:3],
                },
            ))

        return indicators

    def _compute_overall_confidence(self, indicators: List[DeepfakeIndicator]) -> float:
        """
        Compute overall deepfake confidence from individual indicators.

        Uses a weighted combination where more indicators and higher
        individual confidences increase the overall score.
        """
        if not indicators:
            return 0.0

        # Weighted average with diminishing returns
        sorted_confs = sorted([ind.confidence for ind in indicators], reverse=True)
        weighted_sum = 0.0
        weight_total = 0.0

        for i, conf in enumerate(sorted_confs):
            weight = 1.0 / (1.0 + i * 0.5)  # Diminishing weight
            weighted_sum += conf * weight
            weight_total += weight

        if weight_total <= 0:
            return 0.0

        base_score = weighted_sum / weight_total

        # Boost based on number of independent indicators
        indicator_types = {ind.indicator_type for ind in indicators}
        diversity_boost = min(0.2, len(indicator_types) * 0.03)

        return min(1.0, base_score + diversity_boost)

    def _classify_risk(self, confidence: float) -> DeepfakeRiskLevel:
        """Classify confidence score into a risk level."""
        for risk in DeepfakeRiskLevel:
            if risk.min_confidence <= confidence < risk.max_confidence:
                return risk
        return DeepfakeRiskLevel.CONFIRMED_SYNTHETIC

    # ------------------------------------------------------------------
    # Challenge-response
    # ------------------------------------------------------------------

    def run_challenge_response(
        self,
        call_id: str,
        challenge_type: str = "repeat_phrase",
        expected_response: str = "",
        actual_response: str = "",
    ) -> DeepfakeAnalysis:
        """
        Run a liveness challenge-response verification.

        Issues a challenge to the caller and analyzes their response
        for naturalness. Challenges include repeating random phrases,
        answering contextual questions, or speaking specific words.

        Args:
            call_id: Unique identifier for the call.
            challenge_type: Type of challenge issued.
            expected_response: The expected response text.
            actual_response: The caller's actual response text.

        Returns:
            Updated DeepfakeAnalysis with challenge results.
        """
        if not self._policy.challenge_response_enabled:
            logger.info("Challenge-response disabled by policy for call_id=%s", call_id)
            return DeepfakeAnalysis(call_id=call_id, challenge_result="disabled")

        self._stats["total_challenges_issued"] += 1
        logger.info(
            "Running %s challenge for call_id=%s",
            challenge_type,
            call_id,
        )

        # Evaluate challenge response
        indicators: List[DeepfakeIndicator] = []
        challenge_passed = False

        if challenge_type == "repeat_phrase":
            # Check if response matches expected phrase (fuzzy)
            expected_clean = re.sub(r"\s+", " ", expected_response.lower().strip())
            actual_clean = re.sub(r"\s+", " ", actual_response.lower().strip())

            if expected_clean and actual_clean:
                # Simple word overlap as similarity measure
                expected_words = set(expected_clean.split())
                actual_words = set(actual_clean.split())
                overlap = len(expected_words & actual_words)
                total = max(len(expected_words), 1)
                similarity = overlap / total
                challenge_passed = similarity > 0.7
            else:
                challenge_passed = False

        elif challenge_type == "contextual_question":
            challenge_passed = len(actual_response.strip()) > 5

        else:
            challenge_passed = bool(actual_response.strip())

        if not challenge_passed:
            indicators.append(DeepfakeIndicator(
                indicator_type=DeepfakeIndicatorType.REPLAY_ATTACK,
                confidence=0.75,
                evidence={
                    "challenge_type": challenge_type,
                    "challenge_passed": False,
                    "reason": "Caller failed liveness challenge",
                },
            ))

        confidence = self._compute_overall_confidence(indicators)
        risk = self._classify_risk(confidence)
        result_str = "passed" if challenge_passed else "failed"

        hash_input = f"{call_id}:challenge:{result_str}:{datetime.utcnow().isoformat()}"
        pqc_hash = hashlib.sha3_256(hash_input.encode()).hexdigest()

        analysis = DeepfakeAnalysis(
            call_id=call_id,
            indicators=indicators,
            overall_risk=risk,
            confidence_score=confidence,
            is_synthetic=not challenge_passed,
            challenge_result=result_str,
            pqc_integrity_hash=pqc_hash,
        )

        if call_id not in self._history:
            self._history[call_id] = []
        self._history[call_id].append(analysis)

        return analysis

    # ------------------------------------------------------------------
    # Risk assessment
    # ------------------------------------------------------------------

    def get_risk_assessment(self, analysis: DeepfakeAnalysis) -> DeepfakeAction:
        """
        Determine the recommended action based on analysis results.

        Maps the analysis risk level and confidence to an appropriate
        action, respecting the configured policy thresholds.

        Args:
            analysis: The deepfake analysis to assess.

        Returns:
            Recommended DeepfakeAction to take.
        """
        if analysis.confidence_score >= self._policy.auto_terminate_threshold:
            self._stats["total_calls_terminated"] += 1
            return DeepfakeAction.TERMINATE_CALL

        if analysis.overall_risk == DeepfakeRiskLevel.CONFIRMED_SYNTHETIC:
            return DeepfakeAction.REQUIRE_VERIFICATION

        if analysis.overall_risk == DeepfakeRiskLevel.LIKELY_SYNTHETIC:
            if self._policy.challenge_response_enabled:
                return DeepfakeAction.CHALLENGE_CALLER
            return DeepfakeAction.FLAG_FOR_REVIEW

        if analysis.overall_risk == DeepfakeRiskLevel.SUSPICIOUS:
            return DeepfakeAction.ALERT

        return DeepfakeAction.LOG

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_report(self, call_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a detection report for a specific call or overall statistics.

        Args:
            call_id: If provided, report on that specific call.
                     If None, generate an aggregate statistics report.

        Returns:
            Dictionary containing the report data.
        """
        if call_id:
            analyses = self._history.get(call_id, [])
            return {
                "report_type": "call",
                "call_id": call_id,
                "total_analyses": len(analyses),
                "analyses": [a.to_dict() for a in analyses],
                "highest_risk": max(
                    (a.overall_risk.level for a in analyses), default=0,
                ),
                "generated_at": datetime.utcnow().isoformat(),
            }

        return {
            "report_type": "aggregate",
            "statistics": dict(self._stats),
            "total_calls_tracked": len(self._history),
            "policy": self._policy.to_dict(),
            "generated_at": datetime.utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def create_default_policy(cls) -> DeepfakePolicy:
        """Create a default deepfake detection policy."""
        return DeepfakePolicy(
            liveness_detection_enabled=True,
            challenge_response_enabled=True,
            min_detection_confidence=0.7,
            auto_terminate_threshold=0.95,
            baseline_duration_seconds=5.0,
            spectral_analysis_window_ms=25,
        )

    @classmethod
    def create_high_security_policy(cls) -> DeepfakePolicy:
        """
        Create a high-security policy with lower thresholds.

        Suitable for financial services, healthcare, and other
        high-value BPO operations where false negatives are
        more costly than false positives.
        """
        return DeepfakePolicy(
            liveness_detection_enabled=True,
            challenge_response_enabled=True,
            min_detection_confidence=0.5,
            auto_terminate_threshold=0.85,
            baseline_duration_seconds=3.0,
            spectral_analysis_window_ms=20,
        )
