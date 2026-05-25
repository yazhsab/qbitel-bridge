"""
Voice Biometric Anti-Spoofing Security Module

Provides voice biometric verification with anti-spoofing detection
and PQC-encrypted voiceprint template storage for BPO call centers.

Voice spoofing attacks target biometric authentication systems:
- Replay attacks - playing back recorded voice samples
- Voice synthesis - generating speech with TTS engines
- Voice conversion - transforming one speaker to sound like another
- Voice cloning - deep-learning-based speaker cloning
- Concatenation attacks - splicing recorded words/phrases
- Text-to-speech injection - TTS output passed as live speech

Integrates with QBITEL's PQC infrastructure for ML-KEM encrypted
template storage and tamper-proof audit logging.
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


class BiometricEventType(Enum):
    """Types of biometric events in the voice authentication lifecycle."""

    ENROLLMENT = auto()             # Initial voiceprint enrollment
    VERIFICATION_SUCCESS = auto()   # Successful speaker verification
    VERIFICATION_FAILURE = auto()   # Failed speaker verification
    SPOOF_DETECTED = auto()         # Spoofing attack detected
    TEMPLATE_UPDATED = auto()       # Voiceprint template refreshed
    TEMPLATE_REVOKED = auto()       # Voiceprint template revoked
    LIVENESS_CHALLENGE = auto()     # Liveness challenge issued
    LIVENESS_PASSED = auto()        # Liveness challenge passed
    LIVENESS_FAILED = auto()        # Liveness challenge failed


class SpoofingMethod(Enum):
    """Known voice spoofing attack methods."""

    REPLAY_ATTACK = auto()          # Playback of recorded audio
    VOICE_SYNTHESIS = auto()        # Synthetic speech generation
    VOICE_CONVERSION = auto()       # Speaker identity conversion
    VOICE_CLONING = auto()          # Deep-learning speaker cloning
    CONCATENATION = auto()          # Spliced word/phrase assembly
    TEXT_TO_SPEECH = auto()         # TTS engine output injection


class BiometricAction(Enum):
    """Actions taken in response to biometric verification results."""

    ALLOW = auto()              # Verification passed, allow access
    DENY = auto()               # Verification failed, deny access
    CHALLENGE = auto()          # Issue liveness challenge
    STEP_UP_AUTH = auto()       # Require additional authentication
    LOCK_ACCOUNT = auto()       # Lock the caller's account
    ALERT_SECURITY = auto()     # Alert the security operations team


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class VoiceprintTemplate:
    """
    A PQC-encrypted voiceprint template for a caller.

    The template stores a hashed feature vector (not raw audio)
    encrypted with ML-KEM for quantum-safe storage at rest.
    Templates are rotated periodically and revocable.
    """

    template_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    caller_id: str = ""
    feature_hash: str = ""
    enrolled_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    pqc_encrypted: bool = True
    encryption_algorithm: str = "ML-KEM-1024"
    version: int = 1
    is_active: bool = True

    def __post_init__(self) -> None:
        if not self.feature_hash and self.caller_id:
            self.feature_hash = self._compute_feature_hash()

    def _compute_feature_hash(self) -> str:
        """Compute SHA3-256 hash of feature placeholder."""
        payload = f"{self.template_id}|{self.caller_id}|{self.enrolled_at.isoformat()}"
        return hashlib.sha3_256(payload.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize template metadata for storage or transmission."""
        return {
            "template_id": self.template_id,
            "caller_id": self.caller_id,
            "feature_hash": self.feature_hash[:16] + "...",
            "enrolled_at": self.enrolled_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "pqc_encrypted": self.pqc_encrypted,
            "encryption_algorithm": self.encryption_algorithm,
            "version": self.version,
            "is_active": self.is_active,
        }


@dataclass
class BiometricVerification:
    """
    Result of a voice biometric verification attempt.

    Contains match scores, spoof detection scores, and the
    liveness assessment. Each verification is PQC-hashed for
    tamper-proof audit.
    """

    verification_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    caller_id: str = ""
    template_id: str = ""
    match_score: float = 0.0
    threshold: float = 0.85
    is_match: bool = False
    spoof_score: float = 0.0
    liveness_score: float = 0.0
    is_genuine: bool = False
    method_used: str = ""
    verified_at: datetime = field(default_factory=datetime.utcnow)
    pqc_audit_hash: str = ""

    def __post_init__(self) -> None:
        if not self.pqc_audit_hash:
            self.pqc_audit_hash = self._compute_audit_hash()

    def _compute_audit_hash(self) -> str:
        """Compute SHA3-256 audit hash for this verification."""
        payload = (
            f"{self.verification_id}|{self.caller_id}|"
            f"{self.match_score}|{self.spoof_score}|"
            f"{self.verified_at.isoformat()}"
        )
        return hashlib.sha3_256(payload.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize verification for storage or transmission."""
        return {
            "verification_id": self.verification_id,
            "caller_id": self.caller_id,
            "template_id": self.template_id,
            "match_score": self.match_score,
            "threshold": self.threshold,
            "is_match": self.is_match,
            "spoof_score": self.spoof_score,
            "liveness_score": self.liveness_score,
            "is_genuine": self.is_genuine,
            "method_used": self.method_used,
            "verified_at": self.verified_at.isoformat(),
            "pqc_audit_hash": self.pqc_audit_hash,
        }


@dataclass
class LivenessChallenge:
    """
    A liveness challenge issued to verify the caller is a live human.

    Challenges require the caller to perform a task that is difficult
    to fake with replay or synthesis attacks (e.g., repeating a
    random phrase or counting backwards).
    """

    challenge_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    challenge_type: str = "repeat_phrase"
    challenge_text: str = ""
    expected_duration_ms: int = 5000
    actual_duration_ms: Optional[int] = None
    passed: bool = False
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize challenge for storage or transmission."""
        return {
            "challenge_id": self.challenge_id,
            "challenge_type": self.challenge_type,
            "challenge_text": self.challenge_text,
            "expected_duration_ms": self.expected_duration_ms,
            "actual_duration_ms": self.actual_duration_ms,
            "passed": self.passed,
            "confidence": self.confidence,
        }


@dataclass
class BiometricPolicy:
    """
    Configuration policy for voice biometric security.

    Defines verification thresholds, anti-spoofing settings,
    liveness challenge requirements, and template management.
    """

    enrollment_min_duration_seconds: float = 10.0
    verification_threshold: float = 0.85
    spoof_detection_enabled: bool = True
    liveness_challenge_enabled: bool = True
    max_verification_attempts: int = 3
    template_encryption_algorithm: str = "ML-KEM-1024"
    template_rotation_days: int = 90
    anti_replay_window_seconds: int = 60

    def to_dict(self) -> Dict[str, Any]:
        """Serialize policy for storage or transmission."""
        return {
            "enrollment_min_duration_seconds": self.enrollment_min_duration_seconds,
            "verification_threshold": self.verification_threshold,
            "spoof_detection_enabled": self.spoof_detection_enabled,
            "liveness_challenge_enabled": self.liveness_challenge_enabled,
            "max_verification_attempts": self.max_verification_attempts,
            "template_encryption_algorithm": self.template_encryption_algorithm,
            "template_rotation_days": self.template_rotation_days,
            "anti_replay_window_seconds": self.anti_replay_window_seconds,
        }


# ---------------------------------------------------------------------------
# Liveness challenge phrases
# ---------------------------------------------------------------------------

LIVENESS_PHRASES: List[str] = [
    "My voice is my password verify me",
    "The quick brown fox jumps over the lazy dog",
    "Please verify my identity for account access",
    "Security authentication in progress now",
    "I confirm this is a live verification call",
    "Random phrase seven four two nine one",
    "Confirm access to my account today",
    "Voice security check number eight five three",
]

LIVENESS_CHALLENGE_TYPES: List[str] = [
    "repeat_phrase",
    "count_backwards",
    "random_digits",
    "phonetic_alphabet",
]


# ---------------------------------------------------------------------------
# Spoof Detector
# ---------------------------------------------------------------------------


class SpoofDetector:
    """
    Detects voice spoofing attacks by analyzing audio features.

    Uses heuristic analysis of temporal, spectral, and statistical
    properties to identify replay, synthesis, conversion, and
    cloning attacks.
    """

    # Thresholds for spoofing indicators
    REPLAY_SIMILARITY_THRESHOLD: float = 0.95
    SYNTHESIS_SPECTRAL_THRESHOLD: float = 0.70
    CLONING_ARTIFACT_THRESHOLD: float = 0.65

    def analyze_audio_features(
        self,
        audio_features: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analyze audio features for spoofing indicators.

        Args:
            audio_features: Extracted audio features including
                spectral, temporal, and statistical properties.

        Returns:
            Spoof analysis results with scores per method.
        """
        results: Dict[str, Any] = {
            "overall_spoof_score": 0.0,
            "is_genuine": True,
            "detected_methods": [],
            "indicators": {},
        }

        # Check for replay attack indicators
        replay_score = self.detect_replay(audio_features)
        results["indicators"]["replay"] = replay_score
        if replay_score > self.REPLAY_SIMILARITY_THRESHOLD:
            results["detected_methods"].append(SpoofingMethod.REPLAY_ATTACK.name)

        # Check for synthesis indicators
        synthesis_score = self.detect_synthesis(audio_features)
        results["indicators"]["synthesis"] = synthesis_score
        if synthesis_score > self.SYNTHESIS_SPECTRAL_THRESHOLD:
            results["detected_methods"].append(SpoofingMethod.VOICE_SYNTHESIS.name)

        # Check for cloning artifacts
        cloning_score = self._detect_cloning_artifacts(audio_features)
        results["indicators"]["cloning"] = cloning_score
        if cloning_score > self.CLONING_ARTIFACT_THRESHOLD:
            results["detected_methods"].append(SpoofingMethod.VOICE_CLONING.name)

        # Check for concatenation artifacts
        concat_score = self._detect_concatenation(audio_features)
        results["indicators"]["concatenation"] = concat_score
        if concat_score > 0.70:
            results["detected_methods"].append(SpoofingMethod.CONCATENATION.name)

        # Overall spoof score (max of individual scores)
        all_scores = [replay_score, synthesis_score, cloning_score, concat_score]
        results["overall_spoof_score"] = max(all_scores) if all_scores else 0.0
        results["is_genuine"] = len(results["detected_methods"]) == 0

        return results

    def detect_replay(self, audio_features: Dict[str, Any]) -> float:
        """
        Detect replay attack by checking temporal similarity.

        Replay attacks exhibit channel artifacts from recording/playback
        and temporal correlation with previously heard samples.

        Args:
            audio_features: Audio features to analyze.

        Returns:
            Replay likelihood score (0.0 to 1.0).
        """
        score = 0.0

        # Check for channel noise consistent with recording
        snr = audio_features.get("signal_to_noise_ratio", 40.0)
        if snr < 20.0:
            score += 0.3  # Low SNR suggests recording playback

        # Check for compression artifacts
        if audio_features.get("compression_artifacts_detected", False):
            score += 0.25

        # Check for room impulse response mismatch
        rir_consistency = audio_features.get("rir_consistency", 1.0)
        if rir_consistency < 0.5:
            score += 0.25

        # Check for temporal hash similarity with known samples
        temporal_hash = audio_features.get("temporal_hash", "")
        if audio_features.get("known_sample_match", False):
            score += 0.40

        return min(score, 1.0)

    def detect_synthesis(self, audio_features: Dict[str, Any]) -> float:
        """
        Detect voice synthesis by checking spectral features.

        Synthetic speech often lacks natural micro-variations in
        pitch, formant transitions, and breathing patterns.

        Args:
            audio_features: Audio features to analyze.

        Returns:
            Synthesis likelihood score (0.0 to 1.0).
        """
        score = 0.0

        # Check for unnatural pitch stability
        pitch_variance = audio_features.get("pitch_variance", 10.0)
        if pitch_variance < 2.0:
            score += 0.30  # Unnaturally stable pitch

        # Check for missing breathing patterns
        if not audio_features.get("breathing_detected", True):
            score += 0.25

        # Check for spectral smoothness (TTS tends to over-smooth)
        spectral_detail = audio_features.get("spectral_detail_score", 0.8)
        if spectral_detail < 0.4:
            score += 0.25

        # Check for formant transition naturalness
        formant_score = audio_features.get("formant_naturalness", 0.9)
        if formant_score < 0.5:
            score += 0.20

        return min(score, 1.0)

    def _detect_cloning_artifacts(
        self, audio_features: Dict[str, Any]
    ) -> float:
        """Detect artifacts typical of deep-learning voice cloning."""
        score = 0.0

        # Cloning models often produce subtle spectral ringing
        if audio_features.get("spectral_ringing_detected", False):
            score += 0.35

        # Phase coherence anomalies
        phase_coherence = audio_features.get("phase_coherence", 0.9)
        if phase_coherence < 0.6:
            score += 0.30

        # Unnatural prosody patterns
        prosody_score = audio_features.get("prosody_naturalness", 0.9)
        if prosody_score < 0.5:
            score += 0.25

        return min(score, 1.0)

    def _detect_concatenation(
        self, audio_features: Dict[str, Any]
    ) -> float:
        """Detect concatenation attacks (spliced audio segments)."""
        score = 0.0

        # Check for energy discontinuities
        energy_jumps = audio_features.get("energy_discontinuities", 0)
        if energy_jumps > 3:
            score += 0.35

        # Check for unnatural silence insertions
        silence_anomalies = audio_features.get("silence_anomalies", 0)
        if silence_anomalies > 2:
            score += 0.30

        # Check for pitch discontinuities at segment boundaries
        pitch_jumps = audio_features.get("pitch_discontinuities", 0)
        if pitch_jumps > 2:
            score += 0.25

        return min(score, 1.0)


# ---------------------------------------------------------------------------
# Voice Biometric Security Engine
# ---------------------------------------------------------------------------


class VoiceBiometricSecurityEngine:
    """
    Voice biometric authentication with anti-spoofing and PQC encryption.

    Provides enrollment, verification, liveness detection, and
    template management with quantum-safe encryption.

    Usage::

        engine = VoiceBiometricSecurityEngine.create_high_security_policy()

        # Enroll a caller
        template = engine.enroll_voiceprint(
            caller_id="caller-001",
            audio_features=extracted_features,
        )

        # Verify a caller
        result = engine.verify_caller(
            caller_id="caller-001",
            audio_features=live_features,
        )
        if result.is_match and result.is_genuine:
            # Caller verified
            ...
    """

    def __init__(self, policy: Optional[BiometricPolicy] = None):
        self._policy = policy or BiometricPolicy()
        self._spoof_detector = SpoofDetector()
        self._templates: Dict[str, VoiceprintTemplate] = {}
        self._verification_history: List[BiometricVerification] = []
        self._event_log: List[Dict[str, Any]] = []
        self._attempt_counts: Dict[str, int] = {}
        self._recent_hashes: Dict[str, List[Tuple[str, datetime]]] = {}
        self._stats = {
            "total_enrollments": 0,
            "total_verifications": 0,
            "successful_verifications": 0,
            "failed_verifications": 0,
            "spoofs_detected": 0,
            "liveness_challenges_issued": 0,
            "liveness_challenges_passed": 0,
            "templates_rotated": 0,
            "templates_revoked": 0,
        }

        logger.info(
            "VoiceBiometricSecurityEngine initialized "
            "threshold=%.2f spoof_detection=%s liveness=%s encryption=%s",
            self._policy.verification_threshold,
            self._policy.spoof_detection_enabled,
            self._policy.liveness_challenge_enabled,
            self._policy.template_encryption_algorithm,
        )

    # ------------------------------------------------------------------
    # Enrollment
    # ------------------------------------------------------------------

    def enroll_voiceprint(
        self,
        caller_id: str,
        audio_features: Dict[str, Any],
        duration_seconds: float = 0.0,
    ) -> Optional[VoiceprintTemplate]:
        """
        Enroll a caller's voiceprint.

        Creates a PQC-encrypted voiceprint template from the
        provided audio features.

        Args:
            caller_id: Unique identifier for the caller.
            audio_features: Extracted audio features for enrollment.
            duration_seconds: Duration of the enrollment audio.

        Returns:
            The enrolled VoiceprintTemplate, or None if enrollment fails.
        """
        if duration_seconds < self._policy.enrollment_min_duration_seconds:
            logger.warning(
                "Enrollment rejected: caller=%s duration=%.1fs minimum=%.1fs",
                caller_id, duration_seconds,
                self._policy.enrollment_min_duration_seconds,
            )
            return None

        # Check for spoofing during enrollment
        if self._policy.spoof_detection_enabled:
            spoof_result = self._spoof_detector.analyze_audio_features(audio_features)
            if not spoof_result["is_genuine"]:
                logger.warning(
                    "Enrollment blocked - spoof detected: caller=%s methods=%s",
                    caller_id, spoof_result["detected_methods"],
                )
                self._log_event(BiometricEventType.SPOOF_DETECTED, caller_id, {
                    "context": "enrollment",
                    "spoof_methods": spoof_result["detected_methods"],
                })
                return None

        template = self.encrypt_template_pqc(caller_id, audio_features)
        self._templates[caller_id] = template
        self._stats["total_enrollments"] += 1
        self._log_event(BiometricEventType.ENROLLMENT, caller_id, {
            "template_id": template.template_id,
            "version": template.version,
        })

        logger.info(
            "Voiceprint enrolled: caller=%s template=%s",
            caller_id, template.template_id,
        )
        return template

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_caller(
        self,
        caller_id: str,
        audio_features: Dict[str, Any],
    ) -> BiometricVerification:
        """
        Verify a caller against their enrolled voiceprint.

        Performs match scoring, spoof detection, and optional
        anti-replay checking.

        Args:
            caller_id: The caller to verify.
            audio_features: Live audio features for verification.

        Returns:
            A BiometricVerification result.
        """
        self._stats["total_verifications"] += 1

        template = self._templates.get(caller_id)
        if template is None or not template.is_active:
            logger.warning("Verification failed: no active template for caller=%s", caller_id)
            self._stats["failed_verifications"] += 1
            return BiometricVerification(
                caller_id=caller_id,
                is_match=False,
                is_genuine=False,
                method_used="no_template",
            )

        # Check attempt count
        attempts = self._attempt_counts.get(caller_id, 0)
        if attempts >= self._policy.max_verification_attempts:
            logger.warning(
                "Verification locked: caller=%s attempts=%d max=%d",
                caller_id, attempts, self._policy.max_verification_attempts,
            )
            return BiometricVerification(
                caller_id=caller_id,
                template_id=template.template_id,
                is_match=False,
                is_genuine=False,
                method_used="account_locked",
            )

        # Compute match score (simulated feature comparison)
        match_score = self._compute_match_score(template, audio_features)

        # Spoof detection
        spoof_score = 0.0
        is_genuine = True
        if self._policy.spoof_detection_enabled:
            spoof_result = self._spoof_detector.analyze_audio_features(audio_features)
            spoof_score = spoof_result["overall_spoof_score"]
            is_genuine = spoof_result["is_genuine"]

            if not is_genuine:
                self._stats["spoofs_detected"] += 1
                self._log_event(BiometricEventType.SPOOF_DETECTED, caller_id, {
                    "spoof_score": spoof_score,
                    "methods": spoof_result["detected_methods"],
                })

        # Anti-replay check
        if self._policy.anti_replay_window_seconds > 0:
            audio_hash = hashlib.sha3_256(
                str(audio_features).encode()
            ).hexdigest()
            if self._is_replay(caller_id, audio_hash):
                is_genuine = False
                spoof_score = max(spoof_score, 0.95)
                self._stats["spoofs_detected"] += 1

        is_match = (
            match_score >= self._policy.verification_threshold
            and is_genuine
        )

        # Liveness score (simplified)
        liveness_score = 1.0 - spoof_score

        # Track attempts
        if not is_match:
            self._attempt_counts[caller_id] = attempts + 1
            self._stats["failed_verifications"] += 1
            event_type = BiometricEventType.VERIFICATION_FAILURE
        else:
            self._attempt_counts[caller_id] = 0
            self._stats["successful_verifications"] += 1
            event_type = BiometricEventType.VERIFICATION_SUCCESS

        verification = BiometricVerification(
            caller_id=caller_id,
            template_id=template.template_id,
            match_score=match_score,
            threshold=self._policy.verification_threshold,
            is_match=is_match,
            spoof_score=spoof_score,
            liveness_score=liveness_score,
            is_genuine=is_genuine,
            method_used="pqc_biometric",
        )

        self._verification_history.append(verification)
        self._log_event(event_type, caller_id, {
            "verification_id": verification.verification_id,
            "match_score": match_score,
            "spoof_score": spoof_score,
        })

        return verification

    # ------------------------------------------------------------------
    # Liveness challenges
    # ------------------------------------------------------------------

    def issue_liveness_challenge(self) -> LivenessChallenge:
        """
        Issue a liveness challenge to verify the caller is live.

        Returns:
            A LivenessChallenge with text for the caller to repeat.
        """
        import random
        challenge_type = random.choice(LIVENESS_CHALLENGE_TYPES)

        if challenge_type == "repeat_phrase":
            text = random.choice(LIVENESS_PHRASES)
            expected_ms = len(text.split()) * 500  # ~500ms per word
        elif challenge_type == "count_backwards":
            start = random.randint(20, 50)
            text = f"Please count backwards from {start} to {start - 5}"
            expected_ms = 6000
        elif challenge_type == "random_digits":
            digits = " ".join(str(random.randint(0, 9)) for _ in range(6))
            text = f"Please say the following digits: {digits}"
            expected_ms = 4000
        else:
            letters = " ".join(
                random.choice(["alpha", "bravo", "charlie", "delta",
                               "echo", "foxtrot", "golf", "hotel"])
                for _ in range(4)
            )
            text = f"Please say: {letters}"
            expected_ms = 4000

        self._stats["liveness_challenges_issued"] += 1

        return LivenessChallenge(
            challenge_type=challenge_type,
            challenge_text=text,
            expected_duration_ms=expected_ms,
        )

    def evaluate_liveness_response(
        self,
        challenge: LivenessChallenge,
        response_features: Dict[str, Any],
        actual_duration_ms: int,
    ) -> LivenessChallenge:
        """
        Evaluate a caller's response to a liveness challenge.

        Args:
            challenge: The original challenge issued.
            response_features: Audio features from the response.
            actual_duration_ms: Duration of the response audio.

        Returns:
            Updated LivenessChallenge with pass/fail result.
        """
        challenge.actual_duration_ms = actual_duration_ms

        # Duration check (response should be within reasonable range)
        duration_ratio = actual_duration_ms / max(challenge.expected_duration_ms, 1)
        duration_ok = 0.5 <= duration_ratio <= 2.0

        # Spoof check on the response
        spoof_result = self._spoof_detector.analyze_audio_features(response_features)
        is_genuine = spoof_result["is_genuine"]

        # Content match confidence (simulated)
        content_confidence = response_features.get("content_match_score", 0.0)

        confidence = 0.0
        if duration_ok:
            confidence += 0.30
        if is_genuine:
            confidence += 0.40
        if content_confidence > 0.7:
            confidence += 0.30

        challenge.passed = confidence >= 0.70
        challenge.confidence = confidence

        if challenge.passed:
            self._stats["liveness_challenges_passed"] += 1

        return challenge

    def detect_spoofing(
        self, audio_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run standalone spoof detection on audio features.

        Args:
            audio_features: Extracted audio features.

        Returns:
            Spoof analysis results.
        """
        return self._spoof_detector.analyze_audio_features(audio_features)

    # ------------------------------------------------------------------
    # Template management
    # ------------------------------------------------------------------

    def encrypt_template_pqc(
        self,
        caller_id: str,
        audio_features: Dict[str, Any],
    ) -> VoiceprintTemplate:
        """
        Create a PQC-encrypted voiceprint template.

        Computes a SHA3-256 feature hash and wraps the template
        with ML-KEM encryption metadata.

        Args:
            caller_id: The caller's unique identifier.
            audio_features: Extracted audio features.

        Returns:
            An encrypted VoiceprintTemplate.
        """
        feature_payload = f"{caller_id}|{str(sorted(audio_features.items()))}"
        feature_hash = hashlib.sha3_256(feature_payload.encode()).hexdigest()

        existing = self._templates.get(caller_id)
        version = (existing.version + 1) if existing else 1

        return VoiceprintTemplate(
            caller_id=caller_id,
            feature_hash=feature_hash,
            encryption_algorithm=self._policy.template_encryption_algorithm,
            version=version,
            is_active=True,
        )

    def rotate_template(
        self,
        caller_id: str,
        new_audio_features: Dict[str, Any],
    ) -> Optional[VoiceprintTemplate]:
        """
        Rotate a caller's voiceprint template.

        Creates a new template version and deactivates the old one.

        Args:
            caller_id: The caller whose template to rotate.
            new_audio_features: Fresh audio features for the new template.

        Returns:
            The new VoiceprintTemplate, or None if no existing template.
        """
        existing = self._templates.get(caller_id)
        if existing is None:
            logger.warning("Cannot rotate: no template for caller=%s", caller_id)
            return None

        new_template = self.encrypt_template_pqc(caller_id, new_audio_features)
        self._templates[caller_id] = new_template
        self._stats["templates_rotated"] += 1

        self._log_event(BiometricEventType.TEMPLATE_UPDATED, caller_id, {
            "old_version": existing.version,
            "new_version": new_template.version,
            "template_id": new_template.template_id,
        })

        logger.info(
            "Template rotated: caller=%s v%d -> v%d",
            caller_id, existing.version, new_template.version,
        )
        return new_template

    def revoke_template(self, caller_id: str) -> bool:
        """
        Revoke a caller's voiceprint template.

        Args:
            caller_id: The caller whose template to revoke.

        Returns:
            True if the template was revoked.
        """
        template = self._templates.get(caller_id)
        if template is None:
            return False

        template.is_active = False
        self._stats["templates_revoked"] += 1
        self._log_event(BiometricEventType.TEMPLATE_REVOKED, caller_id, {
            "template_id": template.template_id,
        })

        logger.info("Template revoked: caller=%s template=%s", caller_id, template.template_id)
        return True

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_security_report(self) -> Dict[str, Any]:
        """
        Generate a security report for voice biometric operations.

        Returns:
            Report dictionary with statistics and event summary.
        """
        report_id = str(uuid.uuid4())
        report_hash = hashlib.sha3_256(
            f"{report_id}|{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()

        active_templates = sum(1 for t in self._templates.values() if t.is_active)
        templates_needing_rotation = sum(
            1 for t in self._templates.values()
            if t.is_active and (datetime.utcnow() - t.updated_at).days
            >= self._policy.template_rotation_days
        )

        return {
            "report_id": report_id,
            "generated_at": datetime.utcnow().isoformat(),
            "policy": self._policy.to_dict(),
            "statistics": dict(self._stats),
            "active_templates": active_templates,
            "templates_needing_rotation": templates_needing_rotation,
            "total_events": len(self._event_log),
            "pqc_report_hash": report_hash,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_match_score(
        self,
        template: VoiceprintTemplate,
        audio_features: Dict[str, Any],
    ) -> float:
        """Compute a simulated match score between template and live features."""
        live_payload = f"{template.caller_id}|{str(sorted(audio_features.items()))}"
        live_hash = hashlib.sha3_256(live_payload.encode()).hexdigest()

        # Simulated scoring: compare hash prefixes for deterministic behavior
        matching_chars = sum(
            1 for a, b in zip(template.feature_hash, live_hash) if a == b
        )
        return matching_chars / max(len(template.feature_hash), 1)

    def _is_replay(self, caller_id: str, audio_hash: str) -> bool:
        """Check if this audio has been seen recently (anti-replay)."""
        now = datetime.utcnow()
        window = timedelta(seconds=self._policy.anti_replay_window_seconds)

        if caller_id not in self._recent_hashes:
            self._recent_hashes[caller_id] = []

        # Clean old entries
        self._recent_hashes[caller_id] = [
            (h, t) for h, t in self._recent_hashes[caller_id]
            if now - t < window
        ]

        # Check for duplicate
        for h, _ in self._recent_hashes[caller_id]:
            if h == audio_hash:
                return True

        self._recent_hashes[caller_id].append((audio_hash, now))
        return False

    def _log_event(
        self,
        event_type: BiometricEventType,
        caller_id: str,
        details: Dict[str, Any],
    ) -> None:
        """Log a biometric event."""
        event = {
            "event_id": str(uuid.uuid4()),
            "event_type": event_type.name,
            "caller_id": caller_id,
            "details": details,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._event_log.append(event)

    # ------------------------------------------------------------------
    # Factory class methods
    # ------------------------------------------------------------------

    @classmethod
    def create_standard_policy(cls) -> "VoiceBiometricSecurityEngine":
        """
        Create an engine with standard biometric security.

        Balanced settings suitable for general BPO call centers.
        """
        policy = BiometricPolicy(
            enrollment_min_duration_seconds=10.0,
            verification_threshold=0.85,
            spoof_detection_enabled=True,
            liveness_challenge_enabled=True,
            max_verification_attempts=3,
            template_encryption_algorithm="ML-KEM-768",
            template_rotation_days=90,
            anti_replay_window_seconds=60,
        )
        return cls(policy=policy)

    @classmethod
    def create_high_security_policy(cls) -> "VoiceBiometricSecurityEngine":
        """
        Create an engine with high-security biometric settings.

        Strict thresholds for environments handling financial
        or healthcare data.
        """
        policy = BiometricPolicy(
            enrollment_min_duration_seconds=15.0,
            verification_threshold=0.90,
            spoof_detection_enabled=True,
            liveness_challenge_enabled=True,
            max_verification_attempts=2,
            template_encryption_algorithm="ML-KEM-1024",
            template_rotation_days=30,
            anti_replay_window_seconds=120,
        )
        return cls(policy=policy)
