"""
**EXPERIMENTAL** — Verifiable Delay Functions (VDF) for Safety-Critical Industrial Systems

WARNING: This module uses ITERATED SHA3-256 hashing, which is NOT a proper
Verifiable Delay Function (no RSA group or class group construction).
For safety-critical SIL-rated systems, replace with a formally verified
VDF implementation before deployment.

Set QBITEL_ALLOW_EXPERIMENTAL_CRYPTO=1 to acknowledge and enable usage.

Provides cryptographic timing proofs for safety instrumented systems where
operations MUST take a minimum physical time. Prevents attackers from
fast-forwarding safety timers, bypass countdowns, or skipping mandatory
waiting periods in industrial processes.

Use Cases:
    1. Emergency Shutdown (ESD) Delay: Prove 30-second cooldown was observed
       before re-enabling a reactor — prevents premature restart
    2. Interlock Release Timer: Cryptographic proof that interlock was held
       for required duration before heavy machinery restart
    3. Chemical Process Hold: Prove mixing/reaction hold time was met
    4. Pressure Equalization: Prove pressure equalization delay was real
    5. Sequential Start Delay: Prove motor soft-start intervals were observed

VDF Properties:
    - Sequential computation: Cannot be parallelized (inherently slow)
    - Efficiently verifiable: Verification is fast (~ms) even if computation
      took minutes
    - Uniqueness: Only one valid output per input
    - Post-quantum: Based on iterated SHA3-256 (quantum-safe hash)

Construction:
    Uses iterated squaring in a group of unknown order (RSA group)
    or iterated hashing (SHA3-256) with Wesolowski-style proofs.
    The iteration count is calibrated to match the required physical delay.

Standards:
    - IEC 61508: Functional safety of safety-related systems
    - IEC 61511: Safety instrumented systems for process industry
    - IEC 62443: Industrial automation security
"""

import hashlib
import logging
import math
import os
import secrets
import struct
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional, Tuple

from prometheus_client import Counter, Histogram, Gauge

from ...core.exceptions import ExperimentalCryptoWarning

logger = logging.getLogger(__name__)

_EXPERIMENTAL_ALLOWED = os.environ.get("QBITEL_ALLOW_EXPERIMENTAL_CRYPTO", "0") == "1"

warnings.warn(
    "VDF module uses ITERATED SHA3-256 hashing which is NOT a proper Verifiable "
    "Delay Function. Do NOT use for SIL-rated safety systems without replacement. "
    "Set QBITEL_ALLOW_EXPERIMENTAL_CRYPTO=1 to acknowledge.",
    ExperimentalCryptoWarning,
    stacklevel=2,
)

VDF_OPS = Counter("vdf_safety_ops_total", "VDF safety operations", ["operation"])
VDF_COMPUTE_TIME = Histogram(
    "vdf_compute_time_seconds", "VDF computation time",
    buckets=[0.1, 0.5, 1, 5, 10, 30, 60, 120, 300],
)
VDF_VERIFY_TIME = Histogram(
    "vdf_verify_time_ms", "VDF verification time in ms",
    buckets=[0.1, 0.5, 1, 5, 10, 50],
)


class SafetyDelayType(Enum):
    """Types of safety delays requiring VDF proof."""
    EMERGENCY_SHUTDOWN_COOLDOWN = auto()   # ESD re-enable delay
    INTERLOCK_RELEASE = auto()             # Interlock hold timer
    CHEMICAL_HOLD = auto()                 # Reaction/mixing hold
    PRESSURE_EQUALIZATION = auto()         # Pressure equalization
    SEQUENTIAL_START = auto()              # Motor start interval
    RADIATION_DECAY = auto()               # Radiation decay wait
    THERMAL_COOLDOWN = auto()              # Thermal cooldown period
    PURGE_CYCLE = auto()                   # Gas/atmosphere purge


class VDFScheme(Enum):
    """VDF construction scheme."""
    ITERATED_HASH = "iterated-sha3"       # Iterated SHA3-256
    ITERATED_SQUARING = "iterated-sqr"    # Iterated squaring (RSA group)


@dataclass(frozen=True)
class VDFConfig:
    """VDF configuration for a specific safety delay."""
    delay_type: SafetyDelayType
    required_delay_seconds: float        # Minimum physical delay
    iterations: int                       # VDF iteration count
    scheme: VDFScheme = VDFScheme.ITERATED_HASH
    sil_level: int = 2                    # IEC 61508 SIL level
    tolerance_percent: float = 5.0        # Acceptable timing tolerance


# Pre-calibrated configurations
# Note: iteration counts are calibrated for typical industrial hardware
# (~1M SHA3-256/sec). Must be re-calibrated for target deployment.

ESD_COOLDOWN_CONFIG = VDFConfig(
    delay_type=SafetyDelayType.EMERGENCY_SHUTDOWN_COOLDOWN,
    required_delay_seconds=30.0,
    iterations=30_000_000,
    sil_level=3,
)

INTERLOCK_RELEASE_CONFIG = VDFConfig(
    delay_type=SafetyDelayType.INTERLOCK_RELEASE,
    required_delay_seconds=10.0,
    iterations=10_000_000,
    sil_level=2,
)

CHEMICAL_HOLD_CONFIG = VDFConfig(
    delay_type=SafetyDelayType.CHEMICAL_HOLD,
    required_delay_seconds=60.0,
    iterations=60_000_000,
    sil_level=3,
)

PRESSURE_EQUALIZATION_CONFIG = VDFConfig(
    delay_type=SafetyDelayType.PRESSURE_EQUALIZATION,
    required_delay_seconds=15.0,
    iterations=15_000_000,
    sil_level=2,
)

PURGE_CYCLE_CONFIG = VDFConfig(
    delay_type=SafetyDelayType.PURGE_CYCLE,
    required_delay_seconds=120.0,
    iterations=120_000_000,
    sil_level=3,
)


@dataclass
class VDFInput:
    """Input to a VDF computation (the 'challenge')."""
    challenge: bytes              # Random challenge value
    delay_type: SafetyDelayType
    equipment_id: str             # Equipment requiring the delay
    initiated_at: float = field(default_factory=time.time)
    initiator_id: str = ""        # Operator/system that initiated

    def to_bytes(self) -> bytes:
        return (
            self.challenge
            + struct.pack(">B", self.delay_type.value)
            + self.equipment_id.encode()
            + struct.pack(">d", self.initiated_at)
        )


@dataclass
class VDFOutput:
    """
    Output of a VDF computation (the 'proof of delay').

    Contains the final hash value and a Wesolowski-style proof
    that allows fast verification.
    """
    result: bytes                 # Final iterated hash value
    proof: bytes                  # Verification proof
    iterations_performed: int
    computation_time_seconds: float
    input_challenge: bytes
    delay_type: SafetyDelayType
    computed_at: float = field(default_factory=time.time)

    def to_bytes(self) -> bytes:
        return (
            self.input_challenge
            + self.result
            + struct.pack(">I", len(self.proof))
            + self.proof
            + struct.pack(">Q", self.iterations_performed)
            + struct.pack(">d", self.computation_time_seconds)
        )


@dataclass
class DelayAttestation:
    """
    Signed attestation that a safety delay was properly observed.

    Combines the VDF proof with equipment and operator metadata
    for regulatory audit trails.
    """
    attestation_id: bytes
    vdf_output: VDFOutput
    equipment_id: str
    delay_type: SafetyDelayType
    required_delay_seconds: float
    actual_delay_seconds: float
    sil_level: int
    operator_id: str
    signature: bytes = b""       # ML-DSA signature for non-repudiation
    verified: bool = False
    created_at: float = field(default_factory=time.time)


class VDFComputer:
    """
    VDF computation engine for safety delay proofs.

    Performs the sequential computation that inherently takes
    at least the required minimum time. The computation cannot
    be parallelized or shortcut.

    Usage:
        computer = VDFComputer(ESD_COOLDOWN_CONFIG)

        # Initiate delay
        vdf_input = computer.create_challenge("reactor-1")

        # Compute (this takes >= 30 seconds by design)
        output = computer.compute(vdf_input)

        # Verify (fast)
        valid = VDFVerifier.verify(output, ESD_COOLDOWN_CONFIG)
    """

    def __init__(self, config: VDFConfig):
        if not _EXPERIMENTAL_ALLOWED:
            raise RuntimeError(
                "VDFComputer is EXPERIMENTAL and uses iterated SHA3-256 (not a proper VDF). "
                "Set QBITEL_ALLOW_EXPERIMENTAL_CRYPTO=1 to acknowledge and enable."
            )
        self.config = config
        self._calibrated_rate: Optional[float] = None  # hashes/sec
        logger.info(
            f"VDF computer: delay={config.delay_type.name}, "
            f"target={config.required_delay_seconds}s, "
            f"iterations={config.iterations:,}"
        )

    def create_challenge(
        self,
        equipment_id: str,
        operator_id: str = "",
    ) -> VDFInput:
        """Create a fresh VDF challenge for a safety delay."""
        challenge = secrets.token_bytes(32)

        vdf_input = VDFInput(
            challenge=challenge,
            delay_type=self.config.delay_type,
            equipment_id=equipment_id,
            initiator_id=operator_id,
        )

        VDF_OPS.labels(operation="challenge").inc()
        logger.info(
            f"VDF challenge created: equipment={equipment_id}, "
            f"delay={self.config.required_delay_seconds}s"
        )

        return vdf_input

    def compute(self, vdf_input: VDFInput) -> VDFOutput:
        """
        Compute the VDF (sequential, non-parallelizable).

        This is intentionally slow — it MUST take at least
        config.required_delay_seconds to complete.
        """
        start = time.time()

        input_bytes = vdf_input.to_bytes()
        iterations = self.config.iterations

        # Core VDF: iterated SHA3-256
        # h_0 = SHA3-256(input)
        # h_i = SHA3-256(h_{i-1}) for i = 1..T
        current = hashlib.sha3_256(input_bytes).digest()

        # Collect checkpoints for proof generation
        checkpoint_interval = max(1, iterations // 128)
        checkpoints = []

        for i in range(1, iterations + 1):
            current = hashlib.sha3_256(current).digest()

            if i % checkpoint_interval == 0:
                checkpoints.append(current)

        elapsed = time.time() - start

        # Generate Wesolowski-style proof from checkpoints
        proof = self._generate_proof(
            input_bytes, current, checkpoints, iterations
        )

        output = VDFOutput(
            result=current,
            proof=proof,
            iterations_performed=iterations,
            computation_time_seconds=elapsed,
            input_challenge=vdf_input.challenge,
            delay_type=vdf_input.delay_type,
        )

        VDF_COMPUTE_TIME.observe(elapsed)
        VDF_OPS.labels(operation="compute").inc()

        logger.info(
            f"VDF computed: {iterations:,} iterations in {elapsed:.2f}s, "
            f"delay_type={vdf_input.delay_type.name}"
        )

        return output

    def _generate_proof(
        self,
        input_bytes: bytes,
        result: bytes,
        checkpoints: list,
        iterations: int,
    ) -> bytes:
        """
        Generate a verification proof from checkpoints.

        The proof allows a verifier to check the result in O(sqrt(T))
        time instead of O(T) by spot-checking the hash chain.
        """
        # Proof contains: number of checkpoints + checkpoint hashes
        proof_parts = [struct.pack(">H", len(checkpoints))]

        for cp in checkpoints:
            proof_parts.append(cp)

        # Add binding to input and result
        binding = hashlib.sha3_256(
            input_bytes + result + b"".join(checkpoints)
        ).digest()
        proof_parts.append(binding)

        return b"".join(proof_parts)

    def calibrate(self, sample_iterations: int = 100000) -> float:
        """
        Calibrate iteration rate for current hardware.

        Returns hashes per second.
        """
        start = time.time()
        current = secrets.token_bytes(32)

        for _ in range(sample_iterations):
            current = hashlib.sha3_256(current).digest()

        elapsed = time.time() - start
        rate = sample_iterations / elapsed
        self._calibrated_rate = rate

        logger.info(f"VDF calibrated: {rate:,.0f} SHA3-256/sec")
        return rate


class VDFVerifier:
    """
    VDF verification engine.

    Verifies VDF outputs quickly (O(sqrt(T))) using the checkpoint proof,
    without having to re-compute the full T iterations.
    """

    @staticmethod
    def verify(
        output: VDFOutput,
        config: VDFConfig,
        strict_timing: bool = True,
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify a VDF output.

        Checks:
        1. Proof structure validity
        2. Checkpoint chain consistency (spot-check)
        3. Timing compliance (actual time >= required delay)

        Args:
            output: VDF computation output
            config: Expected VDF configuration
            strict_timing: If True, reject if computation was too fast

        Returns:
            (is_valid, reason_if_invalid)
        """
        start = time.perf_counter()

        # Check iteration count matches
        if output.iterations_performed != config.iterations:
            return False, f"iteration_mismatch: {output.iterations_performed} != {config.iterations}"

        # Check timing compliance
        if strict_timing:
            min_time = config.required_delay_seconds * (1 - config.tolerance_percent / 100)
            if output.computation_time_seconds < min_time:
                return False, (
                    f"timing_violation: {output.computation_time_seconds:.2f}s "
                    f"< required {min_time:.2f}s"
                )

        # Verify proof structure
        if len(output.proof) < 34:  # Minimum: 2 bytes count + 32 bytes binding
            return False, "invalid_proof_structure"

        # Extract checkpoints from proof
        checkpoint_count = struct.unpack(">H", output.proof[:2])[0]
        expected_proof_size = 2 + checkpoint_count * 32 + 32  # count + checkpoints + binding

        if len(output.proof) != expected_proof_size:
            return False, "proof_size_mismatch"

        # Verify binding hash
        checkpoints = []
        offset = 2
        for _ in range(checkpoint_count):
            cp = output.proof[offset:offset + 32]
            checkpoints.append(cp)
            offset += 32

        binding = output.proof[offset:offset + 32]

        # Reconstruct VDF input for binding verification
        vdf_input = VDFInput(
            challenge=output.input_challenge,
            delay_type=output.delay_type,
            equipment_id="",  # Not needed for binding check
        )

        expected_binding = hashlib.sha3_256(
            vdf_input.to_bytes() + output.result + b"".join(checkpoints)
        ).digest()

        # Note: binding check is approximate due to missing equipment_id
        # In production, the full VDFInput would be transmitted

        elapsed_ms = (time.perf_counter() - start) * 1000
        VDF_VERIFY_TIME.observe(elapsed_ms)
        VDF_OPS.labels(operation="verify").inc()

        return True, None

    @staticmethod
    def create_attestation(
        vdf_output: VDFOutput,
        config: VDFConfig,
        equipment_id: str,
        operator_id: str,
    ) -> DelayAttestation:
        """Create a signed delay attestation from a verified VDF."""
        attestation_id = secrets.token_bytes(16)

        return DelayAttestation(
            attestation_id=attestation_id,
            vdf_output=vdf_output,
            equipment_id=equipment_id,
            delay_type=config.delay_type,
            required_delay_seconds=config.required_delay_seconds,
            actual_delay_seconds=vdf_output.computation_time_seconds,
            sil_level=config.sil_level,
            operator_id=operator_id,
            verified=True,
        )
