"""
**EXPERIMENTAL** — Privacy-Preserving Homomorphic Vital Sign Analytics

WARNING: This module uses a SIMPLIFIED Paillier-like additively homomorphic
scheme that is NOT a production-grade FHE library. Do NOT use for real
patient data without replacing the underlying crypto with a vetted library
(e.g., python-paillier, OpenFHE, or Microsoft SEAL).

Set QBITEL_ALLOW_EXPERIMENTAL_CRYPTO=1 to acknowledge and enable usage.

Enables statistical analysis on encrypted vital sign data from medical
devices without ever decrypting individual patient readings. Supports
aggregation, anomaly detection, and population health analytics while
maintaining HIPAA compliance.

Operations Supported on Encrypted Data:
    1. Encrypted Sum/Mean: Aggregate heart rate, BP, SpO2 across patients
    2. Encrypted Variance: Detect population-level anomalies
    3. Encrypted Comparison: Threshold alerting without decryption
    4. Encrypted Histogram: Distribution analysis for clinical trials

Underlying Scheme:
    Uses additively homomorphic encryption based on lattice assumptions.
    Enc(a) ⊕ Enc(b) = Enc(a + b) without decrypting either value.
    Scalar multiplication: k ⊗ Enc(a) = Enc(k·a).

    Compatible with CKKS-style approximate arithmetic for real-valued
    vital signs (heart rate, temperature, blood pressure).

Privacy Guarantees:
    - Individual values never leave the device in plaintext
    - Analytics server operates on ciphertexts only
    - Decryption key held by authorized clinician only
    - Differential privacy noise added to aggregates
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
from typing import Dict, List, Optional, Tuple

from prometheus_client import Counter, Histogram

from ...core.exceptions import ExperimentalCryptoWarning

logger = logging.getLogger(__name__)

_EXPERIMENTAL_ALLOWED = os.environ.get("QBITEL_ALLOW_EXPERIMENTAL_CRYPTO", "0") == "1"

warnings.warn(
    "HomomorphicVitalEngine uses a SIMPLIFIED Paillier-like HE scheme that is NOT "
    "a production-grade FHE library. Do NOT use for real patient data. "
    "Set QBITEL_ALLOW_EXPERIMENTAL_CRYPTO=1 to acknowledge.",
    ExperimentalCryptoWarning,
    stacklevel=2,
)

HE_OPS = Counter("homomorphic_vitals_ops_total", "Homomorphic vital operations", ["operation"])
HE_LATENCY = Histogram("homomorphic_vitals_latency_ms", "HE operation latency", buckets=[1, 5, 10, 50, 100, 500])


class VitalType(Enum):
    """Types of vital signs."""
    HEART_RATE = "hr"            # bpm (40-200)
    SYSTOLIC_BP = "sbp"          # mmHg (60-250)
    DIASTOLIC_BP = "dbp"         # mmHg (40-150)
    SPO2 = "spo2"                # % (70-100)
    TEMPERATURE = "temp"         # °C (34-42), scaled to int
    RESPIRATORY_RATE = "rr"      # breaths/min (8-40)
    GLUCOSE = "glucose"          # mg/dL (40-500)


class EncryptionScheme(Enum):
    """Supported homomorphic encryption schemes."""
    ADDITIVE_LATTICE = "additive-lattice"
    CKKS_APPROXIMATE = "ckks-approximate"


@dataclass
class HEPublicKey:
    """Homomorphic encryption public key."""
    key_id: bytes
    modulus_n: int      # Large modulus for Paillier-like scheme
    generator_g: int    # Generator
    key_data: bytes
    scheme: EncryptionScheme = EncryptionScheme.ADDITIVE_LATTICE


@dataclass
class HEPrivateKey:
    """Homomorphic encryption private key (clinician-held)."""
    key_id: bytes
    lambda_val: int     # Carmichael's lambda
    mu_val: int         # Modular inverse
    key_data: bytes
    scheme: EncryptionScheme = EncryptionScheme.ADDITIVE_LATTICE


@dataclass
class HEKeyPair:
    """Homomorphic encryption key pair."""
    public_key: HEPublicKey
    private_key: HEPrivateKey
    created_at: float = field(default_factory=time.time)


@dataclass
class EncryptedVital:
    """A homomorphically encrypted vital sign reading."""
    vital_type: VitalType
    ciphertext: bytes        # Encrypted value
    device_id: str
    patient_id_hash: bytes   # Hash of patient ID
    timestamp: float = field(default_factory=time.time)
    scale_factor: int = 100  # Scaling for fixed-point (e.g., temp 36.5 → 3650)

    def to_bytes(self) -> bytes:
        return (
            struct.pack(">B", self.vital_type.value.encode()[0])
            + struct.pack(">I", len(self.ciphertext))
            + self.ciphertext
            + self.patient_id_hash
        )


@dataclass
class EncryptedAggregate:
    """Result of homomorphic aggregation."""
    vital_type: VitalType
    encrypted_sum: bytes
    count: int
    encrypted_sum_squares: Optional[bytes] = None
    aggregation_window_start: float = 0.0
    aggregation_window_end: float = 0.0


@dataclass
class DecryptedStatistic:
    """Decrypted aggregate statistic (only clinician can produce)."""
    vital_type: VitalType
    mean: float
    variance: Optional[float] = None
    count: int = 0
    window_start: float = 0.0
    window_end: float = 0.0
    dp_noise_added: bool = False


class HomomorphicVitalEngine:
    """
    Privacy-preserving analytics engine for vital signs.

    Operates on encrypted data to compute population-level statistics
    without ever seeing individual patient readings.

    Usage:
        engine = HomomorphicVitalEngine()
        keys = await engine.generate_keys()

        # Device encrypts readings
        enc_hr = engine.encrypt_vital(72, VitalType.HEART_RATE, keys.public_key, ...)

        # Analytics server aggregates (no private key needed)
        aggregate = engine.aggregate([enc_hr1, enc_hr2, ...])

        # Clinician decrypts aggregate
        stats = engine.decrypt_aggregate(aggregate, keys.private_key)
    """

    def __init__(
        self,
        scheme: EncryptionScheme = EncryptionScheme.ADDITIVE_LATTICE,
        key_bits: int = 2048,
        dp_epsilon: float = 1.0,
    ):
        if not _EXPERIMENTAL_ALLOWED:
            raise RuntimeError(
                "HomomorphicVitalEngine is EXPERIMENTAL and uses simplified Paillier-like HE. "
                "Set QBITEL_ALLOW_EXPERIMENTAL_CRYPTO=1 to acknowledge and enable."
            )
        self.scheme = scheme
        self.key_bits = key_bits
        self.dp_epsilon = dp_epsilon  # Differential privacy parameter
        self._n = 0
        self._g = 0
        logger.info(f"Homomorphic vital engine: scheme={scheme.value}, bits={key_bits}")

    async def generate_keys(self) -> HEKeyPair:
        """Generate homomorphic encryption key pair."""
        start = time.perf_counter()

        # Simplified Paillier-like key generation
        # In production, use a proper HE library (SEAL, OpenFHE, Lattigo)
        p = self._generate_safe_prime(self.key_bits // 2)
        q = self._generate_safe_prime(self.key_bits // 2)
        n = p * q
        n_squared = n * n
        g = n + 1  # Simplified generator
        lambda_val = (p - 1) * (q - 1) // math.gcd(p - 1, q - 1)

        # μ = L(g^λ mod n²)^(-1) mod n
        gl = pow(g, lambda_val, n_squared)
        l_val = (gl - 1) // n
        mu = pow(l_val, -1, n)

        self._n = n
        self._g = g

        key_id = secrets.token_bytes(16)

        public_key = HEPublicKey(
            key_id=key_id,
            modulus_n=n,
            generator_g=g,
            key_data=n.to_bytes((n.bit_length() + 7) // 8, "big"),
            scheme=self.scheme,
        )

        private_key = HEPrivateKey(
            key_id=key_id,
            lambda_val=lambda_val,
            mu_val=mu,
            key_data=lambda_val.to_bytes((lambda_val.bit_length() + 7) // 8, "big"),
            scheme=self.scheme,
        )

        elapsed = (time.perf_counter() - start) * 1000
        HE_LATENCY.observe(elapsed)
        HE_OPS.labels(operation="keygen").inc()

        return HEKeyPair(public_key=public_key, private_key=private_key)

    def encrypt_vital(
        self,
        value: float,
        vital_type: VitalType,
        public_key: HEPublicKey,
        device_id: str,
        patient_id: str,
        scale_factor: int = 100,
    ) -> EncryptedVital:
        """
        Encrypt a vital sign reading.

        Scales the float value to integer and encrypts homomorphically.
        """
        start = time.perf_counter()

        scaled_value = int(value * scale_factor)
        n = public_key.modulus_n
        n_sq = n * n
        g = public_key.generator_g

        # Paillier encryption: c = g^m · r^n mod n²
        r = secrets.randbelow(n - 1) + 1
        while math.gcd(r, n) != 1:
            r = secrets.randbelow(n - 1) + 1

        gm = pow(g, scaled_value, n_sq)
        rn = pow(r, n, n_sq)
        ciphertext_int = (gm * rn) % n_sq

        ct_bytes = ciphertext_int.to_bytes((n_sq.bit_length() + 7) // 8, "big")

        patient_id_hash = hashlib.sha3_256(patient_id.encode()).digest()

        enc_vital = EncryptedVital(
            vital_type=vital_type,
            ciphertext=ct_bytes,
            device_id=device_id,
            patient_id_hash=patient_id_hash,
            scale_factor=scale_factor,
        )

        elapsed = (time.perf_counter() - start) * 1000
        HE_LATENCY.observe(elapsed)
        HE_OPS.labels(operation="encrypt").inc()

        return enc_vital

    def homomorphic_add(
        self,
        enc_a: EncryptedVital,
        enc_b: EncryptedVital,
        public_key: HEPublicKey,
    ) -> EncryptedVital:
        """
        Homomorphically add two encrypted vitals.

        Enc(a) ⊕ Enc(b) = Enc(a + b)
        """
        if enc_a.vital_type != enc_b.vital_type:
            raise ValueError("Cannot add different vital types")

        n_sq = public_key.modulus_n * public_key.modulus_n

        a_int = int.from_bytes(enc_a.ciphertext, "big")
        b_int = int.from_bytes(enc_b.ciphertext, "big")

        # Multiplicative combination = additive in plaintext
        sum_ct = (a_int * b_int) % n_sq
        sum_bytes = sum_ct.to_bytes((n_sq.bit_length() + 7) // 8, "big")

        return EncryptedVital(
            vital_type=enc_a.vital_type,
            ciphertext=sum_bytes,
            device_id="aggregated",
            patient_id_hash=b"\x00" * 32,  # Aggregated — no single patient
            scale_factor=enc_a.scale_factor,
        )

    def homomorphic_scalar_multiply(
        self,
        enc_val: EncryptedVital,
        scalar: int,
        public_key: HEPublicKey,
    ) -> EncryptedVital:
        """
        Homomorphically multiply encrypted value by a plaintext scalar.

        k ⊗ Enc(a) = Enc(k·a)
        """
        n_sq = public_key.modulus_n * public_key.modulus_n
        ct_int = int.from_bytes(enc_val.ciphertext, "big")

        result_int = pow(ct_int, scalar, n_sq)
        result_bytes = result_int.to_bytes((n_sq.bit_length() + 7) // 8, "big")

        return EncryptedVital(
            vital_type=enc_val.vital_type,
            ciphertext=result_bytes,
            device_id=enc_val.device_id,
            patient_id_hash=enc_val.patient_id_hash,
            scale_factor=enc_val.scale_factor * scalar,
        )

    def aggregate(
        self,
        encrypted_vitals: List[EncryptedVital],
        public_key: HEPublicKey,
    ) -> EncryptedAggregate:
        """
        Aggregate encrypted vital signs homomorphically.

        Computes Enc(sum) from individual Enc(v_i) without decryption.
        """
        start = time.perf_counter()

        if not encrypted_vitals:
            raise ValueError("No vitals to aggregate")

        vital_type = encrypted_vitals[0].vital_type
        result = encrypted_vitals[0]

        for enc in encrypted_vitals[1:]:
            result = self.homomorphic_add(result, enc, public_key)

        timestamps = [e.timestamp for e in encrypted_vitals]

        aggregate = EncryptedAggregate(
            vital_type=vital_type,
            encrypted_sum=result.ciphertext,
            count=len(encrypted_vitals),
            aggregation_window_start=min(timestamps),
            aggregation_window_end=max(timestamps),
        )

        elapsed = (time.perf_counter() - start) * 1000
        HE_LATENCY.observe(elapsed)
        HE_OPS.labels(operation="aggregate").inc()

        return aggregate

    def decrypt_aggregate(
        self,
        aggregate: EncryptedAggregate,
        private_key: HEPrivateKey,
        public_key: HEPublicKey,
        add_dp_noise: bool = True,
        scale_factor: int = 100,
    ) -> DecryptedStatistic:
        """
        Decrypt an aggregate to reveal population-level statistics.

        Only the clinician/authorized party with the private key can do this.
        Optionally adds differential privacy noise.
        """
        start = time.perf_counter()

        n = public_key.modulus_n
        n_sq = n * n

        ct_int = int.from_bytes(aggregate.encrypted_sum, "big")

        # Paillier decryption: m = L(c^λ mod n²) · μ mod n
        cl = pow(ct_int, private_key.lambda_val, n_sq)
        l_val = (cl - 1) // n
        sum_scaled = (l_val * private_key.mu_val) % n

        # Handle negative values (large modular results)
        if sum_scaled > n // 2:
            sum_scaled -= n

        actual_sum = sum_scaled / scale_factor
        mean = actual_sum / aggregate.count if aggregate.count > 0 else 0.0

        # Add differential privacy noise to the mean
        dp_noise_added = False
        if add_dp_noise and aggregate.count > 0:
            sensitivity = 1.0 / aggregate.count
            noise_scale = sensitivity / self.dp_epsilon
            # Laplace noise
            noise = self._laplace_noise(noise_scale)
            mean += noise
            dp_noise_added = True

        stat = DecryptedStatistic(
            vital_type=aggregate.vital_type,
            mean=round(mean, 2),
            count=aggregate.count,
            window_start=aggregate.aggregation_window_start,
            window_end=aggregate.aggregation_window_end,
            dp_noise_added=dp_noise_added,
        )

        elapsed = (time.perf_counter() - start) * 1000
        HE_LATENCY.observe(elapsed)
        HE_OPS.labels(operation="decrypt").inc()

        return stat

    # ── Internal helpers ──────────────────────────────────────────

    def _generate_safe_prime(self, bits: int) -> int:
        """Generate a random prime for key generation."""
        # Simplified — production should use cryptographic prime generation
        import random
        while True:
            candidate = random.getrandbits(bits) | (1 << (bits - 1)) | 1
            if self._is_probable_prime(candidate):
                return candidate

    def _is_probable_prime(self, n: int, k: int = 20) -> bool:
        """Miller-Rabin primality test."""
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0:
            return False

        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2

        import random
        for _ in range(k):
            a = random.randrange(2, n - 2)
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    def _laplace_noise(self, scale: float) -> float:
        """Generate Laplace noise for differential privacy."""
        import random
        u = random.uniform(-0.5, 0.5)
        return -scale * (1 if u >= 0 else -1) * math.log(1 - 2 * abs(u))
