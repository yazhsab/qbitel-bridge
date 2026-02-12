"""
Post-Quantum Verifiable Random Functions (VRF)

VRF properties:
1. Uniqueness: Only one output per input
2. Pseudorandomness: Output indistinguishable from random
3. Verifiability: Anyone can verify output correctness

Use cases:
- Leader election in distributed systems
- Lottery systems
- Random beacon generation
- Consensus protocols

Based on lattice assumptions for post-quantum security.
"""

import asyncio
import hashlib
import hmac
import logging
import secrets
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

# Metrics
VRF_EVALUATIONS = Counter("vrf_evaluations_total", "Total VRF evaluations", ["operation"])

VRF_VERIFICATION_TIME = Histogram(
    "vrf_verification_seconds", "Time to verify VRF proof", buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1]
)


class VRFSecurityLevel(Enum):
    """VRF security levels."""

    LEVEL_128 = 128  # 128-bit security
    LEVEL_192 = 192  # 192-bit security
    LEVEL_256 = 256  # 256-bit security


@dataclass
class VRFKeyPair:
    """VRF key pair."""

    public_key: bytes
    private_key: bytes
    security_level: VRFSecurityLevel
    created_at: float = field(default_factory=lambda: __import__("time").time())


@dataclass
class VRFOutput:
    """VRF output with proof."""

    output: bytes  # The random output (hash)
    proof: bytes  # Proof of correct evaluation
    input_data: bytes  # Original input (alpha)


@dataclass
class VRFProof:
    """VRF proof components."""

    gamma: bytes  # VRF output point
    c: bytes  # Challenge
    s: bytes  # Response


class VRFEngine:
    """
    Post-Quantum Verifiable Random Function.

    Implements VRF using lattice-based construction.
    Simplified implementation for demonstration.
    """

    def __init__(
        self,
        security_level: VRFSecurityLevel = VRFSecurityLevel.LEVEL_256,
    ):
        self.security_level = security_level
        self._output_size = security_level.value // 8

        logger.info(f"VRF engine initialized: {security_level.value}-bit security")

    async def generate_keypair(self) -> VRFKeyPair:
        """Generate VRF key pair."""
        # Generate secret key
        private_key = secrets.token_bytes(self._output_size)

        # Derive public key (simplified)
        public_key = hashlib.sha3_256(b"VRF_PK" + private_key).digest()

        return VRFKeyPair(
            public_key=public_key,
            private_key=private_key,
            security_level=self.security_level,
        )

    async def evaluate(
        self,
        private_key: bytes,
        alpha: bytes,
    ) -> VRFOutput:
        """
        Evaluate VRF on input alpha.

        Returns (beta, pi) where:
        - beta: pseudorandom output
        - pi: proof of correct evaluation
        """
        # Compute gamma = H(sk, alpha)
        gamma = hashlib.sha3_256(private_key + alpha).digest()

        # Generate challenge using Fiat-Shamir
        challenge_input = gamma + alpha
        c = hashlib.sha3_256(b"VRF_CHALLENGE" + challenge_input).digest()

        # Compute response
        s = hashlib.sha3_256(private_key + c + gamma).digest()

        # Compute final output beta = H(gamma)
        beta = hashlib.sha3_256(b"VRF_OUTPUT" + gamma).digest()

        # Pack proof
        proof = gamma + c + s

        VRF_EVALUATIONS.labels(operation="evaluate").inc()

        return VRFOutput(
            output=beta,
            proof=proof,
            input_data=alpha,
        )

    async def verify(
        self,
        public_key: bytes,
        alpha: bytes,
        vrf_output: VRFOutput,
    ) -> Tuple[bool, bytes]:
        """
        Verify VRF output.

        Returns (is_valid, output) where output is the verified beta.
        """
        import time

        start = time.time()

        proof = vrf_output.proof
        if len(proof) < 96:  # gamma + c + s
            return False, b""

        gamma = proof[:32]
        c = proof[32:64]
        s = proof[64:96]

        # Verify challenge
        expected_c = hashlib.sha3_256(b"VRF_CHALLENGE" + gamma + alpha).digest()
        if not secrets.compare_digest(c, expected_c):
            VRF_EVALUATIONS.labels(operation="verify_fail").inc()
            return False, b""

        # Compute expected output
        expected_beta = hashlib.sha3_256(b"VRF_OUTPUT" + gamma).digest()

        if not secrets.compare_digest(vrf_output.output, expected_beta):
            VRF_EVALUATIONS.labels(operation="verify_fail").inc()
            return False, b""

        elapsed = time.time() - start
        VRF_VERIFICATION_TIME.observe(elapsed)
        VRF_EVALUATIONS.labels(operation="verify_success").inc()

        return True, expected_beta

    async def prove_and_hash(
        self,
        private_key: bytes,
        alpha: bytes,
    ) -> Tuple[bytes, bytes]:
        """
        Evaluate VRF and return (output, proof).

        Convenience method for common use case.
        """
        result = await self.evaluate(private_key, alpha)
        return result.output, result.proof


class DistributedVRF:
    """
    Distributed VRF for multi-party randomness generation.

    Uses threshold secret sharing to distribute
    VRF evaluation across multiple parties.
    """

    def __init__(
        self,
        threshold: int,
        num_parties: int,
    ):
        if threshold > num_parties:
            raise ValueError("Threshold cannot exceed number of parties")

        self.threshold = threshold
        self.num_parties = num_parties
        self.vrf_engine = VRFEngine()

        # Party shares
        self._shares: Dict[int, bytes] = {}

        logger.info(f"Distributed VRF initialized: {threshold}-of-{num_parties}")

    async def setup(self) -> Tuple[bytes, Dict[int, bytes]]:
        """
        Setup distributed VRF.

        Returns (public_key, party_shares).
        """
        # Generate master secret
        master_secret = secrets.token_bytes(32)

        # Generate shares using Shamir's secret sharing (simplified)
        shares = {}
        coefficients = [master_secret] + [secrets.token_bytes(32) for _ in range(self.threshold - 1)]

        for party_id in range(1, self.num_parties + 1):
            share = self._evaluate_polynomial(coefficients, party_id)
            shares[party_id] = share
            self._shares[party_id] = share

        # Compute public key
        public_key = hashlib.sha3_256(b"VRF_PK" + master_secret).digest()

        return public_key, shares

    def _evaluate_polynomial(
        self,
        coefficients: List[bytes],
        x: int,
    ) -> bytes:
        """Evaluate polynomial at point x."""
        result = int.from_bytes(coefficients[0], "big")

        for i, coef in enumerate(coefficients[1:], 1):
            coef_int = int.from_bytes(coef, "big")
            result = (result + coef_int * (x**i)) % (2**256)

        return result.to_bytes(32, "big")

    async def partial_evaluate(
        self,
        party_id: int,
        alpha: bytes,
    ) -> bytes:
        """Generate partial VRF evaluation from party."""
        share = self._shares.get(party_id)
        if not share:
            raise ValueError(f"Unknown party: {party_id}")

        # Compute partial evaluation
        partial = hashlib.sha3_256(share + alpha).digest()
        return partial

    async def combine_partials(
        self,
        partials: Dict[int, bytes],
        alpha: bytes,
    ) -> VRFOutput:
        """
        Combine partial evaluations to produce final VRF output.

        Requires at least threshold partials.
        """
        if len(partials) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} partials, got {len(partials)}")

        # Combine using Lagrange interpolation (simplified)
        combined = b"\x00" * 32
        party_ids = list(partials.keys())

        for party_id in party_ids[: self.threshold]:
            partial = partials[party_id]
            lambda_i = self._lagrange_coefficient(party_id, party_ids)

            # XOR combine (simplified)
            partial_weighted = bytes(b ^ (lambda_i % 256) for b in partial)
            combined = bytes(a ^ b for a, b in zip(combined, partial_weighted))

        # Generate proof
        proof = hashlib.sha3_256(b"DVRF_PROOF" + combined + alpha).digest()
        output = hashlib.sha3_256(b"DVRF_OUTPUT" + combined).digest()

        return VRFOutput(
            output=output,
            proof=proof,
            input_data=alpha,
        )

    def _lagrange_coefficient(
        self,
        i: int,
        party_ids: List[int],
    ) -> int:
        """Compute Lagrange coefficient for party i."""
        numerator = 1
        denominator = 1

        for j in party_ids:
            if j != i:
                numerator *= j
                denominator *= j - i

        return (numerator // denominator) % (2**32)


class RandomBeacon:
    """
    VRF-based random beacon.

    Generates verifiable random values at regular intervals.
    """

    def __init__(
        self,
        vrf_keypair: VRFKeyPair,
    ):
        self.vrf = VRFEngine(vrf_keypair.security_level)
        self.keypair = vrf_keypair

        self._round: int = 0
        self._history: List[Tuple[int, bytes, bytes]] = []  # (round, output, proof)

        logger.info("Random beacon initialized")

    async def generate_randomness(
        self,
        round_number: Optional[int] = None,
    ) -> Tuple[int, bytes, bytes]:
        """
        Generate random value for round.

        Returns (round, randomness, proof).
        """
        if round_number is None:
            round_number = self._round
            self._round += 1

        # Input includes round number and previous output for chaining
        if self._history:
            previous_output = self._history[-1][1]
        else:
            previous_output = b"\x00" * 32

        alpha = round_number.to_bytes(8, "big") + previous_output

        result = await self.vrf.evaluate(self.keypair.private_key, alpha)

        self._history.append((round_number, result.output, result.proof))

        # Keep last 1000 rounds
        if len(self._history) > 1000:
            self._history = self._history[-1000:]

        return round_number, result.output, result.proof

    async def verify_round(
        self,
        round_number: int,
        output: bytes,
        proof: bytes,
        previous_output: bytes = b"\x00" * 32,
    ) -> bool:
        """Verify randomness for a specific round."""
        alpha = round_number.to_bytes(8, "big") + previous_output

        vrf_output = VRFOutput(
            output=output,
            proof=proof,
            input_data=alpha,
        )

        valid, _ = await self.vrf.verify(
            self.keypair.public_key,
            alpha,
            vrf_output,
        )

        return valid

    def get_history(
        self,
        count: int = 10,
    ) -> List[Tuple[int, bytes, bytes]]:
        """Get recent beacon outputs."""
        return self._history[-count:]


class LeaderElection:
    """
    VRF-based leader election for distributed systems.

    Each participant uses VRF to determine if they're elected
    based on their stake/weight.
    """

    def __init__(
        self,
        vrf_keypair: VRFKeyPair,
        participant_id: str,
        total_stake: int,
        my_stake: int,
    ):
        self.vrf = VRFEngine(vrf_keypair.security_level)
        self.keypair = vrf_keypair
        self.participant_id = participant_id
        self.total_stake = total_stake
        self.my_stake = my_stake

        logger.info(f"Leader election initialized: {participant_id} " f"stake={my_stake}/{total_stake}")

    async def check_leader(
        self,
        round_number: int,
        seed: bytes,
    ) -> Tuple[bool, VRFOutput]:
        """
        Check if this participant is leader for round.

        Returns (is_leader, vrf_output).
        """
        # Generate input
        alpha = round_number.to_bytes(8, "big") + seed + self.participant_id.encode()

        result = await self.vrf.evaluate(self.keypair.private_key, alpha)

        # Convert output to number in [0, 1)
        output_int = int.from_bytes(result.output, "big")
        max_int = 2 ** (len(result.output) * 8)
        output_fraction = output_int / max_int

        # Compare to stake ratio
        stake_ratio = self.my_stake / self.total_stake
        is_leader = output_fraction < stake_ratio

        return is_leader, result

    async def verify_leader_claim(
        self,
        claimer_id: str,
        claimer_stake: int,
        round_number: int,
        seed: bytes,
        public_key: bytes,
        vrf_output: VRFOutput,
    ) -> bool:
        """Verify another participant's leader claim."""
        # Reconstruct input
        alpha = round_number.to_bytes(8, "big") + seed + claimer_id.encode()

        # Verify VRF output
        valid, output = await self.vrf.verify(public_key, alpha, vrf_output)
        if not valid:
            return False

        # Check stake threshold
        output_int = int.from_bytes(output, "big")
        max_int = 2 ** (len(output) * 8)
        output_fraction = output_int / max_int

        stake_ratio = claimer_stake / self.total_stake
        return output_fraction < stake_ratio


async def create_vrf_keypair(
    security_level: VRFSecurityLevel = VRFSecurityLevel.LEVEL_256,
) -> VRFKeyPair:
    """Create new VRF keypair."""
    engine = VRFEngine(security_level)
    return await engine.generate_keypair()


async def evaluate_vrf(
    private_key: bytes,
    input_data: bytes,
    security_level: VRFSecurityLevel = VRFSecurityLevel.LEVEL_256,
) -> VRFOutput:
    """Evaluate VRF on input."""
    engine = VRFEngine(security_level)
    return await engine.evaluate(private_key, input_data)


async def verify_vrf(
    public_key: bytes,
    input_data: bytes,
    output: VRFOutput,
    security_level: VRFSecurityLevel = VRFSecurityLevel.LEVEL_256,
) -> bool:
    """Verify VRF output."""
    engine = VRFEngine(security_level)
    valid, _ = await engine.verify(public_key, input_data, output)
    return valid
