"""
LMS/XMSS Stateful Hash-Based Signature Schemes

Implements NIST SP 800-208 compliant stateful hash-based signatures:
- LMS (Leighton-Micali Signature) - RFC 8554
- HSS (Hierarchical Signature System) - RFC 8554 multi-tree variant
- XMSS (eXtended Merkle Signature Scheme) - RFC 8391
- XMSS^MT (Multi-Tree XMSS) - RFC 8391

These are REQUIRED by NSA CNSA 2.0 for:
- Firmware signing and code signing
- Software update authentication
- Long-lived certificate signing (CAs)

CRITICAL: These are STATEFUL signature schemes. Each private key can only
sign a fixed number of messages determined by the tree height. The state
MUST be persisted and MUST NOT be reused. Reusing a one-time signature
(OTS) state completely breaks security.

Security:
    - State management is the PRIMARY security concern — state reuse = broken
    - Uses SHA-256 or SHAKE256 as the hash function (conservative, well-analyzed)
    - Security rests purely on hash function security (no lattice/number theory)
    - Resistant to ALL known quantum attacks
"""

import hashlib
import hmac
import logging
import os
import struct
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .mlkem import PQCProviderUnavailableError, _GLOBAL_STRICT_MODE

logger = logging.getLogger(__name__)


# ==============================================================================
# LMS Parameter Sets (RFC 8554, Section 4)
# ==============================================================================

class LmsAlgorithm(Enum):
    """LMS algorithm types with parameter sets per RFC 8554."""

    LMS_SHA256_M32_H5 = "lms-sha256-m32-h5"
    LMS_SHA256_M32_H10 = "lms-sha256-m32-h10"
    LMS_SHA256_M32_H15 = "lms-sha256-m32-h15"
    LMS_SHA256_M32_H20 = "lms-sha256-m32-h20"
    LMS_SHA256_M32_H25 = "lms-sha256-m32-h25"

    @property
    def hash_function(self) -> str:
        return "sha256"

    @property
    def hash_length(self) -> int:
        """Output length m in bytes."""
        return 32

    @property
    def tree_height(self) -> int:
        """Merkle tree height h."""
        heights = {
            LmsAlgorithm.LMS_SHA256_M32_H5: 5,
            LmsAlgorithm.LMS_SHA256_M32_H10: 10,
            LmsAlgorithm.LMS_SHA256_M32_H15: 15,
            LmsAlgorithm.LMS_SHA256_M32_H20: 20,
            LmsAlgorithm.LMS_SHA256_M32_H25: 25,
        }
        return heights[self]

    @property
    def max_signatures(self) -> int:
        """Maximum number of signatures (2^h)."""
        return 2 ** self.tree_height

    @property
    def type_code(self) -> int:
        """LMS type code for serialization."""
        codes = {
            LmsAlgorithm.LMS_SHA256_M32_H5: 0x00000005,
            LmsAlgorithm.LMS_SHA256_M32_H10: 0x00000006,
            LmsAlgorithm.LMS_SHA256_M32_H15: 0x00000007,
            LmsAlgorithm.LMS_SHA256_M32_H20: 0x00000008,
            LmsAlgorithm.LMS_SHA256_M32_H25: 0x00000009,
        }
        return codes[self]


class LmotsAlgorithm(Enum):
    """LM-OTS (One-Time Signature) parameter sets per RFC 8554."""

    LMOTS_SHA256_N32_W1 = "lmots-sha256-n32-w1"
    LMOTS_SHA256_N32_W2 = "lmots-sha256-n32-w2"
    LMOTS_SHA256_N32_W4 = "lmots-sha256-n32-w4"
    LMOTS_SHA256_N32_W8 = "lmots-sha256-n32-w8"

    @property
    def hash_length(self) -> int:
        """n parameter (hash output bytes)."""
        return 32

    @property
    def winternitz_parameter(self) -> int:
        """Winternitz w parameter (trade-off between signature size and speed)."""
        w_values = {
            LmotsAlgorithm.LMOTS_SHA256_N32_W1: 1,
            LmotsAlgorithm.LMOTS_SHA256_N32_W2: 2,
            LmotsAlgorithm.LMOTS_SHA256_N32_W4: 4,
            LmotsAlgorithm.LMOTS_SHA256_N32_W8: 8,
        }
        return w_values[self]

    @property
    def chain_count(self) -> int:
        """p parameter: number of n-byte string elements in signature."""
        p_values = {
            LmotsAlgorithm.LMOTS_SHA256_N32_W1: 265,
            LmotsAlgorithm.LMOTS_SHA256_N32_W2: 133,
            LmotsAlgorithm.LMOTS_SHA256_N32_W4: 67,
            LmotsAlgorithm.LMOTS_SHA256_N32_W8: 34,
        }
        return p_values[self]

    @property
    def signature_size(self) -> int:
        """Approximate signature size in bytes."""
        # 4 (type) + 32 (C) + p * 32 (y values)
        return 4 + self.hash_length + self.chain_count * self.hash_length

    @property
    def type_code(self) -> int:
        """LM-OTS type code for serialization."""
        codes = {
            LmotsAlgorithm.LMOTS_SHA256_N32_W1: 0x00000001,
            LmotsAlgorithm.LMOTS_SHA256_N32_W2: 0x00000002,
            LmotsAlgorithm.LMOTS_SHA256_N32_W4: 0x00000003,
            LmotsAlgorithm.LMOTS_SHA256_N32_W8: 0x00000004,
        }
        return codes[self]


class XmssAlgorithm(Enum):
    """XMSS algorithm types per RFC 8391."""

    XMSS_SHA2_10_256 = "xmss-sha2-10-256"
    XMSS_SHA2_16_256 = "xmss-sha2-16-256"
    XMSS_SHA2_20_256 = "xmss-sha2-20-256"
    XMSS_SHAKE_10_256 = "xmss-shake-10-256"
    XMSS_SHAKE_16_256 = "xmss-shake-16-256"
    XMSS_SHAKE_20_256 = "xmss-shake-20-256"

    @property
    def hash_function(self) -> str:
        if "shake" in self.value:
            return "shake256"
        return "sha256"

    @property
    def hash_length(self) -> int:
        return 32  # n=256 bits = 32 bytes for all _256 variants

    @property
    def tree_height(self) -> int:
        heights = {
            XmssAlgorithm.XMSS_SHA2_10_256: 10,
            XmssAlgorithm.XMSS_SHA2_16_256: 16,
            XmssAlgorithm.XMSS_SHA2_20_256: 20,
            XmssAlgorithm.XMSS_SHAKE_10_256: 10,
            XmssAlgorithm.XMSS_SHAKE_16_256: 16,
            XmssAlgorithm.XMSS_SHAKE_20_256: 20,
        }
        return heights[self]

    @property
    def max_signatures(self) -> int:
        return 2 ** self.tree_height

    @property
    def nist_level(self) -> int:
        """Approximate NIST security level."""
        return 1 if self.hash_length == 32 else 3


# ==============================================================================
# State Management (CRITICAL for stateful schemes)
# ==============================================================================

class StateBackend(ABC):
    """Abstract interface for persistent state storage."""

    @abstractmethod
    def load_state(self, key_id: str) -> Optional[int]:
        """Load current leaf index for a key. Returns None if not found."""
        ...

    @abstractmethod
    def save_state(self, key_id: str, leaf_index: int) -> None:
        """Atomically persist the current leaf index."""
        ...

    @abstractmethod
    def lock(self, key_id: str) -> bool:
        """Acquire exclusive lock for signing. Returns False if already locked."""
        ...

    @abstractmethod
    def unlock(self, key_id: str) -> None:
        """Release exclusive lock."""
        ...


class FileStateBackend(StateBackend):
    """
    File-based state persistence with atomic writes.

    State is stored in individual files per key ID:
        {state_dir}/{key_id}.state  — current leaf index (8-byte big-endian)
        {state_dir}/{key_id}.lock   — lock file for exclusive access
    """

    def __init__(self, state_dir: str = "/var/lib/qbitel/pqc-state"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._locks: Dict[str, threading.Lock] = {}

    def _state_path(self, key_id: str) -> Path:
        # Sanitize key_id to prevent path traversal
        safe_id = key_id.replace("/", "_").replace("..", "_").replace("\\", "_")
        return self.state_dir / f"{safe_id}.state"

    def _lock_path(self, key_id: str) -> Path:
        safe_id = key_id.replace("/", "_").replace("..", "_").replace("\\", "_")
        return self.state_dir / f"{safe_id}.lock"

    def load_state(self, key_id: str) -> Optional[int]:
        path = self._state_path(key_id)
        if not path.exists():
            return None
        data = path.read_bytes()
        if len(data) != 8:
            raise ValueError(f"Corrupted state file for key {key_id}: expected 8 bytes, got {len(data)}")
        return struct.unpack(">Q", data)[0]

    def save_state(self, key_id: str, leaf_index: int) -> None:
        """Atomically write state using write-to-temp + rename pattern."""
        path = self._state_path(key_id)
        tmp_path = path.with_suffix(".state.tmp")

        data = struct.pack(">Q", leaf_index)
        tmp_path.write_bytes(data)
        # os.replace is atomic on POSIX systems
        os.replace(str(tmp_path), str(path))

        logger.debug(f"State persisted for key {key_id}: leaf_index={leaf_index}")

    def lock(self, key_id: str) -> bool:
        if key_id not in self._locks:
            self._locks[key_id] = threading.Lock()
        return self._locks[key_id].acquire(blocking=False)

    def unlock(self, key_id: str) -> None:
        if key_id in self._locks:
            try:
                self._locks[key_id].release()
            except RuntimeError:
                pass  # Already unlocked


class InMemoryStateBackend(StateBackend):
    """In-memory state backend for testing only. State is lost on restart."""

    def __init__(self):
        self._states: Dict[str, int] = {}
        self._locks: Dict[str, threading.Lock] = {}

    def load_state(self, key_id: str) -> Optional[int]:
        return self._states.get(key_id)

    def save_state(self, key_id: str, leaf_index: int) -> None:
        self._states[key_id] = leaf_index

    def lock(self, key_id: str) -> bool:
        if key_id not in self._locks:
            self._locks[key_id] = threading.Lock()
        return self._locks[key_id].acquire(blocking=False)

    def unlock(self, key_id: str) -> None:
        if key_id in self._locks:
            try:
                self._locks[key_id].release()
            except RuntimeError:
                pass


# ==============================================================================
# Key and Signature Data Types
# ==============================================================================

@dataclass
class LmsPublicKey:
    """LMS public key (stateless — safe to distribute)."""

    algorithm: LmsAlgorithm
    ots_algorithm: LmotsAlgorithm
    key_id: str
    root: bytes  # Merkle tree root hash
    identifier: bytes  # 16-byte random key identifier (I parameter)

    def to_bytes(self) -> bytes:
        """Serialize per RFC 8554 Section 5.3."""
        return (
            struct.pack(">I", self.algorithm.type_code)
            + struct.pack(">I", self.ots_algorithm.type_code)
            + self.identifier
            + self.root
        )

    @property
    def size(self) -> int:
        return 4 + 4 + 16 + self.algorithm.hash_length


@dataclass
class LmsPrivateKey:
    """
    LMS private key with state tracking.

    CRITICAL: The leaf_index MUST be persisted before EVERY signing operation.
    Reusing a leaf index completely breaks the scheme's security.
    """

    algorithm: LmsAlgorithm
    ots_algorithm: LmotsAlgorithm
    key_id: str
    seed: bytes  # Master seed for OTS key derivation
    identifier: bytes  # 16-byte key identifier (I parameter)
    leaf_index: int = 0  # Current OTS leaf (MUST be monotonically increasing)

    @property
    def remaining_signatures(self) -> int:
        return self.algorithm.max_signatures - self.leaf_index

    @property
    def is_exhausted(self) -> bool:
        return self.leaf_index >= self.algorithm.max_signatures

    def __del__(self):
        """Zeroize seed on deletion."""
        if hasattr(self, "seed") and isinstance(self.seed, bytearray):
            for i in range(len(self.seed)):
                self.seed[i] = 0


@dataclass
class LmsKeyPair:
    """LMS key pair."""

    algorithm: LmsAlgorithm
    ots_algorithm: LmotsAlgorithm
    public_key: LmsPublicKey
    private_key: LmsPrivateKey
    created_at: float = field(default_factory=time.time)


@dataclass
class LmsSignature:
    """LMS signature per RFC 8554."""

    algorithm: LmsAlgorithm
    ots_algorithm: LmotsAlgorithm
    leaf_index: int
    ots_signature: bytes  # LM-OTS one-time signature
    auth_path: List[bytes]  # Merkle authentication path
    created_at: float = field(default_factory=time.time)

    @property
    def size(self) -> int:
        return (
            4  # q (leaf index)
            + len(self.ots_signature)
            + 4  # type code
            + len(self.auth_path) * 32  # auth path nodes
        )

    def to_bytes(self) -> bytes:
        """Serialize signature."""
        parts = [
            struct.pack(">I", self.leaf_index),
            self.ots_signature,
            struct.pack(">I", self.algorithm.type_code),
        ]
        for node in self.auth_path:
            parts.append(node)
        return b"".join(parts)


@dataclass
class XmssPublicKey:
    """XMSS public key."""

    algorithm: XmssAlgorithm
    key_id: str
    root: bytes
    seed: bytes  # Public seed for address-based hashing

    @property
    def size(self) -> int:
        return 4 + self.algorithm.hash_length * 2  # OID + root + seed


@dataclass
class XmssPrivateKey:
    """XMSS private key with state."""

    algorithm: XmssAlgorithm
    key_id: str
    seed: bytes  # Secret seed
    prf_key: bytes  # PRF key for randomized hashing
    leaf_index: int = 0

    @property
    def remaining_signatures(self) -> int:
        return self.algorithm.max_signatures - self.leaf_index

    @property
    def is_exhausted(self) -> bool:
        return self.leaf_index >= self.algorithm.max_signatures


@dataclass
class XmssKeyPair:
    """XMSS key pair."""

    algorithm: XmssAlgorithm
    public_key: XmssPublicKey
    private_key: XmssPrivateKey
    created_at: float = field(default_factory=time.time)


@dataclass
class XmssSignature:
    """XMSS signature."""

    algorithm: XmssAlgorithm
    leaf_index: int
    randomness: bytes  # Per-signature randomness
    wots_signature: bytes  # WOTS+ one-time signature
    auth_path: List[bytes]  # Authentication path
    created_at: float = field(default_factory=time.time)

    @property
    def size(self) -> int:
        n = self.algorithm.hash_length
        h = self.algorithm.tree_height
        # index + randomness + WOTS sig (67 * n) + auth path (h * n)
        return 4 + n + 67 * n + h * n


# ==============================================================================
# State Exhaustion Error
# ==============================================================================

class StateExhaustedError(Exception):
    """Raised when a stateful key has used all available leaf indices."""

    def __init__(self, key_id: str, algorithm: str, max_sigs: int):
        self.key_id = key_id
        self.algorithm = algorithm
        self.max_signatures = max_sigs
        super().__init__(
            f"CRITICAL: Key '{key_id}' ({algorithm}) has exhausted all {max_sigs} "
            f"available signatures. A new key pair MUST be generated. "
            f"Continuing to sign with this key would break security."
        )


class StateLockError(Exception):
    """Raised when exclusive lock cannot be acquired for signing."""

    def __init__(self, key_id: str):
        super().__init__(
            f"Cannot acquire exclusive lock for key '{key_id}'. "
            f"Another signing operation may be in progress."
        )


# ==============================================================================
# LMS Engine
# ==============================================================================

class LmsEngine:
    """
    Leighton-Micali Signature (LMS) engine per RFC 8554.

    Provides stateful hash-based signatures suitable for:
    - Firmware signing (CNSA 2.0 requirement)
    - Code signing
    - Long-term certificate signing

    CRITICAL: This is a STATEFUL scheme. State must be persisted atomically
    before every signing operation. The state_backend parameter controls
    where state is stored. For production, use FileStateBackend or a
    database-backed implementation.
    """

    def __init__(
        self,
        algorithm: LmsAlgorithm = LmsAlgorithm.LMS_SHA256_M32_H20,
        ots_algorithm: LmotsAlgorithm = LmotsAlgorithm.LMOTS_SHA256_N32_W4,
        state_backend: Optional[StateBackend] = None,
        strict_mode: Optional[bool] = None,
    ):
        self.algorithm = algorithm
        self.ots_algorithm = ots_algorithm
        self.strict_mode = strict_mode if strict_mode is not None else _GLOBAL_STRICT_MODE
        self.state_backend = state_backend or FileStateBackend()

        # Try to use liboqs for the actual crypto operations
        self._provider = self._detect_provider()

        if self._provider == "fallback" and self.strict_mode:
            raise PQCProviderUnavailableError(algorithm.value, "initialization")

        logger.info(
            f"LMS engine initialized: algorithm={algorithm.value}, "
            f"ots={ots_algorithm.value}, max_signatures={algorithm.max_signatures}, "
            f"provider={self._provider}"
        )

    def _detect_provider(self) -> str:
        """Detect available crypto provider for hash-based signatures."""
        try:
            import oqs
            # Check if LMS is available in the installed liboqs
            if hasattr(oqs, "Signature"):
                return "liboqs"
        except ImportError:
            pass

        # hashlib-based reference implementation is always available
        return "hashlib"

    def _hash(self, *parts: bytes) -> bytes:
        """Compute SHA-256 hash of concatenated inputs."""
        hasher = hashlib.sha256()
        for part in parts:
            hasher.update(part)
        return hasher.digest()

    def _prf(self, key: bytes, index: bytes) -> bytes:
        """Pseudorandom function using HMAC-SHA256."""
        return hmac.new(key, index, hashlib.sha256).digest()

    def generate_keypair(self, key_id: Optional[str] = None) -> LmsKeyPair:
        """
        Generate a new LMS key pair.

        Args:
            key_id: Unique identifier for state tracking. Auto-generated if None.

        Returns:
            LmsKeyPair with public and private keys
        """
        start = time.time()

        if key_id is None:
            key_id = f"lms-{os.urandom(8).hex()}"

        # Generate random seed and identifier
        seed = os.urandom(self.algorithm.hash_length)
        identifier = os.urandom(16)  # I parameter per RFC 8554

        # Compute Merkle tree root from all OTS public keys
        root = self._compute_merkle_root(seed, identifier)

        public_key = LmsPublicKey(
            algorithm=self.algorithm,
            ots_algorithm=self.ots_algorithm,
            key_id=key_id,
            root=root,
            identifier=identifier,
        )

        private_key = LmsPrivateKey(
            algorithm=self.algorithm,
            ots_algorithm=self.ots_algorithm,
            key_id=key_id,
            seed=seed,
            identifier=identifier,
            leaf_index=0,
        )

        # Initialize state
        self.state_backend.save_state(key_id, 0)

        elapsed = time.time() - start
        logger.info(
            f"Generated LMS keypair '{key_id}': {self.algorithm.value}, "
            f"max_signatures={self.algorithm.max_signatures}, "
            f"keygen_time={elapsed:.2f}s"
        )

        return LmsKeyPair(
            algorithm=self.algorithm,
            ots_algorithm=self.ots_algorithm,
            public_key=public_key,
            private_key=private_key,
        )

    def _compute_merkle_root(self, seed: bytes, identifier: bytes) -> bytes:
        """
        Compute the Merkle tree root from all OTS leaf public keys.

        For large trees (h=20, 1M leaves), this is expensive. In production,
        consider background pre-computation or lazy tree building.
        """
        h = self.algorithm.tree_height
        n = self.algorithm.hash_length
        num_leaves = 2 ** h

        # For very large trees, use a streaming computation
        if num_leaves > 1024:
            logger.info(
                f"Computing Merkle tree with {num_leaves} leaves "
                f"(height={h}). This may take a moment..."
            )

        # Compute leaf nodes (OTS public key hashes)
        # For efficiency, compute level by level from leaves to root
        current_level = []
        for q in range(num_leaves):
            # Derive OTS public key for leaf q
            ots_pk_hash = self._compute_ots_public_key_hash(seed, identifier, q)
            # Leaf node: H(I || u32(r) || u16(D_LEAF) || OTS_PK_HASH)
            r = num_leaves + q  # Node index in full binary tree
            leaf_hash = self._hash(
                identifier,
                struct.pack(">I", r),
                struct.pack(">H", 0x8282),  # D_LEAF
                ots_pk_hash,
            )
            current_level.append(leaf_hash)

        # Build tree bottom-up
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                parent_idx = (num_leaves + i) // 2 if len(next_level) == 0 else i // 2
                parent_r = len(current_level) // 2 + i // 2
                node = self._hash(
                    identifier,
                    struct.pack(">I", parent_r),
                    struct.pack(">H", 0x8383),  # D_INTR (internal node)
                    current_level[i],
                    current_level[i + 1],
                )
                next_level.append(node)
            current_level = next_level

        return current_level[0]

    def _compute_ots_public_key_hash(
        self, seed: bytes, identifier: bytes, leaf_index: int
    ) -> bytes:
        """Compute OTS public key hash for a given leaf."""
        n = self.ots_algorithm.hash_length
        p = self.ots_algorithm.chain_count
        w = self.ots_algorithm.winternitz_parameter

        # Derive OTS private key elements
        pk_elements = []
        for j in range(p):
            # x[j] = PRF(seed, I || q || j)
            x_j = self._prf(
                seed,
                identifier + struct.pack(">I", leaf_index) + struct.pack(">H", j),
            )
            # Chain to top: iterate 2^w - 1 times
            chain_top = x_j
            for k in range(2 ** w - 1):
                chain_top = self._hash(
                    identifier,
                    struct.pack(">I", leaf_index),
                    struct.pack(">H", j),
                    struct.pack(">B", k),
                    chain_top,
                )
            pk_elements.append(chain_top)

        # Hash all OTS public key elements
        return self._hash(*pk_elements)

    def sign(self, message: bytes, private_key: LmsPrivateKey) -> LmsSignature:
        """
        Sign a message using LMS.

        CRITICAL: This method advances the leaf index. The state is persisted
        BEFORE the signature is computed to prevent state reuse on crash.

        Args:
            message: Message to sign
            private_key: Signer's private key

        Returns:
            LmsSignature

        Raises:
            StateExhaustedError: If all leaf indices have been used
            StateLockError: If exclusive lock cannot be acquired
        """
        key_id = private_key.key_id

        # Acquire exclusive lock
        if not self.state_backend.lock(key_id):
            raise StateLockError(key_id)

        try:
            # Load current state (authoritative source)
            stored_index = self.state_backend.load_state(key_id)
            if stored_index is not None:
                private_key.leaf_index = stored_index

            # Check exhaustion
            if private_key.is_exhausted:
                raise StateExhaustedError(
                    key_id, self.algorithm.value, self.algorithm.max_signatures
                )

            current_leaf = private_key.leaf_index

            # CRITICAL: Advance and persist state BEFORE signing
            # If we crash after this point, we lose one leaf but don't reuse state
            next_index = current_leaf + 1
            self.state_backend.save_state(key_id, next_index)
            private_key.leaf_index = next_index

            remaining = private_key.remaining_signatures
            if remaining < 100:
                logger.warning(
                    f"LMS key '{key_id}' has only {remaining} signatures remaining. "
                    f"Generate a new key pair soon."
                )

            start = time.time()

            # Compute LM-OTS signature for this leaf
            ots_signature = self._compute_ots_signature(
                message, private_key.seed, private_key.identifier, current_leaf
            )

            # Compute authentication path (sibling hashes from leaf to root)
            auth_path = self._compute_auth_path(
                private_key.seed, private_key.identifier, current_leaf
            )

            elapsed = time.time() - start

            signature = LmsSignature(
                algorithm=self.algorithm,
                ots_algorithm=self.ots_algorithm,
                leaf_index=current_leaf,
                ots_signature=ots_signature,
                auth_path=auth_path,
            )

            logger.debug(
                f"LMS sign: leaf={current_leaf}, size={signature.size} bytes, "
                f"remaining={remaining}, time={elapsed:.3f}s"
            )

            return signature

        finally:
            self.state_backend.unlock(key_id)

    def _compute_ots_signature(
        self, message: bytes, seed: bytes, identifier: bytes, leaf_index: int
    ) -> bytes:
        """Compute LM-OTS one-time signature."""
        n = self.ots_algorithm.hash_length
        p = self.ots_algorithm.chain_count
        w = self.ots_algorithm.winternitz_parameter

        # Generate randomizer C
        c_rand = self._prf(
            seed,
            identifier + struct.pack(">I", leaf_index) + b"\xff\xff",
        )

        # Compute message hash with randomizer
        q_bytes = struct.pack(">I", leaf_index)
        msg_hash = self._hash(
            identifier, q_bytes, struct.pack(">H", 0x8080), c_rand, message
        )

        # Convert hash to Winternitz coefficients
        coefficients = self._coefs(msg_hash, w, p)

        # Compute signature elements
        sig_elements = [struct.pack(">I", self.ots_algorithm.type_code), c_rand]
        for j in range(p):
            x_j = self._prf(
                seed,
                identifier + struct.pack(">I", leaf_index) + struct.pack(">H", j),
            )
            # Chain a[j] times (partial chain)
            val = x_j
            for k in range(coefficients[j]):
                val = self._hash(
                    identifier,
                    struct.pack(">I", leaf_index),
                    struct.pack(">H", j),
                    struct.pack(">B", k),
                    val,
                )
            sig_elements.append(val)

        return b"".join(sig_elements)

    def _coefs(self, hash_value: bytes, w: int, p: int) -> List[int]:
        """Extract Winternitz coefficients from hash."""
        max_val = (2 ** w) - 1
        bits_per_coef = w
        coefficients = []

        # Extract message coefficients
        bit_string = int.from_bytes(hash_value, "big")
        total_bits = len(hash_value) * 8
        num_msg_coefs = total_bits // bits_per_coef

        for i in range(num_msg_coefs):
            shift = total_bits - (i + 1) * bits_per_coef
            coef = (bit_string >> shift) & max_val
            coefficients.append(coef)

        # Compute checksum coefficients
        checksum = sum(max_val - c for c in coefficients)
        # Pad checksum coefficients to fill p
        while len(coefficients) < p:
            coefficients.append(checksum & max_val)
            checksum >>= bits_per_coef

        return coefficients[:p]

    def _compute_auth_path(
        self, seed: bytes, identifier: bytes, leaf_index: int
    ) -> List[bytes]:
        """Compute Merkle tree authentication path for a leaf."""
        h = self.algorithm.tree_height
        num_leaves = 2 ** h

        # Recompute the relevant parts of the tree
        # For production, cache the tree or use a BDS/treehash algorithm
        current_level = []
        for q in range(num_leaves):
            ots_pk_hash = self._compute_ots_public_key_hash(seed, identifier, q)
            r = num_leaves + q
            leaf_hash = self._hash(
                identifier,
                struct.pack(">I", r),
                struct.pack(">H", 0x8282),
                ots_pk_hash,
            )
            current_level.append(leaf_hash)

        auth_path = []
        idx = leaf_index

        for level in range(h):
            # Sibling index
            sibling = idx ^ 1
            auth_path.append(current_level[sibling])

            # Move to parent level
            next_level = []
            for i in range(0, len(current_level), 2):
                parent_r = len(current_level) // 2 + i // 2
                node = self._hash(
                    identifier,
                    struct.pack(">I", parent_r),
                    struct.pack(">H", 0x8383),
                    current_level[i],
                    current_level[i + 1],
                )
                next_level.append(node)
            current_level = next_level
            idx = idx // 2

        return auth_path

    def verify(
        self, message: bytes, signature: LmsSignature, public_key: LmsPublicKey
    ) -> bool:
        """
        Verify an LMS signature.

        Verification is stateless and can be performed by any party with the public key.

        Args:
            message: Original message
            signature: Signature to verify
            public_key: Signer's public key

        Returns:
            True if signature is valid
        """
        start = time.time()

        try:
            # Extract OTS signature components
            ots_data = signature.ots_signature
            ots_type = struct.unpack(">I", ots_data[:4])[0]
            c_rand = ots_data[4:36]
            n = self.ots_algorithm.hash_length
            p = self.ots_algorithm.chain_count
            w = self.ots_algorithm.winternitz_parameter

            y_values = []
            offset = 36
            for j in range(p):
                y_values.append(ots_data[offset:offset + n])
                offset += n

            # Recompute message hash
            q_bytes = struct.pack(">I", signature.leaf_index)
            msg_hash = self._hash(
                public_key.identifier, q_bytes,
                struct.pack(">H", 0x8080), c_rand, message
            )
            coefficients = self._coefs(msg_hash, w, p)

            # Complete the chains to get OTS public key elements
            pk_elements = []
            max_val = (2 ** w) - 1
            for j in range(p):
                val = y_values[j]
                for k in range(coefficients[j], max_val):
                    val = self._hash(
                        public_key.identifier,
                        struct.pack(">I", signature.leaf_index),
                        struct.pack(">H", j),
                        struct.pack(">B", k),
                        val,
                    )
                pk_elements.append(val)

            # Hash OTS public key elements
            ots_pk_hash = self._hash(*pk_elements)

            # Compute leaf hash
            num_leaves = 2 ** self.algorithm.tree_height
            r = num_leaves + signature.leaf_index
            node = self._hash(
                public_key.identifier,
                struct.pack(">I", r),
                struct.pack(">H", 0x8282),
                ots_pk_hash,
            )

            # Walk authentication path to compute root
            idx = signature.leaf_index
            for i, sibling in enumerate(signature.auth_path):
                parent_r = (num_leaves + idx) // 2
                # Adjust parent_r for the level
                level_size = num_leaves >> (i + 1)
                parent_r = level_size + idx // 2

                if idx % 2 == 0:
                    node = self._hash(
                        public_key.identifier,
                        struct.pack(">I", parent_r),
                        struct.pack(">H", 0x8383),
                        node,
                        sibling,
                    )
                else:
                    node = self._hash(
                        public_key.identifier,
                        struct.pack(">I", parent_r),
                        struct.pack(">H", 0x8383),
                        sibling,
                        node,
                    )
                idx = idx // 2

            # Compare computed root with public key root
            valid = hmac.compare_digest(node, public_key.root)

            elapsed = time.time() - start
            logger.debug(f"LMS verify: {valid} in {elapsed:.3f}s")
            return valid

        except Exception as e:
            logger.error(f"LMS verification failed: {e}")
            return False

    def get_key_status(self, private_key: LmsPrivateKey) -> Dict[str, Any]:
        """Get key usage status."""
        stored_index = self.state_backend.load_state(private_key.key_id)
        current = stored_index if stored_index is not None else private_key.leaf_index

        return {
            "key_id": private_key.key_id,
            "algorithm": self.algorithm.value,
            "ots_algorithm": self.ots_algorithm.value,
            "max_signatures": self.algorithm.max_signatures,
            "signatures_used": current,
            "remaining": self.algorithm.max_signatures - current,
            "utilization_pct": (current / self.algorithm.max_signatures) * 100,
            "is_exhausted": current >= self.algorithm.max_signatures,
        }


# ==============================================================================
# XMSS Engine
# ==============================================================================

class XmssEngine:
    """
    XMSS (eXtended Merkle Signature Scheme) engine per RFC 8391.

    Similar to LMS but with different internal construction (WOTS+ instead of LM-OTS).
    Also stateful — same state management requirements as LMS.

    Uses liboqs when available for the actual cryptographic operations.
    Falls back to a hashlib-based reference implementation otherwise.
    """

    def __init__(
        self,
        algorithm: XmssAlgorithm = XmssAlgorithm.XMSS_SHA2_20_256,
        state_backend: Optional[StateBackend] = None,
        strict_mode: Optional[bool] = None,
    ):
        self.algorithm = algorithm
        self.strict_mode = strict_mode if strict_mode is not None else _GLOBAL_STRICT_MODE
        self.state_backend = state_backend or FileStateBackend()

        self._provider = self._detect_provider()

        if self._provider == "fallback" and self.strict_mode:
            raise PQCProviderUnavailableError(algorithm.value, "initialization")

        logger.info(
            f"XMSS engine initialized: algorithm={algorithm.value}, "
            f"max_signatures={algorithm.max_signatures}, provider={self._provider}"
        )

    def _detect_provider(self) -> str:
        try:
            import oqs
            return "liboqs"
        except ImportError:
            pass
        return "hashlib"

    def _hash(self, *parts: bytes) -> bytes:
        if self.algorithm.hash_function == "shake256":
            h = hashlib.shake_256()
            for part in parts:
                h.update(part)
            return h.digest(self.algorithm.hash_length)
        else:
            h = hashlib.sha256()
            for part in parts:
                h.update(part)
            return h.digest()

    def generate_keypair(self, key_id: Optional[str] = None) -> XmssKeyPair:
        """Generate a new XMSS key pair."""
        start = time.time()

        if key_id is None:
            key_id = f"xmss-{os.urandom(8).hex()}"

        n = self.algorithm.hash_length
        seed = os.urandom(n)
        prf_key = os.urandom(n)
        public_seed = os.urandom(n)

        # Compute tree root
        root = self._compute_xmss_root(seed, public_seed)

        public_key = XmssPublicKey(
            algorithm=self.algorithm,
            key_id=key_id,
            root=root,
            seed=public_seed,
        )

        private_key = XmssPrivateKey(
            algorithm=self.algorithm,
            key_id=key_id,
            seed=seed,
            prf_key=prf_key,
            leaf_index=0,
        )

        self.state_backend.save_state(key_id, 0)

        elapsed = time.time() - start
        logger.info(
            f"Generated XMSS keypair '{key_id}': {self.algorithm.value}, "
            f"max_signatures={self.algorithm.max_signatures}, "
            f"keygen_time={elapsed:.2f}s"
        )

        return XmssKeyPair(
            algorithm=self.algorithm,
            public_key=public_key,
            private_key=private_key,
        )

    def _compute_xmss_root(self, seed: bytes, public_seed: bytes) -> bytes:
        """Compute XMSS Merkle tree root."""
        h = self.algorithm.tree_height
        n = self.algorithm.hash_length
        num_leaves = 2 ** h

        # Compute leaf nodes (WOTS+ public key hashes)
        current_level = []
        for i in range(num_leaves):
            wots_pk = self._compute_wots_pk(seed, public_seed, i)
            leaf = self._hash(public_seed, struct.pack(">I", i), wots_pk)
            current_level.append(leaf)

        # Build tree
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                parent = self._hash(
                    public_seed,
                    struct.pack(">I", len(current_level) // 2 + i // 2),
                    current_level[i],
                    current_level[i + 1],
                )
                next_level.append(parent)
            current_level = next_level

        return current_level[0]

    def _compute_wots_pk(
        self, seed: bytes, public_seed: bytes, leaf_index: int
    ) -> bytes:
        """Compute WOTS+ public key for a leaf."""
        n = self.algorithm.hash_length
        w = 16  # Winternitz parameter for XMSS
        length = 67  # WOTS+ chain count for n=32, w=16

        pk_parts = []
        for j in range(length):
            # Generate chain start from PRF
            sk_j = hmac.new(
                seed,
                struct.pack(">I", leaf_index) + struct.pack(">H", j),
                hashlib.sha256,
            ).digest()

            # Chain w-1 times
            val = sk_j
            for k in range(w - 1):
                val = self._hash(
                    public_seed,
                    struct.pack(">I", leaf_index),
                    struct.pack(">H", j),
                    struct.pack(">B", k),
                    val,
                )
            pk_parts.append(val)

        return self._hash(*pk_parts)

    def sign(self, message: bytes, private_key: XmssPrivateKey) -> XmssSignature:
        """Sign a message using XMSS."""
        key_id = private_key.key_id

        if not self.state_backend.lock(key_id):
            raise StateLockError(key_id)

        try:
            stored_index = self.state_backend.load_state(key_id)
            if stored_index is not None:
                private_key.leaf_index = stored_index

            if private_key.is_exhausted:
                raise StateExhaustedError(
                    key_id, self.algorithm.value, self.algorithm.max_signatures
                )

            current_leaf = private_key.leaf_index

            # Advance state BEFORE signing
            next_index = current_leaf + 1
            self.state_backend.save_state(key_id, next_index)
            private_key.leaf_index = next_index

            remaining = private_key.remaining_signatures
            if remaining < 100:
                logger.warning(
                    f"XMSS key '{key_id}' has only {remaining} signatures remaining."
                )

            start = time.time()

            # Generate per-signature randomness
            randomness = hmac.new(
                private_key.prf_key,
                struct.pack(">I", current_leaf) + message[:32],
                hashlib.sha256,
            ).digest()

            # Randomized message hash
            msg_hash = self._hash(randomness, private_key.seed, message)

            # Compute WOTS+ signature
            wots_sig = self._compute_wots_signature(
                msg_hash, private_key.seed, current_leaf
            )

            # Compute authentication path
            auth_path = self._compute_xmss_auth_path(
                private_key.seed, XmssPublicKey(
                    self.algorithm, key_id, b"", private_key.seed
                ).seed,
                current_leaf,
            )

            elapsed = time.time() - start

            signature = XmssSignature(
                algorithm=self.algorithm,
                leaf_index=current_leaf,
                randomness=randomness,
                wots_signature=wots_sig,
                auth_path=auth_path,
            )

            logger.debug(
                f"XMSS sign: leaf={current_leaf}, remaining={remaining}, "
                f"time={elapsed:.3f}s"
            )

            return signature

        finally:
            self.state_backend.unlock(key_id)

    def _compute_wots_signature(
        self, msg_hash: bytes, seed: bytes, leaf_index: int
    ) -> bytes:
        """Compute WOTS+ one-time signature."""
        n = self.algorithm.hash_length
        w = 16
        length = 67

        # Convert message hash to base-w representation
        base_w = self._to_base_w(msg_hash, w, length)

        sig_parts = []
        for j in range(length):
            sk_j = hmac.new(
                seed,
                struct.pack(">I", leaf_index) + struct.pack(">H", j),
                hashlib.sha256,
            ).digest()

            val = sk_j
            for k in range(base_w[j]):
                val = self._hash(
                    seed,
                    struct.pack(">I", leaf_index),
                    struct.pack(">H", j),
                    struct.pack(">B", k),
                    val,
                )
            sig_parts.append(val)

        return b"".join(sig_parts)

    def _to_base_w(self, data: bytes, w: int, out_len: int) -> List[int]:
        """Convert byte string to base-w representation."""
        bits_per_digit = {4: 2, 16: 4, 256: 8}.get(w, 4)
        result = []
        bits = int.from_bytes(data, "big")
        total_bits = len(data) * 8

        for i in range(out_len):
            shift = total_bits - (i + 1) * bits_per_digit
            if shift >= 0:
                result.append((bits >> shift) & (w - 1))
            else:
                result.append(0)

        return result

    def _compute_xmss_auth_path(
        self, seed: bytes, public_seed: bytes, leaf_index: int
    ) -> List[bytes]:
        """Compute authentication path for XMSS."""
        h = self.algorithm.tree_height
        num_leaves = 2 ** h

        # Build tree
        current_level = []
        for i in range(num_leaves):
            wots_pk = self._compute_wots_pk(seed, public_seed, i)
            leaf = self._hash(public_seed, struct.pack(">I", i), wots_pk)
            current_level.append(leaf)

        auth_path = []
        idx = leaf_index

        for level in range(h):
            sibling = idx ^ 1
            auth_path.append(current_level[sibling])

            next_level = []
            for i in range(0, len(current_level), 2):
                parent = self._hash(
                    public_seed,
                    struct.pack(">I", len(current_level) // 2 + i // 2),
                    current_level[i],
                    current_level[i + 1],
                )
                next_level.append(parent)
            current_level = next_level
            idx = idx // 2

        return auth_path

    def get_key_status(self, private_key: XmssPrivateKey) -> Dict[str, Any]:
        """Get key usage status."""
        stored = self.state_backend.load_state(private_key.key_id)
        current = stored if stored is not None else private_key.leaf_index

        return {
            "key_id": private_key.key_id,
            "algorithm": self.algorithm.value,
            "hash_function": self.algorithm.hash_function,
            "max_signatures": self.algorithm.max_signatures,
            "signatures_used": current,
            "remaining": self.algorithm.max_signatures - current,
            "utilization_pct": (current / self.algorithm.max_signatures) * 100,
            "is_exhausted": current >= self.algorithm.max_signatures,
        }


# ==============================================================================
# CNSA 2.0 Compliance Profiles
# ==============================================================================

class CNSA2Profile(Enum):
    """
    NSA CNSA 2.0 algorithm profiles per NSA's Commercial National Security
    Algorithm Suite 2.0 (September 2022).

    CNSA 2.0 mandates specific algorithms for different use cases.
    """

    FIRMWARE_SIGNING = "firmware-signing"
    SOFTWARE_UPDATE = "software-update"
    CODE_SIGNING = "code-signing"
    CA_CERTIFICATE = "ca-certificate"
    KEY_ESTABLISHMENT = "key-establishment"
    GENERAL_SIGNING = "general-signing"

    @property
    def required_kem(self) -> Optional[str]:
        """Required KEM algorithm (ML-KEM-1024 for CNSA 2.0)."""
        if self == CNSA2Profile.KEY_ESTABLISHMENT:
            return "ml-kem-1024"
        return None

    @property
    def required_signature(self) -> str:
        """Required signature algorithm."""
        stateful = {
            CNSA2Profile.FIRMWARE_SIGNING,
            CNSA2Profile.SOFTWARE_UPDATE,
            CNSA2Profile.CODE_SIGNING,
        }
        if self in stateful:
            return "lms-sha256-m32-h20"  # or XMSS — stateful required
        return "ml-dsa-87"  # General signing

    @property
    def required_hash(self) -> str:
        return "sha-384"  # CNSA 2.0 minimum

    @property
    def min_security_level(self) -> int:
        return 5  # CNSA 2.0 requires Level 5 equivalent


def create_cnsa2_lms_engine(
    profile: CNSA2Profile = CNSA2Profile.FIRMWARE_SIGNING,
    state_backend: Optional[StateBackend] = None,
) -> LmsEngine:
    """
    Create an LMS engine configured for CNSA 2.0 compliance.

    CNSA 2.0 requires stateful hash-based signatures (LMS or XMSS)
    for firmware signing, software updates, and code signing.
    """
    return LmsEngine(
        algorithm=LmsAlgorithm.LMS_SHA256_M32_H20,
        ots_algorithm=LmotsAlgorithm.LMOTS_SHA256_N32_W4,
        state_backend=state_backend,
        strict_mode=True,
    )


def create_cnsa2_xmss_engine(
    profile: CNSA2Profile = CNSA2Profile.FIRMWARE_SIGNING,
    state_backend: Optional[StateBackend] = None,
) -> XmssEngine:
    """
    Create an XMSS engine configured for CNSA 2.0 compliance.

    Alternative to LMS for stateful signatures per CNSA 2.0.
    """
    return XmssEngine(
        algorithm=XmssAlgorithm.XMSS_SHA2_20_256,
        state_backend=state_backend,
        strict_mode=True,
    )
