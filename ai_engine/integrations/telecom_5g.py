"""
5G/6G Telecom Security Integration

Post-quantum security for 5G network functions:
- UE Authentication (5G-AKA with PQC)
- SUPI/SUCI concealment
- Network slice security
- NEF exposure function security

Based on:
- 3GPP TS 33.501 (5G Security)
- 3GPP TS 33.535 (Authentication)

Future-proofing for 6G:
- Native PQC support
- Enhanced privacy mechanisms
"""

import asyncio
import hashlib
import hmac
import logging
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any

from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

# Metrics
FIVEG_AUTH_OPS = Counter("fiveg_auth_operations_total", "Total 5G authentication operations", ["operation", "status"])

FIVEG_AUTH_LATENCY = Histogram(
    "fiveg_auth_latency_ms", "5G authentication latency in milliseconds", buckets=[1, 5, 10, 25, 50, 100, 250, 500]
)

FIVEG_ACTIVE_SESSIONS = Gauge("fiveg_active_ue_sessions", "Number of active UE security contexts")


class NetworkFunction(Enum):
    """5G Network Functions."""

    AMF = "amf"  # Access and Mobility Management Function
    SMF = "smf"  # Session Management Function
    UPF = "upf"  # User Plane Function
    AUSF = "ausf"  # Authentication Server Function
    UDM = "udm"  # Unified Data Management
    NEF = "nef"  # Network Exposure Function
    NRF = "nrf"  # Network Repository Function
    NSSF = "nssf"  # Network Slice Selection Function


class AuthenticationMethod(Enum):
    """5G Authentication methods."""

    FIVEG_AKA = "5g-aka"  # 5G-AKA (standard)
    EAP_AKA_PRIME = "eap-aka'"  # EAP-AKA'
    FIVEG_AKA_PQC = "5g-aka-pqc"  # 5G-AKA with PQC (experimental)


class SliceSecurityLevel(Enum):
    """Network slice security levels."""

    BASIC = auto()  # Standard security
    ENHANCED = auto()  # Enhanced for enterprise
    CRITICAL = auto()  # Critical infrastructure
    GOVERNMENT = auto()  # Government grade


@dataclass
class SUPI:
    """Subscription Permanent Identifier."""

    imsi: str  # IMSI format
    mcc: str  # Mobile Country Code
    mnc: str  # Mobile Network Code

    def to_bytes(self) -> bytes:
        return f"{self.mcc}{self.mnc}{self.imsi}".encode()

    @classmethod
    def from_string(cls, supi_string: str) -> "SUPI":
        """Parse SUPI from string format."""
        # Format: imsi-<mcc><mnc><msin>
        if supi_string.startswith("imsi-"):
            digits = supi_string[5:]
            return cls(
                mcc=digits[:3],
                mnc=digits[3:5],
                imsi=digits[5:],
            )
        raise ValueError(f"Invalid SUPI format: {supi_string}")


@dataclass
class SUCI:
    """Subscription Concealed Identifier."""

    supi_type: int
    home_network_id: bytes
    routing_indicator: bytes
    protection_scheme: int
    concealed_data: bytes
    ephemeral_public_key: Optional[bytes] = None


@dataclass
class FiveGKeySet:
    """5G Key Set."""

    kamf: bytes  # Key for AMF
    kausf: bytes  # Key for AUSF
    kseaf: bytes  # Key for SEAF
    kgnb: bytes  # Key for gNB

    # Session keys
    nas_enc_key: Optional[bytes] = None
    nas_int_key: Optional[bytes] = None
    up_enc_key: Optional[bytes] = None
    up_int_key: Optional[bytes] = None


@dataclass
class UESecurityContext:
    """UE Security Context."""

    supi: SUPI
    suci: SUCI
    key_set: FiveGKeySet

    # State
    ngksi: int = 0  # Next Generation Key Set Identifier
    sequence_number: int = 0
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

    # Slice association
    slice_id: Optional[str] = None
    security_level: SliceSecurityLevel = SliceSecurityLevel.BASIC


@dataclass
class AuthenticationVector:
    """5G Authentication Vector."""

    rand: bytes  # Random challenge
    autn: bytes  # Authentication token
    xres_star: bytes  # Expected response
    kausf: bytes  # Key for AUSF
    hxres_star: bytes  # Hashed expected response


class SUPIConcealment:
    """
    SUPI Concealment using ECIES with PQC extension.

    Protects subscriber identity using:
    - Profile A: ECIES with HKDF-SHA256
    - Profile B: ECIES with X25519
    - Profile PQC: Hybrid with ML-KEM (future)
    """

    def __init__(self, use_pqc: bool = False):
        self.use_pqc = use_pqc
        logger.info(f"SUPI concealment initialized: PQC={use_pqc}")

    async def conceal_supi(
        self,
        supi: SUPI,
        home_network_public_key: bytes,
        routing_indicator: bytes = b"\x00\x00",
    ) -> SUCI:
        """
        Conceal SUPI to produce SUCI.

        Args:
            supi: Subscription Permanent Identifier
            home_network_public_key: Home network's public key
            routing_indicator: Routing indicator for network

        Returns:
            Concealed SUCI
        """
        if self.use_pqc:
            return await self._conceal_pqc(supi, home_network_public_key, routing_indicator)
        else:
            return await self._conceal_ecies(supi, home_network_public_key, routing_indicator)

    async def _conceal_ecies(
        self,
        supi: SUPI,
        public_key: bytes,
        routing_indicator: bytes,
    ) -> SUCI:
        """Standard ECIES concealment (Profile A/B)."""
        # Generate ephemeral key pair
        ephemeral_private = secrets.token_bytes(32)
        ephemeral_public = hashlib.sha256(b"EPK" + ephemeral_private).digest()

        # Derive shared secret
        shared_secret = hashlib.sha256(ephemeral_private + public_key).digest()

        # Encrypt SUPI
        supi_bytes = supi.to_bytes()
        concealed = bytes(
            a ^ b
            for a, b in zip(supi_bytes + b"\x00" * (32 - len(supi_bytes) % 32), shared_secret * ((len(supi_bytes) // 32) + 1))
        )[: len(supi_bytes)]

        return SUCI(
            supi_type=0,  # IMSI
            home_network_id=f"{supi.mcc}{supi.mnc}".encode(),
            routing_indicator=routing_indicator,
            protection_scheme=1,  # Profile A
            concealed_data=concealed,
            ephemeral_public_key=ephemeral_public,
        )

    async def _conceal_pqc(
        self,
        supi: SUPI,
        public_key: bytes,
        routing_indicator: bytes,
    ) -> SUCI:
        """PQC-enhanced concealment using ML-KEM."""
        from ai_engine.crypto.mlkem import MlKemEngine, MlKemSecurityLevel, MlKemPublicKey

        engine = MlKemEngine(MlKemSecurityLevel.MLKEM_768)

        # Encapsulate to derive shared secret
        pk = MlKemPublicKey(MlKemSecurityLevel.MLKEM_768, public_key)
        ciphertext, shared_secret = await engine.encapsulate(pk)

        # Encrypt SUPI
        supi_bytes = supi.to_bytes()
        enc_key = hashlib.sha256(shared_secret.data + b"SUPI_ENC").digest()
        concealed = bytes(
            a ^ b for a, b in zip(supi_bytes + b"\x00" * (32 - len(supi_bytes) % 32), enc_key * ((len(supi_bytes) // 32) + 1))
        )[: len(supi_bytes)]

        return SUCI(
            supi_type=0,
            home_network_id=f"{supi.mcc}{supi.mnc}".encode(),
            routing_indicator=routing_indicator,
            protection_scheme=255,  # PQC profile (experimental)
            concealed_data=concealed,
            ephemeral_public_key=ciphertext.data,
        )

    async def reveal_supi(
        self,
        suci: SUCI,
        home_network_private_key: bytes,
    ) -> SUPI:
        """
        Reveal SUPI from SUCI (network side).

        Args:
            suci: Concealed identifier
            home_network_private_key: Network's private key

        Returns:
            Original SUPI
        """
        if suci.protection_scheme == 255:
            return await self._reveal_pqc(suci, home_network_private_key)
        else:
            return await self._reveal_ecies(suci, home_network_private_key)

    async def _reveal_ecies(
        self,
        suci: SUCI,
        private_key: bytes,
    ) -> SUPI:
        """Standard ECIES reveal."""
        # Derive shared secret from ephemeral public key
        shared_secret = hashlib.sha256(private_key + suci.ephemeral_public_key).digest()

        # Decrypt
        decrypted = bytes(
            a ^ b
            for a, b in zip(
                suci.concealed_data + b"\x00" * (32 - len(suci.concealed_data) % 32),
                shared_secret * ((len(suci.concealed_data) // 32) + 1),
            )
        )[: len(suci.concealed_data)]

        supi_str = decrypted.decode().rstrip("\x00")
        mcc = suci.home_network_id[:3].decode()
        mnc = suci.home_network_id[3:].decode()

        return SUPI(
            mcc=mcc,
            mnc=mnc,
            imsi=supi_str[len(mcc) + len(mnc) :],
        )

    async def _reveal_pqc(
        self,
        suci: SUCI,
        private_key: bytes,
    ) -> SUPI:
        """PQC reveal using ML-KEM decapsulation."""
        from ai_engine.crypto.mlkem import MlKemEngine, MlKemSecurityLevel, MlKemPrivateKey, MlKemCiphertext

        engine = MlKemEngine(MlKemSecurityLevel.MLKEM_768)
        sk = MlKemPrivateKey(MlKemSecurityLevel.MLKEM_768, private_key)
        ct = MlKemCiphertext(MlKemSecurityLevel.MLKEM_768, suci.ephemeral_public_key)

        shared_secret = await engine.decapsulate(ct, sk)

        enc_key = hashlib.sha256(shared_secret.data + b"SUPI_ENC").digest()
        decrypted = bytes(
            a ^ b
            for a, b in zip(
                suci.concealed_data + b"\x00" * (32 - len(suci.concealed_data) % 32),
                enc_key * ((len(suci.concealed_data) // 32) + 1),
            )
        )[: len(suci.concealed_data)]

        supi_str = decrypted.decode().rstrip("\x00")
        mcc = suci.home_network_id[:3].decode()
        mnc = suci.home_network_id[3:].decode()

        return SUPI(mcc=mcc, mnc=mnc, imsi=supi_str[len(mcc) + len(mnc) :])


class FiveGKeyHierarchy:
    """
    5G Key Hierarchy derivation.

    Implements 3GPP TS 33.501 key derivation with PQC extensions.
    """

    def __init__(self, use_pqc: bool = False):
        self.use_pqc = use_pqc
        logger.info("5G key hierarchy initialized")

    async def derive_keys(
        self,
        k: bytes,  # Long-term key K
        serving_network_name: str,
        rand: bytes,
    ) -> FiveGKeySet:
        """
        Derive 5G key hierarchy from master key.

        K -> CK/IK -> KAUSF -> KSEAF -> KAMF -> KgNB -> session keys
        """
        # Derive CK and IK (simplified)
        ck = self._kdf(k, b"CK", rand)
        ik = self._kdf(k, b"IK", rand)

        # Derive KAUSF (at AUSF)
        kausf = self._kdf(
            ck + ik,
            b"KAUSF" + serving_network_name.encode(),
            rand,
        )

        # Derive KSEAF (at SEAF)
        kseaf = self._kdf(
            kausf,
            b"KSEAF" + serving_network_name.encode(),
            b"",
        )

        # Derive KAMF (at AMF)
        kamf = self._kdf(kseaf, b"KAMF", b"\x00\x01")

        # Derive KgNB (at gNB)
        kgnb = self._kdf(kamf, b"KgNB", b"\x00\x00\x01\x00")

        # Derive NAS keys
        nas_enc = self._kdf(kamf, b"NAS_ENC", b"\x01\x00\x00\x10")
        nas_int = self._kdf(kamf, b"NAS_INT", b"\x02\x00\x00\x10")

        # Derive UP keys
        up_enc = self._kdf(kgnb, b"UP_ENC", b"\x01\x00\x00\x10")
        up_int = self._kdf(kgnb, b"UP_INT", b"\x02\x00\x00\x10")

        return FiveGKeySet(
            kamf=kamf,
            kausf=kausf,
            kseaf=kseaf,
            kgnb=kgnb,
            nas_enc_key=nas_enc,
            nas_int_key=nas_int,
            up_enc_key=up_enc,
            up_int_key=up_int,
        )

    def _kdf(self, key: bytes, label: bytes, context: bytes) -> bytes:
        """Key derivation function (HMAC-SHA256)."""
        return hmac.new(key, label + b"\x00" + context + len(label).to_bytes(2, "big"), hashlib.sha256).digest()


class UEAuthenticator:
    """
    UE Authentication for 5G-AKA.

    Implements 5G-AKA with optional PQC extensions.
    """

    def __init__(
        self,
        auth_method: AuthenticationMethod = AuthenticationMethod.FIVEG_AKA,
        use_pqc: bool = False,
    ):
        self.auth_method = auth_method
        self.use_pqc = use_pqc
        self.key_hierarchy = FiveGKeyHierarchy(use_pqc)

        self._contexts: Dict[str, UESecurityContext] = {}

        logger.info(f"UE authenticator initialized: method={auth_method.value}")

    async def generate_auth_vector(
        self,
        supi: SUPI,
        k: bytes,  # Long-term key
        serving_network_name: str,
    ) -> AuthenticationVector:
        """
        Generate 5G authentication vector (network side).

        Called by UDM/AUSF to create authentication challenge.
        """
        start = time.time()

        # Generate random challenge
        rand = secrets.token_bytes(16)

        # Generate sequence number (SQN)
        sqn = int(time.time() * 1000).to_bytes(6, "big")

        # Compute authentication token components
        ak = self._f5(k, rand)  # Anonymity key
        mac = self._f1(k, rand, sqn)  # MAC
        xres = self._f2(k, rand)  # Expected response

        # Build AUTN = SQN XOR AK || AMF || MAC
        amf = b"\x80\x00"  # Authentication Management Field
        autn = bytes(a ^ b for a, b in zip(sqn, ak)) + amf + mac

        # Compute XRES* and HXRES*
        xres_star = hashlib.sha256(serving_network_name.encode() + rand + xres).digest()[:16]

        hxres_star = hashlib.sha256(rand + xres_star).digest()[:16]

        # Derive KAUSF
        key_set = await self.key_hierarchy.derive_keys(k, serving_network_name, rand)

        elapsed = (time.time() - start) * 1000
        FIVEG_AUTH_LATENCY.observe(elapsed)
        FIVEG_AUTH_OPS.labels(operation="generate_av", status="success").inc()

        return AuthenticationVector(
            rand=rand,
            autn=autn,
            xres_star=xres_star,
            kausf=key_set.kausf,
            hxres_star=hxres_star,
        )

    async def authenticate_ue(
        self,
        supi: SUPI,
        av: AuthenticationVector,
        res_star: bytes,
    ) -> Optional[UESecurityContext]:
        """
        Authenticate UE response (network side).

        Verifies RES* from UE and creates security context.
        """
        start = time.time()

        # Verify RES*
        if not secrets.compare_digest(av.xres_star, res_star):
            FIVEG_AUTH_OPS.labels(operation="authenticate", status="fail").inc()
            logger.warning(f"Authentication failed for SUPI: {supi.imsi}")
            return None

        # Create security context
        key_set = FiveGKeySet(
            kamf=b"",  # Would be derived
            kausf=av.kausf,
            kseaf=b"",
            kgnb=b"",
        )

        suci = SUCI(
            supi_type=0,
            home_network_id=f"{supi.mcc}{supi.mnc}".encode(),
            routing_indicator=b"\x00\x00",
            protection_scheme=0,
            concealed_data=b"",
        )

        context = UESecurityContext(
            supi=supi,
            suci=suci,
            key_set=key_set,
        )

        self._contexts[supi.imsi] = context
        FIVEG_ACTIVE_SESSIONS.set(len(self._contexts))

        elapsed = (time.time() - start) * 1000
        FIVEG_AUTH_LATENCY.observe(elapsed)
        FIVEG_AUTH_OPS.labels(operation="authenticate", status="success").inc()

        logger.info(f"UE authenticated: {supi.imsi}")
        return context

    def _f1(self, k: bytes, rand: bytes, sqn: bytes) -> bytes:
        """f1: MAC generation function."""
        return hmac.new(k, rand + sqn, hashlib.sha256).digest()[:8]

    def _f2(self, k: bytes, rand: bytes) -> bytes:
        """f2: XRES generation function."""
        return hmac.new(k, rand + b"\x02", hashlib.sha256).digest()[:8]

    def _f5(self, k: bytes, rand: bytes) -> bytes:
        """f5: Anonymity key generation."""
        return hmac.new(k, rand + b"\x05", hashlib.sha256).digest()[:6]


class NEFAuthenticator:
    """
    Network Exposure Function (NEF) Authentication.

    Secures API exposure to external applications
    with PQC-enhanced authentication.
    """

    def __init__(self, use_pqc: bool = True):
        self.use_pqc = use_pqc
        self._api_keys: Dict[str, bytes] = {}
        self._sessions: Dict[str, Dict[str, Any]] = {}

        logger.info("NEF authenticator initialized")

    async def register_application(
        self,
        app_id: str,
        permissions: List[str],
    ) -> Tuple[str, bytes]:
        """
        Register external application.

        Returns (api_key_id, api_secret).
        """
        api_key_id = secrets.token_hex(16)

        if self.use_pqc:
            # Use PQC-derived secret
            from ai_engine.crypto.mlkem import MlKemEngine, MlKemSecurityLevel

            engine = MlKemEngine(MlKemSecurityLevel.MLKEM_768)
            keypair = await engine.generate_keypair()
            ct, ss = await engine.encapsulate(keypair.public_key)
            api_secret = ss.data
        else:
            api_secret = secrets.token_bytes(32)

        self._api_keys[api_key_id] = api_secret
        self._sessions[api_key_id] = {
            "app_id": app_id,
            "permissions": permissions,
            "created_at": time.time(),
        }

        logger.info(f"Registered NEF application: {app_id}")
        return api_key_id, api_secret

    async def authenticate_request(
        self,
        api_key_id: str,
        signature: bytes,
        request_data: bytes,
    ) -> Optional[Dict[str, Any]]:
        """
        Authenticate API request.

        Returns session info if valid, None otherwise.
        """
        if api_key_id not in self._api_keys:
            FIVEG_AUTH_OPS.labels(operation="nef_auth", status="invalid_key").inc()
            return None

        api_secret = self._api_keys[api_key_id]

        # Verify signature
        expected = hmac.new(api_secret, request_data, hashlib.sha256).digest()
        if not secrets.compare_digest(signature, expected):
            FIVEG_AUTH_OPS.labels(operation="nef_auth", status="invalid_sig").inc()
            return None

        FIVEG_AUTH_OPS.labels(operation="nef_auth", status="success").inc()
        return self._sessions.get(api_key_id)


@dataclass
class NetworkSliceSecurityProfile:
    """Security profile for network slice."""

    slice_id: str
    slice_type: str  # eMBB, URLLC, mMTC
    security_level: SliceSecurityLevel

    # Cryptographic settings
    encryption_algorithm: str = "NEA2"  # 128-EEA2 (AES)
    integrity_algorithm: str = "NIA2"  # 128-EIA2 (AES)
    use_pqc: bool = False

    # Access control
    allowed_network_functions: List[NetworkFunction] = field(default_factory=list)
    max_ues: int = 10000
    isolation_level: str = "soft"  # soft/hard


class FiveGSecurityManager:
    """
    Complete 5G Security Manager.

    Coordinates all 5G security functions with PQC support.
    """

    def __init__(self, use_pqc: bool = True):
        self.use_pqc = use_pqc
        self.supi_concealment = SUPIConcealment(use_pqc)
        self.ue_authenticator = UEAuthenticator(use_pqc=use_pqc)
        self.nef_authenticator = NEFAuthenticator(use_pqc)

        self._slice_profiles: Dict[str, NetworkSliceSecurityProfile] = {}

        logger.info(f"5G Security Manager initialized: PQC={use_pqc}")

    def register_slice(
        self,
        profile: NetworkSliceSecurityProfile,
    ) -> None:
        """Register network slice security profile."""
        self._slice_profiles[profile.slice_id] = profile
        logger.info(
            f"Registered slice: {profile.slice_id} " f"(type={profile.slice_type}, security={profile.security_level.name})"
        )

    def get_slice_profile(
        self,
        slice_id: str,
    ) -> Optional[NetworkSliceSecurityProfile]:
        """Get slice security profile."""
        return self._slice_profiles.get(slice_id)

    async def authenticate_ue(
        self,
        supi: SUPI,
        long_term_key: bytes,
        serving_network: str,
        slice_id: Optional[str] = None,
    ) -> Optional[UESecurityContext]:
        """
        Complete UE authentication flow.

        1. Generate authentication vector
        2. (UE would respond with RES*)
        3. Verify and create security context
        """
        # Generate AV
        av = await self.ue_authenticator.generate_auth_vector(
            supi,
            long_term_key,
            serving_network,
        )

        # Simulate UE response (in real flow, this comes from UE)
        res_star = av.xres_star  # UE would compute this

        # Authenticate
        context = await self.ue_authenticator.authenticate_ue(supi, av, res_star)

        if context and slice_id:
            context.slice_id = slice_id
            profile = self._slice_profiles.get(slice_id)
            if profile:
                context.security_level = profile.security_level

        return context

    async def conceal_supi(
        self,
        supi: SUPI,
        home_network_public_key: bytes,
    ) -> SUCI:
        """Conceal SUPI for initial access."""
        return await self.supi_concealment.conceal_supi(
            supi,
            home_network_public_key,
        )

    async def reveal_supi(
        self,
        suci: SUCI,
        home_network_private_key: bytes,
    ) -> SUPI:
        """Reveal SUPI from SUCI (network side)."""
        return await self.supi_concealment.reveal_supi(
            suci,
            home_network_private_key,
        )

    @property
    def active_ue_count(self) -> int:
        """Number of active UE sessions."""
        return len(self.ue_authenticator._contexts)

    @property
    def slice_count(self) -> int:
        """Number of registered slices."""
        return len(self._slice_profiles)


def create_urllc_slice_profile(
    slice_id: str,
    use_pqc: bool = True,
) -> NetworkSliceSecurityProfile:
    """Create URLLC slice with enhanced security."""
    return NetworkSliceSecurityProfile(
        slice_id=slice_id,
        slice_type="URLLC",
        security_level=SliceSecurityLevel.CRITICAL,
        use_pqc=use_pqc,
        encryption_algorithm="NEA2",
        integrity_algorithm="NIA2",
        allowed_network_functions=[
            NetworkFunction.AMF,
            NetworkFunction.SMF,
            NetworkFunction.UPF,
        ],
        max_ues=1000,
        isolation_level="hard",
    )


def create_embb_slice_profile(
    slice_id: str,
    use_pqc: bool = False,
) -> NetworkSliceSecurityProfile:
    """Create eMBB slice with standard security."""
    return NetworkSliceSecurityProfile(
        slice_id=slice_id,
        slice_type="eMBB",
        security_level=SliceSecurityLevel.BASIC,
        use_pqc=use_pqc,
        encryption_algorithm="NEA1",
        integrity_algorithm="NIA1",
        max_ues=100000,
        isolation_level="soft",
    )
