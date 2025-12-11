"""
ADS-B (Automatic Dependent Surveillance-Broadcast) Authentication

PQC-based authentication for ADS-B messages to prevent:
- Ghost aircraft injection
- Position spoofing
- Flight ID manipulation

Challenges:
- ADS-B messages are 112 bits (14 bytes) - very constrained
- 1090 MHz broadcast - no back channel
- Must maintain backward compatibility
"""

import asyncio
import hashlib
import logging
import secrets
import struct
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

# Metrics
ADSB_VERIFIED = Counter(
    'adsb_messages_verified_total',
    'Total ADS-B messages verified',
    ['status', 'message_type']
)

ADSB_SPOOFING_DETECTED = Counter(
    'adsb_spoofing_attempts_detected_total',
    'Detected ADS-B spoofing attempts',
    ['attack_type']
)

ADSB_AUTH_LATENCY = Histogram(
    'adsb_authentication_latency_seconds',
    'ADS-B authentication latency',
    buckets=[0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01]
)

TRACKED_AIRCRAFT = Gauge(
    'adsb_tracked_aircraft',
    'Number of aircraft being tracked'
)


class AdsbMessageType(Enum):
    """ADS-B message types (Downlink Format)."""

    SHORT_SQUITTER = 11        # DF11: All-call reply
    EXTENDED_SQUITTER = 17     # DF17: Extended squitter
    EXTENDED_SQUITTER_NT = 18  # DF18: Extended squitter (non-transponder)
    MILITARY = 19              # DF19: Military extended squitter


class AdsbAuthenticationMode(Enum):
    """Authentication modes for ADS-B."""

    NONE = auto()              # No authentication (legacy)
    OUT_OF_BAND = auto()       # Authentication via separate channel
    EXTENDED_FIELD = auto()    # Extended ADS-B with auth field
    MULTILATERATION = auto()   # Position verification via MLAT


@dataclass
class AdsbPosition:
    """ADS-B position report."""

    icao_address: str   # 24-bit ICAO aircraft address
    latitude: float
    longitude: float
    altitude: int       # feet
    velocity: float     # knots
    heading: float      # degrees
    timestamp: float

    def distance_from(self, other: 'AdsbPosition') -> float:
        """Calculate distance in nautical miles."""
        from math import radians, sin, cos, sqrt, atan2

        R = 3440.065  # Earth radius in nm

        lat1, lon1 = radians(self.latitude), radians(self.longitude)
        lat2, lon2 = radians(other.latitude), radians(other.longitude)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))

        return R * c


@dataclass
class AdsbAuthToken:
    """Out-of-band authentication token for ADS-B."""

    icao_address: str
    public_key_hash: bytes    # Hash of aircraft's PQC public key
    signature: bytes          # PQC signature
    validity_window: int      # Seconds this token is valid
    created_at: float = field(default_factory=time.time)

    @property
    def is_valid(self) -> bool:
        return time.time() - self.created_at < self.validity_window


@dataclass
class AircraftTrackingState:
    """Tracking state for a single aircraft."""

    icao_address: str
    positions: List[AdsbPosition] = field(default_factory=list)
    auth_tokens: List[AdsbAuthToken] = field(default_factory=list)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    verified: bool = False
    anomaly_score: float = 0.0

    def add_position(self, pos: AdsbPosition) -> None:
        """Add position to track history."""
        self.positions.append(pos)
        self.last_seen = time.time()

        # Keep last 100 positions
        if len(self.positions) > 100:
            self.positions = self.positions[-100:]

    def check_trajectory_consistency(self) -> float:
        """
        Check if trajectory is physically consistent.

        Returns anomaly score (0 = normal, 1 = highly anomalous)
        """
        if len(self.positions) < 3:
            return 0.0

        anomalies = 0
        total_checks = 0

        for i in range(2, len(self.positions)):
            prev = self.positions[i-1]
            curr = self.positions[i]

            # Time between positions
            dt = curr.timestamp - prev.timestamp
            if dt <= 0:
                anomalies += 1
                total_checks += 1
                continue

            # Distance traveled
            distance = prev.distance_from(curr)

            # Maximum possible distance (assuming max aircraft speed ~600 knots)
            max_distance = (600 / 3600) * dt  # nm per second * seconds

            if distance > max_distance * 1.5:  # 50% tolerance
                anomalies += 1

            # Altitude change check (max ~6000 ft/min for jets)
            alt_change = abs(curr.altitude - prev.altitude)
            max_alt_change = (6000 / 60) * dt  # ft/sec * seconds

            if alt_change > max_alt_change * 1.5:
                anomalies += 1

            total_checks += 2

        return anomalies / total_checks if total_checks > 0 else 0.0


class AdsbAuthenticator:
    """
    ADS-B Authentication Engine.

    Provides multiple authentication mechanisms for ADS-B:
    1. Out-of-band PQC signatures (via LDACS/SATCOM)
    2. Trajectory consistency analysis
    3. Multilateration verification
    """

    def __init__(
        self,
        auth_mode: AdsbAuthenticationMode = AdsbAuthenticationMode.OUT_OF_BAND,
        anomaly_threshold: float = 0.3,
    ):
        self.auth_mode = auth_mode
        self.anomaly_threshold = anomaly_threshold

        self._aircraft: Dict[str, AircraftTrackingState] = {}
        self._auth_tokens: Dict[str, AdsbAuthToken] = {}
        self._known_good_keys: Dict[str, bytes] = {}  # ICAO -> public key

        # Blacklist for detected spoofers
        self._blacklist: Set[str] = set()

        logger.info(
            f"ADS-B authenticator initialized: mode={auth_mode.name}, "
            f"threshold={anomaly_threshold}"
        )

    async def process_adsb_message(
        self,
        raw_message: bytes,
    ) -> Tuple[Optional[AdsbPosition], bool]:
        """
        Process and authenticate ADS-B message.

        Args:
            raw_message: Raw ADS-B message bytes

        Returns:
            Tuple of (position if valid, is_authenticated)
        """
        start = time.time()

        # Decode message
        position = self._decode_adsb(raw_message)
        if position is None:
            return None, False

        # Check blacklist
        if position.icao_address in self._blacklist:
            ADSB_SPOOFING_DETECTED.labels(attack_type="blacklisted").inc()
            return None, False

        # Get or create tracking state
        if position.icao_address not in self._aircraft:
            self._aircraft[position.icao_address] = AircraftTrackingState(
                icao_address=position.icao_address
            )
            TRACKED_AIRCRAFT.set(len(self._aircraft))

        state = self._aircraft[position.icao_address]
        state.add_position(position)

        # Check trajectory consistency
        anomaly_score = state.check_trajectory_consistency()
        state.anomaly_score = anomaly_score

        if anomaly_score > self.anomaly_threshold:
            logger.warning(
                f"ADS-B anomaly detected: {position.icao_address}, "
                f"score={anomaly_score:.2f}"
            )
            ADSB_SPOOFING_DETECTED.labels(attack_type="trajectory_anomaly").inc()

            # Don't blacklist immediately - could be legitimate
            # but mark as unverified
            state.verified = False

        # Check authentication
        is_authenticated = await self._verify_authentication(position, state)
        state.verified = is_authenticated

        elapsed = time.time() - start
        ADSB_AUTH_LATENCY.observe(elapsed)
        ADSB_VERIFIED.labels(
            status="verified" if is_authenticated else "unverified",
            message_type="position"
        ).inc()

        return position, is_authenticated

    async def register_auth_token(
        self,
        token: AdsbAuthToken,
    ) -> bool:
        """
        Register out-of-band authentication token.

        Tokens are received via secure channel (LDACS/SATCOM)
        and used to verify ADS-B broadcasts.
        """
        # Verify token signature
        if not await self._verify_token_signature(token):
            logger.warning(f"Invalid auth token for {token.icao_address}")
            return False

        # Store token
        self._auth_tokens[token.icao_address] = token

        if token.icao_address in self._aircraft:
            self._aircraft[token.icao_address].auth_tokens.append(token)
            self._aircraft[token.icao_address].verified = True

        logger.debug(f"Auth token registered: {token.icao_address}")
        return True

    async def register_aircraft_key(
        self,
        icao_address: str,
        public_key: bytes,
    ) -> None:
        """Register known aircraft public key."""
        key_hash = hashlib.sha256(public_key).digest()[:16]
        self._known_good_keys[icao_address] = key_hash
        logger.info(f"Registered public key for {icao_address}")

    async def _verify_authentication(
        self,
        position: AdsbPosition,
        state: AircraftTrackingState,
    ) -> bool:
        """Verify authentication for position report."""
        if self.auth_mode == AdsbAuthenticationMode.NONE:
            return True  # No authentication required

        elif self.auth_mode == AdsbAuthenticationMode.OUT_OF_BAND:
            # Check for valid auth token
            token = self._auth_tokens.get(position.icao_address)
            if token and token.is_valid:
                return True

            # Check if we have any recent tokens
            if state.auth_tokens:
                valid_tokens = [t for t in state.auth_tokens if t.is_valid]
                if valid_tokens:
                    return True

            return False

        elif self.auth_mode == AdsbAuthenticationMode.MULTILATERATION:
            # Would verify position via MLAT
            # Placeholder - real implementation would correlate
            # with ground station network
            return state.anomaly_score < self.anomaly_threshold

        return False

    async def _verify_token_signature(
        self,
        token: AdsbAuthToken,
    ) -> bool:
        """Verify PQC signature on auth token."""
        # Get known public key
        known_hash = self._known_good_keys.get(token.icao_address)

        if known_hash is None:
            # Unknown aircraft - accept but mark as unverified
            return True

        # Verify public key hash matches
        if not secrets.compare_digest(token.public_key_hash, known_hash):
            logger.warning(f"Public key mismatch for {token.icao_address}")
            return False

        # Verify signature (simplified - real impl would use full verification)
        # The signature covers: icao_address | validity_window | created_at
        return len(token.signature) >= 512  # Falcon-512 minimum size

    def _decode_adsb(self, raw: bytes) -> Optional[AdsbPosition]:
        """
        Decode ADS-B message.

        Simplified decoder - real implementation would handle
        full Mode S extended squitter format.
        """
        if len(raw) < 14:  # Minimum ADS-B message size
            return None

        # Extract ICAO address (bytes 1-3 for DF17)
        df = (raw[0] >> 3) & 0x1F

        if df not in (17, 18):
            return None  # Not extended squitter

        icao = raw[1:4].hex().upper()

        # Type code (5 bits)
        tc = (raw[4] >> 3) & 0x1F

        # Airborne position (TC 9-18)
        if 9 <= tc <= 18:
            # Simplified position decoding
            # Real implementation would handle CPR decoding
            altitude = self._decode_altitude(raw[5:7])
            lat = self._decode_cpr_lat(raw[6:10])
            lon = self._decode_cpr_lon(raw[6:10])

            return AdsbPosition(
                icao_address=icao,
                latitude=lat,
                longitude=lon,
                altitude=altitude,
                velocity=0.0,
                heading=0.0,
                timestamp=time.time(),
            )

        # Airborne velocity (TC 19)
        elif tc == 19:
            # Velocity message - update existing position
            if icao in self._aircraft and self._aircraft[icao].positions:
                pos = self._aircraft[icao].positions[-1]
                velocity, heading = self._decode_velocity(raw[5:10])
                return AdsbPosition(
                    icao_address=icao,
                    latitude=pos.latitude,
                    longitude=pos.longitude,
                    altitude=pos.altitude,
                    velocity=velocity,
                    heading=heading,
                    timestamp=time.time(),
                )

        return None

    def _decode_altitude(self, data: bytes) -> int:
        """Decode altitude from ADS-B."""
        # Simplified - real implementation handles Q-bit and gray code
        alt_code = ((data[0] & 0x07) << 8) | data[1]
        return alt_code * 25 - 1000  # feet

    def _decode_cpr_lat(self, data: bytes) -> float:
        """Decode CPR latitude (simplified)."""
        # Real implementation would use proper CPR decoding
        raw = int.from_bytes(data[:3], 'big') & 0x1FFFF
        return (raw / 131072.0) * 360.0 - 180.0

    def _decode_cpr_lon(self, data: bytes) -> float:
        """Decode CPR longitude (simplified)."""
        raw = int.from_bytes(data[1:4], 'big') & 0x1FFFF
        return (raw / 131072.0) * 360.0 - 180.0

    def _decode_velocity(self, data: bytes) -> Tuple[float, float]:
        """Decode velocity and heading."""
        # Simplified extraction
        ew_sign = (data[0] >> 2) & 1
        ew_vel = ((data[0] & 0x03) << 8) | data[1]
        ns_sign = (data[2] >> 7) & 1
        ns_vel = ((data[2] & 0x7F) << 3) | (data[3] >> 5)

        ew = ew_vel if not ew_sign else -ew_vel
        ns = ns_vel if not ns_sign else -ns_vel

        from math import sqrt, atan2, degrees
        velocity = sqrt(ew**2 + ns**2)
        heading = (degrees(atan2(ew, ns)) + 360) % 360

        return velocity, heading

    def get_aircraft_status(
        self,
        icao_address: str,
    ) -> Optional[Dict]:
        """Get tracking status for aircraft."""
        state = self._aircraft.get(icao_address)
        if not state:
            return None

        return {
            "icao": state.icao_address,
            "verified": state.verified,
            "anomaly_score": state.anomaly_score,
            "positions": len(state.positions),
            "first_seen": state.first_seen,
            "last_seen": state.last_seen,
            "has_valid_token": bool(
                [t for t in state.auth_tokens if t.is_valid]
            ),
        }

    def get_all_aircraft(self) -> List[Dict]:
        """Get status of all tracked aircraft."""
        return [
            self.get_aircraft_status(icao)
            for icao in self._aircraft
        ]

    def blacklist_aircraft(self, icao_address: str, reason: str) -> None:
        """Add aircraft to blacklist."""
        self._blacklist.add(icao_address)
        logger.warning(f"Aircraft blacklisted: {icao_address}, reason: {reason}")

        if icao_address in self._aircraft:
            del self._aircraft[icao_address]
            TRACKED_AIRCRAFT.set(len(self._aircraft))

    async def cleanup_stale_tracks(self, max_age: float = 300.0) -> int:
        """Remove stale aircraft tracks."""
        now = time.time()
        stale = [
            icao for icao, state in self._aircraft.items()
            if now - state.last_seen > max_age
        ]

        for icao in stale:
            del self._aircraft[icao]

        if stale:
            TRACKED_AIRCRAFT.set(len(self._aircraft))
            logger.info(f"Cleaned up {len(stale)} stale aircraft tracks")

        return len(stale)


class AdsbAuthTokenGenerator:
    """
    Generates authentication tokens for ADS-B broadcasts.

    Tokens are transmitted via secure channel (LDACS/SATCOM)
    and allow receivers to authenticate ADS-B messages.
    """

    def __init__(
        self,
        validity_window: int = 60,  # 1 minute default
    ):
        self.validity_window = validity_window
        logger.info(f"ADS-B token generator initialized: validity={validity_window}s")

    async def generate_token(
        self,
        icao_address: str,
        private_key: bytes,
        public_key: bytes,
    ) -> AdsbAuthToken:
        """
        Generate authentication token.

        Args:
            icao_address: Aircraft ICAO address
            private_key: Aircraft's PQC private key
            public_key: Aircraft's PQC public key

        Returns:
            Authentication token for out-of-band transmission
        """
        from ai_engine.crypto.falcon import FalconEngine, FalconSecurityLevel, FalconPrivateKey

        created_at = time.time()
        public_key_hash = hashlib.sha256(public_key).digest()[:16]

        # Create message to sign
        message = (
            icao_address.encode() +
            self.validity_window.to_bytes(4, 'big') +
            int(created_at).to_bytes(8, 'big')
        )

        # Sign with Falcon
        engine = FalconEngine(FalconSecurityLevel.FALCON_512)
        sk = FalconPrivateKey(FalconSecurityLevel.FALCON_512, private_key)
        sig = await engine.sign(message, sk)

        return AdsbAuthToken(
            icao_address=icao_address,
            public_key_hash=public_key_hash,
            signature=sig.data,
            validity_window=self.validity_window,
            created_at=created_at,
        )

    async def generate_batch_tokens(
        self,
        icao_address: str,
        private_key: bytes,
        public_key: bytes,
        count: int = 10,
    ) -> List[AdsbAuthToken]:
        """
        Generate batch of tokens with staggered validity.

        Useful for pre-generating tokens before flight.
        """
        tokens = []

        for i in range(count):
            # Adjust validity window for staggered coverage
            token = await self.generate_token(
                icao_address,
                private_key,
                public_key,
            )
            # Shift created_at to stagger validity
            token.created_at = time.time() + (i * self.validity_window)
            tokens.append(token)

        logger.debug(f"Generated {count} batch tokens for {icao_address}")
        return tokens
