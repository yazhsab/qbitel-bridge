"""
Legacy Fleet Compatibility Module

Handles backward compatibility with legacy vehicles that only
support classical (ECDSA) cryptography during the 15-20 year
fleet turnover period.
"""

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class CompatibilityProfile(Enum):
    """Vehicle compatibility profiles."""

    LEGACY_ONLY = auto()  # Pre-2025: ECDSA only
    HYBRID = auto()  # 2025-2030: Supports both
    PQC_PREFERRED = auto()  # 2030-2035: PQC preferred
    PQC_ONLY = auto()  # Post-2035: PQC only


@dataclass
class VehicleCapabilities:
    """Detected vehicle capabilities."""

    vehicle_id: str
    model_year: int
    supports_pqc: bool
    supports_hybrid: bool
    preferred_algorithm: str
    detected_at: float


class LegacyVehicleAdapter:
    """
    Adapter for communicating with legacy vehicles.

    Automatically downgrades to classical signatures when
    communicating with legacy vehicles.
    """

    def __init__(self, local_vehicle_id: str):
        self.local_vehicle_id = local_vehicle_id
        self._vehicle_cache: Dict[str, VehicleCapabilities] = {}

        logger.info(f"Legacy adapter initialized for {local_vehicle_id}")

    def register_vehicle(self, capabilities: VehicleCapabilities) -> None:
        """Register detected vehicle capabilities."""
        self._vehicle_cache[capabilities.vehicle_id] = capabilities
        logger.debug(f"Registered vehicle {capabilities.vehicle_id}: pqc={capabilities.supports_pqc}")

    def get_signature_mode(self, target_vehicle_id: str) -> str:
        """Determine signature mode for target vehicle."""
        if target_vehicle_id not in self._vehicle_cache:
            # Unknown vehicle - use hybrid for safety
            return "hybrid"

        caps = self._vehicle_cache[target_vehicle_id]

        if caps.supports_pqc:
            return "pqc"
        elif caps.supports_hybrid:
            return "hybrid"
        else:
            return "classical"

    async def prepare_message(
        self,
        content: bytes,
        target_vehicle_id: str,
        pqc_signature: bytes,
        classical_signature: Optional[bytes] = None,
    ) -> bytes:
        """
        Prepare message with appropriate signature(s).

        For legacy vehicles, only includes classical signature.
        For modern vehicles, includes PQC (or both in hybrid mode).
        """
        mode = self.get_signature_mode(target_vehicle_id)

        if mode == "classical" and classical_signature:
            return content + classical_signature
        elif mode == "hybrid" and classical_signature:
            return content + classical_signature + pqc_signature
        else:
            return content + pqc_signature


class HybridModeManager:
    """
    Manages hybrid classical/PQC operation during transition.

    Handles:
    - Dual signature generation
    - Certificate chain with both key types
    - Automatic mode detection
    """

    def __init__(self):
        self._mode = CompatibilityProfile.HYBRID
        logger.info("Hybrid mode manager initialized")

    def set_mode(self, mode: CompatibilityProfile) -> None:
        """Set the operating mode."""
        self._mode = mode
        logger.info(f"Operating mode set to {mode.name}")

    def should_include_classical(self) -> bool:
        """Check if classical signature should be included."""
        return self._mode in (CompatibilityProfile.LEGACY_ONLY, CompatibilityProfile.HYBRID)

    def should_include_pqc(self) -> bool:
        """Check if PQC signature should be included."""
        return self._mode in (CompatibilityProfile.HYBRID, CompatibilityProfile.PQC_PREFERRED, CompatibilityProfile.PQC_ONLY)

    async def dual_sign(
        self,
        message: bytes,
        classical_private_key: bytes,
        pqc_private_key: bytes,
    ) -> Dict[str, bytes]:
        """Generate both classical and PQC signatures."""
        signatures = {}

        if self.should_include_classical():
            # Would use ECDSA here
            signatures["classical"] = b""  # Placeholder

        if self.should_include_pqc():
            from ai_engine.crypto.falcon import FalconEngine, FalconPrivateKey

            engine = FalconEngine()
            sk = FalconPrivateKey(engine.level, pqc_private_key)
            sig = await engine.sign(message, sk)
            signatures["pqc"] = sig.data

        return signatures
