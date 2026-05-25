"""
Unified PQC Crypto Provider Registry

Centralizes provider detection, caching, and preference ordering for all
post-quantum cryptography modules. Replaces per-module _detect_provider()
with a single registry that enforces the hardened preference chain:

    liboqs (C) → pqcrypto (Rust) → pure-Python (dev only) → error

Environment:
    QBITEL_PQC_ALLOW_FALLBACK=1  Allow pure-Python/insecure fallback (testing only)
"""

import logging
import os
import threading
import warnings
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Strict mode prevents silent fallback to insecure random bytes.
_GLOBAL_STRICT_MODE = os.environ.get("QBITEL_PQC_ALLOW_FALLBACK", "0") != "1"


class CryptoProvider(Enum):
    """Available PQC crypto providers."""

    LIBOQS = "liboqs"
    PQCRYPTO = "pqcrypto"
    KYBER_PY = "kyber-py"
    DILITHIUM_PY = "dilithium-py"
    FALLBACK = "fallback"


class ProviderTier(Enum):
    """Security tier classification for providers."""

    PRODUCTION = "production"     # Constant-time C/Rust implementations
    DEV_ONLY = "dev-only"         # Pure-Python, NOT constant-time
    UNSAFE = "unsafe"             # Random bytes, zero cryptographic value


# Map providers to their security tier
PROVIDER_TIERS: Dict[CryptoProvider, ProviderTier] = {
    CryptoProvider.LIBOQS: ProviderTier.PRODUCTION,
    CryptoProvider.PQCRYPTO: ProviderTier.PRODUCTION,
    CryptoProvider.KYBER_PY: ProviderTier.DEV_ONLY,
    CryptoProvider.DILITHIUM_PY: ProviderTier.DEV_ONLY,
    CryptoProvider.FALLBACK: ProviderTier.UNSAFE,
}


class ProviderRegistry:
    """
    Singleton registry for PQC crypto providers.

    Detects available libraries once at first access, caches results,
    and returns providers in the hardened preference order.
    """

    _instance: Optional["ProviderRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ProviderRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._detected = False
                    inst._available: Dict[CryptoProvider, bool] = {}
                    cls._instance = inst
        return cls._instance

    def _detect_all(self) -> None:
        """Detect all available providers (called once)."""
        if self._detected:
            return

        # liboqs (C bindings via liboqs-python)
        try:
            import oqs  # noqa: F401
            self._available[CryptoProvider.LIBOQS] = True
            logger.info("PQC provider detected: liboqs (C bindings) — PRODUCTION tier")
        except ImportError:
            self._available[CryptoProvider.LIBOQS] = False

        # pqcrypto (Rust FFI)
        try:
            import pqcrypto  # noqa: F401
            self._available[CryptoProvider.PQCRYPTO] = True
            logger.info("PQC provider detected: pqcrypto (Rust FFI) — PRODUCTION tier")
        except ImportError:
            self._available[CryptoProvider.PQCRYPTO] = False

        # kyber-py (pure Python KEM)
        try:
            import kyber  # noqa: F401
            self._available[CryptoProvider.KYBER_PY] = True
            logger.info("PQC provider detected: kyber-py (pure Python) — DEV_ONLY tier")
        except ImportError:
            self._available[CryptoProvider.KYBER_PY] = False

        # dilithium-py (pure Python signatures)
        try:
            import dilithium  # noqa: F401
            self._available[CryptoProvider.DILITHIUM_PY] = True
            logger.info("PQC provider detected: dilithium-py (pure Python) — DEV_ONLY tier")
        except ImportError:
            self._available[CryptoProvider.DILITHIUM_PY] = False

        # Fallback is always "available"
        self._available[CryptoProvider.FALLBACK] = True

        self._detected = True

        prod_available = any(
            self._available.get(p, False)
            for p in (CryptoProvider.LIBOQS, CryptoProvider.PQCRYPTO)
        )
        if not prod_available:
            logger.warning(
                "No PRODUCTION-tier PQC provider found. Install 'liboqs-python' or "
                "'pqcrypto' for constant-time implementations suitable for production."
            )

    def _ensure_detected(self) -> None:
        if not self._detected:
            self._detect_all()

    def get_kem_provider(self, strict_mode: Optional[bool] = None) -> str:
        """
        Get the best available KEM provider.

        Preference: liboqs → pqcrypto → kyber-py (dev) → error/fallback

        Returns:
            Provider name string compatible with existing engine code.
        """
        self._ensure_detected()
        strict = strict_mode if strict_mode is not None else _GLOBAL_STRICT_MODE

        if self._available.get(CryptoProvider.LIBOQS):
            return "liboqs"

        if self._available.get(CryptoProvider.PQCRYPTO):
            return "pqcrypto"

        if self._available.get(CryptoProvider.KYBER_PY):
            warnings.warn(
                "Using pure-Python provider 'kyber-py' which is NOT constant-time. "
                "Install 'liboqs-python' or 'pqcrypto' for production use.",
                DeprecationWarning,
                stacklevel=2,
            )
            return "kyber-py"

        if strict:
            from .mlkem import PQCProviderUnavailableError
            raise PQCProviderUnavailableError("ML-KEM", "initialization")

        return "fallback"

    def get_sig_provider(self, algorithm: str = "dilithium", strict_mode: Optional[bool] = None) -> str:
        """
        Get the best available signature provider.

        Preference: liboqs → pqcrypto → dilithium-py (dev, dilithium only) → error/fallback

        Args:
            algorithm: "dilithium" or "falcon"
        """
        self._ensure_detected()
        strict = strict_mode if strict_mode is not None else _GLOBAL_STRICT_MODE

        if self._available.get(CryptoProvider.LIBOQS):
            return "liboqs"

        if self._available.get(CryptoProvider.PQCRYPTO):
            return "pqcrypto"

        if algorithm == "dilithium" and self._available.get(CryptoProvider.DILITHIUM_PY):
            warnings.warn(
                "Using pure-Python provider 'dilithium-py' which is NOT constant-time. "
                "Install 'liboqs-python' or 'pqcrypto' for production use.",
                DeprecationWarning,
                stacklevel=2,
            )
            return "dilithium-py"

        if strict:
            from .mlkem import PQCProviderUnavailableError
            raise PQCProviderUnavailableError(algorithm, "initialization")

        return "fallback"

    def get_provider_tier(self, provider_name: str) -> ProviderTier:
        """Get the security tier for a provider name."""
        name_map = {
            "liboqs": CryptoProvider.LIBOQS,
            "pqcrypto": CryptoProvider.PQCRYPTO,
            "kyber-py": CryptoProvider.KYBER_PY,
            "dilithium-py": CryptoProvider.DILITHIUM_PY,
            "fallback": CryptoProvider.FALLBACK,
        }
        provider = name_map.get(provider_name, CryptoProvider.FALLBACK)
        return PROVIDER_TIERS[provider]

    def is_production_ready(self) -> bool:
        """Check if at least one PRODUCTION-tier provider is available."""
        self._ensure_detected()
        return any(
            self._available.get(p, False)
            for p in (CryptoProvider.LIBOQS, CryptoProvider.PQCRYPTO)
        )

    def available_providers(self) -> List[str]:
        """List all available provider names."""
        self._ensure_detected()
        return [p.value for p, available in self._available.items() if available and p != CryptoProvider.FALLBACK]

    def status(self) -> Dict[str, Dict[str, str]]:
        """Return status of all providers for diagnostics."""
        self._ensure_detected()
        result = {}
        for provider, available in self._available.items():
            if provider == CryptoProvider.FALLBACK:
                continue
            result[provider.value] = {
                "available": str(available),
                "tier": PROVIDER_TIERS[provider].value,
            }
        return result

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing only)."""
        with cls._lock:
            cls._instance = None
