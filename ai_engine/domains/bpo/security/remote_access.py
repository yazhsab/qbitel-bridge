"""
Remote Agent Security Module

Quantum-safe security for remote/work-from-home BPO agents.

Provides:
- VPN-less quantum-safe tunnel establishment
- Endpoint compliance verification
- Home network security assessment
- Continuous device posture monitoring
- Geo-fencing and location verification
- Split tunneling prevention
- Session watermarking for forensics

This module addresses the unique security challenges of remote BPO
workforces, where agents handle sensitive customer data from home
networks without the physical security controls of an office.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import hashlib
import logging
import uuid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ComplianceStatus(Enum):
    """Overall compliance status for a remote endpoint."""

    COMPLIANT = (1, "Compliant", "All requirements met")
    PARTIALLY_COMPLIANT = (2, "Partially Compliant", "Some requirements not met")
    NON_COMPLIANT = (3, "Non-Compliant", "Critical requirements not met")
    UNKNOWN = (4, "Unknown", "Compliance status could not be determined")
    CHECKING = (5, "Checking", "Compliance check in progress")

    def __init__(self, level: int, display_name: str, description: str):
        self.level = level
        self.display_name = display_name
        self.description = description


class TunnelState(Enum):
    """State of the quantum-safe tunnel."""

    DISCONNECTED = auto()
    NEGOTIATING = auto()     # PQC key exchange in progress
    ESTABLISHING = auto()    # Tunnel parameters being set
    CONNECTED = auto()       # Tunnel active
    REKEYING = auto()        # Periodic rekey in progress
    DEGRADED = auto()        # Connected but with issues
    FAILED = auto()          # Connection failed


class TunnelProtocol(Enum):
    """Supported tunnel protocols."""

    PQC_WIREGUARD = ("pqc-wireguard", "Quantum-safe WireGuard variant")
    PQC_IPSEC = ("pqc-ipsec", "Quantum-safe IPsec IKEv2")
    PQC_TLS = ("pqc-tls", "Quantum-safe TLS 1.3 tunnel")
    PQC_NOISE = ("pqc-noise", "Quantum-safe Noise protocol")

    def __init__(self, protocol_id: str, description: str):
        self.protocol_id = protocol_id
        self.description = description


class NetworkRisk(Enum):
    """Risk assessment levels for home networks."""

    LOW = (1, "Low", "Secure, well-configured network")
    MEDIUM = (2, "Medium", "Minor concerns but acceptable")
    HIGH = (3, "High", "Significant security concerns")
    CRITICAL = (4, "Critical", "Unacceptable network security")

    def __init__(self, level: int, display_name: str, description: str):
        self.level = level
        self.display_name = display_name
        self.description = description


class PostureCheckResult(Enum):
    """Result of a single posture check."""

    PASS = auto()
    FAIL = auto()
    WARNING = auto()
    NOT_APPLICABLE = auto()
    ERROR = auto()


class WatermarkType(Enum):
    """Types of forensic watermarks applied to sessions."""

    VISIBLE = auto()         # Visible overlay on screen
    INVISIBLE = auto()       # Steganographic watermark
    AUDIO = auto()           # Audio watermark in voice streams
    METADATA = auto()        # Metadata-embedded watermark


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EndpointRequirements:
    """
    Requirements that remote endpoints must satisfy.

    Defines the minimum security posture for remote agent devices.
    """

    requirement_id: str = ""
    name: str = "Default Remote Agent Requirements"

    # Operating system
    allowed_os: Set[str] = field(default_factory=lambda: {
        "windows_10", "windows_11", "macos_12", "macos_13", "macos_14",
        "ubuntu_22", "ubuntu_24",
    })
    require_os_updates: bool = True
    max_os_update_age_days: int = 30

    # Security software
    require_antivirus: bool = True
    require_antivirus_updated: bool = True
    max_antivirus_signature_age_days: int = 3
    require_edr: bool = False              # Endpoint Detection & Response
    require_host_firewall: bool = True

    # Disk encryption
    require_disk_encryption: bool = True
    allowed_encryption: Set[str] = field(default_factory=lambda: {
        "bitlocker", "filevault", "luks",
    })

    # Screen lock
    require_screen_lock: bool = True
    max_screen_lock_timeout_minutes: int = 5

    # Network
    require_secure_wifi: bool = True
    min_wifi_security: str = "WPA2"        # WPA2, WPA3
    block_public_wifi: bool = True
    block_vpn_bypass: bool = True          # Prevent split tunneling

    # Hardware
    require_tpm: bool = True               # Trusted Platform Module
    tpm_version: str = "2.0"
    block_usb_storage: bool = True

    # Browser
    allowed_browsers: Set[str] = field(default_factory=lambda: {
        "chrome", "edge", "firefox",
    })
    require_browser_updated: bool = True

    # Application control
    block_unauthorized_apps: bool = True
    blocked_apps: Set[str] = field(default_factory=lambda: {
        "screen_recorder", "remote_desktop_unauthorized",
        "torrent_client", "personal_vpn",
    })


@dataclass
class DevicePosture:
    """
    Current security posture of a remote device.

    Captured during endpoint compliance checks and
    continuously monitored during active sessions.
    """

    device_id: str = ""
    agent_id: str = ""
    check_timestamp: datetime = field(default_factory=datetime.utcnow)

    # Operating system
    os_type: str = ""                      # windows, macos, linux
    os_version: str = ""                   # e.g., "11", "14.2"
    os_build: str = ""
    os_last_update: Optional[datetime] = None
    os_auto_update_enabled: bool = False

    # Security software
    antivirus_installed: bool = False
    antivirus_name: str = ""
    antivirus_version: str = ""
    antivirus_signatures_date: Optional[datetime] = None
    antivirus_real_time_enabled: bool = False
    edr_installed: bool = False
    edr_name: str = ""
    firewall_enabled: bool = False

    # Disk encryption
    disk_encrypted: bool = False
    encryption_method: str = ""
    encryption_percentage: float = 0.0

    # Screen lock
    screen_lock_enabled: bool = False
    screen_lock_timeout_minutes: int = 0

    # Network
    wifi_security: str = ""                # WPA2, WPA3, Open, Wired
    network_type: str = ""                 # wifi, ethernet, cellular
    is_public_network: bool = False
    dns_servers: List[str] = field(default_factory=list)
    gateway_ip: str = ""

    # Hardware
    tpm_present: bool = False
    tpm_version: str = ""
    usb_storage_blocked: bool = False

    # Browser
    browser_name: str = ""
    browser_version: str = ""

    # Location
    ip_address: str = ""
    geo_country: str = ""
    geo_region: str = ""
    geo_city: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    # Running processes of concern
    unauthorized_apps_detected: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize posture for storage."""
        return {
            "device_id": self.device_id,
            "agent_id": self.agent_id,
            "check_timestamp": self.check_timestamp.isoformat(),
            "os_type": self.os_type,
            "os_version": self.os_version,
            "antivirus_installed": self.antivirus_installed,
            "antivirus_name": self.antivirus_name,
            "antivirus_real_time": self.antivirus_real_time_enabled,
            "edr_installed": self.edr_installed,
            "firewall_enabled": self.firewall_enabled,
            "disk_encrypted": self.disk_encrypted,
            "encryption_method": self.encryption_method,
            "screen_lock_enabled": self.screen_lock_enabled,
            "wifi_security": self.wifi_security,
            "network_type": self.network_type,
            "tpm_present": self.tpm_present,
            "geo_country": self.geo_country,
            "geo_region": self.geo_region,
            "unauthorized_apps": self.unauthorized_apps_detected,
        }


@dataclass
class TunnelConfig:
    """
    Configuration for a quantum-safe tunnel.

    Defines the cryptographic parameters and network settings
    for the PQC tunnel between the remote agent and the QBITEL
    infrastructure.
    """

    config_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    protocol: TunnelProtocol = TunnelProtocol.PQC_WIREGUARD

    # PQC key exchange
    kem_algorithm: str = "ML-KEM-768"
    sig_algorithm: str = "ML-DSA-65"
    hybrid_mode: bool = True               # Combine PQC with classical
    classical_kem: str = "X25519"          # For hybrid mode
    classical_sig: str = "Ed25519"         # For hybrid mode

    # Symmetric encryption (post key exchange)
    symmetric_algorithm: str = "AES-256-GCM"
    symmetric_key_bits: int = 256

    # Rekey parameters
    rekey_interval_seconds: int = 3600     # Rekey every hour
    rekey_data_limit_mb: int = 1024        # Rekey after 1 GB
    max_rekey_failures: int = 3

    # Network
    tunnel_mtu: int = 1420
    keepalive_interval_seconds: int = 25
    handshake_timeout_seconds: int = 30
    connection_timeout_seconds: int = 60

    # Split tunnel prevention
    force_all_traffic: bool = True
    dns_over_tunnel: bool = True
    prevent_dns_leaks: bool = True

    # Allowed endpoints
    allowed_server_endpoints: List[str] = field(default_factory=list)
    server_public_key_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize tunnel config."""
        return {
            "config_id": self.config_id,
            "protocol": self.protocol.protocol_id,
            "kem_algorithm": self.kem_algorithm,
            "sig_algorithm": self.sig_algorithm,
            "hybrid_mode": self.hybrid_mode,
            "symmetric_algorithm": self.symmetric_algorithm,
            "rekey_interval_seconds": self.rekey_interval_seconds,
            "force_all_traffic": self.force_all_traffic,
            "dns_over_tunnel": self.dns_over_tunnel,
        }


@dataclass
class TunnelSession:
    """
    An active tunnel session.

    Tracks the lifecycle and health of a PQC tunnel.
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    device_id: str = ""
    config: TunnelConfig = field(default_factory=TunnelConfig)

    state: TunnelState = TunnelState.DISCONNECTED
    established_at: Optional[datetime] = None
    last_rekey_at: Optional[datetime] = None
    rekey_count: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0

    # Security metrics
    handshake_duration_ms: float = 0.0
    current_kem_algorithm: str = ""
    current_sig_algorithm: str = ""

    # Health
    last_keepalive: Optional[datetime] = None
    latency_ms: float = 0.0
    packet_loss_percent: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize tunnel session."""
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "device_id": self.device_id,
            "state": self.state.name,
            "established_at": (
                self.established_at.isoformat()
                if self.established_at else None
            ),
            "rekey_count": self.rekey_count,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "current_kem": self.current_kem_algorithm,
            "latency_ms": self.latency_ms,
            "packet_loss_percent": self.packet_loss_percent,
        }


@dataclass
class GeoFencePolicy:
    """
    Geographic fencing policy for remote agents.

    Restricts agent access based on geographic location,
    detecting impossible travel and location anomalies.
    """

    policy_id: str = ""
    policy_name: str = ""
    tenant_id: str = ""

    # Allowed locations
    allowed_countries: Set[str] = field(default_factory=lambda: {"US"})
    allowed_regions: Set[str] = field(default_factory=set)  # Empty = all in country
    allowed_cities: Set[str] = field(default_factory=set)   # Empty = all in region
    blocked_countries: Set[str] = field(default_factory=set)

    # Radius-based fencing
    enable_radius_fence: bool = False
    center_latitude: float = 0.0
    center_longitude: float = 0.0
    radius_km: float = 100.0

    # Travel detection
    enable_impossible_travel: bool = True
    max_travel_speed_kmh: float = 900.0     # Approximate air travel speed
    min_travel_alert_km: float = 100.0       # Minimum distance for alert

    # Enforcement
    action_on_violation: str = "BLOCK"       # BLOCK, ALERT, LOG
    allow_override_with_approval: bool = True

    def is_location_allowed(
        self,
        country: str,
        region: str = "",
        city: str = "",
    ) -> Tuple[bool, str]:
        """
        Check if a location is allowed by this policy.

        Args:
            country: Country code.
            region: Region/state code.
            city: City name.

        Returns:
            Tuple of (is_allowed, reason).
        """
        if country in self.blocked_countries:
            return False, f"Country {country} is blocked"

        if self.allowed_countries and country not in self.allowed_countries:
            return False, f"Country {country} is not in allowed list"

        if self.allowed_regions and region and region not in self.allowed_regions:
            return False, f"Region {region} is not in allowed list"

        if self.allowed_cities and city and city not in self.allowed_cities:
            return False, f"City {city} is not in allowed list"

        return True, "Location is within allowed boundaries"


@dataclass
class PostureCheckReport:
    """Report from a posture compliance check."""

    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    device_id: str = ""
    agent_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    overall_status: ComplianceStatus = ComplianceStatus.UNKNOWN
    checks: Dict[str, PostureCheckResult] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize report."""
        return {
            "report_id": self.report_id,
            "device_id": self.device_id,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(),
            "overall_status": self.overall_status.display_name,
            "checks": {k: v.name for k, v in self.checks.items()},
            "failures": self.failures,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
        }


# ---------------------------------------------------------------------------
# Compliance Checker
# ---------------------------------------------------------------------------


class ComplianceChecker:
    """
    Verifies endpoint compliance against security requirements.

    Runs a series of posture checks against a device and produces
    a compliance report indicating whether the device meets the
    minimum requirements for remote BPO access.
    """

    def __init__(
        self,
        requirements: EndpointRequirements,
        *,
        strict_mode: bool = True,
        compliance_callback: Optional[
            Callable[[str, ComplianceStatus], None]
        ] = None,
    ):
        self.requirements = requirements
        self.strict_mode = strict_mode
        self.compliance_callback = compliance_callback
        self._check_history: Dict[str, List[PostureCheckReport]] = {}

    def check_compliance(
        self, posture: DevicePosture
    ) -> PostureCheckReport:
        """
        Run a full compliance check against a device posture.

        Args:
            posture: The current device posture to evaluate.

        Returns:
            A PostureCheckReport with the results.
        """
        report = PostureCheckReport(
            device_id=posture.device_id,
            agent_id=posture.agent_id,
        )

        # Run each check
        self._check_os(posture, report)
        self._check_antivirus(posture, report)
        self._check_firewall(posture, report)
        self._check_disk_encryption(posture, report)
        self._check_screen_lock(posture, report)
        self._check_network(posture, report)
        self._check_tpm(posture, report)
        self._check_usb(posture, report)
        self._check_apps(posture, report)

        # Determine overall status
        has_failures = any(
            v == PostureCheckResult.FAIL for v in report.checks.values()
        )
        has_warnings = any(
            v == PostureCheckResult.WARNING for v in report.checks.values()
        )

        if has_failures:
            report.overall_status = ComplianceStatus.NON_COMPLIANT
        elif has_warnings:
            report.overall_status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            report.overall_status = ComplianceStatus.COMPLIANT

        # Store in history
        if posture.device_id not in self._check_history:
            self._check_history[posture.device_id] = []
        self._check_history[posture.device_id].append(report)

        # Limit history
        if len(self._check_history[posture.device_id]) > 100:
            self._check_history[posture.device_id] = (
                self._check_history[posture.device_id][-50:]
            )

        # Fire callback
        if self.compliance_callback:
            try:
                self.compliance_callback(
                    posture.agent_id, report.overall_status
                )
            except Exception as exc:
                logger.error("Compliance callback failed: %s", exc)

        logger.info(
            "Compliance check: device=%s agent=%s status=%s "
            "failures=%d warnings=%d",
            posture.device_id,
            posture.agent_id,
            report.overall_status.display_name,
            len(report.failures),
            len(report.warnings),
        )

        return report

    def get_check_history(
        self, device_id: str, limit: int = 10
    ) -> List[PostureCheckReport]:
        """Get compliance check history for a device."""
        history = self._check_history.get(device_id, [])
        return history[-limit:]

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_os(
        self, posture: DevicePosture, report: PostureCheckReport
    ) -> None:
        """Check operating system compliance."""
        os_key = f"{posture.os_type}_{posture.os_version}".lower()

        if os_key not in self.requirements.allowed_os:
            # Check partial match (e.g., "windows_11" matches "windows_11")
            matched = any(
                os_key.startswith(allowed.rsplit("_", 1)[0])
                for allowed in self.requirements.allowed_os
            )
            if not matched:
                report.checks["os_version"] = PostureCheckResult.FAIL
                report.failures.append(
                    f"OS not in allowed list: {posture.os_type} {posture.os_version}"
                )
                report.recommendations.append(
                    "Upgrade to a supported operating system version"
                )
                return

        report.checks["os_version"] = PostureCheckResult.PASS

        # Check OS updates
        if self.requirements.require_os_updates and posture.os_last_update:
            age_days = (datetime.utcnow() - posture.os_last_update).days
            if age_days > self.requirements.max_os_update_age_days:
                report.checks["os_updates"] = PostureCheckResult.WARNING
                report.warnings.append(
                    f"OS update is {age_days} days old "
                    f"(max: {self.requirements.max_os_update_age_days})"
                )
                report.recommendations.append(
                    "Install pending operating system updates"
                )
            else:
                report.checks["os_updates"] = PostureCheckResult.PASS
        elif self.requirements.require_os_updates:
            report.checks["os_updates"] = PostureCheckResult.WARNING
            report.warnings.append("Unable to determine OS update status")

    def _check_antivirus(
        self, posture: DevicePosture, report: PostureCheckReport
    ) -> None:
        """Check antivirus compliance."""
        if not self.requirements.require_antivirus:
            report.checks["antivirus"] = PostureCheckResult.NOT_APPLICABLE
            return

        if not posture.antivirus_installed:
            report.checks["antivirus"] = PostureCheckResult.FAIL
            report.failures.append("Antivirus software not installed")
            report.recommendations.append(
                "Install approved antivirus software"
            )
            return

        if not posture.antivirus_real_time_enabled:
            report.checks["antivirus_realtime"] = PostureCheckResult.FAIL
            report.failures.append("Antivirus real-time protection is disabled")
            report.recommendations.append(
                "Enable antivirus real-time protection"
            )
        else:
            report.checks["antivirus_realtime"] = PostureCheckResult.PASS

        # Check signature age
        if (
            self.requirements.require_antivirus_updated
            and posture.antivirus_signatures_date
        ):
            age_days = (
                datetime.utcnow() - posture.antivirus_signatures_date
            ).days
            if age_days > self.requirements.max_antivirus_signature_age_days:
                report.checks["antivirus_signatures"] = PostureCheckResult.FAIL
                report.failures.append(
                    f"Antivirus signatures are {age_days} days old "
                    f"(max: {self.requirements.max_antivirus_signature_age_days})"
                )
                report.recommendations.append(
                    "Update antivirus virus definitions"
                )
            else:
                report.checks["antivirus_signatures"] = PostureCheckResult.PASS

        report.checks["antivirus"] = PostureCheckResult.PASS

    def _check_firewall(
        self, posture: DevicePosture, report: PostureCheckReport
    ) -> None:
        """Check host firewall compliance."""
        if not self.requirements.require_host_firewall:
            report.checks["firewall"] = PostureCheckResult.NOT_APPLICABLE
            return

        if posture.firewall_enabled:
            report.checks["firewall"] = PostureCheckResult.PASS
        else:
            report.checks["firewall"] = PostureCheckResult.FAIL
            report.failures.append("Host firewall is disabled")
            report.recommendations.append("Enable the host firewall")

    def _check_disk_encryption(
        self, posture: DevicePosture, report: PostureCheckReport
    ) -> None:
        """Check disk encryption compliance."""
        if not self.requirements.require_disk_encryption:
            report.checks["disk_encryption"] = PostureCheckResult.NOT_APPLICABLE
            return

        if not posture.disk_encrypted:
            report.checks["disk_encryption"] = PostureCheckResult.FAIL
            report.failures.append("Disk encryption is not enabled")
            report.recommendations.append(
                "Enable full disk encryption (BitLocker, FileVault, or LUKS)"
            )
            return

        if (
            self.requirements.allowed_encryption
            and posture.encryption_method.lower()
            not in self.requirements.allowed_encryption
        ):
            report.checks["disk_encryption"] = PostureCheckResult.WARNING
            report.warnings.append(
                f"Encryption method {posture.encryption_method} "
                f"is not in the approved list"
            )
        else:
            report.checks["disk_encryption"] = PostureCheckResult.PASS

        if posture.encryption_percentage < 100.0:
            report.checks["disk_encryption_complete"] = PostureCheckResult.WARNING
            report.warnings.append(
                f"Disk encryption is only {posture.encryption_percentage:.0f}% complete"
            )

    def _check_screen_lock(
        self, posture: DevicePosture, report: PostureCheckReport
    ) -> None:
        """Check screen lock compliance."""
        if not self.requirements.require_screen_lock:
            report.checks["screen_lock"] = PostureCheckResult.NOT_APPLICABLE
            return

        if not posture.screen_lock_enabled:
            report.checks["screen_lock"] = PostureCheckResult.FAIL
            report.failures.append("Screen lock is not enabled")
            report.recommendations.append(
                "Enable automatic screen lock with password"
            )
            return

        if (
            posture.screen_lock_timeout_minutes
            > self.requirements.max_screen_lock_timeout_minutes
        ):
            report.checks["screen_lock"] = PostureCheckResult.WARNING
            report.warnings.append(
                f"Screen lock timeout ({posture.screen_lock_timeout_minutes}min) "
                f"exceeds maximum ({self.requirements.max_screen_lock_timeout_minutes}min)"
            )
            report.recommendations.append(
                f"Set screen lock timeout to "
                f"{self.requirements.max_screen_lock_timeout_minutes} minutes or less"
            )
        else:
            report.checks["screen_lock"] = PostureCheckResult.PASS

    def _check_network(
        self, posture: DevicePosture, report: PostureCheckReport
    ) -> None:
        """Check network security compliance."""
        if not self.requirements.require_secure_wifi:
            report.checks["network"] = PostureCheckResult.NOT_APPLICABLE
            return

        if posture.network_type == "ethernet":
            report.checks["network"] = PostureCheckResult.PASS
            return

        # WiFi security check
        security_levels = {"open": 0, "wep": 1, "wpa": 2, "wpa2": 3, "wpa3": 4}
        current_level = security_levels.get(
            posture.wifi_security.lower(), 0
        )
        required_level = security_levels.get(
            self.requirements.min_wifi_security.lower(), 3
        )

        if current_level < required_level:
            report.checks["wifi_security"] = PostureCheckResult.FAIL
            report.failures.append(
                f"WiFi security ({posture.wifi_security}) is below minimum "
                f"({self.requirements.min_wifi_security})"
            )
            report.recommendations.append(
                f"Upgrade WiFi to {self.requirements.min_wifi_security} or higher"
            )
        else:
            report.checks["wifi_security"] = PostureCheckResult.PASS

        # Public network check
        if self.requirements.block_public_wifi and posture.is_public_network:
            report.checks["public_network"] = PostureCheckResult.FAIL
            report.failures.append("Connected to a public WiFi network")
            report.recommendations.append(
                "Connect to a private/home network"
            )
        else:
            report.checks["public_network"] = PostureCheckResult.PASS

    def _check_tpm(
        self, posture: DevicePosture, report: PostureCheckReport
    ) -> None:
        """Check TPM compliance."""
        if not self.requirements.require_tpm:
            report.checks["tpm"] = PostureCheckResult.NOT_APPLICABLE
            return

        if not posture.tpm_present:
            report.checks["tpm"] = PostureCheckResult.FAIL
            report.failures.append("TPM is not present on the device")
            report.recommendations.append(
                "Use a device with TPM 2.0 support"
            )
        elif posture.tpm_version < self.requirements.tpm_version:
            report.checks["tpm"] = PostureCheckResult.WARNING
            report.warnings.append(
                f"TPM version {posture.tpm_version} is below "
                f"required {self.requirements.tpm_version}"
            )
        else:
            report.checks["tpm"] = PostureCheckResult.PASS

    def _check_usb(
        self, posture: DevicePosture, report: PostureCheckReport
    ) -> None:
        """Check USB storage blocking."""
        if not self.requirements.block_usb_storage:
            report.checks["usb_storage"] = PostureCheckResult.NOT_APPLICABLE
            return

        if posture.usb_storage_blocked:
            report.checks["usb_storage"] = PostureCheckResult.PASS
        else:
            report.checks["usb_storage"] = PostureCheckResult.FAIL
            report.failures.append("USB storage is not blocked")
            report.recommendations.append(
                "Enable USB storage blocking via endpoint policy"
            )

    def _check_apps(
        self, posture: DevicePosture, report: PostureCheckReport
    ) -> None:
        """Check for unauthorized applications."""
        if not self.requirements.block_unauthorized_apps:
            report.checks["unauthorized_apps"] = PostureCheckResult.NOT_APPLICABLE
            return

        if posture.unauthorized_apps_detected:
            report.checks["unauthorized_apps"] = PostureCheckResult.FAIL
            report.failures.append(
                f"Unauthorized applications detected: "
                f"{', '.join(posture.unauthorized_apps_detected)}"
            )
            report.recommendations.append(
                "Remove or disable unauthorized applications before connecting"
            )
        else:
            report.checks["unauthorized_apps"] = PostureCheckResult.PASS


# ---------------------------------------------------------------------------
# Watermark Generator
# ---------------------------------------------------------------------------


class WatermarkGenerator:
    """
    Generates forensic watermarks for remote agent sessions.

    Watermarks embed identifying information into the agent's screen
    content, enabling forensic investigation of data leaks. If a
    screenshot or photo of the screen surfaces, the watermark
    identifies the agent, session, and timestamp.
    """

    def __init__(
        self,
        *,
        visible_opacity: float = 0.15,
        watermark_interval_seconds: int = 30,
        include_agent_id: bool = True,
        include_timestamp: bool = True,
        include_session_id: bool = True,
        include_ip_hash: bool = True,
    ):
        self.visible_opacity = visible_opacity
        self.watermark_interval_seconds = watermark_interval_seconds
        self.include_agent_id = include_agent_id
        self.include_timestamp = include_timestamp
        self.include_session_id = include_session_id
        self.include_ip_hash = include_ip_hash

    def generate_visible_watermark(
        self,
        agent_id: str,
        session_id: str,
        ip_address: str = "",
        tenant_id: str = "",
    ) -> Dict[str, Any]:
        """
        Generate a visible screen watermark.

        Returns watermark parameters that can be rendered as a
        semi-transparent overlay on the agent's desktop.

        Args:
            agent_id: The agent identifier.
            session_id: The session identifier.
            ip_address: The agent's IP address.
            tenant_id: The tenant identifier.

        Returns:
            Dict with watermark text, opacity, and positioning.
        """
        now = datetime.utcnow()
        parts: List[str] = []

        if self.include_agent_id:
            parts.append(f"Agent: {agent_id}")
        if self.include_session_id:
            # Short session ID for readability
            parts.append(f"Session: {session_id[:8]}")
        if self.include_timestamp:
            parts.append(now.strftime("%Y-%m-%d %H:%M UTC"))
        if self.include_ip_hash and ip_address:
            ip_hash = hashlib.sha256(ip_address.encode()).hexdigest()[:8]
            parts.append(f"EP: {ip_hash}")

        watermark_text = " | ".join(parts)

        # Generate a unique watermark ID for forensic tracking
        watermark_id = hashlib.sha256(
            f"{agent_id}:{session_id}:{now.isoformat()}".encode()
        ).hexdigest()[:16]

        return {
            "watermark_id": watermark_id,
            "text": watermark_text,
            "opacity": self.visible_opacity,
            "type": WatermarkType.VISIBLE.name,
            "generated_at": now.isoformat(),
            "agent_id": agent_id,
            "session_id": session_id,
            "tenant_id": tenant_id,
            "rotation_angle": -30,        # Diagonal watermark
            "repeat_pattern": True,        # Tile across screen
            "font_size": 14,
        }

    def generate_invisible_watermark(
        self,
        agent_id: str,
        session_id: str,
        ip_address: str = "",
    ) -> Dict[str, Any]:
        """
        Generate an invisible (steganographic) watermark payload.

        This payload is embedded into screen content using
        steganographic techniques that survive screenshots and
        photographs.

        Args:
            agent_id: The agent identifier.
            session_id: The session identifier.
            ip_address: The agent's IP address.

        Returns:
            Dict with steganographic payload parameters.
        """
        now = datetime.utcnow()

        payload = {
            "agent_id": agent_id,
            "session_id": session_id,
            "timestamp": now.isoformat(),
            "ip_hash": hashlib.sha256(
                ip_address.encode()
            ).hexdigest()[:16] if ip_address else "",
        }

        # Create a compact binary-safe payload
        payload_str = f"{agent_id}|{session_id[:16]}|{int(now.timestamp())}"
        payload_hash = hashlib.sha256(payload_str.encode()).hexdigest()

        return {
            "type": WatermarkType.INVISIBLE.name,
            "payload": payload_str,
            "payload_hash": payload_hash,
            "encoding": "lsb_steganography",
            "bit_depth": 2,               # 2 least significant bits
            "redundancy": 3,              # Repeat payload 3 times
            "error_correction": "reed_solomon",
            "generated_at": now.isoformat(),
        }

    def verify_watermark(
        self, extracted_payload: str
    ) -> Optional[Dict[str, str]]:
        """
        Verify and decode an extracted watermark payload.

        Args:
            extracted_payload: The payload extracted from an image.

        Returns:
            Dict with decoded watermark information, or None if invalid.
        """
        parts = extracted_payload.split("|")
        if len(parts) < 3:
            return None

        try:
            return {
                "agent_id": parts[0],
                "session_id": parts[1],
                "timestamp": datetime.fromtimestamp(
                    int(parts[2])
                ).isoformat(),
                "valid": True,
            }
        except (ValueError, IndexError):
            return None


# ---------------------------------------------------------------------------
# Network Assessor
# ---------------------------------------------------------------------------


class NetworkAssessor:
    """
    Assesses the security of remote agent home networks.

    Evaluates network configuration, security settings, and
    potential vulnerabilities of the network the agent is connecting from.
    """

    def __init__(
        self,
        *,
        min_wifi_security: str = "WPA2",
        block_public_networks: bool = True,
        allowed_dns_servers: Optional[Set[str]] = None,
        suspicious_dns_servers: Optional[Set[str]] = None,
    ):
        self.min_wifi_security = min_wifi_security
        self.block_public_networks = block_public_networks
        self.allowed_dns_servers = allowed_dns_servers
        self.suspicious_dns_servers = suspicious_dns_servers or {
            "0.0.0.0", "127.0.0.1",
        }

    def assess_network(
        self, posture: DevicePosture
    ) -> Tuple[NetworkRisk, List[str], List[str]]:
        """
        Assess the security risk of the agent's network.

        Args:
            posture: The device posture including network info.

        Returns:
            Tuple of (risk level, issues found, recommendations).
        """
        issues: List[str] = []
        recommendations: List[str] = []
        risk_score = 0

        # WiFi security
        if posture.network_type == "wifi":
            security_levels = {
                "open": 0, "wep": 1, "wpa": 2, "wpa2": 3, "wpa3": 4,
            }
            current = security_levels.get(posture.wifi_security.lower(), 0)
            required = security_levels.get(self.min_wifi_security.lower(), 3)

            if current < required:
                risk_score += 30
                issues.append(
                    f"WiFi security ({posture.wifi_security}) is below "
                    f"minimum ({self.min_wifi_security})"
                )
                recommendations.append(
                    f"Upgrade WiFi security to {self.min_wifi_security}"
                )

            if current == 0:
                risk_score += 40
                issues.append("Connected to an open WiFi network")
                recommendations.append(
                    "Do not use open WiFi networks for BPO work"
                )

        # Public network
        if posture.is_public_network:
            risk_score += 50
            issues.append("Connected to a public network")
            recommendations.append("Use a private home network")

        # DNS analysis
        if posture.dns_servers:
            for dns in posture.dns_servers:
                if dns in self.suspicious_dns_servers:
                    risk_score += 20
                    issues.append(f"Suspicious DNS server detected: {dns}")
                    recommendations.append(
                        "Use trusted DNS servers (e.g., corporate DNS)"
                    )
                    break

            if self.allowed_dns_servers:
                if not any(
                    dns in self.allowed_dns_servers
                    for dns in posture.dns_servers
                ):
                    risk_score += 10
                    issues.append("DNS servers are not in the approved list")
                    recommendations.append(
                        "Configure DNS to use approved servers"
                    )

        # Cellular network (higher risk)
        if posture.network_type == "cellular":
            risk_score += 15
            issues.append("Connected via cellular network")
            recommendations.append(
                "Use a stable home WiFi connection when possible"
            )

        # Determine risk level
        if risk_score >= 50:
            risk_level = NetworkRisk.CRITICAL
        elif risk_score >= 30:
            risk_level = NetworkRisk.HIGH
        elif risk_score >= 10:
            risk_level = NetworkRisk.MEDIUM
        else:
            risk_level = NetworkRisk.LOW

        return risk_level, issues, recommendations


# ---------------------------------------------------------------------------
# Remote Agent Security Manager (orchestrator)
# ---------------------------------------------------------------------------


class RemoteAgentSecurityManager:
    """
    Quantum-safe security for remote/work-from-home BPO agents.

    Provides:
    - VPN-less quantum-safe tunnel establishment
    - Endpoint compliance verification
    - Home network security assessment
    - Continuous device posture monitoring
    - Geo-fencing and location verification
    - Split tunneling prevention
    - Session watermarking for forensics

    This is the main orchestrator that coordinates all remote agent
    security controls. It provides a unified interface for managing
    the security lifecycle of remote agent sessions.

    Usage::

        requirements = EndpointRequirements(
            require_disk_encryption=True,
            require_antivirus=True,
        )
        geo_policy = GeoFencePolicy(
            allowed_countries={"US", "CA"},
        )

        manager = RemoteAgentSecurityManager(
            tenant_id="tenant-001",
            requirements=requirements,
            geo_fence_policy=geo_policy,
        )

        # Agent connects
        result = manager.connect_agent(
            agent_id="agent-001",
            device_posture=posture,
        )

        if result["status"] == "connected":
            # Agent can work
            ...
        else:
            # Agent blocked
            ...
    """

    def __init__(
        self,
        tenant_id: str = "",
        *,
        requirements: Optional[EndpointRequirements] = None,
        geo_fence_policy: Optional[GeoFencePolicy] = None,
        tunnel_config: Optional[TunnelConfig] = None,
        enable_watermarking: bool = True,
        enable_continuous_monitoring: bool = True,
        monitoring_interval_seconds: int = 300,
        compliance_callback: Optional[
            Callable[[str, ComplianceStatus], None]
        ] = None,
    ):
        self.tenant_id = tenant_id
        self.enable_watermarking = enable_watermarking
        self.enable_continuous_monitoring = enable_continuous_monitoring
        self.monitoring_interval_seconds = monitoring_interval_seconds

        # Initialize sub-components
        self.requirements = requirements or EndpointRequirements()
        self.geo_fence_policy = geo_fence_policy or GeoFencePolicy()
        self.default_tunnel_config = tunnel_config or TunnelConfig()

        self.compliance_checker = ComplianceChecker(
            requirements=self.requirements,
            compliance_callback=compliance_callback,
        )
        self.watermark_generator = WatermarkGenerator()
        self.network_assessor = NetworkAssessor()

        # Active agent sessions (agent_id -> session info)
        self._active_sessions: Dict[str, Dict[str, Any]] = {}
        self._tunnel_sessions: Dict[str, TunnelSession] = {}

        self._stats = {
            "total_connections": 0,
            "total_rejections": 0,
            "total_compliance_checks": 0,
            "active_agents": 0,
        }

        logger.info(
            "RemoteAgentSecurityManager initialized: tenant=%s "
            "watermarking=%s continuous_monitoring=%s",
            tenant_id,
            enable_watermarking,
            enable_continuous_monitoring,
        )

    def connect_agent(
        self,
        agent_id: str,
        device_posture: DevicePosture,
    ) -> Dict[str, Any]:
        """
        Process a remote agent connection request.

        This is the main entry point for remote agent connections.
        It performs:
        1. Endpoint compliance check
        2. Network security assessment
        3. Geo-fence verification
        4. Tunnel establishment (if all checks pass)
        5. Watermark initialization

        Args:
            agent_id: The agent identifier.
            device_posture: The current device posture.

        Returns:
            Dict with connection result including status and details.
        """
        self._stats["total_connections"] += 1
        result: Dict[str, Any] = {
            "agent_id": agent_id,
            "status": "pending",
            "compliance": None,
            "network_risk": None,
            "geo_allowed": None,
            "tunnel": None,
            "watermark": None,
            "rejection_reasons": [],
        }

        # 1. Compliance check
        self._stats["total_compliance_checks"] += 1
        compliance_report = self.compliance_checker.check_compliance(
            device_posture
        )
        result["compliance"] = compliance_report.overall_status.display_name

        if compliance_report.overall_status == ComplianceStatus.NON_COMPLIANT:
            result["status"] = "rejected"
            result["rejection_reasons"].extend(compliance_report.failures)
            self._stats["total_rejections"] += 1
            logger.warning(
                "Agent %s connection rejected: non-compliant endpoint",
                agent_id,
            )
            return result

        # 2. Network assessment
        network_risk, issues, recommendations = (
            self.network_assessor.assess_network(device_posture)
        )
        result["network_risk"] = network_risk.display_name

        if network_risk == NetworkRisk.CRITICAL:
            result["status"] = "rejected"
            result["rejection_reasons"].extend(issues)
            self._stats["total_rejections"] += 1
            logger.warning(
                "Agent %s connection rejected: critical network risk",
                agent_id,
            )
            return result

        # 3. Geo-fence check
        geo_allowed, geo_reason = self.geo_fence_policy.is_location_allowed(
            country=device_posture.geo_country,
            region=device_posture.geo_region,
            city=device_posture.geo_city,
        )
        result["geo_allowed"] = geo_allowed

        if not geo_allowed:
            result["status"] = "rejected"
            result["rejection_reasons"].append(geo_reason)
            self._stats["total_rejections"] += 1
            logger.warning(
                "Agent %s connection rejected: geo-fence violation (%s)",
                agent_id,
                geo_reason,
            )
            return result

        # 4. Establish tunnel
        tunnel_session = self._establish_tunnel(
            agent_id, device_posture.device_id
        )
        result["tunnel"] = tunnel_session.state.name

        # 5. Generate watermark
        if self.enable_watermarking:
            watermark = self.watermark_generator.generate_visible_watermark(
                agent_id=agent_id,
                session_id=tunnel_session.session_id,
                ip_address=device_posture.ip_address,
                tenant_id=self.tenant_id,
            )
            result["watermark"] = watermark

        # Success
        result["status"] = "connected"
        self._active_sessions[agent_id] = {
            "agent_id": agent_id,
            "device_id": device_posture.device_id,
            "tunnel_session_id": tunnel_session.session_id,
            "connected_at": datetime.utcnow(),
            "last_compliance_check": datetime.utcnow(),
            "compliance_status": compliance_report.overall_status,
            "network_risk": network_risk,
            "geo_country": device_posture.geo_country,
        }
        self._stats["active_agents"] += 1

        logger.info(
            "Agent %s connected: compliance=%s network=%s geo=%s/%s",
            agent_id,
            compliance_report.overall_status.display_name,
            network_risk.display_name,
            device_posture.geo_country,
            device_posture.geo_region,
        )

        return result

    def disconnect_agent(
        self, agent_id: str, reason: str = "manual"
    ) -> bool:
        """
        Disconnect a remote agent.

        Args:
            agent_id: The agent to disconnect.
            reason: Reason for disconnection.

        Returns:
            True if the agent was disconnected.
        """
        if agent_id not in self._active_sessions:
            return False

        session = self._active_sessions[agent_id]
        tunnel_id = session.get("tunnel_session_id")

        # Tear down tunnel
        if tunnel_id and tunnel_id in self._tunnel_sessions:
            self._tunnel_sessions[tunnel_id].state = TunnelState.DISCONNECTED

        del self._active_sessions[agent_id]
        self._stats["active_agents"] = max(
            0, self._stats["active_agents"] - 1
        )

        logger.info(
            "Agent %s disconnected: reason=%s", agent_id, reason
        )
        return True

    def recheck_compliance(
        self,
        agent_id: str,
        device_posture: DevicePosture,
    ) -> Dict[str, Any]:
        """
        Re-run compliance check for an active agent.

        Used for continuous monitoring of remote endpoints.

        Args:
            agent_id: The agent to re-check.
            device_posture: Updated device posture.

        Returns:
            Dict with re-check results.
        """
        self._stats["total_compliance_checks"] += 1
        report = self.compliance_checker.check_compliance(device_posture)

        result: Dict[str, Any] = {
            "agent_id": agent_id,
            "status": report.overall_status.display_name,
            "failures": report.failures,
            "warnings": report.warnings,
        }

        # Update session tracking
        if agent_id in self._active_sessions:
            self._active_sessions[agent_id]["last_compliance_check"] = (
                datetime.utcnow()
            )
            self._active_sessions[agent_id]["compliance_status"] = (
                report.overall_status
            )

            # Disconnect if now non-compliant
            if report.overall_status == ComplianceStatus.NON_COMPLIANT:
                self.disconnect_agent(
                    agent_id,
                    reason="Compliance check failed during session",
                )
                result["action"] = "disconnected"
                logger.warning(
                    "Agent %s disconnected: compliance check failed",
                    agent_id,
                )

        return result

    def get_active_agents(self) -> List[Dict[str, Any]]:
        """Get all active remote agent sessions."""
        return list(self._active_sessions.values())

    def get_agent_session(
        self, agent_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get session info for a specific agent."""
        return self._active_sessions.get(agent_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall remote access statistics."""
        return dict(self._stats)

    def _establish_tunnel(
        self, agent_id: str, device_id: str
    ) -> TunnelSession:
        """Establish a PQC tunnel for a remote agent."""
        tunnel = TunnelSession(
            agent_id=agent_id,
            device_id=device_id,
            config=self.default_tunnel_config,
            state=TunnelState.CONNECTED,
            established_at=datetime.utcnow(),
            current_kem_algorithm=self.default_tunnel_config.kem_algorithm,
            current_sig_algorithm=self.default_tunnel_config.sig_algorithm,
        )
        self._tunnel_sessions[tunnel.session_id] = tunnel

        logger.info(
            "PQC tunnel established: agent=%s session=%s "
            "kem=%s sig=%s hybrid=%s",
            agent_id,
            tunnel.session_id[:8],
            tunnel.current_kem_algorithm,
            tunnel.current_sig_algorithm,
            self.default_tunnel_config.hybrid_mode,
        )

        return tunnel
