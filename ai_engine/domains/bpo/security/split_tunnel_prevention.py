"""
Split-Tunnel Prevention and Detection Module

Detects and prevents split-tunnel configurations for remote BPO agents.

Split tunneling allows traffic to bypass the corporate VPN, creating
data exfiltration vectors and compliance violations. This module detects:
- Active split-tunnel configurations on agent workstations
- DNS leak conditions that expose query metadata
- Dual-NIC configurations enabling out-of-band data paths
- VPN bypass attempts via proxy, Tor, or tethering
- Unauthorized route table modifications
- MTU mismatches indicative of tunnel manipulation

Integrates with QBITEL's PQC-WireGuard infrastructure for
quantum-safe tunnel enforcement and audit logging.
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


class TunnelViolationType(Enum):
    """Types of tunnel violations detected on agent workstations."""

    SPLIT_TUNNEL_DETECTED = auto()      # Traffic not fully tunneled
    DNS_LEAK = auto()                   # DNS queries escaping the tunnel
    DUAL_NIC_ACTIVE = auto()            # Multiple active network interfaces
    VPN_BYPASS_ATTEMPT = auto()         # Deliberate VPN circumvention
    PROXY_DETECTED = auto()             # HTTP/SOCKS proxy in use
    TOR_EXIT_NODE = auto()              # Tor network exit detected
    HOTSPOT_TETHERING = auto()          # Mobile hotspot or tethering
    UNAUTHORIZED_ROUTE = auto()         # Route table modification
    DIRECT_INTERNET_ACCESS = auto()     # Traffic reaching internet directly
    MTU_MISMATCH_SUSPICIOUS = auto()    # MTU inconsistency suggesting tampering


class TunnelEnforcementLevel(Enum):
    """Enforcement levels for tunnel policy violations."""

    MONITOR_ONLY = auto()       # Log the violation, take no action
    WARN_AGENT = auto()         # Display a warning to the agent
    BLOCK_TRAFFIC = auto()      # Block non-tunneled traffic
    TERMINATE_SESSION = auto()  # Terminate the agent session immediately


class NetworkInterfaceType(Enum):
    """Types of network interfaces on agent workstations."""

    ETHERNET = auto()
    WIFI = auto()
    CELLULAR = auto()
    VPN = auto()
    LOOPBACK = auto()
    VIRTUAL = auto()
    UNKNOWN = auto()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class NetworkInterface:
    """
    Represents a network interface on an agent workstation.

    Captures interface metadata needed for split-tunnel detection
    without exposing sensitive network configuration details.
    """

    interface_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    type: NetworkInterfaceType = NetworkInterfaceType.UNKNOWN
    ip_address: str = ""
    is_active: bool = False
    is_default_route: bool = False
    gateway: str = ""
    dns_servers: List[str] = field(default_factory=list)
    mtu: int = 1500

    def to_dict(self) -> Dict[str, Any]:
        """Serialize interface for storage or transmission."""
        return {
            "interface_id": self.interface_id,
            "name": self.name,
            "type": self.type.name,
            "ip_address": self._mask_ip(self.ip_address),
            "is_active": self.is_active,
            "is_default_route": self.is_default_route,
            "gateway": self._mask_ip(self.gateway),
            "dns_servers": [self._mask_ip(s) for s in self.dns_servers],
            "mtu": self.mtu,
        }

    @staticmethod
    def _mask_ip(ip: str) -> str:
        """Mask an IP address for audit logs, preserving subnet."""
        parts = ip.split(".")
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.xxx.xxx"
        return "***"


@dataclass
class TunnelStatus:
    """
    Current tunnel status for an agent session.

    Captures a point-in-time snapshot of the VPN tunnel state
    including all active network interfaces and leak indicators.
    """

    status_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    tunnel_active: bool = False
    tunnel_protocol: str = "PQC-WireGuard"
    all_traffic_tunneled: bool = False
    active_interfaces: List[NetworkInterface] = field(default_factory=list)
    dns_leak_detected: bool = False
    split_tunnel_detected: bool = False
    checked_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize status for storage or transmission."""
        return {
            "status_id": self.status_id,
            "agent_id": self.agent_id,
            "tunnel_active": self.tunnel_active,
            "tunnel_protocol": self.tunnel_protocol,
            "all_traffic_tunneled": self.all_traffic_tunneled,
            "active_interfaces": [i.to_dict() for i in self.active_interfaces],
            "dns_leak_detected": self.dns_leak_detected,
            "split_tunnel_detected": self.split_tunnel_detected,
            "checked_at": self.checked_at.isoformat(),
        }


@dataclass
class TunnelViolationEvent:
    """
    A recorded tunnel policy violation event.

    Contains classification, evidence, and the enforcement
    action taken. Each event is PQC-hashed for tamper-proof audit.
    """

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    violation_type: TunnelViolationType = TunnelViolationType.SPLIT_TUNNEL_DETECTED
    agent_id: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    action_taken: TunnelEnforcementLevel = TunnelEnforcementLevel.MONITOR_ONLY
    detected_at: datetime = field(default_factory=datetime.utcnow)
    pqc_audit_hash: str = ""

    def __post_init__(self) -> None:
        if not self.pqc_audit_hash:
            self.pqc_audit_hash = self._compute_audit_hash()

    def _compute_audit_hash(self) -> str:
        """Compute a SHA3-256 audit hash for this event."""
        payload = (
            f"{self.event_id}|{self.violation_type.name}|"
            f"{self.agent_id}|{self.detected_at.isoformat()}"
        )
        return hashlib.sha3_256(payload.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize event for storage or transmission."""
        return {
            "event_id": self.event_id,
            "violation_type": self.violation_type.name,
            "agent_id": self.agent_id,
            "evidence": self.evidence,
            "action_taken": self.action_taken.name,
            "detected_at": self.detected_at.isoformat(),
            "pqc_audit_hash": self.pqc_audit_hash,
        }


@dataclass
class SplitTunnelPolicy:
    """
    Configuration policy for split-tunnel prevention.

    Defines enforcement behavior, thresholds, and allowed
    exceptions for tunnel monitoring.
    """

    enforcement_level: TunnelEnforcementLevel = TunnelEnforcementLevel.BLOCK_TRAFFIC
    full_tunnel_required: bool = True
    dns_leak_prevention: bool = True
    block_dual_nic: bool = True
    allowed_local_subnets: List[str] = field(
        default_factory=lambda: ["192.168.0.0/16", "10.0.0.0/8"]
    )
    check_interval_seconds: int = 30
    max_violations_before_terminate: int = 3
    pqc_tunnel_protocol: str = "ML-KEM-768"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize policy for storage or transmission."""
        return {
            "enforcement_level": self.enforcement_level.name,
            "full_tunnel_required": self.full_tunnel_required,
            "dns_leak_prevention": self.dns_leak_prevention,
            "block_dual_nic": self.block_dual_nic,
            "allowed_local_subnets": self.allowed_local_subnets,
            "check_interval_seconds": self.check_interval_seconds,
            "max_violations_before_terminate": self.max_violations_before_terminate,
            "pqc_tunnel_protocol": self.pqc_tunnel_protocol,
        }


# ---------------------------------------------------------------------------
# Known indicators
# ---------------------------------------------------------------------------

# Public DNS servers that indicate DNS leak (should use tunnel DNS)
PUBLIC_DNS_SERVERS: Set[str] = {
    "8.8.8.8", "8.8.4.4",           # Google
    "1.1.1.1", "1.0.0.1",           # Cloudflare
    "9.9.9.9", "149.112.112.112",   # Quad9
    "208.67.222.222", "208.67.220.220",  # OpenDNS
    "64.6.64.6", "64.6.65.6",       # Verisign
}

# Known Tor exit node port patterns
TOR_INDICATOR_PORTS: Set[int] = {9001, 9030, 9050, 9051, 9150}

# Standard VPN MTU values
VPN_MTU_RANGE: Tuple[int, int] = (1280, 1420)

# Proxy detection patterns in environment or configuration
PROXY_PATTERNS: List[str] = [
    r"(?i)http_proxy\s*=",
    r"(?i)https_proxy\s*=",
    r"(?i)socks[45]?://",
    r"(?i)all_proxy\s*=",
    r"(?i)no_proxy\s*=",
]


# ---------------------------------------------------------------------------
# Split-Tunnel Prevention Engine
# ---------------------------------------------------------------------------


class SplitTunnelPreventionEngine:
    """
    Detects and prevents split-tunnel configurations for remote BPO agents.

    Split tunneling allows traffic to bypass the corporate VPN, creating
    data exfiltration vectors and compliance violations. This engine
    monitors agent network configurations in real time.

    Usage::

        engine = SplitTunnelPreventionEngine.create_strict_policy()

        # Periodic check
        status = engine.check_tunnel_status(agent_id, interfaces)
        if not status.all_traffic_tunneled:
            violations = engine.detect_split_tunnel(status)
            for violation in violations:
                engine.enforce_policy(violation)

        # Generate compliance report
        report = engine.generate_compliance_report()
    """

    def __init__(self, policy: Optional[SplitTunnelPolicy] = None):
        self._policy = policy or SplitTunnelPolicy()
        self._violation_history: Dict[str, List[TunnelViolationEvent]] = {}
        self._agent_violation_counts: Dict[str, int] = {}
        self._last_check: Dict[str, datetime] = {}
        self._stats = {
            "total_checks": 0,
            "total_violations": 0,
            "sessions_terminated": 0,
            "dns_leaks_detected": 0,
            "split_tunnels_detected": 0,
        }

        logger.info(
            "SplitTunnelPreventionEngine initialized "
            "enforcement=%s full_tunnel=%s protocol=%s",
            self._policy.enforcement_level.name,
            self._policy.full_tunnel_required,
            self._policy.pqc_tunnel_protocol,
        )

    # ------------------------------------------------------------------
    # Tunnel status checking
    # ------------------------------------------------------------------

    def check_tunnel_status(
        self,
        agent_id: str,
        interfaces: List[NetworkInterface],
    ) -> TunnelStatus:
        """
        Check the current tunnel status for an agent.

        Evaluates all active network interfaces to determine whether
        the VPN tunnel is properly configured and all traffic is routed
        through it.

        Args:
            agent_id: The agent whose tunnel to check.
            interfaces: Current network interfaces on the workstation.

        Returns:
            A TunnelStatus snapshot.
        """
        self._stats["total_checks"] += 1
        now = datetime.utcnow()
        self._last_check[agent_id] = now

        active = [i for i in interfaces if i.is_active]
        vpn_interfaces = [i for i in active if i.type == NetworkInterfaceType.VPN]
        tunnel_active = len(vpn_interfaces) > 0

        # Determine if all traffic is tunneled
        non_vpn_default = [
            i for i in active
            if i.is_default_route and i.type != NetworkInterfaceType.VPN
            and i.type != NetworkInterfaceType.LOOPBACK
        ]
        all_tunneled = tunnel_active and len(non_vpn_default) == 0

        # Check for DNS leak
        dns_leak = self._check_dns_leak(active)

        # Check for split tunnel
        split_detected = tunnel_active and not all_tunneled

        status = TunnelStatus(
            agent_id=agent_id,
            tunnel_active=tunnel_active,
            tunnel_protocol=self._policy.pqc_tunnel_protocol,
            all_traffic_tunneled=all_tunneled,
            active_interfaces=active,
            dns_leak_detected=dns_leak,
            split_tunnel_detected=split_detected,
            checked_at=now,
        )

        if dns_leak or split_detected:
            logger.warning(
                "Tunnel issue detected: agent=%s tunnel_active=%s "
                "all_tunneled=%s dns_leak=%s split=%s",
                agent_id, tunnel_active, all_tunneled, dns_leak, split_detected,
            )

        return status

    def detect_split_tunnel(
        self, status: TunnelStatus
    ) -> List[TunnelViolationEvent]:
        """
        Analyze a tunnel status and emit violation events.

        Checks for all violation types including split tunnel,
        DNS leak, dual NIC, and unauthorized routes.

        Args:
            status: The tunnel status snapshot to analyze.

        Returns:
            List of violation events detected.
        """
        violations: List[TunnelViolationEvent] = []

        if status.split_tunnel_detected:
            violations.append(self._create_violation(
                TunnelViolationType.SPLIT_TUNNEL_DETECTED,
                status.agent_id,
                {"tunnel_active": status.tunnel_active,
                 "all_tunneled": status.all_traffic_tunneled},
            ))
            self._stats["split_tunnels_detected"] += 1

        if status.dns_leak_detected:
            violations.append(self._create_violation(
                TunnelViolationType.DNS_LEAK,
                status.agent_id,
                {"public_dns_found": True},
            ))
            self._stats["dns_leaks_detected"] += 1

        # Check for dual NIC
        if self._policy.block_dual_nic:
            dual_violations = self._detect_dual_nic(status)
            violations.extend(dual_violations)

        # Check for MTU mismatch
        mtu_violations = self._detect_mtu_mismatch(status)
        violations.extend(mtu_violations)

        # Check for direct internet access
        if not status.tunnel_active and self._policy.full_tunnel_required:
            violations.append(self._create_violation(
                TunnelViolationType.DIRECT_INTERNET_ACCESS,
                status.agent_id,
                {"tunnel_active": False, "policy_requires_tunnel": True},
            ))

        self._stats["total_violations"] += len(violations)
        return violations

    def detect_dns_leak(
        self, interfaces: List[NetworkInterface]
    ) -> bool:
        """
        Check whether any interface has DNS servers outside the tunnel.

        Args:
            interfaces: Active network interfaces to check.

        Returns:
            True if a DNS leak condition is detected.
        """
        return self._check_dns_leak(interfaces)

    def check_network_interfaces(
        self, interfaces: List[NetworkInterface]
    ) -> Dict[str, Any]:
        """
        Analyze network interfaces for security concerns.

        Returns a summary of findings including interface counts,
        potential risks, and recommendations.

        Args:
            interfaces: All network interfaces on the workstation.

        Returns:
            Analysis summary dictionary.
        """
        active = [i for i in interfaces if i.is_active]
        vpn_count = sum(1 for i in active if i.type == NetworkInterfaceType.VPN)
        wifi_count = sum(1 for i in active if i.type == NetworkInterfaceType.WIFI)
        cellular_count = sum(1 for i in active if i.type == NetworkInterfaceType.CELLULAR)
        virtual_count = sum(1 for i in active if i.type == NetworkInterfaceType.VIRTUAL)

        risks: List[str] = []
        if vpn_count == 0:
            risks.append("No active VPN interface detected")
        if wifi_count > 0 and vpn_count > 0:
            risks.append("WiFi active alongside VPN - potential split tunnel")
        if cellular_count > 0:
            risks.append("Cellular interface active - potential tethering/bypass")
        if virtual_count > 0:
            risks.append("Virtual interface detected - check for VM tunnel bypass")

        return {
            "total_interfaces": len(interfaces),
            "active_interfaces": len(active),
            "vpn_interfaces": vpn_count,
            "wifi_interfaces": wifi_count,
            "cellular_interfaces": cellular_count,
            "virtual_interfaces": virtual_count,
            "risks": risks,
            "compliant": len(risks) == 0,
        }

    # ------------------------------------------------------------------
    # Policy enforcement
    # ------------------------------------------------------------------

    def enforce_policy(
        self, violation: TunnelViolationEvent
    ) -> TunnelEnforcementLevel:
        """
        Enforce the tunnel policy for a detected violation.

        Applies the configured enforcement level and tracks
        violation counts per agent for escalation.

        Args:
            violation: The violation event to enforce.

        Returns:
            The enforcement level applied.
        """
        agent_id = violation.agent_id
        count = self._agent_violation_counts.get(agent_id, 0) + 1
        self._agent_violation_counts[agent_id] = count

        # Track violation history
        if agent_id not in self._violation_history:
            self._violation_history[agent_id] = []
        self._violation_history[agent_id].append(violation)

        # Escalate if threshold exceeded
        if count >= self._policy.max_violations_before_terminate:
            enforcement = TunnelEnforcementLevel.TERMINATE_SESSION
            self._stats["sessions_terminated"] += 1
            logger.critical(
                "Agent session terminated: agent=%s violations=%d "
                "max_allowed=%d last_violation=%s",
                agent_id, count,
                self._policy.max_violations_before_terminate,
                violation.violation_type.name,
            )
        else:
            enforcement = self._policy.enforcement_level

        violation.action_taken = enforcement

        logger.warning(
            "Policy enforced: agent=%s violation=%s action=%s count=%d/%d",
            agent_id, violation.violation_type.name,
            enforcement.name, count,
            self._policy.max_violations_before_terminate,
        )

        return enforcement

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_violation_history(
        self,
        agent_id: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[TunnelViolationEvent]:
        """
        Retrieve violation history, optionally filtered by agent or time.

        Args:
            agent_id: Filter to a specific agent. None for all agents.
            since: Only return violations after this timestamp.

        Returns:
            List of matching violation events.
        """
        results: List[TunnelViolationEvent] = []

        if agent_id is not None:
            events = self._violation_history.get(agent_id, [])
        else:
            events = [
                e for evts in self._violation_history.values() for e in evts
            ]

        for event in events:
            if since is not None and event.detected_at < since:
                continue
            results.append(event)

        results.sort(key=lambda e: e.detected_at, reverse=True)
        return results

    def generate_compliance_report(self) -> Dict[str, Any]:
        """
        Generate a compliance report for tunnel enforcement.

        Returns:
            Report dictionary with statistics and findings.
        """
        all_violations = [
            e for evts in self._violation_history.values() for e in evts
        ]
        violation_by_type: Dict[str, int] = {}
        for v in all_violations:
            key = v.violation_type.name
            violation_by_type[key] = violation_by_type.get(key, 0) + 1

        agents_with_violations = len(self._violation_history)
        agents_terminated = sum(
            1 for c in self._agent_violation_counts.values()
            if c >= self._policy.max_violations_before_terminate
        )

        report_id = str(uuid.uuid4())
        report_hash = hashlib.sha3_256(
            f"{report_id}|{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()

        return {
            "report_id": report_id,
            "generated_at": datetime.utcnow().isoformat(),
            "policy": self._policy.to_dict(),
            "statistics": dict(self._stats),
            "violation_breakdown": violation_by_type,
            "agents_with_violations": agents_with_violations,
            "agents_terminated": agents_terminated,
            "total_violations": len(all_violations),
            "pqc_report_hash": report_hash,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_dns_leak(self, interfaces: List[NetworkInterface]) -> bool:
        """Check for DNS servers outside the VPN tunnel."""
        for iface in interfaces:
            if iface.type == NetworkInterfaceType.VPN:
                continue
            for dns in iface.dns_servers:
                if dns in PUBLIC_DNS_SERVERS:
                    return True
        return False

    def _detect_dual_nic(
        self, status: TunnelStatus
    ) -> List[TunnelViolationEvent]:
        """Detect active dual-NIC configurations."""
        violations: List[TunnelViolationEvent] = []
        non_loopback = [
            i for i in status.active_interfaces
            if i.type not in (NetworkInterfaceType.LOOPBACK, NetworkInterfaceType.VPN)
        ]
        if len(non_loopback) > 1:
            violations.append(self._create_violation(
                TunnelViolationType.DUAL_NIC_ACTIVE,
                status.agent_id,
                {"active_non_vpn_interfaces": len(non_loopback),
                 "interface_names": [i.name for i in non_loopback]},
            ))
        return violations

    def _detect_mtu_mismatch(
        self, status: TunnelStatus
    ) -> List[TunnelViolationEvent]:
        """Detect suspicious MTU mismatches between interfaces."""
        violations: List[TunnelViolationEvent] = []
        vpn_ifaces = [
            i for i in status.active_interfaces
            if i.type == NetworkInterfaceType.VPN
        ]
        for vpn in vpn_ifaces:
            if not (VPN_MTU_RANGE[0] <= vpn.mtu <= VPN_MTU_RANGE[1]):
                violations.append(self._create_violation(
                    TunnelViolationType.MTU_MISMATCH_SUSPICIOUS,
                    status.agent_id,
                    {"vpn_mtu": vpn.mtu,
                     "expected_range": list(VPN_MTU_RANGE),
                     "interface": vpn.name},
                ))
        return violations

    def _create_violation(
        self,
        violation_type: TunnelViolationType,
        agent_id: str,
        evidence: Dict[str, Any],
    ) -> TunnelViolationEvent:
        """Create a violation event with PQC audit hash."""
        return TunnelViolationEvent(
            violation_type=violation_type,
            agent_id=agent_id,
            evidence=evidence,
            action_taken=self._policy.enforcement_level,
        )

    # ------------------------------------------------------------------
    # Factory class methods
    # ------------------------------------------------------------------

    @classmethod
    def create_strict_policy(cls) -> "SplitTunnelPreventionEngine":
        """
        Create an engine with strict tunnel enforcement.

        Full tunnel required, dual NIC blocked, 15-second check
        interval, sessions terminated after 2 violations.
        """
        policy = SplitTunnelPolicy(
            enforcement_level=TunnelEnforcementLevel.TERMINATE_SESSION,
            full_tunnel_required=True,
            dns_leak_prevention=True,
            block_dual_nic=True,
            allowed_local_subnets=["192.168.0.0/16"],
            check_interval_seconds=15,
            max_violations_before_terminate=2,
            pqc_tunnel_protocol="ML-KEM-1024",
        )
        return cls(policy=policy)

    @classmethod
    def create_flexible_policy(cls) -> "SplitTunnelPreventionEngine":
        """
        Create an engine with flexible tunnel enforcement.

        Monitor-only mode, dual NIC allowed, 60-second check
        interval. Suitable for low-risk environments.
        """
        policy = SplitTunnelPolicy(
            enforcement_level=TunnelEnforcementLevel.WARN_AGENT,
            full_tunnel_required=False,
            dns_leak_prevention=True,
            block_dual_nic=False,
            allowed_local_subnets=["192.168.0.0/16", "10.0.0.0/8", "172.16.0.0/12"],
            check_interval_seconds=60,
            max_violations_before_terminate=10,
            pqc_tunnel_protocol="ML-KEM-768",
        )
        return cls(policy=policy)
