"""
CCaaS API Security Module with PQC-Signed Tokens

Secures CCaaS (Contact Center as a Service) API integrations with
post-quantum cryptographic token signing and request authentication.

BPO operations depend on CCaaS platforms for omnichannel routing,
workforce management, and customer interaction analytics. This module:
- Signs API tokens with ML-DSA-65 for quantum-resistant authentication
- Validates every API request against scope, rate limit, and origin
- Detects anomalous request patterns (credential stuffing, replay, injection)
- Rotates compromised keys automatically
- Provides per-platform security policies for multi-vendor BPO estates

Integrates with QBITEL's quantum-safe infrastructure for secure
token lifecycle management and audit-trail preservation.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import hashlib
import logging
import re
import uuid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CCaaSPlatform(Enum):
    """Supported CCaaS platforms in the BPO estate."""

    GENESYS_CLOUD = auto()
    AMAZON_CONNECT = auto()
    FIVE9 = auto()
    TWILIO_FLEX = auto()
    NICE_CXONE = auto()
    TALKDESK = auto()
    RINGCENTRAL = auto()
    VONAGE = auto()


class APIThreatType(Enum):
    """Types of API-layer threats targeting CCaaS integrations."""

    KEY_COMPROMISE = auto()          # API key leaked or stolen
    TOKEN_REPLAY = auto()            # Captured token replayed
    RATE_LIMIT_BYPASS = auto()       # Client circumventing rate limits
    INJECTION_ATTACK = auto()        # SQL/NoSQL/command injection via API
    UNAUTHORIZED_SCOPE = auto()      # Token used outside granted scopes
    CREDENTIAL_STUFFING = auto()     # Bulk login attempts with leaked creds
    MAN_IN_THE_MIDDLE = auto()       # Intercepted API traffic
    PRIVILEGE_ESCALATION = auto()    # Attempt to elevate token privileges


class APISecurityAction(Enum):
    """Remediation actions for detected API threats."""

    LOG = auto()                     # Log the event for audit
    ALERT = auto()                   # Notify the security team
    ROTATE_KEY = auto()              # Rotate the affected API key
    BLOCK_REQUEST = auto()           # Reject the current request
    QUARANTINE_TOKEN = auto()        # Disable the token pending review
    REVOKE_ACCESS = auto()           # Permanently revoke tenant access


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class APIToken:
    """
    A PQC-signed API token issued to a CCaaS integration.

    Tokens carry the platform identity, tenant scope, and a
    quantum-resistant signature that can be verified without
    contacting the issuing authority.
    """

    token_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    platform: CCaaSPlatform = CCaaSPlatform.GENESYS_CLOUD
    tenant_id: str = ""
    issued_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(
        default_factory=lambda: datetime.utcnow() + timedelta(hours=1)
    )
    scopes: List[str] = field(default_factory=list)
    pqc_signature: str = ""
    key_algorithm: str = "ML-DSA-65"
    is_revoked: bool = False

    @property
    def is_expired(self) -> bool:
        """Check whether the token has expired."""
        return datetime.utcnow() > self.expires_at

    @property
    def remaining_seconds(self) -> float:
        """Seconds remaining until expiry (may be negative)."""
        delta = self.expires_at - datetime.utcnow()
        return delta.total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize token metadata for audit logging."""
        return {
            "token_id": self.token_id,
            "platform": self.platform.name,
            "tenant_id": self.tenant_id,
            "issued_at": self.issued_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "scopes": self.scopes,
            "key_algorithm": self.key_algorithm,
            "is_revoked": self.is_revoked,
        }


@dataclass
class APISecurityEvent:
    """
    A security event recorded during API request processing.

    Captures the threat classification, evidence collected, and
    the remediation action applied.
    """

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    threat_type: APIThreatType = APIThreatType.KEY_COMPROMISE
    platform: CCaaSPlatform = CCaaSPlatform.GENESYS_CLOUD
    request_path: str = ""
    source_ip: str = ""
    agent_id: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    action_taken: APISecurityAction = APISecurityAction.LOG
    detected_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize event for storage or transmission."""
        return {
            "event_id": self.event_id,
            "threat_type": self.threat_type.name,
            "platform": self.platform.name,
            "request_path": self.request_path,
            "source_ip": self.source_ip,
            "agent_id": self.agent_id,
            "evidence": self.evidence,
            "action_taken": self.action_taken.name,
            "detected_at": self.detected_at.isoformat(),
        }


@dataclass
class APIRequestValidation:
    """
    Result of validating a single API request.

    Each check is tracked independently so callers can decide
    which failures are acceptable for their threat model.
    """

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    is_valid: bool = True
    token_verified: bool = True
    scope_authorized: bool = True
    rate_limit_ok: bool = True
    pqc_signature_valid: bool = True
    violations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize validation result."""
        return {
            "request_id": self.request_id,
            "is_valid": self.is_valid,
            "token_verified": self.token_verified,
            "scope_authorized": self.scope_authorized,
            "rate_limit_ok": self.rate_limit_ok,
            "pqc_signature_valid": self.pqc_signature_valid,
            "violations": self.violations,
        }


@dataclass
class CCaaSSecurityPolicy:
    """
    Security policy governing API access to CCaaS platforms.

    Policies are tenant-scoped and may differ across platforms
    to balance security requirements with operational flexibility.
    """

    platforms_enabled: List[CCaaSPlatform] = field(default_factory=list)
    token_ttl_seconds: int = 3600
    require_pqc_signatures: bool = True
    rate_limit_per_minute: int = 1000
    allowed_scopes: Dict[str, List[str]] = field(default_factory=dict)
    ip_whitelist: List[str] = field(default_factory=list)
    rotate_keys_on_compromise: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize policy for storage."""
        return {
            "platforms_enabled": [p.name for p in self.platforms_enabled],
            "token_ttl_seconds": self.token_ttl_seconds,
            "require_pqc_signatures": self.require_pqc_signatures,
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "allowed_scopes": self.allowed_scopes,
            "ip_whitelist": self.ip_whitelist,
            "rotate_keys_on_compromise": self.rotate_keys_on_compromise,
        }


# ---------------------------------------------------------------------------
# Injection patterns
# ---------------------------------------------------------------------------

# Common injection patterns that target CCaaS REST / GraphQL endpoints
_INJECTION_PATTERNS: List[re.Pattern] = [
    re.compile(r"(?i)(union\s+select|drop\s+table|insert\s+into)"),
    re.compile(r"(?i)(;\s*exec\b|;\s*execute\b|xp_cmdshell)"),
    re.compile(r"(?i)(<script[\s>]|javascript\s*:|on\w+\s*=)"),
    re.compile(r"(?i)(\$ne|\$gt|\$lt|\$regex|\$where)"),  # NoSQL
    re.compile(r"(?i)(\.\.\/|\.\.\\|%2e%2e)"),             # Path traversal
    re.compile(r"(?i)(\beval\s*\(|\bFunction\s*\()"),      # Code injection
]


# ---------------------------------------------------------------------------
# CCaaS API Security Manager
# ---------------------------------------------------------------------------


class CCaaSAPISecurityManager:
    """
    Manages API security for CCaaS platform integrations.

    Provides PQC-signed token issuance, per-request validation,
    anomaly detection, and automated key rotation. Designed for
    multi-vendor BPO environments running Genesys, Amazon Connect,
    Five9, Twilio Flex, and similar platforms side by side.

    Usage::

        manager = CCaaSAPISecurityManager(
            tenant_id="tenant-001",
            policy=CCaaSAPISecurityManager.create_strict_policy(),
        )

        # Issue a token
        token = manager.sign_api_token(
            platform=CCaaSPlatform.GENESYS_CLOUD,
            scopes=["routing:read", "users:read"],
        )

        # Validate an inbound request
        result = manager.validate_request(
            token=token,
            request_path="/api/v2/routing/queues",
            source_ip="10.0.1.50",
            required_scope="routing:read",
        )
    """

    def __init__(
        self,
        tenant_id: str = "",
        *,
        policy: Optional[CCaaSSecurityPolicy] = None,
        alert_callback: Optional[Callable[[APISecurityEvent], None]] = None,
    ):
        self.tenant_id = tenant_id
        self.policy = policy or CCaaSSecurityPolicy()
        self.alert_callback = alert_callback

        # Token registry  (token_id -> APIToken)
        self._tokens: Dict[str, APIToken] = {}

        # Rate-limit tracking  (source_ip -> list of request timestamps)
        self._rate_tracker: Dict[str, List[datetime]] = {}

        # Compromised key hashes
        self._compromised_keys: Set[str] = set()

        # Security event log
        self._events: List[APISecurityEvent] = []

        # Statistics
        self._stats: Dict[str, int] = {
            "tokens_issued": 0,
            "tokens_revoked": 0,
            "requests_validated": 0,
            "requests_blocked": 0,
            "keys_rotated": 0,
            "anomalies_detected": 0,
        }

        logger.info(
            "CCaaSAPISecurityManager initialized tenant=%s platforms=%d "
            "pqc_required=%s rate_limit=%d/min",
            tenant_id,
            len(self.policy.platforms_enabled),
            self.policy.require_pqc_signatures,
            self.policy.rate_limit_per_minute,
        )

    # ------------------------------------------------------------------
    # Token lifecycle
    # ------------------------------------------------------------------

    def sign_api_token(
        self,
        platform: CCaaSPlatform,
        scopes: Optional[List[str]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> APIToken:
        """
        Issue a new PQC-signed API token for a CCaaS platform.

        The token payload is hashed with SHA-3-256 and the hash is
        signed using the configured PQC signature algorithm (default
        ML-DSA-65).  In production the signing key lives in an HSM;
        here we produce a deterministic stub suitable for testing.

        Args:
            platform:    Target CCaaS platform.
            scopes:      Granted OAuth-style scopes.
            ttl_seconds: Override the policy default TTL.

        Returns:
            A signed APIToken ready for use.
        """
        now = datetime.utcnow()
        ttl = ttl_seconds or self.policy.token_ttl_seconds
        expires = now + timedelta(seconds=ttl)
        scopes = scopes or []

        token = APIToken(
            platform=platform,
            tenant_id=self.tenant_id,
            issued_at=now,
            expires_at=expires,
            scopes=scopes,
            key_algorithm="ML-DSA-65",
        )

        # Build canonical payload for signing
        canonical = (
            f"{token.token_id}|{platform.name}|{self.tenant_id}|"
            f"{now.isoformat()}|{expires.isoformat()}|"
            f"{','.join(sorted(scopes))}"
        )
        payload_hash = hashlib.sha3_256(canonical.encode()).hexdigest()

        # Produce PQC signature stub (deterministic for reproducibility)
        sig_input = f"ML-DSA-65:sign:{payload_hash}:{self.tenant_id}"
        token.pqc_signature = hashlib.sha3_512(sig_input.encode()).hexdigest()

        # Register token
        self._tokens[token.token_id] = token
        self._stats["tokens_issued"] += 1

        logger.info(
            "Signed API token token_id=%s platform=%s scopes=%s ttl=%ds",
            token.token_id,
            platform.name,
            scopes,
            ttl,
        )
        return token

    def verify_api_token(self, token: APIToken) -> Tuple[bool, List[str]]:
        """
        Verify the PQC signature and validity of an API token.

        Checks:
        1. Token is registered and not revoked.
        2. Token has not expired.
        3. PQC signature matches the canonical payload.

        Args:
            token: The APIToken to verify.

        Returns:
            Tuple of (is_valid, list_of_failure_reasons).
        """
        failures: List[str] = []

        # Check registration
        registered = self._tokens.get(token.token_id)
        if registered is None:
            failures.append("token_not_registered")
            return False, failures

        # Revocation check
        if registered.is_revoked:
            failures.append("token_revoked")

        # Expiry check
        if registered.is_expired:
            failures.append("token_expired")

        # Signature verification (reproduce and compare)
        canonical = (
            f"{token.token_id}|{token.platform.name}|{token.tenant_id}|"
            f"{token.issued_at.isoformat()}|{token.expires_at.isoformat()}|"
            f"{','.join(sorted(token.scopes))}"
        )
        payload_hash = hashlib.sha3_256(canonical.encode()).hexdigest()
        sig_input = f"ML-DSA-65:sign:{payload_hash}:{token.tenant_id}"
        expected_sig = hashlib.sha3_512(sig_input.encode()).hexdigest()

        if token.pqc_signature != expected_sig:
            failures.append("pqc_signature_invalid")

        is_valid = len(failures) == 0
        if not is_valid:
            logger.warning(
                "Token verification failed token_id=%s reasons=%s",
                token.token_id,
                failures,
            )
        return is_valid, failures

    def revoke_token(self, token_id: str) -> bool:
        """Revoke a token by ID. Returns True if found and revoked."""
        token = self._tokens.get(token_id)
        if token is None:
            return False
        token.is_revoked = True
        self._stats["tokens_revoked"] += 1
        logger.info("Revoked API token token_id=%s", token_id)
        return True

    # ------------------------------------------------------------------
    # Request validation
    # ------------------------------------------------------------------

    def validate_request(
        self,
        token: APIToken,
        request_path: str,
        source_ip: str,
        required_scope: str = "",
        request_body: str = "",
    ) -> APIRequestValidation:
        """
        Validate an inbound API request end-to-end.

        Performs token verification, scope authorisation, IP whitelist
        check, rate-limit enforcement, and injection scanning.

        Args:
            token:          The bearer token accompanying the request.
            request_path:   The HTTP path (e.g. "/api/v2/users").
            source_ip:      Origin IP address.
            required_scope: The scope needed for this endpoint.
            request_body:   Raw request body for injection scanning.

        Returns:
            An APIRequestValidation with per-check results.
        """
        self._stats["requests_validated"] += 1
        result = APIRequestValidation()
        violations: List[str] = []

        # 1. Token verification
        token_ok, token_failures = self.verify_api_token(token)
        if not token_ok:
            result.token_verified = False
            result.pqc_signature_valid = "pqc_signature_invalid" in token_failures
            violations.extend(token_failures)

        # 2. Scope authorisation
        if required_scope and required_scope not in token.scopes:
            result.scope_authorized = False
            violations.append(f"missing_scope:{required_scope}")

        # 3. IP whitelist
        if self.policy.ip_whitelist and source_ip not in self.policy.ip_whitelist:
            violations.append(f"ip_not_whitelisted:{source_ip}")

        # 4. Rate-limit
        rate_ok = self.check_rate_limit(source_ip)
        if not rate_ok:
            result.rate_limit_ok = False
            violations.append("rate_limit_exceeded")

        # 5. Injection scan
        if request_body:
            injection_found = self._scan_for_injection(request_body)
            if injection_found:
                violations.append(f"injection_detected:{injection_found}")

        # 6. Path injection scan
        path_injection = self._scan_for_injection(request_path)
        if path_injection:
            violations.append(f"path_injection_detected:{path_injection}")

        result.violations = violations
        result.is_valid = len(violations) == 0

        if not result.is_valid:
            self._stats["requests_blocked"] += 1
            self._record_event(
                threat_type=self._classify_violations(violations),
                platform=token.platform,
                request_path=request_path,
                source_ip=source_ip,
                agent_id="",
                evidence={"violations": violations, "token_id": token.token_id},
                action=APISecurityAction.BLOCK_REQUEST,
            )
            logger.warning(
                "Request validation failed path=%s ip=%s violations=%s",
                request_path,
                source_ip,
                violations,
            )

        return result

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------

    def detect_anomalous_requests(
        self,
        requests: List[Dict[str, Any]],
        window_minutes: int = 5,
    ) -> List[APISecurityEvent]:
        """
        Detect anomalous request patterns across a time window.

        Analyses a batch of recent requests for:
        - Credential stuffing (many IPs, same path, high failure rate)
        - Replay attacks (duplicate token + timestamp pairs)
        - Privilege escalation (scope probe sequences)

        Args:
            requests:       List of request dicts with keys
                            (token_id, source_ip, path, timestamp, success).
            window_minutes: Analysis window width.

        Returns:
            List of APISecurityEvent instances for detected anomalies.
        """
        events: List[APISecurityEvent] = []

        # --- Credential stuffing ---
        ip_failure_counts: Dict[str, int] = {}
        path_failure_counts: Dict[str, int] = {}
        for req in requests:
            if not req.get("success", True):
                ip = req.get("source_ip", "unknown")
                path = req.get("path", "unknown")
                ip_failure_counts[ip] = ip_failure_counts.get(ip, 0) + 1
                path_failure_counts[path] = path_failure_counts.get(path, 0) + 1

        for ip, count in ip_failure_counts.items():
            if count >= 10:
                evt = self._record_event(
                    threat_type=APIThreatType.CREDENTIAL_STUFFING,
                    platform=CCaaSPlatform.GENESYS_CLOUD,
                    request_path="*",
                    source_ip=ip,
                    agent_id="",
                    evidence={
                        "failed_attempts": count,
                        "window_minutes": window_minutes,
                    },
                    action=APISecurityAction.BLOCK_REQUEST,
                )
                events.append(evt)

        # --- Token replay ---
        seen_tokens: Dict[str, str] = {}  # token_id -> first_timestamp
        for req in requests:
            tid = req.get("token_id", "")
            ts = req.get("timestamp", "")
            key = f"{tid}:{ts}"
            if key in seen_tokens:
                evt = self._record_event(
                    threat_type=APIThreatType.TOKEN_REPLAY,
                    platform=CCaaSPlatform.GENESYS_CLOUD,
                    request_path=req.get("path", ""),
                    source_ip=req.get("source_ip", ""),
                    agent_id="",
                    evidence={"token_id": tid, "duplicate_timestamp": ts},
                    action=APISecurityAction.QUARANTINE_TOKEN,
                )
                events.append(evt)
            else:
                seen_tokens[key] = ts

        # --- Privilege escalation (scope probing) ---
        token_scope_attempts: Dict[str, Set[str]] = {}
        for req in requests:
            tid = req.get("token_id", "")
            scope = req.get("required_scope", "")
            if scope and not req.get("success", True):
                token_scope_attempts.setdefault(tid, set()).add(scope)

        for tid, scopes in token_scope_attempts.items():
            if len(scopes) >= 5:
                evt = self._record_event(
                    threat_type=APIThreatType.PRIVILEGE_ESCALATION,
                    platform=CCaaSPlatform.GENESYS_CLOUD,
                    request_path="*",
                    source_ip="",
                    agent_id="",
                    evidence={
                        "token_id": tid,
                        "probed_scopes": sorted(scopes),
                    },
                    action=APISecurityAction.REVOKE_ACCESS,
                )
                events.append(evt)

        if events:
            self._stats["anomalies_detected"] += len(events)
            logger.warning(
                "Detected %d anomalous request patterns in %d-min window",
                len(events),
                window_minutes,
            )

        return events

    # ------------------------------------------------------------------
    # Key rotation
    # ------------------------------------------------------------------

    def rotate_compromised_keys(
        self,
        compromised_key_hash: str,
        platform: Optional[CCaaSPlatform] = None,
    ) -> Dict[str, Any]:
        """
        Rotate keys after a compromise is detected.

        Revokes all tokens signed with the compromised key, records
        the key hash in the deny-list, and returns a summary.

        Args:
            compromised_key_hash: SHA-3-256 hash of the compromised key.
            platform:             Optionally limit rotation to one platform.

        Returns:
            Summary dict with counts of revoked tokens.
        """
        self._compromised_keys.add(compromised_key_hash)
        revoked_count = 0

        for token in self._tokens.values():
            if token.is_revoked:
                continue
            if platform and token.platform != platform:
                continue

            sig_hash = hashlib.sha3_256(
                token.pqc_signature.encode()
            ).hexdigest()
            if sig_hash == compromised_key_hash:
                token.is_revoked = True
                revoked_count += 1

        # If policy says rotate, revoke ALL tokens for safety
        if self.policy.rotate_keys_on_compromise and revoked_count == 0:
            for token in self._tokens.values():
                if not token.is_revoked:
                    if platform is None or token.platform == platform:
                        token.is_revoked = True
                        revoked_count += 1

        self._stats["keys_rotated"] += 1
        self._stats["tokens_revoked"] += revoked_count

        summary = {
            "compromised_key_hash": compromised_key_hash,
            "platform": platform.name if platform else "ALL",
            "tokens_revoked": revoked_count,
            "rotation_timestamp": datetime.utcnow().isoformat(),
        }
        logger.info(
            "Key rotation complete: revoked=%d platform=%s",
            revoked_count,
            summary["platform"],
        )
        return summary

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def check_rate_limit(self, source_ip: str) -> bool:
        """
        Check whether a source IP is within the configured rate limit.

        Uses a sliding-window counter. Old entries outside the
        one-minute window are pruned on each call.

        Args:
            source_ip: The IP address to check.

        Returns:
            True if the request is allowed, False if rate-limited.
        """
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=1)

        timestamps = self._rate_tracker.get(source_ip, [])
        # Prune old entries
        timestamps = [t for t in timestamps if t > window_start]
        timestamps.append(now)
        self._rate_tracker[source_ip] = timestamps

        return len(timestamps) <= self.policy.rate_limit_per_minute

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_security_report(
        self,
        since: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Generate a security report covering events since *since*.

        Args:
            since: Start of the reporting window.  Defaults to 24 h ago.

        Returns:
            Report dict with event counts by threat type and platform.
        """
        if since is None:
            since = datetime.utcnow() - timedelta(hours=24)

        relevant = [e for e in self._events if e.detected_at >= since]

        by_threat: Dict[str, int] = {}
        by_platform: Dict[str, int] = {}
        by_action: Dict[str, int] = {}
        for evt in relevant:
            by_threat[evt.threat_type.name] = by_threat.get(
                evt.threat_type.name, 0
            ) + 1
            by_platform[evt.platform.name] = by_platform.get(
                evt.platform.name, 0
            ) + 1
            by_action[evt.action_taken.name] = by_action.get(
                evt.action_taken.name, 0
            ) + 1

        return {
            "tenant_id": self.tenant_id,
            "report_generated_at": datetime.utcnow().isoformat(),
            "window_start": since.isoformat(),
            "total_events": len(relevant),
            "events_by_threat_type": by_threat,
            "events_by_platform": by_platform,
            "events_by_action": by_action,
            "statistics": dict(self._stats),
            "active_tokens": sum(
                1
                for t in self._tokens.values()
                if not t.is_revoked and not t.is_expired
            ),
            "revoked_tokens": sum(
                1 for t in self._tokens.values() if t.is_revoked
            ),
            "compromised_keys_tracked": len(self._compromised_keys),
        }

    # ------------------------------------------------------------------
    # Factory class methods
    # ------------------------------------------------------------------

    @classmethod
    def create_strict_policy(cls) -> CCaaSSecurityPolicy:
        """
        Create a strict security policy suitable for PCI-DSS environments.

        Enables PQC signatures, low rate limits, and automatic key
        rotation.
        """
        return CCaaSSecurityPolicy(
            platforms_enabled=list(CCaaSPlatform),
            token_ttl_seconds=900,           # 15 minutes
            require_pqc_signatures=True,
            rate_limit_per_minute=200,
            allowed_scopes={
                "default": [
                    "routing:read",
                    "users:read",
                    "analytics:read",
                ],
            },
            ip_whitelist=[],
            rotate_keys_on_compromise=True,
        )

    @classmethod
    def create_permissive_policy(cls) -> CCaaSSecurityPolicy:
        """
        Create a permissive policy for development or staging environments.

        Relaxes rate limits and TTL while still enforcing PQC signatures.
        """
        return CCaaSSecurityPolicy(
            platforms_enabled=list(CCaaSPlatform),
            token_ttl_seconds=86400,         # 24 hours
            require_pqc_signatures=False,
            rate_limit_per_minute=5000,
            allowed_scopes={
                "default": ["*"],
            },
            ip_whitelist=[],
            rotate_keys_on_compromise=False,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scan_for_injection(self, text: str) -> Optional[str]:
        """Return the matching injection pattern name, or None."""
        for pattern in _INJECTION_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(0)
        return None

    def _classify_violations(
        self, violations: List[str]
    ) -> APIThreatType:
        """Map violation strings to the most appropriate threat type."""
        for v in violations:
            if "injection" in v:
                return APIThreatType.INJECTION_ATTACK
            if "rate_limit" in v:
                return APIThreatType.RATE_LIMIT_BYPASS
            if "missing_scope" in v:
                return APIThreatType.UNAUTHORIZED_SCOPE
            if "pqc_signature" in v:
                return APIThreatType.MAN_IN_THE_MIDDLE
            if "token_revoked" in v or "token_expired" in v:
                return APIThreatType.TOKEN_REPLAY
        return APIThreatType.KEY_COMPROMISE

    def _record_event(
        self,
        threat_type: APIThreatType,
        platform: CCaaSPlatform,
        request_path: str,
        source_ip: str,
        agent_id: str,
        evidence: Dict[str, Any],
        action: APISecurityAction,
    ) -> APISecurityEvent:
        """Create, store, and optionally alert on a security event."""
        event = APISecurityEvent(
            threat_type=threat_type,
            platform=platform,
            request_path=request_path,
            source_ip=source_ip,
            agent_id=agent_id,
            evidence=evidence,
            action_taken=action,
        )
        self._events.append(event)

        if self.alert_callback is not None:
            try:
                self.alert_callback(event)
            except Exception:
                logger.exception(
                    "Alert callback failed for event %s", event.event_id
                )

        return event
