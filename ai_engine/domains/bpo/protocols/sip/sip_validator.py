"""
SIP Message Validator for BPO Security

Validates SIP messages for call center security including:
- Structure validation (RFC 3261 compliance)
- Header injection attack detection
- Toll fraud prevention (premium rate numbers, international calls)
- Rate limiting validation
- PCI-DSS compliance checks (cardholder data in SIP headers)
- Call duration limit enforcement
- Agent authorization checks
- DTMF relay validation (INFO method with DTMF signals)
"""

import re
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from ai_engine.domains.banking.protocols.validators.base_validator import (
    BaseValidator,
    ValidationResult,
    ValidationSeverity,
)
from ai_engine.domains.bpo.protocols.sip.sip_message import (
    SIPMessage,
    SIPHeader,
    SIPRequestLine,
    SIPResponseLine,
    MessageType,
)
from ai_engine.domains.bpo.protocols.sip.sip_codes import (
    SIPMethod,
    SIPResponseCode,
    SIPHeaderName,
    SIPSecurityProfile,
    PREMIUM_RATE_PREFIXES,
    AGENT_ALLOWED_METHODS,
    CALL_CENTER_SIP_HEADERS,
)


class SIPValidator(BaseValidator):
    """
    Validator for SIP messages in BPO/Call Center environments.

    Validates:
    - Message structure (start line, required headers, body)
    - Header injection attacks (CRLF injection, header smuggling)
    - Toll fraud detection (premium rate numbers, IRSF)
    - Rate limiting per source IP and per agent
    - PCI-DSS compliance (no cardholder data in SIP signaling)
    - Call duration limits
    - Agent authorization (method restrictions, transfer permissions)
    - DTMF relay validation (signal validity, duration bounds)
    - URI validation and request target verification
    """

    # Patterns for security checks
    CRLF_INJECTION_PATTERN = re.compile(r"[\r\n]")
    NULL_BYTE_PATTERN = re.compile(r"\x00")
    SQL_INJECTION_PATTERN = re.compile(
        r"(?:--|;|\b(?:SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|CREATE|EXEC)\b)",
        re.IGNORECASE,
    )
    COMMAND_INJECTION_PATTERN = re.compile(
        r"[|;`$]|\.\./|&&|\|\|",
    )
    HEADER_SMUGGLING_PATTERN = re.compile(
        r"(?:\r\n|\n)(?:Via|From|To|Call-ID|CSeq|Contact|Route|Record-Route)\s*:",
        re.IGNORECASE,
    )

    # PCI-DSS: patterns that match cardholder data
    # Credit card number patterns (Visa, MC, Amex, Discover, etc.)
    CARD_NUMBER_PATTERN = re.compile(
        r"\b(?:"
        r"4\d{3}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}"  # Visa
        r"|5[1-5]\d{2}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}"  # Mastercard
        r"|3[47]\d{2}[\s-]?\d{6}[\s-]?\d{5}"  # Amex
        r"|6(?:011|5\d{2})[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}"  # Discover
        r"|3(?:0[0-5]|[68]\d)\d[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{2}"  # Diners
        r")\b"
    )
    # CVV/CVC pattern (3-4 digits standalone)
    CVV_PATTERN = re.compile(r"\b(?:cvv|cvc|cvv2|cvc2|cid)\s*[:=]\s*\d{3,4}\b", re.IGNORECASE)
    # SSN pattern
    SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

    # Phone number patterns for toll fraud detection
    PHONE_PATTERN = re.compile(r"(?:\+?1?\d{7,15})")

    # Valid DTMF signals
    VALID_DTMF_SIGNALS = set("0123456789*#ABCD")
    DTMF_MAX_DURATION = 10000  # Max duration in ms
    DTMF_MIN_DURATION = 40     # Min duration in ms

    # SIP URI validation
    SIP_URI_PATTERN = re.compile(r"^sips?:[^@\s]+@[^@\s;>]+")

    # Max header sizes (bytes)
    MAX_HEADER_VALUE_LENGTH = 8192
    MAX_HEADER_COUNT = 200
    MAX_VIA_COUNT = 70  # RFC 3261 recommends Max-Forwards of 70
    MAX_BODY_SIZE = 65536  # 64KB
    MAX_URI_LENGTH = 2048

    def __init__(
        self,
        strict: bool = True,
        security_profile: Optional[SIPSecurityProfile] = None,
        allowed_agents: Optional[Set[str]] = None,
        allowed_tenants: Optional[Set[str]] = None,
        rate_limit_window: int = 60,
    ):
        """
        Initialize the SIP validator.

        Args:
            strict: If True, treat warnings as errors
            security_profile: SIP security profile with toll fraud
                              and transport settings
            allowed_agents: Set of authorized agent IDs (None = no check)
            allowed_tenants: Set of authorized tenant IDs (None = no check)
            rate_limit_window: Window in seconds for rate limiting
        """
        super().__init__(strict)
        self.security_profile = security_profile or SIPSecurityProfile()
        self.allowed_agents = allowed_agents
        self.allowed_tenants = allowed_tenants
        self.rate_limit_window = rate_limit_window

        # Rate limiting state
        self._request_counts: Dict[str, List[float]] = defaultdict(list)

    @property
    def name(self) -> str:
        return "SIPValidator"

    @property
    def version(self) -> str:
        return "1.0"

    def validate(self, data: Any) -> ValidationResult:
        """
        Validate a SIP message for security and compliance.

        Args:
            data: SIPMessage object or dictionary

        Returns:
            ValidationResult with any errors or warnings
        """
        result = self._create_result()

        if isinstance(data, SIPMessage):
            self._validate_sip_message(data, result)
        elif isinstance(data, dict):
            self._validate_dict(data, result)
        else:
            result.add_error(
                "SIP_INVALID_INPUT",
                "Input must be a SIPMessage object or dictionary",
                severity=ValidationSeverity.CRITICAL,
            )

        return result

    # ------------------------------------------------------------------
    # Top-level validation orchestration
    # ------------------------------------------------------------------

    def _validate_sip_message(self, msg: SIPMessage, result: ValidationResult) -> None:
        """Run all validation checks on a SIPMessage."""
        # 1. Structure validation
        self._validate_structure(msg, result)

        # 2. Header injection / smuggling detection
        self._validate_header_security(msg, result)

        # 3. URI validation
        self._validate_uri_security(msg, result)

        # 4. Toll fraud detection
        self._validate_toll_fraud(msg, result)

        # 5. Rate limiting
        self._validate_rate_limit(msg, result)

        # 6. PCI-DSS compliance
        self._validate_pci_compliance(msg, result)

        # 7. Call duration limits
        self._validate_call_duration(msg, result)

        # 8. Agent authorization
        self._validate_agent_authorization(msg, result)

        # 9. Tenant authorization
        self._validate_tenant_authorization(msg, result)

        # 10. DTMF relay validation (for INFO messages)
        self._validate_dtmf(msg, result)

        # 11. SDP body security
        if msg.sdp:
            self._validate_sdp_security(msg, result)

    # ------------------------------------------------------------------
    # 1. Structure validation
    # ------------------------------------------------------------------

    def _validate_structure(self, msg: SIPMessage, result: ValidationResult) -> None:
        """Validate SIP message structure per RFC 3261."""
        # Start line
        if msg.is_request:
            if not msg.request_line or not msg.request_line.is_valid:
                result.add_error(
                    "SIP_INVALID_REQUEST_LINE",
                    "Request line is missing or malformed",
                    severity=ValidationSeverity.CRITICAL,
                )
            else:
                # Validate SIP version
                if msg.request_line.sip_version != "SIP/2.0":
                    result.add_error(
                        "SIP_INVALID_VERSION",
                        f"Unsupported SIP version: {msg.request_line.sip_version}",
                        field="sip_version",
                    )

                # Validate method is known
                if msg.request_line.method not in {m.method for m in SIPMethod}:
                    result.add_warning(
                        "SIP_UNKNOWN_METHOD",
                        f"Unknown SIP method: {msg.request_line.method}",
                        field="method",
                    )

                # Validate Request-URI
                if len(msg.request_line.request_uri) > self.MAX_URI_LENGTH:
                    result.add_error(
                        "SIP_URI_TOO_LONG",
                        f"Request-URI exceeds maximum length ({self.MAX_URI_LENGTH})",
                        field="request_uri",
                    )

        elif msg.is_response:
            if not msg.response_line or not msg.response_line.is_valid:
                result.add_error(
                    "SIP_INVALID_RESPONSE_LINE",
                    "Status line is missing or malformed",
                    severity=ValidationSeverity.CRITICAL,
                )
            else:
                # Validate status code range
                if not (100 <= msg.response_line.status_code <= 699):
                    result.add_error(
                        "SIP_INVALID_STATUS_CODE",
                        f"Status code out of range: {msg.response_line.status_code}",
                        field="status_code",
                    )

        # Required headers
        if msg.parse_errors:
            for err in msg.parse_errors:
                if "Missing required header" in err:
                    result.add_error(
                        "SIP_MISSING_REQUIRED_HEADER",
                        err,
                        severity=ValidationSeverity.CRITICAL,
                    )

        # Header count limits
        if len(msg.headers) > self.MAX_HEADER_COUNT:
            result.add_error(
                "SIP_TOO_MANY_HEADERS",
                f"Message contains {len(msg.headers)} headers (max {self.MAX_HEADER_COUNT})",
                severity=ValidationSeverity.ERROR,
            )

        # Via count limit
        via_count = len(msg.via_headers)
        if via_count > self.MAX_VIA_COUNT:
            result.add_error(
                "SIP_TOO_MANY_VIA",
                f"Message contains {via_count} Via headers (max {self.MAX_VIA_COUNT}), possible loop",
                severity=ValidationSeverity.CRITICAL,
            )

        # Max-Forwards validation
        max_fwd = msg.max_forwards
        if max_fwd is not None and max_fwd <= 0:
            result.add_error(
                "SIP_MAX_FORWARDS_ZERO",
                "Max-Forwards is zero or negative, message should be rejected",
                field="Max-Forwards",
                severity=ValidationSeverity.CRITICAL,
            )

        # Body size check
        if msg.body and len(msg.body) > self.MAX_BODY_SIZE:
            result.add_error(
                "SIP_BODY_TOO_LARGE",
                f"Message body exceeds maximum size ({len(msg.body)} > {self.MAX_BODY_SIZE})",
                field="body",
            )

        # Content-Length consistency
        if msg.body:
            declared_length = msg.content_length
            actual_length = len(msg.body.encode("utf-8"))
            if declared_length > 0 and abs(declared_length - actual_length) > 1:
                result.add_warning(
                    "SIP_CONTENT_LENGTH_MISMATCH",
                    f"Content-Length ({declared_length}) does not match actual body size ({actual_length})",
                    field="Content-Length",
                )

        # Call-ID presence
        if not msg.call_id:
            result.add_error(
                "SIP_MISSING_CALL_ID",
                "Call-ID header is required",
                field="Call-ID",
                severity=ValidationSeverity.CRITICAL,
            )

    # ------------------------------------------------------------------
    # 2. Header injection / smuggling detection
    # ------------------------------------------------------------------

    def _validate_header_security(self, msg: SIPMessage, result: ValidationResult) -> None:
        """Detect header injection and smuggling attacks."""
        for header in msg.headers:
            header_ref = f"header:{header.name}"

            # Check header name for injection characters
            if self.CRLF_INJECTION_PATTERN.search(header.name):
                result.add_error(
                    "SIP_HEADER_INJECTION",
                    f"CRLF injection detected in header name: {repr(header.name)}",
                    field=header_ref,
                    severity=ValidationSeverity.CRITICAL,
                )

            # Check header value for CRLF injection
            # Note: multi-line folding should already be resolved by parser
            if self.CRLF_INJECTION_PATTERN.search(header.value):
                result.add_error(
                    "SIP_HEADER_INJECTION",
                    f"CRLF injection detected in {header.name} header value",
                    field=header_ref,
                    severity=ValidationSeverity.CRITICAL,
                )

            # Check for null bytes
            if self.NULL_BYTE_PATTERN.search(header.value):
                result.add_error(
                    "SIP_NULL_BYTE_INJECTION",
                    f"Null byte detected in {header.name} header value",
                    field=header_ref,
                    severity=ValidationSeverity.CRITICAL,
                )

            # Check header value length
            if len(header.value) > self.MAX_HEADER_VALUE_LENGTH:
                result.add_error(
                    "SIP_HEADER_TOO_LONG",
                    f"Header {header.name} value exceeds maximum length "
                    f"({len(header.value)} > {self.MAX_HEADER_VALUE_LENGTH})",
                    field=header_ref,
                )

            # Check for header smuggling (embedded headers in values)
            if self.HEADER_SMUGGLING_PATTERN.search(header.value):
                result.add_error(
                    "SIP_HEADER_SMUGGLING",
                    f"Possible header smuggling detected in {header.name} value",
                    field=header_ref,
                    severity=ValidationSeverity.CRITICAL,
                )

            # Check for SQL injection in header values
            if self.SQL_INJECTION_PATTERN.search(header.value):
                result.add_warning(
                    "SIP_SQL_INJECTION_SUSPECT",
                    f"Possible SQL injection pattern detected in {header.name} header",
                    field=header_ref,
                )

            # Check for command injection
            if self.COMMAND_INJECTION_PATTERN.search(header.value):
                result.add_warning(
                    "SIP_COMMAND_INJECTION_SUSPECT",
                    f"Possible command injection pattern detected in {header.name} header",
                    field=header_ref,
                )

        # Check for duplicate critical headers (should be unique)
        unique_headers = {"Call-ID", "CSeq", "From", "To", "Max-Forwards", "Content-Length"}
        for hdr_name in unique_headers:
            matches = msg.get_headers(hdr_name)
            if len(matches) > 1:
                result.add_error(
                    "SIP_DUPLICATE_UNIQUE_HEADER",
                    f"Header {hdr_name} appears {len(matches)} times but must be unique",
                    field=f"header:{hdr_name}",
                )

    # ------------------------------------------------------------------
    # 3. URI validation
    # ------------------------------------------------------------------

    def _validate_uri_security(self, msg: SIPMessage, result: ValidationResult) -> None:
        """Validate SIP URIs for security issues."""
        uris_to_check: List[tuple] = []

        if msg.is_request and msg.request_line:
            uris_to_check.append(("Request-URI", msg.request_line.request_uri))

        if msg.from_header:
            uris_to_check.append(("From", msg.from_header.uri))

        if msg.to_header:
            uris_to_check.append(("To", msg.to_header.uri))

        contact_value = msg.contact
        if contact_value and contact_value != "*":
            uris_to_check.append(("Contact", contact_value))

        for field_name, uri in uris_to_check:
            if not uri:
                continue

            # Check for null bytes in URIs
            if self.NULL_BYTE_PATTERN.search(uri):
                result.add_error(
                    "SIP_URI_NULL_BYTE",
                    f"Null byte in {field_name} URI: potential bypass attack",
                    field=field_name,
                    severity=ValidationSeverity.CRITICAL,
                )

            # Check for directory traversal in URIs
            if "../" in uri or "..\\" in uri:
                result.add_error(
                    "SIP_URI_TRAVERSAL",
                    f"Directory traversal pattern in {field_name} URI",
                    field=field_name,
                    severity=ValidationSeverity.CRITICAL,
                )

            # Check URI length
            if len(uri) > self.MAX_URI_LENGTH:
                result.add_error(
                    "SIP_URI_TOO_LONG",
                    f"{field_name} URI exceeds maximum length ({len(uri)} > {self.MAX_URI_LENGTH})",
                    field=field_name,
                )

    # ------------------------------------------------------------------
    # 4. Toll fraud detection
    # ------------------------------------------------------------------

    def _validate_toll_fraud(self, msg: SIPMessage, result: ValidationResult) -> None:
        """Detect toll fraud attempts (premium rate, IRSF, international)."""
        if not msg.is_request:
            return

        # Only check INVITE (call initiation) and REFER (call transfer)
        method = msg.method
        if method not in ("INVITE", "REFER"):
            return

        # Extract the destination number
        target_uri = None
        if method == "INVITE" and msg.request_line:
            target_uri = msg.request_line.request_uri
        elif method == "REFER":
            refer_to = msg.get_header_value("Refer-To")
            if refer_to:
                target_uri = refer_to.strip().strip("<>")

        if not target_uri:
            return

        # Extract phone number from URI
        from ai_engine.domains.bpo.protocols.sip.sip_parser import SIPParser
        parser = SIPParser(strict=False)
        phone_number = parser.extract_phone_number(target_uri)

        if not phone_number:
            return

        # Normalize: strip leading + and country code variations
        normalized = phone_number.lstrip("+")

        # Check premium rate prefixes
        if self.security_profile.block_premium_rate:
            for region, prefixes in PREMIUM_RATE_PREFIXES.items():
                for prefix in prefixes:
                    if normalized.startswith(prefix):
                        result.add_error(
                            "SIP_TOLL_FRAUD_PREMIUM",
                            f"Call to premium rate number detected ({region} prefix {prefix}): "
                            f"{self._mask_number(phone_number)}",
                            field="Request-URI",
                            severity=ValidationSeverity.CRITICAL,
                        )
                        return

        # Check international calls
        if self.security_profile.block_international:
            if phone_number.startswith("+") and len(normalized) > 10:
                # Check against allowed country codes
                allowed = self.security_profile.allowed_country_codes or []
                if allowed:
                    country_match = False
                    for cc in allowed:
                        if normalized.startswith(cc):
                            country_match = True
                            break
                    if not country_match:
                        result.add_error(
                            "SIP_TOLL_FRAUD_INTERNATIONAL",
                            f"International call to unauthorized country: "
                            f"{self._mask_number(phone_number)}",
                            field="Request-URI",
                            severity=ValidationSeverity.CRITICAL,
                        )
                else:
                    result.add_error(
                        "SIP_TOLL_FRAUD_INTERNATIONAL",
                        f"International call blocked (no allowed countries configured): "
                        f"{self._mask_number(phone_number)}",
                        field="Request-URI",
                    )

        # Check for unusually long numbers (IRSF indicator)
        if len(normalized) > 15:
            result.add_warning(
                "SIP_SUSPICIOUS_NUMBER_LENGTH",
                f"Unusually long phone number ({len(normalized)} digits): "
                f"possible IRSF attempt",
                field="Request-URI",
            )

    def _mask_number(self, number: str) -> str:
        """Mask a phone number for safe logging (show first 4 and last 2)."""
        if len(number) <= 6:
            return "***"
        return f"{number[:4]}{'*' * (len(number) - 6)}{number[-2:]}"

    # ------------------------------------------------------------------
    # 5. Rate limiting
    # ------------------------------------------------------------------

    def _validate_rate_limit(self, msg: SIPMessage, result: ValidationResult) -> None:
        """Validate request rate against configured limits."""
        if not msg.is_request:
            return

        rate_limit = self.security_profile.rate_limit_per_second
        if rate_limit <= 0:
            return

        # Determine rate limit key (source IP or agent ID)
        rate_key = msg.source_ip or msg.bpo_context.agent_id or msg.call_id or "unknown"

        now = time.time()
        window_start = now - self.rate_limit_window

        # Clean old entries
        self._request_counts[rate_key] = [
            ts for ts in self._request_counts[rate_key]
            if ts > window_start
        ]

        # Check rate
        current_count = len(self._request_counts[rate_key])
        max_in_window = rate_limit * self.rate_limit_window

        if current_count >= max_in_window:
            result.add_error(
                "SIP_RATE_LIMIT_EXCEEDED",
                f"Rate limit exceeded for {rate_key}: "
                f"{current_count} requests in {self.rate_limit_window}s "
                f"(limit: {max_in_window})",
                severity=ValidationSeverity.CRITICAL,
            )
        elif current_count >= max_in_window * 0.8:
            result.add_warning(
                "SIP_RATE_LIMIT_WARNING",
                f"Rate limit approaching for {rate_key}: "
                f"{current_count}/{max_in_window} in {self.rate_limit_window}s",
            )

        # Record this request
        self._request_counts[rate_key].append(now)

    # ------------------------------------------------------------------
    # 6. PCI-DSS compliance
    # ------------------------------------------------------------------

    def _validate_pci_compliance(self, msg: SIPMessage, result: ValidationResult) -> None:
        """
        Validate PCI-DSS compliance in SIP signaling.

        PCI-DSS Requirement 4: Encrypt transmission of cardholder data.
        Cardholder data must never appear in SIP headers or unencrypted
        signaling paths. This check detects accidental data leakage.
        """
        # Check all headers for cardholder data
        for header in msg.headers:
            self._check_pci_data(
                header.value,
                f"header:{header.name}",
                result,
            )
            # Also check parameters
            for param_key, param_value in header.parameters.items():
                self._check_pci_data(
                    param_value,
                    f"header:{header.name};{param_key}",
                    result,
                )

        # Check Request-URI for cardholder data
        if msg.is_request and msg.request_line:
            self._check_pci_data(
                msg.request_line.request_uri,
                "Request-URI",
                result,
            )

        # Check body for cardholder data (except when PCI mode is paused)
        if msg.body and not msg.bpo_context.is_pci_paused:
            self._check_pci_data(msg.body, "body", result)

        # Validate PCI mode transitions
        pci_mode = msg.bpo_context.pci_mode
        if pci_mode and pci_mode.upper() not in ("ACTIVE", "PAUSED", ""):
            result.add_error(
                "SIP_PCI_INVALID_MODE",
                f"Invalid PCI-Mode value: {pci_mode} (expected ACTIVE or PAUSED)",
                field="X-CC-PCI-Mode",
            )

    def _check_pci_data(
        self, value: str, field_name: str, result: ValidationResult
    ) -> None:
        """Check a value for cardholder data patterns."""
        if not value:
            return

        # Credit card numbers
        if self.CARD_NUMBER_PATTERN.search(value):
            result.add_error(
                "SIP_PCI_CARD_NUMBER",
                f"Possible credit card number detected in {field_name}. "
                f"PCI-DSS prohibits cardholder data in SIP signaling.",
                field=field_name,
                severity=ValidationSeverity.CRITICAL,
            )

        # CVV/CVC
        if self.CVV_PATTERN.search(value):
            result.add_error(
                "SIP_PCI_CVV",
                f"Possible CVV/CVC data detected in {field_name}. "
                f"PCI-DSS prohibits sensitive authentication data in SIP signaling.",
                field=field_name,
                severity=ValidationSeverity.CRITICAL,
            )

        # SSN (additional PII check)
        if self.SSN_PATTERN.search(value):
            result.add_warning(
                "SIP_PII_SSN",
                f"Possible SSN detected in {field_name}. "
                f"Sensitive PII should not appear in SIP signaling.",
                field=field_name,
            )

    # ------------------------------------------------------------------
    # 7. Call duration limits
    # ------------------------------------------------------------------

    def _validate_call_duration(self, msg: SIPMessage, result: ValidationResult) -> None:
        """Validate call duration related headers and settings."""
        # Check Session-Expires header (RFC 4028)
        session_expires = msg.get_header_value("Session-Expires")
        if session_expires:
            try:
                # Format: duration;refresher=uac or just duration
                duration_str = session_expires.split(";")[0].strip()
                duration = int(duration_str)

                max_duration = self.security_profile.max_call_duration_seconds
                if duration > max_duration:
                    result.add_error(
                        "SIP_SESSION_DURATION_EXCEEDED",
                        f"Session-Expires ({duration}s) exceeds maximum allowed "
                        f"duration ({max_duration}s)",
                        field="Session-Expires",
                    )

                if duration < 90:
                    result.add_warning(
                        "SIP_SESSION_DURATION_SHORT",
                        f"Session-Expires ({duration}s) is below recommended minimum (90s)",
                        field="Session-Expires",
                    )
            except (ValueError, IndexError):
                result.add_warning(
                    "SIP_INVALID_SESSION_EXPIRES",
                    f"Invalid Session-Expires value: {session_expires}",
                    field="Session-Expires",
                )

        # For INVITE requests, check Min-SE header
        if msg.method == "INVITE":
            min_se = msg.get_header_value("Min-SE")
            if min_se:
                try:
                    min_se_val = int(min_se.split(";")[0].strip())
                    if min_se_val < 90:
                        result.add_warning(
                            "SIP_MIN_SE_TOO_LOW",
                            f"Min-SE ({min_se_val}s) is below RFC 4028 minimum (90s)",
                            field="Min-SE",
                        )
                except (ValueError, IndexError):
                    pass

    # ------------------------------------------------------------------
    # 8. Agent authorization
    # ------------------------------------------------------------------

    def _validate_agent_authorization(
        self, msg: SIPMessage, result: ValidationResult
    ) -> None:
        """Validate agent authorization for BPO operations."""
        agent_id = msg.bpo_context.agent_id
        if not agent_id:
            # No agent context - skip agent-specific checks
            return

        # Check against allowed agents list
        if self.allowed_agents is not None:
            if agent_id not in self.allowed_agents:
                result.add_error(
                    "SIP_UNAUTHORIZED_AGENT",
                    f"Agent {agent_id} is not authorized",
                    field="X-CC-Agent",
                    severity=ValidationSeverity.CRITICAL,
                )

        # Check method restrictions for agents
        if msg.is_request and msg.method:
            allowed_method_names = {m.method for m in AGENT_ALLOWED_METHODS}
            if msg.method not in allowed_method_names:
                result.add_error(
                    "SIP_AGENT_METHOD_FORBIDDEN",
                    f"Agent {agent_id} is not permitted to use method {msg.method}",
                    field="method",
                )

        # Validate transfer authorization (REFER method)
        if msg.method == "REFER":
            if not self.security_profile.allow_transfer:
                result.add_error(
                    "SIP_TRANSFER_FORBIDDEN",
                    f"Call transfer (REFER) is not permitted by security policy",
                    field="method",
                )

        # Validate call forwarding
        if msg.method == "INVITE":
            diversion = msg.get_header_value("Diversion")
            if diversion and not self.security_profile.allow_forward:
                result.add_error(
                    "SIP_FORWARD_FORBIDDEN",
                    "Call forwarding (Diversion header) is not permitted by security policy",
                    field="Diversion",
                )

    # ------------------------------------------------------------------
    # 9. Tenant authorization
    # ------------------------------------------------------------------

    def _validate_tenant_authorization(
        self, msg: SIPMessage, result: ValidationResult
    ) -> None:
        """Validate tenant ID for multi-tenant BPO environments."""
        tenant_id = msg.bpo_context.tenant_id

        if self.allowed_tenants is not None:
            if not tenant_id:
                result.add_error(
                    "SIP_MISSING_TENANT",
                    "X-CC-Tenant header is required in multi-tenant mode",
                    field="X-CC-Tenant",
                    severity=ValidationSeverity.CRITICAL,
                )
            elif tenant_id not in self.allowed_tenants:
                result.add_error(
                    "SIP_UNAUTHORIZED_TENANT",
                    f"Tenant {tenant_id} is not authorized",
                    field="X-CC-Tenant",
                    severity=ValidationSeverity.CRITICAL,
                )

    # ------------------------------------------------------------------
    # 10. DTMF relay validation
    # ------------------------------------------------------------------

    def _validate_dtmf(self, msg: SIPMessage, result: ValidationResult) -> None:
        """
        Validate DTMF relay in INFO method messages.

        DTMF (Dual-Tone Multi-Frequency) signals are sent via SIP INFO
        for telephony keypad input. This validates:
        - Signal is a valid DTMF digit (0-9, *, #, A-D)
        - Duration is within acceptable bounds
        - No injection attacks in DTMF body
        """
        if msg.method != "INFO":
            return

        content_type = msg.content_type
        if not content_type:
            return

        ct_lower = content_type.lower()
        if "dtmf-relay" not in ct_lower and "dtmf" not in ct_lower:
            return

        if not msg.body:
            result.add_warning(
                "SIP_DTMF_EMPTY_BODY",
                "INFO message with DTMF content type has no body",
                field="body",
            )
            return

        # Parse DTMF body
        from ai_engine.domains.bpo.protocols.sip.sip_parser import SIPParser
        parser = SIPParser(strict=False)
        dtmf_data = parser.parse_dtmf_body(msg.body)

        if not dtmf_data:
            result.add_error(
                "SIP_DTMF_INVALID_BODY",
                "Unable to parse DTMF relay body",
                field="body",
            )
            return

        # Validate signal
        signal = dtmf_data.get("signal", "")
        if not signal:
            result.add_error(
                "SIP_DTMF_MISSING_SIGNAL",
                "DTMF signal is required",
                field="body/signal",
            )
        elif signal not in self.VALID_DTMF_SIGNALS:
            result.add_error(
                "SIP_DTMF_INVALID_SIGNAL",
                f"Invalid DTMF signal: {repr(signal)} "
                f"(valid: {', '.join(sorted(self.VALID_DTMF_SIGNALS))})",
                field="body/signal",
            )

        # Validate duration
        duration_str = dtmf_data.get("duration", "")
        if duration_str:
            try:
                duration = int(duration_str)
                if duration < self.DTMF_MIN_DURATION:
                    result.add_warning(
                        "SIP_DTMF_DURATION_SHORT",
                        f"DTMF duration ({duration}ms) is below minimum ({self.DTMF_MIN_DURATION}ms)",
                        field="body/duration",
                    )
                elif duration > self.DTMF_MAX_DURATION:
                    result.add_error(
                        "SIP_DTMF_DURATION_LONG",
                        f"DTMF duration ({duration}ms) exceeds maximum ({self.DTMF_MAX_DURATION}ms)",
                        field="body/duration",
                    )
            except ValueError:
                result.add_error(
                    "SIP_DTMF_INVALID_DURATION",
                    f"Invalid DTMF duration value: {duration_str}",
                    field="body/duration",
                )

        # Check for injection in DTMF body
        if self.SQL_INJECTION_PATTERN.search(msg.body):
            result.add_error(
                "SIP_DTMF_INJECTION",
                "Possible injection attack in DTMF relay body",
                field="body",
                severity=ValidationSeverity.CRITICAL,
            )

    # ------------------------------------------------------------------
    # 11. SDP body security
    # ------------------------------------------------------------------

    def _validate_sdp_security(self, msg: SIPMessage, result: ValidationResult) -> None:
        """Validate SDP body for security compliance."""
        sdp = msg.sdp
        if not sdp:
            return

        # Check SRTP requirement
        if self.security_profile.require_srtp:
            for media in sdp.media_descriptions:
                if media.is_audio or media.is_video:
                    if not media.is_secure:
                        result.add_error(
                            "SIP_SDP_INSECURE_MEDIA",
                            f"Media stream ({media.media_type}) does not use SRTP. "
                            f"Security policy requires encrypted media (SAVP/SAVPF).",
                            field=f"sdp/media/{media.media_type}",
                        )

                    # Check for crypto attribute (SDES key exchange)
                    has_crypto = any(
                        k.startswith("crypto") for k in media.attributes
                    )
                    has_fingerprint = "fingerprint" in media.attributes
                    if not has_crypto and not has_fingerprint:
                        result.add_warning(
                            "SIP_SDP_NO_KEY_EXCHANGE",
                            f"Media stream ({media.media_type}) has no crypto or "
                            f"fingerprint attribute for key exchange",
                            field=f"sdp/media/{media.media_type}",
                        )

        # Check for media on port 0 (disabled media) - informational
        for media in sdp.media_descriptions:
            if media.port == 0:
                result.add_warning(
                    "SIP_SDP_MEDIA_DISABLED",
                    f"Media stream ({media.media_type}) is disabled (port 0)",
                    field=f"sdp/media/{media.media_type}",
                )

        # Check for suspicious IP addresses in SDP
        connection = sdp.connection
        if connection:
            self._validate_sdp_connection(connection, "session", result)

        for media in sdp.media_descriptions:
            if media.connection:
                self._validate_sdp_connection(
                    media.connection, media.media_type, result
                )

    def _validate_sdp_connection(
        self, connection: str, context: str, result: ValidationResult
    ) -> None:
        """Validate SDP connection line for suspicious addresses."""
        # Format: IN IP4 address or IN IP6 address
        parts = connection.split()
        if len(parts) >= 3:
            address = parts[2]

            # Check for private/loopback addresses (may indicate misconfiguration)
            if address.startswith("127.") or address == "::1":
                result.add_warning(
                    "SIP_SDP_LOOPBACK_ADDRESS",
                    f"SDP {context} connection uses loopback address: {address}",
                    field=f"sdp/connection/{context}",
                )

            # Check for 0.0.0.0 (hold scenario or misconfiguration)
            if address == "0.0.0.0":
                result.add_warning(
                    "SIP_SDP_ZERO_ADDRESS",
                    f"SDP {context} connection uses 0.0.0.0 (call hold or misconfigured)",
                    field=f"sdp/connection/{context}",
                )

    # ------------------------------------------------------------------
    # Dictionary validation (for non-parsed data)
    # ------------------------------------------------------------------

    def _validate_dict(self, data: Dict, result: ValidationResult) -> None:
        """Validate SIP data from a dictionary representation."""
        # Basic field validation
        if "method" in data:
            method = data["method"]
            known_methods = {m.method for m in SIPMethod}
            if method not in known_methods:
                result.add_warning(
                    "SIP_UNKNOWN_METHOD",
                    f"Unknown SIP method: {method}",
                    field="method",
                )

        if "status_code" in data:
            try:
                code = int(data["status_code"])
                if not (100 <= code <= 699):
                    result.add_error(
                        "SIP_INVALID_STATUS_CODE",
                        f"Status code out of range: {code}",
                        field="status_code",
                    )
            except (ValueError, TypeError):
                result.add_error(
                    "SIP_INVALID_STATUS_CODE",
                    "Status code must be a numeric value",
                    field="status_code",
                )

        # Check call_id
        if "call_id" in data and not data["call_id"]:
            result.add_error(
                "SIP_MISSING_CALL_ID",
                "Call-ID is required",
                field="call_id",
                severity=ValidationSeverity.CRITICAL,
            )

        # Check headers for PCI data
        if "headers" in data and isinstance(data["headers"], dict):
            for hdr_name, hdr_value in data["headers"].items():
                if isinstance(hdr_value, str):
                    self._check_pci_data(hdr_value, f"header:{hdr_name}", result)

        # Agent authorization
        agent_id = data.get("agent_id", "")
        if agent_id and self.allowed_agents is not None:
            if agent_id not in self.allowed_agents:
                result.add_error(
                    "SIP_UNAUTHORIZED_AGENT",
                    f"Agent {agent_id} is not authorized",
                    field="agent_id",
                    severity=ValidationSeverity.CRITICAL,
                )

        # Tenant authorization
        tenant_id = data.get("tenant_id", "")
        if tenant_id and self.allowed_tenants is not None:
            if tenant_id not in self.allowed_tenants:
                result.add_error(
                    "SIP_UNAUTHORIZED_TENANT",
                    f"Tenant {tenant_id} is not authorized",
                    field="tenant_id",
                    severity=ValidationSeverity.CRITICAL,
                )

    # ------------------------------------------------------------------
    # Rate limit state management
    # ------------------------------------------------------------------

    def reset_rate_limits(self) -> None:
        """Reset all rate limiting counters."""
        self._request_counts.clear()

    def get_rate_limit_status(self, key: str) -> Dict[str, Any]:
        """
        Get current rate limit status for a given key.

        Args:
            key: Rate limit key (IP address, agent ID, etc.)

        Returns:
            Dictionary with current count, limit, and window info
        """
        now = time.time()
        window_start = now - self.rate_limit_window
        count = len([
            ts for ts in self._request_counts.get(key, [])
            if ts > window_start
        ])
        max_in_window = self.security_profile.rate_limit_per_second * self.rate_limit_window

        return {
            "key": key,
            "current_count": count,
            "max_in_window": max_in_window,
            "window_seconds": self.rate_limit_window,
            "utilization_pct": round((count / max_in_window) * 100, 1) if max_in_window > 0 else 0,
        }
