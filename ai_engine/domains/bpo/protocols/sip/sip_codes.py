"""
SIP Protocol Codes and Constants

Defines:
- SIP methods (RFC 3261)
- SIP response codes
- SIP header names
- SDP media types
- Call center specific SIP extensions
"""

from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass


class SIPMethod(Enum):
    """SIP request methods per RFC 3261 and extensions."""

    # Core methods (RFC 3261)
    INVITE = ("INVITE", "Initiate a session/call")
    ACK = ("ACK", "Confirm session establishment")
    BYE = ("BYE", "Terminate a session/call")
    CANCEL = ("CANCEL", "Cancel a pending request")
    REGISTER = ("REGISTER", "Register contact information")
    OPTIONS = ("OPTIONS", "Query capabilities")

    # Extension methods
    REFER = ("REFER", "Transfer a call (RFC 3515)")
    SUBSCRIBE = ("SUBSCRIBE", "Subscribe to event notification (RFC 6665)")
    NOTIFY = ("NOTIFY", "Event notification (RFC 6665)")
    MESSAGE = ("MESSAGE", "Instant message (RFC 3428)")
    INFO = ("INFO", "Mid-session information (RFC 6086)")
    UPDATE = ("UPDATE", "Update session parameters (RFC 3311)")
    PRACK = ("PRACK", "Provisional response acknowledgment (RFC 3262)")
    PUBLISH = ("PUBLISH", "Publish event state (RFC 3903)")

    def __init__(self, method: str, description: str):
        self.method = method
        self.description = description


class SIPResponseCode(Enum):
    """SIP response status codes."""

    # 1xx Provisional
    TRYING = (100, "Trying", "provisional")
    RINGING = (180, "Ringing", "provisional")
    CALL_BEING_FORWARDED = (181, "Call Is Being Forwarded", "provisional")
    QUEUED = (182, "Queued", "provisional")
    SESSION_PROGRESS = (183, "Session Progress", "provisional")

    # 2xx Success
    OK = (200, "OK", "success")
    ACCEPTED = (202, "Accepted", "success")
    NO_NOTIFICATION = (204, "No Notification", "success")

    # 3xx Redirection
    MULTIPLE_CHOICES = (300, "Multiple Choices", "redirection")
    MOVED_PERMANENTLY = (301, "Moved Permanently", "redirection")
    MOVED_TEMPORARILY = (302, "Moved Temporarily", "redirection")

    # 4xx Client Error
    BAD_REQUEST = (400, "Bad Request", "client_error")
    UNAUTHORIZED = (401, "Unauthorized", "client_error")
    PAYMENT_REQUIRED = (402, "Payment Required", "client_error")
    FORBIDDEN = (403, "Forbidden", "client_error")
    NOT_FOUND = (404, "Not Found", "client_error")
    METHOD_NOT_ALLOWED = (405, "Method Not Allowed", "client_error")
    REQUEST_TIMEOUT = (408, "Request Timeout", "client_error")
    BUSY_HERE = (486, "Busy Here", "client_error")
    REQUEST_TERMINATED = (487, "Request Terminated", "client_error")

    # 5xx Server Error
    SERVER_INTERNAL_ERROR = (500, "Server Internal Error", "server_error")
    NOT_IMPLEMENTED = (501, "Not Implemented", "server_error")
    BAD_GATEWAY = (502, "Bad Gateway", "server_error")
    SERVICE_UNAVAILABLE = (503, "Service Unavailable", "server_error")
    GATEWAY_TIMEOUT = (504, "Gateway Time-out", "server_error")

    # 6xx Global Failure
    BUSY_EVERYWHERE = (600, "Busy Everywhere", "global_failure")
    DECLINE = (603, "Decline", "global_failure")
    DOES_NOT_EXIST_ANYWHERE = (604, "Does Not Exist Anywhere", "global_failure")

    def __init__(self, code: int, reason: str, category: str):
        self.code = code
        self.reason = reason
        self.category = category

    @property
    def is_provisional(self) -> bool:
        return 100 <= self.code < 200

    @property
    def is_success(self) -> bool:
        return 200 <= self.code < 300

    @property
    def is_error(self) -> bool:
        return self.code >= 400


class SIPHeaderName(Enum):
    """Standard SIP header names."""

    # Request/Response line headers
    VIA = ("Via", "v", True)
    FROM = ("From", "f", True)
    TO = ("To", "t", True)
    CALL_ID = ("Call-ID", "i", True)
    CSEQ = ("CSeq", None, True)
    MAX_FORWARDS = ("Max-Forwards", None, True)
    CONTACT = ("Contact", "m", False)
    CONTENT_TYPE = ("Content-Type", "c", False)
    CONTENT_LENGTH = ("Content-Length", "l", False)

    # Authentication
    AUTHORIZATION = ("Authorization", None, False)
    WWW_AUTHENTICATE = ("WWW-Authenticate", None, False)
    PROXY_AUTHORIZATION = ("Proxy-Authorization", None, False)
    PROXY_AUTHENTICATE = ("Proxy-Authenticate", None, False)

    # Session
    ALLOW = ("Allow", None, False)
    SUPPORTED = ("Supported", "k", False)
    REQUIRE = ("Require", None, False)
    USER_AGENT = ("User-Agent", None, False)
    SERVER = ("Server", None, False)
    RECORD_ROUTE = ("Record-Route", None, False)
    ROUTE = ("Route", None, False)

    # Call center specific
    REFERRED_BY = ("Referred-By", "b", False)
    REPLACES = ("Replaces", None, False)
    DIVERSION = ("Diversion", None, False)

    def __init__(self, header_name: str, compact_form: Optional[str], required: bool):
        self.header_name = header_name
        self.compact_form = compact_form
        self.required = required


class SDPMediaType(Enum):
    """SDP media types for call center."""

    AUDIO = ("audio", "Voice media stream")
    VIDEO = ("video", "Video media stream")
    APPLICATION = ("application", "Application data (screen sharing, etc.)")
    MESSAGE = ("message", "Messaging/chat stream")

    def __init__(self, media_type: str, description: str):
        self.media_type = media_type
        self.description = description


class SIPTransport(Enum):
    """SIP transport protocols."""

    UDP = ("UDP", 5060, False)
    TCP = ("TCP", 5060, False)
    TLS = ("TLS", 5061, True)
    WSS = ("WSS", 443, True)           # WebSocket Secure for WebRTC
    TLS_PQC = ("TLS-PQC", 5062, True)  # Quantum-safe TLS

    def __init__(self, transport: str, default_port: int, encrypted: bool):
        self.transport = transport
        self.default_port = default_port
        self.encrypted = encrypted


@dataclass
class SIPSecurityProfile:
    """Security profile for SIP communications in BPO context."""

    # Transport security
    require_tls: bool = True
    min_tls_version: str = "TLS 1.3"
    require_pqc_tls: bool = True

    # Authentication
    require_digest_auth: bool = True
    auth_algorithm: str = "SHA-256"
    require_mutual_tls: bool = False

    # Media security
    require_srtp: bool = True
    srtp_crypto_suite: str = "AES_256_CM_HMAC_SHA1_80"
    require_srtp_pqc: bool = True

    # Call center settings
    allow_transfer: bool = True
    allow_forward: bool = True
    max_concurrent_calls: int = 10000
    max_call_duration_seconds: int = 14400  # 4 hours

    # Toll fraud prevention
    block_premium_rate: bool = True
    block_international: bool = False
    allowed_country_codes: List[str] = None
    rate_limit_per_second: int = 100

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "transport": {
                "require_tls": self.require_tls,
                "min_tls_version": self.min_tls_version,
                "require_pqc_tls": self.require_pqc_tls,
            },
            "authentication": {
                "require_digest_auth": self.require_digest_auth,
                "auth_algorithm": self.auth_algorithm,
            },
            "media": {
                "require_srtp": self.require_srtp,
                "srtp_crypto_suite": self.srtp_crypto_suite,
                "require_srtp_pqc": self.require_srtp_pqc,
            },
            "fraud_prevention": {
                "block_premium_rate": self.block_premium_rate,
                "rate_limit_per_second": self.rate_limit_per_second,
            },
        }


# Call center specific SIP headers and extensions
CALL_CENTER_SIP_HEADERS = {
    "X-CC-Queue": "Call center queue identifier",
    "X-CC-Agent": "Assigned agent identifier",
    "X-CC-Campaign": "Campaign/program identifier",
    "X-CC-Tenant": "Multi-tenant identifier",
    "X-CC-Priority": "Call priority level",
    "X-CC-Skill": "Required agent skill group",
    "X-CC-CallType": "Inbound/Outbound/Transfer/Conference",
    "X-CC-RecordingID": "Call recording identifier",
    "X-CC-PCI-Mode": "PCI compliance mode (ACTIVE/PAUSED)",
}

# Premium rate number prefixes (toll fraud risk)
PREMIUM_RATE_PREFIXES = {
    "US": ["900", "976"],
    "UK": ["09"],
    "EU": ["0900", "0906", "0909"],
    "IRSF": ["882", "883"],  # International Revenue Share Fraud
}

# Allowed SIP methods for call center agents
AGENT_ALLOWED_METHODS = [
    SIPMethod.INVITE,
    SIPMethod.ACK,
    SIPMethod.BYE,
    SIPMethod.CANCEL,
    SIPMethod.REFER,    # Call transfer
    SIPMethod.INFO,     # DTMF relay
    SIPMethod.UPDATE,   # Hold/resume
]
