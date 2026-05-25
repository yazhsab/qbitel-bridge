"""
SIP Message Data Structures

Base classes for SIP (Session Initiation Protocol - RFC 3261) message
representation for BPO/Call Center communications including:
- Request and Response line structures
- Header representation (standard and custom X-CC-* headers)
- SDP body parsing
- BPO-specific tenant, queue, and agent context
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


@dataclass
class SIPHeader:
    """Represents a single SIP header field."""

    name: str
    value: str
    parameters: Dict[str, str] = field(default_factory=dict)
    line_number: int = 0

    # Compact form mapping (RFC 3261 Section 7.3.3)
    COMPACT_FORMS: Dict[str, str] = field(default_factory=lambda: {
        "v": "Via",
        "f": "From",
        "t": "To",
        "i": "Call-ID",
        "m": "Contact",
        "c": "Content-Type",
        "l": "Content-Length",
        "k": "Supported",
        "b": "Referred-By",
    }, init=False, repr=False)

    @property
    def canonical_name(self) -> str:
        """Get canonical (long-form) header name."""
        return self.COMPACT_FORMS.get(self.name.lower(), self.name)

    @property
    def display_value(self) -> str:
        """Get value with parameters for display."""
        if not self.parameters:
            return self.value
        params = ";".join(
            f"{k}={v}" if v else k for k, v in self.parameters.items()
        )
        return f"{self.value};{params}"

    def to_sip(self) -> str:
        """Convert to SIP header line format."""
        return f"{self.name}: {self.display_value}"

    def __str__(self) -> str:
        return self.to_sip()


@dataclass
class SIPRequestLine:
    """
    SIP Request-Line (RFC 3261 Section 7.1).

    Format: METHOD Request-URI SIP-Version
    Example: INVITE sip:agent@bpo.example.com SIP/2.0
    """

    method: str = ""
    request_uri: str = ""
    sip_version: str = "SIP/2.0"

    @property
    def is_valid(self) -> bool:
        """Check if the request line has all required components."""
        return bool(self.method and self.request_uri and self.sip_version)

    @property
    def uri_user(self) -> Optional[str]:
        """Extract user part from Request-URI (sip:user@host -> user)."""
        uri = self.request_uri
        if ":" in uri:
            uri = uri.split(":", 1)[1]
        if "@" in uri:
            return uri.split("@", 1)[0]
        return None

    @property
    def uri_host(self) -> Optional[str]:
        """Extract host part from Request-URI."""
        uri = self.request_uri
        if ":" in uri:
            uri = uri.split(":", 1)[1]
        if "@" in uri:
            uri = uri.split("@", 1)[1]
        # Remove port and parameters
        for sep in (":", ";", "?"):
            if sep in uri:
                uri = uri.split(sep, 1)[0]
        return uri if uri else None

    @classmethod
    def from_line(cls, line: str) -> "SIPRequestLine":
        """Parse from a raw request line string."""
        parts = line.strip().split(" ", 2)
        req = cls()
        if len(parts) >= 1:
            req.method = parts[0]
        if len(parts) >= 2:
            req.request_uri = parts[1]
        if len(parts) >= 3:
            req.sip_version = parts[2]
        return req

    def to_sip(self) -> str:
        """Convert to SIP request line format."""
        return f"{self.method} {self.request_uri} {self.sip_version}"

    def __str__(self) -> str:
        return self.to_sip()


@dataclass
class SIPResponseLine:
    """
    SIP Status-Line (RFC 3261 Section 7.2).

    Format: SIP-Version Status-Code Reason-Phrase
    Example: SIP/2.0 200 OK
    """

    sip_version: str = "SIP/2.0"
    status_code: int = 0
    reason_phrase: str = ""

    @property
    def is_valid(self) -> bool:
        """Check if the response line has all required components."""
        return bool(self.sip_version and self.status_code > 0)

    @property
    def is_provisional(self) -> bool:
        """1xx responses."""
        return 100 <= self.status_code < 200

    @property
    def is_success(self) -> bool:
        """2xx responses."""
        return 200 <= self.status_code < 300

    @property
    def is_redirection(self) -> bool:
        """3xx responses."""
        return 300 <= self.status_code < 400

    @property
    def is_client_error(self) -> bool:
        """4xx responses."""
        return 400 <= self.status_code < 500

    @property
    def is_server_error(self) -> bool:
        """5xx responses."""
        return 500 <= self.status_code < 600

    @property
    def is_global_failure(self) -> bool:
        """6xx responses."""
        return 600 <= self.status_code < 700

    @property
    def is_error(self) -> bool:
        """Any error response (4xx, 5xx, 6xx)."""
        return self.status_code >= 400

    @classmethod
    def from_line(cls, line: str) -> "SIPResponseLine":
        """Parse from a raw status line string."""
        parts = line.strip().split(" ", 2)
        resp = cls()
        if len(parts) >= 1:
            resp.sip_version = parts[0]
        if len(parts) >= 2:
            try:
                resp.status_code = int(parts[1])
            except ValueError:
                resp.status_code = 0
        if len(parts) >= 3:
            resp.reason_phrase = parts[2]
        return resp

    def to_sip(self) -> str:
        """Convert to SIP status line format."""
        return f"{self.sip_version} {self.status_code} {self.reason_phrase}"

    def __str__(self) -> str:
        return self.to_sip()


class MessageType(Enum):
    """SIP message type indicator."""

    REQUEST = "REQUEST"
    RESPONSE = "RESPONSE"


@dataclass
class SDPMediaDescription:
    """
    SDP media description (m= line and associated attributes).

    Represents a single media stream in an SDP body.
    """

    media_type: str = ""       # audio, video, application, message
    port: int = 0
    protocol: str = ""         # RTP/AVP, RTP/SAVP, UDP/TLS/RTP/SAVPF
    formats: List[str] = field(default_factory=list)  # payload types
    attributes: Dict[str, str] = field(default_factory=dict)
    connection: str = ""       # c= line value
    bandwidth: str = ""        # b= line value

    @property
    def is_audio(self) -> bool:
        return self.media_type == "audio"

    @property
    def is_video(self) -> bool:
        return self.media_type == "video"

    @property
    def is_secure(self) -> bool:
        """Check if the media uses a secure transport (SAVP/SAVPF)."""
        return "SAVP" in self.protocol.upper()

    @property
    def codecs(self) -> List[str]:
        """Extract codec names from rtpmap attributes."""
        result = []
        for key, value in self.attributes.items():
            if key.startswith("rtpmap"):
                # rtpmap:payload_type codec_name/clock_rate
                parts = value.split(" ", 1)
                if len(parts) == 2:
                    codec_info = parts[1].split("/")
                    result.append(codec_info[0])
        return result

    @property
    def crypto_suites(self) -> List[str]:
        """Extract SRTP crypto suites from crypto attributes."""
        result = []
        for key, value in self.attributes.items():
            if key.startswith("crypto"):
                # crypto:tag suite key-params
                parts = value.split(" ", 2)
                if len(parts) >= 2:
                    result.append(parts[1])
        return result

    def to_sdp(self) -> str:
        """Convert to SDP format."""
        lines = []
        formats_str = " ".join(self.formats)
        lines.append(f"m={self.media_type} {self.port} {self.protocol} {formats_str}")
        if self.connection:
            lines.append(f"c={self.connection}")
        if self.bandwidth:
            lines.append(f"b={self.bandwidth}")
        for key, value in self.attributes.items():
            if value:
                lines.append(f"a={key}:{value}")
            else:
                lines.append(f"a={key}")
        return "\r\n".join(lines)


@dataclass
class SDPBody:
    """
    SDP (Session Description Protocol - RFC 4566) body.

    Contains session-level information and media descriptions
    for SIP INVITE/UPDATE messages.
    """

    # Session-level fields
    version: str = "0"                          # v=
    origin: str = ""                            # o=
    session_name: str = ""                      # s=
    connection: str = ""                        # c= (session-level)
    timing: str = ""                            # t=
    attributes: Dict[str, str] = field(default_factory=dict)  # a= (session-level)
    media_descriptions: List[SDPMediaDescription] = field(default_factory=list)

    # Raw content
    raw_body: str = ""

    @property
    def has_audio(self) -> bool:
        """Check if any audio media stream is present."""
        return any(m.is_audio for m in self.media_descriptions)

    @property
    def has_video(self) -> bool:
        """Check if any video media stream is present."""
        return any(m.is_video for m in self.media_descriptions)

    @property
    def is_secure(self) -> bool:
        """Check if all media streams use secure transport."""
        if not self.media_descriptions:
            return False
        return all(m.is_secure for m in self.media_descriptions)

    @property
    def ice_candidates(self) -> List[str]:
        """Extract ICE candidates from attributes."""
        candidates = []
        for key, value in self.attributes.items():
            if key == "candidate":
                candidates.append(value)
        for media in self.media_descriptions:
            for key, value in media.attributes.items():
                if key == "candidate":
                    candidates.append(value)
        return candidates

    def to_sdp(self) -> str:
        """Convert to SDP format."""
        lines = []
        lines.append(f"v={self.version}")
        if self.origin:
            lines.append(f"o={self.origin}")
        if self.session_name:
            lines.append(f"s={self.session_name}")
        if self.connection:
            lines.append(f"c={self.connection}")
        if self.timing:
            lines.append(f"t={self.timing}")
        for key, value in self.attributes.items():
            if value:
                lines.append(f"a={key}:{value}")
            else:
                lines.append(f"a={key}")
        for media in self.media_descriptions:
            lines.append(media.to_sdp())
        return "\r\n".join(lines)


@dataclass
class SIPViaHeader:
    """Parsed Via header with transport, host, port, and branch."""

    protocol: str = "SIP/2.0"
    transport: str = "UDP"
    host: str = ""
    port: int = 5060
    branch: str = ""
    received: str = ""
    rport: str = ""
    parameters: Dict[str, str] = field(default_factory=dict)

    def to_sip(self) -> str:
        """Convert to Via header value."""
        value = f"{self.protocol}/{self.transport} {self.host}"
        if self.port != 5060:
            value += f":{self.port}"
        if self.branch:
            value += f";branch={self.branch}"
        if self.received:
            value += f";received={self.received}"
        if self.rport:
            value += f";rport={self.rport}"
        for k, v in self.parameters.items():
            if v:
                value += f";{k}={v}"
            else:
                value += f";{k}"
        return value


@dataclass
class SIPAddressHeader:
    """Parsed From/To/Contact header with display name, URI, and tag."""

    display_name: str = ""
    uri: str = ""
    tag: str = ""
    parameters: Dict[str, str] = field(default_factory=dict)

    @property
    def user(self) -> Optional[str]:
        """Extract user from URI (sip:user@host -> user)."""
        uri = self.uri
        if ":" in uri:
            uri = uri.split(":", 1)[1]
        if "@" in uri:
            return uri.split("@", 1)[0]
        return None

    @property
    def host(self) -> Optional[str]:
        """Extract host from URI."""
        uri = self.uri
        if ":" in uri:
            uri = uri.split(":", 1)[1]
        if "@" in uri:
            uri = uri.split("@", 1)[1]
        for sep in (":", ";", "?", ">"):
            if sep in uri:
                uri = uri.split(sep, 1)[0]
        return uri if uri else None

    def to_sip(self) -> str:
        """Convert to SIP address header value."""
        if self.display_name:
            value = f'"{self.display_name}" <{self.uri}>'
        else:
            value = f"<{self.uri}>"
        if self.tag:
            value += f";tag={self.tag}"
        for k, v in self.parameters.items():
            if v:
                value += f";{k}={v}"
            else:
                value += f";{k}"
        return value


@dataclass
class SIPCSeqHeader:
    """Parsed CSeq header with sequence number and method."""

    sequence: int = 0
    method: str = ""

    def to_sip(self) -> str:
        """Convert to CSeq header value."""
        return f"{self.sequence} {self.method}"


@dataclass
class BPOContext:
    """
    BPO-specific context extracted from custom X-CC-* headers.

    Encapsulates call center metadata carried in SIP messages.
    """

    tenant_id: str = ""
    queue_name: str = ""
    agent_id: str = ""
    campaign_id: str = ""
    call_priority: str = ""
    skill_group: str = ""
    call_type: str = ""        # Inbound, Outbound, Transfer, Conference
    recording_id: str = ""
    pci_mode: str = ""         # ACTIVE, PAUSED

    @property
    def is_pci_active(self) -> bool:
        """Check if PCI-DSS compliance mode is active."""
        return self.pci_mode.upper() == "ACTIVE"

    @property
    def is_pci_paused(self) -> bool:
        """Check if PCI-DSS recording is paused for cardholder data entry."""
        return self.pci_mode.upper() == "PAUSED"

    def to_headers(self) -> Dict[str, str]:
        """Convert BPO context to X-CC-* SIP headers."""
        headers = {}
        if self.tenant_id:
            headers["X-CC-Tenant"] = self.tenant_id
        if self.queue_name:
            headers["X-CC-Queue"] = self.queue_name
        if self.agent_id:
            headers["X-CC-Agent"] = self.agent_id
        if self.campaign_id:
            headers["X-CC-Campaign"] = self.campaign_id
        if self.call_priority:
            headers["X-CC-Priority"] = self.call_priority
        if self.skill_group:
            headers["X-CC-Skill"] = self.skill_group
        if self.call_type:
            headers["X-CC-CallType"] = self.call_type
        if self.recording_id:
            headers["X-CC-RecordingID"] = self.recording_id
        if self.pci_mode:
            headers["X-CC-PCI-Mode"] = self.pci_mode
        return headers

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tenant_id": self.tenant_id,
            "queue_name": self.queue_name,
            "agent_id": self.agent_id,
            "campaign_id": self.campaign_id,
            "call_priority": self.call_priority,
            "skill_group": self.skill_group,
            "call_type": self.call_type,
            "recording_id": self.recording_id,
            "pci_mode": self.pci_mode,
        }


@dataclass
class SIPMessage:
    """
    Complete SIP message with headers, body, and BPO context.

    Supports both SIP requests and responses, with full header
    parsing and SDP body extraction for call center operations.
    """

    # Message type
    message_type: MessageType = MessageType.REQUEST

    # Request line (for requests)
    request_line: Optional[SIPRequestLine] = None

    # Response line (for responses)
    response_line: Optional[SIPResponseLine] = None

    # Headers (preserves order and allows duplicates for Via, etc.)
    headers: List[SIPHeader] = field(default_factory=list)

    # Body (raw string)
    body: str = ""

    # Parsed SDP body (if Content-Type is application/sdp)
    sdp: Optional[SDPBody] = None

    # Parsed key headers for convenience
    via_headers: List[SIPViaHeader] = field(default_factory=list)
    from_header: Optional[SIPAddressHeader] = None
    to_header: Optional[SIPAddressHeader] = None
    cseq_header: Optional[SIPCSeqHeader] = None

    # BPO context from X-CC-* headers
    bpo_context: BPOContext = field(default_factory=BPOContext)

    # Metadata
    raw_message: str = ""
    parse_errors: List[str] = field(default_factory=list)
    received_at: Optional[datetime] = None
    source_ip: str = ""
    source_port: int = 0

    # --- Convenience properties ---

    @property
    def is_request(self) -> bool:
        """Check if this is a SIP request."""
        return self.message_type == MessageType.REQUEST

    @property
    def is_response(self) -> bool:
        """Check if this is a SIP response."""
        return self.message_type == MessageType.RESPONSE

    @property
    def method(self) -> Optional[str]:
        """Get the SIP method (for requests, or from CSeq for responses)."""
        if self.is_request and self.request_line:
            return self.request_line.method
        if self.cseq_header:
            return self.cseq_header.method
        return None

    @property
    def status_code(self) -> Optional[int]:
        """Get the status code (for responses only)."""
        if self.is_response and self.response_line:
            return self.response_line.status_code
        return None

    @property
    def call_id(self) -> Optional[str]:
        """Extract Call-ID header value."""
        return self.get_header_value("Call-ID") or self.get_header_value("i")

    @property
    def from_uri(self) -> Optional[str]:
        """Get From URI."""
        if self.from_header:
            return self.from_header.uri
        return None

    @property
    def to_uri(self) -> Optional[str]:
        """Get To URI."""
        if self.to_header:
            return self.to_header.uri
        return None

    @property
    def from_tag(self) -> Optional[str]:
        """Get From tag."""
        if self.from_header:
            return self.from_header.tag
        return None

    @property
    def to_tag(self) -> Optional[str]:
        """Get To tag."""
        if self.to_header:
            return self.to_header.tag
        return None

    @property
    def branch(self) -> Optional[str]:
        """Get top Via branch parameter (transaction ID)."""
        if self.via_headers:
            return self.via_headers[0].branch
        return None

    @property
    def cseq(self) -> Optional[int]:
        """Get CSeq sequence number."""
        if self.cseq_header:
            return self.cseq_header.sequence
        return None

    @property
    def content_type(self) -> Optional[str]:
        """Get Content-Type header value."""
        return self.get_header_value("Content-Type") or self.get_header_value("c")

    @property
    def content_length(self) -> int:
        """Get Content-Length as integer."""
        value = self.get_header_value("Content-Length") or self.get_header_value("l")
        if value:
            try:
                return int(value)
            except ValueError:
                return 0
        return 0

    @property
    def max_forwards(self) -> Optional[int]:
        """Get Max-Forwards as integer."""
        value = self.get_header_value("Max-Forwards")
        if value:
            try:
                return int(value)
            except ValueError:
                return None
        return None

    @property
    def user_agent(self) -> Optional[str]:
        """Get User-Agent header value."""
        return self.get_header_value("User-Agent")

    @property
    def contact(self) -> Optional[str]:
        """Get Contact header value."""
        return self.get_header_value("Contact") or self.get_header_value("m")

    @property
    def dialog_id(self) -> Optional[str]:
        """
        Compute the dialog identifier (Call-ID + from-tag + to-tag).

        Per RFC 3261 Section 12, a dialog is identified by:
        Call-ID + local tag + remote tag.
        """
        cid = self.call_id
        ft = self.from_tag or ""
        tt = self.to_tag or ""
        if cid:
            return f"{cid};{ft};{tt}"
        return None

    # --- BPO convenience properties ---

    @property
    def tenant_id(self) -> str:
        """Get BPO tenant ID from X-CC-Tenant header."""
        return self.bpo_context.tenant_id

    @property
    def queue_name(self) -> str:
        """Get call center queue name from X-CC-Queue header."""
        return self.bpo_context.queue_name

    @property
    def agent_id(self) -> str:
        """Get assigned agent ID from X-CC-Agent header."""
        return self.bpo_context.agent_id

    # --- Header access methods ---

    def get_header(self, name: str) -> Optional[SIPHeader]:
        """Get first header by name (case-insensitive)."""
        name_lower = name.lower()
        for h in self.headers:
            if h.name.lower() == name_lower or h.canonical_name.lower() == name_lower:
                return h
        return None

    def get_headers(self, name: str) -> List[SIPHeader]:
        """Get all headers matching name (case-insensitive)."""
        name_lower = name.lower()
        return [
            h for h in self.headers
            if h.name.lower() == name_lower or h.canonical_name.lower() == name_lower
        ]

    def get_header_value(self, name: str) -> Optional[str]:
        """Get value of first header matching name."""
        h = self.get_header(name)
        return h.value if h else None

    def get_header_values(self, name: str) -> List[str]:
        """Get values of all headers matching name."""
        return [h.value for h in self.get_headers(name)]

    def add_header(self, name: str, value: str, **parameters: str) -> None:
        """Add a header to the message."""
        self.headers.append(SIPHeader(name=name, value=value, parameters=parameters))

    def remove_header(self, name: str) -> int:
        """Remove all headers matching name. Returns count removed."""
        name_lower = name.lower()
        original_count = len(self.headers)
        self.headers = [
            h for h in self.headers
            if h.name.lower() != name_lower and h.canonical_name.lower() != name_lower
        ]
        return original_count - len(self.headers)

    def get_custom_headers(self, prefix: str = "X-CC-") -> Dict[str, str]:
        """Get all custom headers with given prefix."""
        result = {}
        for h in self.headers:
            if h.name.startswith(prefix):
                result[h.name] = h.value
        return result

    # --- Serialization ---

    def to_sip(self) -> str:
        """
        Serialize to SIP message format.

        Produces a complete SIP message string with start line,
        headers, empty line, and body per RFC 3261 Section 7.
        """
        lines = []

        # Start line
        if self.is_request and self.request_line:
            lines.append(self.request_line.to_sip())
        elif self.is_response and self.response_line:
            lines.append(self.response_line.to_sip())

        # Headers
        for header in self.headers:
            lines.append(header.to_sip())

        # Empty line separator
        lines.append("")

        # Body
        if self.body:
            lines.append(self.body)

        return "\r\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result: Dict[str, Any] = {
            "message_type": self.message_type.value,
        }

        if self.is_request and self.request_line:
            result["method"] = self.request_line.method
            result["request_uri"] = self.request_line.request_uri
            result["sip_version"] = self.request_line.sip_version
        elif self.is_response and self.response_line:
            result["status_code"] = self.response_line.status_code
            result["reason_phrase"] = self.response_line.reason_phrase
            result["sip_version"] = self.response_line.sip_version

        result["call_id"] = self.call_id
        result["from"] = self.from_header.to_sip() if self.from_header else None
        result["to"] = self.to_header.to_sip() if self.to_header else None
        result["cseq"] = self.cseq_header.to_sip() if self.cseq_header else None
        result["branch"] = self.branch

        # All headers
        result["headers"] = {}
        for h in self.headers:
            name = h.canonical_name
            if name in result["headers"]:
                if isinstance(result["headers"][name], list):
                    result["headers"][name].append(h.display_value)
                else:
                    result["headers"][name] = [result["headers"][name], h.display_value]
            else:
                result["headers"][name] = h.display_value

        # BPO context
        result["bpo_context"] = self.bpo_context.to_dict()

        # SDP
        if self.sdp:
            result["sdp"] = {
                "has_audio": self.sdp.has_audio,
                "has_video": self.sdp.has_video,
                "is_secure": self.sdp.is_secure,
                "media_count": len(self.sdp.media_descriptions),
            }

        result["content_type"] = self.content_type
        result["content_length"] = self.content_length

        return result

    def __str__(self) -> str:
        if self.is_request and self.request_line:
            return f"SIPMessage({self.request_line.method} {self.request_line.request_uri}, call-id={self.call_id})"
        elif self.is_response and self.response_line:
            return f"SIPMessage({self.response_line.status_code} {self.response_line.reason_phrase}, call-id={self.call_id})"
        return "SIPMessage(unknown)"
