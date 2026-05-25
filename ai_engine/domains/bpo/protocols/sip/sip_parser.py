"""
SIP Message Parser

Parses SIP (RFC 3261) messages from raw text format into structured
SIPMessage objects for BPO/Call Center operations.

Handles:
- Request and response message parsing
- Header extraction (standard + compact forms)
- Multi-line header folding (RFC 3261 Section 7.3.1)
- SDP body parsing (RFC 4566)
- Call center custom X-CC-* headers
- Via, From, To, CSeq structured header parsing
- Required header validation per RFC 3261
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ai_engine.domains.bpo.protocols.sip.sip_message import (
    SIPMessage,
    SIPHeader,
    SIPRequestLine,
    SIPResponseLine,
    SIPViaHeader,
    SIPAddressHeader,
    SIPCSeqHeader,
    SDPBody,
    SDPMediaDescription,
    BPOContext,
    MessageType,
)


class SIPParseError(Exception):
    """Exception raised when SIP message parsing fails."""

    def __init__(self, message: str, position: int = 0, raw_content: str = ""):
        super().__init__(message)
        self.position = position
        self.raw_content = raw_content


# RFC 3261 Section 7.3.3 - compact form mapping
COMPACT_FORM_MAP: Dict[str, str] = {
    "v": "Via",
    "f": "From",
    "t": "To",
    "i": "Call-ID",
    "m": "Contact",
    "c": "Content-Type",
    "l": "Content-Length",
    "k": "Supported",
    "b": "Referred-By",
}

# SIP methods defined in RFC 3261 and extensions
KNOWN_METHODS = {
    "INVITE", "ACK", "BYE", "CANCEL", "REGISTER", "OPTIONS",
    "REFER", "SUBSCRIBE", "NOTIFY", "MESSAGE", "INFO", "UPDATE",
    "PRACK", "PUBLISH",
}

# Required headers per RFC 3261 Section 8.1.1
REQUIRED_REQUEST_HEADERS = {"Via", "To", "From", "Call-ID", "CSeq", "Max-Forwards"}
REQUIRED_RESPONSE_HEADERS = {"Via", "To", "From", "Call-ID", "CSeq"}


class SIPParser:
    """
    Parser for SIP messages.

    Handles:
    - Start line detection (request vs. response)
    - Header extraction with compact form expansion
    - Multi-line header folding (LWS continuation)
    - SDP body parsing
    - Structured header parsing (Via, From, To, CSeq)
    - X-CC-* call center custom header extraction
    - RFC 3261 required header validation
    """

    # Start line patterns
    REQUEST_LINE_PATTERN = re.compile(
        r"^([A-Z]+)\s+(sips?:\S+)\s+(SIP/\d+\.\d+)\s*$"
    )
    RESPONSE_LINE_PATTERN = re.compile(
        r"^(SIP/\d+\.\d+)\s+(\d{3})\s+(.*?)\s*$"
    )

    # Header line pattern
    HEADER_PATTERN = re.compile(r"^([^\s:]+)\s*:\s*(.*)$")

    # Header continuation (LWS - linear whitespace)
    CONTINUATION_PATTERN = re.compile(r"^[ \t]+(.*)$")

    # Via header parsing
    VIA_PATTERN = re.compile(
        r"^(SIP/\d+\.\d+)/(UDP|TCP|TLS|WSS|TLS-PQC|SCTP)\s+"
        r"([^\s;:]+)(?::(\d+))?(.*)$"
    )

    # Address header parsing (From, To, Contact)
    ADDRESS_PATTERN = re.compile(
        r'^(?:"([^"]*)")?\s*<?([^>;\s]+)>?\s*(.*)$'
    )

    # CSeq header parsing
    CSEQ_PATTERN = re.compile(r"^(\d+)\s+([A-Z]+)\s*$")

    # SDP patterns
    SDP_FIELD_PATTERN = re.compile(r"^([a-z])=(.*)$")
    SDP_MEDIA_PATTERN = re.compile(
        r"^(audio|video|application|message)\s+(\d+)(?:/\d+)?\s+(\S+)\s+(.+)$"
    )

    def __init__(self, strict: bool = True):
        """
        Initialize parser.

        Args:
            strict: If True, raise errors for validation failures.
                    If False, collect errors in parse_errors list.
        """
        self.strict = strict

    def parse(self, raw_message: str) -> SIPMessage:
        """
        Parse a SIP message from raw text.

        Args:
            raw_message: Raw SIP message text (request or response)

        Returns:
            Parsed SIPMessage object

        Raises:
            SIPParseError: If parsing fails in strict mode
        """
        message = SIPMessage()
        message.raw_message = raw_message

        # Normalize line endings to CRLF per RFC 3261
        normalized = raw_message.replace("\r\n", "\n").replace("\r", "\n")

        # Split into head and body at empty line
        head, body = self._split_head_body(normalized)

        if not head:
            self._handle_error(message, "Empty message: no start line or headers", 0, raw_message)
            return message

        # Split head into lines
        lines = head.split("\n")

        # Parse start line
        start_line = lines[0].strip()
        if not start_line:
            self._handle_error(message, "Empty start line", 0, raw_message)
            return message

        self._parse_start_line(start_line, message)

        # Parse headers (with multi-line folding)
        raw_headers = self._unfold_headers(lines[1:])
        for line_num, header_line in raw_headers:
            try:
                header = self._parse_header_line(header_line, line_num)
                if header:
                    message.headers.append(header)
            except Exception as e:
                self._handle_error(
                    message,
                    f"Error parsing header at line {line_num}: {str(e)}",
                    line_num,
                    raw_message,
                )

        # Parse structured headers (Via, From, To, CSeq)
        self._parse_structured_headers(message)

        # Extract BPO context from X-CC-* headers
        self._extract_bpo_context(message)

        # Parse body
        if body:
            message.body = body
            content_type = message.content_type
            if content_type and "application/sdp" in content_type.lower():
                try:
                    message.sdp = self._parse_sdp(body)
                except Exception as e:
                    self._handle_error(
                        message,
                        f"Error parsing SDP body: {str(e)}",
                        0,
                        raw_message,
                    )

        # Validate required headers
        self._validate_required_headers(message)

        if self.strict and message.parse_errors:
            raise SIPParseError(
                f"Parse errors: {'; '.join(message.parse_errors)}",
                0,
                raw_message,
            )

        return message

    def _split_head_body(self, text: str) -> Tuple[str, str]:
        """Split message into head (start-line + headers) and body at empty line."""
        # RFC 3261: headers and body separated by a blank line
        separator_idx = text.find("\n\n")
        if separator_idx != -1:
            head = text[:separator_idx]
            body = text[separator_idx + 2:]
            return head, body
        return text, ""

    def _parse_start_line(self, line: str, message: SIPMessage) -> None:
        """Parse the first line to determine if request or response."""
        # Check for response (starts with SIP/x.y)
        response_match = self.RESPONSE_LINE_PATTERN.match(line)
        if response_match:
            message.message_type = MessageType.RESPONSE
            message.response_line = SIPResponseLine(
                sip_version=response_match.group(1),
                status_code=int(response_match.group(2)),
                reason_phrase=response_match.group(3),
            )
            return

        # Check for request (METHOD uri SIP/x.y)
        request_match = self.REQUEST_LINE_PATTERN.match(line)
        if request_match:
            message.message_type = MessageType.REQUEST
            message.request_line = SIPRequestLine(
                method=request_match.group(1),
                request_uri=request_match.group(2),
                sip_version=request_match.group(3),
            )
            return

        # Fallback: try splitting by whitespace
        parts = line.split(None, 2)
        if len(parts) >= 3:
            if parts[0].startswith("SIP/"):
                # Response
                message.message_type = MessageType.RESPONSE
                try:
                    status_code = int(parts[1])
                except ValueError:
                    status_code = 0
                message.response_line = SIPResponseLine(
                    sip_version=parts[0],
                    status_code=status_code,
                    reason_phrase=parts[2],
                )
            elif parts[2].startswith("SIP/"):
                # Request
                message.message_type = MessageType.REQUEST
                message.request_line = SIPRequestLine(
                    method=parts[0],
                    request_uri=parts[1],
                    sip_version=parts[2],
                )
            else:
                self._handle_error(
                    message,
                    f"Unable to parse start line: {line}",
                    0,
                    line,
                )
        else:
            self._handle_error(
                message,
                f"Malformed start line (expected 3 parts): {line}",
                0,
                line,
            )

    def _unfold_headers(self, lines: List[str]) -> List[Tuple[int, str]]:
        """
        Handle multi-line header folding per RFC 3261 Section 7.3.1.

        Headers can continue on the next line if it starts with
        SP (space) or HT (horizontal tab). This method unfolds
        those continuations into single logical header lines.

        Returns list of (line_number, unfolded_header_line) tuples.
        """
        result: List[Tuple[int, str]] = []
        current_line = ""
        current_line_num = 0

        for idx, line in enumerate(lines, start=2):  # start at 2 (line 1 is start line)
            stripped = line.rstrip()

            if not stripped:
                # Empty line marks end of headers
                break

            continuation_match = self.CONTINUATION_PATTERN.match(stripped)
            if continuation_match and current_line:
                # This is a continuation line - append to current header
                current_line += " " + continuation_match.group(1).strip()
            else:
                # Save previous header and start new one
                if current_line:
                    result.append((current_line_num, current_line))
                current_line = stripped
                current_line_num = idx

        # Save last header
        if current_line:
            result.append((current_line_num, current_line))

        return result

    def _parse_header_line(self, line: str, line_number: int) -> Optional[SIPHeader]:
        """Parse a single header line into a SIPHeader object."""
        match = self.HEADER_PATTERN.match(line)
        if not match:
            return None

        name = match.group(1).strip()
        value = match.group(2).strip()

        # Expand compact form to canonical name
        canonical = COMPACT_FORM_MAP.get(name.lower(), name)

        # Parse parameters from value (for headers with ;param=value format)
        base_value, parameters = self._parse_header_parameters(value, canonical)

        return SIPHeader(
            name=canonical,
            value=base_value,
            parameters=parameters,
            line_number=line_number,
        )

    def _parse_header_parameters(
        self, value: str, header_name: str
    ) -> Tuple[str, Dict[str, str]]:
        """
        Parse parameters from a header value.

        For headers like Via, From, To, Contact that have ;param=value syntax.
        We need to be careful not to split URIs that contain semicolons.
        """
        parameters: Dict[str, str] = {}

        # For address-style headers, handle angle brackets carefully
        if header_name in ("Via", "From", "To", "Contact", "Referred-By", "Record-Route", "Route"):
            # Find the end of the URI (after > or after the bare URI)
            in_angle = False
            uri_end = 0
            for i, ch in enumerate(value):
                if ch == "<":
                    in_angle = True
                elif ch == ">" and in_angle:
                    in_angle = False
                    uri_end = i + 1
                    break
            else:
                # No angle brackets found, try to find first semicolon
                # For Via headers, parameters start after the sent-by
                if header_name == "Via":
                    # Via format: SIP/2.0/UDP host:port;params
                    # Find the first semicolon after the host
                    space_idx = value.find(" ")
                    if space_idx != -1:
                        semi_idx = value.find(";", space_idx)
                        if semi_idx != -1:
                            uri_end = semi_idx
                        else:
                            return value, parameters
                    else:
                        return value, parameters
                else:
                    uri_end = 0

            if uri_end > 0 and uri_end < len(value):
                remainder = value[uri_end:]
                base_value = value[:uri_end].strip()

                # Parse parameters from remainder
                parts = remainder.split(";")
                for part in parts:
                    part = part.strip()
                    if not part:
                        continue
                    if "=" in part:
                        k, v = part.split("=", 1)
                        parameters[k.strip()] = v.strip()
                    else:
                        parameters[part] = ""

                return base_value, parameters

        return value, parameters

    def _parse_structured_headers(self, message: SIPMessage) -> None:
        """Parse Via, From, To, CSeq into structured dataclasses."""
        # Parse Via headers
        for via_hdr in message.get_headers("Via"):
            via = self._parse_via_value(via_hdr.value, via_hdr.parameters)
            if via:
                message.via_headers.append(via)

        # Parse From header
        from_hdr = message.get_header("From")
        if from_hdr:
            message.from_header = self._parse_address_value(
                from_hdr.value, from_hdr.parameters
            )

        # Parse To header
        to_hdr = message.get_header("To")
        if to_hdr:
            message.to_header = self._parse_address_value(
                to_hdr.value, to_hdr.parameters
            )

        # Parse CSeq header
        cseq_hdr = message.get_header("CSeq")
        if cseq_hdr:
            message.cseq_header = self._parse_cseq_value(cseq_hdr.value)

    def _parse_via_value(
        self, value: str, parameters: Dict[str, str]
    ) -> Optional[SIPViaHeader]:
        """Parse a Via header value into SIPViaHeader."""
        # Reconstruct full value if parameters were already extracted
        full_value = value
        if parameters:
            param_str = ";".join(
                f"{k}={v}" if v else k for k, v in parameters.items()
            )
            full_value = f"{value};{param_str}"

        match = self.VIA_PATTERN.match(full_value)
        if not match:
            # Try simpler parsing
            parts = full_value.split(";")
            transport_host = parts[0].strip()
            via = SIPViaHeader()

            # Parse transport and host
            tp_parts = transport_host.split("/")
            if len(tp_parts) >= 3:
                via.protocol = f"{tp_parts[0]}/{tp_parts[1]}"
                transport_and_host = tp_parts[2].strip()
                space_idx = transport_and_host.find(" ")
                if space_idx != -1:
                    via.transport = transport_and_host[:space_idx]
                    host_port = transport_and_host[space_idx + 1:].strip()
                else:
                    via.transport = transport_and_host
                    host_port = ""

                if host_port:
                    if ":" in host_port:
                        h, p = host_port.rsplit(":", 1)
                        via.host = h
                        try:
                            via.port = int(p)
                        except ValueError:
                            via.host = host_port
                    else:
                        via.host = host_port

            # Parse parameters
            for param in parts[1:]:
                param = param.strip()
                if "=" in param:
                    k, v = param.split("=", 1)
                    k = k.strip()
                    v = v.strip()
                    if k == "branch":
                        via.branch = v
                    elif k == "received":
                        via.received = v
                    elif k == "rport":
                        via.rport = v
                    else:
                        via.parameters[k] = v
                elif param:
                    if param == "rport":
                        via.rport = ""
                    else:
                        via.parameters[param] = ""

            return via

        via = SIPViaHeader(
            protocol=match.group(1),
            transport=match.group(2),
            host=match.group(3),
            port=int(match.group(4)) if match.group(4) else 5060,
        )

        # Parse parameters from remainder
        remainder = match.group(5)
        if remainder:
            params = remainder.split(";")
            for param in params:
                param = param.strip()
                if not param:
                    continue
                if "=" in param:
                    k, v = param.split("=", 1)
                    k = k.strip()
                    v = v.strip()
                    if k == "branch":
                        via.branch = v
                    elif k == "received":
                        via.received = v
                    elif k == "rport":
                        via.rport = v
                    else:
                        via.parameters[k] = v
                elif param:
                    if param == "rport":
                        via.rport = ""
                    else:
                        via.parameters[param] = ""

        return via

    def _parse_address_value(
        self, value: str, parameters: Dict[str, str]
    ) -> SIPAddressHeader:
        """Parse a From/To/Contact header value into SIPAddressHeader."""
        addr = SIPAddressHeader()

        # Reconstruct full value with parameters for parsing
        full_value = value
        if parameters:
            param_str = ";".join(
                f"{k}={v}" if v else k for k, v in parameters.items()
            )
            full_value = f"{value};{param_str}"

        # Try to extract display name and URI
        match = self.ADDRESS_PATTERN.match(full_value)
        if match:
            addr.display_name = match.group(1) or ""
            addr.uri = match.group(2).strip().strip("<>")
            remainder = match.group(3) or ""

            # Parse tag and other parameters from remainder
            if remainder:
                parts = remainder.split(";")
                for part in parts:
                    part = part.strip()
                    if not part:
                        continue
                    if "=" in part:
                        k, v = part.split("=", 1)
                        k = k.strip()
                        v = v.strip()
                        if k == "tag":
                            addr.tag = v
                        else:
                            addr.parameters[k] = v
                    elif part:
                        addr.parameters[part] = ""
        else:
            # Fallback: use raw value as URI
            addr.uri = value.strip().strip("<>")

        # Check if tag was in pre-parsed parameters
        if not addr.tag and "tag" in parameters:
            addr.tag = parameters.pop("tag")

        # Merge remaining parameters
        for k, v in parameters.items():
            if k != "tag":
                addr.parameters[k] = v

        return addr

    def _parse_cseq_value(self, value: str) -> Optional[SIPCSeqHeader]:
        """Parse a CSeq header value into SIPCSeqHeader."""
        match = self.CSEQ_PATTERN.match(value.strip())
        if match:
            return SIPCSeqHeader(
                sequence=int(match.group(1)),
                method=match.group(2),
            )

        # Fallback: split on whitespace
        parts = value.strip().split()
        if len(parts) >= 2:
            try:
                return SIPCSeqHeader(
                    sequence=int(parts[0]),
                    method=parts[1],
                )
            except ValueError:
                pass

        return None

    def _extract_bpo_context(self, message: SIPMessage) -> None:
        """Extract BPO/call center context from X-CC-* custom headers."""
        ctx = BPOContext()

        custom = message.get_custom_headers("X-CC-")
        ctx.tenant_id = custom.get("X-CC-Tenant", "")
        ctx.queue_name = custom.get("X-CC-Queue", "")
        ctx.agent_id = custom.get("X-CC-Agent", "")
        ctx.campaign_id = custom.get("X-CC-Campaign", "")
        ctx.call_priority = custom.get("X-CC-Priority", "")
        ctx.skill_group = custom.get("X-CC-Skill", "")
        ctx.call_type = custom.get("X-CC-CallType", "")
        ctx.recording_id = custom.get("X-CC-RecordingID", "")
        ctx.pci_mode = custom.get("X-CC-PCI-Mode", "")

        message.bpo_context = ctx

    def _parse_sdp(self, body: str) -> SDPBody:
        """
        Parse an SDP body (RFC 4566) from raw text.

        Args:
            body: Raw SDP body text

        Returns:
            Parsed SDPBody object
        """
        sdp = SDPBody()
        sdp.raw_body = body

        lines = body.strip().split("\n")
        current_media: Optional[SDPMediaDescription] = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            match = self.SDP_FIELD_PATTERN.match(line)
            if not match:
                continue

            field_type = match.group(1)
            field_value = match.group(2).strip()

            if field_type == "v":
                sdp.version = field_value

            elif field_type == "o":
                sdp.origin = field_value

            elif field_type == "s":
                sdp.session_name = field_value

            elif field_type == "c":
                if current_media:
                    current_media.connection = field_value
                else:
                    sdp.connection = field_value

            elif field_type == "t":
                sdp.timing = field_value

            elif field_type == "b":
                if current_media:
                    current_media.bandwidth = field_value

            elif field_type == "m":
                # New media description
                current_media = self._parse_sdp_media_line(field_value)
                if current_media:
                    sdp.media_descriptions.append(current_media)

            elif field_type == "a":
                # Attribute
                if ":" in field_value:
                    attr_name, attr_value = field_value.split(":", 1)
                else:
                    attr_name = field_value
                    attr_value = ""

                if current_media:
                    # Handle duplicate attribute names by appending index
                    base_name = attr_name
                    if base_name in current_media.attributes:
                        idx = 1
                        while f"{base_name}_{idx}" in current_media.attributes:
                            idx += 1
                        attr_name = f"{base_name}_{idx}"
                    current_media.attributes[attr_name] = attr_value
                else:
                    if attr_name in sdp.attributes:
                        idx = 1
                        while f"{attr_name}_{idx}" in sdp.attributes:
                            idx += 1
                        attr_name = f"{attr_name}_{idx}"
                    sdp.attributes[attr_name] = attr_value

        return sdp

    def _parse_sdp_media_line(self, value: str) -> Optional[SDPMediaDescription]:
        """Parse an SDP m= line value."""
        match = self.SDP_MEDIA_PATTERN.match(value)
        if match:
            formats = match.group(4).strip().split()
            return SDPMediaDescription(
                media_type=match.group(1),
                port=int(match.group(2)),
                protocol=match.group(3),
                formats=formats,
            )

        # Fallback: simple split
        parts = value.split()
        if len(parts) >= 4:
            try:
                return SDPMediaDescription(
                    media_type=parts[0],
                    port=int(parts[1]),
                    protocol=parts[2],
                    formats=parts[3:],
                )
            except ValueError:
                pass

        return None

    def _validate_required_headers(self, message: SIPMessage) -> None:
        """Validate that required headers per RFC 3261 are present."""
        present_headers = set()
        for h in message.headers:
            present_headers.add(h.canonical_name)

        if message.is_request:
            required = REQUIRED_REQUEST_HEADERS
        else:
            required = REQUIRED_RESPONSE_HEADERS

        for header_name in required:
            # Check both canonical and compact forms
            if header_name not in present_headers:
                message.parse_errors.append(
                    f"Missing required header: {header_name}"
                )

    def _handle_error(
        self, message: SIPMessage, error_msg: str, position: int, raw_content: str
    ) -> None:
        """Handle a parse error based on strict mode setting."""
        message.parse_errors.append(error_msg)
        if self.strict:
            raise SIPParseError(error_msg, position, raw_content)

    # --- Convenience parsing methods ---

    def parse_dtmf_body(self, body: str) -> Optional[Dict[str, str]]:
        """
        Parse DTMF relay body from INFO method messages.

        Supports both application/dtmf-relay and application/dtmf formats.

        Args:
            body: Raw body content of INFO message

        Returns:
            Dictionary with 'signal' and 'duration' keys, or None
        """
        result: Dict[str, str] = {}

        for line in body.strip().split("\n"):
            line = line.strip()
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip().lower()
                value = value.strip()

                if key == "signal":
                    result["signal"] = value
                elif key == "duration":
                    result["duration"] = value
            elif line.startswith("Signal"):
                # Alternative format: "Signal=X" or "Signal: X"
                for sep in ("=", ":"):
                    if sep in line:
                        result["signal"] = line.split(sep, 1)[1].strip()
                        break
            elif line.startswith("Duration"):
                for sep in ("=", ":"):
                    if sep in line:
                        result["duration"] = line.split(sep, 1)[1].strip()
                        break

        return result if result else None

    def extract_phone_number(self, uri: str) -> Optional[str]:
        """
        Extract a phone number from a SIP URI.

        Handles formats:
        - sip:+14155551234@domain
        - sip:14155551234@domain
        - tel:+14155551234
        - sip:agent@domain;user=phone

        Args:
            uri: SIP or TEL URI string

        Returns:
            Phone number string (digits and optional leading +), or None
        """
        if not uri:
            return None

        # Strip scheme
        if ":" in uri:
            scheme, rest = uri.split(":", 1)
        else:
            rest = uri

        # Strip domain/host
        if "@" in rest:
            user_part = rest.split("@", 1)[0]
        else:
            # TEL URI or bare number
            user_part = rest

        # Strip parameters
        if ";" in user_part:
            user_part = user_part.split(";", 1)[0]

        # Check if it looks like a phone number
        cleaned = user_part.lstrip("+")
        if cleaned.isdigit():
            return user_part

        return None


def parse_sip_message(raw_message: str, strict: bool = True) -> SIPMessage:
    """
    Convenience function to parse a SIP message.

    Args:
        raw_message: Raw SIP message text
        strict: If True, raise errors for validation failures

    Returns:
        Parsed SIPMessage object
    """
    parser = SIPParser(strict=strict)
    return parser.parse(raw_message)
