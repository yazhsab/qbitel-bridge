"""
SIP Protocol Message Generator

Generates realistic SIP (Session Initiation Protocol - RFC 3261) messages
for training the protocol discovery and field detection models.

Message types generated:
- INVITE: Session initiation with SDP body and BPO headers
- REGISTER: Registration with authentication
- BYE: Session termination
- ACK: Acknowledgment
- CANCEL: Request cancellation
- OPTIONS: Server capability query
- INFO: Mid-dialog signaling (DTMF relay)
- REFER: Call transfer
- 1xx Responses: Provisional (100 Trying, 180 Ringing, 183 Session Progress)
- 2xx Responses: Success (200 OK)
- 4xx Responses: Client errors (401 Unauthorized, 403 Forbidden, 486 Busy)
- 5xx Responses: Server errors (500 Internal Error, 503 Service Unavailable)
- PQC-INVITE: Post-quantum crypto SIP with Kyber key exchange

Includes BPO-specific X-CC-* headers per sip_message.py BPOContext:
  X-CC-Tenant, X-CC-Queue, X-CC-Agent, X-CC-Campaign, X-CC-Priority,
  X-CC-Skill, X-CC-CallType, X-CC-RecordingID, X-CC-PCI-Mode
"""

import json
import random
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class SIPGenerator:
    """Generate realistic SIP messages for ML training."""

    # SIP request methods
    METHODS = [
        "INVITE", "REGISTER", "BYE", "ACK", "CANCEL",
        "OPTIONS", "INFO", "REFER",
    ]

    # Response status codes
    RESPONSE_CODES = {
        100: "Trying",
        180: "Ringing",
        183: "Session Progress",
        200: "OK",
        401: "Unauthorized",
        403: "Forbidden",
        404: "Not Found",
        486: "Busy Here",
        487: "Request Terminated",
        500: "Server Internal Error",
        503: "Service Unavailable",
    }

    # Audio codecs for SDP
    CODECS = [
        {"pt": 0, "name": "PCMU", "rate": 8000},
        {"pt": 8, "name": "PCMA", "rate": 8000},
        {"pt": 9, "name": "G722", "rate": 8000},
        {"pt": 18, "name": "G729", "rate": 8000},
        {"pt": 101, "name": "telephone-event", "rate": 8000},
    ]

    # Vendor User-Agents
    VENDOR_USER_AGENTS = [
        "Avaya Aura CM 8.1.3",
        "Avaya Session Manager 8.1",
        "Cisco-CUCM14.0",
        "Cisco/SPA525G-7.6.2d",
        "Genesys Engage SIP Server 9.0.018",
        "Genesys Cloud Edge/1.0",
        "Mitel MiVoice 5000/7.3",
        "Mitel SIP-DECT 8.0",
        "Asterisk PBX 20.5.0",
        "FreeSWITCH-mod_sofia/1.10.9",
        "OpenSIPS/3.4.0",
        "Opal/3.18.8 (Linux)",
        "Opal/3.18.8 (Windows)",
        "Opal/3.18.8 (Mac)",
        "OWASP-ZAP/2.14.0",
        "Opal/3.18.8 (Android)",
        "Opal/3.18.8 (iOS)",
        "OWASP-ZAP/2.14.0",
    ]

    # BPO queue names
    QUEUES = ["sales", "support", "billing", "retention", "collections", "tech"]
    SKILL_GROUPS = ["english", "spanish", "french", "billing", "tech", "sales", "retention"]
    CAMPAIGNS = ["Q4_WINBACK", "UPSELL_2024", "RENEWAL", "COLLECTIONS_Q1", "SURVEY"]
    CALL_TYPES = ["inbound", "outbound", "internal", "callback", "preview_dial"]
    PCI_MODES = ["disabled", "active_masking", "pause_resume", "secure_relay"]

    # Domains
    DOMAINS = [
        "bpo.example.com", "cc.contoso.com", "pbx.acme.corp",
        "sip.callcenter.net", "voice.enterprise.io", "uc.global.com",
    ]

    # Crypto suites for SRTP
    CRYPTO_SUITES = [
        "AES_CM_128_HMAC_SHA1_80",
        "AES_CM_128_HMAC_SHA1_32",
        "AES_256_CM_HMAC_SHA1_80",
        "AEAD_AES_128_GCM",
        "AEAD_AES_256_GCM",
    ]

    def __init__(self, seed: Optional[int] = None):
        if seed:
            random.seed(seed)
        self.cseq_counter = random.randint(1, 10000)
        self.call_id_pool = [
            f"{random.randint(100000, 999999)}@{random.choice(self.DOMAINS)}"
            for _ in range(100)
        ]
        self.tag_pool = [
            f"{random.randint(10000000, 99999999)}" for _ in range(200)
        ]
        self.branch_pool = [
            f"z9hG4bK-{random.randint(100000, 999999)}-{random.randint(1, 999)}"
            for _ in range(200)
        ]
        self.agent_ids = [f"agent_{i:04d}" for i in range(1, 201)]
        self.tenant_ids = [f"tenant_{i:03d}" for i in range(1, 21)]
        self.extensions = [f"{random.randint(1000, 9999)}" for _ in range(200)]

    def _next_cseq(self) -> int:
        self.cseq_counter += 1
        return self.cseq_counter

    def _random_ip(self) -> str:
        return f"{random.randint(10, 192)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"

    def _random_phone(self) -> str:
        return f"+1{random.randint(200, 999)}{random.randint(2000000, 9999999)}"

    def _build_via(self, ip: str, port: int = 5060, transport: str = "UDP") -> str:
        branch = random.choice(self.branch_pool)
        return f"SIP/2.0/{transport} {ip}:{port};branch={branch}"

    def _build_bpo_headers(self) -> Tuple[str, Dict]:
        """Build BPO X-CC-* extension headers."""
        tenant = random.choice(self.tenant_ids)
        queue = random.choice(self.QUEUES)
        agent = random.choice(self.agent_ids)
        campaign = random.choice(self.CAMPAIGNS) if random.random() < 0.6 else ""
        priority = random.randint(1, 10)
        skill = random.choice(self.SKILL_GROUPS)
        call_type = random.choice(self.CALL_TYPES)
        recording_id = f"rec_{random.randint(100000, 999999)}"
        pci_mode = random.choice(self.PCI_MODES)

        lines = [
            f"X-CC-Tenant: {tenant}",
            f"X-CC-Queue: {queue}",
            f"X-CC-Agent: {agent}",
            f"X-CC-Priority: {priority}",
            f"X-CC-Skill: {skill}",
            f"X-CC-CallType: {call_type}",
            f"X-CC-RecordingID: {recording_id}",
            f"X-CC-PCI-Mode: {pci_mode}",
        ]
        if campaign:
            lines.insert(3, f"X-CC-Campaign: {campaign}")

        bpo_meta = {
            "bpo_tenant": tenant,
            "bpo_queue": queue,
            "bpo_agent": agent,
            "bpo_campaign": campaign,
            "bpo_priority": priority,
            "bpo_skill": skill,
            "bpo_call_type": call_type,
            "bpo_recording_id": recording_id,
            "bpo_pci_mode": pci_mode,
        }
        return "\r\n".join(lines), bpo_meta

    def _build_sdp(self, ip: str, port: int = None) -> Tuple[str, Dict]:
        """Build an SDP body for audio with optional SRTP."""
        if port is None:
            port = random.choice([10000, 10002, 10004, 20000, 20002, 30000])

        use_srtp = random.random() < 0.4
        codecs = random.sample(self.CODECS, k=random.randint(2, 4))
        codec_lines = []
        fmtp_lines = []

        for c in codecs:
            codec_lines.append(f"a=rtpmap:{c['pt']} {c['name']}/{c['rate']}")
            if c["name"] == "telephone-event":
                fmtp_lines.append(f"a=fmtp:{c['pt']} 0-16")

        media_fmt = " ".join(str(c["pt"]) for c in codecs)
        proto = "RTP/SAVP" if use_srtp else "RTP/AVP"

        sdp_lines = [
            "v=0",
            f"o=- {random.randint(1000000, 9999999)} {random.randint(1, 10)} IN IP4 {ip}",
            f"s=QBITEL BPO Call",
            f"c=IN IP4 {ip}",
            "t=0 0",
            f"m=audio {port} {proto} {media_fmt}",
        ]
        sdp_lines.extend(codec_lines)
        sdp_lines.extend(fmtp_lines)
        sdp_lines.append(f"a=ptime:20")
        sdp_lines.append(f"a=sendrecv")

        crypto_suite = ""
        if use_srtp:
            crypto_suite = random.choice(self.CRYPTO_SUITES)
            key_material = "".join(
                random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/", k=40)
            )
            sdp_lines.append(f"a=crypto:1 {crypto_suite} inline:{key_material}")

        sdp_body = "\r\n".join(sdp_lines)

        sdp_meta = {
            "sdp_media_ip": ip,
            "sdp_media_port": port,
            "sdp_codecs": [c["name"] for c in codecs],
            "sdp_protocol": proto,
            "sdp_srtp": use_srtp,
        }
        if crypto_suite:
            sdp_meta["sdp_crypto_suite"] = crypto_suite

        return sdp_body, sdp_meta

    def _build_common_headers(
        self,
        method: str,
        from_uri: str,
        to_uri: str,
        call_id: str,
        cseq: int,
        via_ip: str,
        user_agent: str,
        content_type: str = "",
        content_length: int = 0,
    ) -> str:
        """Build common SIP headers."""
        from_tag = random.choice(self.tag_pool)
        to_tag = random.choice(self.tag_pool) if method != "INVITE" else ""

        headers = [
            self._build_via(via_ip),
            f"Max-Forwards: 70",
            f"From: <{from_uri}>;tag={from_tag}",
        ]
        if to_tag:
            headers.append(f"To: <{to_uri}>;tag={to_tag}")
        else:
            headers.append(f"To: <{to_uri}>")

        headers.extend([
            f"Call-ID: {call_id}",
            f"CSeq: {cseq} {method}",
            f"User-Agent: {user_agent}",
            f"Allow: INVITE,ACK,BYE,CANCEL,OPTIONS,INFO,REFER,NOTIFY,SUBSCRIBE",
            f"Supported: replaces,timer,100rel",
        ])

        if content_type:
            headers.append(f"Content-Type: {content_type}")
        headers.append(f"Content-Length: {content_length}")

        return "\r\n".join(headers)

    # ------------------------------------------------------------------
    # Message type generators
    # ------------------------------------------------------------------

    def generate_invite(self) -> Tuple[bytes, Dict]:
        """Generate a SIP INVITE request with SDP and BPO headers."""
        domain = random.choice(self.DOMAINS)
        from_ext = random.choice(self.extensions)
        to_ext = random.choice(self.extensions)
        from_uri = f"sip:{from_ext}@{domain}"
        to_uri = f"sip:{to_ext}@{domain}"
        call_id = random.choice(self.call_id_pool)
        cseq = self._next_cseq()
        via_ip = self._random_ip()
        ua = random.choice(self.VENDOR_USER_AGENTS)

        media_ip = self._random_ip()
        sdp_body, sdp_meta = self._build_sdp(media_ip)
        bpo_headers, bpo_meta = self._build_bpo_headers()

        request_line = f"INVITE {to_uri} SIP/2.0"
        common = self._build_common_headers(
            "INVITE", from_uri, to_uri, call_id, cseq,
            via_ip, ua, "application/sdp", len(sdp_body),
        )

        message_str = f"{request_line}\r\n{common}\r\n{bpo_headers}\r\n\r\n{sdp_body}"
        message = message_str.encode("utf-8")

        meta = {
            "sip_method": "INVITE",
            "from_uri": from_uri,
            "to_uri": to_uri,
            "call_id": call_id,
            "cseq": cseq,
            "user_agent": ua,
            "has_sdp": True,
            "has_bpo_headers": True,
        }
        meta.update(sdp_meta)
        meta.update(bpo_meta)

        fields = self._build_fields(message_str)
        return message, self._finalize_metadata("invite", message, meta, fields)

    def generate_register(self) -> Tuple[bytes, Dict]:
        """Generate a SIP REGISTER request."""
        domain = random.choice(self.DOMAINS)
        ext = random.choice(self.extensions)
        from_uri = f"sip:{ext}@{domain}"
        to_uri = f"sip:{ext}@{domain}"
        contact_uri = f"sip:{ext}@{self._random_ip()}:5060"
        call_id = random.choice(self.call_id_pool)
        cseq = self._next_cseq()
        via_ip = self._random_ip()
        ua = random.choice(self.VENDOR_USER_AGENTS)
        expires = random.choice([60, 120, 300, 600, 3600])

        request_line = f"REGISTER sip:{domain} SIP/2.0"
        common = self._build_common_headers(
            "REGISTER", from_uri, to_uri, call_id, cseq, via_ip, ua,
        )

        # Add auth header 50% of the time
        auth_header = ""
        has_auth = random.random() < 0.5
        if has_auth:
            nonce = hashlib.md5(str(random.random()).encode()).hexdigest()
            response_hash = hashlib.md5(str(random.random()).encode()).hexdigest()
            auth_header = (
                f'\r\nAuthorization: Digest username="{ext}", '
                f'realm="{domain}", nonce="{nonce}", '
                f'uri="sip:{domain}", response="{response_hash}", '
                f'algorithm=MD5'
            )

        message_str = (
            f"{request_line}\r\n{common}\r\n"
            f"Contact: <{contact_uri}>;expires={expires}\r\n"
            f"Expires: {expires}"
            f"{auth_header}\r\n\r\n"
        )
        message = message_str.encode("utf-8")

        meta = {
            "sip_method": "REGISTER",
            "from_uri": from_uri,
            "to_uri": to_uri,
            "call_id": call_id,
            "cseq": cseq,
            "user_agent": ua,
            "contact_uri": contact_uri,
            "expires": expires,
            "has_authorization": has_auth,
            "has_sdp": False,
            "has_bpo_headers": False,
        }

        fields = self._build_fields(message_str)
        return message, self._finalize_metadata("register", message, meta, fields)

    def generate_bye(self) -> Tuple[bytes, Dict]:
        """Generate a SIP BYE request."""
        domain = random.choice(self.DOMAINS)
        from_uri = f"sip:{random.choice(self.extensions)}@{domain}"
        to_uri = f"sip:{random.choice(self.extensions)}@{domain}"
        call_id = random.choice(self.call_id_pool)
        cseq = self._next_cseq()
        via_ip = self._random_ip()
        ua = random.choice(self.VENDOR_USER_AGENTS)

        bpo_headers, bpo_meta = self._build_bpo_headers()

        request_line = f"BYE {to_uri} SIP/2.0"
        common = self._build_common_headers(
            "BYE", from_uri, to_uri, call_id, cseq, via_ip, ua,
        )

        # Add Reason header for call termination cause
        reasons = [
            "Q.850;cause=16;text=\"Normal call clearing\"",
            "Q.850;cause=17;text=\"User busy\"",
            "Q.850;cause=31;text=\"Normal, unspecified\"",
            "SIP;cause=200;text=\"Call completed\"",
            "SIP;cause=487;text=\"Request terminated\"",
        ]
        reason = random.choice(reasons)

        message_str = (
            f"{request_line}\r\n{common}\r\n"
            f"Reason: {reason}\r\n"
            f"{bpo_headers}\r\n\r\n"
        )
        message = message_str.encode("utf-8")

        meta = {
            "sip_method": "BYE",
            "from_uri": from_uri,
            "to_uri": to_uri,
            "call_id": call_id,
            "cseq": cseq,
            "user_agent": ua,
            "reason": reason,
            "has_sdp": False,
            "has_bpo_headers": True,
        }
        meta.update(bpo_meta)

        fields = self._build_fields(message_str)
        return message, self._finalize_metadata("bye", message, meta, fields)

    def generate_ack(self) -> Tuple[bytes, Dict]:
        """Generate a SIP ACK request."""
        domain = random.choice(self.DOMAINS)
        from_uri = f"sip:{random.choice(self.extensions)}@{domain}"
        to_uri = f"sip:{random.choice(self.extensions)}@{domain}"
        call_id = random.choice(self.call_id_pool)
        cseq = self._next_cseq()
        via_ip = self._random_ip()
        ua = random.choice(self.VENDOR_USER_AGENTS)

        request_line = f"ACK {to_uri} SIP/2.0"
        common = self._build_common_headers(
            "ACK", from_uri, to_uri, call_id, cseq, via_ip, ua,
        )

        message_str = f"{request_line}\r\n{common}\r\n\r\n"
        message = message_str.encode("utf-8")

        meta = {
            "sip_method": "ACK",
            "from_uri": from_uri,
            "to_uri": to_uri,
            "call_id": call_id,
            "cseq": cseq,
            "user_agent": ua,
            "has_sdp": False,
            "has_bpo_headers": False,
        }

        fields = self._build_fields(message_str)
        return message, self._finalize_metadata("ack", message, meta, fields)

    def generate_cancel(self) -> Tuple[bytes, Dict]:
        """Generate a SIP CANCEL request."""
        domain = random.choice(self.DOMAINS)
        from_uri = f"sip:{random.choice(self.extensions)}@{domain}"
        to_uri = f"sip:{random.choice(self.extensions)}@{domain}"
        call_id = random.choice(self.call_id_pool)
        cseq = self._next_cseq()
        via_ip = self._random_ip()
        ua = random.choice(self.VENDOR_USER_AGENTS)

        request_line = f"CANCEL {to_uri} SIP/2.0"
        common = self._build_common_headers(
            "CANCEL", from_uri, to_uri, call_id, cseq, via_ip, ua,
        )

        message_str = f"{request_line}\r\n{common}\r\n\r\n"
        message = message_str.encode("utf-8")

        meta = {
            "sip_method": "CANCEL",
            "from_uri": from_uri,
            "to_uri": to_uri,
            "call_id": call_id,
            "cseq": cseq,
            "user_agent": ua,
            "has_sdp": False,
            "has_bpo_headers": False,
        }

        fields = self._build_fields(message_str)
        return message, self._finalize_metadata("cancel", message, meta, fields)

    def generate_options(self) -> Tuple[bytes, Dict]:
        """Generate a SIP OPTIONS request (capability query)."""
        domain = random.choice(self.DOMAINS)
        from_uri = f"sip:monitor@{domain}"
        to_uri = f"sip:{domain}"
        call_id = random.choice(self.call_id_pool)
        cseq = self._next_cseq()
        via_ip = self._random_ip()
        ua = random.choice(self.VENDOR_USER_AGENTS)

        request_line = f"OPTIONS {to_uri} SIP/2.0"
        common = self._build_common_headers(
            "OPTIONS", from_uri, to_uri, call_id, cseq, via_ip, ua,
        )

        message_str = f"{request_line}\r\n{common}\r\nAccept: application/sdp\r\n\r\n"
        message = message_str.encode("utf-8")

        meta = {
            "sip_method": "OPTIONS",
            "from_uri": from_uri,
            "to_uri": to_uri,
            "call_id": call_id,
            "cseq": cseq,
            "user_agent": ua,
            "has_sdp": False,
            "has_bpo_headers": False,
        }

        fields = self._build_fields(message_str)
        return message, self._finalize_metadata("options", message, meta, fields)

    def generate_info(self) -> Tuple[bytes, Dict]:
        """Generate a SIP INFO request (DTMF relay)."""
        domain = random.choice(self.DOMAINS)
        from_uri = f"sip:{random.choice(self.extensions)}@{domain}"
        to_uri = f"sip:{random.choice(self.extensions)}@{domain}"
        call_id = random.choice(self.call_id_pool)
        cseq = self._next_cseq()
        via_ip = self._random_ip()
        ua = random.choice(self.VENDOR_USER_AGENTS)

        digit = random.choice("0123456789*#")
        duration = random.randint(100, 500)
        dtmf_body = f"Signal={digit}\r\nDuration={duration}"

        request_line = f"INFO {to_uri} SIP/2.0"
        common = self._build_common_headers(
            "INFO", from_uri, to_uri, call_id, cseq, via_ip, ua,
            "application/dtmf-relay", len(dtmf_body),
        )

        bpo_headers, bpo_meta = self._build_bpo_headers()

        message_str = (
            f"{request_line}\r\n{common}\r\n"
            f"{bpo_headers}\r\n\r\n{dtmf_body}"
        )
        message = message_str.encode("utf-8")

        meta = {
            "sip_method": "INFO",
            "from_uri": from_uri,
            "to_uri": to_uri,
            "call_id": call_id,
            "cseq": cseq,
            "user_agent": ua,
            "dtmf_digit": digit,
            "dtmf_duration": duration,
            "has_sdp": False,
            "has_bpo_headers": True,
        }
        meta.update(bpo_meta)

        fields = self._build_fields(message_str)
        return message, self._finalize_metadata("info", message, meta, fields)

    def generate_refer(self) -> Tuple[bytes, Dict]:
        """Generate a SIP REFER request (call transfer)."""
        domain = random.choice(self.DOMAINS)
        from_uri = f"sip:{random.choice(self.extensions)}@{domain}"
        to_uri = f"sip:{random.choice(self.extensions)}@{domain}"
        refer_to = f"sip:{random.choice(self.extensions)}@{domain}"
        call_id = random.choice(self.call_id_pool)
        cseq = self._next_cseq()
        via_ip = self._random_ip()
        ua = random.choice(self.VENDOR_USER_AGENTS)

        bpo_headers, bpo_meta = self._build_bpo_headers()

        request_line = f"REFER {to_uri} SIP/2.0"
        common = self._build_common_headers(
            "REFER", from_uri, to_uri, call_id, cseq, via_ip, ua,
        )

        message_str = (
            f"{request_line}\r\n{common}\r\n"
            f"Refer-To: <{refer_to}>\r\n"
            f"Referred-By: <{from_uri}>\r\n"
            f"{bpo_headers}\r\n\r\n"
        )
        message = message_str.encode("utf-8")

        meta = {
            "sip_method": "REFER",
            "from_uri": from_uri,
            "to_uri": to_uri,
            "refer_to": refer_to,
            "call_id": call_id,
            "cseq": cseq,
            "user_agent": ua,
            "has_sdp": False,
            "has_bpo_headers": True,
        }
        meta.update(bpo_meta)

        fields = self._build_fields(message_str)
        return message, self._finalize_metadata("refer", message, meta, fields)

    def generate_response_1xx(self) -> Tuple[bytes, Dict]:
        """Generate a 1xx provisional SIP response."""
        code = random.choice([100, 180, 183])
        reason = self.RESPONSE_CODES[code]
        return self._generate_response(code, reason, "response_1xx", include_sdp=(code == 183))

    def generate_response_2xx(self) -> Tuple[bytes, Dict]:
        """Generate a 200 OK SIP response."""
        return self._generate_response(200, "OK", "response_2xx", include_sdp=True)

    def generate_response_4xx(self) -> Tuple[bytes, Dict]:
        """Generate a 4xx client error SIP response."""
        code = random.choice([401, 403, 404, 486, 487])
        reason = self.RESPONSE_CODES[code]
        return self._generate_response(code, reason, "response_4xx")

    def generate_response_5xx(self) -> Tuple[bytes, Dict]:
        """Generate a 5xx server error SIP response."""
        code = random.choice([500, 503])
        reason = self.RESPONSE_CODES[code]
        return self._generate_response(code, reason, "response_5xx")

    def generate_pqc_invite(self) -> Tuple[bytes, Dict]:
        """Generate a post-quantum crypto SIP INVITE with Kyber key exchange."""
        domain = random.choice(self.DOMAINS)
        from_ext = random.choice(self.extensions)
        to_ext = random.choice(self.extensions)
        from_uri = f"sip:{from_ext}@{domain}"
        to_uri = f"sip:{to_ext}@{domain}"
        call_id = random.choice(self.call_id_pool)
        cseq = self._next_cseq()
        via_ip = self._random_ip()
        ua = random.choice(self.VENDOR_USER_AGENTS)

        media_ip = self._random_ip()
        sdp_body, sdp_meta = self._build_sdp(media_ip)

        # Add PQC key exchange to SDP
        kyber_key = "".join(
            random.choices(
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/",
                k=64,
            )
        )
        pqc_algo = random.choice(["KYBER-768", "KYBER-1024", "ML-KEM-768", "ML-KEM-1024"])
        sdp_body += f"\r\na=pqc-kem:{pqc_algo} {kyber_key}"

        bpo_headers, bpo_meta = self._build_bpo_headers()

        request_line = f"INVITE {to_uri} SIP/2.0"
        common = self._build_common_headers(
            "INVITE", from_uri, to_uri, call_id, cseq,
            via_ip, ua, "application/sdp", len(sdp_body),
        )

        message_str = (
            f"{request_line}\r\n{common}\r\n"
            f"Require: pqc-key-exchange\r\n"
            f"Supported: pqc-kem,replaces,timer\r\n"
            f"{bpo_headers}\r\n\r\n{sdp_body}"
        )
        message = message_str.encode("utf-8")

        meta = {
            "sip_method": "INVITE",
            "from_uri": from_uri,
            "to_uri": to_uri,
            "call_id": call_id,
            "cseq": cseq,
            "user_agent": ua,
            "has_sdp": True,
            "has_bpo_headers": True,
            "pqc_algorithm": pqc_algo,
            "is_pqc": True,
        }
        meta.update(sdp_meta)
        meta.update(bpo_meta)

        fields = self._build_fields(message_str)
        return message, self._finalize_metadata("pqc_invite", message, meta, fields)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _generate_response(
        self,
        status_code: int,
        reason: str,
        msg_type: str,
        include_sdp: bool = False,
    ) -> Tuple[bytes, Dict]:
        """Generate a SIP response message."""
        domain = random.choice(self.DOMAINS)
        from_uri = f"sip:{random.choice(self.extensions)}@{domain}"
        to_uri = f"sip:{random.choice(self.extensions)}@{domain}"
        call_id = random.choice(self.call_id_pool)
        cseq = self._next_cseq()
        via_ip = self._random_ip()
        ua = random.choice(self.VENDOR_USER_AGENTS)
        method = random.choice(["INVITE", "REGISTER", "OPTIONS"])

        status_line = f"SIP/2.0 {status_code} {reason}"

        sdp_body = ""
        sdp_meta = {}
        content_type = ""
        if include_sdp and method == "INVITE":
            media_ip = self._random_ip()
            sdp_body, sdp_meta = self._build_sdp(media_ip)
            content_type = "application/sdp"

        from_tag = random.choice(self.tag_pool)
        to_tag = random.choice(self.tag_pool)

        headers = [
            self._build_via(via_ip),
            f"From: <{from_uri}>;tag={from_tag}",
            f"To: <{to_uri}>;tag={to_tag}",
            f"Call-ID: {call_id}",
            f"CSeq: {cseq} {method}",
            f"User-Agent: {ua}",
        ]

        # Add WWW-Authenticate for 401
        if status_code == 401:
            nonce = hashlib.md5(str(random.random()).encode()).hexdigest()
            headers.append(
                f'WWW-Authenticate: Digest realm="{domain}", '
                f'nonce="{nonce}", algorithm=MD5, qop="auth"'
            )

        # Add Contact for 200 OK
        if status_code == 200:
            headers.append(f"Contact: <sip:{random.choice(self.extensions)}@{self._random_ip()}>")

        if content_type:
            headers.append(f"Content-Type: {content_type}")
        headers.append(f"Content-Length: {len(sdp_body)}")

        header_str = "\r\n".join(headers)
        message_str = f"{status_line}\r\n{header_str}\r\n\r\n{sdp_body}"
        message = message_str.encode("utf-8")

        meta = {
            "sip_status_code": status_code,
            "sip_reason": reason,
            "sip_method": method,
            "from_uri": from_uri,
            "to_uri": to_uri,
            "call_id": call_id,
            "cseq": cseq,
            "user_agent": ua,
            "has_sdp": bool(sdp_body),
            "has_bpo_headers": False,
        }
        meta.update(sdp_meta)

        fields = self._build_fields(message_str)
        return message, self._finalize_metadata(msg_type, message, meta, fields)

    def _build_fields(self, message_str: str) -> List[Dict]:
        """Build field metadata from a SIP message string."""
        msg_bytes = message_str.encode("utf-8")
        fields = [
            {"name": "sip_message", "offset": 0, "length": len(msg_bytes)},
        ]

        # Find request/status line
        first_line_end = message_str.find("\r\n")
        if first_line_end > 0:
            fields.append({"name": "start_line", "offset": 0, "length": first_line_end})

        # Find header section
        body_sep = message_str.find("\r\n\r\n")
        if body_sep > 0:
            header_start = first_line_end + 2 if first_line_end > 0 else 0
            fields.append({
                "name": "headers",
                "offset": header_start,
                "length": body_sep - header_start,
            })
            if body_sep + 4 < len(msg_bytes):
                fields.append({
                    "name": "body",
                    "offset": body_sep + 4,
                    "length": len(msg_bytes) - body_sep - 4,
                })

        return fields

    def _finalize_metadata(
        self, msg_type: str, message: bytes, meta: Dict, fields: list
    ) -> Dict:
        """Create the final metadata dict."""
        meta.update({
            "protocol": "sip",
            "message_type": msg_type,
            "timestamp": datetime.now().isoformat(),
            "message_length": len(message),
            "fields": fields,
            "hash": hashlib.sha256(message).hexdigest(),
        })
        return meta

    def generate_dataset(self, num_samples: int, output_dir: str) -> Dict:
        """Generate a complete dataset of SIP messages."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        generators = [
            ("invite", self.generate_invite, 0.20),
            ("register", self.generate_register, 0.10),
            ("bye", self.generate_bye, 0.10),
            ("ack", self.generate_ack, 0.05),
            ("cancel", self.generate_cancel, 0.05),
            ("options", self.generate_options, 0.05),
            ("info", self.generate_info, 0.08),
            ("refer", self.generate_refer, 0.07),
            ("response_1xx", self.generate_response_1xx, 0.10),
            ("response_2xx", self.generate_response_2xx, 0.08),
            ("response_4xx", self.generate_response_4xx, 0.05),
            ("response_5xx", self.generate_response_5xx, 0.03),
            ("pqc_invite", self.generate_pqc_invite, 0.04),
        ]

        dataset_metadata = {
            "protocol": "sip",
            "version": "RFC 3261",
            "total_samples": num_samples,
            "samples_by_type": {},
            "generated_at": datetime.now().isoformat(),
        }

        sample_idx = 0
        for msg_type, generator, ratio in generators:
            count = int(num_samples * ratio)
            dataset_metadata["samples_by_type"][msg_type] = count

            for i in range(count):
                message, metadata = generator()
                metadata["sample_index"] = sample_idx
                metadata["message_type"] = msg_type

                bin_path = output_path / f"sip_{msg_type}_{sample_idx:06d}.bin"
                with open(bin_path, "wb") as f:
                    f.write(message)

                meta_path = output_path / f"sip_{msg_type}_{sample_idx:06d}.json"
                with open(meta_path, "w") as f:
                    json.dump(metadata, f, indent=2, default=str)

                sample_idx += 1

        with open(output_path / "dataset_metadata.json", "w") as f:
            json.dump(dataset_metadata, f, indent=2)

        return dataset_metadata


def main():
    """Generate SIP dataset."""
    generator = SIPGenerator(seed=42)
    output_dir = Path(__file__).parent.parent / "protocols" / "sip"

    print("Generating SIP dataset...")
    metadata = generator.generate_dataset(num_samples=1000, output_dir=str(output_dir))

    print(f"Generated {metadata['total_samples']} samples")
    print(f"Output directory: {output_dir}")
    for msg_type, count in metadata["samples_by_type"].items():
        print(f"  - {msg_type}: {count} samples")


if __name__ == "__main__":
    main()
