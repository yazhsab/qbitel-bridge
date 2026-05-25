"""
TN3270e Terminal Protocol Message Generator

Generates realistic TN3270e terminal protocol messages (RFC 2355)
for training the protocol discovery and field detection models.

TN3270e Wire Format:
- 5-byte TN3270e header:
  - Data Type (1 byte): DATA_3270=0x00, SCS_DATA=0x01, BIND_IMAGE=0x03,
                         UNBIND=0x04, NVT_DATA=0x05
  - Request Flag (1 byte)
  - Response Flag (1 byte)
  - Sequence Number (2 bytes, big-endian)
- 3270 data stream payload
- EOR marker: 0xFF 0xEF

Message types generated:
1. data_3270: Screen writes with SBA/SF orders and EBCDIC content
2. scs_data: SNA Character Stream print data
3. bind_image: Session BIND image
4. unbind: Session UNBIND notification
5. negotiation: TN3270e option negotiation (DO/WILL/WONT/DONT)

Includes realistic mainframe screen templates:
- Customer lookup screens
- Transaction history displays
- Agent dashboard screens
- Account summary screens
"""

import json
import random
import struct
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class TN3270eGenerator:
    """Generate realistic TN3270e terminal protocol messages for ML training."""

    # ASCII to EBCDIC translation table (partial - printable chars)
    ASCII_TO_EBCDIC = {
        0x20: 0x40,  # space
        0x21: 0x5A,  # !
        0x22: 0x7F,  # "
        0x23: 0x7B,  # #
        0x24: 0x5B,  # $
        0x25: 0x6C,  # %
        0x26: 0x50,  # &
        0x27: 0x7D,  # '
        0x28: 0x4D,  # (
        0x29: 0x5D,  # )
        0x2A: 0x5C,  # *
        0x2B: 0x4E,  # +
        0x2C: 0x6B,  # ,
        0x2D: 0x60,  # -
        0x2E: 0x4B,  # .
        0x2F: 0x61,  # /
        0x30: 0xF0,  # 0
        0x31: 0xF1,  # 1
        0x32: 0xF2,  # 2
        0x33: 0xF3,  # 3
        0x34: 0xF4,  # 4
        0x35: 0xF5,  # 5
        0x36: 0xF6,  # 6
        0x37: 0xF7,  # 7
        0x38: 0xF8,  # 8
        0x39: 0xF9,  # 9
        0x3A: 0x7A,  # :
        0x3B: 0x5E,  # ;
        0x3C: 0x4C,  # <
        0x3D: 0x7E,  # =
        0x3E: 0x6E,  # >
        0x3F: 0x6F,  # ?
        0x40: 0x7C,  # @
        0x41: 0xC1,  # A
        0x42: 0xC2,  # B
        0x43: 0xC3,  # C
        0x44: 0xC4,  # D
        0x45: 0xC5,  # E
        0x46: 0xC6,  # F
        0x47: 0xC7,  # G
        0x48: 0xC8,  # H
        0x49: 0xC9,  # I
        0x4A: 0xD1,  # J
        0x4B: 0xD2,  # K
        0x4C: 0xD3,  # L
        0x4D: 0xD4,  # M
        0x4E: 0xD5,  # N
        0x4F: 0xD6,  # O
        0x50: 0xD7,  # P
        0x51: 0xD8,  # Q
        0x52: 0xD9,  # R
        0x53: 0xE2,  # S
        0x54: 0xE3,  # T
        0x55: 0xE4,  # U
        0x56: 0xE5,  # V
        0x57: 0xE6,  # W
        0x58: 0xE7,  # X
        0x59: 0xE8,  # Y
        0x5A: 0xE9,  # Z
        0x61: 0x81,  # a
        0x62: 0x82,  # b
        0x63: 0x83,  # c
        0x64: 0x84,  # d
        0x65: 0x85,  # e
        0x66: 0x86,  # f
        0x67: 0x87,  # g
        0x68: 0x88,  # h
        0x69: 0x89,  # i
        0x6A: 0x91,  # j
        0x6B: 0x92,  # k
        0x6C: 0x93,  # l
        0x6D: 0x94,  # m
        0x6E: 0x95,  # n
        0x6F: 0x96,  # o
        0x70: 0x97,  # p
        0x71: 0x98,  # q
        0x72: 0x99,  # r
        0x73: 0xA2,  # s
        0x74: 0xA3,  # t
        0x75: 0xA4,  # u
        0x76: 0xA5,  # v
        0x77: 0xA6,  # w
        0x78: 0xA7,  # x
        0x79: 0xA8,  # y
        0x7A: 0xA9,  # z
        0x5F: 0x6D,  # _
    }

    # 3270 Write commands
    WRITE_CMD = 0xF1         # Write
    ERASE_WRITE_CMD = 0xF5   # Erase/Write
    ERASE_WRITE_ALT = 0x7E   # Erase/Write Alternate
    WRITE_STRUCTURED = 0xF3  # Write Structured Field

    # 3270 Orders
    SBA = 0x11   # Set Buffer Address
    SF = 0x1D    # Start Field
    SA = 0x28    # Set Attribute
    IC = 0x13    # Insert Cursor

    # Field attribute byte values
    ATTR_PROTECTED = 0xF0        # Protected, display
    ATTR_UNPROTECTED = 0xC0      # Unprotected, display
    ATTR_INTENSE = 0xF8          # Protected, intensified
    ATTR_HIDDEN = 0x4C           # Non-display (hidden)
    ATTR_NUMERIC = 0xD0          # Protected, numeric

    # TN3270e data types
    DATA_3270 = 0x00
    SCS_DATA = 0x01
    BIND_IMAGE = 0x03
    UNBIND = 0x04

    # Telnet negotiation bytes
    IAC = 0xFF
    DO = 0xFD
    DONT = 0xFE
    WILL = 0xFB
    WONT = 0xFC
    SB = 0xFA
    SE = 0xF0

    # TN3270e negotiation option
    TN3270E_OPTION = 0x28  # 40 decimal

    # EOR marker
    EOR = bytes([0xFF, 0xEF])

    # Screen dimensions
    ROWS = 24
    COLS = 80

    # Screen templates for BPO environments
    SCREEN_TEMPLATES = {
        "customer_lookup": [
            "  CUSTOMER LOOKUP                                        CICS/ESA  ",
            "  ---------------------------------------------------------------  ",
            "  ACCOUNT:  {acct_num}     NAME: {cust_name:30s}                   ",
            "  PHONE:    {phone}        STATUS: {status}                        ",
            "  ADDRESS:  {address:40s}                                          ",
            "  CITY:     {city:20s}  STATE: {state}  ZIP: {zip}                 ",
            "  ---------------------------------------------------------------  ",
            "  BALANCE:  ${balance:>12s}   LAST PMT: ${last_pmt:>10s}           ",
            "  DUE DATE: {due_date}      OVERDUE: {overdue}                     ",
            "  ---------------------------------------------------------------  ",
            "  NOTES: {notes:50s}                                               ",
            "  PF1=HELP  PF3=EXIT  PF5=TRANSACTIONS  PF7=PREV  PF8=NEXT        ",
        ],
        "transaction_history": [
            "  TRANSACTION HISTORY - ACCOUNT: {acct_num}              PAGE {page}",
            "  ---------------------------------------------------------------  ",
            "  DATE       TYPE       AMOUNT      REF#      DESCRIPTION          ",
            "  ---------------------------------------------------------------  ",
            "  {txn1_date} {txn1_type:10s} ${txn1_amt:>10s}  {txn1_ref}  {txn1_desc:20s}",
            "  {txn2_date} {txn2_type:10s} ${txn2_amt:>10s}  {txn2_ref}  {txn2_desc:20s}",
            "  {txn3_date} {txn3_type:10s} ${txn3_amt:>10s}  {txn3_ref}  {txn3_desc:20s}",
            "  {txn4_date} {txn4_type:10s} ${txn4_amt:>10s}  {txn4_ref}  {txn4_desc:20s}",
            "  {txn5_date} {txn5_type:10s} ${txn5_amt:>10s}  {txn5_ref}  {txn5_desc:20s}",
            "  ---------------------------------------------------------------  ",
            "  TOTAL: ${total:>12s}   COUNT: {count}                            ",
            "  PF1=HELP  PF3=EXIT  PF7=PREV  PF8=NEXT  PF12=RETURN             ",
        ],
        "agent_dashboard": [
            "  AGENT DASHBOARD                                  {date}  {time} ",
            "  ---------------------------------------------------------------  ",
            "  AGENT: {agent_id:10s}   QUEUE: {queue:15s}   STATUS: {status}    ",
            "  CALLS TODAY: {calls_today:>4s}   AHT: {aht}   CSAT: {csat}%     ",
            "  ---------------------------------------------------------------  ",
            "  CURRENT CALL:                                                    ",
            "    CALLER:   {caller_phone}    ANI: {ani}                         ",
            "    ACCOUNT:  {acct_num}        WAIT TIME: {wait_time}             ",
            "    QUEUE:    {call_queue:15s}   PRIORITY: {priority}              ",
            "  ---------------------------------------------------------------  ",
            "  ALERTS: {alert:50s}                                              ",
            "  PF1=HELP  PF3=LOGOUT  PF5=HOLD  PF6=TRANSFER  PF9=WRAP          ",
        ],
        "account_summary": [
            "  ACCOUNT SUMMARY                                CICS/PROD         ",
            "  ---------------------------------------------------------------  ",
            "  ACCT#: {acct_num}     TYPE: {acct_type:15s}                      ",
            "  OWNER: {owner:30s}    SSN: ***-**-{ssn_last4}                    ",
            "  OPENED: {open_date}   CLOSED: {close_date}                       ",
            "  ---------------------------------------------------------------  ",
            "  CREDIT LIMIT: ${credit_limit:>12s}                               ",
            "  CURRENT BAL:  ${balance:>12s}                                    ",
            "  AVAILABLE:    ${available:>12s}                                   ",
            "  MIN PAYMENT:  ${min_pmt:>12s}   DUE: {due_date}                  ",
            "  ---------------------------------------------------------------  ",
            "  PF1=HELP  PF3=EXIT  PF5=TXNS  PF6=PAYMENTS  PF12=MENU           ",
        ],
    }

    def __init__(self, seed: Optional[int] = None):
        if seed:
            random.seed(seed)
        self.seq_counter = random.randint(0, 65535)

    def _next_seq(self) -> int:
        """Get next TN3270e sequence number."""
        self.seq_counter = (self.seq_counter + 1) & 0xFFFF
        return self.seq_counter

    def _ascii_to_ebcdic(self, text: str) -> bytes:
        """Convert ASCII text to EBCDIC bytes."""
        result = []
        for ch in text:
            code = ord(ch)
            result.append(self.ASCII_TO_EBCDIC.get(code, 0x40))  # default space
        return bytes(result)

    def _buffer_address(self, row: int, col: int) -> bytes:
        """Encode a 3270 buffer address (12-bit format)."""
        addr = row * self.COLS + col
        # 12-bit addressing for 24x80 screens
        high = (addr >> 6) & 0x3F
        low = addr & 0x3F
        # Encode using 3270 address encoding
        encode_table = (
            0x40, 0xC1, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7,
            0xC8, 0xC9, 0x4A, 0x4B, 0x4C, 0x4D, 0x4E, 0x4F,
            0x50, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7,
            0xD8, 0xD9, 0x5A, 0x5B, 0x5C, 0x5D, 0x5E, 0x5F,
            0x60, 0x61, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7,
            0xE8, 0xE9, 0x6A, 0x6B, 0x6C, 0x6D, 0x6E, 0x6F,
            0xF0, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7,
            0xF8, 0xF9, 0x7A, 0x7B, 0x7C, 0x7D, 0x7E, 0x7F,
        )
        return bytes([encode_table[high], encode_table[low]])

    def _build_tn3270e_header(
        self, data_type: int, req_flag: int = 0, resp_flag: int = 0
    ) -> Tuple[bytes, Dict]:
        """Build the 5-byte TN3270e header."""
        seq = self._next_seq()
        header = struct.pack(">BBBH", data_type, req_flag, resp_flag, seq)

        meta = {
            "data_type": data_type,
            "data_type_name": {
                0x00: "DATA_3270", 0x01: "SCS_DATA",
                0x03: "BIND_IMAGE", 0x04: "UNBIND",
                0x05: "NVT_DATA",
            }.get(data_type, "UNKNOWN"),
            "request_flag": req_flag,
            "response_flag": resp_flag,
            "sequence_number": seq,
        }

        fields = [
            {"name": "tn3270e_header", "offset": 0, "length": 5},
            {"name": "data_type", "offset": 0, "length": 1},
            {"name": "request_flag", "offset": 1, "length": 1},
            {"name": "response_flag", "offset": 2, "length": 1},
            {"name": "sequence_number", "offset": 3, "length": 2},
        ]

        return header, meta, fields

    def _generate_screen_data(self, template_name: str) -> Tuple[bytes, Dict]:
        """Generate 3270 data stream from a screen template."""
        template = self.SCREEN_TEMPLATES[template_name]

        # Generate template values
        values = self._random_screen_values(template_name)

        # Build 3270 data stream
        stream = bytearray()
        # Write command + WCC (Write Control Character)
        write_cmd = random.choice([self.ERASE_WRITE_CMD, self.WRITE_CMD])
        wcc = 0xC3  # Reset + Unlock keyboard
        stream.append(write_cmd)
        stream.append(wcc)

        screen_text = []
        for row_idx, line_template in enumerate(template):
            try:
                line = line_template.format(**values)
            except (KeyError, ValueError):
                line = line_template

            # Pad/truncate to 80 columns
            line = line[:self.COLS].ljust(self.COLS)
            screen_text.append(line)

            # SBA order + field attribute + EBCDIC text
            stream.append(self.SBA)
            stream.extend(self._buffer_address(row_idx, 0))
            stream.append(self.SF)
            stream.append(self.ATTR_PROTECTED)
            stream.extend(self._ascii_to_ebcdic(line))

        # Add input fields for unprotected areas
        # Place cursor at a reasonable input position
        input_row = random.randint(2, 5)
        stream.append(self.SBA)
        stream.extend(self._buffer_address(input_row, 12))
        stream.append(self.SF)
        stream.append(self.ATTR_UNPROTECTED)
        stream.append(self.IC)

        screen_meta = {
            "screen_template": template_name,
            "screen_rows": len(template),
            "write_command": "ERASE_WRITE" if write_cmd == self.ERASE_WRITE_CMD else "WRITE",
            "has_input_fields": True,
            "screen_values": {k: str(v) for k, v in values.items()
                              if not any(s in k for s in ["ssn", "password"])},
        }

        return bytes(stream), screen_meta

    def _random_screen_values(self, template_name: str) -> Dict:
        """Generate random values for screen templates."""
        first_names = ["JOHN", "JANE", "ROBERT", "MARIA", "JAMES", "SARAH",
                       "DAVID", "LISA", "MICHAEL", "JENNIFER"]
        last_names = ["SMITH", "JOHNSON", "WILLIAMS", "BROWN", "JONES",
                      "GARCIA", "MILLER", "DAVIS", "MARTINEZ", "ANDERSON"]
        cities = ["NEW YORK", "CHICAGO", "LOS ANGELES", "HOUSTON", "PHOENIX",
                  "PHILADELPHIA", "SAN ANTONIO", "SAN DIEGO", "DALLAS", "AUSTIN"]
        states = ["NY", "IL", "CA", "TX", "AZ", "PA", "OH", "FL", "GA", "WA"]
        queues = ["SALES", "SUPPORT", "BILLING", "RETENTION", "COLLECTIONS"]
        txn_types = ["PAYMENT", "PURCHASE", "REFUND", "FEE", "INTEREST", "CREDIT"]
        statuses = ["ACTIVE", "INACTIVE", "SUSPENDED", "CLOSED"]

        base = {
            "acct_num": f"{random.randint(1000000000, 9999999999)}",
            "cust_name": f"{random.choice(first_names)} {random.choice(last_names)}",
            "phone": f"{random.randint(200, 999)}-{random.randint(200, 999)}-{random.randint(1000, 9999)}",
            "status": random.choice(statuses),
            "address": f"{random.randint(100, 9999)} {random.choice(['MAIN', 'OAK', 'MAPLE', 'ELM'])} ST",
            "city": random.choice(cities),
            "state": random.choice(states),
            "zip": f"{random.randint(10000, 99999)}",
            "balance": f"{random.randint(0, 50000)}.{random.randint(0, 99):02d}",
            "last_pmt": f"{random.randint(10, 5000)}.{random.randint(0, 99):02d}",
            "due_date": f"{random.randint(1, 12):02d}/{random.randint(1, 28):02d}/2024",
            "overdue": random.choice(["NO", "YES"]),
            "notes": random.choice([
                "CUSTOMER CALLED RE: BILLING ISSUE",
                "PAYMENT ARRANGEMENT REQUESTED",
                "ADDRESS CHANGE PROCESSED",
                "ACCOUNT REVIEW COMPLETED",
                "DISPUTE FILED - PENDING REVIEW",
            ]),
        }

        if template_name == "transaction_history":
            for i in range(1, 6):
                base[f"txn{i}_date"] = f"{random.randint(1, 12):02d}/{random.randint(1, 28):02d}"
                base[f"txn{i}_type"] = random.choice(txn_types)
                base[f"txn{i}_amt"] = f"{random.randint(1, 5000)}.{random.randint(0, 99):02d}"
                base[f"txn{i}_ref"] = f"{random.randint(100000, 999999)}"
                base[f"txn{i}_desc"] = random.choice([
                    "ONLINE PMT", "STORE PURCHASE",
                    "AUTO PAY", "LATE FEE", "INTEREST",
                ])
            base["page"] = f"{random.randint(1, 5)}"
            base["total"] = f"{random.randint(100, 25000)}.{random.randint(0, 99):02d}"
            base["count"] = f"{random.randint(5, 100)}"

        elif template_name == "agent_dashboard":
            base["agent_id"] = f"AGT{random.randint(1000, 9999)}"
            base["queue"] = random.choice(queues)
            base["calls_today"] = f"{random.randint(5, 80)}"
            base["aht"] = f"{random.randint(2, 15)}:{random.randint(0, 59):02d}"
            base["csat"] = f"{random.randint(70, 100)}"
            base["date"] = f"{random.randint(1, 12):02d}/{random.randint(1, 28):02d}/2024"
            base["time"] = f"{random.randint(8, 20):02d}:{random.randint(0, 59):02d}"
            base["caller_phone"] = f"+1{random.randint(200, 999)}{random.randint(2000000, 9999999)}"
            base["ani"] = f"+1{random.randint(200, 999)}{random.randint(2000000, 9999999)}"
            base["wait_time"] = f"{random.randint(0, 30)}:{random.randint(0, 59):02d}"
            base["call_queue"] = random.choice(queues)
            base["priority"] = f"{random.randint(1, 10)}"
            base["alert"] = random.choice([
                "NONE", "HIGH VALUE CUSTOMER",
                "ESCALATION REQUIRED", "PCI SCOPE ACTIVE",
                "FRAUD ALERT - VERIFY IDENTITY",
            ])

        elif template_name == "account_summary":
            base["acct_type"] = random.choice(["CHECKING", "SAVINGS", "CREDIT CARD", "MORTGAGE"])
            base["owner"] = base["cust_name"]
            base["ssn_last4"] = f"{random.randint(1000, 9999)}"
            base["open_date"] = f"{random.randint(1, 12):02d}/{random.randint(1, 28):02d}/{random.randint(2010, 2023)}"
            base["close_date"] = random.choice(["OPEN", "OPEN", "OPEN", "12/31/2023"])
            base["credit_limit"] = f"{random.randint(1000, 100000)}.00"
            base["available"] = f"{random.randint(0, 50000)}.{random.randint(0, 99):02d}"
            base["min_pmt"] = f"{random.randint(25, 500)}.00"

        return base

    # ------------------------------------------------------------------
    # Message type generators
    # ------------------------------------------------------------------

    def generate_data_3270(self) -> Tuple[bytes, Dict]:
        """Generate a TN3270e DATA_3270 message with screen content."""
        template_name = random.choice(list(self.SCREEN_TEMPLATES.keys()))
        header, meta, fields = self._build_tn3270e_header(self.DATA_3270)
        screen_data, screen_meta = self._generate_screen_data(template_name)

        fields.append({"name": "data_3270_payload", "offset": 5, "length": len(screen_data)})
        fields.append({"name": "eor_marker", "offset": 5 + len(screen_data), "length": 2})

        message = header + screen_data + self.EOR
        meta.update(screen_meta)

        return message, self._finalize_metadata("data_3270", message, meta, fields)

    def generate_scs_data(self) -> Tuple[bytes, Dict]:
        """Generate a TN3270e SCS_DATA message (print stream)."""
        header, meta, fields = self._build_tn3270e_header(self.SCS_DATA)

        # SCS commands
        scs_data = bytearray()
        # Transparent (TRN) prefix: 0x35 + length byte
        lines = [
            f"PRINT JOB: PJ{random.randint(10000, 99999)}",
            f"DATE: {random.randint(1, 12):02d}/{random.randint(1, 28):02d}/2024",
            f"AGENT: AGT{random.randint(1000, 9999)}",
            "=" * 40,
            f"ACCOUNT: {random.randint(1000000000, 9999999999)}",
            f"CUSTOMER: {random.choice(['SMITH', 'JOHNSON', 'WILLIAMS', 'BROWN'])}",
            f"TRANSACTION COUNT: {random.randint(1, 50)}",
            f"TOTAL: ${random.randint(100, 50000)}.{random.randint(0, 99):02d}",
            "=" * 40,
            "** END OF REPORT **",
        ]

        for line in lines:
            ebcdic_line = self._ascii_to_ebcdic(line)
            scs_data.extend(ebcdic_line)
            scs_data.append(0x15)  # NL (New Line)

        fields.append({"name": "scs_payload", "offset": 5, "length": len(scs_data)})
        fields.append({"name": "eor_marker", "offset": 5 + len(scs_data), "length": 2})

        message = header + bytes(scs_data) + self.EOR
        meta.update({
            "scs_line_count": len(lines),
            "scs_content_type": "print_report",
        })

        return message, self._finalize_metadata("scs_data", message, meta, fields)

    def generate_bind_image(self) -> Tuple[bytes, Dict]:
        """Generate a TN3270e BIND_IMAGE message."""
        header, meta, fields = self._build_tn3270e_header(self.BIND_IMAGE)

        # BIND image payload (simplified)
        # Format: session type + screen dimensions + device capabilities
        session_type = random.choice([0x01, 0x02, 0x03])  # LU types
        primary_rows = self.ROWS
        primary_cols = self.COLS
        alt_rows = random.choice([24, 32, 43])
        alt_cols = random.choice([80, 132])

        bind_data = struct.pack(
            ">BBBBB",
            session_type,
            primary_rows, primary_cols,
            alt_rows, alt_cols,
        )
        # Add device name in EBCDIC
        device_names = [
            f"TN{random.randint(1000, 9999)}",
            f"LU{random.randint(100, 999)}",
            f"PC{random.randint(1000, 9999)}",
        ]
        device_name = random.choice(device_names)
        bind_data += self._ascii_to_ebcdic(device_name.ljust(8))

        fields.append({"name": "bind_image_payload", "offset": 5, "length": len(bind_data)})
        fields.append({"name": "eor_marker", "offset": 5 + len(bind_data), "length": 2})

        message = header + bind_data + self.EOR
        meta.update({
            "session_type": session_type,
            "primary_screen": f"{primary_rows}x{primary_cols}",
            "alternate_screen": f"{alt_rows}x{alt_cols}",
            "device_name": device_name,
        })

        return message, self._finalize_metadata("bind_image", message, meta, fields)

    def generate_unbind(self) -> Tuple[bytes, Dict]:
        """Generate a TN3270e UNBIND message."""
        header, meta, fields = self._build_tn3270e_header(self.UNBIND)

        # UNBIND reason codes
        reasons = {
            0x01: "NORMAL",
            0x02: "BIND_FORTHCOMING",
            0x07: "VIRTUAL_ROUTE_FAILURE",
            0x08: "SESSION_FAILURE",
            0x0A: "CLEANUP",
            0x0F: "UNRECOVERABLE_ERROR",
        }
        reason_code = random.choice(list(reasons.keys()))
        unbind_data = bytes([reason_code])

        fields.append({"name": "unbind_reason", "offset": 5, "length": 1})
        fields.append({"name": "eor_marker", "offset": 6, "length": 2})

        message = header + unbind_data + self.EOR
        meta.update({
            "unbind_reason_code": reason_code,
            "unbind_reason_name": reasons[reason_code],
        })

        return message, self._finalize_metadata("unbind", message, meta, fields)

    def generate_negotiation(self) -> Tuple[bytes, Dict]:
        """Generate a TN3270e option negotiation message."""
        # TN3270e negotiation uses Telnet option mechanism
        action = random.choice([self.DO, self.WILL, self.WONT, self.DONT])
        action_names = {
            self.DO: "DO", self.WILL: "WILL",
            self.WONT: "WONT", self.DONT: "DONT",
        }

        options = [
            (self.TN3270E_OPTION, "TN3270E"),
            (0x00, "BINARY"),
            (0x01, "ECHO"),
            (0x03, "SUPPRESS_GO_AHEAD"),
            (0x18, "TERMINAL_TYPE"),
            (0x19, "EOR"),
        ]
        option_code, option_name = random.choice(options)

        # Build negotiation message
        neg_data = bytes([self.IAC, action, option_code])

        # Optionally add subnegotiation for terminal type
        sub_neg = b""
        if option_code == 0x18 and action == self.WILL:
            # Terminal type subnegotiation
            term_types = ["IBM-3278-2-E", "IBM-3279-2-E", "IBM-3278-4-E",
                          "IBM-Dynamic", "IBM-3278-5-E"]
            term_type = random.choice(term_types)
            term_ebcdic = term_type.encode("ascii")
            sub_neg = bytes([self.IAC, self.SB, 0x18, 0x00]) + term_ebcdic + bytes([self.IAC, self.SE])

        message = neg_data + sub_neg

        meta = {
            "negotiation_action": action_names[action],
            "option_code": option_code,
            "option_name": option_name,
        }
        if sub_neg:
            meta["terminal_type"] = term_type

        fields = [
            {"name": "iac_byte", "offset": 0, "length": 1},
            {"name": "action_byte", "offset": 1, "length": 1},
            {"name": "option_byte", "offset": 2, "length": 1},
        ]
        if sub_neg:
            fields.append({"name": "subnegotiation", "offset": 3, "length": len(sub_neg)})

        return message, self._finalize_metadata("negotiation", message, meta, fields)

    # ------------------------------------------------------------------
    # Metadata and dataset generation
    # ------------------------------------------------------------------

    def _finalize_metadata(
        self, msg_type: str, message: bytes, meta: Dict, fields: list
    ) -> Dict:
        """Create the final metadata dict."""
        meta.update({
            "protocol": "tn3270e",
            "message_type": msg_type,
            "timestamp": datetime.now().isoformat(),
            "message_length": len(message),
            "fields": fields,
            "hash": hashlib.sha256(message).hexdigest(),
        })
        return meta

    def generate_dataset(self, num_samples: int, output_dir: str) -> Dict:
        """Generate a complete dataset of TN3270e messages."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        generators = [
            ("data_3270", self.generate_data_3270, 0.40),
            ("scs_data", self.generate_scs_data, 0.15),
            ("bind_image", self.generate_bind_image, 0.15),
            ("unbind", self.generate_unbind, 0.15),
            ("negotiation", self.generate_negotiation, 0.15),
        ]

        dataset_metadata = {
            "protocol": "tn3270e",
            "version": "RFC 2355",
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

                bin_path = output_path / f"tn3270e_{msg_type}_{sample_idx:06d}.bin"
                with open(bin_path, "wb") as f:
                    f.write(message)

                meta_path = output_path / f"tn3270e_{msg_type}_{sample_idx:06d}.json"
                with open(meta_path, "w") as f:
                    json.dump(metadata, f, indent=2, default=str)

                sample_idx += 1

        with open(output_path / "dataset_metadata.json", "w") as f:
            json.dump(dataset_metadata, f, indent=2)

        return dataset_metadata


def main():
    """Generate TN3270e dataset."""
    generator = TN3270eGenerator(seed=42)
    output_dir = Path(__file__).parent.parent / "protocols" / "tn3270e"

    print("Generating TN3270e dataset...")
    metadata = generator.generate_dataset(num_samples=1000, output_dir=str(output_dir))

    print(f"Generated {metadata['total_samples']} samples")
    print(f"Output directory: {output_dir}")
    for msg_type, count in metadata["samples_by_type"].items():
        print(f"  - {msg_type}: {count} samples")


if __name__ == "__main__":
    main()
