"""
ARINC Aviation Protocol & Security Dataset Generator

Generates realistic aviation protocol samples and PQC security instruction
pairs for training the QBITEL protocol discovery, field detection, and
LLM fine-tuning models.

Part A -- Protocol Samples (binary + JSON metadata):
  1. ACARS  (Aircraft Communications Addressing and Reporting System)
  2. CPDLC  (Controller-Pilot Data Link Communications)
  3. ADS-C  (Automatic Dependent Surveillance - Contract)
  4. ARINC 429  (avionics data bus 32-bit words)
  5. ARINC 653  (avionics software partition configs)

Part B -- Aviation Security Instruction Pairs (JSONL):
  1. atc_pqc            - PQC for ATC communications
  2. forward_secrecy    - Forward-secure aircraft datalink channels
  3. avionics_security  - DO-326A / DO-356A compliance
  4. satellite_link     - PQC for SATCOM bandwidth-constrained crypto
  5. flight_data_integrity - EFB and nav-database authentication
"""

import json
import math
import random
import struct
import hashlib
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ARINCAviationGenerator:
    """Generate realistic aviation protocol samples and PQC security
    instruction pairs for ML training."""

    # ------------------------------------------------------------------ #
    # Constants                                                            #
    # ------------------------------------------------------------------ #

    # ICAO aircraft registrations
    REGISTRATIONS = [
        "N12345", "N67890", "N54321", "N98765", "N11223",
        "G-ABCD", "G-EFGH", "G-XYZW", "G-BKFW", "G-CIVY",
        "VH-XYZ", "VH-OJA", "VH-QPA", "VH-NME", "VH-BZG",
        "JA8089", "JA731A", "JA816A", "JA612J", "JA822J",
        "D-AIMD", "D-ABYA", "D-AIZQ", "F-GSQA", "F-HPJD",
    ]

    # Flight numbers (airline ICAO designator + numeric)
    FLIGHT_NUMBERS = [
        "UAL123", "UAL456", "UAL789", "UAL1522", "UAL987",
        "BAW456", "BAW117", "BAW283", "BAW15", "BAW9",
        "QFA789", "QFA1", "QFA94", "QFA32", "QFA63",
        "DLH400", "DLH450", "DLH738", "DLH490", "DLH455",
        "AAL100", "AAL191", "AAL77", "DAL60", "DAL200",
        "ANA12", "ANA1", "JAL3", "JAL17", "AFR66",
    ]

    # ICAO airport codes
    AIRPORTS = [
        "KJFK", "KLAX", "KORD", "KATL", "KSFO", "KMIA", "KDEN", "KDFW",
        "EGLL", "EGKK", "EGLC", "LFPG", "EDDF", "EHAM", "LEMD", "LIRF",
        "YSSY", "YMML", "YBBN", "RJTT", "RJAA", "RKSI", "VHHH", "WSSS",
        "OMDB", "OTHH", "VIDP", "ZBAA", "ZSPD", "VTBS",
    ]

    # Major air route waypoints with (lat, lon) for realistic GPS coords
    ROUTE_WAYPOINTS = [
        (40.6413, -73.7781),   # KJFK
        (33.9425, -118.4081),  # KLAX
        (51.4700, -0.4543),    # EGLL
        (-33.9461, 151.1772),  # YSSY
        (35.5494, 139.7798),   # RJTT
        (25.2532, 55.3657),    # OMDB
        (49.0097, 2.5479),     # LFPG
        (1.3502, 103.9944),    # WSSS
        (52.3086, 4.7639),     # EHAM
        (41.9742, -87.9073),   # KORD
        (50.0333, 8.5706),     # EDDF
        (22.3080, 113.9185),   # VHHH
        (28.5665, -81.3388),   # KMCO (Orlando)
        (47.4502, -122.3088),  # KSEA (Seattle)
        (55.9736, -3.1907),    # EGPH (Edinburgh)
    ]

    # ACARS label codes
    ACARS_LABELS = {
        "H1": "general_message",
        "SA": "departure_report",
        "_d": "ads_c_report",
        "Q0": "oooi_event",
        "RA": "arrival_report",
        "5Z": "airline_ops",
        "B6": "takeoff_report",
        "QA": "eta_report",
        "15": "position_report",
        "20": "weather_request",
    }

    # CPDLC uplink messages (controller -> pilot)
    CPDLC_UPLINKS = {
        "UM74": "PROCEED DIRECT TO {position}",
        "UM19": "MAINTAIN {level}",
        "UM20": "CLIMB TO AND MAINTAIN {level}",
        "UM23": "DESCEND TO AND MAINTAIN {level}",
        "UM46": "CROSS {position} AT {level}",
        "UM79": "CLEARED TO {position} VIA {route}",
        "UM106": "MAINTAIN MACH {mach}",
        "UM148": "WHEN CAN YOU ACCEPT {level}",
        "UM169": "TURN LEFT HEADING {heading}",
        "UM170": "TURN RIGHT HEADING {heading}",
        "UM171": "FLY HEADING {heading}",
        "UM215": "TURN LEFT DIRECT TO {position}",
        "UM228": "REPORT MAINTAINING {level}",
    }

    # CPDLC downlink messages (pilot -> controller)
    CPDLC_DOWNLINKS = {
        "DM0": "WILCO",
        "DM1": "UNABLE",
        "DM2": "STANDBY",
        "DM3": "ROGER",
        "DM6": "REQUEST {level}",
        "DM18": "REQUEST DIRECT TO {position}",
        "DM22": "REQUEST HEADING {heading}",
        "DM28": "LEAVING {level}",
        "DM36": "MAINTAINING {level}",
        "DM67": "REQUEST CLIMB TO {level}",
        "DM68": "REQUEST DESCENT TO {level}",
        "DM81": "AT {position}",
        "DM99": "CURRENT POSITION {lat} {lon}",
    }

    # ARINC 429 label definitions (octal label, parameter, units, BNR range)
    ARINC_429_LABELS = {
        0o310: {"name": "barometric_altitude", "units": "ft", "range": (-1000, 65536), "msb_value": 65536},
        0o314: {"name": "computed_airspeed", "units": "knots", "range": (0, 1024), "msb_value": 512},
        0o320: {"name": "magnetic_heading", "units": "deg", "range": (0, 360), "msb_value": 180},
        0o012: {"name": "latitude", "units": "deg", "range": (-90, 90), "msb_value": 90},
        0o013: {"name": "longitude", "units": "deg", "range": (-180, 180), "msb_value": 180},
        0o203: {"name": "mach_number", "units": "mach", "range": (0, 4.096), "msb_value": 2.048},
        0o361: {"name": "total_air_temp", "units": "degC", "range": (-128, 128), "msb_value": 128},
        0o324: {"name": "true_heading", "units": "deg", "range": (0, 360), "msb_value": 180},
        0o331: {"name": "radio_altitude", "units": "ft", "range": (0, 8192), "msb_value": 4096},
        0o350: {"name": "vertical_speed", "units": "ft_min", "range": (-16384, 16384), "msb_value": 16384},
    }

    # ARINC 653 partition names and budgets
    ARINC_653_PARTITIONS = [
        {"name": "FlightManagement", "budget_ms": 25, "period_ms": 50, "dals": "DAL-A"},
        {"name": "DisplaySystem", "budget_ms": 15, "period_ms": 40, "dals": "DAL-B"},
        {"name": "FuelManagement", "budget_ms": 10, "period_ms": 100, "dals": "DAL-B"},
        {"name": "EngineMgmt", "budget_ms": 20, "period_ms": 50, "dals": "DAL-A"},
        {"name": "NavDatabase", "budget_ms": 8, "period_ms": 200, "dals": "DAL-C"},
        {"name": "CommManager", "budget_ms": 12, "period_ms": 80, "dals": "DAL-B"},
        {"name": "HealthMonitor", "budget_ms": 5, "period_ms": 100, "dals": "DAL-A"},
        {"name": "TerrainAware", "budget_ms": 18, "period_ms": 50, "dals": "DAL-A"},
        {"name": "WeatherRadar", "budget_ms": 10, "period_ms": 100, "dals": "DAL-C"},
        {"name": "CabinPressure", "budget_ms": 8, "period_ms": 200, "dals": "DAL-B"},
    ]

    # Instruction pair categories
    INSTRUCTION_CATEGORIES = [
        "atc_pqc",
        "forward_secrecy",
        "avionics_security",
        "satellite_link",
        "flight_data_integrity",
    ]

    DIFFICULTIES = ["basic", "intermediate", "advanced"]

    # ------------------------------------------------------------------ #
    # Initializer                                                          #
    # ------------------------------------------------------------------ #

    def __init__(self, seed: Optional[int] = None):
        if seed:
            random.seed(seed)
        self._msg_ref = random.randint(1, 65535)

    # ------------------------------------------------------------------ #
    # Helper utilities                                                     #
    # ------------------------------------------------------------------ #

    def _next_msg_ref(self) -> int:
        self._msg_ref = (self._msg_ref + 1) % 65536
        return self._msg_ref

    def _random_waypoint(self) -> Tuple[float, float]:
        """Pick a random waypoint and add jitter to simulate en-route."""
        base = random.choice(self.ROUTE_WAYPOINTS)
        lat = base[0] + random.uniform(-5.0, 5.0)
        lon = base[1] + random.uniform(-5.0, 5.0)
        return round(lat, 4), round(lon, 4)

    def _random_level(self) -> str:
        """Return a random flight level string, e.g. FL350."""
        fl = random.choice([250, 270, 290, 310, 330, 350, 370, 390, 410])
        return f"FL{fl}"

    def _random_heading(self) -> int:
        return random.randint(0, 359)

    def _random_timestamp(self, days_back: int = 7) -> str:
        dt = datetime.utcnow() - timedelta(
            days=random.randint(0, days_back),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59),
        )
        return dt.strftime("%Y%m%d%H%M%S")

    def _compute_bcs(self, data: bytes) -> int:
        """Compute ACARS Block Check Sequence (XOR parity)."""
        bcs = 0
        for b in data:
            bcs ^= b
        return bcs & 0x7F

    def _reverse_bits_8(self, val: int) -> int:
        """Reverse the bit order of an 8-bit value (ARINC 429 label encoding)."""
        result = 0
        for _ in range(8):
            result = (result << 1) | (val & 1)
            val >>= 1
        return result

    def _build_fields(self, data: bytes, field_list: List[Dict]) -> List[Dict]:
        """Attach offset information to a pre-built field list."""
        enriched = []
        for f in field_list:
            enriched.append({
                "name": f["name"],
                "offset": f.get("offset", 0),
                "length": f.get("length", len(data)),
            })
        return enriched

    def _finalize_metadata(self, msg_type: str, message: bytes,
                           meta: Dict, fields: List[Dict]) -> Dict:
        meta.update({
            "protocol": "arinc_aviation",
            "message_type": msg_type,
            "timestamp": datetime.utcnow().isoformat(),
            "message_length": len(message),
            "fields": fields,
            "hash": hashlib.sha256(message).hexdigest(),
        })
        return meta

    # ------------------------------------------------------------------ #
    # Part A-1: ACARS message generation                                   #
    # ------------------------------------------------------------------ #

    def generate_acars(self) -> Tuple[bytes, Dict]:
        """Generate a realistic ACARS message (binary + metadata)."""
        reg = random.choice(self.REGISTRATIONS)
        flight = random.choice(self.FLIGHT_NUMBERS)
        label_code = random.choice(list(self.ACARS_LABELS.keys()))
        label_desc = self.ACARS_LABELS[label_code]
        mode = random.choice(["2", "H", "Q"])
        block_id = chr(random.randint(ord("A"), ord("Z")))
        ack = chr(random.randint(ord("A"), ord("Z")))
        msg_num = f"{random.choice('ABCDEFGHIJKLMNOP')}{random.randint(0, 99):02d}{random.choice('AB')}"

        # Build body based on label
        ts = self._random_timestamp(days_back=3)
        lat, lon = self._random_waypoint()
        alt = random.randint(20000, 42000)

        if label_code == "Q0":
            event = random.choice(["OUT", "OFF", "ON", "IN"])
            dep = random.choice(self.AIRPORTS)
            arr = random.choice(self.AIRPORTS)
            body = f"FI {flight}\r\n{event} {ts[:8]} {ts[8:12]}\r\nDEP {dep} ARR {arr}\r\nFUEL {random.randint(5000, 40000)} LBS"
        elif label_code == "SA":
            dep = random.choice(self.AIRPORTS)
            body = f"FI {flight}\r\nDEP {dep} {ts[:8]} {ts[8:12]}\r\nETD {ts[8:12]}\r\nGATE {random.choice('ABCDEFGH')}{random.randint(1, 50)}"
        elif label_code == "_d":
            body = f"ADS-C RPT {reg}\r\nPOS LAT {lat:+010.4f} LON {lon:+011.4f}\r\nALT {alt}\r\nSPD {random.randint(200, 550)} KTS\r\nTRK {random.randint(0, 359)}"
        elif label_code == "15":
            body = f"POS {reg}\r\nLAT {lat:+010.4f} LON {lon:+011.4f}\r\nALT {alt} HDG {random.randint(0, 359):03d}\r\nSAT {random.choice([-60, -55, -50, -45, -40, -35])} C"
        else:
            body = f"FI {flight}/{reg}\r\n{ts[:8]} {ts[8:12]}Z\r\nMSG: ROUTINE OPERATIONS NORMAL\r\nPOS {lat:+010.4f}/{lon:+011.4f} FL{alt // 100}"

        # Encode binary ACARS frame
        SOH = 0x01
        STX = 0x02
        ETX = 0x03

        header = bytes([SOH, ord(mode)])
        header += reg.ljust(7, ".").encode("ascii")
        header += bytes([ord(ack)])
        header += label_code.ljust(2).encode("ascii")
        header += bytes([ord(block_id)])
        header += bytes([STX])

        body_bytes = body.encode("ascii")
        suffix = bytes([ETX])
        bcs_payload = header[1:] + body_bytes + suffix
        bcs = self._compute_bcs(bcs_payload)
        message = header + body_bytes + suffix + bytes([bcs])

        fields = [
            {"name": "soh", "offset": 0, "length": 1},
            {"name": "mode", "offset": 1, "length": 1},
            {"name": "registration", "offset": 2, "length": 7},
            {"name": "ack", "offset": 9, "length": 1},
            {"name": "label", "offset": 10, "length": 2},
            {"name": "block_id", "offset": 12, "length": 1},
            {"name": "stx", "offset": 13, "length": 1},
            {"name": "body", "offset": 14, "length": len(body_bytes)},
            {"name": "etx", "offset": 14 + len(body_bytes), "length": 1},
            {"name": "bcs", "offset": 15 + len(body_bytes), "length": 1},
        ]

        meta = {
            "acars_mode": mode,
            "registration": reg,
            "flight_number": flight,
            "label_code": label_code,
            "label_description": label_desc,
            "block_id": block_id,
            "msg_number": msg_num,
            "body_text": body,
        }

        return message, self._finalize_metadata("acars", message, meta, fields)

    # ------------------------------------------------------------------ #
    # Part A-2: CPDLC message generation                                   #
    # ------------------------------------------------------------------ #

    def generate_cpdlc(self) -> Tuple[bytes, Dict]:
        """Generate a CPDLC ATN-format message (binary + metadata)."""
        is_uplink = random.random() < 0.55
        ref = self._next_msg_ref()
        urgency = random.choice(["NORMAL", "DISTRESS", "URGENT"])
        flight = random.choice(self.FLIGHT_NUMBERS)
        lat, lon = self._random_waypoint()
        position_name = random.choice([
            "WHALE", "BREST", "NATBD", "NOPAC", "SELCL",
            "TESRA", "GALSO", "PIKIL", "DOGAL", "DEVOL",
        ])
        level = self._random_level()
        heading = self._random_heading()
        route = f"{random.choice(self.AIRPORTS)}-{position_name}-{random.choice(self.AIRPORTS)}"

        if is_uplink:
            msg_id = random.choice(list(self.CPDLC_UPLINKS.keys()))
            template = self.CPDLC_UPLINKS[msg_id]
            text = template.format(
                position=position_name, level=level,
                heading=f"{heading:03d}", mach=f"{random.uniform(0.72, 0.86):.2f}",
                route=route,
            )
            direction = "uplink"
        else:
            msg_id = random.choice(list(self.CPDLC_DOWNLINKS.keys()))
            template = self.CPDLC_DOWNLINKS[msg_id]
            text = template.format(
                level=level, position=position_name,
                heading=f"{heading:03d}", lat=f"{lat:+.4f}", lon=f"{lon:+.4f}",
            )
            direction = "downlink"

        # Build ATN-style binary frame
        atn_version = 0x01
        msg_type_flag = 0x01 if is_uplink else 0x02
        encoded_text = text.encode("ascii")
        text_len = len(encoded_text)
        ref_bytes = struct.pack(">H", ref)
        urgency_code = {"NORMAL": 0x00, "URGENT": 0x01, "DISTRESS": 0x02}[urgency]

        header = struct.pack(">BBB", atn_version, msg_type_flag, urgency_code)
        header += ref_bytes
        header += struct.pack(">H", text_len)
        flight_bytes = flight.ljust(8).encode("ascii")
        message = header + flight_bytes + encoded_text

        fields = [
            {"name": "atn_version", "offset": 0, "length": 1},
            {"name": "msg_type_flag", "offset": 1, "length": 1},
            {"name": "urgency", "offset": 2, "length": 1},
            {"name": "msg_reference", "offset": 3, "length": 2},
            {"name": "text_length", "offset": 5, "length": 2},
            {"name": "flight_id", "offset": 7, "length": 8},
            {"name": "message_text", "offset": 15, "length": text_len},
        ]

        meta = {
            "direction": direction,
            "msg_element_id": msg_id,
            "msg_reference": ref,
            "urgency": urgency,
            "flight_number": flight,
            "message_text": text,
        }

        return message, self._finalize_metadata("cpdlc", message, meta, fields)

    # ------------------------------------------------------------------ #
    # Part A-3: ADS-C report generation                                    #
    # ------------------------------------------------------------------ #

    def generate_ads_c(self) -> Tuple[bytes, Dict]:
        """Generate an ADS-C report (binary + metadata)."""
        reg = random.choice(self.REGISTRATIONS)
        flight = random.choice(self.FLIGHT_NUMBERS)
        lat, lon = self._random_waypoint()
        alt = random.randint(20000, 42000)
        ts = self._random_timestamp(days_back=2)
        fom = random.randint(0, 7)  # Figure of Merit (GPS accuracy class)

        contract_type = random.choice(["periodic", "event", "demand"])
        report_type = random.choice(["basic", "extended"])

        airspeed = random.randint(200, 550)
        ground_speed = random.randint(180, 600)
        vertical_rate = random.randint(-3000, 3000)
        track = random.randint(0, 359)
        wind_speed = random.randint(5, 120)
        wind_dir = random.randint(0, 359)
        temperature = random.randint(-70, 20)

        # Binary encoding
        parts = []

        # Header: report type (1B) + contract type code (1B)
        rtype_code = {"basic": 0x01, "extended": 0x02}[report_type]
        ctype_code = {"periodic": 0x01, "event": 0x02, "demand": 0x03}[contract_type]
        parts.append(struct.pack(">BB", rtype_code, ctype_code))

        # Registration (8 bytes padded)
        parts.append(reg.ljust(8).encode("ascii"))

        # Flight number (8 bytes padded)
        parts.append(flight.ljust(8).encode("ascii"))

        # Basic report fields
        parts.append(struct.pack(">i", int(lat * 1e6)))    # lat  microdeg
        parts.append(struct.pack(">i", int(lon * 1e6)))    # lon  microdeg
        parts.append(struct.pack(">I", alt))                # altitude ft
        parts.append(ts.encode("ascii"))                    # timestamp 14B
        parts.append(struct.pack(">B", fom))                # FOM

        offset_extended = 2 + 8 + 8 + 4 + 4 + 4 + 14 + 1  # =45

        if report_type == "extended":
            parts.append(struct.pack(">HHhHBHh",
                                     airspeed, ground_speed, vertical_rate,
                                     track, wind_speed & 0xFF, wind_dir,
                                     temperature))

        message = b"".join(parts)

        fields = [
            {"name": "report_type", "offset": 0, "length": 1},
            {"name": "contract_type", "offset": 1, "length": 1},
            {"name": "registration", "offset": 2, "length": 8},
            {"name": "flight_number", "offset": 10, "length": 8},
            {"name": "latitude", "offset": 18, "length": 4},
            {"name": "longitude", "offset": 22, "length": 4},
            {"name": "altitude", "offset": 26, "length": 4},
            {"name": "timestamp", "offset": 30, "length": 14},
            {"name": "figure_of_merit", "offset": 44, "length": 1},
        ]
        if report_type == "extended":
            ext_fields = [
                {"name": "airspeed", "offset": offset_extended, "length": 2},
                {"name": "ground_speed", "offset": offset_extended + 2, "length": 2},
                {"name": "vertical_rate", "offset": offset_extended + 4, "length": 2},
                {"name": "track", "offset": offset_extended + 6, "length": 2},
                {"name": "wind_speed", "offset": offset_extended + 8, "length": 1},
                {"name": "wind_direction", "offset": offset_extended + 9, "length": 2},
                {"name": "temperature", "offset": offset_extended + 11, "length": 2},
            ]
            fields.extend(ext_fields)

        meta = {
            "registration": reg,
            "flight_number": flight,
            "latitude": lat,
            "longitude": lon,
            "altitude_ft": alt,
            "figure_of_merit": fom,
            "report_type": report_type,
            "contract_type": contract_type,
        }
        if report_type == "extended":
            meta.update({
                "airspeed_kts": airspeed,
                "ground_speed_kts": ground_speed,
                "vertical_rate_fpm": vertical_rate,
                "track_deg": track,
                "wind_speed_kts": wind_speed,
                "wind_direction_deg": wind_dir,
                "temperature_c": temperature,
            })

        return message, self._finalize_metadata("ads_c", message, meta, fields)

    # ------------------------------------------------------------------ #
    # Part A-4: ARINC 429 word generation                                  #
    # ------------------------------------------------------------------ #

    def generate_arinc429(self) -> Tuple[bytes, Dict]:
        """Generate a 32-bit ARINC 429 data word (binary + metadata)."""
        label_octal = random.choice(list(self.ARINC_429_LABELS.keys()))
        label_info = self.ARINC_429_LABELS[label_octal]

        # Label is transmitted LSB first (bit-reversed octal value)
        label_byte = self._reverse_bits_8(label_octal & 0xFF)

        # SDI (Source/Destination Identifier) - 2 bits
        sdi = random.randint(0, 3)

        # Generate realistic data value within range
        low, high = label_info["range"]
        value = random.uniform(low, high)

        # BNR encoding: value / msb_value maps to 19-bit signed integer
        msb_val = label_info["msb_value"]
        scale = (2**18 - 1) / msb_val
        data_int = int(value * scale) & 0x7FFFF  # 19 bits

        # SSM (Sign/Status Matrix) - 2 bits
        # 00=Failure Warning, 01=No Computed Data, 10=Functional Test, 11=Normal
        ssm = random.choices([0b00, 0b01, 0b10, 0b11], weights=[2, 3, 1, 94])[0]

        # Assemble 32-bit word:
        # Bits [0:7]   = Label (8 bits, bit-reversed)
        # Bits [8:9]   = SDI (2 bits)
        # Bits [10:28] = Data (19 bits BNR)
        # Bits [29:30] = SSM (2 bits)
        # Bit  [31]    = Parity (odd parity)
        word = label_byte & 0xFF
        word |= (sdi & 0x03) << 8
        word |= (data_int & 0x7FFFF) << 10
        word |= (ssm & 0x03) << 29

        # Compute odd parity over bits 0-30
        parity = 0
        temp = word & 0x7FFFFFFF
        while temp:
            parity ^= (temp & 1)
            temp >>= 1
        parity ^= 1  # odd parity: total number of 1s must be odd
        word |= (parity & 1) << 31

        message = struct.pack(">I", word)

        fields = [
            {"name": "label", "offset": 0, "length": 1},
            {"name": "sdi", "offset": 1, "length": 1},
            {"name": "data_bnr", "offset": 1, "length": 3},
            {"name": "ssm", "offset": 3, "length": 1},
            {"name": "parity", "offset": 3, "length": 1},
        ]

        ssm_desc = {0b00: "failure_warning", 0b01: "no_computed_data",
                    0b10: "functional_test", 0b11: "normal_operation"}

        meta = {
            "label_octal": oct(label_octal),
            "label_decimal": label_octal,
            "parameter": label_info["name"],
            "units": label_info["units"],
            "value": round(value, 4),
            "data_raw": data_int,
            "sdi": sdi,
            "ssm": ssm,
            "ssm_description": ssm_desc[ssm],
            "parity_bit": parity,
            "word_hex": f"0x{word:08X}",
        }

        return message, self._finalize_metadata("arinc429", message, meta, fields)

    # ------------------------------------------------------------------ #
    # Part A-5: ARINC 653 partition config generation                      #
    # ------------------------------------------------------------------ #

    def generate_arinc653(self) -> Tuple[bytes, Dict]:
        """Generate an ARINC 653 partition configuration (binary + metadata)."""
        num_partitions = random.randint(3, 6)
        partitions = random.sample(self.ARINC_653_PARTITIONS, num_partitions)
        major_frame_ms = random.choice([50, 100, 200])

        # Inter-partition communication channels
        channels = []
        for i in range(random.randint(1, min(4, num_partitions - 1))):
            src = random.randint(0, num_partitions - 1)
            dst = random.randint(0, num_partitions - 1)
            while dst == src:
                dst = random.randint(0, num_partitions - 1)
            channels.append({
                "channel_id": i,
                "source_partition": partitions[src]["name"],
                "dest_partition": partitions[dst]["name"],
                "mode": random.choice(["QUEUING", "SAMPLING"]),
                "max_msg_size": random.choice([64, 128, 256, 512]),
            })

        # Health monitoring actions
        hm_table = []
        error_ids = ["DEADLINE_MISSED", "APPLICATION_ERROR", "NUMERIC_ERROR",
                     "ILLEGAL_REQUEST", "STACK_OVERFLOW", "MEMORY_VIOLATION",
                     "HARDWARE_FAULT", "POWER_FAIL"]
        for eid in random.sample(error_ids, random.randint(3, 6)):
            hm_table.append({
                "error_id": eid,
                "level": random.choice(["PARTITION", "MODULE", "PROCESS"]),
                "action": random.choice(["IGNORE", "WARM_START", "COLD_START", "STOP"]),
            })

        # Build binary representation
        parts = []
        # Header: magic(4B) + version(2B) + num_partitions(1B) + major_frame(2B)
        parts.append(b"A653")
        parts.append(struct.pack(">HBH", 0x0001, num_partitions, major_frame_ms))

        for p in partitions:
            name_bytes = p["name"].ljust(32).encode("ascii")[:32]
            parts.append(name_bytes)
            parts.append(struct.pack(">HH", p["budget_ms"], p["period_ms"]))
            dals_byte = {"DAL-A": 0x01, "DAL-B": 0x02, "DAL-C": 0x03,
                         "DAL-D": 0x04, "DAL-E": 0x05}[p["dals"]]
            parts.append(struct.pack(">B", dals_byte))

        # Channel count + channel records
        parts.append(struct.pack(">B", len(channels)))
        for ch in channels:
            src_bytes = ch["source_partition"].ljust(32).encode("ascii")[:32]
            dst_bytes = ch["dest_partition"].ljust(32).encode("ascii")[:32]
            mode_byte = 0x01 if ch["mode"] == "QUEUING" else 0x02
            parts.append(src_bytes + dst_bytes + struct.pack(">BH", mode_byte, ch["max_msg_size"]))

        # HM table count + entries
        parts.append(struct.pack(">B", len(hm_table)))
        for hm in hm_table:
            hm_id = hm["error_id"].ljust(24).encode("ascii")[:24]
            level_byte = {"PARTITION": 0x01, "MODULE": 0x02, "PROCESS": 0x03}[hm["level"]]
            action_byte = {"IGNORE": 0x00, "WARM_START": 0x01,
                           "COLD_START": 0x02, "STOP": 0x03}[hm["action"]]
            parts.append(hm_id + struct.pack(">BB", level_byte, action_byte))

        message = b"".join(parts)

        fields = [
            {"name": "magic", "offset": 0, "length": 4},
            {"name": "version", "offset": 4, "length": 2},
            {"name": "num_partitions", "offset": 6, "length": 1},
            {"name": "major_frame_ms", "offset": 7, "length": 2},
            {"name": "partition_table", "offset": 9,
             "length": num_partitions * (32 + 4 + 1)},
        ]

        meta = {
            "num_partitions": num_partitions,
            "major_frame_ms": major_frame_ms,
            "partitions": [{"name": p["name"], "budget_ms": p["budget_ms"],
                            "period_ms": p["period_ms"], "dal": p["dals"]}
                           for p in partitions],
            "channels": channels,
            "health_monitoring": hm_table,
        }

        return message, self._finalize_metadata("arinc653", message, meta, fields)

    # ------------------------------------------------------------------ #
    # Part B: Aviation Security Instruction Pairs                          #
    # ------------------------------------------------------------------ #

    def _gen_atc_pqc_pairs(self, count: int) -> List[Dict]:
        instructions = [
            "Design a PQC migration plan for ACARS messaging between airline dispatch and aircraft.",
            "How should ML-KEM be integrated into CPDLC controller-pilot communications?",
            "Evaluate the feasibility of aggregate signatures (ML-DSA) for batch ATC clearance verification.",
            "What is the overhead of adding ML-KEM-768 encapsulation to an ACARS VHF datalink frame?",
            "Propose a hybrid classical/PQC key agreement for the ATN (Aeronautical Telecommunication Network).",
            "How can PQC protect ADS-B surveillance data from spoofing attacks?",
            "Design a certificate rotation scheme for PQC-signed ATC messages under bandwidth constraints.",
            "Analyze the latency impact of SLH-DSA signatures on time-critical CPDLC uplink messages.",
        ]
        contexts = [
            "ACARS VHF bandwidth: 2400 bps. Max message size 220 characters. Current auth: none or ARINC 823 Part 1.",
            "CPDLC over ATN/IPS (ICAO Doc 9896). Current security: AMS (ATN Message Security). Latency budget: < 6 seconds end-to-end.",
            "Fleet: 250 Boeing 787 + 180 Airbus A350. Mixed avionics vendors. 15-year certification cycle.",
            "Oceanic FIR with FANS-1/A CPDLC. SATCOM link (Inmarsat SwiftBroadband). Round-trip latency ~2 seconds.",
        ]
        responses = [
            "## PQC Migration for ACARS\n\n### Challenges\n1. VHF bandwidth (2400 bps) severely limits key/signature sizes\n2. ML-KEM-768 ciphertexts (1088 bytes) exceed single ACARS frame\n3. Legacy avionics lack hardware acceleration for lattice operations\n\n### Phased Approach\n- Phase 1: Deploy hybrid X25519+ML-KEM-768 on ground-side ACARS routers\n- Phase 2: Use ACARS multi-block messages to carry PQC key material\n- Phase 3: Avionics software update (DO-178C DAL-B) to add ML-KEM support\n- Phase 4: Full PQC-only mode with ML-DSA-65 message signatures\n\n### Bandwidth Mitigation\n- Pre-established session keys (reduce per-message overhead)\n- Compressed ML-KEM ciphertexts where possible\n- Signature aggregation for batch messages",
            "## ML-KEM Integration for CPDLC\n\n### Architecture\n1. Initial handshake during CPDLC logon uses ML-KEM-768 for session key establishment\n2. Subsequent messages use AES-256-GCM with PQC-derived keys\n3. Re-keying every 30 minutes or 100 messages (whichever comes first)\n\n### Latency Analysis\n- ML-KEM-768 encapsulation: ~0.2ms (negligible vs 2s SATCOM RTT)\n- Additional payload: 1088 bytes for initial KEM exchange only\n- Per-message overhead: 16-byte GCM tag (acceptable)\n\n### Certification Path\n- EASA/FAA supplemental type certificate (STC)\n- DO-326A airworthiness security assessment required\n- Minimum 24-month certification timeline",
        ]
        return self._build_pairs("atc_pqc", count, instructions, contexts, responses)

    def _gen_forward_secrecy_pairs(self, count: int) -> List[Dict]:
        instructions = [
            "Design a forward-secure ratcheting protocol for aircraft-ground datalink communications.",
            "How can double-ratchet key derivation be adapted for the low-bandwidth ACARS channel?",
            "Evaluate post-quantum forward secrecy for SATCOM channels using ephemeral ML-KEM keys.",
            "What is the key erasure schedule for ensuring forward secrecy on avionics datalinks?",
            "Propose a forward-secret session resumption mechanism for intermittent satellite connectivity.",
            "Analyze the impact of PQC forward-secret ratcheting on CPDLC message ordering guarantees.",
        ]
        contexts = [
            "Aircraft operate in polar regions with satellite gaps up to 20 minutes. Must maintain forward secrecy across connection drops.",
            "FANS-1/A datalink with Iridium NEXT constellation. Uplink/downlink asymmetric bandwidth. Session keys stored in tamper-resistant ARINC 653 partition.",
            "Airline requirement: any compromise of current keys must not reveal past messages. NIST SP 800-227 compliance required.",
        ]
        responses = [
            "## Forward-Secure Ratcheting for Aircraft Datalink\n\n### Protocol Design\n1. **Initial Key Agreement**: ML-KEM-1024 ephemeral exchange during CPDLC logon\n2. **Symmetric Ratchet**: HKDF-SHA-384 chain key derivation per message\n3. **Asymmetric Ratchet**: New ML-KEM-768 ephemeral every 50 messages or 10 minutes\n4. **Key Erasure**: Previous chain keys securely zeroed after derivation\n\n### Satellite Gap Handling\n- Store ratchet state in tamper-resistant memory (ARINC 653 health-monitored partition)\n- On reconnection, resume from saved ratchet state\n- If gap > 30 minutes, perform full re-keying with new ML-KEM ephemeral\n\n### Bandwidth Budget\n- Symmetric ratchet: zero additional bandwidth\n- Asymmetric ratchet: 1088 bytes per re-key (amortized over 50 messages)\n- Average per-message overhead: ~22 bytes (acceptable on SATCOM)",
            "## PQC Forward Secrecy for SATCOM Channels\n\n### Approach\nUse ephemeral ML-KEM-768 key encapsulations piggybacked on periodic ADS-C reports.\n\n### Key Schedule\n1. Generate ephemeral ML-KEM keypair every ADS-C contract period (e.g., every 5 minutes)\n2. Encapsulate shared secret and include ciphertext in ADS-C extended report padding\n3. Derive message keys using HKDF(shared_secret || flight_id || timestamp)\n4. Previous ephemeral keys erased after successful acknowledgment\n\n### Security Properties\n- Compromise of any single epoch key reveals only that epoch's messages\n- Passive adversary recording traffic cannot decrypt past sessions\n- Compliant with NIST SP 800-227 PQC forward secrecy requirements",
        ]
        return self._build_pairs("forward_secrecy", count, instructions, contexts, responses)

    def _gen_avionics_security_pairs(self, count: int) -> List[Dict]:
        instructions = [
            "How should DO-326A airworthiness security requirements be applied to PQC-protected IMA partitions?",
            "Design a secure boot chain for ARINC 653 IMA using ML-DSA signatures.",
            "What are the DO-356A security testing requirements for a PQC cryptographic library on avionics hardware?",
            "Create an ARINC 653 health monitoring configuration for detecting cryptographic failures.",
            "Evaluate the impact of PQC key generation latency on ARINC 653 partition scheduling.",
            "How should post-quantum certificates be distributed to ARINC 653 partitions across IMA modules?",
            "Design a PQC key storage architecture compliant with ED-202A / DO-326A for avionics.",
        ]
        contexts = [
            "IMA platform: 8 ARINC 653 partitions on Honeywell Forge. DAL-A flight management, DAL-B display. DO-178C Level A certification required.",
            "ARINC 653 APEX interface. Secure boot must complete within 15-second power-on-to-ready budget. ML-DSA-65 verification time: ~1.5ms on ARM Cortex-R5.",
            "Airworthiness security assessment per DO-326A Section 5.3. Threat model includes supply chain attacks and post-quantum adversaries.",
        ]
        responses = [
            "## DO-326A PQC Requirements for IMA Partitions\n\n### Security Assessment Activities\n1. **Threat Assessment (Section 5.1)**: Include quantum computing threats to classical asymmetric crypto\n2. **Risk Assessment (Section 5.2)**: Classify PQC protection as HIGH for DAL-A partitions\n3. **Security Requirements (Section 5.3)**:\n   - SR-1: All partition images signed with ML-DSA-65 (NIST FIPS 204)\n   - SR-2: Inter-partition communication authenticated with HMAC-SHA-384 from PQC-derived keys\n   - SR-3: Key material stored in hardware security module (HSM) within health-monitored partition\n   - SR-4: Secure boot verification of all partition binaries before scheduling begins\n\n### Certification Impact\n- ML-DSA library must be developed to DO-178C DAL-A for flight-critical partitions\n- DO-356A testing: Robustness testing of PQC implementation against fault injection\n- EASA/FAA coordination for novel cryptographic means of compliance",
            "## Secure Boot for ARINC 653 IMA with ML-DSA\n\n### Boot Chain\n1. **Stage 0 (ROM)**: Verify Stage 1 loader using ML-DSA-87 root key burned in OTP fuses\n2. **Stage 1 (Bootloader)**: Verify module support software (MSS) signature\n3. **Stage 2 (MSS)**: Verify each ARINC 653 partition image signature (ML-DSA-65)\n4. **Stage 3 (Partition Init)**: Each partition verifies its configuration table\n\n### Timing Budget\n- ML-DSA-87 verify: ~2.5ms (Stage 0)\n- ML-DSA-65 verify x 8 partitions: ~12ms (Stage 2)\n- Total crypto overhead: < 15ms (well within 15-second budget)\n\n### Key Management\n- Root key: burned in OTP, never updated\n- Signing keys: rotated per software load, distributed via airline PKI\n- Revocation: CRL embedded in bootloader update",
        ]
        return self._build_pairs("avionics_security", count, instructions, contexts, responses)

    def _gen_satellite_link_pairs(self, count: int) -> List[Dict]:
        instructions = [
            "Design bandwidth-efficient PQC key exchange for Iridium NEXT satellite datalinks.",
            "How can ML-KEM ciphertexts be compressed for Inmarsat SwiftBroadband constraints?",
            "Evaluate the trade-offs of different PQC KEM parameter sets for L-band satellite channels.",
            "Propose a PQC authentication scheme for SATCOM channel establishment that fits within ARINC 741 framing.",
            "What is the latency overhead of PQC handshake on a 600ms round-trip satellite link?",
            "Design a PQC session management protocol for aircraft transitioning between satellite coverage zones.",
        ]
        contexts = [
            "Iridium NEXT: 66 satellites, L-band, max throughput 352 kbps (SBD: 340 bytes per message). RTT ~100ms low-earth orbit.",
            "Inmarsat SwiftBroadband: 432 kbps shared. Packet size up to 1500 bytes. RTT ~600ms (GEO orbit). Used for FANS-1/A CPDLC.",
            "ARINC 741 satellite data unit (SDU) mounted on aircraft. Interface to CMU (Communications Management Unit). Bandwidth allocation: 8.4 kbps for safety services.",
        ]
        responses = [
            "## Bandwidth-Efficient PQC for Iridium NEXT\n\n### Constraints\n- SBD (Short Burst Data): 340 bytes max per message\n- ML-KEM-768 ciphertext: 1088 bytes (does NOT fit in single SBD)\n- ML-KEM-512 ciphertext: 768 bytes (still exceeds SBD limit)\n\n### Solution: Fragmented KEM Exchange\n1. Split ML-KEM-768 public key (1184 bytes) across 4 SBD messages\n2. Ground station reassembles and performs encapsulation\n3. Ciphertext (1088 bytes) sent in 4 SBD messages to aircraft\n4. Derived AES-256 session key used for subsequent traffic\n\n### Optimization: Pre-provisioned Keys\n- Pre-load ML-KEM public keys during gate operations (WiFi/4G)\n- Only exchange ciphertexts over Iridium (3 SBD messages)\n- Reduces satellite bandwidth by 57%\n\n### Fallback\n- X25519 + ML-KEM-512 hybrid fits in 2 SBD messages\n- Acceptable for non-safety ACARS (AOC messages)",
            "## PQC for Inmarsat SwiftBroadband\n\n### Approach\nML-KEM-768 handshake fits within single 1500-byte Inmarsat packet with room for headers.\n\n### Protocol\n1. Aircraft -> Ground: ML-KEM-768 public key (1184 bytes) + nonce (32 bytes) = 1216 bytes + headers\n2. Ground -> Aircraft: ML-KEM-768 ciphertext (1088 bytes) + ML-DSA-65 signature (3309 bytes)\n   - Signature sent in separate packet due to size\n3. Both sides derive AES-256-GCM session keys via HKDF\n\n### Latency Analysis\n- 2 round trips for full handshake: 2 x 600ms = 1.2 seconds\n- Acceptable for CPDLC (6-second end-to-end budget)\n- Session resumption (PSK mode): 1 round trip = 0.6 seconds",
        ]
        return self._build_pairs("satellite_link", count, instructions, contexts, responses)

    def _gen_flight_data_integrity_pairs(self, count: int) -> List[Dict]:
        instructions = [
            "Design a PQC signature scheme for Electronic Flight Bag (EFB) software and chart updates.",
            "How should navigation database (NavDB) updates be authenticated using post-quantum signatures?",
            "Propose a PQC integrity verification system for ARINC 424 navigation data records.",
            "What are the requirements for PQC-signed Type A/B software loads per DO-178C?",
            "Design a chain-of-custody verification for flight plan data from dispatch to FMS.",
            "Evaluate ML-DSA vs SLH-DSA for signing performance-based navigation (PBN) procedures.",
        ]
        contexts = [
            "EFB Class 2 (iPad-based) running certified charting application. NavDB updated every 28 days (AIRAC cycle). ~15,000 records per update.",
            "FMS navigation database: ARINC 424 format. 600,000+ records. Currently authenticated via CRC-32 checksum only. Spoofed NavDB = wrong waypoints = CFIT risk.",
            "Airline dispatch to aircraft flight plan delivery via ACARS. Current integrity: none. Attack: modified waypoints could reroute aircraft into hostile airspace.",
        ]
        responses = [
            "## PQC Signature for EFB Updates\n\n### Architecture\n1. **Signing Authority**: Airline IT security team signs EFB packages with ML-DSA-65\n2. **Distribution**: Signed packages distributed via airline Wi-Fi or LTE at gate\n3. **Verification**: EFB application verifies ML-DSA-65 signature before installation\n4. **Root of Trust**: ML-DSA-87 root certificate embedded in EFB MDM profile\n\n### NavDB Integrity\n- Each AIRAC cycle package signed as a whole (ML-DSA-65)\n- Individual ARINC 424 records include HMAC-SHA-256 tags derived from PQC session key\n- Merkle tree over all records allows incremental verification\n\n### Performance\n- ML-DSA-65 verification: ~1.5ms on iPad A15 chip\n- Full NavDB verification (15,000 records): < 2 seconds\n- Acceptable for 28-day update cycle",
            "## NavDB Authentication with PQC\n\n### Threat Model\n- Supply chain attack: compromised NavDB at data provider\n- Man-in-middle: modified NavDB during ACARS transfer\n- Insider threat: unauthorized NavDB modification at airline\n\n### Solution: Merkle-Signed ARINC 424\n1. Build Merkle tree over all 600,000 ARINC 424 records (SHA-384)\n2. Sign Merkle root with ML-DSA-87 (high-security root)\n3. Each record carries its Merkle proof (20 hashes x 48 bytes = 960 bytes)\n4. FMS verifies individual records on demand without loading entire DB\n\n### Size Impact\n- ML-DSA-87 signature: 4627 bytes (one per database)\n- Per-record Merkle proof: ~960 bytes (4.2% overhead on typical 23KB record)\n- Total DB size increase: < 5% (acceptable for modern FMS storage)",
        ]
        return self._build_pairs("flight_data_integrity", count, instructions, contexts, responses)

    def _build_pairs(self, category: str, count: int,
                     instructions: List[str], contexts: List[str],
                     responses: List[str]) -> List[Dict]:
        pairs = []
        for _ in range(count):
            pairs.append({
                "pair_id": str(uuid.uuid4()),
                "category": category,
                "difficulty": random.choice(self.DIFFICULTIES),
                "instruction": random.choice(instructions),
                "context": random.choice(contexts),
                "response": random.choice(responses),
            })
        return pairs

    # ------------------------------------------------------------------ #
    # Dataset generation                                                   #
    # ------------------------------------------------------------------ #

    def generate_dataset(self, num_protocol_samples: int,
                         num_instruction_pairs: int,
                         output_dir: str) -> Dict:
        """Generate the full aviation dataset.

        Args:
            num_protocol_samples: Number of protocol binary samples (Part A).
            num_instruction_pairs: Number of instruction pairs (Part B).
            output_dir: Root output directory.

        Returns:
            Dataset metadata dict.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # ---- Part A: Protocol Samples -------------------------------- #
        protocol_generators = [
            ("acars", self.generate_acars, 0.25),
            ("cpdlc", self.generate_cpdlc, 0.25),
            ("ads_c", self.generate_ads_c, 0.20),
            ("arinc429", self.generate_arinc429, 0.20),
            ("arinc653", self.generate_arinc653, 0.10),
        ]

        protocol_dir = output_path / "protocols"
        protocol_dir.mkdir(parents=True, exist_ok=True)

        samples_by_type: Dict[str, int] = {}
        sample_idx = 0

        for msg_type, generator, ratio in protocol_generators:
            count = int(num_protocol_samples * ratio)
            samples_by_type[msg_type] = count

            for _ in range(count):
                message, metadata = generator()
                metadata["sample_index"] = sample_idx

                bin_path = protocol_dir / f"aviation_{msg_type}_{sample_idx:06d}.bin"
                with open(bin_path, "wb") as f:
                    f.write(message)

                meta_path = protocol_dir / f"aviation_{msg_type}_{sample_idx:06d}.json"
                with open(meta_path, "w") as f:
                    json.dump(metadata, f, indent=2, default=str)

                sample_idx += 1

        # ---- Part B: Instruction Pairs ------------------------------- #
        instruction_dir = output_path / "instruction_pairs"
        instruction_dir.mkdir(parents=True, exist_ok=True)

        pairs_per_cat = num_instruction_pairs // len(self.INSTRUCTION_CATEGORIES)
        remainder = num_instruction_pairs % len(self.INSTRUCTION_CATEGORIES)

        category_generators = {
            "atc_pqc": self._gen_atc_pqc_pairs,
            "forward_secrecy": self._gen_forward_secrecy_pairs,
            "avionics_security": self._gen_avionics_security_pairs,
            "satellite_link": self._gen_satellite_link_pairs,
            "flight_data_integrity": self._gen_flight_data_integrity_pairs,
        }

        all_pairs: List[Dict] = []
        pairs_by_category: Dict[str, int] = {}

        for idx, category in enumerate(self.INSTRUCTION_CATEGORIES):
            count = pairs_per_cat + (1 if idx < remainder else 0)
            pairs = category_generators[category](count)
            all_pairs.extend(pairs)
            pairs_by_category[category] = count

        random.shuffle(all_pairs)

        # Write JSONL
        jsonl_path = instruction_dir / "aviation_security_pairs.jsonl"
        with open(jsonl_path, "w") as f:
            for pair in all_pairs:
                f.write(json.dumps(pair, default=str) + "\n")

        # Write JSON for compatibility
        json_path = instruction_dir / "aviation_security_pairs.json"
        with open(json_path, "w") as f:
            json.dump(all_pairs, f, indent=2, default=str)

        # ---- Metadata ------------------------------------------------ #
        dataset_metadata = {
            "protocol": "arinc_aviation",
            "version": "1.0",
            "total_samples": sample_idx + len(all_pairs),
            "protocol_samples": sample_idx,
            "instruction_pairs": len(all_pairs),
            "samples_by_type": samples_by_type,
            "pairs_by_category": pairs_by_category,
            "difficulties": {
                d: sum(1 for p in all_pairs if p["difficulty"] == d)
                for d in self.DIFFICULTIES
            },
            "generated_at": datetime.utcnow().isoformat(),
        }

        with open(output_path / "dataset_metadata.json", "w") as f:
            json.dump(dataset_metadata, f, indent=2)

        return dataset_metadata


def main():
    """Generate ARINC Aviation dataset."""
    generator = ARINCAviationGenerator(seed=42)
    output_dir = Path(__file__).parent.parent / "protocols" / "arinc_aviation"

    print("Generating ARINC Aviation dataset...")
    metadata = generator.generate_dataset(
        num_protocol_samples=1000,
        num_instruction_pairs=200,
        output_dir=str(output_dir),
    )

    print(f"Generated {metadata['protocol_samples']} protocol samples")
    print(f"Generated {metadata['instruction_pairs']} instruction pairs")
    print(f"Output directory: {output_dir}")
    print("\nProtocol samples by type:")
    for msg_type, count in metadata["samples_by_type"].items():
        print(f"  - {msg_type}: {count} samples")
    print("\nInstruction pairs by category:")
    for category, count in metadata["pairs_by_category"].items():
        print(f"  - {category}: {count} pairs")
    print(f"\nDifficulty distribution:")
    for difficulty, count in metadata["difficulties"].items():
        print(f"  - {difficulty}: {count} pairs")


if __name__ == "__main__":
    main()
