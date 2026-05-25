"""
V2X Automotive Protocol Message Generator

Generates realistic IEEE 1609.2 / SAE J2735 V2X protocol samples and
PQC security instruction pairs for training QBITEL models.

Part A - Protocol Samples (binary + JSON metadata):
  - Basic Safety Message (BSM) Part I & II
  - Signal Phase and Timing (SPaT)
  - MAP (MapData)
  - Personal Safety Message (PSM)
  - Emergency Vehicle Alert (EVA)

Part B - V2X Security Instruction Pairs (JSONL):
  - v2x_pqc_integration
  - misbehavior_detection
  - certificate_management
  - group_signatures
  - batch_verification

IEEE 1609.2 Security Layer:
  - SignerIdentifier (digest or certificate)
  - HashAlgorithm (SHA-256 / SHA-384)
  - Signature (ECDSA-256 / ECDSA-384, future: Falcon-512 / ML-DSA-44)

SAE J2735 DER Encoding:
  - PSID 0x20 for BSM, 0x8003 for SPaT/MAP, 0x8002 for PSM
  - UPER (Unaligned Packed Encoding Rules) for payload
"""

import json
import random
import struct
import hashlib
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class V2XAutomotiveGenerator:
    """Generate realistic IEEE 1609.2 / SAE J2735 V2X messages for ML training."""

    # SAE J2735 PSID (Provider Service Identifier) values
    PSID_MAP = {
        "bsm":  0x20,
        "spat": 0x8003,
        "map":  0x8003,
        "psm":  0x8002,
        "eva":  0x20,
    }

    # IEEE 1609.2 protocol version
    PROTOCOL_VERSION = 3

    # Hash algorithms per IEEE 1609.2
    HASH_ALGORITHMS = ["SHA-256", "SHA-384"]

    # Signature algorithms (current + PQC candidates)
    SIG_ALGORITHMS_CURRENT = ["ECDSA-P256", "ECDSA-P384"]
    SIG_ALGORITHMS_PQC = ["Falcon-512", "ML-DSA-44", "ML-DSA-65"]

    # Signer identifier types
    SIGNER_TYPES = ["digest", "certificate", "self"]

    # BSM brake status flags (SAE J2735 BrakeSystemStatus)
    BRAKE_STATUSES = [
        {"wheelBrakes": "0000", "traction": "off", "abs": "off", "scs": "off",
         "brakeBoost": "off", "auxBrakes": "off"},
        {"wheelBrakes": "1111", "traction": "on", "abs": "on", "scs": "on",
         "brakeBoost": "on", "auxBrakes": "off"},
        {"wheelBrakes": "1100", "traction": "on", "abs": "off", "scs": "off",
         "brakeBoost": "off", "auxBrakes": "off"},
        {"wheelBrakes": "0011", "traction": "off", "abs": "on", "scs": "off",
         "brakeBoost": "off", "auxBrakes": "on"},
    ]

    # Realistic US intersection GPS coordinates (lat, lon, name)
    US_INTERSECTIONS = [
        (40.758896, -73.985130, "Times Square, NYC"),
        (34.052235, -118.243683, "Downtown LA"),
        (41.878113, -87.629799, "The Loop, Chicago"),
        (29.760427, -95.369804, "Downtown Houston"),
        (33.448376, -112.074036, "Central Phoenix"),
        (39.739236, -104.990251, "Downtown Denver"),
        (47.606209, -122.332069, "Downtown Seattle"),
        (38.907192, -77.036873, "Washington DC"),
        (42.360081, -71.058884, "Boston Common"),
        (37.774929, -122.419418, "SF Civic Center"),
        (32.715736, -117.161087, "Downtown San Diego"),
        (30.267153, -97.743057, "Austin Capitol"),
        (36.162663, -86.781601, "Downtown Nashville"),
        (35.227085, -80.843124, "Uptown Charlotte"),
        (39.961176, -82.998795, "Downtown Columbus"),
        (25.761681, -80.191788, "Downtown Miami"),
        (33.748997, -84.387985, "Midtown Atlanta"),
        (44.977753, -93.265011, "Downtown Minneapolis"),
        (38.627003, -90.199402, "Downtown St. Louis"),
        (45.523064, -122.676483, "Portland Pioneer Sq"),
    ]

    # Vehicle types for PSM and BSM Part II
    VEHICLE_TYPES = [
        "passenger", "bus", "lightTruck", "heavyTruck", "motorcycle",
        "emergencyVehicle", "specialVehicle", "trailer",
    ]

    # PSM pedestrian types
    PEDESTRIAN_TYPES = [
        "unavailable", "aPEDESTRIAN", "aPEDALCYCLIST", "aPUBLICSAFETYWORKER",
        "anANIMAL",
    ]

    # SPaT movement phases
    SIGNAL_PHASES = [
        "unavailable", "dark", "stop-Then-Proceed", "stop-And-Remain",
        "pre-Movement", "permissive-Movement-Allowed",
        "protected-Movement-Allowed", "permissive-clearance",
        "protected-clearance", "caution-Conflicting-Traffic",
    ]

    # Lane types for MAP
    LANE_TYPES = [
        "vehicle", "crosswalk", "bikeLane", "sidewalk", "median",
        "striping", "trackedVehicle", "parking",
    ]

    # Lane attributes
    LANE_DIRECTIONS = ["ingressPath", "egressPath"]
    LANE_SHARING = ["overlappingLaneDescriptionProvided", "multipleLanesTreatedAsOneLane"]

    # V2X instruction pair categories
    INSTRUCTION_CATEGORIES = [
        "v2x_pqc_integration",
        "misbehavior_detection",
        "certificate_management",
        "group_signatures",
        "batch_verification",
    ]

    DIFFICULTIES = ["basic", "intermediate", "advanced"]

    def __init__(self, seed: Optional[int] = None):
        if seed:
            random.seed(seed)
        self.msg_count = random.randint(0, 127)
        self.temp_id_counter = random.randint(0, 0xFFFFFFFF)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _next_msg_count(self) -> int:
        """Increment and return MsgCount (0-127)."""
        self.msg_count = (self.msg_count + 1) % 128
        return self.msg_count

    def _next_temp_id(self) -> bytes:
        """Generate a 4-byte TemporaryID."""
        self.temp_id_counter = (self.temp_id_counter + 1) % 0xFFFFFFFF
        return struct.pack(">I", self.temp_id_counter)

    def _random_gps(self) -> Tuple[float, float, str]:
        """Pick a random US intersection with small jitter."""
        base_lat, base_lon, name = random.choice(self.US_INTERSECTIONS)
        lat = base_lat + random.uniform(-0.005, 0.005)
        lon = base_lon + random.uniform(-0.005, 0.005)
        return round(lat, 7), round(lon, 7), name

    def _encode_latitude(self, lat: float) -> int:
        """Encode latitude per SAE J2735 (units of 1/10 micro-degree)."""
        return int(lat * 1e7)

    def _encode_longitude(self, lon: float) -> int:
        """Encode longitude per SAE J2735 (units of 1/10 micro-degree)."""
        return int(lon * 1e7)

    def _encode_elevation(self, meters: float) -> int:
        """Encode elevation per SAE J2735 (units of 0.1 m, offset -409.5)."""
        return int((meters + 409.5) * 10)

    def _encode_speed(self, mph: float) -> int:
        """Encode speed per SAE J2735 (units of 0.02 m/s)."""
        ms = mph * 0.44704
        return int(ms / 0.02)

    def _encode_heading(self, degrees: float) -> int:
        """Encode heading per SAE J2735 (units of 0.0125 degrees)."""
        return int(degrees / 0.0125)

    def _random_speed_mph(self) -> float:
        """Generate realistic vehicle speed in mph."""
        scenario = random.choices(
            ["stopped", "urban", "suburban", "highway"],
            weights=[15, 30, 25, 30],
            k=1
        )[0]
        if scenario == "stopped":
            return 0.0
        elif scenario == "urban":
            return round(random.uniform(5, 35), 1)
        elif scenario == "suburban":
            return round(random.uniform(25, 55), 1)
        else:
            return round(random.uniform(55, 120), 1)

    def _random_heading(self) -> float:
        """Generate random heading 0-359.9875 degrees."""
        return round(random.uniform(0, 359.9875), 4)

    def _random_acceleration(self) -> Tuple[float, float, float, float]:
        """Generate lateral, longitudinal, vertical accel and yaw rate."""
        lat_accel = round(random.uniform(-3.0, 3.0), 2)
        lon_accel = round(random.uniform(-4.0, 4.0), 2)
        vert_accel = round(random.uniform(-1.0, 1.0), 2)
        yaw_rate = round(random.uniform(-30.0, 30.0), 2)
        return lat_accel, lon_accel, vert_accel, yaw_rate

    def _build_1609_2_header(self, psid: int) -> Tuple[bytes, Dict]:
        """Build IEEE 1609.2 security header (simplified DER-like)."""
        version = self.PROTOCOL_VERSION
        hash_algo = random.choice(self.HASH_ALGORITHMS)
        signer_type = random.choice(self.SIGNER_TYPES)

        use_pqc = random.random() < 0.25
        if use_pqc:
            sig_algo = random.choice(self.SIG_ALGORITHMS_PQC)
        else:
            sig_algo = random.choice(self.SIG_ALGORITHMS_CURRENT)

        # Construct header bytes
        header = bytearray()
        # Protocol version (1 byte)
        header.append(version)
        # Content type: signed data = 0x03
        header.append(0x03)
        # Hash algorithm identifier
        hash_id = 0x00 if hash_algo == "SHA-256" else 0x01
        header.append(hash_id)
        # PSID (variable length, use 2 bytes for > 127)
        if psid <= 0x7F:
            header.append(psid)
        else:
            header.append(0x80 | ((psid >> 8) & 0x7F))
            header.append(psid & 0xFF)
        # Signer type
        signer_map = {"digest": 0x00, "certificate": 0x01, "self": 0x02}
        header.append(signer_map[signer_type])

        # Signer identifier (8-byte HashedId8 digest or 32-byte cert hash)
        if signer_type == "digest":
            signer_id = bytes([random.randint(0, 255) for _ in range(8)])
        elif signer_type == "certificate":
            signer_id = bytes([random.randint(0, 255) for _ in range(32)])
        else:
            signer_id = b""
        header.extend(signer_id)

        # Signature placeholder (size depends on algorithm)
        sig_sizes = {
            "ECDSA-P256": 64, "ECDSA-P384": 96,
            "Falcon-512": 666, "ML-DSA-44": 2420, "ML-DSA-65": 3293,
        }
        sig_size = sig_sizes.get(sig_algo, 64)
        signature = bytes([random.randint(0, 255) for _ in range(sig_size)])
        header.extend(struct.pack(">H", sig_size))
        header.extend(signature)

        meta = {
            "ieee_1609_2_version": version,
            "content_type": "signedData",
            "hash_algorithm": hash_algo,
            "psid": hex(psid),
            "signer_type": signer_type,
            "signer_id_hex": signer_id.hex() if signer_id else "",
            "signature_algorithm": sig_algo,
            "signature_size_bytes": sig_size,
            "is_pqc_signature": use_pqc,
            "header_length": len(header),
        }

        return bytes(header), meta

    def _build_sec_fields(self, header_bytes: bytes) -> List[Dict]:
        """Build field-level metadata for the security header."""
        fields = [
            {"name": "protocol_version", "offset": 0, "length": 1},
            {"name": "content_type", "offset": 1, "length": 1},
            {"name": "hash_algorithm", "offset": 2, "length": 1},
        ]
        return fields

    def _finalize_metadata(
        self, msg_type: str, message: bytes, meta: Dict, fields: List[Dict]
    ) -> Dict:
        """Create the final metadata dict."""
        meta.update({
            "protocol": "v2x_ieee1609_sae_j2735",
            "message_type": msg_type,
            "timestamp": datetime.now().isoformat(),
            "message_length": len(message),
            "fields": fields,
            "hash": hashlib.sha256(message).hexdigest(),
        })
        return meta

    # ------------------------------------------------------------------
    # Part A: SAE J2735 Protocol Sample Generators
    # ------------------------------------------------------------------

    def generate_bsm(self) -> Tuple[bytes, Dict]:
        """Generate a Basic Safety Message (BSM) Part I + Part II."""
        msg_count = self._next_msg_count()
        temp_id = self._next_temp_id()
        sec_mark = random.randint(0, 59999)  # milliseconds in current minute
        lat, lon, location_name = self._random_gps()
        elevation_m = round(random.uniform(-10, 2500), 1)
        speed_mph = self._random_speed_mph()
        heading = self._random_heading()
        lat_accel, lon_accel, vert_accel, yaw_rate = self._random_acceleration()
        brake = random.choice(self.BRAKE_STATUSES)
        vehicle_width = round(random.uniform(1.5, 2.8), 1)
        vehicle_length = round(random.uniform(3.5, 22.0), 1)

        # Part II optional extensions
        has_part_ii = random.random() < 0.40
        vehicle_type = random.choice(self.VEHICLE_TYPES) if has_part_ii else None
        exterior_lights = None
        if has_part_ii:
            exterior_lights = random.choice([
                "lowBeamHeadlightsOn", "highBeamHeadlightsOn",
                "hazardSignalOn", "allLightsOff",
            ])

        # Build payload (simplified DER / UPER representation)
        payload = bytearray()
        # MessageFrame: messageId = 20 (BSM)
        payload.extend(struct.pack(">B", 20))
        # MsgCount
        payload.extend(struct.pack(">B", msg_count))
        # TemporaryID (4 bytes)
        payload.extend(temp_id)
        # SecMark (2 bytes)
        payload.extend(struct.pack(">H", sec_mark))
        # Latitude (4 bytes, signed)
        payload.extend(struct.pack(">i", self._encode_latitude(lat)))
        # Longitude (4 bytes, signed)
        payload.extend(struct.pack(">i", self._encode_longitude(lon)))
        # Elevation (2 bytes)
        payload.extend(struct.pack(">H", self._encode_elevation(elevation_m)))
        # Speed (2 bytes)
        payload.extend(struct.pack(">H", self._encode_speed(speed_mph)))
        # Heading (2 bytes)
        payload.extend(struct.pack(">H", self._encode_heading(heading)))
        # Acceleration set (8 bytes: 2 each for lat, lon, vert, yaw)
        payload.extend(struct.pack(">hhhh",
            int(lat_accel * 100), int(lon_accel * 100),
            int(vert_accel * 100), int(yaw_rate * 100)))
        # Brake system status (2 bytes packed)
        brake_bits = int(brake["wheelBrakes"], 2) << 4
        brake_bits |= (1 if brake["traction"] == "on" else 0) << 3
        brake_bits |= (1 if brake["abs"] == "on" else 0) << 2
        brake_bits |= (1 if brake["scs"] == "on" else 0) << 1
        brake_bits |= (1 if brake["brakeBoost"] == "on" else 0)
        aux = 1 if brake["auxBrakes"] == "on" else 0
        payload.extend(struct.pack(">BB", brake_bits, aux))
        # Vehicle size (4 bytes: width 2, length 2, in cm)
        payload.extend(struct.pack(">HH", int(vehicle_width * 100), int(vehicle_length * 100)))

        # Part II marker
        if has_part_ii:
            payload.append(0x01)  # Part II present flag
            vtype_map = {v: i for i, v in enumerate(self.VEHICLE_TYPES)}
            payload.append(vtype_map.get(vehicle_type, 0))
            light_codes = {
                "lowBeamHeadlightsOn": 0x01, "highBeamHeadlightsOn": 0x02,
                "hazardSignalOn": 0x04, "allLightsOff": 0x00,
            }
            payload.append(light_codes.get(exterior_lights, 0x00))
        else:
            payload.append(0x00)

        # Prepend security header
        sec_header, sec_meta = self._build_1609_2_header(self.PSID_MAP["bsm"])
        message = sec_header + bytes(payload)

        meta = {
            "msg_count": msg_count,
            "temp_id_hex": temp_id.hex(),
            "sec_mark_ms": sec_mark,
            "latitude": lat,
            "longitude": lon,
            "location_name": location_name,
            "elevation_m": elevation_m,
            "speed_mph": speed_mph,
            "heading_deg": heading,
            "acceleration": {
                "lateral_g": lat_accel,
                "longitudinal_g": lon_accel,
                "vertical_g": vert_accel,
                "yaw_rate_dps": yaw_rate,
            },
            "brake_system_status": brake,
            "vehicle_size": {
                "width_m": vehicle_width,
                "length_m": vehicle_length,
            },
            "has_part_ii": has_part_ii,
            "vehicle_type": vehicle_type,
            "exterior_lights": exterior_lights,
            "security": sec_meta,
        }

        fields = self._build_sec_fields(sec_header)
        fields.extend([
            {"name": "message_id", "offset": len(sec_header), "length": 1},
            {"name": "msg_count", "offset": len(sec_header) + 1, "length": 1},
            {"name": "temporary_id", "offset": len(sec_header) + 2, "length": 4},
            {"name": "sec_mark", "offset": len(sec_header) + 6, "length": 2},
            {"name": "latitude", "offset": len(sec_header) + 8, "length": 4},
            {"name": "longitude", "offset": len(sec_header) + 12, "length": 4},
            {"name": "elevation", "offset": len(sec_header) + 16, "length": 2},
            {"name": "speed", "offset": len(sec_header) + 18, "length": 2},
            {"name": "heading", "offset": len(sec_header) + 20, "length": 2},
            {"name": "acceleration_set", "offset": len(sec_header) + 22, "length": 8},
            {"name": "brake_system_status", "offset": len(sec_header) + 30, "length": 2},
            {"name": "vehicle_size", "offset": len(sec_header) + 32, "length": 4},
        ])

        return message, self._finalize_metadata("bsm", message, meta, fields)

    def generate_spat(self) -> Tuple[bytes, Dict]:
        """Generate a Signal Phase and Timing (SPaT) message."""
        msg_count = self._next_msg_count()
        lat, lon, location_name = self._random_gps()
        intersection_id = random.randint(1, 65535)
        minute_of_year = random.randint(0, 527040)

        # Generate signal groups (3-8 phases per intersection)
        num_groups = random.randint(3, 8)
        signal_groups = []
        for sg_id in range(1, num_groups + 1):
            phase = random.choice(self.SIGNAL_PHASES)
            min_end_time = random.randint(10, 300)  # tenths of second
            max_end_time = min_end_time + random.randint(5, 100)
            likely_time = min_end_time + random.randint(0, max_end_time - min_end_time)
            signal_groups.append({
                "signal_group_id": sg_id,
                "phase_state": phase,
                "min_end_time_ds": min_end_time,
                "max_end_time_ds": max_end_time,
                "likely_time_ds": likely_time,
                "confidence": random.randint(0, 15),
            })

        # Build payload
        payload = bytearray()
        # MessageFrame: messageId = 19 (SPaT)
        payload.extend(struct.pack(">B", 19))
        payload.extend(struct.pack(">B", msg_count))
        payload.extend(struct.pack(">H", intersection_id))
        payload.extend(struct.pack(">I", minute_of_year))
        # Number of signal groups
        payload.extend(struct.pack(">B", num_groups))
        for sg in signal_groups:
            payload.extend(struct.pack(">B", sg["signal_group_id"]))
            phase_idx = self.SIGNAL_PHASES.index(sg["phase_state"])
            payload.extend(struct.pack(">B", phase_idx))
            payload.extend(struct.pack(">HHH",
                sg["min_end_time_ds"], sg["max_end_time_ds"], sg["likely_time_ds"]))
            payload.extend(struct.pack(">B", sg["confidence"]))

        sec_header, sec_meta = self._build_1609_2_header(self.PSID_MAP["spat"])
        message = sec_header + bytes(payload)

        meta = {
            "intersection_id": intersection_id,
            "msg_count": msg_count,
            "minute_of_year": minute_of_year,
            "location_name": location_name,
            "latitude": lat,
            "longitude": lon,
            "num_signal_groups": num_groups,
            "signal_groups": signal_groups,
            "security": sec_meta,
        }

        fields = self._build_sec_fields(sec_header)
        fields.extend([
            {"name": "message_id", "offset": len(sec_header), "length": 1},
            {"name": "msg_count", "offset": len(sec_header) + 1, "length": 1},
            {"name": "intersection_id", "offset": len(sec_header) + 2, "length": 2},
            {"name": "minute_of_year", "offset": len(sec_header) + 4, "length": 4},
            {"name": "signal_group_count", "offset": len(sec_header) + 8, "length": 1},
        ])

        return message, self._finalize_metadata("spat", message, meta, fields)

    def generate_map_data(self) -> Tuple[bytes, Dict]:
        """Generate a MAP (MapData) message."""
        msg_count = self._next_msg_count()
        lat, lon, location_name = self._random_gps()
        intersection_id = random.randint(1, 65535)
        ref_lat = self._encode_latitude(lat)
        ref_lon = self._encode_longitude(lon)

        # Generate lane set (4-12 lanes)
        num_lanes = random.randint(4, 12)
        lane_set = []
        for lane_id in range(1, num_lanes + 1):
            lane_type = random.choice(self.LANE_TYPES)
            direction = random.choice(self.LANE_DIRECTIONS)
            sharing = random.choice(self.LANE_SHARING)
            # Lane node offsets (simplified: 2-6 nodes per lane)
            num_nodes = random.randint(2, 6)
            nodes = []
            for _ in range(num_nodes):
                dx = random.randint(-5000, 5000)   # cm offset
                dy = random.randint(-5000, 5000)
                nodes.append({"dx_cm": dx, "dy_cm": dy})
            lane_width_cm = random.choice([300, 330, 360, 366, 400, 450])
            lane_set.append({
                "lane_id": lane_id,
                "lane_type": lane_type,
                "lane_direction": direction,
                "lane_sharing": sharing,
                "lane_width_cm": lane_width_cm,
                "num_nodes": num_nodes,
                "nodes": nodes,
            })

        # Build payload
        payload = bytearray()
        # MessageFrame: messageId = 18 (MAP)
        payload.extend(struct.pack(">B", 18))
        payload.extend(struct.pack(">B", msg_count))
        payload.extend(struct.pack(">H", intersection_id))
        # Reference point
        payload.extend(struct.pack(">i", ref_lat))
        payload.extend(struct.pack(">i", ref_lon))
        # Number of lanes
        payload.extend(struct.pack(">B", num_lanes))
        for lane in lane_set:
            payload.extend(struct.pack(">B", lane["lane_id"]))
            type_idx = self.LANE_TYPES.index(lane["lane_type"])
            payload.extend(struct.pack(">B", type_idx))
            payload.extend(struct.pack(">H", lane["lane_width_cm"]))
            payload.extend(struct.pack(">B", lane["num_nodes"]))
            for node in lane["nodes"]:
                payload.extend(struct.pack(">hh", node["dx_cm"], node["dy_cm"]))

        sec_header, sec_meta = self._build_1609_2_header(self.PSID_MAP["map"])
        message = sec_header + bytes(payload)

        meta = {
            "intersection_id": intersection_id,
            "msg_count": msg_count,
            "ref_point": {"latitude": lat, "longitude": lon},
            "location_name": location_name,
            "num_lanes": num_lanes,
            "lane_set": lane_set,
            "security": sec_meta,
        }

        fields = self._build_sec_fields(sec_header)
        fields.extend([
            {"name": "message_id", "offset": len(sec_header), "length": 1},
            {"name": "msg_count", "offset": len(sec_header) + 1, "length": 1},
            {"name": "intersection_id", "offset": len(sec_header) + 2, "length": 2},
            {"name": "ref_latitude", "offset": len(sec_header) + 4, "length": 4},
            {"name": "ref_longitude", "offset": len(sec_header) + 8, "length": 4},
            {"name": "lane_count", "offset": len(sec_header) + 12, "length": 1},
        ])

        return message, self._finalize_metadata("map_data", message, meta, fields)

    def generate_psm(self) -> Tuple[bytes, Dict]:
        """Generate a Personal Safety Message (PSM)."""
        msg_count = self._next_msg_count()
        sec_mark = random.randint(0, 59999)
        lat, lon, location_name = self._random_gps()
        basic_type = random.choice(self.PEDESTRIAN_TYPES)

        # Pedestrian speed: 0-15 mph typical
        speed_mph = round(random.uniform(0, 15), 1)
        heading = self._random_heading()
        accuracy = random.randint(1, 15)  # position accuracy 0-15 scale

        # Build payload
        payload = bytearray()
        # MessageFrame: messageId = 32 (PSM)
        payload.extend(struct.pack(">B", 32))
        type_idx = self.PEDESTRIAN_TYPES.index(basic_type)
        payload.extend(struct.pack(">B", type_idx))
        payload.extend(struct.pack(">H", sec_mark))
        payload.extend(struct.pack(">B", msg_count))
        payload.extend(struct.pack(">i", self._encode_latitude(lat)))
        payload.extend(struct.pack(">i", self._encode_longitude(lon)))
        payload.extend(struct.pack(">H", self._encode_speed(speed_mph)))
        payload.extend(struct.pack(">H", self._encode_heading(heading)))
        payload.extend(struct.pack(">B", accuracy))

        sec_header, sec_meta = self._build_1609_2_header(self.PSID_MAP["psm"])
        message = sec_header + bytes(payload)

        meta = {
            "basic_type": basic_type,
            "sec_mark_ms": sec_mark,
            "msg_count": msg_count,
            "latitude": lat,
            "longitude": lon,
            "location_name": location_name,
            "speed_mph": speed_mph,
            "heading_deg": heading,
            "position_accuracy": accuracy,
            "security": sec_meta,
        }

        fields = self._build_sec_fields(sec_header)
        fields.extend([
            {"name": "message_id", "offset": len(sec_header), "length": 1},
            {"name": "basic_type", "offset": len(sec_header) + 1, "length": 1},
            {"name": "sec_mark", "offset": len(sec_header) + 2, "length": 2},
            {"name": "msg_count", "offset": len(sec_header) + 4, "length": 1},
            {"name": "latitude", "offset": len(sec_header) + 5, "length": 4},
            {"name": "longitude", "offset": len(sec_header) + 9, "length": 4},
            {"name": "speed", "offset": len(sec_header) + 13, "length": 2},
            {"name": "heading", "offset": len(sec_header) + 15, "length": 2},
        ])

        return message, self._finalize_metadata("psm", message, meta, fields)

    def generate_eva(self) -> Tuple[bytes, Dict]:
        """Generate an Emergency Vehicle Alert (EVA) message."""
        temp_id = self._next_temp_id()
        timestamp = int(datetime.now().timestamp() * 1000) & 0xFFFFFFFF
        lat, lon, location_name = self._random_gps()
        speed_mph = round(random.uniform(30, 90), 1)
        heading = self._random_heading()

        # RSA event priorities
        rsa_priorities = [
            ("ambulance", 7), ("fire_truck", 7), ("police", 6),
            ("hazmat", 8), ("tow_truck", 3), ("road_maintenance", 2),
        ]
        vehicle_label, priority = random.choice(rsa_priorities)
        response_type = random.choice([
            "emergency", "nonEmergency", "pursuit", "stationary",
        ])

        # Build payload
        payload = bytearray()
        # MessageFrame: messageId = 22 (EVA / RSA)
        payload.extend(struct.pack(">B", 22))
        payload.extend(struct.pack(">I", timestamp))
        payload.extend(temp_id)
        payload.extend(struct.pack(">i", self._encode_latitude(lat)))
        payload.extend(struct.pack(">i", self._encode_longitude(lon)))
        payload.extend(struct.pack(">H", self._encode_speed(speed_mph)))
        payload.extend(struct.pack(">H", self._encode_heading(heading)))
        payload.extend(struct.pack(">B", priority))
        # Response type index
        resp_types = ["emergency", "nonEmergency", "pursuit", "stationary"]
        payload.extend(struct.pack(">B", resp_types.index(response_type)))

        sec_header, sec_meta = self._build_1609_2_header(self.PSID_MAP["eva"])
        message = sec_header + bytes(payload)

        meta = {
            "temp_id_hex": temp_id.hex(),
            "timestamp_ms": timestamp,
            "latitude": lat,
            "longitude": lon,
            "location_name": location_name,
            "speed_mph": speed_mph,
            "heading_deg": heading,
            "vehicle_label": vehicle_label,
            "priority": priority,
            "response_type": response_type,
            "security": sec_meta,
        }

        fields = self._build_sec_fields(sec_header)
        fields.extend([
            {"name": "message_id", "offset": len(sec_header), "length": 1},
            {"name": "timestamp", "offset": len(sec_header) + 1, "length": 4},
            {"name": "temporary_id", "offset": len(sec_header) + 5, "length": 4},
            {"name": "latitude", "offset": len(sec_header) + 9, "length": 4},
            {"name": "longitude", "offset": len(sec_header) + 13, "length": 4},
            {"name": "speed", "offset": len(sec_header) + 17, "length": 2},
            {"name": "heading", "offset": len(sec_header) + 19, "length": 2},
            {"name": "priority", "offset": len(sec_header) + 21, "length": 1},
            {"name": "response_type", "offset": len(sec_header) + 22, "length": 1},
        ])

        return message, self._finalize_metadata("eva", message, meta, fields)

    # ------------------------------------------------------------------
    # Part B: V2X Security Instruction Pair Generators
    # ------------------------------------------------------------------

    def _generate_pqc_integration_pairs(self, count: int) -> List[Dict]:
        """Generate instruction pairs for V2X PQC algorithm integration."""
        pairs = []
        templates = [
            {
                "difficulty": "basic",
                "instructions": [
                    "Explain why Falcon-512 is preferred over ML-DSA-44 for V2X BSM signing.",
                    "What are the latency requirements for V2X message signing and verification?",
                    "Compare signature sizes of ECDSA-P256 vs Falcon-512 vs ML-DSA-44 for V2X use.",
                    "Why is V2X one of the first domains requiring PQC migration?",
                    "Describe the impact of signature size on 802.11p/DSRC channel congestion.",
                ],
                "responses": [
                    "Falcon-512 is preferred for V2X BSM signing because it offers compact signatures (~666 bytes) compared to ML-DSA-44 (~2,420 bytes). In V2X environments, BSMs are broadcast 10 times per second by every vehicle, and 802.11p DSRC channels have limited bandwidth (6-27 Mbps shared). Falcon-512's smaller signature footprint reduces channel congestion by approximately 72% versus ML-DSA-44. Additionally, Falcon-512 verification is fast enough to meet the 100ms end-to-end latency budget for safety-critical BSMs, though its signing requires careful constant-time implementation to avoid side-channel leakage in vehicular HSMs.",
                    "V2X message signing and verification must complete within strict latency bounds. BSMs require end-to-end processing under 100ms (including signing, transmission, reception, and verification). In dense traffic scenarios with 1,000+ vehicles, onboard units must verify all received BSMs within this window. Signing latency should be under 5ms per message. Batch verification techniques are essential when vehicle density exceeds 500 vehicles, allowing verification of 1,000+ signatures per second. SPaT/MAP messages have slightly relaxed timing (up to 1 second) since they change less frequently.",
                    "Signature size comparison for V2X:\n- ECDSA-P256: 64 bytes (current standard)\n- Falcon-512: ~666 bytes (10.4x larger)\n- ML-DSA-44: ~2,420 bytes (37.8x larger)\n- ML-DSA-65: ~3,293 bytes (51.5x larger)\n\nFor a BSM payload of ~38 bytes, ECDSA-P256 keeps the total message compact (~200 bytes with header). Falcon-512 increases it to ~800 bytes, still manageable for 10 Hz broadcast on DSRC. ML-DSA-44 pushes the total past 2.5 KB, potentially exceeding the 802.11p maximum frame size and requiring fragmentation, which is incompatible with safety-critical real-time requirements.",
                    "V2X is among the first domains requiring PQC migration for several reasons: (1) Vehicle lifecycles span 15-20 years, meaning cars manufactured today must resist quantum attacks through 2045+; (2) V2X PKI infrastructure (SCMS) issues long-lived root certificates that cannot be easily rotated; (3) Harvest-now-decrypt-later attacks could record signed BSMs and later forge vehicle identities; (4) Safety-critical applications mean that a compromised V2X signature scheme could enable spoofed collision warnings or traffic signal manipulation, causing physical harm.",
                    "Signature size directly affects 802.11p/DSRC channel congestion. The DSRC Control Channel (CCH) has ~6 Mbps effective throughput shared among all vehicles. With 10 Hz BSM broadcasts, each additional byte per signature adds 80 bits/second per vehicle. In a 1,000-vehicle scenario: ECDSA-P256 consumes ~1.6 Mbps total, Falcon-512 consumes ~6.4 Mbps (near channel capacity), and ML-DSA-44 would require ~19.4 Mbps (exceeding capacity by 3x). This makes Falcon-512 the practical PQC upper bound for DSRC, while C-V2X (LTE PC5) with higher bandwidth can accommodate ML-DSA-44.",
                ],
            },
            {
                "difficulty": "intermediate",
                "instructions": [
                    "Design a hybrid ECDSA+Falcon-512 signature scheme for V2X backward compatibility.",
                    "How should an OBU hardware security module (HSM) manage PQC key storage for V2X?",
                    "Propose a phased PQC migration timeline for US V2X infrastructure (SCMS).",
                    "Analyze the performance impact of PQC signatures on V2X batch verification throughput.",
                ],
                "responses": [
                    "Hybrid ECDSA+Falcon-512 V2X Signature Scheme:\n\n1. **Dual-Sign**: The OBU signs each BSM with both ECDSA-P256 and Falcon-512. The IEEE 1609.2 header carries both signatures in an extension field.\n2. **Receiver Selection**: Legacy OBUs verify only the ECDSA signature (backward compatible). PQC-capable OBUs verify the Falcon-512 signature (quantum-safe). Dual-capable OBUs verify both for maximum assurance.\n3. **Size Overhead**: ~730 bytes total signature (64 + 666), increasing BSM from ~200 to ~870 bytes. This fits within DSRC frame limits.\n4. **Transition Policy**: Phase 1 (years 1-3): dual signatures mandatory. Phase 2 (years 4-7): ECDSA optional, Falcon-512 mandatory. Phase 3 (year 8+): Falcon-512 only.\n5. **Certificate Structure**: Pseudonym certificates include both public keys. SCMS butterfly key expansion generates paired ECDSA+Falcon key sets.",
                    "OBU HSM PQC Key Storage Architecture:\n\n1. **Secure Key Hierarchy**: Root trust anchor (ML-KEM-768 encapsulated) -> Enrollment certificate key pair (ECDSA+Falcon dual) -> Pseudonym certificate pool (20 active Falcon-512 key pairs rotated weekly)\n2. **Memory Requirements**: Falcon-512 private key ~1,281 bytes, public key ~897 bytes. For 20 pseudonym slots: ~43.5 KB dedicated PQC partition (vs ~2.5 KB for ECDSA-only)\n3. **Key Generation**: Falcon key generation requires high-quality randomness and constant-time NTRU sampling. HSM must implement rejection sampling without timing side channels.\n4. **Performance Targets**: HSM must support Falcon-512 sign in <3ms and verify in <1ms to meet V2X timing. Hardware acceleration via dedicated lattice co-processor recommended.\n5. **Tamper Protection**: PQC keys stored in tamper-resistant secure element with voltage glitch and EM fault injection countermeasures.",
                    "Phased PQC Migration for US V2X SCMS:\n\n**Phase 0 - Preparation (2025-2026)**: NIST finalizes Falcon standard (FIPS 206). V2X equipment vendors prototype Falcon-512 in OBU/RSU HSMs. SCMS root CA generates quantum-safe root key pair.\n\n**Phase 1 - Hybrid Deployment (2027-2029)**: New vehicles ship with hybrid ECDSA+Falcon OBUs. RSUs upgraded to accept both signature types. SCMS issues dual pseudonym certificates. Backward compatibility maintained for legacy fleet.\n\n**Phase 2 - PQC Preferred (2030-2033)**: Falcon-512 becomes primary signature. ECDSA carried as fallback only. SCMS butterfly key mechanism updated for lattice-based key expansion. Misbehavior Authority updated with PQC-aware detection rules.\n\n**Phase 3 - PQC Only (2034+)**: ECDSA support deprecated. Legacy OBUs require aftermarket upgrade modules. SCMS root CA rotated to PQC-only hierarchy. Full quantum resistance achieved before projected quantum threat timeline.",
                    "PQC Impact on V2X Batch Verification:\n\nCurrent ECDSA-P256 batch verification achieves ~10,000 verifications/second on automotive-grade processors using multi-scalar multiplication. Falcon-512 verification is inherently fast (~0.1ms single verify) but does not benefit from the same batching optimizations as ECDSA.\n\n**Throughput Analysis**:\n- ECDSA-P256 batch (n=100): ~100us total = 10,000 msgs/sec\n- Falcon-512 single verify: ~100us each = ~10,000 msgs/sec\n- Falcon-512 with tree-based batching: ~60us amortized = ~16,000 msgs/sec\n\nFalcon-512 actually matches or exceeds ECDSA batch performance for single-core verification. The challenge is the 10x larger signature requiring 10x more bandwidth to receive before verification begins. Recommendation: implement Falcon-512 parallel pipeline verification with 4 hardware threads to achieve 40,000+ msgs/sec, sufficient for the densest highway scenarios.",
                ],
            },
            {
                "difficulty": "advanced",
                "instructions": [
                    "Design a crypto-agile V2X security architecture that can hot-swap between ECDSA, Falcon-512, and ML-DSA without OBU firmware updates.",
                ],
                "responses": [
                    "Crypto-Agile V2X Security Architecture:\n\n**1. Algorithm Negotiation Layer (ANL)**\nInsert an Algorithm Negotiation Extension into IEEE 1609.2 SignedData. The ANL header (4 bytes) encodes: algorithm_suite_id (1 byte), fallback_suite_id (1 byte), policy_epoch (2 bytes). Suite IDs map to pre-loaded algorithm implementations in the OBU secure enclave.\n\n**2. Algorithm Suites Registry**\n- Suite 0x01: ECDSA-P256 + SHA-256 (legacy)\n- Suite 0x02: Falcon-512 + SHA-256 (PQC primary)\n- Suite 0x03: ML-DSA-44 + SHA-384 (PQC alternate)\n- Suite 0x04: ECDSA-P256 + Falcon-512 hybrid\n- Suite 0xFF: Reserved for emergency rollback\n\n**3. OBU Secure Enclave Design**\nThe HSM runs a microkernel with pluggable crypto modules loaded as signed WebAssembly (Wasm) bytecode. Each algorithm suite is a Wasm module verified against the SCMS root of trust. New suites deployed via SCMS Policy Distribution Point without firmware OTA.\n\n**4. Suite Selection Protocol**\nRSUs broadcast a CryptoPolicy message (new PSID 0x8010) every 5 seconds containing: current_required_suite, acceptable_suites[], policy_epoch, transition_deadline. OBUs select the highest-priority matching suite from their loaded modules.\n\n**5. Emergency Rollback**\nIf an algorithm is compromised, SCMS issues an Emergency Revocation Notification (ERN) with rollback_to_suite. All OBUs within broadcast range switch within one policy_epoch cycle (5 seconds). ERN messages are signed with the root CA key using the rollback suite.\n\n**6. Key Lifecycle**\nEach pseudonym certificate bundle contains key pairs for all loaded suites. Butterfly key expansion runs in parallel for each algorithm. Storage overhead: ~50 KB per pseudonym slot for tri-algorithm support (ECDSA + Falcon + ML-DSA).",
                ],
            },
        ]

        for _ in range(count):
            template = random.choice(templates)
            instruction = random.choice(template["instructions"])
            response = random.choice(template["responses"])
            pairs.append({
                "pair_id": f"v2x_pqc_{uuid.uuid4().hex[:8]}",
                "category": "v2x_pqc_integration",
                "difficulty": template["difficulty"],
                "instruction": instruction,
                "context": "Domain: V2X automotive security. Standard: IEEE 1609.2 / SAE J2735. PQC migration for connected vehicle infrastructure.",
                "response": response,
            })
        return pairs

    def _generate_misbehavior_detection_pairs(self, count: int) -> List[Dict]:
        """Generate instruction pairs for V2X misbehavior detection."""
        pairs = []
        templates = [
            {
                "difficulty": "basic",
                "instructions": [
                    "What is a BSM spoofing attack and how does it endanger connected vehicles?",
                    "Describe a Sybil attack in the V2X context.",
                    "How can an OBU detect a ghost vehicle attack from forged BSMs?",
                    "What are the common types of V2X misbehavior that the Misbehavior Authority must detect?",
                ],
                "responses": [
                    "A BSM spoofing attack occurs when a malicious actor broadcasts fabricated Basic Safety Messages with false position, speed, or heading data. This can cause nearby vehicles to brake unnecessarily (phantom braking), change lanes into actual traffic, or ignore real hazards. For example, an attacker could spoof a stopped vehicle BSM in the middle of a highway, triggering automated emergency braking in following vehicles. Detection relies on cross-checking BSM data against physical plausibility (e.g., a vehicle cannot teleport or accelerate beyond physical limits) and comparing reported positions with radar/lidar sensor fusion data.",
                    "A Sybil attack in V2X occurs when a single malicious entity generates multiple fake vehicle identities (pseudonyms) simultaneously, broadcasting BSMs that appear to come from many distinct vehicles. This creates the illusion of traffic congestion, road hazards, or a higher vote count in cooperative perception schemes. For example, a single attacker could simulate 50 phantom vehicles to trigger congestion rerouting algorithms, diverting real traffic to benefit the attacker. The SCMS combats Sybil attacks through pseudonym certificate quotas (typically 20 active pseudonyms per vehicle) and linkage authorities that can identify when one enrollment certificate spawns excessive pseudonyms.",
                    "Ghost vehicle detection from forged BSMs uses multiple validation layers:\n1. **Kinematic Plausibility**: Check if reported speed/acceleration/heading changes are physically possible between consecutive BSMs (10 Hz). A vehicle claiming 0-to-60mph in 0.1 seconds is clearly forged.\n2. **Sensor Fusion Cross-Check**: Compare BSM-reported positions with data from onboard radar, lidar, and camera. If a BSM claims a vehicle at 50m ahead but no sensor detects it, flag as suspicious.\n3. **Redundancy Validation**: Check if the same vehicle is reported by BSMs from other nearby vehicles. A ghost vehicle visible to only one BSM source is suspect.\n4. **Map Consistency**: Verify the reported position is on a valid road segment, not in a building or lake.\n5. **Temporal Consistency**: Track BSM sequences for the same TemporaryID. Sudden appearance without approaching trajectory indicates injection.",
                    "The Misbehavior Authority (MA) must detect these V2X misbehavior types:\n1. **Position Falsification**: Incorrect GPS coordinates in BSMs\n2. **Speed/Heading Falsification**: Inconsistent kinematic data\n3. **Sybil Attack**: Single entity using multiple identities\n4. **Denial of Service**: Flooding the DSRC channel with excessive messages\n5. **Replay Attack**: Re-broadcasting old valid BSMs\n6. **Certificate Misuse**: Using revoked or expired pseudonym certificates\n7. **Congestion Attack**: False traffic reports to manipulate routing\n8. **Signal Spoofing**: Forged SPaT messages to manipulate traffic signals\n9. **Ghost Vehicle Injection**: BSMs from non-existent vehicles\n10. **Time Manipulation**: Incorrect timestamps to bypass freshness checks",
                ],
            },
            {
                "difficulty": "intermediate",
                "instructions": [
                    "Design a machine learning pipeline for real-time BSM misbehavior detection on an OBU.",
                    "How should the Misbehavior Authority aggregate and adjudicate reports from millions of vehicles?",
                    "Propose plausibility check thresholds for BSM kinematic validation.",
                ],
                "responses": [
                    "ML Pipeline for OBU Misbehavior Detection:\n\n**Stage 1 - Feature Extraction (per BSM, <1ms)**\nExtract 15 features from each received BSM: lat, lon, speed, heading, acceleration (5), plus delta values from previous BSM of same TemporaryID (position_delta, speed_delta, heading_delta, time_delta, acceleration_delta).\n\n**Stage 2 - Plausibility Filter (rule-based, <0.5ms)**\nApply hard thresholds: max speed 200 km/h, max acceleration 12 m/s^2, max position jump per 100ms = 5.6m at highway speed. Reject obvious violations immediately.\n\n**Stage 3 - Anomaly Scoring (lightweight NN, <2ms)**\nRun a quantized neural network (INT8, ~50KB model) that outputs a misbehavior probability [0,1]. Architecture: 15-input -> 32-unit ReLU -> 16-unit ReLU -> 1-sigmoid. Trained on labeled CAMP dataset with synthetic attacks.\n\n**Stage 4 - Temporal Tracking (Kalman filter, <1ms)**\nMaintain a Kalman filter per tracked TemporaryID. Compare predicted state with reported BSM state. Mahalanobis distance > 3 sigma triggers elevated suspicion.\n\n**Stage 5 - Decision and Report (<0.5ms)**\nIf anomaly_score > 0.7 for 3 consecutive BSMs, generate a Misbehavior Report containing the suspect BSMs, sensor fusion evidence, and confidence level. Transmit to Misbehavior Authority via RSU backhaul.",
                    "Misbehavior Authority Aggregation Architecture:\n\n**Ingestion Layer**: Receive 50M+ misbehavior reports/day via RSU backhaul. Kafka cluster with geo-partitioning (reports routed to regional MA nodes). Message deduplication by report_hash.\n\n**Correlation Engine**: Group reports by target_pseudonym_id and geographic region. Apply weighted voting: reports from multiple independent vehicles carry higher weight. Minimum 3 independent reporters required for action.\n\n**Adjudication Pipeline**:\n1. Automated Tier 1: ML model classifies clear-cut cases (>95% confidence). Handles 90% of volume. Actions: pseudonym revocation via CRL.\n2. Semi-Automated Tier 2: Ambiguous cases reviewed by rule engine with human-set thresholds. Handles 9% of volume. May request additional evidence.\n3. Manual Tier 3: Complex cases (Sybil patterns, coordinated attacks) reviewed by human analysts. Handles 1% of volume.\n\n**Linkage Authority Interaction**: For confirmed misbehavior, MA sends linkage_seed to Linkage Authority to resolve pseudonym to enrollment certificate, enabling revocation of all associated pseudonyms.\n\n**Privacy Preservation**: MA never directly sees vehicle identity. Two Linkage Authorities must collude to link pseudonym to enrollment (dual-authority privacy model).",
                    "BSM Kinematic Plausibility Thresholds:\n\n| Check | Threshold | Rationale |\n|-------|-----------|----------|\n| Max speed | 250 km/h (155 mph) | Covers sports cars + margin |\n| Max acceleration | 12 m/s^2 | Top production car ~1.2g |\n| Max deceleration | 15 m/s^2 | Emergency + downhill |\n| Max lateral accel | 12 m/s^2 | Hard cornering limit |\n| Position jump (100ms) | speed * 0.12 + 2m | Physics + GPS drift |\n| Heading change rate | 120 deg/sec | Maximum steering rate |\n| Elevation change (100ms) | 3m | Steep road + bounce |\n| Speed-heading consistency | cos(heading_delta) * speed_delta < 5 | Catches teleportation |\n| Off-road detection | Position within 15m of road centerline | Map matching tolerance |\n| Sudden appearance | No prior BSM within 300m radius | Ghost vehicle indicator |\n\nThresholds should be dynamically adjusted for road type (highway vs urban), weather conditions (slippery roads allow higher lateral accel due to slides), and vehicle type (trucks vs motorcycles). False positive rate target: <0.01% per vehicle per hour.",
                ],
            },
        ]

        for _ in range(count):
            template = random.choice(templates)
            instruction = random.choice(template["instructions"])
            response = random.choice(template["responses"])
            pairs.append({
                "pair_id": f"v2x_mbd_{uuid.uuid4().hex[:8]}",
                "category": "misbehavior_detection",
                "difficulty": template["difficulty"],
                "instruction": instruction,
                "context": "Domain: V2X misbehavior detection. Vehicles broadcast BSMs at 10 Hz. SCMS provides PKI. Misbehavior Authority adjudicates reports.",
                "response": response,
            })
        return pairs

    def _generate_certificate_management_pairs(self, count: int) -> List[Dict]:
        """Generate instruction pairs for V2X certificate management (SCMS)."""
        pairs = []
        templates = [
            {
                "difficulty": "basic",
                "instructions": [
                    "Explain the role of the Security Credential Management System (SCMS) in V2X.",
                    "What are pseudonym certificates and why are they needed in V2X?",
                    "Describe the butterfly key expansion mechanism used in SCMS.",
                    "How does pseudonym certificate rotation protect V2X privacy?",
                ],
                "responses": [
                    "The SCMS is the PKI backbone of V2X security. It provides: (1) Enrollment Certificates: issued once per vehicle at manufacturing, proving the vehicle is authorized to participate in V2X. (2) Pseudonym Certificates: short-lived certificates (typically valid for 1 week) used to sign BSMs without revealing the vehicle's true identity. (3) Certificate Revocation: distributes CRLs to revoke misbehaving vehicles. (4) Misbehavior Authority: investigates reports and triggers revocation. (5) Linkage Authorities: two separate entities that together can link pseudonyms to enrollment certificates for revocation, but neither alone can break privacy. The SCMS design ensures that even the SCMS operator cannot track individual vehicles under normal operation.",
                    "Pseudonym certificates are short-lived signing certificates used by vehicles to sign BSMs and other V2X messages without revealing their permanent identity. Each vehicle holds a pool of ~20 pseudonym certificates at a time, rotating to a new one periodically (every 5 minutes recommended). This prevents location tracking: an observer cannot link BSMs signed with different pseudonyms to the same vehicle. Without pseudonym rotation, an eavesdropper at two locations could track a vehicle's route by following its consistent signing identity. The SCMS pre-generates batches of pseudonym certificates (typically 3 years' supply) and provisions them to the OBU during manufacturing or service visits.",
                    "Butterfly key expansion is a bandwidth-efficient mechanism for generating large numbers of pseudonym certificates without transmitting each one individually. The process:\n1. The OBU generates a single caterpillar key pair (private seed, public seed).\n2. The OBU sends the public seed to the SCMS Registration Authority.\n3. The RA applies a deterministic key derivation function (cocoon key expansion) to generate thousands of public keys from the single seed.\n4. Each expanded public key becomes the public key for one pseudonym certificate.\n5. The OBU independently derives the corresponding private keys using the same expansion from its private seed.\n\nThis reduces provisioning bandwidth from O(n) to O(1), enabling 3-year certificate pre-generation in a single transaction. The butterfly key mechanism also supports separate expansion for ECDSA and PQC key pairs in hybrid deployments.",
                    "Pseudonym certificate rotation protects privacy through unlinkability: when a vehicle switches from pseudonym A to pseudonym B, an observer cannot determine that both belong to the same vehicle. The recommended rotation strategy:\n- **Temporal rotation**: Switch pseudonyms every 5 minutes during normal driving.\n- **Geographic rotation**: Switch when passing through areas with many vehicles (provides k-anonymity).\n- **Simultaneous rotation**: Multiple vehicles in proximity coordinate rotation at the same instant, preventing observers from linking old and new pseudonyms via timing correlation.\n- **Gap insertion**: Brief transmission pause (1-2 seconds) during rotation prevents linking via spatial continuity.\n\nThe SCMS provisions certificates with overlapping validity periods to ensure seamless rotation without gaps in V2X protection.",
                ],
            },
            {
                "difficulty": "advanced",
                "instructions": [
                    "Design a PQC-ready SCMS architecture that supports Falcon-512 pseudonym certificates alongside ECDSA.",
                    "How should CRL distribution scale to millions of vehicles with PQC-sized certificates?",
                ],
                "responses": [
                    "PQC-Ready SCMS Architecture:\n\n**Root CA Layer**: Dual root keys (ECDSA-P384 + ML-DSA-65). Root certificate contains both public keys. Cross-signed for trust chain validation via either algorithm.\n\n**Intermediate CA Layer**: Separate Enrollment CA and Pseudonym CA, each with dual key pairs. Enrollment certificates issued with hybrid signatures (both algorithms sign the certificate).\n\n**Pseudonym Certificate Generation**:\n- Butterfly key expansion extended for Falcon-512: caterpillar key is a Falcon-512 seed (1,281 bytes private). Expansion uses NTRU-based deterministic derivation to generate cocoon keys.\n- Each pseudonym certificate contains: ECDSA-P256 public key (33 bytes) + Falcon-512 public key (897 bytes) + dual CA signatures.\n- Certificate size: ~1,800 bytes (vs ~117 bytes ECDSA-only). 3-year supply for one vehicle: ~26 MB (vs ~1.7 MB).\n\n**Provisioning Optimization**:\n- Compressed butterfly seeds reduce OTA provisioning to ~5 KB regardless of certificate count.\n- Certificate reconstruction performed on-OBU from seeds + CA signature batch.\n\n**Revocation Adaptation**:\n- CRLs use hash-based linkage values (8 bytes each), unchanged by PQC migration.\n- CRL signatures switch to Falcon-512 for quantum resistance.\n- Delta CRLs distributed hourly, full CRLs weekly.",
                    "Scalable PQC CRL Distribution:\n\n**Challenge**: With 300M+ connected vehicles in the US, a full CRL could contain millions of entries. Falcon-512 CRL signatures are 666 bytes vs 64 bytes for ECDSA, and each CRL must be verified by every vehicle.\n\n**Tiered Distribution Architecture**:\n1. **Regional CRL Shards**: Partition CRL by geographic region (50 state-level shards). Each vehicle downloads only its region's shard + national critical revocations.\n2. **Delta CRL Compression**: Hourly delta CRLs contain only new revocations since last full CRL. Bloom filter encoding reduces delta size to ~1 bit per entry with 1% false positive rate.\n3. **Hash-Based CRL**: Replace explicit certificate listing with hash-chain linkage values. Each entry is 8 bytes (linkage seed hash), enabling 125,000 revocations per MB.\n4. **CDN Edge Caching**: CRL distributed via automotive CDN (existing OTA infrastructure). RSUs cache and rebroadcast CRL deltas over DSRC for vehicles without cellular.\n5. **Signature Amortization**: CRL split into pages of 1,000 entries each. Each page signed independently with Falcon-512. Vehicle verifies only downloaded pages, amortizing signature verification cost.\n6. **OCSP Stapling Alternative**: RSUs perform online certificate status checks and staple freshness proofs to their own broadcasts, reducing per-vehicle CRL download requirements.",
                ],
            },
        ]

        for _ in range(count):
            template = random.choice(templates)
            instruction = random.choice(template["instructions"])
            response = random.choice(template["responses"])
            pairs.append({
                "pair_id": f"v2x_cert_{uuid.uuid4().hex[:8]}",
                "category": "certificate_management",
                "difficulty": template["difficulty"],
                "instruction": instruction,
                "context": "Domain: V2X SCMS certificate management. US DOT SCMS architecture with pseudonym certificates, butterfly keys, and linkage authorities.",
                "response": response,
            })
        return pairs

    def _generate_group_signatures_pairs(self, count: int) -> List[Dict]:
        """Generate instruction pairs for V2X group signatures."""
        pairs = []
        templates = [
            {
                "difficulty": "basic",
                "instructions": [
                    "Explain how group signatures provide anonymous authentication in V2X.",
                    "What is the difference between group signatures and pseudonym certificates for V2X privacy?",
                    "Describe linkable ring signatures and their application to V2X.",
                ],
                "responses": [
                    "Group signatures in V2X allow any authorized vehicle to sign messages on behalf of the group (all registered vehicles) without revealing which specific vehicle produced the signature. A group manager (the SCMS) issues group signing keys to each vehicle. Any verifier can confirm that the signature was produced by a valid group member, but cannot determine which member. If misbehavior is detected, the group manager can open a specific signature to reveal the signer's identity for revocation. This provides stronger anonymity than pseudonym certificates because there is no pseudonym to track, even temporarily.",
                    "Key differences between group signatures and pseudonym certificates for V2X:\n\n**Pseudonym Certificates**: Vehicle signs with one of ~20 short-lived certificates. Privacy relies on rotating pseudonyms frequently. Within a rotation window (5 min), all BSMs from one vehicle are linkable. SCMS must pre-generate and distribute large certificate pools.\n\n**Group Signatures**: Vehicle signs with a single long-lived group key. Every signature is unlinkable, even consecutive BSMs. No certificate rotation needed. Signature size is larger (~200+ bytes vs ~64 bytes for ECDSA pseudonym). Verification is slower (~5ms vs ~0.5ms). Group manager can de-anonymize for revocation.\n\nPseudonym certificates are the current standard (IEEE 1609.2) due to lower computational cost. Group signatures are being researched for future V2X where privacy requirements increase, particularly for autonomous vehicles where extended tracking could reveal home/work patterns.",
                    "Linkable ring signatures allow a signer to produce signatures that are: (1) anonymous within a ring of possible signers, (2) linkable if the same signer produces two signatures with the same linkage tag. In V2X, this enables:\n\n- **Misbehavior Tracking**: If a vehicle sends contradictory BSMs (e.g., claiming to be in two places), the linkage tags on both signatures will match, proving they came from the same vehicle without revealing which vehicle.\n- **Sybil Resistance**: A vehicle attempting to simulate multiple identities would produce signatures with the same linkage tag, exposing the Sybil attack.\n- **Unlinkability Otherwise**: Two honest BSMs with different content produce different linkage tags, preventing tracking.\n\nThe linkage tag is computed as T = H(private_key, epoch). Within the same epoch, the same signer always produces the same tag. Across epochs, tags are unlinkable.",
                ],
            },
            {
                "difficulty": "advanced",
                "instructions": [
                    "Design a lattice-based group signature scheme suitable for real-time V2X BSM authentication.",
                ],
                "responses": [
                    "Lattice-Based Group Signature for V2X:\n\n**Construction** (based on GPV framework with V2X optimizations):\n\n1. **Setup**: Group manager generates lattice trapdoor (A, T_A) where A is m x n matrix over Z_q. Public key is A. Trapdoor T_A enables member key issuance.\n\n2. **Join**: Vehicle receives member key k_i = SamplePre(T_A, u_i) where u_i is the vehicle's enrollment hash. Member key is a short vector such that A * k_i = u_i mod q.\n\n3. **Sign(msg, k_i)**:\n   a. Generate commitment: c = A * r + e (where r, e are short random vectors)\n   b. Compute challenge: ch = H(msg, c)\n   c. Produce response: z = r + ch * k_i (with rejection sampling for zero-knowledge)\n   d. Linkage tag: T = H_epoch(k_i) for epoch-based linkability\n   e. Signature = (c, z, T)\n\n4. **Verify(msg, sigma, A)**:\n   Check that A * z = c + ch * u for some valid u in the group, and that z is sufficiently short (norm bound check).\n\n5. **Open(sigma, T_A)**: Group manager uses trapdoor to extract u_i from the signature, identifying the signer.\n\n**V2X Optimizations**:\n- Dimension n=512, q=12289 for 128-bit post-quantum security\n- Signature size: ~1.5 KB (compact lattice encoding)\n- Sign time: ~2ms on ARM Cortex-A72 (automotive grade)\n- Verify time: ~3ms (meets V2X 100ms budget with margin)\n- Batch verification: NTT-based multi-signature verification achieves 500 sigs/sec\n- Linkage tag adds only 32 bytes overhead\n\n**Limitation**: 3x slower verification than Falcon-512 pseudonym certificates. Suitable for urban (500 vehicle density) but may struggle at highway density (1000+ vehicles) without hardware acceleration.",
                ],
            },
        ]

        for _ in range(count):
            template = random.choice(templates)
            instruction = random.choice(template["instructions"])
            response = random.choice(template["responses"])
            pairs.append({
                "pair_id": f"v2x_gs_{uuid.uuid4().hex[:8]}",
                "category": "group_signatures",
                "difficulty": template["difficulty"],
                "instruction": instruction,
                "context": "Domain: V2X group signatures and anonymous authentication. Privacy-preserving alternatives to pseudonym certificates.",
                "response": response,
            })
        return pairs

    def _generate_batch_verification_pairs(self, count: int) -> List[Dict]:
        """Generate instruction pairs for V2X batch verification."""
        pairs = []
        templates = [
            {
                "difficulty": "basic",
                "instructions": [
                    "Why is batch verification essential for V2X and what throughput is required?",
                    "Explain how ECDSA batch verification works for V2X BSMs.",
                    "What happens when batch verification detects an invalid signature?",
                ],
                "responses": [
                    "Batch verification is essential because vehicles in dense traffic receive BSMs from hundreds of nearby vehicles simultaneously. Each vehicle broadcasts at 10 Hz, so in a 500-vehicle scenario, an OBU must verify 5,000 signatures per second. Individual ECDSA-P256 verification takes ~0.5ms, requiring 2.5 seconds of CPU time per second of received BSMs, which is infeasible. Batch verification amortizes the cost: verifying n signatures together takes approximately the time of 1-3 individual verifications, achieving 10,000+ verifications per second. The V2X requirement is to verify all received BSMs within 100ms end-to-end, meaning batch verification must handle peak loads of 10,000+ msgs/sec in highway merge scenarios.",
                    "ECDSA batch verification for V2X BSMs uses the Bellare-Neven scheme:\n\n1. Collect n BSMs with signatures (r_i, s_i) and public keys P_i.\n2. Generate random coefficients a_1, ..., a_n (small random scalars, typically 128 bits).\n3. Compute the aggregate equation: sum(a_i * s_i^{-1} * H(m_i) * G) + sum(a_i * s_i^{-1} * r_i * P_i) = sum(a_i * R_i).\n4. Use multi-scalar multiplication (Pippenger's algorithm) to evaluate the aggregate in O(n / log n) point operations instead of O(n).\n5. If the aggregate equation holds, all n signatures are valid with overwhelming probability.\n\nPerformance: Batch of 100 ECDSA-P256 signatures verified in ~5ms (50us per signature, vs 500us individually). This 10x speedup is critical for V2X real-time processing.",
                    "When batch verification fails (the aggregate equation does not hold), at least one signature in the batch is invalid, but the batch check does not identify which one. The standard recovery procedure is binary search:\n\n1. Split the batch into two halves.\n2. Batch-verify each half.\n3. Recurse into the failing half(s).\n4. Continue until the individual invalid signature(s) are isolated.\n\nFor a batch of 100 with 1 invalid signature, this requires ~14 verification operations (7 splits) instead of 100 individual checks. The total time is ~7ms vs ~50ms.\n\nOptimization for V2X: Maintain a trust score per TemporaryID. If a batch fails and binary search identifies a suspect, temporarily lower that ID's trust score. Future BSMs from low-trust IDs are verified individually before inclusion in batches, preventing one misbehaving vehicle from degrading batch verification performance across the network.",
                ],
            },
            {
                "difficulty": "advanced",
                "instructions": [
                    "Design a heterogeneous batch verification pipeline that handles mixed ECDSA and Falcon-512 signatures during PQC transition.",
                    "How should an OBU prioritize BSM verification when overloaded in dense traffic?",
                ],
                "responses": [
                    "Heterogeneous Batch Verification Pipeline:\n\n**Architecture**: Dual-lane pipeline with shared priority queue.\n\n**Lane 1 - ECDSA Batch Verifier**:\n- Collects BSMs signed with ECDSA-P256 into batches of 64.\n- Pippenger multi-scalar multiplication on ARM NEON SIMD.\n- Throughput: ~15,000 verifications/sec.\n- Latency: 4ms per batch of 64.\n\n**Lane 2 - Falcon-512 Parallel Verifier**:\n- Falcon-512 does not benefit from algebraic batching like ECDSA.\n- Instead, use task-parallel verification: 4 hardware threads each verify independently.\n- NTT-based verification with pre-computed public key tables.\n- Throughput: ~12,000 verifications/sec (3,000/thread x 4 threads).\n- Latency: 0.3ms per individual signature.\n\n**Hybrid BSMs (ECDSA + Falcon-512)**:\n- Verify ECDSA signature in Lane 1 batch for immediate acceptance.\n- Queue Falcon-512 signature for Lane 2 as background validation.\n- If Falcon-512 fails, revoke the BSM acceptance and issue misbehavior report.\n- This hybrid-verify approach achieves 10,000+ mixed msgs/sec with 95th percentile latency under 10ms.\n\n**Priority Queue**: BSMs ranked by safety criticality (emergency vehicles first, nearby vehicles second, distant vehicles third). During overload, distant vehicle BSMs are dropped before nearby ones.",
                    "OBU BSM Verification Priority Under Overload:\n\n**Priority Tiers** (processed in order):\n\n**Tier 0 - Safety Critical (always verified)**:\n- Emergency Vehicle Alerts (EVA)\n- BSMs from vehicles within 50m and closing\n- BSMs reporting hard braking (deceleration > 4 m/s^2)\n- Queue budget: 20% of verification capacity reserved\n\n**Tier 1 - Immediate Relevance (verified if capacity permits)**:\n- BSMs from vehicles within 150m in same lane or adjacent lanes\n- SPaT messages from approaching intersections\n- BSMs from vehicles with intersecting predicted trajectories\n- Queue budget: 40% of capacity\n\n**Tier 2 - Contextual (best-effort)**:\n- BSMs from vehicles within 300m\n- MAP data messages\n- BSMs contributing to cooperative perception\n- Queue budget: 30% of capacity\n\n**Tier 3 - Background (dropped first)**:\n- BSMs from vehicles beyond 300m\n- Redundant BSMs (same vehicle, <500ms since last verified)\n- Messages from stationary vehicles (parked cars)\n- Queue budget: 10% of capacity, dropped entirely under overload\n\n**Adaptive Threshold**: Monitor verification queue depth. When queue exceeds 200ms backlog, reduce Tier 3 to 0%, Tier 2 to 15%. When queue exceeds 500ms, reduce Tier 1 to 25% and apply probabilistic sampling to Tier 2 (verify every 3rd BSM). Never degrade Tier 0.\n\n**Performance Target**: Maintain 99.9% verification rate for Tier 0 and 95% for Tier 1 at 2,000 vehicles within radio range.",
                ],
            },
        ]

        for _ in range(count):
            template = random.choice(templates)
            instruction = random.choice(template["instructions"])
            response = random.choice(template["responses"])
            pairs.append({
                "pair_id": f"v2x_bv_{uuid.uuid4().hex[:8]}",
                "category": "batch_verification",
                "difficulty": template["difficulty"],
                "instruction": instruction,
                "context": "Domain: V2X batch signature verification. Dense traffic scenarios with 500-2000 vehicles. OBU processing constraints.",
                "response": response,
            })
        return pairs

    # ------------------------------------------------------------------
    # Dataset generation
    # ------------------------------------------------------------------

    def generate_dataset(self, num_samples: int, output_dir: str,
                         num_instruction_pairs: int = 200) -> Dict:
        """Generate a complete V2X dataset with protocol samples and instruction pairs."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Part A: Protocol sample generators with ratios
        generators = [
            ("bsm", self.generate_bsm, 0.40),
            ("spat", self.generate_spat, 0.20),
            ("map_data", self.generate_map_data, 0.15),
            ("psm", self.generate_psm, 0.15),
            ("eva", self.generate_eva, 0.10),
        ]

        dataset_metadata = {
            "protocol": "v2x_ieee1609_sae_j2735",
            "version": "IEEE 1609.2-2022 / SAE J2735-2020",
            "total_protocol_samples": num_samples,
            "total_instruction_pairs": num_instruction_pairs,
            "samples_by_type": {},
            "instruction_pairs_by_category": {},
            "generated_at": datetime.now().isoformat(),
        }

        # Generate protocol samples
        sample_idx = 0
        for msg_type, generator, ratio in generators:
            count = int(num_samples * ratio)
            dataset_metadata["samples_by_type"][msg_type] = count

            for i in range(count):
                message, metadata = generator()
                metadata["sample_index"] = sample_idx

                bin_path = output_path / f"v2x_{msg_type}_{sample_idx:06d}.bin"
                with open(bin_path, "wb") as f:
                    f.write(message)

                meta_path = output_path / f"v2x_{msg_type}_{sample_idx:06d}.json"
                with open(meta_path, "w") as f:
                    json.dump(metadata, f, indent=2, default=str)

                sample_idx += 1

        # Part B: Instruction pairs (JSONL)
        pairs_per_category = num_instruction_pairs // len(self.INSTRUCTION_CATEGORIES)
        all_pairs = []

        pair_generators = {
            "v2x_pqc_integration": self._generate_pqc_integration_pairs,
            "misbehavior_detection": self._generate_misbehavior_detection_pairs,
            "certificate_management": self._generate_certificate_management_pairs,
            "group_signatures": self._generate_group_signatures_pairs,
            "batch_verification": self._generate_batch_verification_pairs,
        }

        for category, gen_fn in pair_generators.items():
            pairs = gen_fn(pairs_per_category)
            dataset_metadata["instruction_pairs_by_category"][category] = len(pairs)
            all_pairs.extend(pairs)

        random.shuffle(all_pairs)

        jsonl_path = output_path / "v2x_instruction_pairs.jsonl"
        with open(jsonl_path, "w") as f:
            for pair in all_pairs:
                f.write(json.dumps(pair, default=str) + "\n")

        # Save dataset metadata
        with open(output_path / "dataset_metadata.json", "w") as f:
            json.dump(dataset_metadata, f, indent=2)

        return dataset_metadata


def main():
    """Generate V2X automotive dataset."""
    generator = V2XAutomotiveGenerator(seed=42)
    output_dir = Path(__file__).parent.parent / "protocols" / "v2x_automotive"

    print("Generating V2X Automotive dataset...")
    metadata = generator.generate_dataset(
        num_samples=1000,
        output_dir=str(output_dir),
        num_instruction_pairs=200,
    )

    print(f"Generated {metadata['total_protocol_samples']} protocol samples")
    print(f"Generated {metadata['total_instruction_pairs']} instruction pairs")
    print(f"Output directory: {output_dir}")
    print("\nProtocol samples by type:")
    for msg_type, count in metadata["samples_by_type"].items():
        print(f"  - {msg_type}: {count} samples")
    print("\nInstruction pairs by category:")
    for category, count in metadata["instruction_pairs_by_category"].items():
        print(f"  - {category}: {count} pairs")


if __name__ == "__main__":
    main()
