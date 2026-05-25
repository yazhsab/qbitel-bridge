"""
IEC 61850 Industrial Protocol Dataset Generator

Generates realistic IEC 61850 substation automation protocol samples
and industrial security instruction pairs for QBITEL ML training.

IEC 61850 Protocol Suite:
- GOOSE (Generic Object Oriented Substation Event): Ethernet multicast, EtherType 0x88B8
- SV (Sampled Values): Ethernet multicast, EtherType 0x88BA
- MMS (Manufacturing Message Specification): TCP/IP, ISO 9506
- R-GOOSE (Routable GOOSE): UDP encapsulation for WAN communication

Part A: 1000 protocol samples (binary + JSON metadata)
Part B: 200 industrial security instruction pairs (JSONL)
"""

import json
import math
import random
import struct
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib


class IEC61850IndustrialGenerator:
    """Generate realistic IEC 61850 protocol samples and industrial security instruction pairs."""

    # --- IEC 61850 Logical Node References ---
    LOGICAL_NODES = {
        "XCBR": {"desc": "Circuit Breaker", "data": ["Pos", "BlkOpn", "BlkCls", "CBOpCap"]},
        "XSWI": {"desc": "Disconnector/Switch", "data": ["Pos", "BlkOpn", "BlkCls"]},
        "MMXU": {"desc": "Measurement", "data": ["TotW", "TotVAr", "TotVA", "Hz", "PPV", "PhV", "A"]},
        "CSWI": {"desc": "Switch Controller", "data": ["Pos", "OpCntRs"]},
        "CILO": {"desc": "Interlocking", "data": ["EnaCls", "EnaOpn"]},
        "PDIS": {"desc": "Distance Protection", "data": ["Op", "Str"]},
        "PTOC": {"desc": "Overcurrent Protection", "data": ["Op", "Str", "TmASt"]},
        "PTOV": {"desc": "Overvoltage Protection", "data": ["Op", "Str"]},
        "MMTR": {"desc": "Metering", "data": ["TotWh", "TotVArh"]},
        "MHAI": {"desc": "Harmonics", "data": ["HA", "HPhV", "HW"]},
    }

    # --- Substation Configuration ---
    SUBSTATIONS = [
        "NORTH_METRO_220", "SOUTH_GRID_500", "EAST_INDUSTRIAL_110",
        "WEST_RENEWABLE_330", "CENTRAL_HUB_400", "RIVER_CROSSING_132",
        "MOUNTAIN_WIND_220", "COASTAL_SOLAR_110", "VALLEY_HYDRO_500",
        "AIRPORT_SUPPLY_66",
    ]

    BAY_NAMES = [
        "LINE_BAY_01", "LINE_BAY_02", "TRANSFORMER_BAY_01", "TRANSFORMER_BAY_02",
        "BUS_COUPLER_01", "BUS_SECTION_01", "CAPACITOR_BAY_01", "REACTOR_BAY_01",
        "FEEDER_BAY_01", "FEEDER_BAY_02", "FEEDER_BAY_03", "GENERATOR_BAY_01",
    ]

    IED_VENDORS = ["ABB_REL670", "SIEMENS_7SJ85", "GE_D60", "SEL_451", "ALSTOM_P14x",
                   "SCHNEIDER_P3U30", "NARI_PCS9882", "TOSHIBA_GRL200"]

    # --- GOOSE Timing ---
    GOOSE_RETRANSMIT_MS = [4, 4, 8, 8, 16, 16, 32, 64, 128, 256, 500, 1000]

    # --- Power System Constants ---
    VOLTAGE_LEVELS_KV = [66, 110, 132, 220, 330, 400, 500]
    NOMINAL_FREQ_HZ = [50.0, 60.0]

    # --- Instruction Pair Categories ---
    INSTRUCTION_CATEGORIES = [
        "scada_pqc",
        "tesla_broadcast",
        "safety_instrumented",
        "supply_chain",
        "network_segmentation",
    ]

    DIFFICULTIES = ["basic", "intermediate", "advanced"]

    def __init__(self, seed: Optional[int] = None):
        if seed:
            random.seed(seed)
        self.goose_st_num = random.randint(1, 100000)
        self.sv_smp_cnt = 0
        self.mms_invoke_id = random.randint(1, 65535)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _mac_bytes(self, prefix: List[int]) -> bytes:
        """Build a 6-byte MAC address from a prefix, filling remaining with random."""
        mac = list(prefix)
        while len(mac) < 6:
            mac.append(random.randint(0, 255))
        return bytes(mac)

    def _iec_timestamp(self, offset_sec: float = 0.0) -> Tuple[bytes, str]:
        """Generate IEC 61850 UTC timestamp (8 bytes) and ISO string."""
        now = datetime.utcnow() + timedelta(seconds=offset_sec)
        epoch = datetime(1970, 1, 1)
        total_sec = (now - epoch).total_seconds()
        sec_int = int(total_sec)
        frac = int((total_sec - sec_int) * (2 ** 24))
        quality = random.choice([0x00, 0x01, 0x04, 0x08])  # valid, leap, clockFailure, notSync
        ts_bytes = struct.pack(">IIB", sec_int, frac << 8, quality)[:8]
        return ts_bytes, now.isoformat() + "Z"

    def _random_ied_ref(self) -> str:
        """Generate a realistic IED reference like AA1J1Q01A1."""
        substation_code = random.choice(["AA1", "BB2", "CC3", "DD4"])
        bay_code = f"J{random.randint(1, 12)}"
        ln_class = random.choice(list(self.LOGICAL_NODES.keys()))
        instance = random.randint(1, 4)
        return f"{substation_code}{bay_code}Q{random.randint(0, 1):02d}{ln_class}{instance}"

    def _data_attribute_ref(self, ln_class: str, ln_inst: int = 1) -> str:
        """Build a fully qualified data attribute reference."""
        node_info = self.LOGICAL_NODES[ln_class]
        data_obj = random.choice(node_info["data"])
        if ln_class == "MMXU":
            # Measurement has deeper hierarchy
            phase = random.choice(["phsA", "phsB", "phsC"])
            return f"{ln_class}{ln_inst}$MX${data_obj}${phase}$cVal$mag$f"
        elif ln_class in ("XCBR", "XSWI", "CSWI"):
            return f"{ln_class}{ln_inst}$ST${data_obj}$stVal"
        elif ln_class in ("PDIS", "PTOC", "PTOV"):
            return f"{ln_class}{ln_inst}$ST${data_obj}$general"
        else:
            return f"{ln_class}{ln_inst}$ST${data_obj}"

    def _generate_quality_flags(self) -> int:
        """Generate IEC 61850 quality flags (2 bytes)."""
        validity = random.choices([0b00, 0b01, 0b10, 0b11], weights=[90, 5, 3, 2])[0]
        overflow = random.choices([0, 1], weights=[98, 2])[0]
        out_of_range = random.choices([0, 1], weights=[97, 3])[0]
        test = random.choices([0, 1], weights=[99, 1])[0]
        return (validity << 14) | (overflow << 11) | (out_of_range << 10) | (test << 2)

    # ------------------------------------------------------------------
    # Part A: Protocol Sample Generators
    # ------------------------------------------------------------------

    def generate_goose_sample(self) -> Tuple[bytes, Dict]:
        """Generate a GOOSE (Generic Object Oriented Substation Event) frame."""
        substation = random.choice(self.SUBSTATIONS)
        bay = random.choice(self.BAY_NAMES)
        ied = random.choice(self.IED_VENDORS)
        ln_class = random.choice(["XCBR", "XSWI", "CSWI", "PDIS", "PTOC"])

        # Ethernet header
        dst_mac = self._mac_bytes([0x01, 0x0C, 0xCD, 0x01])
        src_mac = self._mac_bytes([0x00, 0x30, random.randint(0, 255)])
        ether_type = struct.pack(">H", 0x88B8)

        # GOOSE PDU fields
        gocb_ref = f"{substation}/{bay}LLN0$GO$gcb{random.randint(1, 8):02d}"
        time_allowed = random.choice(self.GOOSE_RETRANSMIT_MS) * 2
        dat_set = f"{substation}/{bay}LLN0$ds{random.randint(1, 8):02d}"
        go_id = f"{bay}_{ln_class}_GOOSE"
        self.goose_st_num += 1
        st_num = self.goose_st_num
        sq_num = random.randint(0, 50)
        conf_rev = random.randint(1, 10)
        nds_com = 0  # needs commissioning = False
        ts_bytes, ts_iso = self._iec_timestamp()

        # Generate boolean status data points (typical protection/control dataset)
        num_entries = random.randint(3, 12)
        all_data = []
        data_bytes = b""
        for i in range(num_entries):
            if random.random() < 0.6:
                # Boolean status
                val = random.choice([True, False])
                data_bytes += struct.pack(">B", 0x83)  # ASN.1 boolean tag
                data_bytes += struct.pack(">B", 1)
                data_bytes += struct.pack(">B", 0xFF if val else 0x00)
                all_data.append({"type": "boolean", "value": val})
            elif random.random() < 0.7:
                # Quality (bitstring)
                q = self._generate_quality_flags()
                data_bytes += struct.pack(">B", 0x84)  # ASN.1 bitstring tag
                data_bytes += struct.pack(">B", 3)
                data_bytes += struct.pack(">BH", 0x04, q)
                all_data.append({"type": "quality", "value": q})
            else:
                # Timestamp
                ts_b, _ = self._iec_timestamp(offset_sec=random.uniform(-0.1, 0.0))
                data_bytes += struct.pack(">B", 0x89)  # ASN.1 UTC time tag
                data_bytes += struct.pack(">B", len(ts_b))
                data_bytes += ts_b
                all_data.append({"type": "timestamp", "value": ts_b.hex()})

        # Build GOOSE PDU (simplified ASN.1 BER encoding)
        gocb_bytes = gocb_ref.encode("ascii")
        dat_set_bytes = dat_set.encode("ascii")
        go_id_bytes = go_id.encode("ascii")

        goose_pdu = b""
        # gocbRef
        goose_pdu += struct.pack(">BB", 0x80, len(gocb_bytes)) + gocb_bytes
        # timeAllowedtoLive
        goose_pdu += struct.pack(">BB", 0x81, 2) + struct.pack(">H", time_allowed)
        # datSet
        goose_pdu += struct.pack(">BB", 0x82, len(dat_set_bytes)) + dat_set_bytes
        # goID
        goose_pdu += struct.pack(">BB", 0x83, len(go_id_bytes)) + go_id_bytes
        # t (timestamp)
        goose_pdu += struct.pack(">BB", 0x84, len(ts_bytes)) + ts_bytes
        # stNum
        goose_pdu += struct.pack(">BB", 0x85, 4) + struct.pack(">I", st_num)
        # sqNum
        goose_pdu += struct.pack(">BB", 0x86, 4) + struct.pack(">I", sq_num)
        # confRev
        goose_pdu += struct.pack(">BB", 0x87, 4) + struct.pack(">I", conf_rev)
        # ndsCom
        goose_pdu += struct.pack(">BB", 0x88, 1) + struct.pack(">B", nds_com)
        # numDatSetEntries
        goose_pdu += struct.pack(">BB", 0x89, 4) + struct.pack(">I", num_entries)
        # allData
        goose_pdu += struct.pack(">BB", 0xAB, len(data_bytes)) + data_bytes

        # Wrap in GOOSE APDU container
        apdu = struct.pack(">BH", 0x61, len(goose_pdu)) + goose_pdu

        # Ethernet frame: dst + src + ethertype + appid(2) + length(2) + reserved(4) + apdu
        app_id = struct.pack(">H", random.randint(0x0000, 0x3FFF))
        frame_payload = app_id + struct.pack(">H", len(apdu) + 8) + b"\x00" * 4 + apdu
        frame = dst_mac + src_mac + ether_type + frame_payload

        metadata = {
            "protocol": "iec61850_goose",
            "message_type": "goose",
            "substation": substation,
            "bay": bay,
            "ied": ied,
            "timestamp": ts_iso,
            "message_length": len(frame),
            "fields": [
                {"name": "dst_mac", "offset": 0, "length": 6, "value": dst_mac.hex()},
                {"name": "src_mac", "offset": 6, "length": 6, "value": src_mac.hex()},
                {"name": "ether_type", "offset": 12, "length": 2, "value": "88b8"},
                {"name": "app_id", "offset": 14, "length": 2, "value": app_id.hex()},
                {"name": "goose_length", "offset": 16, "length": 2},
                {"name": "gocb_ref", "value": gocb_ref},
                {"name": "time_allowed_to_live_ms", "value": time_allowed},
                {"name": "dat_set", "value": dat_set},
                {"name": "go_id", "value": go_id},
                {"name": "st_num", "value": st_num},
                {"name": "sq_num", "value": sq_num},
                {"name": "conf_rev", "value": conf_rev},
                {"name": "nds_com", "value": bool(nds_com)},
                {"name": "num_dat_set_entries", "value": num_entries},
            ],
            "all_data": all_data,
            "logical_node": ln_class,
            "hash": hashlib.sha256(frame).hexdigest(),
        }

        return frame, metadata

    def generate_sv_sample(self) -> Tuple[bytes, Dict]:
        """Generate a Sampled Values (SV) frame with 4 voltage + 4 current channels."""
        substation = random.choice(self.SUBSTATIONS)
        bay = random.choice(self.BAY_NAMES)
        voltage_kv = random.choice(self.VOLTAGE_LEVELS_KV)
        freq = random.choice(self.NOMINAL_FREQ_HZ)
        samples_per_cycle = 80
        smp_synch = random.choice([0, 1, 2])  # 0=none, 1=local, 2=global

        # Ethernet header
        dst_mac = self._mac_bytes([0x01, 0x0C, 0xCD, 0x04])
        src_mac = self._mac_bytes([0x00, 0x30, random.randint(0, 255)])
        ether_type = struct.pack(">H", 0x88BA)

        sv_id = f"{substation}_{bay}_MU01"
        conf_rev = random.randint(1, 5)
        self.sv_smp_cnt = (self.sv_smp_cnt + 1) % (samples_per_cycle * int(freq))

        # Generate 80 samples of 8 channels (4 voltage + 4 current)
        # Each sample: 4 bytes per channel = 32 bytes per sample
        v_peak = voltage_kv * 1000 * math.sqrt(2) / math.sqrt(3)  # Phase-to-neutral peak
        i_max = random.uniform(50, 5000)  # Current in amps
        freq_deviation = random.uniform(-0.1, 0.1)
        actual_freq = freq + freq_deviation

        seq_of_data = b""
        sample_values = []
        for s in range(samples_per_cycle):
            angle = 2 * math.pi * s / samples_per_cycle
            sample_point = {}

            # Voltages: phase A, B, C (120 degrees apart) + neutral
            for phase_idx, phase_name in enumerate(["vA", "vB", "vC"]):
                phase_offset = phase_idx * 2 * math.pi / 3
                noise = random.gauss(0, v_peak * 0.001)
                val = v_peak * math.sin(angle - phase_offset) + noise
                seq_of_data += struct.pack(">i", int(val))
                sample_point[phase_name] = round(val, 2)

            # Neutral voltage (near zero)
            v_neutral = random.gauss(0, v_peak * 0.01)
            seq_of_data += struct.pack(">i", int(v_neutral))
            sample_point["vN"] = round(v_neutral, 2)

            # Currents: phase A, B, C + neutral
            for phase_idx, phase_name in enumerate(["iA", "iB", "iC"]):
                phase_offset = phase_idx * 2 * math.pi / 3
                pf_angle = random.uniform(-0.5, 0.2)  # Power factor angle
                noise = random.gauss(0, i_max * 0.002)
                val = i_max * math.sin(angle - phase_offset - pf_angle) + noise
                seq_of_data += struct.pack(">i", int(val * 1000))  # mA resolution
                sample_point[phase_name] = round(val, 3)

            i_neutral = random.gauss(0, i_max * 0.02)
            seq_of_data += struct.pack(">i", int(i_neutral * 1000))
            sample_point["iN"] = round(i_neutral, 3)

            sample_values.append(sample_point)

        # Build SV PDU (simplified ASN.1 BER)
        sv_id_bytes = sv_id.encode("ascii")
        sv_pdu = b""
        # svID
        sv_pdu += struct.pack(">BB", 0x80, len(sv_id_bytes)) + sv_id_bytes
        # smpCnt
        sv_pdu += struct.pack(">BB", 0x82, 2) + struct.pack(">H", self.sv_smp_cnt)
        # confRev
        sv_pdu += struct.pack(">BB", 0x83, 4) + struct.pack(">I", conf_rev)
        # smpSynch
        sv_pdu += struct.pack(">BB", 0x85, 1) + struct.pack(">B", smp_synch)
        # seqOfData
        if len(seq_of_data) < 128:
            sv_pdu += struct.pack(">BB", 0x87, len(seq_of_data)) + seq_of_data
        else:
            length_bytes = len(seq_of_data).to_bytes(2, "big")
            sv_pdu += struct.pack(">B", 0x87) + b"\x82" + length_bytes + seq_of_data

        # ASDU wrapper
        asdu = struct.pack(">BH", 0x30, len(sv_pdu)) + sv_pdu if len(sv_pdu) > 127 else (
            struct.pack(">BB", 0x30, len(sv_pdu)) + sv_pdu
        )

        # savPDU wrapper
        noASDU = struct.pack(">BB", 0x80, 1) + struct.pack(">B", 1)
        seq_asdu_tag = struct.pack(">BB", 0xA2, len(asdu)) if len(asdu) < 128 else (
            struct.pack(">B", 0xA2) + b"\x82" + len(asdu).to_bytes(2, "big") + b""
        )
        sav_body = noASDU + seq_asdu_tag + asdu
        sav_pdu = struct.pack(">BH", 0x60, len(sav_body)) + sav_body

        # Ethernet frame
        app_id = struct.pack(">H", random.randint(0x4000, 0x7FFF))
        frame_payload = app_id + struct.pack(">H", len(sav_pdu) + 8) + b"\x00" * 4 + sav_pdu
        frame = dst_mac + src_mac + ether_type + frame_payload

        ts_bytes, ts_iso = self._iec_timestamp()

        metadata = {
            "protocol": "iec61850_sv",
            "message_type": "sampled_values",
            "substation": substation,
            "bay": bay,
            "timestamp": ts_iso,
            "message_length": len(frame),
            "fields": [
                {"name": "dst_mac", "offset": 0, "length": 6, "value": dst_mac.hex()},
                {"name": "src_mac", "offset": 6, "length": 6, "value": src_mac.hex()},
                {"name": "ether_type", "offset": 12, "length": 2, "value": "88ba"},
                {"name": "app_id", "offset": 14, "length": 2, "value": app_id.hex()},
                {"name": "sv_id", "value": sv_id},
                {"name": "smp_cnt", "value": self.sv_smp_cnt},
                {"name": "conf_rev", "value": conf_rev},
                {"name": "smp_synch", "value": smp_synch},
            ],
            "voltage_level_kv": voltage_kv,
            "nominal_frequency_hz": freq,
            "actual_frequency_hz": round(actual_freq, 3),
            "samples_per_cycle": samples_per_cycle,
            "channels": ["vA", "vB", "vC", "vN", "iA", "iB", "iC", "iN"],
            "peak_voltage_v": round(v_peak, 1),
            "max_current_a": round(i_max, 1),
            "sample_summary": {
                "first": sample_values[0] if sample_values else {},
                "count": len(sample_values),
            },
            "hash": hashlib.sha256(frame).hexdigest(),
        }

        return frame, metadata

    def generate_mms_sample(self) -> Tuple[bytes, Dict]:
        """Generate an MMS (Manufacturing Message Specification) message."""
        substation = random.choice(self.SUBSTATIONS)
        bay = random.choice(self.BAY_NAMES)
        ied = random.choice(self.IED_VENDORS)
        ln_class = random.choice(list(self.LOGICAL_NODES.keys()))
        ln_inst = random.randint(1, 4)
        data_ref = self._data_attribute_ref(ln_class, ln_inst)
        domain_name = f"{substation}_{bay}_{ied.split('_')[0]}"

        self.mms_invoke_id = (self.mms_invoke_id + 1) % 65536
        invoke_id = self.mms_invoke_id

        mms_op = random.choice(["read", "write", "read_response", "write_response",
                                 "information_report"])

        # Build simplified MMS PDU
        mms_pdu = b""
        mms_value = None
        mms_value_str = None

        if mms_op == "read":
            # ReadRequest: variableAccessSpecification
            ref_bytes = f"{domain_name}/{data_ref}".encode("ascii")
            # ObjectName
            obj_name = struct.pack(">BB", 0xA1, len(ref_bytes)) + ref_bytes
            # variableListName
            var_spec = struct.pack(">BB", 0xA1, len(obj_name)) + obj_name
            mms_pdu = struct.pack(">BB", 0xA4, len(var_spec)) + var_spec
            mms_tag = 0xA0  # confirmedRequestPDU

        elif mms_op == "write":
            ref_bytes = f"{domain_name}/{data_ref}".encode("ascii")
            if ln_class in ("XCBR", "XSWI", "CSWI"):
                mms_value = random.choice([True, False])
                mms_value_str = str(mms_value)
                val_data = struct.pack(">BBB", 0x83, 1, 0xFF if mms_value else 0x00)
            else:
                mms_value = round(random.uniform(0, 1000), 2)
                mms_value_str = str(mms_value)
                val_data = struct.pack(">BB", 0x87, 4) + struct.pack(">f", mms_value)
            obj_name = struct.pack(">BB", 0xA1, len(ref_bytes)) + ref_bytes
            mms_pdu = struct.pack(">BB", 0xA5, len(obj_name) + len(val_data)) + obj_name + val_data
            mms_tag = 0xA0

        elif mms_op == "read_response":
            if ln_class == "MMXU":
                voltage_kv = random.choice(self.VOLTAGE_LEVELS_KV)
                mms_value = round(voltage_kv * 1000 / math.sqrt(3) + random.gauss(0, 50), 2)
                mms_value_str = f"{mms_value}V"
            elif ln_class in ("XCBR", "XSWI"):
                mms_value = random.choice([0, 1, 2, 3])  # off, on, intermediate, bad
                mms_value_str = ["off", "on", "intermediate", "bad_state"][mms_value]
            else:
                mms_value = random.choice([True, False])
                mms_value_str = str(mms_value)
            val_bytes = struct.pack(">BB", 0x87, 4) + struct.pack(">f", float(mms_value) if isinstance(mms_value, (int, float)) else 0.0)
            mms_pdu = struct.pack(">BB", 0xA2, len(val_bytes)) + val_bytes
            mms_tag = 0xA1  # confirmedResponsePDU

        elif mms_op == "write_response":
            success = random.choices([True, False], weights=[95, 5])[0]
            result = struct.pack(">BB", 0x83, 1) + struct.pack(">B", 0x01 if success else 0x00)
            mms_pdu = struct.pack(">BB", 0xA5, len(result)) + result
            mms_value_str = "success" if success else "failure"
            mms_tag = 0xA1

        else:  # information_report
            ref_bytes = f"{domain_name}/{data_ref}".encode("ascii")
            mms_value = round(random.uniform(49.9, 50.1), 3) if "Hz" in data_ref else random.choice([True, False])
            mms_value_str = str(mms_value)
            val_bytes = struct.pack(">BB", 0x87, 4) + struct.pack(">f", float(mms_value) if isinstance(mms_value, (int, float)) else 0.0)
            report_body = struct.pack(">BB", 0xA1, len(ref_bytes)) + ref_bytes + val_bytes
            mms_pdu = struct.pack(">BB", 0xA3, len(report_body)) + report_body
            mms_tag = 0xA3  # unconfirmedPDU

        # Wrap in invoke-id + service
        invoke_bytes = struct.pack(">BB", 0x02, 2) + struct.pack(">H", invoke_id)
        full_pdu = invoke_bytes + mms_pdu
        mms_message = struct.pack(">BB", mms_tag, len(full_pdu)) + full_pdu

        # Simulate TCP/TPKT/COTP wrapping
        cotp = b"\x02\xF0\x80"  # COTP DT TPDU (3 bytes)
        tpkt_payload = cotp + mms_message
        tpkt = struct.pack(">BBH", 3, 0, len(tpkt_payload) + 4) + tpkt_payload

        ts_bytes, ts_iso = self._iec_timestamp()

        metadata = {
            "protocol": "iec61850_mms",
            "message_type": f"mms_{mms_op}",
            "substation": substation,
            "bay": bay,
            "ied": ied,
            "timestamp": ts_iso,
            "message_length": len(tpkt),
            "fields": [
                {"name": "tpkt_version", "offset": 0, "length": 1, "value": 3},
                {"name": "tpkt_length", "offset": 2, "length": 2, "value": len(tpkt)},
                {"name": "cotp", "offset": 4, "length": 3, "value": "02f080"},
                {"name": "mms_pdu_type", "value": hex(mms_tag)},
                {"name": "invoke_id", "value": invoke_id},
                {"name": "mms_operation", "value": mms_op},
                {"name": "domain_name", "value": domain_name},
                {"name": "data_reference", "value": data_ref},
                {"name": "logical_node", "value": ln_class},
            ],
            "mms_value": mms_value_str,
            "hash": hashlib.sha256(tpkt).hexdigest(),
        }

        return tpkt, metadata

    def generate_rgoose_sample(self) -> Tuple[bytes, Dict]:
        """Generate an R-GOOSE (Routable GOOSE over UDP/IP) message."""
        substation = random.choice(self.SUBSTATIONS)
        bay = random.choice(self.BAY_NAMES)
        ln_class = random.choice(["XCBR", "XSWI", "PTOC", "PDIS"])

        # First generate a standard GOOSE payload
        goose_frame, goose_meta = self.generate_goose_sample()
        # Extract the GOOSE PDU portion (skip Ethernet header 14 bytes + appid/len/reserved 8 bytes)
        goose_payload = goose_frame[22:] if len(goose_frame) > 22 else goose_frame

        # Session header (IEC 62351-6 / RFC adaptation)
        spdu_num = random.randint(1, 0xFFFFFFFF)
        session_version = 1
        security_info = random.choice([0x00, 0x01])  # 0x00=none, 0x01=signature present
        ts_bytes, ts_iso = self._iec_timestamp()

        # Build session/SPDU header
        session_hdr = struct.pack(">B", session_version)  # version
        session_hdr += struct.pack(">I", spdu_num)  # SPDU number
        session_hdr += struct.pack(">H", len(goose_payload))  # payload length
        session_hdr += struct.pack(">B", security_info)  # security information

        # If security enabled, add a simulated HMAC (32 bytes)
        if security_info == 0x01:
            hmac_sim = bytes([random.randint(0, 255) for _ in range(32)])
            session_hdr += hmac_sim

        # UDP payload: session header + GOOSE payload
        udp_payload = session_hdr + goose_payload

        # UDP header
        src_port = random.choice([102, 3001, 61850])
        dst_port = random.choice([102, 3001, 61850])
        udp_length = 8 + len(udp_payload)
        udp_header = struct.pack(">HHH", src_port, dst_port, udp_length) + b"\x00\x00"

        # IP header (simplified - 20 bytes, no options)
        src_ip = (10, random.randint(1, 254), random.randint(1, 254), random.randint(1, 254))
        dst_ip = (239, random.randint(0, 255), random.randint(0, 255), random.randint(1, 254))
        total_length = 20 + len(udp_header) + len(udp_payload)
        ip_id = random.randint(0, 65535)
        ttl = random.choice([64, 128])
        ip_header = struct.pack(">BBHHHBBH4s4s",
                                0x45, 0xC0,  # version/IHL, DSCP=CS6 (high priority)
                                total_length, ip_id,
                                0x4000,  # don't fragment
                                ttl, 17,  # TTL, protocol=UDP
                                0,  # checksum placeholder
                                bytes(src_ip), bytes(dst_ip))

        message = ip_header + udp_header + udp_payload

        metadata = {
            "protocol": "iec61850_rgoose",
            "message_type": "routable_goose",
            "substation": substation,
            "bay": bay,
            "timestamp": ts_iso,
            "message_length": len(message),
            "fields": [
                {"name": "ip_src", "value": ".".join(map(str, src_ip))},
                {"name": "ip_dst", "value": ".".join(map(str, dst_ip))},
                {"name": "udp_src_port", "value": src_port},
                {"name": "udp_dst_port", "value": dst_port},
                {"name": "spdu_number", "value": spdu_num},
                {"name": "session_version", "value": session_version},
                {"name": "security_info", "value": "hmac_present" if security_info else "none"},
            ],
            "encapsulated_goose": {
                "gocb_ref": goose_meta["fields"][5]["value"] if len(goose_meta["fields"]) > 5 else "",
                "st_num": goose_meta["fields"][9]["value"] if len(goose_meta["fields"]) > 9 else 0,
            },
            "hash": hashlib.sha256(message).hexdigest(),
        }

        return message, metadata

    # ------------------------------------------------------------------
    # Part B: Industrial Security Instruction Pair Generators
    # ------------------------------------------------------------------

    def _generate_scada_pqc_pairs(self, count: int) -> List[Dict]:
        """PQC for SCADA/ICS systems, IEC 62351 security, Modbus/DNP3 encryption."""
        instructions = [
            "Design a PQC key exchange protocol for SCADA master-to-RTU communication over DNP3.",
            "How should IEC 62351-6 GOOSE authentication be upgraded for post-quantum security?",
            "Evaluate ML-KEM-768 vs ML-KEM-1024 for constrained ICS devices on a Modbus network.",
            "Create a PQC migration roadmap for a 500-node SCADA system with mixed legacy devices.",
            "What IEC 62443 security levels are achievable with current PQC algorithms on ARM Cortex-M4?",
            "Design a hybrid classical/PQC certificate scheme for IEC 62351-3 TLS profiles.",
            "How can Modbus/TCP be secured with PQC without exceeding the 256-byte ADU limit?",
            "Propose a PQC-secured DNP3 Secure Authentication v6 extension.",
            "What are the latency implications of ML-DSA-65 signature verification on GOOSE retransmission?",
            "Design a key management lifecycle for PQC certificates in a multi-vendor substation.",
        ]
        contexts = [
            "Environment: 132kV substation, 24 IEDs (ABB, Siemens, GE mix), IEC 61850 Edition 2, GOOSE trip time requirement <3ms, existing PKI with RSA-2048.",
            "Environment: Oil pipeline SCADA, 200 RTUs over satellite links (600ms latency), DNP3-SA v5, bandwidth limit 9600 baud on some segments.",
            "Environment: Water treatment plant, 50 PLCs on Modbus/TCP, air-gapped OT network, firmware update via USB, no internet connectivity.",
            "Environment: Smart grid deployment, 10,000 smart meters, DLMS/COSEM protocol, NB-IoT backhaul, battery-powered endpoints.",
        ]
        responses = [
            "## PQC Key Exchange for DNP3 SCADA\n\n### Architecture\n1. **Algorithm Selection**: ML-KEM-768 for key encapsulation (1088-byte ciphertext fits DNP3 fragment)\n2. **Session Establishment**: Extend DNP3-SA challenge-response with KEM encapsulation\n3. **Key Derivation**: HKDF-SHA256 from shared secret -> AES-256-GCM session keys\n4. **Rotation**: Re-key every 3600 seconds or 1M messages\n\n### Constraints Addressed\n- RTU memory: ML-KEM-768 requires ~3KB RAM (fits ARM Cortex-M4 with 64KB)\n- Latency: KEM decapsulation ~1.2ms on M4 @ 168MHz (acceptable for non-trip paths)\n- Bandwidth: Key exchange adds 2 DNP3 fragments (within satellite budget)\n\n### Migration Path\n- Phase 1: Hybrid X25519 + ML-KEM-768 for key exchange\n- Phase 2: Pure ML-KEM-1024 after NIST final standardization\n- Phase 3: ML-DSA-65 for firmware signing and configuration authentication",
            "## IEC 62351-6 GOOSE PQC Authentication\n\n### Current State\nIEC 62351-6 uses RSA-2048 or ECDSA-256 for GOOSE HMAC key distribution.\n\n### PQC Upgrade Design\n1. **Signature Scheme**: ML-DSA-44 for GDOI key distribution messages\n2. **HMAC Retention**: Keep HMAC-SHA256 for per-message authentication (symmetric, quantum-safe)\n3. **Key Distribution**: Replace RSA-based GDOI with ML-KEM-768 key transport\n4. **Timing Impact**:\n   - GOOSE authentication overhead: 0.1ms (HMAC only, unchanged)\n   - Key refresh: ML-KEM decap adds 0.8ms (acceptable during non-trip intervals)\n   - Trip time budget: 3ms GOOSE + 0.1ms HMAC = 3.1ms (within 4ms requirement)\n\n### Implementation Notes\n- HMAC keys pre-distributed via PQC-secured GDOI session\n- Fallback to unsigned GOOSE if key unavailable (safety priority)\n- Key validity: 3600s with automatic renegotiation",
        ]
        pairs = []
        for _ in range(count):
            pairs.append({
                "pair_id": str(uuid.uuid4()),
                "category": "scada_pqc",
                "difficulty": random.choice(self.DIFFICULTIES),
                "instruction": random.choice(instructions),
                "context": random.choice(contexts),
                "response": random.choice(responses),
            })
        return pairs

    def _generate_tesla_broadcast_pairs(self, count: int) -> List[Dict]:
        """TESLA protocol for authenticated GOOSE multicast."""
        instructions = [
            "Design a TESLA-based authentication scheme for IEC 61850 GOOSE multicast messages.",
            "How does TESLA delayed key disclosure work for time-critical GOOSE trip signals?",
            "Compare TESLA vs HMAC-based authentication for sampled values at 4800 samples/sec.",
            "What clock synchronization accuracy is needed for TESLA in a substation with PTP?",
            "Design a TESLA key chain initialization protocol using PQC signatures.",
            "How should TESLA handle GOOSE retransmission timing (4ms, 8ms, 16ms schedule)?",
            "Evaluate TESLA overhead for R-GOOSE over WAN with 50ms one-way delay.",
            "What happens to TESLA security if the PTP grandmaster clock drifts by 1ms?",
            "Design a dual-TESLA scheme for redundant GOOSE publishers on parallel LANs.",
            "How can TESLA be combined with IEC 62351 to provide both authentication and confidentiality?",
        ]
        contexts = [
            "Substation: 220kV, 30 IEDs, PTP IEEE 1588 clock sync (<1us accuracy), GOOSE trip requirement <4ms, 50 GOOSE streams active.",
            "WAN scenario: 3 substations interconnected via MPLS, R-GOOSE for inter-trip protection, one-way delay 15-50ms, PTP over Ethernet.",
            "Legacy migration: Existing IEC 61850 Ed1 system, no authentication, upgrading to Ed2 with TESLA, mixed vendor IEDs.",
        ]
        responses = [
            "## TESLA Authentication for GOOSE Multicast\n\n### Design\n1. **Key Chain**: Pre-compute key chain K_n, K_{n-1}, ..., K_0 using SHA-256\n2. **Disclosure Delay**: d = 2 intervals (8ms at 4ms GOOSE rate)\n3. **GOOSE Message**: Append MAC(K_i, GOOSE_PDU) to each frame\n4. **Key Disclosure**: Disclose K_{i-d} in each new GOOSE message\n5. **Verification**: Receiver buffers message, verifies MAC when key disclosed\n\n### Timing Analysis\n- GOOSE publish interval: 4ms (first retransmission)\n- Key disclosure delay: 8ms (2 intervals)\n- Verification latency: 8ms from message receipt to authentication\n- Trip decision: Act on unauthenticated GOOSE for safety (verify post-hoc)\n\n### PQC Integration\n- Key chain commitment signed with ML-DSA-44 (distributed during GDOI setup)\n- Initial key chain bootstrap uses ML-KEM-768 key transport\n- Key chain length: 86400 keys (24 hours at 1 key/second)\n\n### Safety Consideration\nFor protection tripping: GOOSE accepted immediately, TESLA verification logged. False-trip risk mitigated by redundant protection schemes.",
            "## TESLA Key Chain Initialization with PQC\n\n### Protocol\n1. **Publisher** generates key chain: K_n = random, K_i = SHA-256(K_{i+1})\n2. **Publisher** signs commitment: SIG = ML-DSA-44.Sign(SK, K_0 || params)\n3. **Distribution** via GDOI secured with ML-KEM-768:\n   - K_0 (chain anchor)\n   - d (disclosure delay)\n   - T_0 (start time)\n   - interval (key change period)\n4. **Subscribers** verify ML-DSA-44 signature on commitment\n5. **Ongoing**: Each GOOSE carries MAC_i and K_{i-d}\n\n### Clock Requirements\n- PTP accuracy: <1us (IEEE 1588 profile for power)\n- TESLA tolerance: d * interval = 8ms buffer\n- Maximum clock error: 1ms (safe with 8ms buffer)\n- If clock drift >4ms: TESLA keys may be disclosed prematurely, reducing security window",
        ]
        pairs = []
        for _ in range(count):
            pairs.append({
                "pair_id": str(uuid.uuid4()),
                "category": "tesla_broadcast",
                "difficulty": random.choice(self.DIFFICULTIES),
                "instruction": random.choice(instructions),
                "context": random.choice(contexts),
                "response": random.choice(responses),
            })
        return pairs

    def _generate_safety_instrumented_pairs(self, count: int) -> List[Dict]:
        """VDF for safety timers, SIL compliance, emergency shutdown delays."""
        instructions = [
            "How can VDF (Verifiable Delay Functions) replace hardware safety timers in SIS?",
            "Design a PQC-secured emergency shutdown (ESD) sequence for a SIL-3 system.",
            "What are the IEC 61508 certification implications of using PQC in safety functions?",
            "Evaluate the proof-of-elapsed-time guarantees of VDF vs watchdog timers for SIL-2.",
            "Design a safety-critical firmware update protocol for SIS controllers using PQC signatures.",
            "How should PQC key management interact with SIS proof-test intervals?",
            "What is the maximum acceptable PQC signature verification time for a SIL-4 trip path?",
            "Design dual-channel PQC verification for 1oo2D safety architecture.",
            "How do common-cause failures in PQC libraries affect SIS safety integrity?",
            "Create a safety case argument for using ML-DSA in a nuclear safety system.",
        ]
        contexts = [
            "SIS: Triconex TMR controller, SIL-3 certified, ESD response time <500ms, proof-test interval 1 year, 20-year lifecycle.",
            "Safety system: HIMA HIMax, SIL-2 burner management, trip time <100ms, gas detection input to ESD output.",
            "Nuclear: Westinghouse AP1000 diverse protection, SIL-4, 1E-qualified hardware, 60-year design life.",
        ]
        responses = [
            "## VDF-Based Safety Timer for SIS\n\n### Concept\nReplace hardware watchdog timers with cryptographic Verifiable Delay Functions that provide mathematical proof of elapsed time.\n\n### Design\n1. **VDF Selection**: Wesolowski VDF (y = x^(2^T) mod N)\n2. **Time Calibration**: T chosen so evaluation takes exactly the safety delay period\n3. **Verification**: Third party verifies VDF output in O(log T) time\n4. **Safety Timer Flow**:\n   - ESD condition detected -> VDF evaluation starts\n   - VDF completes after calibrated delay -> ESD output activated\n   - Proof accompanies ESD command for audit trail\n\n### SIL Compliance\n- **Deterministic timing**: VDF provides mathematical bound on minimum elapsed time\n- **Tamper evidence**: Cannot shortcut VDF without breaking factoring assumption\n- **Diagnostic coverage**: VDF proof verifiable by independent channel\n- **Common-cause**: VDF + hardware watchdog as diverse redundancy\n\n### Limitations\n- VDF timing depends on CPU speed (requires calibration per hardware)\n- Not suitable for sub-millisecond safety timers (VDF overhead)\n- Requires careful implementation to avoid side-channel timing attacks",
            "## PQC-Secured Emergency Shutdown for SIL-3\n\n### Architecture\n1. **Detection**: Safety sensors -> SIS logic solver (traditional, no PQC in trip path)\n2. **Authentication**: PQC authenticates commands on non-safety communication paths:\n   - Operator ESD commands: ML-DSA-65 signed\n   - Remote maintenance: ML-KEM-768 + ML-DSA-65 mutual auth\n   - Configuration changes: Dual ML-DSA-65 signatures (4-eyes principle)\n3. **Trip Path**: Hardwired, no cryptographic delay in safety function\n4. **Audit Trail**: All safety actions logged with ML-DSA-44 timestamps\n\n### Key Management\n- SIS controller keys stored in tamper-resistant HSM\n- Key rotation aligned with proof-test interval (1 year)\n- Emergency key revocation via hardware switch (not network-dependent)\n\n### IEC 61508 Compliance\n- PQC NOT in the safety function execution path (no impact on PFD)\n- PQC used for defense-in-depth on communication channels\n- Systematic capability: SC 3 (requires rigorous development process for PQC library)",
        ]
        pairs = []
        for _ in range(count):
            pairs.append({
                "pair_id": str(uuid.uuid4()),
                "category": "safety_instrumented",
                "difficulty": random.choice(self.DIFFICULTIES),
                "instruction": random.choice(instructions),
                "context": random.choice(contexts),
                "response": random.choice(responses),
            })
        return pairs

    def _generate_supply_chain_pairs(self, count: int) -> List[Dict]:
        """Firmware signing with PQC, secure boot for PLCs/RTUs."""
        instructions = [
            "Design a PQC firmware signing pipeline for industrial PLCs from multiple vendors.",
            "How should secure boot chain-of-trust be established using ML-DSA for RTUs?",
            "Create a supply chain integrity verification protocol for substation IED firmware.",
            "What PQC algorithms are suitable for firmware signing on 32-bit ARM Cortex-M4 PLCs?",
            "Design a PQC-based SBOM (Software Bill of Materials) attestation for ICS devices.",
            "How can code signing with PQC protect against SolarWinds-style supply chain attacks on SCADA?",
            "Evaluate hash-based signatures (XMSS/LMS) vs lattice-based (ML-DSA) for ICS firmware signing.",
            "Design a rollback-resistant PQC firmware update mechanism for remote RTUs.",
            "How should vendor root-of-trust certificates be managed in a multi-vendor substation?",
            "Create a secure firmware distribution protocol for 10,000 smart grid devices.",
        ]
        contexts = [
            "Deployment: 500 PLCs (Siemens S7-1500, Allen-Bradley, Schneider M580), firmware updates quarterly, current signing with RSA-2048.",
            "Field devices: 200 RTUs on cellular backhaul, limited bandwidth (2G/3G), firmware size 2-8MB, update window: 4-hour maintenance.",
            "Substation: 30 IEDs from 4 vendors, each vendor has own signing key, central SCADA master coordinates updates.",
        ]
        responses = [
            "## PQC Firmware Signing Pipeline for Industrial PLCs\n\n### Architecture\n1. **Signing Infrastructure**:\n   - Offline HSM with ML-DSA-65 signing keys (one per vendor trust domain)\n   - Cross-signed with XMSS for defense-in-depth (stateful hash-based backup)\n   - Dual signature: ML-DSA-65 (primary) + ECDSA-P384 (transition compatibility)\n\n2. **Build Pipeline**:\n   - CI/CD builds firmware image\n   - SBOM generated with dependency hashes\n   - Firmware + SBOM signed: SIG = ML-DSA-65.Sign(vendor_SK, H(firmware || SBOM))\n   - Signature bundle: {firmware, SBOM, ML-DSA-sig, ECDSA-sig, certificate_chain}\n\n3. **Verification on PLC**:\n   - Boot ROM verifies first-stage bootloader (ML-DSA-65)\n   - Bootloader verifies firmware signature before flash write\n   - Verification time: ~12ms on Cortex-M4 @ 168MHz for ML-DSA-65\n   - Memory: ~120KB Flash, ~40KB RAM for verification code\n\n4. **Rollback Protection**:\n   - Monotonic version counter in OTP fuse\n   - Firmware must have version > current fuse value\n   - Anti-rollback prevents downgrade to vulnerable versions",
            "## Hash-Based vs Lattice-Based Signatures for ICS Firmware\n\n### Comparison\n| Metric | XMSS (hash-based) | ML-DSA-65 (lattice) |\n|--------|-------------------|---------------------|\n| Signature size | 2692 bytes | 3309 bytes |\n| Public key | 64 bytes | 1952 bytes |\n| Sign time (M4) | ~150ms | ~8ms |\n| Verify time (M4) | ~5ms | ~12ms |\n| Stateful | Yes (critical) | No |\n| NIST status | SP 800-208 | FIPS 204 |\n\n### Recommendation for ICS\n- **ML-DSA-65** for general firmware signing (stateless, simpler key management)\n- **XMSS** for root CA and long-lived trust anchors (smaller keys, conservative security)\n- **Combined**: XMSS root CA -> ML-DSA intermediate -> firmware signatures",
        ]
        pairs = []
        for _ in range(count):
            pairs.append({
                "pair_id": str(uuid.uuid4()),
                "category": "supply_chain",
                "difficulty": random.choice(self.DIFFICULTIES),
                "instruction": random.choice(instructions),
                "context": random.choice(contexts),
                "response": random.choice(responses),
            })
        return pairs

    def _generate_network_segmentation_pairs(self, count: int) -> List[Dict]:
        """Purdue model zones, PQC for inter-zone communication."""
        instructions = [
            "Design PQC-secured communication between Purdue Level 2 (control) and Level 3 (operations).",
            "How should IEC 62443 zones and conduits be secured with post-quantum cryptography?",
            "Create a PQC-based data diode authentication protocol for OT/IT boundary.",
            "Design a zero-trust architecture for IEC 61850 substations using PQC certificates.",
            "What PQC cipher suites should be configured on industrial firewalls at zone boundaries?",
            "How can PQC-secured jump hosts provide secure remote access to Level 1 devices?",
            "Design a PQC VPN overlay for inter-substation communication over public WAN.",
            "Evaluate the impact of PQC handshake size on industrial DMZ throughput.",
            "How should PQC certificates be distributed in an air-gapped Level 0/1 network?",
            "Create a micro-segmentation strategy for IEC 61850 GOOSE traffic with PQC authentication.",
        ]
        contexts = [
            "Architecture: 5-level Purdue model, Level 0-1 air-gapped, Level 2-3 connected via industrial firewall, Level 3-4 via DMZ. 100 devices total.",
            "Substation network: Station bus (MMS, 100Mbps), Process bus (GOOSE/SV, 1Gbps), Remote access via 4G, Engineering via VPN.",
            "Multi-site: 20 substations connected via MPLS WAN, central SCADA at control center, IEC 62351 deployed for MMS only.",
        ]
        responses = [
            "## PQC for Purdue Model Inter-Zone Communication\n\n### Zone Architecture with PQC\n\n**Level 0-1 (Process/Basic Control)**\n- Air-gapped, no external connectivity\n- PQC: ML-DSA-44 for local firmware verification only\n- Key provisioning: USB token during commissioning\n\n**Level 1-2 Conduit (Control Network)**\n- Protocol: IEC 61850 MMS/GOOSE\n- PQC: HMAC-SHA256 (symmetric, quantum-safe) for GOOSE\n- PQC: TLS 1.3 + ML-KEM-768 for MMS TCP connections\n- Firewall: Allow only IEC 61850 ports (102/TCP, 88B8/ETH, 88BA/ETH)\n\n**Level 2-3 Conduit (DMZ)**\n- PQC: TLS 1.3 with ML-KEM-768 + ML-DSA-65 cipher suite\n- Data diode: One-way from L2->L3 with ML-DSA-65 signed exports\n- Jump host: ML-KEM-768 + ML-DSA-65 mutual authentication\n- Session timeout: 15 minutes, continuous re-authentication\n\n**Level 3-4 Conduit (IT/OT Boundary)**\n- PQC VPN: WireGuard with ML-KEM-1024 key exchange\n- Certificate-based: X.509v3 with ML-DSA-65 signatures\n- Micro-segmentation: Per-application PQC TLS tunnels\n\n### Key Management\n- Offline root CA: XMSS-SHA256 (20-year key chain)\n- Intermediate CAs: ML-DSA-65 (per zone, 2-year validity)\n- Device certificates: ML-DSA-44 (per device, 1-year validity)\n- CRL distribution: Signed with ML-DSA-65, pushed to each zone firewall",
            "## PQC Micro-Segmentation for GOOSE Traffic\n\n### Design\n1. **VLAN-per-Bay**: Each bay's GOOSE traffic isolated in dedicated VLAN\n2. **Cross-Bay GOOSE**: Authenticated with HMAC-SHA256 (symmetric keys)\n3. **Key Distribution**: ML-KEM-768 pairwise key agreement between publishers and subscribers\n4. **Network Enforcement**:\n   - Switch ACLs permit only registered GOOSE multicast groups per port\n   - Unauthorized GOOSE frames dropped at ingress\n   - Rate limiting: Max 1000 GOOSE frames/sec per port\n5. **Monitoring**: SDN controller tracks GOOSE flow patterns, alerts on anomalies\n\n### Segmentation Zones\n- **Protection Zone**: GOOSE between protection IEDs (PDIS, PTOC) - highest priority, lowest latency\n- **Control Zone**: GOOSE for XCBR/XSWI control - medium priority\n- **Monitoring Zone**: MMS measurement data - standard priority\n- Each zone has independent PQC key domains (compromise isolation)",
        ]
        pairs = []
        for _ in range(count):
            pairs.append({
                "pair_id": str(uuid.uuid4()),
                "category": "network_segmentation",
                "difficulty": random.choice(self.DIFFICULTIES),
                "instruction": random.choice(instructions),
                "context": random.choice(contexts),
                "response": random.choice(responses),
            })
        return pairs

    # ------------------------------------------------------------------
    # Dataset generation
    # ------------------------------------------------------------------

    def generate_dataset(self, num_samples: int, output_dir: str) -> Dict:
        """Generate a complete IEC 61850 dataset with protocol samples and instruction pairs."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # --- Part A: Protocol Samples (binary + metadata) ---
        protocol_generators = [
            ("goose", self.generate_goose_sample, 0.30),
            ("sv", self.generate_sv_sample, 0.25),
            ("mms", self.generate_mms_sample, 0.30),
            ("rgoose", self.generate_rgoose_sample, 0.15),
        ]

        protocol_dir = output_path / "protocol_samples"
        protocol_dir.mkdir(parents=True, exist_ok=True)

        samples_by_type = {}
        sample_idx = 0

        for msg_type, generator, ratio in protocol_generators:
            count = int(num_samples * ratio)
            samples_by_type[msg_type] = count

            for i in range(count):
                message, metadata = generator()
                metadata["sample_index"] = sample_idx

                bin_path = protocol_dir / f"iec61850_{msg_type}_{sample_idx:06d}.bin"
                with open(bin_path, "wb") as f:
                    f.write(message)

                meta_path = protocol_dir / f"iec61850_{msg_type}_{sample_idx:06d}.json"
                with open(meta_path, "w") as f:
                    json.dump(metadata, f, indent=2, default=str)

                sample_idx += 1

        # Protocol metadata
        protocol_metadata = {
            "protocol": "iec61850",
            "version": "Edition 2.1",
            "total_samples": sample_idx,
            "samples_by_type": samples_by_type,
            "generated_at": datetime.now().isoformat(),
        }
        with open(protocol_dir / "dataset_metadata.json", "w") as f:
            json.dump(protocol_metadata, f, indent=2)

        # --- Part B: Instruction Pairs (JSONL) ---
        instruction_count = max(num_samples // 5, 200)
        pairs_per_category = instruction_count // len(self.INSTRUCTION_CATEGORIES)
        remainder = instruction_count % len(self.INSTRUCTION_CATEGORIES)

        category_generators = {
            "scada_pqc": self._generate_scada_pqc_pairs,
            "tesla_broadcast": self._generate_tesla_broadcast_pairs,
            "safety_instrumented": self._generate_safety_instrumented_pairs,
            "supply_chain": self._generate_supply_chain_pairs,
            "network_segmentation": self._generate_network_segmentation_pairs,
        }

        all_pairs = []
        pairs_by_category = {}

        for idx, category in enumerate(self.INSTRUCTION_CATEGORIES):
            cat_count = pairs_per_category + (1 if idx < remainder else 0)
            gen_fn = category_generators[category]
            pairs = gen_fn(cat_count)
            all_pairs.extend(pairs)
            pairs_by_category[category] = cat_count

        random.shuffle(all_pairs)

        instruction_dir = output_path / "instruction_pairs"
        instruction_dir.mkdir(parents=True, exist_ok=True)

        with open(instruction_dir / "industrial_security_pairs.jsonl", "w") as f:
            for pair in all_pairs:
                f.write(json.dumps(pair, default=str) + "\n")

        instruction_metadata = {
            "type": "industrial_security_instruction_pairs",
            "total_pairs": len(all_pairs),
            "pairs_by_category": pairs_by_category,
            "difficulties": {
                d: sum(1 for p in all_pairs if p["difficulty"] == d)
                for d in self.DIFFICULTIES
            },
            "generated_at": datetime.now().isoformat(),
        }
        with open(instruction_dir / "dataset_metadata.json", "w") as f:
            json.dump(instruction_metadata, f, indent=2)

        # --- Combined metadata ---
        dataset_metadata = {
            "protocol": "iec61850_industrial",
            "version": "1.0",
            "total_samples": sample_idx + len(all_pairs),
            "protocol_samples": sample_idx,
            "instruction_pairs": len(all_pairs),
            "samples_by_type": samples_by_type,
            "pairs_by_category": pairs_by_category,
            "generated_at": datetime.now().isoformat(),
        }
        with open(output_path / "dataset_metadata.json", "w") as f:
            json.dump(dataset_metadata, f, indent=2)

        return dataset_metadata


def main():
    """Generate IEC 61850 industrial dataset."""
    generator = IEC61850IndustrialGenerator(seed=42)
    output_dir = Path(__file__).parent.parent / "protocols" / "iec61850"

    print("Generating IEC 61850 Industrial dataset...")
    metadata = generator.generate_dataset(num_samples=1000, output_dir=str(output_dir))

    print(f"Generated {metadata['protocol_samples']} protocol samples + {metadata['instruction_pairs']} instruction pairs")
    print(f"Output directory: {output_dir}")
    print("\nProtocol samples by type:")
    for msg_type, count in metadata["samples_by_type"].items():
        print(f"  - {msg_type}: {count} samples")
    print("\nInstruction pairs by category:")
    for category, count in metadata["pairs_by_category"].items():
        print(f"  - {category}: {count} pairs")


if __name__ == "__main__":
    main()
