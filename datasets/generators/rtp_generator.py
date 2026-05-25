"""
RTP/SRTP Protocol Packet Generator

Generates realistic RTP and SRTP binary packets for training the protocol
discovery and field detection models.

RTP Header Structure (RFC 3550):
- Version (2 bits): Always 2
- Padding (1 bit)
- Extension (1 bit)
- CSRC Count (4 bits)
- Marker (1 bit)
- Payload Type (7 bits)
- Sequence Number (16 bits)
- Timestamp (32 bits)
- SSRC (32 bits)
"""

import json
import random
import struct
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple


class RTPGenerator:
    """Generate realistic RTP/SRTP packets for ML training."""

    # Payload Type definitions
    PAYLOAD_TYPES = {
        0: {"name": "PCMU", "clock_rate": 8000, "frame_size": 160},
        8: {"name": "PCMA", "clock_rate": 8000, "frame_size": 160},
        9: {"name": "G722", "clock_rate": 8000, "frame_size": 320},
        18: {"name": "G729", "clock_rate": 8000, "frame_size": 20},
        96: {"name": "dynamic-video", "clock_rate": 90000, "frame_size": None},
        101: {"name": "telephone-event", "clock_rate": 8000, "frame_size": 4},
    }

    # DTMF event codes (RFC 4733)
    DTMF_EVENTS = {
        0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
        5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
        10: "*", 11: "#", 12: "A", 13: "B", 14: "C", 15: "D",
    }

    AUDIO_PAYLOAD_TYPES = [0, 8, 9, 18]

    def __init__(self, seed: Optional[int] = None):
        if seed:
            random.seed(seed)
        self.seq_counter = random.randint(0, 65535)
        self.timestamp_counter = random.randint(0, 2**31)
        self.ssrc_pool = [random.randint(0, 2**32 - 1) for _ in range(50)]

    def _next_seq(self) -> int:
        """Get next sequence number (wraps at 65535)."""
        self.seq_counter = (self.seq_counter + 1) & 0xFFFF
        return self.seq_counter

    def _next_timestamp(self, increment: int = 160) -> int:
        """Get next timestamp with increment."""
        self.timestamp_counter = (self.timestamp_counter + increment) & 0xFFFFFFFF
        return self.timestamp_counter

    def _build_rtp_header(
        self, payload_type: int, marker: bool = False, ssrc: Optional[int] = None
    ) -> Tuple[bytes, Dict]:
        """Build a 12-byte RTP header."""
        seq = self._next_seq()
        pt_info = self.PAYLOAD_TYPES.get(payload_type, {})
        frame_size = pt_info.get("frame_size", 160) or 160
        ts = self._next_timestamp(frame_size)
        if ssrc is None:
            ssrc = random.choice(self.ssrc_pool)

        # Byte 0: V=2, P=0, X=0, CC=0 => 0x80
        byte0 = 0x80
        # Byte 1: M bit | PT
        byte1 = (0x80 if marker else 0x00) | (payload_type & 0x7F)

        header = struct.pack(">BBHII", byte0, byte1, seq, ts, ssrc)

        metadata = {
            "version": 2,
            "padding": False,
            "extension": False,
            "csrc_count": 0,
            "marker_bit": marker,
            "payload_type": payload_type,
            "payload_type_name": pt_info.get("name", "unknown"),
            "sequence_number_value": seq,
            "timestamp_value": ts,
            "ssrc_value": f"0x{ssrc:08X}",
        }

        fields = [
            {"name": "rtp_header", "offset": 0, "length": 12},
            {"name": "version_flags", "offset": 0, "length": 1},
            {"name": "payload_type_field", "offset": 1, "length": 1},
            {"name": "sequence_number", "offset": 2, "length": 2},
            {"name": "timestamp_field", "offset": 4, "length": 4},
            {"name": "ssrc", "offset": 8, "length": 4},
        ]

        return header, metadata, fields

    def generate_rtp_audio(self) -> Tuple[bytes, Dict]:
        """Generate a standard RTP audio packet."""
        pt = random.choice(self.AUDIO_PAYLOAD_TYPES)
        pt_info = self.PAYLOAD_TYPES[pt]
        payload_size = pt_info["frame_size"]

        marker = random.random() < 0.05  # 5% chance of marker bit

        header, meta, fields = self._build_rtp_header(pt, marker=marker)
        payload = bytes([random.randint(0, 255) for _ in range(payload_size)])

        fields.append({"name": "audio_payload", "offset": 12, "length": payload_size})

        message = header + payload
        meta.update({
            "codec": pt_info["name"],
            "payload_size": payload_size,
            "clock_rate": pt_info["clock_rate"],
        })

        return message, self._finalize_metadata("rtp_audio", message, meta, fields)

    def generate_srtp_audio(self) -> Tuple[bytes, Dict]:
        """Generate an SRTP audio packet with authentication tag."""
        pt = random.choice(self.AUDIO_PAYLOAD_TYPES)
        pt_info = self.PAYLOAD_TYPES[pt]
        payload_size = pt_info["frame_size"]
        auth_tag_len = 10  # HMAC-SHA1-80

        marker = random.random() < 0.05

        header, meta, fields = self._build_rtp_header(pt, marker=marker)
        payload = bytes([random.randint(0, 255) for _ in range(payload_size)])
        auth_tag = bytes([random.randint(0, 255) for _ in range(auth_tag_len)])

        fields.append({"name": "encrypted_payload", "offset": 12, "length": payload_size})
        fields.append({"name": "srtp_auth_tag", "offset": 12 + payload_size, "length": auth_tag_len})

        message = header + payload + auth_tag
        meta.update({
            "codec": pt_info["name"],
            "payload_size": payload_size,
            "clock_rate": pt_info["clock_rate"],
            "srtp_auth_tag_length": auth_tag_len,
            "crypto_suite": random.choice([
                "AES_CM_128_HMAC_SHA1_80",
                "AES_CM_256_HMAC_SHA1_80",
                "AEAD_AES_128_GCM",
            ]),
            "is_encrypted": True,
        })

        return message, self._finalize_metadata("srtp_audio", message, meta, fields)

    def generate_dtmf_event(self) -> Tuple[bytes, Dict]:
        """Generate an RFC 4733 DTMF event packet."""
        pt = 101  # telephone-event
        event_code = random.randint(0, 15)
        end_bit = random.choice([0, 1])
        volume = 10
        duration = random.randint(160, 2560)

        marker = (end_bit == 0 and random.random() < 0.3)

        header, meta, fields = self._build_rtp_header(pt, marker=marker)

        # RFC 4733 event payload: 4 bytes
        # Byte 0: event code
        # Byte 1: E(1) | R(1) | volume(6)
        # Bytes 2-3: duration (uint16)
        byte1 = ((end_bit & 0x01) << 7) | (volume & 0x3F)
        payload = struct.pack(">BBH", event_code, byte1, duration)

        fields.append({"name": "dtmf_payload", "offset": 12, "length": 4})

        message = header + payload
        meta.update({
            "dtmf_event_code": event_code,
            "dtmf_digit": self.DTMF_EVENTS.get(event_code, "?"),
            "dtmf_end_bit": bool(end_bit),
            "dtmf_volume": volume,
            "dtmf_duration": duration,
        })

        return message, self._finalize_metadata("dtmf_event", message, meta, fields)

    def generate_rtp_video(self) -> Tuple[bytes, Dict]:
        """Generate an RTP video packet."""
        pt = 96  # Dynamic payload type for video
        payload_size = random.randint(500, 1400)
        is_keyframe = random.random() < 0.1  # 10% keyframes

        header, meta, fields = self._build_rtp_header(pt, marker=is_keyframe)
        payload = bytes([random.randint(0, 255) for _ in range(payload_size)])

        fields.append({"name": "video_payload", "offset": 12, "length": payload_size})

        message = header + payload
        meta.update({
            "codec": "H264",
            "payload_size": payload_size,
            "is_keyframe": is_keyframe,
            "clock_rate": 90000,
        })

        return message, self._finalize_metadata("rtp_video", message, meta, fields)

    def _finalize_metadata(
        self, msg_type: str, message: bytes, meta: Dict, fields: list
    ) -> Dict:
        """Create the final metadata dict."""
        meta.update({
            "protocol": "rtp",
            "message_type": msg_type,
            "timestamp": datetime.now().isoformat(),
            "message_length": len(message),
            "fields": fields,
            "hash": hashlib.sha256(message).hexdigest(),
        })
        return meta

    def generate_dataset(self, num_samples: int, output_dir: str) -> Dict:
        """Generate a complete dataset of RTP packets."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        generators = [
            ("rtp_audio", self.generate_rtp_audio, 0.40),
            ("srtp_audio", self.generate_srtp_audio, 0.25),
            ("dtmf_event", self.generate_dtmf_event, 0.20),
            ("rtp_video", self.generate_rtp_video, 0.15),
        ]

        dataset_metadata = {
            "protocol": "rtp",
            "version": "RFC 3550/3711/4733",
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

                bin_path = output_path / f"rtp_{msg_type}_{sample_idx:06d}.bin"
                with open(bin_path, "wb") as f:
                    f.write(message)

                meta_path = output_path / f"rtp_{msg_type}_{sample_idx:06d}.json"
                with open(meta_path, "w") as f:
                    json.dump(metadata, f, indent=2, default=str)

                sample_idx += 1

        with open(output_path / "dataset_metadata.json", "w") as f:
            json.dump(dataset_metadata, f, indent=2)

        return dataset_metadata


def main():
    """Generate RTP dataset."""
    generator = RTPGenerator(seed=42)
    output_dir = Path(__file__).parent.parent / "protocols" / "rtp"

    print("Generating RTP/SRTP dataset...")
    metadata = generator.generate_dataset(num_samples=1000, output_dir=str(output_dir))

    print(f"Generated {metadata['total_samples']} samples")
    print(f"Output directory: {output_dir}")
    for msg_type, count in metadata["samples_by_type"].items():
        print(f"  - {msg_type}: {count} samples")


if __name__ == "__main__":
    main()
