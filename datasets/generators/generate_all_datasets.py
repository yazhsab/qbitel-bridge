#!/usr/bin/env python3
"""
Master Dataset Generator

Generates all datasets required for CRONOS AI ML training:
1. Protocol samples (ISO-8583, Modbus, HL7)
2. Field detection labeled data
3. Threat intelligence data
4. Security event logs
5. Anomaly detection data
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import random
import hashlib

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from iso8583_generator import ISO8583Generator
from modbus_generator import ModbusGenerator
from hl7_generator import HL7Generator


def generate_field_detection_dataset(protocol_dir: Path, output_dir: Path, split_ratio: tuple = (0.8, 0.1, 0.1)):
    """
    Convert protocol samples with metadata into field detection training format.

    Format: IOB tagging (Inside-Outside-Begin) for each byte
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []

    # Process all protocol directories
    for protocol_path in protocol_dir.iterdir():
        if not protocol_path.is_dir():
            continue

        print(f"  Processing {protocol_path.name}...")

        # Load all samples with metadata
        for meta_file in protocol_path.glob("*.json"):
            if meta_file.name == "dataset_metadata.json":
                continue

            bin_file = meta_file.with_suffix(".bin")
            if not bin_file.exists():
                continue

            with open(meta_file) as f:
                metadata = json.load(f)

            with open(bin_file, "rb") as f:
                message = f.read()

            # Convert to IOB format
            sample = convert_to_iob_format(message, metadata, protocol_path.name)
            if sample:
                all_samples.append(sample)

    # Shuffle and split
    random.shuffle(all_samples)

    n_total = len(all_samples)
    n_train = int(n_total * split_ratio[0])
    n_val = int(n_total * split_ratio[1])

    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:n_train + n_val]
    test_samples = all_samples[n_train + n_val:]

    # Save splits
    for split_name, samples in [("training", train_samples), ("validation", val_samples), ("test", test_samples)]:
        split_dir = output_dir / split_name
        split_dir.mkdir(exist_ok=True)

        for i, sample in enumerate(samples):
            with open(split_dir / f"sample_{i:06d}.json", "w") as f:
                json.dump(sample, f, indent=2)

    # Save schema
    schema = {
        "format": "IOB",
        "tags": ["O", "B-MTI", "I-MTI", "B-BITMAP", "I-BITMAP", "B-FIELD", "I-FIELD",
                 "B-LENGTH", "I-LENGTH", "B-HEADER", "I-HEADER", "B-DATA", "I-DATA"],
        "total_samples": n_total,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "protocols": list(set(s["protocol"] for s in all_samples)),
        "generated_at": datetime.now().isoformat()
    }

    with open(output_dir / "schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    return schema


def convert_to_iob_format(message: bytes, metadata: dict, protocol: str) -> dict:
    """Convert a message with field metadata to IOB format."""
    n_bytes = len(message)
    tags = ["O"] * n_bytes

    fields_info = metadata.get("fields", [])

    for field in fields_info:
        offset = field.get("offset")
        length = field.get("length")
        name = field.get("name", "unknown")

        if offset is None or length is None:
            continue
        if offset < 0 or offset >= n_bytes:
            continue

        end = min(offset + length, n_bytes)

        # Determine tag based on field name
        if "mti" in name.lower() or "message_type" in name.lower():
            tag_prefix = "MTI"
        elif "bitmap" in name.lower():
            tag_prefix = "BITMAP"
        elif "length" in name.lower():
            tag_prefix = "LENGTH"
        elif "header" in name.lower():
            tag_prefix = "HEADER"
        else:
            tag_prefix = "FIELD"

        # Apply IOB tags
        if offset < n_bytes:
            tags[offset] = f"B-{tag_prefix}"
            for i in range(offset + 1, end):
                if i < n_bytes:
                    tags[i] = f"I-{tag_prefix}"

    return {
        "protocol": protocol,
        "message_hex": message.hex(),
        "message_bytes": list(message),
        "tags": tags,
        "length": n_bytes,
        "fields": fields_info,
        "hash": hashlib.sha256(message).hexdigest()
    }


def generate_threat_intelligence_dataset(output_dir: Path, num_samples: int = 1000):
    """Generate synthetic threat intelligence dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # MITRE ATT&CK techniques (subset)
    mitre_techniques = [
        {"technique_id": "T1566.001", "name": "Spearphishing Attachment", "tactics": ["initial-access"], "description": "Adversaries may send spearphishing emails with a malicious attachment in an attempt to gain access to victim systems."},
        {"technique_id": "T1059.001", "name": "PowerShell", "tactics": ["execution"], "description": "Adversaries may abuse PowerShell commands and scripts for execution."},
        {"technique_id": "T1059.003", "name": "Windows Command Shell", "tactics": ["execution"], "description": "Adversaries may abuse the Windows command shell for execution."},
        {"technique_id": "T1053.005", "name": "Scheduled Task", "tactics": ["execution", "persistence", "privilege-escalation"], "description": "Adversaries may abuse the Windows Task Scheduler to perform task scheduling for initial or recurring execution of malicious code."},
        {"technique_id": "T1547.001", "name": "Registry Run Keys", "tactics": ["persistence", "privilege-escalation"], "description": "Adversaries may achieve persistence by adding a program to a Registry run key."},
        {"technique_id": "T1055.001", "name": "DLL Injection", "tactics": ["defense-evasion", "privilege-escalation"], "description": "Adversaries may inject dynamic-link libraries into processes in order to evade process-based defenses."},
        {"technique_id": "T1055.012", "name": "Process Hollowing", "tactics": ["defense-evasion", "privilege-escalation"], "description": "Adversaries may inject malicious code into suspended and hollowed processes."},
        {"technique_id": "T1003.001", "name": "LSASS Memory", "tactics": ["credential-access"], "description": "Adversaries may attempt to access credential material stored in the process memory of LSASS."},
        {"technique_id": "T1071.001", "name": "Web Protocols", "tactics": ["command-and-control"], "description": "Adversaries may communicate using application layer protocols associated with web traffic."},
        {"technique_id": "T1041", "name": "Exfiltration Over C2 Channel", "tactics": ["exfiltration"], "description": "Adversaries may steal data by exfiltrating it over an existing C2 channel."},
    ]

    # Save MITRE techniques
    mitre_dir = output_dir / "mitre_attack"
    mitre_dir.mkdir(exist_ok=True)
    with open(mitre_dir / "techniques.json", "w") as f:
        json.dump({"techniques": mitre_techniques, "count": len(mitre_techniques)}, f, indent=2)

    # Generate IOCs
    iocs = []
    ioc_types = ["ip", "domain", "hash_md5", "hash_sha256", "url", "email"]

    for i in range(num_samples):
        ioc_type = random.choice(ioc_types)

        if ioc_type == "ip":
            value = f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
        elif ioc_type == "domain":
            words = ["malware", "evil", "bad", "hack", "dark", "shadow", "cyber", "attack"]
            tlds = ["com", "net", "org", "io", "xyz", "top", "ru", "cn"]
            value = f"{random.choice(words)}{random.randint(1, 999)}.{random.choice(tlds)}"
        elif ioc_type == "hash_md5":
            value = hashlib.md5(f"malware_{i}".encode()).hexdigest()
        elif ioc_type == "hash_sha256":
            value = hashlib.sha256(f"malware_{i}".encode()).hexdigest()
        elif ioc_type == "url":
            domain = f"malicious{random.randint(1, 999)}.com"
            path = random.choice(["/payload", "/download", "/update", "/login", "/admin"])
            value = f"http://{domain}{path}"
        else:
            value = f"attacker{random.randint(1, 999)}@malicious.com"

        threat_types = ["malware", "apt", "ransomware", "trojan", "botnet", "phishing", "exploit", "c2"]
        ioc = {
            "ioc_id": f"ioc_{i:06d}",
            "type": ioc_type,
            "value": value,
            "threat_type": random.choice(threat_types),
            "confidence": round(random.uniform(0.5, 1.0), 2),
            "severity": random.choice(["low", "medium", "high", "critical"]),
            "tags": random.sample(["apt", "ransomware", "trojan", "c2", "phishing", "exploit"], k=random.randint(1, 3)),
            "first_seen": datetime.now().isoformat(),
            "source": random.choice(["internal", "osint", "commercial", "government"])
        }
        iocs.append(ioc)

    ioc_dir = output_dir / "iocs"
    ioc_dir.mkdir(exist_ok=True)
    with open(ioc_dir / "indicators.json", "w") as f:
        json.dump({"indicators": iocs, "count": len(iocs)}, f, indent=2)

    return {"mitre_techniques": len(mitre_techniques), "iocs": len(iocs)}


def generate_security_events_dataset(output_dir: Path, num_samples: int = 5000):
    """Generate synthetic security event logs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    event_types = {
        "authentication": {
            "success": 0.85,
            "failure": 0.10,
            "lockout": 0.03,
            "mfa_challenge": 0.02
        },
        "network": {
            "connection": 0.60,
            "blocked": 0.15,
            "anomaly": 0.10,
            "scan": 0.10,
            "exfiltration": 0.05
        },
        "file": {
            "access": 0.70,
            "modification": 0.15,
            "deletion": 0.10,
            "suspicious": 0.05
        }
    }

    users = [f"user{i:03d}@company.com" for i in range(100)]
    hosts = [f"workstation-{i:03d}" for i in range(50)] + [f"server-{i:02d}" for i in range(20)]

    events = []
    for i in range(num_samples):
        category = random.choice(list(event_types.keys()))
        subtypes = event_types[category]
        subtype = random.choices(list(subtypes.keys()), weights=list(subtypes.values()))[0]

        # Determine if anomalous (10% anomaly rate)
        is_anomaly = random.random() < 0.10

        event = {
            "event_id": f"evt_{i:08d}",
            "timestamp": datetime.now().isoformat(),
            "category": category,
            "subtype": subtype,
            "user": random.choice(users),
            "source_host": random.choice(hosts),
            "source_ip": f"10.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}",
            "destination_ip": f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}",
            "severity": random.choice(["info", "low", "medium", "high", "critical"]),
            "is_anomaly": is_anomaly,
            "anomaly_score": round(random.uniform(0.7, 1.0), 3) if is_anomaly else round(random.uniform(0.0, 0.3), 3)
        }

        # Add category-specific fields
        if category == "authentication":
            event["auth_method"] = random.choice(["password", "sso", "mfa", "certificate"])
            event["result"] = "success" if subtype == "success" else "failure"
        elif category == "network":
            event["protocol"] = random.choice(["TCP", "UDP", "ICMP", "HTTP", "HTTPS", "DNS"])
            event["port"] = random.choice([22, 80, 443, 445, 3389, 8080, 8443])
            event["bytes_sent"] = random.randint(100, 1000000)
            event["bytes_received"] = random.randint(100, 1000000)
        elif category == "file":
            event["file_path"] = f"/data/files/document_{random.randint(1, 1000)}.{random.choice(['doc', 'pdf', 'xlsx', 'exe', 'dll'])}"
            event["action"] = subtype

        events.append(event)

    # Split by category
    for category in event_types.keys():
        cat_events = [e for e in events if e["category"] == category]
        cat_dir = output_dir / category
        cat_dir.mkdir(exist_ok=True)

        with open(cat_dir / "events.jsonl", "w") as f:
            for event in cat_events:
                f.write(json.dumps(event) + "\n")

    # Save labeled anomalies separately
    anomalies_dir = output_dir / "anomalies"
    anomalies_dir.mkdir(exist_ok=True)
    anomaly_events = [e for e in events if e["is_anomaly"]]
    normal_events = [e for e in events if not e["is_anomaly"]]

    with open(anomalies_dir / "anomalous.jsonl", "w") as f:
        for event in anomaly_events:
            f.write(json.dumps(event) + "\n")

    with open(anomalies_dir / "normal.jsonl", "w") as f:
        for event in normal_events[:len(anomaly_events) * 5]:  # 5:1 ratio for training
            f.write(json.dumps(event) + "\n")

    return {
        "total_events": len(events),
        "anomalies": len(anomaly_events),
        "by_category": {cat: len([e for e in events if e["category"] == cat]) for cat in event_types}
    }


def generate_anomaly_detection_dataset(output_dir: Path, num_samples: int = 2000):
    """Generate time-series anomaly detection dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)

    import math

    normal_dir = output_dir / "normal"
    anomalous_dir = output_dir / "anomalous"
    normal_dir.mkdir(exist_ok=True)
    anomalous_dir.mkdir(exist_ok=True)

    def generate_normal_series(length: int = 100) -> list:
        """Generate normal time series with slight variations."""
        base = 50 + random.uniform(-5, 5)
        noise_level = random.uniform(1, 3)
        trend = random.uniform(-0.05, 0.05)

        series = []
        for i in range(length):
            # Base + trend + seasonality + noise
            value = (
                base +
                trend * i +
                5 * math.sin(2 * math.pi * i / 24) +  # Daily pattern
                random.gauss(0, noise_level)
            )
            series.append(round(value, 2))
        return series

    def generate_anomalous_series(length: int = 100) -> tuple:
        """Generate time series with injected anomalies."""
        series = generate_normal_series(length)
        labels = [0] * length
        anomaly_type = random.choice(["spike", "dip", "shift", "trend_change", "variance_change"])

        if anomaly_type == "spike":
            # Point anomaly - sudden spike
            pos = random.randint(20, length - 20)
            series[pos] = series[pos] + random.uniform(30, 50)
            labels[pos] = 1

        elif anomaly_type == "dip":
            # Point anomaly - sudden dip
            pos = random.randint(20, length - 20)
            series[pos] = series[pos] - random.uniform(30, 50)
            labels[pos] = 1

        elif anomaly_type == "shift":
            # Contextual anomaly - level shift
            pos = random.randint(30, length - 30)
            shift = random.uniform(20, 40) * random.choice([-1, 1])
            for i in range(pos, min(pos + 10, length)):
                series[i] += shift
                labels[i] = 1

        elif anomaly_type == "trend_change":
            # Collective anomaly - sudden trend change
            pos = random.randint(30, length - 30)
            for i in range(pos, length):
                series[i] += (i - pos) * random.uniform(0.5, 1.5)
                labels[i] = 1

        elif anomaly_type == "variance_change":
            # Collective anomaly - variance increase
            pos = random.randint(30, length - 30)
            for i in range(pos, min(pos + 20, length)):
                series[i] += random.gauss(0, 15)
                labels[i] = 1

        return series, labels, anomaly_type

    # Generate normal samples (60%)
    normal_count = int(num_samples * 0.6)
    for i in range(normal_count):
        series = generate_normal_series()
        sample = {
            "id": f"normal_{i:06d}",
            "series": series,
            "labels": [0] * len(series),
            "is_anomalous": False,
            "length": len(series)
        }
        with open(normal_dir / f"sample_{i:06d}.json", "w") as f:
            json.dump(sample, f)

    # Generate anomalous samples (40%)
    anomalous_count = num_samples - normal_count
    for i in range(anomalous_count):
        series, labels, anomaly_type = generate_anomalous_series()
        sample = {
            "id": f"anomalous_{i:06d}",
            "series": series,
            "labels": labels,
            "is_anomalous": True,
            "anomaly_type": anomaly_type,
            "anomaly_positions": [j for j, l in enumerate(labels) if l == 1],
            "length": len(series)
        }
        with open(anomalous_dir / f"sample_{i:06d}.json", "w") as f:
            json.dump(sample, f)

    # Save metadata
    metadata = {
        "total_samples": num_samples,
        "normal_samples": normal_count,
        "anomalous_samples": anomalous_count,
        "series_length": 100,
        "anomaly_types": ["spike", "dip", "shift", "trend_change", "variance_change"],
        "generated_at": datetime.now().isoformat()
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main():
    """Generate all datasets."""
    base_dir = Path(__file__).parent.parent

    print("=" * 60)
    print("CRONOS AI Dataset Generator")
    print("=" * 60)

    # 1. Generate protocol samples
    print("\n[1/5] Generating Protocol Samples...")
    protocols_dir = base_dir / "protocols"

    print("  - ISO 8583...")
    iso_gen = ISO8583Generator(seed=42)
    iso_meta = iso_gen.generate_dataset(1000, str(protocols_dir / "iso8583"))
    print(f"    Generated {iso_meta['total_samples']} ISO 8583 samples")

    print("  - Modbus TCP...")
    modbus_gen = ModbusGenerator(seed=42)
    modbus_meta = modbus_gen.generate_dataset(1000, str(protocols_dir / "modbus"))
    print(f"    Generated {modbus_meta['total_samples']} Modbus samples")

    print("  - HL7 v2.x...")
    hl7_gen = HL7Generator(seed=42)
    hl7_meta = hl7_gen.generate_dataset(1000, str(protocols_dir / "hl7"))
    print(f"    Generated {hl7_meta['total_samples']} HL7 samples")

    # 2. Generate field detection dataset
    print("\n[2/5] Generating Field Detection Dataset...")
    field_dir = base_dir / "field_detection"
    field_meta = generate_field_detection_dataset(protocols_dir, field_dir)
    print(f"    Train: {field_meta['train_samples']}, Val: {field_meta['val_samples']}, Test: {field_meta['test_samples']}")

    # 3. Generate threat intelligence dataset
    print("\n[3/5] Generating Threat Intelligence Dataset...")
    threat_dir = base_dir / "threat_intelligence"
    threat_meta = generate_threat_intelligence_dataset(threat_dir, 1000)
    print(f"    MITRE techniques: {threat_meta['mitre_techniques']}, IOCs: {threat_meta['iocs']}")

    # 4. Generate security events dataset
    print("\n[4/5] Generating Security Events Dataset...")
    events_dir = base_dir / "security_events"
    events_meta = generate_security_events_dataset(events_dir, 5000)
    print(f"    Total events: {events_meta['total_events']}, Anomalies: {events_meta['anomalies']}")

    # 5. Generate anomaly detection dataset
    print("\n[5/5] Generating Anomaly Detection Dataset...")
    anomaly_dir = base_dir / "anomaly_detection"
    anomaly_meta = generate_anomaly_detection_dataset(anomaly_dir, 2000)
    print(f"    Normal: {anomaly_meta['normal_samples']}, Anomalous: {anomaly_meta['anomalous_samples']}")

    # Summary
    print("\n" + "=" * 60)
    print("Dataset Generation Complete!")
    print("=" * 60)
    print(f"\nOutput directory: {base_dir}")
    print("\nGenerated datasets:")
    print(f"  - Protocols: 3,000 samples (ISO-8583, Modbus, HL7)")
    print(f"  - Field Detection: {field_meta['total_samples']} labeled samples")
    print(f"  - Threat Intelligence: {threat_meta['mitre_techniques']} techniques, {threat_meta['iocs']} IOCs")
    print(f"  - Security Events: {events_meta['total_events']} events")
    print(f"  - Anomaly Detection: {anomaly_meta['total_samples']} time series")


if __name__ == "__main__":
    main()
