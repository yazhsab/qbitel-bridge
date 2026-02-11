"""
QBITEL Dataset Loader

Unified interface for loading all ML training datasets.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Generator
import random


class DatasetLoader:
    """Load and iterate over QBITEL datasets."""

    def __init__(self, base_path: Optional[str] = None):
        if base_path:
            self.base_path = Path(base_path)
        else:
            self.base_path = Path(__file__).parent

    def load_protocol_samples(
        self,
        protocol: str,
        limit: Optional[int] = None,
        shuffle: bool = False
    ) -> Generator[Tuple[bytes, Dict], None, None]:
        """
        Load protocol message samples.

        Args:
            protocol: Protocol name (iso8583, modbus, hl7, swift, dnp3, fhir)
            limit: Maximum number of samples to load
            shuffle: Whether to shuffle samples

        Yields:
            Tuple of (message_bytes, metadata_dict)
        """
        protocol_dir = self.base_path / "protocols" / protocol

        if not protocol_dir.exists():
            raise ValueError(f"Protocol directory not found: {protocol_dir}")

        # Find all sample files
        meta_files = list(protocol_dir.glob("*.json"))
        meta_files = [f for f in meta_files if f.name != "dataset_metadata.json"]

        if shuffle:
            random.shuffle(meta_files)

        if limit:
            meta_files = meta_files[:limit]

        for meta_file in meta_files:
            bin_file = meta_file.with_suffix(".bin")
            if not bin_file.exists():
                continue

            with open(meta_file) as f:
                metadata = json.load(f)

            with open(bin_file, "rb") as f:
                message = f.read()

            yield message, metadata

    def load_field_detection_data(
        self,
        split: str = "training",
        limit: Optional[int] = None,
        shuffle: bool = False
    ) -> Generator[Dict, None, None]:
        """
        Load field detection labeled data.

        Args:
            split: Dataset split (training, validation, test)
            limit: Maximum number of samples to load
            shuffle: Whether to shuffle samples

        Yields:
            Sample dict with message_bytes, tags, and metadata
        """
        split_dir = self.base_path / "field_detection" / split

        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")

        sample_files = list(split_dir.glob("sample_*.json"))

        if shuffle:
            random.shuffle(sample_files)

        if limit:
            sample_files = sample_files[:limit]

        for sample_file in sample_files:
            with open(sample_file) as f:
                yield json.load(f)

    def load_threat_intelligence(
        self,
        data_type: str = "iocs"
    ) -> Dict:
        """
        Load threat intelligence data.

        Args:
            data_type: Type of data (mitre_attack, iocs, cve)

        Returns:
            Dict containing the requested data
        """
        if data_type == "mitre_attack":
            path = self.base_path / "threat_intelligence" / "mitre_attack" / "techniques.json"
        elif data_type == "iocs":
            path = self.base_path / "threat_intelligence" / "iocs" / "indicators.json"
        else:
            raise ValueError(f"Unknown data type: {data_type}")

        if not path.exists():
            raise ValueError(f"Data file not found: {path}")

        with open(path) as f:
            return json.load(f)

    def load_security_events(
        self,
        category: Optional[str] = None,
        anomalies_only: bool = False,
        limit: Optional[int] = None
    ) -> Generator[Dict, None, None]:
        """
        Load security event logs.

        Args:
            category: Event category (authentication, network, file) or None for all
            anomalies_only: Only load labeled anomalies
            limit: Maximum number of events to load

        Yields:
            Event dict
        """
        if anomalies_only:
            event_file = self.base_path / "security_events" / "anomalies" / "anomalous.jsonl"
        elif category:
            event_file = self.base_path / "security_events" / category / "events.jsonl"
        else:
            # Load from all categories
            categories = ["authentication", "network", "file"]
            count = 0
            for cat in categories:
                for event in self.load_security_events(category=cat, limit=limit):
                    yield event
                    count += 1
                    if limit and count >= limit:
                        return
            return

        if not event_file.exists():
            raise ValueError(f"Event file not found: {event_file}")

        count = 0
        with open(event_file) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
                    count += 1
                    if limit and count >= limit:
                        break

    def load_anomaly_detection_data(
        self,
        include_normal: bool = True,
        include_anomalous: bool = True,
        limit: Optional[int] = None,
        shuffle: bool = False
    ) -> Generator[Dict, None, None]:
        """
        Load anomaly detection time series data.

        Args:
            include_normal: Include normal samples
            include_anomalous: Include anomalous samples
            limit: Maximum number of samples to load
            shuffle: Whether to shuffle samples

        Yields:
            Sample dict with series, labels, and metadata
        """
        sample_files = []

        if include_normal:
            normal_dir = self.base_path / "anomaly_detection" / "normal"
            if normal_dir.exists():
                sample_files.extend(list(normal_dir.glob("sample_*.json")))

        if include_anomalous:
            anomalous_dir = self.base_path / "anomaly_detection" / "anomalous"
            if anomalous_dir.exists():
                sample_files.extend(list(anomalous_dir.glob("sample_*.json")))

        if shuffle:
            random.shuffle(sample_files)

        if limit:
            sample_files = sample_files[:limit]

        for sample_file in sample_files:
            with open(sample_file) as f:
                yield json.load(f)

    def get_dataset_stats(self) -> Dict:
        """Get statistics about all available datasets."""
        stats = {
            "protocols": {},
            "field_detection": {},
            "threat_intelligence": {},
            "security_events": {},
            "anomaly_detection": {}
        }

        # Protocol stats
        protocols_dir = self.base_path / "protocols"
        if protocols_dir.exists():
            for protocol_dir in protocols_dir.iterdir():
                if protocol_dir.is_dir():
                    meta_file = protocol_dir / "dataset_metadata.json"
                    if meta_file.exists():
                        with open(meta_file) as f:
                            stats["protocols"][protocol_dir.name] = json.load(f)

        # Field detection stats
        field_dir = self.base_path / "field_detection"
        if field_dir.exists():
            schema_file = field_dir / "schema.json"
            if schema_file.exists():
                with open(schema_file) as f:
                    stats["field_detection"] = json.load(f)

        # Threat intelligence stats
        threat_dir = self.base_path / "threat_intelligence"
        if threat_dir.exists():
            mitre_file = threat_dir / "mitre_attack" / "techniques.json"
            ioc_file = threat_dir / "iocs" / "indicators.json"

            if mitre_file.exists():
                with open(mitre_file) as f:
                    data = json.load(f)
                    stats["threat_intelligence"]["mitre_techniques"] = data.get("count", 0)

            if ioc_file.exists():
                with open(ioc_file) as f:
                    data = json.load(f)
                    stats["threat_intelligence"]["iocs"] = data.get("count", 0)

        # Anomaly detection stats
        anomaly_dir = self.base_path / "anomaly_detection"
        if anomaly_dir.exists():
            meta_file = anomaly_dir / "metadata.json"
            if meta_file.exists():
                with open(meta_file) as f:
                    stats["anomaly_detection"] = json.load(f)

        return stats


def main():
    """Display dataset statistics."""
    loader = DatasetLoader()
    stats = loader.get_dataset_stats()

    print("=" * 60)
    print("QBITEL Dataset Statistics")
    print("=" * 60)

    print("\n[Protocols]")
    for protocol, meta in stats["protocols"].items():
        print(f"  {protocol}: {meta.get('total_samples', 0)} samples")

    print("\n[Field Detection]")
    fd = stats["field_detection"]
    print(f"  Total: {fd.get('total_samples', 0)} samples")
    print(f"  Train: {fd.get('train_samples', 0)}, Val: {fd.get('val_samples', 0)}, Test: {fd.get('test_samples', 0)}")

    print("\n[Threat Intelligence]")
    ti = stats["threat_intelligence"]
    print(f"  MITRE Techniques: {ti.get('mitre_techniques', 0)}")
    print(f"  IOCs: {ti.get('iocs', 0)}")

    print("\n[Anomaly Detection]")
    ad = stats["anomaly_detection"]
    print(f"  Total: {ad.get('total_samples', 0)} samples")
    print(f"  Normal: {ad.get('normal_samples', 0)}, Anomalous: {ad.get('anomalous_samples', 0)}")


if __name__ == "__main__":
    main()
