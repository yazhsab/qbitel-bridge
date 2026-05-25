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

    # ------------------------------------------------------------------
    # BPO-specific loaders
    # ------------------------------------------------------------------

    def load_bpo_samples(
        self,
        protocol: str,
        msg_type_filter: Optional[str] = None,
        limit: Optional[int] = None,
        shuffle: bool = False
    ) -> Generator[Tuple[bytes, Dict], None, None]:
        """
        Load BPO protocol message samples.

        Args:
            protocol: BPO protocol name (sip, rtp, tn3270e, cti, ivr)
            msg_type_filter: Filter by message type substring
            limit: Maximum number of samples to load
            shuffle: Whether to shuffle samples

        Yields:
            Tuple of (message_bytes, metadata_dict)
        """
        for message, metadata in self.load_protocol_samples(
            protocol, limit=None, shuffle=shuffle
        ):
            if msg_type_filter:
                msg_type = metadata.get("message_type", "")
                if msg_type_filter.lower() not in msg_type.lower():
                    continue
            yield message, metadata
            if limit:
                limit -= 1
                if limit <= 0:
                    return

    def load_bpo_cdr_data(
        self,
        fraud_only: bool = False,
        fraud_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> Generator[Dict, None, None]:
        """
        Load BPO toll fraud CDR data.

        Args:
            fraud_only: Only load records labeled as fraud
            fraud_type: Filter by specific fraud type (e.g., "IRSF", "PBX_HACK")
            limit: Maximum number of records to load

        Yields:
            CDR record dict
        """
        cdr_file = self.base_path / "security_events" / "bpo_toll_fraud" / "cdrs.jsonl"

        if not cdr_file.exists():
            raise ValueError(f"CDR file not found: {cdr_file}")

        count = 0
        with open(cdr_file) as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)

                if fraud_only and not record.get("is_fraud", False):
                    continue
                if fraud_type and record.get("fraud_type") != fraud_type:
                    continue

                yield record
                count += 1
                if limit and count >= limit:
                    break

    def load_bpo_pci_events(
        self,
        event_type: Optional[str] = None,
        compliant_only: bool = False,
        limit: Optional[int] = None
    ) -> Generator[Dict, None, None]:
        """
        Load BPO PCI voice compliance events.

        Args:
            event_type: Filter by event type (dtmf_payment, recording_pause_resume,
                       pan_detection, compliance_report)
            compliant_only: Only load compliant events
            limit: Maximum number of events to load

        Yields:
            PCI event dict
        """
        events_file = self.base_path / "security_events" / "bpo_pci_voice" / "events.jsonl"

        if not events_file.exists():
            raise ValueError(f"PCI events file not found: {events_file}")

        count = 0
        with open(events_file) as f:
            for line in f:
                if not line.strip():
                    continue
                event = json.loads(line)

                if event_type and event.get("event_type") != event_type:
                    continue
                if compliant_only and not event.get("is_compliant", True):
                    continue

                yield event
                count += 1
                if limit and count >= limit:
                    break

    def load_bpo_llm_pairs(
        self,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Load BPO LLM instruction pairs.

        Args:
            category: Filter by category (security_policy_generation,
                     fraud_pattern_analysis, pci_compliance_assessment,
                     call_anomaly_analysis, protocol_identification)
            difficulty: Filter by difficulty (basic, intermediate, advanced)
            limit: Maximum number of pairs to load

        Returns:
            List of instruction pair dicts
        """
        pairs_file = self.base_path / "security_events" / "bpo_llm_pairs" / "instruction_pairs.json"

        if not pairs_file.exists():
            raise ValueError(f"LLM pairs file not found: {pairs_file}")

        with open(pairs_file) as f:
            all_pairs = json.load(f)

        # Filter
        filtered = all_pairs
        if category:
            filtered = [p for p in filtered if p.get("category") == category]
        if difficulty:
            filtered = [p for p in filtered if p.get("difficulty") == difficulty]
        if limit:
            filtered = filtered[:limit]

        return filtered

    def get_dataset_stats(self) -> Dict:
        """Get statistics about all available datasets."""
        stats = {
            "protocols": {},
            "field_detection": {},
            "threat_intelligence": {},
            "security_events": {},
            "anomaly_detection": {},
            "bpo": {},
        }

        # Protocol stats (includes BPO protocols)
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

        # BPO-specific stats
        bpo_protocols = ["sip", "rtp", "tn3270e", "cti", "ivr"]
        bpo_protocol_stats = {}
        for proto in bpo_protocols:
            proto_dir = self.base_path / "protocols" / proto
            if proto_dir.exists():
                meta_file = proto_dir / "dataset_metadata.json"
                if meta_file.exists():
                    with open(meta_file) as f:
                        bpo_protocol_stats[proto] = json.load(f)
        if bpo_protocol_stats:
            stats["bpo"]["protocols"] = bpo_protocol_stats

        # Toll fraud CDR stats
        cdr_file = self.base_path / "security_events" / "bpo_toll_fraud" / "cdrs.jsonl"
        if cdr_file.exists():
            cdr_count = sum(1 for line in open(cdr_file) if line.strip())
            meta_file = cdr_file.parent / "dataset_metadata.json"
            if meta_file.exists():
                with open(meta_file) as f:
                    stats["bpo"]["toll_fraud_cdrs"] = json.load(f)
            else:
                stats["bpo"]["toll_fraud_cdrs"] = {"total_records": cdr_count}

        # PCI voice event stats
        pci_file = self.base_path / "security_events" / "bpo_pci_voice" / "events.jsonl"
        if pci_file.exists():
            pci_count = sum(1 for line in open(pci_file) if line.strip())
            meta_file = pci_file.parent / "dataset_metadata.json"
            if meta_file.exists():
                with open(meta_file) as f:
                    stats["bpo"]["pci_voice_events"] = json.load(f)
            else:
                stats["bpo"]["pci_voice_events"] = {"total_events": pci_count}

        # LLM instruction pair stats
        llm_file = self.base_path / "security_events" / "bpo_llm_pairs" / "instruction_pairs.json"
        if llm_file.exists():
            with open(llm_file) as f:
                pairs = json.load(f)
            stats["bpo"]["llm_instruction_pairs"] = {
                "total_pairs": len(pairs),
                "categories": list(set(p.get("category", "") for p in pairs)),
            }

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

    print("\n[BPO Datasets]")
    bpo = stats.get("bpo", {})
    bpo_protos = bpo.get("protocols", {})
    if bpo_protos:
        print("  Protocol Samples:")
        for proto, meta in bpo_protos.items():
            print(f"    {proto}: {meta.get('total_samples', 0)} samples")
    cdrs = bpo.get("toll_fraud_cdrs", {})
    if cdrs:
        print(f"  Toll Fraud CDRs: {cdrs.get('total_samples', cdrs.get('total_records', 0))} records")
    pci = bpo.get("pci_voice_events", {})
    if pci:
        print(f"  PCI Voice Events: {pci.get('total_samples', pci.get('total_events', 0))} events")
    llm = bpo.get("llm_instruction_pairs", {})
    if llm:
        print(f"  LLM Instruction Pairs: {llm.get('total_pairs', 0)} pairs")


if __name__ == "__main__":
    main()
