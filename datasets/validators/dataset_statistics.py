"""
QBITEL Dataset Statistics

Comprehensive statistics and analysis for ML training datasets.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter
from dataclasses import dataclass
import statistics


@dataclass
class DatasetStats:
    """Statistics for a dataset."""
    name: str
    total_samples: int
    categories: Dict[str, int]
    size_bytes: int
    avg_sample_size: float
    additional: Dict


class DatasetStatistics:
    """Generate comprehensive statistics for QBITEL datasets."""

    def __init__(self, base_path: Optional[str] = None):
        if base_path:
            self.base_path = Path(base_path)
        else:
            self.base_path = Path(__file__).parent.parent

    def get_all_statistics(self) -> Dict[str, DatasetStats]:
        """Get statistics for all datasets."""
        return {
            "protocols": self.get_protocol_stats(),
            "field_detection": self.get_field_detection_stats(),
            "threat_intelligence": self.get_threat_intelligence_stats(),
            "security_events": self.get_security_events_stats(),
            "anomaly_detection": self.get_anomaly_detection_stats(),
        }

    def get_protocol_stats(self) -> DatasetStats:
        """Get protocol dataset statistics."""
        protocols_dir = self.base_path / "protocols"
        total_samples = 0
        total_size = 0
        categories = {}
        sample_sizes = []
        message_types = Counter()
        field_counts = []

        if not protocols_dir.exists():
            return DatasetStats("protocols", 0, {}, 0, 0, {})

        for protocol_dir in protocols_dir.iterdir():
            if not protocol_dir.is_dir():
                continue

            protocol_name = protocol_dir.name
            bin_files = list(protocol_dir.glob("*.bin"))
            json_files = [f for f in protocol_dir.glob("*.json")
                         if f.name != "dataset_metadata.json"]

            categories[protocol_name] = len(bin_files)
            total_samples += len(bin_files)

            for bin_file in bin_files:
                file_size = bin_file.stat().st_size
                total_size += file_size
                sample_sizes.append(file_size)

            # Analyze message types and fields
            for json_file in list(json_files)[:100]:  # Sample 100
                try:
                    with open(json_file) as f:
                        meta = json.load(f)
                    message_types[meta.get("message_type", "unknown")] += 1
                    field_counts.append(len(meta.get("fields", [])))
                except Exception:
                    pass

        return DatasetStats(
            name="protocols",
            total_samples=total_samples,
            categories=categories,
            size_bytes=total_size,
            avg_sample_size=statistics.mean(sample_sizes) if sample_sizes else 0,
            additional={
                "message_types": dict(message_types.most_common(20)),
                "avg_fields_per_message": statistics.mean(field_counts) if field_counts else 0,
                "min_sample_size": min(sample_sizes) if sample_sizes else 0,
                "max_sample_size": max(sample_sizes) if sample_sizes else 0,
            }
        )

    def get_field_detection_stats(self) -> DatasetStats:
        """Get field detection dataset statistics."""
        fd_dir = self.base_path / "field_detection"
        total_samples = 0
        total_size = 0
        categories = {}
        tag_distribution = Counter()
        samples_per_protocol = Counter()
        fields_per_sample = []

        if not fd_dir.exists():
            return DatasetStats("field_detection", 0, {}, 0, 0, {})

        for split in ["training", "validation", "test"]:
            split_dir = fd_dir / split
            if not split_dir.exists():
                continue

            samples = list(split_dir.glob("sample_*.json"))
            categories[split] = len(samples)
            total_samples += len(samples)

            for sample_file in samples:
                total_size += sample_file.stat().st_size

                try:
                    with open(sample_file) as f:
                        sample = json.load(f)

                    samples_per_protocol[sample.get("protocol", "unknown")] += 1

                    tags = sample.get("tags", [])
                    fields_per_sample.append(len(tags))

                    for tag in tags:
                        tag_distribution[tag.get("tag", "unknown")] += 1
                except Exception:
                    pass

        return DatasetStats(
            name="field_detection",
            total_samples=total_samples,
            categories=categories,
            size_bytes=total_size,
            avg_sample_size=total_size / total_samples if total_samples else 0,
            additional={
                "tag_distribution": dict(tag_distribution.most_common()),
                "protocol_distribution": dict(samples_per_protocol),
                "avg_fields_per_sample": statistics.mean(fields_per_sample) if fields_per_sample else 0,
                "train_val_test_ratio": f"{categories.get('training', 0)}:{categories.get('validation', 0)}:{categories.get('test', 0)}"
            }
        )

    def get_threat_intelligence_stats(self) -> DatasetStats:
        """Get threat intelligence dataset statistics."""
        ti_dir = self.base_path / "threat_intelligence"
        total_size = 0
        categories = {}
        additional = {}

        if not ti_dir.exists():
            return DatasetStats("threat_intelligence", 0, {}, 0, 0, {})

        # MITRE ATT&CK
        mitre_file = ti_dir / "mitre_attack" / "techniques.json"
        if mitre_file.exists():
            total_size += mitre_file.stat().st_size
            with open(mitre_file) as f:
                mitre_data = json.load(f)
            techniques = mitre_data.get("techniques", [])
            categories["mitre_techniques"] = len(techniques)

            tactics = Counter()
            platforms = Counter()
            for tech in techniques:
                for tactic in tech.get("tactics", []):
                    tactics[tactic] += 1
                for platform in tech.get("platforms", []):
                    platforms[platform] += 1

            additional["mitre_tactics"] = dict(tactics)
            additional["mitre_platforms"] = dict(platforms)

        # IOCs
        ioc_file = ti_dir / "iocs" / "indicators.json"
        if ioc_file.exists():
            total_size += ioc_file.stat().st_size
            with open(ioc_file) as f:
                ioc_data = json.load(f)
            indicators = ioc_data.get("indicators", [])
            categories["iocs"] = len(indicators)

            ioc_types = Counter(i.get("type") for i in indicators)
            threat_types = Counter(i.get("threat_type") for i in indicators)
            confidence_dist = Counter(i.get("confidence") for i in indicators)

            additional["ioc_types"] = dict(ioc_types)
            additional["threat_types"] = dict(threat_types)
            additional["confidence_distribution"] = dict(confidence_dist)

        return DatasetStats(
            name="threat_intelligence",
            total_samples=sum(categories.values()),
            categories=categories,
            size_bytes=total_size,
            avg_sample_size=0,
            additional=additional
        )

    def get_security_events_stats(self) -> DatasetStats:
        """Get security events dataset statistics."""
        events_dir = self.base_path / "security_events"
        total_samples = 0
        total_size = 0
        categories = {}
        severity_dist = Counter()
        event_types = Counter()
        hourly_dist = Counter()

        if not events_dir.exists():
            return DatasetStats("security_events", 0, {}, 0, 0, {})

        for category in ["authentication", "network", "file"]:
            event_file = events_dir / category / "events.jsonl"
            if not event_file.exists():
                continue

            total_size += event_file.stat().st_size
            event_count = 0

            with open(event_file) as f:
                for line in f:
                    if line.strip():
                        event_count += 1
                        try:
                            event = json.loads(line)
                            severity_dist[event.get("severity", "unknown")] += 1
                            event_types[event.get("event_type", "unknown")] += 1

                            # Extract hour from timestamp
                            ts = event.get("timestamp", "")
                            if "T" in ts:
                                hour = ts.split("T")[1][:2]
                                hourly_dist[hour] += 1
                        except Exception:
                            pass

            categories[category] = event_count
            total_samples += event_count

        # Anomalies
        anomaly_file = events_dir / "anomalies" / "anomalous.jsonl"
        if anomaly_file.exists():
            total_size += anomaly_file.stat().st_size
            anomaly_count = sum(1 for line in open(anomaly_file) if line.strip())
            categories["anomalies"] = anomaly_count

        return DatasetStats(
            name="security_events",
            total_samples=total_samples,
            categories=categories,
            size_bytes=total_size,
            avg_sample_size=total_size / total_samples if total_samples else 0,
            additional={
                "severity_distribution": dict(severity_dist),
                "top_event_types": dict(event_types.most_common(10)),
                "anomaly_ratio": categories.get("anomalies", 0) / total_samples if total_samples else 0,
            }
        )

    def get_anomaly_detection_stats(self) -> DatasetStats:
        """Get anomaly detection dataset statistics."""
        ad_dir = self.base_path / "anomaly_detection"
        total_samples = 0
        total_size = 0
        categories = {}
        series_lengths = []
        anomaly_ratios = []
        metric_types = Counter()

        if not ad_dir.exists():
            return DatasetStats("anomaly_detection", 0, {}, 0, 0, {})

        for label_type in ["normal", "anomalous"]:
            label_dir = ad_dir / label_type
            if not label_dir.exists():
                continue

            samples = list(label_dir.glob("sample_*.json"))
            categories[label_type] = len(samples)
            total_samples += len(samples)

            for sample_file in samples:
                total_size += sample_file.stat().st_size

                try:
                    with open(sample_file) as f:
                        sample = json.load(f)

                    series = sample.get("series", [])
                    labels = sample.get("labels", [])
                    series_lengths.append(len(series))

                    if labels:
                        anomaly_ratio = sum(labels) / len(labels)
                        anomaly_ratios.append(anomaly_ratio)

                    metric_types[sample.get("metadata", {}).get("metric_type", "unknown")] += 1
                except Exception:
                    pass

        return DatasetStats(
            name="anomaly_detection",
            total_samples=total_samples,
            categories=categories,
            size_bytes=total_size,
            avg_sample_size=total_size / total_samples if total_samples else 0,
            additional={
                "avg_series_length": statistics.mean(series_lengths) if series_lengths else 0,
                "avg_anomaly_ratio": statistics.mean(anomaly_ratios) if anomaly_ratios else 0,
                "metric_types": dict(metric_types),
                "class_balance": f"{categories.get('normal', 0)}:{categories.get('anomalous', 0)}"
            }
        )

    def generate_report(self) -> str:
        """Generate a comprehensive statistics report."""
        stats = self.get_all_statistics()
        lines = []

        lines.append("=" * 70)
        lines.append("QBITEL Dataset Statistics Report")
        lines.append("=" * 70)

        total_samples = 0
        total_size = 0

        for name, dataset_stats in stats.items():
            total_samples += dataset_stats.total_samples
            total_size += dataset_stats.size_bytes

            lines.append(f"\n[{name.upper().replace('_', ' ')}]")
            lines.append("-" * 40)
            lines.append(f"  Total Samples: {dataset_stats.total_samples:,}")
            lines.append(f"  Total Size: {self._format_size(dataset_stats.size_bytes)}")

            if dataset_stats.avg_sample_size > 0:
                lines.append(f"  Avg Sample Size: {self._format_size(dataset_stats.avg_sample_size)}")

            lines.append(f"  Categories:")
            for cat, count in dataset_stats.categories.items():
                lines.append(f"    - {cat}: {count:,}")

            if dataset_stats.additional:
                lines.append(f"  Details:")
                for key, value in dataset_stats.additional.items():
                    if isinstance(value, dict):
                        lines.append(f"    {key}:")
                        for k, v in list(value.items())[:5]:
                            lines.append(f"      - {k}: {v}")
                        if len(value) > 5:
                            lines.append(f"      ... and {len(value) - 5} more")
                    elif isinstance(value, float):
                        lines.append(f"    {key}: {value:.2f}")
                    else:
                        lines.append(f"    {key}: {value}")

        lines.append("\n" + "=" * 70)
        lines.append("OVERALL SUMMARY")
        lines.append("=" * 70)
        lines.append(f"  Total Samples: {total_samples:,}")
        lines.append(f"  Total Size: {self._format_size(total_size)}")
        lines.append(f"  Datasets: {len(stats)}")

        # ML Training Readiness Assessment
        lines.append("\n" + "-" * 40)
        lines.append("ML TRAINING READINESS")
        lines.append("-" * 40)

        readiness = []

        # Protocol Analysis
        proto_stats = stats["protocols"]
        if proto_stats.total_samples >= 1000:
            readiness.append("  ✓ Protocol samples sufficient for BiLSTM-CRF training")
        else:
            readiness.append("  ⚠ Need more protocol samples (target: 1000+)")

        # Field Detection
        fd_stats = stats["field_detection"]
        if fd_stats.total_samples >= 2000:
            readiness.append("  ✓ Field detection labels sufficient for supervised learning")
        else:
            readiness.append("  ⚠ Need more labeled samples (target: 2000+)")

        # Anomaly Detection
        ad_stats = stats["anomaly_detection"]
        if ad_stats.total_samples >= 1000:
            readiness.append("  ✓ Anomaly detection data sufficient for VAE/LSTM training")
        else:
            readiness.append("  ⚠ Need more time series samples (target: 1000+)")

        # Threat Intelligence
        ti_stats = stats["threat_intelligence"]
        if ti_stats.categories.get("iocs", 0) >= 500:
            readiness.append("  ✓ IOC dataset sufficient for threat correlation")
        else:
            readiness.append("  ⚠ Need more IOCs (target: 500+)")

        lines.extend(readiness)

        return "\n".join(lines)

    def _format_size(self, size_bytes: float) -> str:
        """Format size in human readable format."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"


def main():
    """Run dataset statistics."""
    stats = DatasetStatistics()
    report = stats.generate_report()
    print(report)


if __name__ == "__main__":
    main()
