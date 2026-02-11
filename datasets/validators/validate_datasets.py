"""
QBITEL Dataset Validator

Comprehensive validation of all generated ML datasets.
"""

import json
import struct
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import sys


@dataclass
class ValidationResult:
    """Result of a validation check."""
    passed: bool
    message: str
    details: Optional[Dict] = None


class DatasetValidator:
    """Validate QBITEL datasets for ML training readiness."""

    def __init__(self, base_path: Optional[str] = None):
        if base_path:
            self.base_path = Path(base_path)
        else:
            self.base_path = Path(__file__).parent.parent

    def validate_all(self) -> Dict[str, List[ValidationResult]]:
        """Run all validations and return results."""
        results = {
            "protocols": self.validate_protocols(),
            "field_detection": self.validate_field_detection(),
            "threat_intelligence": self.validate_threat_intelligence(),
            "security_events": self.validate_security_events(),
            "anomaly_detection": self.validate_anomaly_detection(),
        }
        return results

    def validate_protocols(self) -> List[ValidationResult]:
        """Validate protocol sample datasets."""
        results = []
        protocols_dir = self.base_path / "protocols"

        if not protocols_dir.exists():
            results.append(ValidationResult(False, "Protocols directory not found"))
            return results

        expected_protocols = ["iso8583", "modbus", "hl7"]

        for protocol in expected_protocols:
            protocol_dir = protocols_dir / protocol

            if not protocol_dir.exists():
                results.append(ValidationResult(
                    False,
                    f"Protocol directory missing: {protocol}"
                ))
                continue

            # Check metadata file
            meta_file = protocol_dir / "dataset_metadata.json"
            if not meta_file.exists():
                results.append(ValidationResult(
                    False,
                    f"{protocol}: Missing dataset_metadata.json"
                ))
                continue

            with open(meta_file) as f:
                metadata = json.load(f)

            # Count samples
            bin_files = list(protocol_dir.glob("*.bin"))
            json_files = [f for f in protocol_dir.glob("*.json")
                         if f.name != "dataset_metadata.json"]

            # Validate sample count
            expected_count = metadata.get("total_samples", 0)
            actual_count = len(bin_files)

            if actual_count == expected_count:
                results.append(ValidationResult(
                    True,
                    f"{protocol}: Sample count matches ({actual_count})",
                    {"expected": expected_count, "actual": actual_count}
                ))
            else:
                results.append(ValidationResult(
                    False,
                    f"{protocol}: Sample count mismatch",
                    {"expected": expected_count, "actual": actual_count}
                ))

            # Validate bin/json pairs
            orphan_bins = []
            orphan_jsons = []

            for bin_file in bin_files:
                json_file = bin_file.with_suffix(".json")
                if not json_file.exists():
                    orphan_bins.append(bin_file.name)

            for json_file in json_files:
                bin_file = json_file.with_suffix(".bin")
                if not bin_file.exists():
                    orphan_jsons.append(json_file.name)

            if orphan_bins or orphan_jsons:
                results.append(ValidationResult(
                    False,
                    f"{protocol}: Unpaired files found",
                    {"orphan_bins": len(orphan_bins), "orphan_jsons": len(orphan_jsons)}
                ))
            else:
                results.append(ValidationResult(
                    True,
                    f"{protocol}: All bin/json files properly paired"
                ))

            # Validate sample structure (check first 5)
            valid_samples = 0
            invalid_samples = []

            for json_file in list(json_files)[:5]:
                try:
                    with open(json_file) as f:
                        sample_meta = json.load(f)

                    # Check required fields
                    required = ["protocol", "message_type", "timestamp", "fields"]
                    missing = [r for r in required if r not in sample_meta]

                    if missing:
                        invalid_samples.append({
                            "file": json_file.name,
                            "missing": missing
                        })
                    else:
                        valid_samples += 1
                except Exception as e:
                    invalid_samples.append({
                        "file": json_file.name,
                        "error": str(e)
                    })

            if invalid_samples:
                results.append(ValidationResult(
                    False,
                    f"{protocol}: Invalid sample structure",
                    {"invalid_samples": invalid_samples}
                ))
            else:
                results.append(ValidationResult(
                    True,
                    f"{protocol}: Sample structure validated ({valid_samples} checked)"
                ))

        return results

    def validate_field_detection(self) -> List[ValidationResult]:
        """Validate field detection labeled datasets."""
        results = []
        fd_dir = self.base_path / "field_detection"

        if not fd_dir.exists():
            results.append(ValidationResult(False, "Field detection directory not found"))
            return results

        # Check schema
        schema_file = fd_dir / "schema.json"
        if schema_file.exists():
            with open(schema_file) as f:
                schema = json.load(f)
            results.append(ValidationResult(
                True,
                f"Schema found: {len(schema.get('tag_types', []))} tag types"
            ))
        else:
            results.append(ValidationResult(False, "Schema file missing"))
            schema = {}

        # Validate splits
        expected_splits = {"training": 0.8, "validation": 0.1, "test": 0.1}
        split_counts = {}

        for split in expected_splits:
            split_dir = fd_dir / split
            if not split_dir.exists():
                results.append(ValidationResult(
                    False,
                    f"Split directory missing: {split}"
                ))
                continue

            samples = list(split_dir.glob("sample_*.json"))
            split_counts[split] = len(samples)

            # Validate sample structure
            invalid_count = 0
            tag_distribution = Counter()

            for sample_file in samples[:10]:  # Check first 10
                try:
                    with open(sample_file) as f:
                        sample = json.load(f)

                    # Check required fields
                    if "message_hex" not in sample or "tags" not in sample:
                        invalid_count += 1
                        continue

                    # Count tag types (tags is a list of strings in IOB format)
                    tags = sample.get("tags", [])
                    if isinstance(tags, list) and len(tags) > 0:
                        for tag in tags:
                            if isinstance(tag, str):
                                tag_distribution[tag] += 1

                except Exception:
                    invalid_count += 1

            if invalid_count > 0:
                results.append(ValidationResult(
                    False,
                    f"{split}: {invalid_count}/10 samples invalid"
                ))
            else:
                results.append(ValidationResult(
                    True,
                    f"{split}: {len(samples)} samples validated"
                ))

        # Check split ratios
        total = sum(split_counts.values())
        if total > 0:
            for split, expected_ratio in expected_splits.items():
                if split in split_counts:
                    actual_ratio = split_counts[split] / total
                    if abs(actual_ratio - expected_ratio) < 0.05:
                        results.append(ValidationResult(
                            True,
                            f"{split}: Ratio {actual_ratio:.1%} (expected {expected_ratio:.0%})"
                        ))
                    else:
                        results.append(ValidationResult(
                            False,
                            f"{split}: Ratio {actual_ratio:.1%} differs from expected {expected_ratio:.0%}"
                        ))

        return results

    def validate_threat_intelligence(self) -> List[ValidationResult]:
        """Validate threat intelligence datasets."""
        results = []
        ti_dir = self.base_path / "threat_intelligence"

        if not ti_dir.exists():
            results.append(ValidationResult(False, "Threat intelligence directory not found"))
            return results

        # Validate MITRE ATT&CK
        mitre_file = ti_dir / "mitre_attack" / "techniques.json"
        if mitre_file.exists():
            with open(mitre_file) as f:
                mitre_data = json.load(f)

            techniques = mitre_data.get("techniques", [])
            tactics = set()
            for tech in techniques:
                tactics.update(tech.get("tactics", []))

            results.append(ValidationResult(
                True,
                f"MITRE ATT&CK: {len(techniques)} techniques, {len(tactics)} tactics",
                {"techniques": len(techniques), "tactics": list(tactics)}
            ))

            # Validate technique structure
            required_fields = ["technique_id", "name", "tactics", "description"]
            for tech in techniques[:3]:
                missing = [f for f in required_fields if f not in tech]
                if missing:
                    results.append(ValidationResult(
                        False,
                        f"MITRE technique missing fields: {missing}"
                    ))
                    break
            else:
                results.append(ValidationResult(
                    True,
                    "MITRE technique structure validated"
                ))
        else:
            results.append(ValidationResult(False, "MITRE ATT&CK file missing"))

        # Validate IOCs
        ioc_file = ti_dir / "iocs" / "indicators.json"
        if ioc_file.exists():
            with open(ioc_file) as f:
                ioc_data = json.load(f)

            indicators = ioc_data.get("indicators", [])
            ioc_types = Counter(i.get("type") for i in indicators)

            results.append(ValidationResult(
                True,
                f"IOCs: {len(indicators)} indicators",
                {"types": dict(ioc_types)}
            ))

            # Validate IOC structure
            required_fields = ["ioc_id", "type", "value", "threat_type", "confidence"]
            for ioc in indicators[:5]:
                missing = [f for f in required_fields if f not in ioc]
                if missing:
                    results.append(ValidationResult(
                        False,
                        f"IOC missing fields: {missing}"
                    ))
                    break
            else:
                results.append(ValidationResult(
                    True,
                    "IOC structure validated"
                ))
        else:
            results.append(ValidationResult(False, "IOC file missing"))

        return results

    def validate_security_events(self) -> List[ValidationResult]:
        """Validate security event datasets."""
        results = []
        events_dir = self.base_path / "security_events"

        if not events_dir.exists():
            results.append(ValidationResult(False, "Security events directory not found"))
            return results

        categories = ["authentication", "network", "file"]
        total_events = 0

        for category in categories:
            cat_dir = events_dir / category
            event_file = cat_dir / "events.jsonl"

            if not event_file.exists():
                results.append(ValidationResult(
                    False,
                    f"{category}: events.jsonl missing"
                ))
                continue

            # Count and validate events
            event_count = 0
            valid_count = 0
            severity_dist = Counter()

            with open(event_file) as f:
                for line in f:
                    if line.strip():
                        event_count += 1
                        try:
                            event = json.loads(line)
                            if all(k in event for k in ["event_id", "timestamp", "category"]):
                                valid_count += 1
                                severity_dist[event.get("severity", "unknown")] += 1
                        except json.JSONDecodeError:
                            pass

            total_events += event_count

            if valid_count == event_count:
                results.append(ValidationResult(
                    True,
                    f"{category}: {event_count} events validated",
                    {"severities": dict(severity_dist)}
                ))
            else:
                results.append(ValidationResult(
                    False,
                    f"{category}: {event_count - valid_count}/{event_count} invalid events"
                ))

        # Check anomalies
        anomaly_file = events_dir / "anomalies" / "anomalous.jsonl"
        if anomaly_file.exists():
            anomaly_count = sum(1 for line in open(anomaly_file) if line.strip())
            results.append(ValidationResult(
                True,
                f"Anomalies: {anomaly_count} labeled events"
            ))
        else:
            results.append(ValidationResult(False, "Anomalies file missing"))

        results.append(ValidationResult(
            True,
            f"Total security events: {total_events}"
        ))

        return results

    def validate_anomaly_detection(self) -> List[ValidationResult]:
        """Validate anomaly detection time series datasets."""
        results = []
        ad_dir = self.base_path / "anomaly_detection"

        if not ad_dir.exists():
            results.append(ValidationResult(False, "Anomaly detection directory not found"))
            return results

        # Check metadata
        meta_file = ad_dir / "metadata.json"
        if meta_file.exists():
            with open(meta_file) as f:
                metadata = json.load(f)
            results.append(ValidationResult(
                True,
                f"Metadata: {metadata.get('total_samples', 0)} total samples"
            ))
        else:
            results.append(ValidationResult(False, "Metadata file missing"))
            metadata = {}

        # Validate normal samples
        normal_dir = ad_dir / "normal"
        if normal_dir.exists():
            normal_samples = list(normal_dir.glob("sample_*.json"))
            normal_count = len(normal_samples)

            # Validate structure
            valid_count = 0
            for sample_file in normal_samples[:10]:
                try:
                    with open(sample_file) as f:
                        sample = json.load(f)
                    if "series" in sample and "labels" in sample:
                        if len(sample["series"]) == len(sample["labels"]):
                            valid_count += 1
                except Exception:
                    pass

            results.append(ValidationResult(
                valid_count == min(10, normal_count),
                f"Normal samples: {normal_count} ({valid_count}/10 structure validated)"
            ))
        else:
            results.append(ValidationResult(False, "Normal samples directory missing"))
            normal_count = 0

        # Validate anomalous samples
        anomalous_dir = ad_dir / "anomalous"
        if anomalous_dir.exists():
            anomalous_samples = list(anomalous_dir.glob("sample_*.json"))
            anomalous_count = len(anomalous_samples)

            # Validate that anomalous samples have anomaly labels
            has_anomalies = 0
            for sample_file in anomalous_samples[:10]:
                try:
                    with open(sample_file) as f:
                        sample = json.load(f)
                    if any(label == 1 for label in sample.get("labels", [])):
                        has_anomalies += 1
                except Exception:
                    pass

            results.append(ValidationResult(
                has_anomalies == min(10, anomalous_count),
                f"Anomalous samples: {anomalous_count} ({has_anomalies}/10 contain anomaly labels)"
            ))
        else:
            results.append(ValidationResult(False, "Anomalous samples directory missing"))
            anomalous_count = 0

        # Check class balance
        total = normal_count + anomalous_count
        if total > 0:
            normal_ratio = normal_count / total
            results.append(ValidationResult(
                True,
                f"Class balance: {normal_ratio:.1%} normal, {1-normal_ratio:.1%} anomalous"
            ))

        return results

    def generate_report(self, results: Dict[str, List[ValidationResult]]) -> str:
        """Generate a validation report."""
        lines = []
        lines.append("=" * 70)
        lines.append("QBITEL Dataset Validation Report")
        lines.append("=" * 70)

        total_passed = 0
        total_failed = 0

        for category, validations in results.items():
            lines.append(f"\n[{category.upper().replace('_', ' ')}]")
            lines.append("-" * 40)

            for result in validations:
                status = "✓" if result.passed else "✗"
                lines.append(f"  {status} {result.message}")

                if result.details and not result.passed:
                    for key, value in result.details.items():
                        lines.append(f"      {key}: {value}")

                if result.passed:
                    total_passed += 1
                else:
                    total_failed += 1

        lines.append("\n" + "=" * 70)
        lines.append("SUMMARY")
        lines.append("=" * 70)
        lines.append(f"  Passed: {total_passed}")
        lines.append(f"  Failed: {total_failed}")
        lines.append(f"  Total:  {total_passed + total_failed}")

        if total_failed == 0:
            lines.append("\n  ✓ ALL VALIDATIONS PASSED - Datasets ready for ML training")
        else:
            lines.append(f"\n  ✗ {total_failed} VALIDATIONS FAILED - Review issues above")

        return "\n".join(lines)


def main():
    """Run dataset validation."""
    validator = DatasetValidator()
    results = validator.validate_all()
    report = validator.generate_report(results)
    print(report)

    # Return exit code based on validation results
    failed = sum(1 for category in results.values()
                 for result in category if not result.passed)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
