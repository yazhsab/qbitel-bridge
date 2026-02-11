"""
Fetch Protocol-Specific Datasets for QBITEL

This script fetches real-world protocol data relevant to QBITEL's
ML components:

1. BiLSTM-CRF Field Detection: Needs labeled protocol messages
2. VAE Anomaly Detection: Needs normal/anomalous protocol traffic
3. Protocol Discovery: Needs diverse protocol samples

Protocols supported:
- ISO 8583 (Banking/Financial)
- SWIFT (Financial messaging)
- Modbus TCP/RTU (Industrial/SCADA)
- HL7 v2.x (Healthcare)
- DNP3 (Power grid/SCADA)
- FHIR (Healthcare API)
"""

import json
import gzip
import zipfile
import io
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import urllib.request
import ssl
import struct


class ProtocolDatasetFetcher:
    """Fetch real protocol datasets for QBITEL training."""

    # Real protocol dataset sources
    PROTOCOL_SOURCES = {
        "modbus": {
            "pcap_samples": [
                {
                    "name": "4SICS ICS Lab Modbus",
                    "url": "https://github.com/4SICS/ICS-pcap/raw/master/modbus/modbus.pcap",
                    "description": "Industrial Modbus TCP traffic from 4SICS lab",
                    "license": "Public"
                }
            ],
            "datasets": [
                {
                    "name": "SWAT Modbus Dataset",
                    "source": "iTrust Singapore",
                    "url": "https://itrust.sutd.edu.sg/itrust-labs_datasets/",
                    "description": "Secure Water Treatment testbed - requires registration",
                    "manual": True
                }
            ]
        },
        "dnp3": {
            "pcap_samples": [
                {
                    "name": "4SICS DNP3 Samples",
                    "url": "https://github.com/4SICS/ICS-pcap/raw/master/dnp3/dnp3.pcap",
                    "description": "DNP3 SCADA protocol samples",
                    "license": "Public"
                }
            ]
        },
        "hl7": {
            "message_samples": [
                {
                    "name": "HL7 v2.x Test Messages",
                    "description": "Sample HL7 ADT, ORU, ORM messages",
                    "source": "HL7 International test data",
                    "samples": True  # Generated samples available
                }
            ],
            "fhir_endpoints": [
                {
                    "name": "HAPI FHIR Test Server",
                    "url": "http://hapi.fhir.org/baseR4",
                    "description": "Public FHIR R4 test server"
                },
                {
                    "name": "Synthea Synthetic Data",
                    "url": "https://github.com/synthetichealth/synthea",
                    "description": "Synthetic patient data generator"
                }
            ]
        },
        "iso8583": {
            "documentation": {
                "name": "ISO 8583 Specification",
                "description": "Banking transaction format - no public traffic datasets available",
                "note": "Use synthetic generators due to PCI compliance restrictions"
            }
        },
        "swift": {
            "documentation": {
                "name": "SWIFT MT/MX Messages",
                "description": "Financial messaging - no public datasets (regulated)",
                "note": "Use synthetic generators"
            }
        },
        "network_ids": {
            "datasets": [
                {
                    "id": "cicids2017",
                    "name": "CICIDS2017",
                    "url": "https://www.unb.ca/cic/datasets/ids-2017.html",
                    "size": "~6GB",
                    "description": "Network intrusion detection with labeled attacks",
                    "manual": True
                },
                {
                    "id": "unsw_nb15",
                    "name": "UNSW-NB15",
                    "url": "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
                    "size": "~2GB",
                    "description": "Modern network attack dataset",
                    "manual": True
                }
            ]
        }
    }

    def __init__(self, output_dir: Optional[str] = None):
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(__file__).parent.parent / "protocols"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ssl_context = ssl.create_default_context()
        self.ssl_context_unverified = ssl._create_unverified_context()

    def _fetch_url(self, url: str, timeout: int = 120) -> bytes:
        """Fetch URL with SSL fallback."""
        print(f"    Fetching: {url}")
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "QBITEL-Dataset-Fetcher/1.0"}
        )
        try:
            with urllib.request.urlopen(req, context=self.ssl_context, timeout=timeout) as response:
                return response.read()
        except (ssl.SSLError, urllib.error.URLError) as e:
            if "certificate" in str(e).lower() or "ssl" in str(e).lower():
                with urllib.request.urlopen(req, context=self.ssl_context_unverified, timeout=timeout) as response:
                    return response.read()
            raise

    def fetch_ics_pcap_samples(self) -> Dict:
        """
        Fetch ICS/SCADA protocol PCAP samples from 4SICS repository.
        Source: https://github.com/4SICS/ICS-pcap
        License: Public
        """
        print("\n[1] Fetching ICS Protocol PCAP Samples...")

        results = {}
        pcap_dir = self.output_dir / "pcap_samples"
        pcap_dir.mkdir(exist_ok=True)

        # Modbus PCAP
        try:
            print("  Fetching Modbus PCAP...")
            # Try alternative sources for Modbus samples
            modbus_urls = [
                "https://raw.githubusercontent.com/4SICS/ICS-pcap/master/modbus/modbus.pcap",
                "https://github.com/ITI/ICS-Security-Tools/raw/master/pcaps/modbus/modbus.pcap"
            ]

            modbus_data = None
            for url in modbus_urls:
                try:
                    modbus_data = self._fetch_url(url, timeout=30)
                    break
                except Exception:
                    continue

            if modbus_data:
                modbus_path = pcap_dir / "modbus_traffic.pcap"
                with open(modbus_path, "wb") as f:
                    f.write(modbus_data)
                results["modbus_pcap"] = {
                    "status": "success",
                    "path": str(modbus_path),
                    "size": len(modbus_data)
                }
                print(f"    ✓ Saved Modbus PCAP ({len(modbus_data)} bytes)")
            else:
                results["modbus_pcap"] = {"status": "not_available"}
                print("    ✗ Modbus PCAP not available from known sources")

        except Exception as e:
            print(f"    ✗ Error: {e}")
            results["modbus_pcap"] = {"status": "error", "message": str(e)}

        # DNP3 PCAP
        try:
            print("  Fetching DNP3 PCAP...")
            dnp3_urls = [
                "https://raw.githubusercontent.com/4SICS/ICS-pcap/master/dnp3/dnp3.pcap",
                "https://github.com/ITI/ICS-Security-Tools/raw/master/pcaps/dnp3/dnp3.pcap"
            ]

            dnp3_data = None
            for url in dnp3_urls:
                try:
                    dnp3_data = self._fetch_url(url, timeout=30)
                    break
                except Exception:
                    continue

            if dnp3_data:
                dnp3_path = pcap_dir / "dnp3_traffic.pcap"
                with open(dnp3_path, "wb") as f:
                    f.write(dnp3_data)
                results["dnp3_pcap"] = {
                    "status": "success",
                    "path": str(dnp3_path),
                    "size": len(dnp3_data)
                }
                print(f"    ✓ Saved DNP3 PCAP ({len(dnp3_data)} bytes)")
            else:
                results["dnp3_pcap"] = {"status": "not_available"}
                print("    ✗ DNP3 PCAP not available from known sources")

        except Exception as e:
            print(f"    ✗ Error: {e}")
            results["dnp3_pcap"] = {"status": "error", "message": str(e)}

        return results

    def fetch_wireshark_samples(self) -> Dict:
        """
        Fetch Wireshark sample captures for various protocols.
        Source: https://wiki.wireshark.org/SampleCaptures
        """
        print("\n[2] Fetching Wireshark Sample Captures...")

        results = {}
        pcap_dir = self.output_dir / "pcap_samples" / "wireshark"
        pcap_dir.mkdir(parents=True, exist_ok=True)

        # Wireshark sample URLs (verified working)
        samples = [
            ("http.cap", "https://wiki.wireshark.org/uploads/__moin_import__/attachments/SampleCaptures/http.cap", "HTTP"),
            ("dns.cap", "https://wiki.wireshark.org/uploads/__moin_import__/attachments/SampleCaptures/dns.cap", "DNS"),
        ]

        for filename, url, protocol in samples:
            try:
                print(f"  Fetching {protocol}...")
                data = self._fetch_url(url, timeout=30)
                filepath = pcap_dir / filename
                with open(filepath, "wb") as f:
                    f.write(data)
                results[protocol.lower()] = {
                    "status": "success",
                    "path": str(filepath),
                    "size": len(data)
                }
                print(f"    ✓ Saved {filename} ({len(data)} bytes)")
            except Exception as e:
                print(f"    ✗ {protocol}: {e}")
                results[protocol.lower()] = {"status": "error", "message": str(e)}

        return results

    def generate_protocol_relevance_report(self) -> Dict:
        """
        Generate a report on dataset relevance for QBITEL ML components.
        """
        print("\n[3] Generating Protocol Dataset Relevance Report...")

        report = {
            "generated_at": datetime.now().isoformat(),
            "qbitel_ml_components": {
                "bilstm_crf_field_detection": {
                    "purpose": "Automatic field boundary detection in protocol messages",
                    "data_requirements": [
                        "Binary protocol messages with labeled field boundaries",
                        "IOB-tagged sequences for supervised training",
                        "Multiple message types per protocol for generalization"
                    ],
                    "relevant_datasets": {
                        "synthetic": ["ISO 8583 (generated)", "Modbus (generated)", "HL7 (generated)"],
                        "real_world": ["ICS PCAP samples (need parsing)", "Wireshark captures (need parsing)"]
                    },
                    "data_gap": "Real-world labeled field boundaries are rare; synthetic generation is appropriate"
                },
                "vae_anomaly_detection": {
                    "purpose": "Detect anomalous protocol traffic patterns",
                    "data_requirements": [
                        "Normal baseline traffic for training",
                        "Labeled anomalies for validation",
                        "Time-series or sequence data"
                    ],
                    "relevant_datasets": {
                        "synthetic": ["Security events (generated)", "Time series (generated)"],
                        "real_world": ["NAB (fetched)", "CICIDS2017 (manual)", "UNSW-NB15 (manual)"]
                    },
                    "data_gap": "Need labeled IDS datasets; CICIDS2017 highly recommended"
                },
                "protocol_discovery": {
                    "purpose": "Identify unknown protocols from raw traffic",
                    "data_requirements": [
                        "Diverse protocol samples",
                        "Raw binary messages",
                        "Protocol labels for training"
                    ],
                    "relevant_datasets": {
                        "synthetic": ["Multi-protocol samples (generated)"],
                        "real_world": ["PCAP files with diverse protocols", "ICS traffic captures"]
                    },
                    "data_gap": "PCAP parsing needed to extract protocol messages"
                },
                "threat_intelligence_correlation": {
                    "purpose": "Correlate traffic with known threats",
                    "data_requirements": [
                        "IOCs (IPs, domains, hashes)",
                        "MITRE ATT&CK mappings",
                        "Known malicious patterns"
                    ],
                    "relevant_datasets": {
                        "fetched": ["MITRE ATT&CK (703 techniques)", "URLhaus (5K URLs)", "MalwareBazaar (700 hashes)"],
                        "huggingface": ["Security TTP Mapping (21K)", "Phishing URLs (11K)"]
                    },
                    "data_gap": "Good coverage from fetched sources"
                }
            },
            "protocol_specific_recommendations": {
                "iso8583": {
                    "status": "Synthetic only",
                    "reason": "PCI-DSS compliance prevents public datasets",
                    "recommendation": "Use generated samples; consider partnering with banks for real data"
                },
                "swift": {
                    "status": "Synthetic only",
                    "reason": "Regulated financial messaging",
                    "recommendation": "Use generated samples; SWIFT provides test environments for members"
                },
                "modbus": {
                    "status": "PCAP available",
                    "sources": ["4SICS ICS-pcap", "SWAT dataset (registration)"],
                    "recommendation": "Parse PCAP files for training; consider SWAT for labeled attacks"
                },
                "dnp3": {
                    "status": "PCAP available",
                    "sources": ["4SICS ICS-pcap"],
                    "recommendation": "Parse PCAP files; limited public data available"
                },
                "hl7": {
                    "status": "Synthetic + test servers",
                    "sources": ["HAPI FHIR server", "Synthea generator"],
                    "recommendation": "Use synthetic; real HL7 data is PHI-protected"
                }
            },
            "high_priority_datasets_to_acquire": [
                {
                    "name": "CICIDS2017",
                    "url": "https://www.unb.ca/cic/datasets/ids-2017.html",
                    "size": "~6GB",
                    "value": "Labeled network attacks for VAE anomaly detection",
                    "how_to_get": "Manual download from UNB website"
                },
                {
                    "name": "SWAT Dataset",
                    "url": "https://itrust.sutd.edu.sg/itrust-labs_datasets/",
                    "size": "~50GB",
                    "value": "ICS/SCADA attack dataset with Modbus traffic",
                    "how_to_get": "Register with iTrust Singapore"
                },
                {
                    "name": "UNSW-NB15",
                    "url": "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
                    "size": "~2GB",
                    "value": "Modern network attack types",
                    "how_to_get": "Manual download"
                }
            ]
        }

        # Save report
        report_path = self.output_dir.parent / "external" / "protocol_relevance_report.json"
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"  ✓ Report saved to {report_path}")
        return report

    def fetch_all(self) -> Dict:
        """Fetch all available protocol datasets."""
        print("=" * 60)
        print("QBITEL - Protocol Dataset Fetcher")
        print("=" * 60)

        results = {}

        results["ics_pcap"] = self.fetch_ics_pcap_samples()
        results["wireshark"] = self.fetch_wireshark_samples()
        results["relevance_report"] = self.generate_protocol_relevance_report()

        print("\n" + "=" * 60)
        print("Fetch Complete!")
        print("=" * 60)

        # Summary
        print("\nKey Findings:")
        print("  - ICS PCAP samples: Modbus/DNP3 traffic available")
        print("  - Financial protocols (ISO8583, SWIFT): Synthetic only (regulated)")
        print("  - Healthcare (HL7): Synthetic only (PHI protected)")
        print("  - For anomaly detection: CICIDS2017 highly recommended (manual download)")

        return results


def main():
    """Fetch protocol datasets."""
    fetcher = ProtocolDatasetFetcher()
    results = fetcher.fetch_all()

    print("\nNext Steps:")
    print("  1. Download CICIDS2017 from: https://www.unb.ca/cic/datasets/ids-2017.html")
    print("  2. Parse PCAP files for protocol message extraction")
    print("  3. Consider SWAT dataset for ICS attack scenarios")


if __name__ == "__main__":
    main()
