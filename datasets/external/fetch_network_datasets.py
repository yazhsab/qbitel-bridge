"""
Fetch Network Security Datasets

Downloads and integrates public network security datasets:
- Wireshark sample captures (protocol analysis)
- NAB anomaly detection dataset
- Public PCAP samples
"""

import json
import csv
import io
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import urllib.request
import ssl


class NetworkDatasetFetcher:
    """Fetch real-world network security datasets."""

    def __init__(self, output_dir: Optional[str] = None):
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(__file__).parent.parent
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ssl_context = ssl.create_default_context()
        self.ssl_context_unverified = ssl._create_unverified_context()

    def _fetch_url(self, url: str, timeout: int = 120) -> bytes:
        """Fetch URL content with SSL fallback."""
        print(f"  Fetching: {url}")
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "CRONOS-AI-Dataset-Fetcher/1.0"}
        )
        try:
            with urllib.request.urlopen(req, context=self.ssl_context, timeout=timeout) as response:
                return response.read()
        except (ssl.SSLError, urllib.error.URLError) as e:
            if "certificate" in str(e).lower() or "ssl" in str(e).lower():
                with urllib.request.urlopen(req, context=self.ssl_context_unverified, timeout=timeout) as response:
                    return response.read()
            raise

    def fetch_nab_dataset(self) -> Dict:
        """
        Fetch Numenta Anomaly Benchmark (NAB) dataset.
        Source: https://github.com/numenta/NAB
        License: AGPL-3.0 (data is Apache 2.0)

        Contains real-world time series with labeled anomalies.
        """
        print("\n[1] Fetching NAB Anomaly Detection Dataset...")

        # NAB dataset files on GitHub
        base_url = "https://raw.githubusercontent.com/numenta/NAB/master/data"

        datasets = {
            "realAWSCloudwatch": ["ec2_cpu_utilization_24ae8d.csv", "ec2_network_in_257a54.csv"],
            "realTraffic": ["speed_6005.csv", "speed_7578.csv"],
            "realTweets": ["Twitter_volume_AAPL.csv", "Twitter_volume_GOOG.csv"],
        }

        # Labels
        labels_url = "https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_labels.json"

        try:
            # Fetch labels
            labels_data = self._fetch_url(labels_url)
            labels = json.loads(labels_data)

            samples = []
            anomaly_dir = self.output_dir / "anomaly_detection" / "nab"
            anomaly_dir.mkdir(parents=True, exist_ok=True)

            for category, files in datasets.items():
                for filename in files:
                    try:
                        url = f"{base_url}/{category}/{filename}"
                        data = self._fetch_url(url, timeout=30)
                        content = data.decode("utf-8")

                        # Parse CSV
                        reader = csv.DictReader(io.StringIO(content))
                        series = []
                        timestamps = []

                        for row in reader:
                            timestamps.append(row.get("timestamp", ""))
                            series.append(float(row.get("value", 0)))

                        # Get anomaly labels for this file
                        label_key = f"{category}/{filename}"
                        anomaly_timestamps = labels.get(label_key, [])

                        # Create binary labels
                        binary_labels = [
                            1 if ts in anomaly_timestamps else 0
                            for ts in timestamps
                        ]

                        sample = {
                            "source": "NAB",
                            "category": category,
                            "filename": filename,
                            "series": series,
                            "timestamps": timestamps,
                            "labels": binary_labels,
                            "anomaly_count": sum(binary_labels),
                            "length": len(series),
                            "metadata": {
                                "metric_type": category,
                                "source_url": url
                            }
                        }
                        samples.append(sample)

                        # Save individual file
                        output_file = anomaly_dir / f"{category}_{filename.replace('.csv', '.json')}"
                        with open(output_file, "w") as f:
                            json.dump(sample, f, indent=2)

                        print(f"    ✓ {category}/{filename}: {len(series)} points, {sum(binary_labels)} anomalies")

                    except Exception as e:
                        print(f"    ✗ Error fetching {filename}: {e}")

            # Save metadata
            metadata = {
                "source": "Numenta Anomaly Benchmark (NAB)",
                "license": "Apache 2.0",
                "fetched_at": datetime.now().isoformat(),
                "total_samples": len(samples),
                "categories": list(datasets.keys())
            }

            with open(anomaly_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"  ✓ Saved {len(samples)} NAB time series")
            return {"nab_samples": len(samples)}

        except Exception as e:
            print(f"  ✗ Error fetching NAB dataset: {e}")
            return {"nab_samples": 0, "error": str(e)}

    def fetch_wireshark_samples_list(self) -> Dict:
        """
        Document available Wireshark sample captures.
        These need to be downloaded manually due to size.

        Source: https://wiki.wireshark.org/SampleCaptures
        """
        print("\n[2] Documenting Wireshark Sample Captures...")

        samples = {
            "protocols": {
                "modbus": {
                    "url": "https://wiki.wireshark.org/Modbus",
                    "description": "Modbus TCP/RTU protocol samples",
                    "manual_download": True
                },
                "dnp3": {
                    "url": "https://wiki.wireshark.org/DNP3",
                    "description": "DNP3 SCADA protocol samples",
                    "manual_download": True
                },
                "hl7": {
                    "url": "https://wiki.wireshark.org/HL7",
                    "description": "HL7 healthcare protocol samples",
                    "manual_download": True
                },
                "http": {
                    "url": "https://wiki.wireshark.org/SampleCaptures#HTTP",
                    "files": ["http.cap", "http_gzip.cap"],
                    "description": "HTTP protocol samples"
                },
                "ssl_tls": {
                    "url": "https://wiki.wireshark.org/SampleCaptures#SSL_with_decryption_keys",
                    "description": "TLS/SSL handshake samples"
                },
                "dns": {
                    "url": "https://wiki.wireshark.org/SampleCaptures#DNS",
                    "description": "DNS protocol samples"
                }
            },
            "security": {
                "malware": {
                    "url": "https://www.malware-traffic-analysis.net/",
                    "description": "Malware traffic captures (requires registration)",
                    "manual_download": True
                },
                "attacks": {
                    "url": "https://www.netresec.com/?page=MACCDC",
                    "description": "MACCDC attack traffic",
                    "manual_download": True
                }
            }
        }

        # Save documentation
        pcap_dir = self.output_dir / "pcap_samples"
        pcap_dir.mkdir(exist_ok=True)

        doc = {
            "source": "Wireshark Sample Captures",
            "generated_at": datetime.now().isoformat(),
            "note": "Most PCAP files require manual download due to size",
            "samples": samples,
            "download_instructions": {
                "step1": "Visit the URLs listed for each protocol",
                "step2": "Download the PCAP files manually",
                "step3": "Place files in datasets/pcap_samples/<protocol>/",
                "step4": "Run integrate_pcap.py to convert to training format"
            }
        }

        with open(pcap_dir / "sample_sources.json", "w") as f:
            json.dump(doc, f, indent=2)

        print("  ✓ Documented available PCAP sample sources")
        return {"pcap_sources_documented": len(samples["protocols"]) + len(samples["security"])}

    def fetch_cicids_info(self) -> Dict:
        """
        Document CICIDS2017 dataset information.
        This dataset is too large for automatic download (~6GB).

        Source: https://www.unb.ca/cic/datasets/ids-2017.html
        """
        print("\n[3] Documenting CICIDS2017 Dataset...")

        info = {
            "name": "CICIDS2017",
            "source": "Canadian Institute for Cybersecurity",
            "url": "https://www.unb.ca/cic/datasets/ids-2017.html",
            "size": "~6 GB",
            "records": "2,830,743 flows",
            "attacks": [
                "Brute Force (FTP, SSH)",
                "DoS (Hulk, GoldenEye, Slowloris, Slowhttptest)",
                "DDoS",
                "Web Attack (SQL Injection, XSS, Brute Force)",
                "Infiltration",
                "Botnet",
                "Port Scan"
            ],
            "features": 80,
            "format": "CSV (labeled flows)",
            "license": "Research use",
            "download_instructions": {
                "step1": "Visit https://www.unb.ca/cic/datasets/ids-2017.html",
                "step2": "Download the CSV files (MachineLearningCVE folder)",
                "step3": "Place in datasets/external/cicids2017/",
                "step4": "Run integrate_cicids.py to convert to CRONOS format"
            },
            "sample_columns": [
                "Flow ID", "Source IP", "Source Port", "Destination IP",
                "Destination Port", "Protocol", "Timestamp", "Flow Duration",
                "Total Fwd Packets", "Total Backward Packets", "Label"
            ]
        }

        external_dir = self.output_dir / "external"
        external_dir.mkdir(exist_ok=True)

        with open(external_dir / "cicids2017_info.json", "w") as f:
            json.dump(info, f, indent=2)

        print("  ✓ Documented CICIDS2017 dataset (manual download required)")
        return {"cicids_documented": True}

    def fetch_all(self) -> Dict:
        """Fetch all available network datasets."""
        print("=" * 60)
        print("CRONOS AI - Network Dataset Fetcher")
        print("=" * 60)

        results = {}

        results.update(self.fetch_nab_dataset())
        results.update(self.fetch_wireshark_samples_list())
        results.update(self.fetch_cicids_info())

        print("\n" + "=" * 60)
        print("Fetch Complete!")
        print("=" * 60)

        return results


def main():
    """Fetch network security datasets."""
    fetcher = NetworkDatasetFetcher()
    results = fetcher.fetch_all()

    print("\nResults:")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
