"""
Fetch Datasets from Major ML Platforms

Supports downloading from:
- Hugging Face Datasets
- Kaggle Datasets
- UCI ML Repository
- OpenML

Note: Some platforms require API keys or authentication.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import urllib.request
import ssl


class MLPlatformFetcher:
    """Fetch datasets from major ML platforms."""

    # Relevant datasets for CRONOS AI
    DATASETS = {
        "huggingface": {
            "security": [
                {
                    "id": "cais/mmlu",
                    "subset": "computer_security",
                    "description": "Security knowledge benchmark",
                    "size": "~10MB"
                },
                {
                    "id": "AI4Sec/cve-cwe-mapping",
                    "description": "CVE to CWE mappings",
                    "size": "~5MB"
                },
            ],
            "network": [
                {
                    "id": "rdpahalmern/ids2018",
                    "description": "Intrusion detection dataset (CSE-CIC-IDS2018)",
                    "size": "~500MB"
                },
            ],
            "nlp_security": [
                {
                    "id": "Hatman/security-paper-titles",
                    "description": "Security research paper titles",
                    "size": "~1MB"
                },
            ],
            "malware": [
                {
                    "id": "yashveersinghsohi/malware_analysis",
                    "description": "Malware analysis features",
                    "size": "~50MB"
                },
            ]
        },
        "kaggle": {
            "network_security": [
                {
                    "id": "dhoogla/cicidss",
                    "name": "CICIDS2017 Network Intrusion",
                    "description": "Network intrusion detection dataset",
                    "size": "~600MB"
                },
                {
                    "id": "sampadab17/network-intrusion-detection",
                    "name": "NSL-KDD",
                    "description": "Classic IDS benchmark dataset",
                    "size": "~20MB"
                },
                {
                    "id": "mrwellsdavid/unsw-nb15",
                    "name": "UNSW-NB15",
                    "description": "Modern network intrusion dataset",
                    "size": "~200MB"
                },
            ],
            "malware": [
                {
                    "id": "claudelemante/malware-data",
                    "name": "Malware Detection Dataset",
                    "description": "PE file features for malware detection",
                    "size": "~50MB"
                },
                {
                    "id": "ang3loliveira/malware-analysis-datasets-pe-section-headers",
                    "name": "PE Section Headers",
                    "description": "PE malware headers analysis",
                    "size": "~100MB"
                },
            ],
            "fraud": [
                {
                    "id": "mlg-ulb/creditcardfraud",
                    "name": "Credit Card Fraud Detection",
                    "description": "Anonymized credit card transactions",
                    "size": "~150MB"
                },
                {
                    "id": "ealaxi/paysim1",
                    "name": "PaySim Fraud",
                    "description": "Synthetic financial fraud dataset",
                    "size": "~500MB"
                },
            ],
            "anomaly": [
                {
                    "id": "boltzmannbrain/nab",
                    "name": "NAB Dataset",
                    "description": "Numenta Anomaly Benchmark",
                    "size": "~10MB"
                },
            ],
            "iot_security": [
                {
                    "id": "francoisxa/ds2ostraffictraces",
                    "name": "IoT Network Traffic",
                    "description": "IoT device network traces",
                    "size": "~1GB"
                },
            ]
        },
        "uci": {
            "kdd99": {
                "url": "https://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz",
                "description": "KDD Cup 1999 - Network intrusion (10%)",
                "size": "~2MB compressed"
            },
            "spambase": {
                "url": "https://archive.ics.uci.edu/static/public/94/spambase.zip",
                "description": "Spam email classification",
                "size": "~500KB"
            },
            "phishing": {
                "url": "https://archive.ics.uci.edu/static/public/327/phishing+websites.zip",
                "description": "Phishing website detection features",
                "size": "~200KB"
            }
        }
    }

    def __init__(self, output_dir: Optional[str] = None):
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(__file__).parent.parent / "external" / "ml_platforms"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ssl_context = ssl.create_default_context()
        self.ssl_context_unverified = ssl._create_unverified_context()

    def _fetch_url(self, url: str, timeout: int = 120) -> bytes:
        """Fetch URL content with SSL fallback."""
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

    def check_huggingface_cli(self) -> bool:
        """Check if Hugging Face datasets library is installed."""
        try:
            import datasets
            return True
        except ImportError:
            return False

    def check_kaggle_cli(self) -> bool:
        """Check if Kaggle CLI is installed and configured."""
        try:
            result = subprocess.run(
                ["kaggle", "--version"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def fetch_huggingface_dataset(self, dataset_id: str, subset: str = None) -> Dict:
        """
        Fetch dataset from Hugging Face.
        Requires: pip install datasets
        """
        print(f"\n  Fetching HuggingFace: {dataset_id}")

        if not self.check_huggingface_cli():
            return {
                "status": "error",
                "message": "Install datasets library: pip install datasets"
            }

        try:
            from datasets import load_dataset

            if subset:
                dataset = load_dataset(dataset_id, subset, trust_remote_code=True)
            else:
                dataset = load_dataset(dataset_id, trust_remote_code=True)

            # Save to disk
            output_path = self.output_dir / "huggingface" / dataset_id.replace("/", "_")
            output_path.mkdir(parents=True, exist_ok=True)

            dataset.save_to_disk(str(output_path))

            # Get stats
            stats = {
                "dataset_id": dataset_id,
                "subset": subset,
                "splits": list(dataset.keys()) if hasattr(dataset, 'keys') else ["train"],
                "path": str(output_path),
                "fetched_at": datetime.now().isoformat()
            }

            print(f"    ✓ Saved to {output_path}")
            return {"status": "success", **stats}

        except Exception as e:
            print(f"    ✗ Error: {e}")
            return {"status": "error", "message": str(e)}

    def fetch_kaggle_dataset(self, dataset_id: str) -> Dict:
        """
        Fetch dataset from Kaggle.
        Requires: pip install kaggle + API key in ~/.kaggle/kaggle.json
        """
        print(f"\n  Fetching Kaggle: {dataset_id}")

        if not self.check_kaggle_cli():
            return {
                "status": "error",
                "message": "Install Kaggle CLI: pip install kaggle, then configure API key"
            }

        try:
            output_path = self.output_dir / "kaggle" / dataset_id.replace("/", "_")
            output_path.mkdir(parents=True, exist_ok=True)

            result = subprocess.run(
                ["kaggle", "datasets", "download", "-d", dataset_id, "-p", str(output_path), "--unzip"],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                print(f"    ✓ Downloaded to {output_path}")
                return {
                    "status": "success",
                    "dataset_id": dataset_id,
                    "path": str(output_path),
                    "fetched_at": datetime.now().isoformat()
                }
            else:
                print(f"    ✗ Error: {result.stderr}")
                return {"status": "error", "message": result.stderr}

        except Exception as e:
            print(f"    ✗ Error: {e}")
            return {"status": "error", "message": str(e)}

    def fetch_uci_dataset(self, name: str) -> Dict:
        """Fetch dataset from UCI ML Repository."""
        if name not in self.DATASETS["uci"]:
            return {"status": "error", "message": f"Unknown UCI dataset: {name}"}

        info = self.DATASETS["uci"][name]
        url = info["url"]

        print(f"\n  Fetching UCI: {name}")
        print(f"    URL: {url}")

        try:
            data = self._fetch_url(url)

            output_path = self.output_dir / "uci"
            output_path.mkdir(parents=True, exist_ok=True)

            # Handle gzipped files
            filename = url.split("/")[-1]
            filepath = output_path / filename

            with open(filepath, "wb") as f:
                f.write(data)

            print(f"    ✓ Downloaded to {filepath}")
            return {
                "status": "success",
                "name": name,
                "path": str(filepath),
                "size_bytes": len(data),
                "fetched_at": datetime.now().isoformat()
            }

        except Exception as e:
            print(f"    ✗ Error: {e}")
            return {"status": "error", "message": str(e)}

    def generate_dataset_catalog(self) -> Dict:
        """Generate a catalog of all available datasets."""
        print("\n[Generating Dataset Catalog]")

        catalog = {
            "generated_at": datetime.now().isoformat(),
            "platforms": {},
            "setup_instructions": {
                "huggingface": {
                    "install": "pip install datasets",
                    "usage": "fetcher.fetch_huggingface_dataset('dataset_id')"
                },
                "kaggle": {
                    "install": "pip install kaggle",
                    "setup": [
                        "1. Go to kaggle.com/account",
                        "2. Click 'Create New API Token'",
                        "3. Save kaggle.json to ~/.kaggle/",
                        "4. chmod 600 ~/.kaggle/kaggle.json"
                    ],
                    "usage": "fetcher.fetch_kaggle_dataset('owner/dataset')"
                },
                "uci": {
                    "install": "No installation required",
                    "usage": "fetcher.fetch_uci_dataset('kdd99')"
                }
            },
            "recommended_for_cronos": {
                "protocol_analysis": [
                    "Wireshark sample captures (manual)",
                    "PCAP datasets from Netresec"
                ],
                "intrusion_detection": [
                    "kaggle: dhoogla/cicidss (CICIDS2017)",
                    "kaggle: sampadab17/network-intrusion-detection (NSL-KDD)",
                    "kaggle: mrwellsdavid/unsw-nb15",
                    "uci: kdd99"
                ],
                "malware_analysis": [
                    "kaggle: claudelemante/malware-data",
                    "huggingface: yashveersinghsohi/malware_analysis",
                    "MalwareBazaar (already integrated)"
                ],
                "fraud_detection": [
                    "kaggle: mlg-ulb/creditcardfraud",
                    "kaggle: ealaxi/paysim1"
                ],
                "anomaly_detection": [
                    "NAB (already integrated)",
                    "kaggle: boltzmannbrain/nab"
                ],
                "threat_intelligence": [
                    "MITRE ATT&CK (already integrated)",
                    "URLhaus (already integrated)",
                    "huggingface: AI4Sec/cve-cwe-mapping"
                ]
            }
        }

        # Add platform details
        for platform, categories in self.DATASETS.items():
            if platform == "uci":
                catalog["platforms"][platform] = {
                    "datasets": categories,
                    "requires_auth": False
                }
            else:
                catalog["platforms"][platform] = {
                    "categories": categories,
                    "requires_auth": platform == "kaggle"
                }

        # Save catalog
        catalog_path = self.output_dir / "dataset_catalog.json"
        with open(catalog_path, "w") as f:
            json.dump(catalog, f, indent=2)

        print(f"  ✓ Catalog saved to {catalog_path}")
        return catalog

    def fetch_recommended_free(self) -> Dict:
        """Fetch recommended datasets that don't require auth."""
        print("=" * 60)
        print("CRONOS AI - ML Platform Dataset Fetcher")
        print("=" * 60)

        results = {}

        # UCI datasets (no auth required)
        print("\n[1] Fetching UCI ML Repository datasets...")
        for name in self.DATASETS["uci"]:
            results[f"uci_{name}"] = self.fetch_uci_dataset(name)

        # Generate catalog
        print("\n[2] Generating dataset catalog...")
        self.generate_dataset_catalog()

        # Check what's available
        print("\n[3] Checking available tools...")
        print(f"  Hugging Face datasets library: {'✓ Installed' if self.check_huggingface_cli() else '✗ Not installed'}")
        print(f"  Kaggle CLI: {'✓ Installed' if self.check_kaggle_cli() else '✗ Not installed'}")

        print("\n" + "=" * 60)
        print("Fetch Complete!")
        print("=" * 60)

        return results

    def interactive_fetch(self):
        """Interactive mode to select and fetch datasets."""
        print("=" * 60)
        print("CRONOS AI - Interactive Dataset Fetcher")
        print("=" * 60)

        print("\nAvailable platforms:")
        print("  1. Hugging Face (requires: pip install datasets)")
        print("  2. Kaggle (requires: pip install kaggle + API key)")
        print("  3. UCI ML Repository (no auth required)")
        print("  4. Fetch all recommended (UCI only)")
        print("  5. Generate catalog only")

        choice = input("\nSelect option (1-5): ").strip()

        if choice == "1":
            if not self.check_huggingface_cli():
                print("\n⚠ Hugging Face datasets not installed.")
                print("Run: pip install datasets")
                return

            print("\nRecommended Hugging Face datasets:")
            for cat, datasets in self.DATASETS["huggingface"].items():
                print(f"\n  [{cat}]")
                for ds in datasets:
                    print(f"    - {ds['id']}: {ds['description']}")

            dataset_id = input("\nEnter dataset ID: ").strip()
            if dataset_id:
                self.fetch_huggingface_dataset(dataset_id)

        elif choice == "2":
            if not self.check_kaggle_cli():
                print("\n⚠ Kaggle CLI not installed or configured.")
                print("Run: pip install kaggle")
                print("Then configure API key from kaggle.com/account")
                return

            print("\nRecommended Kaggle datasets:")
            for cat, datasets in self.DATASETS["kaggle"].items():
                print(f"\n  [{cat}]")
                for ds in datasets:
                    print(f"    - {ds['id']}: {ds['description']}")

            dataset_id = input("\nEnter dataset ID: ").strip()
            if dataset_id:
                self.fetch_kaggle_dataset(dataset_id)

        elif choice == "3":
            print("\nUCI ML Repository datasets:")
            for name, info in self.DATASETS["uci"].items():
                print(f"  - {name}: {info['description']}")

            name = input("\nEnter dataset name: ").strip()
            if name:
                self.fetch_uci_dataset(name)

        elif choice == "4":
            self.fetch_recommended_free()

        elif choice == "5":
            self.generate_dataset_catalog()

        else:
            print("Invalid option")


def main():
    """Main entry point."""
    fetcher = MLPlatformFetcher()

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        fetcher.interactive_fetch()
    else:
        results = fetcher.fetch_recommended_free()

        print("\nResults:")
        for name, result in results.items():
            status = result.get("status", "unknown")
            print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
