"""
Fetch Real-World Threat Intelligence Datasets

Downloads and integrates public threat intelligence feeds:
- MITRE ATT&CK Enterprise Framework
- Abuse.ch ThreatFox IOCs
- Abuse.ch URLhaus malicious URLs
- Abuse.ch MalwareBazaar hashes
"""

import json
import csv
import gzip
import io
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import urllib.request
import ssl


class ThreatIntelFetcher:
    """Fetch real-world threat intelligence data."""

    def __init__(self, output_dir: Optional[str] = None):
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(__file__).parent.parent / "threat_intelligence"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # SSL context for HTTPS - handle certificate issues on some systems
        self.ssl_context = ssl.create_default_context()
        # Fallback unverified context for systems with cert issues
        self.ssl_context_unverified = ssl._create_unverified_context()

    def _fetch_url(self, url: str, timeout: int = 60) -> bytes:
        """Fetch URL content with SSL fallback."""
        print(f"  Fetching: {url}")
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "QBITEL-Dataset-Fetcher/1.0"}
        )
        # Try with verified SSL first, fall back to unverified if certificate issues
        try:
            with urllib.request.urlopen(req, context=self.ssl_context, timeout=timeout) as response:
                return response.read()
        except (ssl.SSLError, urllib.error.URLError) as e:
            if "certificate" in str(e).lower() or "ssl" in str(e).lower():
                print("  (Using unverified SSL - certificate issue detected)")
                with urllib.request.urlopen(req, context=self.ssl_context_unverified, timeout=timeout) as response:
                    return response.read()
            raise

    def fetch_mitre_attack(self) -> Dict:
        """
        Fetch MITRE ATT&CK Enterprise framework from GitHub.
        Source: https://github.com/mitre/cti
        License: Apache 2.0
        """
        print("\n[1] Fetching MITRE ATT&CK Enterprise Framework...")

        url = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"

        try:
            data = self._fetch_url(url)
            attack_data = json.loads(data)

            # Extract techniques
            techniques = []
            tactics_map = {}

            for obj in attack_data.get("objects", []):
                if obj.get("type") == "x-mitre-tactic":
                    tactic_name = obj.get("x_mitre_shortname", "")
                    tactics_map[obj.get("id")] = tactic_name

            for obj in attack_data.get("objects", []):
                if obj.get("type") == "attack-pattern" and not obj.get("revoked", False):
                    external_refs = obj.get("external_references", [])
                    technique_id = None
                    for ref in external_refs:
                        if ref.get("source_name") == "mitre-attack":
                            technique_id = ref.get("external_id")
                            break

                    if technique_id:
                        # Get tactics from kill chain phases
                        tactics = []
                        for phase in obj.get("kill_chain_phases", []):
                            if phase.get("kill_chain_name") == "mitre-attack":
                                tactics.append(phase.get("phase_name"))

                        technique = {
                            "technique_id": technique_id,
                            "name": obj.get("name", ""),
                            "description": obj.get("description", "")[:500],  # Truncate
                            "tactics": tactics,
                            "platforms": obj.get("x_mitre_platforms", []),
                            "detection": obj.get("x_mitre_detection", "")[:300] if obj.get("x_mitre_detection") else "",
                            "data_sources": obj.get("x_mitre_data_sources", []),
                            "is_subtechnique": "." in technique_id,
                        }
                        techniques.append(technique)

            # Save to file
            mitre_dir = self.output_dir / "mitre_attack"
            mitre_dir.mkdir(exist_ok=True)

            output = {
                "source": "MITRE ATT&CK",
                "version": attack_data.get("spec_version", "unknown"),
                "fetched_at": datetime.now().isoformat(),
                "license": "Apache 2.0",
                "count": len(techniques),
                "techniques": techniques
            }

            with open(mitre_dir / "techniques.json", "w") as f:
                json.dump(output, f, indent=2)

            print(f"  ✓ Saved {len(techniques)} MITRE ATT&CK techniques")
            return {"mitre_techniques": len(techniques)}

        except Exception as e:
            print(f"  ✗ Error fetching MITRE ATT&CK: {e}")
            return {"mitre_techniques": 0, "error": str(e)}

    def fetch_threatfox_iocs(self, limit: int = 5000) -> Dict:
        """
        Fetch IOCs from Abuse.ch ThreatFox.
        Source: https://threatfox.abuse.ch
        License: CC0 (Public Domain)
        """
        print("\n[2] Fetching ThreatFox IOCs...")

        url = "https://threatfox.abuse.ch/export/json/recent/"

        try:
            data = self._fetch_url(url)
            threatfox_data = json.loads(data)

            iocs = []
            for ioc_id, ioc_info in list(threatfox_data.items())[:limit]:
                if isinstance(ioc_info, dict):
                    ioc = {
                        "ioc_id": f"tf_{ioc_id}",
                        "type": ioc_info.get("ioc_type", "unknown"),
                        "value": ioc_info.get("ioc", ""),
                        "threat_type": ioc_info.get("threat_type", "unknown"),
                        "malware": ioc_info.get("malware", ""),
                        "malware_alias": ioc_info.get("malware_alias", ""),
                        "confidence": int(ioc_info.get("confidence_level", 50)) / 100,
                        "first_seen": ioc_info.get("first_seen", ""),
                        "last_seen": ioc_info.get("last_seen", ""),
                        "reporter": ioc_info.get("reporter", ""),
                        "tags": ioc_info.get("tags", []) if ioc_info.get("tags") else [],
                        "source": "ThreatFox"
                    }
                    iocs.append(ioc)

            # Save to file
            ioc_dir = self.output_dir / "iocs"
            ioc_dir.mkdir(exist_ok=True)

            output = {
                "source": "Abuse.ch ThreatFox",
                "fetched_at": datetime.now().isoformat(),
                "license": "CC0 (Public Domain)",
                "count": len(iocs),
                "indicators": iocs
            }

            with open(ioc_dir / "threatfox_iocs.json", "w") as f:
                json.dump(output, f, indent=2)

            print(f"  ✓ Saved {len(iocs)} ThreatFox IOCs")
            return {"threatfox_iocs": len(iocs)}

        except Exception as e:
            print(f"  ✗ Error fetching ThreatFox: {e}")
            return {"threatfox_iocs": 0, "error": str(e)}

    def fetch_urlhaus(self, limit: int = 5000) -> Dict:
        """
        Fetch malicious URLs from Abuse.ch URLhaus.
        Source: https://urlhaus.abuse.ch
        License: CC0 (Public Domain)
        """
        print("\n[3] Fetching URLhaus Malicious URLs...")

        url = "https://urlhaus.abuse.ch/downloads/csv_recent/"

        try:
            data = self._fetch_url(url)
            content = data.decode("utf-8")

            urls = []
            reader = csv.reader(io.StringIO(content))

            for i, row in enumerate(reader):
                if i == 0 or row[0].startswith("#"):  # Skip header/comments
                    continue
                if i > limit:
                    break

                if len(row) >= 8:
                    ioc = {
                        "ioc_id": f"urlhaus_{row[0]}",
                        "type": "url",
                        "value": row[2],
                        "threat_type": row[5] if len(row) > 5 else "malware",
                        "status": row[3],
                        "date_added": row[1],
                        "tags": row[6].split(",") if len(row) > 6 and row[6] else [],
                        "confidence": 0.85,
                        "source": "URLhaus"
                    }
                    urls.append(ioc)

            # Save to file
            ioc_dir = self.output_dir / "iocs"
            ioc_dir.mkdir(exist_ok=True)

            output = {
                "source": "Abuse.ch URLhaus",
                "fetched_at": datetime.now().isoformat(),
                "license": "CC0 (Public Domain)",
                "count": len(urls),
                "indicators": urls
            }

            with open(ioc_dir / "urlhaus_urls.json", "w") as f:
                json.dump(output, f, indent=2)

            print(f"  ✓ Saved {len(urls)} URLhaus malicious URLs")
            return {"urlhaus_urls": len(urls)}

        except Exception as e:
            print(f"  ✗ Error fetching URLhaus: {e}")
            return {"urlhaus_urls": 0, "error": str(e)}

    def fetch_malwarebazaar_hashes(self, limit: int = 1000) -> Dict:
        """
        Fetch malware hashes from Abuse.ch MalwareBazaar.
        Source: https://bazaar.abuse.ch
        License: CC0 (Public Domain)
        """
        print("\n[4] Fetching MalwareBazaar Hashes...")

        url = "https://bazaar.abuse.ch/export/csv/recent/"

        try:
            data = self._fetch_url(url)

            # Try to decompress if gzipped
            try:
                content = gzip.decompress(data).decode("utf-8")
            except Exception:
                content = data.decode("utf-8")

            hashes = []
            reader = csv.reader(io.StringIO(content))

            for i, row in enumerate(reader):
                if i == 0 or (row and row[0].startswith("#")):
                    continue
                if i > limit:
                    break

                if len(row) >= 7:
                    ioc = {
                        "ioc_id": f"bazaar_{i}",
                        "type": "hash_sha256",
                        "value": row[1] if len(row) > 1 else "",
                        "md5": row[2] if len(row) > 2 else "",
                        "sha1": row[3] if len(row) > 3 else "",
                        "threat_type": "malware",
                        "file_type": row[5] if len(row) > 5 else "",
                        "signature": row[6] if len(row) > 6 else "",
                        "first_seen": row[0] if row else "",
                        "confidence": 0.95,
                        "source": "MalwareBazaar"
                    }
                    if ioc["value"]:  # Only add if hash exists
                        hashes.append(ioc)

            # Save to file
            ioc_dir = self.output_dir / "iocs"
            ioc_dir.mkdir(exist_ok=True)

            output = {
                "source": "Abuse.ch MalwareBazaar",
                "fetched_at": datetime.now().isoformat(),
                "license": "CC0 (Public Domain)",
                "count": len(hashes),
                "indicators": hashes
            }

            with open(ioc_dir / "malwarebazaar_hashes.json", "w") as f:
                json.dump(output, f, indent=2)

            print(f"  ✓ Saved {len(hashes)} MalwareBazaar hashes")
            return {"malwarebazaar_hashes": len(hashes)}

        except Exception as e:
            print(f"  ✗ Error fetching MalwareBazaar: {e}")
            return {"malwarebazaar_hashes": 0, "error": str(e)}

    def merge_iocs(self) -> Dict:
        """Merge all IOC sources into a unified file."""
        print("\n[5] Merging all IOC sources...")

        ioc_dir = self.output_dir / "iocs"
        all_iocs = []

        ioc_files = [
            "threatfox_iocs.json",
            "urlhaus_urls.json",
            "malwarebazaar_hashes.json",
            "indicators.json"  # Synthetic data
        ]

        for filename in ioc_files:
            filepath = ioc_dir / filename
            if filepath.exists():
                with open(filepath) as f:
                    data = json.load(f)
                    indicators = data.get("indicators", [])
                    all_iocs.extend(indicators)
                    print(f"  - {filename}: {len(indicators)} IOCs")

        # Save merged file
        output = {
            "source": "QBITEL Merged IOCs",
            "fetched_at": datetime.now().isoformat(),
            "sources": ioc_files,
            "count": len(all_iocs),
            "indicators": all_iocs
        }

        with open(ioc_dir / "all_indicators.json", "w") as f:
            json.dump(output, f, indent=2)

        print(f"  ✓ Merged {len(all_iocs)} total IOCs")
        return {"total_iocs": len(all_iocs)}

    def fetch_all(self) -> Dict:
        """Fetch all threat intelligence sources."""
        print("=" * 60)
        print("QBITEL - Threat Intelligence Fetcher")
        print("=" * 60)

        results = {}

        # Fetch each source
        results.update(self.fetch_mitre_attack())
        results.update(self.fetch_threatfox_iocs())
        results.update(self.fetch_urlhaus())
        results.update(self.fetch_malwarebazaar_hashes())
        results.update(self.merge_iocs())

        print("\n" + "=" * 60)
        print("Fetch Complete!")
        print("=" * 60)

        total = sum(v for k, v in results.items() if isinstance(v, int))
        print(f"\nTotal records fetched: {total:,}")

        return results


def main():
    """Fetch all threat intelligence datasets."""
    fetcher = ThreatIntelFetcher()
    results = fetcher.fetch_all()

    print("\nResults:")
    for key, value in results.items():
        if isinstance(value, int):
            print(f"  {key}: {value:,}")


if __name__ == "__main__":
    main()
