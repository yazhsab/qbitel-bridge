# Product 9: Threat Intelligence Platform

## Overview

Threat Intelligence Platform (TIP) is QBITEL's comprehensive threat detection and hunting solution. It integrates with MITRE ATT&CK framework, ingests STIX/TAXII feeds, and provides automated threat hunting to proactively identify and respond to emerging threats.

---

## Problem Solved

### The Challenge

Security teams struggle with threat intelligence:

- **Information overload**: Millions of IOCs, impossible to process manually
- **Disconnected feeds**: Multiple sources with different formats
- **Manual correlation**: Hours to connect indicators to attacks
- **Reactive posture**: Waiting for alerts instead of hunting threats
- **Skill shortage**: Threat hunting requires specialized expertise

### The QBITEL Solution

Threat Intelligence Platform provides:
- **MITRE ATT&CK mapping**: Automatic technique identification
- **STIX/TAXII integration**: Unified feed ingestion
- **Automated hunting**: AI-powered proactive threat detection
- **IOC management**: Centralized indicator repository
- **Threat narratives**: LLM-generated attack explanations

---

## Key Features

### 1. MITRE ATT&CK Framework Integration

Complete coverage of the Enterprise ATT&CK matrix:

**14 Tactics**:
| Tactic | Techniques | Description |
|--------|-----------|-------------|
| Reconnaissance | 10 | Information gathering |
| Resource Development | 8 | Infrastructure preparation |
| Initial Access | 9 | Entry vectors |
| Execution | 14 | Running malicious code |
| Persistence | 19 | Maintaining foothold |
| Privilege Escalation | 13 | Gaining higher access |
| Defense Evasion | 42 | Avoiding detection |
| Credential Access | 17 | Stealing credentials |
| Discovery | 31 | Learning the environment |
| Lateral Movement | 9 | Moving through network |
| Collection | 17 | Gathering target data |
| Command and Control | 16 | Communicating with implants |
| Exfiltration | 9 | Stealing data |
| Impact | 13 | Disruption and destruction |

**Automatic TTP Detection**:
- Real-time mapping of security events to techniques
- Confidence scoring for each mapping
- Attack chain visualization
- Coverage gap analysis

### 2. STIX/TAXII Feed Integration

Unified threat intelligence ingestion:

**Supported Feeds**:
| Feed | Type | Update Frequency |
|------|------|-----------------|
| MISP | STIX 2.1 | Real-time |
| OTX (AlienVault) | STIX 2.0 | Hourly |
| Recorded Future | STIX 2.1 | Real-time |
| CrowdStrike | Proprietary | Real-time |
| VirusTotal | API | On-demand |
| Abuse.ch | CSV/STIX | Hourly |
| Custom TAXII | STIX 2.x | Configurable |

**Indicator Types**:
- IP addresses (IPv4/IPv6)
- Domains and URLs
- File hashes (MD5, SHA1, SHA256)
- Email addresses
- Certificates
- Mutexes
- Registry keys
- YARA rules
- Sigma rules

### 3. Automated Threat Hunting

AI-powered proactive threat detection:

```
Intelligence Input
    ├─ New IOCs from feeds
    ├─ Threat reports
    ├─ Malware analysis
    └─ TTP patterns
           │
           ▼
Hunt Generation
    ├─ YARA rules (files)
    ├─ Sigma rules (logs)
    ├─ Suricata rules (network)
    └─ Custom queries (SIEM)
           │
           ▼
Hunt Execution
    ├─ Log analysis
    ├─ Network traffic
    ├─ Endpoint telemetry
    └─ Cloud audit logs
           │
           ▼
Threat Detection
    ├─ IOC matches
    ├─ Behavioral patterns
    ├─ Anomaly detection
    └─ TTP identification
           │
           ▼
Alert & Response
    ├─ Incident creation
    ├─ Automated response
    └─ Analyst notification
```

### 4. IOC Management

Centralized indicator repository:

- **Deduplication**: Automatic removal of duplicates
- **Enrichment**: Context from multiple sources
- **Aging**: Automatic expiration of stale IOCs
- **Scoring**: Confidence and severity ratings
- **Relationships**: Link indicators to campaigns

### 5. Threat Narratives

LLM-generated attack explanations:

- **Attack summary**: Human-readable description
- **TTP breakdown**: Step-by-step technique analysis
- **Impact assessment**: Business risk evaluation
- **Mitigation guidance**: Recommended countermeasures
- **Historical context**: Similar past attacks

---

## Technical Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│              QBITEL Threat Intelligence Platform                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ STIX/TAXII   │  │   MITRE      │  │   Threat Hunting     │  │
│  │ Feeds        │  │   ATT&CK     │  │   Engine             │  │
│  │              │  │              │  │                      │  │
│  │ - MISP       │  │ - 14 tactics │  │ - YARA rules         │  │
│  │ - OTX        │  │ - 200+ techs │  │ - Sigma rules        │  │
│  │ - Custom     │  │ - Mappings   │  │ - Custom queries     │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         │                 │                      │              │
│         └─────────────────┼──────────────────────┘              │
│                           │                                     │
│                    ┌──────▼──────┐                              │
│                    │    IOC      │                              │
│                    │  Database   │                              │
│                    │             │                              │
│                    │ - 10M+ IOCs │                              │
│                    │ - Enriched  │                              │
│                    │ - Scored    │                              │
│                    └──────┬──────┘                              │
│                           │                                     │
│                    ┌──────▼──────┐                              │
│                    │    LLM      │                              │
│                    │  Analysis   │                              │
│                    │             │                              │
│                    │ - Narratives│                              │
│                    │ - Context   │                              │
│                    │ - Guidance  │                              │
│                    └─────────────┘                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | File | Description |
|-----------|------|-------------|
| ATT&CK Mapper | `mitre_attack_mapper.py` | MITRE ATT&CK integration |
| STIX/TAXII Client | `stix_taxii_client.py` | Feed ingestion |
| Threat Hunter | `threat_hunter.py` | Automated hunting |
| TIP Manager | `tip_manager.py` | Platform orchestration |
| IOC Database | `ioc_repository.py` | Indicator storage |

### Data Models

```python
class MITRETactic(str, Enum):
    RECONNAISSANCE = "reconnaissance"
    RESOURCE_DEVELOPMENT = "resource-development"
    INITIAL_ACCESS = "initial-access"
    EXECUTION = "execution"
    PERSISTENCE = "persistence"
    PRIVILEGE_ESCALATION = "privilege-escalation"
    DEFENSE_EVASION = "defense-evasion"
    CREDENTIAL_ACCESS = "credential-access"
    DISCOVERY = "discovery"
    LATERAL_MOVEMENT = "lateral-movement"
    COLLECTION = "collection"
    COMMAND_AND_CONTROL = "command-and-control"
    EXFILTRATION = "exfiltration"
    IMPACT = "impact"

@dataclass
class ATTACKTechnique:
    technique_id: str          # e.g., T1595, T1071.001
    name: str
    description: str
    tactics: List[MITRETactic]
    platforms: List[str]       # Windows, Linux, macOS, etc.
    data_sources: List[str]
    detection_methods: List[str]
    mitigations: List[str]
    subtechniques: List[str]
    is_subtechnique: bool
    kill_chain_phases: List[str]

@dataclass
class Indicator:
    indicator_id: str
    type: str                  # ip, domain, hash, url, etc.
    value: str
    confidence: float          # 0.0-1.0
    severity: str              # low, medium, high, critical
    source: str
    first_seen: datetime
    last_seen: datetime
    expires_at: datetime
    tags: List[str]
    related_campaigns: List[str]
    related_techniques: List[str]
    enrichment: Dict[str, Any]

@dataclass
class ThreatHunt:
    hunt_id: str
    name: str
    description: str
    hypothesis: str
    techniques: List[str]      # MITRE ATT&CK IDs
    indicators: List[str]      # IOC IDs
    queries: List[HuntQuery]
    status: str                # pending, running, completed
    started_at: datetime
    completed_at: datetime
    results: List[HuntResult]

@dataclass
class HuntResult:
    result_id: str
    hunt_id: str
    timestamp: datetime
    match_type: str            # ioc, behavior, anomaly
    confidence: float
    source_system: str
    evidence: Dict[str, Any]
    techniques_matched: List[str]
    severity: str
    recommended_actions: List[str]
```

---

## API Reference

### Ingest Indicators

```http
POST /api/v1/threat-intel/indicators
Content-Type: application/json
X-API-Key: your_api_key

{
    "indicators": [
        {
            "type": "ip",
            "value": "203.0.113.50",
            "confidence": 0.9,
            "severity": "high",
            "tags": ["c2", "cobalt-strike"],
            "description": "Cobalt Strike C2 server",
            "ttl_days": 30
        },
        {
            "type": "hash",
            "value": "abc123def456...",
            "hash_type": "sha256",
            "confidence": 0.95,
            "severity": "critical",
            "tags": ["ransomware", "lockbit"],
            "malware_family": "LockBit 3.0"
        }
    ],
    "source": "internal-analysis",
    "campaign": "APT29-2025"
}

Response:
{
    "ingested": 2,
    "duplicates": 0,
    "enriched": 2,
    "indicators": [
        {
            "indicator_id": "ioc_abc123",
            "type": "ip",
            "value": "203.0.113.50",
            "enrichment": {
                "asn": "AS12345",
                "country": "RU",
                "reputation_score": 0.1,
                "related_domains": ["malware.example.com"]
            }
        }
    ]
}
```

### Map to MITRE ATT&CK

```http
POST /api/v1/threat-intel/map-to-mitre
Content-Type: application/json
X-API-Key: your_api_key

{
    "event": {
        "event_type": "process_execution",
        "process_name": "powershell.exe",
        "command_line": "powershell -enc JABzAD0ATgBlAHcALQBP...",
        "parent_process": "winword.exe",
        "user": "DOMAIN\\user",
        "timestamp": "2025-01-16T10:30:00Z"
    }
}

Response:
{
    "mappings": [
        {
            "technique_id": "T1059.001",
            "technique_name": "PowerShell",
            "tactic": "execution",
            "confidence": 0.95,
            "evidence": [
                "PowerShell process execution",
                "Encoded command (-enc flag)",
                "Spawned from Office application"
            ]
        },
        {
            "technique_id": "T1566.001",
            "technique_name": "Spearphishing Attachment",
            "tactic": "initial-access",
            "confidence": 0.85,
            "evidence": [
                "PowerShell spawned from Word",
                "Indicates malicious document"
            ]
        },
        {
            "technique_id": "T1027",
            "technique_name": "Obfuscated Files or Information",
            "tactic": "defense-evasion",
            "confidence": 0.90,
            "evidence": [
                "Base64 encoded PowerShell command"
            ]
        }
    ],
    "attack_chain": {
        "initial_access": ["T1566.001"],
        "execution": ["T1059.001"],
        "defense_evasion": ["T1027"]
    },
    "narrative": "This event shows a classic spearphishing attack chain. A malicious Word document (T1566.001) executed an encoded PowerShell command (T1059.001, T1027). The base64 encoding is used to evade detection. Recommend immediate isolation of the affected host and investigation of the decoded payload."
}
```

### Create Threat Hunt

```http
POST /api/v1/threat-intel/hunts
Content-Type: application/json
X-API-Key: your_api_key

{
    "name": "APT29 IOC Hunt",
    "description": "Hunt for APT29 indicators across enterprise",
    "hypothesis": "APT29 may have compromised systems via spearphishing",
    "techniques": ["T1566.001", "T1059.001", "T1071.001"],
    "indicators": ["ioc_abc123", "ioc_def456"],
    "data_sources": [
        "endpoint_logs",
        "network_traffic",
        "email_logs"
    ],
    "time_range": {
        "start": "2025-01-01T00:00:00Z",
        "end": "2025-01-16T23:59:59Z"
    },
    "queries": [
        {
            "type": "sigma",
            "rule": "title: APT29 PowerShell\nlogsource:\n  category: process_creation\ndetection:\n  selection:\n    Image|endswith: '\\powershell.exe'\n    ParentImage|endswith: '\\winword.exe'\n  condition: selection"
        }
    ]
}

Response:
{
    "hunt_id": "hunt_abc123",
    "name": "APT29 IOC Hunt",
    "status": "running",
    "started_at": "2025-01-16T10:30:00Z",
    "estimated_completion": "2025-01-16T11:30:00Z",
    "progress": {
        "data_sources_searched": 0,
        "total_data_sources": 3,
        "events_analyzed": 0,
        "matches_found": 0
    }
}
```

### Get Hunt Results

```http
GET /api/v1/threat-intel/hunts/hunt_abc123/results
X-API-Key: your_api_key

Response:
{
    "hunt_id": "hunt_abc123",
    "status": "completed",
    "completed_at": "2025-01-16T11:25:00Z",
    "summary": {
        "total_events_analyzed": 15000000,
        "matches_found": 23,
        "high_confidence_matches": 5,
        "techniques_detected": ["T1059.001", "T1071.001"],
        "affected_hosts": 3
    },
    "results": [
        {
            "result_id": "result_001",
            "timestamp": "2025-01-10T14:30:00Z",
            "match_type": "behavior",
            "confidence": 0.92,
            "source_system": "workstation-15",
            "technique": "T1059.001",
            "evidence": {
                "process": "powershell.exe",
                "command_line": "powershell -enc ...",
                "parent": "outlook.exe",
                "user": "john.doe"
            },
            "severity": "high",
            "recommended_actions": [
                "Isolate workstation-15",
                "Collect memory dump",
                "Review email received by john.doe"
            ]
        }
    ],
    "iocs_discovered": [
        {
            "type": "ip",
            "value": "198.51.100.50",
            "context": "C2 communication destination"
        }
    ]
}
```

### Check Indicator

```http
POST /api/v1/threat-intel/indicators/check
Content-Type: application/json
X-API-Key: your_api_key

{
    "indicators": [
        {"type": "ip", "value": "203.0.113.50"},
        {"type": "domain", "value": "malware.example.com"},
        {"type": "hash", "value": "abc123def456...", "hash_type": "sha256"}
    ]
}

Response:
{
    "results": [
        {
            "type": "ip",
            "value": "203.0.113.50",
            "found": true,
            "indicator_id": "ioc_abc123",
            "confidence": 0.9,
            "severity": "high",
            "tags": ["c2", "cobalt-strike"],
            "first_seen": "2025-01-10T00:00:00Z",
            "campaigns": ["APT29-2025"],
            "techniques": ["T1071.001"]
        },
        {
            "type": "domain",
            "value": "malware.example.com",
            "found": false
        },
        {
            "type": "hash",
            "value": "abc123def456...",
            "found": true,
            "indicator_id": "ioc_def456",
            "confidence": 0.95,
            "severity": "critical",
            "malware_family": "LockBit 3.0"
        }
    ],
    "overall_risk": "critical"
}
```

### Subscribe to Feed

```http
POST /api/v1/threat-intel/feeds/subscribe
Content-Type: application/json
X-API-Key: your_api_key

{
    "feed_type": "taxii",
    "name": "MISP Feed",
    "url": "https://misp.example.com/taxii2",
    "api_key": "misp_api_key",
    "collection": "default",
    "polling_interval": "1h",
    "filters": {
        "types": ["indicator"],
        "min_confidence": 0.7,
        "tags": ["apt", "ransomware"]
    }
}

Response:
{
    "subscription_id": "sub_abc123",
    "feed_name": "MISP Feed",
    "status": "active",
    "last_poll": null,
    "next_poll": "2025-01-16T11:30:00Z",
    "indicators_received": 0
}
```

---

## Configuration

```yaml
threat_intelligence:
  enabled: true

  mitre_attack:
    enabled: true
    local_database: /var/lib/qbitel/mitre-attack.db
    update_interval: 24h
    auto_mapping: true
    confidence_threshold: 0.7

  stix_taxii:
    enabled: true
    feeds:
      - name: misp
        type: taxii
        url: https://misp.company.com/taxii2
        api_key: ${MISP_API_KEY}
        collection: default
        polling_interval: 1h
        enabled: true

      - name: otx
        type: otx
        url: https://otx.alienvault.com
        api_key: ${OTX_API_KEY}
        polling_interval: 1h
        enabled: true

      - name: abuse_ch
        type: csv
        url: https://feodotracker.abuse.ch/downloads/ipblocklist.csv
        polling_interval: 6h
        enabled: true

    cache:
      enabled: true
      ttl: 30m
      max_size: 100000

  threat_hunting:
    enabled: true
    auto_hunting: true
    hunting_interval: 4h
    min_severity: medium
    max_concurrent_hunts: 5
    data_sources:
      - endpoint_logs
      - network_traffic
      - cloud_audit_logs
      - email_logs

  ioc_management:
    default_ttl_days: 90
    auto_expiration: true
    deduplication: true
    enrichment:
      enabled: true
      providers:
        - virustotal
        - shodan
        - whois
    scoring:
      enabled: true
      decay_rate: 0.1  # Per day

  llm_analysis:
    enabled: true
    provider: ollama
    model: llama3.2:70b
    generate_narratives: true
```

---

## Monitoring & Observability

### Prometheus Metrics

```
# Feed metrics
qbitel_threat_intel_feeds_total{status="active|error"}
qbitel_threat_intel_feed_polls_total{feed}
qbitel_threat_intel_indicators_ingested_total{feed, type}

# IOC metrics
qbitel_threat_intel_iocs_total{type, severity}
qbitel_threat_intel_ioc_matches_total{type}
qbitel_threat_intel_ioc_age_days{type}

# MITRE ATT&CK metrics
qbitel_threat_intel_technique_detections_total{technique_id, tactic}
qbitel_threat_intel_attack_coverage_percent

# Hunt metrics
qbitel_threat_intel_hunts_total{status}
qbitel_threat_intel_hunt_duration_seconds
qbitel_threat_intel_hunt_matches_total{severity}
```

---

## Use Cases

### Use Case 1: Automatic IOC Alerting

```python
from qbitel.threat_intel import ThreatIntelClient

client = ThreatIntelClient(api_key="your_api_key")

# Check firewall logs against threat intel
for log_entry in firewall_logs:
    result = client.check_indicator(
        type="ip",
        value=log_entry.destination_ip
    )

    if result.found and result.severity in ["high", "critical"]:
        # Create alert
        client.create_alert(
            title=f"Malicious IP detected: {log_entry.destination_ip}",
            severity=result.severity,
            techniques=result.techniques,
            evidence=log_entry
        )
```

### Use Case 2: Proactive Threat Hunt

```python
# Hunt for specific APT group
hunt = client.create_hunt(
    name="APT29 Detection",
    techniques=["T1566.001", "T1059.001", "T1071.001", "T1003"],
    time_range={"days": 30},
    data_sources=["endpoint", "network", "email"]
)

# Wait for completion
results = hunt.wait_for_results()

# Review findings
for match in results.high_confidence_matches:
    print(f"Found: {match.technique} on {match.source_system}")
    print(f"Evidence: {match.evidence}")
```

---

## Conclusion

Threat Intelligence Platform transforms reactive security into proactive defense. With MITRE ATT&CK integration, automated feed ingestion, and AI-powered threat hunting, organizations can identify and respond to threats before they cause damage.
